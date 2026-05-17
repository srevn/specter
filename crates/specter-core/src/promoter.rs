//! Promoter — engine-resident dynamic-watch source.
//!
//! A `Promoter` is a peer to `Profile` in the engine: each one carries a
//! `PatternSpec`, a literal-prefix probe state, an `Active` proxy fan-out
//! over matched directories, and a deduplicated map of synthesised
//! dynamic Subs. The engine drives the lifecycle through a state machine;
//! `core::promoter` owns the data shapes and the registry.
//!
//! ## State
//!
//! - `PrefixPending(DescentState)` — the literal prefix doesn't yet exist
//!   on disk. `DescentState.current_prefix` is the deepest existing
//!   ancestor; descent advances one literal segment per probe response
//!   until the prefix materialises.
//! - `Active { proxies }` — literal prefix exists. Each proxy is a Resource
//!   slot carrying a `+1 STRUCTURE` `watch_demand` contribution; events
//!   on a proxy queue an enumeration probe.
//!
//! The two states are mutually exclusive (Rust sum-type). The transition
//! `PrefixPending → Active` is single-shot per Promoter lifetime — once
//! the prefix exists, descent yields to enumeration.
//!
//! ## Single-slot probe
//!
//! At most one outstanding probe per Promoter — a representability
//! property, not a runtime check. `PrefixPending` homes the descent
//! probe on its `DescentState` slot; `Active` homes the enumeration
//! probe on its own `enumerating` slot. The two states are mutually
//! exclusive, so a Promoter holds exactly one probe slot at any
//! instant. Concurrent enumeration requests queue in
//! `pending_enumerations` and drain one at a time — the engine arms
//! the `Active` slot for the popped target and refuses to pop another
//! while it stays armed.
//!
//! ## Dynamic Sub deduplication
//!
//! `dynamic_subs: BTreeMap<ResourceId, SubId>` enforces at most one
//! dynamic Sub per `(promoter_id, anchor_resource)`. Resource-keying is
//! structurally equivalent to path-keying: Tree slot identity is
//! `(parent, segment)`, bijective with the resolved path while the slot
//! is live, and the Sub's `AnchorClaim::Held` contribution keeps the
//! slot from reaping for the dedup entry's lifetime. The dedup entry
//! drops at `on_dynamic_sub_reaped` *before* `reap_profile` releases
//! the anchor contribution, so a re-mint after the slot reaps lands at
//! a fresh `ResourceId` and never collides with stale state. The map
//! is private: the only mutators are [`Promoter::promote`] (insert,
//! dedup-asserted), [`Promoter::forget_dynamic_sub`] (anchor-terminal
//! remove), and [`Promoter::drain_dynamic_subs`] (teardown drain) —
//! the discipline is structural, not a documented convention.

use crate::ids::{ProbeCorrelation, PromoterId, ResourceId, SubId};
use crate::pattern::PatternSpec;
use crate::probe::ProbeSlot;
use crate::profile::DescentState;
use crate::program::ActionProgram;
use crate::scan_config::ProfileIdentity;
use crate::sub::EffectScope;
use compact_str::CompactString;
use slotmap::SlotMap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// Pre-id spec carried on `WatchRegistryDiff::promoters.{added,modified}`.
///
/// Mirrors [`SubAttachRequest`](crate::SubAttachRequest)'s role for the
/// static side: the config layer materialises this from a `[[promoter]]`
/// (or auto-detected `[[watch]]`) block; the engine assigns a
/// [`PromoterId`] at attach. `Clone` serves the rare multi-Engine
/// fan-out. No `Eq`/`PartialEq`: [`ProfileIdentity::config_hash`] is the
/// only identity comparison, never a structural derive.
///
/// `name` is `CompactString`, moved end to end from the already-
/// `CompactString` `PromoterSpec.name` — no `String` round-trip.
#[derive(Clone, Debug)]
pub struct PromoterAttachRequest {
    pub name: CompactString,
    pub pattern_spec: PatternSpec,
    pub identity: ProfileIdentity,
    pub settle: Duration,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub log_output: bool,
}

/// Engine-resident Promoter.
///
/// Mirrors `Profile`'s registry-stored shape and its encapsulation
/// discipline. No `id` field — the slotmap [`PromoterId`] is the
/// identity authority; helper code that needs the id receives it as a
/// parameter. `identity` is the Sub-spec's Profile partition key,
/// threaded verbatim into every synthesised dynamic Sub.
///
/// The seven spec fields are `pub`: frozen at [`Self::from_request`],
/// never written post-construction (the `Sub`-frozen shape — benign
/// all-`pub`). The four **runtime** fields are module-private; the
/// cross-crate write surface is this type's `pub fn`s, never a field
/// assignment. Each runtime mutator owns its invariant structurally
/// (matching `Profile`'s sealed state machine and CLAUDE.md "single
/// source per transition"): the one-shot `PrefixPending → Active` move
/// ([`Self::enter_active_empty`]); the in-`state` linear [`ProbeSlot`]
/// (armed only via [`Self::arm_enumeration`], consumed only via
/// [`Self::take_probe`]); the dedup map ([`Self::promote`] /
/// [`Self::forget_dynamic_sub`] / [`Self::drain_dynamic_subs`]); the
/// enumeration queue ([`Self::enqueue_enumeration`] /
/// [`Self::pop_enumeration`] / [`Self::unregister_proxy_slot`]); and
/// the one-shot fan-out latch ([`Self::latch_fanout_warning`]). Reads
/// project through [`Self::state`] / [`Self::dynamic_subs`] /
/// [`Self::pending_enumerations`].
#[derive(Debug)]
pub struct Promoter {
    pub name: CompactString,
    pub pattern: Arc<PatternSpec>,
    pub identity: ProfileIdentity,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub settle: Duration,
    pub log_output: bool,

    /// State machine (`PrefixPending` XOR `Active`); homes this
    /// Promoter's single linear [`ProbeSlot`]. Private — an out-of-band
    /// `state` write would drop an armed slot past its `Drop` tripwire
    /// or bypass [`ProbeSlot::arm`]'s arm-once assert, defeating the
    /// "one probe per Promoter" representability property. Read via
    /// [`Self::state`]; the only transition is [`Self::enter_active_empty`].
    state: PromoterState,

    /// Deterministic queue of proxies awaiting enumeration. `BTreeSet`
    /// for stable iteration; the single-slot drain pops one at a time.
    /// Private — mutate via [`Self::enqueue_enumeration`] /
    /// [`Self::pop_enumeration`] / [`Self::unregister_proxy_slot`].
    pending_enumerations: BTreeSet<ResourceId>,

    /// `anchor_resource → SubId`. Resource identity is
    /// `(parent, segment)` — bijective with the resolved path while the
    /// slot is live; the Sub's `AnchorClaim::Held` contribution keeps
    /// the slot from reaping for the dedup entry's lifetime. Private:
    /// the dedup invariant (one Sub per `(promoter, anchor)`) is
    /// enforced at [`Self::promote`].
    dynamic_subs: BTreeMap<ResourceId, SubId>,

    /// One-shot fan-out warning latch. Fully private — its only
    /// interaction is the atomic check-and-latch in
    /// [`Self::latch_fanout_warning`], so a pathological pattern warns
    /// once per Promoter lifetime by construction.
    warned_at_threshold: bool,
}

impl Promoter {
    /// Construct an engine-resident Promoter from its frozen spec
    /// ([`PromoterAttachRequest`]) and the engine-computed initial
    /// `state`. The runtime fields start empty/unset — at attach the
    /// Promoter has minted no dynamic Subs and queued no enumeration.
    /// `pattern_spec` is `Arc`-wrapped here (the hot enumeration
    /// dispatcher bumps the refcount per response to release the
    /// registry read borrow). The slotmap [`PromoterId`] assigned by
    /// [`PromoterRegistry::insert`] is the identity authority — no `id`
    /// is embedded. Mirrors `Sub::from_request` for the static side.
    #[must_use]
    pub fn from_request(req: PromoterAttachRequest, state: PromoterState) -> Self {
        Self {
            name: req.name,
            pattern: Arc::new(req.pattern_spec),
            identity: req.identity,
            program: req.program,
            scope: req.scope,
            settle: req.settle,
            log_output: req.log_output,
            state,
            pending_enumerations: BTreeSet::new(),
            dynamic_subs: BTreeMap::new(),
            warned_at_threshold: false,
        }
    }

    /// Immutable projection of the state machine — the sole read seam
    /// for `state`. Mirrors [`Profile::state`](crate::Profile::state):
    /// callers pattern-match the returned `&PromoterState`
    /// (`PrefixPending` XOR `Active`); the write surface is the named
    /// mutators below, never a field assignment.
    #[must_use]
    pub const fn state(&self) -> &PromoterState {
        &self.state
    }

    /// The single `PrefixPending → Active` transition (once per
    /// Promoter lifetime — the prefix materialised, or its claim is
    /// being released). Replaces `state` with
    /// `Active { proxies: ∅, enumerating: ∅ }`; the prior state is
    /// dropped here.
    ///
    /// **Cancel-first contract (structural).** The discarded prior
    /// carries this Promoter's probe slot — the descent slot in
    /// `PrefixPending`, the `enumerating` slot in `Active`. If it is
    /// still armed, [`ProbeSlot`]'s `Drop` tripwire fires. Callers MUST
    /// consume the in-flight probe (`cancel_owner_probe` /
    /// [`Self::take_probe`]) before this — the discard *is* the
    /// enforcement, the Promoter dual of `reap_profile`'s structural
    /// guard. No cancel-first asserts scattered at call sites.
    pub fn enter_active_empty(&mut self) {
        self.state = PromoterState::Active {
            proxies: BTreeMap::new(),
            enumerating: ProbeSlot::empty(),
        };
    }

    /// Mutable descent projection — `Some` only in `PrefixPending`
    /// (`Active` yields `None`). The sole `&mut` seam into the descent
    /// payload (probe arm / segment advance). Symmetric with
    /// [`Profile::descent_state_mut`](crate::Profile::descent_state_mut).
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        self.state.descent_state_mut()
    }

    /// Disarm this Promoter's in-flight probe (descent slot in
    /// `PrefixPending`, enumeration slot in `Active`) and return its
    /// correlation, or `None`. The single state-level consume; the
    /// disarm leaves the state variant intact. Symmetric with
    /// [`Profile::take_probe`](crate::Profile::take_probe).
    pub const fn take_probe(&mut self) -> Option<ProbeCorrelation> {
        self.state.take_probe()
    }

    /// Arm the `Active` enumeration slot for `target` with a freshly
    /// minted `correlation`. Delegates to
    /// [`PromoterState::arm_enumeration`]; its `PrefixPending` arm is
    /// `unreachable!()` (enumeration drains `pending_enumerations`,
    /// non-empty only in `Active`), and [`ProbeSlot::arm`]'s arm-once
    /// assert surfaces a re-arm in every build.
    pub fn arm_enumeration(&mut self, correlation: ProbeCorrelation, target: ResourceId) {
        self.state.arm_enumeration(correlation, target);
    }

    /// Register a proxy at `resource` in the `Active` proxies map with
    /// enumeration cursor `pattern_component_index`. Overwrites an
    /// existing entry (re-registration is gated idempotent at the
    /// caller). `unreachable!()` in `PrefixPending` — the sole callers
    /// (`enter_active`, the enumeration forward pass) guarantee
    /// `Active` by construction; a wrong call surfaces loudly in every
    /// build rather than silently no-op.
    pub fn insert_proxy(&mut self, resource: ResourceId, pattern_component_index: usize) {
        match &mut self.state {
            PromoterState::Active { proxies, .. } => {
                proxies.insert(
                    resource,
                    ProxyState {
                        pattern_component_index,
                    },
                );
            }
            PromoterState::PrefixPending(_) => unreachable!(
                "insert_proxy requires Active: enter_active and the enumeration \
                 forward pass are the sole callers and both ensure Active",
            ),
        }
    }

    /// Drop the proxy at `resource`: remove it from the `Active`
    /// proxies map (no-op if not `Active` / absent) **and** from
    /// `pending_enumerations` (unconditional — a queued enumeration for
    /// a now-gone proxy must not resurrect after the slot reaps). The
    /// inverse of [`Self::insert_proxy`] plus the queue cleanup the two
    /// always pair at the call site.
    pub fn unregister_proxy_slot(&mut self, resource: ResourceId) {
        if let PromoterState::Active { proxies, .. } = &mut self.state {
            proxies.remove(&resource);
        }
        self.pending_enumerations.remove(&resource);
    }

    /// Immutable view of the enumeration queue (introspection and
    /// determinism assertions). Mutated only via
    /// [`Self::enqueue_enumeration`] / [`Self::pop_enumeration`] /
    /// [`Self::unregister_proxy_slot`].
    #[must_use]
    pub const fn pending_enumerations(&self) -> &BTreeSet<ResourceId> {
        &self.pending_enumerations
    }

    /// Queue `resource` for enumeration. `BTreeSet`-idempotent —
    /// concurrent events at one proxy collapse to a single
    /// enumeration. Returns `true` iff newly queued.
    pub fn enqueue_enumeration(&mut self, resource: ResourceId) -> bool {
        self.pending_enumerations.insert(resource)
    }

    /// Pop the next queued enumeration target (`BTreeSet` order — the
    /// deterministic single-slot drain), or `None` when empty.
    pub fn pop_enumeration(&mut self) -> Option<ResourceId> {
        self.pending_enumerations.pop_first()
    }

    /// Immutable view of the dynamic-Sub dedup map — the caller's
    /// dedup gate read (`contains_key`) plus introspection. At most one
    /// entry per `(promoter, anchor_resource)`; the invariant is
    /// enforced at [`Self::promote`].
    #[must_use]
    pub const fn dynamic_subs(&self) -> &BTreeMap<ResourceId, SubId> {
        &self.dynamic_subs
    }

    /// Record the dynamic Sub minted for `anchor_resource`. The dedup
    /// invariant (one Sub per `(promoter, anchor)`) is the caller's
    /// `dynamic_subs().contains_key` gate + early-return; this
    /// `debug_assert!`s the key was absent as the loud dev/CI backstop.
    /// A release-mode escape overwrites the map entry but orphans
    /// nothing: the prior Sub still reaps via its own anchor-terminal
    /// path.
    pub fn promote(&mut self, anchor_resource: ResourceId, sub_id: SubId) {
        let prev = self.dynamic_subs.insert(anchor_resource, sub_id);
        debug_assert!(
            prev.is_none(),
            "promote: dedup invariant breached — anchor {anchor_resource:?} already \
             mapped (caller must gate on dynamic_subs().contains_key)",
        );
    }

    /// Drop the dedup entry for `anchor_resource` (the Sub's
    /// anchor-terminal reap), returning the `SubId` that was mapped, or
    /// `None` if absent (a concurrent [`Self::drain_dynamic_subs`]
    /// already cleared it — benign).
    pub fn forget_dynamic_sub(&mut self, anchor_resource: ResourceId) -> Option<SubId> {
        self.dynamic_subs.remove(&anchor_resource)
    }

    /// Drain every dedup entry (Promoter teardown), returning the
    /// minted `SubId`s for the caller to detach. The map is left empty
    /// so cascading detach paths observe no entries.
    #[must_use]
    pub fn drain_dynamic_subs(&mut self) -> Vec<SubId> {
        let ids = self.dynamic_subs.values().copied().collect();
        self.dynamic_subs.clear();
        ids
    }

    /// One-shot fan-out warning latch. Returns `Some(count)` the first
    /// time `dynamic_subs` exceeds `threshold` and latches so later
    /// crossings return `None` — a pathological pattern warns once per
    /// Promoter lifetime. The check-and-latch is atomic here, so the
    /// one-shot property is structural rather than a caller convention.
    pub fn latch_fanout_warning(&mut self, threshold: usize) -> Option<usize> {
        let count = self.dynamic_subs.len();
        (count > threshold && !self.warned_at_threshold).then(|| {
            self.warned_at_threshold = true;
            count
        })
    }
}

/// Mutually-exclusive Promoter state. `PrefixPending` covers the
/// pre-materialised case; `Active` covers the operating case.
///
/// Each variant homes this Promoter's single probe slot —
/// `PrefixPending` on its `DescentState`, `Active` on `enumerating` —
/// so "at most one probe per Promoter" is structural: there is only
/// ever one slot, selected by which state the Promoter is in.
#[derive(Debug)]
pub enum PromoterState {
    /// Literal-prefix doesn't yet exist on disk. `DescentState.current_prefix`
    /// is the deepest existing ancestor; `remaining_components` are the
    /// literal segments to descend (root excluded).
    PrefixPending(DescentState),

    /// Literal-prefix has materialised. `proxies` keys are Resource slots
    /// holding a `+1 STRUCTURE` `watch_demand` contribution; values carry
    /// the position in `pattern.components` to enumerate next.
    ///
    /// `BTreeMap` for deterministic iteration order across replays.
    ///
    /// `enumerating` is this Promoter's single in-flight enumeration
    /// probe. Armed while a proxy enumeration is outstanding — it holds
    /// both the correlation the response must echo and the proxy
    /// `ResourceId` the probe targets. The wire is path-only, so this
    /// tag is the sole authority for the dispatch key on every outcome
    /// (`SubtreeOk` / `Vanished` / `Failed`). Empty while the Promoter
    /// operates with no enumeration in flight.
    Active {
        proxies: BTreeMap<ResourceId, ProxyState>,
        enumerating: ProbeSlot<ResourceId>,
    },
}

impl PromoterState {
    /// Borrow the descent payload if the state is currently
    /// [`Self::PrefixPending`]. `None` for [`Self::Active`] — descent
    /// only lives in the pre-materialised state.
    ///
    /// Symmetric with [`crate::ProfileState::descent_state`]; the
    /// engine's owner-polymorphic `descent_state` dispatcher routes
    /// to either projection through [`crate::ProbeOwner`].
    #[must_use]
    pub const fn descent_state(&self) -> Option<&DescentState> {
        match self {
            Self::PrefixPending(d) => Some(d),
            Self::Active { .. } => None,
        }
    }

    /// Mutable counterpart to [`Self::descent_state`].
    pub const fn descent_state_mut(&mut self) -> Option<&mut DescentState> {
        match self {
            Self::PrefixPending(d) => Some(d),
            Self::Active { .. } => None,
        }
    }

    /// The correlation of this Promoter's in-flight probe, or `None`.
    /// A total projection over both states: a `PrefixPending` descent
    /// or an `Active` enumeration answers from its armed slot; an empty
    /// slot in either state yields `None`. Owner-symmetric with
    /// [`crate::ProfileState::probe_correlation`].
    #[must_use]
    pub const fn probe_correlation(&self) -> Option<ProbeCorrelation> {
        match self {
            Self::PrefixPending(d) => d.probe_correlation(),
            Self::Active { enumerating, .. } => enumerating.correlation(),
        }
    }

    /// Disarm this Promoter's probe-bearing carrier and return the
    /// prior correlation — the single state-level consume, total over
    /// both states (`PrefixPending` descent slot or `Active`
    /// enumeration slot; an already-empty slot is a `None` no-op). The
    /// disarm leaves the state variant intact, so a route computed
    /// before this call stays valid after it. Owner-symmetric with
    /// [`crate::ProfileState::take_probe`].
    pub const fn take_probe(&mut self) -> Option<ProbeCorrelation> {
        match self {
            Self::PrefixPending(d) => d.disarm_probe(),
            Self::Active { enumerating, .. } => enumerating.disarm(),
        }
    }

    /// Arm the `Active` enumeration slot with a freshly-minted
    /// `correlation` for `target` (the proxy the probe enumerates).
    /// The mint-side twin of [`DescentState::arm_probe`] for the
    /// enumeration carrier; the consume direction is deliberately not
    /// exposed here — it routes through [`Self::take_probe`] so
    /// consume-once stays one law. [`ProbeSlot::arm`] asserts the slot
    /// was empty: a re-arm without an intervening disarm would orphan
    /// the prior correlation, so it must surface in every build.
    ///
    /// `PrefixPending` has no enumeration slot. Reaching that arm is a
    /// caller-discipline breach — enumeration is dispatched only by
    /// draining `pending_enumerations`, which is populated solely while
    /// `Active`. Surfaced loudly rather than silently dropped: a silent
    /// miss would emit a probe whose response then stale-detects
    /// against an empty slot.
    pub fn arm_enumeration(&mut self, correlation: ProbeCorrelation, target: ResourceId) {
        match self {
            Self::Active { enumerating, .. } => enumerating.arm(correlation, target),
            Self::PrefixPending(_) => unreachable!(
                "arm_enumeration requires Active: enumeration drains \
                 pending_enumerations, which is non-empty only in Active",
            ),
        }
    }

    /// The proxy `ResourceId` the in-flight enumeration probe targets,
    /// or `None` (`Active` with no probe out, or `PrefixPending`). The
    /// single read the cancel-gate sites share so they cannot drift:
    /// "is the in-flight enumeration aimed at *this* proxy?" The wire
    /// is path-only, so this slot tag is the sole authority for the
    /// dispatch key across every enumeration outcome.
    #[must_use]
    pub const fn enumeration_target(&self) -> Option<ResourceId> {
        match self {
            Self::Active { enumerating, .. } => enumerating.tag(),
            Self::PrefixPending(_) => None,
        }
    }
}

/// Per-proxy enumeration cursor.
///
/// `pattern_component_index` points at the `PatternComponent` to test
/// children of this proxy against. The first proxy at
/// `PrefixPending → Active` gets index `pattern.literal_prefix_len` (the
/// first non-literal component); deeper sub-proxies advance one position
/// per registration.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ProxyState {
    pub pattern_component_index: usize,
}

/// Slotmap-backed Promoter registry with a name index.
///
/// Mirrors `ProfileMap`'s shape: a `SlotMap` keyed on `PromoterId`
/// plus a `BTreeMap<name, PromoterId>` for configuration-driven lookup
/// at hot-reload time. Every Promoter is operator-named (there is no
/// synthesised Promoter), so `by_name` indexes all of them — the
/// asymmetry with [`SubRegistry`](crate::sub::SubRegistry)'s
/// static-only index.
///
/// `by_name` mirrors the slotmap entry's lifetime: `insert` populates
/// both; `remove` clears both **id-checked** (the entry drops only if
/// it still points at the removed id). Lookup is O(log N) and is
/// load-bearing — the engine's hot-reload shim resolves every
/// `removed`/`modified` Promoter name through [`Self::find_by_name`].
/// The `insert` `debug_assert!` is the dev/CI duplicate-name signal;
/// config validation makes a duplicate unreachable in correct
/// operation, and the id-checked `remove` is the release backstop for
/// the mapping.
#[derive(Debug, Default)]
pub struct PromoterRegistry {
    promoters: SlotMap<PromoterId, Promoter>,
    by_name: BTreeMap<CompactString, PromoterId>,
}

impl PromoterRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a Promoter; the returned slotmap [`PromoterId`] is its
    /// identity authority (the Promoter carries no `id` field). The
    /// `by_name` index is updated in lockstep. Mirrors
    /// [`SubRegistry::insert`](crate::sub::SubRegistry::insert).
    ///
    /// The `debug_assert!` fires on a duplicate name — the dev/CI
    /// signal only; config validation makes a duplicate unreachable in
    /// correct operation, and a release-mode breach is contained by
    /// the id-checked [`Self::remove`].
    pub fn insert(&mut self, promoter: Promoter) -> PromoterId {
        let name = promoter.name.clone();
        let id = self.promoters.insert(promoter);
        debug_assert!(
            !self.by_name.contains_key(&name),
            "duplicate Promoter name escaped config validation: {name:?}",
        );
        self.by_name.insert(name, id);
        id
    }

    /// Remove a Promoter, returning the owned value. The `by_name`
    /// clear is **id-checked** — the entry drops only if it still
    /// points at `id`, so removing a duplicate-name escape's shadowed
    /// id (a release-mode diff bug) cannot clobber the live id's
    /// mapping. Returns `None` for a stale id.
    pub fn remove(&mut self, id: PromoterId) -> Option<Promoter> {
        let p = self.promoters.remove(id)?;
        if self.by_name.get(&p.name) == Some(&id) {
            self.by_name.remove(&p.name);
        }
        Some(p)
    }

    #[must_use]
    pub fn get(&self, id: PromoterId) -> Option<&Promoter> {
        self.promoters.get(id)
    }

    pub fn get_mut(&mut self, id: PromoterId) -> Option<&mut Promoter> {
        self.promoters.get_mut(id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (PromoterId, &Promoter)> {
        self.promoters.iter()
    }

    /// O(log N) lookup by user-facing name. Load-bearing for the
    /// engine's hot-reload resolution shim. Config validation rejects
    /// duplicate names upstream and [`Self::insert`] `debug_assert!`s
    /// the same invariant, so the mapping is 1:1.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<PromoterId> {
        self.by_name.get(name).copied()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.promoters.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.promoters.is_empty()
    }
}

/// Hot-reload diff for the Promoter side.
///
/// Computed by the TOML loader, consumed via
/// `Input::ConfigDiff(WatchRegistryDiff)`. Mirrors
/// [`SubRegistryDiff`](crate::sub::SubRegistryDiff)'s name-keyed
/// shape: `removed` carries operator Promoter names; `modified`
/// carries the new [`PromoterAttachRequest`] (name inside). The engine
/// resolves name → [`PromoterId`] through its own `by_name` index and
/// wholesale-replaces (`reap_promoter_inner` then
/// `attach_promoter_inner`) on each `modified` entry.
#[derive(Clone, Debug, Default)]
pub struct PromoterRegistryDiff {
    pub added: Vec<PromoterAttachRequest>,
    pub removed: Vec<CompactString>,
    pub modified: Vec<PromoterAttachRequest>,
}

#[cfg(test)]
mod tests {
    use super::{
        Promoter, PromoterAttachRequest, PromoterRegistry, PromoterRegistryDiff, PromoterState,
        ProxyState,
    };
    use crate::ids::{ProbeCorrelation, PromoterId, ResourceId, SubId};
    use crate::pattern::PatternSpec;
    use crate::probe::ProbeSlot;
    use crate::profile::{DescentRemaining, DescentState};
    use crate::program::{
        ActionProgram, ArgPart, ArgTemplate, BranchTarget, ExecAction, Placeholder, ProgramBuilder,
        SpawnBody,
    };
    use crate::scan_config::{ProfileIdentity, ScanConfig};
    use crate::sub::{ClassSet, EffectScope};
    use compact_str::CompactString;
    use slotmap::SlotMap;
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    fn program() -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([
                ArgPart::literal("/bin/build"),
                ArgPart::Placeholder(Placeholder::Path),
            ])],
            None,
        )));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// Build an `Active`-state Promoter through the real
    /// [`Promoter::from_request`] constructor (not a parallel struct
    /// literal — the test path and the production path stay one).
    fn build_promoter(name: &str, pattern: &str) -> Promoter {
        let req = PromoterAttachRequest {
            name: CompactString::from(name),
            pattern_spec: PatternSpec::parse(pattern).expect("valid pattern"),
            identity: ProfileIdentity {
                config: ScanConfig::builder().recursive(true).build(),
                max_settle: MAX_SETTLE,
                events: ClassSet::DEFAULT_SUBTREE_ROOT,
            },
            settle: SETTLE,
            program: program(),
            scope: EffectScope::SubtreeRoot,
            log_output: false,
        };
        Promoter::from_request(
            req,
            PromoterState::Active {
                proxies: BTreeMap::new(),
                enumerating: ProbeSlot::empty(),
            },
        )
    }

    /// `insert` minted a key and registered the `by_name` mapping;
    /// `find_by_name` round-trips on the same name and `get` returns
    /// the stored Promoter.
    #[test]
    fn registry_insert_round_trip() {
        let mut reg = PromoterRegistry::new();
        let id = reg.insert(build_promoter("logs", "/var/log/*.log"));
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        let stored = reg.get(id).expect("Promoter stored");
        assert_eq!(stored.name, "logs");
        assert_eq!(reg.find_by_name("logs"), Some(id));
    }

    #[test]
    fn registry_remove_clears_by_name() {
        let mut reg = PromoterRegistry::new();
        let id = reg.insert(build_promoter("logs", "/var/log/*.log"));
        reg.remove(id).expect("returned the Promoter");
        assert!(reg.get(id).is_none());
        assert!(reg.find_by_name("logs").is_none());
        assert_eq!(reg.len(), 0);
    }

    #[test]
    fn registry_find_by_name_absent() {
        let reg = PromoterRegistry::new();
        assert!(reg.find_by_name("missing").is_none());
    }

    #[test]
    fn registry_iter_yields_all_promoters() {
        let mut reg = PromoterRegistry::new();
        reg.insert(build_promoter("a", "/srv/*"));
        reg.insert(build_promoter("b", "/var/*"));
        let mut names: Vec<String> = reg.iter().map(|(_, p)| p.name.to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn registry_remove_stale_id_returns_none() {
        let mut reg = PromoterRegistry::new();
        assert!(reg.remove(PromoterId::default()).is_none());
    }

    /// After a multi-insert/remove sequence, every key `iter()` yields
    /// re-looks-up via `get` and `find_by_name` round-trips. The
    /// slotmap key is the sole identity authority (a `Promoter` carries
    /// no `id`) — this replaces the removed `Promoter.id == minted key`
    /// assertion.
    #[test]
    fn registry_iter_keys_round_trip_through_get() {
        let mut reg = PromoterRegistry::new();
        let a = reg.insert(build_promoter("a", "/a/*"));
        let b = reg.insert(build_promoter("b", "/b/*"));
        let c = reg.insert(build_promoter("c", "/c/*"));
        reg.remove(b);

        let mut iter_keys: Vec<PromoterId> = reg
            .iter()
            .map(|(k, p)| {
                assert_eq!(
                    reg.get(k).expect("iter key resolves via get").name,
                    p.name,
                    "get(k) returns the same entry iter yielded",
                );
                assert_eq!(
                    reg.find_by_name(p.name.as_str()),
                    Some(k),
                    "by_name round-trips on the iterated key",
                );
                k
            })
            .collect();
        iter_keys.sort();

        let mut want = vec![a, c];
        want.sort();
        assert_eq!(iter_keys, want, "iter yields exactly the live keys");
        assert!(reg.get(b).is_none(), "removed key no longer resolves");
        assert_eq!(reg.len(), 2);
    }

    /// Diff is plain data — exercise field construction so changes to
    /// the shape break this test loudly.
    #[test]
    fn promoter_registry_diff_default_is_empty() {
        let d = PromoterRegistryDiff::default();
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified.is_empty());
    }

    #[test]
    fn promoter_registry_diff_carries_added_modified_removed() {
        let req = PromoterAttachRequest {
            name: "logs".into(),
            pattern_spec: PatternSpec::parse("/var/log/*.log").expect("valid"),
            identity: ProfileIdentity {
                config: ScanConfig::builder().recursive(true).build(),
                max_settle: MAX_SETTLE,
                events: ClassSet::DEFAULT_SUBTREE_ROOT,
            },
            settle: SETTLE,
            program: program(),
            scope: EffectScope::SubtreeRoot,
            log_output: false,
        };
        let d = PromoterRegistryDiff {
            added: vec![req.clone()],
            removed: vec!["logs".into()],
            modified: vec![req],
        };
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.removed.len(), 1);
        assert_eq!(d.modified.len(), 1);
    }

    /// Sanity-check that PrefixPending can carry a DescentState — proves
    /// the type composition compiles and accepts the intended payloads.
    #[test]
    fn promoter_state_prefix_pending_carries_descent_state() {
        let state = PromoterState::PrefixPending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![
                CompactString::from("var"),
                CompactString::from("log"),
            ])
            .expect("non-empty by test construction"),
            ProbeSlot::empty(),
        ));
        let PromoterState::PrefixPending(d) = state else {
            panic!("expected PrefixPending");
        };
        assert_eq!(d.remaining_components().len(), 2);
    }

    /// Active proxies map carries `(ResourceId, ProxyState)` entries. The
    /// `pattern_component_index` is the cursor advanced on each
    /// registration; first proxy at materialisation carries
    /// `pattern.literal_prefix_len`.
    #[test]
    fn promoter_state_active_carries_proxy_state() {
        let mut proxies: BTreeMap<ResourceId, ProxyState> = BTreeMap::new();
        proxies.insert(
            ResourceId::default(),
            ProxyState {
                pattern_component_index: 3,
            },
        );
        let state = PromoterState::Active {
            proxies,
            enumerating: ProbeSlot::empty(),
        };
        let PromoterState::Active { proxies, .. } = state else {
            panic!("expected Active");
        };
        assert_eq!(proxies.len(), 1);
    }

    /// Dynamic Sub dedup map round-trips `ResourceId → SubId` through
    /// the sealed [`Promoter::promote`] / [`Promoter::dynamic_subs`]
    /// surface (resource-keying: Tree slot identity is path-bijective
    /// for live slots — cheaper than path-keying, no per-entry string).
    #[test]
    fn promoter_dynamic_subs_round_trip() {
        let mut p = build_promoter("logs", "/var/log/*.log");
        let resource = ResourceId::default();
        let sid = SubId::default();
        p.promote(resource, sid);
        assert_eq!(p.dynamic_subs().get(&resource), Some(&sid));
    }

    /// [`Promoter::promote`] then [`Promoter::forget_dynamic_sub`]
    /// returns the mapped `SubId`; a second forget is a `None` no-op
    /// (the concurrent-drain-already-cleared edge).
    #[test]
    fn promote_then_forget_round_trips_and_is_idempotent() {
        let mut p = build_promoter("logs", "/var/log/*.log");
        let r = ResourceId::default();
        let sid = SubId::default();
        p.promote(r, sid);
        assert_eq!(p.forget_dynamic_sub(r), Some(sid));
        assert_eq!(p.forget_dynamic_sub(r), None);
        assert!(p.dynamic_subs().is_empty());
    }

    /// [`Promoter::drain_dynamic_subs`] returns every minted `SubId`
    /// and leaves the map empty so cascading detach sees no entries.
    #[test]
    fn drain_dynamic_subs_returns_all_and_clears() {
        let mut rk: SlotMap<ResourceId, ()> = SlotMap::with_key();
        let mut sk: SlotMap<SubId, ()> = SlotMap::with_key();
        let mut p = build_promoter("logs", "/var/log/*.log");
        let (r0, r1) = (rk.insert(()), rk.insert(()));
        let (s0, s1) = (sk.insert(()), sk.insert(()));
        p.promote(r0, s0);
        p.promote(r1, s1);
        let mut drained = p.drain_dynamic_subs();
        drained.sort_unstable();
        let mut want = vec![s0, s1];
        want.sort_unstable();
        assert_eq!(drained, want);
        assert!(p.dynamic_subs().is_empty());
        assert!(p.drain_dynamic_subs().is_empty(), "second drain is empty");
    }

    /// The fan-out latch is one-shot and structural: `Some(count)` on
    /// the first crossing, `None` on every later check regardless of
    /// further growth.
    #[test]
    fn latch_fanout_warning_fires_once() {
        let mut rk: SlotMap<ResourceId, ()> = SlotMap::with_key();
        let mut sk: SlotMap<SubId, ()> = SlotMap::with_key();
        let mut p = build_promoter("logs", "/var/log/*.log");
        for _ in 0..3 {
            p.promote(rk.insert(()), sk.insert(()));
        }
        assert_eq!(p.latch_fanout_warning(2), Some(3), "first crossing warns");
        assert_eq!(p.latch_fanout_warning(2), None, "latched — no repeat");
        p.promote(rk.insert(()), sk.insert(()));
        assert_eq!(
            p.latch_fanout_warning(2),
            None,
            "still latched after growth"
        );
        assert_eq!(p.latch_fanout_warning(100), None, "below threshold ⇒ None");
    }

    /// [`Promoter::enter_active_empty`] is the one-shot
    /// `PrefixPending → Active` move: the prior (disarmed) descent slot
    /// drops without tripping the [`ProbeSlot`] guard, and the result
    /// is `Active` with empty proxies + an empty enumeration slot.
    #[test]
    fn enter_active_empty_transitions_from_disarmed_prefix_pending() {
        let req = PromoterAttachRequest {
            name: "logs".into(),
            pattern_spec: PatternSpec::parse("/var/log/*.log").expect("valid"),
            identity: ProfileIdentity {
                config: ScanConfig::builder().recursive(true).build(),
                max_settle: MAX_SETTLE,
                events: ClassSet::DEFAULT_SUBTREE_ROOT,
            },
            settle: SETTLE,
            program: program(),
            scope: EffectScope::SubtreeRoot,
            log_output: false,
        };
        let mut p = Promoter::from_request(
            req,
            PromoterState::PrefixPending(DescentState::new(
                ResourceId::default(),
                DescentRemaining::from_vec(vec![CompactString::from("var")]).expect("non-empty"),
                ProbeSlot::empty(), // disarmed ⇒ Drop guard stays silent
            )),
        );
        assert!(p.state().descent_state().is_some());
        p.enter_active_empty();
        match p.state() {
            PromoterState::Active {
                proxies,
                enumerating,
            } => {
                assert!(proxies.is_empty());
                assert!(enumerating.correlation().is_none());
            }
            PromoterState::PrefixPending(_) => panic!("expected Active after transition"),
        }
    }

    /// `PromoterState::descent_state` borrows the descent in
    /// `PrefixPending`, returns `None` for `Active`.
    #[test]
    fn promoter_state_descent_state_returns_some_only_on_prefix_pending() {
        let pending = PromoterState::PrefixPending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![CompactString::from("var")]).expect("non-empty"),
            ProbeSlot::empty(),
        ));
        assert!(pending.descent_state().is_some());

        let active = PromoterState::Active {
            proxies: BTreeMap::new(),
            enumerating: ProbeSlot::empty(),
        };
        assert!(active.descent_state().is_none());
    }

    /// `descent_state_mut` lets a caller advance the descent in place
    /// when the state is `PrefixPending`.
    #[test]
    fn promoter_state_descent_state_mut_advances_pending() {
        let mut state = PromoterState::PrefixPending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![
                CompactString::from("var"),
                CompactString::from("log"),
            ])
            .expect("non-empty"),
            ProbeSlot::empty(),
        ));

        {
            let d = state
                .descent_state_mut()
                .expect("PrefixPending carries descent");
            d.remaining_components_mut().advance();
        }

        let d = state.descent_state().expect("still PrefixPending");
        assert_eq!(d.remaining_components().len(), 1);

        // Mutator returns None on Active.
        let mut active = PromoterState::Active {
            proxies: BTreeMap::new(),
            enumerating: ProbeSlot::empty(),
        };
        assert!(active.descent_state_mut().is_none());
    }

    /// `probe_correlation` projects the PrefixPending descent slot;
    /// `take_probe` consumes it once and idles it. Total over the
    /// state space — `Active` carries no descent slot.
    #[test]
    fn promoter_probe_correlation_and_take_probe_track_prefix_pending_slot() {
        let c = ProbeCorrelation::from(13);
        let mut s = PromoterState::PrefixPending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![CompactString::from("var")]).expect("non-empty"),
            ProbeSlot::armed(c, ()),
        ));
        assert_eq!(s.probe_correlation(), Some(c));
        assert_eq!(s.take_probe(), Some(c));
        assert_eq!(s.probe_correlation(), None, "slot idled after take");
        assert_eq!(s.take_probe(), None, "second take is a None no-op");

        // PrefixPending + empty ⇒ no correlation, no consume.
        let mut idle = PromoterState::PrefixPending(DescentState::new(
            ResourceId::default(),
            DescentRemaining::from_vec(vec![CompactString::from("var")]).expect("non-empty"),
            ProbeSlot::empty(),
        ));
        assert_eq!(idle.probe_correlation(), None);
        assert_eq!(idle.take_probe(), None);

        // Active holds no descent slot — total projection ⇒ None.
        let mut active = PromoterState::Active {
            proxies: BTreeMap::new(),
            enumerating: ProbeSlot::empty(),
        };
        assert_eq!(active.probe_correlation(), None);
        assert_eq!(active.take_probe(), None);
    }
}
