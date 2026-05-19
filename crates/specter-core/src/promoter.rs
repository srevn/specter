//! Promoter — engine-resident dynamic-watch source.
//!
//! A `Promoter` is a peer to `Profile` in the engine: each one carries a
//! `PatternSpec`, a literal-prefix probe state, and an `Active` proxy
//! fan-out over matched directories. The dynamic Subs it synthesises are
//! owned solely by `SubRegistry` (tagged `source_promoter`); the Promoter
//! keeps no mirror. The engine drives the lifecycle through a state
//! machine; `core::promoter` owns the data shapes and the registry.
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
//! The two states are mutually exclusive (Rust sum-type).
//! `PrefixPending → Active` materialises the prefix
//! ([`Promoter::enter_active_empty`]); the inverse
//! `Active → PrefixPending` ([`Promoter::reenter_prefix_pending`]) is
//! the terminus-loss recovery move. A Promoter whose terminus is
//! `rm -rf`d collapses to `Active { proxies: ∅ }`; the preserved
//! parent-edge watch ([`Promoter::prefix_parent`]) re-enters descent
//! on the parent's next structural event. The transition is therefore
//! bidirectional — the structural mirror of `Profile`'s
//! `Pending ↔ Idle` anchor-loss recovery.
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

use crate::ids::{ProbeCorrelation, PromoterId, ResourceId};
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
/// source per transition"): the bidirectional `PrefixPending` ↔
/// `Active` moves ([`Self::enter_active_empty`] forward,
/// [`Self::reenter_prefix_pending`] the terminus-loss recovery
/// inverse); the in-`state` linear [`ProbeSlot`]
/// (armed only via [`Self::arm_enumeration`], consumed only via
/// [`Self::take_probe`]); the enumeration queue
/// ([`Self::enqueue_enumeration`] / [`Self::pop_enumeration`] /
/// [`Self::unregister_proxy_slot`]); the one-shot fan-out latch
/// ([`Self::latch_fanout_warning`], pre-gated by
/// [`Self::fanout_warned`]); and the parent-edge recovery channel
/// ([`Self::set_prefix_parent`] / [`Self::take_prefix_parent`],
/// projected by [`Self::prefix_parent`]). Reads project through
/// [`Self::state`] /
/// [`Self::pending_enumerations`]. The dynamic Subs this Promoter
/// synthesises are owned solely by `SubRegistry` (tagged
/// `source_promoter`) — there is no Promoter-side mirror to keep
/// coherent, so the dedup gate is a live registry query.
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
    /// [`Self::state`]; the transitions are [`Self::enter_active_empty`]
    /// (`PrefixPending → Active`) and its terminus-loss recovery inverse
    /// [`Self::reenter_prefix_pending`] (`Active → PrefixPending`).
    state: PromoterState,

    /// Deterministic queue of proxies awaiting enumeration. `BTreeSet`
    /// for stable iteration; the single-slot drain pops one at a time.
    /// Private — mutate via [`Self::enqueue_enumeration`] /
    /// [`Self::pop_enumeration`] / [`Self::unregister_proxy_slot`].
    pending_enumerations: BTreeSet<ResourceId>,

    /// One-shot fan-out warning latch. Fully private — its only
    /// interaction is the atomic check-and-latch in
    /// [`Self::latch_fanout_warning`], so a pathological pattern warns
    /// once per Promoter lifetime by construction. [`Self::fanout_warned`]
    /// projects it read-only so the engine can skip the registry count
    /// scan once latched.
    warned_at_threshold: bool,

    /// Parent-edge recovery channel. `Some(parent)` ⇒ this Promoter
    /// contributes a [`crate::ContribKey::PromoterPrefixParent`]
    /// `STRUCTURE` watch at `parent` — the terminus's parent slot,
    /// installed when the literal prefix materialises and preserved
    /// across terminus loss (the downward-only proxy unregister cannot
    /// reach an ancestor). It is the sole recovery channel for an
    /// `Active { proxies: ∅ }` Promoter and is released only at reap or
    /// FD-exhaustion purge of the parent slot. The structural mirror of
    /// `Profile.watch_root_parent`. Private — written only via the
    /// sealed [`Self::set_prefix_parent`] / [`Self::take_prefix_parent`]
    /// pair, projected read-only by [`Self::prefix_parent`].
    prefix_parent: Option<ResourceId>,
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
            warned_at_threshold: false,
            prefix_parent: None,
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

    /// Whether this Promoter can possibly *carry* an `FsEvent` dispatch
    /// responsibility — the membership predicate of
    /// [`PromoterRegistry`]'s `nonsteady` carrier count, the structural
    /// twin of [`Profile::is_nonsteady`](crate::Profile::is_nonsteady).
    ///
    /// A carrier is either a `PrefixPending` descent (`current_prefix
    /// == R`) or a terminus-loss-recovery `Active { proxies: ∅ }`
    /// (`prefix_parent == Some(R)`). This is the tight state-shape set:
    /// a healthy `Active { proxies: ≠∅ }` Promoter — prefix
    /// materialised, terminus live — is **excluded**, so it never pins
    /// the count above zero during a storm. The proxy-emptiness edge is
    /// the one multi-field surface; it is reconciled at every mutator
    /// through [`PromoterRegistry::mutate`].
    #[must_use]
    pub fn is_nonsteady(&self) -> bool {
        match &self.state {
            PromoterState::PrefixPending(_) => true,
            PromoterState::Active { proxies, .. } => proxies.is_empty(),
        }
    }

    /// The forward `PrefixPending → Active` transition (the prefix
    /// materialised, or its claim is being released). Replaces `state`
    /// with `Active { proxies: ∅, enumerating: ∅ }`; the prior state is
    /// dropped here. No longer once-per-lifetime: terminus loss can
    /// drive the inverse [`Self::reenter_prefix_pending`], so a Promoter
    /// may cycle `PrefixPending → Active → PrefixPending → Active …`
    /// across recoveries.
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

    /// The inverse `Active { proxies: ∅, enumerating: ∅ } →
    /// PrefixPending(descent)` move — the terminus-loss recovery
    /// transition, dual of [`Self::enter_active_empty`]. Replaces
    /// `state` with `PrefixPending(descent)`; the prior `Active` is
    /// dropped here.
    ///
    /// **Cancel-first contract (structural).** The discarded `Active`
    /// carries the `enumerating` [`ProbeSlot`]; an armed slot reaching
    /// drop trips its `Drop` tripwire (same enforcement as
    /// [`Self::enter_active_empty`]). It is structurally empty at the
    /// sole caller (`start_promoter_prefix_recovery`, gated by
    /// `classify_event_carriers` on `Active && proxies.is_empty()`):
    /// every path that empties `proxies` while `Active` disarms the
    /// enumeration slot first — the response's consume-once disarm for
    /// the `Vanished` / parent-reverse-pass cascade, `cancel_owner_probe`
    /// for the FD-exhaustion proxy purge — no surviving proxy can
    /// re-arm, and recovery fires in a *later* step, never synchronously
    /// with an in-flight enumeration. The discard *is* the enforcement;
    /// no scattered cancel-first asserts.
    pub fn reenter_prefix_pending(&mut self, descent: DescentState) {
        self.state = PromoterState::PrefixPending(descent);
    }

    /// The cached parent-edge recovery slot, if this Promoter owes a
    /// [`crate::ContribKey::PromoterPrefixParent`] `STRUCTURE`
    /// contribution there. `None` for a root-prefix Promoter
    /// (`terminus == "/"`, no parent) and before the prefix first
    /// materialises. Read seam over the private field;
    /// `Engine::set_promoter_prefix_parent` uses it for the
    /// cache-coherence and idempotence checks,
    /// `classify_event_carriers` for the recovery discriminant. The
    /// structural mirror of
    /// [`Profile::watch_root_parent`](crate::Profile::watch_root_parent).
    #[must_use]
    pub const fn prefix_parent(&self) -> Option<ResourceId> {
        self.prefix_parent
    }

    /// Cache the parent-edge recovery slot. The single write seam,
    /// wrapped by `Engine::set_promoter_prefix_parent` (which also
    /// installs the Tree-side `add_watch` and the cache-coherence
    /// `debug_assert!`). Plain set — idempotence and coherence are the
    /// engine wrapper's concern, not duplicated here. Mirror of
    /// [`Profile::set_watch_root_parent`](crate::Profile::set_watch_root_parent).
    pub const fn set_prefix_parent(&mut self, parent: ResourceId) {
        self.prefix_parent = Some(parent);
    }

    /// Take the cached parent-edge slot, clearing it — the symmetric
    /// deferred-release primitive (`Engine::release_promoter_prefix_parent_claim`
    /// keys the `sub_watch` removal off the returned id). Idempotent: a
    /// second call returns `None`, so a double release cannot
    /// double-remove the contribution. Mirror of
    /// [`Profile::take_watch_root_parent`](crate::Profile::take_watch_root_parent).
    #[must_use]
    pub const fn take_prefix_parent(&mut self) -> Option<ResourceId> {
        self.prefix_parent.take()
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
    /// caller) — but the cursor is invariant for a proxy's lifetime: it
    /// indexes a fixed position in the Promoter's `pattern`, so a
    /// divergent re-insert is a caller bug, caught by the
    /// `debug_assert_eq!` below rather than silently overwritten.
    /// `unreachable!()` in `PrefixPending` — the sole callers
    /// (`enter_active`, the enumeration forward pass) guarantee
    /// `Active` by construction; a wrong call surfaces loudly in every
    /// build rather than silently no-op.
    pub fn insert_proxy(&mut self, resource: ResourceId, pattern_component_index: usize) {
        match &mut self.state {
            PromoterState::Active { proxies, .. } => {
                if let Some(stored) = proxies.get(&resource) {
                    debug_assert_eq!(
                        stored.pattern_component_index, pattern_component_index,
                        "insert_proxy: pattern_component_index must be invariant \
                         for an existing proxy's lifetime (resource = {resource:?})",
                    );
                }
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

    /// Whether the one-shot fan-out warning has already latched.
    /// Read-only projection of `warned_at_threshold` — the engine's
    /// cheap pre-gate so an already-warned (pathological) Promoter
    /// never re-runs the registry count scan
    /// [`Self::latch_fanout_warning`] consumes. Additive to, not a
    /// replacement for, that method's own `warned` short-circuit.
    #[must_use]
    pub const fn fanout_warned(&self) -> bool {
        self.warned_at_threshold
    }

    /// One-shot fan-out warning latch. `count` is the caller's *live*
    /// dynamic-Sub tally for this Promoter, derived from `SubRegistry`
    /// truth (the dedup map this latch once read was deleted — the
    /// Promoter keeps no mirror). Returns `Some(count)` the first time
    /// `count` exceeds `threshold` and latches so later crossings
    /// return `None` — a pathological pattern warns once per Promoter
    /// lifetime. The check-and-latch is atomic here, so the one-shot
    /// property is structural rather than a caller convention;
    /// [`Self::fanout_warned`] lets the caller skip computing `count`
    /// once latched.
    pub fn latch_fanout_warning(&mut self, threshold: usize, count: usize) -> Option<usize> {
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
    /// (`DirEnumerated` / `Vanished` / `Failed`). Empty while the Promoter
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
    /// Live count of Promoters satisfying [`Promoter::is_nonsteady`] —
    /// the promoter half of the engine's O(1) carrier gate. The
    /// Promoter has no single state chokepoint (state and proxy
    /// emptiness are distinct mutators), so unlike
    /// [`ProfileMap::transition_state`] every membership-changing edge
    /// routes through one generic reconcile point,
    /// [`Self::mutate`], plus [`Self::insert`] / [`Self::remove`].
    nonsteady: usize,
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
        // Derived from the actual birth state: a fresh Promoter is
        // born `PrefixPending` (glob prefix absent) or `Active {
        // proxies: ∅ }` (prefix present, no proxies yet) — both
        // nonsteady — but reading the predicate keeps the count exact
        // regardless of the construction path.
        let born_nonsteady = promoter.is_nonsteady();
        let id = self.promoters.insert(promoter);
        if born_nonsteady {
            self.nonsteady += 1;
        }
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
        if p.is_nonsteady() {
            self.nonsteady = self.nonsteady.saturating_sub(1);
        }
        if self.by_name.get(&p.name) == Some(&id) {
            self.by_name.remove(&p.name);
        }
        Some(p)
    }

    /// Live carrier-eligibility count — the promoter half of the O(1)
    /// gate `Engine::classify_event_carriers` consults before its O(Q)
    /// scan. `0` ⟺ every Promoter is a healthy `Active { proxies: ≠∅ }`
    /// (prefix materialised, terminus live), so no Promoter descent /
    /// recovery carrier exists.
    #[must_use]
    pub fn nonsteady(&self) -> usize {
        self.nonsteady
    }

    /// The sole counter-reconciling path for a Promoter mutation:
    /// resolve the id, read [`Promoter::is_nonsteady`] before and
    /// after `f`, and reconcile [`Self::nonsteady`] across the edge.
    /// `f` is the specific state / proxy mutation
    /// ([`Promoter::enter_active_empty`],
    /// [`Promoter::reenter_prefix_pending`],
    /// [`Promoter::insert_proxy`], [`Promoter::unregister_proxy_slot`],
    /// or any composite that also touches membership-invariant fields
    /// like the enumeration queue). Membership-invariant `&mut`
    /// accesses (probe arming, descent advance, `prefix_parent`)
    /// legitimately keep using [`Self::get_mut`]; the debug full-scan
    /// tripwire in `Engine::classify_event_carriers` is the net for a
    /// missed route.
    ///
    /// Returns `None` for a stale id without invoking `f` — so a
    /// construct-armed `f` (a fresh [`crate::ProbeSlot`]) is never
    /// built for a vanished Promoter.
    pub fn mutate<R>(&mut self, id: PromoterId, f: impl FnOnce(&mut Promoter) -> R) -> Option<R> {
        let q = self.promoters.get_mut(id)?;
        let before = q.is_nonsteady();
        let r = f(q);
        let after = q.is_nonsteady();
        match (before, after) {
            (false, true) => self.nonsteady += 1,
            (true, false) => self.nonsteady = self.nonsteady.saturating_sub(1),
            (false, false) | (true, true) => {}
        }
        Some(r)
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
    use crate::ids::{ProbeCorrelation, PromoterId, ResourceId};
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

    /// The fan-out latch is one-shot and structural: `Some(count)` on
    /// the first crossing, `None` on every later check regardless of
    /// further growth. `count` is now a caller-supplied parameter (the
    /// dedup map it once read was deleted); [`Promoter::fanout_warned`]
    /// projects the latch read-only and flips exactly at the crossing.
    #[test]
    fn latch_fanout_warning_fires_once() {
        let mut p = build_promoter("logs", "/var/log/*.log");
        assert!(!p.fanout_warned(), "not warned before any crossing");
        assert_eq!(p.latch_fanout_warning(2, 1), None, "below threshold ⇒ None");
        assert!(!p.fanout_warned(), "still not warned below threshold");
        assert_eq!(
            p.latch_fanout_warning(2, 3),
            Some(3),
            "first crossing warns with the supplied count"
        );
        assert!(p.fanout_warned(), "latched at the crossing");
        assert_eq!(
            p.latch_fanout_warning(2, 4),
            None,
            "latched — no repeat even as the count grows"
        );
        assert_eq!(
            p.latch_fanout_warning(200, 1),
            None,
            "still latched, and below threshold anyway"
        );
    }

    /// `insert_proxy`'s cursor is invariant for a proxy's lifetime — it
    /// indexes a fixed position in the Promoter's `pattern`. A re-insert
    /// at a *divergent* `pattern_component_index` is a caller bug,
    /// caught by the `debug_assert_eq!` (debug builds) rather than
    /// silently overwritten (F-MED-5).
    #[test]
    #[should_panic(expected = "pattern_component_index must be invariant")]
    fn insert_proxy_divergent_index_trips_debug_assert() {
        let mut p = build_promoter("logs", "/var/log/*.log");
        let r = ResourceId::default();
        p.insert_proxy(r, 3);
        // Same proxy resource, different cursor — structurally impossible
        // under the real callers; the assert makes it loud.
        p.insert_proxy(r, 4);
    }

    /// Re-inserting at the *same* cursor is the sanctioned idempotent
    /// path (the [H-5] re-registration gate) — no panic, value stable.
    #[test]
    fn insert_proxy_same_index_is_idempotent() {
        let mut p = build_promoter("logs", "/var/log/*.log");
        let r = ResourceId::default();
        p.insert_proxy(r, 3);
        p.insert_proxy(r, 3);
        let PromoterState::Active { proxies, .. } = p.state() else {
            panic!("build_promoter yields Active");
        };
        assert_eq!(proxies.get(&r).map(|s| s.pattern_component_index), Some(3));
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
