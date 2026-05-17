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
//! a fresh `ResourceId` and never collides with stale state. Mutators
//! are three sites: `try_promote` (insert with contains check),
//! `on_dynamic_sub_reaped` (remove on anchor-terminal), and
//! `reap_promoter_inner` (full drain on Promoter teardown).

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
#[derive(Clone, Debug)]
pub struct PromoterAttachRequest {
    pub name: String,
    pub pattern_spec: PatternSpec,
    pub identity: ProfileIdentity,
    pub settle: Duration,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub log_output: bool,
}

/// Engine-resident Promoter.
///
/// Mirrors `Profile`'s registry-stored shape. No `id` field — the
/// slotmap [`PromoterId`] is the identity authority; helper code that
/// needs the id receives it as a parameter. `identity` is the
/// Sub-spec's Profile partition key, threaded verbatim into every
/// synthesised dynamic Sub.
#[derive(Debug)]
pub struct Promoter {
    pub name: CompactString,
    pub pattern: Arc<PatternSpec>,
    pub identity: ProfileIdentity,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub settle: Duration,
    pub log_output: bool,

    pub state: PromoterState,

    /// Deterministic queue of proxies awaiting enumeration. `BTreeSet` for
    /// stable iteration; insertion is gated on `!already_carries` in
    /// `register_proxy` so re-registration of an already-known proxy is
    /// structurally idempotent on the queue and the per-Resource counter.
    pub pending_enumerations: BTreeSet<ResourceId>,

    /// `anchor_resource → SubId`. Resource identity is `(parent, segment)`
    /// — bijective with the resolved path while the slot is live; the
    /// Sub's `AnchorClaim::Held` contribution keeps the slot from
    /// reaping for the dedup entry's lifetime. Three documented
    /// mutators — `try_promote` (insert with contains check),
    /// `on_dynamic_sub_reaped` (remove on anchor-terminal), and
    /// `reap_promoter_inner` (full drain).
    pub dynamic_subs: BTreeMap<ResourceId, SubId>,

    /// Fanout-warning latch. Set on the first crossing of the threshold;
    /// suppresses repeats so pathological patterns warn once per Promoter
    /// lifetime.
    pub warned_at_threshold: bool,
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
/// `SubRegistryDiff`'s shape; `modified` carries the new spec — the
/// engine wholesale-replaces (`reap_promoter_inner` then
/// `attach_promoter_inner`) on each entry.
#[derive(Clone, Debug, Default)]
pub struct PromoterRegistryDiff {
    pub added: Vec<PromoterAttachRequest>,
    pub removed: Vec<PromoterId>,
    pub modified: Vec<(PromoterId, PromoterAttachRequest)>,
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
    use std::collections::{BTreeMap, BTreeSet};
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

    fn build_promoter(name: &str, pattern: &str) -> Promoter {
        Promoter {
            name: CompactString::from(name),
            pattern: Arc::new(PatternSpec::parse(pattern).expect("valid pattern")),
            identity: ProfileIdentity {
                config: ScanConfig::builder().recursive(true).build(),
                max_settle: MAX_SETTLE,
                events: ClassSet::DEFAULT_SUBTREE_ROOT,
            },
            program: program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            state: PromoterState::Active {
                proxies: BTreeMap::new(),
                enumerating: ProbeSlot::empty(),
            },
            pending_enumerations: BTreeSet::new(),
            dynamic_subs: BTreeMap::new(),
            warned_at_threshold: false,
        }
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
            name: "logs".to_owned(),
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
            removed: vec![PromoterId::default()],
            modified: vec![(PromoterId::default(), req)],
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

    /// Dynamic Sub dedup map round-trips `ResourceId → SubId`. The map
    /// is the unique-key store for the per-(promoter, anchor) dedup;
    /// path-keying was replaced with resource-keying because Tree slot
    /// identity already encodes a path-bijective key for live slots
    /// (cheaper to store, no path-string allocation per entry).
    #[test]
    fn promoter_dynamic_subs_round_trip() {
        let mut p = build_promoter("logs", "/var/log/*.log");
        let resource = ResourceId::default();
        let sid = SubId::default();
        p.dynamic_subs.insert(resource, sid);
        assert_eq!(p.dynamic_subs.get(&resource), Some(&sid));
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
