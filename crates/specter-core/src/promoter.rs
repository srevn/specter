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
//! At most one outstanding probe per Promoter. Concurrent enumerations
//! queue via `pending_enumerations` and drain into a probe via the
//! engine's `dispatch_next_enumeration`. The channel itself lives on
//! the engine (`engine::probe_channel::ProbeChannel`) — the Promoter
//! type holds no probe-side state.
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

use crate::ids::{PromoterId, ResourceId, SubId};
use crate::pattern::PatternSpec;
use crate::profile::DescentState;
use crate::program::ActionProgram;
use crate::scan_config::ScanConfig;
use crate::sub::{ClassSet, EffectScope};
use compact_str::CompactString;
use slotmap::SlotMap;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::Duration;

/// Pre-id spec carried on `WatchRegistryDiff::promoters.{added,modified}`.
///
/// Mirrors `SubAttachRequest`'s role for the static side: the config
/// layer materialises this from a `[[promoter]]` (or auto-detected
/// `[[watch]]`) block; the engine assigns a `PromoterId` at attach.
///
/// `Clone` is derived for the same reason as `SubAttachRequest`: rare
/// fan-out call sites that send the spec to multiple Engines.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct PromoterAttachRequest {
    pub name: String,
    pub pattern_spec: PatternSpec,
    pub config: ScanConfig,
    pub max_settle: Duration,
    pub settle: Duration,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub events: ClassSet,
    pub log_output: bool,
}

/// Engine-resident Promoter.
///
/// Mirrors `Profile`'s registry-stored shape: the `id` field is embedded
/// by `PromoterRegistry::insert`'s `insert_with_key` closure so
/// self-references inside helper code work without round-tripping
/// through the registry.
#[derive(Debug)]
pub struct Promoter {
    pub id: PromoterId,
    pub name: CompactString,
    pub pattern: Arc<PatternSpec>,
    pub config: ScanConfig,
    pub max_settle: Duration,
    pub settle: Duration,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    pub events: ClassSet,
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
/// The variant payload is the *correlation token* of the input the
/// Promoter is currently awaiting. `PrefixPending` carries the
/// `DescentState`; `Active` carries the proxy fan-out.
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
    Active {
        proxies: BTreeMap<ResourceId, ProxyState>,
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

/// Slotmap-backed registry. Mirrors `ProfileMap`'s shape: a `SlotMap`
/// keyed on `PromoterId` plus a `BTreeMap<name, PromoterId>` for
/// configuration-driven lookup at hot-reload time.
///
/// `by_name` mirrors the lifetime of the slotmap entry — `insert`
/// populates both; `remove` clears both. Lookup is O(log N) on names.
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

    /// Insert a Promoter built from the freshly-minted `PromoterId`. The
    /// closure receives the minted key so the Promoter embeds its own
    /// id; mirrors [`SubRegistry::insert`](crate::sub::SubRegistry::insert).
    pub fn insert<F>(&mut self, build: F) -> PromoterId
    where
        F: FnOnce(PromoterId) -> Promoter,
    {
        let id = self.promoters.insert_with_key(build);
        let name = self.promoters[id].name.clone();
        self.by_name.insert(name, id);
        id
    }

    /// Remove a Promoter, returning the owned value. Clears `by_name` in
    /// lockstep. Returns `None` for a stale id.
    pub fn remove(&mut self, id: PromoterId) -> Option<Promoter> {
        let p = self.promoters.remove(id)?;
        self.by_name.remove(&p.name);
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

    /// O(log N) lookup by user-facing name. Uniqueness is the config
    /// layer's responsibility (the validator rejects duplicate names
    /// upstream); `insert` overwrites a same-name entry without
    /// notification.
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
    use crate::ids::{PromoterId, ResourceId, SubId};
    use crate::pattern::PatternSpec;
    use crate::profile::{DescentRemaining, DescentState};
    use crate::program::{
        ActionProgram, ArgPart, ArgTemplate, BranchTarget, ExecAction, Placeholder, ProgramBuilder,
        SpawnBody,
    };
    use crate::scan_config::ScanConfig;
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

    fn build_promoter(id: PromoterId, name: &str, pattern: &str) -> Promoter {
        Promoter {
            id,
            name: CompactString::from(name),
            pattern: Arc::new(PatternSpec::parse(pattern).expect("valid pattern")),
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            settle: SETTLE,
            program: program(),
            scope: EffectScope::SubtreeRoot,
            events: ClassSet::DEFAULT_SUBTREE_ROOT,
            log_output: false,
            state: PromoterState::Active {
                proxies: BTreeMap::new(),
            },
            pending_enumerations: BTreeSet::new(),
            dynamic_subs: BTreeMap::new(),
            warned_at_threshold: false,
        }
    }

    /// `insert` minted a key, embedded it into the value, and registered
    /// the `by_name` mapping. `find_by_name` round-trips on the same
    /// name. `get` returns the embedded Promoter.
    #[test]
    fn registry_insert_round_trip() {
        let mut reg = PromoterRegistry::new();
        let id = reg.insert(|id| build_promoter(id, "logs", "/var/log/*.log"));
        assert_eq!(reg.len(), 1);
        assert!(!reg.is_empty());
        let stored = reg.get(id).expect("Promoter stored");
        assert_eq!(stored.id, id, "embedded id matches minted key");
        assert_eq!(stored.name, "logs");
        assert_eq!(reg.find_by_name("logs"), Some(id));
    }

    #[test]
    fn registry_remove_clears_by_name() {
        let mut reg = PromoterRegistry::new();
        let id = reg.insert(|id| build_promoter(id, "logs", "/var/log/*.log"));
        let removed = reg.remove(id).expect("returned the Promoter");
        assert_eq!(removed.id, id);
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
        reg.insert(|id| build_promoter(id, "a", "/srv/*"));
        reg.insert(|id| build_promoter(id, "b", "/var/*"));
        let mut names: Vec<String> = reg.iter().map(|(_, p)| p.name.to_string()).collect();
        names.sort();
        assert_eq!(names, vec!["a", "b"]);
    }

    #[test]
    fn registry_remove_stale_id_returns_none() {
        let mut reg = PromoterRegistry::new();
        assert!(reg.remove(PromoterId::default()).is_none());
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
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            settle: SETTLE,
            program: program(),
            scope: EffectScope::SubtreeRoot,
            events: ClassSet::DEFAULT_SUBTREE_ROOT,
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
        let state = PromoterState::Active { proxies };
        let PromoterState::Active { proxies } = state else {
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
        let mut p = build_promoter(PromoterId::default(), "logs", "/var/log/*.log");
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
        ));
        assert!(pending.descent_state().is_some());

        let active = PromoterState::Active {
            proxies: BTreeMap::new(),
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
        };
        assert!(active.descent_state_mut().is_none());
    }
}
