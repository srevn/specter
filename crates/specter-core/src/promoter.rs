//! Promoter — engine-resident dynamic-watch source.
//!
//! A `Promoter` is a peer to `Profile` in the engine: each one carries a
//! `PatternSpec`, a literal-prefix probe state, an `Active` proxy fan-out
//! over matched directories, and a deduplicated map of synthesised
//! dynamic Subs. The engine drives the lifecycle through a state machine
//! (Phase 5+); `core::promoter` owns the data shapes and the registry.
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
//! `Promoter.pending_probe: Option<ProbeCorrelation>` mirrors
//! `Profile.pending_probe`: at most one outstanding probe per Promoter.
//! Concurrent enumerations queue via `pending_enumerations`. The slot
//! discipline is owned by `engine::probe_channel` (Phase 2 already wired
//! `mint_owner_correlation` / `cancel_owner_probe` against `ProbeOwner`).
//!
//! ## Dynamic Sub deduplication
//!
//! `dynamic_subs: BTreeMap<PathBuf, SubId>` enforces I-Promoter-5: at most
//! one dynamic Sub per `(promoter_id, resolved_path)`. Mutators are the
//! three sites of I-Promoter-4: `try_promote` (insert with contains
//! check), `on_dynamic_sub_reaped` (remove on anchor-terminal), and
//! `reap_promoter_inner` (full drain on Promoter teardown).

use crate::ids::{PromoterId, ResourceId, SubId};
use crate::op::ProbeCorrelation;
use crate::pattern::PatternSpec;
use crate::profile::DescentState;
use crate::scan_config::ScanConfig;
use crate::sub::{ClassSet, CommandTemplate, EffectScope};
use compact_str::CompactString;
use slotmap::SlotMap;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
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
    pub command: CommandTemplate,
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
    pub pattern: PatternSpec,
    pub config: ScanConfig,
    pub max_settle: Duration,
    pub settle: Duration,
    pub command: CommandTemplate,
    pub scope: EffectScope,
    pub events: ClassSet,
    pub log_output: bool,

    pub state: PromoterState,

    /// I-Promoter-1: at most one outstanding probe. Slot discipline is
    /// owned by `engine::probe_channel`'s
    /// `mint_owner_correlation` / `cancel_owner_probe` against
    /// `ProbeOwner::Promoter(_)`.
    pub pending_probe: Option<ProbeCorrelation>,

    /// Deterministic queue of proxies awaiting enumeration. `BTreeSet` for
    /// stable iteration; insertion is gated on `!already_carries` in
    /// `register_proxy` (Phase 5+) so re-registration of an already-known
    /// proxy is structurally idempotent on the queue and the
    /// per-Resource counter.
    pub pending_enumerations: BTreeSet<ResourceId>,

    /// `(resolved_path) → SubId`. I-Promoter-4 / I-Promoter-5: three
    /// documented mutators — `try_promote` (insert with contains check),
    /// `on_dynamic_sub_reaped` (remove on anchor-terminal), and
    /// `reap_promoter_inner` (full drain).
    pub dynamic_subs: BTreeMap<PathBuf, SubId>,

    /// Fanout-warning latch. Set on the first crossing of the threshold
    /// (Phase 6+ defines the threshold value); suppresses repeats so
    /// pathological patterns warn once per Promoter lifetime.
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
/// Computed by the TOML loader (Phase 10), consumed via
/// `Input::ConfigDiff(WatchRegistryDiff)` (Phase 11). Mirrors
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
    use crate::profile::DescentState;
    use crate::scan_config::ScanConfig;
    use crate::sub::{ArgPart, ArgTemplate, ClassSet, CommandTemplate, EffectScope, Placeholder};
    use compact_str::CompactString;
    use std::collections::{BTreeMap, BTreeSet};
    use std::path::PathBuf;
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    fn cmd() -> CommandTemplate {
        CommandTemplate::new([ArgTemplate::new([
            ArgPart::literal("/bin/build"),
            ArgPart::Placeholder(Placeholder::Path),
        ])])
    }

    fn build_promoter(id: PromoterId, name: &str, pattern: &str) -> Promoter {
        Promoter {
            id,
            name: CompactString::from(name),
            pattern: PatternSpec::parse(pattern).expect("valid pattern"),
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            settle: SETTLE,
            command: cmd(),
            scope: EffectScope::SubtreeRoot,
            events: ClassSet::DEFAULT_SUBTREE_ROOT,
            log_output: false,
            state: PromoterState::Active {
                proxies: BTreeMap::new(),
            },
            pending_probe: None,
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
            command: cmd(),
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
        let state = PromoterState::PrefixPending(DescentState {
            current_prefix: ResourceId::default(),
            remaining_components: vec![CompactString::from("var"), CompactString::from("log")],
        });
        let PromoterState::PrefixPending(d) = state else {
            panic!("expected PrefixPending");
        };
        assert_eq!(d.remaining_components.len(), 2);
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

    /// Dynamic Sub dedup map round-trips path → SubId. The map is the
    /// I-Promoter-5 unique-key store.
    #[test]
    fn promoter_dynamic_subs_round_trip() {
        let mut p = build_promoter(PromoterId::default(), "logs", "/var/log/*.log");
        let path = PathBuf::from("/var/log/foo.log");
        let sid = SubId::default();
        p.dynamic_subs.insert(path.clone(), sid);
        assert_eq!(p.dynamic_subs.get(&path), Some(&sid));
    }
}
