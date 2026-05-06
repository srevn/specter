//! `Profile`, `ProfileMap`, and burst types.
//!
//! `Profile.config_hash` is computed at construction from
//! `(config, max_settle)` and is the lifetime-stable identity of the Profile.
//! `ProfileMap` keeps `(resource, config_hash) → ProfileId` and updates
//! `Resource.profiles` in lockstep — `attach`/`detach` are the only mutators
//! of either index.

use crate::effect::DedupKey;
use crate::ids::{ProfileId, ResourceId, TimerId};
use crate::op::ProbeCorrelation;
use crate::scan_config::{ScanConfig, compute_config_hash};
use crate::snapshot::tree::TreeSnapshot;
use crate::sub::ClassSet;
use crate::tree::Tree;
use slotmap::{SecondaryMap, SlotMap};
use std::collections::{BTreeMap, BTreeSet};
use std::time::Duration;
use tinyvec::TinyVec;

/// One fire cycle.
///
/// A `Burst` lives `Idle → Active(Burst) → Idle`; its `phase` walks
/// `Batching → Verifying [→ Draining → Verifying] → Awaiting → Rebasing`
/// and its `intent` (`Standard | Seed`) decides the terminal action.
///
/// Burst-level state (`intent`, `forced`, `burst_deadline`) survives every
/// phase transition; `phase` carries the **correlation token of the input
/// the burst is currently waiting on**:
/// - `Batching { settle_timer }` — armed debounce timer; the burst is
///   waiting for a quiet gap (or a fresh `FsEvent` to extend it).
/// - `Verifying` — probe in flight. The probe correlation lives on
///   [`Profile::pending_probe`] (the per-Profile probe-channel slot, the
///   single source of truth for "what probe?"); this variant carries no
///   payload of its own.
/// - `Draining` — self-stable, descendant Profiles still resolving;
///   correlated externally by `Profile.dirty_descendants`.
/// - `Awaiting { outstanding, gate_deadline }` — Effects emitted; the
///   engine is waiting for `outstanding` `EffectComplete` arrivals from
///   the actuator. `gate_deadline` is a recovery timer for actuator
///   hangs. Reaching `outstanding == 0` transitions to `Rebasing`.
/// - `Rebasing` — post-fire probe in flight at the anchor. The probe's
///   response captures the post-command tree as the new baseline; the
///   correlation slot is the same `Profile::pending_probe` reused
///   (Verifying and Rebasing are time-disjoint within one burst).
///
/// `dirty_resources` and `force_walk_resources` are accumulators consumed
/// at every `transition_to_verifying`; `probe_target` survives Verifying
/// → Draining → Verifying so the reconfirm probe reuses the original LCA.
#[derive(Debug)]
pub struct Burst {
    pub burst_deadline: TimerId,
    pub phase: BurstPhase,
    pub intent: BurstIntent,
    pub forced: bool,
    /// Resources whose `FsEvent` drove (or is driving) this burst.
    /// Populated cumulatively across the whole burst lifecycle:
    /// • `start_standard_burst` initialises with `{ event_resource }`.
    /// • Every `on_fs_event` during `Active` adds `event_resource`.
    /// Cleared when the `Burst` is dropped (`finish_burst_to_idle`).
    /// Used to compute the LCA target at every `transition_to_verifying`
    /// and as the closure source for `force_walk_resources`.
    pub dirty_resources: BTreeSet<ResourceId>,
    /// Since-last-probe cut of `dirty_resources`. The walker uses this to
    /// refuse mtime-skip on event-dirty paths, closing the coarse-mtime
    /// hole. Same accumulation rule as `dirty_resources`, but cleared at
    /// every `transition_to_verifying` (the engine ships its current
    /// contents as `force_walk` to the walker, then resets).
    pub force_walk_resources: BTreeSet<ResourceId>,
    /// `target_resource` of the most recently emitted probe in this burst.
    /// Mirrors the latest `ProbeRequest.target_resource`. Read by the
    /// Draining→Verifying reconfirm path (`dirty_resources` is empty
    /// there, so LCA would degenerate to the anchor — reuse the prior
    /// target instead) and by `dispatch_standard_ok` to know which
    /// subtree of `Profile.current` to compare against `response_subtree`
    /// for the stability verdict. `None` until the first probe emits.
    pub probe_target: Option<ResourceId>,
}

/// What the burst is waiting on, as a discriminator.
///
/// `Batching` carries its own correlation token (`settle_timer: TimerId`)
/// because timer correlation is per-Burst and has no peer slot to live on.
/// `Verifying` is unit: the probe correlation lives on
/// [`Profile::pending_probe`] — the per-Profile probe-channel slot — so the
/// burst phase only encodes "probe in flight" as state-machine identity.
/// `Draining` is correlated externally by `Profile.dirty_descendants` and
/// carries no token of its own.
///
/// `Awaiting { outstanding, gate_deadline }` is the post-fire phase: the
/// engine has emitted Effects to the actuator and is waiting for their
/// completions to drive `outstanding → 0`. `gate_deadline` is the
/// safety-net `AwaitGateDeadline` timer — a hung child is recovered by
/// force-transitioning to `Rebasing` once the timer expires. `Rebasing`
/// is unit (post-fire probe in flight; correlation lives on
/// [`Profile::pending_probe`], same slot Verifying used — they are
/// time-disjoint within one burst by construction).
#[derive(Debug)]
pub enum BurstPhase {
    /// Activity-gap detection. `settle_timer` is the armed debounce
    /// timer; an `FsEvent` reschedules it (`event_drives_batching`),
    /// timer expiry advances to `Verifying` (`transition_to_verifying`).
    Batching { settle_timer: TimerId },
    /// Probe in flight. The matching `ProbeCorrelation` lives on
    /// [`Profile::pending_probe`]; this variant is unit because the
    /// Profile-side slot is the single source of truth (encoding the
    /// correlation twice would invite drift).
    Verifying,
    /// Self-stable; descendants pending. The stable snapshot lives on
    /// `Profile.current` — `dispatch_standard_ok` updates `current` to
    /// the stable response immediately before transitioning here, so the
    /// reconfirm probe (Draining → Verifying on `dirty_descendants → 0`)
    /// compares against `Profile.current`. Holding a duplicate
    /// `TreeSnapshot` on the variant would only invite drift between the
    /// two references.
    Draining,
    /// Effects emitted; awaiting completion(s) from the actuator.
    /// `outstanding` decrements on each `EffectComplete` for this
    /// Profile's `DedupKey`s; reaching zero transitions to `Rebasing`
    /// (or, when `Profile.reap_pending` is set, finishes the burst
    /// directly without re-probing). `gate_deadline` is the recovery
    /// timer for an actuator that never reports completion — its
    /// expiry forces the burst into `Rebasing` so the engine can
    /// re-establish a baseline against disk reality.
    Awaiting {
        outstanding: u32,
        gate_deadline: TimerId,
    },
    /// Post-fire probe in flight. Correlation lives on
    /// [`Profile::pending_probe`] (same slot Verifying uses — Verifying
    /// and Rebasing are time-disjoint within one burst). The probe's
    /// `Ok` response captures the post-command tree; `dispatch_rebase_ok`
    /// then sets `baseline := current` and finishes the burst to Idle.
    Rebasing,
}

/// Profile state machine.
///
/// Three lifecycle states, mutually exclusive by construction:
/// - `Idle`: no probe in flight, no burst, no descent. Reads/writes baseline
///   and current as-is.
/// - `Pending(DescentState)`: anchor doesn't yet exist on disk; the engine
///   is probing the deepest existing prefix and advancing one path
///   component per response. The anchor's `Profile.resource` slot is
///   `DescentScaffold`-roled and carries no `watch_demand` from this
///   Profile (the prefix carries the `+1`). See `DescentState` invariants.
/// - `Active(Burst)`: anchor is materialized; a stability burst is in
///   flight.
///
/// I5 (at most one outstanding probe per Profile) is enforced as a
/// **field discipline** on [`Profile::pending_probe`]: that slot holds the
/// correlation of the in-flight probe, regardless of which lifecycle state
/// drives it. Pending and Active remain mutually exclusive at the type
/// level, so the dispatch site routes a live response on state identity
/// alone (see [`crate::Engine::on_probe_response`]).
#[derive(Debug, Default)]
pub enum ProfileState {
    #[default]
    Idle,
    /// Pending-path descent in flight. The anchor (`Profile.resource`) is
    /// `DescentScaffold`-roled and carries no `watch_demand` from this
    /// Profile; `DescentState.current_prefix` does. When the anchor
    /// materializes (descent's last component arrives) the engine
    /// transitions Pending → Idle (releasing the prefix's contribution and
    /// bumping the anchor's), then immediately Idle → Active(Seed) via
    /// `start_seed_burst`.
    Pending(DescentState),
    Active(Burst),
}

/// State for a Profile undergoing pending-path descent.
///
/// Lives inline on `ProfileState::Pending` for the duration of descent.
///
/// Invariants:
/// - `current_prefix` carries a `+1` `watch_demand` contribution from this
///   Profile (added at descent registration / advancement; dropped at
///   descent end or rewind).
/// - `remaining_components` is non-empty (the anchor itself is the last
///   component). Empty `remaining_components` is a state-machine bug; the
///   defensive check in the descent dispatch transitions the Profile back
///   to `Idle`.
///
/// I5 ("at most one outstanding probe per Profile") for the Pending
/// lifecycle is enforced by the per-Profile probe channel slot
/// ([`Profile::pending_probe`]) — the same slot used for Active bursts.
/// The descent's variant payload holds no probe-correlation data of its
/// own.
#[derive(Clone, Debug)]
pub struct DescentState {
    /// Deepest existing ancestor currently Watched. The Profile
    /// contributes `+1` to this Resource's `watch_demand`.
    pub current_prefix: ResourceId,
    /// Path components from `current_prefix` (exclusive) down to the
    /// anchor (inclusive). Single-component segments (no `/`).
    pub remaining_components: Vec<String>,
}

/// `Standard` — event-driven burst; preserves baseline; fires Effect on stable.
/// `Seed` — fresh Profile or post-Effect rebase; sets baseline; no Effect.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum BurstIntent {
    #[default]
    Standard,
    Seed,
}

/// Discriminator for a scheduled timer's role within a Burst's lifecycle.
///
/// `Settle` — debounce timer armed during [`BurstPhase::Batching`]. Expiry
/// drives Batching → Verifying.
/// `BurstDeadline` — Burst-level max-settle timer armed at Burst start.
/// Expiry sets `Burst.forced = true` and dispatches by current phase. The
/// timer is structurally relevant only in pre-fire phases (`Batching`,
/// `Verifying`, `Draining`); once the burst transitions to `Awaiting` the
/// fire has already happened, the deadline is moot, and a stale fire is
/// dropped silently by the validation in
/// [`crate::Engine::is_timer_referenced`].
/// `AwaitGateDeadline` — recovery timer armed at
/// [`BurstPhase::Awaiting`] entry. Expiry indicates the actuator is
/// taking longer than expected (likely a hung child); the engine
/// force-transitions to `Rebasing` to re-establish a baseline against
/// disk reality.
///
/// Carried alongside [`TimerId`] on the engine's heap entry and on
/// [`crate::input::Input::TimerExpired`] so dispatch routes directly on
/// the kind without re-deriving from Profile state. The [`TimerId`]
/// continues to act as the lazy-invalidation epoch — `kind` only narrows
/// the validation slot, it does not replace it.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum TimerKind {
    #[default]
    Settle,
    BurstDeadline,
    AwaitGateDeadline,
}

#[derive(Debug)]
pub struct Profile {
    pub resource: ResourceId,
    pub config: ScanConfig,
    pub config_hash: u64,
    pub state: ProfileState,
    /// Engine-side slot for the **probe channel** — the per-Profile
    /// communication primitive between the engine and the Prober pool.
    /// Holds the correlation token of an outstanding `ProbeRequest`, or
    /// `None` if no probe is in flight.
    ///
    /// **Discipline.** Open via `Engine::mint_probe_correlation`; close
    /// via the response-dispatch path (top of `Engine::on_probe_response`)
    /// or via `Engine::cancel_pending_probe`. Open for at most one
    /// outstanding request, regardless of which lifecycle state
    /// (`Pending` or `Active`) drives the emission.
    ///
    /// **Sibling channels.** Distinct from the *watch channel*
    /// (per-Resource, refcounted via `watch_demand`) and the *effect
    /// channel* (per-(Sub, DedupKey), coalesced in the Actuator).
    pub pending_probe: Option<ProbeCorrelation>,
    pub baseline: Option<TreeSnapshot>,
    pub current: Option<TreeSnapshot>,
    pub dirty_descendants: u32,
    pub sub_refcount: u32,
    pub max_settle: Duration,
    /// Settle interval driving `start_standard_burst` and the backoff base.
    /// Cached on construction from the first attached Sub; the engine
    /// recomputes this as `min(remaining_subs.settles)` on `attach_sub`
    /// (existing Profile) and `detach_sub`.
    pub settle: Duration,
    /// True iff the last Sub on this Profile was detached while a burst was
    /// in flight. The active burst runs to completion; `finish_burst_to_idle`
    /// checks this flag, suppresses Effect emission, and reaps the Profile.
    pub reap_pending: bool,
    /// Cached parent Resource that this Profile contributes a watch to.
    /// `attach_sub` sets it; `detach_sub` releases the contribution via the
    /// cached id without re-deriving the parent. `None` if the anchor is
    /// itself a root (no parent in the Tree) — root rename detection is then
    /// unavailable.
    pub watch_root_parent: Option<ResourceId>,
    /// Tracks whether this Profile currently holds a `+1` contribution on
    /// `resource.watch_demand` — set on the path that called
    /// `add_watch_demand(anchor)` (immediate-Seed in `attach_sub_inner`
    /// or descent's anchor materialization), cleared on the matching
    /// `sub_watch_demand(anchor)` (anchor terminal event, reap).
    ///
    /// The flag distinguishes three reap-time lifecycle states that
    /// otherwise look identical in the Profile/descent registry:
    /// **materialized** (`true` ⇒ release anchor), **pending**
    /// (descent in flight ⇒ release descent prefix instead), and
    /// **purged** (`false`, descent already removed by
    /// `Input::WatchOpRejected` ⇒ no contribution to release; the clamp
    /// already did it).
    ///
    /// Without this flag a heuristic like `baseline.is_some() ||
    /// current.is_some()` undercounts `dispatch_seed_vanished` paths
    /// (which clear the snapshots while leaving the anchor's contribution
    /// intact) and a heuristic like `tree.get(anchor).watch_demand > 0`
    /// overcounts in multi-Profile sharing (would steal another
    /// Profile's contribution).
    pub anchor_contribution: bool,
    /// Per-`DedupKey` `dir_hash` (or `leaf_hash`) of the hierarchical
    /// snapshot the engine fired against on the most recent successful
    /// Effect emission for that key.
    ///
    /// After `EffectComplete::Ok` settles, the next stable verdict will
    /// compare `Profile.current.dir_hash()` (or per-file leaf hash) against
    /// the entry here; an identical hash means the post-burst state is the
    /// same one we already fired against, so suppress the duplicate fire.
    /// Cleared on `EffectComplete::Failed` — the failed Effect leaves no
    /// observation to deduplicate against.
    pub last_emitted_dir_hash: BTreeMap<DedupKey, u128>,
    /// User-declared event-class mask for this Profile. Every Sub on a
    /// Profile shares the same `events` by construction (mask folds into
    /// `config_hash`), so this field is the Sub's mask — the "union"
    /// naming is structural: per-Sub contributions OR onto the
    /// Profile's mask, even though the OR is a no-op here. The
    /// per-Resource `events_union` aggregated across covering Profiles
    /// reads this as the per-Profile contribution.
    pub events_union: ClassSet,
    /// True iff covered Leaves need their own FDs. Derived at construction
    /// from `events.intersects(CONTENT | METADATA)` and invariant for the
    /// Profile's lifetime (events are part of `config_hash`, so a mask
    /// change forks a new Profile rather than flipping this flag).
    ///
    /// The walker-side reconciler reads this to decide whether covered
    /// Leaf children get `add_watch_demand` (per-file FDs for in-place
    /// edit detection — closes E2E #3 by default for `subtree-root` Subs
    /// whose default mask includes CONTENT).
    pub has_per_file_fds: bool,
}

impl Profile {
    /// Construct a fresh Profile: state `Idle`, no baseline/current,
    /// refcounts at zero, no reap pending, no watch-root parent recorded.
    /// `config_hash` is computed from `(config, max_settle, events)` and
    /// is stable for the Profile's lifetime — there is no path to a
    /// Profile with an unset or stale hash.
    ///
    /// `events` becomes the Profile's `events_union` and drives
    /// `has_per_file_fds` (true iff CONTENT or METADATA is in the mask).
    /// Every Sub on a Profile shares the same `events`, so
    /// `events_union` is invariant for the Profile's lifetime.
    #[must_use]
    pub fn new(
        resource: ResourceId,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        events: ClassSet,
    ) -> Self {
        let config_hash = compute_config_hash(&config, max_settle, events);
        let has_per_file_fds = events.intersects(ClassSet::CONTENT | ClassSet::METADATA);
        Self {
            resource,
            config,
            config_hash,
            state: ProfileState::Idle,
            pending_probe: None,
            baseline: None,
            current: None,
            dirty_descendants: 0,
            sub_refcount: 0,
            max_settle,
            settle,
            reap_pending: false,
            watch_root_parent: None,
            anchor_contribution: false,
            last_emitted_dir_hash: BTreeMap::new(),
            events_union: events,
            has_per_file_fds,
        }
    }
}

#[derive(Debug, Default)]
pub struct ProfileMap {
    profiles: SlotMap<ProfileId, Profile>,
    by_resource: SecondaryMap<ResourceId, TinyVec<[(u64, ProfileId); 1]>>,
}

impl ProfileMap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up an existing Profile by `(resource, config_hash)`. Returns
    /// `None` if no Profile at this resource matches the hash.
    #[must_use]
    pub fn find(&self, resource: ResourceId, config_hash: u64) -> Option<ProfileId> {
        self.by_resource
            .get(resource)?
            .iter()
            .find(|(h, _)| *h == config_hash)
            .map(|(_, id)| *id)
    }

    /// Insert a fresh Profile and write back-references on both the Tree
    /// (`Resource.profiles`) and the `ProfileMap` (`by_resource`). Caller
    /// has verified `find` returns `None` for `(profile.resource,
    /// profile.config_hash)`; a debug-build assertion guards against repeat.
    ///
    /// Panics if `profile.resource` is stale (no live Tree slot). The Engine
    /// must construct the Resource before attaching a Profile to it.
    pub fn attach(&mut self, tree: &mut Tree, profile: Profile) -> ProfileId {
        let resource = profile.resource;
        let hash = profile.config_hash;
        debug_assert!(
            self.find(resource, hash).is_none(),
            "ProfileMap::attach called twice for the same (resource, config_hash) — caller must `find` first",
        );
        let id = self.profiles.insert(profile);
        // SecondaryMap::entry returns None only if the key has been removed
        // from a primary-tracked SlotMap with a generation that no longer
        // matches. For a freshly-minted ResourceId, we expect `Some`.
        self.by_resource
            .entry(resource)
            .expect("ProfileMap::attach: resource is stale (slot was reaped)")
            .or_default()
            .push((hash, id));
        tree.get_mut(resource)
            .expect("ProfileMap::attach: resource has no live Tree slot")
            .profiles
            .push((hash, id));
        id
    }

    /// Remove a Profile and clear back-references on both indices. The
    /// caller is responsible for any subsequent `tree.try_reap(resource)`
    /// once it confirms no other anchors remain.
    pub fn detach(&mut self, tree: &mut Tree, id: ProfileId) -> Option<Profile> {
        let p = self.profiles.remove(id)?;
        if let Some(v) = self.by_resource.get_mut(p.resource) {
            v.retain(|(h, pid)| !(*pid == id && *h == p.config_hash));
        }
        if let Some(r) = tree.get_mut(p.resource) {
            r.profiles
                .retain(|(h, pid)| !(*pid == id && *h == p.config_hash));
        }
        Some(p)
    }

    #[must_use]
    pub fn get(&self, id: ProfileId) -> Option<&Profile> {
        self.profiles.get(id)
    }

    pub fn get_mut(&mut self, id: ProfileId) -> Option<&mut Profile> {
        self.profiles.get_mut(id)
    }

    /// Iterator over the Profiles attached at `resource`, in
    /// `Resource.profiles` insertion order.
    pub fn at(&self, resource: ResourceId) -> impl Iterator<Item = ProfileId> + '_ {
        self.by_resource
            .get(resource)
            .into_iter()
            .flatten()
            .map(|(_, id)| *id)
    }

    pub fn iter(&self) -> impl Iterator<Item = (ProfileId, &Profile)> {
        self.profiles.iter()
    }

    pub fn iter_mut(&mut self) -> impl Iterator<Item = (ProfileId, &mut Profile)> {
        self.profiles.iter_mut()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.profiles.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.profiles.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassSet, Profile, ProfileMap, ProfileState, ScanConfig, compute_config_hash};
    use crate::resource::ResourceRole;
    use crate::tree::Tree;
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    fn cfg() -> ScanConfig {
        ScanConfig::builder().build()
    }

    #[test]
    fn new_profile_starts_idle_with_zero_refcounts() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(matches!(p.state, ProfileState::Idle));
        assert!(p.baseline.is_none());
        assert!(p.current.is_none());
        assert_eq!(p.dirty_descendants, 0);
        assert_eq!(p.sub_refcount, 0);
        assert_eq!(p.max_settle, MAX_SETTLE);
        assert_eq!(p.settle, SETTLE);
    }

    /// `last_emitted_dir_hash` defaults to an empty map; engine fills it on
    /// first successful Effect emission.
    #[test]
    fn new_profile_initialises_last_emitted_dir_hash_empty() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(p.last_emitted_dir_hash.is_empty());
    }

    /// `has_per_file_fds` defaults to false when `events` excludes both
    /// CONTENT and METADATA. The flag is invariant for the Profile's
    /// lifetime — set once at construction from the events mask.
    #[test]
    fn new_profile_initialises_has_per_file_fds_false_for_empty_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(!p.has_per_file_fds);
        assert_eq!(p.events_union, ClassSet::EMPTY);
    }

    /// `has_per_file_fds` is true when CONTENT is in the mask (closes
    /// E2E #3 by default for `subtree-root`).
    #[test]
    fn new_profile_has_per_file_fds_when_content_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT);
        assert!(p.has_per_file_fds);
        assert_eq!(p.events_union, ClassSet::CONTENT);
    }

    /// `has_per_file_fds` is also true when METADATA is in the mask (a
    /// metadata-only watch needs per-file FDs for chmod / nlink signals).
    #[test]
    fn new_profile_has_per_file_fds_when_metadata_in_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA);
        assert!(p.has_per_file_fds);
    }

    /// STRUCTURE-only watch does not flip `has_per_file_fds` — directory
    /// entries are observed at the parent dir's FD, not at per-file FDs.
    #[test]
    fn new_profile_has_per_file_fds_false_for_structure_only() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::STRUCTURE);
        assert!(!p.has_per_file_fds);
    }

    #[test]
    fn config_hash_matches_compute_config_hash() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let c = cfg();
        let expected = compute_config_hash(&c, MAX_SETTLE, NO_EVENTS);
        let p = Profile::new(r, c, MAX_SETTLE, SETTLE, NO_EVENTS);
        assert_eq!(p.config_hash, expected);
    }

    /// Different `events` mask produces different `config_hash`
    /// (partition-by-mask).
    #[test]
    fn config_hash_partitions_by_events() {
        let mut tree = Tree::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p_content = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::CONTENT);
        let p_meta = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, ClassSet::METADATA);
        assert_ne!(p_content.config_hash, p_meta.config_hash);
    }

    #[test]
    fn attach_writes_both_indices() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        let h = p.config_hash;
        let pid = profiles.attach(&mut tree, p);

        assert_eq!(profiles.find(r, h), Some(pid));
        assert_eq!(tree.get(r).unwrap().profiles(), &[(h, pid)]);
    }

    #[test]
    fn attach_anchors_resource_against_reap() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let _pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        tree.vacate(r);
        assert!(!tree.try_reap(r), "Profile-anchored resource must not reap");
    }

    #[test]
    fn detach_clears_back_references() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let p = Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS);
        let h = p.config_hash;
        let pid = profiles.attach(&mut tree, p);

        let detached = profiles.detach(&mut tree, pid);
        assert!(detached.is_some(), "detach yields the removed Profile");
        assert!(profiles.find(r, h).is_none());
        assert!(tree.get(r).unwrap().profiles().is_empty());
    }

    #[test]
    fn detach_then_reap_succeeds_when_no_other_anchors() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);
        let pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        profiles.detach(&mut tree, pid);
        tree.vacate(r);
        assert!(tree.try_reap(r));
        assert!(tree.get(r).is_none());
    }

    #[test]
    fn at_iterates_profiles_attached_at_resource() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "anchor", ResourceRole::User);

        let pid_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS),
        );
        // Different max_settle ⇒ different config_hash ⇒ distinct Profile.
        let pid_b = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS),
        );

        let mut got: Vec<_> = profiles.at(r).collect();
        got.sort();
        let mut expected = vec![pid_a, pid_b];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn distinct_resources_get_distinct_profiles() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r1 = tree.ensure(None, "a", ResourceRole::User);
        let r2 = tree.ensure(None, "b", ResourceRole::User);

        let p1 = profiles.attach(
            &mut tree,
            Profile::new(r1, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        let p2 = profiles.attach(
            &mut tree,
            Profile::new(r2, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        assert_ne!(p1, p2);
        assert_eq!(profiles.len(), 2);
    }

    #[test]
    #[cfg_attr(
        not(debug_assertions),
        ignore = "debug_assert! is compiled out in release"
    )]
    #[should_panic(expected = "called twice")]
    fn attach_duplicate_panics_in_debug() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "x", ResourceRole::User);
        let _pid = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        // Caller failed to `find` first; second attach hits debug_assert.
        let _pid2 = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
    }
}
