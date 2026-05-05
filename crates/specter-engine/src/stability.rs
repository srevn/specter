//! Stability composition. Holds parent edges between Profiles
//! (nearest covering ancestor) and propagates dirty-descendant deltas.

use crate::coverage::covers;
use slotmap::SecondaryMap;
use specter_core::{BurstPhase, ProfileId, ProfileMap, ProfileState, Tree};
use tinyvec::TinyVec;

/// Parent-edge index per Profile.
///
/// Absent ⇒ root Profile (no covering ancestor). Mutated by the engine on
/// Profile attach/detach and at hot reload; queried by the burst-
/// lifecycle propagation routines.
#[derive(Debug, Default)]
pub struct StabilityIndex {
    parents: SecondaryMap<ProfileId, ProfileId>,
}

impl StabilityIndex {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    #[must_use]
    pub fn parent_of(&self, child: ProfileId) -> Option<ProfileId> {
        self.parents.get(child).copied()
    }

    /// Record `parent` as the nearest covering ancestor of `child`. The
    /// engine calls this exactly once per Profile, when the Profile attaches
    /// (and again on hot reload that changes the topology).
    pub fn set_parent(&mut self, child: ProfileId, parent: ProfileId) {
        debug_assert_ne!(child, parent, "self-parent edge would loop propagate");
        self.parents.insert(child, parent);
    }

    pub fn clear_parent(&mut self, child: ProfileId) {
        self.parents.remove(child);
    }

    /// Compute the parent edge for `child`: the **nearest ancestor
    /// Profile** P' such that `covers(P', child.resource) && P' != child`.
    /// Walks Resource ancestors of `child.resource`; at each ancestor
    /// Resource, picks the smallest covering [`ProfileId`] for a
    /// deterministic tie-break. Returns `None` for root Profiles with no
    /// covering ancestor.
    ///
    /// "Nearest ancestor *Profile*, not Resource" is the easy mistake from
    /// the spec: a Resource ancestor with no Profile is skipped; the walk
    /// continues to the next Resource ancestor.
    #[must_use]
    pub fn compute_parent(
        tree: &Tree,
        profiles: &ProfileMap,
        child: ProfileId,
    ) -> Option<ProfileId> {
        let child_resource = profiles.get(child)?.resource;
        for ancestor in tree.ancestors(child_resource) {
            let nearest = profiles
                .at(ancestor)
                .filter(|&pid| pid != child)
                .filter(|&pid| {
                    profiles
                        .get(pid)
                        .is_some_and(|p| covers(p, child_resource, tree))
                })
                .min();
            if nearest.is_some() {
                return nearest;
            }
        }
        None
    }

    /// Recompute parent edges for every Profile that currently names
    /// `removed_profile` as its parent. Called from `Engine::detach_sub`
    /// after the Profile is detached, so dependent Profiles re-resolve
    /// against the current topology. Profiles whose new edge resolves to
    /// `None` (no covering ancestor remains) have their entry removed.
    ///
    /// O(profiles²) worst-case. v1 typical configs are tens of Subs at
    /// most; profile if it bites.
    pub fn recompute_parent_edges_for_dependents(
        &mut self,
        tree: &Tree,
        profiles: &ProfileMap,
        removed_profile: ProfileId,
    ) {
        let dependents: Vec<ProfileId> = self
            .parents
            .iter()
            .filter(|(_, parent)| **parent == removed_profile)
            .map(|(pid, _)| pid)
            .collect();
        for pid in dependents {
            match Self::compute_parent(tree, profiles, pid) {
                Some(new_parent) => {
                    self.parents.insert(pid, new_parent);
                }
                None => {
                    self.parents.remove(pid);
                }
            }
        }
    }

    /// Recompute parent edges for every Profile yielded by
    /// `profiles_to_check`. Used by `Engine::attach_sub` to re-resolve any
    /// existing Profile whose edge would now name the freshly-added
    /// Profile (the new Profile may interpose between an old child and
    /// its old parent).
    ///
    /// Profiles whose recomputed edge is `None` have their entry removed
    /// from `parents`; otherwise the edge is overwritten.
    pub fn recompute_parent_edges_for_subset<I>(
        &mut self,
        tree: &Tree,
        profiles: &ProfileMap,
        profiles_to_check: I,
    ) where
        I: IntoIterator<Item = ProfileId>,
    {
        for pid in profiles_to_check {
            match Self::compute_parent(tree, profiles, pid) {
                Some(new_parent) => {
                    self.parents.insert(pid, new_parent);
                }
                None => {
                    self.parents.remove(pid);
                }
            }
        }
    }

    /// Walk parent edges from `source` and apply `delta` to each
    /// ancestor's `dirty_descendants`. Returns ancestors whose count just
    /// hit zero **and** are in `BurstPhase::Draining` — that combined
    /// condition is what drives the same-step `Draining → Probing`
    /// reconfirm transition.
    ///
    /// `dirty_descendants` is `u32`; the I4 invariant (`≥ 0`) is enforced
    /// by `debug_assert!` in dev and by clamping in release. The `u32 → i64`
    /// widening lets us compute the post-delta value without overflow
    /// before clamping back into `[0, u32::MAX]`.
    pub fn propagate(
        &self,
        profiles: &mut ProfileMap,
        source: ProfileId,
        delta: i32,
    ) -> TinyVec<[ProfileId; 4]> {
        let mut hit_zero: TinyVec<[ProfileId; 4]> = TinyVec::new();
        let mut current = source;
        while let Some(parent) = self.parents.get(current).copied() {
            let Some(p) = profiles.get_mut(parent) else {
                break;
            };
            let prev = p.dirty_descendants;
            let next = i64::from(prev) + i64::from(delta);
            debug_assert!(next >= 0, "dirty_descendants underflow at {parent:?}");
            let clamped = next.clamp(0, i64::from(u32::MAX));
            // `clamped` is in `[0, u32::MAX]` by construction.
            #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
            let new_value = clamped as u32;
            p.dirty_descendants = new_value;
            if prev > 0 && new_value == 0 && in_draining(&p.state) {
                hit_zero.push(parent);
            }
            current = parent;
        }
        hit_zero
    }
}

/// True iff `state` is `Active` with `BurstPhase::Draining`. Only `Draining`
/// Profiles are interested in the `dirty_descendants → 0` edge — the
/// reconfirm-probe transition is the consumer of `propagate`'s return list.
const fn in_draining(state: &ProfileState) -> bool {
    match state {
        ProfileState::Idle => false,
        // Descent has no Draining phase — the arm is structurally
        // redundant with the wildcard but documents that Pending
        // intentionally never drives the reconfirm cascade.
        ProfileState::Pending(_) => false,
        ProfileState::Active(burst) => matches!(burst.phase, BurstPhase::Draining),
        // `ProfileState` is `non_exhaustive`; any future variant is treated
        // as not-Draining (won't drive a reconfirm probe).
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;
    use proptest::prelude::*;
    use slotmap::KeyData;
    use specter_core::{
        Burst, BurstIntent, BurstPhase, ChildEntry, ClassSet, DirMeta, DirSnapshot,
        ProbeCorrelation, Profile, ProfileState, ResourceKind, ResourceRole, ScanConfig, TimerId,
        TreeSnapshot,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, Instant, UNIX_EPOCH};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    /// Test-default events mask. Stability is orthogonal to the event-class
    /// filter; an empty mask gives a Profile with `has_per_file_fds = false`,
    /// matching the prior test invariants where per-file FDs were not in
    /// scope.
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    fn pid(n: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(n))
    }

    fn cfg() -> ScanConfig {
        ScanConfig::builder().recursive(true).build()
    }

    fn mark_dir(tree: &mut Tree, id: specter_core::ResourceId) {
        tree.get_mut(id).unwrap().kind = ResourceKind::Dir;
    }

    #[test]
    fn parent_of_round_trips_set() {
        let mut idx = StabilityIndex::new();
        let child = pid(1);
        let parent = pid(2);
        assert!(idx.parent_of(child).is_none());
        idx.set_parent(child, parent);
        assert_eq!(idx.parent_of(child), Some(parent));
    }

    #[test]
    fn clear_parent_idempotent() {
        let mut idx = StabilityIndex::new();
        let child = pid(1);
        idx.clear_parent(child);
        idx.set_parent(child, pid(2));
        idx.clear_parent(child);
        idx.clear_parent(child);
        assert!(idx.parent_of(child).is_none());
    }

    #[test]
    fn compute_parent_returns_none_for_orphan_profile() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "root", ResourceRole::User);
        mark_dir(&mut tree, r);
        let pid = profiles.attach(&mut tree, Profile::new(r, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        assert!(StabilityIndex::compute_parent(&tree, &profiles, pid).is_none());
    }

    #[test]
    fn compute_parent_walks_up_to_first_covering_ancestor() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        let b = tree.ensure(Some(a), "b", ResourceRole::User);
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_a = profiles.attach(&mut tree, Profile::new(a, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_b = profiles.attach(&mut tree, Profile::new(b, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        assert_eq!(
            StabilityIndex::compute_parent(&tree, &profiles, p_b),
            Some(p_a),
        );
        assert_eq!(
            StabilityIndex::compute_parent(&tree, &profiles, p_a),
            Some(p_root),
        );
        assert_eq!(
            StabilityIndex::compute_parent(&tree, &profiles, p_root),
            None
        );
    }

    #[test]
    fn compute_parent_skips_non_covering_ancestor() {
        // root has Profile p_root with recursive=false → does not cover deep
        // descendants. The deeper Profile's compute_parent should walk past
        // p_root and return None (no further covering ancestor).
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        let b = tree.ensure(Some(a), "b", ResourceRole::User);
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let _p_root = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ScanConfig::builder().recursive(false).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );
        let p_b = profiles.attach(&mut tree, Profile::new(b, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        // p_root's covers(b) is false (depth > 1, recursive=false), so it's
        // not a candidate. No covering ancestor.
        assert_eq!(StabilityIndex::compute_parent(&tree, &profiles, p_b), None);
    }

    #[test]
    fn compute_parent_excludes_self() {
        // Two co-located Profiles at the anchor; compute_parent for one must
        // not return itself.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "root", ResourceRole::User);
        mark_dir(&mut tree, r);
        let p_a = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(r, cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS),
        );
        // Both at root; root has no Profile *ancestor*; compute_parent walks
        // ancestors of root.resource (none — root is a Tree root).
        assert!(StabilityIndex::compute_parent(&tree, &profiles, p_a).is_none());
        assert!(StabilityIndex::compute_parent(&tree, &profiles, p_b).is_none());
    }

    #[test]
    fn compute_parent_ties_by_smallest_profile_id() {
        // Two co-located covering Profiles at the same ancestor Resource.
        // compute_parent for a deeper Profile picks the smaller ProfileId.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let leaf = tree.ensure(Some(root), "leaf", ResourceRole::User);
        mark_dir(&mut tree, root);
        mark_dir(&mut tree, leaf);
        // Two distinct Profiles at root, distinct config_hashes via differing
        // max_settle (makes them separate Profiles).
        let p_root_a = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS),
        );
        let p_root_b = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS),
        );
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let smaller = std::cmp::min(p_root_a, p_root_b);
        assert_eq!(
            StabilityIndex::compute_parent(&tree, &profiles, p_leaf),
            Some(smaller),
        );
    }

    #[test]
    fn propagate_zero_delta_is_noop() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let leaf = tree.ensure(Some(root), "leaf", ResourceRole::User);
        mark_dir(&mut tree, root);
        mark_dir(&mut tree, leaf);
        let p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let mut idx = StabilityIndex::new();
        idx.set_parent(p_leaf, p_root);
        let hit = idx.propagate(&mut profiles, p_leaf, 0);
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 0);
    }

    #[test]
    fn propagate_chain_sums_delta() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
        let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
        for r in [root, mid, leaf] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_mid = profiles.attach(&mut tree, Profile::new(mid, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let mut idx = StabilityIndex::new();
        idx.set_parent(p_mid, p_root);
        idx.set_parent(p_leaf, p_mid);

        let hit = idx.propagate(&mut profiles, p_leaf, 1);
        // No Profile is in Active(Draining), so hit_zero is empty even
        // when we walk through Profiles.
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_mid).unwrap().dirty_descendants, 1);
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 1);
        // The source Profile itself does not propagate to itself.
        assert_eq!(profiles.get(p_leaf).unwrap().dirty_descendants, 0);
    }

    #[test]
    fn propagate_returns_ancestors_in_draining() {
        // Mid Profile is in Active(Draining) with dirty_descendants > 0; the
        // leaf's burst-end propagates -1 and brings mid's count to 0.
        // `propagate` returns mid's ProfileId so the dispatch can drive the
        // reconfirm probe.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
        let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
        for r in [root, mid, leaf] {
            mark_dir(&mut tree, r);
        }
        let _p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_mid = profiles.attach(&mut tree, Profile::new(mid, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        // Synthesize an Active(Draining) state on p_mid. The snapshot lives
        // on `Profile.current` (set by `dispatch_standard_ok` before
        // `transition_to_draining`).
        let stable_snapshot = TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
            mid,
            DirMeta {
                mtime: UNIX_EPOCH,
                inode: 0,
                device: 0,
            },
            Instant::now(),
            0,
            BTreeMap::<CompactString, ChildEntry>::new(),
        )));
        profiles.get_mut(p_mid).unwrap().current = Some(stable_snapshot);
        profiles.get_mut(p_mid).unwrap().state = ProfileState::Active(Burst {
            started: Instant::now(),
            attempts: 0,
            settle_timer: None,
            burst_deadline: TimerId::default(),
            phase: BurstPhase::Draining,
            intent: BurstIntent::Standard,
            forced: false,
            dirty_resources: std::collections::BTreeSet::new(),
            force_walk_resources: std::collections::BTreeSet::new(),
            probe_target: None,
        });
        profiles.get_mut(p_mid).unwrap().dirty_descendants = 1;

        let mut idx = StabilityIndex::new();
        idx.set_parent(p_leaf, p_mid);

        let hit = idx.propagate(&mut profiles, p_leaf, -1);
        assert_eq!(
            &hit[..],
            &[p_mid][..],
            "Draining ancestor whose count reached 0 is returned",
        );
        // Avoid unused-variable warnings for the silenced ProbeCorrelation use.
        let _ = ProbeCorrelation(0);
    }

    #[test]
    fn propagate_returns_empty_in_idle_only_world() {
        // I3 placeholder: every Profile is Idle in this layer. propagate's
        // hit_zero filter (`prev > 0 && new == 0 && in_draining`) cannot fire
        // — Idle Profiles never have prev > 0 in the first place.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let leaf = tree.ensure(Some(root), "leaf", ResourceRole::User);
        mark_dir(&mut tree, root);
        mark_dir(&mut tree, leaf);
        let p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let mut idx = StabilityIndex::new();
        idx.set_parent(p_leaf, p_root);

        // +1 then -1 returns the counter to zero. The transition does cross
        // the prev>0 → new==0 boundary, but in_draining(Idle) is false, so
        // hit_zero stays empty.
        let _ = idx.propagate(&mut profiles, p_leaf, 1);
        let hit = idx.propagate(&mut profiles, p_leaf, -1);
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "dirty_descendants underflow")]
    fn propagate_underflow_panics_in_debug() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let leaf = tree.ensure(Some(root), "leaf", ResourceRole::User);
        mark_dir(&mut tree, root);
        mark_dir(&mut tree, leaf);
        let p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let mut idx = StabilityIndex::new();
        idx.set_parent(p_leaf, p_root);
        // p_root.dirty_descendants starts at 0; -1 underflows.
        let _ = idx.propagate(&mut profiles, p_leaf, -1);
    }

    #[test]
    fn set_parent_overwrites() {
        let mut idx = StabilityIndex::new();
        let child = pid(1);
        idx.set_parent(child, pid(2));
        idx.set_parent(child, pid(3));
        assert_eq!(idx.parent_of(child), Some(pid(3)));
    }

    #[test]
    fn recompute_for_dependents_after_remove() {
        // Three Profiles A (parent), B (child of A), C (child of A). Remove
        // A from the index by detaching its Profile and recomputing
        // dependents — both B and C re-resolve to None (no other covering
        // ancestor exists).
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
        let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
        for r in [root, mid, leaf] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_mid = profiles.attach(&mut tree, Profile::new(mid, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let mut idx = StabilityIndex::new();
        idx.set_parent(p_leaf, p_mid);
        idx.set_parent(p_mid, p_root);

        // Detach p_root via the registry so compute_parent walks see no A.
        let _ = profiles.detach(&mut tree, p_root);

        idx.recompute_parent_edges_for_dependents(&tree, &profiles, p_root);

        // p_mid had p_root as parent; recomputed yields None (no other
        // covering ancestor). p_leaf still names p_mid as parent (untouched).
        assert!(idx.parent_of(p_mid).is_none());
        assert_eq!(idx.parent_of(p_leaf), Some(p_mid));
    }

    #[test]
    fn recompute_for_subset_picks_new_interposing_profile() {
        // Sequence: leaf has no parent edge. Then a Profile is added at the
        // mid Resource that covers leaf — `recompute_for_subset` over
        // [p_leaf] now resolves p_leaf's edge to the new mid.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
        let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
        for r in [root, mid, leaf] {
            mark_dir(&mut tree, r);
        }
        let _p_root = profiles.attach(&mut tree, Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        let p_leaf = profiles.attach(&mut tree, Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));

        let mut idx = StabilityIndex::new();
        if let Some(parent) = StabilityIndex::compute_parent(&tree, &profiles, p_leaf) {
            idx.set_parent(p_leaf, parent);
        }
        // p_leaf's edge currently points at p_root.

        // Add p_mid; it interposes between p_leaf and p_root.
        let p_mid = profiles.attach(&mut tree, Profile::new(mid, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS));
        idx.recompute_parent_edges_for_subset(&tree, &profiles, [p_leaf]);

        assert_eq!(idx.parent_of(p_leaf), Some(p_mid));
    }

    proptest! {
        #[test]
        fn prop_set_then_parent_of_round_trips(
            c in 1u64..1024, p in 1u64..1024,
        ) {
            prop_assume!(c != p);
            let mut idx = StabilityIndex::new();
            idx.set_parent(pid(c), pid(p));
            prop_assert_eq!(idx.parent_of(pid(c)), Some(pid(p)));
        }
    }
}
