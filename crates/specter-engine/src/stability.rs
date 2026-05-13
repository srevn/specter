//! Parent-edge propagation and draining-cascade dispatch.
//!
//! Storage for the parent edge lives on `Profile.parent_profile` — this
//! module is the namespace of free functions that walk and maintain
//! those edges. There is no `StabilityIndex` struct; the cached edge is
//! per-Profile metadata, and the verb (propagate) reads it directly
//! through `&mut ProfileMap`.

use crate::coverage::nearest_covering_ancestor;
use specter_core::{ActiveBurst, PreFirePhase, ProfileId, ProfileMap, ProfileState, Tree};
use tinyvec::TinyVec;

/// Walk parent edges from `source` and apply `delta` to each ancestor's
/// `dirty_descendants`. Returns ancestors whose count just hit zero
/// **and** are in [`specter_core::PreFirePhase::Draining`] — that combined condition
/// drives the same-step `Draining → Verifying` reconfirm transition.
///
/// `dirty_descendants` is `u32`; the I4 invariant (`≥ 0`) is enforced
/// by `debug_assert!` in dev and clamping in release. The `u32 → i64`
/// widening lets us compute the post-delta value without overflow
/// before clamping back into `[0, u32::MAX]`.
///
/// Defensive: if the cached chain points at a reaped Profile (a
/// transient state between detach and `recompute_parent_edges_for_dependents`,
/// or any missed maintenance bug), the walk terminates rather than
/// trying to mutate a vacated slot.
pub(crate) fn propagate(
    profiles: &mut ProfileMap,
    source: ProfileId,
    delta: i32,
) -> TinyVec<[ProfileId; 4]> {
    let mut hit_zero: TinyVec<[ProfileId; 4]> = TinyVec::new();
    let mut current = source;
    while let Some(parent) = profiles.get(current).and_then(|p| p.parent_profile) {
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

/// Recompute parent edges for every Profile that currently names
/// `removed_profile` as its parent. Called from `Engine::reap_profile`
/// after the Profile is detached from `ProfileMap`, so dependents
/// re-resolve against the current topology. Profiles whose new edge
/// resolves to `None` (no covering ancestor remains) have their
/// `parent_profile` cleared.
///
/// Asymptotic: O(profiles) per call (the iteration scans the whole
/// `ProfileMap` to find dependents). For v1 typical configs (~50
/// Profiles) this is trivially fast. A reverse index
/// `Map<parent, Vec<children>>` would narrow this to O(dependents);
/// deferred until profile-attach rates make it visible.
pub(crate) fn recompute_parent_edges_for_dependents(
    tree: &Tree,
    profiles: &mut ProfileMap,
    removed_profile: ProfileId,
) {
    let dependents: Vec<ProfileId> = profiles
        .iter()
        .filter(|(_, p)| p.parent_profile == Some(removed_profile))
        .map(|(pid, _)| pid)
        .collect();
    for pid in dependents {
        let new_parent = nearest_covering_ancestor(tree, profiles, pid);
        write_parent_edge(profiles, pid, new_parent);
    }
}

/// Recompute parent edges for every Profile yielded by `candidates`.
/// Used by `Engine::attach_sub_inner` to re-resolve any existing
/// Profile whose edge would now name the freshly-added Profile (the
/// new Profile may interpose between an old child and its old parent).
///
/// Profiles whose recomputed edge is `None` have their
/// `parent_profile` cleared; otherwise the edge is overwritten. The
/// caller pre-narrows `candidates` to strict descendants of the new
/// anchor (see `Engine::recompute_dependent_parent_edges`).
pub(crate) fn recompute_parent_edges_for_subset<I>(
    tree: &Tree,
    profiles: &mut ProfileMap,
    candidates: I,
) where
    I: IntoIterator<Item = ProfileId>,
{
    for pid in candidates {
        let new_parent = nearest_covering_ancestor(tree, profiles, pid);
        write_parent_edge(profiles, pid, new_parent);
    }
}

/// Single source for parent-edge writes. The `debug_assert!` against
/// self-parent prevents an infinite `propagate` loop in dev/CI; all
/// engine-side writes converge here so the assertion is unmissable.
/// No-op when `child` is stale (a vacated slot — slotmap returns
/// `None`).
pub(crate) fn write_parent_edge(
    profiles: &mut ProfileMap,
    child: ProfileId,
    parent: Option<ProfileId>,
) {
    if let Some(p) = parent {
        debug_assert_ne!(child, p, "self-parent edge would loop propagate");
    }
    if let Some(profile) = profiles.get_mut(child) {
        profile.parent_profile = parent;
    }
}

/// True iff `state` is `Active(PreFire(Draining))`. Only Draining
/// Profiles are interested in the `dirty_descendants → 0` edge — the
/// reconfirm-probe transition is the consumer of `propagate`'s return
/// list. `Idle` and `Pending` are structurally not-Draining; the
/// descent lifecycle never drives the reconfirm cascade. Post-fire
/// phases are type-impossible (`Draining` lives only on
/// [`PreFirePhase`]); the match's wildcard captures them along with
/// the other non-Draining pre-fire phases.
const fn in_draining(state: &ProfileState) -> bool {
    match state {
        ProfileState::Idle | ProfileState::Pending(_) => false,
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
            matches!(pre.phase, PreFirePhase::Draining)
        }
        ProfileState::Active(ActiveBurst::PostFire(_), _) => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use compact_str::CompactString;
    use specter_core::{
        ActiveBurst, BurstFinish, BurstIntent, ChildEntry, ClassSet, DirMeta, DirSnapshot,
        PreFireBurst, PreFirePhase, Profile, ProfileState, ResourceRole, ScanConfig, TimerId,
        TreeSnapshot,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    /// Test-default events mask. Stability is orthogonal to the
    /// event-class filter; an empty mask gives a Profile with
    /// `has_per_file_fds = false`.
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    fn cfg() -> ScanConfig {
        ScanConfig::builder().recursive(true).build()
    }

    fn mark_dir(tree: &mut Tree, id: specter_core::ResourceId) {
        tree.set_kind(id, specter_core::ResourceKind::Dir);
    }

    /// Anchor a chain `root → mid → leaf` of three User-roled Dir
    /// resources, each with a recursive Profile attached. Returns
    /// `(tree, profiles, p_root, p_mid, p_leaf)` for the test body.
    fn three_level_chain() -> (Tree, ProfileMap, ProfileId, ProfileId, ProfileId) {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
        let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
        for r in [root, mid, leaf] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        let p_mid = profiles.attach(
            &mut tree,
            Profile::new(mid, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        let p_leaf = profiles.attach(
            &mut tree,
            Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        (tree, profiles, p_root, p_mid, p_leaf)
    }

    /// Resolve and write the parent edge for `child` in one step.
    /// Mirrors `Engine::compute_and_set_parent_edge`'s shape so tests
    /// exercise the same code path as production.
    fn resolve_parent(tree: &Tree, profiles: &mut ProfileMap, child: ProfileId) {
        let parent = nearest_covering_ancestor(tree, profiles, child);
        write_parent_edge(profiles, child, parent);
    }

    /// Resolves all three Profiles' parent edges via the
    /// `nearest_covering_ancestor + write_parent_edge` composition
    /// and verifies they end up correctly chained (leaf → mid → root,
    /// root → None).
    #[test]
    fn nearest_covering_ancestor_composes_with_write_parent_edge() {
        let (tree, mut profiles, p_root, p_mid, p_leaf) = three_level_chain();
        resolve_parent(&tree, &mut profiles, p_leaf);
        resolve_parent(&tree, &mut profiles, p_mid);
        resolve_parent(&tree, &mut profiles, p_root);

        assert_eq!(profiles.get(p_leaf).unwrap().parent_profile, Some(p_mid));
        assert_eq!(profiles.get(p_mid).unwrap().parent_profile, Some(p_root));
        assert!(profiles.get(p_root).unwrap().parent_profile.is_none());
    }

    /// Burst-start `+1` propagates through a fully-resolved chain;
    /// each ancestor's `dirty_descendants` increments. Symmetric `-1`
    /// returns it to zero. No Profile is in Draining, so `hit_zero`
    /// stays empty.
    #[test]
    fn propagate_round_trips_through_chain() {
        let (tree, mut profiles, p_root, p_mid, p_leaf) = three_level_chain();
        resolve_parent(&tree, &mut profiles, p_leaf);
        resolve_parent(&tree, &mut profiles, p_mid);

        let hit = propagate(&mut profiles, p_leaf, 1);
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_mid).unwrap().dirty_descendants, 1);
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 1);
        // The source itself does not propagate to itself.
        assert_eq!(profiles.get(p_leaf).unwrap().dirty_descendants, 0);

        let hit = propagate(&mut profiles, p_leaf, -1);
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_mid).unwrap().dirty_descendants, 0);
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 0);
    }

    #[test]
    fn propagate_zero_delta_is_noop() {
        let (tree, mut profiles, p_root, _, p_leaf) = three_level_chain();
        resolve_parent(&tree, &mut profiles, p_leaf);
        let hit = propagate(&mut profiles, p_leaf, 0);
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 0);
    }

    #[test]
    fn propagate_returns_ancestors_in_draining() {
        // Mid Profile is in Active(Draining) with dirty_descendants > 0;
        // the leaf's burst-end propagates -1 and brings mid's count to 0.
        // `propagate` returns mid's ProfileId so dispatch can drive the
        // reconfirm probe.
        let (_tree, mut profiles, _p_root, p_mid, p_leaf) = three_level_chain();
        // Only mid → leaf — the test focuses on a single Draining
        // ancestor; leaving p_root unparented is fine.
        write_parent_edge(&mut profiles, p_leaf, Some(p_mid));

        // Synthesize Active(Draining) on p_mid. The snapshot lives on
        // `Profile.current` (set by `dispatch_standard_ok` before
        // `transition_to_draining`).
        let mid_resource = profiles.get(p_mid).unwrap().resource;
        let stable_snapshot = TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
            mid_resource,
            DirMeta {
                mtime: UNIX_EPOCH,
                inode: 0,
                device: 0,
            },
            0,
            BTreeMap::<CompactString, ChildEntry>::new(),
        )));
        {
            let mid = profiles.get_mut(p_mid).unwrap();
            mid.current = Some(stable_snapshot);
            mid.state = ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    burst_deadline: TimerId::default(),
                    phase: PreFirePhase::Draining,
                    intent: BurstIntent::Standard,
                    forced: false,
                    dirty_resources: std::collections::BTreeSet::new(),
                    force_walk_resources: std::collections::BTreeSet::new(),
                    probe_target: mid_resource,
                    suppressed_resources: std::collections::BTreeSet::new(),
                    last_event_time: None,
                }),
                BurstFinish::ReturnToIdle,
            );
            mid.dirty_descendants = 1;
        }

        let hit = propagate(&mut profiles, p_leaf, -1);
        assert_eq!(
            &hit[..],
            &[p_mid][..],
            "Draining ancestor whose count reached 0 is returned",
        );
    }

    /// I3 placeholder: every Profile is Idle. `propagate`'s `hit_zero`
    /// filter (`prev > 0 && new == 0 && in_draining`) cannot fire —
    /// crossing the `prev > 0 → new == 0` boundary is fine, but
    /// `in_draining(Idle)` is false, so the ancestor is not returned.
    #[test]
    fn propagate_returns_empty_in_idle_only_world() {
        let (tree, mut profiles, p_root, _, p_leaf) = three_level_chain();
        resolve_parent(&tree, &mut profiles, p_leaf);

        let _ = propagate(&mut profiles, p_leaf, 1);
        let hit = propagate(&mut profiles, p_leaf, -1);
        assert!(hit.is_empty());
        assert_eq!(profiles.get(p_root).unwrap().dirty_descendants, 0);
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "dirty_descendants underflow")]
    fn propagate_underflow_panics_in_debug() {
        let (tree, mut profiles, _p_root, _p_mid, p_leaf) = three_level_chain();
        resolve_parent(&tree, &mut profiles, p_leaf);
        // p_root.dirty_descendants starts at 0; -1 underflows.
        let _ = propagate(&mut profiles, p_leaf, -1);
    }

    /// Defensive: a cached `parent_profile` pointing at a vacated slot
    /// must not panic — `propagate` halts at the first dead pointer.
    /// Reproduces the transient stale window between
    /// `ProfileMap::detach` and `recompute_parent_edges_for_dependents`.
    #[test]
    fn propagate_halts_on_stale_parent_pointer() {
        let (mut tree, mut profiles, p_root, p_mid, p_leaf) = three_level_chain();
        write_parent_edge(&mut profiles, p_leaf, Some(p_mid));
        write_parent_edge(&mut profiles, p_mid, Some(p_root));

        // Detach p_root without running the recompute cascade. p_mid
        // still names p_root via `parent_profile`; the propagate walk
        // hits the stale pointer at the second step and breaks cleanly.
        let _ = profiles.detach(&mut tree, p_root);

        let hit = propagate(&mut profiles, p_leaf, 1);
        assert!(hit.is_empty());
        // p_mid was the live first hop — delta applied there.
        assert_eq!(profiles.get(p_mid).unwrap().dirty_descendants, 1);
    }

    /// Detach `removed`, run `recompute_parent_edges_for_dependents`:
    /// dependents whose new edge is `None` have their cache cleared;
    /// dependents that re-resolve to a different ancestor are
    /// rewritten in place.
    #[test]
    fn recompute_for_dependents_clears_or_rewrites_each_child() {
        let (mut tree, mut profiles, p_root, p_mid, p_leaf) = three_level_chain();
        write_parent_edge(&mut profiles, p_leaf, Some(p_mid));
        write_parent_edge(&mut profiles, p_mid, Some(p_root));

        // Detach p_root via the registry so the resolver no longer
        // sees it as a candidate.
        let _ = profiles.detach(&mut tree, p_root);

        recompute_parent_edges_for_dependents(&tree, &mut profiles, p_root);

        // p_mid had p_root as parent; recomputed edge is `None`
        // (no other covering ancestor). p_leaf still names p_mid as
        // parent (unaffected — its parent wasn't reaped).
        assert!(profiles.get(p_mid).unwrap().parent_profile.is_none());
        assert_eq!(profiles.get(p_leaf).unwrap().parent_profile, Some(p_mid));
    }

    /// Sequence: leaf's edge currently points at root. Then a Profile
    /// is added at the mid Resource that covers leaf —
    /// `recompute_parent_edges_for_subset` over `[p_leaf]` rewrites
    /// the edge to the new mid.
    #[test]
    fn recompute_for_subset_picks_new_interposing_profile() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::User);
        let leaf = tree.ensure(Some(mid), "leaf", ResourceRole::User);
        for r in [root, mid, leaf] {
            mark_dir(&mut tree, r);
        }
        let _p_root = profiles.attach(
            &mut tree,
            Profile::new(root, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        let p_leaf = profiles.attach(
            &mut tree,
            Profile::new(leaf, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );

        // Initial parent edge: p_leaf → p_root (no mid yet).
        resolve_parent(&tree, &mut profiles, p_leaf);

        // Interpose p_mid; recompute_parent_edges_for_subset sees
        // p_leaf and rewrites its edge to the closer ancestor.
        let p_mid = profiles.attach(
            &mut tree,
            Profile::new(mid, cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
        );
        recompute_parent_edges_for_subset(&tree, &mut profiles, [p_leaf]);

        assert_eq!(profiles.get(p_leaf).unwrap().parent_profile, Some(p_mid));
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "self-parent edge")]
    fn write_parent_edge_self_loop_panics_in_debug() {
        let (_tree, mut profiles, _, _, p_leaf) = three_level_chain();
        write_parent_edge(&mut profiles, p_leaf, Some(p_leaf));
    }
}
