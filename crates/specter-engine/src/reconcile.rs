//! Newly-discovered descendants reconciliation.
//!
//! Every `ProbeResponse(Ok(snapshot))` runs `reconcile_descendants` *before*
//! the transition body. The pass diffs `snapshot` against the Profile's
//! prior snapshot (`current` first, falling back to `baseline`); for each
//! delta entry it ensures the Resource exists, sets its `kind`, and updates
//! `watch_demand` via `add_watch_demand` / `sub_watch_demand` on the 0↔1
//! edge. The Watch ops appear in `StepOutput.watch_ops` alongside any
//! suppress edges from the burst lifecycle.
//!
//! **Empty-prior synthesis.** When neither `current` nor `baseline` is set —
//! the Seed burst's first probe — the spec's pseudocode would skip the loop
//! and leave descendants Unwatched. We synthesize a `Diff` whose `created`
//! list is the entire `snapshot.entries`; the same code path then runs
//! uniformly. Equivalent to diffing against an empty `TreeSnapshot`, no extra
//! allocation cost beyond the `EntryRef` copies.
//!
//! **Multi-component segments.** Entry segments like `subdir/file.txt` for
//! recursive scans are supported. `Tree::ensure` is single-component by
//! design (`(parent, segment)` is the slot). We walk components within
//! reconcile via `ensure_descendant` / `lookup_descendant`; the lex sort over
//! `(segment_str, kind)` ensures a parent like `subdir` sorts before any
//! descendant `subdir/...`, so intermediate slots exist before their
//! descendants are processed.
//!
//! **Reaping and ordering.** The two-phase order is:
//! 1. Deletions (`delta.deleted` then `delta.renamed.from`), in reverse
//!    lex order — leaves before parents. `try_reap` checks
//!    `has_anchors`; processing leaves first lets parents reap once
//!    their last child is gone.
//! 2. Creations (`delta.created` then `delta.renamed.to`), in forward lex
//!    order — parents before descendants. `ensure_descendant` walks
//!    intermediate components; the parent slot must exist before the
//!    child entry is processed.
//!
//! Deletions-before-creations is load-bearing for the **same-segment
//! kind change** case (`rm foo (File)` then `mkdir foo (Dir)`): the diff
//! reports `Deleted(foo, File, inode_A)` + `Created(foo, Dir, inode_B)`,
//! and the Tree slot at `(parent, "foo")` is shared between them. If we
//! processed creations first, the slot would be re-typed to Dir and
//! Watched; then the deletion pass would look up the same slot (now
//! Dir) and emit Unwatch — silently breaking the new directory's watch.
//! Deletions-first vacates the slot (the file's slot has no Watch
//! contribution to release), and the create pass then
//! `tree.ensure`s a fresh `ResourceId` (generation-incremented) for the
//! new directory, emitting a single Watch.
//!
//! **Reap discipline.** `try_reap` is gated by `watch_demand == 0` so that
//! the multi-Profile case (where another Profile still contributes to
//! `watch_demand`) does not prematurely tear down a still-live slot.
//!
//! **File materialization vs Watch.** All covered diff entries get a Tree
//! slot — `ensure_descendant` runs unconditionally — so
//! `DedupKey::PerFile { sub, resource }` always carries a real `ResourceId`.
//! The Watch op (`add_watch_demand`) is gated to **Dirs only**: v1 doesn't
//! put per-file FDs in the Sensor; pattern matching on
//! `EffectScope::PerStableFile` is at the Profile level and walks the Diff
//! at burst end. Files get Resources, no FDs.

use crate::coverage::covers;
use crate::refcounts::{add_watch_demand, sub_watch_demand};
use specter_core::{
    ChildEntry, DedupKey, Diagnostic, DirSnapshot, EntryKind, Profile, ProfileId, ProfileMap,
    ResourceId, ResourceKind, ResourceRole, SpliceResult, StepOutput, Tree, TreeSnapshot, splice,
};
use std::sync::Arc;

/// Walk `rel_path` component-by-component beneath `anchor`, ensuring each
/// slot. Sets the leaf's `kind` to `leaf_kind`; intermediate components
/// default to `ResourceKind::Dir` only when freshly created (kind was
/// `Unknown`). Returns `None` if `rel_path` is empty (a degenerate case
/// reachable only via spec-violating inputs).
///
/// `pub(crate)` so `transitions::emit_effects_per_stable_file` can reuse
/// the same materialization rules — the diff-entry-to-Resource mapping
/// must agree with what reconcile produced.
pub(crate) fn ensure_descendant(
    tree: &mut Tree,
    anchor: ResourceId,
    rel_path: &str,
    leaf_kind: ResourceKind,
) -> Option<ResourceId> {
    let mut comps = rel_path.split('/').filter(|s| !s.is_empty()).peekable();
    comps.peek()?;
    let mut cur = anchor;
    while let Some(comp) = comps.next() {
        cur = tree.ensure(Some(cur), comp, ResourceRole::User);
        let is_leaf = comps.peek().is_none();
        if let Some(res) = tree.get_mut(cur) {
            if is_leaf {
                // Refresh leaf kind to the snapshot's value — covers create
                // and the rare "kind tracked stale" case (kind change at the
                // same slot is a Vanished → fresh-slot path, not this one).
                res.kind = leaf_kind;
            } else if matches!(res.kind, ResourceKind::Unknown) {
                // Intermediate path component: default to Dir when first
                // created. Don't override an existing kind — P5's pending-
                // path scaffolding stamps `Dir` already.
                res.kind = ResourceKind::Dir;
            }
        }
    }
    Some(cur)
}

/// Walk `rel_path` component-by-component to find an existing slot. Returns
/// `None` if any segment fails to resolve.
///
/// `pub(crate)` so descendant-aware sites outside reconcile (e.g.,
/// `transitions::emit_effects_per_stable_file`,
/// `descent::dispatch_descent_ok`) can resolve a relative path under an
/// anchor without duplicating the component-walk.
pub(crate) fn lookup_descendant(
    tree: &Tree,
    anchor: ResourceId,
    rel_path: &str,
) -> Option<ResourceId> {
    let mut cur = anchor;
    for comp in rel_path.split('/').filter(|s| !s.is_empty()) {
        cur = tree.lookup(Some(cur), comp)?;
    }
    Some(cur)
}

// ---------------------------------------------------------------------------
// reconcile: walk_pair + graft + delete_child
// ---------------------------------------------------------------------------

/// Recursive parallel walk over `(prior_subtree, new_subtree)` at
/// the same logical position in the tree. For each delta entry the pass
/// ensures (or releases) the Tree slot, sets its kind, and updates
/// `watch_demand` via `add_watch_demand` / `sub_watch_demand` on the 0↔1
/// edge for **covered Dirs** (always) and **covered Leaves under
/// `Profile.has_per_file_fds == true`** (B2).
///
/// `add_watch_demand` and `sub_watch_demand` thread the Profile's
/// `events_union` as the per-Resource mask contribution (R2 / D4); the
/// per-Resource union is the OR of every covering Profile's contribution.
/// `sub_watch_demand` requires `&ProfileMap` for the on-decrement
/// recompute, and an explicit `releasing_descendant: Some(profile_id)`
/// signal so the recompute skips this Profile's descendant contribution
/// even though `Profile.current` is still `Some` mid-graft (closes
/// F-MED-4).
///
/// Two-phase order:
/// 1. Deletions in reverse-lex (leaves before parents) so `try_reap`'s
///    `has_anchors` gate sees a vacated child set when it processes the
///    parent.
/// 2. Creations / inode-stable Dir-pair recursion in forward-lex order so
///    intermediate ancestors exist before their descendants are processed.
///
/// `prior == None` ⇒ first observation here (Seed first probe, or a path
/// that became covered between probes). The "Phase 2 / pure create" arm
/// fires for every entry of `new` — same code path for all first-probe
/// scenarios.
///
/// O(1) prune at every level: equal `dir_hash` ⇒ no descendant work.
pub(crate) fn walk_pair(
    prior: Option<&DirSnapshot>,
    new: &DirSnapshot,
    current_id: ResourceId,
    profile: &Profile,
    profile_id: ProfileId,
    tree: &mut Tree,
    profiles: &ProfileMap,
    out: &mut StepOutput,
) {
    // O(1) identity prune at this level.
    if let Some(p) = prior
        && p.dir_hash() == new.dir_hash()
    {
        return;
    }

    // Phase 1 — deletions (reverse lex). A name absent in `new` or
    // present-but-different-(inode, device) is a delete here; the same
    // name + new identity reappears as a create in Phase 2.
    if let Some(p) = prior {
        for (name, prior_child) in p.entries.iter().rev() {
            let delete = match new.entries.get(name) {
                None => true,
                Some(new_child) => !same_inode_device(prior_child, new_child),
            };
            if delete {
                delete_child(
                    tree,
                    profiles,
                    profile,
                    profile_id,
                    current_id,
                    name.as_str(),
                    prior_child,
                    out,
                );
            }
        }
    }

    // Phase 2 — creates and inode-stable Dir-pair recursion (forward lex).
    let prior_entries = prior.map(|p| &p.entries);
    for (name, new_child) in &new.entries {
        let prior_child = prior_entries.and_then(|m| m.get(name));
        let identity_match = prior_child.is_some_and(|p| same_inode_device(p, new_child));

        if !identity_match {
            // Pure create OR delete-then-create (Phase 1 deleted the prior).
            create_child(
                tree,
                profiles,
                profile,
                profile_id,
                current_id,
                name.as_str(),
                new_child,
                out,
            );
            continue;
        }

        // Identity matches; recurse on Dir-Dir, no-op on Leaf-Leaf.
        if let (Some(ChildEntry::Dir(p_dc)), ChildEntry::Dir(n_dc)) = (prior_child, new_child) {
            // Same inode/device on both sides ⇒ same Tree slot. Look it up
            // (it must be live: the prior was observed and inode hasn't
            // flipped). Slot reaped mid-burst is rare; skip gracefully.
            let Some(child_id) = tree.lookup(Some(current_id), name.as_str()) else {
                continue;
            };
            match (p_dc.subtree.as_deref(), n_dc.subtree.as_deref()) {
                (Some(ps), Some(ns)) if ps.dir_hash() != ns.dir_hash() => {
                    walk_pair(
                        Some(ps),
                        ns,
                        child_id,
                        profile,
                        profile_id,
                        tree,
                        profiles,
                        out,
                    );
                }
                (Some(_), Some(_)) | (None, None) => {
                    // Hashes match (covered both sides) or both sides
                    // uncovered: no delta to emit at this Dir slot.
                }
                (Some(_), None) | (None, Some(_)) => {
                    // Coverage flip on the same Dir slot. Structurally
                    // unreachable in v1: a Profile's coverage rule is
                    // pinned by `config_hash`, so a scope change forks a
                    // new Profile rather than flipping subtree presence
                    // at the same slot for the same Profile. The
                    // debug_assert pins the invariant; if a future change
                    // makes it reachable, tests shout.
                    debug_assert!(
                        false,
                        "walk_pair: coverage flip on same Dir slot is unreachable in v1",
                    );
                }
            }
        }
        // Same-inode Leaf-Leaf: content may have changed (caller's diff
        // path emits the per-leaf Effect); no Watch delta because file FDs
        // are bound to slot identity, which is stable across content
        // modifications.
    }
}

/// Splice a probe response subtree into `Profile.current` at `target`,
/// rebuilding `DirSnapshot`s along the path-to-anchor. Emits Watch ops via
/// `walk_pair` against the pre-graft prior. O(1) early-out when hashes
/// match — the post-Effect Seed common case (Effect produced no observable
/// drift) hits this path with zero allocations.
///
/// Single source of truth for "engine just received an Ok response;
/// integrate it into the Profile's view." Used by `dispatch_seed_ok` and
/// `dispatch_standard_ok` alike. Caller handles `baseline` rebasing
/// (different rules for Seed vs Standard).
///
/// **Borrow shape.** Takes `profile_id: ProfileId` (not `&mut Profile`)
/// because `walk_pair`'s `sub_watch_demand` calls need `&ProfileMap`
/// alongside the Profile read. Splitting the &mut Profile borrow at the
/// caller would require two `get_mut` round-trips per graft; threading
/// the id through here lets graft re-borrow under whichever shape each
/// step needs (immutable for `walk_pair`, mutable for the splice write).
pub(crate) fn graft(
    profile_id: ProfileId,
    target: ResourceId,
    response_arc: Arc<DirSnapshot>,
    tree: &mut Tree,
    profiles: &mut ProfileMap,
    out: &mut StepOutput,
) {
    // Identify the prior subtree at this target. None ⇒ Seed first probe,
    // or a path that became covered between probes. Read-only borrow of
    // the Profile; released at the end of the let-block.
    let prior = match profiles.get(profile_id) {
        Some(p) => p.current.as_ref().and_then(|s| s.subtree_at(target, tree)),
        None => return,
    };

    // O(1) early-out: response equals prior at this target ⇒ no Watch
    // ops, no graft, no allocation.
    if let Some(p) = &prior
        && p.dir_hash() == response_arc.dir_hash()
    {
        return;
    }

    // Emit Watch ops + materialise / reap Tree slots. The Profile borrow
    // co-exists with the &ProfileMap shared borrow — both reads, both fine.
    {
        let Some(profile) = profiles.get(profile_id) else {
            return;
        };
        walk_pair(
            prior.as_deref(),
            &response_arc,
            target,
            profile,
            profile_id,
            tree,
            profiles,
            out,
        );
    }

    // walk_pair may have reaped covered descendants via delete_child.
    // PerFile dedup entries keyed at reaped slots become stale — slotmap
    // generations make stale ids non-resolving, so the entries never
    // affect correctness, but they accumulate as memory for the
    // Profile's lifetime. Run the hygiene purge across every Profile
    // (one bursts's deletes can stale entries in *any* covering
    // Profile's map, not just `profile_id`'s).
    purge_per_file_dedup_for_reaped_slots(profiles, tree);

    // Splice the response into current. Rebuilds DirSnapshots along the
    // path from anchor to target, Arc-shares everything off-path,
    // short-circuits when per-level hashes match. Mutable borrow now —
    // the prior and the walk_pair borrow are released.
    //
    // `SpliceResult::CrossedUncovered` flags the engine-contract
    // violation "graft only into observed subtrees" (target outside
    // anchor's tree subtree, or coverage gap on the path-to-target).
    // Carrier is the prior unchanged; the response is intentionally
    // dropped on the floor and the next probe converges. Surface the
    // breach via Diagnostic so operator logs see it instead of a silent
    // information loss.
    let prior_current = profiles.get_mut(profile_id).and_then(|p| p.current.take());
    let result = splice(prior_current, target, response_arc, tree);
    if matches!(result, SpliceResult::CrossedUncovered(_)) {
        out.diagnostics.push(Diagnostic::SpliceCrossedUncovered {
            profile: profile_id,
            target,
        });
    }
    if let Some(p) = profiles.get_mut(profile_id) {
        p.current = Some(result.into_snapshot());
    }
}

/// Drop `last_emitted_dir_hash` entries whose `PerFile` key references a
/// reaped Tree slot. `Subtree`-keyed entries are unaffected — their
/// `(SubId, ProfileId)` key is bounded by Profile lifecycle (the whole
/// map drops at `reap_profile`), and the engine's only natural lifecycle
/// hook for `PerFile` keys is at the Resource level.
///
/// `PerFile` entries become stale when [`delete_child`] reaps a covered
/// leaf via `tree.try_reap`. Slotmap increments the slot's generation on
/// removal, so a stale `ResourceId` never resolves and the stale entry
/// can never collide with a future allocation — this purge is hygiene,
/// not a correctness fix. Without it, the map accumulates unreachable
/// entries proportional to the file-churn rate over the Profile's
/// lifetime.
///
/// Call sites: [`graft`] after `walk_pair` runs, and
/// [`Engine::release_descendant_claim`] after the take-and-walk teardown.
/// Both paths can reap covered leaves; other reap paths (descent rewind,
/// watch-root parent release) target Dir / scaffold slots which are never
/// `PerFile` targets, so those paths don't need this hook.
///
/// Cost: `O(profiles × dedup_size_per_profile)` per call. Typical v1
/// configs are 1–2 profiles with small dedup maps; the cost is
/// negligible compared to the rest of the graft work.
pub(crate) fn purge_per_file_dedup_for_reaped_slots(profiles: &mut ProfileMap, tree: &Tree) {
    for (_, p) in profiles.iter_mut() {
        if p.last_emitted_dir_hash.is_empty() {
            continue;
        }
        p.last_emitted_dir_hash.retain(|k, _| match *k {
            DedupKey::PerFile { resource, .. } => tree.get(resource).is_some(),
            DedupKey::Subtree { .. } => true,
        });
    }
}

/// Extract the `dir_hash` of `current.subtree_at(target)` or `current` itself
/// for File-anchored Profiles. Returns `None` when there is no prior
/// observation at `target` (covered-in-this-probe path; treat as not-stable).
pub(crate) fn current_target_hash(
    profile: &Profile,
    target: ResourceId,
    tree: &Tree,
) -> Option<u128> {
    match profile.current.as_ref()? {
        TreeSnapshot::Dir(_) => profile
            .current
            .as_ref()
            .and_then(|s| s.subtree_at(target, tree))
            .map(|s| s.dir_hash()),
        TreeSnapshot::File(leaf) => Some(leaf.leaf_hash()),
    }
}

/// Reap the Tree slot at `(parent, name)` — and, recursively, every
/// descendant of `prior_child` if it's a Dir. Watch contributions are
/// released on the 1→0 edge for covered Dirs (always) and covered Leaves
/// under `profile.has_per_file_fds` (B2). Idempotent on missing slots.
///
/// Reverse-lex within each Dir's children (leaves before parents) so the
/// `try_reap`-after-`vacate` gate sees a fully-drained slot at every
/// level.
///
/// `sub_watch_demand` threads `profile.events_union` as the contribution
/// (R2) and `Some(profile_id)` as the explicit `releasing_descendant`
/// signal: during graft `Profile.current` is still `Some` while this
/// helper runs, so the recompute would otherwise count this Profile's
/// own descendant claim toward the post-decrement union (F-MED-4). The
/// param skips that contribution precisely.
///
/// `pub(crate)` so [`Engine::release_descendant_claim`] can reuse the
/// walk for whole-snapshot teardown — single source of truth for "walk
/// covered descendants and release their `watch_demand`."
pub(crate) fn delete_child(
    tree: &mut Tree,
    profiles: &ProfileMap,
    profile: &Profile,
    profile_id: ProfileId,
    parent: ResourceId,
    name: &str,
    prior_child: &ChildEntry,
    out: &mut StepOutput,
) {
    let Some(resource) = tree.lookup(Some(parent), name) else {
        return;
    };

    // Phase 1: recurse into the Dir's prior subtree first to reap leaf
    // slots before the parent's slot. Leaf-only `prior_child`s skip the
    // recurse.
    if let ChildEntry::Dir(dc) = prior_child
        && let Some(sub) = dc.subtree.as_deref()
    {
        for (cname, cchild) in sub.entries.iter().rev() {
            delete_child(
                tree,
                profiles,
                profile,
                profile_id,
                resource,
                cname.as_str(),
                cchild,
                out,
            );
        }
    }

    // Phase 2: release this slot's watch contribution if we hold one.
    // The counter-existence guard (`watch_demand > 0`) makes the helper
    // safe over the take-then-walk path of `release_descendant_claim`,
    // where multiple sub-walks may converge on a slot the previous
    // iteration already drained.
    let releases_watch = match prior_child.kind() {
        EntryKind::Dir => covers(profile, resource, tree),
        EntryKind::File | EntryKind::Symlink | EntryKind::Other => {
            covers(profile, resource, tree) && profile.has_per_file_fds
        }
    };
    if releases_watch && tree.get(resource).is_some_and(|r| r.watch_demand > 0) {
        sub_watch_demand(
            tree,
            profiles,
            resource,
            profile.events_union,
            Some(profile_id),
            out,
        );
    }

    // Reap only when fully drained — preserves multi-Profile contributions.
    if tree.get(resource).is_some_and(|r| r.watch_demand == 0) {
        tree.vacate(resource);
        tree.try_reap(resource);
    }
}

/// Materialise the Tree slot at `(parent, name)` — and, recursively,
/// every descendant of `new_child` when it's a Dir whose subtree was
/// observed. Emits `add_watch_demand` on the 0→1 edge (or on a mask
/// widening) for covered Dirs (always) and covered Leaves under
/// `profile.has_per_file_fds` (B2).
///
/// `add_watch_demand` threads `profile.events_union` as the per-Resource
/// mask contribution (R2). The recursive `walk_pair` call into the new
/// Dir's subtree threads `profiles` through so its own `delete_child` /
/// `create_child` recursion is well-formed; on a pure-create walk
/// (`prior == None`) `delete_child` doesn't fire and the borrow is just
/// a structural carrier, but threading it keeps the signature uniform
/// across both `walk_pair` entry points.
fn create_child(
    tree: &mut Tree,
    profiles: &ProfileMap,
    profile: &Profile,
    profile_id: ProfileId,
    parent: ResourceId,
    name: &str,
    new_child: &ChildEntry,
    out: &mut StepOutput,
) {
    let res_kind = match new_child.kind() {
        EntryKind::Dir => ResourceKind::Dir,
        _ => ResourceKind::File,
    };
    let resource = tree.ensure(Some(parent), name, ResourceRole::User);
    if let Some(res) = tree.get_mut(resource) {
        res.kind = res_kind;
    }

    if covers(profile, resource, tree) {
        let need_watch = matches!(new_child, ChildEntry::Dir(_)) || profile.has_per_file_fds;
        if need_watch {
            add_watch_demand(tree, resource, profile.events_union, out);
        }
    }

    // Recurse into the freshly-created Dir's subtree if the walker
    // observed one. `prior=None` ⇒ every descendant is a creation; the
    // same `walk_pair` call.
    if let ChildEntry::Dir(dc) = new_child
        && let Some(sub) = dc.subtree.as_deref()
    {
        walk_pair(
            None,
            sub,
            resource,
            profile,
            profile_id,
            tree,
            profiles,
            out,
        );
    }
}

const fn same_inode_device(a: &ChildEntry, b: &ChildEntry) -> bool {
    a.inode() == b.inode() && a.device() == b.device()
}

#[cfg(test)]
#[allow(clippy::needless_pass_by_value)]
mod tests {
    use super::{graft, walk_pair};
    use compact_str::CompactString;
    use specter_core::{
        ChildEntry, ClassSet, DedupKey, DirChild, DirMeta, DirSnapshot, EntryKind, LeafEntry,
        Profile, ProfileMap, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, SubId,
        Tree, TreeSnapshot, WatchOp,
    };
    use std::collections::BTreeMap;
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    // ---------------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------------

    fn meta(inode: u64) -> DirMeta {
        DirMeta {
            mtime: UNIX_EPOCH,
            inode,
            device: 0,
        }
    }

    fn leaf(kind: EntryKind, inode: u64) -> ChildEntry {
        ChildEntry::Leaf(LeafEntry::new(kind, 0, UNIX_EPOCH, inode, 0))
    }

    fn dir_child(inode: u64, subtree: Option<Arc<DirSnapshot>>) -> ChildEntry {
        ChildEntry::Dir(DirChild {
            inode,
            device: 0,
            subtree,
        })
    }

    fn dir_snap(
        resource: ResourceId,
        inode: u64,
        entries: Vec<(&str, ChildEntry)>,
    ) -> Arc<DirSnapshot> {
        let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        for (name, child) in entries {
            map.insert(CompactString::new(name), child);
        }
        Arc::new(DirSnapshot::new(resource, meta(inode), 0, map))
    }

    fn anchor(per_file: bool) -> (Tree, ProfileMap, ResourceId, specter_core::ProfileId) {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure(None, "root", ResourceRole::User);
        tree.get_mut(r).unwrap().kind = ResourceKind::Dir;
        // `events` chooses the per-leaf gating: CONTENT (or METADATA) ⇒
        // `has_per_file_fds = true` ⇒ Leaves get watch_demand contributions.
        // STRUCTURE-only ⇒ Dir-only contributions.
        let events = if per_file {
            ClassSet::CONTENT
        } else {
            ClassSet::STRUCTURE
        };
        let pid = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                events,
            ),
        );
        // The covered descendants reconciler reads `has_per_file_fds`
        // off the Profile; `Profile::new` already derives it from the
        // events mask, but this test still depends on the constructor's
        // contract — assert it inline so a future change shouts.
        debug_assert_eq!(profiles.get(pid).unwrap().has_per_file_fds, per_file);
        (tree, profiles, r, pid)
    }

    fn count_watch(out: &StepOutput) -> usize {
        out.watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Watch { .. }))
            .count()
    }

    fn count_unwatch(out: &StepOutput) -> usize {
        out.watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
            .count()
    }

    // ---------------------------------------------------------------------------
    // walk_pair — empty-prior synthesis
    // ---------------------------------------------------------------------------

    #[test]
    fn walk_pair_prior_none_creates_dir_and_leaf_entries() {
        // prior=None ⇒ every entry of `new` is a creation. Subtree-only
        // Profile (has_per_file_fds=false) ⇒ Dir gets Watch, Leaf does not.
        let (mut tree, profiles, root, pid) = anchor(false);
        let new = dir_snap(
            root,
            100,
            vec![
                ("a.rs", leaf(EntryKind::File, 1)),
                ("sub", dir_child(2, None)),
            ],
        );
        let mut out = StepOutput::default();
        walk_pair(
            None,
            &new,
            root,
            profiles.get(pid).unwrap(),
            pid,
            &mut tree,
            &profiles,
            &mut out,
        );
        assert_eq!(count_watch(&out), 1, "one Watch for the Dir creation");
        assert!(tree.lookup(Some(root), "a.rs").is_some());
        assert!(tree.lookup(Some(root), "sub").is_some());
    }

    #[test]
    fn walk_pair_per_file_profile_emits_watch_for_leaf_create() {
        // has_per_file_fds=true ⇒ Leaf creates *also* get Watch (B2).
        let (mut tree, profiles, root, pid) = anchor(true);
        let new = dir_snap(
            root,
            100,
            vec![
                ("a.rs", leaf(EntryKind::File, 1)),
                ("sub", dir_child(2, None)),
            ],
        );
        let mut out = StepOutput::default();
        walk_pair(
            None,
            &new,
            root,
            profiles.get(pid).unwrap(),
            pid,
            &mut tree,
            &profiles,
            &mut out,
        );
        assert_eq!(
            count_watch(&out),
            2,
            "Watch for both Dir and Leaf under per-file"
        );
    }

    // ---------------------------------------------------------------------------
    // walk_pair — equal-hash short-circuit
    // ---------------------------------------------------------------------------

    #[test]
    fn walk_pair_equal_dir_hash_short_circuits() {
        // Identical prior and new ⇒ no Watch ops, no Tree mutations.
        let (mut tree, profiles, root, pid) = anchor(false);
        let prior = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        let new = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        assert_eq!(prior.dir_hash(), new.dir_hash());
        let mut out = StepOutput::default();
        walk_pair(
            Some(&prior),
            &new,
            root,
            profiles.get(pid).unwrap(),
            pid,
            &mut tree,
            &profiles,
            &mut out,
        );
        assert_eq!(count_watch(&out), 0);
        assert_eq!(count_unwatch(&out), 0);
    }

    // ---------------------------------------------------------------------------
    // walk_pair — deletions release Watch on covered Dir
    // ---------------------------------------------------------------------------

    #[test]
    fn walk_pair_dir_deletion_releases_watch() {
        // Set up: prior has a covered Dir; new doesn't. Delete should release
        // Watch on the Dir.
        let (mut tree, profiles, root, pid) = anchor(false);
        // Materialise the Dir slot first so the delete path has something to
        // sub_watch_demand against. Pre-populate with the same contribution
        // walk_pair will drop on delete (the Profile's events_union, here
        // STRUCTURE).
        let sub_id = tree.ensure(Some(root), "sub", ResourceRole::User);
        tree.get_mut(sub_id).unwrap().kind = ResourceKind::Dir;
        crate::refcounts::add_watch_demand(
            &mut tree,
            sub_id,
            ClassSet::STRUCTURE,
            &mut StepOutput::default(),
        );

        let prior = dir_snap(root, 100, vec![("sub", dir_child(2, None))]);
        let new = dir_snap(root, 200, vec![]); // bumped root meta to force descent
        let mut out = StepOutput::default();
        walk_pair(
            Some(&prior),
            &new,
            root,
            profiles.get(pid).unwrap(),
            pid,
            &mut tree,
            &profiles,
            &mut out,
        );
        assert_eq!(count_unwatch(&out), 1, "Unwatch for the deleted Dir");
    }

    #[test]
    fn walk_pair_per_file_leaf_deletion_releases_watch() {
        // PerStableFile Profile (has_per_file_fds=true): covered Leaf
        // carries a Watch contribution; delete should release it (B2
        // symmetric to create).
        let (mut tree, profiles, root, pid) = anchor(true);
        let leaf_id = tree.ensure(Some(root), "a.rs", ResourceRole::User);
        tree.get_mut(leaf_id).unwrap().kind = ResourceKind::File;
        crate::refcounts::add_watch_demand(
            &mut tree,
            leaf_id,
            ClassSet::CONTENT,
            &mut StepOutput::default(),
        );

        let prior = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        let new = dir_snap(root, 200, vec![]);
        let mut out = StepOutput::default();
        walk_pair(
            Some(&prior),
            &new,
            root,
            profiles.get(pid).unwrap(),
            pid,
            &mut tree,
            &profiles,
            &mut out,
        );
        assert_eq!(count_unwatch(&out), 1);
    }

    // ---------------------------------------------------------------------------
    // walk_pair — same-name different-inode is delete-then-create
    // ---------------------------------------------------------------------------

    #[test]
    fn walk_pair_inode_change_deletes_then_creates() {
        // `prior.entries["foo"]` has inode 1; `new.entries["foo"]` has inode
        // 2 (delete-then-create). Both Dirs ⇒ Watch released for the deleted,
        // re-emitted for the created. Net: one Unwatch + one Watch.
        let (mut tree, profiles, root, pid) = anchor(false);
        // Pre-materialise the prior Dir slot with the matching mask.
        let foo_id = tree.ensure(Some(root), "foo", ResourceRole::User);
        tree.get_mut(foo_id).unwrap().kind = ResourceKind::Dir;
        crate::refcounts::add_watch_demand(
            &mut tree,
            foo_id,
            ClassSet::STRUCTURE,
            &mut StepOutput::default(),
        );

        let prior = dir_snap(root, 100, vec![("foo", dir_child(1, None))]);
        let new = dir_snap(root, 200, vec![("foo", dir_child(2, None))]);
        let mut out = StepOutput::default();
        walk_pair(
            Some(&prior),
            &new,
            root,
            profiles.get(pid).unwrap(),
            pid,
            &mut tree,
            &profiles,
            &mut out,
        );
        assert_eq!(count_unwatch(&out), 1);
        assert_eq!(count_watch(&out), 1);
    }

    // ---------------------------------------------------------------------------
    // graft — equal-hash early-out (no allocations, no Watch ops)
    // ---------------------------------------------------------------------------

    #[test]
    fn graft_equal_hash_short_circuits() {
        // current.subtree_at(target).dir_hash() == response.dir_hash() ⇒
        // zero allocations, zero Watch ops, current unchanged.
        let (mut tree, mut profiles, root, pid) = anchor(false);
        let snap_a = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        profiles.get_mut(pid).unwrap().current = Some(TreeSnapshot::Dir(Arc::clone(&snap_a)));
        let response = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);

        let prior_arc_count = Arc::strong_count(&snap_a);
        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );
        assert_eq!(count_watch(&out), 0);
        // current Arc is unchanged (still pointing at snap_a — the early-out
        // bypassed splice's path-rebuild).
        let post_arc_count = Arc::strong_count(&snap_a);
        assert!(
            post_arc_count >= prior_arc_count,
            "graft early-out doesn't drop current's Arc",
        );
    }

    #[test]
    fn graft_writes_current_at_target() {
        // No prior subtree at target ⇒ graft splices wholesale.
        let (mut tree, mut profiles, root, pid) = anchor(false);
        let response = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );
        let p = profiles.get(pid).unwrap();
        assert!(p.current.is_some());
        let current = p.current.as_ref().unwrap();
        let TreeSnapshot::Dir(arc) = current else {
            panic!("expected Dir snapshot");
        };
        assert_eq!(arc.dir_hash(), response.dir_hash());
    }

    // ---------------------------------------------------------------------------
    // graft — last_emitted_dir_hash lifecycle (PR 3)
    // ---------------------------------------------------------------------------

    #[test]
    fn graft_purges_per_file_dedup_when_descendant_is_reaped() {
        // Setup: per-file Profile (has_per_file_fds=true) covers root with
        // a covered descendant `a.rs`. Pre-populate `last_emitted_dir_hash`
        // with a PerFile entry keyed at `a.rs`'s ResourceId — simulates a
        // prior PerStableFile Effect having fired against the leaf.
        // Probe response deletes `a.rs`. graft's walk_pair → delete_child
        // reaps the slot; the new purge hook drops the now-stale
        // PerFile entry.
        let (mut tree, mut profiles, root, pid) = anchor(true);
        let a_rs_id = tree.ensure(Some(root), "a.rs", ResourceRole::User);
        tree.get_mut(a_rs_id).unwrap().kind = ResourceKind::File;

        // Prior current = root with a.rs as covered leaf; baseline matches
        // so the diff has no ops other than the delete.
        let prior_current = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        profiles.get_mut(pid).unwrap().current =
            Some(TreeSnapshot::Dir(Arc::clone(&prior_current)));

        // Pre-populate the dedup map. SubId::default() is fine — the
        // purge filter doesn't inspect the sub field. The `profile` field
        // mirrors what production would store: the Profile that owns
        // `last_emitted_dir_hash`.
        let stale_key = DedupKey::PerFile {
            sub: SubId::default(),
            profile: pid,
            resource: a_rs_id,
        };
        profiles
            .get_mut(pid)
            .unwrap()
            .last_emitted_dir_hash
            .insert(stale_key.clone(), 0xdead_beef_u128);

        // a.rs needs a watch_demand contribution so delete_child's
        // sub_watch_demand path is reachable; the per-file Profile would
        // have placed one there at attach via reconcile's create_child
        // pass. Simulate that here directly.
        crate::refcounts::add_watch_demand(
            &mut tree,
            a_rs_id,
            ClassSet::CONTENT,
            &mut StepOutput::default(),
        );

        // Probe response: a.rs is gone.
        let response = dir_snap(root, 200, vec![]);

        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );

        assert!(
            tree.get(a_rs_id).is_none(),
            "delete_child must have reaped a.rs slot",
        );
        let p = profiles.get(pid).unwrap();
        assert!(
            !p.last_emitted_dir_hash.contains_key(&stale_key),
            "stale PerFile entry must be purged after slot reap",
        );
    }

    #[test]
    fn graft_preserves_per_file_dedup_for_live_descendants() {
        // Complement: slots that survive graft retain their dedup
        // entries. a.rs's content changes (different leaf hash) but the
        // slot persists — the entry must NOT be purged.
        let (mut tree, mut profiles, root, pid) = anchor(true);
        let a_rs_id = tree.ensure(Some(root), "a.rs", ResourceRole::User);
        tree.get_mut(a_rs_id).unwrap().kind = ResourceKind::File;

        let prior_current = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        profiles.get_mut(pid).unwrap().current =
            Some(TreeSnapshot::Dir(Arc::clone(&prior_current)));

        let live_key = DedupKey::PerFile {
            sub: SubId::default(),
            profile: pid,
            resource: a_rs_id,
        };
        profiles
            .get_mut(pid)
            .unwrap()
            .last_emitted_dir_hash
            .insert(live_key.clone(), 0x1234_5678_u128);

        // Response: same a.rs name+inode (no delete). Bumped root mtime
        // forces graft to walk the level rather than equal-hash early-out.
        let response = dir_snap(root, 200, vec![("a.rs", leaf(EntryKind::File, 1))]);

        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );

        assert!(
            tree.get(a_rs_id).is_some(),
            "a.rs slot still live (no delete in response)",
        );
        let p = profiles.get(pid).unwrap();
        assert!(
            p.last_emitted_dir_hash.contains_key(&live_key),
            "live PerFile entry must be preserved across graft",
        );
    }

    #[test]
    fn graft_preserves_subtree_dedup_unconditionally() {
        // Subtree-keyed entries are bounded by Profile lifecycle, not
        // Resource lifecycle. The purge hook must leave them alone even
        // when arbitrary descendants get reaped.
        use specter_core::ProfileId;

        let (mut tree, mut profiles, root, pid) = anchor(false);
        let a_rs_id = tree.ensure(Some(root), "a.rs", ResourceRole::User);
        tree.get_mut(a_rs_id).unwrap().kind = ResourceKind::File;

        let prior_current = dir_snap(root, 100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        profiles.get_mut(pid).unwrap().current =
            Some(TreeSnapshot::Dir(Arc::clone(&prior_current)));

        // Subtree key references a *Profile*, not a Resource. SubId
        // and ProfileId values are arbitrary for this test.
        let subtree_key = DedupKey::Subtree {
            sub: SubId::default(),
            profile: ProfileId::default(),
        };
        profiles
            .get_mut(pid)
            .unwrap()
            .last_emitted_dir_hash
            .insert(subtree_key.clone(), 0xfeed_face_u128);

        // Probe response deletes a.rs. The reap fires; the purge runs;
        // the Subtree entry should survive.
        let response = dir_snap(root, 200, vec![]);
        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );

        let p = profiles.get(pid).unwrap();
        assert!(
            p.last_emitted_dir_hash.contains_key(&subtree_key),
            "Subtree-keyed entries are bounded by Profile lifecycle, \
             not Resource lifecycle — purge must leave them alone",
        );
    }

    #[test]
    fn graft_purges_per_file_dedup_across_all_profiles() {
        // Multi-Profile case: a single FsEvent cascade reaps a covered
        // descendant. Both Profiles' dedup maps may carry PerFile entries
        // at the reaped slot; the purge runs across every Profile, not
        // just the one being grafted.
        let (mut tree, mut profiles, root, pid_a) = anchor(true);
        // Second User Profile sharing the same anchor at a different
        // config_hash. `max_settle` is part of `config_hash`, so a
        // different value forks a distinct Profile.
        let pid_b = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ScanConfig::builder().recursive(true).build(),
                Duration::from_secs(12),
                Duration::from_millis(50),
                ClassSet::CONTENT,
            ),
        );
        let a_rs_id = tree.ensure(Some(root), "a.rs", ResourceRole::User);
        tree.get_mut(a_rs_id).unwrap().kind = ResourceKind::File;

        // Both Profiles record a PerFile entry against a.rs. Each entry's
        // `profile` field is the owning Profile's own id — mirrors what
        // `emit_effects_per_stable_file` writes in production.
        for &pid in &[pid_a, pid_b] {
            profiles.get_mut(pid).unwrap().current = Some(TreeSnapshot::Dir(dir_snap(
                root,
                100,
                vec![("a.rs", leaf(EntryKind::File, 1))],
            )));
            profiles.get_mut(pid).unwrap().last_emitted_dir_hash.insert(
                DedupKey::PerFile {
                    sub: SubId::default(),
                    profile: pid,
                    resource: a_rs_id,
                },
                0x00c0_ffee_u128,
            );
        }

        // Match the per-file Profile's contribution on the slot.
        crate::refcounts::add_watch_demand(
            &mut tree,
            a_rs_id,
            ClassSet::CONTENT,
            &mut StepOutput::default(),
        );

        let response = dir_snap(root, 200, vec![]);
        let mut out = StepOutput::default();
        graft(
            pid_a,
            root,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );

        assert!(tree.get(a_rs_id).is_none(), "a.rs reaped");
        // Each Profile's stale key carries its own `profile` field (matching
        // what was inserted above), so we look up per Profile.
        let stale_key_a = DedupKey::PerFile {
            sub: SubId::default(),
            profile: pid_a,
            resource: a_rs_id,
        };
        let stale_key_b = DedupKey::PerFile {
            sub: SubId::default(),
            profile: pid_b,
            resource: a_rs_id,
        };
        assert!(
            !profiles
                .get(pid_a)
                .unwrap()
                .last_emitted_dir_hash
                .contains_key(&stale_key_a),
            "Profile A's stale entry must be purged",
        );
        assert!(
            !profiles
                .get(pid_b)
                .unwrap()
                .last_emitted_dir_hash
                .contains_key(&stale_key_b),
            "Profile B's stale entry (cross-Profile) must also be purged",
        );
    }

    // ---------------------------------------------------------------------------
    // graft — CrossedUncovered surfaces a Diagnostic
    // ---------------------------------------------------------------------------

    #[test]
    fn graft_emits_diagnostic_when_path_crosses_uncovered_intermediate() {
        // Setup: Tree has root → a → b (all live slots). The Profile's
        // `current` snapshot is anchored at root with entry "a" carrying
        // `subtree=None` (uncovered — engine never observed below "a").
        // A probe response arrives for target `b`. Splice navigates the
        // tree chain anchor→a→b successfully but cannot navigate the
        // *snapshot*'s coverage path — `prior.entries["a"].subtree`
        // is None — so it returns CrossedUncovered. Graft must emit
        // Diagnostic::SpliceCrossedUncovered AND keep the prior `current`
        // unchanged so the anchor-rooted invariant on `Profile.current`
        // is preserved.
        let (mut tree, mut profiles, root, pid) = anchor(false);
        let a_id = tree.ensure(Some(root), "a", ResourceRole::User);
        tree.get_mut(a_id).unwrap().kind = ResourceKind::Dir;
        let b_id = tree.ensure(Some(a_id), "b", ResourceRole::User);
        tree.get_mut(b_id).unwrap().kind = ResourceKind::Dir;

        let prior_current = dir_snap(root, 100, vec![("a", dir_child(2, None))]);
        profiles.get_mut(pid).unwrap().current =
            Some(TreeSnapshot::Dir(Arc::clone(&prior_current)));

        let response = dir_snap(b_id, 200, vec![]);

        let mut out = StepOutput::default();
        graft(
            pid,
            b_id,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
        );

        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::SpliceCrossedUncovered { profile, target }
                if *profile == pid && *target == b_id
            )
        });
        assert!(
            has_diag,
            "graft must emit SpliceCrossedUncovered when splice can't \
             navigate the snapshot's coverage path",
        );

        let p = profiles.get(pid).unwrap();
        let TreeSnapshot::Dir(current_arc) = p.current.as_ref().unwrap() else {
            panic!("expected Dir snapshot");
        };
        assert!(
            Arc::ptr_eq(current_arc, &prior_current),
            "graft kept prior current unchanged on CrossedUncovered",
        );
    }
}
