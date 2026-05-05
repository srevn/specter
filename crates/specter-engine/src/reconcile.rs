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
    ChildEntry, DirSnapshot, EntryKind, Profile, ProfileId, ProfileMap, ResourceId, ResourceKind,
    ResourceRole, StepOutput, Tree, TreeSnapshot, splice,
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
/// recompute.
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
                    walk_pair(Some(ps), ns, child_id, profile, tree, profiles, out);
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
            tree,
            profiles,
            out,
        );
    }

    // Splice the response into current. Rebuilds DirSnapshots along the
    // path from anchor to target, Arc-shares everything off-path,
    // short-circuits when per-level hashes match. Mutable borrow now —
    // the prior and the walk_pair borrow are released.
    let prior_current = profiles.get_mut(profile_id).and_then(|p| p.current.take());
    let new_current = splice(prior_current, target, response_arc, tree);
    if let Some(p) = profiles.get_mut(profile_id) {
        p.current = Some(new_current);
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
/// (R2). The recompute path inside `sub_watch_demand` (multi-contributor
/// case only) walks the registry; v1 doesn't track per-(Profile,
/// Resource) contributions for descendants, so a transient over-mask is
/// possible during release — accepted, the next refcount op converges.
fn delete_child(
    tree: &mut Tree,
    profiles: &ProfileMap,
    profile: &Profile,
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
                resource,
                cname.as_str(),
                cchild,
                out,
            );
        }
    }

    // Phase 2: release this slot's watch contribution if we hold one.
    let releases_watch = match prior_child.kind() {
        EntryKind::Dir => covers(profile, resource, tree),
        EntryKind::File | EntryKind::Symlink | EntryKind::Other => {
            covers(profile, resource, tree) && profile.has_per_file_fds
        }
    };
    if releases_watch {
        sub_watch_demand(tree, profiles, resource, profile.events_union, out);
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
        walk_pair(None, sub, resource, profile, tree, profiles, out);
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
        ChildEntry, ClassSet, DirChild, DirMeta, DirSnapshot, EntryKind, LeafEntry, Profile,
        ProfileMap, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, Tree,
        TreeSnapshot, WatchOp,
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
}
