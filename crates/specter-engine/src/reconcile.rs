//! Covered-descendants reconciliation via `Diff`.
//!
//! Every `ProbeResponse(Ok(snapshot))` for a Dir-shaped Profile flows through [`graft`]: splice the
//! response into `Profile.current`, then apply the resulting [`Diff`] to the engine's `Tree` via
//! [`apply_diff_to_tree`]. The Diff is the **single representation of a snapshot change** consumed
//! inside the engine — the same `diff_tree` algorithm core uses for Effect dispatch is the only
//! classifier that mutates Tree state.
//!
//! For each delta entry, [`apply_diff_to_tree`] ensures (or releases) the Tree slot, sets its kind,
//! and installs or releases the per-Resource [`ContribKey::ProfileDescendant`] contribution via
//! [`add_watch`] / [`crate::refcounts::sub_watch`] for covered Dirs (always) and covered Leaves
//! under `Profile.has_per_file_fds`. The Watch ops appear in `StepOutput.watch_ops`, resealed by
//! `ResourceId` at emission.
//!
//! **Empty-prior path.** When `Profile.current` is `None` — the Seed burst's first probe —
//! [`graft`] synthesises the Diff via [`Diff::all_created`](specter_core::Diff::all_created) on the
//! response. Equivalent to diffing against an empty `DirSnapshot`, no empty-snapshot allocation.
//!
//! **Splice-first ordering.** [`graft`] runs `splice` *before* any Tree mutation.
//! [`SpliceResult::CrossedUncovered`] (a contract breach in v1) surfaces a [`Diagnostic`] and
//! short-circuits without touching Tree state — `Profile.current` and `Tree` stay coherent across
//! the breach.
//!
//! **Two-phase reaping and ordering.** [`apply_diff_to_tree`] runs:
//! 1. Phase 1 — `diff.deleted` and `diff.renamed.from`, in reverse iteration. Releases this
//!    Profile's contribution at each slot; [`Tree::try_reap`] reclaims any slot left with no
//!    anchors. Reverse-lex within each list is **performance / cleanliness**, not correctness —
//!    `try_reap` cascades upward through any parent that loses its last anchor, so
//!    leaf-before-parent only avoids the cascade work, it does not enable the reap.
//! 2. Phase 2 — `diff.created` and `diff.renamed.to`, in forward iteration. Ensures each slot under
//!    its rel-path, sets the kind, and installs the contribution if covered.
//!
//! Phase-1-before-Phase-2 is load-bearing for the **same-segment kind change** case (`rm foo
//! (File)` then `mkdir foo (Dir)`): `diff_tree` stages both delete and create with `pair_eligible:
//! false`. Phase 1 reaps the prior slot (generation-incremented on remove); Phase 2's
//! `ensure_child` returns a fresh slot at the new generation. If we processed creations first, the
//! slot would be re-typed to Dir, and the deletion pass would look up the same slot (now Dir) and
//! emit Unwatch — silently breaking the new directory's watch.
//!
//! **Reap discipline.** [`Tree::try_reap`] is gated by the slot's `has_anchors`, so the multi-Profile
//! case (where another Profile still contributes) does not prematurely tear down a still-live slot.
//!
//! **File materialization vs Watch.** Every covered diff entry gets a Tree slot —
//! [`ensure_descendant`] runs unconditionally — so a `PerStableFile` Effect always resolves its
//! diff entry to a real `ResourceId` (`emit_effects_per_stable_file` walks the same Diff at burst
//! end). The Watch op (`add_watch`) is gated independently: covered Dirs always; covered Leaves
//! only when `Profile.has_per_file_fds == true`.

use crate::coverage::covers;
use crate::refcounts::{add_watch, sub_watch_then_try_reap};
use specter_core::{
    ContribKey, Diagnostic, Diff, DirSnapshot, EntryKind, Profile, ProfileId, ProfileMap,
    ResourceId, ResourceKind, ResourceRole, SpliceResult, StepOutput, Tree, TreeSnapshot,
    diff_dir_pair, splice, subtree_at_dir,
};
use std::path::PathBuf;
use std::sync::Arc;

/// Walk `rel_path` component-by-component beneath `anchor`, ensuring each slot. Sets the leaf's
/// `kind` to `leaf_kind`; intermediate components default to `ResourceKind::Dir` only when freshly
/// created (kind was `Unknown`). Returns `None` if `rel_path` is empty (a degenerate case reachable
/// only via spec-violating inputs).
///
/// **Multi-component segments.** `Diff` entries may carry multi-segment rel-paths like
/// `subdir/file.txt` for recursive Profiles. `ensure_child` is single-component by design
/// (`(parent, segment)` is the slot identity); this helper walks each segment in lock-step. The
/// Diff's depth-first lex order over `parent/child` segments ensures parents are processed before
/// their descendants in Phase 2, so intermediate slots typically exist before their descendants are
/// touched — but the helper is robust to out-of-order entries (e.g. `renamed.to` segments) because
/// each component is `ensure_child`d on the way through.
///
/// **Kind refresh.** The leaf is unconditionally `set_kind`-ed to `leaf_kind` even on a
/// pre-existing slot. The same-segment kind-flip case (a `File` becoming a `Dir` under kernel inode
/// reuse) is handled
/// upstream by `diff_tree`'s `_ =>` arm in `diff_same_name`: both the
/// prior File and the new Dir are staged with `pair_eligible: false`, Phase 1 reaps the prior slot
/// (generation increment), and Phase 2's `ensure_child` returns a fresh-generation slot whose kind
/// this call then sets to `Dir`. The leaf-kind write is idempotent for the common (kind-stable)
/// case and load-bearing for the kind-flip case.
///
/// `pub(crate)` so `transitions::emit_effects_per_stable_file` can reuse the same materialization
/// rules — the diff-entry-to-Resource mapping must agree with what reconcile produced.
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
        cur = tree
            .ensure_child(cur, comp, ResourceRole::User)
            .expect("cur is the anchor or just minted by a prior loop iteration");
        let is_leaf = comps.peek().is_none();
        if is_leaf {
            tree.set_kind(cur, leaf_kind);
        } else if tree.get(cur).is_some_and(|r| r.kind().is_none()) {
            // Intermediate path component: default to Dir when first created (kind is Unknown /
            // unprobed). Don't override an existing kind — pending-path scaffolding stamps `Dir`
            // already.
            tree.set_kind(cur, ResourceKind::Dir);
        }
    }
    Some(cur)
}

/// Walk `rel_path` component-by-component to find an existing slot. Returns `None` if any segment
/// fails to resolve OR if `rel_path` contains no non-empty segments (symmetric with
/// [`ensure_descendant`] — neither helper degrades to "return the anchor" for an empty relative
/// path).
///
/// `pub(crate)` so descendant-aware sites outside reconcile (e.g.,
/// `transitions::emit_effects_per_stable_file`, `descent::dispatch_descent_ok`) can resolve a
/// relative path under an anchor without duplicating the component-walk.
///
/// **Empty-`rel_path` discipline.** `Diff` entry segments (`EntryRef.segment: CompactString`) are
/// non-empty by walker construction in v1, but the type admits empty. Returning `Some(anchor)` on
/// the empty case would cascade into [`apply_diff_to_tree`]'s Phase 1 issuing a `sub_watch +
/// try_reap` at the anchor itself — risking anchor reap from a malformed diff entry. The `peek`
/// short-circuit removes that hazard at the entry gate.
pub(crate) fn lookup_descendant(
    tree: &Tree,
    anchor: ResourceId,
    rel_path: &str,
) -> Option<ResourceId> {
    let mut comps = rel_path.split('/').filter(|s| !s.is_empty()).peekable();
    comps.peek()?;
    let mut cur = anchor;
    for comp in comps {
        cur = tree.lookup(Some(cur), comp)?;
    }
    Some(cur)
}

// ---------------------------------------------------------------------------
// reconcile: apply_diff_to_tree + graft + scoped purge
// ---------------------------------------------------------------------------

/// Predicate: this Profile holds a [`ContribKey::ProfileDescendant`] contribution at `r` that the
/// entry's `kind` says should be released.
///
/// - **Dir**: covered Dirs always carry the contribution.
/// - **Leaf** (File / Symlink / Other): covered Leaves carry the contribution iff
///   `profile.has_per_file_fds()` is true.
///
/// Mirrors the gating used by [`apply_diff_to_tree`]'s Phase 2 `add_watch` site, so a contribution
/// installed during a prior probe response is released by the symmetric `sub_watch` on this one.
fn releases_watch(
    profile: &Profile,
    r: ResourceId,
    kind: EntryKind,
    tree: &Tree,
    scratch: &mut PathBuf,
) -> bool {
    match kind {
        EntryKind::Dir => covers(profile, r, tree, scratch),
        EntryKind::File | EntryKind::Symlink | EntryKind::Other => {
            covers(profile, r, tree, scratch) && profile.has_per_file_fds()
        }
    }
}

/// Apply a [`Diff`] to the engine's `Tree` for one Profile. Side-effecting only — no return value.
///
/// The `base` parameter is the Tree node the diff's rel-paths are relative to. For [`graft`], `base
/// == target` (the probe's target). For `Engine::release_descendant_claim`, `base ==
/// profile.resource` (the anchor) and the diff is [`Diff::all_deleted`]. `base` is live at every
/// call site — graft's `target` is `ancestor_chain`-proven (or the anchor itself); claims' `base`
/// is the still-attached anchor (`take_current` ran under a live Profile, so the slot still has
/// anchors) — which is why [`ensure_descendant`]'s `cur` `.expect` below is sound (`cur` starts at
/// `base` or a just-minted child).
///
/// **Two phases.**
///
/// 1. **Phase 1 — deletes.** Iterates `diff.deleted` and `diff.renamed.from` in reverse (via
///    `DoubleEndedIterator::rev`). For each entry the helper looks up the slot under `base`,
///    releases this Profile's [`ContribKey::ProfileDescendant`] if [`releases_watch`] says so, and
///    — when the slot has no remaining anchors — vacates and reaps it. [`Tree::try_reap`] cascades
///    up through any parent that loses its last anchor on the way; reverse iteration here is
///    performance / cleanliness (avoids the cascade work and the intermediate "parent holds reaped
///    child id" states), not a correctness requirement.
///
/// 2. **Phase 2 — creates.** Iterates `diff.created` and `diff.renamed.to` forward. For each entry
///    the helper ensures the slot under `base` (creating intermediate components as needed), sets
///    its kind, and installs the per-Resource [`ContribKey::ProfileDescendant`] contribution via
///    [`add_watch`] for covered Dirs (always) and covered Leaves under `Profile.has_per_file_fds`.
///
/// `diff.modified` entries are Tree-side no-ops — the slot exists and kind is unchanged. Per-file
/// Effect dispatch (`emit_effects_per_stable_file`) consumes them via the same Diff.
///
/// The whole contract is the side effects: Phase 1's watch release / slot reap (with the closing
/// `Unwatch` [`Tree::try_reap`] folds in) and Phase 2's `add_watch`. There is no reaped-slot return
/// — the fire history is per-Sub ([`specter_core::Sub::has_fired`]) and dies with the slotmap
/// entry, so a reaped leaf has nothing to purge by `ResourceId`.
pub(crate) fn apply_diff_to_tree(
    diff: &Diff,
    profile: &Profile,
    profile_id: ProfileId,
    base: ResourceId,
    tree: &mut Tree,
    out: &mut StepOutput,
    scratch: &mut PathBuf,
) {
    let key = ContribKey::ProfileDescendant(profile_id);

    // Phase 1 — deletes + renamed.from, reverse iteration.
    //
    // `slice::Iter` is `DoubleEndedIterator`; `Map<DoubleEnded, _>` is `DoubleEndedIterator`;
    // `Chain<DE, DE>` is `DoubleEndedIterator`. So `.chain(...).rev()` is well-defined and avoids
    // materialising the chain into a buffer just to reverse-iterate. Per Chain's `next_back`
    // semantics the reversed chain yields all `renamed.from` entries first (reverse-lex within that
    // list) followed by all `deleted` entries (reverse-lex within that list); cross-list ordering
    // is not load-bearing.
    //
    // Two paths converge on the slot lifecycle terminus, [`Tree::try_reap`]:
    //
    // - When this Profile contributes a watch at the slot — covered Dir under any events mask, or
    //   covered Leaf under `has_per_file_fds` — [`sub_watch_then_try_reap`] removes the
    //   contribution by key and try-reaps. The multi-Profile case (another Profile still
    //   contributes) short-circuits inside `try_reap` via `has_anchors()`.
    // - When this Profile contributes nothing at the slot — uncovered Leaf under a STRUCTURE-only
    //   Profile, where Phase 2's `ensure_descendant` materialised the slot without an `add_watch` —
    //   we still want to free a now-orphaned Tree slot that this delete reaches. Call `try_reap`
    //   directly so the Tree doesn't leak a never-watched slot.
    //
    // `try_reap` folds in `Tree::vacate` as the closing-emission step, so the kernel-watch protocol
    // owed at reap time (the closing `Unwatch`, if a contribution were ever stranded at the slot)
    // is emitted from inside the terminus rather than the caller — single source per protocol-close
    // edge.
    let phase_1 = diff
        .deleted
        .iter()
        .chain(diff.renamed.iter().map(|r| &r.from));
    for entry in phase_1.rev() {
        let Some(resource) = lookup_descendant(tree, base, entry.segment.as_str()) else {
            continue;
        };
        // Side-effecting; the `bool` (did-reap) return is unused (neither `try_reap` nor
        // `sub_watch_then_try_reap` is `#[must_use]`, so no `let _` ceremony).
        if releases_watch(profile, resource, entry.kind, tree, scratch) {
            sub_watch_then_try_reap(tree, resource, key, out);
        } else {
            tree.try_reap(resource, out);
        }
    }

    // Phase 2 — creates + renamed.to, forward iteration.
    //
    // `diff.created` is in depth-first pre-order, so parents precede their descendants.
    // `diff.renamed.to` follows the baseline-side (`from`) traversal order (not a sort), which is
    // likewise not necessarily descendant-aware, but `ensure_descendant` walks each component via
    // `ensure_child` (idempotent on existing slots), so any out-of-order entry only triggers extra
    // slot materialisation up front — the eventual `add_watch` for each explicit entry still lands
    // correctly because contributions are per-`(slot, key)` and disjoint between unrelated slots.
    let phase_2 = diff
        .created
        .iter()
        .chain(diff.renamed.iter().map(|r| &r.to));
    for entry in phase_2 {
        let Some(resource) =
            ensure_descendant(tree, base, entry.segment.as_str(), entry.kind.into())
        else {
            continue;
        };
        let want_watch = covers(profile, resource, tree, scratch)
            && (matches!(entry.kind, EntryKind::Dir) || profile.has_per_file_fds());
        if want_watch {
            add_watch(tree, resource, key, profile.events(), out);
        }
    }
}

/// Splice a probe response into `Profile.current` at `target`, then apply the resulting [`Diff`] to
/// the engine's `Tree`. Emits Watch ops via [`apply_diff_to_tree`] and commits the new view
/// atomically via [`specter_core::Profile::install_dir_current`].
///
/// Single source of truth for "engine just received an Ok Dir response; integrate it into the
/// Profile's view." The Dir/File dispatch lives one layer up in [`crate::Engine::apply_snapshot`],
/// which extracts the typed `prior` from `Profile.current` and forwards Dir snapshots here.
/// File-anchored Profiles never reach this helper — their `Profile.current` is
/// `TreeSnapshot::File(leaf)`, integrated by an inline `install_file_current` call at
/// `apply_snapshot`'s File arm. The typed [`specter_core::ProbeRequest`] dispatch chain plus the
/// certifier's inline kind guard together guarantee no File-prior + Dir-response pair survives to
/// this call site.
///
/// **Splice-first ordering.** The splice runs *before* any Tree mutation. A
/// [`SpliceResult::CrossedUncovered`] verdict (a v1-unreachable contract breach) surfaces a
/// [`Diagnostic`] and short-circuits without touching Tree state. `Profile.current` is untouched
/// across the breach: the `prior` arg was an Arc clone the caller (`apply_snapshot`) made from
/// `Profile.current`'s handle, so dropping it on the failure path leaves the Profile's own handle
/// alive at its pre-call shape.
///
/// **Diff at TARGET.** The Diff is built between `prior_at_target` (descended from `prior` via
/// [`subtree_at_dir`]) and `response_arc`, so its rel-paths are relative to `target` —
/// [`apply_diff_to_tree`] then passes `target` as the `base`. Diffing at the anchor instead would
/// thread a path-to-target prefix into every entry and force [`ensure_descendant`] /
/// [`lookup_descendant`] to re-walk the prefix per entry.
///
/// **Borrow shape.** Takes `profile_id: ProfileId` (not `&mut Profile`) because the splice write
/// and the immutable Profile read for the `apply_diff_to_tree` argument live on opposite sides of a
/// borrow boundary. Threading the id through lets graft re-borrow under whichever shape each step
/// needs. `prior` arrives typed (`Option<Arc<DirSnapshot>>`) — the caller already extracted it
/// under one Profile borrow, lifting the File-shape detection out of graft's body.
pub(crate) fn graft(
    profile_id: ProfileId,
    target: ResourceId,
    prior: Option<Arc<DirSnapshot>>,
    response_arc: Arc<DirSnapshot>,
    tree: &mut Tree,
    profiles: &mut ProfileMap,
    out: &mut StepOutput,
    scratch: &mut PathBuf,
) {
    let anchor = match profiles.get(profile_id) {
        Some(p) => p.resource(),
        None => return,
    };

    // Navigate the typed Dir prior down to `target`. Reused for the equal-hash early-out and as the
    // diff's baseline; cheap Arc::clone at depth 1 (target == anchor) — same shape as the prior
    // call through the inlined `subtree_at` dispatcher, but without the
    // `TreeSnapshot::Dir(Arc::clone(...))` wrapper allocation the `&TreeSnapshot`-keyed entry point
    // required.
    let prior_at_target = prior
        .as_ref()
        .and_then(|arc| subtree_at_dir(arc, anchor, target, tree));

    // O(1) early-out: response equals prior at this target ⇒ no Watch ops, no graft, no allocation.
    if let Some(p) = &prior_at_target
        && p.dir_hash() == response_arc.dir_hash()
    {
        return;
    }

    // Build the Diff first — pure, no side effects. At TARGET level so `apply_diff_to_tree`'s
    // `lookup_descendant` / `ensure_descendant` walks start at `target` rather than re-walking the
    // path-to-target prefix for every entry. `prior_at_target == None` is the first-graft /
    // freshly-covered-target case; `Diff::all_created(&response)` is the empty-prior shorthand.
    //
    // `diff_dir_pair` takes `&DirSnapshot` directly (Arc derefs in the call coerce), so neither
    // prior nor response needs to be cloned wholesale for the diff — `response_arc` stays available
    // for `splice`'s consume below.
    let diff = match &prior_at_target {
        Some(p) => diff_dir_pair(p, &response_arc),
        None => Diff::all_created(&response_arc),
    };

    // Splice — pure, no Tree mutation. `CrossedUncovered` short- circuits before any
    // `apply_diff_to_tree` work, so `Profile.current` and `Tree` stay coherent across the breach.
    // Even when (a future regression makes) `CrossedUncovered` reachable, the engine cannot
    // diverge: the splice-then-apply ordering keeps `Profile.current` and `Tree` from drifting
    // apart on the breach.
    //
    // Consumes `prior` and `response_arc`. The caller (apply_snapshot) held an independent Arc handle
    // in `Profile.current`, so dropping our `prior` on the CrossedUncovered failure path leaves the
    // Profile's own handle alive at its pre-call shape — no `install_dir_current` rebind needed.
    let new_current = match splice(prior, anchor, target, response_arc, tree) {
        SpliceResult::Spliced(snap) => snap,
        SpliceResult::CrossedUncovered(cause) => {
            out.diagnostics.push(Diagnostic::SpliceCrossedUncovered {
                profile: profile_id,
                target,
                cause,
            });
            return;
        }
    };
    let new_arc = match new_current {
        TreeSnapshot::Dir(arc) => arc,
        TreeSnapshot::File(_) => {
            debug_assert!(
                false,
                "graft: splice over a Dir prior yielded File (profile = {profile_id:?})",
            );
            return;
        }
    };

    // Apply to Tree under a scoped immutable Profile borrow so the `install_dir_current` write
    // below can re-borrow `&mut`. Purely side-effecting (watch release, reap); no reaped-slot
    // return — fire-history is per-Sub now, so a reaped leaf has nothing to purge.
    {
        let Some(profile) = profiles.get(profile_id) else {
            return;
        };
        apply_diff_to_tree(&diff, profile, profile_id, target, tree, out, scratch);
    }

    // Classify-and-graft in one move. The anchor sum's discriminant *is* the kind, so `current =
    // Some(Dir) ⇒ kind == Some(Dir)` is structural — there is no separate `kind` field to write or
    // to disagree with the snapshot variant.
    if let Some(p) = profiles.get_mut(profile_id) {
        p.install_dir_current(new_arc);
    }
}

#[cfg(test)]
#[allow(clippy::needless_pass_by_value)]
mod tests {
    use super::{apply_diff_to_tree, graft};
    use compact_str::CompactString;
    use smallvec::smallvec;
    use specter_core::{
        ChildEntry, ClassSet, Diff, DirChild, DirMeta, DirSnapshot, EntryKind, EntryRef,
        FsIdentity, LeafEntry, Profile, ProfileIdentity, ProfileMap, ResourceId, ResourceKind,
        ResourceRole, ScanConfig, StepOutput, Tree, TreeSnapshot, WatchOp,
    };
    use std::collections::BTreeMap;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, UNIX_EPOCH};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    // ---------------------------------------------------------------------------
    // Fixtures
    // ---------------------------------------------------------------------------

    fn meta(inode: u64) -> DirMeta {
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(inode, 0))
    }

    fn leaf(kind: EntryKind, inode: u64) -> ChildEntry {
        ChildEntry::Leaf(LeafEntry::synthetic(
            kind,
            0,
            UNIX_EPOCH,
            FsIdentity::synthetic(inode, 0),
        ))
    }

    fn dir_uncovered(inode: u64) -> ChildEntry {
        ChildEntry::Dir(DirChild::Uncovered(FsIdentity::synthetic(inode, 0)))
    }

    fn dir_covered(subtree: Arc<DirSnapshot>) -> ChildEntry {
        ChildEntry::Dir(DirChild::Covered(subtree))
    }

    fn dir_snap(inode: u64, entries: Vec<(&str, ChildEntry)>) -> Arc<DirSnapshot> {
        let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        for (name, child) in entries {
            map.insert(CompactString::new(name), child);
        }
        Arc::new(DirSnapshot::new(meta(inode), 0, map))
    }

    fn anchor(per_file: bool) -> (Tree, ProfileMap, ResourceId, specter_core::ProfileId) {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("root", ResourceRole::User);
        tree.set_kind(r, ResourceKind::Dir);
        // `events` chooses the per-leaf gating: CONTENT (or METADATA) ⇒ `has_per_file_fds = true` ⇒
        // Leaves get watch_demand contributions. STRUCTURE-only ⇒ Dir-only contributions.
        let events = if per_file {
            ClassSet::CONTENT
        } else {
            ClassSet::STRUCTURE
        };
        let pid = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                ProfileIdentity::new(
                    ScanConfig::builder().recursive(true).build(),
                    MAX_SETTLE,
                    events,
                ),
                SETTLE,
                None,
            ),
        );
        // The covered descendants reconciler reads `has_per_file_fds` off the Profile;
        // `Profile::new` already derives it from the events mask, but this test still depends on
        // the constructor's contract — assert it inline so a future change shouts.
        debug_assert_eq!(profiles.get(pid).unwrap().has_per_file_fds(), per_file);
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
    // apply_diff_to_tree — empty-prior creations via Diff::all_created
    // ---------------------------------------------------------------------------

    #[test]
    fn apply_diff_subtree_only_profile_watches_dir_creates_only() {
        // Diff::all_created over a fresh response ⇒ every entry is a creation. Subtree-only Profile
        // (has_per_file_fds=false) ⇒ Dir gets Watch, Leaf does not.
        let (mut tree, profiles, root, pid) = anchor(false);
        let response = dir_snap(
            100,
            vec![
                ("a.rs", leaf(EntryKind::File, 1)),
                ("sub", dir_uncovered(2)),
            ],
        );
        let diff = Diff::all_created(&response);

        let mut out = StepOutput::default();
        apply_diff_to_tree(
            &diff,
            profiles.get(pid).unwrap(),
            pid,
            root,
            &mut tree,
            &mut out,
            &mut PathBuf::new(),
        );
        assert_eq!(count_watch(&out), 1, "one Watch for the Dir creation");
        assert!(tree.lookup(Some(root), "a.rs").is_some());
        assert!(tree.lookup(Some(root), "sub").is_some());
    }

    #[test]
    fn apply_diff_per_file_profile_emits_watch_for_leaf_create() {
        // has_per_file_fds=true ⇒ Leaf creates *also* get Watch.
        let (mut tree, profiles, root, pid) = anchor(true);
        let response = dir_snap(
            100,
            vec![
                ("a.rs", leaf(EntryKind::File, 1)),
                ("sub", dir_uncovered(2)),
            ],
        );
        let diff = Diff::all_created(&response);

        let mut out = StepOutput::default();
        apply_diff_to_tree(
            &diff,
            profiles.get(pid).unwrap(),
            pid,
            root,
            &mut tree,
            &mut out,
            &mut PathBuf::new(),
        );
        assert_eq!(
            count_watch(&out),
            2,
            "Watch for both Dir and Leaf under per-file"
        );
    }

    // ---------------------------------------------------------------------------
    // apply_diff_to_tree — deletions release Watch on covered Dir
    // ---------------------------------------------------------------------------

    #[test]
    fn apply_diff_dir_deletion_releases_watch() {
        // Diff carries a single `deleted` entry for a covered Dir; the helper must release the
        // Profile's contribution and emit one Unwatch op.
        let (mut tree, profiles, root, pid) = anchor(false);
        // Materialise the Dir slot first so the delete path has something to `sub_watch` against.
        // Pre-populate with the same contribution apply_diff_to_tree will drop on delete (the
        // Profile's events, here STRUCTURE).
        let sub_id = tree
            .ensure_child(root, "sub", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(sub_id, ResourceKind::Dir);
        crate::refcounts::add_watch(
            &mut tree,
            sub_id,
            specter_core::ContribKey::ProfileDescendant(pid),
            ClassSet::STRUCTURE,
            &mut StepOutput::default(),
        );

        let diff = Diff {
            deleted: smallvec![EntryRef {
                segment: CompactString::new("sub"),
                kind: EntryKind::Dir,
                fs_id: FsIdentity::synthetic(2, 0),
            }],
            ..Default::default()
        };

        let mut out = StepOutput::default();
        apply_diff_to_tree(
            &diff,
            profiles.get(pid).unwrap(),
            pid,
            root,
            &mut tree,
            &mut out,
            &mut PathBuf::new(),
        );
        assert_eq!(count_unwatch(&out), 1, "Unwatch for the deleted Dir");
    }

    #[test]
    fn apply_diff_per_file_leaf_deletion_releases_watch() {
        // PerStableFile Profile (has_per_file_fds=true): covered Leaf carries a Watch contribution;
        // delete should release it (symmetric to create).
        let (mut tree, profiles, root, pid) = anchor(true);
        let leaf_id = tree
            .ensure_child(root, "a.rs", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(leaf_id, ResourceKind::File);
        crate::refcounts::add_watch(
            &mut tree,
            leaf_id,
            specter_core::ContribKey::ProfileDescendant(pid),
            ClassSet::CONTENT,
            &mut StepOutput::default(),
        );

        let diff = Diff {
            deleted: smallvec![EntryRef {
                segment: CompactString::new("a.rs"),
                kind: EntryKind::File,
                fs_id: FsIdentity::synthetic(1, 0),
            }],
            ..Default::default()
        };

        let mut out = StepOutput::default();
        apply_diff_to_tree(
            &diff,
            profiles.get(pid).unwrap(),
            pid,
            root,
            &mut tree,
            &mut out,
            &mut PathBuf::new(),
        );
        assert_eq!(count_unwatch(&out), 1);
    }

    // ---------------------------------------------------------------------------
    // apply_diff_to_tree — same-name different-inode is delete-then-create
    // ---------------------------------------------------------------------------

    #[test]
    fn apply_diff_inode_change_deletes_then_creates() {
        // Diff carries both a `deleted` and a `created` entry for "foo" (delete-then-create across
        // an inode change). Both Dirs ⇒ Watch released for the deleted, re-emitted for the created.
        // Net: one Unwatch + one Watch.
        let (mut tree, profiles, root, pid) = anchor(false);
        // Pre-materialise the prior Dir slot with the matching mask.
        let foo_id = tree
            .ensure_child(root, "foo", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(foo_id, ResourceKind::Dir);
        crate::refcounts::add_watch(
            &mut tree,
            foo_id,
            specter_core::ContribKey::ProfileDescendant(pid),
            ClassSet::STRUCTURE,
            &mut StepOutput::default(),
        );

        let diff = Diff {
            deleted: smallvec![EntryRef {
                segment: CompactString::new("foo"),
                kind: EntryKind::Dir,
                fs_id: FsIdentity::synthetic(1, 0),
            }],
            created: smallvec![EntryRef {
                segment: CompactString::new("foo"),
                kind: EntryKind::Dir,
                fs_id: FsIdentity::synthetic(2, 0),
            }],
            ..Default::default()
        };

        let mut out = StepOutput::default();
        apply_diff_to_tree(
            &diff,
            profiles.get(pid).unwrap(),
            pid,
            root,
            &mut tree,
            &mut out,
            &mut PathBuf::new(),
        );
        assert_eq!(count_unwatch(&out), 1);
        assert_eq!(count_watch(&out), 1);
    }

    // ---------------------------------------------------------------------------
    // graft — equal-hash early-out (no allocations, no Watch ops)
    // ---------------------------------------------------------------------------

    #[test]
    fn graft_equal_hash_short_circuits() {
        // current.subtree_at(target).dir_hash() == response.dir_hash() ⇒ zero allocations, zero
        // Watch ops, current unchanged.
        let (mut tree, mut profiles, root, pid) = anchor(false);
        let snap_a = dir_snap(100, vec![("a.rs", leaf(EntryKind::File, 1))]);
        profiles
            .get_mut(pid)
            .unwrap()
            .install_dir_current(Arc::clone(&snap_a));
        let response = dir_snap(100, vec![("a.rs", leaf(EntryKind::File, 1))]);

        let prior_arc_count = Arc::strong_count(&snap_a);

        let prior = profiles.get(pid).and_then(|p| p.current_dir().cloned());
        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            prior,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
            &mut PathBuf::new(),
        );
        assert_eq!(count_watch(&out), 0);
        // current Arc is unchanged (still pointing at snap_a — the early-out bypassed splice's
        // path-rebuild).
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
        let response = dir_snap(100, vec![("a.rs", leaf(EntryKind::File, 1))]);

        let prior = profiles.get(pid).and_then(|p| p.current_dir().cloned());
        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            prior,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
            &mut PathBuf::new(),
        );
        let p = profiles.get(pid).unwrap();
        assert!(p.current().is_some());
        let current = p.current().unwrap();
        let TreeSnapshot::Dir(arc) = current else {
            panic!("expected Dir snapshot");
        };
        assert_eq!(arc.dir_hash(), response.dir_hash());
    }

    // ---------------------------------------------------------------------------
    // graft — CrossedUncovered surfaces a Diagnostic
    // ---------------------------------------------------------------------------

    #[test]
    fn graft_emits_diagnostic_when_path_crosses_uncovered_intermediate() {
        // Setup: Tree has root → a → b (all live slots). The Profile's `current` snapshot is
        // anchored at root with entry "a" stored as `DirChild::Uncovered(_)` (engine never observed
        // below "a"). A probe response arrives for target `b`. Splice navigates the tree chain
        // anchor→a→b successfully but cannot navigate the *snapshot*'s coverage path —
        // `prior.lookup_covered_dir("a")` returns `None` because "a" is `Uncovered`, not `Covered`
        // — so splice returns CrossedUncovered. Graft must emit Diagnostic::SpliceCrossedUncovered
        // AND keep the prior `current` unchanged so the anchor-rooted invariant on
        // `Profile.current` is preserved.
        let (mut tree, mut profiles, root, pid) = anchor(false);
        let a_id = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(a_id, ResourceKind::Dir);
        let b_id = tree
            .ensure_child(a_id, "b", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(b_id, ResourceKind::Dir);

        let prior_current = dir_snap(100, vec![("a", dir_uncovered(2))]);
        profiles
            .get_mut(pid)
            .unwrap()
            .install_dir_current(Arc::clone(&prior_current));

        let response = dir_snap(200, vec![]);

        let prior = profiles.get(pid).and_then(|p| p.current_dir().cloned());
        let mut out = StepOutput::default();
        graft(
            pid,
            b_id,
            prior,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
            &mut PathBuf::new(),
        );

        let has_diag = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::SpliceCrossedUncovered {
                    profile,
                    target,
                    cause: specter_core::SpliceFailureCause::IntermediateUncovered,
                }
                if *profile == pid && *target == b_id
            )
        });
        assert!(
            has_diag,
            "graft must emit SpliceCrossedUncovered with \
             cause=IntermediateUncovered when splice can't navigate the \
             snapshot's coverage path through a `DirChild::Uncovered` \
             intermediate",
        );

        let p = profiles.get(pid).unwrap();
        let current_arc = p.current_dir().expect("expected Dir snapshot");
        assert!(
            Arc::ptr_eq(current_arc, &prior_current),
            "graft leaves Profile.current's Arc handle untouched on CrossedUncovered \
             (splice consumed our typed prior clone; Profile.current's own handle is independent)",
        );
    }

    // ---------------------------------------------------------------------------
    // graft — kind-flip + inode-reuse regression
    // ---------------------------------------------------------------------------

    #[test]
    fn graft_kind_flip_inode_reuse_retypes_slot_and_installs_watch() {
        // Kind-flip + inode-reuse regression. Failing lifecycle:
        // 1. Prior: `/root/foo` is a covered Leaf (File, inode=42, has per-file watch contribution
        //    under a per-file Profile).
        // 2. Probe response: `/root/foo` is now a covered Dir at the SAME inode (kernel inode reuse
        //    across the kind flip). The response's dir entry carries an observed subtree.
        // 3. Expected post-graft state:
        //    - The old File slot is reaped (generation incremented).
        //    - A fresh-generation Tree slot at `(root, "foo")` exists with kind = Dir.
        //    - The new slot's contributions include `ContribKey::ProfileDescendant(pid)` for the
        //      per-file Profile's events.
        //    - `out.watch_ops` contains a Watch op at the new slot.
        //
        // Keying entity identity on `(inode, device)` alone would, under inode reuse across a kind
        // flip, conclude "same identity ⇒ no Tree-side delta" and skip both the reap of the old
        // File slot and the creation of the new Dir slot. Instead, `diff_tree::diff_same_name`
        // routes kind flips through its
        // `_ =>` arm (pair_eligible: false), which `apply_diff_to_tree`
        // Phase 1 / Phase 2 then applies symmetrically.
        let (mut tree, mut profiles, root, pid) = anchor(true);

        // Pre-materialise the prior File slot at `(root, "foo")` with a watch contribution matching
        // the per-file Profile's mask.
        let prior_foo_id = tree
            .ensure_child(root, "foo", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(prior_foo_id, ResourceKind::File);
        crate::refcounts::add_watch(
            &mut tree,
            prior_foo_id,
            specter_core::ContribKey::ProfileDescendant(pid),
            ClassSet::CONTENT,
            &mut StepOutput::default(),
        );

        // Prior snapshot: root with covered Leaf `foo` (inode 42).
        let prior_current = dir_snap(100, vec![("foo", leaf(EntryKind::File, 42))]);
        profiles
            .get_mut(pid)
            .unwrap()
            .install_dir_current(Arc::clone(&prior_current));

        // Build response: root has covered Dir `foo` at the SAME inode (42), with one descendant
        // File `nested.rs` under it. The covered subtree's `root_meta.fs_id` IS the `foo`
        // directory's kernel identity under the sum-type encoding, so it must be stamped with inode
        // 42 for the kind-flip-with-inode-reuse invariant to hold.
        let nested_subtree = {
            let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
            entries.insert(CompactString::new("nested.rs"), leaf(EntryKind::File, 99));
            Arc::new(DirSnapshot::new(meta(42), 0, entries))
        };
        let response = dir_snap(200, vec![("foo", dir_covered(nested_subtree))]);

        // Sanity: prior and new "foo" entries share `(inode, device)` ⇒ this is the
        // kind-flip-with-inode-reuse case that `walk_pair` mishandled. Confirms the regression's
        // failing pre-condition.
        let prior_foo_child = prior_current.entries().get("foo").unwrap();
        let new_foo_child = response.entries().get("foo").unwrap();
        assert_eq!(
            prior_foo_child.fs_id().inode(),
            new_foo_child.fs_id().inode()
        );
        assert_eq!(
            prior_foo_child.fs_id().device(),
            new_foo_child.fs_id().device()
        );
        assert_ne!(
            prior_foo_child.kind(),
            new_foo_child.kind(),
            "regression invariant: prior File + new Dir at the same inode",
        );

        let prior = profiles.get(pid).and_then(|p| p.current_dir().cloned());
        let mut out = StepOutput::default();
        graft(
            pid,
            root,
            prior,
            Arc::clone(&response),
            &mut tree,
            &mut profiles,
            &mut out,
            &mut PathBuf::new(),
        );

        // 1. Old File slot was reaped — its id is stale on a fresh lookup. `tree.lookup(root,
        //    "foo")` should now resolve to a NEW slot with a different id (generation increment).
        let new_foo_id = tree
            .lookup(Some(root), "foo")
            .expect("graft must materialise a Tree slot at the post-graft target");
        assert_ne!(
            new_foo_id, prior_foo_id,
            "kind-flip must reap the prior slot and allocate a fresh-generation one",
        );
        assert!(
            tree.get(prior_foo_id).is_none(),
            "prior File slot must be reaped after kind flip",
        );

        // 2. New slot's kind is Dir.
        let kind = tree
            .get(new_foo_id)
            .and_then(specter_core::Resource::kind)
            .expect("post-graft slot must have a recorded kind");
        assert_eq!(kind, ResourceKind::Dir, "new slot must be re-typed to Dir");

        // 3. New slot carries the Profile's descendant contribution.
        let has_contribution = tree
            .get(new_foo_id)
            .is_some_and(specter_core::Resource::is_watched);
        assert!(
            has_contribution,
            "new Dir slot must carry the Profile's descendant contribution \
             (kind-flip didn't install Watch — regression)",
        );

        // 4. `out.watch_ops` contains a Watch op at the new slot.
        let watched_new_slot = out
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { resource, .. } if *resource == new_foo_id));
        assert!(
            watched_new_slot,
            "out.watch_ops must contain a Watch op at the new Dir slot",
        );

        // 5. The descendant `nested.rs` was materialised under the new Dir slot. Per-file Profile ⇒
        //    covered Leaf carries a watch contribution too.
        let nested_id = tree
            .lookup(Some(new_foo_id), "nested.rs")
            .expect("descendant `nested.rs` must be materialised");
        let nested_watched = tree
            .get(nested_id)
            .is_some_and(specter_core::Resource::is_watched);
        assert!(
            nested_watched,
            "covered descendant of the new Dir must carry its own watch \
             contribution under a per-file Profile",
        );
    }
}
