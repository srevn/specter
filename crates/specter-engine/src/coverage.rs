//! The coverage relation and the reconfirm query derived from it.
//!
//! [`covers`] walks the segment chain from `profile.resource` (the
//! anchor) down to the candidate `target`, evaluating `max_depth`, the
//! `recursive` flag, the exclude globs, and the file pattern (with
//! directory bypass) along the way. It is the gate for two things in
//! the engine: whether an `FsEvent` at `R` should drive `P`'s burst,
//! and whether `R` contributes to `P`'s `watch_demand`.
//!
//! [`nearest_covering_ancestor`] is its transitive derivation, and
//! [`has_active_standard_descendant`] (via [`chain_reaches`]) is the
//! pure query that replaced the old `dirty_descendants` refcount: it
//! answers, fresh at each consult point, "is some Active-Standard
//! strict-descendant Profile still covering this ancestor?" — the
//! `Draining → Verifying` reconfirm condition. Evaluating it as a
//! query rather than maintaining it as a counter is what makes it
//! robust to mid-burst topology moves; the rationale lives on
//! [`has_active_standard_descendant`].

use smallvec::SmallVec;
use specter_core::{
    Profile, ProfileId, ProfileMap, ProfileState, Resource, ResourceId, ResourceKind, Tree,
};
use std::path::PathBuf;

/// True iff `profile` would scan `target` given its `ScanConfig`.
///
/// **Depth-0 (`target == profile.resource`).** Always `true`. The anchor
/// is part of the Profile's scope by construction — `FsEvent`s at the
/// anchor must drive the anchor's burst, so coverage at the anchor
/// is unconditional. Pattern, exclude, depth, and recursive checks all
/// bypass at this depth.
///
/// **Descendants.** Build the cumulative relative path segment-by-segment
/// from `profile.resource` to `target`. At each step, test the cumulative
/// `Path` against every exclude glob — `target` matches `target/foo` at
/// depth 1, `target/**` matches the same path at depth 2, and the two
/// styles coexist. The file pattern matches the *full* relative path of
/// `target`, only when `target.kind == File`; directories bypass the
/// pattern.
///
/// Returns `false` if `target` is not on the descendant chain of
/// `profile.resource` (sibling, ancestor, or unrelated subtree), or if
/// any node along the chain is stale (its `ResourceId` no longer names
/// a live slot).
#[must_use]
pub fn covers(profile: &Profile, target: ResourceId, tree: &Tree, scratch: &mut PathBuf) -> bool {
    let anchor = profile.resource;

    if target == anchor {
        return true;
    }

    // Walk target → ancestor chain to anchor; collect segments in reverse
    // (target-to-root), then reverse to root-to-target order. Inline cap
    // of 8 covers typical source-tree depths from a workspace anchor
    // (`src/foo/bar/baz/qux/file.rs` is 6 deep); cap 4 spilled on every
    // such path.
    //
    // Termination relies on the `Tree` acyclicity invariant: each
    // `parent()` step strictly ascends, so the walk reaches `anchor`
    // or bottoms out at a root (`None`) in at most `depth(target)`
    // steps. Intentionally not depth-bounded — a defensive cap here
    // would mask a real `Tree`-construction cycle bug instead of
    // surfacing it (mirrors `snapshot::tree::ancestor_chain`).
    let mut rev: SmallVec<[&str; 8]> = SmallVec::new();
    let mut cur = target;
    loop {
        let Some(resource) = tree.get(cur) else {
            return false;
        };
        let Some(segment_str) = tree.name(cur) else {
            return false;
        };
        rev.push(segment_str);
        match resource.parent() {
            Some(p) if p == anchor => break,
            Some(p) => cur = p,
            None => return false,
        }
    }
    rev.reverse();

    let depth = u32::try_from(rev.len()).unwrap_or(u32::MAX);

    if let Some(max) = profile.config().max_depth
        && depth > max
    {
        return false;
    }
    if depth > 1 && !profile.config().recursive {
        return false;
    }

    // Unprobed slots collapse to File-shape (the backend-mask
    // convention shared by `fs_event_to_class`, the kqueue / inotify
    // translators, and `recompute_events_union`). The prior raw-`kind`
    // form let Unknown bypass the pattern entirely — a file freshly
    // touched in the window between create_child's slot materialization
    // and a follow-up event would slip the user's pattern filter. The
    // kind lookup stays gated behind `pattern.is_some()` — the closure
    // runs only on `Some`, matching the prior `if let Some(pat)` cost.
    let pattern = profile.config().pattern.as_ref().filter(|_| {
        matches!(
            tree.get(target)
                .map_or(ResourceKind::File, Resource::kind_or_file),
            ResourceKind::File
        )
    });
    let exclude = &profile.config().exclude;

    // One incremental build into the engine-owned `scratch` (capacity
    // retained across calls; `clear()` per call so the cross-call
    // residue is never observable). The exclude walk tests every
    // cumulative prefix, and its terminal state *is* the full relative
    // path — the pattern matches that directly, so the two prior
    // `PathBuf::new()` allocations + rebuilds collapse to one cleared
    // reuse. Skipped wholesale when neither exclude nor pattern applies
    // (the prior zero-work fast path is preserved). An early `return`
    // mid-walk leaves `scratch` dirty; the next entry's `clear()` is
    // the reset — `scratch` is per-call logically, per-step physically.
    if !exclude.is_empty() || pattern.is_some() {
        scratch.clear();
        for seg in &rev {
            scratch.push(seg);
            for excl in exclude {
                if excl.matches_path(scratch.as_path()) {
                    return false;
                }
            }
        }
        if let Some(pat) = pattern
            && !pat.matches_path(scratch.as_path())
        {
            return false;
        }
    }

    true
}

/// Resolve the nearest covering ancestor Profile of `child` — the
/// derivation companion to [`covers`], and the **query core** of the
/// `Draining → Verifying` reconfirm.
///
/// Walks Resource ancestors of `child.resource`; at each ancestor
/// Resource, picks the smallest covering [`ProfileId`] for a
/// deterministic tie-break. Returns `None` for root Profiles whose
/// ancestor chain holds no covering Profile.
///
/// "Nearest ancestor *Profile*, not Resource" is the easy mistake:
/// a Resource ancestor with no Profile is skipped; the walk
/// continues to the next Resource ancestor.
///
/// **Pure, never cached.** The result is a total function of `(tree,
/// profiles, child)` — no peer state, no stored edge. It used to feed
/// a per-Profile `parent_profile` cache; that cache was deleted
/// because a refcount keyed on a recomputable derivation could not be
/// kept balanced across mid-burst topology moves. The derivation now
/// stands alone: [`chain_reaches`] climbs it hop-by-hop and
/// [`has_active_standard_descendant`] evaluates the reconfirm
/// predicate fresh from it. `pub(crate)` — engine-internal; no
/// cross-crate consumer.
///
/// Each `child → result` step is a strict Resource-ancestor move
/// (`tree.ancestors` is strict, and the same-Resource co-anchor case
/// is excluded), so iterating it ([`chain_reaches`]) terminates
/// structurally — a cycle is unrepresentable, with no self-edge
/// assertion needed.
#[must_use]
pub(crate) fn nearest_covering_ancestor(
    tree: &Tree,
    profiles: &ProfileMap,
    child: ProfileId,
) -> Option<ProfileId> {
    let child_resource = profiles.get(child)?.resource;
    // Cold path (a Draining-phase query, not the per-event hot path):
    // own a local scratch reused across the ancestor loop's `covers`
    // calls. The signature stays clean — threading `&mut PathBuf`
    // through this pure derivation and its `chain_reaches` /
    // `has_active_standard_descendant` callers would muddy their
    // "total function of (tree, profiles, child)" contract for an
    // allocation the cold path does not feel.
    let mut scratch = PathBuf::new();
    for ancestor in tree.ancestors(child_resource) {
        let nearest = profiles
            .at(ancestor)
            .filter(|&pid| pid != child)
            .filter(|&pid| {
                profiles
                    .get(pid)
                    .is_some_and(|p| covers(p, child_resource, tree, &mut scratch))
            })
            .min();
        if nearest.is_some() {
            return nearest;
        }
    }
    None
}

/// Walk `resource` and its strict ancestors looking for Profiles whose
/// [`covers`] predicate accepts `resource`. Returns the matching
/// Profiles in encounter order. P4 single-Profile resolves to 0 or 1.
/// `pub(crate)` — the sole caller is `Engine::on_fs_event`; a coverage
/// derivation co-located with [`covers`] / [`nearest_covering_ancestor`].
///
/// **Pending Profiles are filtered at the source.** A Pending Profile
/// carries no anchor-side `watch_demand` from this Profile — the
/// descent prefix carries it instead (via
/// [`specter_core::ContribKey::ProfileDescent`]); the anchor slot
/// itself only receives the
/// [`specter_core::ContribKey::ProfileAnchor`] contribution at
/// descent-completion time. Events at the prefix route via
/// `classify_event_carriers` / `on_descent_event`; events at the anchor
/// or its descendants are structurally unreachable in production (the
/// anchor's `watch_demand` is 0 ⇒ head guard short-circuits). Filtering
/// here makes the routing contract explicit: covering-Profile dispatch
/// (Standard burst, anchor terminal event) only sees Profiles with a
/// materialized anchor.
#[must_use]
pub(crate) fn covering_profiles(
    tree: &Tree,
    profiles: &ProfileMap,
    resource: ResourceId,
    scratch: &mut PathBuf,
) -> SmallVec<[ProfileId; 2]> {
    let mut out: SmallVec<[ProfileId; 2]> = SmallVec::new();
    let mut cur = Some(resource);
    while let Some(rid) = cur {
        for pid in profiles.at(rid) {
            let Some(p) = profiles.get(pid) else {
                continue;
            };
            if matches!(p.state(), ProfileState::Pending(_)) {
                continue;
            }
            if covers(p, resource, tree, scratch) && !out.contains(&pid) {
                out.push(pid);
            }
        }
        cur = tree.parent(rid);
    }
    out
}

/// True iff some **strict-descendant** Profile of `ancestor`'s subtree
/// is in an Active **Standard** burst (any phase — pre- or post-fire)
/// **and** has `ancestor` on its transitive
/// [`nearest_covering_ancestor`] chain.
///
/// The derived, never-cached replacement for the old
/// `Profile.dirty_descendants > 0` refcount. Same chain semantics —
/// the *transitive nearest-covering-ancestor chain*, **not** the raw
/// Tree subtree and **not** a single direct [`covers`] test (`covers`
/// is not transitive: an intermediate broader Profile keeps a deeper
/// one on `ancestor`'s chain even where `ancestor`'s own
/// `max_depth`/`pattern` would exclude it). Evaluated fresh at each of
/// its two consult points — the `gated_fire` Draining gate and the
/// `finish_burst_to_idle` Draining sweep — never accumulated, so
/// no mid-burst topology move can desynchronise it.
///
/// Iterative DFS over the **strict** Tree descendants of
/// `ancestor.resource` (starts at its children, so `ancestor` itself
/// and any co-anchor Profile sharing its slot are excluded — matching
/// the old refcount never self-counting). The strict subtree is a
/// sound superset of `{D : ancestor ∈ chain(D)}` (every chain link is
/// a Resource-ancestor, so a contributing `D.resource` is necessarily
/// a Tree-descendant of `ancestor.resource`); [`chain_reaches`] is the
/// exact filter. Short-circuits on the first witness.
pub(crate) fn has_active_standard_descendant(
    tree: &Tree,
    profiles: &ProfileMap,
    ancestor: ProfileId,
) -> bool {
    let Some(root) = profiles.get(ancestor).map(|p| p.resource) else {
        return false;
    };
    // Strict descendants only: seed the stack with `root`'s children,
    // never `root` itself.
    let mut stack: Vec<ResourceId> = tree.children_ids(root).collect();
    while let Some(node) = stack.pop() {
        for d in profiles.at(node) {
            if profiles
                .get(d)
                .is_some_and(|p| p.state().in_active_standard_burst())
                && chain_reaches(tree, profiles, d, ancestor)
            {
                return true;
            }
        }
        stack.extend(tree.children_ids(node));
    }
    false
}

/// True iff `ancestor` lies on `descendant`'s transitive
/// [`nearest_covering_ancestor`] chain — i.e. climbing
/// `descendant → nca(descendant) → nca(nca(descendant)) → …` reaches
/// `ancestor`. This is exactly the chain the deleted `propagate` walked
/// via the cached `parent_profile` edge (that edge *was*
/// `nearest_covering_ancestor`'s result), recomputed fresh.
///
/// Terminates structurally: every hop is a strict Resource-ancestor
/// move (see [`nearest_covering_ancestor`]), so the climb is bounded
/// by `descendant`'s Tree depth and a cycle cannot form. A
/// `descendant == ancestor` call (not produced by
/// [`has_active_standard_descendant`]'s strict walk, but well-defined)
/// returns `false`: the chain is strictly above `descendant`.
fn chain_reaches(
    tree: &Tree,
    profiles: &ProfileMap,
    descendant: ProfileId,
    ancestor: ProfileId,
) -> bool {
    let mut cur = descendant;
    while let Some(parent) = nearest_covering_ancestor(tree, profiles, cur) {
        if parent == ancestor {
            return true;
        }
        cur = parent;
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use specter_core::{
        ClassSet, GlobPattern, Profile, ProfileIdentity, ProfileMap, ResourceKind, ResourceRole,
        ScanConfig, ScanConfigBuilder,
    };
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    /// Coverage is orthogonal to the events filter; tests use an empty
    /// mask to keep `Profile::new`'s `has_per_file_fds` derivation off.
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Mark `id`'s `ResourceKind` so the pattern check has a stable answer.
    fn mark(tree: &mut Tree, id: ResourceId, kind: ResourceKind) {
        tree.set_kind(id, kind);
    }

    /// Anchor a Profile with the supplied `ScanConfig`. Caller still owns the
    /// `Tree` (and any descendant Resources they attach).
    fn anchor(tree: &mut Tree, segment: &str, builder: ScanConfigBuilder) -> (ResourceId, Profile) {
        let r = tree.ensure_root(segment, ResourceRole::User);
        mark(tree, r, ResourceKind::Dir);
        let p = Profile::new(
            r,
            ProfileIdentity {
                config: builder.build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        );
        (r, p)
    }

    fn glob(src: &str) -> GlobPattern {
        GlobPattern::compile(src).expect("test glob compiles")
    }

    fn recursive_unbounded() -> ScanConfigBuilder {
        ScanConfig::builder().recursive(true)
    }

    #[test]
    fn target_equals_anchor_dir_is_covered() {
        let mut tree = Tree::new();
        let (anchor_id, profile) = anchor(&mut tree, "root", recursive_unbounded());
        assert!(covers(&profile, anchor_id, &tree));
    }

    #[test]
    fn target_equals_anchor_file_is_covered_no_pattern() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("log.txt", ResourceRole::User);
        mark(&mut tree, r, ResourceKind::File);
        let p = Profile::new(
            r,
            ProfileIdentity {
                config: ScanConfig::builder().build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        );
        assert!(covers(&p, r, &tree));
    }

    #[test]
    fn target_equals_anchor_file_is_covered_with_pattern() {
        // Depth-0 bypasses the pattern check. The user anchored here; the
        // file is part of the Profile's scope by construction.
        let mut tree = Tree::new();
        let r = tree.ensure_root("log.txt", ResourceRole::User);
        mark(&mut tree, r, ResourceKind::File);
        let p = Profile::new(
            r,
            ProfileIdentity {
                config: ScanConfig::builder().pattern(glob("*.rs")).build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        );
        assert!(covers(&p, r, &tree));
    }

    #[test]
    fn target_outside_anchor_subtree_is_uncovered() {
        let mut tree = Tree::new();
        let (_anchor_id, profile) = anchor(&mut tree, "root", recursive_unbounded());
        let sibling = tree.ensure_root("sibling", ResourceRole::User);
        mark(&mut tree, sibling, ResourceKind::Dir);
        assert!(!covers(&profile, sibling, &tree));
    }

    #[test]
    fn ancestor_is_uncovered() {
        // covers(P, R) when R is an ancestor of P.resource — not on the
        // descendant chain, returns false.
        let mut tree = Tree::new();
        let parent = tree.ensure_root("parent", ResourceRole::User);
        mark(&mut tree, parent, ResourceKind::Dir);
        let anchor_id = tree
            .ensure_child(parent, "anchor", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, anchor_id, ResourceKind::Dir);
        let profile = Profile::new(
            anchor_id,
            ProfileIdentity {
                config: recursive_unbounded().build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        );
        assert!(!covers(&profile, parent, &tree));
    }

    #[test]
    fn recursive_true_covers_deep_descendants() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded());
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, a, ResourceKind::Dir);
        let b = tree
            .ensure_child(a, "b", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, b, ResourceKind::Dir);
        let c = tree
            .ensure_child(b, "c.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, c, ResourceKind::File);
        assert!(covers(&profile, a, &tree));
        assert!(covers(&profile, b, &tree));
        assert!(covers(&profile, c, &tree));
    }

    #[test]
    fn recursive_false_covers_depth_one_only() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", ScanConfig::builder().recursive(false));
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, a, ResourceKind::Dir);
        let b = tree
            .ensure_child(a, "b.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, b, ResourceKind::File);
        assert!(covers(&profile, a, &tree));
        assert!(!covers(&profile, b, &tree));
    }

    #[test]
    fn max_depth_caps_descent() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded().max_depth(Some(2)));
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        let b = tree
            .ensure_child(a, "b", ResourceRole::User)
            .expect("test live parent");
        let c = tree
            .ensure_child(b, "c", ResourceRole::User)
            .expect("test live parent");
        for r in [a, b, c] {
            mark(&mut tree, r, ResourceKind::Dir);
        }
        assert!(covers(&profile, a, &tree)); // depth 1
        assert!(covers(&profile, b, &tree)); // depth 2
        assert!(!covers(&profile, c, &tree)); // depth 3 > max
    }

    #[test]
    fn exclude_segment_pattern_drops_at_depth_one() {
        // "target" excludes the `target` directory at depth 1.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().exclude(glob("target")),
        );
        let target = tree
            .ensure_child(root, "target", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, target, ResourceKind::Dir);
        let other = tree
            .ensure_child(root, "src", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, other, ResourceKind::Dir);
        assert!(!covers(&profile, target, &tree));
        assert!(covers(&profile, other, &tree));
    }

    #[test]
    fn exclude_recursive_pattern_drops_at_depth_two() {
        // "target/**" excludes target/foo at depth 2.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().exclude(glob("target/**")),
        );
        let target = tree
            .ensure_child(root, "target", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, target, ResourceKind::Dir);
        let foo = tree
            .ensure_child(target, "foo", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, foo, ResourceKind::Dir);
        assert!(!covers(&profile, foo, &tree));
    }

    #[test]
    fn pattern_matches_file_extension() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().pattern(glob("*.rs")),
        );
        let rs = tree
            .ensure_child(root, "lib.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, rs, ResourceKind::File);
        let c = tree
            .ensure_child(root, "lib.c", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, c, ResourceKind::File);
        assert!(covers(&profile, rs, &tree));
        assert!(!covers(&profile, c, &tree));
    }

    #[test]
    fn pattern_bypasses_directories() {
        // dirs are always covered (we descend through them) regardless
        // of the file pattern.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().pattern(glob("*.rs")),
        );
        let src = tree
            .ensure_child(root, "src", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, src, ResourceKind::Dir);
        assert!(covers(&profile, src, &tree));
    }

    #[test]
    fn pattern_matches_full_relative_path_for_files() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().pattern(glob("src/**/*.rs")),
        );
        let src = tree
            .ensure_child(root, "src", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, src, ResourceKind::Dir);
        let lib = tree
            .ensure_child(src, "lib.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, lib, ResourceKind::File);
        let other_src = tree
            .ensure_child(root, "other.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, other_src, ResourceKind::File);
        assert!(covers(&profile, lib, &tree));
        // `other.rs` lives directly under root and doesn't match `src/**/*.rs`.
        assert!(!covers(&profile, other_src, &tree));
    }

    #[test]
    fn stale_resource_id_returns_false() {
        let mut tree = Tree::new();
        let (_root, profile) = anchor(&mut tree, "root", recursive_unbounded());
        // Build a Resource then drop it via reap so the id is stale.
        let temp = tree
            .ensure_child(profile.resource, "ghost", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, temp, ResourceKind::Dir);
        // Need to drop everything that anchors `temp`. `temp` has no children
        // and no profiles attached, role User — so try_reap should succeed.
        assert!(tree.try_reap(temp, &mut specter_core::StepOutput::default()));
        assert!(tree.get(temp).is_none());
        assert!(!covers(&profile, temp, &tree));
    }

    #[test]
    fn exclude_short_circuits_descendants() {
        // Excluding `a` also rejects `a/b`, since the cumulative path matches
        // at depth 1 already (a Path's glob matcher is checked progressively).
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded().exclude(glob("a")));
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, a, ResourceKind::Dir);
        let b = tree
            .ensure_child(a, "b.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, b, ResourceKind::File);
        assert!(!covers(&profile, a, &tree));
        assert!(!covers(&profile, b, &tree));
    }

    #[test]
    fn star_pattern_descends_through_directories() {
        // globset's `*` does NOT respect `/` — `*.rs` is "any path ending in
        // `.rs`," not basename-only. Pin the semantic so a future glob-engine
        // swap surfaces the deviation as a test failure rather than a
        // silent change in user-visible coverage.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().pattern(glob("*.rs")),
        );
        // Top-level .rs file matches.
        let lib = tree
            .ensure_child(root, "lib.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, lib, ResourceKind::File);
        // Deep .rs file also matches — `*` is path-blind.
        let src = tree
            .ensure_child(root, "src", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, src, ResourceKind::Dir);
        let deep = tree
            .ensure_child(src, "deep.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, deep, ResourceKind::File);
        let deeper_dir = tree
            .ensure_child(src, "foo", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, deeper_dir, ResourceKind::Dir);
        let deepest = tree
            .ensure_child(deeper_dir, "deeper.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, deepest, ResourceKind::File);
        assert!(covers(&profile, lib, &tree));
        assert!(covers(&profile, deep, &tree));
        assert!(covers(&profile, deepest, &tree));
    }

    #[test]
    fn pattern_applies_to_unprobed_kind_targets() {
        // An FsEvent on a descendant whose kind hasn't been classified
        // yet (Resource::kind() == None) must still be filtered by the
        // user's pattern. The prior raw-`kind` form in `covers` let
        // Unknown-kind slots bypass the pattern entirely. The new
        // `kind_or_file()` accessor collapses unprobed to File-shape,
        // matching the convention shared by `fs_event_to_class`.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().pattern(glob("*.rs")),
        );
        let unprobed = tree
            .ensure_child(root, "lib.c", ResourceRole::User)
            .expect("test live parent");
        // Deliberately do NOT call `mark` — kind stays at the default
        // `ResourceKind::Unknown` placeholder.
        assert!(tree.get(unprobed).unwrap().kind().is_none());
        // Pattern is `*.rs`; lib.c does not match. Unprobed must still
        // be filtered out.
        assert!(!covers(&profile, unprobed, &tree));
    }

    #[test]
    fn double_star_exclude_does_not_match_directory_itself() {
        // `target/**` matches `target/foo` but not `target` literally. The
        // directory itself remains covered; only its contents are excluded.
        // Surprises users coming from gitignore (where `target/` excludes the
        // directory and contents); pinned here so the contrast is explicit.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().exclude(glob("target/**")),
        );
        let target = tree
            .ensure_child(root, "target", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, target, ResourceKind::Dir);
        let inside = tree
            .ensure_child(target, "foo", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, inside, ResourceKind::Dir);
        assert!(
            covers(&profile, target, &tree),
            "`target/**` does not match `target` literally — the directory \
             itself stays covered",
        );
        assert!(
            !covers(&profile, inside, &tree),
            "`target/foo` matches `target/**` — contents excluded",
        );
    }

    // ===== nearest_covering_ancestor =====
    //
    // Resolution tests for the `covers` derivation. Walks Resource
    // ancestors of a child Profile's anchor; at each ancestor, picks
    // the smallest covering [`ProfileId`].

    fn cfg_recursive() -> ScanConfig {
        ScanConfig::builder().recursive(true).build()
    }

    fn mark_dir(tree: &mut Tree, id: ResourceId) {
        tree.set_kind(id, ResourceKind::Dir);
    }

    #[test]
    fn nearest_covering_ancestor_returns_none_for_orphan_profile() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("root", ResourceRole::User);
        mark_dir(&mut tree, r);
        let pid = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        assert!(nearest_covering_ancestor(&tree, &profiles, pid).is_none());
    }

    #[test]
    fn nearest_covering_ancestor_walks_up_to_first_covering_ancestor() {
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        let b = tree
            .ensure_child(a, "b", ResourceRole::User)
            .expect("test live parent");
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        let p_a = profiles.attach(
            &mut tree,
            Profile::new(
                a,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                b,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );

        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_b), Some(p_a));
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_a),
            Some(p_root)
        );
        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_root), None);
    }

    #[test]
    fn nearest_covering_ancestor_skips_non_covering_ancestor() {
        // root has Profile p_root with recursive=false → does not cover deep
        // descendants. The deeper Profile's resolution must walk past p_root
        // and return None (no further covering ancestor).
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        let b = tree
            .ensure_child(a, "b", ResourceRole::User)
            .expect("test live parent");
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let _p_root = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ProfileIdentity {
                    config: ScanConfig::builder().recursive(false).build(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                b,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_b), None);
    }

    #[test]
    fn nearest_covering_ancestor_excludes_self() {
        // Two co-located Profiles at the anchor; resolution for one must
        // not return itself.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("root", ResourceRole::User);
        mark_dir(&mut tree, r);
        let p_a = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: Duration::from_secs(6),
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                r,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: Duration::from_secs(12),
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        // Both at root; root has no Profile ancestor; resolution walks
        // ancestors of root.resource (none — root is a Tree root).
        assert!(nearest_covering_ancestor(&tree, &profiles, p_a).is_none());
        assert!(nearest_covering_ancestor(&tree, &profiles, p_b).is_none());
    }

    #[test]
    fn nearest_covering_ancestor_ties_by_smallest_profile_id() {
        // Two co-located covering Profiles at the same ancestor Resource.
        // Resolution for a deeper Profile picks the smaller ProfileId.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        let leaf = tree
            .ensure_child(root, "leaf", ResourceRole::User)
            .expect("test live parent");
        mark_dir(&mut tree, root);
        mark_dir(&mut tree, leaf);
        // Two distinct Profiles at root, distinct config_hashes via differing
        // max_settle (makes them separate Profiles).
        let p_root_a = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: Duration::from_secs(6),
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        let p_root_b = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: Duration::from_secs(12),
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        let p_leaf = profiles.attach(
            &mut tree,
            Profile::new(
                leaf,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );

        let smaller = std::cmp::min(p_root_a, p_root_b);
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_leaf),
            Some(smaller),
        );
    }

    #[test]
    fn nearest_covering_ancestor_skips_resource_ancestor_lacking_profile() {
        // root(profiled, recursive) → a(NO profile) → b(profiled). The
        // walk from `b` skips the Profile-less Resource `a` and resolves
        // to `p_root` two Resource-hops up. (Migrated from the former
        // external `integration::covers_drives_nearest_covering_ancestor`
        // flavor-2 — `nearest_covering_ancestor` is now engine-internal,
        // so the coverage homes inline beside the function it exercises.)
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        let a = tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        let b = tree
            .ensure_child(a, "b", ResourceRole::User)
            .expect("test live parent");
        for r in [root, a, b] {
            mark_dir(&mut tree, r);
        }
        let p_root = profiles.attach(
            &mut tree,
            Profile::new(
                root,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        // `a` deliberately carries no Profile.
        let p_b = profiles.attach(
            &mut tree,
            Profile::new(
                b,
                ProfileIdentity {
                    config: cfg_recursive(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_b),
            Some(p_root),
            "Profile-less Resource ancestor is skipped; walk continues",
        );
        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_root), None);
    }

    // ===== has_active_standard_descendant =====
    //
    // The derived, never-cached replacement for the old
    // `dirty_descendants > 0` refcount: "is some Active-Standard
    // strict-descendant Profile still on this ancestor's transitive
    // nearest-covering-ancestor chain?" Units pin the load-bearing
    // distinctions: transitive chain ≠ subtree+direct-`covers`, strict
    // (co-anchor excluded), Seed never gates, and chain determinism.

    use specter_core::{
        ActiveBurst, BurstFinish, BurstIntent, CertifiedPrior, DirtyProvenance, PreFireBurst,
        PreFirePhase, ProfileState, TimerId,
    };

    /// A Profile at `r` with `builder`'s config, left `Idle`.
    fn attach_idle(
        tree: &mut Tree,
        profiles: &mut ProfileMap,
        r: ResourceId,
        builder: ScanConfigBuilder,
    ) -> ProfileId {
        profiles.attach(
            tree,
            Profile::new(
                r,
                ProfileIdentity {
                    config: builder.build(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        )
    }

    /// A Profile at `r` driven into an Active burst of `intent`. Phase
    /// is `Batching` — the lightest `Active` state (no `ProbeSlot`, so
    /// no correlation plumbing and no armed-slot Drop tripwire at test
    /// teardown). `has_active_standard_descendant` reads only
    /// `ActiveBurst::intent()`, so the phase is immaterial; the synthetic
    /// timer ids are never inspected (these Profiles are never stepped).
    fn attach_active(
        tree: &mut Tree,
        profiles: &mut ProfileMap,
        r: ResourceId,
        builder: ScanConfigBuilder,
        intent: BurstIntent,
    ) -> ProfileId {
        let mut p = Profile::new(
            r,
            ProfileIdentity {
                config: builder.build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        );
        p.transition_state(ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                burst_deadline: TimerId::from(1),
                phase: PreFirePhase::Batching {
                    settle_timer: TimerId::from(2),
                },
                intent,
                forced: false,
                dirty: DirtyProvenance::new(),
                certified: CertifiedPrior::new(),
                probe_target: r,
                last_event_time: None,
            }),
            BurstFinish::ReturnToIdle,
        ));
        profiles.attach(tree, p)
    }

    #[test]
    fn has_active_standard_descendant_follows_transitive_chain_not_subtree() {
        // a(recursive, max_depth=1) → mid(recursive, unbounded) →
        // deep(Active-Standard). `a` does NOT directly cover `deep`
        // (depth 2 > max_depth 1), but `deep`'s nearest-covering-ancestor
        // chain is deep → p_mid → p_a, so the predicate is true *via the
        // intermediate broader Profile*. Pins: the query is the
        // transitive chain, not the raw subtree filtered by a single
        // direct `covers(a, ·)` test.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let a = tree.ensure_root("a", ResourceRole::User);
        let mid = tree
            .ensure_child(a, "mid", ResourceRole::User)
            .expect("test live parent");
        let deep = tree
            .ensure_child(mid, "deep", ResourceRole::User)
            .expect("test live parent");
        for r in [a, mid, deep] {
            mark_dir(&mut tree, r);
        }
        let p_a = attach_idle(
            &mut tree,
            &mut profiles,
            a,
            recursive_unbounded().max_depth(Some(1)),
        );
        let _ = attach_idle(&mut tree, &mut profiles, mid, recursive_unbounded());
        let _ = attach_active(
            &mut tree,
            &mut profiles,
            deep,
            recursive_unbounded(),
            BurstIntent::Standard,
        );

        assert!(
            !covers(profiles.get(p_a).unwrap(), deep, &tree),
            "premise: `a` does not directly cover `deep` (depth 2 > max_depth 1)",
        );
        assert!(
            has_active_standard_descendant(&tree, &profiles, p_a),
            "true via the transitive chain deep → p_mid → p_a",
        );
        // Sanity: with the deep Profile Idle the predicate is false.
        let mut profiles_idle = ProfileMap::new();
        let mut tree2 = Tree::new();
        let a2 = tree2.ensure_root("a", ResourceRole::User);
        let mid2 = tree2
            .ensure_child(a2, "mid", ResourceRole::User)
            .expect("test live parent");
        let deep2 = tree2
            .ensure_child(mid2, "deep", ResourceRole::User)
            .expect("test live parent");
        for r in [a2, mid2, deep2] {
            mark_dir(&mut tree2, r);
        }
        let p_a2 = attach_idle(
            &mut tree2,
            &mut profiles_idle,
            a2,
            recursive_unbounded().max_depth(Some(1)),
        );
        let _ = attach_idle(&mut tree2, &mut profiles_idle, mid2, recursive_unbounded());
        let _ = attach_idle(&mut tree2, &mut profiles_idle, deep2, recursive_unbounded());
        assert!(!has_active_standard_descendant(
            &tree2,
            &profiles_idle,
            p_a2
        ));
    }

    #[test]
    fn has_active_standard_descendant_excludes_co_anchor_profile() {
        // Two Profiles co-located at the ancestor's own Resource. The
        // strict-subtree DFS starts at `root`'s *children*, so a
        // co-anchor Profile in an Active-Standard burst does NOT make
        // the predicate true for its sibling — matching the old refcount
        // never self-counting.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        mark_dir(&mut tree, root);
        let p_a = attach_idle(
            &mut tree,
            &mut profiles,
            root,
            ScanConfig::builder().recursive(true),
        );
        // A second, distinct Profile co-located at the same Resource:
        // differing `recursive` ⇒ differing config_hash (two Profiles
        // sharing a slot must differ in identity). Its config is
        // irrelevant to the assertion — the exclusion is purely
        // structural (the DFS starts at `root`'s children).
        let _p_x = attach_active(
            &mut tree,
            &mut profiles,
            root,
            ScanConfig::builder().recursive(false),
            BurstIntent::Standard,
        );
        assert!(
            !has_active_standard_descendant(&tree, &profiles, p_a),
            "a co-anchor Active-Standard Profile is not a strict descendant",
        );
    }

    #[test]
    fn has_active_standard_descendant_true_for_one_active_among_idle_siblings() {
        // root → {c1(Idle), c2(Active-Standard), c3(Idle)}. A single
        // witness anywhere in the strict subtree suffices.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        let c1 = tree
            .ensure_child(root, "c1", ResourceRole::User)
            .expect("test live parent");
        let c2 = tree
            .ensure_child(root, "c2", ResourceRole::User)
            .expect("test live parent");
        let c3 = tree
            .ensure_child(root, "c3", ResourceRole::User)
            .expect("test live parent");
        for r in [root, c1, c2, c3] {
            mark_dir(&mut tree, r);
        }
        let p_root = attach_idle(
            &mut tree,
            &mut profiles,
            root,
            ScanConfig::builder().recursive(true),
        );
        let _ = attach_idle(
            &mut tree,
            &mut profiles,
            c1,
            ScanConfig::builder().recursive(true),
        );
        let _ = attach_active(
            &mut tree,
            &mut profiles,
            c2,
            ScanConfig::builder().recursive(true),
            BurstIntent::Standard,
        );
        let _ = attach_idle(
            &mut tree,
            &mut profiles,
            c3,
            ScanConfig::builder().recursive(true),
        );
        assert!(has_active_standard_descendant(&tree, &profiles, p_root));
    }

    #[test]
    fn has_active_standard_descendant_false_for_seed_descendant() {
        // A Seed-burst descendant never gates an ancestor (the old
        // refcount took no `+1` for Seed); the same descendant in a
        // Standard burst does. This is the Seed-vs-Standard behavioral
        // distinction, pinned at the unit level.
        let topo = |intent: BurstIntent| {
            let mut tree = Tree::new();
            let mut profiles = ProfileMap::new();
            let root = tree.ensure_root("root", ResourceRole::User);
            let c = tree
                .ensure_child(root, "c", ResourceRole::User)
                .expect("test live parent");
            mark_dir(&mut tree, root);
            mark_dir(&mut tree, c);
            let p_root = attach_idle(
                &mut tree,
                &mut profiles,
                root,
                ScanConfig::builder().recursive(true),
            );
            let _ = attach_active(
                &mut tree,
                &mut profiles,
                c,
                ScanConfig::builder().recursive(true),
                intent,
            );
            has_active_standard_descendant(&tree, &profiles, p_root)
        };
        assert!(!topo(BurstIntent::Seed), "Seed descendant never gates");
        assert!(
            topo(BurstIntent::Standard),
            "the same descendant in a Standard burst does gate",
        );
    }

    #[test]
    fn has_active_standard_descendant_chain_uses_min_profile_id_tie_break() {
        // a(recursive) → mid → deep(Active-Standard), with two co-located
        // covering Profiles at `mid`. `chain_reaches` climbs
        // `nearest_covering_ancestor`, whose co-anchor tie-break is the
        // smallest ProfileId — deterministically resolving `deep`'s first
        // hop and still reaching `p_a`.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let a = tree.ensure_root("a", ResourceRole::User);
        let mid = tree
            .ensure_child(a, "mid", ResourceRole::User)
            .expect("test live parent");
        let deep = tree
            .ensure_child(mid, "deep", ResourceRole::User)
            .expect("test live parent");
        for r in [a, mid, deep] {
            mark_dir(&mut tree, r);
        }
        let p_a = attach_idle(&mut tree, &mut profiles, a, recursive_unbounded());
        // Two distinct co-located Profiles at `mid`: differing
        // `recursive` ⇒ differing config_hash ⇒ separate ProfileIds.
        // Both still cover `deep` (it is their depth-1 child, which a
        // non-recursive Profile also covers), so the chain hop is a real
        // smallest-covering-id tie-break, not a fallback to the only
        // covering one.
        let p_mid_1 = attach_idle(&mut tree, &mut profiles, mid, recursive_unbounded());
        let p_mid_2 = attach_idle(
            &mut tree,
            &mut profiles,
            mid,
            ScanConfig::builder().recursive(false),
        );
        let p_deep = attach_active(
            &mut tree,
            &mut profiles,
            deep,
            recursive_unbounded(),
            BurstIntent::Standard,
        );
        // Both `mid` Profiles cover `deep` (it is their depth-1 child);
        // the tie-break picks the smaller id.
        let smaller = std::cmp::min(p_mid_1, p_mid_2);
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_deep),
            Some(smaller),
            "deep's first chain hop is the deterministic min-id co-anchor",
        );
        assert!(
            has_active_standard_descendant(&tree, &profiles, p_a),
            "the chain still reaches `p_a` through the resolved `mid` Profile",
        );
    }
}
