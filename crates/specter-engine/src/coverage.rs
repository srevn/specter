//! `covers(P, R)` predicate.
//!
//! Walks the segment chain from `profile.resource` (the anchor) down to the
//! candidate `target`, evaluating `max_depth`, the `recursive` flag, the
//! exclude globs, and the file pattern (with directory bypass) along the
//! way. The predicate is the gate for two things in the engine: whether an
//! `FsEvent` at `R` should drive `P`'s burst, and whether `R` contributes
//! to `P`'s `watch_demand`.

use smallvec::SmallVec;
use specter_core::{Profile, ProfileId, ProfileMap, Resource, ResourceId, ResourceKind, Tree};
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
pub fn covers(profile: &Profile, target: ResourceId, tree: &Tree) -> bool {
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

    if !profile.config().exclude.is_empty() {
        let mut rel = PathBuf::new();
        for seg in &rev {
            rel.push(seg);
            for excl in &profile.config().exclude {
                if excl.matches_path(&rel) {
                    return false;
                }
            }
        }
    }

    if let Some(pat) = &profile.config().pattern {
        // Unprobed slots collapse to File-shape (the backend-mask
        // convention shared by `fs_event_to_class`, the kqueue / inotify
        // translators, and `recompute_events_union`). The prior raw-`kind`
        // form let Unknown bypass the pattern entirely — a file freshly
        // touched in the window between create_child's slot
        // materialization and a follow-up event would slip the user's
        // pattern filter.
        let target_kind = tree
            .get(target)
            .map_or(ResourceKind::File, Resource::kind_or_file);
        if matches!(target_kind, ResourceKind::File) {
            let mut rel = PathBuf::new();
            for seg in &rev {
                rel.push(seg);
            }
            if !pat.matches_path(&rel) {
                return false;
            }
        }
    }

    true
}

/// Resolve the nearest covering ancestor Profile of `child` — the
/// derivation companion to [`covers`].
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
/// Coverage-domain by nature: the derivation is purely a `(tree,
/// profiles, child)` function of the `covers` predicate, with no
/// caching or peer state. The cached parent edge lives on
/// `Profile.parent_profile`; engine-side write paths
/// (`Engine::install_parent_edges_for`,
/// `stability::recompute_parent_edges`) call this function and route
/// the result through `stability::write_parent_edge`.
#[must_use]
pub fn nearest_covering_ancestor(
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
}
