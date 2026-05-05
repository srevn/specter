//! `covers(P, R)` predicate.
//!
//! Walks the segment chain from `profile.resource` (the anchor) down to the
//! candidate `target`, evaluating `max_depth`, the `recursive` flag, the
//! exclude globs, and the file pattern (with directory bypass) along the
//! way. The predicate is the gate for two things in the engine: whether an
//! `FsEvent` at `R` should drive `P`'s burst, and whether `R` contributes
//! to `P`'s `watch_demand`.

use smallvec::SmallVec;
use specter_core::{Profile, ResourceId, ResourceKind, Tree};
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
/// any segment along the chain fails to resolve through the `Tree`'s
/// interner.
#[must_use]
pub fn covers(profile: &Profile, target: ResourceId, tree: &Tree) -> bool {
    let anchor = profile.resource;

    if target == anchor {
        return true;
    }

    // Walk target → ancestor chain to anchor; collect segments in reverse
    // (target-to-root), then reverse to root-to-target order. SmallVec
    // avoids both heap allocation in the typical shallow case and
    // `tinyvec`'s `T: Default` bound (`&str` doesn't have a non-static
    // Default).
    let mut rev: SmallVec<[&str; 4]> = SmallVec::new();
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

    if let Some(max) = profile.config.max_depth
        && depth > max
    {
        return false;
    }
    if depth > 1 && !profile.config.recursive {
        return false;
    }

    if !profile.config.exclude.is_empty() {
        let mut rel = PathBuf::new();
        for seg in &rev {
            rel.push(seg);
            for excl in &profile.config.exclude {
                if excl.matches_path(&rel) {
                    return false;
                }
            }
        }
    }

    if let Some(pat) = &profile.config.pattern {
        let target_kind = tree.get(target).map_or(ResourceKind::Unknown, |r| r.kind);
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

#[cfg(test)]
mod tests {
    use super::*;
    use specter_core::{
        ClassSet, GlobPattern, Profile, ResourceKind, ResourceRole, ScanConfig, ScanConfigBuilder,
    };
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    /// Coverage is orthogonal to the events filter; tests use an empty
    /// mask to keep `Profile::new`'s `has_per_file_fds` derivation off.
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Mark `id`'s `ResourceKind` so the pattern check has a stable answer.
    fn mark(tree: &mut Tree, id: ResourceId, kind: ResourceKind) {
        tree.get_mut(id).unwrap().kind = kind;
    }

    /// Anchor a Profile with the supplied `ScanConfig`. Caller still owns the
    /// `Tree` (and any descendant Resources they attach).
    fn anchor(tree: &mut Tree, segment: &str, builder: ScanConfigBuilder) -> (ResourceId, Profile) {
        let r = tree.ensure(None, segment, ResourceRole::User);
        mark(tree, r, ResourceKind::Dir);
        let p = Profile::new(r, builder.build(), MAX_SETTLE, SETTLE, NO_EVENTS);
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
        let r = tree.ensure(None, "log.txt", ResourceRole::User);
        mark(&mut tree, r, ResourceKind::File);
        let p = Profile::new(r, ScanConfig::builder().build(), MAX_SETTLE, SETTLE, NO_EVENTS);
        assert!(covers(&p, r, &tree));
    }

    #[test]
    fn target_equals_anchor_file_is_covered_with_pattern() {
        // Depth-0 bypasses the pattern check. The user anchored here; the
        // file is part of the Profile's scope by construction.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "log.txt", ResourceRole::User);
        mark(&mut tree, r, ResourceKind::File);
        let p = Profile::new(
            r,
            ScanConfig::builder().pattern(glob("*.rs")).build(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
        );
        assert!(covers(&p, r, &tree));
    }

    #[test]
    fn target_outside_anchor_subtree_is_uncovered() {
        let mut tree = Tree::new();
        let (_anchor_id, profile) = anchor(&mut tree, "root", recursive_unbounded());
        let sibling = tree.ensure(None, "sibling", ResourceRole::User);
        mark(&mut tree, sibling, ResourceKind::Dir);
        assert!(!covers(&profile, sibling, &tree));
    }

    #[test]
    fn ancestor_is_uncovered() {
        // covers(P, R) when R is an ancestor of P.resource — not on the
        // descendant chain, returns false.
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "parent", ResourceRole::User);
        mark(&mut tree, parent, ResourceKind::Dir);
        let anchor_id = tree.ensure(Some(parent), "anchor", ResourceRole::User);
        mark(&mut tree, anchor_id, ResourceKind::Dir);
        let profile = Profile::new(
            anchor_id,
            recursive_unbounded().build(),
            MAX_SETTLE,
            SETTLE,
            NO_EVENTS,
        );
        assert!(!covers(&profile, parent, &tree));
    }

    #[test]
    fn recursive_true_covers_deep_descendants() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded());
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        mark(&mut tree, a, ResourceKind::Dir);
        let b = tree.ensure(Some(a), "b", ResourceRole::User);
        mark(&mut tree, b, ResourceKind::Dir);
        let c = tree.ensure(Some(b), "c.rs", ResourceRole::User);
        mark(&mut tree, c, ResourceKind::File);
        assert!(covers(&profile, a, &tree));
        assert!(covers(&profile, b, &tree));
        assert!(covers(&profile, c, &tree));
    }

    #[test]
    fn recursive_false_covers_depth_one_only() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", ScanConfig::builder().recursive(false));
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        mark(&mut tree, a, ResourceKind::Dir);
        let b = tree.ensure(Some(a), "b.rs", ResourceRole::User);
        mark(&mut tree, b, ResourceKind::File);
        assert!(covers(&profile, a, &tree));
        assert!(!covers(&profile, b, &tree));
    }

    #[test]
    fn max_depth_caps_descent() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded().max_depth(Some(2)));
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        let b = tree.ensure(Some(a), "b", ResourceRole::User);
        let c = tree.ensure(Some(b), "c", ResourceRole::User);
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
        let target = tree.ensure(Some(root), "target", ResourceRole::User);
        mark(&mut tree, target, ResourceKind::Dir);
        let other = tree.ensure(Some(root), "src", ResourceRole::User);
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
        let target = tree.ensure(Some(root), "target", ResourceRole::User);
        mark(&mut tree, target, ResourceKind::Dir);
        let foo = tree.ensure(Some(target), "foo", ResourceRole::User);
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
        let rs = tree.ensure(Some(root), "lib.rs", ResourceRole::User);
        mark(&mut tree, rs, ResourceKind::File);
        let c = tree.ensure(Some(root), "lib.c", ResourceRole::User);
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
        let src = tree.ensure(Some(root), "src", ResourceRole::User);
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
        let src = tree.ensure(Some(root), "src", ResourceRole::User);
        mark(&mut tree, src, ResourceKind::Dir);
        let lib = tree.ensure(Some(src), "lib.rs", ResourceRole::User);
        mark(&mut tree, lib, ResourceKind::File);
        let other_src = tree.ensure(Some(root), "other.rs", ResourceRole::User);
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
        let temp = tree.ensure(Some(profile.resource), "ghost", ResourceRole::User);
        mark(&mut tree, temp, ResourceKind::Dir);
        // Need to drop everything that anchors `temp`. `temp` has no children
        // and no profiles attached, role User — so try_reap should succeed.
        assert!(tree.try_reap(temp));
        assert!(tree.get(temp).is_none());
        assert!(!covers(&profile, temp, &tree));
    }

    #[test]
    fn exclude_short_circuits_descendants() {
        // Excluding `a` also rejects `a/b`, since the cumulative path matches
        // at depth 1 already (a Path's glob matcher is checked progressively).
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded().exclude(glob("a")));
        let a = tree.ensure(Some(root), "a", ResourceRole::User);
        mark(&mut tree, a, ResourceKind::Dir);
        let b = tree.ensure(Some(a), "b.rs", ResourceRole::User);
        mark(&mut tree, b, ResourceKind::File);
        assert!(!covers(&profile, a, &tree));
        assert!(!covers(&profile, b, &tree));
    }
}
