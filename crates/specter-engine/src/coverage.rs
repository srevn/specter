//! The coverage relation, its proof-object refinement, and the reconfirm query derived from it.
//!
//! [`classify`] walks the segment chain from `profile.resource` (the anchor) down to the candidate
//! `target`, delegating each prefix's in-scope test to [`specter_core::ScanConfig::accepts`] — the
//! single source of the scope predicate, shared with the walker — then refines the admitted case
//! by the shape's recursion edge ([`specter_core::ScanConfig::descends_into`], via [`descends_at`])
//! into [`CoverageClass`]. The classification gates three things in the engine: whether an
//! `FsEvent` at `R` should drive `P`'s burst, whether `R` contributes to `P`'s `watch_demand`, and
//! how deep a pre-fire probe target may sit. [`covers`] is the boolean projection (`!= Outside`)
//! the chain queries and external consumers keep.
//!
//! [`nearest_covering_ancestor`] is the transitive derivation of [`covers`], and
//! [`has_active_standard_descendant`] (via [`chain_reaches`]) is the pure query that replaced the
//! old `dirty_descendants` refcount: it answers, fresh at each consult point, "is some
//! Active-Standard strict-descendant Profile still covering this ancestor?" — the `Draining →
//! Verifying` reconfirm condition. Evaluating it as a query rather than maintaining it as a counter
//! is what makes it robust to mid-burst topology moves; the rationale lives on
//! [`has_active_standard_descendant`].

use smallvec::SmallVec;
use specter_core::{
    Profile, ProfileId, ProfileMap, ProfileState, Resource, ResourceId, ResourceKind, ScanConfig,
    Tree,
};
use std::path::PathBuf;

/// Where `target` sits relative to `profile`'s **proof object** — the `(path, attribute)` cells
/// that actually fold into `dir_hash` / `leaf_hash` — not merely whether the scope predicate
/// admits it.
///
/// - [`Self::Outside`] — some prefix on the anchor → target chain fails
///   [`specter_core::ScanConfig::accepts`] (or the chain is stale / doesn't reach the anchor).
///   The target contributes nothing to this Profile.
/// - [`Self::Boundary`] — admitted at every prefix, but the scan shape does not descend below it.
///   **Dir-only by construction**: a covered Dir at depth `d` with `descends_into(d)` false. The
///   walker records such a Dir as `DirChild::Uncovered(fs_id)`, so the proof object folds only its
///   *identity* where the parent enumerated it — member churn inside it is invisible to every
///   verdict. `recursive=false` depth-1 Dirs, `max_depth`-bound Dirs, and every `MatchChain`
///   terminus Dir land here.
/// - [`Self::Interior`] — the proof object observes the node's content: the anchor (depth 0,
///   unconditionally), every covered Leaf (`leaf_hash` folds where the parent enumerated it — a
///   leaf's interior *is* its content), and every covered Dir the shape descends into.
///
/// Consumers are exactly three: event routing (`on_fs_event`'s proof-relevance guard), watch
/// installation (`reconcile::wants_descendant_watch`), and the pre-fire target clamp
/// (`burst::resolve_under_anchor`, via [`descends_at`]). The chain queries
/// ([`nearest_covering_ancestor`] / [`chain_reaches`]) deliberately stay on boolean [`covers`]: a
/// minted Profile's anchor sits on the discovery Profile's terminus slot — `Boundary` for the
/// discovery Profile — and minted bursts gate outer Profiles *through* the discovery Profile by
/// resolving their chain across exactly that boundary coverage. Lifting the classification into
/// the chain queries would sever the gate.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum CoverageClass {
    Outside,
    Boundary,
    Interior,
}

/// The shape's recursion edge as the engine consults it: does coverage extend *below* a Dir at
/// `depth`? Pins `same_device = true` — the engine's `Tree` slots carry no device, so every
/// engine-side consumer is device-blind by construction. A mount below the anchor is therefore
/// classified `Interior` here while the snapshot stores it `Uncovered`-by-device; a graft path
/// crossing one surfaces as `SpliceCrossedUncovered`, the one legitimately reachable residue of
/// that blindness.
#[must_use]
pub(crate) fn descends_at(config: &ScanConfig, depth: u32) -> bool {
    config.descends_into(depth, true)
}

/// True iff `profile` would scan `target` given its `ScanConfig` — the boolean projection of
/// [`classify`] (`!= Outside`).
///
/// The chain queries ([`nearest_covering_ancestor`] / [`chain_reaches`]) and cross-crate consumers
/// read coverage at this granularity; the three proof-object consumers read [`classify`] directly.
#[must_use]
pub fn covers(profile: &Profile, target: ResourceId, tree: &Tree, scratch: &mut PathBuf) -> bool {
    !matches!(
        classify(profile, target, tree, scratch),
        CoverageClass::Outside
    )
}

/// Classify `target` against `profile`'s proof object — see [`CoverageClass`] for the vocabulary.
///
/// **Depth-0 (`target == profile.resource`).** Always [`CoverageClass::Interior`]. The anchor is
/// part of the Profile's scope by construction — `FsEvent`s at the anchor must drive the anchor's
/// burst, so coverage at the anchor is unconditional. [`specter_core::ScanConfig::accepts`]
/// bypasses every filter at depth 0 by the same rule.
///
/// **Descendants.** Build the cumulative relative path segment-by-segment from `profile.resource`
/// to `target`, calling `accepts` at each prefix. Intermediate prefixes are typed
/// [`ResourceKind::Dir`] (the Tree invariant: a non-Dir parent can't have children); the final
/// prefix's kind is `target`'s own kind, collapsed to `File` for unprobed slots
/// ([`Resource::kind_or_file`], matching the backend-mask convention shared with
/// `fs_event_to_class` and the kqueue / inotify translators — the File collapse also lands
/// unprobed slots on `Interior`, deliberately conservative-toward-driving: the probe and the
/// target clamp sort out what the slot really is). A failure at any prefix short-circuits to
/// [`CoverageClass::Outside`]; an admitted Dir target then forks `Interior` / `Boundary` on the
/// recursion edge at its own depth ([`descends_at`]).
///
/// Per-prefix evaluation through the same predicate the walker consumes is what makes
/// pattern/exclude/depth/recursive/hidden semantics co-evolve across the two callers. Drift is
/// structurally impossible only because both also measure from the same origin: this builds each
/// prefix's `rel` from the anchor, and the walker strips every dirent against the anchor shipped on
/// `ProbeRequest::Subtree`'s `anchor_path` — same predicate body, same basis. A walker measuring
/// `rel` from a deeper recursion root would silently diverge here.
///
/// Returns [`CoverageClass::Outside`] if `target` is not on the descendant chain of
/// `profile.resource` (sibling, ancestor, or unrelated subtree), or if any node along the chain is
/// stale (its `ResourceId` no longer names a live slot).
#[must_use]
pub(crate) fn classify(
    profile: &Profile,
    target: ResourceId,
    tree: &Tree,
    scratch: &mut PathBuf,
) -> CoverageClass {
    let anchor = profile.resource();

    if target == anchor {
        return CoverageClass::Interior;
    }

    // Walk target → ancestor chain to anchor; collect segments in reverse (target-to-root), then
    // reverse to root-to-target order. Inline cap of 8 covers typical source-tree depths from a
    // workspace anchor (`src/foo/bar/baz/qux/file.rs` is 6 deep); cap 4 spilled on every such path.
    //
    // Termination relies on the `Tree` acyclicity invariant: each `parent()` step strictly ascends,
    // so the walk reaches `anchor` or bottoms out at a root (`None`) in at most `depth(target)`
    // steps. Intentionally not depth-bounded — a defensive cap here would mask a real
    // `Tree`-construction cycle bug instead of surfacing it (mirrors
    // `snapshot::tree::ancestor_chain`).
    let mut rev: SmallVec<[&str; 8]> = SmallVec::new();
    let mut cur = target;
    loop {
        let Some(resource) = tree.get(cur) else {
            return CoverageClass::Outside;
        };
        let Some(segment_str) = tree.name(cur) else {
            return CoverageClass::Outside;
        };
        rev.push(segment_str);
        match resource.parent() {
            Some(p) if p == anchor => break,
            Some(p) => cur = p,
            None => return CoverageClass::Outside,
        }
    }
    rev.reverse();

    let total = rev.len();
    // Hoist `target`'s kind out of the per-prefix loop — every intermediate prefix is `Dir` (Tree
    // invariant), only the final prefix's kind feeds the pattern arm of `accepts`. `kind_or_file`
    // collapses unprobed slots to File-shape, matching the backend- mask convention shared with
    // `fs_event_to_class` and the kqueue / inotify translators — a freshly-touched file in the
    // window between `create_child`'s slot materialization and the follow-up probe is still
    // pattern-filtered (raw-`kind` Unknown would have bypassed the pattern).
    let target_kind = tree
        .get(target)
        .map_or(ResourceKind::File, Resource::kind_or_file);

    let config = profile.config();
    // One incremental build into the engine-owned `scratch` (capacity retained across calls;
    // `clear()` per call so the cross-call residue is never observable). `scratch.as_path()` after
    // `push` is the cumulative relative path the predicate consumes — the same shape the walker
    // passes (`child_path.strip_prefix(anchor_path)`). An early `return` mid-walk leaves `scratch`
    // dirty; the next entry's `clear()` is the reset.
    scratch.clear();
    for (i, seg) in rev.iter().enumerate() {
        scratch.push(seg);
        let depth = u32::try_from(i + 1).unwrap_or(u32::MAX);
        let kind = if i + 1 == total {
            target_kind
        } else {
            ResourceKind::Dir
        };
        if !config.accepts(scratch.as_path(), kind, depth) {
            return CoverageClass::Outside;
        }
    }

    // Every prefix admitted. A non-Dir target is its own proof object (`leaf_hash` folds where the
    // parent enumerated it); an admitted Dir forks on the recursion edge at its own depth — the
    // shape descends into it (its entries fold) or it is a boundary (only its `fs_id` folds).
    let target_depth = u32::try_from(total).unwrap_or(u32::MAX);
    if !matches!(target_kind, ResourceKind::Dir) || descends_at(config, target_depth) {
        CoverageClass::Interior
    } else {
        CoverageClass::Boundary
    }
}

/// Resolve the nearest covering ancestor Profile of `child` — the derivation companion to
/// [`covers`], and the **query core** of the `Draining → Verifying` reconfirm.
///
/// Walks Resource ancestors of `child.resource`; at each ancestor Resource, picks the smallest
/// covering [`ProfileId`] for a deterministic tie-break. Returns `None` for root Profiles whose
/// ancestor chain holds no covering Profile.
///
/// "Nearest ancestor *Profile*, not Resource" is the easy mistake: a Resource ancestor with no
/// Profile is skipped; the walk continues to the next Resource ancestor.
///
/// **Pure, never cached.** The result is a total function of `(tree, profiles, child)` — no peer
/// state, no stored edge. It is not cached into a per-Profile `parent_profile` edge: a refcount
/// keyed on a recomputable derivation cannot be kept balanced across mid-burst topology moves. The
/// derivation stands alone: [`chain_reaches`] climbs it hop-by-hop and
/// [`has_active_standard_descendant`] evaluates the reconfirm predicate fresh from it. `pub(crate)`
/// — engine-internal; no cross-crate consumer.
///
/// Each `child → result` step is a strict Resource-ancestor move (`tree.ancestors` is strict, and
/// the same-Resource co-anchor case is excluded), so iterating it ([`chain_reaches`]) terminates
/// structurally — a cycle is unrepresentable, with no self-edge assertion needed.
#[must_use]
pub(crate) fn nearest_covering_ancestor(
    tree: &Tree,
    profiles: &ProfileMap,
    child: ProfileId,
) -> Option<ProfileId> {
    let child_resource = profiles.get(child)?.resource();
    // Cold path (a Draining-phase query, not the per-event hot path): own a local scratch reused
    // across the ancestor loop's `covers` calls. The signature stays clean — threading `&mut
    // PathBuf` through this pure derivation and its `chain_reaches` /
    // `has_active_standard_descendant` callers would muddy their "total function of (tree,
    // profiles, child)" contract for an allocation the cold path does not feel.
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

/// Walk `resource` and its strict ancestors looking for Profiles whose [`classify`] is not
/// `Outside` for `resource`, pairing each with its [`CoverageClass`] — the per-Profile class feeds
/// the routing guard's proof-relevance fork (the same slot can be `Interior` for one Profile and
/// `Boundary` for a co-located other). Returns matches in encounter order. P4 single-Profile
/// resolves to 0 or 1. `pub(crate)` — the sole caller is `Engine::on_fs_event`; a coverage
/// derivation co-located with [`covers`] / [`nearest_covering_ancestor`].
///
/// **Pending Profiles are filtered at the source.** A Pending Profile carries no anchor-side
/// `watch_demand` from this Profile — the descent prefix carries it instead (via
/// [`specter_core::ContribKey::ProfileDescent`]); the anchor slot itself only receives the
/// [`specter_core::ContribKey::ProfileAnchor`] contribution at descent-completion time. Events at
/// the prefix route via `classify_event_carriers` / `on_descent_event`; events at the anchor or its
/// descendants are structurally unreachable in production (the anchor's `watch_demand` is 0 ⇒ head
/// guard short-circuits). Filtering here makes the routing contract explicit: covering-Profile
/// dispatch (Standard burst, anchor terminal event) only sees Profiles with a materialized anchor.
#[must_use]
pub(crate) fn covering_profiles(
    tree: &Tree,
    profiles: &ProfileMap,
    resource: ResourceId,
    scratch: &mut PathBuf,
) -> SmallVec<[(ProfileId, CoverageClass); 2]> {
    let mut out: SmallVec<[(ProfileId, CoverageClass); 2]> = SmallVec::new();
    let mut cur = Some(resource);
    while let Some(rid) = cur {
        for pid in profiles.at(rid) {
            let Some(p) = profiles.get(pid) else {
                continue;
            };
            if matches!(p.state(), ProfileState::Pending(_)) {
                continue;
            }
            let class = classify(p, resource, tree, scratch);
            if !matches!(class, CoverageClass::Outside) && !out.iter().any(|&(q, _)| q == pid) {
                out.push((pid, class));
            }
        }
        cur = tree.parent(rid);
    }
    out
}

/// True iff some **strict-descendant** Profile of `ancestor`'s subtree is in an Active **Standard**
/// burst (any phase — pre- or post-fire) **and** has `ancestor` on its transitive
/// [`nearest_covering_ancestor`] chain.
///
/// A derived, never-cached predicate over the *transitive nearest-covering-ancestor chain* —
/// **not** the raw Tree subtree and **not** a single direct [`covers`] test (`covers` is not
/// transitive: an intermediate broader Profile keeps a deeper one on `ancestor`'s chain even where
/// `ancestor`'s own `max_depth`/`pattern` would exclude it). Evaluated fresh at each of its two
/// consult points — the `gated_fire` Draining gate and the `finish_burst_to_idle` Draining sweep —
/// never accumulated, so no mid-burst topology move can desynchronise it.
///
/// Iterative DFS over the **strict** Tree descendants of `ancestor.resource` (starts at its
/// children, so `ancestor` itself and any co-anchor Profile sharing its slot are excluded —
/// matching the old refcount never self-counting). The strict subtree is a sound superset of `{D :
/// ancestor ∈ chain(D)}` (every chain link is a Resource-ancestor, so a contributing `D.resource`
/// is necessarily a Tree-descendant of `ancestor.resource`); [`chain_reaches`] is the exact filter.
/// Short-circuits on the first witness.
///
/// **Chain-shaped (`MatchChain`) descendants are excluded.** The gate exists to keep an ancestor's
/// "tree settled" command from racing descendant *command* activity; a discovery burst fires no
/// command (its consequence is a reconcile) and is N=1-short, so holding an ancestor in Draining
/// for it would defer real work for nothing. Both consumers — the `gated_fire` gate and the
/// `finish_burst_to_idle` Draining-exit sweep — re-evaluate this same query, so they inherit the
/// filter together. The exclusion is *not* transitive: [`chain_reaches`] stays shape-agnostic, so a
/// mid-burst **minted** Standard descendant still holds the outer ancestor, resolving its chain
/// *through* the discovery Profile.
pub(crate) fn has_active_standard_descendant(
    tree: &Tree,
    profiles: &ProfileMap,
    ancestor: ProfileId,
) -> bool {
    let Some(root) = profiles.get(ancestor).map(Profile::resource) else {
        return false;
    };
    // Strict descendants only: seed the stack with `root`'s children, never `root` itself.
    let mut stack: Vec<ResourceId> = tree.children_ids(root).collect();
    while let Some(node) = stack.pop() {
        for d in profiles.at(node) {
            if profiles.get(d).is_some_and(|p| {
                p.state().in_active_standard_burst() && p.config().match_chain().is_none()
            }) && chain_reaches(tree, profiles, d, ancestor)
            {
                return true;
            }
        }
        stack.extend(tree.children_ids(node));
    }
    false
}

/// True iff `ancestor` lies on `descendant`'s transitive [`nearest_covering_ancestor`] chain — i.e.
/// climbing `descendant → nca(descendant) → nca(nca(descendant)) → …` reaches `ancestor`. This is
/// exactly the chain the deleted `propagate` walked via the cached `parent_profile` edge (that edge
/// *was* `nearest_covering_ancestor`'s result), recomputed fresh.
///
/// Terminates structurally: every hop is a strict Resource-ancestor move (see
/// [`nearest_covering_ancestor`]), so the climb is bounded by `descendant`'s Tree depth and a cycle
/// cannot form. A `descendant == ancestor` call (not produced by [`has_active_standard_descendant`]'s
/// strict walk, but well-defined) returns `false`: the chain is strictly above `descendant`.
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
    /// Coverage is orthogonal to the events filter; tests use an empty mask to keep
    /// `Profile::new`'s `has_per_file_fds` derivation off.
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Mark `id`'s `ResourceKind` so the pattern check has a stable answer.
    fn mark(tree: &mut Tree, id: ResourceId, kind: ResourceKind) {
        tree.set_kind(id, kind);
    }

    /// The module's one `Profile` fixture: `config` under the supplied `max_settle`, `NO_EVENTS`,
    /// `SETTLE`, unclassified kind. Coverage is a pure function of `(resource, config)`; the other
    /// identity axes are held constant so every test reads as a config-shape variation.
    fn profile_with(r: ResourceId, config: ScanConfig, max_settle: Duration) -> Profile {
        Profile::new(
            r,
            ProfileIdentity::new(config, max_settle, NO_EVENTS),
            SETTLE,
            None,
        )
    }

    /// [`profile_with`] at the canonical `MAX_SETTLE` — the dominant shape. Tests that need two
    /// co-located Profiles fork identity through `profile_with`'s explicit `max_settle` instead.
    fn profile_at(r: ResourceId, config: ScanConfig) -> Profile {
        profile_with(r, config, MAX_SETTLE)
    }

    /// Anchor a Profile with the supplied `ScanConfig`. Caller still owns the `Tree` (and any
    /// descendant Resources they attach).
    fn anchor(tree: &mut Tree, segment: &str, builder: ScanConfigBuilder) -> (ResourceId, Profile) {
        let r = tree.ensure_root(segment, ResourceRole::User);
        mark(tree, r, ResourceKind::Dir);
        (r, profile_at(r, builder.build()))
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
        assert!(covers(&profile, anchor_id, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn target_equals_anchor_file_is_covered_no_pattern() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("log.txt", ResourceRole::User);
        mark(&mut tree, r, ResourceKind::File);
        let p = profile_at(r, ScanConfig::builder().build());
        assert!(covers(&p, r, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn target_equals_anchor_file_is_covered_with_pattern() {
        // Depth-0 bypasses the pattern check. The user anchored here; the file is part of the
        // Profile's scope by construction.
        let mut tree = Tree::new();
        let r = tree.ensure_root("log.txt", ResourceRole::User);
        mark(&mut tree, r, ResourceKind::File);
        let p = profile_at(r, ScanConfig::builder().pattern(glob("*.rs")).build());
        assert!(covers(&p, r, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn target_outside_anchor_subtree_is_uncovered() {
        let mut tree = Tree::new();
        let (_anchor_id, profile) = anchor(&mut tree, "root", recursive_unbounded());
        let sibling = tree.ensure_root("sibling", ResourceRole::User);
        mark(&mut tree, sibling, ResourceKind::Dir);
        assert!(!covers(&profile, sibling, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn ancestor_is_uncovered() {
        // covers(P, R) when R is an ancestor of P.resource — not on the descendant chain, returns
        // false.
        let mut tree = Tree::new();
        let parent = tree.ensure_root("parent", ResourceRole::User);
        mark(&mut tree, parent, ResourceKind::Dir);
        let anchor_id = tree
            .ensure_child(parent, "anchor", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, anchor_id, ResourceKind::Dir);
        let profile = profile_at(anchor_id, recursive_unbounded().build());
        assert!(!covers(&profile, parent, &tree, &mut PathBuf::new()));
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
        assert!(covers(&profile, a, &tree, &mut PathBuf::new()));
        assert!(covers(&profile, b, &tree, &mut PathBuf::new()));
        assert!(covers(&profile, c, &tree, &mut PathBuf::new()));
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
        assert!(covers(&profile, a, &tree, &mut PathBuf::new()));
        assert!(!covers(&profile, b, &tree, &mut PathBuf::new()));
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
        assert!(covers(&profile, a, &tree, &mut PathBuf::new())); // depth 1
        assert!(covers(&profile, b, &tree, &mut PathBuf::new())); // depth 2
        assert!(!covers(&profile, c, &tree, &mut PathBuf::new())); // depth 3 > max
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
        assert!(!covers(&profile, target, &tree, &mut PathBuf::new()));
        assert!(covers(&profile, other, &tree, &mut PathBuf::new()));
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
        assert!(!covers(&profile, foo, &tree, &mut PathBuf::new()));
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
        assert!(covers(&profile, rs, &tree, &mut PathBuf::new()));
        assert!(!covers(&profile, c, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn pattern_bypasses_directories() {
        // dirs are always covered (we descend through them) regardless of the file pattern.
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
        assert!(covers(&profile, src, &tree, &mut PathBuf::new()));
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
        assert!(covers(&profile, lib, &tree, &mut PathBuf::new()));
        // `other.rs` lives directly under root and doesn't match `src/**/*.rs`.
        assert!(!covers(&profile, other_src, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn stale_resource_id_returns_false() {
        let mut tree = Tree::new();
        let (_root, profile) = anchor(&mut tree, "root", recursive_unbounded());
        // Build a Resource then drop it via reap so the id is stale.
        let temp = tree
            .ensure_child(profile.resource(), "ghost", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, temp, ResourceKind::Dir);
        // Need to drop everything that anchors `temp`. `temp` has no children and no profiles
        // attached, role User — so try_reap should succeed.
        assert!(tree.try_reap(temp, &mut specter_core::StepOutput::default()));
        assert!(tree.get(temp).is_none());
        assert!(!covers(&profile, temp, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn exclude_short_circuits_descendants() {
        // Excluding `a` also rejects `a/b`, since the cumulative path matches at depth 1 already (a
        // Path's glob matcher is checked progressively).
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
        assert!(!covers(&profile, a, &tree, &mut PathBuf::new()));
        assert!(!covers(&profile, b, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn star_pattern_descends_through_directories() {
        // globset's `*` does NOT respect `/` — `*.rs` is "any path ending in `.rs`," not
        // basename-only. Pin the semantic so a future glob-engine swap surfaces the deviation as a
        // test failure rather than a silent change in user-visible coverage.
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
        assert!(covers(&profile, lib, &tree, &mut PathBuf::new()));
        assert!(covers(&profile, deep, &tree, &mut PathBuf::new()));
        assert!(covers(&profile, deepest, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn pattern_applies_to_unprobed_kind_targets() {
        // An FsEvent on a descendant whose kind hasn't been classified yet (Resource::kind() ==
        // None) must still be filtered by the user's pattern. The prior raw-`kind` form in `covers`
        // let Unknown-kind slots bypass the pattern entirely. The new `kind_or_file()` accessor
        // collapses unprobed to File-shape, matching the convention shared by `fs_event_to_class`.
        let mut tree = Tree::new();
        let (root, profile) = anchor(
            &mut tree,
            "root",
            recursive_unbounded().pattern(glob("*.rs")),
        );
        let unprobed = tree
            .ensure_child(root, "lib.c", ResourceRole::User)
            .expect("test live parent");
        // Deliberately do NOT call `mark` — kind stays at the default `ResourceKind::Unknown`
        // placeholder.
        assert!(tree.get(unprobed).unwrap().kind().is_none());
        // Pattern is `*.rs`; lib.c does not match. Unprobed must still be filtered out.
        assert!(!covers(&profile, unprobed, &tree, &mut PathBuf::new()));
    }

    #[test]
    fn double_star_exclude_does_not_match_directory_itself() {
        // `target/**` matches `target/foo` but not `target` literally. The directory itself remains
        // covered; only its contents are excluded. Surprises users coming from gitignore (where
        // `target/` excludes the directory and contents); pinned here so the contrast is explicit.
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
            covers(&profile, target, &tree, &mut PathBuf::new()),
            "`target/**` does not match `target` literally — the directory \
             itself stays covered",
        );
        assert!(
            !covers(&profile, inside, &tree, &mut PathBuf::new()),
            "`target/foo` matches `target/**` — contents excluded",
        );
    }

    /// Load-bearing regression pin for the predicate unification. Pre-fix, `covers` reimplemented
    /// its own filter and missed the `hidden` gate — an FS event at a `.hidden` descendant drove
    /// the parent Profile's burst even when the walker had filtered the dirent out of the snapshot.
    /// Multi-Profile setups with mixed `hidden` settings ended up with spurious wake-ups + spurious
    /// watch contributions. After the migration both consumers share `ScanConfig::accepts`, so
    /// `hidden=false` rejects the segment at every depth.
    ///
    /// The basename test is per-segment by construction (the predicate is called at every prefix on
    /// the chain anchor → target). Both "hidden at depth 1" and "hidden at depth 2" are exercised
    /// here because they hit different short-circuit arms of the chain walker: a depth-1 hidden
    /// segment rejects at the first iteration; a depth-2 case proves the cumulative-path walk
    /// doesn't lose the gate when a non-hidden depth-1 segment precedes it.
    #[test]
    fn covers_rejects_hidden_segments_when_hidden_false() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded());
        let hidden_dir = tree
            .ensure_child(root, ".hidden", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, hidden_dir, ResourceKind::Dir);
        let inside_hidden = tree
            .ensure_child(hidden_dir, "leaf.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, inside_hidden, ResourceKind::File);
        let sub = tree
            .ensure_child(root, "src", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, sub, ResourceKind::Dir);
        let hidden_under_sub = tree
            .ensure_child(sub, ".secret", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, hidden_under_sub, ResourceKind::Dir);

        assert!(
            !covers(&profile, hidden_dir, &tree, &mut PathBuf::new()),
            "depth-1 hidden segment must be rejected"
        );
        assert!(
            !covers(&profile, inside_hidden, &tree, &mut PathBuf::new()),
            "descendant of a hidden segment is also out of scope"
        );
        assert!(
            covers(&profile, sub, &tree, &mut PathBuf::new()),
            "sanity: a non-hidden sibling stays covered"
        );
        assert!(
            !covers(&profile, hidden_under_sub, &tree, &mut PathBuf::new()),
            "depth-2 hidden segment must be rejected — the per-segment \
             basename test reaches every prefix"
        );
    }

    // ===== classify (CoverageClass) =====
    //
    // The three-way proof-object refinement consumed by event routing, watch installation, and the
    // pre-fire clamp. These pin the classification's load-bearing distinctions: Boundary is
    // Dir-only (a covered Leaf at the same depth is Interior — its `leaf_hash` is in the proof
    // object), the anchor is Interior unconditionally, and the unprobed File-shape collapse lands
    // on Interior. The admission walk itself (pattern / exclude / hidden / depth arms) is pinned by
    // the `covers` grid above — `covers` is `classify != Outside`, so those tests already exercise
    // the shared body.

    #[test]
    fn classify_recursive_false_forks_boundary_dir_from_interior_leaf_at_depth_one() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", ScanConfig::builder().recursive(false));
        let dir = tree
            .ensure_child(root, "sub", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, dir, ResourceKind::Dir);
        let file = tree
            .ensure_child(root, "f.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, file, ResourceKind::File);
        let deep = tree
            .ensure_child(dir, "deep", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, deep, ResourceKind::Dir);

        let mut scratch = PathBuf::new();
        assert_eq!(
            classify(&profile, dir, &tree, &mut scratch),
            CoverageClass::Boundary,
            "covered depth-1 Dir under recursive=false is not descended into",
        );
        assert_eq!(
            classify(&profile, file, &tree, &mut scratch),
            CoverageClass::Interior,
            "a covered Leaf at the same depth is Interior — leaf_hash is in the proof object",
        );
        assert_eq!(
            classify(&profile, deep, &tree, &mut scratch),
            CoverageClass::Outside,
            "the boundary's interior is Outside, not Boundary",
        );
    }

    #[test]
    fn classify_max_depth_rim_dir_is_boundary_interior_above() {
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", recursive_unbounded().max_depth(Some(2)));
        let d1 = tree
            .ensure_child(root, "d1", ResourceRole::User)
            .expect("test live parent");
        let d2 = tree
            .ensure_child(d1, "d2", ResourceRole::User)
            .expect("test live parent");
        for r in [d1, d2] {
            mark(&mut tree, r, ResourceKind::Dir);
        }
        let f2 = tree
            .ensure_child(d1, "f.rs", ResourceRole::User)
            .expect("test live parent");
        mark(&mut tree, f2, ResourceKind::File);

        let mut scratch = PathBuf::new();
        assert_eq!(
            classify(&profile, d1, &tree, &mut scratch),
            CoverageClass::Interior,
            "depth 1 < max_depth: the shape descends into it",
        );
        assert_eq!(
            classify(&profile, d2, &tree, &mut scratch),
            CoverageClass::Boundary,
            "the max_depth rim Dir is admitted but not descended into",
        );
        assert_eq!(
            classify(&profile, f2, &tree, &mut scratch),
            CoverageClass::Interior,
            "a File on the rim depth is Interior",
        );
    }

    #[test]
    fn classify_anchor_is_interior_unconditionally() {
        // recursive=false would make any non-anchor Dir at this position a Boundary; depth 0
        // bypasses the recursion edge entirely. Same for a File anchor.
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", ScanConfig::builder().recursive(false));
        assert_eq!(
            classify(&profile, root, &tree, &mut PathBuf::new()),
            CoverageClass::Interior,
        );
        let f = tree.ensure_root("log.txt", ResourceRole::User);
        mark(&mut tree, f, ResourceKind::File);
        let fp = profile_at(f, ScanConfig::builder().build());
        assert_eq!(
            classify(&fp, f, &tree, &mut PathBuf::new()),
            CoverageClass::Interior,
        );
    }

    #[test]
    fn classify_unprobed_slot_at_boundary_depth_collapses_to_interior() {
        // An unprobed slot (kind None) collapses to File-shape (`kind_or_file`), so a slot that
        // *would* be Boundary if it turned out to be a Dir classifies Interior — deliberately
        // conservative toward driving; the probe and the target clamp sort out what it really is.
        let mut tree = Tree::new();
        let (root, profile) = anchor(&mut tree, "root", ScanConfig::builder().recursive(false));
        let unprobed = tree
            .ensure_child(root, "sub", ResourceRole::User)
            .expect("test live parent");
        assert!(tree.get(unprobed).unwrap().kind().is_none());
        assert_eq!(
            classify(&profile, unprobed, &tree, &mut PathBuf::new()),
            CoverageClass::Interior,
        );
    }

    // ===== positional covers (MatchChain) =====
    //
    // `covers` reuses its chain walk unchanged — the positional shape only swaps the per-prefix
    // predicate. These pin the covers-specific compositions (cumulative-`rel` depth feeding
    // `matches_at`; the final prefix's `kind_or_file` collapse meeting the kinded chain rule); the
    // predicate arms themselves are the `scan_config` grid's.

    use specter_core::PatternSpec;
    use std::sync::Arc;

    /// Anchor a Profile whose scan shape is the positional chain parsed from `pattern`. The anchor
    /// segment is the pattern's literal-prefix tail by convention; `covers` itself never re-derives
    /// that correspondence (it measures depths from `profile.resource`).
    fn anchor_chain(tree: &mut Tree, segment: &str, pattern: &str) -> (ResourceId, Profile) {
        let r = tree.ensure_root(segment, ResourceRole::User);
        mark(tree, r, ResourceKind::Dir);
        let config = ScanConfig::MatchChain(Arc::new(
            PatternSpec::parse(pattern).expect("test pattern parses"),
        ));
        (r, profile_at(r, config))
    }

    fn child(tree: &mut Tree, parent: ResourceId, segment: &str, kind: ResourceKind) -> ResourceId {
        let r = tree
            .ensure_child(parent, segment, ResourceRole::User)
            .expect("test live parent");
        mark(tree, r, kind);
        r
    }

    #[test]
    fn match_chain_covers_chain_prefixes_and_stops_below_terminus() {
        // `/srv/*/data/*/log`, anchored at `srv` (terminus depth 4). Every matching chain prefix is
        // covered; the slot below the terminus is not — discovery never covers below a terminus, so
        // a minted Profile owns that subtree without overlap; and a non-matching sibling rejects at
        // its position.
        let mut tree = Tree::new();
        let (root, profile) = anchor_chain(&mut tree, "srv", "/srv/*/data/*/log");
        let app1 = child(&mut tree, root, "app1", ResourceKind::Dir);
        let data = child(&mut tree, app1, "data", ResourceKind::Dir);
        let box1 = child(&mut tree, data, "box1", ResourceKind::Dir);
        let log = child(&mut tree, box1, "log", ResourceKind::Dir);
        let below = child(&mut tree, log, "below", ResourceKind::Dir);
        let etc = child(&mut tree, app1, "etc", ResourceKind::Dir);

        let mut scratch = PathBuf::new();
        assert!(covers(&profile, app1, &tree, &mut scratch));
        assert!(covers(&profile, data, &tree, &mut scratch));
        assert!(covers(&profile, box1, &tree, &mut scratch));
        assert!(
            covers(&profile, log, &tree, &mut scratch),
            "a Dir terminus is covered",
        );
        assert!(
            !covers(&profile, below, &tree, &mut scratch),
            "discovery never covers below a terminus",
        );
        assert!(
            !covers(&profile, etc, &tree, &mut scratch),
            "a non-matching segment rejects at its chain position",
        );
    }

    #[test]
    fn classify_match_chain_terminus_dir_boundary_mid_chain_interior_file_terminus_interior() {
        // The discovery shape's proof object is the match set: mid-chain Dirs are walked through
        // (Interior); a Dir terminus is matched-but-unexplored (Boundary — only its membership in
        // the parent's enumeration folds); a File terminus is its own proof object (Interior).
        let mut tree = Tree::new();
        let (root, profile) = anchor_chain(&mut tree, "srv", "/srv/*/log");
        let mid = child(&mut tree, root, "app1", ResourceKind::Dir);
        let dir_terminus = child(&mut tree, mid, "log", ResourceKind::Dir);
        let mut scratch = PathBuf::new();
        assert_eq!(
            classify(&profile, mid, &tree, &mut scratch),
            CoverageClass::Interior,
        );
        assert_eq!(
            classify(&profile, dir_terminus, &tree, &mut scratch),
            CoverageClass::Boundary,
        );

        let (root2, glob_profile) = anchor_chain(&mut tree, "srv2", "/srv2/*/*.log");
        let mid2 = child(&mut tree, root2, "app", ResourceKind::Dir);
        let file_terminus = child(&mut tree, mid2, "x.log", ResourceKind::File);
        assert_eq!(
            classify(&glob_profile, file_terminus, &tree, &mut scratch),
            CoverageClass::Interior,
        );
    }

    #[test]
    fn match_chain_covers_file_terminus_via_glob() {
        // `/srv/*/*.log` — a Glob terminus admitting File targets at depth 2; the terminus pattern
        // still discriminates.
        let mut tree = Tree::new();
        let (root, profile) = anchor_chain(&mut tree, "srv", "/srv/*/*.log");
        let app1 = child(&mut tree, root, "app1", ResourceKind::Dir);
        let log = child(&mut tree, app1, "x.log", ResourceKind::File);
        let txt = child(&mut tree, app1, "x.txt", ResourceKind::File);
        let mut scratch = PathBuf::new();
        assert!(covers(&profile, log, &tree, &mut scratch));
        assert!(!covers(&profile, txt, &tree, &mut scratch));
    }

    #[test]
    fn match_chain_rejects_mid_chain_non_dir_and_unclassified_slots() {
        // Mid-chain positions name directories the chain descends through. A File target there is
        // out of scope — and so is an *unclassified* slot: `covers` collapses Unknown to File-shape
        // (`kind_or_file`), which composes with the kinded chain rule. The walker can never pin
        // this case (it always knows the kind post-`lstat`); only `covers` sees unprobed slots.
        let mut tree = Tree::new();
        let (root, profile) = anchor_chain(&mut tree, "srv", "/srv/*/log");
        let as_file = child(&mut tree, root, "app1", ResourceKind::File);
        let unclassified = tree
            .ensure_child(root, "app2", ResourceRole::User)
            .expect("test live parent");
        assert!(tree.get(unclassified).unwrap().kind().is_none());
        let as_dir = child(&mut tree, root, "app3", ResourceKind::Dir);

        let mut scratch = PathBuf::new();
        assert!(
            !covers(&profile, as_file, &tree, &mut scratch),
            "a mid-chain File is out of scope",
        );
        assert!(
            !covers(&profile, unclassified, &tree, &mut scratch),
            "an unprobed slot collapses to File-shape and rejects mid-chain",
        );
        assert!(
            covers(&profile, as_dir, &tree, &mut scratch),
            "control: a classified Dir at the same position is covered",
        );
    }

    // ===== nearest_covering_ancestor =====
    //
    // Resolution tests for the `covers` derivation. Walks Resource ancestors of a child Profile's
    // anchor; at each ancestor, picks the smallest covering [`ProfileId`].

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
        let pid = profiles.attach(&mut tree, profile_at(r, cfg_recursive()));
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
        let p_root = profiles.attach(&mut tree, profile_at(root, cfg_recursive()));
        let p_a = profiles.attach(&mut tree, profile_at(a, cfg_recursive()));
        let p_b = profiles.attach(&mut tree, profile_at(b, cfg_recursive()));

        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_b), Some(p_a));
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_a),
            Some(p_root)
        );
        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_root), None);
    }

    #[test]
    fn nearest_covering_ancestor_skips_non_covering_ancestor() {
        // root has Profile p_root with recursive=false → does not cover deep descendants. The deeper
        // Profile's resolution must walk past p_root and return None (no further covering ancestor).
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
            profile_at(root, ScanConfig::builder().recursive(false).build()),
        );
        let p_b = profiles.attach(&mut tree, profile_at(b, cfg_recursive()));
        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_b), None);
    }

    #[test]
    fn nearest_covering_ancestor_excludes_self() {
        // Two co-located Profiles at the anchor; resolution for one must not return itself.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let r = tree.ensure_root("root", ResourceRole::User);
        mark_dir(&mut tree, r);
        let p_a = profiles.attach(
            &mut tree,
            profile_with(r, cfg_recursive(), Duration::from_secs(6)),
        );
        let p_b = profiles.attach(
            &mut tree,
            profile_with(r, cfg_recursive(), Duration::from_secs(12)),
        );
        // Both at root; root has no Profile ancestor; resolution walks ancestors of root.resource
        // (none — root is a Tree root).
        assert!(nearest_covering_ancestor(&tree, &profiles, p_a).is_none());
        assert!(nearest_covering_ancestor(&tree, &profiles, p_b).is_none());
    }

    #[test]
    fn nearest_covering_ancestor_ties_by_smallest_profile_id() {
        // Two co-located covering Profiles at the same ancestor Resource. Resolution for a deeper
        // Profile picks the smaller ProfileId.
        let mut tree = Tree::new();
        let mut profiles = ProfileMap::new();
        let root = tree.ensure_root("root", ResourceRole::User);
        let leaf = tree
            .ensure_child(root, "leaf", ResourceRole::User)
            .expect("test live parent");
        mark_dir(&mut tree, root);
        mark_dir(&mut tree, leaf);
        // Two distinct Profiles at root, distinct config_hashes via differing max_settle (makes
        // them separate Profiles).
        let p_root_a = profiles.attach(
            &mut tree,
            profile_with(root, cfg_recursive(), Duration::from_secs(6)),
        );
        let p_root_b = profiles.attach(
            &mut tree,
            profile_with(root, cfg_recursive(), Duration::from_secs(12)),
        );
        let p_leaf = profiles.attach(&mut tree, profile_at(leaf, cfg_recursive()));

        let smaller = std::cmp::min(p_root_a, p_root_b);
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_leaf),
            Some(smaller),
        );
    }

    #[test]
    fn nearest_covering_ancestor_skips_resource_ancestor_lacking_profile() {
        // root(profiled, recursive) → a(NO profile) → b(profiled). The walk from `b` skips the
        // Profile-less Resource `a` and resolves to `p_root` two Resource-hops up. (Migrated from
        // the former external `integration::covers_drives_nearest_covering_ancestor` flavor-2 —
        // `nearest_covering_ancestor` is now engine-internal, so the coverage homes inline beside
        // the function it exercises.)
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
        let p_root = profiles.attach(&mut tree, profile_at(root, cfg_recursive()));
        // `a` deliberately carries no Profile.
        let p_b = profiles.attach(&mut tree, profile_at(b, cfg_recursive()));
        assert_eq!(
            nearest_covering_ancestor(&tree, &profiles, p_b),
            Some(p_root),
            "Profile-less Resource ancestor is skipped; walk continues",
        );
        assert_eq!(nearest_covering_ancestor(&tree, &profiles, p_root), None);
    }

    // ===== has_active_standard_descendant =====
    //
    // A derived, never-cached predicate: "is some Active-Standard strict-descendant Profile still
    // on this ancestor's transitive nearest-covering-ancestor chain?" Units pin the load-bearing
    // distinctions: transitive chain ≠ subtree+direct-`covers`, strict (co-anchor excluded), Seed
    // never gates, and chain determinism.

    use specter_core::{
        ActiveBurst, BurstFinish, BurstIntent, DirtyProvenance, PreFireBurst, PreFirePhase,
        ProfileState, TimerId,
    };

    /// A Profile at `r` with `builder`'s config, left `Idle`.
    fn attach_idle(
        tree: &mut Tree,
        profiles: &mut ProfileMap,
        r: ResourceId,
        builder: ScanConfigBuilder,
    ) -> ProfileId {
        profiles.attach(tree, profile_at(r, builder.build()))
    }

    /// A Profile at `r` driven into an Active burst of `intent`. Phase is `Batching` — the lightest
    /// `Active` state (no `ProbeSlot`, so no correlation plumbing and no armed-slot Drop tripwire
    /// at test teardown). `has_active_standard_descendant` reads only `ActiveBurst::intent()`, so
    /// the phase is immaterial; the synthetic timer ids are never inspected (these Profiles are
    /// never stepped).
    fn attach_active(
        tree: &mut Tree,
        profiles: &mut ProfileMap,
        r: ResourceId,
        builder: ScanConfigBuilder,
        intent: BurstIntent,
    ) -> ProfileId {
        let mut p = profile_at(r, builder.build());
        p.transition_state(ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst::new(
                TimerId::from(1),
                PreFirePhase::Batching {
                    settle_timer: TimerId::from(2),
                },
                intent,
                DirtyProvenance::new(),
                None,
                false,
            )),
            BurstFinish::ReturnToIdle,
        ));
        profiles.attach(tree, p)
    }

    #[test]
    fn has_active_standard_descendant_follows_transitive_chain_not_subtree() {
        // a(recursive, max_depth=1) → mid(recursive, unbounded) → deep(Active-Standard). `a` does
        // NOT directly cover `deep` (depth 2 > max_depth 1), but `deep`'s nearest-covering-ancestor
        // chain is deep → p_mid → p_a, so the predicate is true *via the intermediate broader
        // Profile*. Pins: the query is the transitive chain, not the raw subtree filtered by a
        // single direct `covers(a, ·)` test.
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
            !covers(profiles.get(p_a).unwrap(), deep, &tree, &mut PathBuf::new()),
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
        // Two Profiles co-located at the ancestor's own Resource. The strict-subtree DFS starts at
        // `root`'s *children*, so a co-anchor Profile in an Active-Standard burst does NOT make the
        // predicate true for its sibling — matching the old refcount never self-counting.
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
        // A second, distinct Profile co-located at the same Resource: differing `recursive` ⇒
        // differing config_hash (two Profiles sharing a slot must differ in identity). Its config
        // is irrelevant to the assertion — the exclusion is purely structural (the DFS starts at
        // `root`'s children).
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
        // root → {c1(Idle), c2(Active-Standard), c3(Idle)}. A single witness anywhere in the strict
        // subtree suffices.
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
        // A Seed-burst descendant never gates an ancestor (the old refcount took no `+1` for Seed);
        // the same descendant in a Standard burst does. This is the Seed-vs-Standard behavioral
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
        // a(recursive) → mid → deep(Active-Standard), with two co-located covering Profiles at
        // `mid`. `chain_reaches` climbs `nearest_covering_ancestor`, whose co-anchor tie-break is
        // the smallest ProfileId — deterministically resolving `deep`'s first hop and still
        // reaching `p_a`.
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
        // Two distinct co-located Profiles at `mid`: differing `recursive` ⇒ differing config_hash
        // ⇒ separate ProfileIds. Both still cover `deep` (it is their depth-1 child, which a
        // non-recursive Profile also covers), so the chain hop is a real smallest-covering-id
        // tie-break, not a fallback to the only covering one.
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
        // Both `mid` Profiles cover `deep` (it is their depth-1 child); the tie-break picks the
        // smaller id.
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
