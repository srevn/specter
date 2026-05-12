//! `Tree` — Resource container and slot semantics.
//!
//! The Tree owns one `StringInterner` for segments, a `SlotMap` of
//! `Resource`s (`ResourceId`s are generational), and a flat `roots: Vec`.
//! Identity model: `(parent, segment)` is the slot. Same `(parent, segment)`
//! always returns the same `ResourceId`. Recreation at a vacated-but-anchored
//! slot reuses the id. Reaped slots produce fresh ids on the next `ensure`.
//!
//! Public API takes `&str` segments; the interner is internal.

use crate::ids::ResourceId;
use crate::op::WatchOp;
use crate::output::StepOutput;
use crate::resource::{Resource, ResourceKind, ResourceRole};
use slotmap::SlotMap;
use std::path::PathBuf;
use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};

#[derive(Debug, Default)]
pub struct Tree {
    nodes: SlotMap<ResourceId, Resource>,
    roots: Vec<ResourceId>,
    interner: StringInterner<StringBackend<SymbolU32>>,
}

impl Tree {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Walk `components` from a root downward, ensuring each segment.
    /// Non-leaf components default to [`ResourceRole::DescentScaffold`]
    /// when freshly created; existing slots' roles are preserved (the
    /// `ensure` contract is "role-on-creation only"). The leaf component
    /// is created with `leaf_role`.
    ///
    /// Returns the leaf's [`ResourceId`]; returns the default `ResourceId`
    /// (a stale sentinel) if `components` is empty — debug-asserts in dev.
    /// Empty input is a caller bug; the engine's pending-path materializer
    /// guarantees non-empty by validation.
    ///
    /// Multi-component sole caller is the engine's path-based attach
    /// (`Engine::attach_sub` with `SubAttachRequest::for_path`); reconcile
    /// uses the single-component `ensure` for descendants discovered in
    /// probe responses.
    pub fn ensure_path(&mut self, components: &[&str], leaf_role: ResourceRole) -> ResourceId {
        debug_assert!(
            !components.is_empty(),
            "ensure_path: components must be non-empty",
        );
        if components.is_empty() {
            return ResourceId::default();
        }
        let last = components.len() - 1;
        let mut cur: Option<ResourceId> = None;
        for (i, comp) in components.iter().enumerate() {
            let role = if i == last {
                leaf_role
            } else {
                ResourceRole::DescentScaffold
            };
            let id = self.ensure(cur, comp, role);
            cur = Some(id);
        }
        cur.unwrap_or_default()
    }

    /// In-place role mutation. Sole legitimate use: scaffold materialization
    /// (`DescentScaffold → User`) when a pending path's anchor first
    /// appears in a probe response. Demotion (`User → DescentScaffold`)
    /// is not a defined operation; the API doesn't enforce, but discipline
    /// is single-call-site (the engine's descent module).
    pub fn set_role(&mut self, id: ResourceId, role: ResourceRole) {
        if let Some(r) = self.nodes.get_mut(id) {
            r.role = role;
        }
    }

    /// Set the probed kind on the slot. No-op for stale `id`. The engine
    /// calls this from `reconcile::create_child`, `descent::dispatch`,
    /// and the entry-validate path inside reconcile — every site that
    /// has just observed the inode and classified it. Symmetric with
    /// [`Tree::set_role`]; mirrors the
    /// `Resource.kind` field's `pub(crate)` visibility (see the
    /// rustdoc on [`crate::Resource`]).
    pub fn set_kind(&mut self, id: ResourceId, kind: ResourceKind) {
        if let Some(r) = self.nodes.get_mut(id) {
            r.kind = kind;
        }
    }

    /// Get-or-create a Resource at `(parent, segment)`. Idempotent: returns
    /// the existing slot if one is present at this `(parent, segment)`,
    /// regardless of `role`. The `role` argument applies *only* on creation.
    pub fn ensure(
        &mut self,
        parent: Option<ResourceId>,
        segment: &str,
        role: ResourceRole,
    ) -> ResourceId {
        let sym = self.interner.get_or_intern(segment);
        if let Some(p) = parent {
            if let Some(child_id) = self.nodes[p].children.get(&sym).copied() {
                return child_id;
            }
            let id = self.nodes.insert(Resource::new(Some(p), sym, role));
            self.nodes[p].children.insert(sym, id);
            id
        } else {
            if let Some(id) = self.find_root(sym) {
                return id;
            }
            let id = self.nodes.insert(Resource::new(None, sym, role));
            self.roots.push(id);
            id
        }
    }

    /// Look up a Resource at `(parent, segment)`. Returns `None` if the
    /// segment was never interned or the slot was reaped.
    #[must_use]
    pub fn lookup(&self, parent: Option<ResourceId>, segment: &str) -> Option<ResourceId> {
        let sym = self.interner.get(segment)?;
        match parent {
            Some(p) => self.nodes.get(p)?.children.get(&sym).copied(),
            None => self.find_root(sym),
        }
    }

    fn find_root(&self, sym: SymbolU32) -> Option<ResourceId> {
        self.roots
            .iter()
            .copied()
            .find(|&r| self.nodes.get(r).is_some_and(|n| n.segment == sym))
    }

    /// Finalise the slot's kernel-watch and sensor-suppress protocols,
    /// emitting any closing ops the slot still owes, and reset `kind` to
    /// `Unknown`. The slot is then ready for [`Tree::try_reap`] (no
    /// back-refs) or for re-entry via [`Tree::ensure`] (back-refs
    /// persist).
    ///
    /// `vacate` is the **protocol terminus** for the per-Resource
    /// contributions map and `suppress_count` counter: each branch
    /// acts as the symmetric closer for the matching `add_watch` /
    /// `add_suppress` 0→1 emission. Subsequent `sub_watch` /
    /// `sub_suppress` calls from co-resident bookkeeping short-circuit
    /// on the post-clear / post-zero state (absent key / counter 0).
    ///
    /// **Two production callers, two roles for the defensive branches:**
    ///
    /// - `reconcile::delete_child` invokes `vacate` only when the
    ///   contributions map is already empty (gated by `is_watched()`
    ///   at the call site). The `Unwatch` branch is dormant under this
    ///   caller; the `Unsuppress` branch fires for non-anchor
    ///   descendants whose burst-batching `add_suppress` is owed a
    ///   closing op before slot reap.
    /// - The engine's kernel-watch rejection path
    ///   (`on_watch_op_rejected`) invokes `vacate` to atomically tear
    ///   down every contribution at the rejected slot. Both branches
    ///   are load-bearing here: the `Unwatch` closes the kernel-watch
    ///   protocol, and the `Unsuppress` closes the burst-suppress
    ///   protocol — the per-claimer cleanup loops that follow run
    ///   `sub_watch` / `sub_suppress`, which short-circuit on the
    ///   post-vacate state and rely on `vacate` to have emitted both
    ///   closing ops.
    ///
    /// Emitting both ops unconditionally (rather than asserting on
    /// preconditions) makes any future caller correct by construction:
    /// misuse degrades to "one extra closing op" — the Sensor's
    /// idempotence absorbs the duplicate — rather than to a panic or
    /// a silent kernel-watch leak.
    ///
    /// **What survives.** Children, profiles, the `proxy_promoters`
    /// back-ref, `role`, `parent`, and `segment` all stay untouched.
    /// Of those, children, profiles, and `proxy_promoters` (alongside
    /// the contributions map, which `vacate` itself just cleared)
    /// drive [`Resource::has_anchors`] — i.e., they decide whether a
    /// follow-on [`Tree::try_reap`] keeps the slot alive. Role is
    /// metadata: it records *what* the slot is (User anchor /
    /// watch-root parent / descent scaffold) for diagnostic clarity,
    /// but does not anchor the slot. Vacated-but-anchored slots are
    /// recreated by [`Tree::ensure`] returning the same
    /// [`ResourceId`].
    pub fn vacate(&mut self, id: ResourceId, out: &mut StepOutput) {
        let Some(r) = self.nodes.get_mut(id) else {
            return;
        };
        if !r.contributions.is_empty() {
            out.watch_ops.push(WatchOp::Unwatch { resource: id });
            r.contributions.clear();
        }
        if r.suppress_count > 0 {
            out.watch_ops.push(WatchOp::Unsuppress { resource: id });
            r.suppress_count = 0;
        }
        r.kind = ResourceKind::Unknown;
    }

    /// Remove the slot iff [`Resource::has_anchors`] is `false`, then
    /// cascade the same check up the parent chain. Returns `true` iff the
    /// **caller's** slot was removed (the cascade past it is best-effort
    /// hygiene); the caller's `ResourceId` becomes stale on a `true`
    /// return.
    ///
    /// **Why cascade.** Reaping a slot unlinks it from its parent's
    /// `children` map. If the parent now has no anchors of its own —
    /// no remaining children, no profiles, no Promoter back-refs, no
    /// contributions — it is also orphaned and should reap. Without the
    /// cascade, every release helper that targets a leaf slot would
    /// silently leave its now-orphaned ancestor chain behind, since
    /// `try_reap` is a local op. The cascade is structurally bounded by
    /// the tree depth from `id` to its root (filesystem path depth,
    /// single-digit in practice) and gated at every step by
    /// `has_anchors`, so it never tears down a slot still claimed by
    /// some live owner.
    ///
    /// **Cascade stop conditions.** The walk halts as soon as it
    /// encounters a parent that still has anchors (the normal case — a
    /// sibling child, a co-resident Profile / Promoter, or another
    /// contribution keeps it alive) or reaches a root (parent =
    /// `None`).
    pub fn try_reap(&mut self, id: ResourceId) -> bool {
        let Some(r) = self.nodes.get(id) else {
            return false;
        };
        if r.has_anchors() {
            return false;
        }

        let mut current = id;
        loop {
            // Invariant: `nodes[current]` is live and `has_anchors() ==
            // false`. The first iteration enters here from the gate
            // above; subsequent iterations enter after the cascade
            // check below.
            let node = &self.nodes[current];
            let parent = node.parent;
            let segment = node.segment;

            // Unlink from parent's `children` map or `roots` vector
            // before removing the slot itself. Both operations are
            // cheap (BTreeMap by-key remove / Vec retain).
            match parent {
                Some(p) => {
                    if let Some(parent_node) = self.nodes.get_mut(p) {
                        parent_node.children.remove(&segment);
                    }
                }
                None => {
                    self.roots.retain(|x| *x != current);
                }
            }
            self.nodes.remove(current);

            // Advance to the parent and re-test. Stop on roots or when
            // the parent still carries an anchor.
            let Some(parent_id) = parent else {
                return true;
            };
            let Some(parent_node) = self.nodes.get(parent_id) else {
                return true;
            };
            if parent_node.has_anchors() {
                return true;
            }
            current = parent_id;
        }
    }

    #[must_use]
    pub fn parent(&self, id: ResourceId) -> Option<ResourceId> {
        self.nodes.get(id)?.parent
    }

    /// Iterator over strict ancestors (excludes `id` itself). Yields parent,
    /// grandparent, ..., until a root is reached.
    pub fn ancestors(&self, id: ResourceId) -> impl Iterator<Item = ResourceId> + '_ {
        std::iter::successors(self.parent(id), move |&p| self.parent(p))
    }

    /// Iterator over direct children of `id`. Order is the `BTreeMap`
    /// iteration order over `SymbolU32` (interner-insertion-derived) —
    /// deterministic within one Tree but not lex by segment string. Sites
    /// that need lex order resolve segment strings at the emission point.
    pub fn children_ids(&self, id: ResourceId) -> impl Iterator<Item = ResourceId> + '_ {
        self.nodes
            .get(id)
            .into_iter()
            .flat_map(|n| n.children.values().copied())
    }

    /// Resolved name (segment string) of `id`, if the slot exists.
    #[must_use]
    pub fn name(&self, id: ResourceId) -> Option<&str> {
        let sym = self.nodes.get(id)?.segment;
        self.interner.resolve(sym)
    }

    /// Path formed by joining segments from the root chain down to `id`.
    /// Returns `None` if `id` is stale or any segment fails to resolve.
    #[must_use]
    pub fn path_of(&self, id: ResourceId) -> Option<PathBuf> {
        let mut segments: Vec<&str> = Vec::new();
        let mut cur = id;
        loop {
            let r = self.nodes.get(cur)?;
            segments.push(self.interner.resolve(r.segment)?);
            match r.parent {
                Some(p) => cur = p,
                None => break,
            }
        }
        segments.reverse();
        let mut path = PathBuf::new();
        for seg in segments {
            path.push(seg);
        }
        Some(path)
    }

    #[must_use]
    pub fn get(&self, id: ResourceId) -> Option<&Resource> {
        self.nodes.get(id)
    }

    pub fn get_mut(&mut self, id: ResourceId) -> Option<&mut Resource> {
        self.nodes.get_mut(id)
    }

    #[must_use]
    pub fn roots(&self) -> &[ResourceId] {
        &self.roots
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::Tree;
    use crate::op::WatchOp;
    use crate::output::StepOutput;
    use crate::resource::ResourceRole;
    use proptest::prelude::*;
    use std::path::PathBuf;

    /// Throwaway `StepOutput` for tests that don't inspect the emitted
    /// ops. Keeping it as a tiny helper makes the in-file tests below
    /// read closer to their pre-refactor shape.
    fn discard() -> StepOutput {
        StepOutput::default()
    }

    fn any_role() -> impl Strategy<Value = ResourceRole> {
        prop_oneof![
            Just(ResourceRole::User),
            Just(ResourceRole::WatchRootParent),
            Just(ResourceRole::DescentScaffold),
        ]
    }

    fn any_segment() -> impl Strategy<Value = String> {
        "[a-zA-Z][a-zA-Z0-9_.-]{0,8}".prop_map(String::from)
    }

    proptest! {
        #[test]
        fn prop_ensure_idempotent(seg in any_segment(), role_a in any_role(), role_b in any_role()) {
            let mut tree = Tree::new();
            let id1 = tree.ensure(None, &seg, role_a);
            let id2 = tree.ensure(None, &seg, role_b);
            prop_assert_eq!(id1, id2);
            prop_assert_eq!(tree.len(), 1);
        }

        #[test]
        fn prop_lookup_round_trip(seg in any_segment()) {
            let mut tree = Tree::new();
            prop_assert!(tree.lookup(None, &seg).is_none());
            let id = tree.ensure(None, &seg, ResourceRole::User);
            prop_assert_eq!(tree.lookup(None, &seg), Some(id));
        }

        #[test]
        fn prop_reap_invalidates(seg in any_segment()) {
            let mut tree = Tree::new();
            let id = tree.ensure(None, &seg, ResourceRole::User);
            tree.vacate(id, &mut discard());
            prop_assert!(tree.try_reap(id));
            prop_assert!(tree.get(id).is_none());
            prop_assert!(tree.lookup(None, &seg).is_none());
            prop_assert!(tree.is_empty());
        }

        #[test]
        fn prop_rename_invalidates_id(
            s_old in any_segment(),
            s_new in any_segment(),
        ) {
            prop_assume!(s_old != s_new);
            prop_assume!(s_old != "sibling" && s_new != "sibling");
            let mut tree = Tree::new();
            let parent = tree.ensure(None, "p", ResourceRole::User);
            let _sibling = tree.ensure(Some(parent), "sibling", ResourceRole::User);
            let id_old = tree.ensure(Some(parent), &s_old, ResourceRole::User);
            tree.vacate(id_old, &mut discard());
            prop_assert!(tree.try_reap(id_old));
            let id_new = tree.ensure(Some(parent), &s_new, ResourceRole::User);
            prop_assert_ne!(id_old, id_new);
        }

        #[test]
        fn prop_path_of_inverse_of_walk(
            segments in proptest::collection::vec(any_segment(), 1..6),
        ) {
            let mut tree = Tree::new();
            let mut parent = None;
            let mut last = None;
            for seg in &segments {
                let id = tree.ensure(parent, seg, ResourceRole::User);
                parent = Some(id);
                last = Some(id);
            }
            let id = last.unwrap();
            let mut expected = PathBuf::new();
            for seg in &segments {
                expected.push(seg);
            }
            prop_assert_eq!(tree.path_of(id), Some(expected));
        }
    }

    /// Role is metadata: a vacated `WatchRootParent` slot with no
    /// structural anchors (children, profiles, proxy back-refs,
    /// contributions) is reapable. The previous behavior — role alone
    /// pinning the slot — leaked watch-root parent slots after every
    /// Profile reap. See `has_anchors`'s rustdoc for the contract.
    #[test]
    fn try_reap_succeeds_for_role_only_slot_post_vacate() {
        let mut tree = Tree::new();
        let id = tree.ensure(None, "watch-root", ResourceRole::WatchRootParent);
        tree.vacate(id, &mut discard());
        assert!(
            tree.try_reap(id),
            "role is metadata; vacated slot with no structural anchors reaps",
        );
        assert!(tree.get(id).is_none());
    }

    #[test]
    fn try_reap_refused_with_children() {
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "parent", ResourceRole::User);
        let _child = tree.ensure(Some(parent), "child", ResourceRole::User);
        tree.vacate(parent, &mut discard());
        assert!(!tree.try_reap(parent), "parent with child must not reap");
        assert!(tree.get(parent).is_some());
    }

    /// Reaping a leaf unlinks it from its parent's `children`, which may
    /// orphan the parent. The cascade walks up and reaps each ancestor
    /// that no longer has any anchors, stopping at the first ancestor
    /// that still does. With `ensure_path`'s `DescentScaffold`
    /// intermediates anchored only by the chain to a now-reaped leaf, the
    /// cascade frees the whole prefix on a single `try_reap` of the leaf.
    #[test]
    fn try_reap_cascades_through_role_only_ancestors() {
        let mut tree = Tree::new();
        let leaf = tree.ensure_path(&["a", "b", "c"], ResourceRole::User);
        let a = tree.lookup(None, "a").unwrap();
        let b = tree.lookup(Some(a), "b").unwrap();
        assert!(matches!(
            tree.get(a).unwrap().role,
            ResourceRole::DescentScaffold,
        ));
        assert!(matches!(
            tree.get(b).unwrap().role,
            ResourceRole::DescentScaffold,
        ));

        tree.vacate(leaf, &mut discard());
        assert!(tree.try_reap(leaf), "leaf reaps on the empty edge");

        assert!(tree.get(leaf).is_none());
        assert!(tree.get(b).is_none(), "b cascaded — only the leaf held it");
        assert!(tree.get(a).is_none(), "a cascaded — only b held it");
        assert!(tree.is_empty());
    }

    /// The cascade stops at the first ancestor that still has any
    /// anchor — here, a sibling subtree. The intermediate ancestor
    /// shared by the reaped leaf and the surviving sibling stays alive.
    #[test]
    fn try_reap_cascade_halts_at_anchored_ancestor() {
        let mut tree = Tree::new();
        let root = tree.ensure(None, "root", ResourceRole::User);
        let mid = tree.ensure(Some(root), "mid", ResourceRole::DescentScaffold);
        let a = tree.ensure(Some(mid), "a", ResourceRole::User);
        let _b = tree.ensure(Some(mid), "b", ResourceRole::User);

        tree.vacate(a, &mut discard());
        assert!(tree.try_reap(a), "a reaps — no anchors of its own");

        assert!(tree.get(a).is_none());
        assert!(
            tree.get(mid).is_some(),
            "mid still has sibling `b` as a child — cascade halts",
        );
        assert!(tree.get(root).is_some());
    }

    /// Multi-claimant retention: a slot anchored only by a co-resident
    /// contribution survives the reap of one claim. The cascade does not
    /// fire because the slot itself never becomes empty.
    #[test]
    fn try_reap_refused_with_live_contribution() {
        let mut tree = Tree::new();
        let id = tree.ensure(None, "root", ResourceRole::User);
        tree.get_mut(id).unwrap().contributions.insert(
            crate::resource::ContribKey::ProfileAnchor(crate::ids::ProfileId::default()),
            crate::sub::ClassSet::STRUCTURE,
        );
        assert!(
            !tree.try_reap(id),
            "live contribution is itself a retention anchor",
        );
        assert!(tree.get(id).is_some());
    }

    #[test]
    fn ensure_at_same_slot_after_vacate_keeps_role() {
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "p", ResourceRole::User);
        let id_first = tree.ensure(Some(parent), "child", ResourceRole::DescentScaffold);
        // First insertion has the DescentScaffold role.
        assert_eq!(
            tree.get(id_first).unwrap().role,
            ResourceRole::DescentScaffold
        );

        // ensure again with a different role: must not change the existing role.
        let id_second = tree.ensure(Some(parent), "child", ResourceRole::User);
        assert_eq!(id_first, id_second);
        assert_eq!(
            tree.get(id_first).unwrap().role,
            ResourceRole::DescentScaffold
        );
    }

    #[test]
    fn vacate_clears_kind_keeps_children_on_drained_slot() {
        // Drained slot (no contributions, suppress == 0): vacate's
        // contract is "reset `kind` to Unknown on a slot whose
        // refcounts have already been drained". Children, role, and
        // back-refs survive.
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "p", ResourceRole::User);
        let _child = tree.ensure(Some(parent), "c", ResourceRole::User);
        tree.set_kind(parent, crate::resource::ResourceKind::Dir);
        // `contributions` empty and `suppress == 0` by construction
        // (no refcount edges emitted) — vacate's precondition holds.

        tree.vacate(parent, &mut discard());

        let r = tree.get(parent).unwrap();
        assert_eq!(r.kind, crate::resource::ResourceKind::Unknown);
        assert_eq!(r.watch_demand(), 0);
        assert_eq!(r.suppress_count, 0);
        assert_eq!(r.children().len(), 1, "children survive vacate");
    }

    #[test]
    fn vacate_emits_unwatch_when_contributions_nonempty() {
        // Defensive branch: a future caller that reaches vacate
        // without first draining the contributions map would have
        // left a live FD orphaned at the sensor. The protocol-closer
        // contract emits the `Unwatch` and clears the map atomically,
        // so the misuse degrades to "one extra closing op" rather
        // than a panic / silent kernel-watch leak.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "x", ResourceRole::User);
        // Simulate a stranded contribution by inserting directly into
        // the map — the production path goes through
        // `engine::refcounts::add_watch`.
        tree.get_mut(r).unwrap().contributions.insert(
            crate::resource::ContribKey::ProfileAnchor(crate::ids::ProfileId::default()),
            crate::sub::ClassSet::STRUCTURE,
        );

        let mut out = StepOutput::default();
        tree.vacate(r, &mut out);

        assert_eq!(tree.get(r).unwrap().watch_demand(), 0);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
    }

    #[test]
    fn vacate_emits_unsuppress_when_suppress_count_nonzero() {
        // Load-bearing branch: non-anchor descendants bumped during a
        // Burst's `Batching` window have an outstanding
        // `suppress_count` when `release_descendant_claim` reaches
        // them through `delete_child` mid-anchor-loss. Vacate's
        // emission pairs the prior `Suppress` with the closing
        // `Unsuppress` before the slot reaps — keeps the sensor's
        // per-Resource suppress bookkeeping balanced.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "x", ResourceRole::User);
        tree.get_mut(r).unwrap().suppress_count = 1;

        let mut out = StepOutput::default();
        tree.vacate(r, &mut out);

        assert_eq!(tree.get(r).unwrap().suppress_count, 0);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unsuppress { resource } if resource == r,
        ));
    }

    #[test]
    fn vacate_emits_both_closing_ops_when_both_counters_nonzero() {
        // Combined branch: both protocols owed at vacate time. Order
        // is `Unwatch` before `Unsuppress`. `StepOutput::sort_for_emission`
        // ultimately re-orders by `ResourceId`; the relative order
        // within a single Resource's ops is preserved by the sort's
        // stability.
        let mut tree = Tree::new();
        let r = tree.ensure(None, "x", ResourceRole::User);
        {
            let res = tree.get_mut(r).unwrap();
            // Two distinct contribution keys ⇒ refcount of 2.
            res.contributions.insert(
                crate::resource::ContribKey::ProfileAnchor(crate::ids::ProfileId::default()),
                crate::sub::ClassSet::STRUCTURE,
            );
            res.contributions.insert(
                crate::resource::ContribKey::ProfileParent(crate::ids::ProfileId::default()),
                crate::sub::ClassSet::STRUCTURE,
            );
            res.suppress_count = 3;
        }

        let mut out = StepOutput::default();
        tree.vacate(r, &mut out);

        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand(), 0);
        assert_eq!(res.suppress_count, 0);
        assert_eq!(out.watch_ops.len(), 2);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unwatch { resource } if resource == r,
        ));
        assert!(matches!(
            out.watch_ops[1],
            WatchOp::Unsuppress { resource } if resource == r,
        ));
    }

    #[test]
    fn ancestors_walks_to_root() {
        let mut tree = Tree::new();
        let r0 = tree.ensure(None, "root", ResourceRole::User);
        let r1 = tree.ensure(Some(r0), "a", ResourceRole::User);
        let r2 = tree.ensure(Some(r1), "b", ResourceRole::User);
        let r3 = tree.ensure(Some(r2), "c", ResourceRole::User);

        let chain: Vec<_> = tree.ancestors(r3).collect();
        assert_eq!(chain, vec![r2, r1, r0]);
    }

    #[test]
    fn path_of_handles_absolute_root_segment() {
        let mut tree = Tree::new();
        let root = tree.ensure(None, "/home", ResourceRole::User);
        let user = tree.ensure(Some(root), "user", ResourceRole::User);
        let project = tree.ensure(Some(user), "project", ResourceRole::User);

        assert_eq!(
            tree.path_of(project),
            Some(PathBuf::from("/home/user/project"))
        );
    }

    #[test]
    fn path_of_returns_none_for_stale_id() {
        let mut tree = Tree::new();
        let id = tree.ensure(None, "x", ResourceRole::User);
        tree.vacate(id, &mut discard());
        assert!(tree.try_reap(id));
        assert!(tree.path_of(id).is_none());
    }

    #[test]
    fn distinct_roots_are_independent() {
        let mut tree = Tree::new();
        let r1 = tree.ensure(None, "/a", ResourceRole::User);
        let r2 = tree.ensure(None, "/b", ResourceRole::User);
        assert_ne!(r1, r2);
        assert_eq!(tree.roots().len(), 2);
    }

    #[test]
    fn ensure_path_creates_intermediate_scaffolds() {
        // Non-leaf components are DescentScaffold; leaf is User.
        let mut tree = Tree::new();
        let leaf = tree.ensure_path(&["a", "b", "c"], ResourceRole::User);

        assert_eq!(tree.name(leaf), Some("c"));
        let b = tree.parent(leaf).unwrap();
        let a = tree.parent(b).unwrap();
        assert!(tree.parent(a).is_none(), "a is a root");

        assert!(matches!(
            tree.get(a).unwrap().role,
            ResourceRole::DescentScaffold
        ));
        assert!(matches!(
            tree.get(b).unwrap().role,
            ResourceRole::DescentScaffold
        ));
        assert!(matches!(tree.get(leaf).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn ensure_path_preserves_existing_user_role() {
        // `Tree::ensure` is role-on-creation only; ensure_path inherits.
        let mut tree = Tree::new();
        let _a = tree.ensure(None, "a", ResourceRole::User);
        let leaf = tree.ensure_path(&["a", "b"], ResourceRole::User);
        let a = tree.lookup(None, "a").unwrap();
        assert!(
            matches!(tree.get(a).unwrap().role, ResourceRole::User),
            "a's role preserved"
        );
        assert!(matches!(tree.get(leaf).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn ensure_path_single_component_uses_leaf_role() {
        let mut tree = Tree::new();
        let id = tree.ensure_path(&["only"], ResourceRole::User);
        assert!(matches!(tree.get(id).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn set_role_promotes_scaffold_to_user() {
        // Scaffold materialization at descent's anchor branch.
        let mut tree = Tree::new();
        let id = tree.ensure(None, "x", ResourceRole::DescentScaffold);
        tree.set_role(id, ResourceRole::User);
        assert!(matches!(tree.get(id).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn set_role_on_stale_id_is_noop() {
        let mut tree = Tree::new();
        let id = tree.ensure(None, "x", ResourceRole::User);
        assert!(tree.try_reap(id));
        tree.set_role(id, ResourceRole::User);
        // No panic; lookups still return None.
        assert!(tree.get(id).is_none());
    }
}
