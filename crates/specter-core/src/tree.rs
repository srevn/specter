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

    /// Clear in-memory state at `id` (kind, refcounts) without removing the
    /// slot. Children, profiles, and role survive — those are anchors.
    /// Vacated-but-anchored slots are recreated by `ensure` returning
    /// the same `ResourceId`.
    pub fn vacate(&mut self, id: ResourceId) {
        if let Some(r) = self.nodes.get_mut(id) {
            r.kind = ResourceKind::Unknown;
            r.watch_demand = 0;
            r.suppress_count = 0;
        }
    }

    /// Remove the slot iff `Resource::has_anchors()` is `false`. Returns
    /// `true` iff the slot was removed; the caller's `ResourceId` then
    /// becomes stale (lookups return `None`).
    pub fn try_reap(&mut self, id: ResourceId) -> bool {
        let Some(r) = self.nodes.get(id) else {
            return false;
        };
        if r.has_anchors() {
            return false;
        }
        let parent = r.parent;
        let segment = r.segment;
        match parent {
            Some(p) => {
                if let Some(parent_node) = self.nodes.get_mut(p) {
                    parent_node.children.remove(&segment);
                }
            }
            None => {
                self.roots.retain(|x| *x != id);
            }
        }
        self.nodes.remove(id);
        true
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
    use crate::resource::ResourceRole;
    use proptest::prelude::*;
    use std::path::PathBuf;

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
            tree.vacate(id);
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
            let mut tree = Tree::new();
            let parent = tree.ensure(None, "p", ResourceRole::User);
            let id_old = tree.ensure(Some(parent), &s_old, ResourceRole::User);
            tree.vacate(id_old);
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

    #[test]
    fn try_reap_refused_with_anchor_role() {
        let mut tree = Tree::new();
        let id = tree.ensure(None, "watch-root", ResourceRole::WatchRootParent);
        tree.vacate(id);
        assert!(!tree.try_reap(id), "infrastructure role must not reap");
        assert!(tree.get(id).is_some());
    }

    #[test]
    fn try_reap_refused_with_children() {
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "parent", ResourceRole::User);
        let _child = tree.ensure(Some(parent), "child", ResourceRole::User);
        tree.vacate(parent);
        assert!(!tree.try_reap(parent), "parent with child must not reap");
        assert!(tree.get(parent).is_some());
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
    fn vacate_clears_kind_and_refcounts_keeps_children() {
        let mut tree = Tree::new();
        let parent = tree.ensure(None, "p", ResourceRole::User);
        let _child = tree.ensure(Some(parent), "c", ResourceRole::User);
        tree.set_kind(parent, crate::resource::ResourceKind::Dir);
        tree.get_mut(parent).unwrap().watch_demand = 3;
        tree.get_mut(parent).unwrap().suppress_count = 2;

        tree.vacate(parent);

        let r = tree.get(parent).unwrap();
        assert_eq!(r.kind, crate::resource::ResourceKind::Unknown);
        assert_eq!(r.watch_demand, 0);
        assert_eq!(r.suppress_count, 0);
        assert_eq!(r.children().len(), 1, "children survive vacate");
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
        tree.vacate(id);
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
