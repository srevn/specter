//! `Resource` and friends.
//!
//! `Resource` lives inside `Tree`'s `SlotMap`. The structurally load-bearing
//! fields (`parent`, `segment`, `children`, `profiles`) are `pub(crate)` —
//! mutating them outside the routes that maintain the corresponding indices
//! corrupts the Tree. Read access is via the accessor methods. The pure-data
//! fields (`kind`, `watch_demand`, `suppress_count`, `role`) are `pub`; the
//! engine writes them directly.

use crate::ids::{ProfileId, ResourceId};
use std::collections::BTreeMap;
use string_interner::symbol::SymbolU32;
use tinyvec::TinyVec;

#[derive(Debug)]
pub struct Resource {
    pub(crate) parent: Option<ResourceId>,
    pub(crate) segment: SymbolU32,
    pub(crate) children: BTreeMap<SymbolU32, ResourceId>,
    pub(crate) profiles: TinyVec<[(u64, ProfileId); 1]>,
    pub kind: ResourceKind,
    pub watch_demand: u32,
    pub suppress_count: u32,
    pub role: ResourceRole,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResourceKind {
    File,
    Dir,
    #[default]
    Unknown,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResourceRole {
    #[default]
    User,
    WatchRootParent,
    DescentScaffold,
}

impl Resource {
    pub(crate) fn new(parent: Option<ResourceId>, segment: SymbolU32, role: ResourceRole) -> Self {
        Self {
            parent,
            segment,
            children: BTreeMap::new(),
            profiles: TinyVec::new(),
            kind: ResourceKind::Unknown,
            watch_demand: 0,
            suppress_count: 0,
            role,
        }
    }

    /// Slot retention rule: `Tree::try_reap` removes the slot iff this returns `false`.
    #[must_use]
    pub fn has_anchors(&self) -> bool {
        !self.children.is_empty()
            || !self.profiles.is_empty()
            || matches!(
                self.role,
                ResourceRole::WatchRootParent | ResourceRole::DescentScaffold
            )
    }

    #[must_use]
    pub const fn parent(&self) -> Option<ResourceId> {
        self.parent
    }

    #[must_use]
    pub const fn segment(&self) -> SymbolU32 {
        self.segment
    }

    #[must_use]
    pub const fn children(&self) -> &BTreeMap<SymbolU32, ResourceId> {
        &self.children
    }

    /// `(config_hash, profile)` pairs anchoring this Resource. Mutated only
    /// by `ProfileMap::attach`/`detach`, which keep `Resource.profiles` and
    /// `ProfileMap.by_resource` in lockstep.
    #[must_use]
    pub fn profiles(&self) -> &[(u64, ProfileId)] {
        &self.profiles
    }
}

#[cfg(test)]
mod tests {
    use super::{Resource, ResourceKind, ResourceRole};
    use crate::ids::ResourceId;
    use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};

    fn dummy_segment() -> SymbolU32 {
        let mut interner = StringInterner::<StringBackend<SymbolU32>>::new();
        interner.get_or_intern("seg")
    }

    #[test]
    fn fresh_resource_has_no_anchors_when_user() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::User);
        assert!(!r.has_anchors());
    }

    #[test]
    fn fresh_resource_anchored_when_watch_root_parent() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::WatchRootParent);
        assert!(r.has_anchors());
    }

    #[test]
    fn fresh_resource_anchored_when_descent_scaffold() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::DescentScaffold);
        assert!(r.has_anchors());
    }

    #[test]
    fn anchored_when_children_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User);
        let child_seg = dummy_segment();
        r.children.insert(child_seg, ResourceId::default());
        assert!(r.has_anchors());
    }

    #[test]
    fn anchored_when_profiles_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User);
        r.profiles.push((42, crate::ids::ProfileId::default()));
        assert!(r.has_anchors());
    }

    #[test]
    fn defaults_for_kind_and_role() {
        assert_eq!(ResourceKind::default(), ResourceKind::Unknown);
        assert_eq!(ResourceRole::default(), ResourceRole::User);
    }
}
