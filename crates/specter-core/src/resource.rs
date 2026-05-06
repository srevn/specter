//! `Resource` and friends.
//!
//! `Resource` lives inside `Tree`'s `SlotMap`. The structurally load-bearing
//! fields (`parent`, `segment`, `children`, `profiles`) are `pub(crate)` —
//! mutating them outside the routes that maintain the corresponding indices
//! corrupts the Tree. Read access is via the accessor methods. The pure-data
//! fields (`kind`, `watch_demand`, `suppress_count`, `role`) are `pub`; the
//! engine writes them directly.

use crate::ids::{ProfileId, ResourceId};
use crate::sub::ClassSet;
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
    /// Per-Resource OR of every covering Profile's contribution (R2 / D4).
    /// The kqueue translator (sensor side) reads this off `WatchOpts.events`
    /// to compute fflags. Maintained by the engine's refcount helpers in
    /// lockstep with `watch_demand` — added on +1, recomputed on −1 when
    /// the refcount stays non-zero, cleared on 1→0 alongside the
    /// `Unwatch` op.
    ///
    /// `pub` (not `pub(crate)`) — same visibility as `watch_demand` and
    /// `suppress_count`. The engine reads it directly via
    /// `tree.get(r).events_union`; the sensor never reads it (it sees the
    /// per-resource mask through `WatchOp::Watch.opts.events`).
    pub events_union: ClassSet,
    pub role: ResourceRole,
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ResourceKind {
    File,
    Dir,
    #[default]
    Unknown,
}

impl ResourceKind {
    /// "Effective" kind for backend-mask decisions: [`Self::Unknown`]
    /// collapses to [`Self::File`].
    ///
    /// Single declaration of the "treat unclassified slots as File-shape"
    /// convention shared by:
    /// - `sensor::kqueue::translate::class_set_to_fflags` (CONTENT /
    ///   METADATA branches register file bits on Unknown).
    /// - `sensor::kqueue::normalize::kevent_to_fs_event`
    ///   (NOTE_LINK / NOTE_WRITE on Unknown surface as File-shape
    ///   FsEvents).
    /// - `engine::transitions::fs_event_to_class` (terminal events on
    ///   Unknown classify as CONTENT).
    ///
    /// Inotify's analogue (when the port lands) shares it. Note that
    /// `compute_cwd` deliberately treats Unknown as Dir-shape (anchor
    /// path itself, not its parent) — that's a different concern
    /// (subprocess working directory), not a backend-mask decision, and
    /// stays out of this helper.
    #[must_use]
    pub const fn effective(self) -> Self {
        match self {
            Self::Unknown => Self::File,
            other => other,
        }
    }
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
            events_union: ClassSet::EMPTY,
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
    use super::{ClassSet, Resource, ResourceKind, ResourceRole};
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

    #[test]
    fn effective_kind_collapses_unknown_to_file() {
        assert_eq!(ResourceKind::Unknown.effective(), ResourceKind::File);
        assert_eq!(ResourceKind::File.effective(), ResourceKind::File);
        assert_eq!(ResourceKind::Dir.effective(), ResourceKind::Dir);
    }

    /// Fresh `Resource` initialises `events_union` to `EMPTY`. Refcount
    /// helpers OR contributions onto this field as covering Profiles
    /// attach (R2 / D4).
    #[test]
    fn fresh_resource_events_union_is_empty() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::User);
        assert_eq!(r.events_union, ClassSet::EMPTY);
    }
}
