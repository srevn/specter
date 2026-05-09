//! `Resource` and friends.
//!
//! `Resource` lives inside `Tree`'s `SlotMap`. The structurally load-bearing
//! fields (`parent`, `segment`, `children`, `profiles`) are `pub(crate)` —
//! mutating them outside the routes that maintain the corresponding indices
//! corrupts the Tree. Read access is via the accessor methods.
//!
//! `kind` is `pub(crate)` — three external read sites historically
//! disagreed on what `Unknown` means (pattern bypass vs File-shape vs
//! not-Dir). Forcing reads through [`Resource::kind`] (returns
//! `Option<ResourceKind>`) and [`Resource::kind_or_file`] (collapses
//! Unknown to File-shape, the backend-mask convention) makes that
//! choice explicit at every call site. Writes go through
//! [`crate::Tree::set_kind`], same pattern as `Tree::set_role`.
//!
//! The remaining pure-data fields (`watch_demand`, `suppress_count`,
//! `events_union`, `role`) are `pub`; the engine writes them directly.

use crate::ids::{ProfileId, PromoterId, ResourceId};
use crate::sub::ClassSet;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use string_interner::symbol::SymbolU32;
use tinyvec::TinyVec;

#[derive(Debug)]
pub struct Resource {
    pub(crate) parent: Option<ResourceId>,
    pub(crate) segment: SymbolU32,
    pub(crate) children: BTreeMap<SymbolU32, ResourceId>,
    pub(crate) profiles: TinyVec<[(u64, ProfileId); 1]>,
    /// Promoter back-ref. Maintained in lockstep with
    /// `Promoter.proxies` by the engine's promoter-side helpers
    /// (`register_proxy` / `unregister_proxy`). Inline cap 1 covers
    /// the typical case: most Resources have zero proxies, and
    /// cross-Promoter sharing on the same slot is rare.
    ///
    /// `pub` (joining `watch_demand`, `suppress_count`,
    /// `events_union`, `role`) — the engine mutates the back-ref
    /// directly via the helpers above; the typed accessor
    /// [`Resource::proxy_promoters`] is the public read surface.
    pub proxy_promoters: SmallVec<[PromoterId; 1]>,
    /// Probed kind of this slot. `ResourceKind::Unknown` is the
    /// pre-classification placeholder — fresh slots created by
    /// `Tree::ensure`, `Tree::vacate`-reset slots, and descent
    /// scaffolds all start here. The engine writes the classified
    /// kind via [`crate::Tree::set_kind`] once a probe response or
    /// reconcile pass observes the inode. Read via [`Resource::kind`]
    /// (returns `Option<ResourceKind>`, with `Unknown` as `None`) or
    /// [`Resource::kind_or_file`] (Unknown → File, the backend-mask
    /// convention).
    pub(crate) kind: ResourceKind,
    pub watch_demand: u32,
    pub suppress_count: u32,
    /// Per-Resource OR of every covering Profile's contribution.
    /// The kqueue translator (sensor side) reads this off
    /// `WatchOp::Watch.events` to compute fflags. Maintained by the
    /// engine's refcount helpers in lockstep with `watch_demand` — added
    /// on +1, recomputed on −1 when the refcount stays non-zero, cleared
    /// on 1→0 alongside the `Unwatch` op.
    ///
    /// `pub` (not `pub(crate)`) — same visibility as `watch_demand` and
    /// `suppress_count`. The engine reads it directly via
    /// `tree.get(r).events_union`; the sensor never reads it (it sees the
    /// per-resource mask through `WatchOp::Watch.events`).
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

    /// Verification predicate for `WatchOp::Watch.kind` against the
    /// inode the watcher's open fd resolved to.
    ///
    /// Returns `true` when `self` (the engine's expected kind on the
    /// `WatchOp`) matches `observed` (the watcher's `fstat` of the
    /// freshly opened fd), with [`Self::Unknown`] acting as a
    /// wildcard. Backends use it from their fresh-watch path to reject
    /// installs where the path's on-disk kind diverges from the
    /// engine's expectation — closing the TOCTOU window between
    /// `stat(path)` and `inotify_add_watch(path)` (linux) or
    /// `open(path)` and `kevent(EV_ADD)` (kqueue).
    ///
    /// `Unknown` is the engine's sentinel for unclassified slots
    /// (descent prefix placeholder, post-`add_watch_demand` before the
    /// first probe). Treating it as a wildcard lets the watcher
    /// proceed against whatever inode resolved and cache the observed
    /// kind for downstream normalization / mask translation.
    #[must_use]
    pub const fn matches_or_unknown(self, observed: Self) -> bool {
        matches!(self, Self::Unknown) || self as u8 == observed as u8
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
            proxy_promoters: SmallVec::new(),
            kind: ResourceKind::Unknown,
            watch_demand: 0,
            suppress_count: 0,
            events_union: ClassSet::EMPTY,
            role,
        }
    }

    /// Slot retention rule: `Tree::try_reap` removes the slot iff this returns `false`.
    ///
    /// `proxy_promoters` joins `children`, `profiles`, and the two anchored
    /// roles as a retention signal: a Resource that backs a live Promoter
    /// proxy must outlive the proxy.
    #[must_use]
    pub fn has_anchors(&self) -> bool {
        !self.children.is_empty()
            || !self.profiles.is_empty()
            || !self.proxy_promoters.is_empty()
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

    /// Promoter back-references at this slot. Each entry corresponds to a
    /// live `Promoter.proxies` entry keyed by this Resource. Maintained in
    /// lockstep by the engine's `register_proxy` / `unregister_proxy`
    /// helpers (Phase 5+).
    #[must_use]
    pub fn proxy_promoters(&self) -> &[PromoterId] {
        &self.proxy_promoters
    }

    /// Probed kind of this slot. `None` means the slot has not yet been
    /// classified — descent prefix placeholder, freshly-`ensure`'d slot
    /// before the first probe response, or post-`vacate` slot whose
    /// kind was reset. Consumers must explicitly handle the unprobed
    /// case.
    ///
    /// Use [`Resource::kind_or_file`] when the call site wants the
    /// backend-mask "Unknown collapses to File" convention.
    #[must_use]
    pub const fn kind(&self) -> Option<ResourceKind> {
        match self.kind {
            ResourceKind::File => Some(ResourceKind::File),
            ResourceKind::Dir => Some(ResourceKind::Dir),
            ResourceKind::Unknown => None,
        }
    }

    /// Probed kind, with the unprobed case collapsed to
    /// [`ResourceKind::File`]. This is the backend-mask convention: the
    /// kqueue / inotify translators register file-shape bits for
    /// unclassified slots, terminal events on Unknown classify as
    /// CONTENT, etc. See [`ResourceKind::effective`] for the same
    /// semantic on a bare `ResourceKind` value.
    #[must_use]
    pub const fn kind_or_file(&self) -> ResourceKind {
        self.kind.effective()
    }

    /// Raw kind including [`ResourceKind::Unknown`]. Use only when the
    /// consumer needs to **preserve** the unprobed signal — the
    /// kqueue / inotify watcher's [`ResourceKind::matches_or_unknown`]
    /// verification expects `Unknown` as the engine's intentional
    /// wildcard, so [`crate::WatchOp::Watch`] construction sites pass
    /// it through unchanged.
    ///
    /// All other engine-side sites should prefer [`Resource::kind`]
    /// (Option, `None` for unprobed) or [`Resource::kind_or_file`]
    /// (collapses unprobed to File-shape). The accessor exists to make
    /// "I want the raw value as a wildcard" an explicit choice
    /// distinguishable from a stale-bypass bug.
    #[must_use]
    pub const fn kind_raw(&self) -> ResourceKind {
        self.kind
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
    fn anchored_when_proxy_promoters_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User);
        r.proxy_promoters.push(crate::ids::PromoterId::default());
        assert!(r.has_anchors());
        assert_eq!(r.proxy_promoters().len(), 1);
    }

    #[test]
    fn fresh_resource_has_empty_proxy_promoters() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::User);
        assert!(r.proxy_promoters().is_empty());
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

    /// `matches_or_unknown` is the watcher-side verification predicate
    /// for `WatchOp::Watch.kind`. It matches when both kinds agree OR
    /// the expected kind is `Unknown` (the engine's wildcard).
    #[test]
    fn matches_or_unknown_accepts_exact_matches() {
        assert!(ResourceKind::File.matches_or_unknown(ResourceKind::File));
        assert!(ResourceKind::Dir.matches_or_unknown(ResourceKind::Dir));
    }

    #[test]
    fn matches_or_unknown_rejects_kind_disagreement() {
        assert!(!ResourceKind::File.matches_or_unknown(ResourceKind::Dir));
        assert!(!ResourceKind::Dir.matches_or_unknown(ResourceKind::File));
    }

    #[test]
    fn matches_or_unknown_treats_unknown_expected_as_wildcard() {
        assert!(ResourceKind::Unknown.matches_or_unknown(ResourceKind::File));
        assert!(ResourceKind::Unknown.matches_or_unknown(ResourceKind::Dir));
        assert!(ResourceKind::Unknown.matches_or_unknown(ResourceKind::Unknown));
    }

    #[test]
    fn matches_or_unknown_is_one_directional_in_unknown() {
        // `expected` Unknown is a wildcard; `observed` Unknown is not —
        // the watcher's fstat must always classify to a concrete kind,
        // and a concrete-expected kind paired with an unknown-observed
        // signals a broken sensor invariant rather than a wildcard.
        assert!(!ResourceKind::File.matches_or_unknown(ResourceKind::Unknown));
        assert!(!ResourceKind::Dir.matches_or_unknown(ResourceKind::Unknown));
    }

    /// Fresh `Resource` initialises `events_union` to `EMPTY`. Refcount
    /// helpers OR contributions onto this field as covering Profiles
    /// attach.
    #[test]
    fn fresh_resource_events_union_is_empty() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::User);
        assert_eq!(r.events_union, ClassSet::EMPTY);
    }
}
