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
//! `contributions` is `pub(crate)` — the engine's refcount helpers
//! (`add_watch` / `sub_watch`) are the sole legitimate mutators. Read
//! access for the per-Resource demand summary goes through
//! [`Resource::watch_demand`] (number of contributors) and
//! [`Resource::events_union`] (OR over contributions' masks).
//!
//! `suppress_count` and `role` are `pub`; the engine writes them directly.

use crate::ids::{ProfileId, PromoterId, ResourceId};
use crate::sub::ClassSet;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use string_interner::symbol::SymbolU32;
use tinyvec::TinyVec;

/// Identity of a single contributor to a Resource's contributions map.
///
/// Each `(Resource, ContribKey)` pair holds at most one entry — the
/// value is the contributor's `ClassSet` mask, which the per-Resource
/// union OR-folds for the kqueue / inotify fflags. The six variants
/// partition the cross-layer "who claims me" graph by owner role: a
/// Profile holds at most one claim of each kind per Resource (anchor
/// / parent / descent / descendant); a Promoter holds at most one
/// (`PrefixPending` XOR `Active`-proxy) per Resource.
///
/// Each variant carries the owner id so the contribution can be
/// removed by key without re-deriving from owner state — contribution
/// attribution is **data**, not a derivation. The engine's refcount
/// helpers ([`crate::Tree::vacate`], `add_watch` / `sub_watch`) read
/// and write the map directly; there is no walk-the-registry
/// recompute.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum ContribKey {
    /// Profile is anchored at this Resource —
    /// `Profile.anchor_claim == AnchorClaim::Held` AND
    /// `Profile.resource == resource`. Mask is `Profile.events_union`.
    ProfileAnchor(ProfileId),
    /// Profile's watch-root parent points at this Resource —
    /// `Profile.watch_root_parent == Some(resource)`. Mask is
    /// `STRUCTURE` (parent-edge watch is for anchor-reappearance
    /// detection only).
    ProfileParent(ProfileId),
    /// Profile is in `Pending` descent with `current_prefix ==
    /// resource`. Mask is `STRUCTURE` (descent prefix watch is for
    /// next-segment materialisation only).
    ProfileDescent(ProfileId),
    /// Profile holds a covered-descendant claim at this Resource
    /// (`resource != Profile.resource` AND
    /// `covers(Profile, resource, tree) == true` for a covered Dir,
    /// or under `Profile.has_per_file_fds` for a covered Leaf). Mask
    /// is `Profile.events_union`. Per-resource fan-out is
    /// 1-to-N across the snapshot but each (Resource, Profile) pair
    /// contributes at most one entry.
    ProfileDescendant(ProfileId),
    /// Promoter is in `PrefixPending` descent with `current_prefix ==
    /// resource`. Mask is `STRUCTURE`. Mutually exclusive with
    /// [`Self::PromoterProxy`] for the same Promoter.
    PromoterPrefix(PromoterId),
    /// Promoter is in `Active` state with a proxy entry at this
    /// Resource (`proxies.contains_key(&resource)`). Mask is
    /// `STRUCTURE`. Mutually exclusive with [`Self::PromoterPrefix`]
    /// for the same Promoter.
    PromoterProxy(PromoterId),
}

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
    /// Per-Resource map of contributors to the kernel-watch demand.
    /// `contributions.len()` is the refcount; `OR` over the values is
    /// the per-Resource events mask passed to the sensor on
    /// `WatchOp::Watch`. Maintained by the engine's `add_watch` /
    /// `sub_watch` helpers (see `specter-engine::refcounts`) — direct
    /// mutation outside those helpers (and [`crate::Tree::vacate`])
    /// breaks the 0↔non-empty Watch / Unwatch invariant.
    ///
    /// **Source of truth.** Coverage / Profile-state / Promoter-state
    /// are no longer walked to recompute the union; the map is
    /// directly read. Each call site that bumps or releases a
    /// contribution passes the explicit [`ContribKey`], so removal is
    /// O(log k) by key, not O(registry).
    ///
    /// `pub` joins `suppress_count` — the engine's refcount helpers
    /// mutate the field directly. Outside the engine, the read surface
    /// is [`Resource::watch_demand`] / [`Resource::events_union`] /
    /// [`Resource::contributions`].
    pub contributions: BTreeMap<ContribKey, ClassSet>,
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
    /// The inotify backend shares the same convention. The actuator's
    /// `compute_cwd` is a different concern (subprocess working
    /// directory) and does not consume Unknown: emitted Effects carry
    /// `anchor_kind ∈ { File, Dir }` by construction.
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
    /// (descent prefix placeholder, post-`add_watch` before the first
    /// probe). Treating it as a wildcard lets the watcher proceed
    /// against whatever inode resolved and cache the observed kind
    /// for downstream normalization / mask translation.
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
            contributions: BTreeMap::new(),
            suppress_count: 0,
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

    /// Number of distinct contributors holding watch-demand at this
    /// Resource. Derived from [`Self::contributions`]; `0` ⇔ the
    /// Resource is not watched.
    ///
    /// Replaces the old `watch_demand: u32` field. Callers comparing
    /// `> 0` should prefer [`Self::is_watched`] for clarity; the
    /// numeric accessor exists for tests and diagnostic logs that
    /// quote the count.
    #[must_use]
    pub fn watch_demand(&self) -> u32 {
        // Typical fan-out is single-digit; cast is safe well below
        // `u32::MAX`. Saturating cast as defence-in-depth.
        u32::try_from(self.contributions.len()).unwrap_or(u32::MAX)
    }

    /// True iff this Resource has at least one contributor, i.e., the
    /// kernel-watch refcount is `> 0`.
    #[must_use]
    pub fn is_watched(&self) -> bool {
        !self.contributions.is_empty()
    }

    /// OR-fold of every contributor's `ClassSet` mask — the
    /// per-Resource events mask the sensor sees on `WatchOp::Watch`.
    /// `ClassSet::EMPTY` when the Resource has no contributors.
    ///
    /// Replaces the old `events_union: ClassSet` cached field. The
    /// fold is O(k) over the contributions map; k is bounded by
    /// typical multi-Profile fan-out (single-digit).
    #[must_use]
    pub fn events_union(&self) -> ClassSet {
        self.contributions
            .values()
            .copied()
            .fold(ClassSet::EMPTY, |a, b| a | b)
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
    /// helpers.
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
    use super::{ClassSet, ContribKey, Resource, ResourceKind, ResourceRole};
    use crate::ids::{ProfileId, ResourceId};
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

    /// Fresh `Resource` carries an empty contributions map ⇒
    /// `events_union()` returns `EMPTY` and `watch_demand()` returns
    /// `0`. Refcount helpers insert into the map as covering Profiles
    /// / Promoters attach.
    #[test]
    fn fresh_resource_events_union_is_empty() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::User);
        assert_eq!(r.events_union(), ClassSet::EMPTY);
        assert_eq!(r.watch_demand(), 0);
        assert!(!r.is_watched());
    }

    /// `watch_demand()` counts distinct contributors; `events_union()`
    /// OR-folds their masks. Two contributors with disjoint masks
    /// produce a union containing both.
    #[test]
    fn contributions_map_yields_count_and_union() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User);
        r.contributions.insert(
            ContribKey::ProfileAnchor(ProfileId::default()),
            ClassSet::CONTENT,
        );
        // A second `ProfileAnchor` from a different Profile would
        // normally collide on the slotmap key; use a distinct
        // [`ContribKey`] variant to keep the test free of slotmap
        // setup boilerplate.
        r.contributions.insert(
            ContribKey::ProfileParent(ProfileId::default()),
            ClassSet::STRUCTURE,
        );
        assert_eq!(r.watch_demand(), 2);
        assert_eq!(r.events_union(), ClassSet::CONTENT | ClassSet::STRUCTURE);
        assert!(r.is_watched());
    }

    /// Same-key re-insert overwrites the prior mask; the count and
    /// the union reflect the freshest value.
    #[test]
    fn contributions_same_key_overwrites_mask() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User);
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        r.contributions.insert(key, ClassSet::CONTENT);
        r.contributions
            .insert(key, ClassSet::CONTENT | ClassSet::METADATA);
        assert_eq!(r.watch_demand(), 1);
        assert_eq!(r.events_union(), ClassSet::CONTENT | ClassSet::METADATA);
    }
}
