//! `Resource` and friends.
//!
//! `Resource` lives inside `Tree`'s `SlotMap`. The structurally
//! load-bearing fields (`parent`, `segment`, `children`, `profiles`)
//! are `pub(crate)` — mutating them outside the routes that maintain
//! the corresponding indices corrupts the Tree. Cross-crate read
//! access is via the accessor methods (`parent()`, `children()`,
//! `profiles()`); a slot's own segment string is read through
//! [`crate::Tree::name`] (no standalone `segment()` accessor — the
//! key type is a Tree-internal detail).
//!
//! `kind` is `pub(crate)` — three external read sites historically
//! disagreed on what `Unknown` means (pattern bypass vs File-shape vs
//! not-Dir). Forcing reads through [`Resource::kind`] (returns
//! `Option<ResourceKind>`) and [`Resource::kind_or_file`] (collapses
//! Unknown to File-shape, the backend-mask convention) makes that
//! choice explicit at every call site. Writes go through
//! [`crate::Tree::set_kind`], same pattern as `Tree::set_role`.
//!
//! `contributions` and `suppress_count` are `pub(crate)` — every
//! engine-side mutation flows through the typed methods on `Resource`
//! ([`Resource::insert_contribution`] / [`Resource::remove_contribution`]
//! / [`Resource::clear_contributions`] for the contributions map;
//! [`Resource::inc_suppress`] / [`Resource::dec_suppress`] /
//! [`Resource::clear_suppress`] for the counter). The mutators return
//! the edge information the refcount helpers need (`bool` for `0 ↔ 1`
//! edges, `usize` / `u32` for the prior count on `clear_*`), so the
//! 0↔non-empty / 0↔1 emission decisions sit at the engine helper layer
//! without leaking the underlying field shape. Read access for the
//! demand summary goes through [`Resource::contributions`],
//! [`Resource::watch_demand`], [`Resource::events_union`],
//! [`Resource::suppress_count`].
//!
//! `role` is `pub`; the engine writes it directly. Role is metadata
//! (no refcount edges), so a typed setter would buy nothing.

use crate::ids::{ProfileId, PromoterId, ResourceId};
use crate::sub::ClassSet;
use compact_str::CompactString;
use smallvec::SmallVec;
use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

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
    /// `Profile.resource == resource`. Mask is `Profile.events`.
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
    /// is `Profile.events`. Per-resource fan-out is
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
    /// This slot's own path segment — the second half of the
    /// `(parent, segment)` identity. Owned outright (`CompactString`,
    /// inline and allocation-free for the typical ≤24-byte path
    /// component) and **write-once**: set in `Resource::new`, never
    /// reparented (a rename mints a fresh slot under a new
    /// `ResourceId`; the chain is append-only). The parent's
    /// `children` map holds an equal key — the duplication is the
    /// intrusive-tree trade-off that buys O(1) self-naming
    /// ([`crate::Tree::name`]) without a reverse parent-map scan, and
    /// it is bounded by the live-slot count: the segment dies with
    /// the slot, so there is no second arena to keep in lockstep.
    pub(crate) segment: CompactString,
    /// Path from the root chain down to this slot, materialised once at
    /// construction. **Function-of-data, by construction**: the
    /// equality `path == join(root..=self segments)` holds because
    /// `(parent, segment)` are write-once — set only in
    /// [`Resource::new`] and never reparented (a rename mints a fresh
    /// slot under a new `ResourceId`; the chain is append-only). So
    /// `path` cannot drift from the chain it projects.
    ///
    /// `Arc<Path>` so every reader ([`crate::Tree::path_of`], the
    /// engine's probe / watch emission) is an `Arc::clone` refcount
    /// bump — never a parent-walk or re-allocation — and the same
    /// allocation ships read-only across the engine→sensor actor
    /// boundary.
    pub(crate) path: Arc<Path>,
    /// Direct children keyed by segment string. Each key equals the
    /// child's own `segment`. `BTreeMap` ⇒ iteration is lexicographic
    /// by segment — deterministic and local to this directory, with
    /// no dependency on global Tree attach history
    /// ([`crate::Tree::children_ids`]).
    pub(crate) children: BTreeMap<CompactString, ResourceId>,
    pub(crate) profiles: SmallVec<[(u64, ProfileId); 1]>,
    /// Promoter back-ref. Maintained in lockstep with
    /// `Promoter.proxies` by the engine's promoter-side helpers
    /// (`register_proxy` / `unregister_proxy`). Inline cap 1 covers
    /// the typical case: most Resources have zero proxies, and
    /// cross-Promoter sharing on the same slot is rare.
    pub proxy_promoters: SmallVec<[PromoterId; 1]>,
    /// Probed kind of this slot. `ResourceKind::Unknown` is the
    /// pre-classification placeholder — fresh slots created by
    /// `Tree::ensure_root` / `Tree::ensure_child`, `Tree::vacate`-reset
    /// slots, and descent scaffolds all start here. The engine writes
    /// the classified kind via [`crate::Tree::set_kind`] once a probe
    /// response or reconcile pass observes the inode. Read via
    /// [`Resource::kind`] (returns `Option<ResourceKind>`, with
    /// `Unknown` as `None`) or [`Resource::kind_or_file`]
    /// (Unknown → File, the backend-mask convention).
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
    /// `pub(crate)` — the legitimate external mutators are the typed
    /// methods on `Resource` ([`Resource::insert_contribution`],
    /// [`Resource::remove_contribution`],
    /// [`Resource::clear_contributions`]). Outside the engine, the
    /// read surface is [`Resource::contributions`] /
    /// [`Resource::watch_demand`] / [`Resource::events_union`].
    pub(crate) contributions: BTreeMap<ContribKey, ClassSet>,
    /// Suppression refcount. Event delivery is silenced iff `> 0`. The
    /// 0↔1 edges drive the sensor's `Suppress` / `Unsuppress`
    /// emissions; intermediate counts are bookkeeping only.
    ///
    /// `pub(crate)` — mutated via [`Resource::inc_suppress`] /
    /// [`Resource::dec_suppress`] / [`Resource::clear_suppress`].
    /// Read surface is [`Resource::suppress_count`].
    pub(crate) suppress_count: u32,
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
    pub(crate) fn new(
        parent: Option<ResourceId>,
        segment: CompactString,
        role: ResourceRole,
        path: Arc<Path>,
    ) -> Self {
        Self {
            parent,
            segment,
            path,
            children: BTreeMap::new(),
            profiles: SmallVec::new(),
            proxy_promoters: SmallVec::new(),
            kind: ResourceKind::Unknown,
            contributions: BTreeMap::new(),
            suppress_count: 0,
            role,
        }
    }

    /// Slot retention rule: `Tree::try_reap` removes the slot iff this
    /// returns `false`.
    ///
    /// Retention is **structural** — a slot stays alive while *something*
    /// claims it. Four canonical claimants:
    ///
    /// - `children` — a descendant slot's `parent` edge points here.
    /// - `profiles` — one or more Profiles are anchored at this slot.
    /// - `proxy_promoters` — one or more Promoter proxies are pinned here.
    /// - `contributions` — at least one [`ContribKey`] entry holds a
    ///   kernel-watch demand here (Profile anchor / parent / descent /
    ///   descendant, or Promoter prefix / proxy).
    ///
    /// [`ResourceRole`] is **metadata, not retention**. The role tag
    /// records *what* the slot is (User anchor / watch-root parent /
    /// descent scaffold) for diagnostic clarity; whether the slot
    /// *survives* is a question for the structural claimants above. The
    /// canonical retention question is "does any owner still hold this
    /// slot?", and the contributions map (in lockstep with owner state
    /// via [`crate::Tree::vacate`] and the engine's refcount helpers)
    /// answers it directly.
    ///
    /// **Why all four fields, not just contributions.** The three
    /// back-ref vectors (`children`, `profiles`, `proxy_promoters`)
    /// describe live ownership *without* implying a kernel-watch demand:
    /// a Pending Profile's User-roled leaf carries `profiles` but no
    /// contribution at the leaf (the leaf's only contribution arrives
    /// at materialization); a non-leaf descent scaffold carries
    /// `children` but no contribution of its own (its descent
    /// contribution belongs to its deepest-existing descendant). The
    /// union of all four is "anything reaches into this slot from
    /// somewhere."
    #[must_use]
    pub fn has_anchors(&self) -> bool {
        !self.children.is_empty()
            || !self.profiles.is_empty()
            || !self.proxy_promoters.is_empty()
            || !self.contributions.is_empty()
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

    /// Materialised path from the root chain to this slot. `Arc::clone`
    /// at the call site is a refcount bump — the join was paid once at
    /// construction. See the field rustdoc for the by-construction
    /// invariant.
    #[must_use]
    pub const fn path(&self) -> &Arc<Path> {
        &self.path
    }

    #[must_use]
    pub const fn children(&self) -> &BTreeMap<CompactString, ResourceId> {
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

    /// Read-only view of the per-Resource contributions map.
    ///
    /// Sole mutators are [`Self::insert_contribution`],
    /// [`Self::remove_contribution`], and [`Self::clear_contributions`].
    /// The engine's [`add_watch`] / [`sub_watch`] helpers and
    /// [`crate::Tree::vacate`]'s protocol terminus are the legitimate
    /// production callers.
    ///
    /// [`add_watch`]: ../../specter_engine/refcounts/fn.add_watch.html
    /// [`sub_watch`]: ../../specter_engine/refcounts/fn.sub_watch.html
    #[must_use]
    pub const fn contributions(&self) -> &BTreeMap<ContribKey, ClassSet> {
        &self.contributions
    }

    /// Insert or overwrite the contribution at `key` with `mask`.
    /// Returns the prior mask iff `key` was already present.
    ///
    /// Engine helpers use the return value (along with
    /// [`Self::events_union`] before/after) to detect the 0→1 existence
    /// edge and union-widening transitions that drive `WatchOp::Watch`
    /// emission. Tests may use it to assert overwrite semantics.
    pub fn insert_contribution(&mut self, key: ContribKey, mask: ClassSet) -> Option<ClassSet> {
        self.contributions.insert(key, mask)
    }

    /// Remove the contribution at `key`. Returns the prior mask iff
    /// `key` was present; `None` is the idempotent absent-key path —
    /// safe against post-`vacate` slots and slots a prior sub-walk in
    /// the same step already drained.
    pub fn remove_contribution(&mut self, key: ContribKey) -> Option<ClassSet> {
        self.contributions.remove(&key)
    }

    /// Atomically clear every contribution. Returns the prior count
    /// (`> 0` ⇒ caller should emit the closing `WatchOp::Unwatch`;
    /// `0` ⇒ no-op already drained). Used by [`crate::Tree::vacate`]'s
    /// protocol terminus.
    pub fn clear_contributions(&mut self) -> usize {
        let n = self.contributions.len();
        self.contributions.clear();
        n
    }

    /// Current suppression refcount. Event delivery is silenced iff
    /// `> 0`.
    #[must_use]
    pub const fn suppress_count(&self) -> u32 {
        self.suppress_count
    }

    /// Saturating `+1` on [`Self::suppress_count`]. Returns `true` iff
    /// this call traversed the `0 → 1` edge — i.e., suppression just
    /// activated, and the caller (engine's `add_suppress`) should emit
    /// `WatchOp::Suppress`.
    pub const fn inc_suppress(&mut self) -> bool {
        let prev = self.suppress_count;
        self.suppress_count = prev.saturating_add(1);
        prev == 0
    }

    /// `-1` on [`Self::suppress_count`], saturating at 0. Returns
    /// `true` iff this call traversed the `1 → 0` edge — i.e.,
    /// suppression just deactivated, and the caller (engine's
    /// `sub_suppress`) should emit `WatchOp::Unsuppress`. Returns
    /// `false` on the no-op path (counter was already 0; reachable
    /// post-[`crate::Tree::vacate`] when symmetric drain enters here
    /// after the terminus already emitted the closing op).
    pub const fn dec_suppress(&mut self) -> bool {
        let prev = self.suppress_count;
        if prev == 0 {
            return false;
        }
        self.suppress_count = prev - 1;
        prev == 1
    }

    /// Atomically zero the suppression refcount. Returns the prior
    /// count (`> 0` ⇒ caller should emit the closing
    /// `WatchOp::Unsuppress`; `0` ⇒ no-op already zeroed). Used by
    /// [`crate::Tree::vacate`]'s protocol terminus.
    pub const fn clear_suppress(&mut self) -> u32 {
        std::mem::replace(&mut self.suppress_count, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::{ClassSet, ContribKey, Resource, ResourceKind, ResourceRole};
    use crate::ids::{ProfileId, ResourceId};
    use compact_str::CompactString;

    fn dummy_segment() -> CompactString {
        CompactString::const_new("seg")
    }

    fn dummy_path() -> std::sync::Arc<std::path::Path> {
        std::sync::Arc::from(std::path::Path::new(""))
    }

    /// Role is metadata; a fresh slot with no children, no profiles, no
    /// proxy back-refs, and no contributions has no anchors regardless
    /// of its role tag.
    #[test]
    fn fresh_resource_has_no_anchors_regardless_of_role() {
        for role in [
            ResourceRole::User,
            ResourceRole::WatchRootParent,
            ResourceRole::DescentScaffold,
        ] {
            let r = Resource::new(None, dummy_segment(), role, dummy_path());
            assert!(
                !r.has_anchors(),
                "role-only retention was dropped; fresh {role:?} slot is not anchored",
            );
        }
    }

    /// A live contribution is itself a retention anchor — paired with
    /// the slot's owner-side bookkeeping, it is the canonical "this
    /// slot is claimed" signal that drives the kernel-watch lifetime.
    #[test]
    fn anchored_when_contribution_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        r.insert_contribution(
            ContribKey::ProfileAnchor(ProfileId::default()),
            ClassSet::STRUCTURE,
        );
        assert!(r.has_anchors());
    }

    #[test]
    fn anchored_when_children_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        let child_seg = dummy_segment();
        r.children.insert(child_seg, ResourceId::default());
        assert!(r.has_anchors());
    }

    #[test]
    fn anchored_when_profiles_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        r.profiles.push((42, crate::ids::ProfileId::default()));
        assert!(r.has_anchors());
    }

    #[test]
    fn anchored_when_proxy_promoters_present() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        r.proxy_promoters.push(crate::ids::PromoterId::default());
        assert!(r.has_anchors());
        assert_eq!(r.proxy_promoters().len(), 1);
    }

    #[test]
    fn fresh_resource_has_empty_proxy_promoters() {
        let r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
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
        let r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        assert_eq!(r.events_union(), ClassSet::EMPTY);
        assert_eq!(r.watch_demand(), 0);
        assert!(!r.is_watched());
    }

    /// `watch_demand()` counts distinct contributors; `events_union()`
    /// OR-folds their masks. Two contributors with disjoint masks
    /// produce a union containing both.
    #[test]
    fn contributions_map_yields_count_and_union() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        r.insert_contribution(
            ContribKey::ProfileAnchor(ProfileId::default()),
            ClassSet::CONTENT,
        );
        // A second `ProfileAnchor` from a different Profile would
        // normally collide on the slotmap key; use a distinct
        // [`ContribKey`] variant to keep the test free of slotmap
        // setup boilerplate.
        r.insert_contribution(
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
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        let prior = r.insert_contribution(key, ClassSet::CONTENT);
        assert!(prior.is_none(), "fresh key: no prior mask");
        let overwritten = r.insert_contribution(key, ClassSet::CONTENT | ClassSet::METADATA);
        assert_eq!(
            overwritten,
            Some(ClassSet::CONTENT),
            "re-insert returns the prior mask",
        );
        assert_eq!(r.watch_demand(), 1);
        assert_eq!(r.events_union(), ClassSet::CONTENT | ClassSet::METADATA);
    }

    /// `remove_contribution` returns the prior mask iff the key was
    /// present. The idempotent absent-key path returns `None` so
    /// callers (engine's `sub_watch`) can short-circuit silently
    /// against post-`vacate` slots.
    #[test]
    fn remove_contribution_returns_prior_mask_or_none() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        let key = ContribKey::ProfileAnchor(ProfileId::default());
        assert!(
            r.remove_contribution(key).is_none(),
            "absent key: idempotent no-op",
        );
        r.insert_contribution(key, ClassSet::CONTENT);
        assert_eq!(r.remove_contribution(key), Some(ClassSet::CONTENT));
        assert!(r.contributions().is_empty());
    }

    /// `clear_contributions` returns the prior count. The vacate
    /// terminus uses `> 0` to decide whether to emit `Unwatch`.
    #[test]
    fn clear_contributions_returns_prior_count() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        assert_eq!(r.clear_contributions(), 0, "empty: no-op");
        r.insert_contribution(
            ContribKey::ProfileAnchor(ProfileId::default()),
            ClassSet::CONTENT,
        );
        r.insert_contribution(
            ContribKey::ProfileParent(ProfileId::default()),
            ClassSet::STRUCTURE,
        );
        assert_eq!(r.clear_contributions(), 2);
        assert!(r.contributions().is_empty());
    }

    /// `inc_suppress` reports the `0 → 1` edge; intermediate bumps
    /// return `false`. This is what the engine's `add_suppress` helper
    /// uses to decide whether to emit `WatchOp::Suppress`.
    #[test]
    fn inc_suppress_reports_zero_to_one_edge_once() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        assert!(r.inc_suppress(), "0 → 1 edge");
        assert_eq!(r.suppress_count(), 1);
        assert!(!r.inc_suppress(), "1 → 2: intermediate, no edge");
        assert_eq!(r.suppress_count(), 2);
        assert!(!r.inc_suppress(), "2 → 3: intermediate, no edge");
        assert_eq!(r.suppress_count(), 3);
    }

    /// `dec_suppress` reports the `1 → 0` edge; intermediate
    /// decrements and the saturating zero-floor path return `false`.
    #[test]
    fn dec_suppress_reports_one_to_zero_edge_and_saturates_at_zero() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        assert!(!r.dec_suppress(), "no-op at 0: no edge, no underflow panic");
        assert_eq!(r.suppress_count(), 0);
        r.inc_suppress();
        r.inc_suppress();
        assert_eq!(r.suppress_count(), 2);
        assert!(!r.dec_suppress(), "2 → 1: intermediate");
        assert!(r.dec_suppress(), "1 → 0: edge");
        assert_eq!(r.suppress_count(), 0);
    }

    /// `clear_suppress` returns the prior count. Vacate uses `> 0` to
    /// decide whether to emit `Unsuppress`.
    #[test]
    fn clear_suppress_returns_prior_count() {
        let mut r = Resource::new(None, dummy_segment(), ResourceRole::User, dummy_path());
        assert_eq!(r.clear_suppress(), 0, "already zero: no-op");
        r.inc_suppress();
        r.inc_suppress();
        r.inc_suppress();
        assert_eq!(r.clear_suppress(), 3);
        assert_eq!(r.suppress_count(), 0);
    }
}
