//! `Tree` — Resource container and slot semantics.
//!
//! The Tree owns one `StringInterner` for segments, a `SlotMap` of
//! `Resource`s (`ResourceId`s are generational), and a flat `roots: Vec`.
//! Identity model: `(parent, segment)` is the slot. Same `(parent, segment)`
//! always returns the same `ResourceId`. Recreation at a vacated-but-anchored
//! slot reuses the id. Reaped slots produce fresh ids on the next
//! `ensure_root` / `ensure_child`.
//!
//! Public API takes `&str` segments; the interner is internal.

use crate::ids::ResourceId;
use crate::op::WatchOp;
use crate::output::StepOutput;
use crate::resource::{Resource, ResourceKind, ResourceRole};
use compact_str::CompactString;
use slotmap::SlotMap;
use std::path::{Component, Path, PathBuf};
use string_interner::{StringInterner, backend::StringBackend, symbol::SymbolU32};

/// Synthetic segment representing the filesystem root `/`.
///
/// Every absolute attach decomposes to `[FS_ROOT_SEGMENT, ...real
/// segments]` so descents have a guaranteed-existing starting ancestor;
/// [`Tree::path_of`] reconstructs an absolute path back out because
/// `PathBuf::push("/")` resets the buffer to absolute. The constant
/// lives in [`Tree`] rather than in the engine because the path-parsing
/// invariant it anchors is Tree-shape, not engine-lifecycle.
pub const FS_ROOT_SEGMENT: &str = "/";

/// Reason an absolute-path attach request was rejected during
/// [`Tree::parse_attach_path`].
///
/// The engine maps each variant to
/// [`crate::Diagnostic::AttachPathInvalid`] with the matching
/// [`Self::hint`] string so operators see the same actionable message
/// regardless of which caller (static config, hot reload, fuzz harness)
/// produced the malformed path.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum AttachPathError {
    /// `is_absolute() == false`. The bin's `canonicalize_lenient`
    /// already filters this for static config, but hot-reload diff
    /// applies and direct test fixtures can bypass the bin's discipline.
    NotAbsolute,
    /// At least one path segment is not valid UTF-8. The Tree's segment
    /// store is `&str`-keyed; non-UTF-8 segments are unrepresentable.
    NonUtf8,
    /// A `Component::Normal` payload was the empty string. Defensive
    /// against hand-constructed `PathBuf`s — `PathBuf` itself normalises
    /// repeated separators.
    EmptyComponent,
    /// `.` or `..` component. Caller must canonicalise before attach;
    /// the Tree's `(parent, segment)` identity model has no notion of
    /// dot navigation.
    Relative,
    /// `Component::Prefix(_)` — a Windows path prefix (e.g. `C:`).
    /// Unix v1 only.
    WindowsPrefix,
}

impl AttachPathError {
    /// Static operator-facing message paired with each rejection variant.
    /// Kept stable so the engine's [`crate::Diagnostic::AttachPathInvalid`]
    /// hint matches the pre-refactor strings byte-for-byte.
    #[must_use]
    pub const fn hint(self) -> &'static str {
        match self {
            Self::NotAbsolute => "path must be absolute (engine requires fully-qualified paths)",
            Self::NonUtf8 => "non-UTF-8 path segment (engine requires UTF-8)",
            Self::EmptyComponent => "empty path segment",
            Self::Relative => "non-canonical attach path (`.`/`..`); canonicalize before attach",
            Self::WindowsPrefix => "Windows path prefix not supported on Unix v1",
        }
    }
}

/// Structural-precondition fault from [`Tree::ensure_child`] /
/// [`Tree::ensure_path`].
///
/// Production callers reach both methods with live parents and
/// non-empty inputs by construction and `.expect()` the `Result`; the
/// typed variant pins which invariant the caller is claiming.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum StaleIdError {
    /// [`Tree::ensure_child`] called with a `parent` that doesn't name
    /// a live slot (reaped, never-existed, or the slotmap null key
    /// `ResourceId::default()`).
    StaleParent(ResourceId),

    /// [`Tree::ensure_path`] called with `components.is_empty()`.
    EmptyComponents,
}

/// Validated Tree-path produced by [`Tree::parse_attach_path`].
///
/// **Type invariants** (enforced by the sole constructor):
/// - `segments()` is non-empty.
/// - `segments()[0] == FS_ROOT_SEGMENT`.
/// - Every other `segments()[i]` is a non-empty UTF-8 string containing
///   no path separators, no `.` / `..`, and no Windows prefix.
///
/// Downstream consumers (`Engine::materialize_path_or_pending`,
/// `Engine::attach_sub_inner`'s descent setup) take `&TreePath` and
/// rely on these invariants without re-checking. The opaque field
/// guarantees the only producer is the parser.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TreePath {
    segments: Vec<CompactString>,
}

impl TreePath {
    /// Validated segments. `[0] == FS_ROOT_SEGMENT`; non-empty.
    #[must_use]
    pub fn segments(&self) -> &[CompactString] {
        &self.segments
    }

    /// Segment count. Always `>= 1` by type invariant.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.segments.len()
    }

    /// Always `false` by type invariant. Method present for API
    /// completeness so clippy's `len_without_is_empty` is happy.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }
}

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

    /// Parse an absolute, UTF-8 attach path into a validated
    /// [`TreePath`]. `Component::RootDir` lowers to the synthetic
    /// [`FS_ROOT_SEGMENT`] so the engine has a single shared root for
    /// every absolute attach; [`Tree::path_of`] reconstructs an absolute
    /// path on the way back out (`PathBuf::push("/")` resets to absolute).
    ///
    /// Rejection categories (each maps to a distinct
    /// [`AttachPathError`] variant — see [`AttachPathError::hint`]):
    /// - non-absolute paths;
    /// - paths containing non-UTF-8 bytes;
    /// - relative components (`.` / `..`);
    /// - Windows path prefixes;
    /// - empty path segments (defense-in-depth against hand-constructed
    ///   `PathBuf`s — `PathBuf` itself normalises double separators).
    ///
    /// **Why on [`Tree`].** The validated invariants — non-empty,
    /// root-anchored, segment shape — are Tree-shape invariants, not
    /// engine-lifecycle invariants. The parser lives next to the type
    /// that consumes the result (`Tree::ensure_child`, `Tree::lookup`) so a
    /// future core-side path constructor can produce `TreePath`s without
    /// borrowing engine code.
    ///
    /// **Post-condition.** On `Ok`, `path.segments()` is non-empty and
    /// `path.segments()[0] == FS_ROOT_SEGMENT`; downstream callers rely
    /// on both invariants without re-checking.
    pub fn parse_attach_path(path: &Path) -> Result<TreePath, AttachPathError> {
        if !path.is_absolute() {
            return Err(AttachPathError::NotAbsolute);
        }

        // Single upfront UTF-8 check on the whole path. On Unix,
        // `Path::to_str` returns `Some` iff every byte is valid UTF-8;
        // a `Some` result means every `Component::Normal`'s byte-slice
        // is also UTF-8. The loop body's `to_str().expect(...)` is
        // sound under this precondition.
        if path.to_str().is_none() {
            return Err(AttachPathError::NonUtf8);
        }

        let mut segments: Vec<CompactString> = Vec::new();
        for c in path.components() {
            match c {
                Component::RootDir => segments.push(CompactString::const_new(FS_ROOT_SEGMENT)),
                Component::Normal(s) => {
                    let name = s.to_str().expect("path UTF-8 verified above");
                    if name.is_empty() {
                        return Err(AttachPathError::EmptyComponent);
                    }
                    segments.push(CompactString::from(name));
                }
                Component::CurDir | Component::ParentDir => {
                    return Err(AttachPathError::Relative);
                }
                Component::Prefix(_) => {
                    return Err(AttachPathError::WindowsPrefix);
                }
            }
        }

        // `is_absolute()` guarantees `Component::RootDir` was emitted,
        // which puts `FS_ROOT_SEGMENT` at `segments[0]`. The TreePath
        // type invariant rests on this; the assertion pins the contract
        // against future regressions or hand-constructed paths that
        // confuse the components iterator.
        debug_assert!(
            !segments.is_empty() && segments[0].as_str() == FS_ROOT_SEGMENT,
            "Tree::parse_attach_path post-condition: absolute path → segments[0] == FS_ROOT_SEGMENT",
        );

        Ok(TreePath { segments })
    }

    /// Walk `components` root-down, ensuring each segment. Non-leaf
    /// components default to [`ResourceRole::DescentScaffold`] on
    /// creation; the leaf takes `leaf_role`. Existing slots' roles are
    /// preserved (role-on-creation contract).
    ///
    /// Returns [`StaleIdError::EmptyComponents`] iff `components` is
    /// empty. Production callers pass [`TreePath::segments`] (non-empty
    /// by type invariant) and `.expect()` the `Result`.
    pub fn ensure_path(
        &mut self,
        components: &[&str],
        leaf_role: ResourceRole,
    ) -> Result<ResourceId, StaleIdError> {
        let (first, rest) = components
            .split_first()
            .ok_or(StaleIdError::EmptyComponents)?;
        let root_role = if rest.is_empty() {
            leaf_role
        } else {
            ResourceRole::DescentScaffold
        };
        let mut cur = self.ensure_root(first, root_role);
        let last_idx = rest.len().saturating_sub(1);
        for (i, seg) in rest.iter().enumerate() {
            let role = if i == last_idx {
                leaf_role
            } else {
                ResourceRole::DescentScaffold
            };
            cur = self
                .ensure_child(cur, seg, role)
                .expect("cur was minted by ensure_root or the previous loop iteration");
        }
        Ok(cur)
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

    /// Promote a `DescentScaffold`-roled slot to `new_role`. No-op if
    /// the slot is already `User` / `WatchRootParent` (preserves the
    /// existing role — never demote a real role to its scaffold
    /// origin) or if the slot is stale.
    ///
    /// Captures the common attach/promoter pattern:
    /// > "a slot that came from `ensure_path` as a scaffold has now
    /// > acquired a real purpose (anchor of a User Profile, parent of
    /// > one, or proxy of a Promoter) — flip its tag for diagnostic
    /// > clarity."
    ///
    /// Role is metadata: retention runs through the structural
    /// claimants on [`Resource::has_anchors`] (`children`, `profiles`,
    /// `proxy_promoters`, `contributions`), so the tag mutation is
    /// observer-only. The helper exists to keep the four-line "get +
    /// matches + set_role" idiom from drifting across call sites.
    pub fn promote_scaffold(&mut self, id: ResourceId, new_role: ResourceRole) {
        if let Some(r) = self.nodes.get_mut(id)
            && matches!(r.role, ResourceRole::DescentScaffold)
        {
            r.role = new_role;
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

    /// Get-or-create a root-level Resource. Idempotent on `segment`;
    /// `role` applies only on creation. Infallible — a root has no
    /// parent to be stale against.
    pub fn ensure_root(&mut self, segment: &str, role: ResourceRole) -> ResourceId {
        let sym = self.interner.get_or_intern(segment);
        if let Some(id) = self.find_root(sym) {
            return id;
        }
        let id = self.nodes.insert(Resource::new(None, sym, role));
        self.roots.push(id);
        id
    }

    /// Get-or-create a Resource at `(parent, segment)`. Idempotent;
    /// `role` applies only on creation. Returns
    /// [`StaleIdError::StaleParent`] iff `parent` doesn't name a live
    /// slot (reaped, never-existed, or `ResourceId::default()`).
    ///
    /// Production callers `.expect()` the `Result` with a panic
    /// message pinning whichever structural invariant keeps `parent`
    /// alive. The staleness check runs before `get_or_intern`, so a
    /// faulted call cannot grow the interner.
    pub fn ensure_child(
        &mut self,
        parent: ResourceId,
        segment: &str,
        role: ResourceRole,
    ) -> Result<ResourceId, StaleIdError> {
        if self.nodes.get(parent).is_none() {
            return Err(StaleIdError::StaleParent(parent));
        }
        let sym = self.interner.get_or_intern(segment);
        if let Some(child_id) = self.nodes[parent].children.get(&sym).copied() {
            return Ok(child_id);
        }
        let id = self.nodes.insert(Resource::new(Some(parent), sym, role));
        self.nodes[parent].children.insert(sym, id);
        Ok(id)
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
    /// back-refs) or for re-entry via [`Tree::ensure_child`] (back-refs
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
    /// - [`Tree::try_reap`] folds `vacate` into the slot lifecycle
    ///   terminus, calling it for every slot entering the cascade.
    ///   The reap precondition (`has_anchors() == false`) guarantees
    ///   `contributions` is empty here, so the `Unwatch` branch is
    ///   dormant; the `Unsuppress` branch fires for slots that still
    ///   owe burst-suppress accounting (e.g., a descendant whose
    ///   `suppress_count` was bumped `0→1` by
    ///   `event_drives_batching` and is being torn down by
    ///   reconcile's Phase 1).
    /// - The engine's kernel-watch rejection path
    ///   (`on_watch_op_rejected`) invokes `vacate` directly to
    ///   atomically tear down every contribution at the rejected slot.
    ///   Both branches are load-bearing here: the `Unwatch` closes the
    ///   kernel-watch protocol, and the `Unsuppress` closes the
    ///   burst-suppress protocol — the per-claimer cleanup loops that
    ///   follow run `sub_watch` / `sub_suppress`, which short-circuit
    ///   on the post-vacate state and rely on `vacate` to have emitted
    ///   both closing ops. This is the only standalone-clamp call
    ///   site; every other reap path flows through `try_reap`'s
    ///   folded-in vacate.
    ///
    /// Emitting both ops unconditionally (rather than asserting on
    /// preconditions) makes any future caller correct by construction:
    /// misuse degrades to "one extra closing op" — the Sensor's
    /// idempotence absorbs the duplicate — rather than to a panic or
    /// a silent kernel-watch leak.
    ///
    /// **Op emission order: `Unwatch` then `Unsuppress`.** `Unwatch`
    /// closes the kernel-watch protocol; `Unsuppress` would re-open
    /// the event window if it landed first. The relative order
    /// within a single slot survives
    /// [`StepOutput::sort_for_emission`]'s stable `ResourceId` sort,
    /// so the sensor sees the watch torn down before suppression
    /// lifts.
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
    /// recreated by [`Tree::ensure_child`] returning the same
    /// [`ResourceId`].
    pub fn vacate(&mut self, id: ResourceId, out: &mut StepOutput) {
        let Some(r) = self.nodes.get_mut(id) else {
            return;
        };
        if r.clear_contributions() > 0 {
            out.watch_ops.push(WatchOp::Unwatch { resource: id });
        }
        if r.clear_suppress() > 0 {
            out.watch_ops.push(WatchOp::Unsuppress { resource: id });
        }
        r.kind = ResourceKind::Unknown;
    }

    /// Remove the slot iff [`Resource::has_anchors`] is `false`, then
    /// cascade the same check up the parent chain. Returns `true` iff the
    /// **caller's** slot was removed (the cascade past it is best-effort
    /// hygiene); the caller's `ResourceId` becomes stale on a `true`
    /// return.
    ///
    /// **Slot lifecycle terminus.** Each cascade iteration calls
    /// [`Tree::vacate`] as the closing-emission step before unlinking
    /// and removing the slot. The slot is reapable here by definition
    /// (`has_anchors() == false`), so the contributions map is empty by
    /// invariant — `vacate`'s `Unwatch` branch is dormant. The
    /// `Unsuppress` branch fires for slots that still owe burst-suppress
    /// accounting (e.g., a descendant whose `suppress_count` was bumped
    /// `0→1` by `event_drives_batching` and is being torn down by
    /// reconcile's Phase 1). Folding `vacate` into the terminus
    /// guarantees the per-slot protocol owed at reap time is emitted
    /// before the slot leaves the Tree, regardless of caller.
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
    pub fn try_reap(&mut self, id: ResourceId, out: &mut StepOutput) -> bool {
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
            //
            // `vacate` is the closing-emission step of the slot
            // lifecycle terminus. `contributions` is empty here
            // (has_anchors's contract), so the `Unwatch` branch is
            // dormant; the `Unsuppress` branch fires when this slot
            // still owes a burst-suppress closing op.
            self.vacate(current, out);

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
    use super::{StaleIdError, Tree};
    use crate::ids::ResourceId;
    use crate::op::WatchOp;
    use crate::output::StepOutput;
    use crate::resource::ResourceRole;
    use proptest::prelude::*;
    use std::path::PathBuf;

    #[test]
    fn stale_id_error_variants_are_constructible() {
        let _ = StaleIdError::EmptyComponents;
        let _ = StaleIdError::StaleParent(ResourceId::default());
    }

    #[test]
    fn ensure_root_is_idempotent_on_segment() {
        let mut tree = Tree::new();
        let id1 = tree.ensure_root("alpha", ResourceRole::User);
        let id2 = tree.ensure_root("alpha", ResourceRole::DescentScaffold);
        assert_eq!(id1, id2);
        assert!(matches!(tree.get(id1).unwrap().role, ResourceRole::User));
        assert_eq!(tree.roots().len(), 1);
    }

    #[test]
    fn ensure_root_distinct_segments_distinct_ids() {
        let mut tree = Tree::new();
        let a = tree.ensure_root("/alpha", ResourceRole::User);
        let b = tree.ensure_root("/beta", ResourceRole::User);
        assert_ne!(a, b);
        assert_eq!(tree.roots().len(), 2);
    }

    #[test]
    fn ensure_root_after_reap_mints_fresh_slot() {
        let mut tree = Tree::new();
        let id1 = tree.ensure_root("gamma", ResourceRole::User);
        assert!(tree.try_reap(id1, &mut discard()));
        let id2 = tree.ensure_root("gamma", ResourceRole::User);
        assert_ne!(id1, id2);
    }

    #[test]
    fn ensure_child_creates_and_is_idempotent() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("/p", ResourceRole::User);
        let id1 = tree
            .ensure_child(parent, "child", ResourceRole::User)
            .expect("parent is live");
        let id2 = tree
            .ensure_child(parent, "child", ResourceRole::DescentScaffold)
            .expect("parent is live");
        assert_eq!(id1, id2);
        assert!(matches!(tree.get(id1).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn ensure_child_returns_stale_parent_for_reaped_id() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("/p", ResourceRole::User);
        assert!(tree.try_reap(parent, &mut discard()));
        let err = tree.ensure_child(parent, "child", ResourceRole::User);
        assert_eq!(err, Err(StaleIdError::StaleParent(parent)));
    }

    /// The slotmap null key collides with reaped-id semantics — surface
    /// the disjointness hazard the pre-refactor sentinel return hid.
    #[test]
    fn ensure_child_returns_stale_parent_for_default_id() {
        let mut tree = Tree::new();
        let null = ResourceId::default();
        let err = tree.ensure_child(null, "child", ResourceRole::User);
        assert_eq!(err, Err(StaleIdError::StaleParent(null)));
    }

    /// Stale-parent check must precede `get_or_intern` so a fuzz
    /// stream of novel segments at dead parents can't grow the interner.
    #[test]
    fn ensure_child_does_not_intern_segment_on_stale_parent() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("/p", ResourceRole::User);
        assert!(tree.try_reap(parent, &mut StepOutput::default()));
        let interner_before = tree.interner.len();
        let _ = tree.ensure_child(parent, "never_interned_segment", ResourceRole::User);
        assert_eq!(tree.interner.len(), interner_before);
    }

    #[test]
    fn ensure_child_returns_existing_slot_after_vacate() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("p", ResourceRole::User);
        let child = tree
            .ensure_child(parent, "c", ResourceRole::User)
            .expect("parent live");
        tree.vacate(child, &mut discard());
        let again = tree
            .ensure_child(parent, "c", ResourceRole::DescentScaffold)
            .expect("parent still live");
        assert_eq!(child, again);
        assert!(matches!(tree.get(again).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn ensure_path_returns_empty_components_for_empty_input() {
        let mut tree = Tree::new();
        let err = tree.ensure_path(&[], ResourceRole::User);
        assert_eq!(err, Err(StaleIdError::EmptyComponents));
    }

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
            let id1 = tree.ensure_root(&seg, role_a);
            let id2 = tree.ensure_root(&seg, role_b);
            prop_assert_eq!(id1, id2);
            prop_assert_eq!(tree.len(), 1);
        }

        #[test]
        fn prop_lookup_round_trip(seg in any_segment()) {
            let mut tree = Tree::new();
            prop_assert!(tree.lookup(None, &seg).is_none());
            let id = tree.ensure_root(&seg, ResourceRole::User);
            prop_assert_eq!(tree.lookup(None, &seg), Some(id));
        }

        #[test]
        fn prop_reap_invalidates(seg in any_segment()) {
            let mut tree = Tree::new();
            let id = tree.ensure_root(&seg, ResourceRole::User);
            prop_assert!(tree.try_reap(id, &mut discard()));
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
            let parent = tree.ensure_root("p", ResourceRole::User);
            let _sibling = tree.ensure_child(parent, "sibling", ResourceRole::User).expect("test live parent");
            let id_old = tree.ensure_child(parent, &s_old, ResourceRole::User).expect("test live parent");
            prop_assert!(tree.try_reap(id_old, &mut discard()));
            let id_new = tree.ensure_child(parent, &s_new, ResourceRole::User).expect("test live parent");
            prop_assert_ne!(id_old, id_new);
        }

        #[test]
        fn prop_path_of_inverse_of_walk(
            segments in proptest::collection::vec(any_segment(), 1..6),
        ) {
            let mut tree = Tree::new();
            let mut parent: Option<ResourceId> = None;
            let mut last = None;
            for seg in &segments {
                let id = match parent {
                    None => tree.ensure_root(seg, ResourceRole::User),
                    Some(p) => tree
                        .ensure_child(p, seg, ResourceRole::User)
                        .expect("test live parent"),
                };
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
        let id = tree.ensure_root("watch-root", ResourceRole::WatchRootParent);
        assert!(
            tree.try_reap(id, &mut discard()),
            "role is metadata; vacated slot with no structural anchors reaps",
        );
        assert!(tree.get(id).is_none());
    }

    #[test]
    fn try_reap_refused_with_children() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("parent", ResourceRole::User);
        let _child = tree
            .ensure_child(parent, "child", ResourceRole::User)
            .expect("test live parent");
        assert!(
            !tree.try_reap(parent, &mut discard()),
            "parent with child must not reap",
        );
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
        let leaf = tree
            .ensure_path(&["a", "b", "c"], ResourceRole::User)
            .expect("non-empty fixture");
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

        assert!(
            tree.try_reap(leaf, &mut discard()),
            "leaf reaps on the empty edge",
        );

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
        let root = tree.ensure_root("root", ResourceRole::User);
        let mid = tree
            .ensure_child(root, "mid", ResourceRole::DescentScaffold)
            .expect("test live parent");
        let a = tree
            .ensure_child(mid, "a", ResourceRole::User)
            .expect("test live parent");
        let _b = tree
            .ensure_child(mid, "b", ResourceRole::User)
            .expect("test live parent");

        assert!(
            tree.try_reap(a, &mut discard()),
            "a reaps — no anchors of its own",
        );

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
        let id = tree.ensure_root("root", ResourceRole::User);
        tree.get_mut(id).unwrap().insert_contribution(
            crate::resource::ContribKey::ProfileAnchor(crate::ids::ProfileId::default()),
            crate::sub::ClassSet::STRUCTURE,
        );
        assert!(
            !tree.try_reap(id, &mut discard()),
            "live contribution is itself a retention anchor",
        );
        assert!(tree.get(id).is_some());
    }

    #[test]
    fn ensure_at_same_slot_after_vacate_keeps_role() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("p", ResourceRole::User);
        let id_first = tree
            .ensure_child(parent, "child", ResourceRole::DescentScaffold)
            .expect("test live parent");
        // First insertion has the DescentScaffold role.
        assert_eq!(
            tree.get(id_first).unwrap().role,
            ResourceRole::DescentScaffold
        );

        // ensure again with a different role: must not change the existing role.
        let id_second = tree
            .ensure_child(parent, "child", ResourceRole::User)
            .expect("test live parent");
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
        let parent = tree.ensure_root("p", ResourceRole::User);
        let _child = tree
            .ensure_child(parent, "c", ResourceRole::User)
            .expect("test live parent");
        tree.set_kind(parent, crate::resource::ResourceKind::Dir);
        // `contributions` empty and `suppress == 0` by construction
        // (no refcount edges emitted) — vacate's precondition holds.

        tree.vacate(parent, &mut discard());

        let r = tree.get(parent).unwrap();
        assert_eq!(r.kind, crate::resource::ResourceKind::Unknown);
        assert_eq!(r.watch_demand(), 0);
        assert_eq!(r.suppress_count(), 0);
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
        let r = tree.ensure_root("x", ResourceRole::User);
        // Simulate a stranded contribution by inserting directly via
        // the typed mutator — the production path goes through
        // `engine::refcounts::add_watch`.
        tree.get_mut(r).unwrap().insert_contribution(
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
        let r = tree.ensure_root("x", ResourceRole::User);
        tree.get_mut(r).unwrap().inc_suppress();

        let mut out = StepOutput::default();
        tree.vacate(r, &mut out);

        assert_eq!(tree.get(r).unwrap().suppress_count(), 0);
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unsuppress { resource } if resource == r,
        ));
    }

    /// `Tree::try_reap` folds in `Tree::vacate` as its closing-emission
    /// step. The reap precondition (`has_anchors() == false`) guarantees
    /// `contributions` is empty here, so the `Unwatch` branch is
    /// dormant; the `Unsuppress` branch fires for slots that still owe
    /// burst-suppress accounting at reap time (e.g., a descendant whose
    /// `suppress_count` was bumped `0→1` by `event_drives_batching` and
    /// is being torn down by reconcile's Phase 1). Pin the fire path so
    /// the folding contract is regression-protected.
    #[test]
    fn try_reap_emits_unsuppress_for_drained_slot_with_residual_suppress() {
        let mut tree = Tree::new();
        let r = tree.ensure_root("x", ResourceRole::User);
        tree.get_mut(r).unwrap().inc_suppress();
        // Slot is reapable: contributions empty, no children / profiles /
        // proxies, but suppress_count > 0 ⇒ vacate's Unsuppress branch
        // fires from inside the terminus.

        let mut out = StepOutput::default();
        assert!(
            tree.try_reap(r, &mut out),
            "drained slot with residual suppress is reapable",
        );
        assert!(tree.get(r).is_none(), "slot was reaped");
        assert_eq!(out.watch_ops.len(), 1);
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unsuppress { resource } if resource == r,
        ));
    }

    /// Cascade variant of the above: a child slot reaps and orphans its
    /// parent; the parent then enters the cascade with its own pending
    /// `suppress_count > 0`. Both slots emit `Unsuppress` from inside
    /// `try_reap`'s folded-in vacate, in cascade order (leaf first,
    /// parent second).
    #[test]
    fn try_reap_cascade_emits_unsuppress_for_each_drained_slot() {
        let mut tree = Tree::new();
        let parent = tree.ensure_root("parent", ResourceRole::DescentScaffold);
        let child = tree
            .ensure_child(parent, "child", ResourceRole::User)
            .expect("test live parent");
        tree.get_mut(child).unwrap().inc_suppress();
        tree.get_mut(parent).unwrap().inc_suppress();

        let mut out = StepOutput::default();
        assert!(tree.try_reap(child, &mut out));
        assert!(tree.get(child).is_none());
        assert!(
            tree.get(parent).is_none(),
            "cascade reaped the now-orphaned parent",
        );
        assert_eq!(out.watch_ops.len(), 2, "one Unsuppress per cascaded slot");
        assert!(matches!(
            out.watch_ops[0],
            WatchOp::Unsuppress { resource } if resource == child,
        ));
        assert!(matches!(
            out.watch_ops[1],
            WatchOp::Unsuppress { resource } if resource == parent,
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
        let r = tree.ensure_root("x", ResourceRole::User);
        {
            let res = tree.get_mut(r).unwrap();
            // Two distinct contribution keys ⇒ refcount of 2.
            res.insert_contribution(
                crate::resource::ContribKey::ProfileAnchor(crate::ids::ProfileId::default()),
                crate::sub::ClassSet::STRUCTURE,
            );
            res.insert_contribution(
                crate::resource::ContribKey::ProfileParent(crate::ids::ProfileId::default()),
                crate::sub::ClassSet::STRUCTURE,
            );
            // Bump the counter to 3 — three independent suppress edges'
            // worth of accounting, simulating a slot mid-burst.
            res.inc_suppress();
            res.inc_suppress();
            res.inc_suppress();
        }

        let mut out = StepOutput::default();
        tree.vacate(r, &mut out);

        let res = tree.get(r).unwrap();
        assert_eq!(res.watch_demand(), 0);
        assert_eq!(res.suppress_count(), 0);
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
        let r0 = tree.ensure_root("root", ResourceRole::User);
        let r1 = tree
            .ensure_child(r0, "a", ResourceRole::User)
            .expect("test live parent");
        let r2 = tree
            .ensure_child(r1, "b", ResourceRole::User)
            .expect("test live parent");
        let r3 = tree
            .ensure_child(r2, "c", ResourceRole::User)
            .expect("test live parent");

        let chain: Vec<_> = tree.ancestors(r3).collect();
        assert_eq!(chain, vec![r2, r1, r0]);
    }

    #[test]
    fn path_of_handles_absolute_root_segment() {
        let mut tree = Tree::new();
        let root = tree.ensure_root("/home", ResourceRole::User);
        let user = tree
            .ensure_child(root, "user", ResourceRole::User)
            .expect("test live parent");
        let project = tree
            .ensure_child(user, "project", ResourceRole::User)
            .expect("test live parent");

        assert_eq!(
            tree.path_of(project),
            Some(PathBuf::from("/home/user/project"))
        );
    }

    #[test]
    fn path_of_returns_none_for_stale_id() {
        let mut tree = Tree::new();
        let id = tree.ensure_root("x", ResourceRole::User);
        assert!(tree.try_reap(id, &mut discard()));
        assert!(tree.path_of(id).is_none());
    }

    #[test]
    fn distinct_roots_are_independent() {
        let mut tree = Tree::new();
        let r1 = tree.ensure_root("/a", ResourceRole::User);
        let r2 = tree.ensure_root("/b", ResourceRole::User);
        assert_ne!(r1, r2);
        assert_eq!(tree.roots().len(), 2);
    }

    #[test]
    fn ensure_path_creates_intermediate_scaffolds() {
        let mut tree = Tree::new();
        let leaf = tree
            .ensure_path(&["a", "b", "c"], ResourceRole::User)
            .expect("non-empty fixture");

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
        let mut tree = Tree::new();
        let _a = tree.ensure_root("a", ResourceRole::User);
        let leaf = tree
            .ensure_path(&["a", "b"], ResourceRole::User)
            .expect("non-empty fixture");
        let a = tree.lookup(None, "a").unwrap();
        assert!(matches!(tree.get(a).unwrap().role, ResourceRole::User));
        assert!(matches!(tree.get(leaf).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn ensure_path_single_component_uses_leaf_role() {
        let mut tree = Tree::new();
        let id = tree
            .ensure_path(&["only"], ResourceRole::User)
            .expect("non-empty fixture");
        assert!(matches!(tree.get(id).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn set_role_promotes_scaffold_to_user() {
        // Scaffold materialization at descent's anchor branch.
        let mut tree = Tree::new();
        let id = tree.ensure_root("x", ResourceRole::DescentScaffold);
        tree.set_role(id, ResourceRole::User);
        assert!(matches!(tree.get(id).unwrap().role, ResourceRole::User));
    }

    #[test]
    fn set_role_on_stale_id_is_noop() {
        let mut tree = Tree::new();
        let id = tree.ensure_root("x", ResourceRole::User);
        assert!(tree.try_reap(id, &mut discard()));
        tree.set_role(id, ResourceRole::User);
        // No panic; lookups still return None.
        assert!(tree.get(id).is_none());
    }

    // ===== parse_attach_path =====
    //
    // The parser is the seam between user-supplied `PathBuf` (bin's
    // TOML loader, hot-reload diff, test fixtures) and the Tree's
    // `&str` segment world. The post-condition (`segments[0] ==
    // FS_ROOT_SEGMENT`) is load-bearing for every downstream consumer.

    use super::{AttachPathError, FS_ROOT_SEGMENT};
    use compact_str::CompactString;
    use std::path::Path;

    #[test]
    fn parse_attach_path_preserves_root_marker() {
        let p = Tree::parse_attach_path(Path::new("/tmp")).expect("absolute parses");
        assert_eq!(
            p.segments()
                .iter()
                .map(CompactString::as_str)
                .collect::<Vec<_>>(),
            vec![FS_ROOT_SEGMENT, "tmp"],
        );
    }

    #[test]
    fn parse_attach_path_deep_path_preserves_each_segment() {
        let p = Tree::parse_attach_path(Path::new("/var/log/myapp")).expect("absolute parses");
        assert_eq!(
            p.segments()
                .iter()
                .map(CompactString::as_str)
                .collect::<Vec<_>>(),
            vec![FS_ROOT_SEGMENT, "var", "log", "myapp"],
        );
    }

    #[test]
    fn parse_attach_path_root_only_path_is_single_segment() {
        let p = Tree::parse_attach_path(Path::new("/")).expect("root-only parses");
        assert_eq!(p.len(), 1);
        assert_eq!(p.segments()[0].as_str(), FS_ROOT_SEGMENT);
    }

    #[test]
    fn parse_attach_path_empty_is_not_absolute() {
        // An empty `Path` is non-absolute on Unix; the gate fires before
        // any component-level work — the diagnostic's hint is "absolute"
        // rather than "empty". `EmptyComponent` is reserved for the
        // hand-constructed paths where `Component::Normal` carries an
        // empty `OsStr`.
        assert_eq!(
            Tree::parse_attach_path(Path::new("")),
            Err(AttachPathError::NotAbsolute),
        );
    }

    #[test]
    fn parse_attach_path_relative_segments_rejected() {
        assert_eq!(
            Tree::parse_attach_path(Path::new("foo")),
            Err(AttachPathError::NotAbsolute),
        );
        assert_eq!(
            Tree::parse_attach_path(Path::new("foo/bar")),
            Err(AttachPathError::NotAbsolute),
        );
    }

    #[test]
    fn parse_attach_path_parent_dir_rejected() {
        assert_eq!(
            Tree::parse_attach_path(Path::new("/var/../log")),
            Err(AttachPathError::Relative),
        );
    }

    #[cfg(unix)]
    #[test]
    fn parse_attach_path_non_utf8_rejected() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;
        use std::path::PathBuf;

        let bad_seg = OsStr::from_bytes(&[0xFF, 0xFE]);
        let mut path = PathBuf::from("/foo");
        path.push(bad_seg);
        path.push("bar");

        assert_eq!(
            Tree::parse_attach_path(&path),
            Err(AttachPathError::NonUtf8),
        );
    }

    #[test]
    fn attach_path_error_hint_matches_pre_refactor_strings() {
        // The hint strings are operator-visible (driver logs them and the
        // engine forwards them via `Diagnostic::AttachPathInvalid.hint`).
        // Pinning the exact substrings keeps the bin's grep / dashboard
        // matchers stable across the move from engine::decompose_attach_path
        // to Tree::parse_attach_path.
        assert!(
            AttachPathError::NotAbsolute.hint().contains("absolute"),
            "NotAbsolute hint must include 'absolute'",
        );
        assert!(
            AttachPathError::NonUtf8.hint().contains("non-UTF-8"),
            "NonUtf8 hint must include 'non-UTF-8'",
        );
        assert!(
            AttachPathError::EmptyComponent.hint().contains("empty"),
            "EmptyComponent hint must include 'empty'",
        );
        assert!(
            AttachPathError::Relative.hint().contains("non-canonical"),
            "Relative hint must include 'non-canonical'",
        );
        assert!(
            AttachPathError::WindowsPrefix.hint().contains("Windows"),
            "WindowsPrefix hint must include 'Windows'",
        );
    }
}
