//! Hierarchical snapshot — `DirSnapshot`, `LeafEntry`, `DirChild`,
//! `ChildEntry`, `DirMeta`, and the engine-facing `TreeSnapshot` enum
//! (re-exported at the lib root as [`crate::TreeSnapshot`]).
//!
//! ## Identity model
//!
//! - Snapshots are tree-shaped. Each `DirSnapshot` owns one directory's
//!   `lstat` triple ([`DirMeta`]), the `ScanConfig` hash they were captured
//!   under (`captured_with`), and a `BTreeMap<CompactString, ChildEntry>`
//!   of direct children (string-keyed, not interner-relative — keeps the
//!   hash cross-process stable).
//! - Children are either [`LeafEntry`] (file/symlink/other; no recursion)
//!   or [`DirChild`] (directory; carries an `Option<Arc<DirSnapshot>>`).
//!   `subtree: None` means *uncovered* — the walker stored the entry but
//!   did not recurse (excluded glob, beyond `max_depth`, or
//!   `recursive=false`).
//! - `DirSnapshot::root_resource` is *advisory*. Engine paths that need to
//!   navigate the snapshot use `&Tree` lookups; a stale id here doesn't
//!   break correctness, it only makes diagnostics less useful.
//!
//! ## Hashing
//!
//! - [`LeafEntry::leaf_hash`] folds `(kind, size, mtime, inode, device)` to
//!   a 128-bit signature, cached in `OnceLock`.
//! - [`DirSnapshot::dir_hash`] folds `captured_with`, the directory's own
//!   `root_meta` (mtime/inode/device), an entry-count length-prefix, then
//!   each `(name, ChildEntry)` pair in lex order. Per-`Dir` child carries
//!   `(inode, device, subtree.dir_hash())`; subtree=None contributes a
//!   constant `0u128`. The fold *includes* `root_meta`, so `dir_hash(d)` is
//!   a complete signature of `d`'s observable state — two-source-of-truth
//!   drift between `dir_hash` and the parent's `subtree_mtime` is impossible.
//! - 128-bit width (`siphasher::sip128`): pair-comparison space at scale
//!   (`O(levels × bursts × profiles)`) makes 64-bit collision probability
//!   uncomfortable; 128 bits is astronomically safe.
//!
//! ## Mutability and concurrency
//!
//! - `DirSnapshot` and `LeafEntry` are immutable post-construction. The
//!   only interior mutability is each one's `OnceLock<u128>` hash cache.
//! - `OnceLock<u128>` is `Sync` (and `Send`); both types are `Send + Sync`
//!   (compile-time pinned via `_SEND_SYNC` in tests).
//! - Splice and graft build *new* `Arc<DirSnapshot>`s — never
//!   mutate-through-Arc — so engine and walker can share `Arc<DirSnapshot>`
//!   handles without locks.

use crate::diff::{Diff, EntryRef, Rename};
use crate::hash::{Hasher128Ext, hash_systemtime_into, hasher_128};
use crate::ids::ResourceId;
use crate::snapshot::EntryKind;
use crate::tree::Tree;
use compact_str::CompactString;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::hash::Hash;
use std::sync::{Arc, OnceLock};
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// DirMeta
// ---------------------------------------------------------------------------

/// `lstat` triple of a directory: the load-bearing fields for the V5
/// walker's mtime-skip plus inode/device guards against the
/// delete-and-recreate-at-same-path case.
///
/// `mtime` drives the skip; `inode`/`device` defend against the case
/// where the directory was unlinked and recreated at the same path
/// between probes (the recreation gets a fresh inode under POSIX
/// semantics — same name, different identity).
#[derive(Copy, Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub struct DirMeta {
    pub mtime: SystemTime,
    pub inode: u64,
    pub device: u64,
}

// ---------------------------------------------------------------------------
// LeafEntry
// ---------------------------------------------------------------------------

/// Direct child that is *not* a directory: file, symlink, or other.
///
/// `leaf_hash` is the lazy 128-bit signature of `(kind, size, mtime,
/// inode, device)` — see [`LeafEntry::leaf_hash`]. Cached on first read;
/// `Clone` preserves the cache.
pub struct LeafEntry {
    pub kind: EntryKind,
    pub size: u64,
    pub mtime: SystemTime,
    pub inode: u64,
    pub device: u64,
    leaf_hash: OnceLock<u128>,
}

impl LeafEntry {
    #[must_use]
    pub const fn new(
        kind: EntryKind,
        size: u64,
        mtime: SystemTime,
        inode: u64,
        device: u64,
    ) -> Self {
        Self {
            kind,
            size,
            mtime,
            inode,
            device,
            leaf_hash: OnceLock::new(),
        }
    }

    /// Lazy 128-bit signature; cached for the entry's lifetime.
    #[must_use]
    pub fn leaf_hash(&self) -> u128 {
        *self.leaf_hash.get_or_init(|| compute_leaf_hash(self))
    }

    /// Pre-populate the lazy `leaf_hash` cache with `h` and return `self`.
    ///
    /// Sole intended caller is the walker, which transfers a cached hash
    /// from a prior `LeafEntry` whose identity fields are known to match
    /// (the lookup goes through [`DirSnapshot::leaf_hash_if_unchanged`]).
    /// Skips the SipHash24 fold the engine would otherwise pay on the
    /// next stability comparison (and on the parent's `dir_hash` fold,
    /// which reads each child leaf's hash transitively).
    ///
    /// **Precondition.** `h` must equal `compute_leaf_hash(self)` for
    /// `self`'s identity fields (kind, size, mtime, inode, device).
    /// Passing any other value poisons the cache permanently — every
    /// subsequent [`leaf_hash`](Self::leaf_hash) reads the stale value,
    /// breaking stability comparison and parent `dir_hash` folds.
    ///
    /// Idempotent: setting an already-populated cell discards the new
    /// value (the `Err` arm of `OnceLock::set`); call sites prepopulate
    /// the cache before any reader has had a chance to fill it via
    /// [`leaf_hash`](Self::leaf_hash).
    #[must_use]
    pub fn with_cached_hash(self, h: u128) -> Self {
        // OnceLock::set on a fresh cell never errors. The discard is
        // intentional — preserving the cached value is an optimisation,
        // not load-bearing for correctness.
        let _ = self.leaf_hash.set(h);
        self
    }
}

impl Clone for LeafEntry {
    fn clone(&self) -> Self {
        let out = Self::new(self.kind, self.size, self.mtime, self.inode, self.device);
        if let Some(&v) = self.leaf_hash.get() {
            // OnceLock::set on a fresh cell never errors. The discard is
            // intentional — preserving the cached value is an optimisation,
            // not load-bearing for correctness.
            let _ = out.leaf_hash.set(v);
        }
        out
    }
}

impl PartialEq for LeafEntry {
    fn eq(&self, other: &Self) -> bool {
        // Equality is over data, not the derived cache: two LeafEntries
        // with identical fields but only one with a populated leaf_hash
        // must compare equal.
        self.kind == other.kind
            && self.size == other.size
            && self.mtime == other.mtime
            && self.inode == other.inode
            && self.device == other.device
    }
}
impl Eq for LeafEntry {}

impl std::fmt::Debug for LeafEntry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Excludes `leaf_hash` (cached derived view); use
        // `finish_non_exhaustive` so the manual impl truthfully states
        // it doesn't enumerate every field.
        f.debug_struct("LeafEntry")
            .field("kind", &self.kind)
            .field("size", &self.size)
            .field("mtime", &self.mtime)
            .field("inode", &self.inode)
            .field("device", &self.device)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// DirChild
// ---------------------------------------------------------------------------

/// Direct child that *is* a directory. Carries inode/device for rename
/// detection and an optional `Arc<DirSnapshot>` for the recursive subtree.
///
/// `subtree: None` means *uncovered*: excluded by glob, beyond
/// `max_depth`, or `recursive=false` — three causes, indistinguishable to
/// the engine. The walker stored the entry but did not recurse; the parent's
/// [`dir_hash`] contributes `(inode, device, 0u128)` for the subtree slot.
///
/// Subtree mtime is **not** stored on `DirChild` — the canonical mtime
/// lives at `subtree.root_meta.mtime`, and the parent fold pulls it
/// transitively via the child's `dir_hash`. One indirection beats
/// two-fields-that-must-agree.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirChild {
    pub inode: u64,
    pub device: u64,
    pub subtree: Option<Arc<DirSnapshot>>,
}

// ---------------------------------------------------------------------------
// ChildEntry
// ---------------------------------------------------------------------------

/// One direct child of a `DirSnapshot`: either a leaf or a directory.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ChildEntry {
    Leaf(LeafEntry),
    Dir(DirChild),
}

impl ChildEntry {
    /// Inode of the underlying entry — same accessor for Leaf and Dir.
    /// Used by the engine reconciler for inode-stable Dir pairs and by
    /// `diff_tree`'s rename pairing.
    #[must_use]
    pub const fn inode(&self) -> u64 {
        match self {
            Self::Leaf(l) => l.inode,
            Self::Dir(d) => d.inode,
        }
    }

    #[must_use]
    pub const fn device(&self) -> u64 {
        match self {
            Self::Leaf(l) => l.device,
            Self::Dir(d) => d.device,
        }
    }

    /// `EntryKind` projection for downstream `Diff` emission. Dir always
    /// projects to `EntryKind::Dir`; Leaf preserves its tag.
    #[must_use]
    pub const fn kind(&self) -> EntryKind {
        match self {
            Self::Leaf(l) => l.kind,
            Self::Dir(_) => EntryKind::Dir,
        }
    }
}

// ---------------------------------------------------------------------------
// DirSnapshot
// ---------------------------------------------------------------------------

/// One directory's snapshot. Recursive via `ChildEntry::Dir`'s
/// `Option<Arc<DirSnapshot>>`. The `Arc` discipline lets splice and the
/// walker's mtime-skip share subtrees across snapshots without copying.
///
/// All public fields except `dir_hash` are immutable post-construction.
/// `dir_hash: OnceLock<u128>` is the only interior mutability. Both
/// `OnceLock` and `u128` are `Send + Sync`, so `DirSnapshot` is too
/// (compile-time pinned in tests).
pub struct DirSnapshot {
    pub root_resource: ResourceId,
    pub root_meta: DirMeta,
    pub captured_with: u64,
    pub entries: BTreeMap<CompactString, ChildEntry>,
    dir_hash: OnceLock<u128>,
}

impl DirSnapshot {
    /// Sole constructor. Takes already-built entries; doesn't sort
    /// (`BTreeMap` is sorted-by-key by construction). The hash cache
    /// starts empty and fills on first read.
    #[must_use]
    pub const fn new(
        root_resource: ResourceId,
        root_meta: DirMeta,
        captured_with: u64,
        entries: BTreeMap<CompactString, ChildEntry>,
    ) -> Self {
        Self {
            root_resource,
            root_meta,
            captured_with,
            entries,
            dir_hash: OnceLock::new(),
        }
    }

    /// Lazy 128-bit signature of `(captured_with, root_meta, entries)`.
    /// The cache is process-local and never invalidated; the snapshot is
    /// immutable post-construction.
    #[must_use]
    pub fn dir_hash(&self) -> u128 {
        *self.dir_hash.get_or_init(|| compute_dir_hash(self))
    }

    /// Look up `name` and return the prior leaf's cached `leaf_hash` iff
    /// the prior entry is a `Leaf`, its identity fields match `fresh`,
    /// and its hash cache is already populated.
    ///
    /// Used by the walker to transfer cache across re-enumeration: when
    /// a parent directory's mtime bumps but a child leaf is observably
    /// unchanged, the prior leaf's hash is reusable for the freshly-
    /// `lstat`ed leaf. The result composes with
    /// [`LeafEntry::with_cached_hash`].
    ///
    /// Returns `None` for: missing entry, kind mismatch (`Dir` at this
    /// name), any identity-field difference (the leaf changed), or a
    /// prior leaf whose hash was never computed (rare — most baselines
    /// have had `dir_hash()` called on the parent, which forces every
    /// child leaf hash; the no-cache case falls back to lazy computation
    /// in the engine, identical to the pre-optimization behaviour).
    ///
    /// Identity equality goes through [`LeafEntry`]'s `PartialEq`, which
    /// deliberately excludes the cache.
    #[must_use]
    pub fn leaf_hash_if_unchanged(&self, name: &str, fresh: &LeafEntry) -> Option<u128> {
        let ChildEntry::Leaf(prior) = self.entries.get(name)? else {
            return None;
        };
        if prior == fresh {
            prior.leaf_hash.get().copied()
        } else {
            None
        }
    }
}

impl Clone for DirSnapshot {
    fn clone(&self) -> Self {
        let out = Self::new(
            self.root_resource,
            self.root_meta,
            self.captured_with,
            self.entries.clone(),
        );
        if let Some(&v) = self.dir_hash.get() {
            let _ = out.dir_hash.set(v);
        }
        out
    }
}

impl PartialEq for DirSnapshot {
    fn eq(&self, other: &Self) -> bool {
        // Excludes the `dir_hash` cache (derived view).
        self.root_resource == other.root_resource
            && self.root_meta == other.root_meta
            && self.captured_with == other.captured_with
            && self.entries == other.entries
    }
}
impl Eq for DirSnapshot {}

impl std::fmt::Debug for DirSnapshot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Excludes `dir_hash` (cached derived view); `finish_non_exhaustive`
        // states the omission honestly.
        f.debug_struct("DirSnapshot")
            .field("root_resource", &self.root_resource)
            .field("root_meta", &self.root_meta)
            .field("captured_with", &self.captured_with)
            .field("entries", &self.entries)
            .finish_non_exhaustive()
    }
}

// ---------------------------------------------------------------------------
// TreeSnapshot (engine-facing top-level)
// ---------------------------------------------------------------------------

/// Engine-facing snapshot. File-anchored Profiles carry one [`LeafEntry`];
/// Dir-anchored Profiles carry an `Arc<DirSnapshot>` (the recursive tree).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TreeSnapshot {
    File(LeafEntry),
    Dir(Arc<DirSnapshot>),
}

impl TreeSnapshot {
    /// Stability verdict. One `dir_hash` (or `leaf_hash`) comparison; O(1)
    /// after the cache is filled.
    ///
    /// Kind mismatch (File vs Dir) is never stable — kind changes route
    /// through `Vanished` at the probe layer; this arm is defence-in-depth.
    #[must_use]
    pub fn stable_against(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::File(a), Self::File(b)) => a.leaf_hash() == b.leaf_hash(),
            (Self::Dir(a), Self::Dir(b)) => a.dir_hash() == b.dir_hash(),
            _ => false,
        }
    }

    /// Walk this snapshot down to the directory at `target`, following the
    /// segment chain `tree.parent(target) → ... → root_resource`. Returns
    /// `None` for any of:
    ///
    /// - `TreeSnapshot::File` (no recursion possible).
    /// - `target` outside the snapshot's anchor subtree (the parent walk
    ///   bottoms out before reaching `root_resource`).
    /// - The chain crosses a `Leaf` or an uncovered `Dir`
    ///   (`subtree: None`).
    /// - Any segment fails to resolve via `tree.name` (slot reaped).
    ///
    /// `DirSnapshot::root_resource` is advisory; navigation uses `&Tree`,
    /// not snapshot identity.
    #[must_use]
    pub fn subtree_at(&self, target: ResourceId, tree: &Tree) -> Option<Arc<DirSnapshot>> {
        subtree_at_impl(self, target, tree)
    }
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

/// `1u8` for Leaf, `2u8` for Dir. Defends against the (theoretical)
/// collision where a name + leaf-fields hash equals a name + dir-fields
/// hash; one byte per entry.
const LEAF_TAG: u8 = 1;
const DIR_TAG: u8 = 2;

/// Constant `subtree_hash` contribution for an uncovered branch
/// (`subtree: None`). The engine has no observation about uncovered
/// subtrees, so the contribution is constant. A config reload that
/// changes the rule produces a different `Profile` (different
/// `config_hash`), which seeds fresh.
const UNCOVERED_SUBTREE_HASH: u128 = 0;

fn compute_leaf_hash(l: &LeafEntry) -> u128 {
    let mut h = hasher_128();
    (l.kind as u8).hash(&mut h);
    l.size.hash(&mut h);
    hash_systemtime_into(l.mtime, &mut h);
    l.inode.hash(&mut h);
    l.device.hash(&mut h);
    h.finish_128_u128()
}

fn compute_dir_hash(d: &DirSnapshot) -> u128 {
    let mut h = hasher_128();

    // Header: ScanConfig hash + the directory's own lstat triple.
    // root_meta is folded HERE (not on the parent's contribution), so
    // dir_hash(d) is a complete signature of d's observable state.
    d.captured_with.hash(&mut h);
    hash_systemtime_into(d.root_meta.mtime, &mut h);
    d.root_meta.inode.hash(&mut h);
    d.root_meta.device.hash(&mut h);

    // Length prefix: belt-and-suspenders alongside SipHash24's
    // prefix-freeness. Keeps the golden test legible.
    (d.entries.len() as u64).hash(&mut h);

    // Sequential lex-order fold (BTreeMap iterates in lex order). XOR
    // was rejected: sequential preserves ordering information and avoids
    // commutative-fold subtleties at no real cost (entries are already
    // sorted by construction).
    for (name, child) in &d.entries {
        name.as_str().hash(&mut h);
        match child {
            ChildEntry::Leaf(l) => {
                LEAF_TAG.hash(&mut h);
                l.leaf_hash().hash(&mut h);
            }
            ChildEntry::Dir(c) => {
                // `(inode, device)` are folded twice on the `Some(subtree)`
                // path: once via the DirChild and once transitively through
                // the subtree's `root_meta` inside `dir_hash`. The values
                // agree by walker construction (both lstats target the same
                // dirent), so the duplication is harmless. The asymmetry is
                // necessary: when `subtree = None` (uncovered), only the
                // DirChild fold contributes inode/device, since
                // `UNCOVERED_SUBTREE_HASH` is a constant. Folding the
                // DirChild's inode/device unconditionally keeps the
                // covered/uncovered cases distinguishable.
                DIR_TAG.hash(&mut h);
                c.inode.hash(&mut h);
                c.device.hash(&mut h);
                let sub: u128 = c
                    .subtree
                    .as_ref()
                    .map_or(UNCOVERED_SUBTREE_HASH, |s| s.dir_hash());
                sub.hash(&mut h);
            }
        }
    }

    h.finish_128_u128()
}

// ---------------------------------------------------------------------------
// subtree_at
// ---------------------------------------------------------------------------

fn subtree_at_impl(
    snap: &TreeSnapshot,
    target: ResourceId,
    tree: &Tree,
) -> Option<Arc<DirSnapshot>> {
    let TreeSnapshot::Dir(root) = snap else {
        return None;
    };
    let chain = ancestor_chain(target, root.root_resource, tree)?;

    // Descend from `root` by following segment names. `chain[0] == anchor`
    // matches `root` already, so we start at `chain[1]`.
    let mut current: Arc<DirSnapshot> = Arc::clone(root);
    for &id in chain.iter().skip(1) {
        let name = tree.name(id)?;
        let next = match current.entries.get(name)? {
            ChildEntry::Dir(dc) => dc.subtree.as_ref()?,
            ChildEntry::Leaf(_) => return None,
        };
        current = Arc::clone(next);
    }
    Some(current)
}

/// Walk `tree.parent` from `target` up to `anchor` and return the
/// inclusive chain `[anchor, mid_1, ..., target]`. Returns `None` when
/// `target` is not in `anchor`'s subtree (the parent walk bottoms out
/// before reaching `anchor`).
///
/// Sole helper for navigation that needs to follow the path from an
/// anchor down to one of its descendants — `subtree_at_impl` consumes
/// it as the descent guide for snapshot navigation; `splice` consumes
/// it to know which intermediate `DirSnapshot`s need rebuilding.
fn ancestor_chain(
    target: ResourceId,
    anchor: ResourceId,
    tree: &Tree,
) -> Option<SmallVec<[ResourceId; 8]>> {
    let mut chain: SmallVec<[ResourceId; 8]> = SmallVec::new();
    let mut cur = target;
    loop {
        chain.push(cur);
        if cur == anchor {
            chain.reverse();
            return Some(chain);
        }
        cur = tree.parent(cur)?;
    }
}

// ---------------------------------------------------------------------------
// splice
// ---------------------------------------------------------------------------

/// Outcome of [`splice`].
///
/// The carried [`TreeSnapshot`] is always the view the caller should adopt
/// as the new current — the variants only differentiate whether the splice
/// path encountered a contract violation the caller should surface as a
/// diagnostic.
#[derive(Debug)]
pub enum SpliceResult {
    /// Splice succeeded. The new view integrates `replacement` at `target`
    /// (or is the trivial wholesale-replace when prior was `None` /
    /// `File(_)` / target-equals-anchor).
    Spliced(TreeSnapshot),
    /// Splice could not navigate from the prior anchor down to `target`
    /// (target outside anchor's tree subtree, or path crossed a
    /// `subtree: None` intermediate). The carried snapshot is the prior
    /// unchanged — `replacement` was not integrated. Caller emits
    /// [`crate::Diagnostic::SpliceCrossedUncovered`] so the contract
    /// violation is visible in operator logs.
    ///
    /// Engine contract: "graft only into observed subtrees". Reaching
    /// this variant in v1 implies a state-machine bug; the variant
    /// exists to surface it without crashing the engine.
    CrossedUncovered(TreeSnapshot),
}

impl SpliceResult {
    /// Consume the result and return its carried [`TreeSnapshot`].
    /// Equivalent for the caller in both variants — the variant tag is
    /// the only difference, and is consulted before this call to decide
    /// whether to emit a Diagnostic.
    #[must_use]
    pub fn into_snapshot(self) -> TreeSnapshot {
        match self {
            Self::Spliced(s) | Self::CrossedUncovered(s) => s,
        }
    }
}

/// Tree-zipper splice that replaces the subtree at `target`.
///
/// Produces a new [`TreeSnapshot`] whose subtree at `target` equals
/// `replacement`, sharing all off-path subtrees with `prior` via `Arc`.
/// Rebuilds at most `depth(target)` `DirSnapshot`s along the
/// path-to-anchor.
///
/// Returns [`SpliceResult::Spliced`] with `TreeSnapshot::Dir(replacement)`
/// (Arc-cheap) for the trivial cases:
/// - `prior == None` (first graft).
/// - `prior == Some(File(_))` (kind change at the anchor).
/// - `target == prior.root_resource` and the hashes differ (new root).
///
/// Returns [`SpliceResult::Spliced`] with `TreeSnapshot::Dir(prior)`
/// (no allocation) when:
/// - `target == prior.root_resource` and `dir_hash` matches (G7-trivial).
/// - The recursive splice short-circuited at every level via `Arc::ptr_eq`
///   or `dir_hash` equality (G7 propagation).
///
/// Returns [`SpliceResult::CrossedUncovered`] carrying the prior unchanged
/// when the engine's "graft only into observed subtrees" contract is
/// violated:
/// - `target` is outside the anchor's tree subtree (parent walk bottoms
///   out before reaching `anchor`).
/// - The path from anchor to target crosses a `subtree: None` intermediate
///   (snapshot coverage gap), a missing entry, or a slot reaped mid-graft.
///
/// Structurally unreachable in v1. The CrossedUncovered fallback preserves
/// the prior view (no integration); the caller emits a Diagnostic so the
/// contract breach is observable.
#[must_use]
pub fn splice(
    prior: Option<TreeSnapshot>,
    target: ResourceId,
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> SpliceResult {
    let Some(TreeSnapshot::Dir(root)) = prior else {
        // None or File(_) prior: replace wholesale.
        return SpliceResult::Spliced(TreeSnapshot::Dir(replacement));
    };
    let anchor = root.root_resource;
    if target == anchor {
        if root.dir_hash() == replacement.dir_hash() {
            return SpliceResult::Spliced(TreeSnapshot::Dir(root));
        }
        return SpliceResult::Spliced(TreeSnapshot::Dir(replacement));
    }

    let Some(chain) = ancestor_chain(target, anchor, tree) else {
        // Target outside anchor's tree subtree. Keep prior unchanged;
        // surface the contract violation. Behaviour change vs. pre-PR:
        // the prior `wholesale-replace with replacement` left
        // `Profile.current` rooted at `target` (not anchor), violating
        // the snapshot navigation invariants. Keeping prior preserves
        // the invariant and lets the next observation converge.
        return SpliceResult::CrossedUncovered(TreeSnapshot::Dir(root));
    };

    // chain is [anchor, mid_1, ..., mid_k, target]; we already consumed
    // the anchor as `root`, so descend with chain[1..].
    match splice_dir(&root, &chain[1..], replacement, tree) {
        Some(new_root) => {
            if Arc::ptr_eq(&new_root, &root) {
                SpliceResult::Spliced(TreeSnapshot::Dir(root))
            } else {
                SpliceResult::Spliced(TreeSnapshot::Dir(new_root))
            }
        }
        None => SpliceResult::CrossedUncovered(TreeSnapshot::Dir(root)),
    }
}

/// Recursive splice helper. Returns `Some(arc)` on a successful per-level
/// rebuild (or G7 short-circuit); returns `None` when navigation can't
/// proceed (slot reaped mid-graft, snapshot coverage gap, or missing
/// entry). The top-level [`splice`] translates `None` into
/// [`SpliceResult::CrossedUncovered`] preserving the prior unchanged.
fn splice_dir(
    prior: &Arc<DirSnapshot>,
    rest: &[ResourceId],
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> Option<Arc<DirSnapshot>> {
    let Some((&next_id, deeper)) = rest.split_first() else {
        // We're at target. G7-leaf: hash-equal ⇒ keep prior Arc; the
        // splice is a no-op observationally.
        if prior.dir_hash() == replacement.dir_hash() {
            return Some(Arc::clone(prior));
        }
        return Some(replacement);
    };
    // Slot reaped mid-graft. Engine contract says this can't happen for
    // an observed subtree; surface as CrossedUncovered.
    let name = tree.name(next_id)?;
    // Path crossed an uncovered branch (subtree=None) or missing entry.
    // We don't synthesise empty intermediates — that would lie to
    // dir_hash. Surface as CrossedUncovered; the engine keeps its prior
    // view and converges on the next probe.
    let pc: Arc<DirSnapshot> = prior.entries.get(name).and_then(|c| match c {
        ChildEntry::Dir(dc) => dc.subtree.clone(),
        ChildEntry::Leaf(_) => None,
    })?;
    let new_child = splice_dir(&pc, deeper, replacement, tree)?;

    // G7 per-level: child unchanged ⇒ parent unchanged; propagate
    // Arc::ptr_eq up the spine without rebuilding.
    if Arc::ptr_eq(&new_child, &pc) || new_child.dir_hash() == pc.dir_hash() {
        return Some(Arc::clone(prior));
    }

    let mut new_entries = prior.entries.clone();
    new_entries.insert(
        CompactString::new(name),
        ChildEntry::Dir(DirChild {
            inode: new_child.root_meta.inode,
            device: new_child.root_meta.device,
            subtree: Some(new_child),
        }),
    );
    // Preserve prior's `captured_with` on the rebuilt parent: it is
    // conceptually "still the same observation as prior, with one child
    // subtree spliced in", and `captured_with` is constant within a
    // Profile by construction.
    Some(Arc::new(DirSnapshot::new(
        prior.root_resource,
        prior.root_meta,
        prior.captured_with,
        new_entries,
    )))
}

// ---------------------------------------------------------------------------
// diff_tree
// ---------------------------------------------------------------------------

/// [`Diff`] over two parallel [`TreeSnapshot`] trees.
///
/// Walks in lock-step, pruning equal-`dir_hash` subtrees. Output ordering
/// is lex-by-segment within each list — depth-first lex traversal happens
/// to coincide with flat lex sort of `parent/child` segments.
///
/// Cross-level rename detection: the per-level walk collects deltas
/// keyed by `(device, inode)`; a post-pass pairs `Created` and `Deleted`
/// across the entire walk into `Renamed`.
#[must_use]
pub fn diff_tree(baseline: &TreeSnapshot, current: &TreeSnapshot) -> Diff {
    let mut out = Diff::default();
    match (baseline, current) {
        (TreeSnapshot::File(b), TreeSnapshot::File(c)) => diff_file_pair(b, c, &mut out),
        (TreeSnapshot::Dir(b), TreeSnapshot::Dir(c)) => {
            if b.dir_hash() == c.dir_hash() {
                return out; // O(1) prune at root
            }
            let mut staged_created: Vec<StagedEntry> = Vec::new();
            let mut staged_deleted: Vec<StagedEntry> = Vec::new();
            collect_dir_pair(
                b,
                c,
                "",
                &mut out.modified,
                &mut staged_created,
                &mut staged_deleted,
            );
            pair_renames(staged_created, staged_deleted, &mut out);
        }
        // Kind mismatch (File vs Dir) at the anchor: structurally
        // unreachable in v1 — Profile kind is fixed at attach time and
        // a kind change at the anchor surfaces as Vanished, not as a
        // diff. The empty Diff is the safe release behaviour; the
        // debug_assert flags any future contract drift in tests.
        _ => {
            debug_assert!(
                false,
                "diff_tree: File↔Dir mismatch at the anchor is unreachable in v1; \
                 anchor kind changes are reported via Vanished, not diff",
            );
        }
    }
    out
}

#[derive(Clone, Debug)]
struct StagedEntry {
    rel: CompactString,
    kind: EntryKind,
    inode: u64,
    device: u64,
    /// When `false`, `pair_renames` skips this entry's `(device, inode)`
    /// from rename matching and routes it directly to `out.created` /
    /// `out.deleted`. Used for parent slots whose identity has flipped
    /// (kind change at the same name, Dir replaced at a different inode):
    /// such slots represent observably-different entities and are not
    /// rename candidates, even when their inodes coincide. Descendants of
    /// these slots remain eligible — genuine moves into / out of the slot
    /// surface as Renames.
    pair_eligible: bool,
}

fn collect_dir_pair(
    prior: &DirSnapshot,
    new: &DirSnapshot,
    rel_prefix: &str,
    modified: &mut SmallVec<[EntryRef; 4]>,
    staged_created: &mut Vec<StagedEntry>,
    staged_deleted: &mut Vec<StagedEntry>,
) {
    if prior.dir_hash() == new.dir_hash() {
        return;
    }

    let mut left = prior.entries.iter().peekable();
    let mut right = new.entries.iter().peekable();
    loop {
        match (left.peek(), right.peek()) {
            (None, None) => break,
            (Some((ln, lc)), None) => {
                stage_deleted(ln, lc, rel_prefix, staged_deleted);
                left.next();
            }
            (None, Some((rn, rc))) => {
                stage_created(rn, rc, rel_prefix, staged_created);
                right.next();
            }
            (Some((ln, lc)), Some((rn, rc))) => match ln.as_str().cmp(rn.as_str()) {
                std::cmp::Ordering::Less => {
                    stage_deleted(ln, lc, rel_prefix, staged_deleted);
                    left.next();
                }
                std::cmp::Ordering::Greater => {
                    stage_created(rn, rc, rel_prefix, staged_created);
                    right.next();
                }
                std::cmp::Ordering::Equal => {
                    diff_same_name(
                        ln,
                        lc,
                        rc,
                        rel_prefix,
                        modified,
                        staged_created,
                        staged_deleted,
                    );
                    left.next();
                    right.next();
                }
            },
        }
    }
}

fn diff_same_name(
    name: &CompactString,
    pc: &ChildEntry,
    nc: &ChildEntry,
    rel_prefix: &str,
    modified: &mut SmallVec<[EntryRef; 4]>,
    staged_created: &mut Vec<StagedEntry>,
    staged_deleted: &mut Vec<StagedEntry>,
) {
    let rel = compose_rel(rel_prefix, name);
    match (pc, nc) {
        (ChildEntry::Leaf(p), ChildEntry::Leaf(n)) => {
            if p.inode != n.inode || p.device != n.device {
                // Same name, different inode ⇒ delete-then-create. Stage
                // as pair_eligible: each side may legitimately pair with
                // a cross-level entry sharing its inode (the user moved
                // the prior file out and a different one in).
                staged_deleted.push(StagedEntry {
                    rel: rel.clone(),
                    kind: p.kind,
                    inode: p.inode,
                    device: p.device,
                    pair_eligible: true,
                });
                staged_created.push(StagedEntry {
                    rel,
                    kind: n.kind,
                    inode: n.inode,
                    device: n.device,
                    pair_eligible: true,
                });
            } else if p.leaf_hash() != n.leaf_hash() {
                modified.push(EntryRef {
                    segment: rel,
                    kind: n.kind,
                    inode: n.inode,
                });
            }
        }
        (ChildEntry::Dir(p), ChildEntry::Dir(n)) => {
            if p.inode != n.inode || p.device != n.device {
                // Same-name dir-replace at a different inode: parent slot
                // represents a different entity. Stage parent ineligible
                // (it must surface as Deleted + Created, never collapse
                // to a same-rel "Rename"), and recurse both subtrees so
                // descendants surface as Deleted/Created or pair as
                // cross-level Renames against the rest of the walk.
                staged_deleted.push(StagedEntry {
                    rel: rel.clone(),
                    kind: EntryKind::Dir,
                    inode: p.inode,
                    device: p.device,
                    pair_eligible: false,
                });
                staged_created.push(StagedEntry {
                    rel: rel.clone(),
                    kind: EntryKind::Dir,
                    inode: n.inode,
                    device: n.device,
                    pair_eligible: false,
                });
                stage_descendants_deleted(&rel, pc, staged_deleted);
                stage_descendants_created(&rel, nc, staged_created);
            } else {
                match (p.subtree.as_deref(), n.subtree.as_deref()) {
                    (Some(ps), Some(ns)) if ps.dir_hash() != ns.dir_hash() => {
                        collect_dir_pair(ps, ns, &rel, modified, staged_created, staged_deleted);
                    }
                    (Some(_), Some(_)) | (None, None) => {
                        // Hashes match (covered both sides) or both sides
                        // uncovered: no delta to emit at this Dir slot.
                    }
                    (Some(_), None) | (None, Some(_)) => {
                        // Coverage flip on the same Dir slot. Structurally
                        // unreachable in v1: a Profile's coverage rule is
                        // pinned by `config_hash`, so a scope change forks
                        // a new Profile rather than flipping subtree
                        // presence at the same slot. Mirrors the assert in
                        // the engine's `walk_pair`.
                        debug_assert!(
                            false,
                            "diff_same_name: coverage flip on same Dir slot is unreachable in v1",
                        );
                    }
                }
            }
        }
        // Kind change at same name (Leaf↔Dir): the slot represents
        // logically-different entities across the two snapshots. Stage
        // the parent as ineligible (so pair_renames doesn't try to
        // collapse it into a nonsensical same-name "Rename" when the
        // kernel reuses the inode across the kind flip) and recurse the
        // Dir side(s) so descendants surface — either as Deleted/Created
        // or as cross-level Renames.
        _ => {
            staged_deleted.push(StagedEntry {
                rel: rel.clone(),
                kind: pc.kind(),
                inode: pc.inode(),
                device: pc.device(),
                pair_eligible: false,
            });
            staged_created.push(StagedEntry {
                rel: rel.clone(),
                kind: nc.kind(),
                inode: nc.inode(),
                device: nc.device(),
                pair_eligible: false,
            });
            stage_descendants_deleted(&rel, pc, staged_deleted);
            stage_descendants_created(&rel, nc, staged_created);
        }
    }
}

/// Stage every descendant of `parent` (if `parent` is a covered Dir) as
/// Deleted, with `parent_rel` as the rel-prefix. Called from
/// `diff_same_name`'s ineligible-parent paths (kind change, Dir-replace
/// at different inode). Leaves and uncovered Dirs are no-ops.
fn stage_descendants_deleted(parent_rel: &str, parent: &ChildEntry, staged: &mut Vec<StagedEntry>) {
    if let ChildEntry::Dir(d) = parent
        && let Some(sub) = d.subtree.as_deref()
    {
        for (cname, cchild) in &sub.entries {
            stage_deleted(cname, cchild, parent_rel, staged);
        }
    }
}

/// Symmetric counterpart of [`stage_descendants_deleted`].
fn stage_descendants_created(parent_rel: &str, parent: &ChildEntry, staged: &mut Vec<StagedEntry>) {
    if let ChildEntry::Dir(d) = parent
        && let Some(sub) = d.subtree.as_deref()
    {
        for (cname, cchild) in &sub.entries {
            stage_created(cname, cchild, parent_rel, staged);
        }
    }
}

fn stage_deleted(
    name: &CompactString,
    pc: &ChildEntry,
    rel_prefix: &str,
    staged: &mut Vec<StagedEntry>,
) {
    let rel = compose_rel(rel_prefix, name);
    staged.push(StagedEntry {
        rel: rel.clone(),
        kind: pc.kind(),
        inode: pc.inode(),
        device: pc.device(),
        pair_eligible: true,
    });
    // For Dir deletions, recurse to emit each descendant as Deleted.
    // Output is a flat Diff for the Effect API; it doesn't care about
    // reap order. The recursive walk preserves lex within each level.
    if let ChildEntry::Dir(d) = pc
        && let Some(sub) = d.subtree.as_deref()
    {
        for (cname, cchild) in &sub.entries {
            stage_deleted(cname, cchild, &rel, staged);
        }
    }
}

fn stage_created(
    name: &CompactString,
    nc: &ChildEntry,
    rel_prefix: &str,
    staged: &mut Vec<StagedEntry>,
) {
    let rel = compose_rel(rel_prefix, name);
    staged.push(StagedEntry {
        rel: rel.clone(),
        kind: nc.kind(),
        inode: nc.inode(),
        device: nc.device(),
        pair_eligible: true,
    });
    if let ChildEntry::Dir(d) = nc
        && let Some(sub) = d.subtree.as_deref()
    {
        for (cname, cchild) in &sub.entries {
            stage_created(cname, cchild, &rel, staged);
        }
    }
}

/// Pair Created/Deleted entries by `(device, inode)` to recover Renames.
///
/// The index uses `BTreeMap::insert` semantics, so when a `(device, inode)`
/// collides (the pathological hardlink case of multiple Created at the
/// same inode) the *last* index wins. The `paired` set guarantees one
/// Created can match at most one Deleted.
///
/// **Pairing rules.** A `(deleted, created)` pair becomes a `Rename` iff
/// (1) both sides are `pair_eligible`, (2) the `(device, inode)` matches,
/// (3) the `kind` matches, and (4) the `rel` differs. Same-`rel` candidates
/// are structurally impossible for eligible entries (parent kind changes
/// and Dir-replace-at-different-inode stage their parents ineligible;
/// other staging paths cannot produce same-rel collisions in the global
/// buffer) — pinned by the `debug_assert` below. Cross-kind candidates
/// arise from kernel inode reuse across unrelated operations and are
/// not renames; they fall through to Created+Deleted.
///
/// Output order: unpaired Created/Deleted are emitted in collection order
/// (depth-first lex on each side); Renamed entries are emitted in
/// `staged_deleted`'s iteration order (also depth-first lex on the
/// baseline side).
fn pair_renames(
    staged_created: Vec<StagedEntry>,
    staged_deleted: Vec<StagedEntry>,
    out: &mut Diff,
) {
    let mut by_key: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    for (i, c) in staged_created.iter().enumerate() {
        if c.pair_eligible {
            by_key.insert((c.device, c.inode), i);
        }
    }
    let mut paired: BTreeSet<usize> = BTreeSet::new();
    let mut leftover_deleted: Vec<StagedEntry> = Vec::with_capacity(staged_deleted.len());

    for d in staged_deleted {
        if !d.pair_eligible {
            // Ineligible parent (kind change or Dir-replace): never a
            // rename. Route to out.deleted in lex order via the shared
            // leftover queue.
            leftover_deleted.push(d);
            continue;
        }
        match by_key.get(&(d.device, d.inode)) {
            Some(&ci) if !paired.contains(&ci) => {
                let c = &staged_created[ci];
                debug_assert!(
                    c.rel != d.rel,
                    "staging invariant: eligible same-rel pairs should be \
                     reduced upstream (modified, dir-recursion, or marked \
                     ineligible) and never reach pair_renames",
                );
                if c.kind != d.kind {
                    // Cross-kind inode collision (kernel reuse across
                    // unrelated operations). Not a rename — let both
                    // sides surface as Created/Deleted.
                    leftover_deleted.push(d);
                    continue;
                }
                out.renamed.push(Rename {
                    from: EntryRef {
                        segment: d.rel,
                        kind: d.kind,
                        inode: d.inode,
                    },
                    to: EntryRef {
                        segment: c.rel.clone(),
                        kind: c.kind,
                        inode: c.inode,
                    },
                });
                paired.insert(ci);
            }
            _ => leftover_deleted.push(d),
        }
    }

    for (i, c) in staged_created.into_iter().enumerate() {
        if !paired.contains(&i) {
            out.created.push(EntryRef {
                segment: c.rel,
                kind: c.kind,
                inode: c.inode,
            });
        }
    }
    for d in leftover_deleted {
        out.deleted.push(EntryRef {
            segment: d.rel,
            kind: d.kind,
            inode: d.inode,
        });
    }
}

fn diff_file_pair(b: &LeafEntry, c: &LeafEntry, out: &mut Diff) {
    if b.inode == c.inode && b.device == c.device {
        if b.leaf_hash() != c.leaf_hash() {
            out.modified.push(EntryRef {
                segment: CompactString::new(""),
                kind: c.kind,
                inode: c.inode,
            });
        }
    } else {
        // Inode change at the file Profile's anchor: same-segment kind/
        // identity flip. Emit Deleted + Created (no Rename: a file Profile
        // sees its anchor as one fact, not a moved name).
        out.deleted.push(EntryRef {
            segment: CompactString::new(""),
            kind: b.kind,
            inode: b.inode,
        });
        out.created.push(EntryRef {
            segment: CompactString::new(""),
            kind: c.kind,
            inode: c.inode,
        });
    }
}

fn compose_rel(prefix: &str, name: &CompactString) -> CompactString {
    if prefix.is_empty() {
        name.clone()
    } else {
        let mut s = CompactString::new(prefix);
        s.push('/');
        s.push_str(name);
        s
    }
}

#[cfg(test)]
#[path = "tree_tests.rs"]
mod tests;
