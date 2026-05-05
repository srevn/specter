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

/// Tree-zipper splice that replaces the subtree at `target`.
///
/// Produces a new [`TreeSnapshot`] whose subtree at `target` equals
/// `replacement`, sharing all off-path subtrees with `prior` via `Arc`.
/// Rebuilds at most `depth(target)` `DirSnapshot`s along the
/// path-to-anchor.
///
/// Returns `TreeSnapshot::Dir(replacement)` (Arc-cheap) for the trivial
/// cases:
/// - `prior == None` (first graft).
/// - `prior == Some(File(_))` (kind change at the anchor).
/// - `target == prior.root_resource` and the hashes differ (new root).
///
/// Returns `TreeSnapshot::Dir(prior)` (no allocation) when:
/// - `target == prior.root_resource` and `dir_hash` matches (G7-trivial).
/// - The recursive splice short-circuited at every level via `Arc::ptr_eq`
///   or `dir_hash` equality (G7 propagation).
///
/// Defensive fallback: if the path from anchor to target can't be resolved
/// in `tree`, returns `TreeSnapshot::Dir(replacement)` — the engine's
/// contract is "graft only into observed subtrees", so this path indicates a
/// contract violation. We prefer a wholesale replace over corrupting the
/// prior.
#[must_use]
pub fn splice(
    prior: Option<TreeSnapshot>,
    target: ResourceId,
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> TreeSnapshot {
    let Some(TreeSnapshot::Dir(root)) = prior else {
        // None or File(_) prior: replace wholesale.
        return TreeSnapshot::Dir(replacement);
    };
    let anchor = root.root_resource;
    if target == anchor {
        if root.dir_hash() == replacement.dir_hash() {
            return TreeSnapshot::Dir(root);
        }
        return TreeSnapshot::Dir(replacement);
    }

    let Some(chain) = ancestor_chain(target, anchor, tree) else {
        return TreeSnapshot::Dir(replacement);
    };

    // chain is [anchor, mid_1, ..., mid_k, target]; we already consumed
    // the anchor as `root`, so descend with chain[1..].
    let new_root = splice_dir(&root, &chain[1..], replacement, tree);
    if Arc::ptr_eq(&new_root, &root) {
        TreeSnapshot::Dir(root)
    } else {
        TreeSnapshot::Dir(new_root)
    }
}

fn splice_dir(
    prior: &Arc<DirSnapshot>,
    rest: &[ResourceId],
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> Arc<DirSnapshot> {
    let Some((&next_id, deeper)) = rest.split_first() else {
        // We're at target. G7-leaf: hash-equal ⇒ keep prior Arc; the
        // splice is a no-op observationally.
        if prior.dir_hash() == replacement.dir_hash() {
            return Arc::clone(prior);
        }
        return replacement;
    };
    let Some(name) = tree.name(next_id) else {
        // Slot reaped mid-graft. Engine contract says this can't happen
        // for an observed subtree; fall back to prior unchanged.
        return Arc::clone(prior);
    };
    let prior_child: Option<Arc<DirSnapshot>> = prior.entries.get(name).and_then(|c| match c {
        ChildEntry::Dir(dc) => dc.subtree.clone(),
        ChildEntry::Leaf(_) => None,
    });
    let Some(pc) = prior_child else {
        // Path crossed an uncovered branch (subtree=None) or a missing
        // entry. We don't synthesise empty intermediates — that would lie
        // to dir_hash. Fall back to prior; the engine re-probes if it
        // needs the deeper view.
        return Arc::clone(prior);
    };
    let new_child = splice_dir(&pc, deeper, replacement, tree);

    // G7 per-level: child unchanged ⇒ parent unchanged; propagate
    // Arc::ptr_eq up the spine without rebuilding.
    if Arc::ptr_eq(&new_child, &pc) || new_child.dir_hash() == pc.dir_hash() {
        return Arc::clone(prior);
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
    Arc::new(DirSnapshot::new(
        prior.root_resource,
        prior.root_meta,
        prior.captured_with,
        new_entries,
    ))
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
        // Kind mismatch (File vs Dir): can't happen in v1 — Profile
        // kind is fixed at attach time and a kind change at the anchor
        // is reported as Vanished. Empty diff is the safe answer.
        _ => {}
    }
    out
}

#[derive(Clone, Debug)]
struct StagedEntry {
    rel: CompactString,
    kind: EntryKind,
    inode: u64,
    device: u64,
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
                // both; rename pairing in the post-pass may join them
                // (it skips when both segments match, so this stays as
                // a Deleted+Created pair).
                staged_deleted.push(StagedEntry {
                    rel: rel.clone(),
                    kind: p.kind,
                    inode: p.inode,
                    device: p.device,
                });
                staged_created.push(StagedEntry {
                    rel,
                    kind: n.kind,
                    inode: n.inode,
                    device: n.device,
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
                staged_deleted.push(StagedEntry {
                    rel: rel.clone(),
                    kind: EntryKind::Dir,
                    inode: p.inode,
                    device: p.device,
                });
                staged_created.push(StagedEntry {
                    rel,
                    kind: EntryKind::Dir,
                    inode: n.inode,
                    device: n.device,
                });
            } else if let (Some(ps), Some(ns)) = (p.subtree.as_deref(), n.subtree.as_deref())
                && ps.dir_hash() != ns.dir_hash()
            {
                collect_dir_pair(ps, ns, &rel, modified, staged_created, staged_deleted);
            }
            // (Some, None) or (None, Some): coverage flip on the same
            // dir. Diff doesn't emit a delta for coverage changes — the
            // engine's reconciler handles depth/exclude transitions on
            // the next probe.
        }
        // Kind change at same name ⇒ delete + create with the recorded
        // kinds.
        _ => {
            staged_deleted.push(StagedEntry {
                rel: rel.clone(),
                kind: pc.kind(),
                inode: pc.inode(),
                device: pc.device(),
            });
            staged_created.push(StagedEntry {
                rel,
                kind: nc.kind(),
                inode: nc.inode(),
                device: nc.device(),
            });
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
/// The index is `BTreeMap::insert`,
/// so when a `(device, inode)` collides (the pathological hardlink case
/// of multiple Created at the same inode) the *last* index wins. The
/// `paired` set guarantees one Created can match at most one Deleted.
///
/// Output order: unpaired Created/Deleted are emitted in collection order
/// (depth-first lex); Renamed entries are emitted in `staged_deleted`'s
/// iteration order (also depth-first lex on the baseline side).
fn pair_renames(
    staged_created: Vec<StagedEntry>,
    staged_deleted: Vec<StagedEntry>,
    out: &mut Diff,
) {
    let mut by_key: BTreeMap<(u64, u64), usize> = BTreeMap::new();
    for (i, c) in staged_created.iter().enumerate() {
        by_key.insert((c.device, c.inode), i);
    }
    let mut paired: BTreeSet<usize> = BTreeSet::new();
    let mut leftover_deleted: Vec<StagedEntry> = Vec::with_capacity(staged_deleted.len());

    for d in staged_deleted {
        match by_key.get(&(d.device, d.inode)) {
            Some(&ci) if !paired.contains(&ci) => {
                let c = &staged_created[ci];
                if c.rel == d.rel {
                    // Same name + same inode: the entry didn't actually
                    // change identity. Mark the Created as paired so it
                    // isn't emitted; drop the Deleted (no delta).
                    paired.insert(ci);
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
