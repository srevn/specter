//! Hierarchical snapshot — `DirSnapshot`, `LeafEntry`, `DirChild`, `ChildEntry`, `DirMeta`, and the
//! engine-facing `TreeSnapshot` enum (re-exported at the lib root as [`crate::TreeSnapshot`]).
//!
//! ## Identity model
//!
//! - Snapshots are tree-shaped. Each `DirSnapshot` owns one directory's `lstat` triple ([`DirMeta`]),
//!   the `ScanConfig` hash they were captured under (`captured_with`), and a `BTreeMap<CompactString,
//!   ChildEntry>` of direct children (string-keyed — keeps the hash cross-process stable).
//! - Children are either [`LeafEntry`] (file/symlink/other; no recursion) or [`DirChild`]
//!   (directory; a sum type with two variants — `Covered(Arc<DirSnapshot>)` and
//!   `Uncovered(FsIdentity)`). `Uncovered` means the walker stored the entry but did not recurse
//!   (`recursive=false`, beyond `max_depth`, or cross-filesystem); the sum makes the coverage
//!   discrimination structural rather than `Option`-tagged.
//! - A `DirSnapshot` carries no engine-side identity. The walker speaks paths; the engine speaks
//!   resources; navigation helpers ([`subtree_at_dir`], [`TreeSnapshot::subtree_at`]) take an
//!   explicit `anchor: ResourceId` so the caller's anchor invariant lives at the call site rather
//!   than as a stamp on the wire payload.
//!
//! ## Hashing
//!
//! - [`LeafEntry::leaf_hash`] is a 128-bit fingerprint of `(kind, size, mtime, fs_id)`. A leaf's
//!   mtime is its per-file content fingerprint, so it belongs in the identity.
//! - [`DirSnapshot::dir_hash`] folds `captured_with`, the directory's own `fs_id`, an entry-count
//!   length-prefix, then each `(name, ChildEntry)` pair in lex order, with a per-variant tag
//!   distinguishing the three child shapes: `Leaf` contributes `leaf_hash`, `Dir(Covered)`
//!   contributes the subtree's `dir_hash`, and `Dir(Uncovered)` contributes the raw `fs_id` (the
//!   walker has no observation beyond the directory's identity). `root_meta.mtime` is **not** folded
//!   — `dir_hash` is filter-aware identity ("are these snapshots observably the same to the user?"),
//!   and a directory's mtime bumps on every dirent-block change including filtered-out entries
//!   (`.DS_Store`, hidden files, excluded paths) the user-configured filter would never present. The
//!   walker's mtime-skip optimisation reads `root_meta.mtime` as a struct field on [`DirSnapshot`]
//!   (its own kernel-aware identity); no consumer needs the value composed into the hash.
//! - Both hashes are computed **eagerly at construction** and stored as plain `u128` fields. The
//!   walker pays SipHash24 on its worker thread (parallel pool); the engine driver reads the field
//!   via a `const fn` accessor. Eager construction collapses the prior `OnceLock<u128>` cache
//!   discipline (manual `Clone` / `PartialEq` / `Debug` impls, `with_cached_hash` poisoning hazard)
//!   into a function-of-data invariant: `leaf_hash(l) == compute_leaf_hash(l.fields)` and
//!   `dir_hash(d) == compute_dir_hash(d.fields)` hold by construction, not by convention.
//! - The walker may inherit a baseline leaf's hash via [`LeafEntry::from_metadata_or_inherit`] when
//!   identity fields match — a pure performance optimisation (skips one SipHash24 fold per
//!   unchanged leaf in a dirent-bumped directory), semantically equivalent to recomputing since the
//!   inherited value is identical to what recomputation would produce.
//! - 128-bit width (`siphasher::sip128`): pair-comparison space at scale (`O(levels × bursts ×
//!   profiles)`) makes 64-bit collision probability uncomfortable; 128 bits is astronomically safe.
//! - `FsIdentity` folds into a digest only via [`crate::fs_id::encode_into`] (`inode` then
//!   `device`, each a little-endian `u64`); nested digests (`leaf_hash`, `dir_hash`) fold through
//!   `StableHasher::put_u128`, an endian-explicit width. Pinned by
//!   `fs_id::tests::encode_into_matches_inode_then_device` plus the snapshot goldens so a field
//!   reorder or encoding change fires at nextest time before goldens silently drift.
//!
//! ## Mutability and concurrency
//!
//! - `DirSnapshot` and `LeafEntry` are fully immutable post-construction — every field is a plain
//!   value or an `Arc<...>`, with no interior mutability. Both types are `Send + Sync` trivially;
//!   pinned by a compile-time `Send + Sync` assertion in tests.
//! - Splice and graft build *new* `Arc<DirSnapshot>`s — never mutate-through-Arc — so engine and
//!   walker can share `Arc<DirSnapshot>` handles without locks.

use crate::diag::SpliceFailureCause;
use crate::diff::{Diff, EntryRef, Rename};
use crate::fs_id::{FsIdentity, encode_into};
use crate::hash::{hasher_128, put_systemtime_into};
use crate::ids::ResourceId;
use crate::snapshot::EntryKind;
use crate::tree::Tree;
use compact_str::CompactString;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::SystemTime;

// ---------------------------------------------------------------------------
// DirMeta
// ---------------------------------------------------------------------------

/// `lstat` pair of a directory: the load-bearing fields for the V5 walker's mtime-skip plus the
/// kernel identity guarding against the delete-and-recreate-at-same-path case.
///
/// `mtime` drives the skip; `fs_id` defends against the case where the directory was unlinked and
/// recreated at the same path between probes (the recreation gets a fresh inode under POSIX
/// semantics — same name, different identity).
///
/// ## Atomicity invariant
///
/// `mtime` and `fs_id` are meaningful only as the pair read from a *single* `lstat`: across two
/// `lstat`s the kernel can bump the mtime and recycle the inode independently, so a torn `(mtime,
/// fs_id)` assembled from separate observations names no coherent directory state. The fields are
/// private and the sole production constructor, [`DirMeta::from_metadata`], reads both halves from
/// one `&Metadata`. The pair feeds both `compute_dir_hash` (via `fs_id`) and the walker's
/// whole-`DirMeta` mtime-skip compare, so a torn pair is correctness-adjacent, not hygiene — the
/// invariant is discharged by the type, not by caller convention.
///
/// `Ord`/`PartialOrd` are intentionally absent: nothing orders `DirMeta` (the `pair_renames` index
/// keys on `FsIdentity`). `Eq`/ `PartialEq` are load-bearing — the walker's mtime-skip compares the
/// whole pair and [`DirSnapshot`]'s derived `PartialEq` composes it.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct DirMeta {
    mtime: SystemTime,
    fs_id: FsIdentity,
}

impl DirMeta {
    /// Construct from a single freshly-`lstat`ed `Metadata`.
    ///
    /// Both halves are read here, inside the constructor, from the *same* `&Metadata` — this is
    /// what discharges the atomicity invariant at the type boundary. `Metadata::modified` and
    /// `MetadataExt::ino`/`dev` read fields the sensor's `lstat` already populated; they are *not*
    /// syscalls, so this stays consistent with `core`'s no-I/O discipline (I1).
    ///
    /// The `UNIX_EPOCH` mtime fallback (platform without a usable modified-time) is centralised
    /// here — the single production source of a `DirMeta` mtime, so the sentinel cannot vary across
    /// walker call sites.
    #[cfg(unix)]
    #[must_use]
    pub fn from_metadata(meta: &std::fs::Metadata) -> Self {
        Self {
            mtime: meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            fs_id: FsIdentity::from_metadata(meta),
        }
    }

    /// The directory mtime observed at `lstat` time — the walker's mtime-skip key.
    #[must_use]
    pub const fn mtime(self) -> SystemTime {
        self.mtime
    }

    /// The directory's kernel identity observed at `lstat` time.
    #[must_use]
    pub const fn fs_id(self) -> FsIdentity {
        self.fs_id
    }

    /// Test-only constructor from explicit halves, bypassing the single-`lstat` provenance
    /// [`DirMeta::from_metadata`] enforces. Compiled only under `cfg(test)` or the `testkit`
    /// feature, mirroring [`FsIdentity::synthetic`].
    #[cfg(any(test, feature = "testkit"))]
    #[must_use]
    pub const fn synthetic(mtime: SystemTime, fs_id: FsIdentity) -> Self {
        Self { mtime, fs_id }
    }
}

// ---------------------------------------------------------------------------
// LeafEntry
// ---------------------------------------------------------------------------

/// Direct child that is *not* a directory: file, symlink, or other.
///
/// `leaf_hash` is the 128-bit fingerprint of `(kind, size, mtime, fs_id)` — see
/// [`LeafEntry::leaf_hash`]. All five fields are private: every production leaf is built from one
/// `&Metadata` via [`LeafEntry::from_metadata`] / [`LeafEntry::from_metadata_or_inherit`], so the
/// four identity fields are atomic (one `lstat`) and `leaf_hash == compute_leaf_hash(fields)` holds
/// by construction. `Clone` / `PartialEq` / `Debug` auto-derive correctly because the hash is a
/// pure function of the data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct LeafEntry {
    kind: EntryKind,
    size: u64,
    mtime: SystemTime,
    fs_id: FsIdentity,
    leaf_hash: u128,
}

impl LeafEntry {
    /// The single hash-computing constructor: every leaf whose hash is *computed* (rather than
    /// inherited) routes through here, so the eager-hash invariant `leaf_hash ==
    /// compute_leaf_hash(fields)` has exactly one production enforcement point. Private — callers
    /// reach it via `from_metadata` / `synthetic`.
    fn from_parts(kind: EntryKind, size: u64, mtime: SystemTime, fs_id: FsIdentity) -> Self {
        Self {
            kind,
            size,
            mtime,
            fs_id,
            leaf_hash: compute_leaf_hash(kind, size, mtime, fs_id),
        }
    }

    /// Construct from a single freshly-`lstat`ed `Metadata`, computing `leaf_hash` eagerly. `kind`
    /// is derived from `meta.file_type()` (file → `File`, symlink → `Symlink`, else → `Other`).
    ///
    /// **Precondition:** `meta` is *not* a directory. There is no `is_dir` arm — a directory
    /// `FileType` would degrade to `EntryKind::Other`. Leaf construction is never reached for a
    /// directory by the walker's dispatch (`enumerate_dir` routes `is_dir` dirents to
    /// `build_dir_child`); the precondition holds by caller contract and is not re-checked here.
    ///
    /// `Metadata` field reads are not syscalls — consistent with `core`'s no-I/O discipline (I1).
    #[cfg(unix)]
    #[must_use]
    pub fn from_metadata(meta: &std::fs::Metadata) -> Self {
        Self::from_parts(
            entry_kind_from_file_type(meta.file_type()),
            meta.len(),
            meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
            FsIdentity::from_metadata(meta),
        )
    }

    /// As [`LeafEntry::from_metadata`], but inherits `baseline`'s `leaf_hash` iff every identity
    /// field matches — eliding one SipHash24 fold per unchanged leaf in a dirent-bumped directory.
    /// Observably equivalent to recomputation: `baseline` was itself built through `from_parts` (or
    /// `synthetic`), so on a full identity match `baseline.leaf_hash == compute_leaf_hash(kind,
    /// size, mtime, fs_id)` already — the inherited value *is* the recomputed one. This is the
    /// single deliberate bypass of `from_parts`; the identity gate makes "transferring the wrong
    /// hash" unrepresentable. Same non-directory precondition as [`LeafEntry::from_metadata`].
    #[cfg(unix)]
    #[must_use]
    pub fn from_metadata_or_inherit(meta: &std::fs::Metadata, baseline: Option<&Self>) -> Self {
        let kind = entry_kind_from_file_type(meta.file_type());
        let size = meta.len();
        let mtime = meta.modified().unwrap_or(SystemTime::UNIX_EPOCH);
        let fs_id = FsIdentity::from_metadata(meta);
        if let Some(b) = baseline
            && b.kind == kind
            && b.size == size
            && b.mtime == mtime
            && b.fs_id == fs_id
        {
            return Self {
                kind,
                size,
                mtime,
                fs_id,
                leaf_hash: b.leaf_hash,
            };
        }
        Self::from_parts(kind, size, mtime, fs_id)
    }

    /// The leaf's entry kind (file / symlink / other) at `lstat` time.
    #[must_use]
    pub const fn kind(&self) -> EntryKind {
        self.kind
    }

    /// The leaf's byte size at `lstat` time.
    #[must_use]
    pub const fn size(&self) -> u64 {
        self.size
    }

    /// The leaf's mtime — its per-file content-fingerprint component.
    #[must_use]
    pub const fn mtime(&self) -> SystemTime {
        self.mtime
    }

    /// The leaf's kernel identity observed at `lstat` time.
    #[must_use]
    pub const fn fs_id(&self) -> FsIdentity {
        self.fs_id
    }

    /// 128-bit fingerprint of `(kind, size, mtime, fs_id)`. Computed at construction time; this
    /// accessor is `const fn` and reads the stored field.
    #[must_use]
    pub const fn leaf_hash(&self) -> u128 {
        self.leaf_hash
    }

    /// Test-only constructor from explicit identity fields, computing `leaf_hash` so fixtures
    /// uphold the eager-hash invariant too. Bypasses the single-`lstat` provenance
    /// [`LeafEntry::from_metadata`] enforces; compiled only under `cfg(test)` or the `testkit`
    /// feature, mirroring [`FsIdentity::synthetic`].
    #[cfg(any(test, feature = "testkit"))]
    #[must_use]
    pub fn synthetic(kind: EntryKind, size: u64, mtime: SystemTime, fs_id: FsIdentity) -> Self {
        Self::from_parts(kind, size, mtime, fs_id)
    }
}

// ---------------------------------------------------------------------------
// Leaf-kind derivation
// ---------------------------------------------------------------------------

/// Map a `std::fs::FileType` to the leaf [`EntryKind`] the snapshot stores. **Not total over the
/// filesystem:** there is deliberately no `is_dir` arm — a directory `FileType` falls through to
/// `EntryKind::Other`. The contract is that leaf construction is never reached for a directory (the
/// walker's `enumerate_dir` routes `is_dir` dirents to `build_dir_child`); callers of
/// [`LeafEntry::from_metadata`] uphold that precondition.
#[cfg(unix)]
#[must_use]
fn entry_kind_from_file_type(ft: std::fs::FileType) -> EntryKind {
    entry_kind_from_flags(ft.is_file(), ft.is_symlink())
}

/// The leaf-kind decision over just the two booleans it depends on: `is_file` ⇒ `File`; else
/// `is_symlink` ⇒ `Symlink`; else `Other` (directory / fifo / socket / block / char all map to
/// `Other`). The signature *structurally* excludes an `is_dir` input, so the "no `is_dir` arm"
/// guarantee is enforced by the type, not by a comment. `const fn` and fs-free, so it is
/// unit-testable in `core` without I/O.
#[must_use]
const fn entry_kind_from_flags(is_file: bool, is_symlink: bool) -> EntryKind {
    if is_file {
        EntryKind::File
    } else if is_symlink {
        EntryKind::Symlink
    } else {
        EntryKind::Other
    }
}

// ---------------------------------------------------------------------------
// DirChild
// ---------------------------------------------------------------------------

/// Direct child that *is* a directory.
///
/// Sum type encoding the walker's covered/uncovered distinction structurally — `Covered` carries
/// the recursive `Arc<DirSnapshot>` (whose `root_meta.fs_id` is the kernel identity); `Uncovered`
/// carries the `FsIdentity` directly.
///
/// `Uncovered` means *the walker stored the entry but did not recurse* because the scan shape's
/// recursion edge (`ScanConfig::descends_into`) refused the level: `Subtree`'s `recursive=false`,
/// beyond-`max_depth`, or cross-filesystem gates (the child's `fs_id.device` differs from the
/// anchor's `root_dev`), or `MatchChain`'s terminus depth (a matched directory at the chain's end
/// is membership, not content — the pruned walk stops there by design). The walker never mints
/// `Uncovered` for transient I/O failures (raced unlink, kind-flip, EACCES on the subdir's
/// `read_dir`); those surface as `Covered(empty_or_partial_arc)` via the walker's `read_dir`
/// benign-empty contract, distinct from the uncovered variant. The structural consequence: within a
/// Profile (whose `config_hash` freezes the scan shape and its depth bounds, and whose cross-fs
/// identity bifurcates through `fs_id` rather than this variant), the `(Covered, Uncovered)` and
/// `(Uncovered, Covered)` transitions on the *same* `fs_id` are unreachable.
///
/// Two boundary cases to keep distinct:
/// - **`exclude` glob**: filtered entries are absent from the parent's `entries` map entirely — the
///   walker never constructs a `DirChild` for them, so neither variant applies.
/// - **`read_dir` failure (EACCES, EIO, …)**: the parent's `lstat` succeeded but enumeration of
///   *this* directory's contents failed. The walker emits a *covered-but-empty*
///   `DirChild::Covered(arc)` where `arc.entries` is empty — the engine sees a known-empty subtree,
///   not an uncovered slot. The walker contract in `specter-sensor::prober::walk::probe_subtree` is
///   authoritative.
///
/// Subtree mtime is **not** stored on `DirChild` — the canonical mtime lives at
/// `Covered(_).root_meta.mtime` and is consumed by the walker's mtime-skip directly off the
/// [`DirSnapshot`] struct field. The parent `dir_hash` fold deliberately omits subtree mtime
/// (filter-aware identity is independent of kernel-side dirent-block churn).
///
/// Both variants project a uniform `fs_id()` (sourced from the subtree's `root_meta.fs_id` on
/// `Covered`, or stored directly on `Uncovered`); the hash fold uses a per-variant tag so the two
/// shapes contribute distinct payloads even when their identities coincide.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DirChild {
    /// The walker recursed and stored the directory's snapshot. The kernel identity lives at
    /// `arc.root_meta.fs_id`.
    Covered(Arc<DirSnapshot>),
    /// The walker stored the entry but did not recurse — one of the three static-config gates fired
    /// (`!recursive`, `max_depth`, or cross-fs). Carries the kernel identity directly.
    Uncovered(FsIdentity),
}

impl DirChild {
    /// Kernel identity of the directory. For `Covered`, sourced from the subtree's
    /// `root_meta.fs_id`; for `Uncovered`, the stored value. Single source of truth per variant.
    #[must_use]
    pub fn fs_id(&self) -> FsIdentity {
        match self {
            Self::Covered(s) => s.root_meta.fs_id,
            Self::Uncovered(id) => *id,
        }
    }
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
    /// Kernel identity of the underlying entry — same accessor for Leaf and Dir. Used by the engine
    /// reconciler for fs_id-stable Dir pairs and by `diff_tree`'s rename pairing.
    #[must_use]
    pub fn fs_id(&self) -> FsIdentity {
        match self {
            Self::Leaf(l) => l.fs_id,
            Self::Dir(d) => d.fs_id(),
        }
    }

    /// `EntryKind` projection for downstream `Diff` emission. Dir always projects to
    /// `EntryKind::Dir`; Leaf preserves its tag.
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

/// One directory's snapshot. Recursive via `ChildEntry::Dir`'s `Option<Arc<DirSnapshot>>`. The
/// `Arc` discipline lets splice and the walker's mtime-skip share subtrees across snapshots without
/// copying.
///
/// ## Immutability boundary
///
/// All data fields are private and there is no mutator, so **no crate outside this module can
/// desync the eager `dir_hash` or assemble a snapshot from torn observations** — the engine/sensor
/// lifecycle the seal closes. `root_meta` is an atomic-by-construction [`DirMeta`] (one `lstat`,
/// via [`DirMeta::from_metadata`]); the sole constructor [`Self::new`] folds `dir_hash` eagerly
/// from the inputs. Within this module every constructor folds eagerly, so the module — not a
/// caller convention — is the trust boundary: there is no public surface, in any crate, that
/// installs a `dir_hash` disagreeing with the data. `Clone` / `PartialEq` / `Debug` auto-derive
/// correctly because the hash is a pure function of the rest. `Send + Sync` are trivially derived;
/// compile-time pinned in tests.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DirSnapshot {
    root_meta: DirMeta,
    captured_with: u64,
    entries: BTreeMap<CompactString, ChildEntry>,
    dir_hash: u128,
}

impl DirSnapshot {
    /// Sole constructor. Takes already-built entries; doesn't sort (`BTreeMap` is sorted-by-key by
    /// construction). Folds the 128-bit `dir_hash` over the inputs eagerly — every child
    /// `Arc<DirSnapshot>` in `entries` already carries its own eagerly-computed hash, so the fold
    /// is a pure read.
    #[must_use]
    pub fn new(
        root_meta: DirMeta,
        captured_with: u64,
        entries: BTreeMap<CompactString, ChildEntry>,
    ) -> Self {
        let dir_hash = compute_dir_hash(&root_meta, captured_with, &entries);
        Self {
            root_meta,
            captured_with,
            entries,
            dir_hash,
        }
    }

    /// The directory's atomic `lstat` pair ([`DirMeta`]) at capture time. `DirMeta` is `Copy`; the
    /// walker's mtime-skip compares the whole pair against a fresh `lstat`.
    #[must_use]
    pub const fn root_meta(&self) -> DirMeta {
        self.root_meta
    }

    /// The `ScanConfig` hash this directory was captured under — equal values mean two snapshots
    /// share one filter regime.
    #[must_use]
    pub const fn captured_with(&self) -> u64 {
        self.captured_with
    }

    /// The direct children, string-keyed and lex-ordered by `BTreeMap`. Borrowed read-only — the
    /// map is immutable post-construction.
    #[must_use]
    pub const fn entries(&self) -> &BTreeMap<CompactString, ChildEntry> {
        &self.entries
    }

    /// 128-bit fingerprint of `(captured_with, root_meta.fs_id, entries)`. `root_meta.mtime` is
    /// intentionally absent from the fold — `dir_hash` is filter-aware identity, while the raw
    /// `lstat` mtime lives on the [`DirSnapshot::root_meta`] struct field for kernel-aware
    /// comparisons (the walker's mtime-skip). Computed at construction; this accessor is `const fn`
    /// and reads the stored field.
    #[must_use]
    pub const fn dir_hash(&self) -> u128 {
        self.dir_hash
    }

    /// Look up `name` and return the entry's `LeafEntry` iff present and the entry is a leaf.
    /// Returns `None` for missing entries and for `Dir` entries.
    ///
    /// Sole intended caller is the walker's per-leaf cache-transfer site, which feeds the result into
    /// [`LeafEntry::from_metadata_or_inherit`] to elide the SipHash24 fold on identity-matching
    /// baselines. The identity check that gates the inheritance lives in `from_metadata_or_inherit`,
    /// not here — this primitive returns the raw reference; callers compose the gate.
    #[must_use]
    pub fn lookup_leaf(&self, name: &str) -> Option<&LeafEntry> {
        match self.entries.get(name)? {
            ChildEntry::Leaf(l) => Some(l),
            ChildEntry::Dir(_) => None,
        }
    }

    /// Look up `name` and return the covered subtree iff present and the entry is a
    /// `Dir(Covered(_))`. Returns `None` for missing entries, for `Leaf` entries, and for
    /// `Dir(Uncovered(_))`.
    ///
    /// Three call sites consume the covered-dir slot: the walker's recursive baseline lookup,
    /// [`subtree_at_dir`]'s descent step, and `splice_dir`'s prior-child resolution. Each needs
    /// "`Dir` entry that is `Covered`" → `Arc<DirSnapshot>` as a single named operation; this
    /// primitive collapses the `entries.get(name)` + variant match into one call.
    #[must_use]
    pub fn lookup_covered_dir(&self, name: &str) -> Option<&Arc<Self>> {
        match self.entries.get(name)? {
            ChildEntry::Dir(DirChild::Covered(s)) => Some(s),
            ChildEntry::Dir(DirChild::Uncovered(_)) | ChildEntry::Leaf(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// TreeSnapshot (engine-facing top-level)
// ---------------------------------------------------------------------------

/// Engine-facing snapshot. File-anchored Profiles carry one [`LeafEntry`]; Dir-anchored Profiles
/// carry an `Arc<DirSnapshot>` (the recursive tree).
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TreeSnapshot {
    File(LeafEntry),
    Dir(Arc<DirSnapshot>),
}

impl TreeSnapshot {
    /// Anchor-rooted snapshot hash. Dispatches to the variant's eager digest:
    /// [`DirSnapshot::dir_hash`] for `Dir`, [`LeafEntry::leaf_hash`] for `File`. Both are 128-bit
    /// SipHash-2-4 digests computed once at snapshot construction and stored as plain fields — the
    /// read is a field access, not a lazy-cache fill.
    #[must_use]
    pub fn hash(&self) -> u128 {
        match self {
            Self::Dir(arc) => arc.dir_hash(),
            Self::File(leaf) => leaf.leaf_hash(),
        }
    }

    /// Stability verdict. One `dir_hash` (or `leaf_hash`) comparison; O(1) after the cache is filled.
    ///
    /// Kind mismatch (File vs Dir) is never stable — kind changes route through `Vanished` at the
    /// probe layer; this arm is defence-in-depth.
    #[must_use]
    pub fn stable_against(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::File(a), Self::File(b)) => a.leaf_hash() == b.leaf_hash(),
            (Self::Dir(a), Self::Dir(b)) => a.dir_hash() == b.dir_hash(),
            _ => false,
        }
    }

    /// Walk this snapshot down to the directory at `target`, following the segment chain
    /// `tree.parent(target) → ... → anchor`. Returns `None` for any of:
    ///
    /// - `TreeSnapshot::File` (no recursion possible).
    /// - `target` outside `anchor`'s subtree (the parent walk bottoms out before reaching `anchor`).
    /// - The chain crosses a `Leaf` or a `DirChild::Uncovered(_)` intermediate.
    /// - Any segment fails to resolve via `tree.name` (slot reaped).
    ///
    /// `anchor` is the engine-side `ResourceId` the caller knows the snapshot is rooted at — for
    /// [`crate::Profile::current`], this is `profile.resource`. Navigation uses `&Tree`
    /// exclusively; `DirSnapshot` carries no engine identity of its own.
    #[must_use]
    pub fn subtree_at(
        &self,
        anchor: ResourceId,
        target: ResourceId,
        tree: &Tree,
    ) -> Option<Arc<DirSnapshot>> {
        match self {
            Self::Dir(root) => subtree_at_dir(root, anchor, target, tree),
            Self::File(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Hash helpers
// ---------------------------------------------------------------------------

// Per-child variant discriminators folded into `compute_dir_hash` so the encoding is unambiguous
// across variants. `0u8` is reserved (never emitted) — the gap defends against zero-padded inputs
// masquerading as a real tag and leaves room for future variants without renumbering the existing
// ones.

/// Tag for `ChildEntry::Leaf`.
const LEAF_TAG: u8 = 1;

/// Tag for `DirChild::Covered` — a Dir child the snapshot recursed into; its subtree is observable.
const DIR_COVERED_TAG: u8 = 2;

/// Tag for `DirChild::Uncovered` — a Dir child the snapshot did not descend; its subtree is opaque.
const DIR_UNCOVERED_TAG: u8 = 3;

fn compute_leaf_hash(kind: EntryKind, size: u64, mtime: SystemTime, fs_id: FsIdentity) -> u128 {
    let mut h = hasher_128();
    h.put_u8(kind as u8);
    h.put_u64(size);
    put_systemtime_into(mtime, &mut h);
    encode_into(fs_id, &mut h);
    h.finish_u128()
}

fn compute_dir_hash(
    root_meta: &DirMeta,
    captured_with: u64,
    entries: &BTreeMap<CompactString, ChildEntry>,
) -> u128 {
    let mut h = hasher_128();

    // Header: ScanConfig hash + the directory's own kernel identity. `root_meta.mtime` is **not**
    // folded — filter-aware identity is independent of kernel-side dirent-block churn (filtered-out
    // entries bump mtime without changing the user-visible state). The walker reads `root_meta.mtime`
    // directly off the struct field for its mtime-skip; no consumer needs it via the hash.
    h.put_u64(captured_with);
    encode_into(root_meta.fs_id, &mut h);

    // Length prefix: belt-and-suspenders alongside SipHash24's prefix-freeness. Keeps the golden
    // test legible.
    h.put_u64(entries.len() as u64);

    // Sequential lex-order fold (BTreeMap iterates in lex order). XOR was rejected: sequential
    // preserves ordering information and avoids commutative-fold subtleties at no real cost
    // (entries are already sorted by construction).
    //
    // Per-variant tags keep the three child shapes unambiguous at the hash level — without a tag, a
    // `Leaf` with `leaf_hash = X` and a `Dir(Covered)` whose subtree's `dir_hash = X` would fold
    // identically into the parent (vanishingly unlikely under SipHash24, but the tag makes the
    // discrimination structural):
    // - `Leaf`          contributes `leaf_hash` (transitively folds
    //                   `(kind, size, mtime, fs_id)`).
    // - `Dir(Covered)`   contributes `dir_hash` (transitively folds
    //                   `(root_meta.fs_id, entries...)`).
    // - `Dir(Uncovered)` contributes the raw `fs_id` — the walker has no observation beyond the
    //   directory's identity.
    for (name, child) in entries {
        h.put_str(name.as_str());
        match child {
            ChildEntry::Leaf(l) => {
                h.put_u8(LEAF_TAG);
                h.put_u128(l.leaf_hash());
            }
            ChildEntry::Dir(DirChild::Covered(s)) => {
                h.put_u8(DIR_COVERED_TAG);
                h.put_u128(s.dir_hash());
            }
            ChildEntry::Dir(DirChild::Uncovered(fs_id)) => {
                h.put_u8(DIR_UNCOVERED_TAG);
                encode_into(*fs_id, &mut h);
            }
        }
    }

    h.finish_u128()
}

// ---------------------------------------------------------------------------
// subtree_at_dir
// ---------------------------------------------------------------------------

/// Descend from `root` (a Dir-shaped anchor snapshot) by following the segment chain to `target`
/// and return the matching subtree (or `None` if navigation cannot reach `target`).
///
/// Same semantics as [`TreeSnapshot::subtree_at`] but typed for the Dir-only call sites — graft /
/// splice plumbing — so they avoid the `TreeSnapshot::Dir(Arc::clone(root))` wrapper required to
/// reach the `&TreeSnapshot`-keyed entry point. The `Arc::clone` at `chain.len() == 1` (target ==
/// anchor) is intrinsic: the return type owns an `Arc<DirSnapshot>`, and at depth 1 we can either
/// clone the input or have a degenerate path that consumes it. Cloning keeps the helper
/// non-consuming.
///
/// `anchor` is supplied by the caller — typically `profile.resource` for navigation off
/// [`crate::Profile::current`]. The snapshot itself carries no engine identity; navigation is
/// `&Tree`-driven from `anchor` down to `target`.
///
/// `None` arms:
/// - `target` is outside `anchor`'s tree subtree (ancestor walk bottoms out before reaching
///   `anchor`).
/// - The chain crosses a [`ChildEntry::Leaf`] (snapshot identity flip at an intermediate segment).
/// - The chain crosses a [`DirChild::Uncovered`] intermediate.
/// - Any segment fails to resolve via [`Tree::name`] (slot reaped).
#[must_use]
pub fn subtree_at_dir(
    root: &Arc<DirSnapshot>,
    anchor: ResourceId,
    target: ResourceId,
    tree: &Tree,
) -> Option<Arc<DirSnapshot>> {
    let chain = ancestor_chain(target, anchor, tree)?;

    // Descend from `root` by following segment names. `chain[0] == anchor` matches `root` already,
    // so we start at `chain[1]`. Any non-covered intermediate (Leaf, Uncovered Dir, or missing
    // entry) yields None via `lookup_covered_dir`'s unified gate.
    let mut current: Arc<DirSnapshot> = Arc::clone(root);
    for &id in chain.iter().skip(1) {
        let name = tree.name(id)?;
        let next = current.lookup_covered_dir(name)?;
        current = Arc::clone(next);
    }
    Some(current)
}

/// Walk `tree.parent` from `target` up to `anchor` and return the inclusive chain `[anchor, mid_1,
/// ..., target]`. Returns `None` when `target` is not in `anchor`'s subtree (the parent walk
/// bottoms out before reaching `anchor`).
///
/// Sole helper for navigation that needs to follow the path from an anchor down to one of its
/// descendants — [`subtree_at_dir`] consumes it as the descent guide for snapshot navigation;
/// [`splice`] consumes it to know which intermediate `DirSnapshot`s need rebuilding.
///
/// Termination relies on the [`Tree`] acyclicity invariant: each `tree.parent` step strictly
/// ascends, so the walk reaches `anchor` or bottoms out at a root (`None`) in at most
/// `depth(target)` steps. The loop is intentionally not depth-bounded here — the bound is a
/// property of [`Tree`] construction, not a guard this hot path re-checks.
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
/// On `Spliced` the caller adopts the carried view as the new current; on `CrossedUncovered` the
/// caller surfaces a diagnostic and leaves its own prior handle untouched. `splice` consumes the
/// caller's `prior` Arc, but `Profile.current`'s own handle is independent of that (the engine always
/// clones before calling), so the caller already owns the unchanged prior view across the breach.
#[derive(Debug)]
pub enum SpliceResult {
    /// Splice succeeded. The new view integrates `replacement` at `target` (or is the trivial
    /// wholesale-replace when prior was `None` / target-equals-anchor).
    Spliced(TreeSnapshot),
    /// Splice could not navigate from the prior anchor down to `target`. The payload demuxes the
    /// three structural failure modes: [`SpliceFailureCause::TargetOutsideAnchorSubtree`] (target
    /// outside anchor's tree subtree), [`SpliceFailureCause::SlotReapedMidGraft`] (interior slot's
    /// generation moved mid-graft), and [`SpliceFailureCause::IntermediateUncovered`] (path crossed
    /// a [`DirChild::Uncovered`], a missing entry, or a `Leaf` at an interior segment).
    ///
    /// The caller leaves its prior view in place and emits
    /// [`crate::Diagnostic::SpliceCrossedUncovered`] carrying the cause so the contract violation
    /// is visible in operator logs with the failure mode pre-classified.
    ///
    /// Engine contract is "graft only into observed subtrees". After the walker-race fix, only
    /// [`SpliceFailureCause::IntermediateUncovered`] remains reachable through legitimate filesystem
    /// state, and only via cross-filesystem boundaries. The other two are v1-unreachable.
    CrossedUncovered(SpliceFailureCause),
}

/// Tree-zipper splice that replaces the subtree at `target` within a Dir-shaped prior.
///
/// Produces a new [`TreeSnapshot`] whose subtree at `target` equals `replacement`, sharing all
/// off-path subtrees with `prior` via `Arc`. Rebuilds at most `depth(target)` `DirSnapshot`s along
/// the path-to-anchor.
///
/// `anchor` is the Profile's anchor `ResourceId` — the engine-side identity the caller knows
/// `prior` is rooted at. It drives `ancestor_chain`'s walk and isn't compared against any snapshot
/// field; the wire payload is path-and-content only.
///
/// **File-anchored Profiles never call this helper.** Their `Profile.current` is
/// `TreeSnapshot::File(leaf)`, integrated by an inline write at the relevant `dispatch_*_ok`. The
/// typed [`crate::ProbeRequest`] contract guarantees File-anchored Profiles emit `AnchorFile`
/// requests whose `AnchorOk(LeafEntry)` payloads never reach `graft` / `splice`; the Dir-only
/// signature here is the engine-side half of that contract.
///
/// Returns [`SpliceResult::Spliced`] with `TreeSnapshot::Dir(replacement)` (Arc-cheap) for the
/// trivial cases:
/// - `prior == None` (first graft).
/// - `target == anchor` and the hashes differ (new root).
///
/// Returns [`SpliceResult::Spliced`] with `TreeSnapshot::Dir(prior)` (no allocation) when:
/// - `target == anchor` and `dir_hash` matches (G7-trivial).
/// - The recursive splice short-circuited at every level via `dir_hash` equality (G7 propagation).
///
/// Returns [`SpliceResult::CrossedUncovered`] when the engine's "graft only into observed subtrees"
/// contract is violated. The carried [`SpliceFailureCause`] demuxes the three structural triggers:
/// - [`SpliceFailureCause::TargetOutsideAnchorSubtree`] — parent walk bottoms out before reaching
///   `anchor`.
/// - [`SpliceFailureCause::SlotReapedMidGraft`] — an interior segment's slot was reaped mid-graft
///   (`Tree::name` ⇒ `None`).
/// - [`SpliceFailureCause::IntermediateUncovered`] — the path from anchor to target crosses a
///   [`DirChild::Uncovered`] intermediate (snapshot coverage gap), a missing entry, or a `Leaf` at
///   an interior segment.
///
/// After the walker-race fix, only the cross-fs subset of
/// [`SpliceFailureCause::IntermediateUncovered`] is reachable through legitimate filesystem state;
/// the other two remain v1-unreachable. The caller's prior handle stays alive across the breach
/// (it's an independent Arc clone), so no integration occurs; the caller emits a Diagnostic so the
/// contract breach is observable.
#[must_use]
pub fn splice(
    prior: Option<Arc<DirSnapshot>>,
    anchor: ResourceId,
    target: ResourceId,
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> SpliceResult {
    match prior {
        None => SpliceResult::Spliced(TreeSnapshot::Dir(replacement)),
        Some(root) => splice_dir_prior(root, anchor, target, replacement, tree),
    }
}

/// Dir-prior splice path. Extracted so [`splice`]'s top-level match reads as one branch per `prior`
/// shape rather than mixing Dir-only flow into the dispatcher.
fn splice_dir_prior(
    root: Arc<DirSnapshot>,
    anchor: ResourceId,
    target: ResourceId,
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> SpliceResult {
    if target == anchor {
        if root.dir_hash() == replacement.dir_hash() {
            return SpliceResult::Spliced(TreeSnapshot::Dir(root));
        }
        return SpliceResult::Spliced(TreeSnapshot::Dir(replacement));
    }

    let Some(chain) = ancestor_chain(target, anchor, tree) else {
        // Target outside anchor's tree subtree. The caller keeps its prior view (independent Arc
        // clone) and surfaces the contract violation. Wholesale-replacing with `replacement` would
        // leave `Profile.current` rooted at `target` (not anchor) and violate the snapshot
        // navigation invariants.
        return SpliceResult::CrossedUncovered(SpliceFailureCause::TargetOutsideAnchorSubtree);
    };

    // chain is [anchor, mid_1, ..., mid_k, target]; we already consumed the anchor as `root`, so
    // descend with chain[1..]. The recursive helper threads the typed [`SpliceFailureCause`] up via
    // `?` from whichever interior site fails; the failure-site discrimination (slot reaped vs.
    // uncovered intermediate) lives at the recursion leaves rather than being reconstructed at this
    // dispatcher.
    match splice_dir(&root, &chain[1..], replacement, tree) {
        Ok(new_root) => SpliceResult::Spliced(TreeSnapshot::Dir(new_root)),
        Err(cause) => SpliceResult::CrossedUncovered(cause),
    }
}

/// Recursive splice helper. Returns `Ok(arc)` on a successful per-level rebuild (or G7
/// short-circuit); returns `Err(cause)` when navigation can't proceed at this level. The two
/// failure modes:
/// - [`SpliceFailureCause::SlotReapedMidGraft`] — `tree.name(next_id)` returned `None` for an
///   interior segment; the slot's generation moved between burst start and graft commit.
/// - [`SpliceFailureCause::IntermediateUncovered`] — the prior snapshot's `lookup_covered_dir(name)`
///   returned `None` (entry is absent, a `Leaf`, or stored as `DirChild::Uncovered`).
///
/// The typed error threads through the recursive call via `?`, so deeper failures surface at the
/// dispatcher with the originating site's classification intact. [`splice_dir_prior`] wraps the
/// result into [`SpliceResult::CrossedUncovered`] preserving the prior unchanged.
///
/// **Performance.** Each ancestor on the path-to-anchor whose child hash changed clones its
/// `BTreeMap<CompactString, ChildEntry>` to install one updated slot — `O(Σ fanout_per_ancestor)`
/// per splice, plus one `compute_dir_hash` fold per rebuilt ancestor. Worst-case for a 1000-entry
/// directory at depth-3 splice: ~3000 `ChildEntry` clones plus the BTreeMap's interior-node allocs
/// (4-8 entries per node ⇒ ~150 node allocs per level). The per-level G7 short-circuit prunes
/// ancestors above the deepest observable change, so the realistic cost is over the *changed*
/// prefix of the spine, not the full path.
fn splice_dir(
    prior: &Arc<DirSnapshot>,
    rest: &[ResourceId],
    replacement: Arc<DirSnapshot>,
    tree: &Tree,
) -> Result<Arc<DirSnapshot>, SpliceFailureCause> {
    let Some((&next_id, deeper)) = rest.split_first() else {
        // We're at target. G7-leaf: hash-equal ⇒ keep prior Arc; the splice is a no-op
        // observationally.
        if prior.dir_hash() == replacement.dir_hash() {
            return Ok(Arc::clone(prior));
        }
        return Ok(replacement);
    };
    // Slot reaped mid-graft. Engine contract says this can't happen for an observed subtree;
    // surface as SlotReapedMidGraft so operators can demux it from the legitimately-reachable
    // IntermediateUncovered.
    let name = tree
        .name(next_id)
        .ok_or(SpliceFailureCause::SlotReapedMidGraft)?;
    // Path crossed an uncovered branch (DirChild::Uncovered), missing entry, or a Leaf at this
    // interior segment. We don't synthesise empty intermediates — that would lie to `dir_hash`.
    // Surface as IntermediateUncovered; the engine keeps its prior view and converges on the next
    // probe.
    let pc: Arc<DirSnapshot> = Arc::clone(
        prior
            .lookup_covered_dir(name)
            .ok_or(SpliceFailureCause::IntermediateUncovered)?,
    );
    let new_child = splice_dir(&pc, deeper, replacement, tree)?;

    // G7 per-level: propagate `Arc::clone(prior)` up the spine when the rebuilt child's `dir_hash`
    // matches — nothing observable changed at this level.
    if new_child.dir_hash() == pc.dir_hash() {
        return Ok(Arc::clone(prior));
    }

    // `prior.lookup_covered_dir(name)` above proved the key exists, so we mutate the cloned entry in
    // place — `BTreeMap::insert(K, V)` would drop a fresh `CompactString::new(name)` on the floor
    // here (insert updates the value but never the key when the key was already present in the map).
    let mut new_entries = prior.entries.clone();
    *new_entries.get_mut(name).expect(
        "entry at `name` is present in `new_entries` because the prior \
         `lookup_covered_dir(name)` resolution succeeded above",
    ) = ChildEntry::Dir(DirChild::Covered(new_child));
    // Preserve prior's `captured_with` on the rebuilt parent: it is conceptually "still the same
    // observation as prior, with one child subtree spliced in", and `captured_with` is constant
    // within a Profile by construction.
    Ok(Arc::new(DirSnapshot::new(
        prior.root_meta,
        prior.captured_with,
        new_entries,
    )))
}

// ---------------------------------------------------------------------------
// diff_tree
// ---------------------------------------------------------------------------

/// [`Diff`] over two parallel [`DirSnapshot`] trees rooted at the same anchor / target.
///
/// Walks in lock-step, pruning equal-`dir_hash` subtrees. Each output list is in **stable depth-first
/// pre-order**: `BTreeMap` iteration (lexicographic within a directory) then fixed recursion, so a
/// directory entry is immediately followed by its whole subtree, before the directory's lexical
/// siblings. Deterministic and replay-stable (`Diff: PartialEq`) but **not** a flat lexicographic
/// sort of `parent/child` paths: the two diverge whenever a sibling sorts between a directory and its
/// children (`d.txt` between `d` and `d/file`, since `/` = 0x2F sorts after `.` = 0x2E). `renamed` is
/// in baseline-side traversal order, not sorted by `from` (see `pair_renames`).
///
/// Cross-level rename detection: the per-level walk collects deltas keyed by `fs_id`; a
/// `pair_renames` post-pass then pairs `Created` and `Deleted` across the whole walk into `Renamed`.
///
/// This is the typed Dir/Dir surface — its production consumer is `specter_engine::reconcile::graft`
/// (Dir prior + Dir response by construction: no variant dispatch, no wrapper-`Arc` clone). The
/// [`TreeSnapshot`]-keyed [`diff_tree`] is the other production surface (anchor-shape dispatch,
/// consumed by the engine's `emit_effects`); both are real paths, not test-only.
#[must_use]
pub fn diff_dir_pair(baseline: &DirSnapshot, current: &DirSnapshot) -> Diff {
    let mut out = Diff::default();
    if baseline.dir_hash() == current.dir_hash() {
        return out; // O(1) prune at root
    }
    let mut staged_created: Vec<StagedEntry> = Vec::new();
    let mut staged_deleted: Vec<StagedEntry> = Vec::new();
    collect_dir_pair(
        baseline,
        current,
        "",
        &mut out.modified,
        &mut staged_created,
        &mut staged_deleted,
    );
    pair_renames(staged_created, staged_deleted, &mut out);
    out
}

/// [`Diff`] over two parallel [`TreeSnapshot`] trees.
///
/// Anchor-shape-dispatching surface: matches the [`TreeSnapshot`] variant pair to [`diff_dir_pair`]
/// (Dir/Dir) or the private File/File walker. The production consumer is the engine's
/// `emit_effects` (diffs owned `baseline()` / `current()` [`TreeSnapshot`]s on the driver thread);
/// test fixtures also use it to diff over both anchor shapes. The typed Dir/Dir hot path
/// (`specter_engine::reconcile::graft`) calls [`diff_dir_pair`] directly to skip the variant match.
/// Per-list ordering is [`diff_dir_pair`]'s — stable depth-first pre-order.
#[must_use]
pub fn diff_tree(baseline: &TreeSnapshot, current: &TreeSnapshot) -> Diff {
    match (baseline, current) {
        (TreeSnapshot::Dir(b), TreeSnapshot::Dir(c)) => diff_dir_pair(b, c),
        (TreeSnapshot::File(b), TreeSnapshot::File(c)) => {
            let mut out = Diff::default();
            diff_file_pair(b, c, &mut out);
            out
        }
        // Kind mismatch (File vs Dir) at the anchor: structurally unreachable in v1 — Profile kind
        // is fixed at attach time and a kind change at the anchor surfaces as Vanished, not as a
        // diff. The empty Diff is the safe release behaviour; the debug_assert flags any future
        // contract drift in tests.
        _ => {
            debug_assert!(
                false,
                "diff_tree: File↔Dir mismatch at the anchor is unreachable in v1; \
                 anchor kind changes are reported via Vanished, not diff",
            );
            Diff::default()
        }
    }
}

#[derive(Clone, Debug)]
struct StagedEntry {
    rel: CompactString,
    kind: EntryKind,
    fs_id: FsIdentity,
    /// When `false`, `pair_renames` skips this entry's `fs_id` from rename matching and routes it
    /// directly to `out.created` / `out.deleted`. Used for parent slots whose identity has flipped
    /// (kind change at the same name, Dir replaced at a different inode): such slots represent
    /// observably-different entities and are not rename candidates, even when their inodes
    /// coincide. Descendants of these slots remain eligible — genuine moves into / out of the slot
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
            if p.fs_id != n.fs_id {
                // Same name, different kernel identity ⇒ delete-then-create. Stage as
                // pair_eligible: each side may legitimately pair with a cross-level entry sharing
                // its `fs_id` (the user moved the prior file out and a different one in).
                staged_deleted.push(StagedEntry {
                    rel: rel.clone(),
                    kind: p.kind,
                    fs_id: p.fs_id,
                    pair_eligible: true,
                });
                staged_created.push(StagedEntry {
                    rel,
                    kind: n.kind,
                    fs_id: n.fs_id,
                    pair_eligible: true,
                });
            } else if p.leaf_hash() != n.leaf_hash() {
                modified.push(EntryRef {
                    segment: rel,
                    kind: n.kind,
                    fs_id: n.fs_id,
                });
            }
        }
        (ChildEntry::Dir(p), ChildEntry::Dir(n)) => {
            if p.fs_id() == n.fs_id() {
                match (p, n) {
                    (DirChild::Covered(ps), DirChild::Covered(ns)) => {
                        if ps.dir_hash() != ns.dir_hash() {
                            collect_dir_pair(
                                ps,
                                ns,
                                &rel,
                                modified,
                                staged_created,
                                staged_deleted,
                            );
                        }
                    }
                    (DirChild::Uncovered(_), DirChild::Uncovered(_)) => {
                        // Both sides uncovered: no observation, no delta.
                    }
                    (DirChild::Covered(_), DirChild::Uncovered(_))
                    | (DirChild::Uncovered(_), DirChild::Covered(_)) => {
                        // Same-fs_id coverage flip: v1-unreachable. The walker's Uncovered gates
                        // (`!recursive`, `max_depth`, cross-fs) are `config_hash`-frozen per
                        // Profile or change `fs_id`, so the outer `p.fs_id() == n.fs_id()` guard
                        // would already have failed (reachability argument unchanged).
                        //
                        // Degrade, don't abort: a panic here stalls the single-threaded engine driver
                        // for a compounding routing breach. Release stages the slot ineligible
                        // (Deleted + Created, never a Rename) and recurses both sides, surfacing the
                        // dropped coverage delta rather than silently leaking Tree watches.
                        debug_assert!(
                            false,
                            "diff_same_name: same-fs_id (Covered, Uncovered) is \
                             v1-unreachable — Uncovered gates are config-frozen \
                             or change fs_id",
                        );
                        staged_deleted.push(StagedEntry {
                            rel: rel.clone(),
                            kind: EntryKind::Dir,
                            fs_id: p.fs_id(),
                            pair_eligible: false,
                        });
                        staged_created.push(StagedEntry {
                            rel: rel.clone(),
                            kind: EntryKind::Dir,
                            fs_id: n.fs_id(),
                            pair_eligible: false,
                        });
                        stage_descendants_deleted(&rel, pc, staged_deleted);
                        stage_descendants_created(&rel, nc, staged_created);
                    }
                }
            } else {
                // Same-name dir-replace at a different kernel identity: parent slot represents a
                // different entity. Stage parent ineligible (it must surface as Deleted + Created,
                // never collapse to a same-rel "Rename"), and recurse both subtrees so descendants
                // surface as Deleted/Created or pair as cross-level Renames against the rest of the
                // walk.
                staged_deleted.push(StagedEntry {
                    rel: rel.clone(),
                    kind: EntryKind::Dir,
                    fs_id: p.fs_id(),
                    pair_eligible: false,
                });
                staged_created.push(StagedEntry {
                    rel: rel.clone(),
                    kind: EntryKind::Dir,
                    fs_id: n.fs_id(),
                    pair_eligible: false,
                });
                stage_descendants_deleted(&rel, pc, staged_deleted);
                stage_descendants_created(&rel, nc, staged_created);
            }
        }
        // Kind change at same name (Leaf↔Dir): the slot represents logically-different entities
        // across the two snapshots. Stage the parent as ineligible (so pair_renames doesn't try to
        // collapse it into a nonsensical same-name "Rename" when the kernel reuses the inode across
        // the kind flip) and recurse the Dir side(s) so descendants surface — either as
        // Deleted/Created or as cross-level Renames.
        _ => {
            staged_deleted.push(StagedEntry {
                rel: rel.clone(),
                kind: pc.kind(),
                fs_id: pc.fs_id(),
                pair_eligible: false,
            });
            staged_created.push(StagedEntry {
                rel: rel.clone(),
                kind: nc.kind(),
                fs_id: nc.fs_id(),
                pair_eligible: false,
            });
            stage_descendants_deleted(&rel, pc, staged_deleted);
            stage_descendants_created(&rel, nc, staged_created);
        }
    }
}

/// Stage every descendant of `parent` (if `parent` is a covered Dir) as Deleted, with `parent_rel`
/// as the rel-prefix. Called from `diff_same_name`'s ineligible-parent paths (kind change,
/// Dir-replace at different inode). Leaves and uncovered Dirs are no-ops.
fn stage_descendants_deleted(parent_rel: &str, parent: &ChildEntry, staged: &mut Vec<StagedEntry>) {
    if let ChildEntry::Dir(DirChild::Covered(sub)) = parent {
        for (cname, cchild) in &sub.entries {
            stage_deleted(cname, cchild, parent_rel, staged);
        }
    }
}

/// Symmetric counterpart of [`stage_descendants_deleted`].
fn stage_descendants_created(parent_rel: &str, parent: &ChildEntry, staged: &mut Vec<StagedEntry>) {
    if let ChildEntry::Dir(DirChild::Covered(sub)) = parent {
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
        fs_id: pc.fs_id(),
        pair_eligible: true,
    });
    // For Dir deletions, recurse to emit each descendant as Deleted. Output is a flat Diff for the
    // Effect API; it doesn't care about reap order. The recursive walk preserves lex within each
    // level.
    if let ChildEntry::Dir(DirChild::Covered(sub)) = pc {
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
        fs_id: nc.fs_id(),
        pair_eligible: true,
    });
    if let ChildEntry::Dir(DirChild::Covered(sub)) = nc {
        for (cname, cchild) in &sub.entries {
            stage_created(cname, cchild, &rel, staged);
        }
    }
}

// ---------------------------------------------------------------------------
// Diff single-snapshot constructors
// ---------------------------------------------------------------------------
//
// `all_created` / `all_deleted` are the snapshot ↔ empty transformations. They exist as named
// methods (rather than `diff_tree(empty, snap)` calls) to avoid allocating an empty `DirSnapshot`
// per invocation; both reuse the same depth-first pre-order staging recursion as `diff_tree`.
//
// **Asymmetry vs `modified` / `renamed`.** The four `Diff` categories do not have symmetric
// single-snapshot reductions:
//
// - `created` / `deleted`: snapshot ↔ empty has a natural meaning (every entry appeared /
//   disappeared relative to the empty state).
// - `modified`: pairs entries across two non-empty snapshots by identity (same `(inode, device)`,
//   different `leaf_hash`). One snapshot alone cannot express "modified relative to what" — there
//   is no implicit prior identity.
// - `renamed`: pairs entries across two non-empty snapshots by both `(inode, device)` and
//   `segment`. Requires both endpoints.
//
// So `all_modified` / `all_renamed` have no semantically grounded definition, only an arbitrary one
// (e.g. "every entry as Modified relative to itself") that no engine path would need. They are
// omitted by design.

impl Diff {
    /// Construct a [`Diff`] where every entry of `snap` (recursively, into covered subtrees)
    /// appears as a `Created` entry, in stable depth-first pre-order (the same order
    /// [`diff_dir_pair`] emits).
    ///
    /// Equivalent to `diff_tree(empty_dirsnapshot, snap)` without the empty `DirSnapshot`
    /// allocation. Sole intended caller is `specter_engine::reconcile::graft`'s first-graft path
    /// (`Profile.current == None` ⇒ every entry of the response is new from the engine's
    /// perspective).
    ///
    /// `modified` / `renamed` are empty by construction — there is no prior snapshot to pair
    /// entries against; see the module-level asymmetry rationale.
    #[must_use]
    pub fn all_created(snap: &DirSnapshot) -> Self {
        let mut staged: Vec<StagedEntry> = Vec::new();
        for (name, child) in &snap.entries {
            stage_created(name, child, "", &mut staged);
        }
        let mut out = Self::default();
        out.created.reserve(staged.len());
        for s in staged {
            out.created.push(EntryRef {
                segment: s.rel,
                kind: s.kind,
                fs_id: s.fs_id,
            });
        }
        out
    }

    /// Symmetric counterpart of [`Diff::all_created`]: every entry of `snap` appears as a `Deleted`
    /// entry, in stable depth-first pre-order. Used by
    /// `specter_engine::Engine::release_descendant_claim` for whole-snapshot teardown.
    ///
    /// See [`Diff::all_created`] for the modified/renamed asymmetry rationale.
    #[must_use]
    pub fn all_deleted(snap: &DirSnapshot) -> Self {
        let mut staged: Vec<StagedEntry> = Vec::new();
        for (name, child) in &snap.entries {
            stage_deleted(name, child, "", &mut staged);
        }
        let mut out = Self::default();
        out.deleted.reserve(staged.len());
        for s in staged {
            out.deleted.push(EntryRef {
                segment: s.rel,
                kind: s.kind,
                fs_id: s.fs_id,
            });
        }
        out
    }
}

/// Pair Created/Deleted entries by `fs_id` to recover Renames.
///
/// The index uses `BTreeMap::insert` semantics, so when an `fs_id` collides (the pathological
/// hardlink case of multiple Created at the same inode) the *last* index wins. The `paired` set
/// guarantees one Created can match at most one Deleted.
///
/// **Pairing rules.** A `(deleted, created)` pair becomes a `Rename` iff (1) both sides are
/// `pair_eligible`, (2) the `fs_id` matches, (3) the `kind` matches, and (4) the `rel` differs.
/// Same-`rel` candidates are structurally impossible for eligible entries (parent kind changes and
/// Dir-replace-at-different-fs_id stage their parents ineligible; other staging paths cannot
/// produce same-rel collisions in the global buffer) — pinned by the `debug_assert` below.
/// Cross-kind candidates arise from kernel inode reuse across unrelated operations and are not
/// renames; they fall through to Created+Deleted.
///
/// Output order: unpaired Created/Deleted are emitted in collection order (depth-first pre-order on
/// each side); Renamed entries are emitted in `staged_deleted`'s iteration order — the
/// baseline-side depth-first pre-order, not sorted by `from`.
///
/// The BTreeMap is keyed lookup-only (never iterated), so the canonical `FsIdentity` ord
/// (inode-first by declaration order) supersedes the pre-migration `(device, inode)` tuple ord with
/// no observable effect.
fn pair_renames(
    staged_created: Vec<StagedEntry>,
    staged_deleted: Vec<StagedEntry>,
    out: &mut Diff,
) {
    let mut by_key: BTreeMap<FsIdentity, usize> = BTreeMap::new();
    for (i, c) in staged_created.iter().enumerate() {
        if c.pair_eligible {
            by_key.insert(c.fs_id, i);
        }
    }
    let mut paired: BTreeSet<usize> = BTreeSet::new();
    let mut leftover_deleted: Vec<StagedEntry> = Vec::with_capacity(staged_deleted.len());

    for d in staged_deleted {
        if !d.pair_eligible {
            // Ineligible parent (kind change or Dir-replace): never a rename. Route to out.deleted
            // in lex order via the shared leftover queue.
            leftover_deleted.push(d);
            continue;
        }
        match by_key.get(&d.fs_id) {
            Some(&ci) if !paired.contains(&ci) => {
                let c = &staged_created[ci];
                debug_assert!(
                    c.rel != d.rel,
                    "staging invariant: eligible same-rel pairs should be \
                     reduced upstream (modified, dir-recursion, or marked \
                     ineligible) and never reach pair_renames",
                );
                if c.kind != d.kind {
                    // Cross-kind inode collision (kernel reuse across unrelated operations). Not a
                    // rename — let both sides surface as Created/Deleted.
                    leftover_deleted.push(d);
                    continue;
                }
                out.renamed.push(Rename {
                    from: EntryRef {
                        segment: d.rel,
                        kind: d.kind,
                        fs_id: d.fs_id,
                    },
                    to: EntryRef {
                        segment: c.rel.clone(),
                        kind: c.kind,
                        fs_id: c.fs_id,
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
                fs_id: c.fs_id,
            });
        }
    }
    for d in leftover_deleted {
        out.deleted.push(EntryRef {
            segment: d.rel,
            kind: d.kind,
            fs_id: d.fs_id,
        });
    }
}

fn diff_file_pair(b: &LeafEntry, c: &LeafEntry, out: &mut Diff) {
    if b.fs_id == c.fs_id {
        if b.leaf_hash() != c.leaf_hash() {
            out.modified.push(EntryRef {
                segment: CompactString::new(""),
                kind: c.kind,
                fs_id: c.fs_id,
            });
        }
    } else {
        // Kernel-identity change at the file Profile's anchor: same-segment kind/identity flip.
        // Emit Deleted + Created (no Rename: a file Profile sees its anchor as one fact, not a
        // moved name).
        out.deleted.push(EntryRef {
            segment: CompactString::new(""),
            kind: b.kind,
            fs_id: b.fs_id,
        });
        out.created.push(EntryRef {
            segment: CompactString::new(""),
            kind: c.kind,
            fs_id: c.fs_id,
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
