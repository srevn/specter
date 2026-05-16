//! `probe_anchor_file`, `probe_subtree`, `probe_descent` — pure-IO walkers.
//!
//! All three probes return [`ProbeOutcome`]; on success they emit either
//! a leaf observation (`AnchorOk(LeafEntry)`) or a recursive
//! `Arc<DirSnapshot>` tree (`SubtreeOk`). Kind mismatches collapse to
//! `Vanished` ("a file probe that finds a directory, or vice versa,
//! returns `Vanished`"). Errors at the *root* anchor map to
//! `Failed { errno }`; errors mid-walk on a *subtree* skip-and-continue
//! with `tracing::warn!` — `exclude` is the user-facing surface for
//! declaring expected-EACCES paths.
//!
//! Three controls live on the [`ProbeRequest::Subtree`] variant:
//! - `baseline_subtree`: the engine's last-known view of the target's
//!   subtree. Equal `(mtime, fs_id)` against the freshly `lstat`-ed
//!   directory ⇒ return `Arc::clone(prior)` (mtime-skip). The skip
//!   cascades into recursion via each child's
//!   `DirChild::Covered(arc)`, looked up by name through
//!   [`specter_core::DirSnapshot::lookup_covered_dir`].
//! - `force_walk`: a `BTreeSet<Arc<Path>>` of paths the walker must
//!   enumerate regardless of mtime — populated by the engine from
//!   kqueue-driven `dirty_resources`. The walker tests "is any forced path
//!   at-or-under this dir?" via `Path::starts_with`.
//! - `forced`: defensive bypass for max-settle force-fire. When `true`,
//!   every recursion frame enumerates regardless of `baseline_subtree` or
//!   `force_walk`.
//!
//! [`ProbeRequest::AnchorFile`] runs a single `lstat` (no controls — a
//! leaf has no descendants to skip). [`ProbeRequest::Descent`] hardcodes
//! a minimal override config (`recursive=false`, `hidden=true`, no
//! exclude/pattern, no `max_depth`) — the Profile's user-facing filters
//! would mask the very segment descent is searching for.
//!
//! Symlinks are never traversed (`symlink_metadata` ≡ `lstat`); they
//! appear as `EntryKind::Symlink` leaves when encountered as direct
//! children. v1 has no `follow_symlinks` opt-in. Cross-filesystem descent
//! is refused: subdir entries with a `dev` differing from the root anchor's
//! `dev` are emitted as `DirChild::Uncovered(fs_id)` (uncovered-by-mount).
//!
//! `exclude` and `pattern` are tested against the *cumulative relative path*
//! from the anchor (`subdir/file.c`, not just `file.c`). This matches the
//! engine's `coverage::covers`, keeping walker behaviour consistent with
//! the predicate the engine consumes.

use compact_str::CompactString;
use specter_core::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, FsIdentity, LeafEntry, ProbeOutcome, ScanConfig,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;

/// Recursion-invariant inputs shared across every frame of one subtree
/// probe. Built once at probe entry ([`probe_subtree`]) from the
/// `ProbeRequest::Subtree` payload, then threaded by reference into
/// [`snapshot_dir`], [`enumerate_dir`], and [`build_dir_child`].
/// Per-frame inputs (`path`, `baseline`, `depth`, `cmeta`, `name`) stay
/// as positional arguments to those callees.
///
/// Separating invariant from per-frame at the type level makes the
/// distinction structural: a reader at any call site sees `ctx`
/// (unchanging across the recursion) plus the dirent-scope inputs that
/// vary. The three methods name the walker's three coverage decisions:
/// [`should_recurse`](Self::should_recurse) (the
/// `Covered`/`Uncovered(fs_id)` gate at the dirent),
/// [`try_mtime_skip`](Self::try_mtime_skip) (the no-op-when-unchanged
/// primitive), and
/// [`has_forced_at_or_under`](Self::has_forced_at_or_under) (the
/// `force_walk` override that refuses skip).
///
/// `root_dev` is the *anchor*'s device, captured once in
/// [`probe_subtree`] from the top-level `lstat`. It is intentionally
/// distinct from each recursion frame's `root_meta.fs_id.device` — the
/// cross-filesystem gate refuses to descend whenever a child's device
/// differs from the anchor's, regardless of whether the recursion has
/// already crossed a sub-mount earlier.
///
/// `Copy + Clone` because the struct is three thin/fat pointers + two
/// `u64`s + one `bool`. Passing by reference at recursion frequency is
/// the convention here; the `Copy` derive is for the cheap "snapshot a
/// `ctx` value into a closure" cases that arise during evolution.
#[derive(Clone, Copy)]
struct WalkContext<'a> {
    anchor_path: &'a Path,
    config: &'a ScanConfig,
    force_walk: &'a BTreeSet<Arc<Path>>,
    forced: bool,
    captured_with: u64,
    root_dev: u64,
}

impl WalkContext<'_> {
    /// True iff a child directory at `depth_after_descent` on
    /// `child_dev` is in-scope for recursive descent. Folds three
    /// statically-knowable gates:
    /// - `self.config.recursive`
    /// - `depth_after_descent < max_depth.unwrap_or(u32::MAX)`
    /// - `child_dev == self.root_dev` (cross-filesystem refusal)
    ///
    /// Negation drives `DirChild::Uncovered(fs_id)` emission in
    /// [`build_dir_child`]. This is the only source of `Uncovered`
    /// emissions in the walker; transient I/O (raced unlink, EACCES,
    /// ENOTDIR mid-walk) surfaces as `Covered(empty_or_partial_arc)`
    /// instead, via [`enumerate_dir`]'s benign-empty contract.
    ///
    /// Mirrors the engine's `coverage::covers(profile, R)` predicate,
    /// modulo the cross-filesystem gate (which `covers` does not
    /// consult — the engine's `Tree` doesn't carry `device`).
    #[must_use]
    fn should_recurse(&self, depth_after_descent: u32, child_dev: u64) -> bool {
        self.config.recursive
            && depth_after_descent < self.config.max_depth.unwrap_or(u32::MAX)
            && child_dev == self.root_dev
    }

    /// Returns `Some(Arc::clone(baseline))` when the directory at
    /// `path` with freshly-`lstat`ed `root_meta` is observationally
    /// identical to the baseline subtree. Three predicates folded:
    /// - `!self.forced` (no defensive bypass), AND
    /// - no path in `self.force_walk` lies at-or-under `path`, AND
    /// - `baseline.root_meta == *root_meta` (mtime + inode + device).
    ///
    /// On `Some`, the caller short-circuits one whole recursion frame:
    /// zero readdir, zero leaf `lstat`, zero allocation. Composes
    /// recursively through each child's `DirChild::Covered(arc)` — an
    /// equal-mtime tree elides the entire walk.
    #[must_use]
    fn try_mtime_skip(
        &self,
        path: &Path,
        root_meta: &DirMeta,
        baseline: Option<&Arc<DirSnapshot>>,
    ) -> Option<Arc<DirSnapshot>> {
        if self.forced || self.has_forced_at_or_under(path) {
            return None;
        }
        let prior = baseline?;
        if prior.root_meta() != *root_meta {
            return None;
        }
        Some(Arc::clone(prior))
    }

    /// Returns `true` iff any path in `self.force_walk` is at-or-under
    /// `path`.
    ///
    /// Why `Path::starts_with` and not `==`: imagine `path = /a` and
    /// `self.force_walk = {/a/b/c}`. If we skip at `/a`, we never
    /// recurse into `/a/b/c` and miss the kernel's signal. Component-
    /// wise `starts_with` catches this — at `/a`,
    /// `(/a/b/c).starts_with(/a)` is true ⇒ refuse skip ⇒ enumerate
    /// children. At `/a/b`, the same path triggers the same refusal
    /// until we reach `/a/b/c`'s leaf, after which sibling subtrees
    /// are mtime-skip-eligible again.
    ///
    /// Byte-lex via `BTreeSet::range` would erroneously match `/ab`
    /// when probing `/a`; we need component-wise `Path::starts_with`.
    #[must_use]
    fn has_forced_at_or_under(&self, path: &Path) -> bool {
        self.force_walk.iter().any(|p| p.starts_with(path))
    }
}

/// Anchor-file probe. Single `lstat` against `target_path`.
///
/// Returns:
/// - `AnchorOk(LeafEntry)` for a regular file.
/// - `Vanished` when the path doesn't exist *or* is not a regular file
///   (kind mismatch — symlink, directory, FIFO, etc.).
/// - `Failed { errno }` for any other I/O error.
pub(super) fn probe_anchor_file(target_path: &Path) -> ProbeOutcome {
    let meta = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeOutcome::Vanished,
        Err(e) => {
            return ProbeOutcome::Failed {
                errno: e.raw_os_error().unwrap_or(libc::EIO),
            };
        }
    };
    if !meta.is_file() {
        return ProbeOutcome::Vanished;
    }
    // The `is_file` guard above upholds `from_metadata`'s non-directory
    // precondition; `entry_kind_from_file_type` resolves it to `File`.
    let leaf = LeafEntry::from_metadata(&meta);
    ProbeOutcome::AnchorOk(leaf)
}

/// Subtree probe. Recursive DFS walk against `target_path` honoring
/// `recursive`, `hidden`, `exclude`, `pattern`, and `max_depth`.
///
/// Each recursion frame may short-circuit via mtime-skip when:
/// - `forced == false`, AND
/// - no path in `force_walk` lies at-or-under the current directory, AND
/// - a baseline subtree is provided whose `root_meta` (mtime + inode +
///   device) equals the freshly-`lstat`ed directory.
///
/// On skip, the frame returns `Arc::clone(baseline)` — zero allocation,
/// zero readdir, zero leaf `lstat`. Otherwise it enumerates one level,
/// stamps a fresh `DirSnapshot`, and recurses for covered Dir children
/// (passing each child's prior subtree from the baseline so the skip
/// composes recursively).
///
/// Errors: root errors propagate (`NotFound → Vanished`, kind mismatch
/// → `Vanished`, anything else → `Failed { errno }`). Mid-walk
/// `read_dir` errors on a *subdirectory* (EACCES, EIO, ENOENT after a
/// raced delete, ENOTDIR after a kind-flip race) skip-and-continue with
/// `tracing::warn!`; the affected subtree becomes
/// `DirChild::Covered(empty_or_partial_arc)` — *covered-but-empty*.
/// The uncovered variant `DirChild::Uncovered(fs_id)` is reserved for
/// the three statically-knowable gates fronted by
/// [`WalkContext::should_recurse`] and applied in [`build_dir_child`]:
/// `!recursive`, `depth + 1 >= max_depth`, or `cmeta.dev() != root_dev`
/// (cross-filesystem). The walker never mints `Uncovered` for transient
/// I/O — the redundant per-recursion `lstat` that historically opened
/// that race surface no longer exists.
pub(super) fn probe_subtree(
    target_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    force_walk: &BTreeSet<Arc<Path>>,
    forced: bool,
) -> ProbeOutcome {
    let root_meta_raw = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeOutcome::Vanished,
        Err(e) => {
            return ProbeOutcome::Failed {
                errno: e.raw_os_error().unwrap_or(libc::EIO),
            };
        }
    };
    if !root_meta_raw.is_dir() {
        return ProbeOutcome::Vanished;
    }
    let root_meta = DirMeta::from_metadata(&root_meta_raw);
    let ctx = WalkContext {
        anchor_path: target_path,
        config,
        force_walk,
        forced,
        captured_with,
        root_dev: root_meta.fs_id().device(),
    };
    let arc = snapshot_dir(&ctx, target_path, root_meta, baseline, 0);
    ProbeOutcome::SubtreeOk(arc)
}

/// Descent prefix probe. Single-level enumeration of `target_path` with
/// a hardcoded override config: `recursive=false`, `hidden=true`, no
/// `exclude`, no `pattern`, no `max_depth`. The walker owns the override
/// config because the engine's user-facing filters would mask the very
/// segment descent is searching for; descent dispatch reads
/// `arc.entries.get(name)` directly and (for Profile descent) discards
/// the snapshot.
///
/// `captured_with` is stamped as `0` — descent dispatch never reads the
/// field (the snapshot is consumed by the engine and dropped before any
/// consumer compares hashes), so the value is observationally
/// irrelevant. Callers should not rely on a particular sentinel.
pub(super) fn probe_descent(target_path: &Path) -> ProbeOutcome {
    let cfg = ScanConfig::builder()
        .recursive(false)
        .hidden(true)
        .max_depth(None)
        .build();
    probe_subtree(target_path, &cfg, 0, None, &BTreeSet::new(), false)
}

/// Build one directory's snapshot frame. Shared by two callers:
/// 1. [`probe_subtree`], after the root `lstat` produces `root_meta`
///    from the freshly-`lstat`ed anchor.
/// 2. [`build_dir_child`], with a `cmeta`-derived `root_meta` for a
///    covered subdir dirent.
///
/// The mtime-skip primitive ([`WalkContext::try_mtime_skip`]) is the
/// same at every recursion depth: equal `root_meta` plus no
/// `force_walk` override plus `!forced` ⇒ return the baseline
/// `Arc::clone`. On non-skip, enumerate one level and stamp a fresh
/// `DirSnapshot`.
///
/// Infallible by construction. Any failure inside the recursive
/// [`enumerate_dir`] (raced unlink surfacing as `read_dir` `ENOENT`,
/// kind-flip surfacing as `ENOTDIR`, EACCES on the subdir's
/// `read_dir`, etc.) routes through the existing `Covered(empty_arc)`
/// contract — the same semantic pinned by
/// `probe_subtree_unreadable_subdir_emits_dir_child_some_empty`. As a
/// consequence, `DirChild::Uncovered(fs_id)` is reserved for the
/// static-config gates fronted by [`WalkContext::should_recurse`]
/// (`recursive=false`, `max_depth`, cross-fs) and never minted for
/// transient I/O.
#[must_use]
fn snapshot_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    root_meta: DirMeta,
    baseline: Option<&Arc<DirSnapshot>>,
    depth: u32,
) -> Arc<DirSnapshot> {
    if let Some(skipped) = ctx.try_mtime_skip(path, &root_meta, baseline) {
        return skipped;
    }
    let entries = enumerate_dir(ctx, path, baseline.map(Arc::as_ref), depth);
    Arc::new(DirSnapshot::new(root_meta, ctx.captured_with, entries))
}

/// Read one directory level, applying filters and recursing into covered
/// Dir children. Returns the constructed entries map.
///
/// Errors at this level are skip-and-continue. `read_dir` failure on
/// `path` (EACCES, EIO, ENOENT after a raced delete, ENOTDIR after a
/// kind-flip race, etc.) returns the already-accumulated `BTreeMap` —
/// empty when the failure was at the `read_dir` open boundary,
/// partially-populated if dirent iteration errored mid-walk after some
/// entries had already been folded in. The caller ([`snapshot_dir`])
/// wraps the result in `Arc::new(DirSnapshot::new(…))`, so the parent
/// emits `DirChild::Covered(empty_or_partial_arc)` — covered-but-empty,
/// distinct from the uncovered variant `DirChild::Uncovered(fs_id)`
/// reserved for the static-config gates in [`build_dir_child`]
/// (`recursive=false`, beyond `max_depth`, cross-filesystem boundary).
fn enumerate_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    baseline: Option<&DirSnapshot>,
    depth: u32,
) -> BTreeMap<CompactString, ChildEntry> {
    let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();

    let read_dir = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return entries,
        Err(e) => {
            tracing::warn!(
                anchor = ?ctx.anchor_path,
                ?path,
                ?e,
                "probe_subtree readdir failed; skipping subtree"
            );
            return entries;
        }
    };

    for dirent_result in read_dir {
        let dirent = match dirent_result {
            Ok(d) => d,
            Err(e) => {
                tracing::trace!(?path, ?e, "probe_subtree dirent error; skipping");
                continue;
            }
        };
        let child_path = dirent.path();
        let name_os = dirent.file_name();
        let Some(name_str) = name_os.to_str() else {
            tracing::trace!(?child_path, "probe_subtree non-UTF-8 filename; skipping");
            continue;
        };
        if !ctx.config.hidden && name_str.starts_with('.') {
            continue;
        }
        let Ok(rel) = child_path.strip_prefix(ctx.anchor_path) else {
            tracing::trace!(?child_path, "probe_subtree strip_prefix failed; skipping");
            continue;
        };
        if ctx.config.exclude.iter().any(|g| g.matches_path(rel)) {
            continue;
        }
        let Ok(cmeta) = std::fs::symlink_metadata(&child_path) else {
            continue;
        };
        let is_dir = cmeta.file_type().is_dir();

        // Pattern semantics: directories are always covered (we descend
        // through them); files (and symlinks/other) are gated by the
        // pattern.
        if let Some(pat) = &ctx.config.pattern
            && !is_dir
            && !pat.matches_path(rel)
        {
            continue;
        }

        let key = CompactString::new(name_str);
        let child_entry = if is_dir {
            build_dir_child(ctx, &child_path, baseline, depth, &cmeta, name_str)
        } else {
            build_leaf_child(&cmeta, name_str, baseline)
        };

        entries.insert(key, child_entry);
    }

    entries
}

/// Build a `ChildEntry::Dir` for one directory dirent. Recurses via
/// [`snapshot_dir`] when the entry is in-scope per
/// [`WalkContext::should_recurse`] (recursive walk, within `max_depth`,
/// same filesystem); emits `DirChild::Uncovered(fs_id)` otherwise.
///
/// `Uncovered(fs_id)` is emitted iff [`WalkContext::should_recurse`]
/// returns `false`. Every other path enters [`snapshot_dir`], whose
/// infallible return is wrapped unconditionally in
/// `DirChild::Covered(arc)`. Transient I/O failures inside the
/// recursive walk surface as `DirChild::Covered(empty_or_partial_arc)`
/// via [`enumerate_dir`]'s benign-empty contract, never as
/// `Uncovered`.
fn build_dir_child(
    ctx: &WalkContext<'_>,
    child_path: &Path,
    baseline: Option<&DirSnapshot>,
    depth: u32,
    cmeta: &std::fs::Metadata,
    name: &str,
) -> ChildEntry {
    let fs_id = FsIdentity::from_metadata(cmeta);
    if !ctx.should_recurse(depth + 1, cmeta.dev()) {
        // Uncovered branch: not recursive, beyond max_depth, or cross-fs.
        // Walker stores the entry but does not recurse.
        return ChildEntry::Dir(DirChild::Uncovered(fs_id));
    }
    // Build the subdir's DirMeta from the caller-held `cmeta`: a second
    // `symlink_metadata(child_path)` would be redundant in the happy
    // path and a race surface in the unhappy one (concurrent unlink /
    // kind-flip could make it disagree with the is_dir just checked).
    let root_meta = DirMeta::from_metadata(cmeta);
    // Pull the child's prior subtree from baseline so mtime-skip composes
    // recursively. BTreeMap key match by string segment is the snapshot's
    // native lookup; `lookup_covered_dir` collapses the "Dir entry + covered"
    // gate into one named operation.
    let child_baseline = baseline.and_then(|b| b.lookup_covered_dir(name));
    let arc = snapshot_dir(ctx, child_path, root_meta, child_baseline, depth + 1);
    ChildEntry::Dir(DirChild::Covered(arc))
}

/// Build a `ChildEntry::Leaf` for one non-directory dirent. Inherits
/// the baseline leaf's `leaf_hash` when the prior entry's identity
/// matches — re-enumeration of an unchanged leaf elides the SipHash24
/// fold the walker would otherwise pay. Identity mismatch recomputes
/// the hash from the freshly-`lstat`ed fields. Kind, size, mtime, and
/// `fs_id` all derive from the one `cmeta`, so the leaf is atomic by
/// construction.
///
/// The caller's `is_dir` dispatch in [`enumerate_dir`] upholds
/// `LeafEntry::from_metadata`'s non-directory precondition (dirents
/// with `is_dir` route to [`build_dir_child`], never here).
fn build_leaf_child(
    cmeta: &std::fs::Metadata,
    name: &str,
    baseline: Option<&DirSnapshot>,
) -> ChildEntry {
    let baseline_leaf = baseline.and_then(|b| b.lookup_leaf(name));
    ChildEntry::Leaf(LeafEntry::from_metadata_or_inherit(cmeta, baseline_leaf))
}
