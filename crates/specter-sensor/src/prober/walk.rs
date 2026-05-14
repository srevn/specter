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
//!   cascades into recursion via `entries[name].subtree`.
//! - `force_walk`: a `BTreeSet<PathBuf>` of paths the walker must
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
//! `dev` are emitted as `DirChild { subtree: None }` (uncovered-by-mount).
//!
//! `exclude` and `pattern` are tested against the *cumulative relative path*
//! from the anchor (`subdir/file.c`, not just `file.c`). This matches the
//! engine's `coverage::covers`, keeping walker behaviour consistent with
//! the predicate the engine consumes.

use compact_str::CompactString;
use specter_core::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, EntryKind, FsIdentity, LeafEntry, ProbeOutcome,
    ResourceId, ScanConfig,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

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
    let leaf = LeafEntry::new(
        EntryKind::File,
        meta.len(),
        meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        FsIdentity {
            inode: meta.ino(),
            device: meta.dev(),
        },
    );
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
/// zero readdir, zero leaf `lstat`s. Otherwise it enumerates one level,
/// stamps a fresh `DirSnapshot`, and recurses for covered Dir children
/// (passing each child's prior subtree from the baseline so the skip
/// composes recursively).
///
/// Errors: root errors propagate (`NotFound → Vanished`, kind mismatch
/// → `Vanished`, anything else → `Failed { errno }`). Mid-walk
/// `read_dir` errors on a *subdirectory* (EACCES, EIO, ENOENT after a
/// raced delete) skip-and-continue with `tracing::warn!`; the affected
/// subtree becomes `DirChild { subtree: Some(empty_or_partial_arc) }` —
/// *covered-but-empty*. The uncovered sentinel
/// `DirChild { subtree: None }` is reserved for `recursive=false`,
/// beyond `max_depth`, cross-filesystem (`dev` differs from
/// `root_dev`), and mid-walk `lstat`/kind-flip failures on the subdir
/// itself.
pub(super) fn probe_subtree(
    target_path: &Path,
    target_resource: ResourceId,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    force_walk: &BTreeSet<PathBuf>,
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
    let root_meta = DirMeta {
        mtime: root_meta_raw.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        fs_id: FsIdentity {
            inode: root_meta_raw.ino(),
            device: root_meta_raw.dev(),
        },
    };

    // Top-level mtime-skip. Bypassed when forced, or when any forced path
    // lies under target_path (we'd elide visiting it).
    if !forced
        && !any_forced_under(target_path, force_walk)
        && let Some(prior) = baseline
        && prior.root_meta == root_meta
    {
        return ProbeOutcome::SubtreeOk(Arc::clone(prior));
    }

    let entries = enumerate_dir(
        target_path,
        target_path,
        config,
        captured_with,
        baseline.map(Arc::as_ref),
        force_walk,
        forced,
        0,
        root_meta.fs_id.device,
    );
    ProbeOutcome::SubtreeOk(Arc::new(DirSnapshot::new(
        target_resource,
        root_meta,
        captured_with,
        entries,
    )))
}

/// Descent prefix probe. Single-level enumeration of `target_path` with
/// a hardcoded override config: `recursive=false`, `hidden=true`, no
/// `exclude`, no `pattern`, no `max_depth`. The walker owns the override
/// config because the engine's user-facing filters would mask the very
/// segment descent is searching for; descent dispatch reads
/// `arc.entries.get(name)` directly and (for Profile descent) discards
/// the snapshot.
///
/// `target_resource` is stamped onto the response's
/// `DirSnapshot.root_resource` so consumers reading the snapshot
/// directly (engine dispatch arms with no per-state target field) can
/// identify the prefix without an extra channel field. Profile descent
/// dispatch reads `descent.current_prefix` as the source of truth and
/// `debug_assert!`s the stamp matches.
///
/// `captured_with` is stamped as `0` — descent dispatch never reads the
/// field (the snapshot is consumed by the engine and dropped before any
/// consumer compares hashes), so the value is observationally
/// irrelevant. Callers should not rely on a particular sentinel.
pub(super) fn probe_descent(target_path: &Path, target_resource: ResourceId) -> ProbeOutcome {
    let cfg = ScanConfig::builder()
        .recursive(false)
        .hidden(true)
        .max_depth(None)
        .build();
    probe_subtree(
        target_path,
        target_resource,
        &cfg,
        0,
        None,
        &BTreeSet::new(),
        false,
    )
}

/// Returns `true` iff any path in `force_walk` is at-or-under `path`.
///
/// Why `Path::starts_with` and not `==`: imagine `path = /a` and
/// `force_walk = {/a/b/c}`. If we skip at `/a`, we never recurse into
/// `/a/b/c` and miss the kernel's signal. Component-wise `starts_with`
/// catches this — at `/a`, `(/a/b/c).starts_with(/a)` is true ⇒ refuse
/// skip ⇒ enumerate children. At `/a/b`, the same path triggers the same
/// refusal until we reach `/a/b/c`'s leaf, after which sibling subtrees
/// are mtime-skip-eligible again.
///
/// Byte-lex via `BTreeSet::range` would erroneously match `/ab` when
/// probing `/a`; we need component-wise `Path::starts_with`.
fn any_forced_under(path: &Path, force_walk: &BTreeSet<PathBuf>) -> bool {
    force_walk.iter().any(|p| p.starts_with(path))
}

/// Read one directory level, applying filters and recursing into covered
/// Dir children. Returns the constructed entries map.
///
/// Errors at this level are skip-and-continue. `read_dir` failure on
/// `path` (EACCES, EIO, ENOENT after a raced delete, etc.) returns the
/// already-accumulated `BTreeMap` — empty when the failure was at the
/// `read_dir` open boundary, partially-populated if dirent iteration
/// errored mid-walk after some entries had already been folded in. The
/// caller ([`walk_subdir`] or [`probe_subtree`]) wraps the result in
/// `Arc::new(DirSnapshot::new(…))`, so the parent emits
/// `DirChild { subtree: Some(empty_or_partial_arc) }` — covered-but-
/// empty, distinct from the uncovered `DirChild { subtree: None }`
/// sentinel reserved for `recursive=false`, beyond `max_depth`,
/// cross-filesystem boundaries, and mid-walk `lstat`/kind-flip failures
/// on a subdir itself.
fn enumerate_dir(
    path: &Path,
    anchor_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&DirSnapshot>,
    force_walk: &BTreeSet<PathBuf>,
    forced: bool,
    depth: u32,
    root_dev: u64,
) -> BTreeMap<CompactString, ChildEntry> {
    let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();

    let read_dir = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return entries,
        Err(e) => {
            tracing::warn!(?path, ?e, "probe_subtree readdir failed; skipping subtree");
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
        if !config.hidden && name_str.starts_with('.') {
            continue;
        }
        let Ok(rel) = child_path.strip_prefix(anchor_path) else {
            tracing::trace!(?child_path, "probe_subtree strip_prefix failed; skipping");
            continue;
        };
        if config.exclude.iter().any(|g| g.matches_path(rel)) {
            continue;
        }
        let Ok(cmeta) = std::fs::symlink_metadata(&child_path) else {
            continue;
        };
        let file_type = cmeta.file_type();
        let is_dir = file_type.is_dir();

        // Pattern semantics: directories are always covered (we descend
        // through them); files (and symlinks/other) are gated by the
        // pattern.
        if let Some(pat) = &config.pattern
            && !is_dir
            && !pat.matches_path(rel)
        {
            continue;
        }

        let key = CompactString::new(name_str);
        let child_entry = if is_dir {
            build_dir_child(
                &child_path,
                anchor_path,
                config,
                captured_with,
                baseline,
                force_walk,
                forced,
                depth,
                root_dev,
                &cmeta,
                name_str,
            )
        } else {
            build_leaf_child(&cmeta, file_type, name_str, baseline)
        };

        entries.insert(key, child_entry);
    }

    entries
}

/// Build a `ChildEntry::Dir` for one directory dirent. Recurses via
/// [`walk_subdir`] when the entry is in-scope (recursive walk, within
/// `max_depth`, same filesystem); emits `subtree: None` for uncovered
/// branches.
#[allow(clippy::too_many_arguments)]
fn build_dir_child(
    child_path: &Path,
    anchor_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&DirSnapshot>,
    force_walk: &BTreeSet<PathBuf>,
    forced: bool,
    depth: u32,
    root_dev: u64,
    cmeta: &std::fs::Metadata,
    name: &str,
) -> ChildEntry {
    let recurse = config.recursive
        && depth + 1 < config.max_depth.unwrap_or(u32::MAX)
        && cmeta.dev() == root_dev;
    if !recurse {
        // Uncovered branch: not recursive, beyond max_depth, or cross-fs.
        // Walker stores the entry but does not recurse.
        return ChildEntry::Dir(DirChild {
            fs_id: FsIdentity {
                inode: cmeta.ino(),
                device: cmeta.dev(),
            },
            subtree: None,
        });
    }
    // Pull the child's prior subtree from baseline so mtime-skip composes
    // recursively. BTreeMap key match by string segment is the snapshot's
    // native lookup.
    let child_baseline = baseline
        .and_then(|b| b.entries.get(name))
        .and_then(|c| match c {
            ChildEntry::Dir(dc) => dc.subtree.as_ref(),
            ChildEntry::Leaf(_) => None,
        });
    let sub = walk_subdir(
        child_path,
        anchor_path,
        config,
        captured_with,
        child_baseline,
        force_walk,
        forced,
        depth + 1,
        root_dev,
    );
    ChildEntry::Dir(DirChild {
        fs_id: FsIdentity {
            inode: cmeta.ino(),
            device: cmeta.dev(),
        },
        subtree: sub,
    })
}

/// Build a `ChildEntry::Leaf` for one non-directory dirent. Transfers
/// the baseline's cached `leaf_hash` when the prior entry's identity
/// matches — re-enumeration of an unchanged leaf skips the SipHash24
/// fold the engine would otherwise pay during stability comparison.
/// Cache miss falls back to lazy compute on the engine thread, identical
/// to pre-optimisation behaviour.
fn build_leaf_child(
    cmeta: &std::fs::Metadata,
    file_type: std::fs::FileType,
    name: &str,
    baseline: Option<&DirSnapshot>,
) -> ChildEntry {
    let kind = if file_type.is_file() {
        EntryKind::File
    } else if file_type.is_symlink() {
        EntryKind::Symlink
    } else {
        EntryKind::Other
    };
    let leaf = LeafEntry::new(
        kind,
        cmeta.len(),
        cmeta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        FsIdentity {
            inode: cmeta.ino(),
            device: cmeta.dev(),
        },
    );
    let leaf = match baseline.and_then(|b| b.leaf_hash_if_unchanged(name, &leaf)) {
        Some(h) => leaf.with_cached_hash(h),
        None => leaf,
    };
    ChildEntry::Leaf(leaf)
}

/// Recursive helper: probe one level deeper.
///
/// Returns `Some(Arc<DirSnapshot>)` on success (including partial
/// enumeration after a `read_dir` warn) and `None` for mid-walk
/// `Vanished` / `Failed` / kind-flip cases — the parent emits
/// `DirChild { subtree: None }` for `None` returns.
fn walk_subdir(
    path: &Path,
    anchor_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    force_walk: &BTreeSet<PathBuf>,
    forced: bool,
    depth: u32,
    root_dev: u64,
) -> Option<Arc<DirSnapshot>> {
    let Ok(raw) = std::fs::symlink_metadata(path) else {
        return None;
    };
    if !raw.is_dir() {
        return None;
    }
    let root_meta = DirMeta {
        mtime: raw.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        fs_id: FsIdentity {
            inode: raw.ino(),
            device: raw.dev(),
        },
    };

    // Per-level mtime-skip — same primitive as the root probe.
    if !forced
        && !any_forced_under(path, force_walk)
        && let Some(prior) = baseline
        && prior.root_meta == root_meta
    {
        return Some(Arc::clone(prior));
    }

    let entries = enumerate_dir(
        path,
        anchor_path,
        config,
        captured_with,
        baseline.map(Arc::as_ref),
        force_walk,
        forced,
        depth,
        root_dev,
    );

    // `target_resource` for sub-snapshots is `ResourceId::default()` —
    // the engine resolves child-resource identity at receive-time.
    Some(Arc::new(DirSnapshot::new(
        ResourceId::default(),
        root_meta,
        captured_with,
        entries,
    )))
}
