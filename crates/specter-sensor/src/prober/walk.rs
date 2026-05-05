//! `probe_file` and `probe_dir` — pure-IO walkers.
//!
//! Both probes return [`ProbeResult`]; on success they emit a
//! [`TreeSnapshot`] (file leaf or recursive `Arc<DirSnapshot>` tree).
//! Kind mismatches collapse to `Vanished` ("a file probe that finds a
//! directory (or vice versa) returns `ProbeResult::Vanished`").
//! Errors at the *root* anchor map to `Failed { errno }`; errors mid-walk
//! on a *subtree* skip-and-continue with `tracing::warn!` —
//! `exclude` is the user-facing surface for declaring expected-EACCES paths.
//!
//! Three controls live on the request:
//! - `baseline_subtree`: the engine's last-known view of the target's
//!   subtree. Equal `(mtime, inode, device)` against the freshly `lstat`-ed
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
    ChildEntry, DirChild, DirMeta, DirSnapshot, EntryKind, LeafEntry, ProbeResult, ResourceId,
    ScanConfig, TreeSnapshot,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::SystemTime;

/// File probe. Single `lstat` against `target_path`.
///
/// Returns:
/// - `Ok(TreeSnapshot::File(LeafEntry))` for a regular file.
/// - `Vanished` when the path doesn't exist *or* is not a regular file
///   (kind mismatch — symlink, directory, FIFO, etc.).
/// - `Failed { errno }` for any other I/O error.
pub(super) fn probe_file(target_path: &Path) -> ProbeResult {
    let meta = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeResult::Vanished,
        Err(e) => {
            return ProbeResult::Failed {
                errno: e.raw_os_error().unwrap_or(libc::EIO),
            };
        }
    };
    if !meta.is_file() {
        return ProbeResult::Vanished;
    }
    let leaf = LeafEntry::new(
        EntryKind::File,
        meta.len(),
        meta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        meta.ino(),
        meta.dev(),
    );
    ProbeResult::Ok(TreeSnapshot::File(leaf))
}

/// Directory probe. Recursive DFS walk against `target_path` honoring
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
/// → `Vanished`, anything else → `Failed { errno }`); subtree errors
/// during walk skip-and-continue with `tracing::warn!` and the affected
/// subtree becomes `DirChild { subtree: None }`.
pub(super) fn probe_dir(
    target_path: &Path,
    target_resource: ResourceId,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    force_walk: &BTreeSet<PathBuf>,
    forced: bool,
) -> ProbeResult {
    let root_meta_raw = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeResult::Vanished,
        Err(e) => {
            return ProbeResult::Failed {
                errno: e.raw_os_error().unwrap_or(libc::EIO),
            };
        }
    };
    if !root_meta_raw.is_dir() {
        return ProbeResult::Vanished;
    }
    let root_meta = DirMeta {
        mtime: root_meta_raw.modified().unwrap_or(SystemTime::UNIX_EPOCH),
        inode: root_meta_raw.ino(),
        device: root_meta_raw.dev(),
    };

    // Top-level mtime-skip. Bypassed when forced, or when any forced path
    // lies under target_path (we'd elide visiting it).
    if !forced
        && !any_forced_under(target_path, force_walk)
        && let Some(prior) = baseline
        && prior.root_meta == root_meta
    {
        return ProbeResult::Ok(TreeSnapshot::Dir(Arc::clone(prior)));
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
        root_meta.device,
    );
    ProbeResult::Ok(TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        target_resource,
        root_meta,
        captured_with,
        entries,
    ))))
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
/// Errors are skip-and-continue: a missing/unreadable subdir yields no
/// entry (in the case of a raced delete) or a `DirChild { subtree: None }`
/// (in the case of an unreadable but enumerable parent — `read_dir`
/// errored after `lstat` succeeded). Empty `BTreeMap` is the honest
/// representation of "we tried, found nothing readable."
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
            tracing::warn!(?path, ?e, "probe_dir readdir failed; skipping subtree");
            return entries;
        }
    };

    for dirent_result in read_dir {
        let dirent = match dirent_result {
            Ok(d) => d,
            Err(e) => {
                tracing::trace!(?path, ?e, "probe_dir dirent error; skipping");
                continue;
            }
        };
        let child_path = dirent.path();
        let name_os = dirent.file_name();
        let Some(name_str) = name_os.to_str() else {
            tracing::trace!(?child_path, "probe_dir non-UTF-8 filename; skipping");
            continue;
        };
        if !config.hidden && name_str.starts_with('.') {
            continue;
        }
        let Ok(rel) = child_path.strip_prefix(anchor_path) else {
            tracing::trace!(?child_path, "probe_dir strip_prefix failed; skipping");
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
        let is_file = file_type.is_file();
        let is_symlink = file_type.is_symlink();

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
            // Cross-fs and depth gates for recursion.
            let recurse = config.recursive
                && depth + 1 < config.max_depth.unwrap_or(u32::MAX)
                && cmeta.dev() == root_dev;
            if recurse {
                // Pull the child's prior subtree from baseline so
                // mtime-skip composes recursively. BTreeMap key match by
                // string segment is the snapshot's native lookup.
                let child_baseline = baseline
                    .and_then(|b| b.entries.get(name_str))
                    .and_then(|c| match c {
                        ChildEntry::Dir(dc) => dc.subtree.as_ref(),
                        ChildEntry::Leaf(_) => None,
                    });
                let sub = walk_subdir(
                    &child_path,
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
                    inode: cmeta.ino(),
                    device: cmeta.dev(),
                    subtree: sub,
                })
            } else {
                // Uncovered branch: not recursive, beyond max_depth, or
                // cross-fs. Walker stores the entry but does not recurse.
                ChildEntry::Dir(DirChild {
                    inode: cmeta.ino(),
                    device: cmeta.dev(),
                    subtree: None,
                })
            }
        } else {
            let kind = if is_file {
                EntryKind::File
            } else if is_symlink {
                EntryKind::Symlink
            } else {
                EntryKind::Other
            };
            ChildEntry::Leaf(LeafEntry::new(
                kind,
                cmeta.len(),
                cmeta.modified().unwrap_or(SystemTime::UNIX_EPOCH),
                cmeta.ino(),
                cmeta.dev(),
            ))
        };

        entries.insert(key, child_entry);
    }

    entries
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
        inode: raw.ino(),
        device: raw.dev(),
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
