//! Race-stable kind verification (Phase B6 / § 1.2 of the inotify port
//! plan). The watcher's fresh-watch path opens the user-supplied path
//! with `O_PATH | O_NOFOLLOW`, `fstat`s the resulting fd to discover the
//! inode's actual kind, and verifies it against the engine's
//! `WatchOp::Watch.kind`. A disagreement maps to
//! [`WatchFailure::Resource`] (`ENOTDIR`) so the engine routes through
//! the path-fatal recovery channel rather than installing a
//! kind-incoherent watch. Engine-emitted `Unknown` is a wildcard that
//! defers to the observed kind — verification cannot fail when `kind ==
//! ResourceKind::Unknown`.
//!
//! Companion of `inotify_dir_file_swap.rs`: that test exercises the
//! atomic-rename-during-watch race; this test pins the static
//! disagreement case (engine expected one shape, the path is the other).

#![cfg(target_os = "linux")]

use slotmap::SlotMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatchFailure};
use tempfile::TempDir;

#[test]
fn dir_watch_on_regular_file_returns_resource_enotdir() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("plain.txt");
    std::fs::write(&file_path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // Engine asserts `ResourceKind::Dir`; on-disk the path is a regular
    // file. The watcher's fstat verification fails and surfaces ENOTDIR
    // — classified as `Resource` by `WatchFailureExt::from_io`.
    let res = w.watch(r, &file_path, ResourceKind::Dir, ClassSet::STRUCTURE);
    assert_eq!(
        res,
        Err(WatchFailure::Resource {
            errno: libc::ENOTDIR,
        }),
        "expected Resource(ENOTDIR) on Dir-watch over regular file; got {res:?}"
    );

    drop(w);
}

#[test]
fn file_watch_on_directory_returns_resource_enotdir() {
    let tmp = TempDir::new().unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // Engine asserts `ResourceKind::File`; on-disk the path is a
    // directory. The same kind-mismatch arm fires.
    let res = w.watch(r, tmp.path(), ResourceKind::File, ClassSet::CONTENT);
    assert_eq!(
        res,
        Err(WatchFailure::Resource {
            errno: libc::ENOTDIR,
        }),
        "expected Resource(ENOTDIR) on File-watch over directory; got {res:?}"
    );

    drop(w);
}

#[test]
fn unknown_kind_accepts_any_inode_shape_dir() {
    // `ResourceKind::Unknown` is the wildcard the descent placeholder
    // emits before the parent probe has classified the slot. The
    // watcher accepts whatever inode resolves and caches the observed
    // kind for downstream normalization.
    let tmp = TempDir::new().unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, tmp.path(), ResourceKind::Unknown, ClassSet::STRUCTURE)
        .expect("Unknown kind on Dir path must succeed (wildcard)");

    drop(w);
}

#[test]
fn unknown_kind_accepts_any_inode_shape_file() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, ResourceKind::Unknown, ClassSet::CONTENT)
        .expect("Unknown kind on File path must succeed (wildcard)");

    drop(w);
}
