//! Atomic rename across the kind boundary — a Dir at the watched path
//! is replaced by a regular File (or vice versa) between the engine's
//! emit of `WatchOp::Watch { kind, path }` and the watcher's
//! `inotify_add_watch` install.
//!
//! Without the [`crate::inotify::ffi::open_o_path`] race-free chain,
//! the watcher would install on the new inode (kind-disagreement) and
//! silently observe events for the wrong shape. With the chain —
//! `O_PATH` open pins the inode,
//! `fstat` yields the race-stable kind, kind verification rejects on
//! disagreement — the install fails with [`WatchFailure::Resource`]
//! (`ENOTDIR`) and the engine reseeds via descent.
//!
//! This test simulates the post-swap state: the engine's stale `kind`
//! is `Dir` (its last classification), the path now points at a
//! regular file. The kind-disagreement arm should fire deterministically.

#![cfg(target_os = "linux")]

use slotmap::SlotMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use specter_sensor::{DrainWindow, FsWatcher, InotifyWatcher, WatchFailure};
use tempfile::TempDir;

#[test]
fn watch_with_stale_dir_kind_after_swap_to_file_returns_resource() {
    // Setup: tmp/anchor was a Dir; we simulate that by removing the
    // path and recreating it as a regular file. The engine's last
    // classification (the `kind` argument) is still `Dir`.
    let tmp = TempDir::new().unwrap();
    let anchor = tmp.path().join("anchor");
    std::fs::create_dir(&anchor).unwrap();
    std::fs::remove_dir(&anchor).unwrap();
    std::fs::write(&anchor, "now a file").unwrap();

    let mut w = InotifyWatcher::new(DrainWindow::disabled()).unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // Engine's stale `Dir` classification meets observed `File`. The
    // watcher's fstat verification surfaces ENOTDIR — the engine
    // routes through `finalize_anchor_lost` and reseeds via descent
    // rather than installing a kind-incoherent watch.
    let res = w.watch(r, &anchor, ResourceKind::Dir, ClassSet::STRUCTURE);
    assert_eq!(
        res,
        Err(WatchFailure::Resource {
            errno: libc::ENOTDIR,
        }),
        "expected Resource(ENOTDIR) on Dir → File swap; got {res:?}"
    );

    drop(w);
}

#[test]
fn watch_with_stale_file_kind_after_swap_to_dir_returns_resource() {
    // Reverse case: engine's last classification is `File`; the path
    // now points at a directory. Same kind-disagreement arm.
    let tmp = TempDir::new().unwrap();
    let anchor = tmp.path().join("anchor");
    std::fs::write(&anchor, "was a file").unwrap();
    std::fs::remove_file(&anchor).unwrap();
    std::fs::create_dir(&anchor).unwrap();

    let mut w = InotifyWatcher::new(DrainWindow::disabled()).unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    let res = w.watch(r, &anchor, ResourceKind::File, ClassSet::CONTENT);
    assert_eq!(
        res,
        Err(WatchFailure::Resource {
            errno: libc::ENOTDIR,
        }),
        "expected Resource(ENOTDIR) on File → Dir swap; got {res:?}"
    );

    drop(w);
}
