//! FD-pressure round-trip — `setrlimit(RLIMIT_NOFILE, low)` lowers the
//! per-process FD ceiling; subsequent `KqueueWatcher::watch` calls
//! eventually return `Err(EMFILE)` (or `ENFILE`).
//!
//! The rlimit reduction is process-scoped; cargo runs each
//! `tests/*.rs` as a separate binary, so this test's reduction does
//! not affect any other test. macOS / FreeBSD
//! only.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use nix::sys::resource::{Resource, setrlimit};
use slotmap::SlotMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use specter_sensor::{DrainWindow, FsWatcher, KqueueWatcher, WatchFailure};
use tempfile::TempDir;

#[test]
fn watch_eventually_returns_emfile_under_low_rlimit() {
    // Pre-create files BEFORE lowering rlimit. `std::fs::write` opens
    // the file, which requires its own fd; doing this first keeps the
    // setup phase from itself hitting EMFILE.
    let tmp = TempDir::new().unwrap();
    let mut paths = Vec::with_capacity(200);
    for i in 0..200 {
        let p = tmp.path().join(format!("f{i}"));
        std::fs::write(&p, "").unwrap();
        paths.push(p);
    }

    // Lower the per-process FD ceiling. 64 leaves room for stdio (3)
    // + the test runner's pipes + the `KqueueWatcher`'s own kqueue fd
    // + tempfile's cleanup fds, while still being tight enough that
    // ~50 successful watches exhaust the budget.
    setrlimit(Resource::RLIMIT_NOFILE, 64, 64).expect("setrlimit");

    let mut w = KqueueWatcher::new(DrainWindow::default()).expect("kqueue_new under rlimit");
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    let mut emfile_seen = false;
    for p in &paths {
        let r = sm.insert(());
        if let Err(failure) = w.watch(r, p, ResourceKind::File, ClassSet::EMPTY) {
            assert!(
                matches!(
                    failure,
                    WatchFailure::Pressure {
                        errno: libc::EMFILE | libc::ENFILE,
                    }
                ),
                "expected Pressure(EMFILE|ENFILE), got {failure:?}",
            );
            emfile_seen = true;
            break;
        }
    }
    assert!(emfile_seen, "FD pressure never reached; raise probe count");

    drop(w);
}
