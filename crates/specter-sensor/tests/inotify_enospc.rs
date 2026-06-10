//! `ENOSPC` round-trip — the kernel's `max_user_watches` ceiling eventually rejects
//! `inotify_add_watch` with `ENOSPC`, which the watcher classifies as [`WatchFailure::Pressure`]
//! (per [`specter_sensor::WatchFailureExt::from_io`]). The engine clamps `watch_demand := 0` on the
//! affected resource; the next reconcile is the natural retry path.
//!
//! The kernel ceiling is per-user (typically `524288` on modern distros, much lower in containers).
//! This test reads `/proc/sys/fs/inotify/max_user_watches` and adapts:
//!
//! - If the limit is small enough to exhaust within ~10k watches, do so and assert the failure shape.
//! - Otherwise, skip cleanly with an informational message — exhausting 500k+ watches is too slow
//!   and would impact concurrent tests sharing the per-user limit.
//!
//! Linux only.

#![cfg(target_os = "linux")]

use slotmap::SlotMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatchFailure};
use tempfile::TempDir;

/// Read the per-user inotify watch ceiling. `Err` ⇒ skip the test.
fn read_max_user_watches() -> Option<usize> {
    std::fs::read_to_string("/proc/sys/fs/inotify/max_user_watches")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

#[test]
fn watch_eventually_returns_pressure_under_low_max_user_watches() {
    // Cap at 10000 watches: enough to exhaust the kernel's default in a tight container (typically
    // `8192`), still bounded enough to be reasonable elapsed time on a real-disk runner. Skip if
    // the ceiling is materially higher.
    const TEST_CAP: usize = 10_000;

    let Some(max) = read_max_user_watches() else {
        eprintln!(
            "skipping inotify_enospc: cannot read \
             /proc/sys/fs/inotify/max_user_watches"
        );
        return;
    };

    if max > TEST_CAP {
        eprintln!(
            "skipping inotify_enospc: max_user_watches = {max}, > {TEST_CAP} \
             test cap (would be slow and impact concurrent tests sharing the \
             per-user limit; rerun with a stricter sysctl or in a \
             low-limit container to exercise this path)"
        );
        return;
    }

    let tmp = TempDir::new().unwrap();
    // Pre-create paths to watch. We need at least `max + 1` distinct inodes; create `max + 100` for
    // headroom.
    let count = max + 100;
    let mut paths: Vec<std::path::PathBuf> = Vec::with_capacity(count);
    for i in 0..count {
        let p = tmp.path().join(format!("f{i}"));
        std::fs::write(&p, "").unwrap();
        paths.push(p);
    }

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    let mut pressure_seen = false;
    let mut installed = 0usize;
    for p in &paths {
        let r = sm.insert(());
        match w.watch(r, p, ResourceKind::File, ClassSet::EMPTY) {
            Ok(()) => installed += 1,
            Err(failure) => {
                assert!(
                    matches!(
                        failure,
                        WatchFailure::Pressure {
                            errno: libc::ENOSPC
                        }
                    ),
                    "expected Pressure(ENOSPC) when exhausting max_user_watches; \
                     got {failure:?} after {installed} successful watches"
                );
                pressure_seen = true;
                break;
            }
        }
    }
    assert!(
        pressure_seen,
        "exhausted {installed} watches without hitting ENOSPC; \
         max_user_watches = {max}, count = {count} — the per-user counter \
         has free capacity from concurrent processes? rerun in isolation."
    );

    drop(w);
}
