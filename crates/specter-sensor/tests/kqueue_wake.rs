//! Cross-thread wake-handle correctness — `wake()` interrupts an
//! in-flight `poll_until`, concurrent wakes coalesce in the kernel,
//! and a wake after the watcher has been dropped is a no-op (Arc keeps
//! the kqueue fd alive). macOS / FreeBSD only.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{ClassSet, ResourceId, ResourceKind};
use specter_sensor::{DrainWindow, FsWatcher, KqueueWatcher, WatcherEvent};
use std::fs;
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
fn wake_interrupts_long_poll_until() {
    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
    let wake = w.wake_handle();

    // Spawn the wake-issuing thread first; the main thread blocks in
    // poll_until and gets interrupted.
    let waker = std::thread::spawn(move || {
        std::thread::sleep(Duration::from_millis(80));
        wake.wake();
    });

    let mut events: Vec<WatcherEvent> = Vec::new();
    let start = Instant::now();
    let n = w
        .poll_until(Some(Instant::now() + Duration::from_secs(10)), &mut events)
        .unwrap();
    let elapsed = start.elapsed();

    waker.join().unwrap();

    assert_eq!(n, 0, "wake produces no fs events");
    assert!(
        elapsed < Duration::from_secs(2),
        "wake should interrupt within ~80ms; took {elapsed:?}"
    );

    drop(w);
}

#[test]
fn multiple_concurrent_wakes_coalesce() {
    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
    let wake = w.wake_handle();

    let mut threads = Vec::new();
    for _ in 0..4 {
        let h = wake.clone();
        threads.push(std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(60));
            h.wake();
        }));
    }

    let mut events: Vec<WatcherEvent> = Vec::new();
    let n = w
        .poll_until(Some(Instant::now() + Duration::from_secs(2)), &mut events)
        .unwrap();

    for t in threads {
        t.join().unwrap();
    }

    // EVFILT_USER + EV_CLEAR coalesces concurrent triggers — at most
    // one user event arrives per drain. The watcher filters it out, so
    // `n == 0`. The kernel may subsequently re-trigger if more wakes
    // arrive after the drain; that's tested separately below.
    assert_eq!(n, 0, "concurrent wakes coalesce → 0 fs events");

    drop(w);
}

#[test]
fn wake_after_drop_does_not_panic() {
    let watcher = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
    let wake = watcher.wake_handle();

    // Drop the watcher; the wake handle's Arc<OwnedFd> keeps the
    // kqueue fd alive, so wake() still succeeds at the syscall level.
    // No consumer drains the resulting event — kernel reaps when the
    // last Arc clone drops below.
    drop(watcher);
    wake.wake();
    wake.wake(); // Idempotent at the kernel level.

    drop(wake); // Final Arc drop reaps the kqueue fd.
}

#[test]
fn wake_handle_clone_box_is_independent() {
    let w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
    let h1 = w.wake_handle();
    let h2 = h1.clone();
    drop(h1);
    h2.wake(); // Still alive — h2 holds its own Arc clone.
}

#[test]
fn poll_until_returns_promptly_with_zero_deadline() {
    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
    let mut events: Vec<WatcherEvent> = Vec::new();

    let start = Instant::now();
    // Past deadline → non-blocking poll.
    let past = start
        .checked_sub(Duration::from_secs(1))
        .expect("1s before Instant::now() is representable");
    let _ = w.poll_until(Some(past), &mut events);
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(100),
        "past deadline should yield non-blocking poll; took {elapsed:?}"
    );
    assert_eq!(events.len(), 0);

    drop(w);
}

/// Live drain-window wake interaction (F-MED-1 regression detector).
///
/// The watcher's deferred-drain phase (phase 2) is gated on five
/// terms; the fifth, added in this audit, is `!phase1_woke`. A wake
/// observed alongside real events in phase 1 must suppress phase 2 —
/// otherwise the watcher would burn the full drain window before
/// returning to the bin's loop, delaying the application of queued
/// `WatchOp`s.
///
/// The test primes `last_event_at` with one watcher-side cycle (so
/// the recency gate would otherwise open), then enqueues a wake and
/// a real event before the next `poll_until`. Phase 1 reads both;
/// the `!phase1_woke` gate must keep phase 2 closed.
///
/// **Timing distinguishes the fix from the bug.** With the fix:
/// `elapsed ≈ 0` (one `kevent` round-trip). Without the fix: phase 2
/// enters with the window-bounded deadline and, with no further
/// events queued, blocks the full window (≈ 50 ms). The 20 ms
/// assertion threshold is comfortably between the two regimes on any
/// sane host.
#[test]
fn wake_during_phase1_suppresses_phase2() {
    const DRAIN_WINDOW: Duration = Duration::from_millis(50);

    let tmp = TempDir::new().unwrap();
    let file = tmp.path().join("f.txt");
    fs::write(&file, "v0").unwrap();

    let mut w = KqueueWatcher::new(DrainWindow::new(DRAIN_WINDOW)).unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());
    w.watch(r, &file, ResourceKind::File, ClassSet::CONTENT)
        .expect("watch file ok");

    let mut events: Vec<WatcherEvent> = Vec::new();

    // Prime cycle: one real event + drain so the watcher's
    // `last_event_at` is set. Without this, the next call's recency
    // gate is closed and phase 2 is skipped regardless of the
    // wake-fired term — the test would pass on the buggy code too.
    fs::write(&file, "v1").unwrap();
    let _ = w
        .poll_until(Some(Instant::now() + Duration::from_secs(1)), &mut events)
        .expect("prime drain ok");
    events.clear();

    let wake = w.wake_handle();

    // Enqueue wake and a real fs event before the next `poll_until`
    // entry. Both end up queued kernel-side; phase 1's
    // `kevent_drain` returns the pair in one batch.
    wake.wake();
    fs::write(&file, "v2").unwrap();
    // Small settle so the kernel's vnode-event delivery finishes
    // before the watcher's syscall samples the queue. 10 ms is well
    // inside the 50 ms recency window.
    std::thread::sleep(Duration::from_millis(10));

    let start = Instant::now();
    let _ = w
        .poll_until(Some(Instant::now() + Duration::from_secs(1)), &mut events)
        .expect("poll_until ok");
    let elapsed = start.elapsed();

    assert!(
        elapsed < Duration::from_millis(20),
        "wake observed in phase 1 must suppress phase 2; \
         elapsed {elapsed:?} suggests phase 2 burned the {DRAIN_WINDOW:?} window",
    );

    drop(w);
}
