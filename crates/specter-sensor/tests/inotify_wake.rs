//! Cross-thread wake-handle correctness — `wake()` interrupts an
//! in-flight `poll_until`, concurrent wakes coalesce in the eventfd
//! counter, and a wake after the watcher has been dropped is a no-op
//! (Arc keeps the eventfd alive). Mirror of `kqueue_wake.rs`. Linux
//! only.

#![cfg(target_os = "linux")]

use specter_sensor::{FsWatcher, InotifyWatcher, WatcherEvent};
use std::time::{Duration, Instant};

#[test]
fn wake_interrupts_long_poll_until() {
    let mut w = InotifyWatcher::new().unwrap();
    let wake = w.wake_handle();

    // Spawn the wake-issuing thread first; the main thread blocks in
    // `poll_until` and gets interrupted.
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
    let mut w = InotifyWatcher::new().unwrap();
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

    // The eventfd counter accumulates concurrent writes; a single
    // `eventfd_read` consumes the entire counter atomically. The
    // watcher emits no fs events on a wake-only return; `n == 0`.
    assert_eq!(n, 0, "concurrent wakes coalesce → 0 fs events");

    drop(w);
}

#[test]
fn wake_after_drop_does_not_panic() {
    let watcher = InotifyWatcher::new().unwrap();
    let wake = watcher.wake_handle();

    // Drop the watcher; the wake handle's `Arc<OwnedFd>` keeps the
    // eventfd alive, so `wake()` still succeeds at the syscall level.
    // No consumer drains the resulting counter — kernel reaps when the
    // last Arc clone drops below.
    drop(watcher);
    wake.wake();
    wake.wake(); // Idempotent at the kernel level.

    drop(wake); // Final Arc drop reaps the eventfd.
}

#[test]
fn wake_handle_clone_box_is_independent() {
    let w = InotifyWatcher::new().unwrap();
    let h1 = w.wake_handle();
    let h2 = h1.clone();
    drop(h1);
    h2.wake(); // Still alive — h2 holds its own Arc clone.
}

#[test]
fn poll_until_returns_promptly_with_zero_deadline() {
    let mut w = InotifyWatcher::new().unwrap();
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
