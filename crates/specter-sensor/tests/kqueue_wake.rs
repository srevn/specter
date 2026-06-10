//! Non-blocking-drain semantics on the kqueue watcher.
//!
//! The trait contract: [`FsWatcher::drain_ready`] is non-blocking, [`AsFd::as_fd`] is the readiness
//! substrate, and the caller blocks via a reactor (mio::Poll) on the fd. macOS / FreeBSD only —
//! kqueue is BSD-only.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a `Vec` while preserving its
// allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// `drain_ready` on a watcher with no pending kernel records returns `Ok(0)` promptly without
/// blocking. The trait's non-blocking contract: the caller blocks via a reactor on `AsFd::as_fd`,
/// not inside the watcher.
#[test]
fn drain_ready_returns_promptly_on_empty_queue() {
    let mut w = KqueueWatcher::new().unwrap();
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let start = Instant::now();
    let n = w.drain_ready(&mut buf).expect("drain ok");
    let elapsed = start.elapsed();
    assert_eq!(n, 0, "empty queue must produce no events");
    assert!(buf.is_empty());
    assert!(
        elapsed < Duration::from_millis(50),
        "drain_ready must not block; took {elapsed:?}",
    );
    drop(w);
}

/// A `mio::Poll` registered on the watcher's `AsFd::as_fd()` observes the canonical "register →
/// poll → drain → idle" sequence the production driver exercises against a real kqueue fd. BSD twin
/// of the `MockFsWatcher::as_fd_becomes_readable_after_inject` testkit test — pins that the trait's
/// readiness contract holds against the kernel-backed fd.
#[test]
fn poll_then_drain_returns_kernel_events() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());
    w.watch(r, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir ok");

    let mut poll = Poll::new().unwrap();
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register");

    // Trigger a kernel event.
    std::fs::write(tmp.path().join("a"), "x").unwrap();

    // Block on the reactor; drain on every readable edge until the expected event lands or the
    // deadline elapses.
    let mut events = Events::with_capacity(4);
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut saw_event = false;
    let mut buf: Vec<WatcherEvent> = Vec::new();
    while Instant::now() < deadline && !saw_event {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(100));
        poll.poll(&mut events, Some(timeout)).expect("poll");
        buf.clear();
        w.drain_ready(&mut buf).expect("drain");
        for ev in buf.drain(..) {
            if let WatcherEvent::Fs { resource, event } = ev
                && resource == r
                && event == FsEvent::StructureChanged
            {
                saw_event = true;
            }
        }
    }
    assert!(
        saw_event,
        "registered reactor must observe StructureChanged"
    );
    drop(w);
}
