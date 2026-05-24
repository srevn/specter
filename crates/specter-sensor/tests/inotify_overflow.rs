//! `IN_Q_OVERFLOW` round-trip — when the per-instance event queue
//! overflows, the kernel emits a synthetic record with `wd = -1` and
//! `mask = IN_Q_OVERFLOW`. The watcher lifts it to
//! [`WatcherEvent::Overflow`] (`scope: Global`); the bin then posts
//! [`Input::SensorOverflow`] to the engine, which reseeds every
//! in-scope Profile.
//!
//! The per-instance queue size is `/proc/sys/fs/inotify/max_queued_events`
//! (default `16384`). To exercise overflow without root, we'd need to
//! generate >16k events between drains — feasible but slow. Easier:
//! lower the queue cap by writing to that sysctl. Both paths require
//! some privilege (root can write `/proc/sys/...`; non-root just needs
//! to be patient).
//!
//! This test attempts the **non-root patient** path: it generates a
//! large burst of structural events on a Dir watch without intervening
//! drains, expects the kernel to overflow, and asserts that
//! [`WatcherEvent::Overflow`] is emitted. If the burst doesn't trigger
//! overflow within reasonable bounds, we skip cleanly.
//!
//! Linux only.

#![allow(clippy::iter_with_drain)]
#![cfg(target_os = "linux")]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, OverflowScope, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Read the per-instance queue cap. `None` ⇒ skip.
fn read_max_queued_events() -> Option<usize> {
    std::fs::read_to_string("/proc/sys/fs/inotify/max_queued_events")
        .ok()
        .and_then(|s| s.trim().parse().ok())
}

#[test]
fn massive_event_burst_emits_overflow() {
    // Cap at 25k events generated. If `max_queued_events` is higher,
    // skip cleanly — the test would otherwise generate millions of
    // child files and run for minutes.
    const GEN_CAP: usize = 25_000;

    let Some(queue_cap) = read_max_queued_events() else {
        eprintln!(
            "skipping inotify_overflow: cannot read \
             /proc/sys/fs/inotify/max_queued_events"
        );
        return;
    };

    if queue_cap >= GEN_CAP {
        eprintln!(
            "skipping inotify_overflow: max_queued_events = {queue_cap}, \
             >= {GEN_CAP} test cap. Lower with `echo 1024 | sudo tee \
             /proc/sys/fs/inotify/max_queued_events` and rerun, or run on \
             a low-limit container."
        );
        return;
    }

    let tmp = TempDir::new().unwrap();
    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = sm.insert(());

    // Watch the dir with STRUCTURE — every child create fires
    // `IN_CREATE`, queueing an event into the inotify instance.
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir");

    // Burst: create `queue_cap + headroom` files without any drain
    // intervening. The kernel queues each `IN_CREATE` on the
    // per-instance queue; once it crosses `max_queued_events`, the
    // kernel drops further events and emits the synthetic
    // `IN_Q_OVERFLOW` record.
    let target = queue_cap + 2_000;
    for i in 0..target {
        // `OpenOptions::create_new` minimizes per-file overhead vs
        // `std::fs::write`. The kernel only sees `IN_CREATE`.
        if std::fs::OpenOptions::new()
            .create_new(true)
            .write(true)
            .open(tmp.path().join(format!("c{i}")))
            .is_err()
        {
            // RLIMIT_NOFILE or disk pressure: bail and skip.
            eprintln!("skipping inotify_overflow: file creation failed at i={i}");
            return;
        }
    }

    // Register the watcher's inotify fd with mio so the drain loop
    // blocks via the reactor.
    let mut poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register inotify fd");

    // Drain everything; expect at least one `WatcherEvent::Overflow`.
    let deadline = Instant::now() + Duration::from_secs(5);
    let mut events = Events::with_capacity(8);
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut overflow_seen = false;
    let mut total_events = 0usize;
    while Instant::now() < deadline && !overflow_seen {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        buf.clear();
        if w.drain_ready(&mut buf).is_err() {
            break;
        }
        if buf.is_empty() {
            // No records this edge — keep blocking on the reactor
            // until the deadline.
            continue;
        }
        for ev in buf.drain(..) {
            total_events += 1;
            if let WatcherEvent::Overflow { scope } = ev {
                assert_eq!(
                    scope,
                    OverflowScope::Global,
                    "inotify must emit Global-scoped overflow"
                );
                overflow_seen = true;
                break;
            }
        }
    }

    assert!(
        overflow_seen,
        "expected WatcherEvent::Overflow after burst of {target} events \
         (max_queued_events = {queue_cap}); drained {total_events} events"
    );

    drop(w);
}
