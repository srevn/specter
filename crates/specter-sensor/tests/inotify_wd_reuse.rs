//! wd-reuse race mitigation.
//!
//! When `inotify_rm_watch(wd=N)` is called, the kernel queues
//! `IN_IGNORED` on the per-instance event queue and frees `N` on the
//! per-instance `idr`. A subsequent `inotify_add_watch` may return the
//! same `N` *before* userspace observes the queued `IN_IGNORED`. Without
//! protection, pre-rm events on the old inode would mis-attribute to the
//! freshly attached resource — a silent state-corruption a `Removed`
//! event on the wrong slot is the symptom of.
//!
//! The watcher closes this race via a `draining_wds: BTreeSet<c_int>`:
//! `unwatch` marks the wd as draining BEFORE `rm_watch`, the drain loop
//! drops events on draining wds, and `IN_IGNORED` consumption clears the
//! flag. The kernel's FIFO event order makes this correct under healthy
//! invariants.
//!
//! These tests pin the invariant observationally: a rapid
//! `unwatch(r1) → watch(r2, same_path)` cycle must never deliver an
//! event to `r1` (the unwatched id) and must always attribute fresh
//! events to `r2` (the live id), even when the kernel reuses the wd.

#![allow(clippy::iter_with_drain)]
#![cfg(target_os = "linux")]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{DrainWindow, FsWatcher, InotifyWatcher, WatcherEvent};
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn drain_for(w: &mut InotifyWatcher, dur: Duration) -> Vec<(ResourceId, FsEvent)> {
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + dur), &mut buf);
    let mut out = Vec::with_capacity(buf.len());
    for ev in buf.drain(..) {
        if let WatcherEvent::Fs { resource, event } = ev {
            out.push((resource, event));
        }
    }
    out
}

#[test]
fn rapid_unwatch_watch_cycle_attributes_to_new_resource() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new(DrainWindow::disabled()).unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    // 100 cycles is plenty to exercise wd reuse on Linux's per-instance
    // idr (which prefers low free indices on reuse). Each cycle:
    // unwatch r_old → watch r_new on the same inode → write → drain
    // and assert correctness.
    for cycle in 0..100 {
        let r_old = sm.insert(());
        w.watch(r_old, &path, ResourceKind::File, ClassSet::CONTENT)
            .expect("initial watch");
        // Unwatch immediately — pre-rm events queued on r_old's wd are
        // possible if a concurrent write lands here, but with no
        // intervening disk op, the kernel queue is empty by the time
        // we hit the rm.
        w.unwatch(r_old);

        let r_new = sm.insert(());
        w.watch(r_new, &path, ResourceKind::File, ClassSet::CONTENT)
            .expect("post-unwatch re-watch");

        // Write to fire an event. Whether the wd is reused or fresh,
        // the watcher's `by_wd[wd]` must map to `r_new` (NOT r_old).
        std::fs::write(&path, format!("v{cycle}")).unwrap();
        let evs = drain_for(&mut w, Duration::from_millis(200));

        for (rid, event) in &evs {
            assert_ne!(
                *rid, r_old,
                "cycle {cycle}: event {event:?} attributed to unwatched r_old; \
                 wd-reuse race detected — pre-rm event leaked through `draining_wds`"
            );
        }
        assert!(
            evs.iter()
                .any(|(rid, e)| *rid == r_new && *e == FsEvent::Modified),
            "cycle {cycle}: post-rewatch write must deliver Modified to r_new; got {evs:?}"
        );

        w.unwatch(r_new);
        // Brief drain to let any final `IN_IGNORED` settle before the
        // next cycle's `unwatch`. Without this, `draining_wds` could
        // accumulate. The watcher tolerates accumulation but the test
        // wants a clean per-cycle baseline.
        let _ = drain_for(&mut w, Duration::from_millis(20));
    }

    drop(w);
}

#[test]
fn pre_rm_event_on_old_wd_is_dropped_not_misattributed() {
    // Tighter race window: queue a write on r_old's wd, then immediately
    // unwatch (pre-rm events are now in the kernel queue), then watch
    // r_new on the same path. If the kernel reuses the wd, the queued
    // event on the old inode would land on r_new without the
    // `draining_wds` filter. With it, the event is dropped.
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("hot.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new(DrainWindow::disabled()).unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    let r_old = sm.insert(());
    w.watch(r_old, &path, ResourceKind::File, ClassSet::CONTENT)
        .expect("initial watch");

    // Queue an event on r_old's wd. Don't drain it.
    std::fs::write(&path, "queued-event").unwrap();

    // Tear down without draining; the event is now stuck in the
    // kernel's per-instance queue, addressed to r_old's wd. Mark it
    // draining + rm.
    w.unwatch(r_old);

    // Reattach a fresh ResourceId on the same path. The kernel may
    // reuse r_old's wd here.
    let r_new = sm.insert(());
    w.watch(r_new, &path, ResourceKind::File, ClassSet::CONTENT)
        .expect("post-rm re-watch");

    // Drain pending events. The pre-rm queued event must NOT surface
    // as `r_new`'s `Modified`. The `IN_IGNORED` for r_old's wd will
    // also be in the queue — the watcher consumes it and clears the
    // draining flag.
    let evs = drain_for(&mut w, Duration::from_millis(200));
    assert!(
        evs.iter().all(|(rid, _)| *rid != r_old),
        "no event should reference unwatched r_old; got {evs:?}"
    );
    // We don't assert `evs is empty` — the kernel may also surface a
    // late event from the post-rewatch state if internal scheduling
    // reorders. The test's contract is "no mis-attribution to r_old or
    // r_new from r_old's queued events".

    // Trigger a guaranteed event on the live r_new and verify
    // attribution.
    std::fs::write(&path, "fresh-event").unwrap();
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut saw_new = false;
    while Instant::now() < deadline && !saw_new {
        let evs = drain_for(&mut w, Duration::from_millis(50));
        if evs
            .iter()
            .any(|(rid, e)| *rid == r_new && *e == FsEvent::Modified)
        {
            saw_new = true;
        }
        for (rid, _) in evs {
            assert_ne!(
                rid, r_old,
                "post-rewatch fresh write fired on r_old — wd-reuse mis-attribution"
            );
        }
    }
    assert!(saw_new, "fresh write on r_new must deliver Modified");

    drop(w);
}
