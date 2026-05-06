//! `EV_DISABLE` / `EV_ENABLE` round-trip — suppress silences delivery,
//! unsuppress restores it. macOS / FreeBSD only.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn drain_for(w: &mut KqueueWatcher, dur: Duration) -> Vec<(ResourceId, FsEvent)> {
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + dur), &mut buf);
    let mut out = Vec::with_capacity(buf.len());
    for ev in buf {
        match ev {
            WatcherEvent::Fs { resource, event } => out.push((resource, event)),
            WatcherEvent::Overflow { scope } => {
                panic!("kqueue must not emit WatcherEvent::Overflow; got scope={scope:?}");
            }
        }
    }
    out
}

#[test]
fn suppress_silences_subsequent_events() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .unwrap();
    w.suppress(r);

    std::fs::write(tmp.path().join("a.txt"), "x").unwrap();
    let suppressed = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !suppressed.iter().any(|(rid, _)| *rid == r),
        "suppress should silence delivery; got {suppressed:?}"
    );

    w.unsuppress(r);
    std::fs::write(tmp.path().join("b.txt"), "y").unwrap();
    let restored = drain_for(&mut w, Duration::from_secs(1));
    assert!(
        restored
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::StructureChanged),
        "unsuppress should restore delivery; got {restored:?}"
    );

    drop(w);
}

#[test]
fn suppress_on_unwatched_resource_is_noop() {
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // Never watched — should log warn and drop, but not panic.
    w.suppress(r);
    w.unsuppress(r);
    drop(w);
}

#[test]
fn suppress_then_unwatch_then_unsuppress_does_not_panic() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .unwrap();
    w.suppress(r);
    w.unwatch(r);
    // EV_ENABLE on a closed fd hits ENOENT inside ffi; the watcher logs
    // warn and drops — no return value to assert beyond "we get here."
    w.unsuppress(r);
    drop(w);
}

#[test]
fn double_suppress_is_idempotent_at_kernel_level() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .unwrap();
    w.suppress(r);
    w.suppress(r); // No error; kernel re-applies EV_DISABLE harmlessly.

    std::fs::write(tmp.path().join("c.txt"), "z").unwrap();
    let out = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !out.iter().any(|(rid, _)| *rid == r),
        "double-suppress still silences; got {out:?}"
    );

    drop(w);
}
