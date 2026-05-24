//! `IN_IGNORED` cleanup — the kernel's per-wd reap signal.
//!
//! Two paths reach the cleanup branch in
//! [`InotifyWatcher::poll_until`]:
//!
//! 1. **Watcher-initiated** — `unwatch(r)` calls `inotify_rm_watch(wd)`,
//!    the kernel queues `IN_IGNORED` synchronously, and the watcher
//!    consumes it on the next drain. The wd is in `draining_wds`; the
//!    `IN_IGNORED` arm clears the flag (per-resource state was already
//!    cleared at unwatch time).
//!
//! 2. **Spontaneous** — the kernel reaps the watch because the watched
//!    inode was deleted or the filesystem unmounted. The preceding
//!    `IN_DELETE_SELF` / `IN_UNMOUNT` already produced
//!    [`FsEvent::Removed`] / [`FsEvent::Revoked`]; this `IN_IGNORED`
//!    cleans the watcher's per-resource maps so a future kernel-side
//!    wd reuse can't mis-attribute through a stale `by_wd[wd]`.
//!
//! These tests pin the spontaneous-reap path observationally: after a
//! file's inode is destroyed and the watcher drains, a subsequent
//! `watch(r, …)` succeeds — a stale `by_resource[r]` entry would have
//! routed through the re-watch path with potentially stale state.
//!
//! Mirror-style coverage of the watcher-initiated path lives in
//! `inotify_wd_reuse.rs`, which exercises the same draining-flag
//! lifecycle through its end-to-end re-attribution invariant.

#![allow(clippy::iter_with_drain)]
#![cfg(target_os = "linux")]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatcherEvent};
use std::time::{Duration, Instant};
use tempfile::TempDir;

fn drain_until<F: Fn(&(ResourceId, FsEvent)) -> bool>(
    w: &mut InotifyWatcher,
    pred: F,
    overall: Duration,
) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + overall;
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut out: Vec<(ResourceId, FsEvent)> = Vec::new();
    while Instant::now() < deadline {
        buf.clear();
        let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(50)), &mut buf);
        for ev in buf.drain(..) {
            if let WatcherEvent::Fs { resource, event } = ev {
                out.push((resource, event));
            }
        }
        if out.iter().any(&pred) {
            return out;
        }
    }
    out
}

#[test]
fn delete_self_then_in_ignored_clears_per_resource_state() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("vanishing.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // EMPTY events suffice: `IN_DELETE_SELF` is in the identity floor.
    w.watch(r, &path, ResourceKind::File, ClassSet::EMPTY)
        .unwrap();

    // Delete the file. The kernel queues `IN_DELETE_SELF` then
    // `IN_IGNORED` on the wd in FIFO order.
    std::fs::remove_file(&path).unwrap();

    // Drain the `Removed` event. The drain loop also consumes the
    // `IN_IGNORED` (case 2 in `poll_until`), which clears the
    // `by_resource[r]` entry and removes the wd from `by_wd`.
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Removed,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Removed),
        "expected Removed before IN_IGNORED cleanup; got {out:?}"
    );

    // Recreate the file at the same path and re-watch the same
    // ResourceId. With the spontaneous-reap cleanup, this goes through
    // the fresh-watch path (no entry in `by_resource[r]`) and installs
    // anew. A subsequent write must fire `Modified`, proving the
    // cleanup left no stale state behind.
    std::fs::write(&path, "y").unwrap();
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .expect("post-spontaneous-reap watch must succeed via fresh-watch path");

    std::fs::write(&path, "z").unwrap();
    let post = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        post.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "post-cleanup watch must deliver Modified; got {post:?}"
    );

    drop(w);
}

#[test]
fn unwatch_then_redrain_clears_draining_flag() {
    // Watcher-initiated path: `unwatch(r)` marks the wd as draining and
    // calls `rm_watch`. The kernel queues `IN_IGNORED` synchronously;
    // the next drain consumes it and clears `draining_wds[wd]`. We
    // can't observe `draining_wds` directly from outside the crate, so
    // we test the behavioral equivalent: after the drain, the same
    // ResourceId can be watched again at a fresh path without conflict.
    let tmp = TempDir::new().unwrap();
    let p1 = tmp.path().join("a.txt");
    let p2 = tmp.path().join("b.txt");
    std::fs::write(&p1, "x").unwrap();
    std::fs::write(&p2, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &p1, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();
    w.unwatch(r);

    // Allow the drain loop to consume the `IN_IGNORED` for the prior
    // wd. A small deadline is enough; the kernel queues `IN_IGNORED`
    // synchronously at `rm_watch`.
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(100)), &mut buf);

    // Fresh watch on a different path. If draining state had leaked,
    // the new wd would land in `draining_wds` (silently dropping its
    // events). This watch + write + drain proves the flag was cleared.
    w.watch(r, &p2, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();
    std::fs::write(&p2, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::Modified),
        "post-IN_IGNORED-drain re-watch must deliver Modified; got {out:?}"
    );

    drop(w);
}
