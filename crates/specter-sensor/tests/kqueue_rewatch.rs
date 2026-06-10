//! Re-registration on mask change — the per-entry `KqueueEntry.fflags` cache +
//! `EV_ADD`-overwrites-fflags semantics.
//!
//! These tests exercise [`KqueueWatcher::watch`]'s re-watch path: a second `watch()` call on a
//! resource that already holds an entry. The watcher diffs the cached fflags against the
//! translator's output for the new `(events, kind)` and re-registers via `EV_ADD` when they differ.
//! macOS / FreeBSD only.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Build a `mio::Poll` registered on the watcher's kqueue fd. The drain helpers block via the
/// reactor; `drain_ready` is non-blocking by trait.
fn poll_for(w: &KqueueWatcher) -> Poll {
    let poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register kqueue fd");
    poll
}

/// Push every [`WatcherEvent::Fs`] in `buf` into `out`; `panic!` on [`WatcherEvent::Overflow`]
/// (kqueue must not emit it under v1).
fn collect_fs(buf: &mut Vec<WatcherEvent>, out: &mut Vec<(ResourceId, FsEvent)>) {
    for ev in buf.drain(..) {
        match ev {
            WatcherEvent::Fs { resource, event } => out.push((resource, event)),
            WatcherEvent::Overflow { scope } => {
                panic!("kqueue must not emit WatcherEvent::Overflow; got scope={scope:?}");
            }
        }
    }
}

/// Drain at least one event matching `pred` or hit `overall` deadline. Spurious wakes drain to
/// `Ok(0)` and the loop re-blocks on the next `poll`.
fn drain_until<F: Fn(&(ResourceId, FsEvent)) -> bool>(
    w: &mut KqueueWatcher,
    pred: F,
    overall: Duration,
) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + overall;
    let mut poll = poll_for(w);
    let mut events = Events::with_capacity(8);
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut out: Vec<(ResourceId, FsEvent)> = Vec::new();
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        buf.clear();
        if w.drain_ready(&mut buf).is_err() {
            break;
        }
        collect_fs(&mut buf, &mut out);
        if out.iter().any(&pred) {
            return out;
        }
    }
    out
}

/// Drain every `Fs` event the watcher emits over the next `dur`. Bounded-time drain — the
/// negative-assertion tests below rely on returning the full window's events so callers can
/// `assert!(!.iter() .any(...))`.
fn drain_for(w: &mut KqueueWatcher, dur: Duration) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + dur;
    let mut poll = poll_for(w);
    let mut events = Events::with_capacity(8);
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut out: Vec<(ResourceId, FsEvent)> = Vec::new();
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        buf.clear();
        if w.drain_ready(&mut buf).is_err() {
            break;
        }
        collect_fs(&mut buf, &mut out);
    }
    out
}

/// Re-watch with a widened mask should make new event classes deliverable. Specifically: a
/// CONTENT-only registration filters out `MetadataChanged`; widening to `CONTENT | METADATA` (a
/// fresh `Watch` op the engine emits when `Resource.events_union` changes) must re-register the FD
/// with `NOTE_ATTRIB`, and a subsequent chmod then fires `MetadataChanged`.
#[test]
fn rewatch_with_widened_mask_delivers_new_classes() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "initial").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // First registration: CONTENT only. NOTE_ATTRIB is NOT installed, so chmod must not fire
    // MetadataChanged.
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    // Drain any pending registration acks / spurious events.
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // chmod: should NOT fire MetadataChanged since NOTE_ATTRIB isn't registered.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms).unwrap();
    let early = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !early
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "CONTENT-only mask must not deliver MetadataChanged; got {early:?}"
    );

    // Re-register with widened mask. Same path, same resource — the watcher takes the re-watch path,
    // diffs cached fflags vs new, and re-registers via EV_ADD with NOTE_ATTRIB now in the mask.
    w.watch(
        r,
        &path,
        ResourceKind::File,
        ClassSet::CONTENT | ClassSet::METADATA,
    )
    .unwrap();

    // Now chmod fires MetadataChanged.
    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o644);
    std::fs::set_permissions(&path, perms).unwrap();
    let post = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        post.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "after re-register with METADATA, chmod should fire MetadataChanged; got {post:?}"
    );

    drop(w);
}

/// Re-watch with a narrowed mask should drop classes from the registration. Specifically: an
/// initial `STRUCTURE | METADATA` registration on a Dir delivers both `StructureChanged` on child
/// write and `MetadataChanged` on chmod. Re-registering with `STRUCTURE` only must remove
/// `NOTE_ATTRIB`, so chmod no longer fires.
#[test]
fn rewatch_with_narrowed_mask_drops_classes() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    // First: STRUCTURE | METADATA on the dir.
    w.watch(
        r,
        tmp.path(),
        ResourceKind::Dir,
        ClassSet::STRUCTURE | ClassSet::METADATA,
    )
    .unwrap();
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // Verify METADATA is currently registered: chmod fires.
    let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
    perms.set_mode(0o700);
    std::fs::set_permissions(tmp.path(), perms).unwrap();
    let pre = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        pre.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "pre-narrow: chmod should fire MetadataChanged; got {pre:?}"
    );

    // Narrow to STRUCTURE only. Re-watch path: re-registers without NOTE_ATTRIB.
    w.watch(r, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .unwrap();
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // chmod again — must NOT fire MetadataChanged anymore.
    let mut perms = std::fs::metadata(tmp.path()).unwrap().permissions();
    perms.set_mode(0o755);
    std::fs::set_permissions(tmp.path(), perms).unwrap();
    let post = drain_for(&mut w, Duration::from_millis(300));
    assert!(
        !post
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::MetadataChanged),
        "post-narrow: METADATA dropped, chmod must not fire MetadataChanged; got {post:?}"
    );

    // STRUCTURE still works — child write fires StructureChanged.
    std::fs::write(tmp.path().join("child.txt"), "x").unwrap();
    let post_struct = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::StructureChanged,
        Duration::from_secs(2),
    );
    assert!(
        post_struct
            .iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::StructureChanged),
        "post-narrow: STRUCTURE still active, child write should fire StructureChanged; got {post_struct:?}"
    );

    drop(w);
}

/// Re-watch with the same mask is a no-op — the watcher's cache catches it and skips the syscall.
/// Observationally: the fd's behavior is unchanged. We exercise this by checking that a normal
/// write still fires `ContentChanged` after a same-mask re-watch (i.e., re-watching didn't
/// accidentally clear the fflags).
#[test]
fn rewatch_with_same_mask_preserves_registration() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();
    // Same mask twice: hits the cache-diff `noop` branch.
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    std::fs::write(&path, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::ContentChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::ContentChanged),
        "post no-op rewatch, write should still fire ContentChanged; got {out:?}"
    );

    drop(w);
}

/// `unwatch` clears the per-FD fflags cache so a subsequent fresh `watch` opens a new FD. This is
/// the cache-lifecycle invariant.
#[test]
fn unwatch_then_watch_starts_fresh() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = sm.insert(());

    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();
    w.unwatch(r);
    // Fresh watch (FD reopened, cache repopulated). Observable check: a subsequent write fires
    // ContentChanged normally.
    w.watch(r, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    std::fs::write(&path, "y").unwrap();
    let out = drain_until(
        &mut w,
        |(rid, e)| *rid == r && *e == FsEvent::ContentChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(rid, e)| *rid == r && *e == FsEvent::ContentChanged),
        "post unwatch+watch, write should fire ContentChanged; got {out:?}"
    );

    drop(w);
}
