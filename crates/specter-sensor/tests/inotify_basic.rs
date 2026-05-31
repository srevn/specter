//! Real-fs round-trip on Linux inotify. Each test installs one watch,
//! performs one filesystem operation, and asserts that the corresponding
//! [`FsEvent`] arrives at [`FsWatcher::drain_ready`]. Mirror of
//! `kqueue_basic.rs`.
//!
//! Each test passes the minimum [`ClassSet`] needed to fire the event it
//! asserts on (identity floor + class-aware mapping):
//! - Terminal events ([`FsEvent::Removed`], [`FsEvent::Renamed`]) work
//!   with [`ClassSet::EMPTY`] because `IDENTITY_FLOOR =
//!   IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT` is OR-ed onto every
//!   registration.
//! - [`FsEvent::StructureChanged`] on a Dir needs [`ClassSet::STRUCTURE`].
//! - [`FsEvent::ContentChanged`] on a File needs [`ClassSet::CONTENT`].
//! - [`FsEvent::MetadataChanged`] needs [`ClassSet::METADATA`].

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation. Required here because the helper
// reuses the same buffer across the drain-loop's iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(target_os = "linux")]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatchFailure, WatcherEvent};
use std::ffi::OsStr;
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Build a `mio::Poll` registered on the watcher's inotify fd. The
/// helper centralises the "register `as_fd()` for READABLE" boilerplate
/// the drain loops below share — `drain_ready` is non-blocking by
/// trait, so the caller blocks via the reactor.
fn poll_for(w: &InotifyWatcher) -> Poll {
    let poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register inotify fd");
    poll
}

/// Drain events from `w` into a `(ResourceId, FsEvent)` accumulator until
/// at least one matches `pred` or the deadline elapses. Returns the
/// accumulated events. Inotify can emit [`WatcherEvent::Overflow`], so
/// the helper records but does not panic on it.
fn drain_until<F: Fn(&(ResourceId, FsEvent)) -> bool>(
    w: &mut InotifyWatcher,
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

fn fresh_id(sm: &mut SlotMap<ResourceId, ()>) -> ResourceId {
    sm.insert(())
}

#[test]
fn watch_dir_observes_structure_changed_on_create() {
    let tmp = TempDir::new().unwrap();
    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = fresh_id(&mut sm);

    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir ok");

    std::fs::write(tmp.path().join("foo.c"), "x").unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_dir && *e == FsEvent::StructureChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_dir && *e == FsEvent::StructureChanged),
        "expected StructureChanged on dir, got {out:?}"
    );

    drop(w);
}

#[test]
fn watch_file_observes_content_changed_on_write() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "initial").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    w.watch(r_file, &path, ResourceKind::File, ClassSet::CONTENT)
        .expect("watch file ok");

    std::fs::write(&path, "updated more bytes here").unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::ContentChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::ContentChanged),
        "expected ContentChanged on file, got {out:?}"
    );

    drop(w);
}

#[test]
fn watch_file_observes_removed_on_unlink() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("doomed.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    // EMPTY events suffice: IN_DELETE_SELF is in the identity floor.
    w.watch(r_file, &path, ResourceKind::File, ClassSet::EMPTY)
        .expect("watch file ok");

    std::fs::remove_file(&path).unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::Removed,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::Removed),
        "expected Removed on file, got {out:?}"
    );

    drop(w);
}

#[test]
fn watch_file_observes_renamed_on_rename() {
    let tmp = TempDir::new().unwrap();
    let src = tmp.path().join("src.txt");
    let dst = tmp.path().join("dst.txt");
    std::fs::write(&src, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    // EMPTY events suffice: IN_MOVE_SELF is in the identity floor.
    w.watch(r_file, &src, ResourceKind::File, ClassSet::EMPTY)
        .unwrap();

    std::fs::rename(&src, &dst).unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::Renamed,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::Renamed),
        "expected Renamed on file, got {out:?}"
    );

    drop(w);
}

#[test]
fn watch_file_observes_metadata_changed_on_chmod() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("perm.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    w.watch(r_file, &path, ResourceKind::File, ClassSet::METADATA)
        .unwrap();

    let mut perms = std::fs::metadata(&path).unwrap().permissions();
    perms.set_mode(0o600);
    std::fs::set_permissions(&path, perms).unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::MetadataChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::MetadataChanged),
        "expected MetadataChanged on file, got {out:?}"
    );

    drop(w);
}

#[test]
fn watch_path_with_nul_byte_returns_error() {
    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = fresh_id(&mut sm);

    let bad: &OsStr = OsStrExt::from_bytes(b"/tmp/has\0nul");
    let bad_path = std::path::Path::new(bad);

    // Path with embedded NUL → `CString::new` rejects → `Error::other`
    // which carries no `raw_os_error`, so the trait wrapper hits the
    // `_ → Invariant { errno: 0 }` arm.
    let res = w.watch(r, bad_path, ResourceKind::Unknown, ClassSet::EMPTY);
    assert_eq!(
        res,
        Err(WatchFailure::Invariant { errno: 0 }),
        "expected Invariant on NUL byte; got {res:?}"
    );
}

#[test]
fn watch_nonexistent_path_returns_resource_enoent() {
    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = fresh_id(&mut sm);

    // `Unknown` kind: `open_o_path` fails with ENOENT before fstat, so
    // kind verification never runs. The trait wrapper classifies ENOENT
    // as `WatchFailure::Resource` per `WatchFailureExt::from_io`.
    let res = w.watch(
        r,
        std::path::Path::new("/this/path/does/not/exist/specter"),
        ResourceKind::Unknown,
        ClassSet::EMPTY,
    );
    assert_eq!(
        res,
        Err(WatchFailure::Resource {
            errno: libc::ENOENT,
        }),
    );
}

#[test]
fn unwatch_after_event_does_not_panic_on_subsequent_poll() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    // CONTENT so the kernel actually queues an event for the write
    // below — exercising the late-event-drain path the test's contract
    // covers.
    w.watch(r_file, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    std::fs::write(&path, "changed").unwrap();
    w.unwatch(r_file);

    // Late event drain — kernel may still deliver an event whose wd is
    // now in `draining_wds`. Watcher drops the event silently; the
    // `IN_IGNORED` consumption clears the flag. Test contract is "no
    // panic / no error".
    let mut poll = poll_for(&w);
    let mut events = Events::with_capacity(8);
    let mut out: Vec<WatcherEvent> = Vec::new();
    let deadline = Instant::now() + Duration::from_millis(200);
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        if w.drain_ready(&mut out).is_err() {
            break;
        }
    }
    drop(w);
}
