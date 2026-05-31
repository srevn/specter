//! Real-fs round-trip. Each test sets up a watcher, runs one filesystem
//! operation, and asserts that the corresponding `FsEvent` arrives via
//! `drain_ready` driven by a `mio::Poll` registered on the watcher's
//! `AsFd::as_fd`. macOS / FreeBSD only — kqueue is BSD-only.
//!
//! Each test passes the minimum [`ClassSet`] needed to fire the event it
//! asserts on (identity floor + class-aware mapping):
//! - Terminal events (`Removed`, `Renamed`, `Revoked`) work with `EMPTY`
//!   because `IDENTITY_FLOOR = NOTE_DELETE | NOTE_RENAME | NOTE_REVOKE`
//!   is OR-ed onto every registration.
//! - `StructureChanged` on a Dir needs [`ClassSet::STRUCTURE`].
//! - `ContentChanged` on a File needs [`ClassSet::CONTENT`].
//! - `MetadataChanged` needs [`ClassSet::METADATA`].

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation. Required here because the helper
// reuses the same buffer across the drain-loop's iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::ffi::OsStr;
use std::os::fd::{AsFd, AsRawFd};
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Build a `mio::Poll` registered on the watcher's kqueue fd. The
/// helper centralises the "register `as_fd()` for READABLE" boilerplate
/// the drain loops below share — `drain_ready` is non-blocking by
/// trait, so the caller blocks via the reactor.
fn poll_for(w: &KqueueWatcher) -> Poll {
    let poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register kqueue fd");
    poll
}

/// Drain events from `w` into a `(ResourceId, FsEvent)` accumulator
/// until at least one matches `pred` or the deadline elapses. Returns
/// the accumulated events.
///
/// Blocks via `mio::Poll` on the watcher's `AsFd::as_fd`; pumps every
/// readable edge through `drain_ready`. Spurious wakes are harmless —
/// `drain_ready` is idempotent on an empty queue (`Ok(0)`).
///
/// kqueue must not emit [`WatcherEvent::Overflow`] under v1 (`EV_CLEAR`
/// coalesces but never silently drops at the kernel level); the helper
/// `panic!`s if it sees one so a future regression here surfaces as a
/// loud test failure rather than silent event loss.
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
        for ev in buf.drain(..) {
            match ev {
                WatcherEvent::Fs { resource, event } => out.push((resource, event)),
                WatcherEvent::Overflow { scope } => {
                    panic!("kqueue must not emit WatcherEvent::Overflow; got scope={scope:?}");
                }
            }
        }
        if out.iter().any(&pred) {
            return out;
        }
    }
    out
}

/// Drain whatever the watcher emits for `dur`. Returns the raw
/// [`WatcherEvent`] sequence (no `Overflow` panic — late-event tests
/// inspect both variants in `out`).
fn drain_for(w: &mut KqueueWatcher, dur: Duration) -> Vec<WatcherEvent> {
    let deadline = Instant::now() + dur;
    let mut poll = poll_for(w);
    let mut events = Events::with_capacity(8);
    let mut out: Vec<WatcherEvent> = Vec::new();
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        if w.drain_ready(&mut out).is_err() {
            break;
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
    let mut w = KqueueWatcher::new().unwrap();
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

    let mut w = KqueueWatcher::new().unwrap();
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

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    // EMPTY events suffice: NOTE_DELETE is in the identity floor.
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

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    // EMPTY events suffice: NOTE_RENAME is in the identity floor.
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

    let mut w = KqueueWatcher::new().unwrap();
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
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = fresh_id(&mut sm);

    let bad: &OsStr = OsStrExt::from_bytes(b"/tmp/has\0nul");
    let bad_path = std::path::Path::new(bad);

    // `Unknown` kind: open fails before fstat, so kind verification
    // never runs.
    let res = w.watch(r, bad_path, ResourceKind::Unknown, ClassSet::EMPTY);
    assert!(res.is_err());
}

#[test]
fn watch_nonexistent_path_returns_enoent() {
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = fresh_id(&mut sm);

    // `Unknown` kind: open fails with ENOENT before fstat, so kind
    // verification never runs.
    let res = w.watch(
        r,
        std::path::Path::new("/this/path/does/not/exist/specter"),
        ResourceKind::Unknown,
        ClassSet::EMPTY,
    );
    assert_eq!(
        res,
        Err(specter_sensor::WatchFailure::Resource {
            errno: libc::ENOENT,
        }),
    );
}

#[test]
fn unwatch_after_event_does_not_panic_on_subsequent_poll() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    // CONTENT so the kernel actually queues an event for the write
    // below — exercising the late-event-drain path the test's contract
    // covers.
    w.watch(r_file, &path, ResourceKind::File, ClassSet::CONTENT)
        .unwrap();

    std::fs::write(&path, "changed").unwrap();
    w.unwatch(r_file);

    // Late event drain — kernel may still deliver an event for the
    // unwatched fd. Watcher emits anyway; the test's contract is
    // "no panic / no error."
    let _ = drain_for(&mut w, Duration::from_millis(200));
    drop(w);
}
