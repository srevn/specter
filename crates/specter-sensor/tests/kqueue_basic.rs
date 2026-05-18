//! Real-fs round-trip. Each test sets up a watcher, runs one filesystem
//! operation, and asserts that the corresponding `FsEvent` arrives at
//! `poll_until`. macOS / FreeBSD only — kqueue is BSD-only.
//!
//! Each test passes the minimum [`ClassSet`] needed to fire the event it
//! asserts on (identity floor + class-aware mapping):
//! - Terminal events (`Removed`, `Renamed`, `Revoked`) work with `EMPTY`
//!   because `IDENTITY_FLOOR = NOTE_DELETE | NOTE_RENAME | NOTE_REVOKE`
//!   is OR-ed onto every registration.
//! - `StructureChanged` on a Dir needs [`ClassSet::STRUCTURE`].
//! - `Modified` on a File needs [`ClassSet::CONTENT`].
//! - `MetadataChanged` needs [`ClassSet::METADATA`].

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation. Required here because the helper
// reuses the same buffer across the drain-loop's iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{DrainWindow, FsWatcher, KqueueWatcher, WatcherEvent};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Drain events from `w` into a `(ResourceId, FsEvent)` accumulator
/// until at least one matches `pred` or the deadline elapses. Returns
/// the accumulated events.
///
/// Loops with short inner deadlines because kqueue may need a couple of
/// round-trips on some systems before delivering the post-fs-op event.
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
    let mut buf: Vec<WatcherEvent> = Vec::new();
    let mut out: Vec<(ResourceId, FsEvent)> = Vec::new();
    while Instant::now() < deadline {
        buf.clear();
        let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(50)), &mut buf);
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

fn fresh_id(sm: &mut SlotMap<ResourceId, ()>) -> ResourceId {
    sm.insert(())
}

#[test]
fn watch_dir_observes_structure_changed_on_create() {
    let tmp = TempDir::new().unwrap();
    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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
fn watch_file_observes_modified_on_write() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "initial").unwrap();

    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    w.watch(r_file, &path, ResourceKind::File, ClassSet::CONTENT)
        .expect("watch file ok");

    std::fs::write(&path, "updated more bytes here").unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::Modified),
        "expected Modified on file, got {out:?}"
    );

    drop(w);
}

#[test]
fn watch_file_observes_removed_on_unlink() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("doomed.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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

    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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

    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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
    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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
    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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

    let mut w = KqueueWatcher::new(DrainWindow::disabled()).unwrap();
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
    let mut out: Vec<WatcherEvent> = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(200)), &mut out);
    drop(w);
}
