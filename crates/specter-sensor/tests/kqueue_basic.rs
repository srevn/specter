//! Real-fs round-trip. Each test sets up a watcher, runs one filesystem
//! operation, and asserts that the corresponding `FsEvent` arrives at
//! `poll_until`. macOS / FreeBSD only — kqueue is BSD-only.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{FsEvent, ResourceId, WatchOpts};
use specter_sensor::{FsWatcher, KqueueWatcher};
use std::ffi::OsStr;
use std::os::unix::ffi::OsStrExt;
use std::os::unix::fs::PermissionsExt;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Drain events from `w` until at least one matches `pred` or the
/// deadline elapses. Returns the accumulated events. Loops with short
/// inner deadlines because kqueue may need a couple of round-trips on
/// some systems before delivering the post-fs-op event.
fn drain_until<F: Fn(&(ResourceId, FsEvent)) -> bool>(
    w: &mut KqueueWatcher,
    pred: F,
    overall: Duration,
) -> Vec<(ResourceId, FsEvent)> {
    let deadline = Instant::now() + overall;
    let mut out = Vec::new();
    while Instant::now() < deadline {
        let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(50)), &mut out);
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
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = fresh_id(&mut sm);

    w.watch(r_dir, tmp.path(), WatchOpts::default())
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

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    w.watch(r_file, &path, WatchOpts::default())
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

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    w.watch(r_file, &path, WatchOpts::default())
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
    w.watch(r_file, &src, WatchOpts::default()).unwrap();

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
    w.watch(r_file, &path, WatchOpts::default()).unwrap();

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

    let res = w.watch(r, bad_path, WatchOpts::default());
    assert!(res.is_err());
}

#[test]
fn watch_nonexistent_path_returns_enoent() {
    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r = fresh_id(&mut sm);

    let res = w.watch(
        r,
        std::path::Path::new("/this/path/does/not/exist/specter"),
        WatchOpts::default(),
    );
    assert!(res.is_err());
    assert_eq!(res.err().unwrap().raw_os_error(), Some(libc::ENOENT));
}

#[test]
fn unwatch_after_event_does_not_panic_on_subsequent_poll() {
    let tmp = TempDir::new().unwrap();
    let path = tmp.path().join("file.txt");
    std::fs::write(&path, "x").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_file = fresh_id(&mut sm);
    w.watch(r_file, &path, WatchOpts::default()).unwrap();

    std::fs::write(&path, "changed").unwrap();
    w.unwatch(r_file);

    // Late event drain — kernel may still deliver an event for the
    // unwatched fd. Watcher emits anyway; the test's contract is
    // "no panic / no error."
    let mut out = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(200)), &mut out);
    drop(w);
}
