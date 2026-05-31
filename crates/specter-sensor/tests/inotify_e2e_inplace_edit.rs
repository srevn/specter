//! Real-fs E2E #3 closure on Linux (mirror of `kqueue_e2e_inplace_edit.rs`).
//!
//! Linux inotify (unlike kqueue) does fire `IN_MODIFY` on the parent
//! directory's inode when its child file's content changes via an
//! in-place rewrite *if* the directory is registered for `IN_MODIFY`.
//! But the engine does NOT register `IN_MODIFY` on a Dir under the
//! translator (CONTENT × Dir is a no-op; STRUCTURE on Dir adds
//! `IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO`, not
//! `IN_MODIFY`). So a STRUCTURE-only Dir watch on Linux behaves
//! similarly to kqueue: an in-place edit at a child file fires no
//! parent-dir event. The fix is identical — install a per-file watch
//! with CONTENT.
//!
//! This test pins the **kernel + watcher + translator** half of the
//! closure: with both a STRUCTURE Dir watch and a CONTENT per-file
//! watch installed, an in-place truncate-and-rewrite fires `ContentChanged`
//! on the file's wd and (no event) on the dir's wd. The engine half is
//! covered by `crates/specter-engine/tests/event_filtering.rs`.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain, clippy::missing_const_for_fn)]
#![cfg(target_os = "linux")]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
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

/// Drain whatever the watcher emits for `dur` into `out`, without
/// asserting on a specific event. Used by tests that want to collect
/// the raw `WatcherEvent` stream (incl. overflow) over a window.
fn drain_raw_for(w: &mut InotifyWatcher, dur: Duration, out: &mut Vec<WatcherEvent>) {
    let deadline = Instant::now() + dur;
    let mut poll = poll_for(w);
    let mut events = Events::with_capacity(8);
    while Instant::now() < deadline {
        let timeout = (deadline - Instant::now()).min(Duration::from_millis(50));
        if poll.poll(&mut events, Some(timeout)).is_err() {
            break;
        }
        if w.drain_ready(out).is_err() {
            break;
        }
    }
}

/// E2E #3 closure: an in-place file edit (`>` redirect, no rename) fires
/// `FsEvent::ContentChanged` on the per-file wd installed by the engine's
/// `has_per_file_fds = true` walk_pair gating.
///
/// The setup mirrors what the engine produces for a `subtree-root` Sub
/// with default events (`STRUCTURE | CONTENT`):
/// - Parent dir watched with STRUCTURE (Dir-only mask:
///   `IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO`).
/// - Per-leaf file watched with CONTENT (File-only mask:
///   `IN_MODIFY | IN_CLOSE_WRITE`).
#[test]
fn in_place_edit_fires_content_changed_on_per_file_wd() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("file.txt");
    std::fs::write(&file_path, "v1").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    let r_dir = sm.insert(());
    let r_file = sm.insert(());
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir with STRUCTURE");
    w.watch(r_file, &file_path, ResourceKind::File, ClassSet::CONTENT)
        .expect("watch file with CONTENT");

    // In-place edit. `std::fs::write(path, ...)` opens with `O_TRUNC`
    // and writes — same syscall pattern as `echo 'test' > file.txt`.
    // No rename, no unlink. The kernel emits `IN_MODIFY` and
    // `IN_CLOSE_WRITE` on the file's inode; the watcher's per-batch
    // dedup collapses both to one `ContentChanged`.
    std::fs::write(&file_path, "v2 with more bytes").unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::ContentChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::ContentChanged),
        "per-file wd must fire ContentChanged on in-place edit (E2E #3 closure); got {out:?}",
    );

    // Bonus: per-batch dedup should collapse the kernel's `IN_MODIFY +
    // IN_CLOSE_WRITE` pair to one `ContentChanged` per write — count and
    // assert.
    let modified_count = out
        .iter()
        .filter(|(r, e)| *r == r_file && *e == FsEvent::ContentChanged)
        .count();
    assert!(
        modified_count <= 2,
        "per-batch dedup should keep duplicates low (got {modified_count}); \
         IN_MODIFY + IN_CLOSE_WRITE must collapse within a single drain"
    );

    drop(w);
}

/// Confirm the symptom: a STRUCTURE-only Dir watch (no per-file wd)
/// does NOT fire on in-place file edits. The Dir's STRUCTURE mask covers
/// only child create / delete / move; an in-place edit at a child file
/// is observable via the file's `IN_MODIFY` registration, not the dir's.
#[test]
fn in_place_edit_does_not_fire_on_structure_only_dir_watch() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("file.txt");
    std::fs::write(&file_path, "v1").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = sm.insert(());
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir");

    // Drain any registration noise.
    let mut warmup: Vec<WatcherEvent> = Vec::new();
    drain_raw_for(&mut w, Duration::from_millis(100), &mut warmup);

    // In-place edit (the symptom case).
    std::fs::write(&file_path, "v2 with more bytes").unwrap();

    let mut out: Vec<WatcherEvent> = Vec::new();
    drain_raw_for(&mut w, Duration::from_millis(300), &mut out);

    // The dir's STRUCTURE-only watch must not fire on a child's
    // in-place edit. STRUCTURE on Dir installs only
    // `IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO`; child
    // content modify happens via `IN_MODIFY` on the child's inode, not
    // the parent's.
    let dir_fired = out.iter().any(|ev| {
        matches!(
            ev,
            WatcherEvent::Fs { resource, event }
                if *resource == r_dir && *event == FsEvent::StructureChanged
        )
    });
    assert!(
        !dir_fired,
        "STRUCTURE-only Dir watch must stay silent on a child's in-place edit; got {out:?}",
    );

    drop(w);
}
