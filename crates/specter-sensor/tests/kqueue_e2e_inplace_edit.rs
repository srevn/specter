//! In-place file edit (`echo 'test' > file.txt`) inside a subtree-root watched directory does
//! **not** bump the parent directory's mtime on APFS / HFS+, so a `STRUCTURE`-only watch on the
//! parent emits no event. The fix: when the user's events mask includes `CONTENT`, the engine sets
//! `has_per_file_fds = true` and registers a per-file FD on every covered Leaf (driven by
//! `apply_diff_to_tree` / `ensure_descendant` in `crates/specter-engine/src/reconcile.rs`). The
//! kernel then emits `NOTE_WRITE` plus `NOTE_EXTEND` on the file's own FD; the watcher normalizes
//! that to `FsEvent::ContentChanged`.
//!
//! This test pins the **kernel + watcher + translator** half of the closure: when the watcher
//! installs both a Dir watch (STRUCTURE) and a per-file watch (CONTENT), an in-place edit fires
//! `ContentChanged` on the file's FD AND `StructureChanged` on the dir's FD. The engine half is
//! covered by `crates/specter-engine/tests/event_filtering.rs`'s
//! `it_ef_1_default_subtree_root_emits_per_file_watch_on_leaves`.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a `Vec` while preserving its
// allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain, clippy::missing_const_for_fn)]
#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Build a `mio::Poll` registered on the watcher's kqueue fd. The drain loops below block via the
/// reactor; `drain_ready` is non-blocking by trait.
fn poll_for(w: &KqueueWatcher) -> Poll {
    let poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register kqueue fd");
    poll
}

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

/// Drain whatever the watcher emits over the next `dur`. Used by the negative test below to confirm
/// the absence of an event over a bounded interval.
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

/// In-place file edit (`>` redirect, no rename) fires `FsEvent::ContentChanged` on the per-file FD
/// installed by the engine's `has_per_file_fds = true` reconciler gating (`apply_diff_to_tree`).
///
/// The setup mirrors what the engine produces for a `subtree-root` Sub with default events
/// (`STRUCTURE | CONTENT`):
/// - Parent dir watched with STRUCTURE (Dir-only fflags: NOTE_WRITE | NOTE_EXTEND | NOTE_LINK).
/// - Per-leaf file watched with CONTENT (File-only fflags: NOTE_WRITE | NOTE_EXTEND).
///
/// On APFS / HFS+, an in-place rewrite of `file.txt` does NOT modify the parent dir's mtime — the
/// dir's `NOTE_WRITE` does not fire. Without the per-file FD, the engine would never see this
/// change. With the per-file FD, the kernel emits `NOTE_WRITE` on the file directly, which the
/// watcher normalizes to `FsEvent::ContentChanged`.
#[test]
fn in_place_edit_fires_content_changed_on_per_file_fd() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("file.txt");
    std::fs::write(&file_path, "v1").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    // Engine's typical registration for a subtree-root × default events Sub: dir with STRUCTURE,
    // file with CONTENT.
    let r_dir = sm.insert(());
    let r_file = sm.insert(());
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir with STRUCTURE");
    w.watch(r_file, &file_path, ResourceKind::File, ClassSet::CONTENT)
        .expect("watch file with CONTENT");

    // In-place edit. The shell's `>` redirect on macOS opens the existing file with O_TRUNC then
    // writes — same syscall pattern as `echo 'test' > file.txt`. Crucially this does NOT rename or
    // unlink, so the parent dir's mtime stays put.
    std::fs::write(&file_path, "v2 with more bytes").unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::ContentChanged,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::ContentChanged),
        "per-file FD must fire ContentChanged on in-place edit; got {out:?}",
    );

    drop(w);
}

/// Confirm the symptom: a STRUCTURE-only Dir watch (no per-file FD) does NOT fire on in-place file
/// edits. This is the exact failure mode the event-filtering primitive's default mask was designed
/// to fix.
///
/// The test asserts the absence of an event for ~300 ms — long enough to be confident that the
/// kernel isn't going to deliver one. If APFS or HFS+ ever changes its mtime semantics this test
/// will fail and the design rationale needs revisiting.
#[test]
fn in_place_edit_does_not_fire_on_dir_watch_alone() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("file.txt");
    std::fs::write(&file_path, "v1").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = sm.insert(());
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir");

    // Drain any registration ack noise so the post-edit drain is clean.
    let _ = drain_for(&mut w, Duration::from_millis(100));

    // In-place edit (the symptom case).
    std::fs::write(&file_path, "v2 with more bytes").unwrap();

    let out = drain_for(&mut w, Duration::from_millis(300));

    // The dir's STRUCTURE watch must NOT fire on an in-place edit. If it does, the documented
    // design assumption (APFS/HFS+ doesn't bump parent mtime on in-place writes) is violated and
    // the rationale needs revisiting.
    let dir_fired = out.iter().any(|ev| {
        matches!(
            ev,
            WatcherEvent::Fs { resource, event }
                if *resource == r_dir && *event == FsEvent::StructureChanged
        )
    });
    assert!(
        !dir_fired,
        "design assumption: in-place edit does not bump parent dir mtime → \
         dir's STRUCTURE watch must stay silent; got {out:?}",
    );

    drop(w);
}
