//! In-place file edit (`echo 'test' > file.txt`) inside a subtree-root
//! watched directory does **not** bump the parent directory's mtime on
//! APFS / HFS+, so a `STRUCTURE`-only watch on the parent emits no event.
//! The fix: when the user's events mask includes `CONTENT`, the engine
//! sets `has_per_file_fds = true` and registers a per-file FD on every
//! covered Leaf (driven by `walk_pair` / `create_child` in
//! `crates/specter-engine/src/reconcile.rs`). The kernel then emits
//! `NOTE_WRITE` plus `NOTE_EXTEND` on the file's own FD; the watcher
//! normalizes that to `FsEvent::Modified`.
//!
//! This test pins the **kernel + watcher + translator** half of the
//! closure: when the watcher installs both a Dir watch (STRUCTURE) and a
//! per-file watch (CONTENT), an in-place edit fires `Modified` on the
//! file's FD AND `StructureChanged` on the dir's FD. The engine half is
//! covered by `crates/specter-engine/tests/event_filtering.rs`'s
//! `it_ef_1_default_subtree_root_emits_per_file_watch_on_leaves`.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain, clippy::missing_const_for_fn)]
#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::time::{Duration, Instant};
use tempfile::TempDir;

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

/// In-place file edit (`>` redirect, no rename) fires
/// `FsEvent::Modified` on the per-file FD installed by the engine's
/// `has_per_file_fds = true` walk_pair gating.
///
/// The setup mirrors what the engine produces for a `subtree-root` Sub
/// with default events (`STRUCTURE | CONTENT`):
/// - Parent dir watched with STRUCTURE (Dir-only fflags: NOTE_WRITE |
///   NOTE_EXTEND | NOTE_LINK).
/// - Per-leaf file watched with CONTENT (File-only fflags: NOTE_WRITE |
///   NOTE_EXTEND).
///
/// On APFS / HFS+, an in-place rewrite of `file.txt` does NOT modify
/// the parent dir's mtime — the dir's `NOTE_WRITE` does not fire.
/// Without the per-file FD, the engine would never see this change.
/// With the per-file FD, the kernel emits `NOTE_WRITE` on the file
/// directly, which the watcher normalizes to `FsEvent::Modified`.
#[test]
fn in_place_edit_fires_modified_on_per_file_fd() {
    let tmp = TempDir::new().unwrap();
    let file_path = tmp.path().join("file.txt");
    std::fs::write(&file_path, "v1").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();

    // Engine's typical registration for a subtree-root × default events
    // Sub: dir with STRUCTURE, file with CONTENT.
    let r_dir = sm.insert(());
    let r_file = sm.insert(());
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir with STRUCTURE");
    w.watch(r_file, &file_path, ResourceKind::File, ClassSet::CONTENT)
        .expect("watch file with CONTENT");

    // In-place edit. The shell's `>` redirect on macOS opens the
    // existing file with O_TRUNC then writes — same syscall pattern as
    // `echo 'test' > file.txt`. Crucially this does NOT rename or
    // unlink, so the parent dir's mtime stays put.
    std::fs::write(&file_path, "v2 with more bytes").unwrap();

    let out = drain_until(
        &mut w,
        |(r, e)| *r == r_file && *e == FsEvent::Modified,
        Duration::from_secs(2),
    );
    assert!(
        out.iter()
            .any(|(r, e)| *r == r_file && *e == FsEvent::Modified),
        "per-file FD must fire Modified on in-place edit; got {out:?}",
    );

    drop(w);
}

/// Confirm the symptom: a STRUCTURE-only Dir watch (no per-file FD)
/// does NOT fire on in-place file edits. This is the exact failure mode
/// the event-filtering primitive's default mask was designed to fix.
///
/// The test asserts the absence of an event for ~300 ms — long enough
/// to be confident that the kernel isn't going to deliver one. If APFS
/// or HFS+ ever changes its mtime semantics this test will fail and the
/// design rationale needs revisiting.
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
    let mut warmup: Vec<WatcherEvent> = Vec::new();
    let _ = w.poll_until(
        Some(Instant::now() + Duration::from_millis(100)),
        &mut warmup,
    );

    // In-place edit (the symptom case).
    std::fs::write(&file_path, "v2 with more bytes").unwrap();

    let mut out: Vec<WatcherEvent> = Vec::new();
    let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(300)), &mut out);

    // The dir's STRUCTURE watch must NOT fire on an in-place edit. If
    // it does, the documented design assumption (APFS/HFS+ doesn't bump
    // parent mtime on in-place writes) is violated and the rationale
    // needs revisiting.
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
