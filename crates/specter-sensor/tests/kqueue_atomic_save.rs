//! Atomic save: `cp file file.tmp; mv file.tmp file` replaces the
//! destination file's inode atomically. The watched file fd's old
//! inode is unlinked → terminal `FsEvent` (Removed or Renamed,
//! depending on kernel-side semantics); the watched dir fd sees a
//! `StructureChanged`.
//!
//! The engine vacates the slot on the terminal event; the next dir
//! `StructureChanged` triggers a reprobe that discovers the new file
//! at the same path with a fresh `ResourceId`.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a
// `Vec` while preserving its allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, KqueueWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
fn atomic_save_emits_terminal_on_old_inode_and_structure_on_dir() {
    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("main.c");
    let staging = tmp.path().join("main.c.tmp");
    std::fs::write(&target, "v1").unwrap();

    let mut w = KqueueWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = sm.insert(());
    let r_file = sm.insert(());

    // Dir watch needs STRUCTURE so the kernel registers NOTE_WRITE on
    // the directory's vnode and a child rename produces
    // StructureChanged. The file watch can stay with EMPTY events: the
    // identity floor (NOTE_DELETE | NOTE_RENAME | NOTE_REVOKE) covers
    // the terminal event we assert on for the file's old inode.
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir");
    w.watch(r_file, &target, ResourceKind::File, ClassSet::EMPTY)
        .expect("watch file");

    // Register the kqueue fd on a `mio::Poll` for the block half of the
    // drain loop: the watcher's `drain_ready` is non-blocking by
    // contract, so the caller blocks on `AsFd::as_fd` and pumps every
    // ready-edge through `drain_ready`. Spurious wakes are absorbed by
    // an empty drain (`Ok(0)`).
    let mut poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register kqueue fd");

    // Atomic save: write staging file, then rename over target.
    std::fs::write(&staging, "v2 with more content").unwrap();
    std::fs::rename(&staging, &target).unwrap();

    let deadline = Instant::now() + Duration::from_secs(2);
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
        let file_terminal = out
            .iter()
            .any(|(r, e)| *r == r_file && matches!(e, FsEvent::Removed | FsEvent::Renamed));
        let dir_changed = out
            .iter()
            .any(|(r, e)| *r == r_dir && *e == FsEvent::StructureChanged);
        if file_terminal && dir_changed {
            drop(w);
            return;
        }
    }

    panic!("missing terminal-on-file or StructureChanged-on-dir; got {out:?}");
}
