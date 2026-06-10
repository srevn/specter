//! Atomic save: `cp file file.tmp; mv file.tmp file` replaces the destination file's inode
//! atomically. The watched file's old inode is unlinked → terminal [`FsEvent`] (`Removed` or
//! `Renamed`); the watched parent dir sees a [`FsEvent::StructureChanged`] (the kernel emits
//! `IN_MOVED_TO` on the rename's destination component, which the translator routes through
//! `IN_CREATE | IN_DELETE | IN_MOVED_FROM | IN_MOVED_TO` → `StructureChanged`).
//!
//! The engine vacates the slot on the terminal event; the next dir `StructureChanged` triggers a
//! reprobe that discovers the new file at the same path with a fresh `ResourceId`. Mirror of
//! `kqueue_atomic_save.rs`.

// `iter_with_drain`: `buf.drain(..)` is the canonical way to consume a `Vec` while preserving its
// allocation across drain-loop iterations.
#![allow(clippy::iter_with_drain)]
#![cfg(target_os = "linux")]

use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, ResourceKind};
use specter_sensor::{FsWatcher, InotifyWatcher, WatcherEvent};
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, Instant};
use tempfile::TempDir;

#[test]
fn atomic_save_emits_terminal_on_old_inode_and_structure_on_dir() {
    let tmp = TempDir::new().unwrap();
    let target = tmp.path().join("main.c");
    let staging = tmp.path().join("main.c.tmp");
    std::fs::write(&target, "v1").unwrap();

    let mut w = InotifyWatcher::new().unwrap();
    let mut sm = SlotMap::<ResourceId, ()>::with_key();
    let r_dir = sm.insert(());
    let r_file = sm.insert(());

    // Dir watch needs STRUCTURE so the kernel registers `IN_CREATE | IN_DELETE | IN_MOVED_FROM |
    // IN_MOVED_TO` on the directory and a child rename produces `StructureChanged`. The file watch
    // can stay with EMPTY events: the identity floor (`IN_DELETE_SELF | IN_MOVE_SELF | IN_UNMOUNT`)
    // covers the terminal event we assert on for the file's old inode.
    w.watch(r_dir, tmp.path(), ResourceKind::Dir, ClassSet::STRUCTURE)
        .expect("watch dir");
    w.watch(r_file, &target, ResourceKind::File, ClassSet::EMPTY)
        .expect("watch file");

    // Register the watcher's inotify fd with mio; `drain_ready` is non-blocking by trait, so the
    // caller blocks via the reactor.
    let mut poll = Poll::new().expect("mio Poll");
    let raw = w.as_fd().as_raw_fd();
    poll.registry()
        .register(&mut SourceFd(&raw), Token(0), Interest::READABLE)
        .expect("register inotify fd");

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
            if let WatcherEvent::Fs { resource, event } = ev {
                out.push((resource, event));
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
