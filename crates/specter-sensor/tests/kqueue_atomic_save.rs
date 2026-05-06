//! Atomic save: `cp file file.tmp; mv file.tmp file` replaces the
//! destination file's inode atomically. The watched file fd's old
//! inode is unlinked → terminal `FsEvent` (Removed or Renamed,
//! depending on kernel-side semantics); the watched dir fd sees a
//! `StructureChanged`.
//!
//! The engine vacates the slot on the terminal event; the next dir
//! `StructureChanged` triggers a reprobe that discovers the new file
//! at the same path with a fresh `ResourceId`.

#![cfg(any(target_os = "macos", target_os = "freebsd"))]

use slotmap::SlotMap;
use specter_core::{ClassSet, FsEvent, ResourceId, WatchOpts};
use specter_sensor::{FsWatcher, KqueueWatcher};
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
    w.watch(
        r_dir,
        tmp.path(),
        WatchOpts {
            events: ClassSet::STRUCTURE,
        },
    )
    .expect("watch dir");
    w.watch(r_file, &target, WatchOpts::default())
        .expect("watch file");

    // Atomic save: write staging file, then rename over target.
    std::fs::write(&staging, "v2 with more content").unwrap();
    std::fs::rename(&staging, &target).unwrap();

    let deadline = Instant::now() + Duration::from_secs(2);
    let mut out = Vec::new();
    while Instant::now() < deadline {
        let _ = w.poll_until(Some(Instant::now() + Duration::from_millis(50)), &mut out);
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
