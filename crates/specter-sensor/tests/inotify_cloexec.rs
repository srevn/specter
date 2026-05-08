//! CLOEXEC discipline (§ 1.4 of the inotify port plan).
//!
//! The actuator's spawn path uses fork+exec; any fd without `CLOEXEC`
//! leaks into every spawned command. The watcher's three persistent
//! fds — `inotify_fd`, `wake_fd` (eventfd), `epoll_fd` — must all carry
//! the flag:
//!
//! - `inotify_fd`: leaked → child holds an unrelated inotify instance
//!   that prevents kernel-side cleanup at watcher drop.
//! - `wake_fd`: leaked → child can issue wakes (or block our wakes)
//!   nondeterministically; eventfd lifetime escapes the supervisor.
//! - `epoll_fd`: leaked → child inherits an epoll instance referencing
//!   the parent's inotify_fd / wake_fd; kernel-resource bloat per spawn.
//!
//! This test forks a child via `Command::new` with the actuator's
//! `pre_exec`-driven discipline (forces fork+exec on Linux), then reads
//! `/proc/<child_pid>/fd/` and asserts no symlink target matches the
//! `anon_inode:inotify` / `anon_inode:[eventfd]` /
//! `anon_inode:[eventpoll]` magic strings — the kernel-side proc class
//! for the watcher's three fds. (The kernel emits `inotify` without
//! brackets but `[eventfd]` / `[eventpoll]` with brackets; both shapes
//! are stable across modern kernels.)
//!
//! The child execs `/bin/sleep` with a brief argument so the
//! introspection window is stable.

#![cfg(target_os = "linux")]

use specter_sensor::{DrainWindow, FsWatcher, InotifyWatcher};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

/// Read every entry of `/proc/<pid>/fd/` and return the readlink target
/// for each. `None` ⇒ the directory cannot be read (child gone, or
/// `/proc` restricted by mount options).
fn child_fd_targets(pid: u32) -> Option<Vec<String>> {
    let dir = format!("/proc/{pid}/fd");
    let mut out = Vec::new();
    for entry in std::fs::read_dir(&dir).ok()?.flatten() {
        if let Ok(target) = std::fs::read_link(entry.path()) {
            out.push(target.to_string_lossy().into_owned());
        }
    }
    Some(out)
}

#[test]
fn watcher_fds_are_cloexec() {
    let watcher = InotifyWatcher::new(DrainWindow::default()).expect("InotifyWatcher::new");

    // Sanity: the parent itself must hold the watcher's three fd
    // classes right now (otherwise the test premise is broken).
    let parent_targets = child_fd_targets(std::process::id()).expect("read /proc/self/fd");
    for marker in [
        "anon_inode:inotify",
        "anon_inode:[eventfd]",
        "anon_inode:[eventpoll]",
    ] {
        assert!(
            parent_targets.iter().any(|t| t == marker),
            "parent must hold an `{marker}` fd from InotifyWatcher; \
             got parent fds: {parent_targets:?}"
        );
    }

    // Spawn-and-introspect a long-running child. The pre_exec hook
    // mirrors `OsSpawner::spawn` so the spawn path is fork+exec rather
    // than posix_spawn (the actuator's production discipline).
    let mut cmd = Command::new("/bin/sleep");
    cmd.arg("2")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: pre_exec hook is an empty `Ok(())`; matches the
    // actuator's `OsSpawner::spawn` discipline. The hook performs no
    // I/O, no allocation, no signal-unsafe work.
    #[allow(unsafe_code)]
    unsafe {
        cmd.pre_exec(|| Ok(()));
    }
    let mut child = cmd.spawn().expect("spawn /bin/sleep");
    let child_pid = child.id();

    // Give the child a moment to reach `sleep` (post-exec). Without
    // the sleep, `/proc/<pid>/fd/` may briefly show pre-exec fd table
    // contents from the fork.
    thread::sleep(Duration::from_millis(100));

    let child_targets =
        child_fd_targets(child_pid).expect("read /proc/<child>/fd; child should still be alive");

    // Assert no inherited anon-inode fd of the watcher's classes.
    for marker in [
        "anon_inode:inotify",
        "anon_inode:[eventfd]",
        "anon_inode:[eventpoll]",
    ] {
        assert!(
            !child_targets.iter().any(|t| t == marker),
            "child inherited an `{marker}` fd; CLOEXEC discipline broken — \
             leak in InotifyWatcher's fd open path. child fds: {child_targets:?}"
        );
    }

    // Reap the child cleanly (avoid zombies in CI).
    let _ = child.kill();
    let _ = child.wait();
    drop(watcher);
}

/// Sanity check: the watcher's wake handle survives an actuator-style
/// fork+exec without UB. The watcher's fds must not leak into the
/// child (covered by the previous test); the wake handle's `Arc` must
/// continue to function in the parent afterwards.
#[test]
fn wake_handle_survives_actuator_style_spawn() {
    let watcher = InotifyWatcher::new(DrainWindow::default()).unwrap();
    let wake = watcher.wake_handle();

    let mut cmd = Command::new("/bin/true");
    cmd.stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: see `watcher_fds_are_cloexec`.
    #[allow(unsafe_code)]
    unsafe {
        cmd.pre_exec(|| Ok(()));
    }
    let _ = cmd.spawn().unwrap().wait();

    // Wake the watcher post-spawn. No panic, no UB.
    wake.wake();

    drop(watcher);
}
