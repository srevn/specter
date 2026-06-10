//! CLOEXEC discipline.
//!
//! The actuator's spawn path uses fork+exec; any fd without `CLOEXEC` leaks into every spawned
//! command. The watcher's single persistent fd — `inotify_fd` — must carry the flag: if leaked, the
//! child holds an unrelated inotify instance that prevents kernel-side cleanup at watcher drop.
//! (Wake / reactor fds live on the caller's side under the mio integration and are not the
//! watcher's responsibility.)
//!
//! This test forks a child via `Command::new` with the actuator's `pre_exec`-driven discipline
//! (forces fork+exec on Linux), then reads `/proc/<child_pid>/fd/` and asserts no symlink target
//! matches the `anon_inode:inotify` magic string — the kernel-side proc class for the watcher's fd.
//! The child execs `/bin/sleep` with a brief argument so the introspection window is stable.

#![cfg(target_os = "linux")]

use specter_sensor::InotifyWatcher;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

/// Read every entry of `/proc/<pid>/fd/` and return the readlink target for each. `None` ⇒ the
/// directory cannot be read (child gone, or `/proc` restricted by mount options).
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
    let watcher = InotifyWatcher::new().expect("InotifyWatcher::new");

    // Sanity: the parent itself must hold an `anon_inode:inotify` fd right now (otherwise the test
    // premise is broken).
    let parent_targets = child_fd_targets(std::process::id()).expect("read /proc/self/fd");
    assert!(
        parent_targets.iter().any(|t| t == "anon_inode:inotify"),
        "parent must hold an `anon_inode:inotify` fd from InotifyWatcher; \
         got parent fds: {parent_targets:?}"
    );

    // Spawn-and-introspect a long-running child. The pre_exec hook mirrors `OsSpawner::spawn` so
    // the spawn path is fork+exec rather than posix_spawn (the actuator's production discipline).
    let mut cmd = Command::new("/bin/sleep");
    cmd.arg("2")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // SAFETY: pre_exec hook is an empty `Ok(())`; matches the actuator's `OsSpawner::spawn`
    // discipline. The hook performs no I/O, no allocation, no signal-unsafe work.
    #[allow(unsafe_code)]
    unsafe {
        cmd.pre_exec(|| Ok(()));
    }
    let mut child = cmd.spawn().expect("spawn /bin/sleep");
    let child_pid = child.id();

    // Give the child a moment to reach `sleep` (post-exec). Without the sleep, `/proc/<pid>/fd/`
    // may briefly show pre-exec fd table contents from the fork.
    thread::sleep(Duration::from_millis(100));

    let child_targets =
        child_fd_targets(child_pid).expect("read /proc/<child>/fd; child should still be alive");

    // Assert no inherited inotify fd. The watcher's `inotify_fd` is the only persistent kernel
    // resource it owns under the mio integration; if it leaks, the actuator's spawn discipline is
    // broken at `inotify_init1`'s CLOEXEC argument.
    assert!(
        !child_targets.iter().any(|t| t == "anon_inode:inotify"),
        "child inherited an `anon_inode:inotify` fd; CLOEXEC discipline broken — \
         leak in InotifyWatcher's fd open path. child fds: {child_targets:?}"
    );

    // Reap the child cleanly (avoid zombies in CI).
    let _ = child.kill();
    let _ = child.wait();
    drop(watcher);
}
