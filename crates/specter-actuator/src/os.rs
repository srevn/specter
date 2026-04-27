//! Production [`Spawner`] impl using `std::process::Command` +
//! `nix::sys::signal`.
//!
//! Stdin/stdout/stderr are routed to `/dev/null` (v1's `log_output =
//! false`). cwd is validated by `Command::spawn` at spawn time;
//! failure surfaces as an `io::Result::Err`.
//!
//! `Command::spawn` is forced down the fork+exec path via a no-op
//! [`CommandExt::pre_exec`] hook (see `OsSpawner::spawn`'s safety note).
//! macOS `posix_spawn` — the default Rust std fast path on Darwin —
//! returns `EBADF` when the parent process has more than ~10,200 open
//! file descriptors (the kernel's `OPEN_MAX = 10240`); the kqueue
//! watcher opens one `O_EVTONLY` fd per watched directory, so trees
//! with more than ~10k directories trip it. fork+exec has no such
//! cap and is the load-bearing fix for deep-tree workloads.
//!
//! The PID-reuse race during shutdown signaling is closed in two layers:
//! a shared `Arc<AtomicBool>` flag set by [`OsChildWaiter::wait`] before
//! returning, plus ESRCH-collapse at the syscall layer.

use crate::spawner::{ChildSignaler, ChildWaiter, SpawnHandles, Spawner};
use specter_core::EffectOutcome;
use std::io;
use std::os::unix::process::{CommandExt, ExitStatusExt};
use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

/// Production `Spawner`. Spawns via `std::process::Command`. Stdin/
/// stdout/stderr → `/dev/null` (v1 `log_output = false`). cwd
/// is passed to `Command::current_dir` and validated at spawn time.
#[derive(Debug, Default)]
pub struct OsSpawner;

impl OsSpawner {
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Spawner for OsSpawner {
    fn spawn(
        &self,
        argv: &[String],
        env: &[(String, String)],
        cwd: &Path,
    ) -> io::Result<SpawnHandles> {
        if argv.is_empty() {
            return Err(io::Error::new(io::ErrorKind::InvalidInput, "argv is empty"));
        }
        let mut cmd = Command::new(&argv[0]);
        cmd.args(&argv[1..])
            .envs(env.iter().map(|(k, v)| (k.as_str(), v.as_str())))
            .current_dir(cwd)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .stderr(Stdio::null());
        // SAFETY: the pre_exec hook is an empty `Ok(())` — it performs no
        // I/O, no allocation, no signal-unsafe work. Its sole purpose is
        // to disqualify Rust std's `posix_spawn` fast path (which
        // requires no pre_exec hook) so the spawn falls back to
        // fork+exec. macOS `posix_spawn` returns EBADF once the parent
        // crosses ~10,200 open file descriptors (the kernel's
        // `OPEN_MAX`); the kqueue watcher opens one fd per watched
        // directory, and trees of ~10k+ directories therefore trip the
        // limit on the very first Effect spawn. fork+exec iterates the
        // child's fd table without that cap.
        #[allow(unsafe_code)]
        unsafe {
            cmd.pre_exec(|| Ok(()));
        }
        let child = cmd.spawn()?;
        let pid = child.id();
        let dead = Arc::new(AtomicBool::new(false));
        Ok(SpawnHandles {
            pid,
            waiter: Box::new(OsChildWaiter {
                child,
                dead: Arc::clone(&dead),
            }),
            signaler: Box::new(OsChildSignaler { pid, dead }),
        })
    }
}

struct OsChildWaiter {
    child: Child,
    dead: Arc<AtomicBool>,
}

impl ChildWaiter for OsChildWaiter {
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
        let mut child = self.child;
        let result = child.wait();
        // Set dead unconditionally before returning, so the controller
        // sees a coherent "child reaped, signals are no-ops" state
        // regardless of wait success or failure (closes PID-reuse race
        // at the protocol layer; ESRCH-collapse is the syscall fallback).
        self.dead.store(true, Ordering::SeqCst);
        let status = result?;
        Ok(if status.success() {
            EffectOutcome::Ok
        } else if let Some(sig) = status.signal() {
            EffectOutcome::Failed {
                exit_code: None,
                signal: Some(sig),
            }
        } else {
            EffectOutcome::Failed {
                exit_code: status.code(),
                signal: None,
            }
        })
    }
}

struct OsChildSignaler {
    pid: u32,
    dead: Arc<AtomicBool>,
}

impl ChildSignaler for OsChildSignaler {
    fn signal_term(&self) -> io::Result<()> {
        if self.dead.load(Ordering::SeqCst) {
            return Ok(());
        }
        signal_pid(self.pid, nix::sys::signal::Signal::SIGTERM)
    }
    fn signal_kill(&self) -> io::Result<()> {
        if self.dead.load(Ordering::SeqCst) {
            return Ok(());
        }
        signal_pid(self.pid, nix::sys::signal::Signal::SIGKILL)
    }
}

#[allow(clippy::cast_possible_wrap)]
fn signal_pid(pid: u32, sig: nix::sys::signal::Signal) -> io::Result<()> {
    use nix::errno::Errno;
    use nix::unistd::Pid;
    let pid = Pid::from_raw(pid as i32);
    match nix::sys::signal::kill(pid, sig) {
        Ok(()) => Ok(()),
        Err(Errno::ESRCH) => Ok(()), // already gone
        Err(e) => Err(io::Error::from_raw_os_error(e as i32)),
    }
}

#[cfg(test)]
mod tests {
    #![allow(unsafe_code)] // mirrors the production site (`os.rs::OsSpawner::spawn`); the test exercises it.

    use super::*;
    use crate::spawner::Spawner;
    use std::os::fd::OwnedFd;
    use std::path::Path;

    /// macOS `posix_spawn` returns `EBADF` once the parent process holds more
    /// than ~10,200 open file descriptors (the kernel's `OPEN_MAX = 10240`).
    /// Specter's kqueue watcher opens one `O_EVTONLY` fd per watched
    /// directory, so trees with ~10k+ directories trip this limit on the
    /// first Effect spawn — the user-visible symptom is "deep tree, file
    /// changed, command silently never runs". The fix routes spawn through
    /// fork+exec via a no-op `pre_exec` hook.
    ///
    /// This test pre-opens enough `O_EVTONLY` fds to push the process across
    /// the `OPEN_MAX` boundary, then asserts that `OsSpawner::spawn`
    /// succeeds. macOS-only: Linux/glibc `posix_spawn` is implemented as
    /// fork+exec under the hood and has no equivalent cap, so the test
    /// would be a no-op there (and would simply burn ~10k fds).
    #[cfg(target_os = "macos")]
    #[test]
    fn spawn_succeeds_above_macos_posix_spawn_open_max() {
        // The kernel's `OPEN_MAX` is 10240 on every supported macOS version.
        // Open `OPEN_MAX + headroom` fds so we are unambiguously past the
        // failure threshold for the legacy posix_spawn path; even if a
        // future macOS update raises the limit, this test still exercises
        // the fork+exec route at scale.
        const FDS_TO_OPEN: usize = 10_500;

        // The process's `RLIMIT_NOFILE` may be lower than what we need;
        // skip cleanly if so rather than failing for an environment reason
        // unrelated to the spawn behavior we want to assert. CI on macOS
        // typically allows 16k or more.
        let nofile_soft = unsafe {
            let mut rlim: libc::rlimit = std::mem::zeroed();
            if libc::getrlimit(libc::RLIMIT_NOFILE, &raw mut rlim) != 0 {
                return;
            }
            rlim.rlim_cur
        };
        if nofile_soft < FDS_TO_OPEN as u64 + 256 {
            eprintln!(
                "skipping spawn_succeeds_above_macos_posix_spawn_open_max: \
                 RLIMIT_NOFILE soft = {nofile_soft}, need >= {}",
                FDS_TO_OPEN + 256,
            );
            return;
        }

        // Open `FDS_TO_OPEN` directory fds with `O_EVTONLY`, the same flag
        // the kqueue watcher uses. We stat any always-present path; the fd
        // count is what matters, not what's behind it.
        let cstr = std::ffi::CString::new("/").unwrap();
        let mut fds: Vec<OwnedFd> = Vec::with_capacity(FDS_TO_OPEN);
        let o_evtonly: i32 = 0x8000;
        for _ in 0..FDS_TO_OPEN {
            let raw = unsafe { libc::open(cstr.as_ptr(), o_evtonly) };
            if raw < 0 {
                // If we couldn't open enough fds (RLIMIT_NOFILE, EMFILE),
                // skip — the test's premise (cross OPEN_MAX) hasn't been met.
                eprintln!(
                    "skipping spawn_succeeds_above_macos_posix_spawn_open_max: \
                     open() failed at fd #{} (errno={})",
                    fds.len(),
                    std::io::Error::last_os_error(),
                );
                return;
            }
            fds.push(unsafe { <OwnedFd as std::os::fd::FromRawFd>::from_raw_fd(raw) });
        }
        assert!(
            fds.len() >= FDS_TO_OPEN,
            "must open enough fds to trip OPEN_MAX"
        );

        // The actual assertion: `OsSpawner::spawn` must succeed. Without the
        // `pre_exec` hook, Rust std would route through posix_spawn and fail
        // with EBADF here. With the hook, fork+exec is used and succeeds.
        let spawner = OsSpawner::new();
        let cwd = Path::new("/tmp");
        let argv: Vec<String> = vec!["/bin/sh".into(), "-c".into(), "exit 0".into()];
        let env: Vec<(String, String)> = Vec::new();
        let handles = spawner
            .spawn(&argv, &env, cwd)
            .expect("spawn must succeed under high fd pressure (fork+exec route)");
        let outcome = handles
            .waiter
            .wait()
            .expect("wait must succeed for a spawned child");
        assert_eq!(
            outcome,
            EffectOutcome::Ok,
            "child exited cleanly; outcome should be Ok",
        );

        // Drop the OwnedFds explicitly; closing 10k+ fds at end-of-test
        // adds visible time to the test runner output and we'd rather log
        // the close-time once than have it linger in `Drop`.
        drop(fds);
    }
}
