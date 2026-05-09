//! `KqueueConfigWatcher` — kqueue-backed [`ConfigWatcher`] for the
//! daemon's own config file.
//!
//! Single-threaded: one thread owns the watcher value and calls
//! [`wait`](ConfigWatcher::wait) in a loop; the
//! [`wake_handle`](ConfigWatcher::wake_handle) is the only cross-thread
//! surface. The kqueue fd is `Arc`-shared with every wake handle so
//! drop of the watcher does not invalidate outstanding handles — a
//! stale `wake()` becomes a no-op-equivalent, never UB. (Same lifecycle
//! discipline as [`crate::kqueue::wake`].)
//!
//! # Watch shape
//!
//! Two registrations on the same kqueue fd:
//!
//! - **File fd** — registered against the canonicalised config file
//!   with the full file mask (`NOTE_WRITE | NOTE_EXTEND | NOTE_DELETE
//!   | NOTE_RENAME | NOTE_LINK | NOTE_REVOKE | NOTE_ATTRIB`). Catches
//!   in-place edits, atomic-rename source side, terminal flags
//!   (delete / move / revoke), and `chmod` / `chown` (NOTE_ATTRIB; the
//!   driver's lstat filter then sees the mode / ownership delta).
//!   `NOTE_ATTRIB` also fires on `setxattr` / `chflags` / `utimes` —
//!   noisy on macOS where LaunchServices writes
//!   `com.apple.lastuseddate#PS` on every Finder open / Quick Look —
//!   but the driver's `FileMeta` fingerprints only mode / uid / gid
//!   (not ctime), so those wakes collapse to a no-op at the
//!   convergence point. Dropped to `None` on any terminal flag.
//!
//! - **Parent dir fd** — registered with bare `NOTE_WRITE`. Fires on
//!   any dir-contents change (atomic save's rename, delete, recreate,
//!   sibling activity). Held for the watcher's lifetime; if it goes
//!   stale (parent dir moved / removed) the watcher is documented as
//!   restart-required.
//!
//! Kqueue events carry no name on the parent fd, so the watcher
//! cannot pre-classify "this was about my basename" without an extra
//! syscall. Every parent pulse becomes an `Ok(true)` return; the
//! driver's lstat-vs-`FileMeta` filter at the convergence point
//! suppresses no-op pulses.
//!
//! # Opportunistic re-open
//!
//! The file fd is dropped on terminal flags. The watcher does not own
//! a state machine for "waiting for recreate" — the next parent
//! `NOTE_WRITE` event simply triggers [`Self::try_reopen`]. Failure
//! (`ENOENT` because recreate hasn't happened yet, or any other errno)
//! logs at `warn!` and the next parent event retries. Two `if
//! self.file_fd.is_none() { ... }` lines, no state machine.
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `file_fd` (`Option<OwnedFd>`) — kernel auto-removes the vnode
//!   registration when the fd closes.
//! - `parent_fd` (`OwnedFd`) — same.
//! - `parent_path`, `config_basename` — heap drops, no syscalls.
//! - `kq` (`Arc<OwnedFd>`) — decrements; if last clone, the kqueue fd
//!   closes, kernel-reaping the `EVFILT_USER` ident and any queued
//!   events.
//!
//! Wake handles holding `Arc` clones keep the kqueue fd alive past the
//! watcher's drop — `wake()` from those becomes a no-op-equivalent
//! (no consumer drains the resulting trigger), with no UB.

use crate::ConfigWatcher;
use crate::WakeHandle;
use crate::kqueue::wake::KqueueWakeHandle;
use crate::kqueue::{fd, ffi};
use std::ffi::OsString;
use std::io;
use std::os::fd::OwnedFd;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Wake-up ident reserved on the watcher's kqueue `EVFILT_USER` filter.
///
/// kqueue keys events by `(ident, filter)`; this watcher's kqueue is a
/// fresh fd, separate from [`crate::kqueue::watcher::KqueueWatcher`]'s,
/// so the wake-ident namespace is independent. The distinct value
/// (vs. the engine watcher's `0xDEAD_BEEF`) is purely for debug-log
/// readability — there is no kernel-level collision risk to avoid.
const WAKE_IDENT: usize = 0xC0FF_EE00;

/// `udata` correlation tokens for the two vnode registrations.
///
/// Plain `1` / `2` — eye-catcher patterns would be noise here. The
/// engine watcher encodes a `ResourceId.as_ffi()` into `udata` and
/// lives on a *different* kqueue fd, so a `KeyData` round-trip
/// cannot collide with these constants.
const FILE_UDATA: u64 = 1;
const PARENT_UDATA: u64 = 2;

/// Per-FD vnode mask installed on the file watch.
///
/// Every flag relevant to a config file:
///
/// | Flag         | Why we listen                                        |
/// |--------------|------------------------------------------------------|
/// | `NOTE_WRITE` | content modification (in-place edit)                 |
/// | `NOTE_EXTEND`| append / truncate                                    |
/// | `NOTE_DELETE`| terminal — entry unlinked                            |
/// | `NOTE_RENAME`| terminal — entry renamed                             |
/// | `NOTE_LINK`  | hardlink count change (rare for configs, cheap)      |
/// | `NOTE_REVOKE`| terminal — fd revoked                                |
/// | `NOTE_ATTRIB`| `chmod` / `chown` — closes the recovery gap for      |
/// |              | `EACCES`-after-`chmod`. Also fires on `setxattr` /   |
/// |              | `chflags` / macOS LaunchServices `lastuseddate`      |
/// |              | xattr; the driver's lstat filter (mode + ownership)  |
/// |              | rejects those without a re-parse.                    |
const FILE_FFLAGS: u32 = libc::NOTE_WRITE
    | libc::NOTE_EXTEND
    | libc::NOTE_DELETE
    | libc::NOTE_RENAME
    | libc::NOTE_LINK
    | libc::NOTE_REVOKE
    | libc::NOTE_ATTRIB;

/// Per-FD vnode mask installed on the parent directory.
///
/// Bare `NOTE_WRITE` — fires on any dir-contents change (entry add /
/// remove / rename, including atomic save's rename and delete /
/// recreate). The terminal flags on the parent (`NOTE_DELETE` /
/// `NOTE_RENAME`) are intentionally *not* watched: parent-dir loss is
/// a documented restart-required limitation, and the silent-skip
/// behaviour matches the SIGHUP-only fallback.
const PARENT_FFLAGS: u32 = libc::NOTE_WRITE;

/// Maximum events drained per `kevent` syscall. The config watcher is
/// only deciding "any event vs. wake-only," so the batch size only
/// shapes how many drain syscalls a parent burst takes — 16 fits
/// every realistic editor pattern in one drain while staying small on
/// the stack (`16 * sizeof(libc::kevent) ≈ 512 bytes` on macOS).
const EVENT_BATCH: usize = 16;

/// kqueue-backed [`ConfigWatcher`] for the daemon's config file.
///
/// One file fd + one parent-dir fd registered on a fresh kqueue;
/// see the module docs for the full lifecycle and edge-case
/// matrix. Construct with [`Self::new`]; drive with
/// [`ConfigWatcher::wait`].
#[derive(Debug)]
pub struct KqueueConfigWatcher {
    /// File-side fd. Dropped to `None` on any terminal flag
    /// (`NOTE_DELETE` / `NOTE_RENAME` / `NOTE_REVOKE`); restored by
    /// [`Self::try_reopen`] on the next parent event. The kernel
    /// auto-removes the vnode registration when the fd closes, so a
    /// drop here is sufficient — no explicit deregister syscall.
    file_fd: Option<OwnedFd>,
    /// Parent-dir fd. Held for the watcher's lifetime; never dropped
    /// short of the watcher itself. If the parent inode is unlinked
    /// underneath us, this fd keeps the inode pinned but observes no
    /// further events — the documented restart-required path.
    ///
    /// The field is read by no method — it exists purely so its `Drop`
    /// closes the fd, which the kernel uses as the deregister signal
    /// for the vnode watch installed against it. Without storage in
    /// the struct the registration would be torn down at the end of
    /// `new()`.
    #[allow(dead_code)]
    parent_fd: OwnedFd,
    /// Canonicalised parent path. Used by [`Self::try_reopen`] to
    /// rebuild the full file path; held as `PathBuf` (not just an
    /// `Arc<Path>`) because tests + diagnostics want owned values
    /// readily.
    parent_path: PathBuf,
    /// Final path component of the canonical config path. Used by
    /// [`Self::try_reopen`] to rebuild the full file path. Stored as
    /// `OsString` to round-trip raw bytes losslessly across platforms.
    config_basename: OsString,
    /// `Arc` so wake handles can hold their own clones without
    /// borrowing from the watcher; drop of the last clone closes the
    /// kqueue fd. Mirrors [`crate::kqueue::watcher::KqueueWatcher`]'s
    /// kq sharing discipline.
    kq: Arc<OwnedFd>,
}

impl KqueueConfigWatcher {
    /// Construct a watcher bound to `path`.
    ///
    /// Steps (each failure is fatal — the bin warn-logs and falls
    /// back to SIGHUP-only):
    ///
    /// 1. `canonicalize(path)` — resolves every symlink. ELOOP on a
    ///    cyclic symlink, ENOENT if the file doesn't exist. Subsequent
    ///    leaf-symlink retargets after this point are documented as
    ///    restart-required.
    /// 2. Split into `parent_path` + `config_basename`. Pathological
    ///    paths (`/`, names with no file component) yield `InvalidInput`.
    /// 3. Create a fresh kqueue fd; register `EVFILT_USER` for the
    ///    wake ident.
    /// 4. Open the parent dir; register vnode with [`PARENT_FFLAGS`].
    /// 5. Open the config file; register vnode with [`FILE_FFLAGS`].
    ///    `ENOENT` here (TOCTOU after canonicalize) is non-fatal — we
    ///    leave `file_fd = None` and re-open on the next parent event.
    pub fn new(path: &Path) -> io::Result<Self> {
        let canonical = path.canonicalize()?;
        let parent_path = canonical
            .parent()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "config path has no parent directory",
                )
            })?
            .to_path_buf();
        let config_basename = canonical
            .file_name()
            .ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "config path has no file-name component",
                )
            })?
            .to_os_string();

        let kq = Arc::new(ffi::kqueue_new()?);
        ffi::register_user_event(&kq, WAKE_IDENT)?;

        let parent_fd = fd::open_for_watch(&parent_path)?;
        ffi::register_vnode(&kq, &parent_fd, PARENT_UDATA, PARENT_FFLAGS)?;

        // The TOCTOU window here is the price of `Config::from_path_with_meta`
        // running before the watcher: bytes captured atomically with
        // `FileMeta`, then watcher constructed. An edit landing between
        // the two collapses to "file vanished" → leave `file_fd = None`;
        // the bin's post-init `FileMeta::from_path` lstat compares
        // against the captured meta to drive an immediate
        // `reload_signal_tx` pulse if the on-disk state diverged.
        let file_fd = match fd::open_for_watch(&canonical) {
            Ok(fd) => {
                ffi::register_vnode(&kq, &fd, FILE_UDATA, FILE_FFLAGS)?;
                Some(fd)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };

        tracing::debug!(
            path = %canonical.display(),
            parent = %parent_path.display(),
            file_present = file_fd.is_some(),
            "kqueue config-watcher initialised"
        );

        Ok(Self {
            file_fd,
            parent_fd,
            parent_path,
            config_basename,
            kq,
        })
    }

    /// Attempt to re-open + re-register the file fd at the cached
    /// path. Idempotent: if `file_fd` is already `Some`, no-ops. Any
    /// failure other than `NotFound` is logged at `warn!`; the next
    /// parent event retries.
    fn try_reopen(&mut self) {
        if self.file_fd.is_some() {
            return;
        }
        let path = self.parent_path.join(&self.config_basename);
        match fd::open_for_watch(&path) {
            Ok(fd) => match ffi::register_vnode(&self.kq, &fd, FILE_UDATA, FILE_FFLAGS) {
                Ok(()) => {
                    tracing::debug!(
                        path = %path.display(),
                        "kqueue config-watcher reopened file"
                    );
                    self.file_fd = Some(fd);
                }
                Err(e) => {
                    tracing::warn!(
                        ?e,
                        path = %path.display(),
                        "kqueue config-watcher re-register failed",
                    );
                }
            },
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // Recreate hasn't materialised yet; the next parent
                // NOTE_WRITE event triggers another attempt.
            }
            Err(e) => {
                tracing::warn!(
                    ?e,
                    path = %path.display(),
                    "kqueue config-watcher re-open failed",
                );
            }
        }
    }
}

impl ConfigWatcher for KqueueConfigWatcher {
    /// Block on `kevent_drain` until events arrive (or the deadline
    /// elapses, or a wake fires). Per-event dispatch:
    ///
    /// - **Wake event** (`is_user_event(WAKE_IDENT)`) — skipped silently
    ///   without flipping `real_seen`. A wake-only return becomes
    ///   `Ok(false)`.
    /// - **`FILE_UDATA`** — `real_seen = true`. If any terminal flag
    ///   fires (`NOTE_DELETE` / `NOTE_RENAME` / `NOTE_REVOKE`), the
    ///   file fd is dropped. The kernel auto-removes the vnode
    ///   registration on close.
    /// - **`PARENT_UDATA`** — `real_seen = true`. If `file_fd` is
    ///   currently `None`, attempt [`Self::try_reopen`] in the same
    ///   pass — coalesced delete-recreate bursts therefore restore the
    ///   watch within a single drain.
    /// - **Other udata** — logged at `trace!`. Should not occur given
    ///   we only register the two known idents on this kqueue, but the
    ///   defensive arm beats a panic on a future kernel surprise.
    ///
    /// `EINTR` is retried inside [`ffi::kevent_drain`]. Any other
    /// `io::Error` propagates verbatim — the bin's wrapper logs at
    /// `error!` and exits the watcher thread; SIGHUP-only operation
    /// continues.
    fn wait(&mut self, deadline: Option<Instant>) -> io::Result<bool> {
        let timeout = deadline.map(|d| {
            let dur = d.saturating_duration_since(Instant::now());
            ffi::duration_to_timespec(dur)
        });
        let mut events = [ffi::Kevent::zeroed(); EVENT_BATCH];
        let n = ffi::kevent_drain(&self.kq, &mut events, timeout)?;

        let mut real_seen = false;
        for ev in &events[..n] {
            if ev.is_user_event(WAKE_IDENT) {
                continue;
            }
            match ev.udata() {
                FILE_UDATA => {
                    let f = ev.fflags();
                    if f & (libc::NOTE_DELETE | libc::NOTE_RENAME | libc::NOTE_REVOKE) != 0 {
                        // Drop the fd — kernel removes the vnode reg.
                        // try_reopen runs on the next parent event;
                        // this pass may already include that event in
                        // the same batch (delete + parent NOTE_WRITE
                        // coalesce), so the recovery can complete in
                        // one drain.
                        self.file_fd = None;
                    }
                    real_seen = true;
                }
                PARENT_UDATA => {
                    real_seen = true;
                    if self.file_fd.is_none() {
                        self.try_reopen();
                    }
                }
                other => {
                    tracing::trace!(
                        udata = format_args!("{other:#x}"),
                        "kqueue config-watcher: unexpected udata; dropped"
                    );
                }
            }
        }
        Ok(real_seen)
    }

    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(KqueueWakeHandle::new(Arc::clone(&self.kq), WAKE_IDENT))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfigWatcher;
    use std::fs;
    use std::os::unix::fs::symlink;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Watchdog deadline for `wait` calls in tests. Plenty of headroom
    /// for a kqueue drain on any sane CI host while still bounding a
    /// stuck test below CI's per-test timeout.
    fn watchdog() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    /// Drain pending events from the watcher until either nothing
    /// arrives within a short deadline or the cap is hit. Used to
    /// flush a startup TOCTOU pulse in delete-recreate / atomic-save
    /// tests where the act of opening + watching can race ahead of
    /// the test's edit (no event lands on a wait the test is about
    /// to issue, but a stale event from setup can land on the wait
    /// the test does issue).
    ///
    /// Returns on the first non-`Ok(true)` outcome — `Ok(false)`
    /// (deadline expired with nothing real, expected exit) or `Err`
    /// (test-fatal but caller decides; we just stop draining).
    fn drain_quiet<W: ConfigWatcher>(w: &mut W) {
        for _ in 0..16 {
            let deadline = Instant::now() + Duration::from_millis(20);
            if !matches!(w.wait(Some(deadline)), Ok(true)) {
                return;
            }
        }
    }

    #[test]
    fn wakes_on_in_place_edit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = KqueueConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        fs::write(&cfg, b"b").expect("in-place edit");

        let r = w.wait(Some(watchdog())).expect("wait ok");
        assert!(r, "in-place edit must wake the watcher (Ok(true))");
    }

    #[test]
    fn wakes_on_atomic_save() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = KqueueConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Write tempfile, then rename — the canonical editor save shape.
        let tmp = dir.path().join("specter.toml.tmp");
        fs::write(&tmp, b"b").expect("write tmp");
        fs::rename(&tmp, &cfg).expect("atomic rename");

        let r = w.wait(Some(watchdog())).expect("wait ok");
        assert!(r, "atomic save must wake the watcher (Ok(true))");
    }

    #[test]
    fn wakes_on_delete_recreate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = KqueueConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Phase 1: delete. File NOTE_DELETE drops file_fd; parent
        // NOTE_WRITE pulses. Either or both events satisfy Ok(true).
        fs::remove_file(&cfg).expect("unlink");
        assert!(
            w.wait(Some(watchdog())).expect("wait ok"),
            "delete must wake the watcher (Ok(true))"
        );
        assert!(w.file_fd.is_none(), "file_fd dropped after terminal flag");

        // Phase 2: recreate. Parent NOTE_WRITE pulses; try_reopen
        // succeeds inside the same wait().
        fs::write(&cfg, b"c").expect("recreate");
        assert!(
            w.wait(Some(watchdog())).expect("wait ok"),
            "recreate must wake the watcher (Ok(true))"
        );
        assert!(
            w.file_fd.is_some(),
            "file_fd reattached by try_reopen on parent event"
        );

        // Phase 3: post-reopen edit pulses through the new file fd —
        // proves the re-register actually rebound the vnode filter.
        drain_quiet(&mut w);
        fs::write(&cfg, b"d").expect("post-recreate edit");
        assert!(
            w.wait(Some(watchdog())).expect("wait ok"),
            "post-reopen edit must wake the watcher (Ok(true))"
        );
    }

    #[test]
    fn wake_handle_returns_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = KqueueConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        let wh = w.wake_handle();
        let fired = Arc::new(AtomicBool::new(false));
        let fired_thr = Arc::clone(&fired);
        let h = thread::Builder::new()
            .name("test-wake".into())
            .spawn(move || {
                thread::sleep(Duration::from_millis(50));
                fired_thr.store(true, Ordering::SeqCst);
                wh.wake();
            })
            .expect("spawn wake thread");

        let r = w.wait(Some(watchdog())).expect("wait ok");
        h.join().expect("wake thread join");

        assert!(
            fired.load(Ordering::SeqCst),
            "wake thread must have fired before watchdog"
        );
        assert!(
            !r,
            "wake-only return must be Ok(false) (no real fs event observed)"
        );
    }

    #[test]
    fn drops_on_init_when_path_resolves_to_symlink_loop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        // Two-link cycle: a → b, b → a. Either side canonicalised
        // resolves through enough hops to trip ELOOP on macOS / FreeBSD.
        symlink(&b, &a).expect("symlink a→b");
        symlink(&a, &b).expect("symlink b→a");

        let r = KqueueConfigWatcher::new(&a);
        let err = r.expect_err("symlink loop must fail watcher init");
        assert_eq!(
            err.raw_os_error(),
            Some(libc::ELOOP),
            "errno passes through unchanged from canonicalize"
        );
    }
}
