//! `KqueueConfigWatcher` — kqueue-backed [`ConfigWatcher`] for the daemon's own config file.
//!
//! Single-threaded: one thread owns the watcher value and drives
//! [`drain_ready`](ConfigWatcher::drain_ready). The kqueue fd is exposed via [`AsFd`] so a reactor
//! can register it for edge- triggered readiness; `drain_ready` is non-blocking and drains the
//! kqueue to empty on every call.
//!
//! # Watch shape
//!
//! Two registrations on the same kqueue fd:
//!
//! - **File fd** — registered against the canonicalised config file with the full file mask
//!   (`NOTE_WRITE | NOTE_EXTEND | NOTE_DELETE | NOTE_RENAME | NOTE_LINK | NOTE_REVOKE |
//!   NOTE_ATTRIB`). Catches in-place edits, atomic-rename source side, terminal flags (delete /
//!   move / revoke), and `chmod` / `chown` (NOTE_ATTRIB; the driver's lstat filter then sees the
//!   mode / ownership delta). `NOTE_ATTRIB` also fires on `setxattr` / `chflags` / `utimes` — noisy
//!   on macOS where LaunchServices writes `com.apple.lastuseddate#PS` on every Finder open / Quick
//!   Look — but the driver's `FileMeta` fingerprints only mode / uid / gid (not ctime), so those
//!   wakes collapse to a no-op at the convergence point. Dropped to `None` on any terminal flag.
//!
//! - **Parent dir fd** — registered with bare `NOTE_WRITE`. Fires on any dir-contents change
//!   (atomic save's rename, delete, recreate, sibling activity). Held for the watcher's lifetime;
//!   if it goes stale (parent dir moved / removed) the watcher is documented as restart-required.
//!
//! Kqueue events carry no name on the parent fd, so the watcher cannot pre-classify "this was about
//! my basename" without an extra syscall. Every parent pulse becomes an `Ok(true)` return; the
//! driver's lstat-vs-`FileMeta` filter at the convergence point suppresses no-op pulses.
//!
//! # Opportunistic re-open
//!
//! The file fd is dropped on terminal flags. The watcher does not own a state machine for "waiting
//! for recreate" — after the drain-to-empty loop completes, [`ConfigWatcher::drain_ready`] checks
//! the *final* `file_fd` state and calls `try_reopen` iff a real event landed and the fd is now
//! `None`. The decision is post-loop-of-loops, so the kqueue-unspecified intra-batch order of the
//! parent's `NOTE_WRITE` and the file's terminal `NOTE_DELETE`/`NOTE_RENAME`/`NOTE_REVOKE` is
//! irrelevant, and so is the kernel splitting one logical burst across two `kevent` drains within a
//! single `drain_ready` invocation: every ordering converges on the same post-loop state.
//! `try_reopen` is idempotent (`is_some()` short-circuit) and `ENOENT`-fast, so a call on a
//! not-yet-recreated file is cheap and the next real event retries.
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `file_fd` (`Option<OwnedFd>`) — kernel auto-removes the vnode registration when the fd closes.
//! - `parent_fd` ([`OwnedFd`]) — same.
//! - `parent_path`, `config_basename` — heap drops, no syscalls.
//! - `kq` ([`OwnedFd`]) closes, kernel-reaping any queued events.

use crate::ConfigWatcher;
use crate::kqueue::ffi;
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::path::{Path, PathBuf};

/// `udata` correlation tokens for the two vnode registrations.
///
/// Plain `1` / `2` — eye-catcher patterns would be noise here. The engine watcher encodes a
/// `ResourceId.as_ffi()` into `udata` and lives on a *different* kqueue fd, so a `KeyData`
/// round-trip cannot collide with these constants.
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
/// Bare `NOTE_WRITE` — fires on any dir-contents change (entry add / remove / rename, including
/// atomic save's rename and delete / recreate). The terminal flags on the parent (`NOTE_DELETE` /
/// `NOTE_RENAME`) are intentionally *not* watched: parent-dir loss is a documented restart-required
/// limitation, and the silent-skip behaviour matches the SIGHUP-only fallback.
const PARENT_FFLAGS: u32 = libc::NOTE_WRITE;

/// Maximum events drained per `kevent` syscall. The config watcher is only deciding "any event vs.
/// wake-only," so the batch size only shapes how many drain syscalls a parent burst takes — 16 fits
/// every realistic editor pattern in one drain while staying small on the stack (`16 *
/// sizeof(libc::kevent) ≈ 512 bytes` on macOS).
const EVENT_BATCH: usize = 16;

/// kqueue-backed [`ConfigWatcher`] for the daemon's config file.
///
/// One file fd + one parent-dir fd registered on a fresh kqueue; see the module docs for the full
/// lifecycle and edge-case matrix. Construct with [`Self::new`]; drive with
/// [`ConfigWatcher::drain_ready`].
#[derive(Debug)]
pub struct KqueueConfigWatcher {
    /// File-side fd. Dropped to `None` on any terminal flag (`NOTE_DELETE` / `NOTE_RENAME` /
    /// `NOTE_REVOKE`); restored by [`Self::try_reopen`] on the next parent event. The kernel
    /// auto-removes the vnode registration when the fd closes, so a drop here is sufficient — no
    /// explicit deregister syscall.
    file_fd: Option<OwnedFd>,
    /// Parent-dir fd. Held for the watcher's lifetime; never dropped short of the watcher itself.
    /// If the parent inode is unlinked underneath us, this fd keeps the inode pinned but observes
    /// no further events — the documented restart-required path.
    ///
    /// The field is read by no method — it exists purely so its `Drop` closes the fd, which the
    /// kernel uses as the deregister signal for the vnode watch installed against it. Without
    /// storage in the struct the registration would be torn down at the end of `new()`.
    #[allow(dead_code)]
    parent_fd: OwnedFd,
    /// Canonicalised parent path. Used by [`Self::try_reopen`] to rebuild the full file path; held as
    /// `PathBuf` (not just an `Arc<Path>`) because tests + diagnostics want owned values readily.
    parent_path: PathBuf,
    /// Final path component of the canonical config path. Used by [`Self::try_reopen`] to rebuild the
    /// full file path. Stored as `OsString` to round-trip raw bytes losslessly across platforms.
    config_basename: OsString,
    /// The kqueue fd. Exposed through [`AsFd`] so a reactor can register it for edge-triggered
    /// readiness; drop closes the fd and kernel-reaps any queued events.
    kq: OwnedFd,
}

impl KqueueConfigWatcher {
    /// Construct a watcher bound to `path`.
    ///
    /// Steps (each failure is fatal — the bin warn-logs and falls back to SIGHUP-only):
    ///
    /// 1. `canonicalize(path)` — resolves every symlink. ELOOP on a cyclic symlink, ENOENT if the
    ///    file doesn't exist. Subsequent leaf-symlink retargets after this point are documented as
    ///    restart-required.
    /// 2. Split into `parent_path` + `config_basename`. Pathological paths (`/`, names with no file
    ///    component) yield `InvalidInput`.
    /// 3. Create a fresh kqueue fd.
    /// 4. Open the parent dir; register vnode with `PARENT_FFLAGS`.
    /// 5. Open the config file; register vnode with `FILE_FFLAGS`. `ENOENT` here (TOCTOU after
    ///    canonicalize) is non-fatal — we leave `file_fd = None` and re-open on the next parent
    ///    event.
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

        let kq = ffi::kqueue_new()?;

        let parent_fd = ffi::open_for_watch(&parent_path)?;
        ffi::register_vnode(&kq, &parent_fd, PARENT_UDATA, PARENT_FFLAGS)?;

        // The TOCTOU window here is the price of `Config::from_path_with_meta` running before the
        // watcher: bytes captured atomically with `FileMeta`, then watcher constructed. An edit
        // landing between the two collapses to "file vanished" → leave `file_fd = None`; the bin's
        // post-init `FileMeta::from_path` lstat compares against the captured meta to drive an
        // immediate `reload_signal_tx` pulse if the on-disk state diverged.
        let file_fd = match ffi::open_for_watch(&canonical) {
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

    /// Attempt to re-open + re-register the file fd at the cached path. Idempotent: if `file_fd` is
    /// already `Some`, no-ops. Any failure other than `NotFound` is logged at `warn!`; the next
    /// parent event retries.
    fn try_reopen(&mut self) {
        if self.file_fd.is_some() {
            return;
        }
        let path = self.parent_path.join(&self.config_basename);
        match ffi::open_for_watch(&path) {
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
                // Recreate hasn't materialised yet; the next parent NOTE_WRITE event triggers
                // another attempt.
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
    /// Non-blocking drain-to-empty of the kqueue queue. Loops on `ffi::kevent_drain` (zero
    /// `timespec`) until the kernel returns `0`, dispatching each record:
    ///
    /// - **`FILE_UDATA`** — `real_seen = true`. If any terminal flag fires (`NOTE_DELETE` /
    ///   `NOTE_RENAME` / `NOTE_REVOKE`), the file fd is dropped. The kernel auto-removes the vnode
    ///   registration on close.
    /// - **`PARENT_UDATA`** — `real_seen = true`. No inline reopen — see the post-loop recovery
    ///   below.
    /// - **Other udata** — logged at `trace!`. Should not occur given we only register the two known
    ///   idents on this kqueue, but the defensive arm beats a panic on a future kernel surprise.
    ///
    /// After the drain-to-empty loop, `try_reopen` runs iff a real event landed *and* the final
    /// `file_fd` state is `None`. The recovery decision lives outside the loop on purpose: the
    /// kernel may split one logical atomic-save burst across two `kevent` batches (terminal flag in
    /// batch *k*, parent `NOTE_WRITE` in batch *k+1*), and an inside-the-loop `try_reopen` would
    /// race the in-progress inode swap. The post-loop placement makes the decision against the
    /// invocation's final state, independent of both the kqueue-unspecified intra-batch order and
    /// the kernel's across-batch fragmentation.
    ///
    /// `EINTR` is retried inside `ffi::kevent_drain`. Any other `io::Error` propagates verbatim —
    /// the caller logs at `error!` and exits the watcher loop; SIGHUP-only operation continues.
    fn drain_ready(&mut self) -> io::Result<bool> {
        let mut events = [ffi::Kevent::zeroed(); EVENT_BATCH];
        let mut real_seen = false;
        loop {
            let n = ffi::kevent_drain(&self.kq, &mut events)?;
            if n == 0 {
                // Kernel queue drained; edge-triggered contract satisfied.
                break;
            }
            for ev in &events[..n] {
                match ev.udata() {
                    FILE_UDATA => {
                        let f = ev.fflags();
                        if f & (libc::NOTE_DELETE | libc::NOTE_RENAME | libc::NOTE_REVOKE) != 0 {
                            // Drop the fd — kernel removes the vnode reg. The post-loop-of-loops
                            // recovery restores it against the invocation's final state.
                            self.file_fd = None;
                        }
                        real_seen = true;
                    }
                    PARENT_UDATA => {
                        real_seen = true;
                    }
                    other => {
                        tracing::trace!(
                            udata = format_args!("{other:#x}"),
                            "kqueue config-watcher: unexpected udata; dropped"
                        );
                    }
                }
            }
        }

        // Order-independent recovery: decide against the *final* state of `file_fd` after the
        // entire drain-to-empty loop. Parent-knote-before-file-knote, file-knote-before-parent-
        // knote, both, neither, fragmented across kernel batches — every ordering converges here.
        // `try_reopen` is idempotent and `ENOENT`-fast, so a call on a not-yet-recreated file is
        // cheap and the next real event retries.
        if real_seen && self.file_fd.is_none() {
            self.try_reopen();
        }
        Ok(real_seen)
    }
}

impl AsFd for KqueueConfigWatcher {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.kq.as_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfigWatcher;
    use std::fs;
    use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
    use std::os::unix::fs::symlink;
    use std::time::{Duration, Instant};

    /// Watchdog deadline for drain calls in tests. Plenty of headroom for a kqueue drain on any
    /// sane CI host while still bounding a stuck test below CI's per-test timeout.
    fn watchdog() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    /// Block on `libc::poll(POLLIN, ...)` for `fd` until readable or `deadline` elapses. Returns
    /// `Ok(true)` on readable, `Ok(false)` on timeout, `Err` on syscall error. Retries `EINTR`
    /// internally with per-iteration remaining-budget recompute.
    ///
    /// This is the test-side substitute for the watcher's old `wait(Some(deadline))` block: the
    /// watcher is non-blocking and exposes [`AsFd`], so a test that needs to "wait for readability
    /// under a watchdog" polls the fd directly.
    #[allow(unsafe_code)]
    fn wait_fd_readable_until(fd: BorrowedFd<'_>, deadline: Instant) -> io::Result<bool> {
        loop {
            let remaining = deadline.saturating_duration_since(Instant::now());
            let timeout_ms = i32::try_from(remaining.as_millis()).unwrap_or(i32::MAX);
            let mut pfd = libc::pollfd {
                fd: fd.as_raw_fd(),
                events: libc::POLLIN,
                revents: 0,
            };
            // SAFETY: `pfd` is a stack binding holding one valid `pollfd`; the pointer is live for
            // the duration of the syscall. `1` matches the slice length.
            let n = unsafe { libc::poll(std::ptr::from_mut(&mut pfd), 1, timeout_ms) };
            if n > 0 {
                return Ok(pfd.revents & libc::POLLIN != 0);
            }
            if n == 0 {
                return Ok(false);
            }
            let e = io::Error::last_os_error();
            if e.raw_os_error() == Some(libc::EINTR) {
                continue;
            }
            return Err(e);
        }
    }

    /// One pulse-or-timeout step: wait for the watcher's fd to become readable under `deadline`,
    /// then drain. Returns `Ok(false)` on timeout (caller asserts a wake was expected), `Ok(true)`
    /// on a substantive drain, `Err` on syscall error.
    fn fire_drain<W: ConfigWatcher>(w: &mut W, deadline: Instant) -> io::Result<bool> {
        if !wait_fd_readable_until(w.as_fd(), deadline)? {
            return Ok(false);
        }
        w.drain_ready()
    }

    /// Drain pending events from the watcher until either nothing arrives within a short deadline
    /// or the cap is hit. Used to flush a startup TOCTOU pulse in delete-recreate / atomic-save
    /// tests where the act of opening + watching can race ahead of the test's edit.
    ///
    /// Returns on the first non-`Ok(true)` outcome — `Ok(false)` (deadline expired with nothing
    /// real, expected exit) or `Err` (caller-fatal; we just stop draining).
    fn drain_quiet<W: ConfigWatcher>(w: &mut W) {
        for _ in 0..16 {
            let deadline = Instant::now() + Duration::from_millis(20);
            if !matches!(fire_drain(w, deadline), Ok(true)) {
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

        let r = fire_drain(&mut w, watchdog()).expect("drain ok");
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

        let r = fire_drain(&mut w, watchdog()).expect("drain ok");
        assert!(r, "atomic save must wake the watcher (Ok(true))");

        // Post-loop recovery against the invocation's final state: a coalesced delete + parent
        // NOTE_WRITE batch restores `file_fd` in this same drain regardless of intra-batch
        // ordering. A kernel-split fragment can leave `file_fd` holding a stale (but `Some`) fd;
        // the end-to-end `atomic_save_then_in_place_edit` test catches that mode.
        assert!(w.file_fd.is_some(), "atomic save must leave a live file fd");
    }

    /// End-to-end check that an atomic save followed by an in-place edit on the recreated file
    /// surfaces a second pulse. Requires the post-loop recovery to install a live file fd on the
    /// new inode against any intra-batch ordering of the file's terminal knote and the parent's
    /// `NOTE_WRITE`; a stranded `file_fd = None` (or a stale fd on the dead inode) would silently
    /// swallow the in-place edit and block the second drain past the watchdog.
    #[test]
    fn atomic_save_then_in_place_edit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = KqueueConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Atomic save: write staging, rename over. Both knotes (file terminal + parent NOTE_WRITE)
        // fire on this kqueue in kernel-unspecified relative order.
        let tmp = dir.path().join("specter.toml.tmp");
        fs::write(&tmp, b"b").expect("write tmp");
        fs::rename(&tmp, &cfg).expect("atomic rename");

        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
            "atomic save must wake the watcher (Ok(true))"
        );
        // Drain any trailing fragments so subsequent state reflects the watcher's settled post-save
        // view.
        drain_quiet(&mut w);
        assert!(
            w.file_fd.is_some(),
            "post-loop reopen must restore the file fd on the new inode"
        );

        // In-place edit on the recreated file pulses iff the file fd points at the new inode
        // (rather than the unlinked old one or being null).
        fs::write(&cfg, b"c").expect("in-place edit on recreated file");
        let r = fire_drain(&mut w, watchdog()).expect("drain ok");
        assert!(
            r,
            "in-place edit on recreated file must wake the watcher (Ok(true))"
        );
    }

    #[test]
    fn wakes_on_delete_recreate() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = KqueueConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Phase 1: delete. File NOTE_DELETE drops file_fd; parent NOTE_WRITE pulses. Either or both
        // events satisfy Ok(true).
        fs::remove_file(&cfg).expect("unlink");
        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
            "delete must wake the watcher (Ok(true))"
        );
        assert!(w.file_fd.is_none(), "file_fd dropped after terminal flag");

        // Phase 2: recreate. Parent NOTE_WRITE pulses; try_reopen succeeds inside the same
        // drain_ready().
        fs::write(&cfg, b"c").expect("recreate");
        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
            "recreate must wake the watcher (Ok(true))"
        );
        assert!(
            w.file_fd.is_some(),
            "file_fd reattached by try_reopen on parent event"
        );

        // Phase 3: post-reopen edit pulses through the new file fd — proves the re-register
        // actually rebound the vnode filter.
        drain_quiet(&mut w);
        fs::write(&cfg, b"d").expect("post-recreate edit");
        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
            "post-reopen edit must wake the watcher (Ok(true))"
        );
    }

    #[test]
    fn drops_on_init_when_path_resolves_to_symlink_loop() {
        let dir = tempfile::tempdir().expect("tempdir");
        let a = dir.path().join("a");
        let b = dir.path().join("b");
        // Two-link cycle: a → b, b → a. Either side canonicalised resolves through enough hops to
        // trip ELOOP on macOS / FreeBSD.
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
