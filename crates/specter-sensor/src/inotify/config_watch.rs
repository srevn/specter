//! `InotifyConfigWatcher` — inotify-backed [`ConfigWatcher`] for the
//! daemon's own config file. Linux only.
//!
//! Single-threaded: one thread owns the watcher value and calls
//! [`wait`](ConfigWatcher::wait) in a loop; the
//! [`wake_handle`](ConfigWatcher::wake_handle) is the only cross-thread
//! surface. The eventfd is `Arc`-shared with every wake handle so drop
//! of the watcher does not invalidate outstanding handles — a stale
//! `wake()` becomes a no-op-equivalent, never UB. (Same lifecycle
//! discipline as [`crate::inotify::wake`].)
//!
//! # Watch shape
//!
//! Two `inotify_add_watch` registrations on the same inotify_fd:
//!
//! - **File watch (`file_wd`)** — installed against the canonicalised
//!   config file via the `O_PATH` + `/proc/self/fd/N` race-free
//!   pattern (mirror of [`crate::inotify::watcher::InotifyWatcher`]).
//!   The file mask is [`FILE_MASK`]: `IN_MODIFY | IN_DELETE_SELF |
//!   IN_MOVE_SELF | IN_ATTRIB | IN_CLOSE_WRITE`. Catches in-place
//!   edits, terminal flags (delete / move), `chmod` / `chown`
//!   (`IN_ATTRIB`; the driver's lstat filter then sees the mode /
//!   ownership delta), and editor close-after-write
//!   (`IN_CLOSE_WRITE`). `IN_ATTRIB` also fires on `setxattr` and
//!   `utimes`, but the driver's `FileMeta` fingerprints only mode /
//!   uid / gid (not ctime), so those wakes collapse to a no-op at
//!   the convergence point. Dropped to `None` on `IN_IGNORED` only —
//!   intermediate flags like `IN_DELETE_SELF` / `IN_MOVE_SELF` just
//!   signal a real-event pulse and let the kernel's subsequent
//!   `IN_IGNORED` finalise the drop. This avoids an ordering hazard:
//!   if `file_wd` were nullified at `IN_DELETE_SELF`, the trailing
//!   `IN_IGNORED` for that wd would be misclassified by the
//!   parent-loss check (`rec.wd != self.file_wd` ⇒ falls through to
//!   the parent-wd arm) and force a spurious `Err`.
//!
//! - **Parent-dir watch (`parent_wd`)** — installed on the
//!   canonicalised parent directory with [`PARENT_MASK`]: `IN_CREATE |
//!   IN_MOVED_TO | IN_DELETE | IN_MOVED_FROM`. Held for the watcher's
//!   lifetime. Kernel-side reap (`IN_IGNORED` on `parent_wd`) means
//!   the parent path is gone; auto-reload cannot recover. `wait()`
//!   propagates `Err`; the bin's wrapper logs and exits the watcher
//!   thread; SIGHUP-only operation continues. Parent-dir loss /
//!   parent-symlink retarget are documented restart-required
//!   limitations.
//!
//! Unlike kqueue, inotify's parent records carry the basename of the
//! affected child. The watcher compares each parent record's `name`
//! field to the cached [`Self::config_basename`]; sibling activity
//! drops at the watcher edge with no extra syscall — strictly more
//! efficient than the kqueue branch, where the driver's lstat filter
//! is the only suppression point for unrelated parent traffic.
//!
//! # Opportunistic re-open
//!
//! When a parent record arrives with `name == config_basename` and
//! `file_wd` is `None`, [`Self::try_reopen`] runs after the per-call
//! record loop finishes. The call is **deferred** rather than inline
//! because [`record::parse`] borrows [`Self::read_buf`] for the
//! iterator's lifetime; a `&mut self` method invocation inside the
//! loop would conflict with that borrow. Disjoint-field NLL handles
//! the field-level mutations on `file_wd` / `parent_wd` etc. inline,
//! but a method call on `&mut self` widens to the whole struct.
//!
//! Failure (`ENOENT` because recreate has not happened yet, any other
//! errno) logs at `warn!` and the next parent event retries — two
//! `if file_wd.is_none() { try_reopen(); }` lines, no state machine.
//!
//! # IN_IGNORED disposition
//!
//! Per `inotify(7)`, the kernel emits `IN_IGNORED` for any wd whose
//! watch is being torn down. For this watcher there are at most two
//! sources:
//!
//! - **`file_wd`** — kernel-side reap of the file watch (the watched
//!   inode lost its last hardlink, was unmounted, was renamed across
//!   filesystems, or `IN_EXCL_UNLINK` triggered). We nullify
//!   `file_wd` and continue iterating the same batch — a coalesced
//!   delete-recreate burst (atomic save) often packs the parent
//!   `IN_MOVED_TO` ahead of the file `IN_IGNORED`, so the recovery
//!   window may already have passed by the time we drop `file_wd`.
//!   In that case the next batch's first parent event re-arms
//!   `try_reopen` (one settle-window of latency, no correctness
//!   impact — the engine's lstat filter covers the gap).
//!
//! - **`parent_wd`** — kernel-side reap of the parent directory
//!   watch (rmdir, unmount, cross-filesystem rename of the parent
//!   itself). We propagate `Err`; the bin exits the watcher thread.
//!
//! A stale wd on a third path (record's wd matches neither
//! `file_wd` nor `parent_wd`) is dropped silently. This watcher
//! holds at most one `file_wd` at a time and never explicitly
//! reuses it; the kernel's wd-allocator returns a fresh integer on
//! every successful `inotify_add_watch`, so no wd-routing table is
//! required (cf. [`crate::inotify::watcher::InotifyWatcher`]'s
//! `draining_wds`, which only matters when multiple resources can
//! share the wd-namespace).
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `inotify_fd` (`OwnedFd`) — drops first. The kernel reaps every
//!   per-watch descriptor on this instance; the resulting
//!   `IN_IGNORED` records queue onto a stream no consumer reads
//!   (benign).
//! - `wake_fd` (`Arc<OwnedFd>`) — decrements; if last clone, the
//!   eventfd closes and any queued counter is discarded. Wake
//!   handles holding `Arc` clones outlive the watcher; `wake()`
//!   from those becomes a no-op-equivalent (the counter accumulates
//!   on a fd no consumer drains).
//! - `epoll_fd` (`OwnedFd`) — drops last. The epoll instance closes;
//!   the kernel had already removed the registrations as the
//!   `inotify_fd` / `wake_fd` closed.
//!
//! No explicit deregister syscalls: closing each fd is the kernel's
//! deregister signal.

use crate::ConfigWatcher;
use crate::WakeHandle;
use crate::inotify::wake::InotifyWakeHandle;
use crate::inotify::{ffi, record};
use libc::c_int;
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

/// Token tagging the inotify fd in this watcher's epoll instance.
///
/// The config watcher's epoll fd is independent of
/// [`crate::inotify::watcher::InotifyWatcher`]'s, so the token
/// namespace is private — plain `1` / `2` mirror the kqueue config
/// watcher's `FILE_UDATA` / `PARENT_UDATA` discipline. Eye-catcher
/// patterns would be noise here.
const INOTIFY_TOKEN: u64 = 1;

/// Token tagging the wake (eventfd) in this watcher's epoll instance.
const WAKE_TOKEN: u64 = 2;

/// `inotify_add_watch` mask installed on the parent directory.
///
/// Every flag relevant to a config file's parent dir:
///
/// | Flag             | Why we listen                                       |
/// |------------------|-----------------------------------------------------|
/// | `IN_CREATE`      | basename appears (recreate-after-delete)            |
/// | `IN_MOVED_TO`    | basename appears (atomic-save target side)          |
/// | `IN_DELETE`      | basename disappears (`unlink`, `rm`)                |
/// | `IN_MOVED_FROM`  | basename disappears (rename target gone)            |
///
/// The terminal flags on the parent itself (`IN_DELETE_SELF` /
/// `IN_MOVE_SELF`) and identity-floor flags (`IN_UNMOUNT`,
/// `IN_IGNORED`) are kernel-emitted regardless of mask; the watcher
/// observes them in `wait()` regardless. Setting them here
/// explicitly would be redundant.
///
/// Not included: `IN_EXCL_UNLINK` — that only matters for file
/// watches (it suppresses post-unlink events on the watched inode);
/// at the dir level it has no effect.
const PARENT_MASK: u32 =
    libc::IN_CREATE | libc::IN_MOVED_TO | libc::IN_DELETE | libc::IN_MOVED_FROM;

/// `inotify_add_watch` mask installed on the file.
///
/// Every flag relevant to a config file:
///
/// | Flag              | Why we listen                                       |
/// |-------------------|-----------------------------------------------------|
/// | `IN_MODIFY`       | content modification (in-place edit)                |
/// | `IN_DELETE_SELF`  | terminal — inode unlinked                           |
/// | `IN_MOVE_SELF`    | terminal — inode moved                              |
/// | `IN_ATTRIB`       | `chmod` / `chown` — closes the recovery gap for     |
/// |                   | `EACCES`-after-`chmod`. Also fires on `setxattr` /  |
/// |                   | `utimes`; the driver's lstat filter (mode +         |
/// |                   | ownership) rejects those without a re-parse.        |
/// | `IN_CLOSE_WRITE`  | editor close-after-write (atomic-save final beat)   |
///
/// `IN_IGNORED` is kernel-emitted regardless of mask; the
/// `wait()` drain handles it explicitly.
///
/// Not included: `IN_EXCL_UNLINK`. The watcher pairs `O_PATH` +
/// `/proc/self/fd/N` for race-free install; the inode it pins is
/// not held open by this watcher beyond the install transaction
/// (the OwnedFd from `open_o_path` drops at end of scope, per
/// `inotify(7)`'s "the watch follows the inode, not the fd"
/// semantics). `IN_EXCL_UNLINK`'s purpose — suppress post-unlink
/// events on an open inode — is moot when nothing in this watcher
/// holds the inode open. The engine watcher uses it because *other*
/// processes (the actuator pool) may hold descendant fds; the
/// config watcher has no analogous concern.
const FILE_MASK: u32 = libc::IN_MODIFY
    | libc::IN_DELETE_SELF
    | libc::IN_MOVE_SELF
    | libc::IN_ATTRIB
    | libc::IN_CLOSE_WRITE;

/// Drain buffer size in bytes. Per `inotify(7)` the per-event minimum
/// is `sizeof(struct inotify_event) + NAME_MAX + 1` ≈ 273 bytes;
/// 16 KiB drains a typical editor save in one `read` syscall and is
/// well above the floor. Mirrors
/// [`crate::inotify::watcher::InotifyWatcher`]'s `READ_BUF_BYTES`.
const READ_BUF_BYTES: usize = 16 * 1024;

/// Maximum epoll events drained per `epoll_wait` syscall. We register
/// exactly two fds (`inotify_fd`, `wake_fd`); both can be ready
/// simultaneously. A third slot would never be populated.
const EPOLL_BATCH: usize = 2;

/// inotify-backed [`ConfigWatcher`] for the daemon's config file.
///
/// One file wd + one parent-dir wd registered on a fresh inotify_fd,
/// multiplexed with a wake eventfd via a private epoll instance; see
/// the module docs for the full lifecycle and edge-case matrix.
/// Construct with [`Self::new`]; drive with [`ConfigWatcher::wait`].
#[derive(Debug)]
pub struct InotifyConfigWatcher {
    /// Single inotify fd for both watches. Owned exclusively by the
    /// watcher; close ⇒ kernel auto-reaps every per-watch descriptor
    /// (per `inotify(7)`). Plain [`OwnedFd`] (no `Arc`) — only the
    /// watcher's owning thread reads from it; cross-thread wake uses
    /// the separate `wake_fd` eventfd.
    inotify_fd: OwnedFd,

    /// Eventfd for cross-thread wake. `Arc` so wake handles can hold
    /// their own clones without borrowing from the watcher; drop of
    /// the last clone closes the fd. See [`InotifyWakeHandle`] for
    /// the full lifecycle discipline.
    wake_fd: Arc<OwnedFd>,

    /// Epoll fd watching `(inotify_fd, wake_fd)`. Owned, not Arc'd —
    /// only `wait` reads from it (via `epoll_wait`); wake handles
    /// never touch it. Its `Drop` closes the fd, tearing down the
    /// epoll instance when the watcher ends.
    epoll_fd: OwnedFd,

    /// File-side watch descriptor. Set on successful initial install
    /// (or `None` if the file vanished between `canonicalize` and
    /// `inotify_add_watch`). Dropped to `None` on `IN_IGNORED` for
    /// this wd; restored by [`Self::try_reopen`] on the next parent
    /// event with basename match.
    file_wd: Option<c_int>,

    /// Parent-dir watch descriptor. Held for the watcher's lifetime;
    /// kernel-side reap (`IN_IGNORED` on this wd) propagates `Err`
    /// to the bin and exits the watcher thread.
    parent_wd: c_int,

    /// Canonicalised parent path. Used by [`Self::try_reopen`] to
    /// rebuild the full file path; held as `PathBuf` (not just an
    /// `Arc<Path>`) because tests + diagnostics want owned values
    /// readily.
    parent_path: PathBuf,

    /// Final path component of the canonical config path, captured
    /// once at construction. Used by [`Self::try_reopen`] to rebuild
    /// the full file path, and by `wait()` for the parent-event
    /// basename filter (`rec.name == self.config_basename.as_bytes()`).
    /// Stored as `OsString` to round-trip raw bytes losslessly across
    /// non-UTF-8 names.
    config_basename: OsString,

    /// Drain buffer for `read_inotify`. Sized at construction and
    /// reused across `wait` calls — the hot path performs no
    /// allocation.
    read_buf: Vec<u8>,
}

impl InotifyConfigWatcher {
    /// Construct a watcher bound to `path`.
    ///
    /// Steps (each failure is fatal — the bin warn-logs and falls
    /// back to SIGHUP-only):
    ///
    /// 1. `canonicalize(path)` — resolves every symlink. ELOOP on a
    ///    cyclic symlink, ENOENT if the file doesn't exist.
    ///    Subsequent leaf-symlink retargets after this point are
    ///    documented as restart-required.
    /// 2. Split into `parent_path` + `config_basename`.
    /// 3. Create the inotify fd, eventfd, epoll fd; register the
    ///    inotify and wake fds on the epoll under distinct tokens.
    /// 4. `inotify_add_watch(parent_path, PARENT_MASK)` — captures
    ///    the parent wd.
    /// 5. `open_o_path(canonical) → /proc/self/fd/N → inotify_add_watch(FILE_MASK)`
    ///    — race-free install for the file. `ENOENT` from the open
    ///    is non-fatal (TOCTOU after canonicalize): leave
    ///    `file_wd = None` and re-attach on the next parent event.
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

        let inotify_fd = ffi::inotify_init()?;
        let wake_fd = Arc::new(ffi::eventfd_create()?);
        let epoll_fd = ffi::epoll_create()?;
        ffi::epoll_register(&epoll_fd, &inotify_fd, INOTIFY_TOKEN)?;
        ffi::epoll_register(&epoll_fd, &wake_fd, WAKE_TOKEN)?;

        let parent_wd = ffi::inotify_add_watch(&inotify_fd, &parent_path, PARENT_MASK)?;

        // The TOCTOU window here is the price of `Config::from_path_with_meta`
        // running before the watcher: bytes captured atomically with
        // `FileMeta`, then watcher constructed. An edit landing
        // between the two collapses to "file vanished" → leave
        // `file_wd = None`; the bin's post-init `FileMeta::from_path`
        // lstat compares against the captured meta to drive an
        // immediate reload pulse if the on-disk state diverged.
        let file_wd = match ffi::open_o_path(&canonical) {
            Ok(fd) => {
                let proc_path = format!("/proc/self/fd/{}", fd.as_raw_fd());
                let proc_path_ref = Path::new(&proc_path);
                let wd = ffi::inotify_add_watch(&inotify_fd, proc_path_ref, FILE_MASK)?;
                // `fd` drops at end of scope; the kernel watches the
                // inode the fd resolved to at `add_watch` time,
                // independent of fd lifetime (per `inotify(7)`).
                Some(wd)
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => None,
            Err(e) => return Err(e),
        };

        tracing::debug!(
            path = %canonical.display(),
            parent = %parent_path.display(),
            parent_wd,
            file_present = file_wd.is_some(),
            "inotify config-watcher initialised"
        );

        Ok(Self {
            inotify_fd,
            wake_fd,
            epoll_fd,
            file_wd,
            parent_wd,
            parent_path,
            config_basename,
            read_buf: vec![0u8; READ_BUF_BYTES],
        })
    }

    /// Attempt to re-install the file watch at the cached path.
    /// Idempotent: if `file_wd` is already `Some`, no-ops.
    ///
    /// Failures other than `NotFound` log at `warn!`; the next
    /// parent event retries. Mirrors the kqueue config watcher's
    /// re-open discipline.
    fn try_reopen(&mut self) {
        if self.file_wd.is_some() {
            return;
        }
        let path = self.parent_path.join(&self.config_basename);
        match ffi::open_o_path(&path) {
            Ok(fd) => {
                let proc_path = format!("/proc/self/fd/{}", fd.as_raw_fd());
                let proc_path_ref = Path::new(&proc_path);
                match ffi::inotify_add_watch(&self.inotify_fd, proc_path_ref, FILE_MASK) {
                    Ok(wd) => {
                        tracing::debug!(
                            path = %path.display(),
                            wd,
                            "inotify config-watcher reopened file"
                        );
                        self.file_wd = Some(wd);
                    }
                    Err(e) => {
                        tracing::warn!(
                            ?e,
                            path = %path.display(),
                            "inotify config-watcher re-add failed",
                        );
                    }
                }
                // `fd` drops at end of scope (success and failure paths).
            }
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                // Recreate hasn't materialised yet; the next parent
                // event triggers another attempt.
            }
            Err(e) => {
                tracing::warn!(
                    ?e,
                    path = %path.display(),
                    "inotify config-watcher re-open failed",
                );
            }
        }
    }
}

impl ConfigWatcher for InotifyConfigWatcher {
    /// Block on `epoll_wait` until events arrive (or the deadline
    /// elapses, or a wake fires). Per-record dispatch:
    ///
    /// - **`IN_IGNORED` on `file_wd`** — kernel reaped the file
    ///   watch. Nullify `self.file_wd`; subsequent parent events in
    ///   the same batch may already trigger `try_reopen` (deferred
    ///   to after the loop — see module docs on the borrow
    ///   discipline).
    /// - **`IN_IGNORED` on `parent_wd`** — kernel reaped the parent
    ///   watch. Return `Err` immediately; the bin exits the watcher
    ///   thread.
    /// - **`IN_IGNORED` on a stale wd** — silently dropped. This
    ///   watcher never explicitly reuses a wd, so a stale value
    ///   means kernel-side reap of a wd we already nullified.
    /// - **Parent record** (`rec.wd == parent_wd`) — basename
    ///   filter: `rec.name == config_basename` ⇒ `real_seen = true`
    ///   and arm `try_reopen` if `file_wd` is `None`. Mismatched
    ///   basename drops at the watcher edge.
    /// - **File record** (`rec.wd == file_wd`) — `real_seen = true`.
    ///   The driver's lstat-vs-`FileMeta` filter at the convergence
    ///   point decides whether the change was substantive.
    /// - **Other wd** — drop silently. Structurally unreachable in
    ///   v1 (only two registrations), but defensive against a
    ///   future kernel surprise.
    ///
    /// `EINTR` is retried inside the FFI helpers. Any other
    /// `io::Error` propagates verbatim — the bin's wrapper logs at
    /// `error!` and exits the watcher thread; SIGHUP-only operation
    /// continues.
    fn wait(&mut self, deadline: Option<Instant>) -> io::Result<bool> {
        // `None` blocks indefinitely (-1); `Some(d)` past the
        // deadline saturates to `Duration::ZERO` ⇒ `0 ms`
        // non-blocking poll.
        let timeout_ms = deadline.map_or(-1, |d| {
            ffi::duration_to_ms(d.saturating_duration_since(Instant::now()))
        });
        let mut epoll_events = [libc::epoll_event { events: 0, u64: 0 }; EPOLL_BATCH];
        let n_ready = ffi::epoll_wait(&self.epoll_fd, &mut epoll_events, timeout_ms)?;

        if n_ready == 0 {
            // Deadline arrived with nothing ready. Wake / event
            // would have populated at least one slot.
            return Ok(false);
        }

        let mut wake_fired = false;
        let mut inotify_data = false;
        for ev in &epoll_events[..n_ready] {
            match ev.u64 {
                INOTIFY_TOKEN => inotify_data = true,
                WAKE_TOKEN => wake_fired = true,
                other => tracing::warn!(
                    token = format_args!("{other:#018x}"),
                    "inotify config-watcher: unrecognised epoll token (structural break)"
                ),
            }
        }

        if wake_fired {
            // Drain the eventfd counter to clear `EPOLLIN` on the
            // wake fd. The actual counter value is observationally
            // irrelevant — any non-zero accumulation collapses to
            // "wake delivered." A drain failure is reachable only
            // on a structural break (the watcher's `Arc<OwnedFd>`
            // keeps `wake_fd` alive for the watcher's lifetime);
            // log at `trace` and proceed.
            if let Err(e) = ffi::eventfd_drain(&self.wake_fd) {
                tracing::trace!(error = ?e, "inotify config-watcher wake-fd drain failed (benign)");
            }
        }

        if !inotify_data {
            // Wake-only return path. The bin's loop re-checks the
            // shutdown flag before the next `wait`.
            return Ok(false);
        }

        let n_bytes = ffi::read_inotify(&self.inotify_fd, &mut self.read_buf)?;
        if n_bytes == 0 {
            // EAGAIN under `IN_NONBLOCK`. Reachable on a concurrent
            // drain that races us between `epoll_wait` and `read` —
            // structurally unreachable under the single-reader
            // discipline, defended for completeness.
            return Ok(false);
        }

        let mut real_seen = false;
        let mut needs_reopen = false;

        // The iterator borrows `self.read_buf` for the loop scope.
        // Disjoint-field NLL allows mutations on `self.file_wd`
        // (Copy `Option<i32>`) inline; `self.try_reopen()` is a
        // `&mut self` method call that would conflict — defer it
        // via `needs_reopen` and run it after the loop.
        for rec in record::parse(&self.read_buf[..n_bytes]) {
            if rec.mask & libc::IN_IGNORED != 0 {
                if Some(rec.wd) == self.file_wd {
                    self.file_wd = None;
                } else if rec.wd == self.parent_wd {
                    return Err(io::Error::other("config parent dir watch lost"));
                }
                // Stale wd (already-nullified file_wd or unknown):
                // drop silently. No multi-resource routing here.
                continue;
            }

            // Defensive: kernel-emitted overflow signal. The config
            // watcher's per-instance queue is normally not at risk
            // (we generate a handful of events per editor save),
            // but paranoia is cheap. Surface as a generic pulse so
            // the driver's lstat filter takes the hit on uncertain
            // state. wd is `-1` for IN_Q_OVERFLOW per
            // `inotify(7)`, which falls outside our two known wds.
            if rec.mask & libc::IN_Q_OVERFLOW != 0 {
                tracing::warn!("inotify config-watcher: kernel queue overflow; signalling pulse");
                real_seen = true;
                continue;
            }

            if rec.wd == self.parent_wd {
                if rec.name == self.config_basename.as_bytes() {
                    real_seen = true;
                    if self.file_wd.is_none() {
                        needs_reopen = true;
                    }
                }
                // Unmatched basename: dropped at the watcher edge.
            } else if Some(rec.wd) == self.file_wd {
                real_seen = true;
            }
            // Unknown wd (neither parent nor file): drop silently.
            // Reachable only across a kernel-side reap window — we
            // observed a record on a wd whose `IN_IGNORED` we
            // already consumed in this same batch.
        }

        if needs_reopen {
            self.try_reopen();
        }
        Ok(real_seen)
    }

    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(InotifyWakeHandle::new(Arc::clone(&self.wake_fd)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfigWatcher;
    use std::fs;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::{Duration, Instant};

    /// Watchdog deadline for `wait` calls in tests. Plenty of
    /// headroom for an inotify drain on any sane CI host while still
    /// bounding a stuck test below CI's per-test timeout.
    fn watchdog() -> Instant {
        Instant::now() + Duration::from_secs(5)
    }

    /// Drain pending events from the watcher until either nothing
    /// arrives within a short deadline or the cap is hit. Used to
    /// flush a startup TOCTOU pulse (or coalesced setup events) so
    /// the test's subsequent edit lands on a clean drain.
    ///
    /// Returns on the first non-`Ok(true)` outcome — `Ok(false)` is
    /// the expected exit; `Err` would be test-fatal but is left for
    /// the next `wait` call to surface.
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

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
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

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
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

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Phase 1: delete. file_wd's IN_DELETE_SELF + IN_IGNORED;
        // parent's IN_DELETE(specter.toml). Either or both events
        // satisfy Ok(true). Multiple `wait` calls may be needed if
        // events split across kernel batches.
        fs::remove_file(&cfg).expect("unlink");
        assert!(
            w.wait(Some(watchdog())).expect("wait ok"),
            "delete must wake the watcher (Ok(true))"
        );
        // Drain any trailing events (notably IN_IGNORED on file_wd
        // if not in the prior batch) so the post-condition is
        // observable.
        drain_quiet(&mut w);
        assert!(
            w.file_wd.is_none(),
            "file_wd dropped after kernel IN_IGNORED"
        );

        // Phase 2: recreate. Parent IN_CREATE(specter.toml) pulses;
        // basename match → try_reopen runs after the wait loop.
        fs::write(&cfg, b"c").expect("recreate");
        assert!(
            w.wait(Some(watchdog())).expect("wait ok"),
            "recreate must wake the watcher (Ok(true))"
        );
        assert!(
            w.file_wd.is_some(),
            "file_wd reattached by try_reopen on parent event"
        );

        // Phase 3: post-reopen edit pulses through the new file_wd —
        // proves the re-add actually rebound the inotify watch.
        drain_quiet(&mut w);
        fs::write(&cfg, b"d").expect("post-recreate edit");
        assert!(
            w.wait(Some(watchdog())).expect("wait ok"),
            "post-reopen edit must wake the watcher (Ok(true))"
        );
    }

    #[test]
    fn basename_filter_rejects_sibling_writes() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Create a sibling: parent_wd sees IN_CREATE(sibling.toml).
        // Basename filter drops it; file_wd's inode is unaffected.
        // The whole batch is `real_seen = false`.
        fs::write(dir.path().join("sibling.toml"), b"x").expect("write sibling");

        // Short deadline — anything that would have fired has
        // landed by now; the basename filter dropped it.
        let deadline = Instant::now() + Duration::from_millis(200);
        let r = w.wait(Some(deadline)).expect("wait ok");
        assert!(
            !r,
            "sibling write must NOT wake the watcher (basename filter)"
        );
    }

    #[test]
    fn wake_handle_returns_false() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
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
    fn parent_wd_lost_returns_err() {
        let dir = tempfile::tempdir().expect("tempdir");
        let parent = dir.path().to_path_buf();
        let cfg = parent.join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Trigger parent-wd loss: rmfile (so the dir is empty),
        // then rmdir. The kernel emits IN_DELETE_SELF + IN_IGNORED
        // on the parent's wd; `wait` propagates `Err` on the
        // IN_IGNORED.
        fs::remove_file(&cfg).expect("rm file");
        fs::remove_dir(&parent).expect("rm parent dir");

        // Drain until Err, capped by watchdog. Earlier batches may
        // surface `Ok(true)` for the file events / parent
        // IN_DELETE; the parent IN_IGNORED is the test's terminal
        // observation.
        let deadline = watchdog();
        loop {
            match w.wait(Some(deadline)) {
                Ok(true) => {}
                Ok(false) => panic!("wait returned Ok(false) before parent IN_IGNORED"),
                Err(e) => {
                    let msg = format!("{e}");
                    assert!(
                        msg.contains("parent dir watch lost"),
                        "unexpected error: {msg}"
                    );
                    return;
                }
            }
        }
    }
}
