//! `InotifyConfigWatcher` — inotify-backed [`ConfigWatcher`] for the daemon's own config file.
//! Linux only.
//!
//! Single-threaded: one thread owns the watcher value and drives
//! [`drain_ready`](ConfigWatcher::drain_ready). The inotify fd is exposed via [`AsFd`] so a reactor
//! can register it for edge- triggered readiness; `drain_ready` is non-blocking and drains the
//! inotify queue to empty on every call.
//!
//! # Watch shape
//!
//! Two `inotify_add_watch` registrations on the same inotify_fd:
//!
//! - **File watch (`file_wd`)** — installed against the canonicalised config file via the `O_PATH`
//!   `/proc/self/fd/N` race-free pattern (mirror of [`crate::inotify::watcher::InotifyWatcher`]).
//!   The file mask is [`FILE_MASK`]: `IN_MODIFY | IN_DELETE_SELF | IN_MOVE_SELF | IN_ATTRIB |
//!   IN_CLOSE_WRITE`. Catches in-place edits, terminal flags (delete / move), `chmod` / `chown`
//!   (`IN_ATTRIB`; the driver's lstat filter then sees the mode / ownership delta), and editor
//!   close-after-write (`IN_CLOSE_WRITE`). `IN_ATTRIB` also fires on `setxattr` and `utimes`, but
//!   the driver's `FileMeta` fingerprints only mode / uid / gid (not ctime), so those wakes
//!   collapse to a no-op at the convergence point. Dropped to `None` on `IN_IGNORED` only —
//!   intermediate flags like `IN_DELETE_SELF` / `IN_MOVE_SELF` just signal a real-event pulse and
//!   let the kernel's subsequent `IN_IGNORED` finalise the drop. This avoids an ordering hazard: if
//!   `file_wd` were nullified at `IN_DELETE_SELF`, the trailing `IN_IGNORED` for that wd would be
//!   misclassified by the parent-loss check (`rec.wd != self.file_wd` ⇒ falls through to the
//!   parent-wd arm) and force a spurious `Err`.
//!
//! - **Parent-dir watch (`parent_wd`)** — installed on the canonicalised parent directory with
//!   [`PARENT_MASK`]: `IN_CREATE | IN_MOVED_TO | IN_DELETE | IN_MOVED_FROM`. Held for the watcher's
//!   lifetime. Kernel-side reap (`IN_IGNORED` on `parent_wd`) means the parent path is gone;
//!   auto-reload cannot recover. `drain_ready()` propagates `Err`; the bin's wrapper logs and exits
//!   the watcher thread; SIGHUP-only operation continues. Parent-dir loss / parent-symlink retarget
//!   are documented restart-required limitations.
//!
//! Unlike kqueue, inotify's parent records carry the basename of the affected child. The watcher
//! compares each parent record's `name` field to the cached [`Self::config_basename`]; sibling
//! activity drops at the watcher edge with no extra syscall — strictly more efficient than the
//! kqueue branch, where the driver's lstat filter is the only suppression point for unrelated
//! parent traffic.
//!
//! # Opportunistic re-open
//!
//! After the drain-to-empty loop completes, [`Self::drain_ready`] runs [`Self::try_reopen`] iff a
//! basename-matched parent record was observed *and* the final `file_wd` state is `None`. The
//! decision is post-loop-of-loops, so the intra-batch ordering of the parent's `IN_MOVED_TO` /
//! `IN_CREATE` and the file's `IN_IGNORED` (the kernel reap of the old wd) is irrelevant, and so is
//! the kernel splitting one logical atomic-save burst across two `read_inotify` calls within a
//! single `drain_ready` invocation — every ordering converges on the same post-loop state.
//!
//! Post-loop is also a borrow-checker concession: [`record::parse`] borrows [`Self::read_buf`] for
//! the iterator's lifetime; a `&mut self` method invocation inside the loop would conflict with
//! that borrow. Disjoint-field NLL handles the field-level mutation on `file_wd` (Copy
//! `Option<i32>`) inline; the post-loop `try_reopen` call runs after the borrow ends.
//!
//! `saw_basename_parent` (not bare `real_seen`) gates the recovery because file-side records
//! nulling `file_wd` mid-drain are not a recovery signal *unless* a parent event also confirms "the
//! basename is back." A lone `IN_DELETE_SELF`/`IN_IGNORED` without a parent record means the file
//! is gone and the recreate has not yet happened; reopening on that signal alone is premature.
//!
//! Failure (`ENOENT` because recreate has not happened yet, any other errno) logs at `warn!` and
//! the next basename-matched parent event retries.
//!
//! # IN_IGNORED disposition
//!
//! Per `inotify(7)`, the kernel emits `IN_IGNORED` for any wd whose watch is being torn down. For
//! this watcher there are at most two sources:
//!
//! - **`file_wd`** — kernel-side reap of the file watch (the watched inode lost its last hardlink,
//!   was unmounted, was renamed across filesystems, or `IN_EXCL_UNLINK` triggered). We nullify
//!   `file_wd` and continue iterating the same batch — a coalesced delete-recreate burst (atomic
//!   save) often packs the parent `IN_MOVED_TO` ahead of the file `IN_IGNORED`, so the recovery
//!   window may already have passed by the time we drop `file_wd`. In that case the next batch's
//!   first parent event re-arms `try_reopen` (one settle-window of latency, no correctness impact —
//!   the engine's lstat filter covers the gap).
//!
//! - **`parent_wd`** — kernel-side reap of the parent directory watch (rmdir, unmount,
//!   cross-filesystem rename of the parent itself). We propagate `Err`; the bin exits the watcher
//!   thread.
//!
//! A stale wd on a third path (record's wd matches neither `file_wd` nor `parent_wd`) is dropped
//! silently. This watcher holds at most one `file_wd` at a time and never explicitly reuses it; the
//! kernel's wd-allocator returns a fresh integer on every successful `inotify_add_watch`, so no
//! wd-routing table is required (cf. [`crate::inotify::watcher::InotifyWatcher`]'s `draining_wds`,
//! which only matters when multiple resources can share the wd-namespace).
//!
//! # Drop semantics
//!
//! Default field-order drop:
//! - `inotify_fd` ([`OwnedFd`]) drops first → the kernel reaps both per-watch descriptors on this
//!   instance (per `inotify(7)`) and queues the resulting `IN_IGNORED` records on a stream no
//!   consumer reads (benign).
//! - `file_wd`, `parent_wd`, `parent_path`, `config_basename`, `read_buf` — `Option<c_int>` /
//!   `c_int` integers and heap strings / buffer; no kernel obligations of their own. Closing the
//!   inotify fd above is the sole deregister signal.

use crate::ConfigWatcher;
use crate::inotify::{ffi, record};
use libc::c_int;
use std::ffi::OsString;
use std::io;
use std::os::fd::{AsFd, BorrowedFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};

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
/// The terminal flags on the parent itself (`IN_DELETE_SELF` / `IN_MOVE_SELF`) and identity-floor
/// flags (`IN_UNMOUNT`, `IN_IGNORED`) are kernel-emitted regardless of mask; the watcher observes
/// them in `drain_ready()` regardless. Setting them here explicitly would be redundant.
///
/// Not included: `IN_EXCL_UNLINK` — that only matters for file watches (it suppresses post-unlink
/// events on the watched inode); at the dir level it has no effect.
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
/// `IN_IGNORED` is kernel-emitted regardless of mask; the `drain_ready()` drain handles it
/// explicitly.
///
/// Not included: `IN_EXCL_UNLINK`. The watcher pairs `O_PATH` + `/proc/self/fd/N` for race-free
/// install; the inode it pins is not held open by this watcher beyond the install transaction (the
/// OwnedFd from `open_o_path` drops at end of scope, per `inotify(7)`'s "the watch follows the
/// inode, not the fd" semantics). `IN_EXCL_UNLINK`'s purpose — suppress post-unlink events on an
/// open inode — is moot when nothing in this watcher holds the inode open. The engine watcher uses
/// it because *other* processes (the actuator pool) may hold descendant fds; the config watcher has
/// no analogous concern.
const FILE_MASK: u32 = libc::IN_MODIFY
    | libc::IN_DELETE_SELF
    | libc::IN_MOVE_SELF
    | libc::IN_ATTRIB
    | libc::IN_CLOSE_WRITE;

/// Drain buffer size in bytes. Per `inotify(7)` the per-event minimum is `sizeof(struct
/// inotify_event) + NAME_MAX + 1` ≈ 273 bytes; 16 KiB drains a typical editor save in one `read`
/// syscall and is well above the floor. Mirrors [`crate::inotify::watcher::InotifyWatcher`]'s
/// `READ_BUF_BYTES`.
const READ_BUF_BYTES: usize = 16 * 1024;

/// inotify-backed [`ConfigWatcher`] for the daemon's config file.
///
/// One file wd + one parent-dir wd registered on a fresh inotify_fd; see the module docs for the
/// full lifecycle and edge-case matrix. Construct with [`Self::new`]; drive with
/// [`ConfigWatcher::drain_ready`].
#[derive(Debug)]
pub struct InotifyConfigWatcher {
    /// Single inotify fd for both watches. Owned exclusively by the watcher; close ⇒ kernel
    /// auto-reaps every per-watch descriptor (per `inotify(7)`). Exposed through [`AsFd`] so a
    /// reactor can register it for edge-triggered readiness; drop closes the fd and kernel-reaps
    /// any queued events.
    inotify_fd: OwnedFd,

    /// File-side watch descriptor. Set on successful initial install (or `None` if the file
    /// vanished between `canonicalize` and `inotify_add_watch`). Dropped to `None` on `IN_IGNORED`
    /// for this wd; restored by [`Self::try_reopen`] on the next parent event with basename match.
    file_wd: Option<c_int>,

    /// Parent-dir watch descriptor. Held for the watcher's lifetime; kernel-side reap (`IN_IGNORED`
    /// on this wd) propagates `Err` to the bin and exits the watcher thread.
    parent_wd: c_int,

    /// Canonicalised parent path. Used by [`Self::try_reopen`] to rebuild the full file path; held as
    /// `PathBuf` (not just an `Arc<Path>`) because tests + diagnostics want owned values readily.
    parent_path: PathBuf,

    /// Final path component of the canonical config path, captured once at construction. Used by
    /// [`Self::try_reopen`] to rebuild the full file path, and by `drain_ready()` for the
    /// parent-event basename filter (`rec.name == self.config_basename.as_bytes()`). Stored as
    /// `OsString` to round-trip raw bytes losslessly across non-UTF-8 names.
    config_basename: OsString,

    /// Drain buffer for `read_inotify`. Sized at construction and reused across `drain_ready` calls
    /// — the hot path performs no allocation.
    read_buf: Vec<u8>,
}

impl InotifyConfigWatcher {
    /// Construct a watcher bound to `path`.
    ///
    /// Steps (each failure is fatal — the bin warn-logs and falls back to SIGHUP-only):
    ///
    /// 1. `canonicalize(path)` — resolves every symlink. ELOOP on a cyclic symlink, ENOENT if the
    ///    file doesn't exist. Subsequent leaf-symlink retargets after this point are documented as
    ///    restart-required.
    /// 2. Split into `parent_path` + `config_basename`.
    /// 3. Create the inotify fd.
    /// 4. `inotify_add_watch(parent_path, PARENT_MASK)` — captures the parent wd.
    /// 5. `open_o_path(canonical) → /proc/self/fd/N → inotify_add_watch(FILE_MASK)` — race-free
    ///    install for the file. `ENOENT` from the open is non-fatal (TOCTOU after canonicalize):
    ///    leave `file_wd = None` and re-attach on the next parent event.
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
        let parent_wd = ffi::inotify_add_watch(&inotify_fd, &parent_path, PARENT_MASK)?;

        // The TOCTOU window here is the price of `Config::from_path_with_meta` running before the
        // watcher: bytes captured atomically with `FileMeta`, then watcher constructed. An edit
        // landing between the two collapses to "file vanished" → leave `file_wd = None`; the bin's
        // post-init `FileMeta::from_path` lstat compares against the captured meta to drive an
        // immediate reload pulse if the on-disk state diverged.
        let file_wd = match ffi::open_o_path(&canonical) {
            Ok(fd) => {
                let wd = ffi::inotify_add_watch_fd(&inotify_fd, &fd, FILE_MASK)?;
                // `fd` drops at end of scope; the kernel watches the inode the fd resolved to at
                // `add_watch` time, independent of fd lifetime (per `inotify(7)`).
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
            file_wd,
            parent_wd,
            parent_path,
            config_basename,
            read_buf: vec![0u8; READ_BUF_BYTES],
        })
    }

    /// Attempt to re-install the file watch at the cached path. Idempotent: if `file_wd` is already
    /// `Some`, no-ops.
    ///
    /// Failures other than `NotFound` log at `warn!`; the next parent event retries. Mirrors the
    /// kqueue config watcher's re-open discipline.
    fn try_reopen(&mut self) {
        if self.file_wd.is_some() {
            return;
        }
        let path = self.parent_path.join(&self.config_basename);
        match ffi::open_o_path(&path) {
            Ok(fd) => {
                match ffi::inotify_add_watch_fd(&self.inotify_fd, &fd, FILE_MASK) {
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
                // Recreate hasn't materialised yet; the next parent event triggers another attempt.
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
    /// Non-blocking drain-to-empty of the inotify queue. Loops on [`ffi::read_inotify`] (which
    /// under `IN_NONBLOCK` returns `Ok(0)` on an empty queue) until the kernel signals `EAGAIN`,
    /// dispatching each record:
    ///
    /// - **`IN_IGNORED` on `file_wd`** — kernel reaped the file watch. Nullify `self.file_wd`; the
    ///   post-loop recovery (below) restores it iff a basename-matched parent record also fired in
    ///   this drain.
    /// - **`IN_IGNORED` on `parent_wd`** — kernel reaped the parent watch. Return `Err`
    ///   immediately; the bin exits the watcher thread.
    /// - **`IN_IGNORED` on a stale wd** — silently dropped. This watcher never explicitly reuses a
    ///   wd, so a stale value means kernel-side reap of a wd we already nullified.
    /// - **Parent record** (`rec.wd == parent_wd`) — basename filter: `rec.name == config_basename`
    ///   ⇒ `real_seen = true` and `saw_basename_parent = true`. Mismatched basename drops at the
    ///   watcher edge.
    /// - **File record** (`rec.wd == file_wd`) — `real_seen = true`. The driver's lstat-vs-`FileMeta`
    ///   filter at the convergence point decides whether the change was substantive.
    /// - **Other wd** — drop silently. Structurally unreachable in v1 (only two registrations), but
    ///   defensive against a future kernel surprise.
    ///
    /// After the drain-to-empty loop completes, `try_reopen` runs iff a basename-matched parent
    /// record was observed *and* the final `file_wd` state is `None`. The recovery decision lives
    /// outside the loop on purpose: the kernel may split one logical atomic-save burst across two
    /// `read_inotify` batches (basename-matched parent record in batch *k*, file-wd `IN_IGNORED` in
    /// batch *k+1*), and an inside-the-loop `try_reopen` would race the in-progress inode swap. The
    /// post-loop placement makes the decision against the invocation's final state, independent of
    /// both the intra-batch record order and the kernel's across-batch fragmentation. See the
    /// module's "Opportunistic re-open" section.
    ///
    /// The parent-`IN_IGNORED` arm intentionally returns mid-loop: parent-dir loss is terminal, and
    /// continuing the drain on a torn-down watch would only consume `IN_IGNORED`-tail stragglers
    /// against an already-doomed state.
    ///
    /// `EINTR` is retried inside [`ffi::read_inotify`]. Any other `io::Error` propagates verbatim —
    /// the bin's wrapper logs at `error!` and exits the watcher thread; SIGHUP-only operation
    /// continues.
    fn drain_ready(&mut self) -> io::Result<bool> {
        let mut real_seen = false;
        let mut saw_basename_parent = false;

        // The inner iterator borrows `self.read_buf` for its scope. Disjoint-field NLL allows
        // mutations on `self.file_wd` (Copy `Option<i32>`) inline; `self.try_reopen()` is a `&mut
        // self` method call that would conflict — it runs post-loop, after every borrow ends.
        //
        // Per-record dispatch parallels the engine watcher
        // (`super::watcher::InotifyWatcher::drain_ready`): control-records first (`IN_Q_OVERFLOW`,
        // `IN_IGNORED`), then per-wd routing.
        loop {
            let n_bytes = ffi::read_inotify(&self.inotify_fd, &mut self.read_buf)?;
            if n_bytes == 0 {
                // EAGAIN under IN_NONBLOCK ⇒ kernel queue drained; edge-triggered contract satisfied.
                break;
            }
            for rec in record::parse(&self.read_buf[..n_bytes]) {
                // 1. IN_Q_OVERFLOW: kernel-emitted overflow signal. The config watcher's
                //    per-instance queue is normally not at risk (a handful of events per editor
                //    save), but paranoia is cheap. Surface as a generic pulse so the driver's lstat
                //    filter takes the hit on uncertain state. wd is `-1` per `inotify(7)`, falling
                //    outside `file_wd` / `parent_wd` — ordering relative to `IN_IGNORED` is
                //    intent-preserving, not load-bearing.
                if rec.mask & libc::IN_Q_OVERFLOW != 0 {
                    tracing::warn!(
                        "inotify config-watcher: kernel queue overflow; signalling pulse"
                    );
                    real_seen = true;
                    continue;
                }

                // 2. IN_IGNORED: cleanup signal for a watch.
                if rec.mask & libc::IN_IGNORED != 0 {
                    if Some(rec.wd) == self.file_wd {
                        self.file_wd = None;
                    } else if rec.wd == self.parent_wd {
                        // Terminal: parent watch is gone, no recovery. Drop the in-flight partial
                        // state (real_seen, saw_basename_parent) and propagate Err so the bin tears
                        // the watcher thread down.
                        return Err(io::Error::other("config parent dir watch lost"));
                    }
                    // Stale wd (already-nullified file_wd or unknown): drop silently. No
                    // multi-resource routing here.
                    continue;
                }

                // 3. Per-wd routing.
                if rec.wd == self.parent_wd {
                    if rec.name == self.config_basename.as_bytes() {
                        real_seen = true;
                        saw_basename_parent = true;
                    }
                    // Unmatched basename: dropped at the watcher edge.
                } else if Some(rec.wd) == self.file_wd {
                    real_seen = true;
                }
                // Unknown wd (neither parent nor file): drop silently. Reachable only across a
                // kernel-side reap window — we observed a record on a wd whose `IN_IGNORED` we
                // already consumed in this same drain.
            }
        }

        // Order-independent recovery: decide against the *final* state of `file_wd` after the entire
        // drain-to-empty loop. A basename-matched parent record is the proof "the watched basename is
        // back"; a bare file-side `IN_IGNORED` without a parent event is not a recovery signal (the
        // file is gone, recreate hasn't happened yet), so we gate on `saw_basename_parent` rather
        // than the weaker `real_seen`. `try_reopen` is idempotent and `ENOENT`-fast; a call on a
        // not-yet-recreated file is cheap and the next basename-matched parent event retries.
        if saw_basename_parent && self.file_wd.is_none() {
            self.try_reopen();
        }
        Ok(real_seen)
    }
}

impl AsFd for InotifyConfigWatcher {
    fn as_fd(&self) -> BorrowedFd<'_> {
        self.inotify_fd.as_fd()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ConfigWatcher;
    use std::fs;
    use std::os::fd::{AsFd, AsRawFd, BorrowedFd};
    use std::time::{Duration, Instant};

    /// Watchdog deadline for drain calls in tests. Plenty of headroom for an inotify drain on any
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
    /// or the cap is hit. Used to flush a startup TOCTOU pulse (or coalesced setup events) so the
    /// test's subsequent edit lands on a clean drain.
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

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
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

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Write tempfile, then rename — the canonical editor save shape.
        let tmp = dir.path().join("specter.toml.tmp");
        fs::write(&tmp, b"b").expect("write tmp");
        fs::rename(&tmp, &cfg).expect("atomic rename");

        let r = fire_drain(&mut w, watchdog()).expect("drain ok");
        assert!(r, "atomic save must wake the watcher (Ok(true))");

        // Post-loop recovery against the invocation's final state: a coalesced drain containing the
        // parent's basename-matched `IN_MOVED_TO` plus the file wd's `IN_IGNORED` restores
        // `file_wd` in this same drain regardless of intra-batch ordering. A kernel-split fragment
        // can leave `file_wd` holding a stale (but `Some`) wd; the end-to-end
        // `atomic_save_then_in_place_edit` test catches that mode.
        assert!(w.file_wd.is_some(), "atomic save must leave a live file wd");
    }

    /// End-to-end check that an atomic save followed by an in-place edit on the recreated file
    /// surfaces a second pulse. Requires the post-loop recovery to install a live file wd on the
    /// new inode against any intra-batch ordering of the basename-matched parent record and the
    /// file wd's `IN_IGNORED`; a stranded `file_wd = None` (or a stale wd on the dead inode) would
    /// silently swallow the in-place edit and block the second drain past the watchdog.
    #[test]
    fn atomic_save_then_in_place_edit() {
        let dir = tempfile::tempdir().expect("tempdir");
        let cfg = dir.path().join("specter.toml");
        fs::write(&cfg, b"a").expect("write seed");

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Atomic save: write staging, rename over. The kernel emits `IN_MOVED_FROM`(staging) +
        // `IN_MOVED_TO`(basename) on the parent and `IN_MOVE_SELF` + `IN_IGNORED` on the file's wd,
        // possibly across multiple batches.
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
            w.file_wd.is_some(),
            "post-loop reopen must restore the file wd on the new inode"
        );

        // In-place edit on the recreated file pulses iff the file wd points at the new inode
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

        let mut w = InotifyConfigWatcher::new(&cfg).expect("watcher init");
        drain_quiet(&mut w);

        // Phase 1: delete. file_wd's IN_DELETE_SELF + IN_IGNORED; parent's IN_DELETE(specter.toml).
        // Either or both events satisfy Ok(true). Multiple drain calls may be needed if events
        // split across kernel batches.
        fs::remove_file(&cfg).expect("unlink");
        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
            "delete must wake the watcher (Ok(true))"
        );
        // Drain any trailing events (notably IN_IGNORED on file_wd if not in the prior batch) so
        // the post-condition is observable.
        drain_quiet(&mut w);
        assert!(
            w.file_wd.is_none(),
            "file_wd dropped after kernel IN_IGNORED"
        );

        // Phase 2: recreate. Parent IN_CREATE(specter.toml) pulses; basename match → try_reopen
        // runs after the drain loop.
        fs::write(&cfg, b"c").expect("recreate");
        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
            "recreate must wake the watcher (Ok(true))"
        );
        assert!(
            w.file_wd.is_some(),
            "file_wd reattached by try_reopen on parent event"
        );

        // Phase 3: post-reopen edit pulses through the new file_wd — proves the re-add actually
        // rebound the inotify watch.
        drain_quiet(&mut w);
        fs::write(&cfg, b"d").expect("post-recreate edit");
        assert!(
            fire_drain(&mut w, watchdog()).expect("drain ok"),
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

        // Create a sibling: parent_wd sees IN_CREATE(sibling.toml). The fd becomes readable, but
        // the basename filter inside drain_ready drops the record; the drain returns `Ok(false)`.
        // file_wd's inode is unaffected.
        fs::write(dir.path().join("sibling.toml"), b"x").expect("write sibling");

        // Short deadline — anything that would have fired has landed by now; the basename filter
        // dropped it.
        let deadline = Instant::now() + Duration::from_millis(200);
        let r = fire_drain(&mut w, deadline).expect("drain ok");
        assert!(
            !r,
            "sibling write must NOT wake the watcher (basename filter)"
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

        // Trigger parent-wd loss: rmfile (so the dir is empty), then rmdir. The kernel emits
        // IN_DELETE_SELF + IN_IGNORED on the parent's wd; `drain_ready` propagates `Err` on the
        // IN_IGNORED.
        fs::remove_file(&cfg).expect("rm file");
        fs::remove_dir(&parent).expect("rm parent dir");

        // Drain until Err, capped by watchdog. Earlier batches may surface `Ok(true)` for the file
        // events / parent IN_DELETE; the parent IN_IGNORED is the test's terminal observation.
        let deadline = watchdog();
        loop {
            match fire_drain(&mut w, deadline) {
                Ok(true) => {}
                Ok(false) => panic!("drain returned Ok(false) before parent IN_IGNORED"),
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
