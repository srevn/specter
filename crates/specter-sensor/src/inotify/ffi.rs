//! Thin libc wrappers over the inotify / eventfd / epoll syscalls — the
//! lone `unsafe` surface in this module. Each function below is a direct
//! syscall; module-level `#[allow(unsafe_code)]` keeps the audit boundary
//! at the file edge, mirroring [`crate::kqueue::ffi`].
//!
//! ## CLOEXEC discipline
//!
//! Every fd opened here carries `CLOEXEC`. The actuator's spawn path uses
//! fork+exec ([`crate::FsWatcher`] coexists in the same process); any fd
//! without `CLOEXEC` would leak into every spawned command. A leaked
//! `inotify_fd` would prevent kernel-side cleanup at watcher drop; a
//! leaked `eventfd` would make wakes nondeterministic across child
//! lifetimes; a leaked `epoll_fd` would inflate the watcher's
//! kernel-resource footprint per spawn.
//!
//! ## NONBLOCK discipline
//!
//! `inotify_init1(IN_NONBLOCK)` and `eventfd(EFD_NONBLOCK)` arm the read
//! side as non-blocking so the watcher's drain loop never wedges between
//! `epoll_wait` (which says "data ready") and the actual read (which the
//! kernel may have drained on a prior iteration in concurrent corner
//! cases). The `EAGAIN` short-circuit returns `Ok(0)` rather than
//! propagating an error — empty drain on a wake is a normal outcome.

#![allow(unsafe_code)]
// Helper consumers land per-phase: `eventfd_write` is wired by Phase B4
// (`super::wake`); the inotify/epoll/`O_PATH`/`fstat`/read helpers are
// wired by Phase B5–B9 (`super::watcher`). `dead_code` would otherwise
// spam every helper between B1 and B9. Remove this allow once Phase B9
// lands `poll_until` and every helper has a consumer.
#![allow(dead_code)]

use libc::{
    self, EFD_CLOEXEC, EFD_NONBLOCK, EPOLL_CLOEXEC, EPOLL_CTL_ADD, EPOLLIN, IN_CLOEXEC,
    IN_NONBLOCK, c_int, c_void, epoll_event,
};
use specter_core::ResourceKind;
use std::ffi::CString;
use std::io::{self, Error};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::time::Duration;

/// Convert a `Path` into a NUL-terminated C string for syscalls. Embedded
/// NULs (impossible from a real Linux path component but defensible from
/// fuzzed input) surface as a typed `io::Error::other`. The watcher's
/// trait wrapper classifies that as [`crate::WatchFailure::Invariant`]
/// (errno = 0 hits the `_` arm in `WatchFailureExt::from_io`), which is
/// the correct routing — a NUL-bearing path is a programmer / config
/// error, not a kernel-pressure or path-fatal signal.
fn path_to_cstring(path: &Path) -> io::Result<CString> {
    CString::new(path.as_os_str().as_bytes())
        .map_err(|_| Error::other("path contains an interior NUL byte"))
}

/// Create an inotify instance with `IN_NONBLOCK | IN_CLOEXEC`.
///
/// `IN_NONBLOCK` lets the watcher's drain loop short-circuit `EAGAIN` to
/// `Ok(0)` between `epoll_wait` notifications; `IN_CLOEXEC` plugs the
/// fork+exec leak (see module docs).
pub(super) fn inotify_init() -> io::Result<OwnedFd> {
    // SAFETY: `inotify_init1` returns a fresh non-negative fd or -1.
    // The flag set is a valid bit-or of two libc constants. No memory or
    // fd ownership crosses the boundary.
    let raw = unsafe { libc::inotify_init1(IN_NONBLOCK | IN_CLOEXEC) };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ the kernel handed us a fresh owned fd.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Add or replace a watch on `path` with the supplied `mask`.
///
/// Per `inotify(7)`:
/// - "If the pathname referred to by pathname is already being watched,
///   then the existing watch is updated (and `IN_MASK_CREATE` is not
///   used)."
/// - The returned `wd` is non-negative on success; on update of an
///   existing path the kernel returns the same `wd`, on a path resolving
///   to a different inode it returns a fresh `wd`.
///
/// The watcher's [`super::watcher::InotifyWatcher::watch`] (Phase B6)
/// uses the wd-equality check to detect inode swaps under an atomic
/// rename of the watched path — the load-bearing race the
/// `/proc/self/fd/N` install is designed to close.
pub(super) fn inotify_add_watch(fd: &OwnedFd, path: &Path, mask: u32) -> io::Result<c_int> {
    let cstr = path_to_cstring(path)?;
    // SAFETY: `cstr` is a valid NUL-terminated C string for the duration
    // of the call (lifetime extends to the `?`-returning paths above);
    // `mask` is a valid `IN_*` bit set; `fd` is a valid open inotify_fd.
    let wd = unsafe { libc::inotify_add_watch(fd.as_raw_fd(), cstr.as_ptr(), mask) };
    if wd < 0 {
        return Err(Error::last_os_error());
    }
    Ok(wd)
}

/// Remove the watch identified by `wd`.
///
/// Per `inotify(7)`, the kernel queues `IN_IGNORED` for the wd before
/// freeing the descriptor on the per-instance `idr`. Callers must drop
/// any pre-existing events queued on `wd` until `IN_IGNORED` is observed
/// — see the wd-reuse race mitigation (Phase B7).
///
/// `EINVAL` from a stale wd (the inode was already deleted, kernel
/// already reaped) is not an error: caller treats as "already gone."
pub(super) fn inotify_rm_watch(fd: &OwnedFd, wd: c_int) -> io::Result<()> {
    // SAFETY: `fd` is a valid open inotify_fd; `wd` is a `c_int` payload
    // the kernel either accepts (returns 0) or rejects with `EINVAL`.
    let n = unsafe { libc::inotify_rm_watch(fd.as_raw_fd(), wd) };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    Ok(())
}

/// Read pending inotify records into `buf`.
///
/// `EAGAIN` (kernel's `IN_NONBLOCK` short-circuit on an empty queue)
/// returns `Ok(0)` rather than an error — consistent with kqueue's
/// "no events available" outcome. `EINTR` retries internally so a stray
/// signal during the syscall doesn't surface as a false drain failure.
///
/// Truncated records are impossible on this code path: the kernel
/// returns `EINVAL` if `buf.len()` is below the per-event floor
/// (`sizeof(struct inotify_event) + NAME_MAX + 1`, ~273 bytes), and the
/// watcher sizes its buffer well above that.
pub(super) fn read_inotify(fd: &OwnedFd, buf: &mut [u8]) -> io::Result<usize> {
    loop {
        // SAFETY: `buf` is a `&mut [u8]` with valid length; `fd` is a
        // valid open inotify_fd. The kernel writes whole records into
        // the prefix; the trailing tail is undefined but the caller
        // consumes only the returned `n` bytes.
        let n = unsafe { libc::read(fd.as_raw_fd(), buf.as_mut_ptr().cast::<c_void>(), buf.len()) };
        if n >= 0 {
            return Ok(usize::try_from(n).unwrap_or(0));
        }
        let e = Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EAGAIN) => return Ok(0),
            Some(libc::EINTR) => {}
            _ => return Err(e),
        }
    }
}

/// Open `path` with `O_PATH | O_NOFOLLOW`. The fd binds to a specific
/// inode regardless of subsequent path-level renames; used by the
/// watcher's race-free install in Phase B6.
///
/// `O_PATH` permits `fstat` even without read permission and does not
/// pin the inode against `unlink` — exactly the discipline kqueue's
/// `O_EVTONLY` provides on Darwin.
pub(super) fn open_o_path(path: &Path) -> io::Result<OwnedFd> {
    let cstr = path_to_cstring(path)?;
    // SAFETY: `cstr` is a valid NUL-terminated C string; `flags` is a
    // valid `O_*` bit set. `open` returns a non-negative fd or -1.
    let raw = unsafe { libc::open(cstr.as_ptr(), libc::O_PATH | libc::O_NOFOLLOW) };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ `open` handed us a fresh owned fd.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Stat an `O_PATH` fd to determine [`ResourceKind`].
///
/// Used by the watcher's verification step against the engine's
/// expected `kind` on a fresh `WatchOp::Watch` — the `fstat` happens on
/// the same fd that subsequently feeds `inotify_add_watch` via
/// `/proc/self/fd/N`, so the kind we read here is the kind the kernel
/// will resolve at install time.
pub(super) fn fstat_kind(fd: &OwnedFd) -> io::Result<ResourceKind> {
    let mut s = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fd` is a valid open fd; `s` is a writable `*mut libc::stat`.
    // `fstat` returns 0 and populates every field on success.
    let n = unsafe { libc::fstat(fd.as_raw_fd(), s.as_mut_ptr()) };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `fstat` returned 0 above ⇒ every field of `s` is initialized.
    let s = unsafe { s.assume_init() };
    Ok(match s.st_mode & libc::S_IFMT {
        libc::S_IFDIR => ResourceKind::Dir,
        libc::S_IFREG => ResourceKind::File,
        _ => ResourceKind::Unknown,
    })
}

/// Create an eventfd with `EFD_NONBLOCK | EFD_CLOEXEC`.
///
/// The wake channel for cross-thread `poll_until` interruption: any
/// number of [`crate::WakeHandle::wake`] callers bump the kernel-side
/// counter; the watcher's `epoll_wait` fires; the watcher drains the
/// counter atomically (a single `read` consumes the entire accumulated
/// value).
pub(super) fn eventfd_create() -> io::Result<OwnedFd> {
    // SAFETY: `eventfd` returns a fresh non-negative fd or -1. The flag
    // set is a valid bit-or of two libc constants; the initial value
    // (zero) is a valid `c_uint`.
    let raw = unsafe { libc::eventfd(0, EFD_NONBLOCK | EFD_CLOEXEC) };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ the kernel handed us a fresh owned fd.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Bump the eventfd counter by `value`.
///
/// Concurrent writes accumulate kernel-side; a single
/// [`eventfd_drain`] consumes the entire counter in one shot. Callers
/// pass `1` for a single wake — the actual numeric value is
/// observationally irrelevant under the watcher's semantics ("any
/// non-zero counter ⇒ drained ⇒ wake delivered").
pub(super) fn eventfd_write(fd: &OwnedFd, value: u64) -> io::Result<()> {
    // SAFETY: `fd` is a valid open eventfd. `eventfd_write` performs a
    // single 8-byte write; libc handles the byte-order plumbing the
    // kernel's eventfd driver expects.
    let n = unsafe { libc::eventfd_write(fd.as_raw_fd(), value) };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    Ok(())
}

/// Drain the eventfd counter atomically. Returns the consumed value, or
/// `0` if the eventfd was empty (`EAGAIN` on `EFD_NONBLOCK`).
///
/// `EINTR` retries internally; otherwise mirrors [`read_inotify`]'s
/// failure shape.
pub(super) fn eventfd_drain(fd: &OwnedFd) -> io::Result<u64> {
    let mut value: u64 = 0;
    loop {
        // SAFETY: `fd` is a valid open eventfd; `&raw mut value` is a
        // writable `*mut u64`. `eventfd_read` writes the consumed counter
        // into `value` on success.
        let n = unsafe { libc::eventfd_read(fd.as_raw_fd(), &raw mut value) };
        if n == 0 {
            return Ok(value);
        }
        let e = Error::last_os_error();
        match e.raw_os_error() {
            Some(libc::EAGAIN) => return Ok(0),
            Some(libc::EINTR) => {}
            _ => return Err(e),
        }
    }
}

/// Create an epoll instance with `EPOLL_CLOEXEC`.
pub(super) fn epoll_create() -> io::Result<OwnedFd> {
    // SAFETY: `epoll_create1` returns a fresh non-negative fd or -1.
    let raw = unsafe { libc::epoll_create1(EPOLL_CLOEXEC) };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ the kernel handed us a fresh owned fd.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Register `fd` on `epoll` for `EPOLLIN`, tagging events with `token`.
/// `token` is echoed back in the [`epoll_event`] `u64` field on each
/// `epoll_wait` return; the watcher uses distinct tokens to discriminate
/// `inotify_fd`-readable from `wake_fd`-readable.
pub(super) fn epoll_register(epoll: &OwnedFd, fd: &OwnedFd, token: u64) -> io::Result<()> {
    // `EPOLLIN` is `c_int` in libc; `epoll_event.events` is `u32`.
    // `EPOLLIN`'s value (`0x1`) fits trivially; the cast is bound-safe.
    #[allow(clippy::cast_sign_loss)]
    let mut ev = epoll_event {
        events: EPOLLIN as u32,
        u64: token,
    };
    // SAFETY: `epoll`, `fd` are valid open fds. `&raw mut ev` is a
    // writable `*mut epoll_event`; the kernel reads the events/u64
    // fields and does not retain the pointer past the syscall.
    let n = unsafe {
        libc::epoll_ctl(
            epoll.as_raw_fd(),
            EPOLL_CTL_ADD,
            fd.as_raw_fd(),
            &raw mut ev,
        )
    };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    Ok(())
}

/// Block on `epoll_wait` until at least one fd is ready or `timeout_ms`
/// elapses. Returns the count of populated slots in `out`. `EINTR`
/// retries internally so signal delivery during the wait is invisible
/// to the watcher.
///
/// `timeout_ms = -1` blocks indefinitely; `timeout_ms = 0` is a
/// non-blocking poll. Convert from `Duration` via [`duration_to_ms`].
pub(super) fn epoll_wait(
    epoll: &OwnedFd,
    out: &mut [epoll_event],
    timeout_ms: c_int,
) -> io::Result<usize> {
    let maxevents = c_int::try_from(out.len()).unwrap_or(c_int::MAX);
    loop {
        // SAFETY: `out` is a mutable slice of `epoll_event`; the kernel
        // writes whole `epoll_event` values into the first `n` (returned)
        // slots and treats the rest as undefined. The slice's start
        // pointer is correctly aligned (epoll_event is `repr(packed)` on
        // x86_64 but Vec/slice storage honours the type's layout).
        let n =
            unsafe { libc::epoll_wait(epoll.as_raw_fd(), out.as_mut_ptr(), maxevents, timeout_ms) };
        if n >= 0 {
            return Ok(usize::try_from(n).unwrap_or(0));
        }
        let e = Error::last_os_error();
        if e.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        return Err(e);
    }
}

/// Convert a `Duration` to a millisecond timeout suitable for
/// `epoll_wait`. Saturates at `c_int::MAX` (~24 days) — well above any
/// engine-supplied deadline; saturation here is documentary, not
/// load-bearing.
#[must_use]
pub(super) fn duration_to_ms(d: Duration) -> c_int {
    let ms = d.as_millis().min(c_int::MAX as u128);
    c_int::try_from(ms).unwrap_or(c_int::MAX)
}
