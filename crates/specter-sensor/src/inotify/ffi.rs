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
//!
//! ## Deadline tracking
//!
//! The wait primitive ([`epoll_wait`]) takes an `Option<Instant>` — not
//! a pre-computed millisecond timeout — and owns deadline tracking
//! across `EINTR` retries. The remaining budget is recomputed inside
//! the retry loop on every iteration so wall-clock progress between
//! syscall re-entries is preserved. This is the only layer that *can*
//! re-derive the remaining budget, so deadline math belongs here
//! rather than at every caller (mirror of [`crate::kqueue::ffi`]).

#![allow(unsafe_code)]

use libc::{
    self, EFD_CLOEXEC, EFD_NONBLOCK, EPOLL_CLOEXEC, EPOLL_CTL_ADD, EPOLLIN, IN_CLOEXEC,
    IN_NONBLOCK, c_int, c_void, epoll_event,
};
use specter_core::ResourceKind;
use std::ffi::{CStr, CString};
use std::io::{self, Error};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::time::{Duration, Instant};

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
/// The watcher's [`super::watcher::InotifyWatcher::watch`] uses the
/// wd-equality check to detect inode swaps under an atomic
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

/// Add or replace a watch on the inode referred to by `watched_fd`, via
/// the kernel's `/proc/self/fd/N` magic-symlink resolution.
///
/// Fused variant of [`inotify_add_watch`] that closes the TOCTOU window
/// between [`open_o_path`] / [`fstat_kind`] and the watch install: the
/// caller's `O_PATH` fd binds to a specific inode at open time, and
/// `inotify_add_watch` on the procfs path resolves to that exact inode
/// regardless of intervening renames at the original pathname.
///
/// Stack-formats the procfs path into a fixed-size buffer — no heap
/// allocation on the hot path. The magic-symlink resolution is an FFI
/// concern (mirror of [`super::super::kqueue::ffi::register_vnode`]'s
/// fd-bound registration), not a watcher concern; callers hand over the
/// `O_PATH` fd plus mask and read back the watch descriptor.
///
/// Returns the watch descriptor on success, matching
/// [`inotify_add_watch`]'s contract. The realistic error set is
/// `EACCES` (no `inotify(7)` permission on the inode), `ENOSPC`
/// (`/proc/sys/fs/inotify/max_user_watches` exceeded), `ENOMEM`.
pub(super) fn inotify_add_watch_fd(
    inotify_fd: &OwnedFd,
    watched_fd: &OwnedFd,
    mask: u32,
) -> io::Result<c_int> {
    let mut buf = [0u8; PROC_FD_PATH_BUF];
    let cstr = format_proc_fd_path(&mut buf, watched_fd.as_raw_fd());
    // SAFETY: `cstr.as_ptr()` is a valid `*const c_char` for the
    // duration of the syscall (`buf` lives on the stack across this
    // call); `mask` is a valid `IN_*` bit set; `inotify_fd` is a valid
    // open inotify_fd.
    let wd = unsafe { libc::inotify_add_watch(inotify_fd.as_raw_fd(), cstr.as_ptr(), mask) };
    if wd < 0 {
        return Err(Error::last_os_error());
    }
    Ok(wd)
}

/// `/proc/self/fd/<c_int>\0` stack buffer size. `/proc/self/fd/` is
/// 14 bytes; a non-negative `c_int` (i32 on every Linux target) is at
/// most 10 decimal digits (`i32::MAX = 2_147_483_647`); plus NUL = 25
/// bytes. 32 is comfortable round-power-of-two storage.
const PROC_FD_PATH_BUF: usize = 32;

/// Format `/proc/self/fd/<fd>` into the start of `buf` and return a
/// borrowed [`CStr`] view of the populated prefix.
///
/// Panics on a negative fd — [`OwnedFd::as_raw_fd`] is non-negative by
/// API contract, so a panic here is a structural break (the watcher
/// would otherwise format the two's-complement value as the procfs
/// path, silently watching whatever inode that aliased fd refers to).
fn format_proc_fd_path(buf: &mut [u8; PROC_FD_PATH_BUF], fd: RawFd) -> &CStr {
    const PREFIX: &[u8] = b"/proc/self/fd/";
    let fd_unsigned =
        u32::try_from(fd).expect("OwnedFd::as_raw_fd is non-negative by API contract");
    buf[..PREFIX.len()].copy_from_slice(PREFIX);
    let digits_written = write_decimal_u32(&mut buf[PREFIX.len()..], fd_unsigned);
    let nul_pos = PREFIX.len() + digits_written;
    buf[nul_pos] = 0;
    // SAFETY: `buf[..=nul_pos]` ends in NUL and contains no interior
    // NULs: PREFIX is constant ASCII; `write_decimal_u32` writes only
    // ASCII digits (`b'0'..=b'9'`); position `nul_pos` is the first
    // NUL byte placed above.
    unsafe { CStr::from_bytes_with_nul_unchecked(&buf[..=nul_pos]) }
}

/// Write `value` as decimal ASCII digits into the start of `buf`,
/// returning the count of bytes written. `buf` must have at least 10
/// bytes of capacity (`u32::MAX = 4_294_967_295` is 10 digits).
fn write_decimal_u32(buf: &mut [u8], mut value: u32) -> usize {
    if value == 0 {
        buf[0] = b'0';
        return 1;
    }
    // Digits accumulate least-significant-first into a scratch buffer,
    // then reverse-copy into `buf`. Avoids the two-pass "compute length
    // first" overhead at the cost of 10 stack bytes.
    let mut tmp = [0u8; 10];
    let mut len = 0;
    while value > 0 {
        // `value % 10` is in `0..=9`; the cast to `u8` is exact.
        #[allow(clippy::cast_possible_truncation)]
        let digit = (value % 10) as u8;
        tmp[len] = b'0' + digit;
        value /= 10;
        len += 1;
    }
    for i in 0..len {
        buf[i] = tmp[len - 1 - i];
    }
    len
}

/// Remove the watch identified by `wd`.
///
/// Per `inotify(7)`, the kernel queues `IN_IGNORED` for the wd before
/// freeing the descriptor on the per-instance `idr`. Callers must drop
/// any pre-existing events queued on `wd` until `IN_IGNORED` is observed
/// — see the wd-reuse race mitigation in the watcher.
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

/// Open `path` with `O_PATH | O_NOFOLLOW | O_CLOEXEC`. The fd binds to
/// a specific inode regardless of subsequent path-level renames; used
/// by the watcher's race-free install.
///
/// `O_PATH` permits `fstat` even without read permission and does not
/// pin the inode against `unlink` — exactly the discipline kqueue's
/// `O_EVTONLY` provides on Darwin. `O_CLOEXEC` covers the
/// `open → fstat → add_watch → drop(fd)` window: the actuator's
/// fork+exec can race against any of those steps, and a leaked
/// `O_PATH` fd in the child would prolong the inode's reference count
/// for the child's lifetime. Plugging the leak at open() time is
/// uniform with the watcher's three persistent fds.
pub(super) fn open_o_path(path: &Path) -> io::Result<OwnedFd> {
    let cstr = path_to_cstring(path)?;
    // SAFETY: `cstr` is a valid NUL-terminated C string; `flags` is a
    // valid `O_*` bit set. `open` returns a non-negative fd or -1.
    let raw = unsafe {
        libc::open(
            cstr.as_ptr(),
            libc::O_PATH | libc::O_NOFOLLOW | libc::O_CLOEXEC,
        )
    };
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

/// Block on `epoll_wait` until at least one fd is ready or the
/// deadline elapses. Returns the count of populated slots in `out`.
/// `EINTR` retries internally so signal delivery during the wait is
/// invisible to the watcher.
///
/// `deadline = None` blocks indefinitely (`timeout_ms = -1`); `Some(d)`
/// arms the kernel-side wait at a per-iteration millisecond timeout
/// re-derived from `d.saturating_duration_since(Instant::now())`. The
/// remaining budget is recomputed inside the retry loop so an
/// `EINTR` re-entry does not re-arm the full original budget — the
/// caller's deadline is total wall-clock budget, not per-syscall.
///
/// A `Some(d)` with `d <= Instant::now()` collapses to a zero timeout
/// (non-blocking poll); `epoll_wait(2)` returns immediately with `0`
/// fds ready if none are.
pub(super) fn epoll_wait(
    epoll: &OwnedFd,
    out: &mut [epoll_event],
    deadline: Option<Instant>,
) -> io::Result<usize> {
    let maxevents = c_int::try_from(out.len()).unwrap_or(c_int::MAX);
    loop {
        // Re-derive the remaining budget on every iteration so an
        // `EINTR` retry resumes against wall-clock progress, not the
        // original deadline budget. `None` keeps the indefinite block
        // (`-1` per `epoll_wait(2)`).
        let timeout_ms = match deadline {
            None => -1,
            Some(d) => duration_to_ms(d.saturating_duration_since(Instant::now())),
        };
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
fn duration_to_ms(d: Duration) -> c_int {
    let ms = d.as_millis().min(c_int::MAX as u128);
    c_int::try_from(ms).unwrap_or(c_int::MAX)
}

#[cfg(test)]
mod tests {
    use super::{
        PROC_FD_PATH_BUF, duration_to_ms, epoll_create, epoll_wait, format_proc_fd_path,
        write_decimal_u32,
    };
    use std::time::{Duration, Instant};

    #[test]
    fn duration_to_ms_zero_yields_zero() {
        assert_eq!(duration_to_ms(Duration::ZERO), 0);
    }

    #[test]
    fn duration_to_ms_one_ms_rounds_down() {
        assert_eq!(duration_to_ms(Duration::from_millis(1)), 1);
        assert_eq!(duration_to_ms(Duration::from_micros(999)), 0);
    }

    /// `u32::MAX = 4_294_967_295` is 10 digits — the boundary the
    /// scratch buffer in `write_decimal_u32` is sized for, and the
    /// boundary that determines [`PROC_FD_PATH_BUF`]. A regression
    /// in the scratch sizing surfaces as an index-out-of-bounds
    /// panic here; the zero branch is covered alongside to pin the
    /// distinct early-return code path.
    #[test]
    fn write_decimal_u32_covers_zero_and_max() {
        let mut buf = [0u8; 10];
        let n = write_decimal_u32(&mut buf, 0);
        assert_eq!(&buf[..n], b"0");

        let mut buf = [0u8; 10];
        let n = write_decimal_u32(&mut buf, u32::MAX);
        assert_eq!(&buf[..n], b"4294967295");
    }

    /// `i32::MAX` (= `2_147_483_647`, 10 digits) is the worst-case
    /// fd value [`std::os::fd::OwnedFd::as_raw_fd`] could legitimately
    /// return. The buffer must accommodate prefix (14) + digits (10) +
    /// NUL (1) = 25 bytes; this pins the [`PROC_FD_PATH_BUF`] sizing.
    #[test]
    fn format_proc_fd_path_i32_max_fits_buffer() {
        let mut buf = [0u8; PROC_FD_PATH_BUF];
        let cstr = format_proc_fd_path(&mut buf, i32::MAX);
        assert_eq!(cstr.to_bytes(), b"/proc/self/fd/2147483647");
        assert_eq!(cstr.to_bytes_with_nul().len(), 25);
    }

    /// Live syscall: open an inotify instance and an `O_PATH` fd on
    /// the temp dir, then verify the fused helper round-trips
    /// against the real kernel. Exercises the full allocation-free
    /// path that the watcher's hot paths now depend on.
    #[test]
    fn inotify_add_watch_fd_round_trip() {
        let dir = tempfile::tempdir().expect("tempdir");
        let inotify_fd = super::inotify_init().expect("inotify_init");
        let target_fd = super::open_o_path(dir.path()).expect("open_o_path");
        let wd = super::inotify_add_watch_fd(&inotify_fd, &target_fd, libc::IN_CREATE)
            .expect("inotify_add_watch_fd");
        assert!(wd >= 0, "wd must be non-negative");
        let _ = super::inotify_rm_watch(&inotify_fd, wd);
    }

    /// `epoll_wait` with a past deadline must non-blocking-poll: the
    /// per-iteration `saturating_duration_since` inside the retry
    /// loop clamps the elapsed budget to `Duration::ZERO`, which
    /// `duration_to_ms` encodes as `0`. Empty epoll instance ⇒
    /// kernel returns `0` ready fds immediately.
    ///
    /// Pins the deadline-honoured-across-`EINTR` conversion path. We
    /// cannot reliably inject `EINTR` in a portable test; the
    /// past-deadline + empty-instance path exercises the same
    /// conversion every retry iteration uses, so any regression in
    /// the per-iter recompute surfaces as a long block here.
    #[test]
    fn epoll_wait_past_deadline_returns_promptly() {
        let epoll = epoll_create().expect("epoll");
        let mut out = [libc::epoll_event { events: 0, u64: 0 }; 2];
        let past = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("1s before Instant::now() is representable");
        let start = Instant::now();
        let n = epoll_wait(&epoll, &mut out, Some(past)).expect("wait ok");
        let elapsed = start.elapsed();
        assert_eq!(n, 0, "no fds were registered");
        assert!(
            elapsed < Duration::from_millis(100),
            "past deadline must non-blocking-poll; took {elapsed:?}"
        );
    }

    /// `epoll_wait` with a finite future deadline must honour it on
    /// an empty instance: the wait blocks ≈ `deadline - now`, then
    /// returns `0` ready fds. Companion to
    /// `epoll_wait_past_deadline_returns_promptly` — exercises the
    /// `Some(d)` branch with a non-zero remaining budget.
    #[test]
    fn epoll_wait_honours_future_deadline() {
        let epoll = epoll_create().expect("epoll");
        let mut out = [libc::epoll_event { events: 0, u64: 0 }; 2];
        let budget = Duration::from_millis(60);
        let start = Instant::now();
        let n = epoll_wait(&epoll, &mut out, Some(start + budget)).expect("wait ok");
        let elapsed = start.elapsed();
        assert_eq!(n, 0, "no fds were registered");
        // Lower bound is loose: `epoll_wait` rounds the budget down to
        // whole milliseconds; the kernel may also return slightly
        // early on some hosts, so we allow ~10ms slack.
        assert!(
            elapsed >= budget.saturating_sub(Duration::from_millis(10)),
            "wait should approach deadline; took {elapsed:?} for {budget:?}"
        );
        // Upper bound: the wait must not significantly exceed the
        // deadline. A regression in the per-iteration recompute (e.g.
        // re-arming the original budget on `EINTR`) would show up as
        // a multi-window overrun here.
        assert!(
            elapsed < budget + Duration::from_millis(150),
            "wait should not significantly exceed deadline; took {elapsed:?} for {budget:?}"
        );
    }
}
