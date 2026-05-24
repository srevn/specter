//! Thin libc wrappers — the lone `unsafe` surface in this module.
//! Module-level `#[allow(unsafe_code)]` keeps the audit boundary at
//! the file edge; every direct syscall in the kqueue backend lives
//! here. Mirror of [`crate::inotify::ffi`]'s discipline.
//!
//! The surface is two-flavoured:
//!
//! - **kqueue primitives** ([`kqueue_new`], [`register_vnode`],
//!   [`kevent_drain`]). The `Kevent` newtype is
//!   `#[repr(transparent)]` so we can hand a `&mut [Kevent]` to
//!   `kevent(2)` as a `*mut libc::kevent`. Accessors return raw
//!   `flags` / `fflags` / `udata`; the `udata` token is opaque at
//!   this layer — consumers encode/decode at their own boundary.
//!
//! - **Path-to-FD primitives** ([`open_for_watch`], [`stat_kind`]).
//!   The watcher's race-free install pattern: open with the
//!   platform's "event-only" flag, `fstat` to discover the kind.
//!   The inotify backend integrates the same shape directly into
//!   its `ffi` module ([`crate::inotify::ffi::open_o_path`] /
//!   [`crate::inotify::ffi::fstat_kind`]); we mirror that here so
//!   the `unsafe` surface per backend is one file.
//!
//! ## Non-blocking discipline
//!
//! [`kevent_drain`] passes a zero `timespec` so `kevent(2)` returns
//! immediately whether or not events are queued. The trait surface
//! ([`crate::FsWatcher::drain_ready`]) wraps this in a drain-to-
//! empty loop; callers block at the reactor level on
//! [`std::os::fd::AsFd::as_fd`] of the watcher, never inside the FFI.

#![allow(unsafe_code)]

use libc::{c_int, kevent, kqueue, timespec};
use specter_core::ResourceKind;
use std::ffi::CString;
use std::io::{self, Error};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

/// One `libc::kevent` slot. Constructed via `Kevent::zeroed()` for output
/// arrays and via the per-op builders (`vnode_register`, etc.) for input
/// changes. Copy because `libc::kevent` is plain POD on macOS / FreeBSD.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(super) struct Kevent(libc::kevent);

impl Kevent {
    pub(super) const fn zeroed() -> Self {
        // SAFETY: `libc::kevent` is plain old data — every field is an
        // integer or a pointer. Zero is a valid bit pattern for all.
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }

    pub(super) const fn flags(&self) -> u16 {
        self.0.flags
    }

    pub(super) const fn fflags(&self) -> u32 {
        self.0.fflags
    }

    /// Raw correlation token attached at registration time. The FFI
    /// treats `udata` as opaque; consumers encode/decode it at their
    /// own boundary. `udata == 0` is the "no payload" sentinel —
    /// consumers treat a zero round-trip as a dropped event and
    /// reserve non-zero values for live dispatch.
    pub(super) fn udata(&self) -> u64 {
        self.0.udata as u64
    }
}

/// `kqueue(2)`. Fresh queue fd; owned by the watcher and exposed
/// to its reactor through the [`std::os::fd::AsFd`] supertrait.
pub(super) fn kqueue_new() -> io::Result<OwnedFd> {
    // SAFETY: kqueue() takes no arguments and returns a fresh fd or -1.
    let raw = unsafe { kqueue() };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ `kqueue` returned a fresh fd we now own.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Register (or re-register) a vnode watch with the caller-supplied
/// fflags mask, edge-triggered. `udata` is an opaque correlation
/// token; events round-trip it via [`Kevent::udata`] so the watcher
/// needs no fd↔id map. Callers should pick non-zero values —
/// `udata == 0` is the "no payload" sentinel reserved for kernel
/// records that lose their payload (cancelled / dropped).
///
/// `fflags` is the caller's responsibility — the kqueue translator
/// (`super::translate::class_set_to_fflags`) is the single producer of
/// the mask in the watcher's `watch()` path. `EV_ADD` on an existing
/// `(fd, EVFILT_VNODE)` entry overwrites the prior fflags atomically.
pub(super) fn register_vnode(
    kq: &OwnedFd,
    target: &OwnedFd,
    udata: u64,
    fflags: u32,
) -> io::Result<()> {
    vnode_change(kq, target, udata, libc::EV_ADD | libc::EV_CLEAR, fflags)
}

#[allow(clippy::similar_names)]
fn vnode_change(
    kq: &OwnedFd,
    target: &OwnedFd,
    udata: u64,
    flags: u16,
    fflags: u32,
) -> io::Result<()> {
    let mut ev = Kevent::zeroed();
    // `OwnedFd::as_raw_fd` returns a non-negative `RawFd` by API contract
    // (the type wraps an owned, opened descriptor). The conversion to
    // `usize` is therefore lossless. Panic on the unreachable path
    // rather than silently registering against fd 0 (stdin) — a
    // misregistration there would corrupt the watcher's view of every
    // resource keyed at fd 0's slot.
    ev.0.ident = usize::try_from(target.as_raw_fd())
        .expect("OwnedFd::as_raw_fd is non-negative by API contract");
    ev.0.filter = libc::EVFILT_VNODE;
    ev.0.flags = flags;
    ev.0.fflags = fflags;
    ev.0.udata = udata as *mut _;
    kevent_change(kq, &ev.0)
}

/// Drain pending events into `out`. Retries on `EINTR`.
///
/// Non-blocking: passes a zero `timespec` (NOT a NULL pointer — that
/// would block indefinitely, the opposite of what the trait
/// contract needs) so `kevent(2)` returns immediately whether or not
/// events are queued. The trait surface
/// ([`crate::FsWatcher::drain_ready`]) wraps this in a drain-to-
/// empty loop that terminates when this returns `0`; callers block
/// at the reactor level on the kqueue's [`std::os::fd::AsFd`]
/// surface, never inside this helper.
///
/// Returns the number of slots in `out` populated by the kernel.
pub(super) fn kevent_drain(kq: &OwnedFd, out: &mut [Kevent]) -> io::Result<usize> {
    let len_c: c_int = c_int::try_from(out.len()).unwrap_or(c_int::MAX);
    // SAFETY: `timespec` is plain POD — `tv_sec`/`tv_nsec` are
    // integers. A zero bit pattern is `{ tv_sec: 0, tv_nsec: 0 }`,
    // the non-blocking signal documented for `kevent(2)`.
    let ts: timespec = unsafe { MaybeUninit::zeroed().assume_init() };
    loop {
        // SAFETY: `out` is a `&mut [Kevent]` of length `out.len()`;
        // `Kevent` is `#[repr(transparent)]` over `libc::kevent`, so
        // the slice's start pointer is a valid `*mut libc::kevent`
        // for `len_c` elements. The kernel writes only the first
        // `n` (returned) slots and treats the rest as undefined;
        // callers consume only `out[..n]`. `ts` is a stack binding
        // that outlives the syscall; `&ts` is a non-NULL pointer to
        // a zero `timespec`.
        let n = unsafe {
            kevent(
                kq.as_raw_fd(),
                std::ptr::null(),
                0,
                out.as_mut_ptr().cast::<libc::kevent>(),
                len_c,
                std::ptr::from_ref(&ts),
            )
        };
        if n >= 0 {
            // n was non-negative; cast to usize is exact.
            return Ok(usize::try_from(n).unwrap_or(0));
        }
        let e = Error::last_os_error();
        if e.raw_os_error() == Some(libc::EINTR) {
            continue;
        }
        return Err(e);
    }
}

/// Apply one vnode change (register / disable / enable). Single-shot:
/// `kevent` with `nchanges = 1` and `nevents = 0`.
fn kevent_change(kq: &OwnedFd, ev: &libc::kevent) -> io::Result<()> {
    // SAFETY: `ev` is a valid `*const libc::kevent` (single element);
    // `nchanges = 1`, `nevents = 0`, so the kernel reads but does not
    // write. `timeout = NULL` makes the call non-blocking for changes.
    let n = unsafe {
        kevent(
            kq.as_raw_fd(),
            std::ptr::from_ref(ev),
            1,
            std::ptr::null_mut(),
            0,
            std::ptr::null(),
        )
    };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    Ok(())
}

/// Open `path` with the kqueue-friendly flag set for the current
/// target. macOS uses `O_EVTONLY` (Darwin-private flag for "open for
/// event monitoring only" — won't pin the file against `unlink`);
/// FreeBSD falls back to `O_RDONLY`. Both unconditionally apply
/// `O_NOFOLLOW` — symlinks at the anchor path fail with `ELOOP`
/// rather than silently traversing. v1 has no follow-symlinks opt-in.
///
/// Errors propagate verbatim (`EMFILE` / `ENFILE` / `ENOENT` /
/// `EACCES` / `ELOOP` are the FD-pressure / pending-path / symlink
/// cases the engine surfaces via `WatchOpRejected`).
pub(super) fn open_for_watch(path: &Path) -> io::Result<OwnedFd> {
    let cstr = CString::new(path.as_os_str().as_bytes())
        .map_err(|_| io::Error::other("path contains NUL byte"))?;

    #[cfg(target_os = "macos")]
    let flags = libc::O_EVTONLY | libc::O_NOFOLLOW;

    #[cfg(target_os = "freebsd")]
    let flags = libc::O_RDONLY | libc::O_NOFOLLOW;

    // SAFETY: `cstr` is a valid NUL-terminated C string for the
    // duration of the call; `flags` is a valid `O_*` bit set. `open`
    // returns a non-negative fd or -1 on error.
    let raw = unsafe { libc::open(cstr.as_ptr(), flags) };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ `open` handed us a fresh fd we now own.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Stat the open fd to determine `ResourceKind`. Used by the watcher's
/// per-resource kind cache — `NOTE_WRITE` on a Dir means structural
/// change; on a File, content modification.
pub(super) fn stat_kind(fd: &OwnedFd) -> io::Result<ResourceKind> {
    let mut s = MaybeUninit::<libc::stat>::uninit();
    // SAFETY: `fd` is a valid open fd; `s` is a writable
    // `*mut libc::stat`. `fstat` returns 0 on success, populating `s`.
    let n = unsafe { libc::fstat(fd.as_raw_fd(), s.as_mut_ptr()) };
    if n < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `fstat` returned 0 above, so every field of `s` is now
    // initialized.
    let s = unsafe { s.assume_init() };
    Ok(match s.st_mode & libc::S_IFMT {
        libc::S_IFDIR => ResourceKind::Dir,
        libc::S_IFREG => ResourceKind::File,
        _ => ResourceKind::Unknown,
    })
}

#[cfg(test)]
mod tests {
    use super::{Kevent, kevent_drain, kqueue_new};
    use std::time::{Duration, Instant};

    #[test]
    fn kevent_zeroed_is_default_state() {
        let ev = Kevent::zeroed();
        // `EVFILT_*` constants are negative on macOS / FreeBSD; zero
        // is a valid (and unused) bit pattern that we never treat as
        // a real filter, confirming the zero-init is "untriggered".
        // `udata` of zero is the "no payload" sentinel — consumers
        // skip the event on a zero round-trip.
        assert_eq!(ev.flags(), 0);
        assert_eq!(ev.fflags(), 0);
        assert_eq!(ev.udata(), 0, "zero-init udata round-trips to zero");
    }

    /// `kevent_drain` on an empty queue must return promptly: the
    /// zero `timespec` armed inside makes `kevent(2)` return
    /// immediately with `0` events. Pins the non-blocking contract —
    /// a regression that swapped the zero `timespec` for a NULL
    /// pointer (the historical "block forever" sentinel) would
    /// deadlock here on a fresh kqueue, and any non-zero `timespec`
    /// would force a deliberate sleep.
    #[test]
    fn kevent_drain_empty_queue_returns_promptly() {
        let kq = kqueue_new().expect("kqueue");
        let mut out = [Kevent::zeroed(); 4];
        let start = Instant::now();
        let n = kevent_drain(&kq, &mut out).expect("drain ok");
        let elapsed = start.elapsed();
        assert_eq!(n, 0, "no events were registered");
        assert!(
            elapsed < Duration::from_millis(100),
            "non-blocking drain must return promptly; took {elapsed:?}"
        );
    }
}
