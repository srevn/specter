//! Thin `libc::kevent` wrappers — the lone `unsafe` surface in this
//! crate. Each function below is a direct syscall; module-level
//! `#[allow(unsafe_code)]` keeps the audit boundary at the file edge.
//!
//! The `Kevent` newtype is `#[repr(transparent)]` so we can hand a
//! `&mut [Kevent]` to `kevent(2)` as a `*mut libc::kevent`. Accessors
//! decode the raw flags / fflags / `udata` shape into typed Rust values.

#![allow(unsafe_code)]

use libc::{c_int, kevent, kqueue, timespec};
use slotmap::{Key, KeyData};
use specter_core::ResourceId;
use std::io::{self, Error};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};

/// One `libc::kevent` slot. Constructed via `Kevent::zeroed()` for output
/// arrays and via the per-op builders (`vnode_register`, etc.) for input
/// changes. Copy because `libc::kevent` is plain POD on macOS / FreeBSD.
#[derive(Copy, Clone, Debug)]
#[repr(transparent)]
pub(crate) struct Kevent(libc::kevent);

impl Kevent {
    pub(crate) const fn zeroed() -> Self {
        // SAFETY: `libc::kevent` is plain old data — every field is an
        // integer or a pointer. Zero is a valid bit pattern for all.
        Self(unsafe { MaybeUninit::zeroed().assume_init() })
    }

    pub(crate) const fn flags(&self) -> u16 {
        self.0.flags
    }

    pub(crate) const fn fflags(&self) -> u32 {
        self.0.fflags
    }

    /// `true` iff this kevent corresponds to the `EVFILT_USER` wake
    /// ident reserved at watcher init. Wake events are filtered out
    /// before normalization — they have no `ResourceId` payload.
    pub(crate) const fn is_user_event(&self, wake_ident: usize) -> bool {
        self.0.filter == libc::EVFILT_USER && self.0.ident == wake_ident
    }

    /// Decode `udata` back to a `ResourceId`. Returns `None` if the udata
    /// is the zero sentinel (every wake event carries `udata = 0`).
    pub(crate) fn resource_id(&self) -> Option<ResourceId> {
        let raw = self.0.udata as u64;
        if raw == 0 {
            return None;
        }
        Some(ResourceId::from(KeyData::from_ffi(raw)))
    }
}

/// `kqueue(2)`. Fresh queue fd; held inside `Arc<OwnedFd>` by the
/// watcher and shared with every wake handle.
pub(crate) fn kqueue_new() -> io::Result<OwnedFd> {
    // SAFETY: kqueue() takes no arguments and returns a fresh fd or -1.
    let raw = unsafe { kqueue() };
    if raw < 0 {
        return Err(Error::last_os_error());
    }
    // SAFETY: `raw >= 0` ⇒ `kqueue` returned a fresh fd we now own.
    Ok(unsafe { OwnedFd::from_raw_fd(raw) })
}

/// Register `EVFILT_USER` for the wake ident. The watcher does this once
/// at construction; only the watcher's `poll_until` consumes the wakes.
pub(crate) fn register_user_event(kq: &OwnedFd, wake_ident: usize) -> io::Result<()> {
    let mut ev = Kevent::zeroed();
    ev.0.ident = wake_ident;
    ev.0.filter = libc::EVFILT_USER;
    ev.0.flags = libc::EV_ADD | libc::EV_CLEAR;
    ev.0.fflags = libc::NOTE_FFNOP;
    kevent_change(kq, &ev.0)
}

/// Trigger the wake ident — issues `NOTE_TRIGGER` so any in-flight
/// `kevent_drain` returns promptly. Idempotent on the kernel side
/// (concurrent triggers coalesce).
pub(crate) fn trigger_user_event(kq: &OwnedFd, wake_ident: usize) -> io::Result<()> {
    let mut ev = Kevent::zeroed();
    ev.0.ident = wake_ident;
    ev.0.filter = libc::EVFILT_USER;
    ev.0.flags = libc::EV_ENABLE;
    ev.0.fflags = libc::NOTE_TRIGGER;
    kevent_change(kq, &ev.0)
}

/// Register a vnode watch with the caller-supplied fflags mask,
/// edge-triggered. `udata` carries the engine's `ResourceId.as_ffi()` so
/// events round-trip the id without the watcher needing a separate fd↔id
/// map.
///
/// `fflags` is the caller's responsibility — the kqueue translator
/// (`super::translate::class_set_to_fflags`) is the single producer of the
/// mask in the watcher's `watch()` path. The same call doubles as a
/// re-registration: `EV_ADD` on an existing `(fd, EVFILT_VNODE)` entry
/// overwrites the prior fflags without affecting other kevent state
/// (notably the `EV_DISABLE` bit, which is preserved per the kqueue man
/// page). The watcher exploits this to update the registered mask
/// without closing or reopening the fd.
pub(crate) fn register_vnode(
    kq: &OwnedFd,
    target: &OwnedFd,
    r: ResourceId,
    fflags: u32,
) -> io::Result<()> {
    vnode_change(kq, target, r, libc::EV_ADD | libc::EV_CLEAR, fflags)
}

/// `EV_DISABLE` — silences delivery without removing the registration.
///
/// `fflags` should be the **currently-registered** mask for this vnode
/// (the value the watcher's `registered_fflags` cache holds). Passing
/// the live mask matters: empirically, macOS xnu's `EV_DISABLE` /
/// `EV_ENABLE` paths overwrite the registered fflags with whatever the
/// caller supplies — passing `0` would silently clear `NOTE_WRITE` /
/// `NOTE_ATTRIB` / etc. on the next re-enable. FreeBSD preserves fflags
/// across disable/enable per `kqueue_register`, so the value is a no-op
/// there. Treating both backends identically (always pass the cached
/// mask) is correct on both. Only the disable bit is intentionally
/// changed.
pub(crate) fn disable_vnode(
    kq: &OwnedFd,
    target: &OwnedFd,
    r: ResourceId,
    fflags: u32,
) -> io::Result<()> {
    vnode_change(kq, target, r, libc::EV_DISABLE, fflags)
}

/// `EV_ENABLE` — restores delivery on a previously disabled
/// registration. See [`disable_vnode`] for the fflags-passthrough
/// rationale.
pub(crate) fn enable_vnode(
    kq: &OwnedFd,
    target: &OwnedFd,
    r: ResourceId,
    fflags: u32,
) -> io::Result<()> {
    vnode_change(kq, target, r, libc::EV_ENABLE, fflags)
}

#[allow(clippy::similar_names)]
fn vnode_change(
    kq: &OwnedFd,
    target: &OwnedFd,
    r: ResourceId,
    flags: u16,
    fflags: u32,
) -> io::Result<()> {
    let mut ev = Kevent::zeroed();
    // `OwnedFd` guarantees a non-negative raw fd, so the cast widens
    // (`i32` → `usize`) without sign-loss.
    ev.0.ident = usize::try_from(target.as_raw_fd()).unwrap_or(0);
    ev.0.filter = libc::EVFILT_VNODE;
    ev.0.flags = flags;
    ev.0.fflags = fflags;
    ev.0.udata = r.data().as_ffi() as *mut _;
    kevent_change(kq, &ev.0)
}

/// Drain pending events into `out`. Retries on `EINTR`. `timeout = None`
/// blocks indefinitely; `Some(ts)` arms the kernel-side wait. Returns
/// the number of slots in `out` populated by the kernel.
pub(crate) fn kevent_drain(
    kq: &OwnedFd,
    out: &mut [Kevent],
    timeout: Option<timespec>,
) -> io::Result<usize> {
    let len = out.len();
    let len_c: c_int = c_int::try_from(len).unwrap_or(c_int::MAX);
    loop {
        let timeout_ptr = timeout
            .as_ref()
            .map_or(std::ptr::null(), std::ptr::from_ref);
        // SAFETY: `out` is a `&mut [Kevent]` of length `len`; `Kevent` is
        // `#[repr(transparent)]` over `libc::kevent`, so the slice's start
        // pointer is a valid `*mut libc::kevent` for `len` elements. The
        // kernel writes only the first `n` (returned) slots and treats
        // the rest as undefined; callers consume only `out[..n]`.
        let n = unsafe {
            kevent(
                kq.as_raw_fd(),
                std::ptr::null(),
                0,
                out.as_mut_ptr().cast::<libc::kevent>(),
                len_c,
                timeout_ptr,
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

/// Apply one change (vnode register/disable/enable, or user-event
/// trigger). Single-shot: `kevent` with `nchanges = 1` and `nevents = 0`.
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

/// Convert a `Duration` (always non-negative — caller clamps) to a
/// kqueue-friendly `timespec`. `Duration::ZERO` means "non-blocking
/// poll"; `kevent` accepts a zero timespec and returns immediately if
/// no events are pending.
///
/// `tv_sec`/`tv_nsec` are signed (`time_t = i64`, `c_long = i64` on
/// 64-bit). The wrapping casts are bounded:
/// - `dur.as_secs()` returns `u64`; durations exceeding `i64::MAX`
///   seconds (~292 billion years) are physically impossible from any
///   `Instant`-derived deadline, and `Duration::MAX` itself only goes
///   to `u64::MAX` seconds.
/// - `subsec_nanos()` returns `u32` capped at `999_999_999`.
#[allow(clippy::cast_possible_wrap)]
pub(crate) fn duration_to_timespec(dur: std::time::Duration) -> timespec {
    timespec {
        tv_sec: dur.as_secs() as libc::time_t,
        tv_nsec: libc::c_long::from(dur.subsec_nanos() as i32),
    }
}
