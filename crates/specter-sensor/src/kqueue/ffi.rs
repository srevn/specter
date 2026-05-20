//! Thin `libc::kevent` wrappers — the lone `unsafe` surface in this
//! crate. Each function below is a direct syscall; module-level
//! `#[allow(unsafe_code)]` keeps the audit boundary at the file edge.
//!
//! The `Kevent` newtype is `#[repr(transparent)]` so we can hand a
//! `&mut [Kevent]` to `kevent(2)` as a `*mut libc::kevent`. Accessors
//! return raw `flags` / `fflags` / `udata`; the `udata` token is opaque
//! at this layer — consumers encode/decode at their own boundary.
//!
//! ## Deadline tracking
//!
//! The wait primitives ([`kevent_drain`]) take an `Option<Instant>` —
//! not a pre-computed `timespec` — and own deadline tracking across
//! `EINTR` retries. The remaining budget is recomputed inside the retry
//! loop on every iteration so wall-clock progress between syscall
//! re-entries is preserved. This is the only layer that *can* re-derive
//! the remaining budget, so deadline math belongs here rather than at
//! every caller.

#![allow(unsafe_code)]

use libc::{c_int, kevent, kqueue, timespec};
use std::io::{self, Error};
use std::mem::MaybeUninit;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::time::Instant;

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

    /// `true` iff this kevent corresponds to the `EVFILT_USER` wake
    /// ident reserved at watcher init. Wake events are filtered out
    /// before normalization — they have no `ResourceId` payload.
    pub(super) const fn is_user_event(&self, wake_ident: usize) -> bool {
        self.0.filter == libc::EVFILT_USER && self.0.ident == wake_ident
    }

    /// Raw correlation token attached at registration time. The FFI
    /// treats `udata` as opaque; consumers encode/decode it at their
    /// own boundary. `udata == 0` is the "no payload" sentinel —
    /// `register_user_event` leaves `udata` unset, so every wake event
    /// round-trips to zero, and consumers should reserve zero for
    /// non-vnode dispatch.
    pub(super) fn udata(&self) -> u64 {
        self.0.udata as u64
    }
}

/// `kqueue(2)`. Fresh queue fd; held inside `Arc<OwnedFd>` by the
/// watcher and shared with every wake handle.
pub(super) fn kqueue_new() -> io::Result<OwnedFd> {
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
pub(super) fn register_user_event(kq: &OwnedFd, wake_ident: usize) -> io::Result<()> {
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
pub(super) fn trigger_user_event(kq: &OwnedFd, wake_ident: usize) -> io::Result<()> {
    let mut ev = Kevent::zeroed();
    ev.0.ident = wake_ident;
    ev.0.filter = libc::EVFILT_USER;
    ev.0.flags = libc::EV_ENABLE;
    ev.0.fflags = libc::NOTE_TRIGGER;
    kevent_change(kq, &ev.0)
}

/// Register (or re-register) a vnode watch with the caller-supplied
/// fflags mask, edge-triggered. `udata` is an opaque correlation
/// token; events round-trip it via [`Kevent::udata`] so the watcher
/// needs no fd↔id map. Callers should pick non-zero values —
/// `udata == 0` is the "no payload" sentinel reserved for the
/// `EVFILT_USER` wake event.
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
/// `deadline = None` blocks indefinitely (null `timespec`); `Some(d)`
/// arms the kernel-side wait at a per-iteration `timespec` re-derived
/// from `d.saturating_duration_since(Instant::now())`. The remaining
/// budget is recomputed inside the retry loop so a signal-induced
/// `EINTR` re-entry does not re-arm the full original budget — the
/// caller's deadline is total wall-clock budget, not per-syscall.
///
/// A `Some(d)` with `d <= Instant::now()` collapses to a zero timespec
/// (non-blocking poll); `kevent(2)` returns immediately with `0` events
/// if no events are pending.
///
/// Returns the number of slots in `out` populated by the kernel.
pub(super) fn kevent_drain(
    kq: &OwnedFd,
    out: &mut [Kevent],
    deadline: Option<Instant>,
) -> io::Result<usize> {
    let len_c: c_int = c_int::try_from(out.len()).unwrap_or(c_int::MAX);
    loop {
        // Re-derive the remaining budget on every iteration so an
        // `EINTR` retry resumes against wall-clock progress, not the
        // original deadline budget. `None` keeps the indefinite block
        // via a null timespec pointer; the `Some` arm holds the
        // freshly-built `timespec` on the stack across the syscall.
        let ts;
        let timeout_ptr = match deadline {
            None => std::ptr::null(),
            Some(d) => {
                ts = duration_to_timespec(d.saturating_duration_since(Instant::now()));
                std::ptr::from_ref(&ts)
            }
        };
        // SAFETY: `out` is a `&mut [Kevent]` of length `out.len()`;
        // `Kevent` is `#[repr(transparent)]` over `libc::kevent`, so the
        // slice's start pointer is a valid `*mut libc::kevent` for
        // `len_c` elements. The kernel writes only the first `n`
        // (returned) slots and treats the rest as undefined; callers
        // consume only `out[..n]`. `timeout_ptr` is either NULL or
        // points to `ts` whose binding outlives the syscall.
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
fn duration_to_timespec(dur: std::time::Duration) -> timespec {
    timespec {
        tv_sec: dur.as_secs() as libc::time_t,
        tv_nsec: libc::c_long::from(dur.subsec_nanos() as i32),
    }
}

#[cfg(test)]
mod tests {
    use super::{Kevent, duration_to_timespec, kevent_drain, kqueue_new};
    use std::time::{Duration, Instant};

    #[test]
    fn duration_to_timespec_zero_yields_zero_components() {
        let ts = duration_to_timespec(Duration::ZERO);
        assert_eq!(ts.tv_sec, 0);
        assert_eq!(ts.tv_nsec, 0);
    }

    #[test]
    fn duration_to_timespec_one_sec_one_nano() {
        let ts = duration_to_timespec(Duration::new(1, 1));
        assert_eq!(ts.tv_sec, 1);
        assert_eq!(ts.tv_nsec, 1);
    }

    #[test]
    fn kevent_zeroed_is_default_state() {
        let ev = Kevent::zeroed();
        // `EVFILT_*` constants are negative on macOS / FreeBSD; zero is
        // a valid (and unused) bit pattern that we never treat as a
        // real filter, confirming the zero-init is "untriggered". `udata`
        // of zero is the "no payload" sentinel — the wake event leaves
        // it unset; consumers skip the event on a zero round-trip.
        assert_eq!(ev.flags(), 0);
        assert_eq!(ev.fflags(), 0);
        assert_eq!(ev.udata(), 0, "zero-init udata round-trips to zero");
        // Zero `filter` is not `EVFILT_USER` (a negative value on both
        // BSDs), so an arbitrary user-ident probe rejects.
        assert!(
            !ev.is_user_event(0xDEAD_BEEF),
            "zero-init does not look like a user event"
        );
    }

    /// `kevent_drain` with a past deadline must non-blocking-poll: the
    /// per-iteration `saturating_duration_since` inside the retry loop
    /// clamps the elapsed budget to `Duration::ZERO`, which
    /// `duration_to_timespec` encodes as a zero `timespec`. Empty
    /// queue ⇒ kernel returns `0` events immediately.
    ///
    /// Pins the deadline-honoured-across-`EINTR` conversion path. We
    /// cannot reliably inject `EINTR` in a portable test; the
    /// past-deadline + empty-queue path exercises the same conversion
    /// every retry iteration uses, so any regression in the per-iter
    /// recompute surfaces as a long block here.
    #[test]
    fn kevent_drain_past_deadline_returns_promptly() {
        let kq = kqueue_new().expect("kqueue");
        let mut out = [Kevent::zeroed(); 4];
        let past = Instant::now()
            .checked_sub(Duration::from_secs(1))
            .expect("1s before Instant::now() is representable");
        let start = Instant::now();
        let n = kevent_drain(&kq, &mut out, Some(past)).expect("drain ok");
        let elapsed = start.elapsed();
        assert_eq!(n, 0, "no events were registered");
        assert!(
            elapsed < Duration::from_millis(100),
            "past deadline must non-blocking-poll; took {elapsed:?}"
        );
    }

    /// `kevent_drain` with a finite future deadline must honour it on
    /// an empty queue: the wait blocks ≈ `deadline - now`, then
    /// returns `0` events. Companion to
    /// `kevent_drain_past_deadline_returns_promptly` — exercises the
    /// `Some(d)` branch with a non-zero remaining budget.
    #[test]
    fn kevent_drain_honours_future_deadline() {
        let kq = kqueue_new().expect("kqueue");
        let mut out = [Kevent::zeroed(); 4];
        let budget = Duration::from_millis(60);
        let start = Instant::now();
        let n = kevent_drain(&kq, &mut out, Some(start + budget)).expect("drain ok");
        let elapsed = start.elapsed();
        assert_eq!(n, 0, "no events were registered");
        // Lower bound is loose: `kevent` may return slightly early on
        // some kernels, but it must approach the deadline.
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
