//! `KqueueWakeHandle` — cross-thread wake-up signal for an in-flight
//! `KqueueWatcher::poll_until`.
//!
//! Backed by `EVFILT_USER` + `NOTE_TRIGGER` on the watcher's kqueue fd.
//! Both macOS (since 10.6) and FreeBSD (since 8.0) support this filter
//! type with identical semantics — no self-pipe fallback is needed.
//!
//! # Lifecycle and the `Arc<OwnedFd>` discipline
//!
//! The handle holds an `Arc<OwnedFd>` clone of the watcher's kqueue fd.
//! As long as *any* clone exists (the watcher itself plus every wake
//! handle in flight), the kernel-side fd stays open and `wake()` is
//! valid. Drop of the last clone closes the fd, kernel-reaping every
//! pending event including queued user triggers.
//!
//! `wake()` after the watcher has been dropped is a no-op-equivalent: a
//! `NOTE_TRIGGER` lands on a kqueue with no consumer; the next drop of
//! a wake-handle clone reaps it. No use-after-free is possible — the
//! Arc keeps the underlying fd live.

use crate::WakeHandle;
use crate::kqueue::ffi;
use std::os::fd::OwnedFd;
use std::sync::Arc;

/// Cross-thread wake-up handle for `KqueueWatcher::poll_until`.
///
/// Cheap to clone (`Arc` + `usize`). Multiple handles may coexist;
/// concurrent `wake()` calls coalesce into one user-event delivery on
/// the watcher's kqueue. Idempotent on consecutive wakes within one
/// `poll_until` window.
#[derive(Debug, Clone)]
pub(crate) struct KqueueWakeHandle {
    kq: Arc<OwnedFd>,
    wake_ident: usize,
}

impl KqueueWakeHandle {
    pub(super) const fn new(kq: Arc<OwnedFd>, wake_ident: usize) -> Self {
        Self { kq, wake_ident }
    }
}

impl WakeHandle for KqueueWakeHandle {
    fn wake(&self) {
        if let Err(e) = ffi::trigger_user_event(&self.kq, self.wake_ident) {
            // Reachable when the watcher's kqueue fd has been closed
            // underneath us (last Arc dropped while a clone is still
            // triggering). The handle itself stays sound; subsequent
            // wakes silently hit the same dead fd — no consumer will
            // drain the resulting `NOTE_TRIGGER`. Benign during
            // shutdown (watcher dropped while a wake handle still
            // triggers); log at `debug` rather than `warn` to avoid
            // operational noise on a routine teardown race. Mirror
            // of [`super::super::inotify::wake`]'s twin.
            tracing::debug!(
                ident = self.wake_ident,
                error = ?e,
                "kqueue wake() syscall failed (typically watcher dropped); consumer stale"
            );
        }
    }

    fn clone_box(&self) -> Box<dyn WakeHandle> {
        Box::new(self.clone())
    }
}
