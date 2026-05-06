//! `InotifyWakeHandle` — cross-thread wake-up signal for an in-flight
//! [`super::watcher::InotifyWatcher::poll_until`].
//!
//! Backed by an eventfd (`EFD_NONBLOCK | EFD_CLOEXEC`). The watcher's
//! epoll instance listens on `(inotify_fd, wake_fd)`; concurrent
//! [`wake`](Self::wake) calls accumulate kernel-side, and a single
//! `eventfd_read` from `poll_until` drains the entire counter
//! atomically. Mirror of [`crate::kqueue::wake::KqueueWakeHandle`] with
//! the eventfd in place of `EVFILT_USER`.
//!
//! ## Lifecycle and the `Arc<OwnedFd>` discipline
//!
//! The handle holds an `Arc<OwnedFd>` clone of the watcher's eventfd.
//! As long as *any* clone exists (the watcher itself plus every wake
//! handle in flight), the kernel-side fd stays open and `wake()` is
//! valid. Drop of the last clone closes the fd, kernel-reaping the
//! eventfd's pending counter — a queued non-zero value is silently
//! discarded by the kernel at close time, with no UB.
//!
//! `wake()` after the watcher has been dropped is a no-op-equivalent:
//! the eventfd_write lands on a counter no consumer will drain, and the
//! Arc keeps the fd live until the last handle clone drops. No
//! use-after-free is possible.

use crate::WakeHandle;
use crate::inotify::ffi;
use std::os::fd::OwnedFd;
use std::sync::Arc;

/// Cross-thread wake-up handle for [`super::watcher::InotifyWatcher::poll_until`].
///
/// Cheap to clone (one `Arc` increment). Multiple handles may coexist;
/// concurrent `wake()` calls accumulate in the eventfd counter, which a
/// single `eventfd_read` consumes atomically. Idempotent on consecutive
/// wakes within one `poll_until` window.
#[derive(Debug, Clone)]
pub struct InotifyWakeHandle {
    wake_fd: Arc<OwnedFd>,
}

impl InotifyWakeHandle {
    /// Construct a handle backed by `wake_fd`. The watcher creates the
    /// eventfd in its constructor (Phase B5), wraps it in `Arc`, and
    /// hands clones to every caller of [`crate::FsWatcher::wake_handle`].
    pub(super) const fn new(wake_fd: Arc<OwnedFd>) -> Self {
        Self { wake_fd }
    }
}

impl WakeHandle for InotifyWakeHandle {
    fn wake(&self) {
        if let Err(e) = ffi::eventfd_write(&self.wake_fd, 1) {
            // Reachable when the eventfd has been closed underneath us
            // (last Arc dropped while a clone is still triggering). The
            // handle itself stays sound; subsequent wakes silently hit
            // the same dead fd. Logging at warn keeps the consumer-stale
            // case visible in operational dashboards.
            tracing::warn!(
                error = ?e,
                "inotify wake() failed; consumer may be stale"
            );
        }
    }

    fn clone_box(&self) -> Box<dyn WakeHandle> {
        Box::new(self.clone())
    }
}
