//! `InotifyWakeHandle` — cross-thread wake for
//! [`super::watcher::InotifyWatcher::poll_until`].
//!
//! Stub — the real eventfd-backed implementation lands in Phase B4.
//! The placeholder type satisfies [`crate::WakeHandle`] so the lib's
//! `DefaultWatcher` re-export and the `wake_handle` trait method type-
//! check on Linux during Phases B1–B3.

use crate::WakeHandle;

/// Stub wake handle.
///
/// Replaced in Phase B4 with the eventfd-backed `Arc<OwnedFd>` form. The
/// placeholder is intentionally unconstructible outside this module —
/// only [`super::watcher::InotifyWatcher`]'s stub `wake_handle` reaches
/// it, and that path is unreachable until Phase B5 makes
/// `InotifyWatcher::new` succeed.
#[derive(Debug, Clone)]
pub struct InotifyWakeHandle {
    _phantom: (),
}

impl InotifyWakeHandle {
    pub(super) const fn placeholder() -> Self {
        Self { _phantom: () }
    }
}

impl WakeHandle for InotifyWakeHandle {
    fn wake(&self) {
        // Unreachable while the watcher's `new` is stubbed (Phase B5).
        // Logging here would be misleading — the call path doesn't exist
        // on a healthy bin during the helper-cluster phases. Real
        // implementation lands in Phase B4.
    }

    fn clone_box(&self) -> Box<dyn WakeHandle> {
        Box::new(self.clone())
    }
}
