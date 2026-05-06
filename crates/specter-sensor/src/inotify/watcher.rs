//! `InotifyWatcher` — inotify-backed `FsWatcher` impl.
//!
//! Stub — the real implementation lands in Phase B5–B9. The placeholder
//! satisfies [`crate::FsWatcher`] so lib's `DefaultWatcher` alias and
//! `default_watcher` factory type-check on Linux during the helper-
//! cluster phases (B1–B4). `Self::new` returns `ENOSYS` so any caller
//! that flows past `default_watcher` reaches a defined "not yet
//! implemented" state rather than panicking.

use crate::inotify::wake::InotifyWakeHandle;
use crate::{FsWatcher, WakeHandle, WatchFailure, WatcherEvent};
use specter_core::{ClassSet, ResourceId, ResourceKind};
use std::io;
use std::path::Path;
use std::time::Instant;

/// inotify-backed watcher. Stub — replaced in Phase B5 with the
/// epoll-driven, fd-state-bearing implementation.
#[derive(Debug)]
pub struct InotifyWatcher {
    _phantom: (),
}

impl InotifyWatcher {
    /// Construct a fresh watcher. Until Phase B5 lands the real syscall-
    /// driven body, this returns `ENOSYS`. The bin's `default_watcher`
    /// surfaces the failure as a fatal startup error — symmetric with
    /// the kqueue branch's behaviour when its own `kqueue_new` fails.
    pub fn new() -> io::Result<Self> {
        Err(io::Error::from_raw_os_error(libc::ENOSYS))
    }
}

impl FsWatcher for InotifyWatcher {
    fn watch(
        &mut self,
        _r: ResourceId,
        _path: &Path,
        _kind: ResourceKind,
        _events: ClassSet,
    ) -> Result<(), WatchFailure> {
        // Unreachable on a healthy bin during the helper-cluster phases:
        // `Self::new` errors out before any `WatchOp::Watch` reaches the
        // trait surface. Returning `Invariant` rather than panicking
        // keeps the failure path explicit if a test fixture wires a
        // post-`new`-success path.
        Err(WatchFailure::Invariant {
            errno: libc::ENOSYS,
        })
    }

    fn unwatch(&mut self, _r: ResourceId) {}

    fn suppress(&mut self, _r: ResourceId) {}

    fn unsuppress(&mut self, _r: ResourceId) {}

    fn poll_until(
        &mut self,
        _deadline: Option<Instant>,
        _out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure> {
        Err(WatchFailure::Invariant {
            errno: libc::ENOSYS,
        })
    }

    fn wake_handle(&self) -> Box<dyn WakeHandle> {
        Box::new(InotifyWakeHandle::placeholder())
    }
}
