//! `specter-sensor` — kqueue Watcher + Prober pool. The traits are
//! platform-agnostic; the kqueue implementation is BSD-only and lives in
//! a `#[cfg]`-gated module.

// Sensor legitimately needs `unsafe` for kqueue FFI; `warn` is looser
// than the workspace `deny`. Per-call-site `#[allow(unsafe_code)]` then
// silences the warning at the FFI boundary itself, keeping the audit
// surface narrow.
#![warn(unsafe_code)]

use specter_core::{FsEvent, ProbeRequest, ProfileId, ResourceId, WatchOpts};
use std::io;
use std::path::Path;
use std::time::Instant;

/// Single-threaded filesystem watcher.
///
/// One thread blocks in [`poll_until`](FsWatcher::poll_until); the
/// mutators ([`watch`](FsWatcher::watch) / [`unwatch`](FsWatcher::unwatch)
/// / [`suppress`](FsWatcher::suppress) /
/// [`unsuppress`](FsWatcher::unsuppress)) run on the same thread between
/// `poll_until` calls. Cross-thread coordination — fresh `WatchOp`s
/// arriving on a channel — is the bin's responsibility: it pushes into
/// the channel and calls [`WakeHandle::wake`] on a handle captured before
/// spawning the watcher thread, which interrupts the watcher's in-flight
/// `poll_until` so it can drain the channel and reblock.
///
/// The trait is `Send` (one thread owns the watcher) but not `Sync`
/// (mutators take `&mut self`). The wake handle ([`WakeHandle`]) is the
/// only cross-thread surface.
///
/// # Bin loop pattern
///
/// ```ignore
/// let mut events = Vec::with_capacity(64);
/// loop {
///     // 1. Apply pending WatchOps from the channel.
///     while let Ok(op) = ops_rx.try_recv() {
///         match op {
///             WatchOp::Watch { resource, path, opts } => {
///                 if let Err(e) = watcher.watch(resource, &path, opts) {
///                     // FD-pressure or similar — engine handles via
///                     // `Input::WatchOpRejected`.
///                     engine_inbound.send(/* … */);
///                 }
///             }
///             WatchOp::Unwatch { resource } => watcher.unwatch(resource),
///             WatchOp::Suppress { resource } => watcher.suppress(resource),
///             WatchOp::Unsuppress { resource } => watcher.unsuppress(resource),
///         }
///     }
///     // 2. Block until the deadline, an event, or a wake.
///     events.clear();
///     watcher.poll_until(engine_deadline, &mut events)?;
///     for (resource, event) in events.drain(..) {
///         engine_inbound.send(Input::FsEvent { resource, event });
///     }
/// }
/// ```
///
/// Coalescing under `EV_CLEAR`: multiple writes between drains are
/// reported as one event. The engine's `Settling` state already debounces
/// by rescheduling on every event, so callers must not assume per-write
/// delivery — only "at least one event when something changed."
pub trait FsWatcher: Send {
    /// Install a watch. Returns syscall errors verbatim — `EMFILE` /
    /// `ENFILE` / `ENOENT` / `EACCES` propagate. The bin packages a
    /// non-`Ok` return as `Input::WatchOpRejected { resource, op, errno }`
    /// for the engine, which clamps `watch_demand` to zero and waits for
    /// the parent's next `StructureChanged` to retry.
    fn watch(&mut self, r: ResourceId, path: &Path, opts: WatchOpts) -> io::Result<()>;

    /// Remove a watch. Idempotent on stale ids — the kernel's vnode
    /// registration is cleaned up by closing the underlying fd.
    fn unwatch(&mut self, r: ResourceId);

    /// Silence event delivery on a watched resource. Idempotent; no-op
    /// (with `tracing::warn!`) if `r` is not currently watched. The
    /// underlying kqueue registration is preserved — re-enabling restores
    /// delivery without re-stat-ing.
    fn suppress(&mut self, r: ResourceId);

    /// Restore event delivery. Idempotent; no-op (with `tracing::warn!`)
    /// if `r` is not currently suppressed.
    fn unsuppress(&mut self, r: ResourceId);

    /// Block until the next event(s), the deadline, or a wake. Pushes
    /// normalized `(ResourceId, FsEvent)` pairs into `out` and returns
    /// the count pushed.
    ///
    /// `deadline = None` means "no deadline; block until event or wake."
    /// A returned count of zero is normal: either the deadline arrived
    /// or only a wake fired.
    ///
    /// `EINTR` is retried internally; other syscall errors propagate.
    fn poll_until(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<(ResourceId, FsEvent)>,
    ) -> io::Result<usize>;

    /// Capture a wake handle for cross-thread interruption of
    /// `poll_until`. Cloneable via [`WakeHandle::clone_box`]; concurrent
    /// wakes coalesce in the kernel. Idempotent.
    fn wake_handle(&self) -> Box<dyn WakeHandle>;
}

/// Cross-thread wake-up signal for an in-flight
/// [`FsWatcher::poll_until`].
///
/// Implementations must be cheap to clone (the kqueue impl is two
/// pointer-sized fields wrapping `Arc<OwnedFd> + usize`) and tolerate
/// `wake()` after the watcher's lifecycle has ended — a stale wake is a
/// no-op-equivalent, never UB.
pub trait WakeHandle: Send + Sync {
    /// Issue a wake. The next (or in-flight) `poll_until` returns
    /// promptly; no event is delivered to `out`. Idempotent on the
    /// kernel side — concurrent wakes coalesce into one returned event.
    fn wake(&self);

    /// Clone the handle into a fresh `Box<dyn WakeHandle>`. Keeps the
    /// trait object cloneable without forcing the implementor to be
    /// `Sized`.
    fn clone_box(&self) -> Box<dyn WakeHandle>;
}

impl Clone for Box<dyn WakeHandle> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Multi-threaded probe worker pool.
///
/// `submit` is fire-and-forget (I8); the response lands on the bin's
/// `engine_inbound` channel as `Input::ProbeResponse(...)`. `cancel` is
/// best-effort: queued probes whose `(profile, correlation)` pair
/// no longer matches the prober's per-Profile expectation are skipped
/// silently at worker-recv time. In-flight probes complete to
/// completion; the engine discards their responses via stale-correlation
/// discipline.
///
/// `Send + Sync` so the bin can hold an `Arc<dyn Prober>` (or
/// `Arc<WorkerProber>`) and share it across threads — the engine driver
/// thread submits, signal handlers may cancel.
pub trait Prober: Send + Sync {
    /// Queue a probe request. Returns immediately. The work item runs
    /// on a worker thread; the response is delivered via the
    /// `Sender<Input>` captured at constructor time.
    fn submit(&self, req: ProbeRequest);

    /// Best-effort cancel of any *queued* probe for `profile`. In-flight
    /// probes are not interrupted; the engine drops their responses via
    /// stale-correlation discipline. After `cancel`, a fresh `submit` for
    /// the same profile runs normally — cancellation is per-correlation,
    /// not per-profile.
    fn cancel(&self, profile: ProfileId);
}

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
mod kqueue;

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub use kqueue::{KqueueWakeHandle, KqueueWatcher};

#[cfg(unix)]
mod prober;

#[cfg(unix)]
pub use prober::{DEFAULT_CONCURRENCY, WorkerProber};

#[cfg(feature = "testkit")]
pub mod testkit;
