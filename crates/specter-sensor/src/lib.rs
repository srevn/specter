//! `specter-sensor` — kqueue Watcher + Prober pool. The traits are
//! platform-agnostic; the kqueue implementation is BSD-only and lives in
//! a `#[cfg]`-gated module.

// Sensor legitimately needs `unsafe` for kqueue FFI; `warn` is looser
// than the workspace `deny`. Per-call-site `#[allow(unsafe_code)]` then
// silences the warning at the FFI boundary itself, keeping the audit
// surface narrow.
#![warn(unsafe_code)]

use specter_core::{ClassSet, FsEvent, ProbeOwner, ProbeRequest, ResourceId, ResourceKind};
use std::io;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

// Re-exported alongside the trait so the bin can name `WatcherEvent` and
// its variant payloads (`OverflowScope`, `WatchFailure`) via one crate
// path. `OverflowScope` lives in `core` because the engine consumes it
// as `Input::SensorOverflow.scope`, but the sensor → bin call site never
// touches `core` directly. The `pub use` doubles as the in-module import
// the trait + `WatcherEvent` definitions below need.
pub use specter_core::{OverflowScope, WatchFailure};

/// Cross-thread, runtime-tunable drain window for the watcher's
/// deferred-drain phase.
///
/// The bin owns one [`DrainWindow`]; clones are cheap (`Arc` bump). The
/// engine driver writes the value at startup and on hot reload; the
/// watcher thread reads it on every `poll_until` iteration. The
/// `AtomicU64` is the cross-thread surface — no lock, no channel.
///
/// **Semantics.**
/// - A value of `Duration::ZERO` (the initial state) disables deferred
///   drain entirely. The watcher returns from `poll_until` as soon as
///   the first `kevent_drain` / `epoll_wait` returns events.
/// - A non-zero value arms a second drain pass after the first returns
///   real events, **subject to the recency check** documented at each
///   backend's `poll_until`. The check ensures W_edit single touches in
///   quiet periods skip the second drain (zero latency cost) while
///   sustained bursts catch it from the second drain onwards.
///
/// **Ordering.** Both `set` and `get` use `Ordering::Relaxed`. Engine
/// correctness does not depend on which window value the watcher used
/// for any given drain (settle deadlines are engine-timer driven; the
/// window only shapes batch granularity), so the cheaper memory order
/// is correct.
#[derive(Debug, Clone)]
pub struct DrainWindow(Arc<AtomicU64>);

impl DrainWindow {
    /// Construct a fresh handle in the disabled state (`Duration::ZERO`).
    /// The bin computes the derived value (typically `min(settle) / 4`
    /// clamped to the audit's bounds) and calls [`Self::set`] before
    /// spawning the watcher thread.
    #[must_use]
    pub fn new() -> Self {
        Self(Arc::new(AtomicU64::new(0)))
    }

    /// Atomically update the window. Subsequent watcher reads observe
    /// the new value; at most one drain uses the prior value across a
    /// reload. Saturates at `u64::MAX` nanoseconds (`~584 years`) for
    /// pathologically large `Duration`s — well past any reasonable
    /// settle / window derivation.
    pub fn set(&self, d: Duration) {
        let nanos = u64::try_from(d.as_nanos()).unwrap_or(u64::MAX);
        self.0.store(nanos, Ordering::Relaxed);
    }

    /// Read the current window. Returns `Duration::ZERO` when unset
    /// (the initial / disabled state).
    #[must_use]
    pub fn get(&self) -> Duration {
        Duration::from_nanos(self.0.load(Ordering::Relaxed))
    }
}

impl Default for DrainWindow {
    fn default() -> Self {
        Self::new()
    }
}

/// Sensor-side extension on [`WatchFailure`] that classifies an
/// `io::Error` from a watch-install syscall.
///
/// `WatchFailure` lives in `specter-core`, which is `libc`-banned per
/// `deny.toml`, so the errno-name match cannot live there. This trait
/// keeps the constructor reachable as `WatchFailure::from_io(&e)` while
/// localising every `libc` reference to backends that actually link it.
pub trait WatchFailureExt: Sized {
    /// Map an `io::Error` (the kqueue / inotify watcher syscall surface)
    /// into the typed variant. Backends call this at the trait boundary —
    /// the kernel error vocabulary stops here.
    fn from_io(e: &io::Error) -> Self;
}

impl WatchFailureExt for WatchFailure {
    fn from_io(e: &io::Error) -> Self {
        let errno = e.raw_os_error().unwrap_or(0);
        match errno {
            libc::EMFILE | libc::ENFILE | libc::ENOSPC => Self::Pressure { errno },
            libc::ENOENT | libc::EACCES | libc::ELOOP | libc::ENOTDIR => Self::Resource { errno },
            _ => Self::Invariant { errno },
        }
    }
}

/// One observation produced by [`FsWatcher::poll_until`].
///
/// Two variants:
///
/// - [`Fs`](Self::Fs) — a per-resource filesystem event. The dominant
///   variant; every `WatchOp::Watch` install can produce these.
/// - [`Overflow`](Self::Overflow) — a kernel-level "events were dropped"
///   signal that has no `ResourceId` attached. inotify emits this on
///   `IN_Q_OVERFLOW` (the `IDR` overflow → queue-wide → `Global` scope);
///   kqueue never emits it under v1 because `EV_CLEAR` coalesces but
///   never silently drops at the kernel level.
///
/// The bin lifts each variant into the engine's input vocabulary:
/// `Fs` → `Input::FsEvent`; `Overflow` → `Input::SensorOverflow`.
/// The engine's response to `Overflow` is to reseed every in-scope
/// Profile.
#[derive(Debug, Clone)]
pub enum WatcherEvent {
    Fs {
        resource: ResourceId,
        event: FsEvent,
    },
    Overflow {
        scope: OverflowScope,
    },
}

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
/// let mut events: Vec<WatcherEvent> = Vec::with_capacity(64);
/// loop {
///     // 1. Apply pending WatchOps from the channel.
///     while let Ok(op) = ops_rx.try_recv() {
///         match op {
///             WatchOp::Watch { resource, path, kind, events } => {
///                 if let Err(failure) = watcher.watch(resource, &path, kind, events) {
///                     // Pressure / Resource / Invariant — engine demuxes via
///                     // `Input::WatchOpRejected`.
///                     engine_inbound.send(/* … failure … */);
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
///     for ev in events.drain(..) {
///         match ev {
///             WatcherEvent::Fs { resource, event } => {
///                 engine_inbound.send(Input::FsEvent { resource, event });
///             }
///             WatcherEvent::Overflow { scope } => {
///                 // Surfaced as `Input::SensorOverflow`.
///                 engine_inbound.send(/* … scope … */);
///             }
///         }
///     }
/// }
/// ```
///
/// Coalescing under `EV_CLEAR`: multiple writes between drains are
/// reported as one event. The engine's `Settling` state already debounces
/// by rescheduling on every event, so callers must not assume per-write
/// delivery — only "at least one event when something changed."
pub trait FsWatcher: Send {
    /// Install (or re-register) a watch. Returns a typed
    /// [`WatchFailure`] on rejection: backends classify their kernel
    /// errno set (e.g. via [`WatchFailureExt::from_io`]) at the trait
    /// boundary, and the engine demuxes on the variant rather than on
    /// raw errno values. The bin packages a non-`Ok` return as
    /// `Input::WatchOpRejected { resource, failure }` for the engine,
    /// which clamps `watch_demand` to zero and waits for the parent's
    /// next `StructureChanged` to retry.
    ///
    /// `kind` is the engine's authoritative classification of the slot
    /// (`File` / `Dir` / `Unknown`). The watcher's fresh-watch path
    /// uses it as a verification step against the inode the freshly
    /// opened fd resolved to: a kind disagreement maps to
    /// [`WatchFailure::Resource`] (`ENOTDIR` / `EISDIR`) so the engine
    /// routes through the same path-fatal recovery channel. `Unknown` is
    /// a wildcard — the watcher accepts whatever inode resolves and
    /// caches the observed kind for downstream normalization. Re-watch
    /// paths reuse the cached kind and ignore this argument.
    ///
    /// `events` is the per-Resource event-class union. The watcher diffs
    /// it against the cached per-FD mask and re-registers iff different;
    /// `ClassSet::EMPTY` degrades to identity-floor-only delivery.
    fn watch(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> Result<(), WatchFailure>;

    /// Remove a watch. Idempotent on stale ids. The sensor releases its
    /// kernel-level registration (kqueue: closes the watched fd; inotify:
    /// `inotify_rm_watch(wd)`) and clears every internal map keyed by
    /// `r`. Pending kernel-level events for the resource (kqueue: queued
    /// events on the closed fd; inotify: events queued on the soon-to-be-
    /// reaped wd before `IN_IGNORED`) are sensor-internal and never cross
    /// the trait boundary.
    fn unwatch(&mut self, r: ResourceId);

    /// Silence event delivery on a watched resource. Idempotent; no-op
    /// (with `tracing::warn!`) if `r` is not currently watched.
    ///
    /// Events arriving while suppressed are dropped at the watcher
    /// boundary; the post-unsuppress event stream contains only events
    /// that occur after [`unsuppress`](Self::unsuppress). Implementations
    /// realise this as a userspace filter consulted by `poll_until`
    /// before lifting an event onto the engine's input channel —
    /// kernel-level disable mechanisms are not used because their
    /// queue-and-replay semantics would deliver a coalesced phantom
    /// on re-enable, breaking the engine's "no events for `r` while
    /// suppressed" expectation. The kernel registration is unchanged
    /// across suppress/unsuppress; no re-stat happens on either edge.
    fn suppress(&mut self, r: ResourceId);

    /// Restore event delivery. Idempotent; no-op (with `tracing::warn!`)
    /// if `r` is not currently suppressed. See [`suppress`](Self::suppress)
    /// for the drop-at-boundary contract.
    fn unsuppress(&mut self, r: ResourceId);

    /// Block until the next event(s), the deadline, or a wake. Pushes
    /// normalized [`WatcherEvent`]s into `out` and returns the count
    /// pushed *this call*.
    ///
    /// Two variants ride the same channel:
    ///
    /// - [`Fs`](WatcherEvent::Fs) — per-resource filesystem event;
    ///   the dominant variant.
    /// - [`Overflow`](WatcherEvent::Overflow) — kernel-level "events
    ///   were dropped" signal carrying an [`OverflowScope`]. inotify
    ///   emits `Global` on `IN_Q_OVERFLOW`; FSEvents would emit
    ///   per-stream; kqueue never emits this under v1.
    ///
    /// `deadline = None` means "no deadline; block until event or wake."
    /// A returned count of zero is normal: either the deadline arrived
    /// or only a wake fired.
    ///
    /// `EINTR` is retried internally. Syscall errors map to a typed
    /// [`WatchFailure`] (kqueue: `EMFILE` from a full kernel queue →
    /// [`Pressure`](WatchFailure::Pressure); everything else →
    /// [`Invariant`](WatchFailure::Invariant)) — symmetric with
    /// [`watch`](Self::watch). The bin treats a `poll_until` failure
    /// as terminal for the watcher thread (no recovery path).
    fn poll_until(
        &mut self,
        deadline: Option<Instant>,
        out: &mut Vec<WatcherEvent>,
    ) -> Result<usize, WatchFailure>;

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

/// Single-threaded watcher for the daemon's own config file.
///
/// Distinct from [`FsWatcher`]: that trait is the engine's per-Resource
/// surface, with `watch` / `unwatch` / `suppress` mutators and a vector
/// drain. The config watcher has exactly one watch target (the running
/// process's config path) and no engine vocabulary at the boundary —
/// just "kernel said something happened" or "wake / deadline arrived."
///
/// One thread owns the watcher and calls [`wait`](Self::wait) in a loop;
/// the bin's wrapper thread translates `Ok(true)` into a pulse on the
/// `config_event` channel, leaving the lstat-vs-meta filter and settle
/// debounce to the engine driver. The wake handle ([`WakeHandle`]) is
/// the only cross-thread surface — same discipline as [`FsWatcher`].
///
/// **Why no engine vocabulary?** The kqueue parent-dir filter cannot
/// see basename, so every dir-contents change in the config's parent
/// becomes a pulse — the watcher cannot pre-classify "this was about
/// the config file" without a syscall. The driver's lstat-vs-`FileMeta`
/// filter is the natural place to suppress noise *and* the place that
/// owns the prior-meta-known state, so the watcher stays a minimal
/// kernel-event pump.
///
/// # Bin loop pattern
///
/// ```ignore
/// loop {
///     if shutdown_flag.load(SeqCst) { return; }
///     match watcher.wait(None) {
///         Ok(true)  => { let _ = config_event_tx.try_send(()); }
///         Ok(false) => { /* wake; loop and re-check shutdown */ }
///         Err(e)    => { tracing::error!(?e, "config-watcher exit"); return; }
///     }
/// }
/// ```
pub trait ConfigWatcher: Send {
    /// Block until: (a) a kernel event fires on the config file or its
    /// parent directory (returns `Ok(true)`), (b) `deadline` elapses
    /// (returns `Ok(false)`), (c) a wake fires (returns `Ok(false)`),
    /// or (d) a syscall error occurs (returns `Err`).
    ///
    /// Production passes `None` — block forever; the watcher has no
    /// timers of its own. The deadline is a kernel-level pass-through
    /// (`kevent` / `epoll_wait` already accept the timespec); tests use
    /// `Some(deadline)` as a watchdog without spawning a wake-thread.
    /// Settle and lstat-vs-meta filtering are driver-side concerns
    /// regardless.
    ///
    /// `Ok(true)` is a *raw* pulse — the watcher doesn't decide whether
    /// the change was substantive. Drivers debounce and lstat-filter.
    ///
    /// `EINTR` is retried internally. Other syscall errors propagate;
    /// the bin's wrapper logs at `error!` and exits the watcher thread.
    /// SIGHUP-only operation continues to work.
    fn wait(&mut self, deadline: Option<Instant>) -> io::Result<bool>;

    /// Capture a wake handle for cross-thread interruption of an
    /// in-flight `wait`. Cloneable via [`WakeHandle::clone_box`];
    /// concurrent wakes coalesce in the kernel. Idempotent.
    ///
    /// Reuses [`WakeHandle`] so the bin's shutdown path can wake either
    /// the engine watcher or the config watcher through one trait
    /// object — uniform discipline.
    fn wake_handle(&self) -> Box<dyn WakeHandle>;
}

/// Multi-threaded probe worker pool.
///
/// `submit` is fire-and-forget; the response lands on the bin's
/// `engine_inbound` channel as `Input::ProbeResponse(...)`. `cancel` is
/// best-effort: queued probes whose `(owner, correlation)` pair no
/// longer matches the prober's per-owner expectation are skipped
/// silently at worker-recv time. In-flight probes complete to
/// completion; the engine discards their responses via
/// stale-correlation discipline.
///
/// `Send + Sync` so the bin can hold an `Arc<dyn Prober>` (or
/// `Arc<WorkerProber>`) and share it across threads — the engine driver
/// thread submits, signal handlers may cancel.
pub trait Prober: Send + Sync {
    /// Queue a probe request. Returns immediately. The work item runs
    /// on a worker thread; the response is delivered via the
    /// `Sender<Input>` captured at constructor time.
    fn submit(&self, req: ProbeRequest);

    /// Best-effort cancel of any *queued* probe for `owner`. In-flight
    /// probes are not interrupted; the engine drops their responses via
    /// stale-correlation discipline. After `cancel`, a fresh `submit`
    /// for the same owner runs normally — cancellation is
    /// per-correlation, not per-owner.
    fn cancel(&self, owner: ProbeOwner);
}

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
mod kqueue;

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub use kqueue::{KqueueConfigWatcher, KqueueWakeHandle, KqueueWatcher};

#[cfg(target_os = "linux")]
mod inotify;

#[cfg(target_os = "linux")]
pub use inotify::{InotifyConfigWatcher, InotifyWakeHandle, InotifyWatcher};

#[cfg(unix)]
mod prober;

#[cfg(unix)]
pub use prober::{DEFAULT_CONCURRENCY, WorkerProber};

#[cfg(feature = "testkit")]
pub mod testkit;

/// Concrete platform watcher type — chosen at compile time so the
/// bin holds a typed value (no `Box<dyn>` in the watcher hot loop).
///
/// One alias per backend, each `cfg`-gated to its target. Adding a
/// backend is one block (alias + module + `FsWatcher` impl); the
/// factory below already routes through this alias.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub type DefaultWatcher = KqueueWatcher;

#[cfg(target_os = "linux")]
pub type DefaultWatcher = InotifyWatcher;

/// Construct the platform's default watcher with the supplied drain
/// window.
///
/// Returns the same concrete type as [`DefaultWatcher`] — no
/// trait-object overhead. See module docs on [`FsWatcher`] for the
/// invariants the returned watcher must uphold (`Send`, single-threaded
/// `poll_until` consumer, cross-thread mutation only via the bin's
/// channel + [`WakeHandle`] discipline).
///
/// `drain_window` is consumed by reference cheaply via `Arc` clone;
/// the bin keeps its own clone for hot-reload writes.
pub fn default_watcher(drain_window: DrainWindow) -> io::Result<DefaultWatcher> {
    DefaultWatcher::new(drain_window)
}

/// Concrete platform config-watcher type — chosen at compile time so
/// the bin holds a typed value (no `Box<dyn>` in the auto-reload loop).
///
/// One alias per backend, each `cfg`-gated to its target. Adding a
/// backend is one block (alias + module + `ConfigWatcher` impl); the
/// factory below already routes through this alias.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub type DefaultConfigWatcher = KqueueConfigWatcher;

#[cfg(target_os = "linux")]
pub type DefaultConfigWatcher = InotifyConfigWatcher;

/// Construct the platform's default config-watcher for the supplied
/// path.
///
/// Returns the same concrete type as [`DefaultConfigWatcher`] — no
/// trait-object overhead. The watcher canonicalises `path` once at
/// construction; symlink retarget at the leaf (or any path-component
/// move) is a documented restart-required limitation.
///
/// On error (`canonicalize` failure / parent dir unreadable / kqueue
/// or inotify init failure), the bin warn-logs and continues without
/// auto-reload — SIGHUP still works.
pub fn default_config_watcher(path: &Path) -> io::Result<DefaultConfigWatcher> {
    DefaultConfigWatcher::new(path)
}
