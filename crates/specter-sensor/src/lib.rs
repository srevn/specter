//! `specter-sensor` — dual-backend Watcher (kqueue / inotify) + Prober
//! pool. The traits are platform-agnostic; the backend implementations
//! are `#[cfg]`-gated modules (kqueue on BSD / macOS, inotify on Linux).

// Sensor legitimately needs `unsafe` for kqueue / inotify / epoll /
// eventfd FFI; `warn` is the looser-than-workspace setting that lets
// individual sites opt in to silenced unsafe at the audit boundary.
//
// The two FFI-only modules (`kqueue/ffi.rs`, `inotify/ffi.rs`) carry a
// module-level `#![allow(unsafe_code)]`: every `unsafe` block in those
// files is a syscall wrapper, and pinning the audit boundary at the
// file edge scales to ~30 blocks without per-block ceremony. Adding a
// non-FFI helper to either file would inherit the silence — so don't.
//
// Per-call-site `#[allow(unsafe_code)]` is reserved for the rare
// isolated `unsafe` block that lives outside an FFI-only module
// (currently none in sensor; see `actuator/src/os.rs:325` for the
// pattern in another crate).
#![warn(unsafe_code)]

use specter_core::{ClassSet, FsEvent, ProbeOwner, ProbeRequest, ResourceId, ResourceKind};
use std::io;
use std::path::Path;
use std::time::Instant;

// Re-exported alongside the trait so the bin can name `WatcherEvent` and
// its variant payloads (`OverflowScope`, `WatchFailure`) via one crate
// path. `OverflowScope` lives in `core` because the engine consumes it
// as `Input::SensorOverflow.scope`, but the sensor → bin call site never
// touches `core` directly. The `pub use` doubles as the in-module import
// the trait + `WatcherEvent` definitions below need. `ProbeResponse` is
// the payload [`ProberResponseSender::send`] carries — re-exported so
// implementors don't have to reach across the `specter_core` crate
// boundary to name the type.
pub use specter_core::{OverflowScope, ProbeResponse, WatchFailure};

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
    ///
    /// # Preconditions
    ///
    /// Classifies errors from **watch-install and watcher-poll
    /// syscalls only** (`inotify_add_watch` / `open(O_EVTONLY)` /
    /// `kevent` / `epoll_wait`). `ENOSPC` is interpreted as inotify
    /// watch-limit (`max_user_watches`) exhaustion and maps to
    /// [`WatchFailure::Pressure`]; passing an `io::Error` from a
    /// data-path syscall (where `ENOSPC` means "disk full") would
    /// misclassify a real I/O fault as watch-pressure. Every backend
    /// call site honours this; the precondition is the contract for
    /// any future caller.
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
/// mutators ([`watch`](FsWatcher::watch) /
/// [`unwatch`](FsWatcher::unwatch)) run on the same thread between
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
    /// `EINTR` is retried internally. Syscall errors are classified
    /// via [`WatchFailureExt::from_io`] into a typed [`WatchFailure`]
    /// — symmetric with [`watch`](Self::watch). The bin treats a
    /// `poll_until` failure as terminal for the watcher thread (no
    /// recovery path).
    ///
    /// # Lifecycle invariants the implementor must honour
    ///
    /// - **Intra-batch order independence.** Recovery decisions that
    ///   span a single drained batch (e.g., the post-loop reopen
    ///   after an atomic-save coalesces terminal + parent records
    ///   into one drain) must be made *after* the per-record loop,
    ///   against the batch's final state. The kernel's record order
    ///   within one drain is unspecified across both backends; per-
    ///   arm decisions that read intermediate state are order-
    ///   sensitive bugs.
    ///
    /// - **Deadline honoured across `EINTR`.** A `Some(deadline)`
    ///   is *total wall-clock budget*, not a per-syscall budget.
    ///   The implementor's `EINTR` retry loop must recompute the
    ///   remaining budget on every iteration; a re-armed full
    ///   original budget would multiply the effective deadline by
    ///   the number of interruptions.
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
/// surface, with `watch` / `unwatch` mutators and a vector
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
/// # Caller loop pattern
///
/// ```ignore
/// loop {
///     if should_exit() { return; }
///     match watcher.wait(None) {
///         Ok(true)  => { let _ = config_event_tx.try_send(()); }
///         Ok(false) => { /* wake; loop and re-check exit condition */ }
///         Err(e)    => { tracing::error!(?e, "config-watcher exit"); return; }
///     }
/// }
/// ```
///
/// The exit condition is the caller's choice (an `AtomicBool` flag,
/// a separate shutdown channel, a poisoned mutex — whatever the
/// caller's shutdown primitive is); the trait surface does not
/// prescribe one.
pub trait ConfigWatcher: Send {
    /// Block until: (a) a kernel event fires on the config file or its
    /// parent directory (returns `Ok(true)`), (b) `deadline` elapses
    /// (returns `Ok(false)`), (c) a wake fires (returns `Ok(false)`),
    /// or (d) a syscall error occurs (returns `Err`).
    ///
    /// Production passes `None` — block forever; the watcher has no
    /// timers of its own. A `Some(deadline)` threads through to the
    /// backend's wait primitive (`kevent` / `epoll_wait`), which owns
    /// the per-iteration remaining-budget recompute across `EINTR`;
    /// tests use it as a watchdog without spawning a wake-thread.
    /// Settle and lstat-vs-meta filtering are driver-side concerns
    /// regardless.
    ///
    /// `Ok(true)` is a *raw* pulse — the watcher doesn't decide whether
    /// the change was substantive. Drivers debounce and lstat-filter.
    ///
    /// `EINTR` is retried internally. Other syscall errors propagate;
    /// the bin's wrapper logs at `error!` and exits the watcher thread.
    /// SIGHUP-only operation continues to work.
    ///
    /// # Lifecycle invariants the implementor must honour
    ///
    /// - **Intra-batch order independence.** File-loss recovery
    ///   decisions (the post-loop reopen after an atomic-save's
    ///   coalesced parent + file-terminal records) must be made
    ///   *after* the per-record loop, against the batch's final
    ///   state. The kernel's record order within one drain is
    ///   unspecified across both backends; per-arm decisions reading
    ///   intermediate state would strand the watcher under one
    ///   batch ordering and recover under the other.
    ///
    /// - **Wake produces `Ok(false)`.** A wake-only drain reports
    ///   "nothing substantive observed"; the bin's wrapper loops
    ///   and re-checks shutdown. Mixing the wake into the truthy
    ///   pulse would force an unnecessary engine-side settle cycle.
    ///
    /// - **Deadline honoured across `EINTR`.** A `Some(deadline)`
    ///   is *total wall-clock budget*, not a per-syscall budget.
    ///   The implementor's `EINTR` retry loop must recompute the
    ///   remaining budget on every iteration; a re-armed full
    ///   original budget would multiply the effective deadline by
    ///   the number of interruptions.
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
/// silently at worker-recv time. In-flight probes complete; the engine
/// discards their responses via stale-correlation discipline.
///
/// # Threading
///
/// `Send + Sync` so the bin can hold an `Arc<dyn Prober>` (or
/// `Arc<WorkerProber>`) and share it across threads. v1 issues every
/// `submit` and `cancel` from the bin's engine driver thread (the
/// sole `StepOutput` forwarder); signal handlers route through
/// channels and the shared shutdown flag into the engine step, never
/// directly into the prober. The trait bounds reserve the option of
/// future cross-thread sharing without breaking ABI; the
/// implementation guarantees only that single-submitter discipline.
/// The discipline is the correctness floor, not an optimisation —
/// `submit` and `cancel` for the same owner do not commute (reordering
/// loses the cancel and runs the stale correlation).
pub trait Prober: Send + Sync {
    /// Queue a probe request. Returns immediately. The work item runs
    /// on a worker thread; the resulting [`ProbeResponse`] ships to the
    /// [`ProberResponseSender`] wired in at constructor time.
    fn submit(&self, req: ProbeRequest);

    /// Best-effort cancel of any *queued* probe for `owner`. In-flight
    /// probes are not interrupted; the engine drops their responses via
    /// stale-correlation discipline. After `cancel`, a fresh `submit`
    /// for the same owner runs normally — cancellation is
    /// per-correlation, not per-owner.
    fn cancel(&self, owner: ProbeOwner);
}

/// Sink for probe responses produced by a [`Prober`] implementation.
///
/// The sensor crate owns *what* it delivers ([`ProbeResponse`]); the
/// bin owns *where* it lands (the engine's `Input::ProbeResponse(_)`
/// channel). This trait is the seam: implementors translate the
/// sensor's response vocabulary into whatever transport the bin holds.
///
/// # Threading
///
/// `Send + Sync + 'static` so the pool can hold a single
/// `Arc<dyn ProberResponseSender>` and share it across every worker
/// thread without re-cloning the underlying transport per worker. The
/// inner transport (a crossbeam `Sender`, an in-memory queue, a
/// channel-multiplexer) is the implementor's choice; the trait
/// constrains only the wire vocabulary.
///
/// # Semantics
///
/// Fire-and-forget. A successful [`send`](Self::send) leaves no further
/// obligation on the caller; an [`Err`](SendError) means the consumer
/// is gone (the engine driver dropped its receiver) and the calling
/// worker should exit its loop. The trait does not carry the rejected
/// payload back on error — workers do not retry, the bin owns
/// shutdown-cause logging at the appropriate severity, and dropping
/// the response on the floor is the documented contract once the
/// receiver disappears.
pub trait ProberResponseSender: Send + Sync + 'static {
    /// Deliver one probe response. Returns `Ok(())` on enqueue;
    /// `Err(SendError::Disconnected)` if the consumer is gone — at
    /// which point the caller (a worker) exits its loop.
    fn send(&self, response: ProbeResponse) -> Result<(), SendError>;
}

/// Sender-side error vocabulary for [`ProberResponseSender::send`].
///
/// One variant today; reserved as an `enum` rather than `()` so future
/// transports (bounded backpressure, batch submit) can extend the
/// vocabulary without churning every worker call site.
#[derive(Debug)]
pub enum SendError {
    /// The consumer dropped its receiver. No further `send` will
    /// succeed on this sender; the calling worker should exit.
    Disconnected,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => f.write_str("prober response consumer disconnected"),
        }
    }
}

impl std::error::Error for SendError {}

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
mod kqueue;

#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub use kqueue::{KqueueConfigWatcher, KqueueWatcher};

#[cfg(target_os = "linux")]
mod inotify;

#[cfg(target_os = "linux")]
pub use inotify::{InotifyConfigWatcher, InotifyWatcher};

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

/// Construct the platform's default watcher.
///
/// Returns the same concrete type as [`DefaultWatcher`] — no
/// trait-object overhead. See module docs on [`FsWatcher`] for the
/// invariants the returned watcher must uphold (`Send`, single-threaded
/// `poll_until` consumer, cross-thread mutation only via the bin's
/// channel + [`WakeHandle`] discipline).
pub fn default_watcher() -> io::Result<DefaultWatcher> {
    DefaultWatcher::new()
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
