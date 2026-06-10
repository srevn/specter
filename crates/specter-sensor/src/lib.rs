//! `specter-sensor` — dual-backend Watcher (kqueue / inotify) + Prober pool. The traits are
//! platform-agnostic; the backend implementations are `#[cfg]`-gated modules (kqueue on BSD /
//! macOS, inotify on Linux).

// Sensor legitimately needs `unsafe` for kqueue / inotify / epoll / eventfd FFI; `warn` is the
// looser-than-workspace setting that lets individual sites opt in to silenced unsafe at the audit
// boundary.
//
// The two FFI-only modules (`kqueue/ffi.rs`, `inotify/ffi.rs`) carry a module-level
// `#![allow(unsafe_code)]`: every `unsafe` block in those files is a syscall wrapper, and pinning
// the audit boundary at the file edge scales to ~30 blocks without per-block ceremony. Adding a
// non-FFI helper to either file would inherit the silence — so don't.
//
// Per-call-site `#[allow(unsafe_code)]` is reserved for the rare isolated `unsafe` block that lives
// outside an FFI-only module (currently none in sensor; see `actuator/src/os.rs:325` for the
// pattern in another crate).
#![warn(unsafe_code)]

use specter_core::{ClassSet, FsEvent, ProbeRequest, ProfileId, ResourceId, ResourceKind};
use std::io;
use std::os::fd::AsFd;
use std::path::Path;

// Re-exported alongside the trait so the bin can name `WatcherEvent` and its variant payloads
// (`OverflowScope`, `WatchFailure`) via one crate path. `OverflowScope` lives in `core` because the
// engine consumes it as `Input::SensorOverflow.scope`, but the sensor → bin call site never touches
// `core` directly. The `pub use` doubles as the in-module import the trait + `WatcherEvent`
// definitions below need. `ProbeResponse` is the payload [`ProberResponseSender::send`] carries —
// re-exported so implementors don't have to reach across the `specter_core` crate boundary to name
// the type. [`SendError`] is the workspace-shared sender-error vocabulary — re-exported so
// `sensor::SendError` is a stable path for callers.
pub use specter_core::{OverflowScope, ProbeFailure, ProbeResponse, SendError, WatchFailure};

/// Sensor-side extension on [`WatchFailure`] that classifies an `io::Error` from a watch-install
/// syscall.
///
/// `WatchFailure` lives in `specter-core`, which is `libc`-banned per `deny.toml`, so the errno-name
/// match cannot live there. This trait keeps the constructor reachable as `WatchFailure::from_io(&e)`
/// while localising every `libc` reference to backends that actually link it.
pub trait WatchFailureExt: Sized {
    /// Map an `io::Error` (the kqueue / inotify watcher syscall surface) into the typed variant.
    /// Backends call this at the trait boundary — the kernel error vocabulary stops here.
    ///
    /// # Preconditions
    ///
    /// Classifies errors from **watch-install and watcher-drain syscalls only**
    /// (`inotify_add_watch` / `open(O_EVTONLY)` / `kevent` / `read(inotify_fd)`). `ENOSPC` is
    /// interpreted as inotify watch-limit (`max_user_watches`) exhaustion and maps to
    /// [`WatchFailure::Pressure`]; passing an `io::Error` from a data-path syscall (where `ENOSPC`
    /// means "disk full") would misclassify a real I/O fault as watch-pressure. Every backend call
    /// site honours this; the precondition is the contract for any future caller.
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

/// Sensor-side extension on [`ProbeFailure`] that classifies an `io::Error` from a probe-root
/// syscall.
///
/// Cross-crate dual of [`WatchFailureExt`] — `ProbeFailure` lives in `specter-core` (libc-banned),
/// so the errno-name match lives here. The constructor reads as `ProbeFailure::from_io(&e)` once
/// the trait is in scope, mirroring the watch-side ergonomics.
pub trait ProbeFailureExt: Sized {
    /// Map an `io::Error` from the walker's probe-root syscalls (`std::fs::symlink_metadata`) into
    /// the typed routing variant. Called once at the walker stamp sites; the engine never
    /// re-derives the classification from a raw `i32`.
    ///
    /// # Preconditions
    ///
    /// Classifies errors from **probe-root syscalls only** — `symlink_metadata(target_path)`
    /// against the probe's anchor / descent prefix / proxy. Mid-walk faults skip-and-continue in
    /// the walker and never reach this trait. `ENOSPC` here is the process-FD-pressure surface
    /// (root-`lstat` allocates a brief kernel-internal FD), not "disk full"; mirrors
    /// [`WatchFailureExt::from_io`]'s `ENOSPC` precondition.
    fn from_io(e: &io::Error) -> Self;
}

impl ProbeFailureExt for ProbeFailure {
    fn from_io(e: &io::Error) -> Self {
        match e.raw_os_error() {
            Some(n @ (libc::EMFILE | libc::ENFILE | libc::ENOSPC | libc::EAGAIN)) => {
                Self::Transient { errno: n }
            }
            other => Self::Anchor {
                errno: other.unwrap_or(libc::EIO),
            },
        }
    }
}

/// One observation produced by [`FsWatcher::drain_ready`].
///
/// Two variants:
///
/// - [`Fs`](Self::Fs) — a per-resource filesystem event. The dominant variant; every
///   `WatchOp::Watch` install can produce these.
/// - [`Overflow`](Self::Overflow) — a kernel-level "events were dropped" signal that has no
///   `ResourceId` attached. inotify emits this on `IN_Q_OVERFLOW` (the `IDR` overflow → queue-wide
///   → `Global` scope); kqueue never emits it under v1 because `EV_CLEAR` coalesces but never
///   silently drops at the kernel level.
///
/// The bin lifts each variant into the engine's input vocabulary: `Fs` → `Input::FsEvent`;
/// `Overflow` → `Input::SensorOverflow`. The engine's response to `Overflow` is to reseed every
/// in-scope Profile.
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

/// Single-owner filesystem watcher.
///
/// `Send + AsFd`. One thread at a time owns the watcher and drives every
/// [`watch`](FsWatcher::watch) / [`unwatch`](FsWatcher::unwatch) /
/// [`drain_ready`](FsWatcher::drain_ready) call. The `Send` bound is load-bearing for the bin's
/// boot-time ownership transfer: the watcher is constructed on the main thread and *moved* into the
/// driver's `DriverHub` for the lifetime of the process. Once inside that owner, every call is on
/// the same thread by construction — there is no internal synchronization. Blocking is the caller's
/// responsibility: a reactor (mio::Poll, libc::poll, etc.) registers [`AsFd::as_fd`] in
/// edge-triggered mode and invokes `drain_ready` only when the reactor reports the fd readable. The
/// trait itself is non-blocking — there is no wake mechanism, no deadline, no internal block;
/// cross-thread coordination lives in the reactor (e.g. `mio::Waker`), not on the watcher.
///
/// # `AsFd` contract
///
/// `as_fd()` is the load-bearing surface for reactor integration. The implementor's obligations:
///
/// - **Long-lived FD.** Return a [`std::os::fd::BorrowedFd`] backed by an owned kernel resource
///   (kqueue / inotify_init1 / equivalent) acquired in the watcher's constructor and held until
///   drop. Re-acquiring the resource per call would invalidate any external mio / epoll
///   registration the reactor took out against the returned RawFd.
/// - **Readability ⇒ events queued.** The kernel must mark the fd readable iff at least one record
///   is queued for delivery via [`drain_ready`](FsWatcher::drain_ready). A spurious wake (readable,
///   drain returns no events) is permitted but should be rare; a missed wake (events queued, fd
///   stays non-readable) silently strands the records.
/// - **Edge-triggered support.** Reactors register with edge semantics (`Interest::READABLE`); the
///   fd must transition from "drained" to "non-drained" via the kernel's internal queue state on
///   every fresh event arrival. [`drain_ready`](FsWatcher::drain_ready)'s drain-to-empty discipline
///   (see below) is the implementor's complement to this contract — partial drains leave the queue
///   non-empty, the next arrival can't fire an edge, and the record is silently stranded.
/// - **`BorrowedFd` lifetime.** The returned borrow is the call expression's scope; reactors
///   capture the underlying RawFd (`as_fd().as_raw_fd()` is `Copy`) before the borrow ends. Holding
///   the BorrowedFd across `&mut self` calls is unnecessary.
///
/// # Caller pattern (illustrative)
///
/// ```ignore
/// let mut events: Vec<WatcherEvent> = Vec::with_capacity(64);
/// poll.registry().register(
///     &mut SourceFd(&watcher.as_fd().as_raw_fd()),
///     WATCHER_TOKEN,
///     Interest::READABLE,
/// )?;
/// loop {
///     poll.poll(&mut mio_events, Some(timeout))?;
///     for ev in &mio_events {
///         if ev.token() == WATCHER_TOKEN {
///             events.clear();
///             watcher.drain_ready(&mut events)?;
///             for w in events.drain(..) {
///                 // dispatch into engine_inbound (FsEvent / SensorOverflow)
///             }
///         }
///     }
///     // Apply pending WatchOps between drains — same thread.
/// }
/// ```
///
/// Coalescing under `EV_CLEAR`: multiple writes between drains are reported as one event. The
/// engine's `Settling` state already debounces by rescheduling on every event, so callers must not
/// assume per-write delivery — only "at least one event when something changed."
pub trait FsWatcher: Send + AsFd {
    /// Install (or re-register) a watch. Returns a typed [`WatchFailure`] on rejection: backends
    /// classify their kernel errno set (e.g. via [`WatchFailureExt::from_io`]) at the trait boundary,
    /// and the engine demuxes on the variant rather than on raw errno values. The bin packages a
    /// non-`Ok` return as `Input::WatchOpRejected { resource, failure }` for the engine, which clamps
    /// `watch_demand` to zero and waits for the parent's next `StructureChanged` to retry.
    ///
    /// `kind` is the engine's authoritative classification of the slot (`File` / `Dir` / `Unknown`).
    /// The watcher's fresh-watch path uses it as a verification step against the inode the freshly
    /// opened fd resolved to: a kind disagreement maps to [`WatchFailure::Resource`] (`ENOTDIR` /
    /// `EISDIR`) so the engine routes through the same path-fatal recovery channel. `Unknown` is a
    /// wildcard — the watcher accepts whatever inode resolves and caches the observed kind for
    /// downstream normalization. Re-watch paths reuse the cached kind and ignore this argument.
    ///
    /// `events` is the per-Resource event-class union. The watcher diffs it against the cached
    /// per-FD mask and re-registers iff different; `ClassSet::EMPTY` degrades to
    /// identity-floor-only delivery.
    fn watch(
        &mut self,
        r: ResourceId,
        path: &Path,
        kind: ResourceKind,
        events: ClassSet,
    ) -> Result<(), WatchFailure>;

    /// Remove a watch. Idempotent on stale ids. The sensor releases its kernel-level registration
    /// (kqueue: closes the watched fd; inotify: `inotify_rm_watch(wd)`) and clears every internal
    /// map keyed by `r`. Pending kernel-level events for the resource (kqueue: queued events on the
    /// closed fd; inotify: events queued on the soon-to-be- reaped wd before `IN_IGNORED`) are
    /// sensor-internal and never cross the trait boundary.
    fn unwatch(&mut self, r: ResourceId);

    /// Drain every record currently queued on the underlying kqueue / inotify fd into `out`,
    /// looping internally until the kernel reports `EAGAIN` (or `EAGAIN`-equivalent: a `kevent(2)`
    /// call with a zero `timespec` returning `n == 0`). Returns the count of [`WatcherEvent`]s
    /// pushed *this call*.
    ///
    /// Non-blocking by contract. The caller owns blocking via a reactor on [`AsFd::as_fd`]; this
    /// method translates "kernel says fd readable" into "here are the events."
    ///
    /// Two variants ride `out`:
    ///
    /// - [`Fs`](WatcherEvent::Fs) — per-resource filesystem event; the dominant variant.
    /// - [`Overflow`](WatcherEvent::Overflow) — kernel-level "events were dropped" signal carrying
    ///   an [`OverflowScope`]. inotify emits `Global` on `IN_Q_OVERFLOW`; FSEvents would emit
    ///   per-stream; kqueue never emits this under v1.
    ///
    /// # Why drain-to-empty internally
    ///
    /// Edge-triggered readiness REQUIRES drain-to-empty: a partial drain leaves residual kernel-side
    /// records, and the next event must transition the fd from "not readable" to "readable" to fire
    /// an edge. If the kernel queue is still non-empty when the next record arrives, the edge does
    /// not fire and the new record is silently stranded. Pinning the loop *inside* the watcher makes
    /// the invariant a structural property of the trait, not a caller discipline.
    ///
    /// # Lifecycle invariants the implementor must honour
    ///
    /// - **Intra-batch order independence.** Recovery decisions that span a single drained batch
    ///   (e.g., the post-loop reopen after an atomic-save coalesces terminal + parent records into
    ///   one drain) must be made *after* the per-record loop, against the batch's final state. The
    ///   kernel's record order within one drain is unspecified across both backends; per-arm
    ///   decisions that read intermediate state are order-sensitive bugs. The drain-to-empty loop
    ///   further widens this guarantee across multiple syscall batches within a single invocation —
    ///   the decision still belongs *after* the outer loop, not per inner batch.
    ///
    /// # Errors
    ///
    /// `EINTR` is retried inside the FFI helpers. Any other syscall error classifies through
    /// [`WatchFailureExt::from_io`] and terminates the drain mid-stream; `out` may be partially
    /// populated on `Err`. The caller treats any `Err` as terminal for the watcher.
    fn drain_ready(&mut self, out: &mut Vec<WatcherEvent>) -> Result<usize, WatchFailure>;
}

/// Single-threaded watcher for the daemon's own config file.
///
/// Distinct from [`FsWatcher`]: that trait is the engine's per- Resource surface, with `watch` /
/// `unwatch` mutators and a vector drain. The config watcher has exactly one watch target (the
/// running process's config path) and no engine vocabulary at the boundary — just "kernel said
/// something happened (substantively)" or "nothing yet."
///
/// `Send + AsFd`: same discipline as [`FsWatcher`]. One thread at a time owns the watcher and
/// drives [`drain_ready`](Self::drain_ready); the `Send` bound supports the bin's boot-time
/// hand-off into the driver thread the same way it does for `FsWatcher`. A reactor blocks on
/// [`AsFd::as_fd`] and invokes `drain_ready` only when the reactor reports the fd readable. The
/// trait itself is non-blocking and owns no wake mechanism.
///
/// The `AsFd` contract is the same as [`FsWatcher`]'s: a long-lived kernel-resource-backed
/// [`std::os::fd::BorrowedFd`], kernel marks readable iff at least one record is queued for
/// [`drain_ready`](Self::drain_ready), edge-triggered transitions on every fresh event, borrow
/// lifetime is the call expression's scope.
///
/// **Why no engine vocabulary?** The kqueue parent-dir filter cannot see basename, so every
/// dir-contents change in the config's parent becomes a pulse — the watcher cannot pre-classify
/// "this was about the config file" without a syscall. The driver's lstat-vs- `FileMeta` filter is
/// the natural place to suppress noise *and* the place that owns the prior-meta-known state, so the
/// watcher stays a minimal kernel-event pump.
///
/// # Caller pattern (illustrative)
///
/// ```ignore
/// poll.registry().register(
///     &mut SourceFd(&watcher.as_fd().as_raw_fd()),
///     CONFIG_TOKEN,
///     Interest::READABLE,
/// )?;
/// loop {
///     poll.poll(&mut events, None)?;
///     for ev in &events {
///         if ev.token() == CONFIG_TOKEN {
///             match watcher.drain_ready() {
///                 Ok(true)  => { let _ = config_event_tx.try_send(()); }
///                 Ok(false) => { /* spurious wake / non-substantive */ }
///                 Err(e)    => { tracing::error!(?e, "config-watcher exit"); return; }
///             }
///         }
///     }
/// }
/// ```
pub trait ConfigWatcher: Send + AsFd {
    /// Drain every record currently queued on the underlying inotify / kqueue fd, looping
    /// internally to the EAGAIN-equivalent. Returns `true` iff at least one substantive event was
    /// observed (file-side record, or a basename-matched parent record on inotify; any parent pulse
    /// on kqueue since the watcher cannot pre-classify by basename without an extra syscall).
    ///
    /// Same non-blocking + drain-to-empty discipline as [`FsWatcher::drain_ready`]. `Ok(true)` is a
    /// *raw* pulse — the watcher does not decide whether the change is substantive at any deeper
    /// level than the basename match. Drivers debounce and lstat-filter.
    ///
    /// # Lifecycle invariants the implementor must honour
    ///
    /// - **Intra-batch order independence.** File-loss recovery decisions (the post-loop reopen after
    ///   an atomic-save's coalesced parent + file-terminal records) must be made *after* every batch
    ///   drained in this invocation, against the invocation's final state. The kernel's record order
    ///   within one drain is unspecified across both backends, and the drain-to-empty loop may span
    ///   multiple syscall batches within one invocation; per-arm decisions reading intermediate state
    ///   would strand the watcher under one ordering and recover under the other.
    ///
    /// `EINTR` is retried internally. Other syscall errors propagate; the caller logs at `error!`
    /// and exits the watcher loop. SIGHUP-only operation continues to work.
    fn drain_ready(&mut self) -> io::Result<bool>;
}

/// Multi-threaded probe worker pool.
///
/// `submit` is fire-and-forget; the response lands on the bin's `engine_inbound` channel as
/// `Input::ProbeResponse(...)`. `cancel` is best-effort: queued probes whose `(owner, correlation)`
/// pair no longer matches the prober's per-owner expectation are skipped silently at worker-recv
/// time. In-flight probes complete; the engine discards their responses via stale-correlation
/// discipline.
///
/// # Threading
///
/// `Send + Sync` so the bin can hold an `Arc<dyn Prober>` (or `Arc<WorkerProber>`) and share it
/// across threads. v1 issues every `submit` and `cancel` from the bin's engine driver thread (the
/// sole `StepOutput` forwarder); signal handlers route through channels and the shared shutdown
/// flag into the engine step, never directly into the prober. The trait bounds reserve the option
/// of future cross-thread sharing without breaking ABI; the implementation guarantees only that
/// single-submitter discipline. The discipline is the correctness floor, not an optimisation —
/// `submit` and `cancel` for the same owner do not commute (reordering loses the cancel and runs
/// the stale correlation).
pub trait Prober: Send + Sync {
    /// Queue a probe request. Returns immediately. The work item runs on a worker thread; the
    /// resulting [`ProbeResponse`] ships to the [`ProberResponseSender`] wired in at constructor
    /// time.
    fn submit(&self, req: ProbeRequest);

    /// Best-effort cancel of any *queued* probe for `owner`. In-flight probes are not interrupted;
    /// the engine drops their responses via stale-correlation discipline. After `cancel`, a fresh
    /// `submit` for the same owner runs normally — cancellation is per-correlation, not per-owner.
    fn cancel(&self, owner: ProfileId);
}

/// Sink for probe responses produced by a [`Prober`] implementation.
///
/// The sensor crate owns *what* it delivers ([`ProbeResponse`]); the bin owns *where* it lands (the
/// engine's `Input::ProbeResponse(_)` channel). This trait is the seam: implementors translate the
/// sensor's response vocabulary into whatever transport the bin holds.
///
/// # Threading
///
/// `Send + Sync + 'static` so the pool can hold a single `Arc<dyn ProberResponseSender>` and share
/// it across every worker thread without re-cloning the underlying transport per worker. The inner
/// transport (a crossbeam `Sender`, an in-memory queue, a channel-multiplexer) is the implementor's
/// choice; the trait constrains only the wire vocabulary.
///
/// # Semantics
///
/// Fire-and-forget. A successful [`send`](Self::send) leaves no further obligation on the caller; an
/// [`Err`]([`SendError`]) means the consumer is gone (the engine driver dropped its receiver) and the
/// calling worker should exit its loop. The trait does not carry the rejected payload back on error —
/// workers do not retry, the bin owns shutdown-cause logging at the appropriate severity, and
/// dropping the response on the floor is the documented contract once the receiver disappears.
pub trait ProberResponseSender: Send + Sync + 'static {
    /// Deliver one probe response. Returns `Ok(())` on enqueue; `Err(SendError::Disconnected)` if
    /// the consumer is gone — at which point the caller (a worker) exits its loop.
    fn send(&self, response: ProbeResponse) -> Result<(), SendError>;
}

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
pub use prober::{WorkerProber, default_concurrency};

#[cfg(feature = "testkit")]
pub mod testkit;

/// Concrete platform watcher type — chosen at compile time so the bin holds a typed value (no
/// `Box<dyn>` in the watcher hot loop).
///
/// One alias per backend, each `cfg`-gated to its target. Adding a backend is one block (alias +
/// module + `FsWatcher` impl); the factory below already routes through this alias.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub type DefaultWatcher = KqueueWatcher;

#[cfg(target_os = "linux")]
pub type DefaultWatcher = InotifyWatcher;

/// Construct the platform's default watcher.
///
/// Returns the same concrete type as [`DefaultWatcher`] — no trait-object overhead. See module docs
/// on [`FsWatcher`] for the invariants the returned watcher must uphold (`Send + AsFd`,
/// single-threaded `drain_ready` consumer driven by a reactor blocking on [`AsFd::as_fd`]).
pub fn default_watcher() -> io::Result<DefaultWatcher> {
    DefaultWatcher::new()
}

/// Concrete platform config-watcher type — chosen at compile time so the bin holds a typed value
/// (no `Box<dyn>` in the auto-reload loop).
///
/// One alias per backend, each `cfg`-gated to its target. Adding a backend is one block (alias +
/// module + `ConfigWatcher` impl); the factory below already routes through this alias.
#[cfg(any(target_os = "macos", target_os = "freebsd"))]
pub type DefaultConfigWatcher = KqueueConfigWatcher;

#[cfg(target_os = "linux")]
pub type DefaultConfigWatcher = InotifyConfigWatcher;

/// Construct the platform's default config-watcher for the supplied path.
///
/// Returns the same concrete type as [`DefaultConfigWatcher`] — no trait-object overhead. The
/// watcher canonicalises `path` once at construction; symlink retarget at the leaf (or any
/// path-component move) is a documented restart-required limitation.
///
/// On error (`canonicalize` failure / parent dir unreadable / kqueue or inotify init failure), the
/// bin warn-logs and continues without auto-reload — SIGHUP still works.
pub fn default_config_watcher(path: &Path) -> io::Result<DefaultConfigWatcher> {
    DefaultConfigWatcher::new(path)
}
