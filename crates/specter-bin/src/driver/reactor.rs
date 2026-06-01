//! [`Reactor`] — owner of the mio [`Poll`] surface and every
//! kernel-driven [`Source`](mio::event::Source) the driver reacts to
//! that is not an IPC connection.
//!
//! Constructed once by `App::run`; owned by [`super::EngineDriver`]
//! for the rest of the daemon's lifetime. Holds the Poll, every static
//! fd source (watcher, config-watcher, signal pipe), the cross-thread
//! channel receivers, and the canonical `Arc<mio::Waker>` (held as
//! the `waker` field) the prober + actuator
//! wake-bearing senders clone via [`Reactor::wake_handle`]. The IPC
//! listener and per-conn map live on [`super::Hub`], which registers
//! against a [`Registry::try_clone()`] handle minted through
//! [`Reactor::registry_clone`]; both halves share the same underlying
//! selector.
//!
//! # Drop order
//!
//! Field order on [`Reactor`] is the drop order, and the explicit
//! [`Drop`] impl below performs `Registry::deregister` for every
//! static [`Source`](mio::event::Source) BEFORE the field-order drop
//! reaches their owning fields. The discipline:
//!
//! 1. **`Drop::drop` runs first** — explicit `deregister` for the
//!    watcher fd, the config-watcher fd (when present), and the
//!    signal pipe fd. Errors are best-effort (`NotFound` is benign
//!    on an already-closed fd). The deregisters use the Poll's own
//!    registry, which is still live at this point.
//! 2. **`signals`** drops — closes the [`SignalPipe`]'s read end and
//!    unregisters our handler chain entries from
//!    `signal_hook_registry`. The registry's static handler table
//!    itself is process-global and is not torn down by the
//!    unregister — other deliveries (e.g. test rigs holding their
//!    own [`SignalPipe`]) are unaffected.
//! 3. **`config_watcher`** drops (if present) — closes the kqueue /
//!    inotify fd that the config-side watcher held.
//! 4. **`watcher`** drops — closes the kqueue / inotify fd.
//! 5. **`events`** drops — a plain `Vec<event::Event>`; no resource
//!    implications.
//! 6. **`waker`** drops — releases the Reactor's `Arc<mio::Waker>`
//!    reference. Any external clones still held in
//!    [`WakingSink`](crate::driver::WakingSink)s
//!    contribute the remaining refcount; the underlying
//!    [`mio::Waker`] closes when the last clone drops. Placed BEFORE
//!    `poll` in field order so the refcount decrement happens while
//!    the selector is still alive — matching mio's "Source dies
//!    before Selector" lifecycle convention.
//! 7. **`poll`** drops — the underlying selector loses one
//!    Arc-reference. The [`super::Hub`] has already dropped
//!    (field order on [`super::EngineDriver`] puts `ipc` before
//!    `reactor`), so its [`Registry::try_clone()`] handle has
//!    released its Arc-reference too. The selector closes here as
//!    the last reference dies.
//! 8. **`prober_response_rx` / `effect_complete_rx`** drop together
//!    with the surrounding struct — pure crossbeam `Receiver` drops,
//!    which signal `Disconnected` to the paired senders (the prober
//!    pool's worker threads, the actuator's controller thread). Those
//!    threads exit their loops on the next observed `Disconnected`,
//!    so Reactor drop is the structural shutdown signal for both.
//!
//! The Reactor anchors the canonical `Arc<mio::Waker>` via its
//! `waker` field; [`Reactor::wake_handle`] emits clones for external
//! senders. Late `wake()` calls against a torn-down Poll are silent
//! no-ops (mio's documented contract).
//!
//! # Lifetime anchoring
//!
//! Every kernel-fd source registered against this Poll has its
//! Rust-side ownership structurally bounded below by this Reactor's
//! lifetime. The watcher, config-watcher, signal pipe, and the Waker
//! are direct fields; the Hub's listener and per-conn streams are
//! downstream via the [`EngineDriver`](super::EngineDriver) field
//! order. No external consumer needs to track an fd's lifetime to
//! reason about whether the kernel state is still observable — if
//! the Reactor exists, the kernel state is intact.
//!
//! The `waker` field is the structural cap on the "Waker lifetime ≥
//! Poll lifetime" invariant mio's rustdoc requires
//! (`mio-1.2.0/src/waker.rs:16-17`). Linux exposes the contract:
//! `mio::Waker` is an [`OwnedFd`](std::os::fd::OwnedFd) wrapping the
//! eventfd, and closing it auto-removes the fd from epoll's interest
//! AND ready lists — a Drop-fired `wake()` against the last Arc clone
//! would be stranded. macOS satisfies the contract incidentally: the
//! `NOTE_TRIGGER` event is queued on the kqueue's own state, so the
//! Waker's selector-clone close does not lose the pending trigger.
//! The Reactor's anchor closes the platform divergence — no
//! permutation of external [`WakingSink`](crate::driver::WakingSink) drop
//! orderings can take the
//! Arc refcount to 0 while `Poll` is alive, so a Drop-fired wake
//! always reaches the next `poll.poll()` call.
//!
//! # Visibility
//!
//! Every export is `pub(super)` or `pub(crate)`. The crate-visible
//! surface is [`Reactor::new`] + [`Reactor::wake_handle`] +
//! [`Reactor::registry_clone`] (all called by `App::run`); every
//! other method is `pub(super)` — only the surrounding `driver`
//! module reaches them. `tick.rs` drives [`Reactor::poll_and_drain`];
//! `forward.rs` drives [`Reactor::apply_watch_ops`]; the static
//! token constants are imported by [`super::ipc::hub`] for the
//! listener registration.

use crate::driver::WakeHandle;
use crossbeam::channel::Receiver;
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Registry, Token};
use specter_core::{FsEvent, Input, OverflowScope, ResourceId, WatchFailure, WatchOp};
use specter_sensor::{
    ConfigWatcher, DefaultConfigWatcher, DefaultWatcher, FsWatcher, WatcherEvent,
};
use std::io;
use std::os::fd::{AsFd, AsRawFd};
use std::time::Duration;

use crate::signals::SignalPipe;

/// Static token assignments for the always-present Sources. Per-conn
/// tokens (allocated on [`super::Hub`]) start at
/// [`super::ipc::hub::TOKEN_CONN_BASE`]; the gap between `0..=4`
/// (static) and `0x100..` (per-conn) makes the dispatch arm
/// `token if token.0 >= TOKEN_CONN_BASE` an unambiguous catch-all
/// without colliding with the static set.
pub(super) const TOKEN_WATCHER: Token = Token(0);
pub(super) const TOKEN_CONFIG_WATCHER: Token = Token(1);
pub(super) const TOKEN_SIGNAL: Token = Token(2);
pub(super) const TOKEN_WAKER: Token = Token(3);
/// Listener token. Defined here alongside the other static tokens so
/// the dispatch loop's arm set is grep-able to one file; consumed by
/// [`super::Hub::new`] for the listener registration.
pub(super) const TOKEN_LISTENER: Token = Token(4);

/// Owner of the mio reactor surface for kernel-driven, non-IPC sources.
/// See module rustdoc for the drop-order discipline the field order +
/// explicit [`Drop`] impl encode.
///
/// Generic over `W: FsWatcher` so tests can substitute the sensor
/// crate's `specter_sensor::testkit::MockFsWatcher` (whose
/// `UnixStream::pair()` readiness substrate lets reactor-integration
/// tests run against a real `mio::Poll` for free). Production uses
/// the platform [`DefaultWatcher`] — the type parameter's default
/// keeps app.rs free of `<DefaultWatcher>` boilerplate.
///
/// [`FsWatcher`] already requires `Send + AsFd`, so the bound is
/// minimal — the trait carries the AsFd surface every register call
/// needs and the Send required to construct on one thread and move
/// onto the driver thread.
pub(crate) struct Reactor<W: FsWatcher = DefaultWatcher> {
    /// The signal pipeline's reactor-visible surface. Owns the read
    /// end of the signal-hook pipe that the `sa_sigaction` handlers
    /// write to. The handlers stay installed for the life of this
    /// value; drop unregisters our chain entries from
    /// `signal_hook_registry`.
    signals: SignalPipe,
    /// Optional config watcher — absent under `--no-config-watch`
    /// (then `None`) or on watcher-init failure (then logged + `None`).
    /// When present, registered against the Poll registry at
    /// construction; [`Self::poll_and_drain`] dispatches via
    /// [`Self::drain_config_watcher`].
    config_watcher: Option<DefaultConfigWatcher>,
    /// The kqueue / inotify watcher (or a `MockFsWatcher` in tests).
    /// Always present (its init failure is a startup-fatal
    /// `ExitCode::from(1)` upstream of Reactor construction).
    watcher: W,
    /// Pre-allocated event buffer. Owned on Reactor so the per-tick
    /// `poll` call reuses the allocation. `Events::with_capacity(64)`
    /// covers the steady-state burst (≤5 static sources + ≤8 IPC
    /// conns × 2 directions); the kernel coalesces ready edges so the
    /// worst case is bounded.
    events: Events,
    /// Anchored [`WakeHandle`] — the Reactor IS the canonical owner;
    /// external wake-bearing sinks receive clones via
    /// [`Self::wake_handle`]. Carries one live `Arc<mio::Waker>`
    /// reference for the Reactor's lifetime, so no permutation of
    /// external [`super::WakingSink`] drops can take the refcount to
    /// 0 while [`Poll`] is alive. See the module rustdoc's "Lifetime
    /// anchoring" section for the cross-platform rationale.
    ///
    /// Field order places `waker` BEFORE `poll`: on Reactor drop the
    /// Arc-refcount decrement happens while the selector is still
    /// alive, matching mio's "Source dies before Selector" lifecycle
    /// convention.
    waker: WakeHandle,
    /// The mio reactor. Drops last (after the explicit [`Drop`] runs
    /// its deregisters) so registered Sources can finalize their fd
    /// close ahead of the underlying selector's invalidation.
    poll: Poll,
    /// Receiver for the prober pool's wake'd channel. Drained on the
    /// `TOKEN_WAKER` arm.
    prober_response_rx: Receiver<Input>,
    /// Receiver for the actuator's wake'd channel. Drained on the
    /// `TOKEN_WAKER` arm.
    effect_complete_rx: Receiver<Input>,
}

/// Partitioned drain output of one [`Reactor::poll_and_drain`] call.
///
/// The mio reactor's `iter()` yields events in unspecified Token
/// order. `poll_and_drain` dispatches each event to the appropriate
/// `drain_*` helper, which appends to the matching `DrainedTick`
/// field. The caller ([`super::EngineDriver::tick`]) then consumes
/// each field in the canonical order — listener accept → sensor inputs
/// → signals → effects → IPC — to preserve the per-tick drain
/// discipline the engine's lossy-hint contract depends on.
///
/// Consumption discipline: every field is drained with
/// `std::mem::take` (or by-value `drain`) so a second read returns
/// an empty Vec, making "drain each source at most once per tick"
/// structurally enforced rather than caller-disciplined. The
/// `listener_ready` and `config_event_pulse` bools are non-draining
/// (the consumer reads them as gates rather than walking them).
#[derive(Default)]
pub(super) struct DrainedTick {
    /// Per-resource fs events drained from the watcher fd. The
    /// engine's tick lifts each into `Input::FsEvent { resource, event }`.
    pub(super) fs_events: Vec<(ResourceId, FsEvent)>,
    /// Kernel-level overflow markers from the watcher (inotify's
    /// `IN_Q_OVERFLOW`; kqueue never emits). The engine's tick lifts
    /// each into `Input::SensorOverflow { scope }`.
    pub(super) sensor_overflows: Vec<OverflowScope>,
    /// Drained prober responses. The wake'd channel preserves the
    /// engine's per-response Input shape, so each entry is already
    /// an `Input::ProbeResponse(_)`.
    pub(super) probe_responses: Vec<Input>,
    /// Drained effect completions. Each entry is already an
    /// `Input::EffectComplete { .. }` envelope.
    pub(super) effect_completions: Vec<Input>,
    /// Signals queued on the signal-hook pipe since the last
    /// `poll_and_drain` call. The tick walks them in arrival order;
    /// dispatch on each is one `EngineDriver::dispatch_signal` call.
    pub(super) signals: Vec<i32>,
    /// `true` iff the config watcher drained at least one substantive
    /// event this tick. The tick re-arms its `config_settle_until`
    /// deadline on `true`; `false` carries no information.
    pub(super) config_event_pulse: bool,
    /// `true` iff `TOKEN_LISTENER` fired this tick. The tick delegates
    /// accept to [`super::Hub::drain_accept`] gated on this bool —
    /// making the accept step explicit in `tick.rs` rather than
    /// implicit inside the dispatch loop here.
    pub(super) listener_ready: bool,
    /// `true` iff `Reactor::drain_effect_completions` observed
    /// [`crossbeam::channel::TryRecvError::Disconnected`] this tick.
    ///
    /// The actuator thread's `Box<dyn EffectCompleteSender>` adapter
    /// wraps a [`super::WakingSink`]; when the actuator's `run` closure
    /// exits (clean or panic), the Box drops, the inner [`super::WakingSink`]
    /// drops, and its [`Drop`] closes the `Sender<Input>` BEFORE pulsing
    /// the wake edge. The driver's next `try_recv` on the paired
    /// `effect_complete_rx` returns Disconnected, surfacing here as
    /// `true`. The tick body then routes through
    /// [`super::EngineDriver::begin_shutdown`] — the actuator-gone
    /// signal that closes the [`super::EngineDriver::run`] loop end-to-end.
    ///
    /// `false` in steady state and after a no-op drain (Empty arm). The
    /// drain visits queued completions FIRST (Ok arm) and only sets
    /// this flag when the Disconnected variant lands, so a healthy
    /// pulse-with-completions tick never trips the flag.
    pub(super) actuator_gone: bool,
    /// Per-conn tokens whose readiness this tick included WRITABLE.
    /// The tick's drain pass walks these calling
    /// [`super::Hub::drain_writable`] on each.
    pub(super) ready_writes: Vec<Token>,
    /// Per-conn tokens whose readiness this tick included READABLE.
    /// The tick's drain pass walks these calling
    /// [`super::Hub::read_conn_into_lines`] on each.
    pub(super) ready_reads: Vec<Token>,
}

impl<W: FsWatcher> Reactor<W> {
    /// Construct the Reactor, allocating its [`mio::Poll`] and
    /// anchoring its [`WakeHandle`].
    ///
    /// `watcher` is any [`FsWatcher`] (production passes
    /// [`DefaultWatcher`]; tests pass `MockFsWatcher`). `config_watcher`
    /// is the platform default type — the bin does not currently mock
    /// it. `signals` is the bin's [`SignalPipe`] returned from
    /// [`crate::signals::register_signal_handlers`].
    /// `prober_response_rx` / `effect_complete_rx` are the consumer
    /// halves of the wake'd channels paired with the
    /// `WakingProberResponseSender` / `WakingEffectCompleteSender`
    /// adapters built at `App::run` time using
    /// [`Self::wake_handle`] clones.
    ///
    /// The constructor mints the [`WakeHandle`] (the sole call site
    /// of [`WakeHandle::new`] in the bin, which is itself the sole
    /// call site of [`mio::Waker::new`] — making mio's "one Waker
    /// per Poll" contract structural by typing) and anchors it as
    /// the `waker` field. Downstream consumers receive clones via
    /// [`Self::wake_handle`]; the Hub registers against a
    /// [`Registry::try_clone()`] obtained through
    /// [`Self::registry_clone`].
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `Poll::new`, [`WakeHandle::new`],
    /// or any of the three static `Source` registrations. All paths
    /// are programmer-error or kernel-pressure failures (`EMFILE` on
    /// the Waker fd) — the caller treats any error as startup-fatal.
    pub(crate) fn new(
        watcher: W,
        mut config_watcher: Option<DefaultConfigWatcher>,
        signals: SignalPipe,
        prober_response_rx: Receiver<Input>,
        effect_complete_rx: Receiver<Input>,
    ) -> io::Result<Self> {
        let poll = Poll::new()?;
        let waker = WakeHandle::new(poll.registry(), TOKEN_WAKER)?;

        // Watcher fd registration goes through SourceFd: the watcher
        // owns the kqueue / inotify fd directly via `AsFd`, and mio
        // captures the raw fd number internally — the `BorrowedFd`
        // lifetime is the call-expression scope, which is fine
        // because `as_raw_fd()` returns a `Copy` `RawFd`.
        let watcher_raw = watcher.as_fd().as_raw_fd();
        poll.registry().register(
            &mut SourceFd(&watcher_raw),
            TOKEN_WATCHER,
            Interest::READABLE,
        )?;

        if let Some(cw) = config_watcher.as_mut() {
            let cw_raw = cw.as_fd().as_raw_fd();
            poll.registry().register(
                &mut SourceFd(&cw_raw),
                TOKEN_CONFIG_WATCHER,
                Interest::READABLE,
            )?;
        }

        let signal_raw = signals.as_fd().as_raw_fd();
        poll.registry()
            .register(&mut SourceFd(&signal_raw), TOKEN_SIGNAL, Interest::READABLE)?;

        Ok(Self {
            signals,
            config_watcher,
            watcher,
            events: Events::with_capacity(64),
            waker,
            poll,
            prober_response_rx,
            effect_complete_rx,
        })
    }

    /// Mint a clone of the anchored [`WakeHandle`].
    ///
    /// Total fn — [`Arc::clone`](std::sync::Arc::clone) is infallible
    /// and structurally cheap (one atomic refcount bump). Every
    /// downstream wake-bearing sink calls this to receive a clone
    /// whose lifetime is bounded BELOW by the Reactor's: the Arc
    /// refcount has a floor of 1 (the Reactor's own `waker` field)
    /// for the entire span of the Reactor's existence, so no
    /// permutation of external clone-drop orderings can take the
    /// refcount to 0 while [`Poll`] is alive. See the module
    /// rustdoc's "Lifetime anchoring" section for the cross-platform
    /// rationale.
    #[must_use]
    pub(crate) fn wake_handle(&self) -> WakeHandle {
        self.waker.clone()
    }

    /// Mint a [`Registry::try_clone()`] handle for the
    /// [`super::Hub`].
    ///
    /// Fallible — [`Registry::try_clone`] is a syscall that can fail
    /// under kernel pressure (selector clone OOM). Callers translate
    /// the error to startup-fatal; this method exists so the call
    /// site sees the fallibility explicitly rather than hiding it
    /// behind a tuple-return contract the constructor would carry
    /// forever.
    ///
    /// The returned [`Registry`] handle shares the underlying
    /// selector with the Reactor's [`Poll`] — registrations against
    /// either land in the same kernel-side state.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from [`Registry::try_clone`].
    pub(crate) fn registry_clone(&self) -> io::Result<Registry> {
        self.poll.registry().try_clone()
    }

    /// Block on mio's Poll with `timeout`, drain every ready static
    /// Source non-blockingly, and return the partitioned drained state.
    ///
    /// `timeout` follows mio's convention: `None` blocks forever;
    /// `Some(Duration::ZERO)` polls once non-blockingly; any
    /// positive duration is the upper wait bound.
    ///
    /// Every static-source drain helper is internally drain-to-empty.
    /// The mio reactor's edge-triggered convention REQUIRES drain-to-
    /// empty on every ready fd: a partial drain leaves kernel-side
    /// state non-empty, the next arrival can't transition empty→
    /// non-empty, and the edge silently misses. Each `drain_*` helper
    /// loops internally until the underlying source reports
    /// `WouldBlock` / `EAGAIN`.
    ///
    /// Listener and per-conn readiness are surfaced as flags / token
    /// vectors on the returned [`DrainedTick`]; the caller delegates
    /// to [`super::Hub`] for accept / per-conn drain.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `poll.poll` or any drain that
    /// reports a non-`WouldBlock` syscall error. The caller treats
    /// any error as terminal for the Reactor (mio errors here are
    /// programmer-error / kernel-pressure).
    pub(super) fn poll_and_drain(&mut self, timeout: Option<Duration>) -> io::Result<DrainedTick> {
        let mut out = DrainedTick::default();
        match self.poll.poll(&mut self.events, timeout) {
            Ok(()) => {}
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {
                // EINTR — a signal arrived mid-poll. The signal is
                // queued on the signal-hook pipe and will surface on
                // the next iteration's TOKEN_SIGNAL event. Return an
                // empty DrainedTick; the caller's loop re-blocks.
                return Ok(out);
            }
            Err(e) => return Err(e),
        }
        // Snapshot every ready event's `(token, readable, writable)`
        // tuple into a local vector BEFORE invoking any `&mut self`
        // drain helper. Each per-Source drain (`drain_watcher` etc.)
        // needs `&mut self` to reach `self.watcher` / `self.signals`
        // / channel receivers, but `self.events.iter()` holds an
        // immutable borrow of `self.events` (a field of `self`).
        // Lifting the iteration into a snapshot releases that borrow
        // before the drains run. `Token` is `Copy` and the readiness
        // flags are bools, so this is bytewise cheap — the events
        // buffer holds ≤64 entries (`Events::with_capacity(64)` at
        // construction), so the allocation is bounded and small.
        let ready: Vec<(Token, bool, bool)> = self
            .events
            .iter()
            .map(|ev| (ev.token(), ev.is_readable(), ev.is_writable()))
            .collect();
        for (token, readable, writable) in ready {
            match token {
                TOKEN_WATCHER => self.drain_watcher(&mut out)?,
                TOKEN_CONFIG_WATCHER => self.drain_config_watcher(&mut out)?,
                TOKEN_SIGNAL => self.drain_signals(&mut out),
                TOKEN_WAKER => {
                    self.drain_prober_responses(&mut out);
                    self.drain_effect_completions(&mut out);
                }
                TOKEN_LISTENER => out.listener_ready = true,
                t if t.0 >= super::ipc::hub::TOKEN_CONN_BASE => {
                    // Per-conn readiness — collect tokens here, defer
                    // the actual byte shoveling to the Hub's
                    // post-drain pass (read_conn_into_lines /
                    // drain_writable).
                    if readable {
                        out.ready_reads.push(t);
                    }
                    if writable {
                        out.ready_writes.push(t);
                    }
                }
                _ => {
                    // Unknown token — defensive. Shouldn't fire under
                    // any production registration; a future
                    // registration that forgot to add a dispatch arm
                    // would surface here as a silent no-op. The
                    // structural fix is "add the arm"; logging here
                    // would noise-spam on a real bug.
                }
            }
        }
        Ok(out)
    }

    /// Apply a slice of [`WatchOp`]s to the owned watcher inline.
    /// Returns the rejected ops as `(resource, failure)` pairs so the
    /// caller can queue each as a deferred `Input::WatchOpRejected`
    /// for the next tick's engine step.
    ///
    /// The engine emits ops, the driver applies them against the owned
    /// watcher, the rejected ops queue for replay — all on one thread,
    /// without a `watch_ops` channel between any two of them.
    ///
    /// # Why a deferred-queue return rather than inline engine step
    ///
    /// `Input::WatchOpRejected` is a sensor-class barrier input that
    /// drives the engine's claim-purge path; routing it through the
    /// deferred-inputs queue (per the same-tick replay discipline)
    /// preserves the tick's drain ordering — the next tick's
    /// `replay_deferred_inputs` call runs the rejection through
    /// `engine.step` BEFORE the mio poll re-blocks, so the engine's
    /// claim-purge fires this tick's `forward` cycle.
    pub(super) fn apply_watch_ops(&mut self, ops: &[WatchOp]) -> Vec<(ResourceId, WatchFailure)> {
        let mut rejected = Vec::new();
        for op in ops {
            match op {
                WatchOp::Watch {
                    resource,
                    path,
                    kind,
                    events,
                } => {
                    if let Err(failure) = self.watcher.watch(*resource, path, *kind, *events) {
                        rejected.push((*resource, failure));
                    }
                }
                WatchOp::Unwatch { resource } => {
                    self.watcher.unwatch(*resource);
                }
            }
        }
        rejected
    }

    /// Drain the watcher fd to EAGAIN. Pushes each [`WatcherEvent`]
    /// into the matching [`DrainedTick`] field.
    ///
    /// # Errors
    ///
    /// Maps [`WatchFailure`] to [`io::Error`] via the failure's
    /// `errno()` — every drain error is structurally a syscall
    /// failure on the watcher fd. The caller ([`Self::poll_and_drain`])
    /// propagates upward; the Reactor's caller treats any error as
    /// terminal.
    fn drain_watcher(&mut self, out: &mut DrainedTick) -> io::Result<()> {
        let mut buf: Vec<WatcherEvent> = Vec::with_capacity(64);
        self.watcher
            .drain_ready(&mut buf)
            .map_err(|f| io::Error::other(format!("watcher drain failed: errno={}", f.errno())))?;
        for ev in buf {
            match ev {
                WatcherEvent::Fs { resource, event } => {
                    out.fs_events.push((resource, event));
                }
                WatcherEvent::Overflow { scope } => {
                    out.sensor_overflows.push(scope);
                }
            }
        }
        Ok(())
    }

    /// Drain the config watcher fd to EAGAIN. Sets
    /// `out.config_event_pulse = true` if any substantive event was
    /// observed; a `false` from `drain_ready` is a spurious wake or
    /// a non-basename-matched parent event the watcher already
    /// filtered out.
    fn drain_config_watcher(&mut self, out: &mut DrainedTick) -> io::Result<()> {
        if let Some(cw) = self.config_watcher.as_mut()
            && cw.drain_ready()?
        {
            out.config_event_pulse = true;
        }
        Ok(())
    }

    /// Drain queued signals from the signal-hook pipe. Each signal is
    /// pushed into `out.signals` in arrival order; the tick walks
    /// them and dispatches via `EngineDriver::dispatch_signal`.
    fn drain_signals(&mut self, out: &mut DrainedTick) {
        for sig in self.signals.pending() {
            out.signals.push(sig);
        }
    }

    /// Drain the prober response channel. Loops `try_recv` until
    /// `Empty`; `Disconnected` is ignored here (the channel's senders
    /// are the prober pool's `WakingProberResponseSender` clones,
    /// which disconnect only on pool shutdown — observed elsewhere
    /// via the Reactor's drop).
    ///
    /// Takes `&self` because [`crossbeam::channel::Receiver::try_recv`]
    /// only needs a shared borrow — the receiver's internal state is
    /// thread-safe. The `&mut DrainedTick` is the only mutable surface.
    fn drain_prober_responses(&self, out: &mut DrainedTick) {
        while let Ok(input) = self.prober_response_rx.try_recv() {
            out.probe_responses.push(input);
        }
    }

    /// Drain the effect completion channel.
    ///
    /// Mirror-shape of [`Self::drain_prober_responses`] on the Ok /
    /// Empty arms — both loop `try_recv` until the channel reports no
    /// more queued items. Diverges on the **Disconnected** arm: a
    /// disconnect on this channel is the actuator-gone signal (see
    /// [`DrainedTick::actuator_gone`] for the load-bearing rationale),
    /// so the drain sets the flag and breaks.
    ///
    /// Queued completions drain BEFORE the flag is set — the loop
    /// continues until `try_recv` reports either `Empty` (steady-state
    /// pause) or `Disconnected` (actuator's
    /// `Box<dyn EffectCompleteSender>` adapter has dropped). A
    /// post-disconnect tick that observes any leftover Ok arms first
    /// is the correct behavior: the actuator's late completions are
    /// still valid engine input, and the tick body routes the
    /// shutdown decision through [`super::EngineDriver::begin_shutdown`]
    /// after the standard drain pass anyway.
    ///
    /// Takes `&self` because [`crossbeam::channel::Receiver::try_recv`]
    /// only needs a shared borrow — the receiver's internal state is
    /// thread-safe. The `&mut DrainedTick` is the only mutable surface.
    fn drain_effect_completions(&self, out: &mut DrainedTick) {
        use crossbeam::channel::TryRecvError;
        loop {
            match self.effect_complete_rx.try_recv() {
                Ok(input) => out.effect_completions.push(input),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    out.actuator_gone = true;
                    break;
                }
            }
        }
    }

    /// Test-only mutable borrow of the owned watcher. Lets the driver's
    /// `mod tests` reach into a [`specter_sensor::testkit::MockFsWatcher`]
    /// to inject events between tick passes — the production seam
    /// (kernel-side fd → mio readiness → `drain_watcher`) is the same;
    /// only the *source* of the readiness edge differs.
    ///
    /// Gated behind `#[cfg(test)]` so production builds carry no extra
    /// surface; production never reaches into the watcher outside
    /// Reactor. `pub(super)` keeps the accessor scoped to the
    /// `driver` module.
    #[cfg(test)]
    pub(super) const fn watcher_mut(&mut self) -> &mut W {
        &mut self.watcher
    }
}

impl<W: FsWatcher> Drop for Reactor<W> {
    /// Explicit `deregister` for every static Source ahead of the
    /// field-order drop that closes their fds. mio's contract calls
    /// for "deregister before drop"; relying on the source's own
    /// `Drop` to release the registration is a contract violation,
    /// even though in practice the field-order drop reaches each fd
    /// before the selector closes.
    ///
    /// Errors are best-effort. `NotFound` is benign on an
    /// already-closed fd (a future cleanup path may close the fd
    /// before this Drop runs). A non-`NotFound` error here is a
    /// programmer-error worth knowing about but not worth panicking
    /// — Drop must not unwind. The log channel may already be gone
    /// at this teardown phase, which is fine: the next process boot
    /// re-creates every Source against a fresh selector regardless.
    fn drop(&mut self) {
        let registry = self.poll.registry();
        let watcher_raw = self.watcher.as_fd().as_raw_fd();
        let _ = registry.deregister(&mut SourceFd(&watcher_raw));
        if let Some(cw) = self.config_watcher.as_mut() {
            let cw_raw = cw.as_fd().as_raw_fd();
            let _ = registry.deregister(&mut SourceFd(&cw_raw));
        }
        let signal_raw = self.signals.as_fd().as_raw_fd();
        let _ = registry.deregister(&mut SourceFd(&signal_raw));
        // Field-order drop (signals → config_watcher → watcher →
        // events → waker → poll → receivers) runs after this method
        // returns. The `waker` field's drop releases the Reactor's
        // [`Arc<mio::Waker>`] reference while the Poll is still alive
        // — matching mio's "Source dies before Selector" convention
        // (see the module rustdoc's "Lifetime anchoring" section for
        // the load-bearing cross-platform rationale).
    }
}

impl<W: FsWatcher> std::fmt::Debug for Reactor<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Reactor")
            .field("config_watcher", &self.config_watcher.is_some())
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::register_signal_handlers;
    use crossbeam::channel::unbounded;
    use specter_sensor::testkit::MockFsWatcher;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    /// `poll_and_drain(Some(ZERO))` returns immediately with no events
    /// queued. Pins the non-blocking poll path. No wake fires and no
    /// Hub is constructed, so neither [`Reactor::wake_handle`] nor
    /// [`Reactor::registry_clone`] is called.
    #[test]
    fn poll_and_drain_zero_timeout_returns_empty() {
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let mut reactor =
            Reactor::new(watcher, None, signals, prober_rx, effect_rx).expect("reactor init");
        let drained = reactor
            .poll_and_drain(Some(Duration::ZERO))
            .expect("non-blocking poll succeeds");
        assert!(drained.fs_events.is_empty());
        assert!(drained.sensor_overflows.is_empty());
        assert!(drained.probe_responses.is_empty());
        assert!(drained.effect_completions.is_empty());
        assert!(drained.signals.is_empty());
        assert!(!drained.config_event_pulse);
        assert!(!drained.listener_ready);
        assert!(!drained.actuator_gone);
        assert!(drained.ready_reads.is_empty());
        assert!(drained.ready_writes.is_empty());
    }

    /// Dropping the effect-completion channel's last sender clone
    /// surfaces as `DrainedTick.actuator_gone == true` on the next
    /// `poll_and_drain`. Models the production actuator-thread closure
    /// exit: the thread's `Box<dyn EffectCompleteSender>` drops, the
    /// inner `WakingSink::Drop` closes its `Sender<Input>` clone, and
    /// the driver-side `try_recv` returns `Disconnected`.
    #[test]
    fn drain_effect_completions_sets_actuator_gone_on_disconnect() {
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (effect_tx, effect_rx) = unbounded::<Input>();

        let mut reactor =
            Reactor::new(watcher, None, signals, prober_rx, effect_rx).expect("reactor init");
        let waker = reactor.wake_handle();

        // Drop the only effect-completion sender. The Reactor-side
        // receiver has no other senders connected — the next try_recv
        // returns Disconnected.
        drop(effect_tx);
        waker.wake().expect("wake");

        let drained = reactor
            .poll_and_drain(Some(Duration::from_millis(500)))
            .expect("poll succeeds after wake");
        assert!(
            drained.actuator_gone,
            "Disconnected on effect_complete_rx surfaces as actuator_gone",
        );
        assert!(
            drained.effect_completions.is_empty(),
            "no in-flight completions queued before the disconnect",
        );
    }

    /// Queued completions drain via the Ok arm BEFORE the
    /// Disconnected arm sets `actuator_gone`. A clean-but-terminal
    /// shape: the actuator emits a final completion then exits,
    /// surfacing both the message AND the disconnect on the same
    /// drain pass.
    #[test]
    fn drain_effect_completions_drains_queued_before_signalling_disconnect() {
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (effect_tx, effect_rx) = unbounded::<Input>();

        let mut reactor =
            Reactor::new(watcher, None, signals, prober_rx, effect_rx).expect("reactor init");
        let waker = reactor.wake_handle();

        effect_tx
            .send(Input::TimerExpired {
                profile: specter_core::ProfileId::default(),
                kind: specter_core::TimerKind::Settle,
                id: specter_core::TimerId::default(),
            })
            .expect("queue completion");
        drop(effect_tx);
        waker.wake().expect("wake");

        let drained = reactor
            .poll_and_drain(Some(Duration::from_millis(500)))
            .expect("poll succeeds after wake");
        assert_eq!(
            drained.effect_completions.len(),
            1,
            "queued completion drained before Disconnected lands",
        );
        assert!(
            drained.actuator_gone,
            "post-drain try_recv observes Disconnected on the same tick",
        );
    }

    /// `waker.wake()` on an external clone causes the next
    /// `poll_and_drain` call to return immediately even with a generous
    /// timeout. Pins the wake fire-edge → mio dispatch path.
    #[test]
    fn waker_wake_unblocks_poll_and_drain() {
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let mut reactor =
            Reactor::new(watcher, None, signals, prober_rx, effect_rx).expect("reactor init");
        let waker = reactor.wake_handle();

        // Push an Input into the prober channel — without a wake,
        // the poll wouldn't see it (it polls fd-readiness, not
        // channel-non-emptiness). The wake fires the TOKEN_WAKER
        // edge, the drain helper try_recv's the message.
        prober_tx
            .send(Input::TimerExpired {
                profile: specter_core::ProfileId::default(),
                kind: specter_core::TimerKind::Settle,
                id: specter_core::TimerId::default(),
            })
            .expect("send into wake'd channel");
        waker.wake().expect("wake");

        let start = Instant::now();
        let drained = reactor
            .poll_and_drain(Some(Duration::from_secs(5)))
            .expect("poll succeeds after wake");
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_millis(500),
            "wake should unblock immediately, took {elapsed:?}",
        );
        assert_eq!(drained.probe_responses.len(), 1);
    }

    /// `apply_watch_ops` against the mock watcher emits no rejection
    /// for an accepting watcher. The MockFsWatcher accepts every
    /// `watch` call by default; pins the no-rejection path.
    #[test]
    fn apply_watch_ops_no_rejection_on_accepting_watcher() {
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let mut reactor =
            Reactor::new(watcher, None, signals, prober_rx, effect_rx).expect("reactor init");

        let mut slotmap: slotmap::SlotMap<ResourceId, ()> = slotmap::SlotMap::with_key();
        let r = slotmap.insert(());
        let path: Arc<std::path::Path> = Arc::from(PathBuf::from("/tmp").as_path());
        let rejected = reactor.apply_watch_ops(&[WatchOp::Watch {
            resource: r,
            path,
            kind: specter_core::ResourceKind::Unknown,
            events: specter_core::ClassSet::EMPTY,
        }]);
        assert!(rejected.is_empty(), "MockFsWatcher accepts by default");
    }

    /// `apply_watch_ops` surfaces a rejection when the watcher
    /// returns `Err`. The MockFsWatcher's `fail_next_watch` arms a
    /// one-shot failure for the next call.
    #[test]
    fn apply_watch_ops_surfaces_rejection_on_watcher_error() {
        let mut watcher = MockFsWatcher::new();
        // EMFILE = 24 on every supported target; hardcoded to avoid a
        // libc dev-dep in the bin's test surface.
        let failure = WatchFailure::Pressure { errno: 24 };
        watcher.fail_next_watch(failure);
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let mut reactor =
            Reactor::new(watcher, None, signals, prober_rx, effect_rx).expect("reactor init");

        let mut slotmap: slotmap::SlotMap<ResourceId, ()> = slotmap::SlotMap::with_key();
        let r = slotmap.insert(());
        let path: Arc<std::path::Path> = Arc::from(PathBuf::from("/tmp").as_path());
        let rejected = reactor.apply_watch_ops(&[WatchOp::Watch {
            resource: r,
            path,
            kind: specter_core::ResourceKind::Unknown,
            events: specter_core::ClassSet::EMPTY,
        }]);
        assert_eq!(rejected.len(), 1);
        assert_eq!(rejected[0].0, r);
        assert_eq!(rejected[0].1, failure);
    }
}
