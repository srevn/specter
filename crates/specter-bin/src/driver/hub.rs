//! [`DriverHub`] — owner of the mio [`Poll`] surface and every
//! kernel-side [`Source`](mio::event::Source) the driver reacts to.
//!
//! Constructed once by `App::run`; owned by [`super::EngineDriver`]
//! for the rest of the daemon's lifetime. Replaces the four bridge
//! threads (watcher / config-watcher / signal / IPC server's
//! accept loop + per-conn workers) with one main-thread reactor.
//!
//! # Drop order
//!
//! Field order on [`DriverHub`] is the drop order. The discipline:
//!
//! 1. **`listener`** drops first — closes the bound socket fd; any
//!    in-flight accept on this Poll returns "fd was closed."
//! 2. **`conns`** drops — each [`super::conns::ConnState`]'s stream
//!    Drop fires; the corresponding mio-side registration was
//!    explicitly deregistered at [`DriverHub::terminate_conn`] time
//!    so a `ConnState` reaching Drop without going through it would
//!    strand a registration. The conn map is the only path that
//!    removes entries — the audit grep
//!    `conns.remove\|conns.drain` returns only `terminate_conn`.
//! 3. **`signals`** drops — closes the [`SignalPipe`]'s read end
//!    and unregisters our handler chain entries from
//!    `signal_hook_registry`. The registry's static handler table
//!    itself is process-global and is not torn down by the
//!    unregister — other deliveries (e.g. test rigs holding their
//!    own [`SignalPipe`]) are unaffected.
//! 4. **`config_watcher`** drops (if present) — closes the kqueue /
//!    inotify fd that the config-side watcher held.
//! 5. **`watcher`** drops — closes the kqueue / inotify fd.
//! 6. **`events`** drops — a plain Vec<event::Event>; no resource
//!    implications.
//! 7. **`poll`** drops last — the registry is invalidated. Any
//!    straggler Source Drop that tries to deregister silently fails
//!    (mio's contract) — which is fine because every Source above
//!    already closed its fd; deregistration was best-effort cleanup.
//!    The [`WakeHandle`] returned from [`DriverHub::new`] is the
//!    sole anchor — external senders (the [`WakingSink`]-bearing
//!    adapters used by the prober pool and actuator thread) hold
//!    their own clones; those clones outlive the Hub and drop when
//!    their owners exit. A late `wake()` against a torn-down Poll is
//!    a silent no-op (mio's documented contract).
//! 8. **`prober_response_rx` / `effect_complete_rx`** drop together
//!    with the surrounding struct — pure crossbeam Receiver Drop,
//!    which signals Disconnected to the paired senders (the prober
//!    pool's worker threads, the actuator's controller thread). Those
//!    threads exit their loops on the Disconnected they observe next,
//!    so Hub drop is the structural shutdown signal for both.
//!
//! # Visibility
//!
//! Every export is `pub(super)`. The only consumer is the surrounding
//! `driver` module — `tick.rs` drives [`DriverHub::next_inputs`],
//! `forward.rs` drives [`DriverHub::apply_watch_ops`], the IPC
//! handler reaches per-conn helpers.

use crate::driver::WakeHandle;
use crate::driver::conns::{ConnState, MAX_REQUEST_LINE_BYTES, PushOutcome};
use crate::ipc::framing::serialize_line;
use crate::ipc::protocol::{ERR_BUSY, ResponsePayload};
use crate::ipc::wire::WireDiagnostic;
use crate::signals::SignalPipe;
use crossbeam::channel::Receiver;
use mio::unix::SourceFd;
use mio::{Events, Interest, Poll, Token};
use specter_core::{
    Diagnostic, FsEvent, Input, OverflowScope, ResourceId, SubId, WatchFailure, WatchOp,
};
use specter_sensor::{
    ConfigWatcher, DefaultConfigWatcher, DefaultWatcher, FsWatcher, WatcherEvent,
};
use std::borrow::Cow;
use std::collections::BTreeMap;
use std::io;
use std::os::fd::{AsFd, AsRawFd};
use std::time::{Duration, SystemTime};

/// Static token assignments for the always-present Sources. Per-conn
/// tokens are allocated dynamically starting at [`TOKEN_CONN_BASE`];
/// the gap between `0..=4` (static) and `0x100..` (per-conn) makes the
/// dispatch arm `token if token.0 >= TOKEN_CONN_BASE` an unambiguous
/// catch-all without colliding with the static set.
pub(super) const TOKEN_WATCHER: Token = Token(0);
pub(super) const TOKEN_CONFIG_WATCHER: Token = Token(1);
pub(super) const TOKEN_SIGNAL: Token = Token(2);
pub(super) const TOKEN_WAKER: Token = Token(3);
pub(super) const TOKEN_LISTENER: Token = Token(4);

/// First per-conn Token value. Chosen well above the static set so a
/// new static Source can be added (Token(5..0xFF) are reserved) without
/// migrating per-conn allocation; the gap also makes a debug-mode
/// `assert!(token.0 >= TOKEN_CONN_BASE)` in per-conn arms a cheap
/// structural check against accidental static-token aliasing.
pub(super) const TOKEN_CONN_BASE: usize = 0x100;

/// Concurrent IPC client cap. Bound by the operator hand-control
/// envelope (one engineer typically holds ≤2 sessions: a `tail` and
/// a verb shell); the cap survives mostly as a DoS floor against a
/// misbehaving client opening connections in a loop.
pub(super) const MAX_IPC_CONNS: usize = 8;

/// Owner of the mio reactor surface. See module rustdoc for the
/// drop-order discipline the field order encodes.
///
/// Generic over `W: FsWatcher` so tests can substitute the sensor
/// crate's [`specter_sensor::testkit::MockFsWatcher`] (whose
/// `UnixStream::pair()` readiness substrate lets reactor-integration
/// tests run against a real `mio::Poll` for free). Production uses
/// the platform [`DefaultWatcher`] — the type parameter's default
/// keeps app.rs free of `<DefaultWatcher>` boilerplate.
///
/// [`FsWatcher`] already requires `Send + AsFd`, so the bound is
/// minimal — the trait carries the AsFd surface every register call
/// needs and the Send required to construct on one thread and move
/// onto the driver thread.
pub(crate) struct DriverHub<W: FsWatcher = DefaultWatcher> {
    /// The bound operator-IPC socket. Drops first to close the bind
    /// fd so any racing client connect sees the socket gone before
    /// any other Hub state is torn down.
    listener: mio::net::UnixListener,
    /// Per-conn state map. `BTreeMap` over `HashMap`: the conn count
    /// is small (≤[`MAX_IPC_CONNS`]) and `BTreeMap` carries no random
    /// state for the iteration order — the dispatch loop walks in
    /// Token order, which makes test assertions deterministic without
    /// extra sort calls.
    conns: BTreeMap<Token, ConnState>,
    /// The signal pipeline's reactor-visible surface. Owns the read
    /// end of the signal-hook pipe that the `sa_sigaction` handlers
    /// write to. The handlers stay installed for the life of this
    /// value: drop unregisters our chain entries from
    /// `signal_hook_registry`, see the module-level drop-order block.
    signals: SignalPipe,
    /// Optional config watcher — absent under `--no-config-watch`
    /// (then `None`) or on watcher-init failure (then logged + `None`).
    /// When present, registered against the Poll registry at
    /// construction; the `next_inputs` dispatch drains via
    /// `drain_config_watcher`.
    config_watcher: Option<DefaultConfigWatcher>,
    /// The kqueue / inotify watcher (or a `MockFsWatcher` in tests).
    /// Always present (its init failure is a startup-fatal
    /// `ExitCode::from(1)` upstream of Hub construction).
    watcher: W,
    /// Pre-allocated event buffer. Owned on Hub so the per-tick
    /// `poll` call reuses the allocation. `Events::with_capacity(64)`
    /// covers the steady-state burst (≤5 static sources + ≤8 IPC
    /// conns × 2 directions); the kernel coalesces ready edges so
    /// the worst case is bounded.
    events: Events,
    /// The mio reactor. Drops last so registered Sources can finalize
    /// their fd close ahead of the registry's invalidation.
    poll: Poll,
    /// Receiver for the prober pool's wake'd channel. Drained on the
    /// `TOKEN_WAKER` arm.
    prober_response_rx: Receiver<Input>,
    /// Receiver for the actuator's wake'd channel. Drained on the
    /// `TOKEN_WAKER` arm.
    effect_complete_rx: Receiver<Input>,
    /// Monotone counter for fresh per-conn Token allocation. Starts at
    /// [`TOKEN_CONN_BASE`]; wraps back to the base on overflow (a
    /// purely theoretical concern — `usize::MAX - 0x100` accepts is
    /// &gt;10^18 on 64-bit; an operator daemon hits the heat death of
    /// the universe first). The wrap exists so the type-level
    /// `usize::wrapping_add` doesn't introduce an undefined edge.
    next_conn_token: usize,
}

/// Partitioned drain output of one [`DriverHub::next_inputs`] call.
///
/// The mio reactor's `iter()` yields events in unspecified Token
/// order. `next_inputs` dispatches each event to the appropriate
/// `drain_*` helper, which appends to the matching `DrainedTick`
/// field. The caller (`tick.rs`) then consumes each field in the
/// canonical order — sensor inputs → signals → effects → IPC — to
/// preserve the per-tick drain discipline the engine's lossy-hint
/// contract depends on.
///
/// Consumption discipline: every field is drained with
/// `std::mem::take` (or by-value `drain`) so a second read returns
/// an empty Vec, making "drain each source at most once per tick"
/// structurally enforced rather than caller-disciplined.
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
    /// `next_inputs` call. The tick walks them in arrival order;
    /// dispatch on each is one `EngineDriver::dispatch_signal` call.
    pub(super) signals: Vec<i32>,
    /// `true` iff the config watcher drained at least one substantive
    /// event this tick. The tick re-arms its `config_settle_until`
    /// deadline on `true`; `false` carries no information.
    pub(super) config_event_pulse: bool,
    /// Per-conn tokens whose readiness this tick included WRITABLE.
    /// The tick's drain pass walks these calling
    /// [`DriverHub::drain_writable`] on each.
    pub(super) ready_writes: Vec<Token>,
    /// Per-conn tokens whose readiness this tick included READABLE.
    /// The tick's drain pass walks these calling
    /// [`DriverHub::read_conn_into_lines`] on each.
    pub(super) ready_reads: Vec<Token>,
}

/// Outcome of one [`DriverHub::enqueue_response`] call.
///
/// The Hub-side wrapper around
/// [`crate::driver::conns::ConnState::push_response`] threads the
/// per-conn capacity verdict back to the caller along with a
/// "conn-not-in-map" signal — the IPC handler's Subscribe arm needs
/// to know whether the ack actually landed before flipping the role.
/// Other handlers (`Reload`, `Disable`, `Enable`, projection paths)
/// `let _ = ...` the outcome benignly: a refused or gone conn is
/// already on the path to termination, and re-acking is pointless.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(super) enum EnqueueOutcome {
    /// Bytes pushed into the write_queue; will flush on the next
    /// WRITABLE drain.
    Accepted,
    /// Response did not fit. `close_after_flush` is armed; if the
    /// queue was empty at the time, the conn has been terminated
    /// already (this call internally invoked
    /// [`DriverHub::try_terminate_if_idle`]). If the queue had bytes,
    /// [`DriverHub::drain_writable`] will terminate when those drain.
    Refused,
    /// `token` is not in the conn map — the caller addressed a conn
    /// that closed between an earlier point in this tick and the
    /// enqueue (a read drain that observed EOF terminated it, or a
    /// write failure removed it). Discriminated from `Refused` so the
    /// Subscribe handler can avoid a no-op role flip on a gone conn.
    ConnGone,
}

/// Outcome of one [`DriverHub::read_conn_into_lines`] call.
///
/// The two variants distinguish the termination semantics the read
/// drain triggers:
///
/// - `Continue` — the read end is alive (or pending). The caller
///   processes any drained lines and then pairs with
///   [`DriverHub::try_terminate_if_idle`] in case an oversize-line
///   guard armed `close_after_flush` against an empty queue.
/// - `PeerGone` — peer EOF or a non-recoverable read transport error.
///   The caller terminates the conn unconditionally; any pending
///   write-queue bytes are wasted because the peer's read end has
///   closed.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(super) enum ReadOutcome {
    Continue,
    PeerGone,
}

impl<W: FsWatcher> DriverHub<W> {
    /// Construct the Hub from already-allocated kernel resources.
    ///
    /// `listener` is the bound `std::os::unix::net::UnixListener` from
    /// `sockpath::bind_socket_atomic`; we re-wrap it into mio's flavor
    /// after setting non-blocking. `watcher` is any
    /// [`FsWatcher`] (production passes [`DefaultWatcher`]; tests pass
    /// `MockFsWatcher`). `config_watcher` is the platform default
    /// type — the bin does not currently mock it. `signals` is the
    /// bin's [`SignalPipe`] returned from
    /// [`crate::signals::register_signal_handlers`].
    /// `prober_response_rx` / `effect_complete_rx` are the consumer
    /// halves of the wake'd channels paired with the
    /// `WakingProberResponseSender` / `WakingEffectCompleteSender`
    /// adapters (constructed at `App::run` with a clone of the
    /// [`WakeHandle`] returned from this constructor).
    ///
    /// Returns `(Self, WakeHandle)`: the [`WakeHandle`] is cloned for
    /// every external sender that needs to wake the driver. Sharing
    /// one Waker across senders honors mio's "one Waker per Poll"
    /// contract — and the [`WakeHandle`] newtype makes that
    /// invariant structural: [`WakeHandle::new`] is the sole call
    /// site of [`mio::Waker::new`] in the bin, so a second
    /// construction site would require a fresh `use mio::Waker`
    /// import.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `Poll::new`, [`WakeHandle::new`],
    /// `set_nonblocking`, or any of the four `Source` registrations.
    /// All five are programmer-error or kernel-pressure failures
    /// (`EMFILE` on Waker fd) — the caller treats any error as
    /// startup-fatal.
    pub(crate) fn new(
        listener: std::os::unix::net::UnixListener,
        watcher: W,
        mut config_watcher: Option<DefaultConfigWatcher>,
        signals: SignalPipe,
        prober_response_rx: Receiver<Input>,
        effect_complete_rx: Receiver<Input>,
    ) -> io::Result<(Self, WakeHandle)> {
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

        // Listener: must be non-blocking before mio observes it
        // (mio's `from_std` does not set the flag), or the next
        // `accept()` would block the reactor thread on an empty
        // listen queue.
        listener.set_nonblocking(true)?;
        let mut listener = mio::net::UnixListener::from_std(listener);
        poll.registry()
            .register(&mut listener, TOKEN_LISTENER, Interest::READABLE)?;

        Ok((
            Self {
                listener,
                conns: BTreeMap::new(),
                signals,
                config_watcher,
                watcher,
                events: Events::with_capacity(64),
                poll,
                prober_response_rx,
                effect_complete_rx,
                next_conn_token: TOKEN_CONN_BASE,
            },
            waker,
        ))
    }

    /// Block on mio's Poll with `timeout`, drain every ready Source
    /// non-blockingly, and return the partitioned drained state.
    ///
    /// `timeout` follows mio's convention: `None` blocks forever;
    /// `Some(Duration::ZERO)` polls once non-blockingly; any
    /// positive duration is the upper wait bound.
    ///
    /// Every drain helper is internally drain-to-empty. The mio
    /// reactor's edge-triggered convention REQUIRES drain-to-empty on
    /// every ready fd: a partial drain leaves kernel-side state
    /// non-empty, the next arrival can't transition empty→non-empty,
    /// and the edge silently misses. Each `drain_*` helper loops
    /// internally until the underlying source reports WouldBlock /
    /// `EAGAIN`.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `poll.poll` or any drain that
    /// reports a non-`WouldBlock` syscall error. The caller treats
    /// any error as terminal for the Hub (mio errors here are
    /// programmer-error / kernel-pressure).
    pub(super) fn next_inputs(&mut self, timeout: Option<Duration>) -> io::Result<DrainedTick> {
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
                TOKEN_LISTENER => self.drain_accept()?,
                t if t.0 >= TOKEN_CONN_BASE => {
                    // Per-conn readiness — collect tokens here, defer
                    // the actual byte shoveling to the consumer's
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
    /// into the matching `DrainedTick` field.
    ///
    /// # Errors
    ///
    /// Maps [`WatchFailure`] to [`io::Error`] via the failure's
    /// `errno()` — every drain error is structurally a syscall
    /// failure on the watcher fd. The caller (`next_inputs`)
    /// propagates upward; the Hub's caller treats any error as
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
    /// Empty; Disconnected is ignored here (the channel's senders are
    /// the prober pool's `WakingProberResponseSender` clones, which
    /// disconnect only on pool shutdown — observed elsewhere via the
    /// Hub's drop).
    ///
    /// Takes `&self` because [`crossbeam::channel::Receiver::try_recv`]
    /// only needs a shared borrow — the receiver's internal state is
    /// thread-safe. The `&mut DrainedTick` is the only mutable surface.
    fn drain_prober_responses(&self, out: &mut DrainedTick) {
        while let Ok(input) = self.prober_response_rx.try_recv() {
            out.probe_responses.push(input);
        }
    }

    /// Drain the effect completion channel. Same semantics as
    /// [`Self::drain_prober_responses`].
    fn drain_effect_completions(&self, out: &mut DrainedTick) {
        while let Ok(input) = self.effect_complete_rx.try_recv() {
            out.effect_completions.push(input);
        }
    }

    /// Accept every pending connection up to [`MAX_IPC_CONNS`].
    ///
    /// Edge-triggered: loops until `accept()` returns `WouldBlock`.
    /// New conns are inserted into `self.conns` with a fresh per-conn
    /// Token registered for `READABLE` against the Poll registry.
    ///
    /// # Cap behavior
    ///
    /// On reaching [`MAX_IPC_CONNS`], extra accepts get a structured
    /// `ERR_BUSY` JSON response written best-effort + the stream
    /// dropped. The cap rejects rather than queues — operator IPC is
    /// not throughput-sensitive, and a queue would let a misbehaving
    /// client wedge the daemon's resource budget.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `register` (mio programmer-error;
    /// always startup-fatal on a fresh fd). `accept()` errors other
    /// than `WouldBlock` propagate too — under normal operation
    /// these are `ECONNABORTED` (client closed between SYN and
    /// accept), which is rare enough to be terminal here.
    fn drain_accept(&mut self) -> io::Result<()> {
        loop {
            match self.listener.accept() {
                Ok((stream, _addr)) => {
                    if self.conns.len() >= MAX_IPC_CONNS {
                        // Cap reached. Drop the stream after writing
                        // a structured error so the client gets a
                        // clean refusal rather than a connection
                        // reset.
                        if let Err(e) = write_busy_then_drop(stream) {
                            tracing::debug!(
                                ?e,
                                "ipc busy-response write failed (peer likely already gone)",
                            );
                        }
                        continue;
                    }
                    let token = self.allocate_conn_token();
                    let mut conn = ConnState::new(stream, token);
                    self.poll
                        .registry()
                        .register(&mut conn.stream, token, Interest::READABLE)?;
                    self.conns.insert(token, conn);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        Ok(())
    }

    /// Mint a fresh per-conn [`Token`]. Monotone from
    /// [`TOKEN_CONN_BASE`]; wraps back to the base on overflow (a
    /// purely theoretical concern at scale, but the wrap exists so
    /// the type-level `wrapping_add` is structurally bounded).
    const fn allocate_conn_token(&mut self) -> Token {
        let raw = self.next_conn_token;
        // Wrapping increment: on `usize::MAX → 0`, snap back to
        // TOKEN_CONN_BASE so a hypothetical wrap doesn't collide with
        // the static set.
        let next = raw.wrapping_add(1);
        self.next_conn_token = if next < TOKEN_CONN_BASE {
            TOKEN_CONN_BASE
        } else {
            next
        };
        Token(raw)
    }

    /// Serialize a [`Diagnostic`] once and append its JSON line to
    /// every subscriber conn's write_queue.
    ///
    /// Subscriber storage IS the per-conn map: every conn whose role
    /// is [`crate::driver::conns::ConnRole::Sub`] is a subscriber,
    /// the per-Sub filter and back-pressure marker accumulator live
    /// on the role payload, and the line bytes are pushed directly
    /// into [`ConnState::write_queue`] on the reactor thread — no
    /// envelope, no worker-thread re-serialize hop.
    ///
    /// **`diag_sub` is caller-computed** — the projection
    /// `&Diagnostic → Option<SubId>` is an exhaustive walk of the
    /// core enum that lives next to its only call site
    /// ([`crate::driver::forward::diag_sub_id`]); this method takes
    /// the resolved value to keep the Diagnostic-walking concern out
    /// of the reactor module.
    ///
    /// **One serialize per dispatch.** The JSON bytes are built once
    /// before the conn loop and appended verbatim per subscriber via
    /// [`crate::driver::conns::ConnState::try_dispatch_diag`], which
    /// owns the per-conn five-axis verdict (close, role, filter,
    /// capacity, marker flush).
    ///
    /// **Interest re-arming is deferred** to
    /// [`Self::arm_writable_interests`], which runs once at the end
    /// of the tick's drain pass. Per-tick byte pushes therefore
    /// translate into one interest rearm per ready conn rather than
    /// N (one per `dispatch_to_subscribers` call) — the
    /// re-registration syscall amortizes across the whole tick.
    pub(super) fn dispatch_to_subscribers(
        &mut self,
        diag: &Diagnostic,
        at: SystemTime,
        diag_sub: Option<SubId>,
    ) {
        if self.conns.is_empty() {
            return;
        }
        let wire = WireDiagnostic::from((diag, at));
        let line = serialize_line(&wire)
            .expect("WireDiagnostic serialization is infallible by construction");
        for conn in self.conns.values_mut() {
            // The five-axis verdict (close, role, filter, capacity,
            // marker dance) and the per-conn missed-window
            // bookkeeping live on ConnState — the Hub's job here is
            // serialize-once + iterate.
            let _ = conn.try_dispatch_diag(&line, diag_sub, at);
        }
    }

    /// Borrow a per-conn state mutably by [`Token`]. Returns `None`
    /// if the conn closed between the tick's drain and the caller's
    /// reach (e.g., an oversize line set `close_after_flush`, the
    /// flush completed, and `terminate_conn` ran).
    ///
    /// Used by the IPC dispatcher's `Subscribe` arm to flip the conn's
    /// role *after* the ack bytes have been enqueued — the
    /// ack-before-fanout ordering pinned by the wire-side regression
    /// test.
    pub(super) fn conn_mut(&mut self, token: Token) -> Option<&mut ConnState> {
        self.conns.get_mut(&token)
    }

    /// Borrow a per-conn state immutably by [`Token`]. Returns `None`
    /// when the conn is no longer in the map — same lifecycle as
    /// [`Self::conn_mut`] (terminate_conn between tick boundaries can
    /// remove the entry).
    ///
    /// Read-only complement to `conn_mut` for handler-side predicates
    /// that need to inspect the conn's role / `close_after_flush` /
    /// queue state without mutating, and for test introspection of
    /// the post-tick per-conn shape. The Subscribe handler gates on
    /// the conn's existing role here before deciding whether to
    /// flip via [`crate::driver::conns::ConnState::transition_to_sub`]
    /// or refuse with
    /// [`crate::ipc::protocol::ERR_ALREADY_SUBSCRIBED`].
    pub(super) fn conn_ref(&self, token: Token) -> Option<&ConnState> {
        self.conns.get(&token)
    }

    /// Drain the per-conn read end into LF-delimited lines.
    ///
    /// Edge-triggered: loops `read(2)` until `WouldBlock`. EOF
    /// (`Ok(0)`) returns [`ReadOutcome::PeerGone`] so the caller
    /// terminates unconditionally. `Interrupted` retries; any other
    /// read error returns `PeerGone` too (the peer-side is gone, no
    /// further bytes will arrive).
    ///
    /// Complete lines are sliced off the front of `conn.read_buf`
    /// via `drain(..=nl)` (one allocation per line, no tail copy).
    /// Each line carries its trailing `\n` — the dispatcher trims
    /// it once before serde sees the bytes (mirroring
    /// `BufRead::read_line`'s convention).
    ///
    /// **Oversize line guard.** A line exceeding
    /// [`MAX_REQUEST_LINE_BYTES`] is structurally hostile (operator
    /// IPC verbs are small JSON objects; the largest verb is
    /// `Subscribe { name: <CompactString> }` at ~60 bytes). The
    /// guard arms `close_after_flush` via
    /// [`crate::driver::conns::ConnState::arm_close_after_flush`],
    /// breaks the line loop, and returns [`ReadOutcome::Continue`] —
    /// the caller pairs this with `try_terminate_if_idle` (so an
    /// armed-close + empty-queue terminates inline rather than
    /// lingering for a WRITABLE edge that will never come) AND lets
    /// `drain_writable` flush any partially-queued response bytes
    /// before teardown.
    ///
    /// **Read-accumulator size guard.** A peer streaming bytes
    /// without ever emitting an LF would grow `read_buf` unboundedly
    /// across ticks — every tick more bytes arrive, no LF, no line
    /// gets dispatched, no post-process termination fires. The
    /// post-loop check on `read_buf.len()` arms `close_after_flush`
    /// for that case; the same `try_terminate_if_idle` pairing
    /// terminates the conn at end-of-tick.
    ///
    /// # Errors
    ///
    /// Returns `Err(NotFound)` only if `token` is not in the conn
    /// map — that would be a tick-body bug (we drain tokens that
    /// arrived this tick; if the conn isn't there, we never accepted
    /// it). All transport-level read failures map to
    /// `Ok(ReadOutcome::PeerGone)` so the caller's terminate path
    /// is uniform.
    pub(super) fn read_conn_into_lines(
        &mut self,
        token: Token,
        out: &mut Vec<Vec<u8>>,
    ) -> io::Result<ReadOutcome> {
        use std::io::Read;
        let conn = self.conns.get_mut(&token).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("ipc read: no conn for {token:?}"),
            )
        })?;
        let mut buf = [0u8; 4096];
        let mut peer_gone = false;
        loop {
            match conn.stream.read(&mut buf) {
                Ok(0) => {
                    peer_gone = true;
                    break;
                }
                Ok(n) => conn.read_buf.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    tracing::debug!(?token, ?e, "ipc read failed; closing conn");
                    peer_gone = true;
                    break;
                }
            }
        }
        while let Some(nl) = conn.read_buf.iter().position(|&b| b == b'\n') {
            // `drain(..=nl)` consumes the inclusive prefix in O(n)
            // — the tail of `read_buf` shifts forward once per line.
            // For the steady-state "one verb per line" cadence the
            // tail is empty, so the shift is a no-op.
            let line: Vec<u8> = conn.read_buf.drain(..=nl).collect();
            if line.len() > MAX_REQUEST_LINE_BYTES {
                tracing::warn!(
                    ?token,
                    len = line.len(),
                    "ipc request line exceeds MAX_REQUEST_LINE_BYTES; closing conn",
                );
                conn.arm_close_after_flush();
                break;
            }
            out.push(line);
        }
        if conn.read_buf.len() > MAX_REQUEST_LINE_BYTES {
            tracing::warn!(
                ?token,
                buffered = conn.read_buf.len(),
                "ipc read accumulator exceeds MAX_REQUEST_LINE_BYTES with no LF; closing conn",
            );
            conn.arm_close_after_flush();
        }
        Ok(if peer_gone {
            ReadOutcome::PeerGone
        } else {
            ReadOutcome::Continue
        })
    }

    /// Serialize a response value and append the JSON line to the
    /// conn's write_queue (the trailing `\n` is added by
    /// [`serialize_line`]; framing is line-delimited).
    ///
    /// Capacity-gated via
    /// [`crate::driver::conns::ConnState::push_response`]: if the
    /// projected queue length would exceed
    /// [`crate::driver::conns::WRITE_QUEUE_HIGH_WATER`], the queue
    /// is left untouched, `close_after_flush` is armed, and this
    /// returns [`EnqueueOutcome::Refused`]. The refusal path then
    /// runs [`Self::try_terminate_if_idle`] inline — when the queue
    /// was empty at the time of the refusal (the "oversize response
    /// into an idle conn" linger case), no WRITABLE edge would ever
    /// observe the armed flag, so the terminate has to happen here.
    /// When the queue had bytes (a normal response queued first,
    /// then an over-water one), the in-flight bytes drain through
    /// [`Self::drain_writable`] and the close-flag is observed at
    /// flush time.
    ///
    /// Serializer failure is `expect`'d: the daemon's wire types
    /// ([`crate::ipc::protocol::ResponsePayload`] /
    /// [`crate::ipc::wire::WireDiagnostic`]) are `Serialize`-derive
    /// over plain-data fields and structurally cannot fail. The
    /// wire-side `wire_diagnostic_round_trips_via_serde` regression
    /// covers the projection invariant; reaching this panic surfaces
    /// a programmer-error in the same shape the fan-out path uses,
    /// rather than swallowing the error as a stale `let _ = ...`
    /// `io::Error` would.
    pub(super) fn enqueue_response<T: serde::Serialize>(
        &mut self,
        token: Token,
        response: &T,
    ) -> EnqueueOutcome {
        let bytes = serialize_line(response)
            .expect("ipc response serialization is infallible by construction");
        // Scope the conn borrow so the post-push `try_terminate_if_idle`
        // can reach `&mut self` cleanly on the Refused arm. PushOutcome
        // is Copy, so the binding outlives the borrow without effort.
        let outcome = match self.conns.get_mut(&token) {
            Some(conn) => conn.push_response(&bytes),
            None => return EnqueueOutcome::ConnGone,
        };
        match outcome {
            PushOutcome::Accepted => EnqueueOutcome::Accepted,
            PushOutcome::Refused => {
                tracing::warn!(
                    ?token,
                    response_len = bytes.len(),
                    "ipc response over write-queue high-water; arming close",
                );
                // push_response already armed close_after_flush. The
                // queue may be empty (over-water response into an idle
                // conn) in which case no WRITABLE edge ever drains and
                // observes the flag — terminate inline. If the queue
                // had bytes, the conn stays in the map and
                // drain_writable handles termination on the flush edge.
                self.try_terminate_if_idle(token);
                EnqueueOutcome::Refused
            }
        }
    }

    /// Terminate `token` iff the conn is armed for close AND the
    /// write_queue is empty (nothing left to flush). Returns `true`
    /// on terminate, `false` otherwise (queue still holding bytes,
    /// close not armed, or conn already gone).
    ///
    /// Called at two natural moments:
    /// 1. After [`super::EngineDriver::drain_ipc_lines`] finishes
    ///    processing a conn's lines — the post-process pass folds in
    ///    any response bytes the handler may have pushed, so the
    ///    queue state at THIS point is the conn's settled state for
    ///    the tick.
    /// 2. From [`Self::enqueue_response`]'s `Refused` arm — an
    ///    over-water response that didn't fit into a previously-
    ///    empty queue must terminate now rather than waiting for a
    ///    WRITABLE edge that will never come (the queue is empty;
    ///    no edge will trigger).
    ///
    /// Termination needs the mio Poll registry to deregister the
    /// stream, which is why this lives on Hub rather than on
    /// `ConnState`.
    pub(super) fn try_terminate_if_idle(&mut self, token: Token) -> bool {
        let should_terminate = self
            .conns
            .get(&token)
            .is_some_and(|c| c.close_after_flush && c.write_queue.is_empty());
        if should_terminate {
            self.terminate_conn(token);
        }
        should_terminate
    }

    /// Drain the conn's write_queue to the kernel non-blockingly.
    ///
    /// Edge-triggered: loops `write(2)` against the queue's front
    /// slice until WouldBlock, the queue empties, or the peer is
    /// gone. `Interrupted` retries; an `Ok(0)` is treated as
    /// "no progress this iteration" and breaks the loop (the next
    /// WRITABLE edge re-enters).
    ///
    /// **Queue-empty bookkeeping.** When the queue empties:
    /// 1. Re-register the conn with `READABLE` only — the WRITABLE
    ///    interest was a transient "drain me" flag; leaving it
    ///    armed against an empty queue would have mio fire on every
    ///    socket-send-buffer-room edge.
    /// 2. If `close_after_flush` is set (oversize line, over-water
    ///    response, etc.), return `Ok(true)` so the caller runs
    ///    `terminate_conn`. The arm-for-close → drain → close sequence
    ///    guarantees the last queued bytes reach the wire before
    ///    teardown.
    ///
    /// Returns `Ok(true)` ⇒ caller terminates this conn; `Ok(false)`
    /// ⇒ keep the conn open (queue empty + no close flag, or queue
    /// still has bytes and the next WRITABLE edge will continue).
    ///
    /// # Errors
    ///
    /// `NotFound` for an unknown token; any other transport error
    /// (peer-gone, etc.) maps to `Ok(true)` so the caller's terminate
    /// path is uniform with the read drain.
    pub(super) fn drain_writable(&mut self, token: Token) -> io::Result<bool> {
        use std::io::Write;
        let conn = self.conns.get_mut(&token).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("ipc drain_writable: no conn for {token:?}"),
            )
        })?;
        while !conn.write_queue.is_empty() {
            // `as_slices().0` is the contiguous front slice — no
            // need to defragment for the write. `VecDeque::drain(..n)`
            // peels off the consumed prefix in O(n) (the tail shifts
            // forward); under steady-state the queue is small and
            // the shift is negligible.
            let front = conn.write_queue.as_slices().0;
            match conn.stream.write(front) {
                Ok(0) => break,
                Ok(n) => {
                    conn.write_queue.drain(..n);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    tracing::debug!(?token, ?e, "ipc write failed; closing conn");
                    return Ok(true);
                }
            }
        }
        if conn.write_queue.is_empty() {
            // Disarm WRITABLE — leaving it armed on an empty queue
            // would fire on every send-buffer-room edge.
            self.poll
                .registry()
                .reregister(&mut conn.stream, token, Interest::READABLE)?;
            if conn.close_after_flush {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Remove a conn from the map, deregistering its mio source
    /// first so no straggler event for this Token reaches the
    /// dispatch loop after teardown.
    ///
    /// The deregister-before-drop discipline matches mio's
    /// "deregister explicitly before drop" contract — a
    /// `ConnState::stream` that drops without going through this
    /// function would strand a registration, and the next `poll`
    /// would fire events against a Token whose conn has gone
    /// (silent miss; harmless in practice but technically a
    /// programmer-error).
    ///
    /// Deregister errors are logged at debug — the most common
    /// cause is "fd already closed" (a write failure earlier in the
    /// tick); the path is best-effort, the next `terminate_conn`
    /// for the same Token would be a `None` on the map.
    ///
    /// Audit grep: `conns.remove\|conns.drain` returns only this
    /// function — the single removal path keeps the
    /// deregister-before-drop discipline structurally enforced.
    pub(super) fn terminate_conn(&mut self, token: Token) {
        let Some(mut conn) = self.conns.remove(&token) else {
            return;
        };
        if let Err(e) = self.poll.registry().deregister(&mut conn.stream) {
            tracing::debug!(
                ?token,
                ?e,
                "ipc deregister failed (fd likely already closed)"
            );
        }
        // `conn` drops here — `ConnState::stream`'s Drop closes
        // the fd; the read/write buffers drop their allocations.
    }

    /// Walk every conn and re-register the mio interest to include
    /// `WRITABLE` for any conn with pending bytes.
    ///
    /// Called once at the end of the tick's drain pass — see
    /// [`Self::dispatch_to_subscribers`] for the "amortize one
    /// rearm across the whole tick" rationale. A conn whose queue
    /// is empty stays at `READABLE` only (the disarm happens in
    /// `drain_writable` when the queue empties via a successful
    /// write).
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `Registry::reregister` — a
    /// programmer-error class failure (the conn's fd is gone,
    /// implying we missed a `terminate_conn` somewhere). The tick
    /// treats the error as terminal.
    pub(super) fn arm_writable_interests(&mut self) -> io::Result<()> {
        for conn in self.conns.values_mut() {
            if !conn.write_queue.is_empty() {
                self.poll.registry().reregister(
                    &mut conn.stream,
                    conn.token,
                    Interest::READABLE | Interest::WRITABLE,
                )?;
            }
        }
        Ok(())
    }

    /// Test-only mutable borrow of the owned watcher. Lets the driver's
    /// `mod tests` reach into a [`specter_sensor::testkit::MockFsWatcher`]
    /// to inject events between tick passes — the production seam
    /// (kernel-side fd → mio readiness → `drain_watcher`) is the same;
    /// only the *source* of the readiness edge differs.
    ///
    /// Gated behind `#[cfg(test)]` so production builds carry no extra
    /// surface; production never reaches into the watcher outside Hub.
    /// `pub(super)` keeps the accessor scoped to the `driver` module,
    /// matching the rest of Hub's private surface.
    #[cfg(test)]
    pub(super) const fn watcher_mut(&mut self) -> &mut W {
        &mut self.watcher
    }

    /// Test-only read of the conn-map size. Used to assert
    /// "accept happened" / "terminate happened" without observing the
    /// wire (the wire-side assertion is the load-bearing one; this is
    /// belt-and-braces for tests where the conn lifecycle is the
    /// subject and the wire payload is incidental).
    #[cfg(test)]
    pub(super) fn conn_count(&self) -> usize {
        self.conns.len()
    }
}

impl<W: FsWatcher> std::fmt::Debug for DriverHub<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DriverHub")
            .field("conn_count", &self.conns.len())
            .field("config_watcher", &self.config_watcher.is_some())
            .field("next_conn_token", &self.next_conn_token)
            .finish_non_exhaustive()
    }
}

/// Best-effort blocking write of a structured `ERR_BUSY` response to
/// a stream we are about to drop. The conn count is at the cap; the
/// peer gets one short JSON line and the stream closes.
///
/// `set_nonblocking(false)` + `set_write_timeout(Some(500ms))` bounds
/// the wait: a healthy peer receives ~80 bytes in microseconds; a
/// wedged peer with a full receive buffer hits the timeout and the
/// caller logs the failure. The bound is generous enough to ride out
/// scheduler contention, tight enough that a hostile client cannot
/// stall the accept path more than half a second.
fn write_busy_then_drop(stream: mio::net::UnixStream) -> io::Result<()> {
    use std::io::Write;
    let mut std_stream: std::os::unix::net::UnixStream = stream.into();
    std_stream.set_nonblocking(false)?;
    std_stream.set_write_timeout(Some(Duration::from_millis(500)))?;
    let resp = ResponsePayload::Err {
        code: Cow::Borrowed(ERR_BUSY),
        error: "max concurrent connections".into(),
    };
    let bytes = serialize_line(&resp)?;
    std_stream.write_all(&bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signals::register_signal_handlers;
    use crossbeam::channel::unbounded;
    use specter_sensor::testkit::MockFsWatcher;
    use std::path::PathBuf;
    use std::time::Instant;

    /// `next_inputs(Some(ZERO))` returns immediately with no events
    /// queued. Pins the non-blocking poll path.
    #[test]
    fn next_inputs_zero_timeout_returns_empty() {
        let tmp = tempfile::TempDir::new().unwrap();
        let sock = tmp.path().join("specter.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind tmp socket");
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let (mut hub, _waker) =
            DriverHub::new(listener, watcher, None, signals, prober_rx, effect_rx)
                .expect("hub init");
        let drained = hub
            .next_inputs(Some(Duration::ZERO))
            .expect("non-blocking poll succeeds");
        assert!(drained.fs_events.is_empty());
        assert!(drained.sensor_overflows.is_empty());
        assert!(drained.probe_responses.is_empty());
        assert!(drained.effect_completions.is_empty());
        assert!(drained.signals.is_empty());
        assert!(!drained.config_event_pulse);
        assert!(drained.ready_reads.is_empty());
        assert!(drained.ready_writes.is_empty());
    }

    /// `waker.wake()` on an external clone causes the next
    /// `next_inputs` call to return immediately even with a generous
    /// timeout. Pins the wake fire-edge → mio dispatch path.
    #[test]
    fn waker_wake_unblocks_next_inputs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let sock = tmp.path().join("specter.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind tmp socket");
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let (mut hub, waker) =
            DriverHub::new(listener, watcher, None, signals, prober_rx, effect_rx)
                .expect("hub init");

        // Push an Input into the prober channel — without a wake,
        // the next_inputs poll wouldn't see it (it polls fd-readiness,
        // not channel-non-emptiness). The wake fires the TOKEN_WAKER
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
        let drained = hub
            .next_inputs(Some(Duration::from_secs(5)))
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
        let tmp = tempfile::TempDir::new().unwrap();
        let sock = tmp.path().join("specter.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind tmp socket");
        let watcher = MockFsWatcher::new();
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let (mut hub, _waker) =
            DriverHub::new(listener, watcher, None, signals, prober_rx, effect_rx)
                .expect("hub init");

        let mut slotmap: slotmap::SlotMap<ResourceId, ()> = slotmap::SlotMap::with_key();
        let r = slotmap.insert(());
        let path: std::sync::Arc<std::path::Path> =
            std::sync::Arc::from(PathBuf::from("/tmp").as_path());
        let rejected = hub.apply_watch_ops(&[WatchOp::Watch {
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
        let tmp = tempfile::TempDir::new().unwrap();
        let sock = tmp.path().join("specter.sock");
        let listener = std::os::unix::net::UnixListener::bind(&sock).expect("bind tmp socket");
        let mut watcher = MockFsWatcher::new();
        // EMFILE = 24 on every supported target; hardcoded to avoid a
        // libc dev-dep in the bin's test surface.
        let failure = WatchFailure::Pressure { errno: 24 };
        watcher.fail_next_watch(failure);
        let signals = register_signal_handlers().expect("signal pipe init");
        let (_prober_tx, prober_rx) = unbounded::<Input>();
        let (_effect_tx, effect_rx) = unbounded::<Input>();

        let (mut hub, _waker) =
            DriverHub::new(listener, watcher, None, signals, prober_rx, effect_rx)
                .expect("hub init");

        let mut slotmap: slotmap::SlotMap<ResourceId, ()> = slotmap::SlotMap::with_key();
        let r = slotmap.insert(());
        let path: std::sync::Arc<std::path::Path> =
            std::sync::Arc::from(PathBuf::from("/tmp").as_path());
        let rejected = hub.apply_watch_ops(&[WatchOp::Watch {
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
