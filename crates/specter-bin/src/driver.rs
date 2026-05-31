//! Engine driver — the bin's main-thread loop, split across focused
//! submodules with the spine here.
//!
//! [`EngineDriver`] owns the [`Engine`], the [`Loader`], a
//! [`state::DriverState`] (process-level facts: start instants +
//! reload counters + socket path), an operator-runtime disable
//! override set, the actuator-coordination channels, the prober
//! [`Arc`] clone, the deferred-input queue, and the two mio-reactor
//! halves: [`reactor::Reactor`] (kernel-driven non-IPC sources) and
//! [`ipc::Hub`] (the operator-IPC listener + per-conn map). This
//! module holds the struct and its lifecycle ([`EngineDriver::new`],
//! [`EngineDriver::run_initial_attach`], [`EngineDriver::run`]) plus
//! the cancel-first shutdown drain (`begin_shutdown`). The load-bearing
//! work lives next to it:
//!
//! - [`tick`] — one pass of the drain order (accept → sensor →
//!   timers → reload → config-settle → effects → ipc → block). The
//!   hot loop; new inbound-path work lands there.
//! - [`reload`] — the SIGHUP + auto-reload settle pipeline.
//! - [`forward`] — ships a `StepOutput` downstream, maps a
//!   `Diagnostic` to tracing, and fans diagnostics out to live IPC
//!   subscribers via [`ipc::Hub::dispatch_to_subscribers`].
//! - [`state`] — driver-owned process facts (startup instants,
//!   reload counters, socket path) consumed by the IPC `status`
//!   surface.
//! - [`reactor`] — owner of the mio reactor surface for kernel-driven
//!   non-IPC sources (watcher, config-watcher, signal pipe, waker,
//!   channel receivers).
//! - [`ipc`] — the daemon-side IPC concern: kernel-fd owner ([`ipc::Hub`]),
//!   per-conn state ([`ipc::conns`]), verb dispatch
//!   ([`ipc::dispatch`]), and engine-state projection
//!   ([`ipc::project`]) — all registered against the same Poll selector
//!   as the Reactor via a [`mio::Registry::try_clone()`] handle.
//!
//! `run_initial_attach` walks `loader.current_config` in source order,
//! attaching each Sub / Promoter and forwarding the resulting output
//! immediately so the watcher / prober see work as it lands. `run`
//! wraps [`EngineDriver::tick`] until shutdown. All file I/O is on
//! this thread — no Mutex.

mod forward;
mod ipc;
mod reactor;
mod reload;
mod state;
mod tick;
mod wake;

use crate::actuator::ActuatorIO;
use crate::app::CliLogOverrides;
use crate::loader::Loader;
use crate::observability::ObservabilityHandle;
use compact_str::CompactString;
use specter_core::Input;
use specter_engine::Engine;
use specter_sensor::{DefaultWatcher, FsWatcher, Prober};
use std::collections::{BTreeSet, VecDeque};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

pub(crate) use ipc::Hub;
pub(crate) use reactor::Reactor;
pub(crate) use state::{DriverState, ReloadTrigger};
pub(crate) use wake::{WakeHandle, WakingSink};

/// Reason the driver loop exited. Returned from [`EngineDriver::run`].
///
/// v1 has only the `Shutdown` variant — every path that could exit the
/// loop without a shutdown signal (sensor channel disconnect) currently
/// also routes through `TickOutcome::Shutdown` per [`EngineDriver::tick`].
/// The enum exists so v2 (recovery / restart) has a structural seam
/// without breaking the [`EngineDriver::run`] return type.
#[derive(Debug, Eq, PartialEq)]
pub enum ExitReason {
    /// SIGINT / SIGTERM dispatched (operator-driven, normal path), OR
    /// a downstream channel disconnected (actuator thread crash; v1
    /// treats both as terminal-graceful).
    Shutdown,
}

/// Outcome of a single [`EngineDriver::tick`] call. The loop wrapper
/// matches on this; explicit enum is friendlier than a bool.
#[derive(Debug, Eq, PartialEq)]
pub enum TickOutcome {
    /// Inputs drained; loop again.
    Continue,
    /// Operator signal or sensor disconnect. The tick has already run
    /// the cancel-first probe drain ([`EngineDriver::begin_shutdown`]),
    /// so the engine holds no armed probe: tearing the driver down
    /// (the bin's `drop(driver)`) will not trip the linear `ProbeSlot`
    /// Drop guard.
    Shutdown,
}

/// Engine driver — see module rustdoc.
///
/// **Generic over `W: FsWatcher`** so tests can substitute the
/// sensor crate's `MockFsWatcher` (whose socketpair-backed `AsFd`
/// surface lets reactor-integration tests run against a real
/// `mio::Poll`). Production uses the platform [`DefaultWatcher`];
/// the type-parameter default keeps `app.rs` and call sites free of
/// `<DefaultWatcher>` boilerplate (inference fills the type from the
/// `Reactor<W>` passed to [`Self::new`]).
///
/// **Field order is the drop order.** [`Engine`] FIRST so the probe
/// tripwire (`specter_core::probe`) runs against a fully-armed
/// engine until [`Self::begin_shutdown`] has drained it.
/// [`ipc::Hub`] drops BEFORE [`reactor::Reactor`] so
/// its explicit `Drop` impl can deregister the listener + every
/// live conn stream against the still-live Poll selector;
/// [`reactor::Reactor`] drops LAST so the mio reactor fds (watcher,
/// signal pipe, waker) outlive every channel-sender clone the
/// driver holds — a stray send-to-disconnected from a midway drop
/// can't fire on a partially-torn-down reactor.
///
/// **Shutdown is co-owned with the actuator.** [`Self::run`] exits
/// only when (a) `effect_complete_rx` observes
/// [`crossbeam::channel::TryRecvError::Disconnected`] (the actuator
/// thread exited — clean or crashed — and its
/// `Box<dyn EffectCompleteSender>` adapter dropped the inner
/// [`wake::WakingSink`], which closes its `Sender<Input>` clone via
/// the [`wake::WakingSink::drop`] close-then-wake protocol), or (b)
/// the operator escalates a second SIGINT/SIGTERM within
/// [`crate::signals::HARD_EXIT_WINDOW`] and
/// [`dispatch_signal_inner`] reaches the `HardExit` arm. The first
/// SIGINT/SIGTERM pulses `shutdown_actuator_tx` and arms
/// `first_term` (the IPC-mutating-verb gate); the driver keeps
/// ticking — mutating IPC verbs refuse with
/// [`crate::ipc::protocol::WireErrorCode::ShuttingDown`]; effect
/// completions and probe responses arriving during the actuator's
/// SIGTERM → grace → SIGKILL → reap-drain pipeline continue to flow
/// through [`forward::EngineDriver::forward`]. [`crate::signals::SignalPipe`]
/// lives inside [`reactor::Reactor`] lives inside this driver — all
/// three stay alive until the actuator-gone signal lands — so the
/// second-tap hard-exit handshake stays installed across the entire
/// actuator-grace window, leaving no handshake orphaned mid-grace.
pub struct EngineDriver<W: FsWatcher = DefaultWatcher> {
    engine: Engine,
    loader: Loader,
    config_path: PathBuf,
    /// CLI overrides applied to `[log]` at startup. Re-applied on every
    /// SIGHUP-driven reload so CLI precedence stays consistent across
    /// the process lifetime (`CLI > config > default`).
    cli_log_overrides: CliLogOverrides,
    /// Subscriber handle for runtime updates (`set_level`,
    /// `reopen_file`). Held here so `dispatch_reload` can fire both on
    /// SIGHUP without going through the loader.
    obs_handle: ObservabilityHandle,
    /// Process-level facts (startup instants + reload counters +
    /// socket path). Constructed at boot via [`DriverState::new`] and
    /// mutated only through [`DriverState::record_reload`] — the edge
    /// method guarantees the counter fields move together. Consumed
    /// by the IPC `status` surface.
    driver_state: DriverState,
    prober: Arc<dyn Prober>,
    /// Auto-reload settle deadline — armed by the config-event drain,
    /// expires after [`CONFIG_SETTLE`] of quiet, at which point the
    /// driver runs the lstat-vs-`loader.config_meta` filter and (on
    /// drift) calls [`Self::dispatch_reload`]. Reset to `None` on
    /// expiry and re-armed per pulse (settle resets, so sustained
    /// bursts defer the reload until the edits actually settle).
    ///
    /// Two consumers:
    /// - The [`Self::tick`] block-timeout math feeds the deadline into
    ///   `mio::Poll::poll`'s timeout so the driver wakes precisely
    ///   when the window expires.
    /// - [`Self::apply_config_settle_expiry`] gates the lstat call
    ///   on `now >= deadline` so the engine thread never lstats
    ///   before the settle window has elapsed.
    config_settle_until: Option<Instant>,
    /// Operator-IPC runtime disable overrides — names of Subs the
    /// operator disabled via `specter disable` and has not yet
    /// re-enabled. Empty at boot — the set is process-local and not
    /// persisted across restarts. Read by the IPC `status` projection
    /// (`len()`) and the IPC `list`/`show` projections (set
    /// membership); mutated by the IPC `disable` / `enable` handlers
    /// (which also filter the next reload's diff so a runtime-disabled
    /// Sub is not re-attached).
    disabled_runtime: BTreeSet<CompactString>,
    /// Driver-side actuator-coordination channels — effects pipe +
    /// the three shutdown handshake legs. Wired in by [`Self::new`]
    /// from `App::run`'s [`ActuatorIO::pair`] allocation;
    /// [`Self::dispatch_signal_with_exit_fn`] pulses
    /// `shutdown_actuator_tx` / `hard_shutdown_actuator_tx` and
    /// waits on `hard_shutdown_done_rx`. [`Self::forward`] dispatches
    /// every emitted `EffectOp` through `effects_tx`.
    actuator_io: ActuatorIO,
    /// Timestamp of the first SIGINT / SIGTERM the driver observed.
    /// `None` until the first signal lands; the next SIGINT / SIGTERM
    /// within [`crate::signals::HARD_EXIT_WINDOW`] escalates to
    /// the hard-exit path. Outside the window, the field re-arms
    /// with the new timestamp.
    ///
    /// **Dual role: shutdown-in-flight gate.** `first_term.is_some()`
    /// is the IPC-mutating-verb gate read by
    /// [`Self::handle_ipc_line`] — `Reload` / `Disable` / `Enable` /
    /// `Absorb` requests arriving after the first termination signal refuse
    /// with [`crate::ipc::protocol::WireErrorCode::ShuttingDown`].
    /// Read-only verbs and `Subscribe` (bin-local mutation) stay
    /// accessible so operators can `specter tail` the wind-down. One
    /// source of truth — adding a parallel `shutdown_in_flight: bool`
    /// would shadow this field's semantics; the time-window math and
    /// the gate share the same `Some(_)` discriminant by
    /// construction.
    first_term: Option<Instant>,
    /// Replay queue for engine [`Input`]s the driver wants the next
    /// tick to process *before* the mio Poll re-blocks.
    ///
    /// Lifecycle: `forward()`'s inline `apply_watch_ops` returns the
    /// rejected ops; each is queued here as
    /// `Input::WatchOpRejected { resource, failure }`. The next
    /// tick's `replay_deferred_inputs` (called at the top of `tick`,
    /// before any fresh mio drain) runs `engine.step` on each in
    /// FIFO order. When the queue is non-empty the block timeout
    /// collapses to `Duration::ZERO` so the replay isn't deferred
    /// behind a long wait.
    ///
    /// The queue is unbounded by type but soft-capped in
    /// `replay_deferred_inputs` (debug-asserted against a sane
    /// upper bound) so pathological engine emission shape that
    /// produced unbounded rejections would fail loud rather than
    /// silently grow.
    deferred_inputs: VecDeque<Input>,
    /// The operator-IPC kernel surface — bound listener + per-conn
    /// map + per-conn Token allocator + a [`mio::Registry::try_clone()`]
    /// handle pointing at the [`reactor::Reactor`]'s Poll selector.
    /// Owned by the driver for the lifetime of the daemon;
    /// constructed once in `App::run` from the registry clone the
    /// Reactor handed back.
    ///
    /// Field-order BEFORE `reactor` so the explicit `Drop` impl on
    /// [`ipc::Hub`] can deregister the listener + every live conn
    /// stream against the still-live Poll selector; once the Hub is
    /// gone, its Registry clone drops, leaving the Reactor's own Poll
    /// holding the last Arc-reference to the underlying selector. See
    /// [`ipc::hub`]'s module rustdoc for the Hub-internal field drop
    /// order.
    ipc: Hub,
    /// The mio reactor surface for kernel-driven non-IPC sources —
    /// kqueue/inotify watcher, config-watcher, signal pipe, and the
    /// `Arc<Waker>` shared with the prober + actuator wake-bearing
    /// senders. Owned by the driver for the lifetime of the daemon;
    /// constructed once in `App::run` and threaded through
    /// [`Self::new`].
    ///
    /// Field-order LAST so the static-source fds (watcher, signal
    /// pipe, waker) outlive every channel-sender clone the driver
    /// holds — a stray send-to-disconnected from a midway drop
    /// can't fire on a partially-torn-down Reactor. See
    /// [`reactor::Reactor`]'s module rustdoc for the Reactor-internal
    /// field drop order.
    reactor: Reactor<W>,
}

impl<W: FsWatcher> std::fmt::Debug for EngineDriver<W> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineDriver")
            .field("loader", &self.loader)
            .field("config_path", &self.config_path)
            .field("cli_log_overrides", &self.cli_log_overrides)
            .field("obs_handle", &self.obs_handle)
            .field("driver_state", &self.driver_state)
            .field("disabled_runtime", &self.disabled_runtime)
            .field("ipc", &self.ipc)
            .field("reactor", &self.reactor)
            .finish_non_exhaustive()
    }
}

impl<W: FsWatcher> EngineDriver<W> {
    /// Borrow the config file path this driver was constructed against.
    /// The path lives for the driver's lifetime — `App::run` consumes
    /// `args.config` into the constructor below, so the boot-time
    /// startup-TOCTOU lstat reads it back through this accessor.
    pub(crate) fn config_path(&self) -> &std::path::Path {
        &self.config_path
    }

    /// Build the driver from preconstructed pieces.
    ///
    /// The [`Reactor`] is constructed in `App::run` first because it
    /// anchors the canonical [`mio::Waker`] (via its `waker` field,
    /// reached through [`Reactor::wake_handle`]) that the prober +
    /// actuator wrappers clone before the driver gets built. The
    /// [`Hub`]'s [`mio::Registry::try_clone()`] handle is minted by
    /// a follow-up [`Reactor::registry_clone`] call — the boot order
    /// is therefore Reactor → registry_clone → Hub → wake_handle
    /// clones → wrappers → EngineDriver. Passing both halves in
    /// here keeps the construction order honest.
    ///
    /// **Argument order matches drop order.** `ipc` precedes `reactor`
    /// in the parameter list as a syntactic mirror of the field order
    /// (which IS the drop order); a future contributor wiring a new
    /// driver would write the constructor call in the same syntactic
    /// shape as the struct.
    #[must_use]
    pub(crate) fn new(
        engine: Engine,
        loader: Loader,
        config_path: PathBuf,
        socket_path: PathBuf,
        cli_log_overrides: CliLogOverrides,
        obs_handle: ObservabilityHandle,
        prober: Arc<dyn Prober>,
        actuator_io: ActuatorIO,
        ipc: Hub,
        reactor: Reactor<W>,
    ) -> Self {
        Self {
            engine,
            loader,
            config_path,
            cli_log_overrides,
            obs_handle,
            driver_state: DriverState::new(socket_path),
            prober,
            config_settle_until: None,
            disabled_runtime: BTreeSet::new(),
            actuator_io,
            first_term: None,
            deferred_inputs: VecDeque::new(),
            ipc,
            reactor,
        }
    }

    /// Attach every active Sub and Promoter from
    /// `loader.current_config` in source order. Disabled entries are
    /// filtered out via [`Config::active_watches`] /
    /// [`Config::active_promoters`] — they remain in the raw `Vec`s
    /// for introspection but never reach the engine, mirroring the
    /// "disabled = absent" discipline the diff layer applies to
    /// hot-reload.
    ///
    /// Each [`StepOutput`] is forwarded as we go so the watcher /
    /// prober receive ops as the engine emits them — a single
    /// startup-sized `ConfigDiff` would batch the entire attach into
    /// one output and stall the watcher behind the post-call
    /// `forward`. Hot-reload (in `reload.rs`) deliberately uses the
    /// inverse pattern — a single batched `Input::ConfigDiff` — because
    /// reload diffs are typically small. Revisit if those diffs grow
    /// large enough to stall the watcher behind a single `forward`.
    ///
    /// No bin-side reconciliation: the engine owns `name → id`
    /// resolution through its registries' `by_name` indices. The
    /// `SubAttached` / `PromoterAttached` diagnostics are pure operator
    /// narration, logged via `forward`.
    ///
    /// Returns [`ControlFlow::Break`] if any `forward` observed a
    /// downstream channel disconnect (actuator-thread death surfaces
    /// as `effects_tx` returning `Err(Disconnected)`). On `Break` we
    /// run [`Self::begin_shutdown`] before returning — an attached Sub
    /// leaves the Profile in a Seed-Verifying state with an armed
    /// `ProbeSlot`, and a caller that just drops the driver would trip
    /// `ProbeSlot::drop`'s linear-edge tripwire. Containing the probe
    /// drain inside `run_initial_attach` keeps the lifecycle discipline
    /// encapsulated; the caller (`app.rs`) stays a thin branch on the
    /// `ControlFlow` return.
    pub(crate) fn run_initial_attach(&mut self) -> ControlFlow<()> {
        let now = Instant::now();
        // Snapshot the active spec lists: `self.engine.step` needs
        // `&mut self`, so the `&self` borrow on `loader.current_config`
        // cannot be held across the loop.
        let watch_specs: Vec<_> = self
            .loader
            .current_config()
            .active_watches()
            .cloned()
            .collect();
        let promoter_specs: Vec<_> = self
            .loader
            .current_config()
            .active_promoters()
            .cloned()
            .collect();
        for spec in watch_specs {
            let req = spec.to_attach_request();
            let out = self.engine.step(Input::AttachSub(req), now);
            if self.forward(out).is_break() {
                let _ = self.begin_shutdown();
                return ControlFlow::Break(());
            }
        }
        for spec in promoter_specs {
            let req = spec.to_attach_request();
            let out = self.engine.step(Input::AttachPromoter(req), now);
            if self.forward(out).is_break() {
                let _ = self.begin_shutdown();
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }

    /// Loop wrapping [`Self::tick`] until shutdown.
    ///
    /// MUST NOT be wrapped in `catch_unwind`: `ProbeSlot`'s in-unwind
    /// silence (`specter_core::probe`) depends on a mid-`step` panic
    /// being fatal — catching it here would let the daemon carry on
    /// with a probe-bearing state torn down mid-flight.
    pub fn run(&mut self) -> ExitReason {
        loop {
            match self.tick() {
                TickOutcome::Continue => {}
                TickOutcome::Shutdown => return ExitReason::Shutdown,
            }
        }
    }

    /// Inline dispatch for a single signal value drained off the
    /// Reactor's `TOKEN_SIGNAL` arm.
    ///
    /// Production wrapper: delegates to
    /// [`Self::dispatch_signal_with_exit_fn`] with the real
    /// `std::process::exit` as the escalation closure. Tests call the
    /// `_with_exit_fn` variant directly with a recording closure so
    /// the test process survives the assertion.
    ///
    /// Returns [`ControlFlow::Continue`] for SIGHUP (the reload either
    /// applies or fails — both are non-terminal) AND for the first
    /// SIGINT/SIGTERM (shutdown initiated; the driver stays alive
    /// through the actuator's grace and exits via
    /// [`super::reactor::DrainedTick::actuator_gone`] when the
    /// actuator-thread closure drops its sender). Returns
    /// [`ControlFlow::Break`] only for the second-within-window
    /// (hard-exit escalation) — only reachable from tests because
    /// production's `exit_fn` does not return.
    ///
    /// Called from [`tick`]'s signal-drain pass: the Reactor's
    /// `TOKEN_SIGNAL` arm hands every queued signum to this method in
    /// arrival order. No cross-thread coordination, no global-handler
    /// racing — the dispatch runs on the same thread that owns the
    /// engine.
    pub(crate) fn dispatch_signal(&mut self, sig: i32, now: Instant) -> ControlFlow<()> {
        self.dispatch_signal_with_exit_fn(sig, now, |code| std::process::exit(code))
    }

    /// Test-friendly variant of [`Self::dispatch_signal`] that takes
    /// an injectable `exit_fn` closure in place of
    /// [`std::process::exit`]. Production calls this with
    /// `|code| std::process::exit(code)`; tests pass a closure that
    /// records the requested exit code so the test runner survives
    /// the hard-exit branch.
    ///
    /// The pure signal-classification work runs in
    /// [`dispatch_signal_inner`] against `&mut self.first_term` and
    /// `&self.actuator_io`; this method then resolves the action's
    /// outer effect (the SIGHUP arm calls [`Self::dispatch_reload`];
    /// the first SIGINT/SIGTERM arm continues — the driver stays
    /// alive through the actuator's grace and exits via the
    /// actuator-gone signal on `effect_complete_rx`; the
    /// second-within-window arm escalates to `exit_fn` and returns
    /// `Break` for test sanity). Splitting the work this way lets
    /// the inner function take field references that don't conflict
    /// with the `&mut self` the reload arm needs — the borrow checker
    /// is satisfied because the two borrows are disjoint in time.
    ///
    /// **`SignalAction::Shutdown` returns `Continue`.** The first
    /// termination signal pulses `shutdown_actuator_tx` and arms
    /// `first_term` (the IPC-mutating-verb gate; see
    /// [`crate::ipc::protocol::WireErrorCode::ShuttingDown`]) but
    /// does NOT exit the loop. The driver keeps ticking through the
    /// actuator's SIGTERM → grace → SIGKILL → reap-drain phases; the
    /// actuator's eventual exit drops its `WakingSink`'s
    /// `Sender<Input>` clone, which surfaces on the driver's next
    /// `try_recv` as Disconnected → `DrainedTick.actuator_gone =
    /// true` → tick body routes through [`Self::begin_shutdown`].
    /// `SignalPipe` lives inside the Reactor (inside this driver),
    /// so the second-tap hard-exit handshake stays installed across
    /// the entire actuator-grace window — no handshake is orphaned
    /// mid-grace.
    pub(crate) fn dispatch_signal_with_exit_fn<F: FnOnce(i32)>(
        &mut self,
        sig: i32,
        now: Instant,
        exit_fn: F,
    ) -> ControlFlow<()> {
        let action =
            dispatch_signal_inner(sig, now, &mut self.first_term, &self.actuator_io, exit_fn);
        match action {
            SignalAction::None | SignalAction::Shutdown => ControlFlow::Continue(()),
            SignalAction::Reload => {
                // SIGHUP routes through the same shared apply path
                // every other reload source uses. `dispatch_reload`'s
                // own error handling logs and continues; the only way
                // out of `Continue` here is the post-apply `forward`
                // observing shutdown mid-stream — the `Break` is
                // propagated through this method's return so the
                // outer tick can resolve to `Shutdown`.
                self.dispatch_reload(state::ReloadTrigger::Sighup, now)
            }
            SignalAction::HardExit => ControlFlow::Break(()),
        }
    }

    /// Cancel-first shutdown teardown, run once when [`Self::tick`]
    /// resolves to shutdown (operator signal or sensor disconnect).
    ///
    /// The linear `ProbeSlot` Drop tripwire panics if the `Engine` is
    /// dropped (the bin's `drop(driver)`) with a probe still armed,
    /// and a graceful shutdown routinely coincides with one in flight
    /// (settle / verify / rebase / descent). Disarm every owner's slot
    /// and forward the resulting `Cancel`s to the prober — the same
    /// disarm-then-`Cancel` discipline the engine applies to its
    /// internal abandon sites, now at the process boundary. After this
    /// returns the engine holds no armed probe, so dropping it is
    /// silent and [`TickOutcome::Shutdown`] means "drained, safe to
    /// tear down".
    #[must_use]
    fn begin_shutdown(&mut self) -> TickOutcome {
        let out = self.engine.cancel_all_in_flight_probes();
        // INVARIANT: cancel_all_in_flight_probes emits exclusively
        // `ProbeOp::Cancel` ops (see `engine::probe::cancel_owner_probe`
        // — the disarm-then-`Cancel` choke this drain iterates over).
        // `watch_ops` and `effects` are therefore structurally empty,
        // so `forward`'s outbound `crossbeam::select!` arms never
        // execute on this `StepOutput`; the `ControlFlow` return is
        // structurally `Continue`. The cancels dispatch through
        // `forward`'s probe arm directly to the prober (no channel,
        // no shutdown race), so the discard is intentional. A future
        // refactor adding non-probe ops to
        // `cancel_all_in_flight_probes` must thread `Break` here.
        debug_assert!(
            out.watch_ops.is_empty() && out.effects().is_empty(),
            "cancel_all_in_flight_probes must emit only ProbeOp::Cancel",
        );
        let _ = self.forward(out);
        TickOutcome::Shutdown
    }
}

/// Outcome of [`dispatch_signal_inner`]: what side effect the caller
/// owes the engine driver after the inner function's classification
/// + actuator-coord pulse has run.
///
/// The pure inner function returns this discriminator; the wrapping
/// method ([`EngineDriver::dispatch_signal_with_exit_fn`]) then runs
/// the lifecycle effect tied to each arm. This split keeps the inner
/// function pure (only touches `first_term` + `actuator_io`) so unit
/// tests can exercise every branch without constructing a full
/// [`EngineDriver`].
#[derive(Debug, Eq, PartialEq)]
enum SignalAction {
    /// Unknown signal. The dispatch caller returns
    /// [`ControlFlow::Continue`].
    None,
    /// SIGHUP. The dispatch caller runs `dispatch_reload` with
    /// [`state::ReloadTrigger::Sighup`].
    Reload,
    /// First SIGINT/SIGTERM observed (or first after the
    /// [`crate::signals::HARD_EXIT_WINDOW`] expired). The inner
    /// function has already armed `first_term`, logged the
    /// observation, and pulsed `shutdown_actuator_tx`. The dispatch
    /// caller returns [`ControlFlow::Continue`] — the driver stays
    /// alive through the actuator's grace and exits via the
    /// actuator-gone signal on `effect_complete_rx` (see
    /// [`super::reactor::DrainedTick::actuator_gone`]).
    Shutdown,
    /// Second SIGINT/SIGTERM within
    /// [`crate::signals::HARD_EXIT_WINDOW`]. The inner function
    /// pre-empted the actuator's grace, waited on the confirm pulse,
    /// and called `exit_fn(HARD_EXIT_CODE)`. Production's `exit_fn`
    /// does not return; this variant is only reachable from tests
    /// passing a recording closure. The dispatch caller returns
    /// [`ControlFlow::Break`] so the test's outer loop can observe
    /// the hard-exit intent and resolve to [`TickOutcome::Shutdown`].
    HardExit,
}

/// Pure dispatch core for [`EngineDriver::dispatch_signal_with_exit_fn`].
///
/// Separated from the method so the borrow checker can hold
/// `&mut first_term` + `&io` simultaneously without conflicting with
/// the `&mut self` that the SIGHUP arm's outer effect
/// ([`EngineDriver::dispatch_reload`]) needs — the inner function
/// returns its action discriminator, then drops every borrow before
/// the caller takes the `&mut self` it needs for the reload step.
fn dispatch_signal_inner<F: FnOnce(i32)>(
    sig: i32,
    now: Instant,
    first_term: &mut Option<Instant>,
    io: &ActuatorIO,
    exit_fn: F,
) -> SignalAction {
    use crate::signals::{HARD_EXIT_CODE, HARD_EXIT_WINDOW, HARD_SHUTDOWN_CONFIRM_TIMEOUT};
    use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
    match sig {
        SIGHUP => {
            tracing::info!("SIGHUP — config reload");
            SignalAction::Reload
        }
        SIGINT | SIGTERM => {
            if let Some(prev) = *first_term
                && now.duration_since(prev) < HARD_EXIT_WINDOW
            {
                // Synchronous stderr: `exit_fn` typically calls
                // `std::process::exit`, which skips destructors —
                // the tracing-appender's worker thread dies with the
                // process. `eprintln!` lands the line before exit; a
                // `tracing::*` event might be silently dropped on a
                // stalled appender.
                eprintln!(
                    "specter: second termination within {}s — exiting hard",
                    HARD_EXIT_WINDOW.as_secs(),
                );
                // Pre-empt the actuator's SIGTERM grace so it
                // SIGKILLs running children before we abort the
                // process — otherwise stubborn children survive as
                // PID-1 orphans.
                let _ = io.hard_shutdown_actuator_tx.try_send(());
                // Wait for the actuator's "phase 3 SIGKILL fanout
                // complete" pulse. Three terminal paths, all OK:
                //   - `Ok(())` — confirmation received; the kernel
                //     has been told to kill every running child.
                //   - `Err(Disconnected)` — actuator thread already
                //     exited (sender dropped); kernel reap pending,
                //     parent safe to die.
                //   - `Err(Timeout)` — fallback bound for a wedged
                //     actuator; parent dies, kernel reaps on exit.
                let _ = io
                    .hard_shutdown_done_rx
                    .recv_timeout(HARD_SHUTDOWN_CONFIRM_TIMEOUT);
                exit_fn(HARD_EXIT_CODE);
                SignalAction::HardExit
            } else {
                *first_term = Some(now);
                tracing::info!(signal = sig, "termination signal — shutdown initiated");
                let _ = io.shutdown_actuator_tx.try_send(());
                SignalAction::Shutdown
            }
        }
        _ => SignalAction::None,
    }
}

#[cfg(test)]
mod tests;
