//! `App::run` — the bin's lifecycle entry point.
//!
//! Builds the reactor surface, spawns the one surviving worker thread
//! (the actuator), constructs the [`EngineDriver`] on the main thread,
//! runs initial attach, enters the main loop, and runs the shutdown
//! sequence on exit.
//!
//! # Init order
//!
//! Load-bearing across the prologue:
//!
//! 1. **Signal handlers first** —
//!    [`signals::register_signal_handlers`] installs `sa_sigaction`
//!    for SIGHUP / SIGINT / SIGTERM before any other production action
//!    runs. The returned [`signals::SignalPipe`] owns the
//!    signal-pipeline's internal pipe; every signal arriving during
//!    init lands in that pipe and surfaces on the first reactor
//!    tick's `TOKEN_SIGNAL` drain. Without this lift, SIGTERM during
//!    config load would fall through to the kernel default
//!    (immediate process death) and bypass orderly shutdown.
//! 2. **Config + observability** — fail-fast on parse or filter
//!    errors; tracing-subscriber installation here makes every
//!    downstream init step routable through `tracing::*` rather than
//!    `eprintln!`.
//! 3. **Sensor watcher + config watcher** — kernel fd handles that
//!    will register against the reactor's [`mio::Poll`]; constructed
//!    before the [`Reactor`] takes ownership.
//! 4. **IPC socket bind** — `sockpath::bind_socket_atomic` writes the
//!    socket file with the correct permissions; the returned
//!    [`UnlinkGuard`](sockpath::UnlinkGuard) cleans up on graceful
//!    shutdown or panic.
//! 5. **Channels** — [`ActuatorIO::pair`] for the actuator seam, plus
//!    two `unbounded::<Input>()` channels paired with the Reactor's
//!    [`WakeHandle`] for the prober + actuator wake'd-channels.
//! 6. **Reactor construction** —
//!    [`Reactor::new`] consumes the watcher, config watcher,
//!    [`signals::SignalPipe`], and the two channel receivers; returns
//!    the Reactor, a clone of its [`WakeHandle`] for the wake-bearing
//!    senders, and a [`mio::Registry::try_clone()`] handle the
//!    Hub takes to register against the same Poll selector.
//! 7. **Hub construction** — [`Hub::new`] consumes the
//!    bound listener and the Registry clone; registers the listener
//!    against the shared selector and owns the per-conn map for
//!    every accepted client.
//! 8. **Waking senders + worker spawns** —
//!    [`WakingProberResponseSender`] and [`WakingEffectCompleteSender`]
//!    are constructed with [`WakingSink`]s holding [`WakeHandle`]
//!    clones; the prober pool + actuator thread spawn after the
//!    Reactor is built so they hold the one Waker it minted.
//! 9. **Engine driver** — [`EngineDriver::new`] takes ownership of
//!    every preceding piece; runs on the main thread.
//!
//! # Single-threaded reactor
//!
//! The daemon has FOUR threads at idle: driver (main) + actuator +
//! N prober workers + tracing-appender (if file destination). No
//! watcher thread, no config-watcher thread, no signal thread, no
//! IPC server thread, no per-conn worker threads. Every kernel-event
//! source the daemon cares about (watcher fd, config-watcher fd,
//! signal pipe, listener fd, per-conn streams, and the [`mio::Waker`]
//! the worker threads pulse) is registered against ONE [`mio::Poll`]
//! owned by [`Reactor`] — the [`Hub`] registers against the
//! same selector via a [`mio::Registry::try_clone()`] handle. The
//! driver's `tick` body is one drain pass over the partitioned ready
//! set.

use crate::actuator::ActuatorIO;
use crate::driver::{EngineDriver, Hub, Reactor, ReloadTrigger, WakingSink};
use crate::ipc::sockpath;
use crate::loader::Loader;
use crate::observability;
use crate::signals;
use specter_actuator::{EffectCompleteSender, RunWiring, SubprocessActuator, default_spawner};
use specter_config::{Config, DaemonArgs, FileMeta};
use specter_core::{EffectCompletion, Input, SendError};
use specter_engine::Engine;
use specter_sensor::{
    ProbeResponse, ProberResponseSender, WorkerProber, default_config_watcher, default_watcher,
};
use std::io;
use std::num::NonZeroUsize;
use std::process::ExitCode;
use std::sync::Arc;
use std::thread::{self, JoinHandle};
use std::time::Instant;

/// Run the bin against the parsed daemon arguments.
///
/// Loads + validates the config, initializes tracing, builds the
/// reactor + the one surviving worker thread (the actuator), drives
/// the engine to completion, and runs the shutdown sequence. Returns
/// `ExitCode::SUCCESS` on graceful exit; `ExitCode::from(1)` on
/// startup failure (config / watcher / prober / thread spawn /
/// reactor / IPC server / listener bind).
pub fn run(args: DaemonArgs) -> ExitCode {
    // Destructure at the function head — every field is consumed
    // exactly once by the lifecycle below (`config` moves into the
    // driver constructor; `log_path` moves into `CliLogOverrides`;
    // `log_level`, `log_destination`, `concurrency`, `probe_concurrency`,
    // and `no_config_watch` are `Copy`). The bare bindings retire the
    // `args.<field>.clone()` chains the pre-Phase-3.3 shape carried at
    // the driver-constructor and merge_cli sites — the driver consumes
    // `config` directly and the boot-time TOCTOU lstat reads back
    // through [`EngineDriver::config_path`].
    let DaemonArgs {
        config,
        log_level,
        log_destination,
        log_path,
        concurrency,
        probe_concurrency,
        no_config_watch,
    } = args;

    // 1. Register signal handlers BEFORE any other production action.
    //    `SignalPipe::new` installs `sa_sigaction` for HANDLED_SIGNALS
    //    synchronously on construction; any signal arriving in the
    //    rest of init is captured by the signal-pipeline's internal
    //    pipe and surfaces on the first reactor tick's `TOKEN_SIGNAL`
    //    drain. `eprintln!` (not `tracing::error!`): the subscriber
    //    isn't installed yet.
    let signals = match signals::register_signal_handlers() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("specter: signal-hook init failed: {e}");
            return ExitCode::from(1);
        }
    };

    // 2. Load config (fail-fast, pre-tracing). `from_path_with_meta`
    //    captures `FileMeta` atomically with the bytes via a single
    //    `File` handle — closing the startup TOCTOU between the
    //    content read and a separate path-level lstat. The captured
    //    value seeds `loader.config_meta` and is consulted by the
    //    auto-reload settle filter to decide whether a watcher pulse
    //    reflects substantive change.
    let (initial_config, initial_meta) = match Config::from_path_with_meta(&config) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("specter: config load failed:\n{e}");
            return ExitCode::from(1);
        }
    };

    // 3. Tracing — CLI overrides applied on top of `[log]` (cli wins).
    //    `merge_cli` returns a bare `ValidationIssue` (not wrapped in
    //    `ConfigError::Validate`): the issue's own `Display` carries
    //    the field + detail + kind, so we forward it verbatim.
    let log_cfg =
        match initial_config
            .log
            .clone()
            .merge_cli(log_level, log_destination, log_path.as_deref())
        {
            Ok(c) => c,
            Err(issue) => {
                eprintln!("specter: log config invalid: {issue}");
                return ExitCode::from(1);
            }
        };
    // `_obs_guard` holds the file appender's worker thread alive for
    // the entire process lifetime. Drop ordering is load-bearing: if
    // the engine driver owned the guard, every `tracing::*` event
    // between `drop(driver)` and end-of-`run` ("specter exited
    // cleanly", thread join errors) would land on a dropped appender
    // and be silently discarded. Keeping it on `App::run`'s stack
    // frame defers the appender shutdown until after every join
    // completes.
    let (obs_handle, _obs_guard) = match observability::init(&log_cfg) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("specter: observability init failed: {e}");
            return ExitCode::from(1);
        }
    };
    // Emit the same `disabled_*` summary as the config-loaded log so
    // an operator booting Specter with a mostly-disabled config sees
    // *which* entries are suppressed at startup, not just a watch
    // count that omits them.
    let (disabled_watches, disabled_promoters) = initial_config.disabled_names();
    tracing::info!(
        level = ?log_cfg.level,
        destination = ?log_cfg.destination,
        path = ?log_cfg.path.as_ref().map(|p| p.display().to_string()),
        watches = initial_config.watches.len(),
        promoters = initial_config.promoters.len(),
        ?disabled_watches,
        ?disabled_promoters,
        config = %config.display(),
        "specter starting"
    );

    // 4. Bin-side reload state — handed to the engine driver and
    //    mutated only via `Loader::rotate_apply` / `rotate_meta_only`
    //    (the sole-writer claim on `Loader`'s module rustdoc). Fields
    //    are private; `Loader::new` is the one production construction
    //    site that touches them.
    let loader = Loader::new(initial_config, log_cfg, initial_meta);

    // 5. Kqueue on macOS / FreeBSD, inotify on Linux. The watcher
    //    moves into the Reactor below, which registers its fd against
    //    [`mio::Poll`].
    let watcher = match default_watcher() {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(?e, "watcher init failed");
            return ExitCode::from(1);
        }
    };

    // 6. Auto-reload — config-watcher init (default-on; opt-out via
    //    `--no-config-watch` / `SPECTER_NO_CONFIG_WATCH`). A spawn
    //    failure under `--no-config-watch` keeps `None` (no
    //    auto-reload); a spawn failure with auto-reload on logs and
    //    degrades to SIGHUP-only.
    let config_watcher = if no_config_watch {
        tracing::info!("auto-reload disabled via --no-config-watch");
        None
    } else {
        match default_config_watcher(&config) {
            Ok(w) => Some(w),
            Err(e) => {
                tracing::warn!(?e, "config watcher init failed; SIGHUP-only reload");
                None
            }
        }
    };

    // 7. Operator IPC socket — resolve, recover from stale, bind via
    //    atomic-rename + chmod 0600 BEFORE Hub construction
    //    (the Hub takes ownership of the listener fd). The
    //    `unlink_guard` armed here unlinks the socket on graceful
    //    shutdown (via explicit `unlink_now` after the driver drops)
    //    and on panic (Drop runs unconditionally), so the next boot
    //    never trips over our own residue.
    let socket_path = sockpath::default_socket_path();
    if let Err(e) = sockpath::check_stale_or_remove(&socket_path) {
        tracing::error!(
            ?e,
            path = %socket_path.display(),
            "ipc socket path unusable; daemon refusing to start",
        );
        return ExitCode::from(1);
    }
    let (listener, unlink_guard) = match sockpath::bind_socket_atomic(&socket_path) {
        Ok(pair) => pair,
        Err(e) => {
            tracing::error!(
                ?e,
                path = %socket_path.display(),
                "ipc bind_socket_atomic failed",
            );
            return ExitCode::from(1);
        }
    };

    // 8. Channel allocation. Two wake'd channels for the prober pool
    //    and the actuator's completion stream; one bundle-pair for
    //    the actuator's effects + shutdown legs.
    //
    //    The wake'd channels are `unbounded` because the wake edge
    //    is the back-pressure surface — a bounded channel would
    //    require a `try_send` arm in the sender wrappers, which adds
    //    a drop path the engine's recovery contract does not cover
    //    for prober responses (a dropped `ProbeResponse` would leak
    //    a probe correlation; there is no `gate_deadline` for
    //    probes). Practically the queue depth is bounded by
    //    `probe_concurrency` (≤2×num_cpus) + actuator concurrency
    //    (operator-configured) — at default settings, ≤16 entries
    //    queued at once.
    let (actuator_io, actuator_wiring) = ActuatorIO::pair();
    let (prober_tx, prober_rx) = crossbeam::channel::unbounded::<Input>();
    let (effect_complete_tx, effect_complete_rx) = crossbeam::channel::unbounded::<Input>();

    // 9. Build the Reactor first. It mints the Poll, registers every
    //    static fd source (watcher, config-watcher, signal pipe),
    //    mints the [`WakeHandle`] returned for the wake-bearing
    //    senders, and hands back a [`mio::Registry::try_clone()`]
    //    handle the Hub registers its listener and per-conn
    //    streams against. Every static Source is registered against
    //    the Reactor's [`mio::Poll`] before this call returns — a
    //    signal arriving during the rest of init will fire the
    //    reactor's `TOKEN_SIGNAL` on the first poll.
    let (reactor, wake, registry_for_ipc) = match Reactor::new(
        watcher,
        config_watcher,
        signals,
        prober_rx,
        effect_complete_rx,
    ) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(?e, "Reactor construction failed");
            return ExitCode::from(1);
        }
    };

    // 9b. Build the Hub with the Registry clone. The listener
    //     registers against the same underlying selector as the
    //     Reactor's Poll; the per-conn map starts empty and grows
    //     through [`Hub::drain_accept`] as clients connect.
    let ipc = match Hub::new(listener, registry_for_ipc) {
        Ok(p) => p,
        Err(e) => {
            tracing::error!(?e, "Hub construction failed");
            return ExitCode::from(1);
        }
    };

    // 10. Build wrapped senders sharing the Reactor's [`WakeHandle`].
    //     mio mandates one Waker per Poll — both adapters wrap a
    //     [`WakingSink`] built on a `wake.clone()` so the worker
    //     threads pulse the same wake edge. The
    //     [`crate::driver::wake`] module's `WakeHandle::new` is the
    //     sole `mio::Waker::new` site in the bin; constructing
    //     `WakingSink` requires holding a `WakeHandle`, so a future
    //     wake-bearing sink inherits the "one Waker" invariant by
    //     typing rather than convention.
    let prober_sink: Arc<dyn ProberResponseSender> = Arc::new(WakingProberResponseSender(
        WakingSink::new(prober_tx, wake.clone()),
    ));
    let effect_sink: Box<dyn EffectCompleteSender> = Box::new(WakingEffectCompleteSender(
        WakingSink::new(effect_complete_tx, wake),
    ));

    // 11. Spawn the prober pool. The pool takes the `Arc<dyn>`
    //     sender directly and clones it per worker — workers wake
    //     the Reactor on every probe response.
    let probe_concurrency = probe_concurrency.unwrap_or_else(specter_sensor::default_concurrency);
    let prober = match WorkerProber::new(prober_sink, probe_concurrency) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            tracing::error!(?e, "prober init failed");
            return ExitCode::from(1);
        }
    };

    // 12. Spawn the actuator thread. Its `effect_complete_tx`
    //     wrapper wakes the Reactor on every reaped completion.
    let actuator_concurrency = concurrency.unwrap_or_else(specter_actuator::default_concurrency);
    let actuator_handle =
        match spawn_actuator_thread(actuator_concurrency, actuator_wiring, effect_sink) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(?e, "failed to spawn actuator thread");
                return ExitCode::from(1);
            }
        };

    // 13. Engine driver — main thread. `config` and `log_path` move
    //     into the driver here; subsequent reads of the config path
    //     (the boot-time TOCTOU lstat at step #15) go through
    //     [`EngineDriver::config_path`].
    let cli_log_overrides = CliLogOverrides {
        level: log_level,
        destination: log_destination,
        path: log_path,
    };
    let mut driver = EngineDriver::new(
        Engine::new(),
        loader,
        config,
        socket_path,
        cli_log_overrides,
        obs_handle,
        prober.clone(),
        actuator_io,
        ipc,
        reactor,
    );

    // 14. Initial attach against the boot-parsed config.
    //
    //     MUST precede the startup-TOCTOU close (#15):
    //     `dispatch_reload`'s contract is "engine and loader are in
    //     sync; rotate them both forward atomically." Calling it
    //     before initial-attach violates that precondition — the
    //     diff's `added` bucket would attach Subs against an empty
    //     engine, rotate the loader to the post-TOCTOU config, then
    //     `run_initial_attach` would walk the rotated loader and
    //     double-attach those Subs, tripping
    //     `SubRegistry::insert`'s `debug_assert!` (and
    //     `PromoterRegistry::insert`'s equivalent on a TOML edit
    //     that adds a Promoter during the boot window).
    //
    //     On `Break` (downstream `effects_tx` disconnect mid-attach),
    //     `run_initial_attach` internally drains in-flight probes
    //     via `begin_shutdown` before returning — the lifecycle
    //     discipline lives on the method, not on this caller.
    let initial_attach_break = driver.run_initial_attach().is_break();

    // 15. Startup-TOCTOU close — the config can change between the
    //     initial `from_path_with_meta` capture (line 116) and the
    //     config-watcher's registration (the `Reactor::new`
    //     return). A single `lstat` here catches the drift; on
    //     inequality the driver runs `dispatch_reload(Startup)`
    //     directly. Initial-attach has already run (#14), so the
    //     engine is in sync with the loader's pre-rotation state and
    //     the diff's `added` / `removed` / `modified` buckets
    //     dispatch cleanly. Attribution via
    //     [`ReloadTrigger::Startup`] so operators see "boot-time
    //     drift caught and applied" in `status.last_reload_via`.
    //
    //     The lstat-Err branch is "no drift": a missing config at
    //     startup is unusual but recovers via the auto-reload settle
    //     pipeline on the next config-watcher pulse.
    //
    //     The `!initial_attach_break` guard is load-bearing: the
    //     prior call drained probes through `begin_shutdown` on its
    //     `Break` path; re-arming probes mid-shutdown via
    //     `dispatch_reload` would violate the linear-edge invariant.
    //     On a `Break` from `dispatch_reload`'s own apply-branch
    //     `forward`, the inner drain runs symmetrically (see the
    //     [`EngineDriver::dispatch_reload`] rustdoc).
    let drift_break = if !initial_attach_break
        && let Ok(post_meta) = FileMeta::from_path(driver.config_path())
        && post_meta != initial_meta
    {
        tracing::info!("config changed during startup; running reload now");
        driver
            .dispatch_reload(ReloadTrigger::Startup, Instant::now())
            .is_break()
    } else {
        false
    };

    // 16. Run. Skip the main loop on any `Break` observed above —
    //     both `run_initial_attach` and `dispatch_reload` drain
    //     their own in-flight probes through `begin_shutdown` on
    //     `Break`, so the engine is probe-free and the driver is
    //     safe to drop below.
    //
    //     `driver.run()` returns only when (a) the actuator's
    //     `WakingSink` dropped (closing its `Sender<Input>` clone)
    //     and the driver's next tick observed
    //     `DrainedTick.actuator_gone == true`, OR (b) a
    //     second-within-window SIGINT/SIGTERM escalated through
    //     `dispatch_signal_inner`'s HardExit arm (production's
    //     `exit_fn` is `std::process::exit`, which does not return
    //     — the join below is unreachable in that arm). The first
    //     SIGINT no longer breaks the loop; the driver keeps ticking
    //     through the actuator's SIGTERM → grace → SIGKILL →
    //     reap-drain pipeline so `SignalPipe` stays installed for
    //     the entire window (closing the orphan defect the old
    //     "first signal exits loop" shape opened).
    if initial_attach_break || drift_break {
        tracing::info!("shutdown observed during boot; engine drained");
    } else {
        let exit_reason = driver.run();
        tracing::info!(?exit_reason, "engine driver exited");
    }

    // 17. Shutdown sequence. By the time we get here the actuator
    //     thread has already exited (its `WakingSink::Drop` was the
    //     load-bearing edge that unblocked `driver.run()` via
    //     `DrainedTick.actuator_gone`). The drop chain below is
    //     pure cleanup — no further cross-thread coordination.
    //
    //     `drop(driver)` runs the driver's field-order drop:
    //     `actuator_io` (its `effects_tx` clone) → `ipc` (Hub's
    //     Drop deregisters listener + conns from the still-live
    //     Poll selector) → `reactor` (deregisters watcher /
    //     config-watcher / signal pipe; closes the Poll). The
    //     waker drop decrements the `Arc<Waker>` refcount; the
    //     prober's wrapper still holds its clone, but it's inert
    //     once Poll is gone (mio's `wake()` returns Err on a
    //     destroyed Poll, which the wrapper ignores).
    //
    //     `effects_tx` dropping after the actuator already exited
    //     is moot — there's no receiver to disconnect. The Arc
    //     refcount on the `Arc<dyn Prober>` clone the driver held
    //     is released, leaving only this scope's `prober` Arc; the
    //     `try_unwrap` below succeeds.
    drop(driver);

    // Unlink the bound socket path. Safe to do now: the listener fd
    // closed when the Hub dropped (as part of the driver's
    // field-order drop above). An operator reconnecting sees ENOENT,
    // which is structurally correct (the daemon is gone). On panic
    // anywhere between bind and here, `unlink_guard`'s Drop runs and
    // cleans up.
    unlink_guard.unlink_now();

    // Actuator: `join()` is immediate here on every non-panic path.
    // The actuator-gone signal that closed `driver.run()` IS the
    // thread's exit — the closure has already returned by the time
    // we reach this join. On a panic that unwound the closure, the
    // unwind also drops the `Box<dyn EffectCompleteSender>` (which
    // fires the same `WakingSink::Drop`), so the closed-channel
    // edge surfaces identically and `driver.run()` exits the same
    // way. Either way, `join()` returns immediately — clean exits
    // return `Ok(())`; panics return `Err(payload)` so the operator
    // log carries the actual panic text rather than a Debug-formatted
    // `Box<dyn Any>` shell.
    if let Err(payload) = actuator_handle.join() {
        let msg = payload
            .downcast_ref::<&'static str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .unwrap_or("(non-string panic payload)");
        tracing::error!(message = %msg, "actuator thread panicked");
    }

    // Prober: now that the driver is dropped, only `App` holds the
    // Arc. `try_unwrap` succeeds → explicit `shutdown` returns the
    // per-worker join `Vec` so this site fans out the operator-narration
    // log at the right teardown phase. The `Drop` impl on `WorkerProber`
    // is the safety net (boot-fail unwind, panic recovery, leaked-Arc
    // late drop); on the happy path here it fires against an already-
    // drained pool and is a structural no-op.
    match Arc::try_unwrap(prober) {
        Ok(mut p) => {
            for (worker, r) in p.shutdown() {
                if let Err(e) = r {
                    tracing::warn!(worker, ?e, "prober worker join error");
                }
            }
        }
        Err(arc) => {
            // Refcount leak: a clone outlives `drop(driver)` above.
            // Drop our clone; `WorkerProber::drop` runs when the
            // leaked clone eventually drops, joining workers and
            // warn-logging any join failures from there. The kernel
            // is no longer the workers' only reaper.
            tracing::error!(
                refcount = Arc::strong_count(&arc),
                "prober Arc leaked; workers join via WorkerProber::drop on leaked clone teardown",
            );
            drop(arc);
        }
    }

    tracing::info!("specter exited cleanly");
    ExitCode::SUCCESS
}

/// CLI overrides for `[log]`. Captured at startup so SIGHUP-driven
/// reloads can re-apply them on top of the freshly-parsed config (CLI
/// wins, matching the startup precedence).
#[derive(Clone, Debug, Default)]
pub(crate) struct CliLogOverrides {
    pub level: Option<specter_config::LogLevel>,
    pub destination: Option<specter_config::LogDestination>,
    pub path: Option<std::path::PathBuf>,
}

/// Bin's [`ProberResponseSender`] impl — content-lift over a single
/// [`WakingSink`].
///
/// The sensor crate does not name [`Input`] (it would couple worker
/// code to the engine's inbound vocabulary); this newtype is the one
/// place the [`ProbeResponse`] payload lifts into the engine's
/// `Input::ProbeResponse(_)` envelope before crossing the sink's
/// send-then-wake protocol.
///
/// One instance per process — held inside the prober's `Arc<dyn>`
/// so every worker shares the single underlying transport + waker.
/// Drops when the pool's last worker exits (the workers' `Arc<dyn>`
/// clones are the only refs once `App::run` returns).
struct WakingProberResponseSender(WakingSink);

impl ProberResponseSender for WakingProberResponseSender {
    fn send(&self, response: ProbeResponse) -> Result<(), SendError> {
        self.0.send(Input::ProbeResponse(response))
    }
}

/// Bin's [`EffectCompleteSender`] impl — content-lift over a single
/// [`WakingSink`].
///
/// Mirror-shape of [`WakingProberResponseSender`] for the actuator's
/// [`EffectCompletion`] envelope: lifts into
/// `Input::EffectComplete(_)` before crossing the sink's send-then-
/// wake protocol. The two adapters share the same [`WakingSink`]
/// shape — a third wake-bearing sink drops in as a one-line content
/// lift with the same structural guarantees.
///
/// One instance per actuator-thread spawn — boxed into the
/// actuator's `Box<dyn EffectCompleteSender>` constructor argument.
/// Drops when the actuator's `run` returns at shutdown.
struct WakingEffectCompleteSender(WakingSink);

impl EffectCompleteSender for WakingEffectCompleteSender {
    fn send(&self, completion: EffectCompletion) -> Result<(), SendError> {
        self.0.send(Input::EffectComplete(completion))
    }
}

/// Spawn the actuator thread. Constructs [`SubprocessActuator`] with
/// the resolved concurrency, runs the controller blocking until
/// either `wiring.effects_rx` disconnects or `wiring.shutdown_rx`
/// fires.
///
/// `engine_in` is the wake-bearing [`EffectCompleteSender`] —
/// production passes a boxed [`WakingEffectCompleteSender`]
/// constructed at `App::run` wiring time with a clone of the
/// [`WakeHandle`] returned from [`Reactor::new`]. The actuator's
/// controller borrows the sink via `&dyn` so it never names the
/// engine's `Input` vocabulary on its own thread. The closure here
/// owns the `Box<dyn>` for the actuator-thread lifetime and passes
/// `&*engine_in` into `run`; on closure exit the Box drops cleanly.
///
/// `wiring.hard_shutdown_done_tx` is the back-channel the actuator
/// pulses after phase 3 SIGKILL fanout; the driver's hard-exit path
/// (running inside the reactor thread) waits on its paired receiver
/// before `process::exit(130)` so the parent never aborts mid-fanout.
///
/// Returns [`io::Error`] on `thread::Builder::spawn` failure; the
/// caller translates to a startup-fail [`ExitCode`], same shape as
/// every other init path in [`run`].
fn spawn_actuator_thread(
    concurrency: NonZeroUsize,
    wiring: RunWiring,
    engine_in: Box<dyn EffectCompleteSender>,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("specter-actuator".into())
        .spawn(move || {
            let spawner = default_spawner();
            let mut act = SubprocessActuator::new(concurrency);
            // No `catch_unwind` wrapper: on panic, the closure
            // unwinds, `SubprocessActuator::drop` runs the SIGTERM
            // + SIGKILL fanout safety net, and the thread exits
            // with the panic payload intact. The caller's
            // `actuator_handle.join()` becomes the load-bearing
            // observation point — its `Err(payload)` arm extracts
            // the panic message for operator logs (see `run`'s
            // teardown).
            act.run(wiring, &*engine_in, spawner.as_ref());
        })
}

#[cfg(test)]
mod tests {
    use super::{WakingEffectCompleteSender, WakingProberResponseSender};
    use crate::driver::{WakeHandle, WakingSink};
    use mio::{Events, Poll, Token};
    use specter_actuator::EffectCompleteSender;
    use specter_core::{
        DedupKey, EffectCompletion, EffectOutcome, Input, ProbeCorrelation, ProbeOutcome,
        ProbeOwner, ProbeResponse, ProfileId, SubId,
    };
    use specter_sensor::ProberResponseSender;
    use std::time::Duration;

    /// Constructing a [`WakingProberResponseSender`] manually + sending
    /// a [`ProbeResponse`] through it deposits the
    /// `Input::ProbeResponse` on the channel AND fires the paired
    /// `mio::Waker` so a subsequent `Poll::poll` returns immediately
    /// with `TOKEN_WAKER` ready. Pins the send-THEN-wake protocol the
    /// reactor depends on for cross-thread liveness.
    #[test]
    fn waking_prober_response_sender_pulses_waker_after_send() {
        let mut poll = Poll::new().expect("mio Poll");
        let waker_token = Token(0xABC);
        let wake = WakeHandle::new(poll.registry(), waker_token).expect("WakeHandle");
        let (tx, rx) = crossbeam::channel::unbounded::<Input>();
        let sender = WakingProberResponseSender(WakingSink::new(tx, wake));

        // Construct a minimal ProbeResponse. `Vanished` is the
        // narrowest outcome (no payload). The owner/correlation are
        // arbitrary — the wrapper threads the value through verbatim.
        let response = ProbeResponse {
            owner: ProbeOwner::Profile(ProfileId::default()),
            correlation: ProbeCorrelation::from(7),
            outcome: ProbeOutcome::Vanished,
        };
        sender.send(response).expect("send into wake'd channel");

        // The channel must hold the lifted Input.
        match rx.try_recv().expect("Input on channel post-send") {
            Input::ProbeResponse(_) => {}
            other => panic!("expected Input::ProbeResponse, got {other:?}"),
        }

        // The wake edge must be live: a poll with a generous timeout
        // returns immediately with TOKEN_WAKER ready.
        let mut events = Events::with_capacity(4);
        poll.poll(&mut events, Some(Duration::from_secs(2)))
            .expect("poll unblocks via wake");
        let observed: Vec<Token> = events.iter().map(mio::event::Event::token).collect();
        assert!(
            observed.contains(&waker_token),
            "TOKEN_WAKER must fire post-send; got {observed:?}",
        );
    }

    /// Mirror-shape proof for [`WakingEffectCompleteSender`]. The two
    /// adapters share one [`WakeHandle`] in production; this test
    /// constructs a separate handle to keep the two adapters'
    /// contracts independently testable.
    #[test]
    fn waking_effect_complete_sender_pulses_waker_after_send() {
        let mut poll = Poll::new().expect("mio Poll");
        let waker_token = Token(0xDEF);
        let wake = WakeHandle::new(poll.registry(), waker_token).expect("WakeHandle");
        let (tx, rx) = crossbeam::channel::unbounded::<Input>();
        let sender = WakingEffectCompleteSender(WakingSink::new(tx, wake));

        let sub = SubId::default();
        let profile = ProfileId::default();
        sender
            .send(EffectCompletion {
                sub,
                key: DedupKey::Subtree { sub, profile },
                outcome: EffectOutcome::Ok,
            })
            .expect("send into wake'd channel");

        match rx.try_recv().expect("Input on channel post-send") {
            Input::EffectComplete(_) => {}
            other => panic!("expected Input::EffectComplete, got {other:?}"),
        }

        let mut events = Events::with_capacity(4);
        poll.poll(&mut events, Some(Duration::from_secs(2)))
            .expect("poll unblocks via wake");
        let observed: Vec<Token> = events.iter().map(mio::event::Event::token).collect();
        assert!(
            observed.contains(&waker_token),
            "TOKEN_WAKER must fire post-send; got {observed:?}",
        );
    }
}
