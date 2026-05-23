//! `App::run` — the bin's lifecycle entry point.
//!
//! Builds channels, spawns the signal / watcher / actuator threads,
//! constructs the [`EngineDriver`] on the main thread, runs initial
//! attach, enters the main loop, and runs the shutdown sequence
//! on exit.
//!
//! Init order is load-bearing:
//! 1. **Signal handlers first** — [`signals::register_signal_handlers`]
//!    installs `sa_sigaction` for SIGHUP / SIGINT / SIGTERM before any
//!    other production action runs. A signal arriving during the rest
//!    of init is captured by signal-hook's internal pipe and surfaces
//!    on the signal thread's first iteration; without this lift,
//!    SIGTERM during config load would fall through to the kernel
//!    default (immediate process death) and bypass orderly shutdown.
//! 2. **Prober next** — workers must be ready to receive probes before
//!    initial attach emits the first `ProbeOp::Probe`.
//! 3. **Watcher** — `KqueueWatcher::new` must succeed; the wake handle
//!    is captured here.
//! 4. **Actuator** — must be ready before initial attach or the first
//!    tick can emit Effects.
//! 5. **Engine driver** runs on the main thread.

use crate::channels::{Channels, ConfigWatcherSide, IpcServerSide};
use crate::driver::EngineDriver;
use crate::ipc::{server as ipc_server, sockpath};
use crate::loader::Loader;
use crate::observability;
use crate::signals::{register_signal_handlers, spawn_signal_thread};
use crossbeam::channel::bounded;
use specter_actuator::{SubprocessActuator, default_spawner};
use specter_config::{Config, DaemonArgs, FileMeta};
use specter_core::{Input, WatchOp};
use specter_engine::Engine;
use specter_sensor::{
    ConfigWatcher, DrainWindow, FsWatcher, WakeHandle, WatcherEvent, WorkerProber,
    default_config_watcher, default_watcher,
};
use std::io;
use std::num::NonZeroUsize;
use std::ops::ControlFlow;
use std::os::unix::net::UnixListener;
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::Duration;

/// Fixed trailing-latency window for the watcher's deferred-drain pass.
///
/// **Not an inbound-volume lever.** Inbound volume is owned by driver
/// same-tick coalescing (accumulate regime) and per-event engine cost
/// (keeps-up regime); one watcher-side scalar provably cannot serve a
/// per-Profile volume constraint, so this knob no longer scales with
/// `settle`. It is purely the latency budget the watcher trades for
/// batch granularity on its second drain pass — see
/// [`specter_sensor::DrainWindow`] for the deferred-drain semantics.
///
/// `50ms` is the top of the historical `[10ms, 50ms]` band — the value
/// default-`settle` configs already resolved to. The watcher's recency
/// gate skips the second drain for single touches in quiet periods, so
/// a quiet-period edit pays none of this. Fixed, not operator-tunable,
/// under the "minimal config surface" rule.
///
/// Lives next to its sole consumer ([`run`] hands it to
/// [`default_watcher`] at startup) — `Loader` is bin-side reload state
/// and never reads this constant, so colocation with the watcher
/// initialisation is the clearer home.
const WATCHER_DRAIN_WINDOW: Duration = Duration::from_millis(50);

/// Run the bin against the parsed daemon arguments.
///
/// Loads + validates the config, initializes tracing, starts every
/// long-lived thread, drives the engine to completion, and runs the
/// shutdown sequence. Returns `ExitCode::SUCCESS` on graceful
/// exit; `ExitCode::from(1)` on startup failure (config / kqueue /
/// prober / thread spawn).
///
/// `DaemonArgs` is taken by value because every field is consumed
/// (config moves into the driver; concurrency knobs are extracted then
/// dropped); the `needless_pass_by_value` allow documents the intent.
#[allow(clippy::needless_pass_by_value)]
pub fn run(args: DaemonArgs) -> ExitCode {
    // Register signal handlers before any other production action.
    // signal-hook's `sa_sigaction` captures any signal arriving during
    // the rest of init into its internal pipe (owned by the returned
    // `Signals`); the signal thread drains the pipe when it starts.
    // SIGTERM during config load is now bounded by "queued until
    // signal thread runs" rather than "kernel-default kill" — see the
    // module rustdoc for the init-order rationale. `eprintln!` not
    // `tracing::error!`: the subscriber isn't installed yet.
    let signals = match register_signal_handlers() {
        Ok(s) => s,
        Err(e) => {
            eprintln!("specter: signal-hook init failed: {e}");
            return ExitCode::from(1);
        }
    };

    // Load config (fail-fast, pre-tracing). `from_path_with_meta`
    // captures `FileMeta` atomically with the bytes via a single `File`
    // handle — closing the startup TOCTOU between the content read
    // and a separate path-level lstat. The captured value seeds
    // `loader.config_meta` and is consulted by the auto-reload settle
    // filter to decide whether a watcher pulse reflects substantive
    // change.
    let (initial_config, initial_meta) = match Config::from_path_with_meta(&args.config) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("specter: config load failed:\n{e}");
            return ExitCode::from(1);
        }
    };

    // Tracing — CLI overrides applied on top of `[log]` (cli wins).
    // `merge_cli` returns a bare `ValidationIssue` (not wrapped in
    // `ConfigError::Validate`): the issue's own `Display` carries the
    // field + detail + kind, so we forward it verbatim.
    let log_cfg = match initial_config.log.clone().merge_cli(
        args.log_level,
        args.log_destination,
        args.log_path.clone(),
    ) {
        Ok(c) => c,
        Err(issue) => {
            eprintln!("specter: log config invalid: {issue}");
            return ExitCode::from(1);
        }
    };
    // `_obs_guard` holds the file appender's worker thread alive for the
    // entire process lifetime. Drop ordering is load-bearing: if the
    // engine driver owned the guard, every `tracing::*` event between
    // `drop(driver)` and end-of-`run` ("specter exited cleanly", thread
    // join errors) would land on a dropped appender and be silently
    // discarded. Keeping it on `App::run`'s stack frame defers the
    // appender shutdown until after every join completes.
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
        config = %args.config.display(),
        "specter starting"
    );

    // Bin-side reload state — handed to the engine driver and mutated
    // only via `Loader::rotate_apply` / `Loader::rotate_meta_only`
    // (the sole-writer claim on `Loader`'s module rustdoc). The
    // struct-literal construction here is the one production site
    // outside those rotation methods that touches the fields.
    let loader = Loader {
        current_config: initial_config,
        current_log: log_cfg,
        config_meta: initial_meta,
    };

    // Kqueue (or Linux inotify, when that backend lands) + wake handle.
    let watcher = match default_watcher(DrainWindow::new(WATCHER_DRAIN_WINDOW)) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(?e, "watcher init failed");
            return ExitCode::from(1);
        }
    };
    let wake_handle: Box<dyn WakeHandle> = watcher.wake_handle();

    // Channels — allocates every unconditional pair into per-thread
    // bundles. Each bundle below partial-moves into its consumer; no
    // dispenser remainder survives the spawn sequence.
    let chans = Channels::new();

    // Shutdown coordination. Constructed before the prober so workers
    // can capture the flag at spawn time; the signal thread, the
    // watcher / config-watcher loops, and the bin's shutdown sequence
    // all clone it in below.
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    // Prober (workers spawn inside `WorkerProber::new`). The constructor
    // borrows the watcher bundle's `sensor_in_tx` and clones it once
    // per worker internally; the borrow ends here, leaving the bundle
    // free to move into the watcher thread below.
    let probe_concurrency = args
        .probe_concurrency
        .map_or(specter_sensor::DEFAULT_CONCURRENCY, NonZeroUsize::get);
    let prober = match WorkerProber::new(
        &chans.watcher.sensor_in_tx,
        probe_concurrency,
        &shutdown_flag,
    ) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            tracing::error!(?e, "prober init failed");
            return ExitCode::from(1);
        }
    };

    // Capture the startup-TOCTOU sender clone before `chans.signal`
    // moves into the signal thread. Field-level `Sender::clone` —
    // `SignalSide` itself does not impl `Clone`. The clone is
    // released below once the auto-reload branch has had its chance
    // to fire the startup pulse, so the channel's sender refcount
    // reflects the steady-state graph for the rest of the process.
    let toctou_reload_tx = chans.signal.reload_signal_tx.clone();

    // Signal thread (drains the pre-registered signal queue). The
    // `Signals` constructed at the top of `run` moves in here; any
    // signal that arrived during init is already queued in
    // signal-hook's internal pipe and surfaces on the first
    // `signals.forever()` iteration.
    let signal_handle = match spawn_signal_thread(
        signals,
        chans.signal,
        Arc::clone(&shutdown_flag),
        wake_handle.clone(),
    ) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!(?e, "failed to spawn signal thread");
            return ExitCode::from(1);
        }
    };

    // Watcher thread.
    let watcher_handle =
        match spawn_watcher_thread(watcher, chans.watcher, Arc::clone(&shutdown_flag)) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(?e, "failed to spawn watcher thread");
                return ExitCode::from(1);
            }
        };

    // Actuator thread.
    let actuator_concurrency = args
        .concurrency
        .map_or(specter_actuator::DEFAULT_CONCURRENCY, NonZeroUsize::get);
    let actuator_handle = match spawn_actuator_thread(actuator_concurrency, chans.actuator) {
        Ok(h) => h,
        Err(e) => {
            tracing::error!(?e, "failed to spawn actuator thread");
            return ExitCode::from(1);
        }
    };

    // Engine driver — main thread.
    let config_path = args.config;
    let cli_log_overrides = CliLogOverrides {
        level: args.log_level,
        destination: args.log_destination,
        path: args.log_path,
    };

    // Operator IPC socket — resolve, recover from stale, bind via
    // atomic-rename + chmod 0600 BEFORE the engine driver constructs
    // (the driver records the bound path on `DriverState` for the
    // `status` projection). The `unlink_guard` armed here unlinks
    // the socket on graceful shutdown (via explicit `disarm` after
    // ipc-thread join) and on panic (Drop runs unconditionally), so
    // the next boot never trips over our own residue.
    //
    // Path resolution today reads the per-platform default
    // (`XDG_RUNTIME_DIR/specter.sock` on Linux, `$TMPDIR/specter.sock`
    // on macOS/BSD); a `--socket` flag + `[control] socket` config
    // block can plug in here without further changes downstream.
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

    // Auto-reload — config-watcher init (default-on; opt-out via
    // `--no-config-watch` / `SPECTER_NO_CONFIG_WATCH`).
    //
    // The `config_event` channel pair is allocated inline below
    // *only* when the config watcher thread will spawn. Under
    // `--no-config-watch` (or a watcher init failure) the engine
    // bundle's `config_event_rx` is `None`, and the driver's tick
    // skips both the drain and the `Select` arm — crossbeam can't
    // report a non-existent receiver as ready, so no keepalive
    // sender on the stack is required.
    //
    // **Startup-TOCTOU close** (when the watcher initialises). The
    // bin's `Config::from_path_with_meta` captured `(initial_config,
    // initial_meta)` atomically — the bound `File` handle pinned
    // the inode for both bytes and meta. Between that capture and
    // `default_config_watcher` constructing the kqueue / inotify
    // registration, the operator can land an edit (`vim`'s
    // atomic-save flow, or an in-place `echo > file`). The watcher
    // would observe the pre-edit state (or no events at all if the
    // edit completed before registration). A single
    // `FileMeta::from_path` lstat after watcher init catches the
    // race: if the on-disk identity differs from `initial_meta`,
    // queue a SIGHUP-style pulse on the captured `toctou_reload_tx`
    // clone so the driver's first tick handles it immediately (no
    // settle delay, unlike steady-state pulses). The driver's
    // `handle_reload` re-reads the file with a fresh atomic capture
    // and rotates `loader.config_meta` to the post-edit identity.
    let (config_event_rx, config_watcher_handles) = if args.no_config_watch {
        tracing::info!("auto-reload disabled via --no-config-watch");
        (None, None)
    } else {
        match default_config_watcher(&config_path) {
            Ok(watcher) => {
                let (config_event_tx, config_event_rx) = bounded::<()>(1);
                match FileMeta::from_path(&config_path) {
                    Ok(post_init_meta) if post_init_meta != initial_meta => {
                        // Meta inequality is the *detection* fact; the
                        // `try_send` outcome is the *delivery* fact. Log
                        // the detection unconditionally — `Err(Full)`
                        // means an operator-issued SIGHUP raced the
                        // lstat between `spawn_signal_thread` above and
                        // here, and the queued pulse will handle this
                        // same drift; silently dropping the *pulse* is
                        // correct (one reload covers both drifts),
                        // silently dropping the *log* is the bug.
                        // `FileMeta` covers mtime / size / mode, so an
                        // in-place `echo > file` is caught alongside
                        // the atomic-save (write-tmp → rename) flow.
                        tracing::info!("config changed during startup; reload queued");
                        let _ = toctou_reload_tx.try_send(());
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(
                        ?e,
                        "post-init config lstat failed; skipping startup-TOCTOU pulse"
                    ),
                }
                let cw_wake = watcher.wake_handle();
                let cw_handle = match spawn_config_watcher_thread(
                    watcher,
                    ConfigWatcherSide { config_event_tx },
                    Arc::clone(&shutdown_flag),
                ) {
                    Ok(h) => h,
                    Err(e) => {
                        tracing::error!(?e, "failed to spawn config-watcher thread");
                        return ExitCode::from(1);
                    }
                };
                (Some(config_event_rx), Some((cw_handle, cw_wake)))
            }
            Err(e) => {
                tracing::warn!(?e, "config watcher init failed; SIGHUP-only reload");
                (None, None)
            }
        }
    };
    // Startup race window closed — release the TOCTOU clone so the
    // reload-signal channel's sender refcount tracks the steady-state
    // graph (signal thread + the driver's own clones, no stragglers).
    drop(toctou_reload_tx);

    let mut driver = EngineDriver::new(
        Engine::new(),
        loader,
        config_path,
        socket_path,
        cli_log_overrides,
        obs_handle,
        chans.engine.finalize(config_event_rx),
        prober.clone(),
        wake_handle.clone(),
    );
    // `chans.engine` and `chans.signal` are now consumed; `chans.watcher`
    // / `chans.actuator` moved into their threads above; `chans.ipc_server`
    // moves into the IPC server thread below. After that, `chans` is
    // fully partial-moved and drops silently at end-of-scope.

    // IPC server handle. `None` ⇒ either initial-attach observed
    // shutdown before we got here (we never spawned), or the spawn
    // call itself failed (rare; `EAGAIN` under process-wide thread
    // pressure). Either way the shutdown sequence below treats the
    // `Option` uniformly.
    let mut ipc_handle: Option<JoinHandle<()>> = None;
    // Wrap the bound listener so the failure path can decide whether
    // to spawn it into the server thread or drop it (closing the fd
    // immediately; `unlink_guard` then removes the path on graceful
    // shutdown or panic).
    let mut listener_slot: Option<UnixListener> = Some(listener);

    if driver.run_initial_attach().is_break() {
        // Shutdown observed during initial attach (operator signal
        // mid-startup or a downstream channel disconnect). The driver
        // has already drained its in-flight probes via
        // `begin_shutdown`, so dropping it below is safe — skip the
        // main loop and route directly to the shared teardown. We
        // never spawn the IPC server, so `ipc_handle` stays `None`
        // and the listener_slot's `UnixListener` drops at end-of-run.
        tracing::info!("shutdown observed during initial attach; engine drained");
    } else {
        // IPC server thread — spawned AFTER `run_initial_attach` so
        // the engine is in steady state before the first `status`
        // request lands (otherwise the projection would lie). The
        // `chans.ipc_server` bundle's `ipc_request_tx` moves in here;
        // clones per accepted client come off it inside
        // `ipc_server_run`.
        let listener = listener_slot
            .take()
            .expect("listener_slot is Some at this branch");
        match spawn_ipc_server_thread(listener, chans.ipc_server, Arc::clone(&shutdown_flag)) {
            Ok(h) => ipc_handle = Some(h),
            Err(e) => {
                // Spawn failure leaves the listener consumed (it
                // moved into the closure on a successful spawn; on
                // failure the closure was dropped, taking the
                // listener with it — kernel reclaims the fd
                // immediately). The control surface is partial; we
                // refuse to enter the main loop and fall through to
                // the shared shutdown so worker threads exit
                // cleanly.
                tracing::error!(?e, "failed to spawn ipc server thread");
            }
        }
        if ipc_handle.is_some() {
            let exit_reason = driver.run();
            tracing::info!(?exit_reason, "engine driver exited");
        }
    }

    // Shutdown sequence — broadcast intent before tearing the driver
    // down, so every consumer of `shutdown_flag` observes `true`
    // synchronously with the channel disconnects that drive its exit.
    // The wake handles held by `App` are still the linchpin for the
    // watchers' blocking syscalls; the flag is the load-bearing hint
    // for the prober workers' `out.send`-failure log severity.
    //
    // Order is load-bearing on two edges:
    //
    // 1. **Flag before `drop(driver)`.** `drop(driver)` releases
    //    `sensor_in_rx`; the next `out.send` from a worker mid-probe
    //    fails synchronously. The worker reads `shutdown_flag` on
    //    that path to discriminate clean teardown (`debug!`) from
    //    mid-runtime engine loss (`warn!`). Publishing the flag
    //    first means the channel-internal acquire on the worker side
    //    observes a flag already set to `true`.
    //
    // 2. **Flag before wake.** The watcher / config-watcher loops
    //    check `shutdown_flag` at the *top* of their bodies. A wake
    //    before the store would race the loop's flag read against
    //    the `wait`-return path; flag-first guarantees the next
    //    iteration sees `true` and exits cleanly.
    shutdown_flag.store(true, Ordering::SeqCst);
    drop(driver); // releases driver's clones (engine, prober, wake_handle, txs).
    // Drop any unused listener (initial-attach-break path) before the
    // wake fan-out: this closes the bound fd, but the socket file on
    // disk persists until `unlink_guard` drops (also at end of `run`).
    drop(listener_slot);
    wake_handle.wake();
    if let Some((_, ref cw_wake)) = config_watcher_handles {
        cw_wake.wake();
    }

    // Join the IPC server thread first — its accept loop exits on
    // `shutdown_flag.load(true)`; per-conn worker threads are
    // detached and the OS reaps them on process exit. The accept
    // loop typically returns within `ACCEPT_IDLE_SLEEP` of the flag
    // store above.
    if let Some(h) = ipc_handle
        && let Err(e) = h.join()
    {
        tracing::error!(?e, "ipc server thread panicked");
    }
    // Surrender unlink responsibility *after* the IPC server thread
    // has joined: no surviving thread holds the listener fd. An
    // operator who reconnects after this point sees ENOENT, which
    // is structurally correct (the daemon is gone). On panic
    // anywhere between bind and here, `unlink_guard`'s Drop runs
    // and cleans up.
    unlink_guard.disarm();

    if let Err(e) = watcher_handle.join() {
        tracing::error!(?e, "watcher thread panicked");
    }
    if let Err(e) = actuator_handle.join() {
        tracing::error!(?e, "actuator thread panicked");
    }
    if let Some((cw_handle, _)) = config_watcher_handles
        && let Err(e) = cw_handle.join()
    {
        tracing::error!(?e, "config-watcher thread panicked");
    }

    // Prober: now that engine driver is dropped, only `App` holds the
    // Arc. `try_unwrap` succeeds → `shutdown` joins workers.
    match Arc::try_unwrap(prober) {
        Ok(p) => {
            for (worker, r) in p.shutdown() {
                if let Err(e) = r {
                    tracing::warn!(worker, ?e, "prober worker join error");
                }
            }
        }
        Err(arc) => {
            tracing::error!(
                refcount = Arc::strong_count(&arc),
                "prober Arc leaked; abandoning workers (kernel reaps on process exit)",
            );
            // Best-effort: drop our Arc clone so the workers exit on
            // queue disconnect when the leaked clone eventually drops
            // (or, in pathological cases, on process exit).
            drop(arc);
        }
    }

    // Signal thread is daemon-style; signal-hook's static handlers don't
    // expose a programmatic teardown that doesn't race in-flight signals.
    // The thread will be reaped by the OS on process exit. We don't join
    // — joining would block forever waiting for `signals.forever()` to
    // return, which only happens via `process::exit(130)` (the hard-exit
    // path).
    drop(signal_handle);

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

/// Spawn the watcher thread. Owns the [`DefaultWatcher`] for its
/// lifetime; drop closes the underlying fd(s) on exit.
///
/// Returns the [`JoinHandle`] on success and the underlying
/// [`io::Error`] on `thread::Builder::spawn` failure (typically
/// `EAGAIN` under process-wide thread-limit pressure). The caller
/// translates the error to a startup-fail [`ExitCode`], mirroring the
/// uniform "startup failure → exit 1" contract every other init path
/// in [`run`] honours.
fn spawn_watcher_thread(
    mut watcher: specter_sensor::DefaultWatcher,
    sides: crate::channels::WatcherSide,
    shutdown_flag: Arc<AtomicBool>,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("specter-watcher".into())
        .spawn(move || {
            // Wrap the loop in catch_unwind so a watcher-side panic
            // doesn't propagate into the bin's process abort.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                watcher_loop(&mut watcher, &sides, &shutdown_flag);
            }));
            if let Err(payload) = result {
                tracing::error!(
                    "watcher thread panicked; payload size = {}",
                    std::mem::size_of_val(&payload),
                );
            }
            // `watcher` drops here, closing the kqueue fd.
        })
}

/// Watcher event loop body. Generic over the watcher type so sibling
/// tests can drive it with `MockFsWatcher` without spinning up kqueue.
///
/// `clippy::iter_with_drain` allow: `events.drain(..)` is the canonical
/// way to consume a `Vec` while preserving its allocation. `into_iter()`
/// would drop the buffer between poll iterations and force a fresh
/// allocation per drain — defeating the `Vec::with_capacity(64)` we
/// initialise with.
#[allow(clippy::iter_with_drain)]
pub(crate) fn watcher_loop<W: FsWatcher>(
    watcher: &mut W,
    sides: &crate::channels::WatcherSide,
    shutdown_flag: &AtomicBool,
) {
    let mut events: Vec<WatcherEvent> = Vec::with_capacity(64);
    loop {
        // Apply pending watch ops first.
        loop {
            match sides.watch_ops_rx.try_recv() {
                Ok(op) => {
                    if apply_watch_op(watcher, op, &sides.sensor_in_tx).is_break() {
                        return;
                    }
                }
                Err(crossbeam::channel::TryRecvError::Empty) => break,
                Err(crossbeam::channel::TryRecvError::Disconnected) => return,
            }
        }
        if shutdown_flag.load(Ordering::SeqCst) {
            return;
        }
        events.clear();
        match watcher.poll_until(None, &mut events) {
            Ok(_) => {
                // Drain the buffer in place so the next iteration's
                // `events.clear()` reuses the same allocation.
                for ev in events.drain(..) {
                    match ev {
                        WatcherEvent::Fs { resource, event } => {
                            // Engine inbound gone ⇒ stop the watcher rather
                            // than spin forever sending into the void. The
                            // same discipline the `watch_ops_rx`
                            // `Disconnected` arm applies; `sensor_in` is
                            // unbounded, so a send error is unambiguously
                            // "engine dead", never back-pressure.
                            if sides
                                .sensor_in_tx
                                .send(Input::FsEvent { resource, event })
                                .is_err()
                            {
                                return;
                            }
                        }
                        WatcherEvent::Overflow { scope } => {
                            // inotify's `IN_Q_OVERFLOW` lifts here on
                            // Linux; kqueue never emits Overflow
                            // (`EV_CLEAR` coalesces but never silently
                            // drops). The engine's `on_sensor_overflow`
                            // handler reseeds every in-scope Profile
                            // and emits `Diagnostic::SensorOverflow`.
                            // Engine-gone ⇒ stop, as in the `Fs` arm
                            // above.
                            if sides
                                .sensor_in_tx
                                .send(Input::SensorOverflow { scope })
                                .is_err()
                            {
                                return;
                            }
                        }
                    }
                }
            }
            Err(failure) => {
                tracing::error!(?failure, "watcher poll error; thread exiting");
                return;
            }
        }
    }
}

/// Spawn the config-watcher thread. Owns the platform's
/// [`specter_sensor::DefaultConfigWatcher`] for its lifetime; drop
/// closes the underlying fd(s) on exit.
///
/// Mirrors [`spawn_watcher_thread`]'s discipline: the loop body sits
/// in a free function so sibling tests can drive it with any
/// [`ConfigWatcher`] implementation, and a `catch_unwind` around the
/// loop body localises a watcher-side panic to "thread exits;
/// SIGHUP-only continues" — the rest of the bin keeps running.
///
/// Returns [`io::Error`] on `thread::Builder::spawn` failure; the
/// caller translates to a startup-fail [`ExitCode`], same shape as
/// every other init path in [`run`].
fn spawn_config_watcher_thread(
    mut watcher: specter_sensor::DefaultConfigWatcher,
    sides: crate::channels::ConfigWatcherSide,
    shutdown_flag: Arc<AtomicBool>,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("specter-config-watcher".into())
        .spawn(move || {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                config_watcher_loop(&mut watcher, &sides, &shutdown_flag);
            }));
            if let Err(payload) = result {
                tracing::error!(
                    "config-watcher thread panicked; payload size = {}",
                    std::mem::size_of_val(&payload),
                );
            }
            // `watcher` drops here, closing the kqueue / inotify fd(s).
        })
}

/// Config-watcher event loop. Generic over the watcher type so sibling
/// tests can drive it with a stub implementation without touching real
/// kernel resources.
///
/// Loop semantics mirror the trait's rustdoc:
/// - **Top-of-loop shutdown check.** Read `shutdown_flag` *before*
///   blocking; a flag-set + wake from the bin's shutdown sequence
///   guarantees the next `wait` returns `Ok(false)` (wake) and the
///   subsequent iteration exits.
/// - **`Ok(true)`** — kernel observed an event for the config file or
///   its parent dir. Try-send a single `()` pulse on
///   [`crate::channels::ConfigWatcherSide::config_event_tx`]. The
///   channel is `bounded(1)`; sustained editor bursts coalesce at
///   the kernel-queue layer, and the driver's `config_settle_until`
///   does the time-based debounce. A `Full` rejection on `try_send`
///   is the desired no-op (a pulse is already queued).
/// - **`Ok(false)`** — wake or deadline. Production passes
///   `wait(None)` so deadline never fires; this branch is purely
///   shutdown-driven. Falls through to the next iteration's flag
///   check.
/// - **`Err(e)`** — syscall error. `error!`-log and exit; SIGHUP-only
///   reload continues to work via the existing signal pipeline.
pub(crate) fn config_watcher_loop<W: ConfigWatcher>(
    watcher: &mut W,
    sides: &crate::channels::ConfigWatcherSide,
    shutdown_flag: &AtomicBool,
) {
    loop {
        if shutdown_flag.load(Ordering::SeqCst) {
            return;
        }
        match watcher.wait(None) {
            Ok(true) => {
                // `try_send` Full is the desired coalescing path; the
                // driver's settle window debounces irrespective of how
                // many pulses fired against the bounded(1) slot.
                let _ = sides.config_event_tx.try_send(());
            }
            Ok(false) => {
                // Wake or deadline; production never sets a deadline
                // so the only way here is a wake from the bin's
                // shutdown path. Fall through to re-check the flag.
            }
            Err(e) => {
                tracing::error!(
                    ?e,
                    "config-watcher syscall failed; thread exiting (SIGHUP still works)"
                );
                return;
            }
        }
    }
}

/// Apply one [`WatchOp`] to the watcher, packaging failures as
/// [`Input::WatchOpRejected`] back to the engine. Generic over the
/// watcher type for the same testability reason as
/// [`watcher_loop`].
///
/// Takes `op` by value to move `WatchOp::Watch`'s `path` out for the
/// borrowing `watcher.watch` call without cloning; the rejection
/// payload carries only `resource` + `failure` (the engine demuxes on
/// the typed `failure`, never the rejected op's shape).
///
/// Returns [`ControlFlow::Break`] iff the rejection send observed the
/// engine inbound channel disconnected (the sole `sensor_in_rx` holder
/// gone). `sensor_in` is unbounded, so a send error is unambiguously
/// "engine dead" — the caller must stop the watcher rather than spin,
/// the same discipline the `watch_ops_rx` `Disconnected` arm applies.
/// (`ControlFlow` is itself `#[must_use]` — the caller cannot drop it
/// silently.)
pub(crate) fn apply_watch_op<W: FsWatcher>(
    watcher: &mut W,
    op: WatchOp,
    sensor_in_tx: &crossbeam::channel::Sender<Input>,
) -> ControlFlow<()> {
    match op {
        WatchOp::Watch {
            resource,
            path,
            kind,
            events,
        } => {
            if let Err(failure) = watcher.watch(resource, &path, kind, events) {
                return match sensor_in_tx.send(Input::WatchOpRejected { resource, failure }) {
                    Ok(()) => ControlFlow::Continue(()),
                    Err(_) => ControlFlow::Break(()),
                };
            }
        }
        WatchOp::Unwatch { resource } => watcher.unwatch(resource),
    }
    ControlFlow::Continue(())
}

/// Spawn the IPC server thread. Owns the bound [`UnixListener`] for
/// its lifetime; drop closes the bound fd on exit. Per-connection
/// worker threads are detached inside the accept loop — they observe
/// shutdown via their own `shutdown_flag` clone and the per-conn
/// write timeout, never via this `JoinHandle`.
///
/// Mirrors [`spawn_watcher_thread`]'s discipline: the loop body sits
/// inside a `catch_unwind` so a server-side panic doesn't propagate
/// into the bin's process abort path.
///
/// Returns [`io::Error`] on `thread::Builder::spawn` failure; the
/// caller falls through to the standard shutdown sequence so worker
/// threads exit cleanly.
fn spawn_ipc_server_thread(
    listener: UnixListener,
    side: IpcServerSide,
    shutdown_flag: Arc<AtomicBool>,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("specter-ipc-server".into())
        .spawn(move || {
            // `move ||` on the inner closure transfers ownership of
            // `listener`, `side`, and `shutdown_flag` into
            // `ipc_server_run` — the thread-entry shape (by-value
            // arguments) is what the function expects, and the
            // `AssertUnwindSafe` wrapper bridges the missing
            // `UnwindSafe` impls those types carry by default.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(move || {
                ipc_server::ipc_server_run(listener, side, shutdown_flag);
            }));
            if let Err(payload) = result {
                tracing::error!(
                    "ipc server thread panicked; payload size = {}",
                    std::mem::size_of_val(&payload),
                );
            }
            // `listener` drops inside `ipc_server_run` on return,
            // closing the bound fd.
        })
}

/// Spawn the actuator thread. Constructs [`SubprocessActuator`] with
/// the resolved concurrency, runs the controller blocking until either
/// `effects_rx` disconnects or `shutdown_actuator_rx` fires.
///
/// `sides.hard_shutdown_done_tx` is the back-channel the actuator
/// pulses after phase 3 SIGKILL fanout; the signal thread waits on
/// its paired receiver before `process::exit(130)` so the parent
/// never aborts mid-fanout.
///
/// Returns [`io::Error`] on `thread::Builder::spawn` failure; the
/// caller translates to a startup-fail [`ExitCode`], same shape as
/// every other init path in [`run`].
fn spawn_actuator_thread(
    concurrency: usize,
    sides: crate::channels::ActuatorSide,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("specter-actuator".into())
        .spawn(move || {
            let spawner = default_spawner();
            let mut act = SubprocessActuator::new(concurrency);
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                act.run(
                    sides.effects_rx,
                    sides.shutdown_actuator_rx,
                    sides.hard_shutdown_actuator_rx,
                    sides.effect_in_tx,
                    spawner.as_ref(),
                    sides.hard_shutdown_done_tx,
                );
            }));
            if let Err(payload) = result {
                tracing::error!(
                    "actuator thread panicked; payload size = {}",
                    std::mem::size_of_val(&payload),
                );
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::{Channels, ConfigWatcherSide};
    use crossbeam::channel::bounded;
    use slotmap::SlotMap;
    use specter_core::{ClassSet, ResourceId, ResourceKind, WatchFailure};
    use specter_sensor::testkit::MockFsWatcher;

    /// Mint a fresh non-null `ResourceId`. Required because slotmap's
    /// `SecondaryMap` rejects the null/default key — and `MockFsWatcher`
    /// stores its installed-watch state in a `SecondaryMap`.
    fn fresh_resource_id() -> ResourceId {
        let mut map: SlotMap<ResourceId, ()> = SlotMap::with_key();
        map.insert(())
    }

    #[test]
    fn apply_watch_op_watch_calls_watcher() {
        let chans = Channels::new();
        let sides = chans.watcher;
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        // Connected channel: the return is Continue; these tests assert
        // the watcher-side effect, not the disconnect signal.
        let _ = apply_watch_op(
            &mut watcher,
            WatchOp::Watch {
                resource: r,
                path: Arc::from(std::path::Path::new("/tmp")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            &sides.sensor_in_tx,
        );
        assert!(watcher.installed.contains_key(r));
    }

    #[test]
    fn apply_watch_op_watch_failure_emits_rejection() {
        let chans = Channels::new();
        let watcher_sides = chans.watcher;
        let engine_side = chans.engine.finalize(None);
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        // EMFILE = 24 on macOS / FreeBSD / Linux. Hard-coded so the
        // bin's tests don't pull `libc` as a direct dev-dep.
        watcher.fail_next_watch(WatchFailure::Pressure { errno: 24 });
        // Connected channel: the return is Continue; these tests assert
        // the watcher-side effect, not the disconnect signal.
        let _ = apply_watch_op(
            &mut watcher,
            WatchOp::Watch {
                resource: r,
                path: Arc::from(std::path::Path::new("/tmp")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            &watcher_sides.sensor_in_tx,
        );
        // Reject delivered to engine inbound.
        let recv = engine_side.sensor_in_rx.try_recv().expect("rejection sent");
        match recv {
            Input::WatchOpRejected { failure, .. } => {
                assert_eq!(failure, WatchFailure::Pressure { errno: 24 });
            }
            other => panic!("expected WatchOpRejected, got {other:?}"),
        }
    }

    #[test]
    fn apply_watch_op_unwatch_clears_state() {
        let chans = Channels::new();
        let sides = chans.watcher;
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        watcher
            .watch(
                r,
                std::path::Path::new("/tmp"),
                ResourceKind::Unknown,
                ClassSet::EMPTY,
            )
            .unwrap();
        // Connected channel: the return is Continue; these tests assert
        // the watcher-side effect, not the disconnect signal.
        let _ = apply_watch_op(
            &mut watcher,
            WatchOp::Unwatch { resource: r },
            &sides.sensor_in_tx,
        );
        assert!(!watcher.installed.contains_key(r));
    }

    #[test]
    fn watcher_loop_drains_ops_and_exits_on_shutdown_flag() {
        let chans = Channels::new();
        // Stage a watch op into the bounded channel before the watcher
        // bundle moves; the loop drains it on the first iteration via
        // `watch_ops_rx.try_recv` Empty.
        let watch_ops_tx = chans.engine.watch_ops_tx.clone();
        let sides = chans.watcher;
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        watch_ops_tx
            .try_send(WatchOp::Watch {
                resource: r,
                path: Arc::from(std::path::Path::new("/tmp")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            })
            .unwrap();
        watcher.inject(r, specter_core::FsEvent::Modified);
        let flag = Arc::new(AtomicBool::new(true));
        watcher_loop(&mut watcher, &sides, &flag);
        // The op was applied (state changed); the event was queued but
        // poll_until on a flag=true loop returns immediately on the next
        // iteration without polling. Either way, no panic; clean exit.
        assert!(watcher.installed.contains_key(r));
    }

    #[test]
    fn watcher_loop_exits_on_disconnect() {
        let chans = Channels::new();
        let sides = chans.watcher;
        let mut watcher = MockFsWatcher::new();
        // Drop the engine bundle (sole holder of `watch_ops_tx`) — the
        // loop's try_recv returns Disconnected and the loop exits. The
        // signal / actuator bundles also drop here, releasing their
        // half of the unconditional channels.
        drop(chans.engine);
        drop(chans.actuator);
        drop(chans.signal);
        let flag = Arc::new(AtomicBool::new(false));
        watcher_loop(&mut watcher, &sides, &flag);
    }

    /// Stub `ConfigWatcher` that returns scripted `wait` outcomes from a
    /// `VecDeque`. Drives the bin's loop body without touching kernel
    /// resources — the loop's discipline (top-of-iteration shutdown
    /// check, pulse-on-true, fall-through-on-false, exit-on-Err) can be
    /// asserted in isolation.
    ///
    /// On exhaustion of the script, returns `Ok(false)` indefinitely so
    /// the loop exits via the `shutdown_flag` path the test sets up.
    #[derive(Debug, Default)]
    struct ScriptedConfigWatcher {
        outcomes: std::collections::VecDeque<std::io::Result<bool>>,
        wait_calls: usize,
        wake_calls: Arc<std::sync::atomic::AtomicUsize>,
    }

    impl specter_sensor::ConfigWatcher for ScriptedConfigWatcher {
        fn wait(&mut self, _deadline: Option<std::time::Instant>) -> std::io::Result<bool> {
            self.wait_calls += 1;
            self.outcomes.pop_front().unwrap_or(Ok(false))
        }
        fn wake_handle(&self) -> Box<dyn WakeHandle> {
            Box::new(StubWake {
                count: Arc::clone(&self.wake_calls),
            })
        }
    }

    #[derive(Debug)]
    struct StubWake {
        count: Arc<std::sync::atomic::AtomicUsize>,
    }
    impl WakeHandle for StubWake {
        fn wake(&self) {
            self.count
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        fn clone_box(&self) -> Box<dyn WakeHandle> {
            Box::new(Self {
                count: Arc::clone(&self.count),
            })
        }
    }

    /// Allocate a [`ConfigWatcherSide`] paired with its receiver. The
    /// config-event channel is no longer materialised by
    /// [`Channels::new`]; it is the conditional auto-reload edge
    /// `App::run` allocates inline. Tests below mirror that pattern.
    fn config_watcher_pair() -> (ConfigWatcherSide, crossbeam::channel::Receiver<()>) {
        let (config_event_tx, config_event_rx) = bounded::<()>(1);
        (ConfigWatcherSide { config_event_tx }, config_event_rx)
    }

    /// `Ok(true)` arrival emits exactly one pulse on `config_event_tx`.
    /// Verifies the loop body's "watch event ⇒ try_send" mapping.
    #[test]
    fn config_watcher_loop_emits_pulse_on_ok_true() {
        let (sides, config_event_rx) = config_watcher_pair();
        let mut watcher = ScriptedConfigWatcher {
            outcomes: std::iter::once(Ok(true)).collect(),
            ..Default::default()
        };
        // Flag set so the iteration after the scripted Ok(true) (which
        // re-enters with empty deque ⇒ Ok(false)) sees flag=true and
        // exits.
        let flag = Arc::new(AtomicBool::new(false));
        // Emulate the production shutdown sequence: scripted single
        // event, then bound to exit. We arm the flag before the second
        // iteration by spawning a wakeup-style mutator (in a
        // single-threaded test this is just storing true between calls).
        // Simpler: arrange the script with [Ok(true)] then mutate the
        // flag inside the test by giving the watcher a scripted Ok(false)
        // that triggers shutdown.
        watcher.outcomes.push_back(Ok(false));
        // Use a small helper thread to flip the flag after the second
        // wait returns; but in test we can just precompute: queue events
        // to drive the loop deterministically.
        let flag_handle = Arc::clone(&flag);
        std::thread::scope(|s| {
            s.spawn(|| {
                // Yield long enough for the loop to consume both
                // outcomes and start a third wait. The third `wait` hits
                // the empty-deque fallback `Ok(false)`; the next flag
                // check then exits.
                std::thread::sleep(std::time::Duration::from_millis(10));
                flag_handle.store(true, Ordering::SeqCst);
            });
            config_watcher_loop(&mut watcher, &sides, &flag);
        });
        // Drained pulse delivered.
        assert!(
            config_event_rx.try_recv().is_ok(),
            "Ok(true) ⇒ pulse on config_event_rx"
        );
    }

    /// `Ok(false)` does NOT emit a pulse; the loop falls through to the
    /// next iteration's flag check. Combined with the flag-set, the loop
    /// exits without any pulse on the channel.
    #[test]
    fn config_watcher_loop_no_pulse_on_ok_false() {
        let (sides, config_event_rx) = config_watcher_pair();
        let mut watcher = ScriptedConfigWatcher {
            outcomes: std::iter::once(Ok(false)).collect(),
            ..Default::default()
        };
        let flag = Arc::new(AtomicBool::new(false));
        let flag_handle = Arc::clone(&flag);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                flag_handle.store(true, Ordering::SeqCst);
            });
            config_watcher_loop(&mut watcher, &sides, &flag);
        });
        assert!(config_event_rx.try_recv().is_err(), "Ok(false) ⇒ no pulse");
    }

    /// `Err` from `wait` ⇒ the loop logs and exits — without pulsing,
    /// without panicking.
    #[test]
    fn config_watcher_loop_exits_on_err() {
        let (sides, config_event_rx) = config_watcher_pair();
        let mut watcher = ScriptedConfigWatcher {
            outcomes: std::iter::once(Err(std::io::Error::other("synthetic"))).collect(),
            ..Default::default()
        };
        let flag = Arc::new(AtomicBool::new(false));
        // No flag flip needed — Err exits immediately.
        config_watcher_loop(&mut watcher, &sides, &flag);
        assert_eq!(watcher.wait_calls, 1, "single wait, then Err");
        assert!(config_event_rx.try_recv().is_err(), "no pulse on Err");
    }

    /// Top-of-loop shutdown check: with the flag pre-set to `true`, the
    /// loop exits without ever calling `wait`. This is the exit path the
    /// bin's shutdown sequence relies on after `wake()` returns the
    /// thread to the loop top.
    #[test]
    fn config_watcher_loop_exits_immediately_on_pre_set_flag() {
        let (sides, _config_event_rx) = config_watcher_pair();
        let mut watcher = ScriptedConfigWatcher::default();
        let flag = Arc::new(AtomicBool::new(true));
        config_watcher_loop(&mut watcher, &sides, &flag);
        assert_eq!(
            watcher.wait_calls, 0,
            "pre-set flag short-circuits the wait"
        );
    }

    /// Bounded(1) channel + repeated `Ok(true)` ⇒ pulses coalesce. The
    /// loop's `try_send` returns `Full` on saturation; the loop must
    /// not panic and must continue iterating. Verifies the
    /// "kernel-queue layer coalescing" rationale documented on the
    /// loop body.
    #[test]
    fn config_watcher_loop_coalesces_under_pressure() {
        let (sides, config_event_rx) = config_watcher_pair();
        // Three Ok(true) — only one pulse fits in the bounded(1) slot.
        let mut watcher = ScriptedConfigWatcher {
            outcomes: [Ok(true), Ok(true), Ok(true)].into_iter().collect(),
            ..Default::default()
        };
        let flag = Arc::new(AtomicBool::new(false));
        let flag_handle = Arc::clone(&flag);
        std::thread::scope(|s| {
            s.spawn(|| {
                std::thread::sleep(std::time::Duration::from_millis(10));
                flag_handle.store(true, Ordering::SeqCst);
            });
            config_watcher_loop(&mut watcher, &sides, &flag);
        });
        // Exactly one pulse drained — the other two coalesced at the
        // bounded(1) slot via `try_send` Full.
        assert!(config_event_rx.try_recv().is_ok(), "first pulse");
        assert!(
            config_event_rx.try_recv().is_err(),
            "no second pulse — coalesced"
        );
    }

    /// The watcher's deferred-drain window is a fixed trailing-latency
    /// constant — no longer config-derived. Pins the value and its
    /// in-band placement so a future change is a conscious latency
    /// decision, not accidental drift. The historical band was
    /// `[10ms, 50ms]`; `50ms` is the prior default-`settle` resolution.
    #[test]
    fn watcher_drain_window_is_fixed_at_band_ceiling() {
        assert_eq!(WATCHER_DRAIN_WINDOW, Duration::from_millis(50));
        assert!(
            WATCHER_DRAIN_WINDOW >= Duration::from_millis(10)
                && WATCHER_DRAIN_WINDOW <= Duration::from_millis(50),
            "constant must stay within the historical trailing-latency band",
        );
    }
}
