//! `App::run` — the bin's lifecycle entry point.
//!
//! Builds channels, spawns the signal / watcher / actuator threads,
//! constructs the [`EngineDriver`] on the main thread, runs initial
//! attach, enters the main loop, and runs the shutdown sequence
//! on exit.
//!
//! Spawn order is load-bearing:
//! 1. **Signals first** — `Signals::new` registers handlers immediately,
//!    so SIGTERM/SIGHUP arriving during init don't fall through to the
//!    kernel default.
//! 2. **Prober next** — workers must be ready to receive probes before
//!    initial attach emits the first `ProbeOp::Probe`.
//! 3. **Watcher** — `KqueueWatcher::new` must succeed; the wake handle
//!    is captured here.
//! 4. **Actuator** — must be ready before initial attach or the first
//!    tick can emit Effects.
//! 5. **Engine driver** runs on the main thread.

use crate::channels::Channels;
use crate::driver::EngineDriver;
use crate::loader::Loader;
use crate::observability;
use crate::signals::spawn_signal_thread;
use specter_actuator::{SubprocessActuator, default_spawner};
use specter_config::{Cli, Config, FileMeta};
use specter_core::{Input, WatchOp};
use specter_engine::Engine;
use specter_sensor::{
    ConfigWatcher, DrainWindow, FsWatcher, WakeHandle, WatcherEvent, WorkerProber,
    default_config_watcher, default_watcher,
};
use std::process::ExitCode;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};

/// Run the bin against `cli`.
///
/// Loads + validates the config, initializes tracing, starts every
/// long-lived thread, drives the engine to completion, and runs the
/// shutdown sequence. Returns `ExitCode::SUCCESS` on graceful
/// exit; `ExitCode::from(1)` on startup failure (config / kqueue /
/// prober / thread spawn).
///
/// `Cli` is taken by value because every field is consumed (config
/// moves into the driver; concurrency knobs are extracted then
/// dropped); the `needless_pass_by_value` allow documents the intent.
#[allow(clippy::needless_pass_by_value)]
pub fn run(cli: Cli) -> ExitCode {
    // Load config (fail-fast, pre-tracing). `from_path_with_meta`
    // captures `FileMeta` atomically with the bytes via a single `File`
    // handle — closing the startup TOCTOU between the content read
    // and a separate path-level lstat. The captured value seeds
    // `loader.config_meta` and is consulted by the auto-reload settle
    // filter to decide whether a watcher pulse reflects substantive
    // change.
    let (initial_config, initial_meta) = match Config::from_path_with_meta(&cli.config) {
        Ok(pair) => pair,
        Err(e) => {
            eprintln!("specter: config load failed:\n{e}");
            return ExitCode::from(1);
        }
    };

    // Tracing — CLI overrides applied on top of `[log]` (cli wins).
    let log_cfg = match initial_config.log.clone().merge_cli(
        cli.log_level,
        cli.log_destination,
        cli.log_path.clone(),
    ) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("specter: log config invalid:\n{e}");
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
    let disabled_watches: Vec<&str> = initial_config
        .watches
        .iter()
        .filter(|s| !s.enabled)
        .map(|s| s.name.as_str())
        .collect();
    let disabled_promoters: Vec<&str> = initial_config
        .promoters
        .iter()
        .filter(|p| !p.enabled)
        .map(|p| p.name.as_str())
        .collect();
    tracing::info!(
        level = ?log_cfg.level,
        destination = ?log_cfg.destination,
        path = ?log_cfg.path.as_ref().map(|p| p.display().to_string()),
        watches = initial_config.watches.len(),
        promoters = initial_config.promoters.len(),
        ?disabled_watches,
        ?disabled_promoters,
        config = %cli.config.display(),
        "specter starting"
    );

    // Bookkeeping for the watcher's deferred-drain phase. The bin holds
    // one `DrainWindow` and gives the watcher its own clone so the
    // Atomic store on hot reload reaches both threads without a lock.
    // Set once before `default_watcher` so the watcher reads the
    // derived value on its very first `poll_until`.
    let loader = Loader::new(initial_config, log_cfg, initial_meta);
    let drain_window = DrainWindow::new();
    drain_window.set(loader.derive_drain_window());

    // Kqueue (or Linux inotify, when that backend lands) + wake handle.
    let watcher = match default_watcher(drain_window.clone()) {
        Ok(w) => w,
        Err(e) => {
            tracing::error!(?e, "watcher init failed");
            return ExitCode::from(1);
        }
    };
    let wake_handle: Box<dyn WakeHandle> = watcher.wake_handle();

    // Channels.
    let mut chans = Channels::new();

    // Prober (workers spawn inside `WorkerProber::new`).
    let probe_concurrency = cli
        .probe_concurrency
        .map_or(specter_sensor::DEFAULT_CONCURRENCY, |n| n as usize);
    let prober = match WorkerProber::new(&chans.sensor_in_tx, probe_concurrency) {
        Ok(p) => Arc::new(p),
        Err(e) => {
            tracing::error!(?e, "prober init failed");
            return ExitCode::from(1);
        }
    };

    // Shutdown coordination.
    let shutdown_flag = Arc::new(AtomicBool::new(false));

    // Signal thread (registers handlers immediately).
    let signal_handle = spawn_signal_thread(
        chans.signal_side(),
        Arc::clone(&shutdown_flag),
        wake_handle.clone(),
    );

    // Watcher thread.
    let watcher_handle = spawn_watcher_thread(
        watcher,
        chans.take_watcher_side(),
        Arc::clone(&shutdown_flag),
    );

    // Actuator thread.
    let actuator_concurrency = cli
        .concurrency
        .map_or(specter_actuator::DEFAULT_CONCURRENCY, |n| n as usize);
    let actuator_handle = spawn_actuator_thread(actuator_concurrency, chans.take_actuator_side());

    // Engine driver — main thread.
    let config_path = cli.config;
    let cli_log_overrides = CliLogOverrides {
        level: cli.log_level,
        destination: cli.log_destination,
        path: cli.log_path,
    };

    // Auto-reload — config-watcher init (default-on; opt-out via
    // `--no-config-watch` / `SPECTER_NO_CONFIG_WATCH`).
    //
    // **Keepalive discipline.** Always project a `config_event_tx`
    // clone onto the stack regardless of whether the watcher is
    // spawned (`--no-config-watch`, init failure, or watcher-thread
    // exit on `Err`). Without this, the engine's `config_event_rx`
    // would observe Disconnected on every path that doesn't keep a
    // live producer; crossbeam's `Select::ready_timeout` reports
    // disconnected arms as immediately-ready, busy-looping the
    // driver tick. The keepalive on the stack outlives the driver
    // (which holds the rx); when `App::run` returns, both halves
    // drop together and the channel cleanly tears down.
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
    // queue a SIGHUP-style pulse on `reload_signal_tx` so the
    // driver's first tick handles it immediately (no settle delay,
    // unlike steady-state pulses). The driver's `handle_reload`
    // re-reads the file with a fresh atomic capture and rotates
    // `loader.config_meta` to the post-edit identity.
    let _config_event_keepalive = chans.config_watcher_side().config_event_tx;
    let config_watcher_handles = if cli.no_config_watch {
        tracing::info!("auto-reload disabled via --no-config-watch");
        None
    } else {
        match default_config_watcher(&config_path) {
            Ok(watcher) => {
                match FileMeta::from_path(&config_path) {
                    Ok(post_init_meta) if post_init_meta != initial_meta => {
                        if chans.reload_signal_tx.try_send(()).is_ok() {
                            tracing::info!(
                                "config changed during startup; reload queued via SIGHUP path"
                            );
                        }
                    }
                    Ok(_) => {}
                    Err(e) => tracing::warn!(
                        ?e,
                        "post-init config lstat failed; skipping startup-TOCTOU pulse"
                    ),
                }
                let cw_wake = watcher.wake_handle();
                let cw_handle = spawn_config_watcher_thread(
                    watcher,
                    chans.config_watcher_side(),
                    Arc::clone(&shutdown_flag),
                );
                Some((cw_handle, cw_wake))
            }
            Err(e) => {
                tracing::warn!(?e, "config watcher init failed; SIGHUP-only reload",);
                None
            }
        }
    };

    let mut driver = EngineDriver::new(
        Engine::new(),
        loader,
        config_path,
        cli_log_overrides,
        obs_handle,
        chans.take_engine_side(),
        prober.clone(),
        wake_handle.clone(),
        drain_window,
    );
    drop(chans); // originals release; per-thread clones keep channels alive.

    driver.run_initial_attach();
    let exit_reason = driver.run();
    tracing::info!(?exit_reason, "engine driver exited");

    // Shutdown sequence.
    drop(driver); // releases driver's clones (engine, prober, wake_handle, txs).

    // Belt + braces: ensure both watchers exit even if the engine
    // driver returned via Disconnected (signal thread didn't fire).
    // The wake handles held by `App` are the linchpin — without
    // them the watchers' blocking `poll_until` / `wait` would
    // never return.
    //
    // Order is load-bearing: store the flag before waking so the
    // watcher thread, which checks `shutdown_flag` at the *top* of
    // its loop, observes the new value when its `wait` returns
    // (`Ok(false)`) and exits cleanly. A wake before the store
    // would race the loop's flag read against the watcher's
    // `wait`-return path.
    shutdown_flag.store(true, Ordering::SeqCst);
    wake_handle.wake();
    if let Some((_, ref cw_wake)) = config_watcher_handles {
        cw_wake.wake();
    }

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
fn spawn_watcher_thread(
    mut watcher: specter_sensor::DefaultWatcher,
    sides: crate::channels::WatcherSide,
    shutdown_flag: Arc<AtomicBool>,
) -> JoinHandle<()> {
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
        .expect("spawn watcher thread")
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
                Ok(op) => apply_watch_op(watcher, op, &sides.sensor_in_tx),
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
                            let _ = sides.sensor_in_tx.send(Input::FsEvent { resource, event });
                        }
                        WatcherEvent::Overflow { scope } => {
                            // inotify's `IN_Q_OVERFLOW` lifts here on
                            // Linux; kqueue never emits Overflow under
                            // v1 (`EV_CLEAR` coalesces but never
                            // silently drops). The engine's
                            // `on_sensor_overflow` handler reseeds every
                            // in-scope Profile and emits
                            // `Diagnostic::SensorOverflow`.
                            let _ = sides.sensor_in_tx.send(Input::SensorOverflow { scope });
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
fn spawn_config_watcher_thread(
    mut watcher: specter_sensor::DefaultConfigWatcher,
    sides: crate::channels::ConfigWatcherSide,
    shutdown_flag: Arc<AtomicBool>,
) -> JoinHandle<()> {
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
        .expect("spawn config-watcher thread")
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
/// Takes `op` by value because the rejection path constructs the
/// `WatchOpRejected` payload with the same path; on success we do not
/// need to clone.
pub(crate) fn apply_watch_op<W: FsWatcher>(
    watcher: &mut W,
    op: WatchOp,
    sensor_in_tx: &crossbeam::channel::Sender<Input>,
) {
    match op {
        WatchOp::Watch {
            resource,
            path,
            kind,
            events,
        } => {
            if let Err(failure) = watcher.watch(resource, &path, kind, events) {
                let rejected = WatchOp::Watch {
                    resource,
                    path,
                    kind,
                    events,
                };
                let _ = sensor_in_tx.send(Input::WatchOpRejected {
                    resource,
                    op: rejected,
                    failure,
                });
            }
        }
        WatchOp::Unwatch { resource } => watcher.unwatch(resource),
        WatchOp::Suppress { resource } => watcher.suppress(resource),
        WatchOp::Unsuppress { resource } => watcher.unsuppress(resource),
    }
}

/// Spawn the actuator thread. Constructs [`SubprocessActuator`] with
/// the resolved concurrency, runs the controller blocking until either
/// `effects_rx` disconnects or `shutdown_actuator_rx` fires.
fn spawn_actuator_thread(
    concurrency: usize,
    sides: crate::channels::ActuatorSide,
) -> JoinHandle<()> {
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
                );
            }));
            if let Err(payload) = result {
                tracing::error!(
                    "actuator thread panicked; payload size = {}",
                    std::mem::size_of_val(&payload),
                );
            }
        })
        .expect("spawn actuator thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::Channels;
    use slotmap::SlotMap;
    use specter_core::{ClassSet, ResourceId, ResourceKind, WatchFailure};
    use specter_sensor::testkit::MockFsWatcher;

    /// Mint a fresh non-null `ResourceId`. Required because slotmap's
    /// `SecondaryMap` rejects the null/default key — and `MockFsWatcher`
    /// stores its installed/suppressed state in `SecondaryMap`s.
    fn fresh_resource_id() -> ResourceId {
        let mut map: SlotMap<ResourceId, ()> = SlotMap::with_key();
        map.insert(())
    }

    #[test]
    fn apply_watch_op_watch_calls_watcher() {
        let mut chans = Channels::new();
        let sides = chans.take_watcher_side();
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        apply_watch_op(
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
        let mut chans = Channels::new();
        let sides = chans.take_watcher_side();
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        // EMFILE = 24 on macOS / FreeBSD / Linux. Hard-coded so the
        // bin's tests don't pull `libc` as a direct dev-dep.
        watcher.fail_next_watch(WatchFailure::Pressure { errno: 24 });
        apply_watch_op(
            &mut watcher,
            WatchOp::Watch {
                resource: r,
                path: Arc::from(std::path::Path::new("/tmp")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            &sides.sensor_in_tx,
        );
        // Reject delivered to engine inbound.
        let engine_side = chans.take_engine_side();
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
        let mut chans = Channels::new();
        let sides = chans.take_watcher_side();
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
        apply_watch_op(
            &mut watcher,
            WatchOp::Unwatch { resource: r },
            &sides.sensor_in_tx,
        );
        assert!(!watcher.installed.contains_key(r));
    }

    #[test]
    fn apply_watch_op_suppress_marks_suppressed() {
        let mut chans = Channels::new();
        let sides = chans.take_watcher_side();
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
        apply_watch_op(
            &mut watcher,
            WatchOp::Suppress { resource: r },
            &sides.sensor_in_tx,
        );
        assert!(watcher.suppressed.contains_key(r));
    }

    #[test]
    fn watcher_loop_drains_ops_and_exits_on_shutdown_flag() {
        let mut chans = Channels::new();
        let sides = chans.take_watcher_side();
        let mut watcher = MockFsWatcher::new();
        let r = fresh_resource_id();
        // Queue an op + an event before entering the loop.
        chans
            .watch_ops_tx
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
        let mut chans = Channels::new();
        let sides = chans.take_watcher_side();
        let mut watcher = MockFsWatcher::new();
        // Drop watch_ops_tx (and the rest of chans) — the loop's try_recv
        // returns Disconnected and the loop exits.
        drop(chans);
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

    /// `Ok(true)` arrival emits exactly one pulse on `config_event_tx`.
    /// Verifies the loop body's "watch event ⇒ try_send" mapping.
    #[test]
    fn config_watcher_loop_emits_pulse_on_ok_true() {
        let mut chans = Channels::new();
        let sides = chans.config_watcher_side();
        let engine = chans.take_engine_side();
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
            engine.config_event_rx.try_recv().is_ok(),
            "Ok(true) ⇒ pulse on config_event_rx"
        );
    }

    /// `Ok(false)` does NOT emit a pulse; the loop falls through to the
    /// next iteration's flag check. Combined with the flag-set, the loop
    /// exits without any pulse on the channel.
    #[test]
    fn config_watcher_loop_no_pulse_on_ok_false() {
        let mut chans = Channels::new();
        let sides = chans.config_watcher_side();
        let engine = chans.take_engine_side();
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
        assert!(
            engine.config_event_rx.try_recv().is_err(),
            "Ok(false) ⇒ no pulse"
        );
    }

    /// `Err` from `wait` ⇒ the loop logs and exits — without pulsing,
    /// without panicking.
    #[test]
    fn config_watcher_loop_exits_on_err() {
        let mut chans = Channels::new();
        let sides = chans.config_watcher_side();
        let engine = chans.take_engine_side();
        let mut watcher = ScriptedConfigWatcher {
            outcomes: std::iter::once(Err(std::io::Error::other("synthetic"))).collect(),
            ..Default::default()
        };
        let flag = Arc::new(AtomicBool::new(false));
        // No flag flip needed — Err exits immediately.
        config_watcher_loop(&mut watcher, &sides, &flag);
        assert_eq!(watcher.wait_calls, 1, "single wait, then Err");
        assert!(
            engine.config_event_rx.try_recv().is_err(),
            "no pulse on Err"
        );
    }

    /// Top-of-loop shutdown check: with the flag pre-set to `true`, the
    /// loop exits without ever calling `wait`. This is the exit path the
    /// bin's shutdown sequence relies on after `wake()` returns the
    /// thread to the loop top.
    #[test]
    fn config_watcher_loop_exits_immediately_on_pre_set_flag() {
        let chans = Channels::new();
        let sides = chans.config_watcher_side();
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
        let mut chans = Channels::new();
        let sides = chans.config_watcher_side();
        let engine = chans.take_engine_side();
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
        assert!(engine.config_event_rx.try_recv().is_ok(), "first pulse");
        assert!(
            engine.config_event_rx.try_recv().is_err(),
            "no second pulse — coalesced"
        );
    }
}
