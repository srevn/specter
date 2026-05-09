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
use specter_config::{Cli, Config};
use specter_core::{Input, WatchOp};
use specter_engine::Engine;
use specter_sensor::{
    DrainWindow, FsWatcher, WakeHandle, WatcherEvent, WorkerProber, default_watcher,
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
    // filter (subsequent phases) to decide whether a watcher pulse
    // reflects substantive change.
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
    tracing::info!(
        level = ?log_cfg.level,
        destination = ?log_cfg.destination,
        path = ?log_cfg.path.as_ref().map(|p| p.display().to_string()),
        watches = initial_config.watches.len(),
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

    // Belt + braces: ensure the watcher exits even if the engine driver
    // returned via Disconnected (signal thread didn't fire). The wake
    // handle held by `App` is the linchpin — without it, the watcher's
    // `poll_until` would block forever.
    shutdown_flag.store(true, Ordering::SeqCst);
    wake_handle.wake();

    if let Err(e) = watcher_handle.join() {
        tracing::error!(?e, "watcher thread panicked");
    }
    if let Err(e) = actuator_handle.join() {
        tracing::error!(?e, "actuator thread panicked");
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
                path: "/tmp".into(),
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
                path: "/tmp".into(),
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
                path: "/tmp".into(),
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
}
