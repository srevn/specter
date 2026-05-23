//! Engine-driver unit tests — single-tick drive of `EngineDriver`
//! over the `TestRig` mock-channel harness: drain order, the
//! SIGHUP / auto-reload settle pipeline, and the `forward`
//! wake-per-send protocol.
//!
//! Wired by `#[cfg(test)] mod tests;` in `driver.rs`. Imports below
//! are explicit (no `use super::*;`) so the driver spine carries no
//! cfg(test)-only re-exports — the test surface is what this file
//! references, nothing more.

use super::state::ReloadTrigger;
use super::{EngineDriver, TickOutcome};
use crate::app::CliLogOverrides;
use crate::channels::{ActuatorSide, Channels, WatcherSide};
use crate::loader::Loader;
use crossbeam::channel::Sender;
use specter_config::{Config, FileMeta};
use specter_core::{Input, StepOutput, SubId, WatchOp};
use specter_engine::Engine;
use specter_sensor::FsWatcher;
use specter_sensor::testkit::{MockFsWatcher, MockProber, MockWaker};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Sentinel meta used in fixtures whose config file may not exist
/// on disk. Inode 0 is reserved by every supported kernel and
/// `mode = 0` cannot occur in a real lstat (the kernel always sets
/// file-type bits); this value never compares equal to a real
/// `FileMeta::from_path` capture, so tests that *do* exercise the
/// meta-rotation path can assert "rotated to a real value" by
/// comparing against a fresh `FileMeta::from_path` (which differs
/// from this sentinel in every field).
fn dummy_meta() -> FileMeta {
    FileMeta {
        inode: 0,
        device: 0,
        mtime_sec: 0,
        mtime_nsec: 0,
        size: 0,
        mode: 0,
        uid: 0,
        gid: 0,
    }
}

/// Bundle of handles a test holds to drive [`EngineDriver`] without the
/// [`crate::app`] orchestration layer.
struct TestRig {
    driver: EngineDriver,
    /// Held to keep the `watcher_side` receivers alive (and so tests
    /// can assert what would have been delivered to the watcher).
    watcher_side: WatcherSide,
    actuator_side: ActuatorSide,
    prober: Arc<MockProber>,
    waker: Arc<MockWaker>,
    sensor_in_tx: Sender<Input>,
    effect_in_tx: Sender<Input>,
    reload_tx: Sender<()>,
    shutdown_tx: Sender<()>,
    /// Cloned config-event sender. Holding this clone alive in the
    /// rig keeps the engine's `config_event_rx` connected (otherwise
    /// `drop(chans)` would release the only sender and the
    /// driver's `Select` arm would observe Disconnected). Tests
    /// `try_send(())` here to simulate watcher pulses.
    config_event_tx: Sender<()>,
    /// Cloned `watch_ops_tx`. The driver holds its own clone via
    /// `engine_side`; tests that need to fill the bounded channel
    /// from the outside use this one. The watcher-side `watch_ops_rx`
    /// stays the sole receiver.
    watch_ops_tx: Sender<WatchOp>,
    /// Cloned `ipc_request_tx`. The driver holds its own clone via
    /// `engine_side.ipc_request_rx`; tests that need to drive the
    /// IPC drain queue an `IpcRequest` through this sender.
    ipc_request_tx: Sender<crate::ipc::protocol::IpcRequest>,
}

fn rig_for(config: Config, config_path: PathBuf) -> TestRig {
    let chans = Channels::new();
    // Field-level clones for the test's producer-side handles. Each
    // clone targets the same underlying channel as the bundle field
    // it's lifted off; the bundle moves below, the clones stay.
    let sensor_in_tx = chans.watcher.sensor_in_tx.clone();
    let effect_in_tx = chans.actuator.effect_in_tx.clone();
    let reload_tx = chans.signal.reload_signal_tx.clone();
    let shutdown_tx = chans.signal.shutdown_engine_tx.clone();
    let watch_ops_tx = chans.engine.watch_ops_tx.clone();
    let ipc_request_tx = chans.ipc_server.ipc_request_tx.clone();

    // Auto-reload — the rig always exercises the wired-on path,
    // mirroring `App::run`'s inline allocation. Tests that need the
    // arm absent can construct an `EngineSide` via
    // `EnginePieces::finalize(None)` directly.
    let (config_event_tx, config_event_rx) = crossbeam::channel::bounded::<()>(1);

    let actuator_side = chans.actuator;
    let watcher_side = chans.watcher;
    let engine_side = chans.engine.finalize(Some(config_event_rx));
    // `chans.signal` / `chans.ipc_server`'s only role for the rig
    // was the clones captured above; the bundles themselves drop at
    // end-of-scope when `chans` does. Senders surviving through
    // clones keep each channel alive across the driver loop.

    let watcher = MockFsWatcher::new();
    let waker = Arc::clone(&watcher.waker);
    let wake_handle = watcher.wake_handle();
    let prober: Arc<MockProber> = Arc::new(MockProber::new());

    let log_cfg = config.log.clone();
    // Tests don't drive the SIGHUP API meaningfully and would race
    // each other on the global subscriber slot if every rig called
    // `observability::init`. `noop()` returns a structurally-correct
    // handle whose `set_level` / `reopen_file` are silent no-ops —
    // tests assert the *driver*'s reload-pipeline behaviour, not the
    // subscriber's filter state.
    let obs_handle = crate::observability::ObservabilityHandle::noop();
    let loader = Loader {
        current_config: config,
        current_log: log_cfg,
        config_meta: dummy_meta(),
    };
    // Synthetic socket path — the rig never binds a listener, but
    // `EngineDriver::new` requires the value (load-bearing for the
    // `status` projection's `socket_path` field). A fixed string
    // keeps test fixtures deterministic without polluting `/tmp`.
    let socket_path = PathBuf::from("/tmp/specter-test.sock");
    let driver = EngineDriver::new(
        Engine::new(),
        loader,
        config_path,
        socket_path,
        CliLogOverrides::default(),
        obs_handle,
        engine_side,
        prober.clone(),
        wake_handle,
    );
    TestRig {
        driver,
        watcher_side,
        actuator_side,
        prober,
        waker,
        sensor_in_tx,
        effect_in_tx,
        reload_tx,
        shutdown_tx,
        config_event_tx,
        watch_ops_tx,
        ipc_request_tx,
    }
}

fn config_with_one_watch(path: &std::path::Path) -> Config {
    let toml = format!(
        r#"
[log]
level = "warn"

[[watch]]
name      = "build"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
"#,
        path.display(),
    );
    Config::from_str(&toml).expect("test config parses")
}

#[test]
fn empty_run_returns_continue_after_select_timeout() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    // No inputs queued; select times out after FOREVER_TIMEOUT, but
    // since the engine has no timers we'd block forever. Skirt this
    // by triggering shutdown immediately.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

#[test]
fn run_initial_attach_attaches_static_sub_and_emits_watch_op() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    // One Sub was attached → the engine's static `by_name` index
    // resolves that operator name to a live SubId.
    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");
    assert!(rig.driver.engine.subs().get(sid).is_some());

    // The attach emitted a Watch op → forwarded to watch_ops_tx.
    let mut watch_ops = Vec::new();
    while let Ok(op) = rig.watcher_side.watch_ops_rx.try_recv() {
        watch_ops.push(op);
    }
    assert!(!watch_ops.is_empty(), "attach emits at least one Watch op");

    // Wake handle was poked (since ≥1 WatchOp was sent).
    assert!(*rig.waker.woken.lock().unwrap() >= 1);

    // The Seed burst emitted a probe → forwarded to prober.submit.
    let submitted = rig.prober.take_submitted();
    assert_eq!(submitted.len(), 1);

    // The attach left a Seed-Verifying probe armed. Production
    // always loops to a shutdown tick, which drains it via
    // `begin_shutdown`; model that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

#[test]
fn shutdown_signal_returns_shutdown_outcome() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

#[test]
fn sensor_in_disconnect_returns_shutdown() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    // Drop every sender into sensor_in_rx (the engine side holds the
    // only receiver). Drop our test clone first; the watcher_side's
    // sensor_in_tx clone still keeps it alive — drop that too.
    drop(rig.sensor_in_tx);
    let WatcherSide {
        watch_ops_rx,
        sensor_in_tx,
    } = rig.watcher_side;
    drop(sensor_in_tx);
    drop(watch_ops_rx); // not needed for this assertion

    // Now sensor_in_rx is disconnected; tick observes via try_recv.
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

#[test]
fn effect_in_disconnect_shuts_down() {
    // Every inbound drain is Terminal on Disconnect. Previously
    // `effect_in_rx` Disconnect was silently treated as `Empty`
    // (crossbeam's `Select::ready_timeout` then reports the same arm
    // as immediately-ready, busy-looping the driver until SIGTERM).
    // This test pins the corrected policy: disconnect alone drives
    // the tick to `Shutdown`, no shutdown pulse required.
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    drop(rig.effect_in_tx);
    let ActuatorSide {
        effects_rx,
        shutdown_actuator_rx,
        hard_shutdown_actuator_rx,
        effect_in_tx,
        hard_shutdown_done_tx,
    } = rig.actuator_side;
    drop(effect_in_tx);
    drop(effects_rx);
    drop(shutdown_actuator_rx);
    drop(hard_shutdown_actuator_rx);
    drop(hard_shutdown_done_tx);

    // No shutdown_tx pulse — the disconnect alone must drive Shutdown.
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

#[test]
fn reload_with_invalid_path_logs_and_keeps_config() {
    // Config file at a non-existent path: handle_reload returns early
    // without touching loader.current_config.
    let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config.clone(), cfg_path);

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
    // Current config preserved.
    assert_eq!(rig.driver.loader.current_config, config);
}

#[test]
fn reload_with_no_changes_rotates_config_silently() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_text = format!(
        r#"
[[watch]]
name      = "build"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &cfg_text).unwrap();
    let initial = Config::from_str(&cfg_text).expect("test config");

    let mut rig = rig_for(initial.clone(), cfg_path);
    let _ = rig.driver.run_initial_attach();
    let sid_before = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    // No changes → the attached Sub is the same identity (no
    // reap/re-attach churned its SubId).
    let sid_after = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' still attached");
    assert_eq!(sid_before, sid_after);
    assert_eq!(rig.driver.loader.current_config, initial);
}

#[test]
fn reload_added_watch_attaches_in_engine() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
settle    = "100ms"
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_none());

    // Operator edits config; sends SIGHUP (we simulate via the channel).
    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    // Both Subs now attached in the engine.
    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_some());
}

#[test]
fn reload_removed_watch_detaches_in_engine() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_some());

    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_none());
}

#[test]
fn fs_event_drained_before_effect_complete_so_fire_tail_absorbs() {
    // Sensor inputs drain BEFORE effect completions: an EffectComplete
    // could move an Awaiting burst into Rebasing, and any FsEvent
    // queued in the same tick should reach the engine first so the
    // fire-tail (`PostFirePhase::Awaiting` / `Rebasing`) can absorb it
    // and fold the disk change into the post-fire rebase. Push an
    // EffectComplete first, then an FsEvent; tick sees the FsEvent first
    // because of the drain order — even though EffectComplete was queued earlier.
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let r = specter_core::ResourceId::default();
    rig.effect_in_tx
        .send(Input::EffectComplete {
            sub: SubId::default(),
            key: specter_core::DedupKey::Subtree {
                sub: SubId::default(),
                profile: specter_core::ProfileId::default(),
            },
            result: specter_core::EffectOutcome::Ok,
        })
        .unwrap();
    rig.sensor_in_tx
        .send(Input::FsEvent {
            resource: r,
            event: specter_core::FsEvent::Modified,
        })
        .unwrap();
    rig.shutdown_tx.try_send(()).expect("shutdown send");

    // Tick processes both, then sees shutdown. The engine handles
    // each input atomically in the order step is called; the drain
    // order is what we're testing — the bin's contract is that
    // sensor inputs reach engine.step before effect completions.
    let outcome = rig.driver.tick();
    assert_eq!(outcome, TickOutcome::Shutdown);
    // We don't assert on engine state here — the FsEvent + EC for
    // unknown ids both produce diagnostics, and the order of
    // diagnostics confirms drain order. For a behavioral test, see
    // `tests/e2e_*` integration tests where ordering surfaces as
    // observable subprocess behavior.
}

#[test]
fn forward_wakes_after_each_send_to_break_full_channel_deadlock() {
    // Regression for the deep-tree startup deadlock. The Seed burst
    // against a tree with many directories emits a single `StepOutput`
    // whose `watch_ops` exceeds the bounded(1024) `watch_ops_tx`
    // capacity. With a wake-once-at-end protocol, the engine's `forward`
    // would fill the channel, block on `Sender::send` at op 1025, and
    // never reach the trailing `wake_handle.wake()` — leaving the watcher
    // asleep in `kevent` until SIGTERM forced a separate wake. The
    // contract is one wake **per successful send**, so the watcher always
    // sees a fresh `EVFILT_USER` trigger to drain by, kernel-coalesced.
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let rig = rig_for(config, cfg_path);
    let TestRig {
        mut driver,
        watcher_side,
        waker,
        watch_ops_tx,
        ..
    } = rig;
    // `..` defers drop of unmatched fields to end of scope (partial-move
    // semantics retain them on the residual `rig` storage). The rig's
    // `watch_ops_tx` clone targets the same bounded channel the drainer
    // recvs on; leaving it alive past `drop(driver)` would keep the
    // drainer's `recv()` blocked. Drop it eagerly so the engine-side
    // clone is the only remaining producer when the driver dies.
    drop(watch_ops_tx);

    // Drain in a thread so the bounded channel can flow; without a
    // drainer this test would block at the channel-bound boundary even
    // with the fix in place — wake-per-send unsticks the *kqueue*
    // consumer, not crossbeam's `bounded` send.
    let watch_ops_rx = watcher_side.watch_ops_rx;
    let drainer = std::thread::spawn(move || {
        let mut count = 0usize;
        while watch_ops_rx.recv().is_ok() {
            count += 1;
        }
        count
    });

    let n_ops: usize = 5;
    let mut out = StepOutput::default();
    for i in 0..n_ops {
        out.watch_ops.push(specter_core::WatchOp::Watch {
            resource: specter_core::ResourceId::default(),
            path: Arc::from(PathBuf::from(format!("/p/{i}"))),
            kind: specter_core::ResourceKind::Unknown,
            events: specter_core::ClassSet::EMPTY,
        });
    }

    let outcome = driver.forward(out);
    assert_eq!(
        outcome,
        ControlFlow::Continue(()),
        "every send succeeded; no shutdown was signalled",
    );
    drop(driver); // last `watch_ops_tx` released; drainer's recv() disconnects.

    let received = drainer.join().expect("drainer thread panicked");
    assert_eq!(received, n_ops, "all ops must reach the watcher");

    let woken = usize::try_from(*waker.woken.lock().expect("MockWaker poisoned"))
        .expect("wake count fits in usize");
    assert_eq!(
        woken, n_ops,
        "expected wake-per-send (n={n_ops}); got {woken}",
    );
}

// ===== engine-owned identity across attach / reload =====
//
// The bin no longer mirrors `name → id`: the engine's
// `SubRegistry`/`PromoterRegistry` `by_name` indices are the sole
// authority. Tests in this section drive `run_initial_attach` /
// `handle_reload` and assert attachment through the engine
// accessors (`engine.subs().find_by_name` /
// `engine.promoters().find_by_name`).

/// Build a config with a single dynamic [[watch]] entry. The path
/// uses brace expansion, exercising the `is_dynamic` auto-detect
/// path (the brace `{` is one of `*?[{`). Literal prefix is the
/// supplied `tmp` directory so the validator's path-canonicalisation
/// pass succeeds.
fn config_with_one_promoter(path: &std::path::Path) -> Config {
    let toml = format!(
        r#"
[log]
level = "warn"

[[watch]]
name      = "logs"
path      = "{}/{{a,b}}/access.log"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
"#,
        path.display(),
    );
    Config::from_str(&toml).expect("test config parses")
}

/// `run_initial_attach` for a static-only config attaches one Sub
/// per `[[watch]]` into the engine's static `by_name` index and
/// leaves the Promoter registry empty.
#[test]
fn run_initial_attach_attaches_static_only_config() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    // Engine's static `by_name` resolves the operator name; no
    // Promoter was created.
    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");
    assert!(rig.driver.engine.subs().get(sid).is_some());
    assert!(rig.driver.engine.promoters().is_empty());

    // The attach left a Seed-Verifying probe armed. Production
    // always loops to a shutdown tick, which drains it via
    // `begin_shutdown`; model that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// `run_initial_attach` extension: a config with a dynamic
/// `[[watch]]` (auto-detected at config load) routes through
/// `attach_promoter` and registers the Promoter in the engine's
/// `PromoterRegistry` `by_name` index.
#[test]
fn run_initial_attach_registers_promoter_for_dynamic_watch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_promoter(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    // No static Sub; the engine's Promoter registry carries the
    // dynamic entry under its operator name.
    assert!(rig.driver.engine.subs().is_empty());
    let pid = rig
        .driver
        .engine
        .promoters()
        .find_by_name("logs")
        .expect("Promoter 'logs' registered");
    assert!(rig.driver.engine.promoters().get(pid).is_some());

    // The promoter attach left a descent probe armed. Production
    // always loops to a shutdown tick, which drains it via
    // `begin_shutdown`; model that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// Mixed static + dynamic config: the initial-attach loop walks
/// both spec lists and populates both maps in one run, with a
/// single forward per attach so the watcher receives WatchOps
/// incrementally.
#[test]
fn run_initial_attach_handles_mixed_static_and_dynamic() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_text = format!(
        r#"
[log]
level = "warn"

[[watch]]
name      = "build"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"

[[watch]]
name      = "logs"
path      = "{0}/{{a,b}}/access.log"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str(&cfg_text).expect("mixed config parses");
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    assert!(rig.driver.engine.subs().find_by_name("build").is_some());
    assert!(rig.driver.engine.promoters().find_by_name("logs").is_some());

    // The attaches left probes armed (static Seed-Verifying +
    // dynamic descent). Production always loops to a shutdown tick,
    // which drains them via `begin_shutdown`; model that here.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// Disabled entries on either side are skipped: the config carries
/// one enabled + one disabled static watch and one enabled + one
/// disabled dynamic watch. After initial attach, only the enabled
/// names resolve through the engine's registries; the disabled
/// entries leave no engine residue (not attached, registries hold
/// exactly one each).
#[test]
fn run_initial_attach_skips_disabled_entries() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_text = format!(
        r#"
[log]
level = "warn"

[[watch]]
name      = "build"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"

[[watch]]
name      = "build_off"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
enabled   = false

[[watch]]
name      = "logs"
path      = "{0}/{{a,b}}/access.log"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"

[[watch]]
name      = "logs_off"
path      = "{0}/disabled/*"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
enabled   = false
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str(&cfg_text).expect("disabled mix parses");
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    let subs = rig.driver.engine.subs();
    let promoters = rig.driver.engine.promoters();
    assert!(
        subs.find_by_name("build").is_some(),
        "enabled static attached"
    );
    assert!(
        subs.find_by_name("build_off").is_none(),
        "disabled static skipped"
    );
    assert!(
        promoters.find_by_name("logs").is_some(),
        "enabled dynamic attached"
    );
    assert!(
        promoters.find_by_name("logs_off").is_none(),
        "disabled dynamic skipped"
    );
    assert_eq!(subs.len(), 1, "only the enabled static is attached");
    assert_eq!(promoters.len(), 1, "only the enabled dynamic is registered");

    // The enabled attaches left probes armed. Production always
    // loops to a shutdown tick, which drains them via
    // `begin_shutdown`; model that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// Reload that adds a fresh dynamic [[watch]] registers a Promoter
/// in the engine via the `Input::ConfigDiff` step.
#[test]
fn reload_added_promoter_registers_in_engine() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = String::new(); // empty config → no watches
    let new_text = format!(
        r#"
[[watch]]
name      = "logs"
path      = "{}/{{a,b}}/access.log"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.promoters().is_empty());

    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert!(rig.driver.engine.promoters().find_by_name("logs").is_some());
}

/// Reload that removes a dynamic [[watch]] reaps the Promoter from
/// the engine: the diff's name-keyed `removed` list drives the
/// `reap_promoter_inner`, after which `find_by_name` no longer
/// resolves.
#[test]
fn reload_removed_promoter_detaches_in_engine() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = format!(
        r#"
[[watch]]
name      = "logs"
path      = "{}/{{a,b}}/access.log"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new_text = String::new();
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.promoters().find_by_name("logs").is_some());

    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert!(rig.driver.engine.promoters().find_by_name("logs").is_none());
}

/// Reload that modifies a dynamic [[watch]] (e.g., changes the
/// command) replaces the old `PromoterId` with a freshly-minted
/// one, keyed by the same name. The engine processes the diff's
/// name-keyed `modified` entry as reap-then-attach, so
/// `find_by_name("logs")` resolves to a different id afterward.
#[test]
fn reload_modified_promoter_replaces_id_in_engine() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = format!(
        r#"
[[watch]]
name      = "logs"
path      = "{}/{{a,b}}/access.log"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new_text = format!(
        r#"
[[watch]]
name      = "logs"
path      = "{}/{{a,b}}/access.log"
actions   = [{{ exec = ["echo"] }}]
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    let old_pid = rig
        .driver
        .engine
        .promoters()
        .find_by_name("logs")
        .expect("Promoter 'logs' registered pre-reload");

    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    let new_pid = rig
        .driver
        .engine
        .promoters()
        .find_by_name("logs")
        .expect("Promoter 'logs' still registered post-reload");
    assert_ne!(new_pid, old_pid, "modify mints a fresh PromoterId");
}

/// Static→dynamic migration via path edit: a `[[watch]]` named
/// "foo" with a literal path edits to a glob path. `is_dynamic`
/// flips, so the diff emits `subs.removed + promoters.added`.
/// Engine registries converge: the static Sub vanishes from
/// `subs().by_name`; a Promoter appears in `promoters().by_name`.
#[test]
fn reload_static_to_dynamic_migration_swaps_engine_registries() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = format!(
        r#"
[[watch]]
name      = "foo"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new_text = format!(
        r#"
[[watch]]
name      = "foo"
path      = "{}/*"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().find_by_name("foo").is_some());
    assert!(rig.driver.engine.promoters().is_empty());

    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert!(
        rig.driver.engine.subs().find_by_name("foo").is_none(),
        "static `foo` reaped from the Sub registry",
    );
    assert!(
        rig.driver.engine.promoters().find_by_name("foo").is_some(),
        "dynamic `foo` registered in the Promoter registry",
    );
}

/// Reverse direction: a dynamic [[watch]] flips to a literal
/// path. Diff emits `promoters.removed + subs.added`; engine
/// registries mirror the swap.
#[test]
fn reload_dynamic_to_static_migration_swaps_engine_registries() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = format!(
        r#"
[[watch]]
name      = "foo"
path      = "{}/*"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new_text = format!(
        r#"
[[watch]]
name      = "foo"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, &initial_text).unwrap();
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().is_empty());
    assert!(rig.driver.engine.promoters().find_by_name("foo").is_some());

    std::fs::write(&cfg_path, &new_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert!(
        rig.driver.engine.promoters().find_by_name("foo").is_none(),
        "dynamic `foo` reaped from the Promoter registry",
    );
    assert!(
        rig.driver.engine.subs().find_by_name("foo").is_some(),
        "static `foo` registered in the Sub registry",
    );
}

/// `read_and_parse_config` on a valid file returns
/// `Some((Config, FileMeta))` with the parsed `[[watch]]` blocks
/// populated and `FileMeta` matching the on-disk lstat. Pins the
/// helper's happy-path contract — both the SIGHUP and the
/// auto-reload settle-expiry paths rely on this signature, and the
/// meta-rotation discipline in `handle_reload` depends on the
/// captured value being lstat-equivalent in the absence of
/// concurrent edits.
#[test]
fn read_and_parse_config_returns_some_on_valid_file() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(
        &cfg_path,
        format!(
            r#"
[[watch]]
name    = "build"
path    = "{}"
actions = [{{ exec = ["true"] }}]
"#,
            tmp.path().display(),
        ),
    )
    .unwrap();
    let config = Config::from_str("").expect("empty config parses");
    let rig = rig_for(config, cfg_path.clone());
    let (parsed_config, parsed_meta) = rig
        .driver
        .read_and_parse_config()
        .expect("valid file parses to Some");
    assert_eq!(parsed_config.watches.len(), 1);
    assert_eq!(parsed_config.watches[0].name, "build");
    // No concurrent edits between the helper's atomic capture and
    // this fresh path-level stat — both must observe the same
    // inode-level identity.
    let lstat = FileMeta::from_path(&cfg_path).expect("lstat ok");
    assert_eq!(parsed_meta, lstat);
    assert_ne!(
        parsed_meta,
        dummy_meta(),
        "captured meta is real, not the placeholder"
    );
}

/// SIGHUP reload that introduces a substantive diff (added watch)
/// rotates `loader.config_meta` to the post-edit lstat. Pins the
/// apply-branch half of the meta-rotation discipline.
#[test]
fn reload_rotates_config_meta_on_apply_branch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let v1_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let v2_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
settle    = "100ms"
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let initial = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    assert_eq!(
        rig.driver.loader.config_meta,
        dummy_meta(),
        "rig starts with placeholder meta",
    );

    // Substantive edit — diff is non-empty (one added watch).
    std::fs::write(&cfg_path, &v2_text).unwrap();
    let expected_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.loader.config_meta, expected_meta,
        "apply-branch reload rotates loader.config_meta to the on-disk identity",
    );
    // Confirm the apply branch ran (added "b" attached in engine).
    assert!(
        rig.driver.engine.subs().find_by_name("b").is_some(),
        "v2's added watch attached — apply-branch path was exercised",
    );
}

/// SIGHUP reload whose new content differs only in metadata
/// (re-write of identical bytes; mtime moves, content identical)
/// takes the empty-diff branch, but **must still rotate
/// `loader.config_meta`** — otherwise the auto-reload
/// settle filter would observe `lstat != stored_meta` on every
/// subsequent pulse for the same already-applied edit and loop
/// `handle_reload` against unchanged content. Pins the
/// empty-diff half of the meta-rotation discipline.
#[test]
fn reload_rotates_config_meta_on_empty_diff_branch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let cfg_text = format!(
        r#"
[[watch]]
name      = "build"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &cfg_text).unwrap();
    let initial = Config::from_str(&cfg_text).expect("v1 parses");

    let mut rig = rig_for(initial.clone(), cfg_path.clone());
    let _ = rig.driver.run_initial_attach();
    let sid_before = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");

    // Re-save the same content. Real `FileMeta::from_path` after
    // this returns nonzero inode + nonzero mode (file-type bits),
    // which is enough to distinguish from `dummy_meta()` and to
    // observe rotation.
    std::fs::write(&cfg_path, &cfg_text).unwrap();
    let expected_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");
    assert_ne!(
        expected_meta,
        dummy_meta(),
        "real lstat is non-placeholder — comparison is meaningful",
    );

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.loader.config_meta, expected_meta,
        "empty-diff reload rotates loader.config_meta — \
         skipping rotation here would loop the auto-reload settle filter",
    );
    // Confirm the empty-diff branch ran (loader state unchanged
    // semantically — same config; the attached Sub keeps its
    // identity, no reap/re-attach churn).
    assert_eq!(
        rig.driver.loader.current_config, initial,
        "v1 ≡ v1 → empty-diff branch was exercised",
    );
    let sid_after = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' still attached");
    assert_eq!(
        sid_before, sid_after,
        "SubId unchanged across empty-diff reload"
    );
}

/// Parse-fail with a SUCCESSFUL post-fail lstat MUST rotate
/// `loader.config_meta` to the post-fail value. Closes the
/// chmod-EACCES recovery loop:
///
/// 1. operator chmod 000 → daemon EACCES on parse → meta rotates
///    to the mode-000 lstat
/// 2. operator chmod 644 → next pulse's lstat (mode-644) differs
///    from stored → re-fires `handle_reload`
///
/// Without this rotation, stored meta would freeze at the
/// pre-tighten state and auto-recovery would silently break —
/// `FileMeta` keys on mode/uid/gid precisely so chmod-only
/// transitions surface.
///
/// This test exercises the parse-fail polarity via malformed TOML
/// (file exists → post-fail lstat succeeds); chmod-EACCES is the
/// same code path with a different errno. The complementary
/// post-fail-lstat-fails polarity (meta preserved) is pinned by
/// the auto-reload settle-expiry tests against unreachable paths.
#[test]
fn reload_parse_failure_rotates_meta_when_post_fail_lstat_succeeds() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let v1_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let v1_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");
    let v1_config = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(v1_config.clone(), cfg_path.clone());
    rig.driver.loader.config_meta = v1_meta;

    // Overwrite with malformed TOML — parse fails, but the file
    // still exists so the post-fail lstat succeeds. Sleep first so
    // mtime advances at least one nanosecond past `v1_meta` on
    // coarse-resolution filesystems.
    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&cfg_path, "not valid toml [[[").unwrap();
    let v2_lstat = FileMeta::from_path(&cfg_path).expect("v2 lstat ok");
    assert_ne!(
        v1_meta, v2_lstat,
        "v2 must lstat-differ from v1 — otherwise the rotation \
         couldn't be observed",
    );

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.loader.config_meta, v2_lstat,
        "parse-fail with successful post-fail lstat rotates meta — \
         closes the chmod-EACCES recovery loop",
    );
    assert_eq!(
        rig.driver.loader.current_config, v1_config,
        "parse-fail preserves the running config — only meta advanced",
    );
}

/// A reload whose requested destination/path differs from the
/// running runtime values MUST NOT rotate the running shape into
/// `loader.current_log` — the appender doesn't move at runtime, so
/// the rotated value reflects what is *applied* (the running shape),
/// not what was requested.
///
/// Load-bearing structural invariant behind the phantom-warning
/// fix: the next reload's `apply_log_reload` compares against this
/// preserved running shape, so a destination flip-back to the
/// daemon's startup value sees `applied == requested` and refrains
/// from re-firing "restart to apply" on every reload until restart.
/// Asserting the shape invariant pins the cause; the warning
/// suppression is a downstream consequence.
#[test]
fn reload_with_destination_mismatch_preserves_running_log() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let log_path = tmp.path().join("specter.log");

    let v_file = format!(
        r#"
[log]
destination = "file"
path        = "{}"
level       = "info"
"#,
        log_path.display(),
    );
    let v_stderr = r#"
[log]
destination = "stderr"
level       = "info"
"#;

    std::fs::write(&cfg_path, &v_file).unwrap();
    let initial = Config::from_str(&v_file).expect("initial parses");
    let initial_log = initial.log.clone();

    let mut rig = rig_for(initial, cfg_path.clone());
    rig.driver.loader.config_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");

    // Sanity: rig starts with the file-destination running shape.
    assert_eq!(
        rig.driver.loader.current_log.destination,
        specter_config::LogDestination::File,
    );

    // Edit to `dest = stderr`. `apply_log_reload` sees a mismatch
    // (requested stderr, running file), logs the "restart to apply"
    // error, and returns the *running* shape unchanged. The rotation
    // writes that running shape into `current_log`, so the
    // destination field stays at File across this reload.
    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&cfg_path, v_stderr).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.loader.current_log.destination,
        specter_config::LogDestination::File,
        "destination request must not rotate the running destination",
    );
    assert_eq!(
        rig.driver.loader.current_log.path, initial_log.path,
        "running path preserved alongside the destination",
    );
}

// ===== Auto-reload settle pipeline =====
//
// Tests below exercise the `config_event` channel + `tick`'s drain
// step + the `apply_config_settle_expiry` helper end-to-end, with
// pulses driven by hand (no watcher backend wired yet). The helper
// takes an explicit `now: Instant` so tests can span the 100 ms
// settle window deterministically without sleeping.

/// `tick`'s config-event drain converts a pulse into an armed
/// settle deadline. Settle window = `now + CONFIG_SETTLE` (100 ms),
/// so the freshly-armed deadline lies in the future relative to
/// the tick's `Instant::now()`.
#[test]
fn config_event_pulse_via_tick_arms_settle_window() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let before_tick = Instant::now();
    rig.config_event_tx.try_send(()).expect("pulse send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    let armed = rig
        .driver
        .config_settle_until
        .expect("settle armed by drain");
    // Lower bound: the tick captured `now` at-or-after `before_tick`,
    // so the armed deadline is at-or-after `before_tick + 100ms`.
    assert!(
        armed >= before_tick + Duration::from_millis(100),
        "armed deadline must be at least CONFIG_SETTLE in the future",
    );
    // Upper bound (sanity): the deadline isn't far in the future
    // (allow a generous 1 s slack for slow CI).
    assert!(
        armed <= Instant::now() + Duration::from_secs(1),
        "armed deadline shouldn't drift more than a second past now",
    );
}

/// Settle resets per pulse. Two consecutive ticks each draining a
/// pulse push the deadline strictly forward — a sustained editor
/// burst defers the reload window until quiet.
#[test]
fn repeat_config_pulses_via_tick_defer_settle_expiry() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    rig.config_event_tx.try_send(()).expect("first pulse");
    // Drain-side check only — `tick`'s outcome is incidental (no
    // shutdown queued); the test asserts on `config_settle_until`.
    let _ = rig.driver.tick();
    let t1 = rig.driver.config_settle_until.expect("first settle armed");

    // Yield enough time for `Instant::now()` to advance — `tick()`
    // captures `now` afresh per call, so we just need a measurable
    // delta. Sleeping `2 ms` is well within scheduler granularity
    // on every supported platform.
    std::thread::sleep(Duration::from_millis(2));

    rig.config_event_tx.try_send(()).expect("second pulse");
    let _ = rig.driver.tick();
    let t2 = rig.driver.config_settle_until.expect("second settle armed");

    assert!(
        t2 > t1,
        "second pulse defers the deadline (t1={t1:?}, t2={t2:?})",
    );
}

/// Helper short-circuits when no pulse has armed the deadline.
/// Pre-pulse state must remain unchanged (no spurious reload).
#[test]
fn apply_config_settle_expiry_no_op_when_unarmed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config.clone(), cfg_path);

    let snapshot_meta = rig.driver.loader.config_meta;
    let _ = rig.driver.apply_config_settle_expiry(Instant::now());
    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(
        rig.driver.loader.config_meta, snapshot_meta,
        "unarmed expiry must not touch loader.config_meta",
    );
    assert_eq!(rig.driver.loader.current_config, config);
}

/// Helper short-circuits when `now < deadline`. Deadline stays
/// armed so a future tick (after the window elapses) can fire it.
#[test]
fn apply_config_settle_expiry_no_op_within_window() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let now = Instant::now();
    let deadline = now + Duration::from_millis(50);
    rig.driver.config_settle_until = Some(deadline);

    let _ = rig.driver.apply_config_settle_expiry(now);

    assert_eq!(
        rig.driver.config_settle_until,
        Some(deadline),
        "in-window call must not clear the deadline",
    );
}

/// `now == deadline` is the boundary case for the `>=` test in the
/// helper. The deadline is consumed (cleared); the lstat filter
/// then runs and (against an unchanged file) silent-drops.
#[test]
fn apply_config_settle_expiry_fires_at_exact_deadline() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    // Empty config on disk so the lstat capture has a real meta to
    // compare against.
    std::fs::write(&cfg_path, "").unwrap();
    let config = Config::from_str("").expect("empty config parses");
    let real_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");

    let mut rig = rig_for(config, cfg_path);
    rig.driver.loader.config_meta = real_meta;

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);

    let _ = rig.driver.apply_config_settle_expiry(deadline);

    assert_eq!(
        rig.driver.config_settle_until, None,
        "exact-deadline match clears the slot",
    );
    assert_eq!(
        rig.driver.loader.config_meta, real_meta,
        "lstat agreed with stored meta — no reload, meta unchanged",
    );
}

/// Settle expiry whose lstat agrees with `loader.config_meta`
/// silently drops the pulse: no `handle_reload`, no parse, no log
/// (beyond TRACE). This is the kqueue-parent-spillover case — a
/// sibling write fires a pulse, settle expires, lstat shows the
/// config file is unchanged → skip.
#[test]
fn apply_config_settle_expiry_skips_reload_on_unchanged_meta() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let cfg_text = format!(
        r#"
[[watch]]
name      = "build"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &cfg_text).unwrap();
    let real_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");
    let initial = Config::from_str(&cfg_text).expect("v1 parses");

    let mut rig = rig_for(initial.clone(), cfg_path);
    // Seed loader.config_meta to the real on-disk lstat so the
    // helper's `m != self.loader.config_meta` returns false.
    rig.driver.loader.config_meta = real_meta;
    let _ = rig.driver.run_initial_attach();
    let sid_snapshot = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");

    // Fire expiry with a `now` past the deadline — helper takes the
    // `now >= deadline` branch.
    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));
    // Settle slot consumed even on a silent drop (the deadline was
    // serviced; future pulses arm a fresh window).
    assert_eq!(rig.driver.config_settle_until, None);
    // No reload ⇒ loader state untouched.
    assert_eq!(
        rig.driver.loader.config_meta, real_meta,
        "silent-drop does not rotate config_meta",
    );
    assert_eq!(
        rig.driver.loader.current_config, initial,
        "silent-drop does not rotate current_config",
    );
    let sid_after = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' still attached");
    assert_eq!(
        sid_snapshot, sid_after,
        "silent-drop does not perturb attached Sub ids",
    );

    // The initial attach left a Seed-Verifying probe armed.
    // Production always loops to a shutdown tick, which drains it
    // via `begin_shutdown`; model that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// Settle expiry whose lstat detects drift (file was edited)
/// triggers `handle_reload`, which rotates `loader.config_meta`
/// and `loader.current_config`. The end-to-end gate for the
/// drift-driven auto-reload path.
#[test]
fn apply_config_settle_expiry_triggers_reload_on_meta_drift() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let v1_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let v1_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");
    let v1_config = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(v1_config, cfg_path.clone());
    rig.driver.loader.config_meta = v1_meta;
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_none());

    // Edit the file — atomic write replaces inode; mtime moves.
    let v2_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    // Sleep briefly so the FS-resolved mtime ticks at least one
    // nanosecond past `v1_meta` even on coarse-resolution FSs
    // (and so that the inode allocator doesn't reuse `v1`'s slot
    // immediately on FSs that recycle eagerly).
    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&cfg_path, &v2_text).unwrap();
    let v2_lstat = FileMeta::from_path(&cfg_path).expect("v2 lstat ok");
    assert_ne!(
        v1_meta, v2_lstat,
        "v2 must lstat-differ from v1 — otherwise the helper's filter \
         can't drive the reload",
    );

    // Force settle expiry.
    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));
    // Reload happened: settle consumed, meta rotated to the
    // post-edit identity, config now has v2's "b" watch.
    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(
        rig.driver.loader.config_meta, v2_lstat,
        "drift-driven reload rotated config_meta to the v2 lstat",
    );
    assert!(
        rig.driver.engine.subs().find_by_name("a").is_some(),
        "v2 still has watch 'a' — preserved across reload",
    );
    assert!(
        rig.driver.engine.subs().find_by_name("b").is_some(),
        "v2's added watch 'b' attached during reload",
    );
    assert_eq!(rig.driver.loader.current_config.watches.len(), 2);

    // The initial attach and the drift-driven reload left
    // Seed-Verifying probes armed. Production always loops to a
    // shutdown tick, which drains them via `begin_shutdown`; model
    // that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// Lstat error (file missing, EACCES on parent, etc.) routes
/// through the "treat-as-changed" branch: helper calls
/// `handle_reload`, which fails to read, logs, and preserves
/// loader state. The settle slot is consumed (no internal looping);
/// the next pulse fires a fresh attempt.
///
/// Specifically pins the **post-fail-lstat-also-fails** sub-case
/// of `handle_reload`'s parse-fail rotation: when the open fails
/// AND the post-fail lstat fails too, there is no fresh meta to
/// rotate to, so the existing meta is preserved. The next pulse's
/// `config_meta_changed` treats lstat-Err as drift and re-fires
/// the retry loop.
#[test]
fn apply_config_settle_expiry_treats_missing_path_as_changed() {
    // Path that cannot be lstat'd: `/dev/null` is a character
    // device, so `lstat("/dev/null/no/such")` returns ENOTDIR.
    // Both the open-for-parse and the post-fail lstat hit the
    // same error, so the post-fail rotation is a no-op.
    let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config.clone(), cfg_path);
    let pre_meta = rig.driver.loader.config_meta;

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));
    // Settle slot is consumed even when lstat fails — the helper
    // doesn't loop on its own; the next external pulse arms a
    // fresh window.
    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(
        rig.driver.loader.config_meta, pre_meta,
        "parse-fail must not rotate meta — would suppress retry pulses",
    );
    assert_eq!(rig.driver.loader.current_config, config);
}

/// End-to-end gate: pulse → tick (drain arms settle) →
/// helper-driven expiry → reload runs against drift. Pins the
/// drain step's interaction with the helper without spinning
/// 100 ms of real time.
#[test]
fn pulse_then_helper_expiry_runs_full_pipeline_on_drift() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let v1_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let v1_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");
    let v1_config = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(v1_config, cfg_path.clone());
    rig.driver.loader.config_meta = v1_meta;
    let _ = rig.driver.run_initial_attach();

    // Edit the file.
    std::thread::sleep(Duration::from_millis(10));
    let v2_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v2_text).unwrap();

    // Drain arms settle.
    rig.config_event_tx.try_send(()).expect("pulse send");
    let _ = rig.driver.tick();
    let armed = rig.driver.config_settle_until.expect("drain armed settle");

    // Force-expire via the helper. (Skirts the 100ms wall-clock
    // wait that an end-to-end-with-tick test would need; the
    // helper's contract is identical regardless of who calls it.)
    let _ = rig
        .driver
        .apply_config_settle_expiry(armed + Duration::from_millis(1));
    assert_eq!(rig.driver.config_settle_until, None);
    assert!(
        rig.driver.engine.subs().find_by_name("b").is_some(),
        "drift-driven reload attached the new watch",
    );
    assert_ne!(
        rig.driver.loader.config_meta, v1_meta,
        "post-reload meta must differ from the pre-edit identity",
    );

    // The initial attach and the drift-driven reload left
    // Seed-Verifying probes armed. Production always loops to a
    // shutdown tick, which drains them via `begin_shutdown`; model
    // that here before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

// ===== DriverState reload counters =====
//
// `DriverState::record_reload` is the sole writer for
// `reload_count` / `last_reload_at` / `last_reload_via`; the bump
// fires inside `handle_reload` immediately after
// `read_and_parse_config` returns `Some`. Three critical
// scenarios pin the contract:
//
// - SIGHUP-driven apply-diff reload bumps with `trigger = Sighup`
//   (exercises the `tick`-side caller wiring).
// - Auto-reload settle-drift reload bumps with `trigger = AutoReload`
//   (exercises the `apply_config_settle_expiry`-side caller wiring).
// - Parse-fail reload (either trigger) does NOT bump — the early
//   return short-circuits upstream of the record call.
//
// The empty-diff success branch and the SIGHUP-driven apply-diff
// branch reach the same bump line; one apply-diff test suffices to
// pin both. The parse-fail no-bump assertion is structurally
// trigger-agnostic (the record is never called); one trigger's
// coverage suffices.

/// SIGHUP against a substantive diff: the post-parse bump records
/// the operator pulse with `trigger = Sighup`, count moves from 0
/// to 1, and the wall-clock stamp is populated.
#[test]
fn handle_reload_via_sighup_bumps_counters_with_sighup_trigger() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let v1_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let v2_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let initial = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    let _ = rig.driver.run_initial_attach();

    // Pre-bump: fresh-process zero state.
    assert_eq!(rig.driver.driver_state.reload_count, 0);
    assert!(rig.driver.driver_state.last_reload_at.is_none());
    assert!(rig.driver.driver_state.last_reload_via.is_none());

    // Substantive edit + SIGHUP pulse.
    std::fs::write(&cfg_path, &v2_text).unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.driver_state.reload_count, 1,
        "successful SIGHUP reload bumps the counter",
    );
    assert!(
        rig.driver.driver_state.last_reload_at.is_some(),
        "successful SIGHUP reload stamps the wall-clock",
    );
    assert_eq!(
        rig.driver.driver_state.last_reload_via,
        Some(ReloadTrigger::Sighup),
        "tick.rs caller threads `Sighup` into handle_reload",
    );
}

/// Auto-reload settle expiry against drifted on-disk meta:
/// `apply_config_settle_expiry` calls
/// `handle_reload(AutoReload, _)`, the post-parse bump records the
/// trigger as `AutoReload`. Pins the second caller's threading.
#[test]
fn handle_reload_via_auto_reload_bumps_counters_with_auto_reload_trigger() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let v1_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let v1_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");
    let v1_config = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(v1_config, cfg_path.clone());
    rig.driver.loader.config_meta = v1_meta;
    let _ = rig.driver.run_initial_attach();

    // Pre-bump: fresh-process zero state.
    assert_eq!(rig.driver.driver_state.reload_count, 0);
    assert_eq!(rig.driver.driver_state.last_reload_via, None);

    // Edit the file so the lstat filter detects drift.
    let v2_text = format!(
        r#"
[[watch]]
name      = "a"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "b"
path      = "{0}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&cfg_path, &v2_text).unwrap();

    // Force settle expiry — bypasses the 100ms wall-clock wait;
    // the helper's contract is the same regardless of caller.
    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));

    assert_eq!(
        rig.driver.driver_state.reload_count, 1,
        "successful auto-reload bumps the counter",
    );
    assert!(rig.driver.driver_state.last_reload_at.is_some());
    assert_eq!(
        rig.driver.driver_state.last_reload_via,
        Some(ReloadTrigger::AutoReload),
        "apply_config_settle_expiry threads `AutoReload` into handle_reload",
    );

    // The drift-driven reload left a Seed-Verifying probe armed —
    // drain via begin_shutdown before the rig drops.
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// Parse-fail reload (file present, malformed TOML): the early
/// return in `handle_reload` short-circuits before
/// `record_reload`, so the counters stay at their fresh-process
/// zero. Pins the no-bump-on-failure half of the contract.
#[test]
fn handle_reload_does_not_bump_counters_on_parse_fail() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    // Write malformed TOML — parse fails, but the file exists so
    // the post-fail lstat succeeds (covers the meta-rotation
    // branch without bumping reload counters).
    std::fs::write(&cfg_path, "not valid toml [[[").unwrap();
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.driver_state.reload_count, 0,
        "parse-fail must not bump the counter",
    );
    assert!(
        rig.driver.driver_state.last_reload_at.is_none(),
        "parse-fail must not stamp the wall-clock",
    );
    assert!(
        rig.driver.driver_state.last_reload_via.is_none(),
        "parse-fail must not record a trigger",
    );
}

// ===== Outbound disconnect + shutdown-race + attach-break drain =====
//
// Defensive Terminal symmetry on the other inbound arms (`reload_signal_rx`,
// `config_event_rx`) is covered structurally by
// `effect_in_disconnect_shuts_down` — the three `Disconnected` match arms
// in `tick.rs` are byte-identical.

/// A downstream `watch_ops_tx` disconnect is a kernel-drift hazard —
/// the engine's state would silently diverge from the watcher's view,
/// wedging future bursts in `Verifying` against stale baselines. It
/// is terminal: `forward`'s per-send `crossbeam::select!` observes
/// `Err` on the disconnected send arm and returns `Break`, which
/// routes the tick through `begin_shutdown`.
#[test]
fn watch_ops_disconnect_shuts_down() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    // Capture the attach request before `config` moves into the rig.
    let req = config
        .active_watches()
        .next()
        .expect("one active watch")
        .to_attach_request();
    let mut rig = rig_for(config, cfg_path);

    // Disconnect `watch_ops_rx` (and drop the watcher-side
    // `sensor_in_tx` clone; the rig keeps its own clone for the
    // AttachSub queue below). The driver's `watch_ops_tx` clone in
    // `engine_side` is still alive — sends from `forward` will
    // observe Err, not Disconnected-on-send-side.
    let WatcherSide {
        watch_ops_rx,
        sensor_in_tx,
    } = rig.watcher_side;
    drop(watch_ops_rx);
    drop(sensor_in_tx);

    // Queue an AttachSub via `sensor_in_tx`. `drain_sensor`'s barrier
    // arm runs `engine.step`, which emits a Watch op; `forward`'s
    // select! send arm fires with `Err`, `forward` returns `Break`,
    // `drain_sensor` returns `Break(())`, and `tick` routes through
    // `begin_shutdown`.
    rig.sensor_in_tx
        .send(Input::AttachSub(req))
        .expect("attach send");

    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
}

/// A full `watch_ops_tx` (bounded(1024)) combined with a queued
/// shutdown pulse must not deadlock. The per-send `crossbeam::select!`
/// in `forward` races `send` vs `recv(shutdown_engine_rx)` — the
/// queued shutdown wins and `forward` returns `Break` without
/// dropping into a blocking send.
#[test]
fn forward_shutdown_wins_when_full_channel_signaled() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let rig = rig_for(config, cfg_path);
    let TestRig {
        mut driver,
        watcher_side,
        shutdown_tx,
        watch_ops_tx,
        ..
    } = rig;

    // Fill `watch_ops_tx` to capacity from the test's clone. The
    // driver's clone targets the same bounded channel, so a
    // subsequent send from `forward` would block.
    for _ in 0..1024 {
        watch_ops_tx
            .try_send(WatchOp::Unwatch {
                resource: specter_core::ResourceId::default(),
            })
            .expect("first 1024 fit");
    }

    // Pre-pulse the shutdown receiver so its select arm is
    // immediately ready when `forward` races.
    shutdown_tx.try_send(()).expect("shutdown pulse");

    // Non-empty `StepOutput`: the send arm would block on a full
    // channel; the recv-on-shutdown arm wins and `Break` is returned.
    let mut out = StepOutput::default();
    out.watch_ops.push(WatchOp::Watch {
        resource: specter_core::ResourceId::default(),
        path: Arc::from(PathBuf::from("/p/0")),
        kind: specter_core::ResourceKind::Unknown,
        events: specter_core::ClassSet::EMPTY,
    });
    let outcome = driver.forward(out);
    assert_eq!(outcome, ControlFlow::Break(()));

    // Drop in safe order: `watcher_side` outlived `forward` so the
    // bounded channel was full-and-blocking, not full-and-disconnected.
    drop(driver);
    drop(watcher_side);
}

/// Forward dispatches a `StepOutput` with both `cancel_effects` and
/// `effects` populated as `EffectOp::Cancel` then `EffectOp::Submit`
/// on the same `effects_tx` channel, cancels first.
///
/// Pins the wire contract: the engine→actuator channel carries
/// `EffectOp`, and cancel-effects (gate-deadline abandonment)
/// dispatch *before* any submit in the same step. The same-step
/// collision is unconstructable in production (`handle_gate_deadline`
/// emits no Effects; fire-and-settle emits no cancel), but a future
/// emission site that crossed the two streams would inherit the
/// right "kill stale before spawn new" ordering from this dispatch
/// shape.
#[test]
fn forward_dispatches_cancel_before_submit_on_effects_tx() {
    use slotmap::KeyData;
    use specter_core::testkit::single_exec_program;
    use specter_core::{
        ArgPart, ArgTemplate, CorrelationId, Effect, EffectCommon, EffectOp, ProfileId,
    };

    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let rig = rig_for(config, cfg_path);
    let TestRig {
        mut driver,
        actuator_side,
        ..
    } = rig;

    // Two distinct profile ids; BTreeSet iteration order is intrinsic
    // to the key (`KeyData` Ord), so pid_a < pid_b yields pid_a first.
    let pid_a = ProfileId::from(KeyData::from_ffi(0x10));
    let pid_b = ProfileId::from(KeyData::from_ffi(0x20));

    // Minimal Effect — `EffectOp::Submit` is the dominant variant
    // width, but the dispatch contract under test is the order, not
    // the Submit payload's content.
    let submit_effect = {
        let common = EffectCommon {
            sub: specter_core::SubId::default(),
            profile: ProfileId::from(KeyData::from_ffi(0x30)),
            anchor: specter_core::ResourceId::default(),
            correlation: CorrelationId::default(),
            forced: false,
            capture_output: false,
            sub_name: compact_str::CompactString::new(""),
            program: single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])]),
            anchor_path: Arc::from(PathBuf::new()),
            anchor_kind: specter_core::ResourceKind::Dir,
            exclude: Arc::from(Vec::<compact_str::CompactString>::new()),
        };
        Effect::subtree(common, None)
    };

    let mut out = StepOutput::default();
    out.push_cancel_effect(pid_a);
    out.push_cancel_effect(pid_b);
    out.push_effect(submit_effect);
    out.sort_for_emission();

    let outcome = driver.forward(out);
    assert_eq!(outcome, ControlFlow::Continue(()));

    // Drain `effects_rx` and assert the order. Cancels first
    // (pid_a < pid_b by `KeyData` Ord), then the Submit.
    let mut received: Vec<EffectOp> = Vec::new();
    while let Ok(op) = actuator_side.effects_rx.try_recv() {
        received.push(op);
    }
    assert_eq!(
        received.len(),
        3,
        "two cancels + one submit reached the wire"
    );
    assert!(
        matches!(received[0], EffectOp::Cancel { profile } if profile == pid_a),
        "first dispatched op must be the lower-keyed cancel; got {:?}",
        received[0],
    );
    assert!(
        matches!(received[1], EffectOp::Cancel { profile } if profile == pid_b),
        "second dispatched op must be the higher-keyed cancel; got {:?}",
        received[1],
    );
    assert!(
        matches!(received[2], EffectOp::Submit(_)),
        "submit dispatches after all cancels (defense in depth); got {:?}",
        received[2],
    );

    // Drop in safe order — keep the driver alive past actuator_side
    // (we drained it but never disconnected). The probe slot armed
    // here? None — `forward` doesn't touch probes for this fixture.
    drop(driver);
    drop(actuator_side);
}

/// When `run_initial_attach`'s `forward` returns `Break`, the
/// Seed-Verifying probe slot armed by the just-attached Profile must
/// be disarmed before returning — otherwise dropping the driver
/// would trip `ProbeSlot::drop`'s linear-edge tripwire. The
/// assertion is structural: `drop(rig.driver)` must not panic.
#[test]
fn run_initial_attach_break_drains_probes() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    // Disconnect `watch_ops_rx` before attach. The first `forward`
    // inside `run_initial_attach` hits a disconnected `watch_ops_tx`
    // and returns `Break`; `run_initial_attach` then calls
    // `begin_shutdown` and returns `Break`.
    let WatcherSide {
        watch_ops_rx,
        sensor_in_tx,
    } = rig.watcher_side;
    drop(watch_ops_rx);
    drop(sensor_in_tx);

    let outcome = rig.driver.run_initial_attach();
    assert_eq!(
        outcome,
        ControlFlow::Break(()),
        "watch_ops disconnect during attach ⇒ Break",
    );

    // begin_shutdown ran inside run_initial_attach; the probe slot
    // armed by the attached Seed is disarmed. Dropping the driver
    // here must not trip the `ProbeSlot::drop` linear-edge tripwire.
    drop(rig.driver);
}

// ===== IPC drain (driver/ipc.rs) =====
//
// Exercises [`super::EngineDriver::drain_ipc`] and `handle_ipc`
// through the rig's `ipc_request_tx` clone. Validates:
//
// - `Status` projection round-trips through the reply channel.
// - `Subscribe { name: None }` adds an unfiltered subscriber and
//   acknowledges synchronously.
// - `Subscribe { name: Some(unknown) }` returns `Err {
//   ERR_UNKNOWN_SUB }` without touching the broker.
// - `Subscribe { name: Some(attached) }` resolves the name to a
//   `SubId`, registers the subscriber with that filter, and acks
//   carrying the resolved `WireId`.
// - `Reload` routes through the driver-side pipeline and bumps
//   `driver_state.reload_count` + `last_reload_via = Ipc`.
// - Empty IPC queue returns `Continue`; disconnected producer
//   returns `Break`.

use crate::ipc::protocol::{ERR_UNKNOWN_SUB, IpcRequest, RequestPayload, ResponsePayload, WireId};
use compact_str::CompactString;

/// Mint a `bounded(1)` reply channel + wrap a payload in an
/// `IpcRequest`. Mirrors the per-conn thread's `reply_tx`
/// discipline so the test rig observes the same reply window the
/// production handler does.
fn ipc_request_with_reply(
    payload: RequestPayload,
) -> (IpcRequest, crossbeam::channel::Receiver<ResponsePayload>) {
    let (reply_tx, reply_rx) = crossbeam::channel::bounded::<ResponsePayload>(1);
    (IpcRequest { payload, reply_tx }, reply_rx)
}

/// Empty queue ⇒ `drain_ipc` returns `Continue` immediately. The
/// `try_recv` Empty arm is the production fast-path on every tick
/// with no operator-IPC pressure.
#[test]
fn drain_ipc_empty_returns_continue() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(
        outcome,
        ControlFlow::Continue(()),
        "no queued requests ⇒ Continue (drain is a no-op)",
    );
}

/// Producer disconnect ⇒ `drain_ipc` returns `Break`. The IPC
/// server thread is the sole producer; its death means the bin's
/// shutdown path is in flight.
#[test]
fn drain_ipc_disconnect_returns_break() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    // Drop the producer-side sender — the driver's `ipc_request_rx`
    // observes Disconnected on the next `try_recv`.
    drop(rig.ipc_request_tx);

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(
        outcome,
        ControlFlow::Break(()),
        "disconnected producer ⇒ Break (shutdown in flight)",
    );
}

/// `Status` round-trips through the reply channel. The reply
/// carries a `ResponsePayload::Status` whose `socket_path`
/// matches the rig's synthetic path — proves the projection ran
/// against the driver's actual state, not a hard-coded fixture.
#[test]
fn drain_ipc_status_replies_with_projection() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Status);
    rig.ipc_request_tx.send(req).expect("queue status request");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    let reply = reply_rx.try_recv().expect("status reply present");
    match reply {
        ResponsePayload::Status(status) => {
            assert_eq!(
                status.socket_path,
                PathBuf::from("/tmp/specter-test.sock"),
                "projection read the driver's socket_path",
            );
            assert_eq!(status.sub_total, 0, "no subs attached in this fixture");
        }
        other => panic!("expected Status, got {other:?}"),
    }
}

/// `Subscribe { name: None }` adds an unfiltered subscriber and
/// replies `SubscribeAck { sub: None }`. The broker is left with
/// exactly one subscriber after the drain.
#[test]
fn drain_ipc_subscribe_unfiltered_adds_subscriber() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (event_tx, _event_rx) = crossbeam::channel::bounded(16);
    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Subscribe {
        tx: event_tx,
        name: None,
    });
    rig.ipc_request_tx.send(req).expect("queue subscribe");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    let reply = reply_rx.try_recv().expect("subscribe ack present");
    match reply {
        ResponsePayload::SubscribeAck { sub: None } => {} // OK
        other => panic!("expected SubscribeAck(None), got {other:?}"),
    }
    // The broker now holds one subscriber — `forward()` would
    // dispatch every future diagnostic to its channel.
    assert_eq!(
        rig.driver.broker.len(),
        1,
        "broker holds the new unfiltered subscriber",
    );
}

/// `Subscribe { name: Some("nope") }` against an empty engine
/// returns `Err { ERR_UNKNOWN_SUB }` and DOES NOT register a
/// subscriber. The race window is closed structurally: the
/// resolve happens on the driver thread, atomic with
/// `add_subscriber`, so the client cannot observe an in-between
/// state.
#[test]
fn drain_ipc_subscribe_unknown_name_errors() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (event_tx, _event_rx) = crossbeam::channel::bounded(16);
    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Subscribe {
        tx: event_tx,
        name: Some(CompactString::const_new("nope")),
    });
    rig.ipc_request_tx.send(req).expect("queue subscribe");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    let reply = reply_rx.try_recv().expect("err reply present");
    match reply {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, ERR_UNKNOWN_SUB);
            assert!(
                error.contains("no watch named nope"),
                "error carries the resolution detail; got {error:?}",
            );
        }
        other => panic!("expected Err(ERR_UNKNOWN_SUB), got {other:?}"),
    }
    assert_eq!(
        rig.driver.broker.len(),
        0,
        "unknown name MUST NOT add a subscriber",
    );
}

/// `Subscribe { name: Some("build") }` against a config with a
/// `build` watch attached resolves the name to a SubId, registers
/// the subscriber with that filter, and acks carrying the resolved
/// WireId. The add-before-ack ordering holds structurally: the
/// `add_subscriber` happens before the ack reaches the reply
/// channel.
#[test]
fn drain_ipc_subscribe_known_name_resolves_and_acks() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    // Attach the static sub.
    let _ = rig.driver.run_initial_attach();
    let expected_sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("build attached");

    let (event_tx, _event_rx) = crossbeam::channel::bounded(16);
    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Subscribe {
        tx: event_tx,
        name: Some(CompactString::const_new("build")),
    });
    rig.ipc_request_tx.send(req).expect("queue subscribe");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    let reply = reply_rx.try_recv().expect("subscribe ack present");
    match reply {
        ResponsePayload::SubscribeAck { sub: Some(wire_id) } => {
            assert_eq!(
                wire_id,
                WireId::from(expected_sid),
                "ack carries the resolved WireId",
            );
        }
        other => panic!("expected SubscribeAck(Some), got {other:?}"),
    }
    assert_eq!(rig.driver.broker.len(), 1, "filtered subscriber registered");

    // Drain probes before dropping — the attached sub armed a Seed
    // probe. begin_shutdown drains via cancel_all_in_flight_probes.
    let _ = rig.driver.begin_shutdown();
}

/// `Reload` via the IPC drain re-reads the on-disk config, rotates
/// the loader, bumps `reload_count`, and stamps `last_reload_via =
/// Ipc`. Empty-diff branch (the on-disk content matches the
/// initial-loaded content) — still a successful reload that
/// honours the operator's request.
#[test]
fn drain_ipc_reload_via_pipeline_records_ipc_trigger() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    // Write a tiny valid config so the reload's parse succeeds.
    std::fs::write(&cfg_path, "").expect("write empty config");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Reload);
    rig.ipc_request_tx.send(req).expect("queue reload");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    // The Reload arm ACKs `Ok` regardless of forward-side shutdown
    // observation — we got here without breaking, so the reload
    // applied.
    let reply = reply_rx.try_recv().expect("ok reply present");
    assert!(matches!(reply, ResponsePayload::Ok), "got {reply:?}");

    assert_eq!(
        rig.driver.driver_state.reload_count, 1,
        "reload_count bumped",
    );
    assert!(matches!(
        rig.driver.driver_state.last_reload_via,
        Some(ReloadTrigger::Ipc),
    ));
}

// ===== IPC Disable / Enable handlers =====
//
// Exercises [`super::EngineDriver::handle_disable`] /
// [`super::EngineDriver::handle_enable`] through the IPC drain.
// Each test pins one branch of the handlers' precondition gates or
// happy paths; together they cover every reply variant the operator
// can observe.
//
// Engine state assertions go through `engine.subs().find_by_name`
// (the static `by_name` index — the same surface the projection
// helpers query). `disabled_runtime` mutations are asserted against
// the rig's `driver.disabled_runtime` (accessible from this child
// module).

use crate::ipc::protocol::{ERR_DYNAMIC_SUB_NO_OP, ERR_NOT_DISABLED, ERR_TOML_DISABLED};

/// Disable happy path: the dynamic-shape gate passes (no `@`), the
/// engine resolves the name, the precondition passes (not yet
/// disabled), `disabled_runtime` records the name BEFORE the
/// engine's [`Input::DetachSub`] step, and the sub leaves
/// `engine.subs()`.
#[test]
fn drain_ipc_disable_static_sub_detaches_and_records_override() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().find_by_name("build").is_some());

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Disable {
        name: CompactString::const_new("build"),
    });
    rig.ipc_request_tx.send(req).expect("queue disable");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    let reply = reply_rx.try_recv().expect("ok reply present");
    assert!(matches!(reply, ResponsePayload::Ok), "got {reply:?}");
    assert!(
        rig.driver.engine.subs().find_by_name("build").is_none(),
        "engine detached the sub",
    );
    assert!(
        rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("build")),
        "runtime override recorded",
    );

    // Detach reaps the Profile; no probe remains armed. Belt-and-
    // braces drain in case the Seed-Verifying probe armed on attach
    // wasn't already disarmed by the detach path.
    let _ = rig.driver.begin_shutdown();
}

/// Disable against a name the engine doesn't know returns
/// [`ERR_UNKNOWN_SUB`] with no state mutation.
#[test]
fn drain_ipc_disable_unknown_name_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Disable {
        name: CompactString::const_new("ghost"),
    });
    rig.ipc_request_tx.send(req).expect("queue disable");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    match reply_rx.try_recv().expect("err reply present") {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, ERR_UNKNOWN_SUB);
            assert!(error.contains("no watch named ghost"));
        }
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(
        rig.driver.disabled_runtime.is_empty(),
        "unknown name MUST NOT touch the override set",
    );
}

/// Disable against an `@`-bearing name with no registry entry is
/// refused with [`ERR_UNKNOWN_SUB`] — a typo (operator addressing a
/// non-existent dynamic-shape name) reports the structural truth
/// (the name does not resolve), not a misleading dynamic-sub
/// classification. The dynamic-vs-static discrimination is a
/// property of the resolved Sub; this case never reaches that gate
/// because the lookup is empty.
#[test]
fn drain_ipc_disable_unknown_dynamic_shape_name_returns_unknown_sub() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Disable {
        name: CompactString::const_new("promoter@/tmp/x"),
    });
    rig.ipc_request_tx.send(req).expect("queue disable");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    match reply_rx.try_recv().expect("err reply present") {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, ERR_UNKNOWN_SUB);
            assert!(error.contains("no watch named promoter@/tmp/x"));
        }
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(rig.driver.disabled_runtime.is_empty());
}

/// Disable against a real dynamic (promoter-spawned) Sub returns
/// [`ERR_DYNAMIC_SUB_NO_OP`] — the gate reads `source_promoter` off
/// the resolved Sub, not the lexical shape of the name. The dynamic
/// Sub stays in the engine; `disabled_runtime` is not touched.
#[test]
fn drain_ipc_disable_dynamic_sub_returns_dynamic_no_op() {
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, ClassSet, EffectScope, ExecAction, ProfileIdentity,
        PromoterId, ScanConfig, SubAttachAnchor, SubAttachRequest, SubParams,
    };

    // Build a trivial single-op program inline — the dynamic-Sub
    // path only needs one exec stub for the SubParams to be valid.
    fn trivial_program() -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal("/bin/true")])],
            None,
        )));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    // Inject a dynamic Sub directly — production-side these are
    // minted by Promoter::try_promote, but the disable handler reads
    // only `source_promoter` on the resolved Sub, so a synthetic
    // attach is sufficient to pin the gate.
    let dynamic_name = "promoter@/tmp/dyn_anchor";
    let req = SubAttachRequest::from_parts(
        SubAttachAnchor::Path(PathBuf::from("/tmp/dyn_anchor")),
        ProfileIdentity {
            config: ScanConfig::builder().build(),
            max_settle: Duration::from_hours(1),
            events: ClassSet::DEFAULT_SUBTREE_ROOT,
        },
        SubParams {
            name: CompactString::const_new(dynamic_name),
            program: trivial_program(),
            scope: EffectScope::SubtreeRoot,
            settle: Duration::from_millis(100),
            log_output: false,
            source_promoter: Some(PromoterId::default()),
        },
    );
    let _ = rig
        .driver
        .engine
        .step(Input::AttachSub(req), Instant::now());
    assert!(
        rig.driver
            .engine
            .subs()
            .find_by_name(dynamic_name)
            .is_some(),
        "dynamic Sub indexed under the new contract",
    );

    let (ipc_req, reply_rx) = ipc_request_with_reply(RequestPayload::Disable {
        name: CompactString::const_new(dynamic_name),
    });
    rig.ipc_request_tx.send(ipc_req).expect("queue disable");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    match reply_rx.try_recv().expect("err reply present") {
        ResponsePayload::Err { code, .. } => {
            assert_eq!(code, ERR_DYNAMIC_SUB_NO_OP);
        }
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(
        rig.driver.disabled_runtime.is_empty(),
        "dynamic-sub gate must not pollute disabled_runtime",
    );
    assert!(
        rig.driver
            .engine
            .subs()
            .find_by_name(dynamic_name)
            .is_some(),
        "engine state untouched on the dynamic-sub gate",
    );

    let _ = rig.driver.begin_shutdown();
}

/// Second `disable` for a name already in `disabled_runtime`
/// returns [`ERR_NOT_DISABLED`] (precondition violation) without
/// re-emitting a detach.
#[test]
fn drain_ipc_disable_already_disabled_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    // Pre-populate the override set as if a prior disable ran.
    // Engine still has the sub — the precondition check fires
    // before the engine state is mutated.
    rig.driver
        .disabled_runtime
        .insert(CompactString::const_new("build"));

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Disable {
        name: CompactString::const_new("build"),
    });
    rig.ipc_request_tx.send(req).expect("queue disable");
    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    match reply_rx.try_recv().expect("err reply present") {
        ResponsePayload::Err { code, .. } => assert_eq!(code, ERR_NOT_DISABLED),
        other => panic!("expected Err, got {other:?}"),
    }
    // Engine state untouched — the sub was never detached on this
    // path (no fresh DetachSub fired).
    assert!(
        rig.driver.engine.subs().find_by_name("build").is_some(),
        "second disable is a precondition-failure no-op on engine state",
    );

    let _ = rig.driver.begin_shutdown();
}

/// Enable happy path: the override is cleared AND
/// [`Input::AttachSub`] re-attaches the sub the TOML still carries
/// active.
#[test]
fn drain_ipc_enable_clears_override_and_reattaches() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    // Simulate a prior disable: override set + sub detached from
    // engine. Drive the disable handler so the rig models the
    // post-disable state faithfully.
    let (req, _) = ipc_request_with_reply(RequestPayload::Disable {
        name: CompactString::const_new("build"),
    });
    rig.ipc_request_tx.send(req).expect("queue disable");
    let _ = rig.driver.drain_ipc(Instant::now());
    assert!(rig.driver.engine.subs().find_by_name("build").is_none());
    assert_eq!(rig.driver.disabled_runtime.len(), 1);

    // Now enable: clear the override + re-attach.
    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Enable {
        name: CompactString::const_new("build"),
    });
    rig.ipc_request_tx.send(req).expect("queue enable");
    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    let reply = reply_rx.try_recv().expect("ok reply present");
    assert!(matches!(reply, ResponsePayload::Ok), "got {reply:?}");
    assert!(
        rig.driver.engine.subs().find_by_name("build").is_some(),
        "engine re-attached the sub",
    );
    assert!(
        rig.driver.disabled_runtime.is_empty(),
        "runtime override cleared",
    );

    let _ = rig.driver.begin_shutdown();
}

/// Enable against a name not in `disabled_runtime` returns
/// [`ERR_NOT_DISABLED`] without mutating state (no override to
/// clear, no engine step to fire).
#[test]
fn drain_ipc_enable_not_disabled_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Enable {
        name: CompactString::const_new("nothing"),
    });
    rig.ipc_request_tx.send(req).expect("queue enable");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    match reply_rx.try_recv().expect("err reply present") {
        ResponsePayload::Err { code, .. } => assert_eq!(code, ERR_NOT_DISABLED),
        other => panic!("expected Err, got {other:?}"),
    }
}

/// Enable against a runtime-disabled name whose TOML entry is no
/// longer active returns [`ERR_TOML_DISABLED`], BUT the runtime
/// override IS cleared as a side effect — the operator's
/// "no-longer-suppress" intent is honoured regardless of the
/// TOML's reattach gate.
#[test]
fn drain_ipc_enable_toml_disabled_clears_override_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    // Config with NO active watches — the override target has no
    // TOML anchor.
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    rig.driver
        .disabled_runtime
        .insert(CompactString::const_new("orphan"));

    let (req, reply_rx) = ipc_request_with_reply(RequestPayload::Enable {
        name: CompactString::const_new("orphan"),
    });
    rig.ipc_request_tx.send(req).expect("queue enable");

    let outcome = rig.driver.drain_ipc(Instant::now());
    assert_eq!(outcome, ControlFlow::Continue(()));

    match reply_rx.try_recv().expect("err reply present") {
        ResponsePayload::Err { code, .. } => assert_eq!(code, ERR_TOML_DISABLED),
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(
        rig.driver.disabled_runtime.is_empty(),
        "override cleared on the TOML-disabled failure path",
    );
}

// ===== disabled_runtime: diff filter + post-apply prune =====
//
// The bin keeps an in-memory set of operator-disabled Sub names that
// must survive every reload. Two driver-side gates carry the
// discipline:
//
// - [`super::EngineDriver::compute_watch_diff`] filters the four Sub
//   buckets BEFORE the engine sees them — an attach / re-attach /
//   re-bind / detach for a runtime-disabled Sub would churn the
//   engine on a Sub the operator already suppressed.
// - [`super::EngineDriver::prune_disabled_runtime_against_current_config`]
//   runs AFTER the loader rotation; it retains only those override
//   names whose `[[watch]]` entry still exists in the freshly-applied
//   TOML (regardless of `enabled`), so a TOML-disabled entry's
//   runtime override survives ("off twice") and a TOML-removed entry
//   evaporates.

/// Filter discipline: every name in `disabled_runtime` is stripped
/// from each of the four Sub-diff buckets simultaneously. Fixture
/// places one disabled name in EACH bucket — a regression dropping
/// any of the four `.retain` calls surfaces here.
#[test]
fn compute_watch_diff_filters_disabled_runtime_from_all_four_buckets() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let anchor = tmp.path().display();

    // Old config: three of the four eventually-disabled subs are
    // present so we can drive `removed` / `modified_identity` /
    // `modified_params`. The fourth lands as `added` against the new
    // config.
    let old_text = format!(
        r#"
[[watch]]
name      = "to_be_removed"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
max_settle = "500ms"

[[watch]]
name      = "to_be_modified_identity"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
max_settle = "500ms"

[[watch]]
name      = "to_be_modified_params"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
max_settle = "500ms"
"#,
    );
    let old = Config::from_str(&old_text).expect("old config parses");

    // New config: `to_be_removed` is gone; `to_be_modified_identity`
    // flips `max_settle` (an identity-partition field per
    // `requires_new_profile`); `to_be_modified_params` flips `settle`
    // (a params-only field); `to_be_added` is fresh.
    let new_text = format!(
        r#"
[[watch]]
name      = "to_be_modified_identity"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
max_settle = "1s"

[[watch]]
name      = "to_be_modified_params"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
settle    = "100ms"
max_settle = "500ms"

[[watch]]
name      = "to_be_added"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
settle    = "50ms"
max_settle = "500ms"
"#,
    );
    let new = Config::from_str(&new_text).expect("new config parses");

    let mut rig = rig_for(old, cfg_path);

    // Sanity: the unfiltered diff populates every bucket. Without
    // this, a fixture drift could pass a filter test that strips
    // already-empty buckets.
    let unfiltered = specter_config::diff(&rig.driver.loader.current_config, &new);
    assert_eq!(unfiltered.subs.added.len(), 1);
    assert_eq!(unfiltered.subs.removed.len(), 1);
    assert_eq!(unfiltered.subs.modified_identity.len(), 1);
    assert_eq!(unfiltered.subs.modified_params.len(), 1);

    // Disable every name the diff would touch.
    for name in [
        "to_be_added",
        "to_be_removed",
        "to_be_modified_identity",
        "to_be_modified_params",
    ] {
        rig.driver
            .disabled_runtime
            .insert(CompactString::const_new(name));
    }

    let filtered = rig.driver.compute_watch_diff(&new);
    assert!(filtered.subs.added.is_empty(), "added bucket filtered");
    assert!(filtered.subs.removed.is_empty(), "removed bucket filtered");
    assert!(
        filtered.subs.modified_identity.is_empty(),
        "modified_identity bucket filtered",
    );
    assert!(
        filtered.subs.modified_params.is_empty(),
        "modified_params bucket filtered",
    );
}

/// Filter discipline (negative case): names absent from
/// `disabled_runtime` pass through every bucket unchanged. Confirms
/// the filter is additive — daemons with an empty
/// `disabled_runtime` compute the same diff as if the filter were
/// not applied.
#[test]
fn compute_watch_diff_with_empty_disabled_runtime_passes_diff_through() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let initial = Config::from_str("").expect("empty config parses");
    let new_text = format!(
        r#"
[[watch]]
name      = "added"
path      = "{}"
actions   = [{{ exec = ["true"] }}]
"#,
        tmp.path().display(),
    );
    let new = Config::from_str(&new_text).expect("new config parses");

    let rig = rig_for(initial, cfg_path);
    let diff = rig.driver.compute_watch_diff(&new);
    assert_eq!(diff.subs.added.len(), 1, "added survives empty filter");
    assert_eq!(diff.subs.added[0].params.name, "added");
}

/// Prune discipline: the post-apply pass over `disabled_runtime`
/// retains names whose `[[watch]]` entry still exists in TOML
/// (regardless of `enabled`) and drops names that left the file
/// entirely. Tests every retention branch under one fixture so a
/// future refactor that swaps the membership check (e.g., uses
/// `active_watches` and drops TOML-disabled retention) regresses
/// here.
#[test]
fn prune_disabled_runtime_retains_toml_entries_drops_removed_names() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let anchor = tmp.path().display();
    let initial_text = format!(
        r#"
[[watch]]
name      = "kept_active"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "kept_toml_disabled"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
enabled   = false
"#,
    );
    let initial = Config::from_str(&initial_text).expect("initial parses");

    let mut rig = rig_for(initial, cfg_path);

    // Three runtime overrides: two anchored against TOML rows that
    // still exist (one active, one disabled), one anchored against a
    // name the TOML doesn't carry.
    for name in ["kept_active", "kept_toml_disabled", "gone_from_toml"] {
        rig.driver
            .disabled_runtime
            .insert(CompactString::const_new(name));
    }

    rig.driver.prune_disabled_runtime_against_current_config();

    assert!(
        rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("kept_active")),
        "runtime override over an active TOML row stays",
    );
    assert!(
        rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("kept_toml_disabled")),
        "runtime override over a TOML-disabled row stays \
         (operator's 'off twice' preference is preserved)",
    );
    assert!(
        !rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("gone_from_toml")),
        "runtime override over a TOML-removed name evaporates",
    );
    assert_eq!(
        rig.driver.disabled_runtime.len(),
        2,
        "only the two TOML-anchored names survive",
    );
}

/// End-to-end pipeline ordering: `handle_reload` runs the prune
/// AFTER `rotate_apply`, so the helper reads the freshly-applied
/// `current_config`. Pinning this guards against a refactor that
/// reorders the prune above the rotation (which would have it read
/// the old TOML).
#[test]
fn handle_reload_runs_prune_against_post_rotation_config() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let anchor = tmp.path().display();
    let v1_text = format!(
        r#"
[[watch]]
name      = "doomed"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
"#,
    );
    std::fs::write(&cfg_path, &v1_text).unwrap();
    let initial = Config::from_str(&v1_text).expect("v1 parses");

    let mut rig = rig_for(initial, cfg_path.clone());
    rig.driver
        .disabled_runtime
        .insert(CompactString::const_new("doomed"));

    // Rewrite the on-disk config to drop the watch entirely. The
    // reload's parse picks this up; rotate_apply commits it; prune
    // observes the post-rotation state.
    std::fs::write(&cfg_path, "").unwrap();
    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert!(
        rig.driver.disabled_runtime.is_empty(),
        "post-reload prune dropped the override whose TOML row vanished",
    );
}
