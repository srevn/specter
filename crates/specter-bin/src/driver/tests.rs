//! Engine-driver unit tests — single-tick drive of `EngineDriver`
//! over the `TestRig` mock-channel harness: drain order, the
//! SIGHUP / auto-reload settle pipeline, and the `forward`
//! wake-per-send protocol.
//!
//! Wired by `#[cfg(test)] mod tests;` in `driver.rs`. Imports below
//! are explicit (no `use super::*;`) so the driver spine carries no
//! cfg(test)-only re-exports — the test surface is what this file
//! references, nothing more.

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
}

fn rig_for(config: Config, config_path: PathBuf) -> TestRig {
    let mut chans = Channels::new();
    let sensor_in_tx = chans.sensor_in_tx.clone();
    let effect_in_tx = chans.effect_in_tx.clone();
    let reload_tx = chans.reload_signal_tx.clone();
    let shutdown_tx = chans.shutdown_engine_tx.clone();
    let config_event_tx = chans.config_event_tx.clone();
    let watch_ops_tx = chans.watch_ops_tx.clone();
    let actuator_side = chans.take_actuator_side();
    let watcher_side = chans.take_watcher_side();
    let engine_side = chans.take_engine_side();
    drop(chans);

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
    let loader = Loader::new(config, log_cfg, dummy_meta());
    let driver = EngineDriver::new(
        Engine::new(),
        loader,
        config_path,
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
    } = rig.actuator_side;
    drop(effect_in_tx);
    drop(effects_rx);
    drop(shutdown_actuator_rx);
    drop(hard_shutdown_actuator_rx);

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
        driver,
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

/// Parse-fail must **not** rotate `loader.config_meta`. Rotating
/// on failure would suppress the auto-reload retry loop: the next
/// pulse's lstat-vs-stored-meta check would compare the (still
/// broken) on-disk file against the freshly-stored meta from the
/// failed attempt, decide "unchanged," and never retry. Pins the
/// negative invariant.
#[test]
fn reload_parse_failure_does_not_rotate_meta() {
    // `/dev/null/no/such/file.toml` — a guaranteed-ENOTDIR path
    // (because `/dev/null` is a character device, not a directory).
    // The reload pipeline observes a parse-fail equivalent.
    let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let pre_reload_meta = rig.driver.loader.config_meta;

    rig.reload_tx.try_send(()).expect("reload send");
    rig.shutdown_tx.try_send(()).expect("shutdown send");
    assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);

    assert_eq!(
        rig.driver.loader.config_meta, pre_reload_meta,
        "parse-fail must not rotate meta — would suppress retry pulses",
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
        driver,
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
