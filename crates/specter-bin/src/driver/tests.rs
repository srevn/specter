//! Engine-driver unit tests — single-tick drive of [`EngineDriver`]
//! over the [`TestRig`] mio-integrated harness.
//!
//! Wired by `#[cfg(test)] mod tests;` in `driver.rs`. The rig
//! constructs a real [`mio::Poll`] via [`DriverHub::new`] against the
//! sensor crate's [`MockFsWatcher`] (whose socketpair-backed `AsFd`
//! surface lets reactor-integration tests run against a real reactor
//! without any platform watcher backend). Tests inject `FsEvent`s
//! through `hub.watcher_mut().inject(...)`, drive signals through
//! [`EngineDriver::dispatch_signal`] directly (real signals would
//! race nextest's process-wide handlers), and exercise IPC through a
//! real bound socket on a tempdir path.

use super::WakeHandle;
use super::conns::{ConnRole, WRITE_QUEUE_HIGH_WATER};
use super::hub::{EnqueueOutcome, TOKEN_CONN_BASE};
use super::state::ReloadTrigger;
use super::{DriverHub, EngineDriver, TickOutcome};
use crate::app::CliLogOverrides;
use crate::channels::ActuatorIO;
use crate::ipc::protocol::{ResponsePayload, WireErrorCode, WireId, WireRequest};
use crate::ipc::wire::WireTime;
use crate::loader::Loader;
use compact_str::CompactString;
use crossbeam::channel::Sender;
use specter_actuator::RunWiring;
use specter_config::{Config, FileMeta};
use specter_core::{Diagnostic, Input, ProbeOwner, ResourceId, StepOutput, SubId};
use specter_engine::Engine;
use specter_sensor::testkit::{MockFsWatcher, MockProber};
use std::io::{Read, Write};
use std::ops::ControlFlow;
use std::os::unix::net::UnixStream;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

// ============================================================
// Fixtures + rig
// ============================================================

/// Sentinel meta used in fixtures whose config file may not exist
/// on disk. Inode 0 is reserved by every supported kernel and
/// `mode = 0` cannot occur in a real lstat (the kernel always sets
/// file-type bits); this value never compares equal to a real
/// [`FileMeta::from_path`] capture, so tests that *do* exercise the
/// meta-rotation path can assert "rotated to a real value" by
/// comparing against a fresh [`FileMeta::from_path`] (which differs
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

/// Bundle of handles a test holds to drive [`EngineDriver`] without
/// the [`crate::app`] orchestration layer.
///
/// Field order is the drop order — the driver (owning the Hub) drops
/// before `_tmp`, so the listener fd closes before the tempdir reaps
/// the socket file.
struct TestRig {
    driver: EngineDriver<MockFsWatcher>,
    /// Held so the actuator-thread side of the bundle survives —
    /// `ActuatorIO::pair` returns the consumer halves here; without
    /// holding them, the driver's `effects_tx` senders would observe
    /// Disconnected on the first `try_send`.
    actuator_side: RunWiring,
    /// Shared `Arc<MockProber>` clone the driver received as the
    /// `Arc<dyn Prober>`. Tests use this clone to drain `take_submitted` /
    /// `take_cancelled` recordings.
    prober: Arc<MockProber>,
    /// Producer-side handle for the prober response channel. Tests
    /// `send` here followed by `waker.wake()` to simulate the
    /// production [`crate::app::WakingProberResponseSender`]'s
    /// send-then-wake protocol.
    prober_response_tx: Sender<Input>,
    /// Producer-side handle for the effect completion channel. Same
    /// send-then-wake protocol as `prober_response_tx`.
    effect_complete_tx: Sender<Input>,
    /// Shared [`WakeHandle`] clone — the Hub holds one clone, the
    /// rig holds another. Tests fire `waker.wake()` after writing to
    /// `prober_response_tx` / `effect_complete_tx` to mirror the
    /// production wake-after-send semantics.
    waker: WakeHandle,
    /// Bound socket path. Tests may [`UnixStream::connect`] here to
    /// drive IPC clients. Lives through the rig's lifetime so the
    /// socket file survives until `_tmp` drops.
    socket_path: PathBuf,
    /// Tempdir guard — last field so the driver (and its Hub's
    /// listener fd) drops before the tempdir reaps the socket file.
    _tmp: tempfile::TempDir,
}

/// Build a [`TestRig`] for the supplied config + config_path. Every
/// kernel-side resource is freshly allocated per call: a bound
/// `UnixListener`, a fresh `mio::Poll` (via `DriverHub::new`), a
/// fresh `Signals` iterator, two unbounded crossbeam channels for the
/// wake'd Input streams, and a fresh `MockFsWatcher` with its
/// socketpair-backed readiness substrate.
fn rig_for(config: Config, config_path: PathBuf) -> TestRig {
    let tmp = tempfile::TempDir::new().expect("tempdir");
    let socket_path = tmp.path().join("specter-test.sock");
    let listener =
        std::os::unix::net::UnixListener::bind(&socket_path).expect("bind tmp ipc socket");
    let watcher = MockFsWatcher::new();
    let signals = crate::signals::register_signal_handlers().expect("signal pipe init");
    let (prober_response_tx, prober_response_rx) = crossbeam::channel::unbounded::<Input>();
    let (effect_complete_tx, effect_complete_rx) = crossbeam::channel::unbounded::<Input>();
    let (hub, waker) = DriverHub::new(
        listener,
        watcher,
        None,
        signals,
        prober_response_rx,
        effect_complete_rx,
    )
    .expect("hub init");
    let prober: Arc<MockProber> = Arc::new(MockProber::new());
    let (actuator_io, actuator_side) = ActuatorIO::pair();

    let log_cfg = config.log.clone();
    // `noop()` avoids racing every rig on the global tracing
    // subscriber slot — tests assert on the driver's reload-pipeline
    // behaviour, not the subscriber's filter state.
    let obs_handle = crate::observability::ObservabilityHandle::noop();
    let loader = Loader {
        current_config: config,
        current_log: log_cfg,
        config_meta: dummy_meta(),
    };

    let driver = EngineDriver::new(
        Engine::new(),
        loader,
        config_path,
        socket_path.clone(),
        CliLogOverrides::default(),
        obs_handle,
        prober.clone(),
        actuator_io,
        hub,
    );

    TestRig {
        driver,
        actuator_side,
        prober,
        prober_response_tx,
        effect_complete_tx,
        waker,
        socket_path,
        _tmp: tmp,
    }
}

/// Single-watch config used by attach / reload / IPC tests.
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

/// Dynamic single-watch config. Brace expansion makes `is_dynamic`
/// auto-detect; the literal prefix is the supplied tempdir so the
/// validator's path-canonicalisation pass succeeds.
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

/// Connect a [`UnixStream`] client to the rig's bound socket. The
/// stream is left in blocking mode by default (callers can flip it
/// non-blocking if they want to interleave reads with `tick`s) and
/// carries a short read timeout so a non-arriving response surfaces
/// as a `WouldBlock`-style error rather than hanging the test.
fn ipc_connect(rig: &TestRig) -> UnixStream {
    let client = UnixStream::connect(&rig.socket_path).expect("connect to rig socket");
    client
        .set_read_timeout(Some(Duration::from_millis(50)))
        .expect("set read timeout");
    client
}

/// Pre-arm a zero-duration block timeout on the next `tick` by
/// flagging `config_settle_until = Some(now)`. The settle-expiry
/// helper consumes the slot once the tick passes its expiry check, so
/// each call covers exactly one tick — the timeout collapses to
/// `Duration::ZERO`, `mio::Poll::poll` returns immediately, and the
/// drain pass runs without waiting on a real deadline.
fn arm_zero_timeout(rig: &mut TestRig) {
    rig.driver.config_settle_until = Some(Instant::now());
}

/// Drive the rig's tick to the deadline, polling for a complete
/// LF-delimited JSON response on `client`. Returns the parsed
/// [`ResponsePayload`] or `None` on timeout. Each loop iteration arms
/// a zero-duration block timeout (so `tick` returns promptly) and
/// reads whatever bytes the kernel has buffered.
fn run_until_response(rig: &mut TestRig, client: &mut UnixStream) -> Option<ResponsePayload> {
    let deadline = Instant::now() + Duration::from_secs(2);
    let mut buf: Vec<u8> = Vec::new();
    while Instant::now() < deadline {
        arm_zero_timeout(rig);
        let _ = rig.driver.tick();
        let mut chunk = [0u8; 1024];
        match client.read(&mut chunk) {
            Ok(0) => return None,
            Ok(n) => {
                buf.extend_from_slice(&chunk[..n]);
                if let Some(nl) = buf.iter().position(|&b| b == b'\n') {
                    let line = &buf[..nl];
                    return Some(
                        serde_json::from_slice(line).expect("response payload parses as JSON"),
                    );
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => return None,
        }
    }
    None
}

/// Write a [`WireRequest`] LF-delimited to a client stream.
fn write_request(client: &mut UnixStream, req: &WireRequest) {
    let mut bytes = serde_json::to_vec(req).expect("serialize wire request");
    bytes.push(b'\n');
    client.write_all(&bytes).expect("write request");
}

/// Round-trip a single request → response over `client`, driving the
/// rig's tick loop until the response surfaces. Panics on timeout
/// (every covered verb completes within milliseconds).
fn ipc_round_trip(
    rig: &mut TestRig,
    client: &mut UnixStream,
    req: &WireRequest,
) -> ResponsePayload {
    write_request(client, req);
    run_until_response(rig, client).expect("IPC response within test deadline")
}

// ============================================================
// Empty-tick + shutdown semantics
// ============================================================

/// A SIGTERM dispatched directly via [`EngineDriver::dispatch_signal`]
/// returns [`ControlFlow::Break`]. The pure dispatch path is also
/// pinned by `dispatch_signal_inner_tests` in `driver.rs`; this test
/// covers the method-level wrapper from the rig's surface.
#[test]
fn dispatch_signal_sigterm_returns_break() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let outcome = rig
        .driver
        .dispatch_signal(signal_hook::consts::SIGTERM, Instant::now());
    assert_eq!(outcome, ControlFlow::Break(()));
    // Probe drain has no work (no Sub attached). begin_shutdown is
    // safe to call regardless — pins that the cancel-first drain is
    // idempotent when no probes are armed.
    let _ = rig.driver.begin_shutdown();
}

// ============================================================
// run_initial_attach: static / dynamic / mixed / disabled
// ============================================================

#[test]
fn run_initial_attach_attaches_static_sub_and_emits_watch_op() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");
    assert!(rig.driver.engine.subs().get(sid).is_some());

    // The attach emitted a Watch op inline (via `forward`'s
    // `apply_watch_ops`). The Seed burst emitted a probe forwarded to
    // the prober's `submit` recording.
    let submitted = rig.prober.take_submitted();
    assert_eq!(submitted.len(), 1, "Seed burst emits one probe");

    // begin_shutdown drains the armed Seed probe so the rig drops
    // silently; production loops to a Shutdown tick which would do
    // the same.
    let _ = rig.driver.begin_shutdown();
}

/// Static-only config attaches one Sub per `[[watch]]` into the
/// engine's `by_name` index and leaves the Promoter registry empty.
#[test]
fn run_initial_attach_attaches_static_only_config() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("static Sub 'build' attached");
    assert!(rig.driver.engine.subs().get(sid).is_some());
    assert!(rig.driver.engine.promoters().is_empty());

    let _ = rig.driver.begin_shutdown();
}

/// A config with a dynamic `[[watch]]` routes through
/// `attach_promoter` and registers the Promoter in the engine's
/// registry.
#[test]
fn run_initial_attach_registers_promoter_for_dynamic_watch() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_promoter(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();

    assert!(rig.driver.engine.subs().is_empty());
    let pid = rig
        .driver
        .engine
        .promoters()
        .find_by_name("logs")
        .expect("Promoter 'logs' registered");
    assert!(rig.driver.engine.promoters().get(pid).is_some());

    let _ = rig.driver.begin_shutdown();
}

/// Mixed static + dynamic config: the initial-attach loop walks both
/// spec lists and populates both maps in one run.
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

    let _ = rig.driver.begin_shutdown();
}

/// Disabled entries on either side are skipped at initial attach —
/// neither the engine's Sub registry nor its Promoter registry sees
/// the disabled rows.
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
    assert!(subs.find_by_name("build").is_some());
    assert!(subs.find_by_name("build_off").is_none());
    assert!(promoters.find_by_name("logs").is_some());
    assert!(promoters.find_by_name("logs_off").is_none());
    assert_eq!(subs.len(), 1);
    assert_eq!(promoters.len(), 1);

    let _ = rig.driver.begin_shutdown();
}

// ============================================================
// Reload pipeline: SIGHUP → dispatch_reload
// ============================================================

/// Reload against an invalid path keeps the running config — the
/// parse early-return preserves `loader.current_config`.
#[test]
fn reload_with_invalid_path_logs_and_keeps_config() {
    // `/dev/null/no/such` returns ENOTDIR on lstat; the parse never
    // sees the bytes; the early-return preserves running state.
    let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config.clone(), cfg_path);

    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(rig.driver.loader.current_config, config);
}

/// Empty-diff reload preserves Sub identity: re-saving the same TOML
/// bytes runs the reload pipeline but the engine's `by_name` resolves
/// to the same SubId across the rotation.
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

    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    let sid_after = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("still attached");
    assert_eq!(sid_before, sid_after);
    assert_eq!(rig.driver.loader.current_config, initial);

    let _ = rig.driver.begin_shutdown();
}

/// Reload that adds a Sub attaches it through the diff-driven
/// `Input::ConfigDiff` step.
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

    std::fs::write(&cfg_path, &new_text).unwrap();
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_some());

    let _ = rig.driver.begin_shutdown();
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
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_none());

    let _ = rig.driver.begin_shutdown();
}

/// Reload that adds a dynamic [[watch]] registers a Promoter.
#[test]
fn reload_added_promoter_registers_in_engine() {
    let tmp = tempfile::TempDir::new().unwrap();
    let initial_text = String::new();
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
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.engine.promoters().find_by_name("logs").is_some());

    let _ = rig.driver.begin_shutdown();
}

/// Reload that removes a dynamic [[watch]] reaps the Promoter.
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
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.engine.promoters().find_by_name("logs").is_none());

    let _ = rig.driver.begin_shutdown();
}

/// Reload modifying a dynamic [[watch]] mints a fresh PromoterId
/// under the same name.
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
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    let new_pid = rig
        .driver
        .engine
        .promoters()
        .find_by_name("logs")
        .expect("Promoter 'logs' still registered post-reload");
    assert_ne!(new_pid, old_pid);

    let _ = rig.driver.begin_shutdown();
}

/// Static→dynamic migration via path edit. Diff emits
/// `subs.removed + promoters.added`; engine registries swap.
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
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.engine.subs().find_by_name("foo").is_none());
    assert!(rig.driver.engine.promoters().find_by_name("foo").is_some());

    let _ = rig.driver.begin_shutdown();
}

/// Dynamic→static migration via path edit.
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
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.engine.promoters().find_by_name("foo").is_none());
    assert!(rig.driver.engine.subs().find_by_name("foo").is_some());

    let _ = rig.driver.begin_shutdown();
}

// ============================================================
// read_and_parse_config + meta rotation discipline
// ============================================================

/// `read_and_parse_config` on a valid file returns
/// `Some((Config, FileMeta))` whose meta matches a fresh
/// path-level lstat.
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
    let lstat = FileMeta::from_path(&cfg_path).expect("lstat ok");
    assert_eq!(parsed_meta, lstat);
    assert_ne!(parsed_meta, dummy_meta());
}

/// Apply-branch reload rotates `loader.config_meta` to the post-edit
/// lstat AND attaches the added Sub.
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
    assert_eq!(rig.driver.loader.config_meta, dummy_meta());

    std::fs::write(&cfg_path, &v2_text).unwrap();
    let expected_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");

    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(rig.driver.loader.config_meta, expected_meta);
    assert!(rig.driver.engine.subs().find_by_name("b").is_some());

    let _ = rig.driver.begin_shutdown();
}

/// Empty-diff reload must still rotate `loader.config_meta` —
/// otherwise the auto-reload settle filter would loop against an
/// already-applied edit.
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
        .expect("attached");

    std::fs::write(&cfg_path, &cfg_text).unwrap();
    let expected_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");
    assert_ne!(expected_meta, dummy_meta());

    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(rig.driver.loader.config_meta, expected_meta);
    assert_eq!(rig.driver.loader.current_config, initial);
    let sid_after = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("still attached");
    assert_eq!(sid_before, sid_after);

    let _ = rig.driver.begin_shutdown();
}

/// Parse-fail with a successful post-fail lstat rotates
/// `loader.config_meta` (closes the chmod-EACCES recovery loop).
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

    // Sleep so mtime advances at least one nanosecond past `v1_meta`
    // on coarse-resolution filesystems.
    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&cfg_path, "not valid toml [[[").unwrap();
    let v2_lstat = FileMeta::from_path(&cfg_path).expect("v2 lstat ok");
    assert_ne!(v1_meta, v2_lstat);

    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(rig.driver.loader.config_meta, v2_lstat);
    assert_eq!(rig.driver.loader.current_config, v1_config);
}

/// Destination mismatch on reload preserves the running log shape
/// in `loader.current_log` — the appender doesn't hot-reload, so
/// the rotation must reflect what is applied.
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

    assert_eq!(
        rig.driver.loader.current_log.destination,
        specter_config::LogDestination::File,
    );

    std::thread::sleep(Duration::from_millis(10));
    std::fs::write(&cfg_path, v_stderr).unwrap();
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(
        rig.driver.loader.current_log.destination,
        specter_config::LogDestination::File,
    );
    assert_eq!(rig.driver.loader.current_log.path, initial_log.path);
}

// ============================================================
// Auto-reload settle pipeline
// ============================================================

/// Helper short-circuits when no pulse has armed the deadline.
#[test]
fn apply_config_settle_expiry_no_op_when_unarmed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config.clone(), cfg_path);

    let snapshot_meta = rig.driver.loader.config_meta;
    let _ = rig.driver.apply_config_settle_expiry(Instant::now());
    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(rig.driver.loader.config_meta, snapshot_meta);
    assert_eq!(rig.driver.loader.current_config, config);
}

/// Helper short-circuits when `now < deadline`. Deadline stays armed.
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

    assert_eq!(rig.driver.config_settle_until, Some(deadline));
}

/// `now == deadline` is the boundary case. The deadline is consumed
/// and the lstat filter then runs.
#[test]
fn apply_config_settle_expiry_fires_at_exact_deadline() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, "").unwrap();
    let config = Config::from_str("").expect("empty config parses");
    let real_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");

    let mut rig = rig_for(config, cfg_path);
    rig.driver.loader.config_meta = real_meta;

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);

    let _ = rig.driver.apply_config_settle_expiry(deadline);

    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(rig.driver.loader.config_meta, real_meta);
}

/// Settle expiry whose lstat agrees with `loader.config_meta` silently
/// drops the pulse — the kqueue-parent-spillover case.
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
    rig.driver.loader.config_meta = real_meta;
    let _ = rig.driver.run_initial_attach();
    let sid_snapshot = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("attached");

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));

    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(rig.driver.loader.config_meta, real_meta);
    assert_eq!(rig.driver.loader.current_config, initial);
    let sid_after = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("still attached");
    assert_eq!(sid_snapshot, sid_after);

    let _ = rig.driver.begin_shutdown();
}

/// Settle expiry whose lstat detects drift triggers `dispatch_reload`,
/// rotating config and meta.
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
    let v2_lstat = FileMeta::from_path(&cfg_path).expect("v2 lstat ok");
    assert_ne!(v1_meta, v2_lstat);

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));

    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(rig.driver.loader.config_meta, v2_lstat);
    assert!(rig.driver.engine.subs().find_by_name("a").is_some());
    assert!(rig.driver.engine.subs().find_by_name("b").is_some());
    assert_eq!(rig.driver.loader.current_config.watches.len(), 2);

    let _ = rig.driver.begin_shutdown();
}

/// Lstat error routes through the "treat-as-changed" branch:
/// `dispatch_reload` runs but fails to read; loader state preserved.
/// Settle slot consumed regardless.
#[test]
fn apply_config_settle_expiry_treats_missing_path_as_changed() {
    let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config.clone(), cfg_path);
    let pre_meta = rig.driver.loader.config_meta;

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));

    assert_eq!(rig.driver.config_settle_until, None);
    assert_eq!(rig.driver.loader.config_meta, pre_meta);
    assert_eq!(rig.driver.loader.current_config, config);
}

// ============================================================
// DriverState reload counters
// ============================================================

/// SIGHUP-driven reload bumps with `trigger = Sighup`.
#[test]
fn dispatch_reload_via_sighup_bumps_counters_with_sighup_trigger() {
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

    assert_eq!(rig.driver.driver_state.reload_count, 0);
    assert!(rig.driver.driver_state.last_reload_at.is_none());
    assert!(rig.driver.driver_state.last_reload_via.is_none());

    std::fs::write(&cfg_path, &v2_text).unwrap();
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(rig.driver.driver_state.reload_count, 1);
    assert!(rig.driver.driver_state.last_reload_at.is_some());
    assert_eq!(
        rig.driver.driver_state.last_reload_via,
        Some(ReloadTrigger::Sighup),
    );

    let _ = rig.driver.begin_shutdown();
}

/// Auto-reload settle expiry threads `AutoReload` into
/// `dispatch_reload`.
#[test]
fn dispatch_reload_via_auto_reload_bumps_counters_with_auto_reload_trigger() {
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

    assert_eq!(rig.driver.driver_state.reload_count, 0);
    assert_eq!(rig.driver.driver_state.last_reload_via, None);

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

    let deadline = Instant::now();
    rig.driver.config_settle_until = Some(deadline);
    let _ = rig
        .driver
        .apply_config_settle_expiry(deadline + Duration::from_millis(1));

    assert_eq!(rig.driver.driver_state.reload_count, 1);
    assert!(rig.driver.driver_state.last_reload_at.is_some());
    assert_eq!(
        rig.driver.driver_state.last_reload_via,
        Some(ReloadTrigger::AutoReload),
    );

    let _ = rig.driver.begin_shutdown();
}

/// Parse-fail reload does NOT bump the counters — the early return
/// short-circuits before `record_reload`.
#[test]
fn dispatch_reload_does_not_bump_counters_on_parse_fail() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, "not valid toml [[[").unwrap();
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert_eq!(rig.driver.driver_state.reload_count, 0);
    assert!(rig.driver.driver_state.last_reload_at.is_none());
    assert!(rig.driver.driver_state.last_reload_via.is_none());
}

// ============================================================
// forward: effects + cancel ordering
// ============================================================

/// `forward` dispatches `cancel_effects` ahead of `effects` over the
/// same `effects_tx` channel. The same-step collision is
/// unconstructable in production but the ordering pins the contract.
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
    let mut rig = rig_for(config, cfg_path);

    let pid_a = ProfileId::from(KeyData::from_ffi(0x10));
    let pid_b = ProfileId::from(KeyData::from_ffi(0x20));

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

    let outcome = rig.driver.forward(out);
    assert_eq!(outcome, ControlFlow::Continue(()));

    let mut received: Vec<EffectOp> = Vec::new();
    while let Ok(op) = rig.actuator_side.effects_rx.try_recv() {
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
        "submit dispatches after all cancels; got {:?}",
        received[2],
    );
}

// ============================================================
// Drop-order discipline + initial-attach probe drain
// ============================================================

/// Drop-order test: probe armed via initial-attach, begin_shutdown
/// drains it, then dropping the rig is silent. A rig drop with an
/// armed probe would trip `ProbeSlot::drop`'s linear-edge tripwire
/// (panic in every build) — the test asserts the cancel-first drain
/// holds even when the Profile started its life with a Seed-Verifying
/// burst.
#[test]
fn drop_after_begin_shutdown_is_silent_with_armed_probe() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    let _ = rig.driver.run_initial_attach();
    // The Seed burst left a probe armed on the attached Profile.
    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("attached");
    let pid = rig
        .driver
        .engine
        .subs()
        .get(sid)
        .map(specter_core::Sub::profile)
        .expect("Sub has a Profile");
    assert!(
        rig.driver
            .engine
            .pending_probe_for(ProbeOwner::Profile(pid))
            .is_some(),
        "Seed-Verifying probe armed at attach time",
    );

    let _ = rig.driver.begin_shutdown();
    assert!(
        rig.driver
            .engine
            .pending_probe_for(ProbeOwner::Profile(pid))
            .is_none(),
        "begin_shutdown drained the probe",
    );

    // Drop is silent — no `ProbeSlot::drop` tripwire panic. Test
    // passing IS the assertion.
    drop(rig);
}

// ============================================================
// Deferred-input queue: WatchOpRejected replay
// ============================================================

/// A rejected watch op queues into `deferred_inputs` and replays on
/// the next tick before the mio Poll re-blocks. The replay drives
/// the engine's claim-purge path in the SAME tick the original
/// `forward` cycle ran, so an `Input::WatchOpRejected` never lingers
/// across the block boundary even though the watcher's rejection is
/// observed synchronously inside `apply_watch_ops`.
#[test]
fn watch_op_rejection_queues_deferred_input_and_replays_next_tick() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);

    // Arm a one-shot Watch failure on the MockFsWatcher. The next
    // `WatchOp::Watch` returns `Err(Pressure { errno: 24 (EMFILE) })`,
    // forward queues an `Input::WatchOpRejected` onto deferred_inputs.
    rig.driver
        .hub
        .watcher_mut()
        .fail_next_watch(specter_sensor::WatchFailure::Pressure { errno: 24 });

    let _ = rig.driver.run_initial_attach();

    assert_eq!(
        rig.driver.deferred_inputs.len(),
        1,
        "rejected watch op queued for replay",
    );
    match &rig.driver.deferred_inputs[0] {
        Input::WatchOpRejected { failure, .. } => assert_eq!(
            *failure,
            specter_sensor::WatchFailure::Pressure { errno: 24 },
        ),
        other => panic!("expected Input::WatchOpRejected, got {other:?}"),
    }

    // Drive one tick. `replay_deferred_inputs` runs first, consuming
    // the queued `WatchOpRejected`; the engine's claim-purge fires.
    // Pre-arm `config_settle_until` so the tick's block timeout is
    // ZERO and we don't wait on an actual deadline.
    arm_zero_timeout(&mut rig);
    let _ = rig.driver.tick();

    assert!(
        rig.driver.deferred_inputs.is_empty(),
        "deferred input consumed by replay_deferred_inputs",
    );

    let _ = rig.driver.begin_shutdown();
}

// ============================================================
// disabled_runtime: diff filter + post-apply prune
// ============================================================

#[test]
fn compute_watch_diff_filters_disabled_runtime_from_all_four_buckets() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let anchor = tmp.path().display();

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

    let unfiltered = specter_config::diff(&rig.driver.loader.current_config, &new);
    assert_eq!(unfiltered.subs.added.len(), 1);
    assert_eq!(unfiltered.subs.removed.len(), 1);
    assert_eq!(unfiltered.subs.modified_identity.len(), 1);
    assert_eq!(unfiltered.subs.modified_params.len(), 1);

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
    assert!(filtered.subs.added.is_empty());
    assert!(filtered.subs.removed.is_empty());
    assert!(filtered.subs.modified_identity.is_empty());
    assert!(filtered.subs.modified_params.is_empty());
}

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
    assert_eq!(diff.subs.added.len(), 1);
    assert_eq!(diff.subs.added[0].params.name, "added");
}

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
    );
    assert!(
        rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("kept_toml_disabled")),
    );
    assert!(
        !rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("gone_from_toml")),
    );
    assert_eq!(rig.driver.disabled_runtime.len(), 2);
}

/// `dispatch_reload` runs the prune AFTER `rotate_apply`.
#[test]
fn dispatch_reload_runs_prune_against_post_rotation_config() {
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

    std::fs::write(&cfg_path, "").unwrap();
    let _ = rig
        .driver
        .dispatch_reload(ReloadTrigger::Sighup, Instant::now());

    assert!(rig.driver.disabled_runtime.is_empty());
}

// ============================================================
// IPC verb dispatch — over real UnixStream clients
// ============================================================

/// `Status` round-trips: write a Status request, drive ticks until
/// the response surfaces, parse it back into `ResponsePayload::Status`,
/// and assert the projection observed the driver's actual socket
/// path.
#[test]
fn ipc_status_replies_with_projection() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let expected_socket = rig.socket_path.clone();
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(&mut rig, &mut client, &WireRequest::Status);
    match reply {
        ResponsePayload::Status(status) => {
            assert_eq!(
                status.socket_path,
                crate::ipc::wire::WirePath::from(&expected_socket),
            );
            assert_eq!(status.sub_total, 0);
        }
        other => panic!("expected Status, got {other:?}"),
    }
}

/// Subscribe { name: None } enqueues an unfiltered-tail ack and
/// flips the conn role to `Sub`. The `conn_count` reflects the
/// surviving conn (one entry).
#[test]
fn ipc_subscribe_unfiltered_acks_and_registers_subscriber() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe { name: None },
    );
    match reply {
        ResponsePayload::SubscribeAck { sub: None } => {}
        other => panic!("expected SubscribeAck(None), got {other:?}"),
    }
    // The Sub-role conn is still in the conn map — the new
    // subscriber storage (one ConnRole::Sub conn ≡ one subscriber).
    assert_eq!(rig.driver.hub.conn_count(), 1);
}

/// Subscribe { name: Some("nope") } against an empty engine returns
/// `Err { code: WireErrorCode::UnknownSub }` and DOES NOT flip the
/// conn role.
#[test]
fn ipc_subscribe_unknown_name_errors_without_registering() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe {
            name: Some(CompactString::const_new("nope")),
        },
    );
    match reply {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, WireErrorCode::UnknownSub);
            assert!(
                error.contains("no watch named nope"),
                "error carries the resolution detail; got {error:?}",
            );
        }
        other => panic!("expected Err(WireErrorCode::UnknownSub), got {other:?}"),
    }
    // The conn stays alive as a Reqs conn (no role flip happened),
    // so conn_count is still 1.
    assert_eq!(rig.driver.hub.conn_count(), 1);
}

/// Subscribe { name: Some("build") } against a config with a `build`
/// watch attached resolves the name to a `SubId` and acks carrying
/// the resolved `WireId`.
#[test]
fn ipc_subscribe_known_name_resolves_and_acks() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    let expected_sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("build attached");

    let mut client = ipc_connect(&rig);
    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe {
            name: Some(CompactString::const_new("build")),
        },
    );
    match reply {
        ResponsePayload::SubscribeAck { sub: Some(wire_id) } => {
            assert_eq!(wire_id, WireId::from(expected_sid));
        }
        other => panic!("expected SubscribeAck(Some), got {other:?}"),
    }
    assert_eq!(rig.driver.hub.conn_count(), 1);

    let _ = rig.driver.begin_shutdown();
}

/// A second `Subscribe` on a conn that already flipped to
/// [`ConnRole::Sub`] is a precondition violation. The handler gate
/// refuses with [`WireErrorCode::AlreadySubscribed`] before reaching
/// `transition_to_sub`, so the first subscription's `filter` and
/// `missed` window survive unchanged.
///
/// Pins the fix structurally: the wire surface carries an `Err`
/// for the second Subscribe, and the conn's role inspection shows
/// the first subscription's state intact.
#[test]
fn subscribe_twice_returns_err_already_subscribed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    let mut client = ipc_connect(&rig);

    // First Subscribe: `name = None` → unfiltered tail. The handler
    // acks with `sub = None` and flips the conn role to
    // `Sub { filter: None, missed: 0, first_dropped_at: None }`.
    let reply1 = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe { name: None },
    );
    assert!(
        matches!(reply1, ResponsePayload::SubscribeAck { sub: None }),
        "first Subscribe acks with sub=None; got {reply1:?}",
    );

    // First accepted conn lands on `TOKEN_CONN_BASE`.
    let token = mio::Token(TOKEN_CONN_BASE);
    {
        let conn = rig
            .driver
            .hub
            .conn_ref(token)
            .expect("conn lives across the first round-trip");
        assert!(
            matches!(
                conn.role,
                ConnRole::Sub {
                    filter: None,
                    missed: 0,
                    first_dropped_at: None,
                }
            ),
            "post-first-Subscribe role is fresh Sub state",
        );
    }

    // Second Subscribe on the SAME conn: `name = Some("build")`. The
    // gate refuses with WireErrorCode::AlreadySubscribed before reaching
    // `transition_to_sub`.
    let reply2 = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe {
            name: Some(CompactString::const_new("build")),
        },
    );
    match reply2 {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, WireErrorCode::AlreadySubscribed);
            assert!(
                error.contains("already in subscribe mode"),
                "error carries the precondition detail; got {error:?}",
            );
        }
        other => panic!("expected Err(WireErrorCode::AlreadySubscribed); got {other:?}"),
    }

    // The first subscription's role is untouched — `filter == None`
    // (not `Some(sid_build)`), `missed == 0` (no window opened).
    let conn = rig
        .driver
        .hub
        .conn_ref(token)
        .expect("conn still in map after refusal");
    assert!(
        matches!(
            conn.role,
            ConnRole::Sub {
                filter: None,
                missed: 0,
                first_dropped_at: None,
            }
        ),
        "the first subscription's state survives; got {:?}",
        conn.role,
    );

    let _ = rig.driver.begin_shutdown();
}

/// Regression sibling of [`subscribe_twice_returns_err_already_subscribed`]:
/// when a missed window has accumulated on the first subscription,
/// a refused second Subscribe must NOT reset `missed` or
/// `first_dropped_at`. The handler gate fires before any state
/// mutation, so the back-pressure accounting carries through
/// unchanged.
#[test]
fn subscribe_after_err_does_not_overwrite_prior_subscription() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    let mut client = ipc_connect(&rig);
    let _ = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe { name: None },
    );
    let token = mio::Token(TOKEN_CONN_BASE);

    // Synthesize a pre-existing missed window on the conn. Direct
    // mutation lets the test pin gate behavior without driving the
    // full fan-out throttle path.
    let stamped_at = SystemTime::UNIX_EPOCH + Duration::from_secs(1_700_000_000);
    {
        let conn = rig
            .driver
            .hub
            .conn_mut(token)
            .expect("conn lives after Subscribe ack flushed");
        match &mut conn.role {
            ConnRole::Sub {
                missed,
                first_dropped_at,
                ..
            } => {
                *missed = 7;
                *first_dropped_at = Some(stamped_at);
            }
            ConnRole::Reqs => panic!("expected Sub after first Subscribe"),
        }
    }

    // Second Subscribe — refused.
    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe {
            name: Some(CompactString::const_new("build")),
        },
    );
    assert!(
        matches!(reply, ResponsePayload::Err { code, .. } if code == WireErrorCode::AlreadySubscribed),
        "second Subscribe is refused; got {reply:?}",
    );

    // The missed bookkeeping survives the refusal: gate fires
    // before `transition_to_sub` runs, so neither field is reset.
    let conn = rig
        .driver
        .hub
        .conn_ref(token)
        .expect("conn still in map after refusal");
    match &conn.role {
        ConnRole::Sub {
            missed,
            first_dropped_at,
            ..
        } => {
            assert_eq!(
                *missed, 7,
                "missed window preserved across refused Subscribe"
            );
            assert_eq!(*first_dropped_at, Some(stamped_at));
        }
        ConnRole::Reqs => panic!("conn unexpectedly fell back to Reqs"),
    }

    let _ = rig.driver.begin_shutdown();
}

/// An oversize response refused against a previously-empty queue
/// terminates the conn inline within the same call — the linger
/// path is structurally closed. [`super::hub::DriverHub::enqueue_response`]'s
/// `Refused` arm internally calls
/// [`super::hub::DriverHub::try_terminate_if_idle`], which (queue
/// empty + close armed) deregisters the stream and removes the entry
/// from the conn map.
///
/// Synthesises the oversize payload via a `ResponsePayload::Err`
/// whose `error: String` is padded past the cap; the serialized JSON
/// envelope adds ~50 bytes of `{"kind":"err","code":"…","error":"…"}`
/// framing, so the padded payload comfortably exceeds the cap with
/// no need for a custom carrier or a `cfg(test)` constant override.
/// The code value [`WireErrorCode::Busy`] is incidental — the test
/// pins the oversize-refusal mechanism, not the error vocabulary.
#[test]
fn oversize_response_terminates_conn_inline_when_queue_empty() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let _client = ipc_connect(&rig);
    arm_zero_timeout(&mut rig);
    let _ = rig.driver.tick();
    assert_eq!(
        rig.driver.hub.conn_count(),
        1,
        "client accepted on the first tick",
    );

    let token = mio::Token(TOKEN_CONN_BASE);

    let padding = "x".repeat(WRITE_QUEUE_HIGH_WATER + 1);
    let huge = ResponsePayload::Err {
        code: WireErrorCode::Busy,
        error: padding,
    };
    let outcome = rig.driver.hub.enqueue_response(token, &huge);
    assert_eq!(
        outcome,
        EnqueueOutcome::Refused,
        "oversize response is refused by the capacity gate",
    );
    assert_eq!(
        rig.driver.hub.conn_count(),
        0,
        "conn terminated inline: refusal armed close, queue was empty, \
         try_terminate_if_idle ran",
    );
    assert!(
        rig.driver.hub.conn_ref(token).is_none(),
        "conn removed from map",
    );

    let _ = rig.driver.begin_shutdown();
}

/// An oversize response refused against a queue that already holds
/// bytes (a normal response from a prior verb) does NOT terminate
/// inline — `try_terminate_if_idle`'s queue-empty precondition is
/// not met. The conn stays in the map, `drain_writable` flushes the
/// existing bytes on the next WRITABLE edge, observes
/// `close_after_flush`, and terminates on the flush edge.
///
/// The flush-then-terminate half of the linger fix. Together with
/// [`oversize_response_terminates_conn_inline_when_queue_empty`] the
/// two tests cover both shapes (queue-empty-at-arm and
/// queue-non-empty-at-arm) of an over-water-armed close.
#[test]
fn oversize_response_arms_close_then_flushes_then_terminates() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    let _client = ipc_connect(&rig);
    arm_zero_timeout(&mut rig);
    let _ = rig.driver.tick();
    let token = mio::Token(TOKEN_CONN_BASE);

    // Queue a normal-sized response first so the write_queue is
    // non-empty when the oversize one is refused.
    let small = ResponsePayload::Ok;
    assert_eq!(
        rig.driver.hub.enqueue_response(token, &small),
        EnqueueOutcome::Accepted,
    );

    let padding = "x".repeat(WRITE_QUEUE_HIGH_WATER + 1);
    let huge = ResponsePayload::Err {
        code: WireErrorCode::Busy,
        error: padding,
    };
    assert_eq!(
        rig.driver.hub.enqueue_response(token, &huge),
        EnqueueOutcome::Refused,
    );

    // Inline-terminate did NOT fire — queue still holds the small
    // response. close_after_flush is armed for the next drain.
    assert_eq!(
        rig.driver.hub.conn_count(),
        1,
        "conn lives, queue non-empty"
    );
    {
        let conn = rig.driver.hub.conn_ref(token).expect("conn lives");
        assert!(conn.close_after_flush, "armed");
        assert!(!conn.write_queue.is_empty(), "small response queued");
    }

    // Drive ticks: arm_writable_interests adds WRITABLE; drain_writable
    // flushes the small response; queue empties; close_after_flush
    // observed; terminate fires.
    for _ in 0..5 {
        arm_zero_timeout(&mut rig);
        let _ = rig.driver.tick();
        if rig.driver.hub.conn_count() == 0 {
            break;
        }
    }
    assert_eq!(
        rig.driver.hub.conn_count(),
        0,
        "flushed-then-terminated within a handful of ticks",
    );

    let _ = rig.driver.begin_shutdown();
}

/// End-to-end witness: the Missed marker carries the FIRST-DROP
/// timestamp, not the flush-time stamp. Operators reading a
/// `_missed` line on the wire see when the drops began (the
/// start-of-window time), which is the load-bearing detail for
/// incident forensics — the marker reaches the wire well after the
/// drops happened, so a flush-time stamp would point at the wrong
/// part of the timeline.
///
/// Drives the full fan-out path via
/// [`super::hub::DriverHub::dispatch_to_subscribers`]: a saturated
/// queue throttles the diag (missed = 1, first_dropped_at = at_drop),
/// the queue clears (simulating drain), a second dispatch lands at
/// at_flush AND flushes the marker. The marker's wire `at` is
/// at_drop.
#[test]
fn missed_marker_uses_first_dropped_at_when_flushed() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();
    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("build attached");

    let mut client = ipc_connect(&rig);
    let _ = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Subscribe { name: None },
    );
    let token = mio::Token(TOKEN_CONN_BASE);

    // Sanity: the conn is in Sub mode after the ack flushed.
    {
        let conn = rig
            .driver
            .hub
            .conn_ref(token)
            .expect("conn lives post-Subscribe");
        assert!(matches!(conn.role, ConnRole::Sub { .. }));
    }

    // Pre-fill the queue near high-water so the next dispatch
    // overflows the capacity gate.
    {
        let conn = rig.driver.hub.conn_mut(token).expect("conn lives");
        let fill = WRITE_QUEUE_HIGH_WATER - 10;
        conn.write_queue.extend(std::iter::repeat_n(b'x', fill));
    }

    // Dispatch a diag at at_drop — drops (combined would overflow).
    let at_drop = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
    let wire_at_drop = WireTime::from(at_drop);
    let diag = Diagnostic::SubAttached {
        sub: sid,
        name: CompactString::const_new("build"),
        source_promoter: None,
    };
    rig.driver
        .hub
        .dispatch_to_subscribers(&diag, at_drop, &wire_at_drop, None);
    {
        let conn = rig.driver.hub.conn_ref(token).expect("conn lives");
        match &conn.role {
            ConnRole::Sub {
                missed,
                first_dropped_at,
                ..
            } => {
                assert_eq!(*missed, 1);
                assert_eq!(*first_dropped_at, Some(at_drop));
            }
            ConnRole::Reqs => panic!("expected Sub role"),
        }
    }

    // Simulate the wire draining — clear the queue so the next
    // dispatch fits the marker + diag.
    {
        let conn = rig.driver.hub.conn_mut(token).expect("conn lives");
        conn.write_queue.clear();
    }

    // Dispatch a fresh diag at at_flush — fits, AND the marker
    // flushes ahead of it carrying at_drop as its `at`.
    let at_flush = SystemTime::UNIX_EPOCH + Duration::from_secs(500);
    let wire_at_flush = WireTime::from(at_flush);
    rig.driver
        .hub
        .dispatch_to_subscribers(&diag, at_flush, &wire_at_flush, None);
    let conn = rig.driver.hub.conn_ref(token).expect("conn lives");
    match &conn.role {
        ConnRole::Sub {
            missed,
            first_dropped_at,
            ..
        } => {
            assert_eq!(*missed, 0, "marker flushed; missed reset");
            assert_eq!(*first_dropped_at, None, "first_dropped_at reset on flush");
        }
        ConnRole::Reqs => panic!("expected Sub role"),
    }

    let queued: Vec<u8> = conn.write_queue.iter().copied().collect();
    let lines: Vec<&[u8]> = queued
        .split(|&b| b == b'\n')
        .filter(|l| !l.is_empty())
        .collect();
    assert_eq!(lines.len(), 2, "marker + diag");

    let marker_v: serde_json::Value =
        serde_json::from_slice(lines[0]).expect("marker is valid JSON");
    assert_eq!(marker_v["diag"], "_missed");
    assert_eq!(marker_v["count"], 1);
    let expected_at_drop = humantime::format_rfc3339_seconds(at_drop).to_string();
    let expected_at_flush = humantime::format_rfc3339_seconds(at_flush).to_string();
    assert_eq!(
        marker_v["at"].as_str().expect("at is a string"),
        expected_at_drop,
        "marker carries first-drop time",
    );
    assert_ne!(
        marker_v["at"].as_str().unwrap(),
        expected_at_flush,
        "marker MUST NOT carry flush-time",
    );

    let _ = rig.driver.begin_shutdown();
}

/// IPC `Reload` routes through the driver-side reload pipeline and
/// records `last_reload_via = Ipc`.
///
/// Seeds `loader.config_meta` with the on-disk lstat so the
/// per-tick `apply_config_settle_expiry` (driven by the test rig's
/// zero-timeout arming) is a silent drop. Without this seed the
/// settle-expiry's lstat filter would observe drift against the
/// rig's `dummy_meta()` placeholder and fire an extra reload via
/// `AutoReload`, inflating `reload_count`.
#[test]
fn ipc_reload_via_pipeline_records_ipc_trigger() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    std::fs::write(&cfg_path, "").expect("write empty config");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path.clone());
    rig.driver.loader.config_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(&mut rig, &mut client, &WireRequest::Reload);
    assert!(matches!(reply, ResponsePayload::Ok), "got {reply:?}");

    assert_eq!(rig.driver.driver_state.reload_count, 1);
    assert!(matches!(
        rig.driver.driver_state.last_reload_via,
        Some(ReloadTrigger::Ipc),
    ));
}

/// Disable happy path over IPC: removes the Sub from the engine
/// and records the runtime override.
#[test]
fn ipc_disable_static_sub_detaches_and_records_override() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();
    assert!(rig.driver.engine.subs().find_by_name("build").is_some());

    let mut client = ipc_connect(&rig);
    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Disable {
            name: CompactString::const_new("build"),
        },
    );
    assert!(matches!(reply, ResponsePayload::Ok), "got {reply:?}");

    assert!(rig.driver.engine.subs().find_by_name("build").is_none());
    assert!(
        rig.driver
            .disabled_runtime
            .contains(&CompactString::const_new("build")),
    );

    let _ = rig.driver.begin_shutdown();
}

#[test]
fn ipc_disable_unknown_name_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Disable {
            name: CompactString::const_new("ghost"),
        },
    );
    match reply {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, WireErrorCode::UnknownSub);
            assert!(error.contains("no watch named ghost"));
        }
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(rig.driver.disabled_runtime.is_empty());
}

#[test]
fn ipc_disable_unknown_dynamic_shape_name_returns_unknown_sub() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Disable {
            name: CompactString::const_new("promoter@/tmp/x"),
        },
    );
    match reply {
        ResponsePayload::Err { code, error } => {
            assert_eq!(code, WireErrorCode::UnknownSub);
            assert!(error.contains("no watch named promoter@/tmp/x"));
        }
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(rig.driver.disabled_runtime.is_empty());
}

/// Disable against a real dynamic (promoter-spawned) Sub returns
/// [`WireErrorCode::DynamicSubNoOp`]. Inject a dynamic Sub directly
/// so the gate (which reads `source_promoter`) fires.
#[test]
fn ipc_disable_dynamic_sub_returns_dynamic_no_op() {
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, ClassSet, EffectScope, ExecAction, ProfileIdentity,
        PromoterId, ScanConfig, SubAttachAnchor, SubAttachRequest, SubParams,
    };

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
            .is_some()
    );

    let mut client = ipc_connect(&rig);
    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Disable {
            name: CompactString::const_new(dynamic_name),
        },
    );
    match reply {
        ResponsePayload::Err { code, .. } => assert_eq!(code, WireErrorCode::DynamicSubNoOp),
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(rig.driver.disabled_runtime.is_empty());
    assert!(
        rig.driver
            .engine
            .subs()
            .find_by_name(dynamic_name)
            .is_some()
    );

    let _ = rig.driver.begin_shutdown();
}

#[test]
fn ipc_disable_already_disabled_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    rig.driver
        .disabled_runtime
        .insert(CompactString::const_new("build"));

    let mut client = ipc_connect(&rig);
    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Disable {
            name: CompactString::const_new("build"),
        },
    );
    match reply {
        ResponsePayload::Err { code, .. } => assert_eq!(code, WireErrorCode::NotDisabled),
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(rig.driver.engine.subs().find_by_name("build").is_some());

    let _ = rig.driver.begin_shutdown();
}

/// Enable happy path: clears the override AND re-attaches via
/// Input::AttachSub.
#[test]
fn ipc_enable_clears_override_and_reattaches() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();

    // Drive a real disable round-trip first so the override is
    // recorded by the production path, mirroring the lifecycle a
    // disable→enable client sees.
    let mut client_a = ipc_connect(&rig);
    let _ = ipc_round_trip(
        &mut rig,
        &mut client_a,
        &WireRequest::Disable {
            name: CompactString::const_new("build"),
        },
    );
    assert!(rig.driver.engine.subs().find_by_name("build").is_none());
    assert_eq!(rig.driver.disabled_runtime.len(), 1);

    let mut client_b = ipc_connect(&rig);
    let reply = ipc_round_trip(
        &mut rig,
        &mut client_b,
        &WireRequest::Enable {
            name: CompactString::const_new("build"),
        },
    );
    assert!(matches!(reply, ResponsePayload::Ok), "got {reply:?}");
    assert!(rig.driver.engine.subs().find_by_name("build").is_some());
    assert!(rig.driver.disabled_runtime.is_empty());

    let _ = rig.driver.begin_shutdown();
}

#[test]
fn ipc_enable_not_disabled_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    let mut client = ipc_connect(&rig);

    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Enable {
            name: CompactString::const_new("nothing"),
        },
    );
    match reply {
        ResponsePayload::Err { code, .. } => assert_eq!(code, WireErrorCode::NotDisabled),
        other => panic!("expected Err, got {other:?}"),
    }
}

/// Enable against a runtime-disabled name whose TOML entry no longer
/// exists clears the override AND returns [`WireErrorCode::TomlDisabled`].
#[test]
fn ipc_enable_toml_disabled_clears_override_returns_err() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);
    rig.driver
        .disabled_runtime
        .insert(CompactString::const_new("orphan"));

    let mut client = ipc_connect(&rig);
    let reply = ipc_round_trip(
        &mut rig,
        &mut client,
        &WireRequest::Enable {
            name: CompactString::const_new("orphan"),
        },
    );
    match reply {
        ResponsePayload::Err { code, .. } => assert_eq!(code, WireErrorCode::TomlDisabled),
        other => panic!("expected Err, got {other:?}"),
    }
    assert!(rig.driver.disabled_runtime.is_empty());
}

// ============================================================
// Subscribe ack-ordering on the wire
// ============================================================

/// Subscribe → diag emission → ack-before-diag on the wire. The
/// handler enqueues ack bytes BEFORE flipping the conn role, so a
/// same-tick diag pushed via `forward` AFTER the role flip lands
/// AFTER the ack on the wire.
///
/// Sequencing:
/// 1. Client writes Subscribe.
/// 2. Drive ticks until the conn is accepted AND `handle_subscribe`
///    has run (ack bytes enqueued, role flipped to Sub). We detect
///    this by polling `hub.conn_count() == 1` AND a per-tick check
///    on the role through the wire (the ack bytes must be queued —
///    not yet flushed if WRITABLE hasn't fired, but the role is the
///    structural witness). Easier surrogate: drive ticks until the
///    client's `read` returns the ack bytes.
/// 3. Once the ack is on the wire (so the role HAS flipped),
///    `forward` the diagnostic. The diag's bytes land after the ack
///    bytes that were just consumed; they enter the conn's empty
///    write_queue.
/// 4. Drive ticks to flush the diag.
/// 5. Assert both lines parsed, ack first, diag second.
#[test]
fn subscribe_ack_precedes_diag_on_wire() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = config_with_one_watch(tmp.path());
    let mut rig = rig_for(config, cfg_path);
    let _ = rig.driver.run_initial_attach();
    let sid = rig
        .driver
        .engine
        .subs()
        .find_by_name("build")
        .expect("attached");

    let mut client = ipc_connect(&rig);
    client
        .set_read_timeout(Some(Duration::from_millis(50)))
        .expect("set read timeout");
    write_request(&mut client, &WireRequest::Subscribe { name: None });

    // Drive ticks until the ack lands on the wire. The presence of
    // the ack proves the role flipped to Sub — only then can the
    // diag enqueue against this conn.
    let mut ack_bytes: Vec<u8> = Vec::new();
    let deadline_ack = Instant::now() + Duration::from_secs(2);
    while Instant::now() < deadline_ack {
        arm_zero_timeout(&mut rig);
        let _ = rig.driver.tick();
        let mut chunk = [0u8; 1024];
        match client.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                ack_bytes.extend_from_slice(&chunk[..n]);
                if ack_bytes.contains(&b'\n') {
                    break;
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => break,
        }
    }
    assert!(
        ack_bytes.contains(&b'\n'),
        "ack must arrive within deadline; got {:?}",
        String::from_utf8_lossy(&ack_bytes),
    );
    let resp0: ResponsePayload =
        serde_json::from_slice(ack_bytes.split(|&b| b == b'\n').next().unwrap())
            .expect("ack line parses as ResponsePayload");
    assert!(
        matches!(resp0, ResponsePayload::SubscribeAck { .. }),
        "first wire line is the SubscribeAck; got {resp0:?}",
    );

    // Role has flipped to Sub by now (ack proved it). Synthesize a
    // diag and route it through `forward` — production fan-out path
    // sees role=Sub and pushes the diag bytes into write_queue.
    let mut out = StepOutput::default();
    out.diagnostics.push(specter_core::Diagnostic::SubAttached {
        sub: sid,
        name: CompactString::const_new("build"),
        source_promoter: None,
    });
    let _ = rig.driver.forward(out);

    // Drive ticks to flush the diag.
    let deadline_diag = Instant::now() + Duration::from_secs(2);
    let mut diag_bytes: Vec<u8> = Vec::new();
    while Instant::now() < deadline_diag {
        arm_zero_timeout(&mut rig);
        let _ = rig.driver.tick();
        let mut chunk = [0u8; 1024];
        match client.read(&mut chunk) {
            Ok(0) => break,
            Ok(n) => {
                diag_bytes.extend_from_slice(&chunk[..n]);
                if diag_bytes.contains(&b'\n') {
                    break;
                }
            }
            Err(e)
                if e.kind() == std::io::ErrorKind::WouldBlock
                    || e.kind() == std::io::ErrorKind::TimedOut => {}
            Err(_) => break,
        }
    }
    assert!(
        diag_bytes.contains(&b'\n'),
        "diag must arrive within deadline; got {:?}",
        String::from_utf8_lossy(&diag_bytes),
    );
    let diag_v: serde_json::Value =
        serde_json::from_slice(diag_bytes.split(|&b| b == b'\n').next().unwrap())
            .expect("diag line parses as JSON");
    assert_eq!(
        diag_v.get("diag").and_then(|v| v.as_str()),
        Some("SubAttached"),
        "second wire line is the diag; got {diag_v:?}",
    );

    let _ = rig.driver.begin_shutdown();
}

// ============================================================
// Channel wake-after-send: prober + effect senders
// ============================================================

/// The prober/effect senders' send-then-wake protocol is provable
/// through the rig's `prober_response_tx` + `waker.wake()` pair: a
/// send-then-wake fires the `TOKEN_WAKER` edge and the next
/// `next_inputs` call drains the message.
#[test]
fn prober_response_send_then_wake_drains_through_token_waker() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    rig.prober_response_tx
        .send(Input::TimerExpired {
            profile: specter_core::ProfileId::default(),
            kind: specter_core::TimerKind::Settle,
            id: specter_core::TimerId::default(),
        })
        .expect("send into wake'd channel");
    rig.waker.wake().expect("fire wake edge");

    let start = Instant::now();
    arm_zero_timeout(&mut rig);
    let _ = rig.driver.tick();
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_millis(500),
        "wake should unblock immediately, took {elapsed:?}",
    );
    // The message is consumed by `tick`'s drain pass; the rig's
    // prober_response_tx clone keeps the channel alive.
}

/// The effect-complete sender protocol mirrors the prober one. Pin
/// the channel routing — send-then-wake delivers an
/// `Input::EffectComplete` through the Hub's `TOKEN_WAKER` arm.
#[test]
fn effect_complete_send_then_wake_drains_through_token_waker() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    rig.effect_complete_tx
        .send(Input::EffectComplete(specter_core::EffectCompletion {
            sub: SubId::default(),
            key: specter_core::DedupKey::Subtree {
                sub: SubId::default(),
                profile: specter_core::ProfileId::default(),
            },
            outcome: specter_core::EffectOutcome::Ok,
        }))
        .expect("send into wake'd channel");
    rig.waker.wake().expect("fire wake edge");

    arm_zero_timeout(&mut rig);
    let _ = rig.driver.tick();
    // No panic / no hang ⇒ the tick consumed the EffectComplete via
    // the Hub's TOKEN_WAKER arm.
}

// ============================================================
// Drain order: sensor inputs precede effect completions
// ============================================================

/// Sensor inputs drain BEFORE effect completions: pre-arm the
/// fs-event queue + the effect-complete channel, drive one tick, and
/// observe that the FsEvent reached engine.step first (via the
/// MockProber recording — an unknown FsEvent produces no probe; an
/// EffectComplete for an unknown sub emits no probe either; the
/// drain order itself is what we pin via the routing pattern).
///
/// We can't directly observe step ordering from the engine surface
/// (no per-tick `last_input` accessor), but we CAN observe that
/// neither input crashes the engine and the tick returns Continue —
/// any drain-order regression that tried to handle effect_complete
/// first against an FsEvent-bearing resource would surface as a
/// state-machine routing bug. The structural ordering is enforced
/// in `tick.rs`; this test is a regression smoke that the wiring
/// reaches both inputs.
#[test]
fn fs_event_and_effect_complete_both_drain_in_one_tick() {
    let tmp = tempfile::TempDir::new().unwrap();
    let cfg_path = tmp.path().join("specter.toml");
    let config = Config::from_str("").expect("empty config parses");
    let mut rig = rig_for(config, cfg_path);

    // Inject an FsEvent via the MockFsWatcher (sets a readable edge
    // on the watcher fd; drain_watcher reads it on the next poll).
    let r = ResourceId::default();
    rig.driver
        .hub
        .watcher_mut()
        .inject(r, specter_core::FsEvent::Modified);
    // Queue an EffectComplete via the wake'd channel.
    rig.effect_complete_tx
        .send(Input::EffectComplete(specter_core::EffectCompletion {
            sub: SubId::default(),
            key: specter_core::DedupKey::Subtree {
                sub: SubId::default(),
                profile: specter_core::ProfileId::default(),
            },
            outcome: specter_core::EffectOutcome::Ok,
        }))
        .expect("queue effect complete");
    rig.waker.wake().expect("wake");

    // Both inputs reach the engine in one tick. The drain order
    // (sensor before effects) is enforced inside `tick`; observation
    // is the absence of a panic / hang and the tick returning
    // Continue.
    arm_zero_timeout(&mut rig);
    let outcome = rig.driver.tick();
    assert_eq!(outcome, TickOutcome::Continue);
}
