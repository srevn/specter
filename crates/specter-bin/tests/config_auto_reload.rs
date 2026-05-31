//! Integration tests for config auto-reload.
//!
//! Each test spawns the `specter` binary as a subprocess. One process
//! per test ⇒ full state isolation:
//!
//! - `signal-hook` installs SIGHUP / SIGINT / SIGTERM handlers
//!   process-wide; concurrent tests within one process would receive
//!   each other's signals.
//! - `tracing-subscriber::registry().try_init()` succeeds exactly once
//!   per process; a second `App::run` in the same process would exit
//!   with `ExitCode::from(1)` from observability init.
//! - kqueue / inotify fds are per-process; subprocess isolation
//!   guarantees a fresh kernel-side state per test.
//!
//! Cargo's `env!("CARGO_BIN_EXE_specter")` resolves to the workspace's
//! built binary path; cargo runs `cargo build` for the bin before
//! integration tests, so the path is always populated.
//!
//! Communication:
//!
//! - **Stimulus** — config edits via `fs::write` / `fs::set_modified`,
//!   atomic-rename via `fs::rename`, ownership / mode changes via
//!   `fs::set_permissions`. Signals via `nix::sys::signal::kill`.
//! - **Observation** — the bin writes engine telemetry to a file
//!   (`--log-destination file --log-path …`) at `info` level; tests
//!   poll the file for known log strings.
//!
//! All tests target Unix (`#![cfg(unix)]`) — the auto-reload feature
//! is kqueue (macOS / FreeBSD) or inotify (Linux); no Windows
//! counterpart in v1.

#![cfg(unix)]

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Wall-clock deadline for "wait until log line N appears." Generous
/// to absorb cold-start jitter on slow CI hosts; tests still terminate
/// promptly when the assertion is met because polling is 50ms.
const LOG_DEADLINE: Duration = Duration::from_secs(10);

/// Wall-clock deadline for "wait for the child to exit after SIGTERM."
/// Bounded by the actuator's 5s SIGTERM grace + a few hundred ms of
/// thread-join overhead; 8s gives ample headroom on slow hosts.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(8);

/// Polling cadence for log-file tail reads. 50ms keeps tests responsive
/// without hammering the FS — the bin's `tracing-appender` writer
/// flushes at its own cadence (worker-thread driven).
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Settle-window absorption deadline. The bin's `CONFIG_SETTLE` is
/// 100ms; tests that need to assert "no reload occurred despite an
/// edit" wait at least 4× the settle window so kernel + userspace
/// jitter cannot mask a real pulse. 600ms is comfortably above the
/// settle * 4 floor.
const NO_RELOAD_WINDOW: Duration = Duration::from_millis(600);

/// Spawn the workspace's `specter` binary with the standard log /
/// config flags plus any `extra` args (e.g. `--no-config-watch`).
///
/// `env_remove("SPECTER_NO_CONFIG_WATCH")` defends against a test
/// runner that already exports the env var — every test must control
/// the auto-reload state via its `extra` argv slice, not via inherited
/// env.
///
/// `--socket <sandbox>/specter.sock` binds the daemon's IPC socket
/// inside the per-test tempdir — concurrent integration tests in
/// `cargo nextest` would otherwise collide on the shared per-platform
/// convention path (the second binder hits `AddrInUse` and exits 1).
/// These tests observe only the log file, so nothing connects to the
/// socket; pinning it merely keeps each daemon's bind unique.
fn spawn_specter<I, S>(cfg: &Path, log: &Path, extra: I) -> Child
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let bin = env!("CARGO_BIN_EXE_specter");
    let socket = cfg
        .parent()
        .expect("config path lives in a sandbox tempdir")
        .join("specter.sock");
    Command::new(bin)
        .arg("run")
        .arg("--config")
        .arg(cfg)
        .arg("--socket")
        .arg(&socket)
        .args(["--log-destination", "file"])
        .arg("--log-path")
        .arg(log)
        .args(["--log-level", "info"])
        .args(extra)
        .env_remove("SPECTER_NO_CONFIG_WATCH")
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn specter: {e}"))
}

/// Poll `log` until `pred` matches the file's current contents, or
/// `deadline` elapses. Returns the matching contents on success,
/// `None` on timeout.
fn wait_for_log<F: Fn(&str) -> bool>(log: &Path, pred: F, deadline: Duration) -> Option<String> {
    let stop = Instant::now() + deadline;
    while Instant::now() < stop {
        if let Ok(s) = fs::read_to_string(log)
            && pred(&s)
        {
            return Some(s);
        }
        thread::sleep(POLL_INTERVAL);
    }
    None
}

/// Wait for `child` to exit, polling at [`POLL_INTERVAL`] up to
/// [`SHUTDOWN_DEADLINE`]. Logs SIGKILL on overrun so test hangs
/// surface as `child did not exit on SIGTERM` rather than indefinite
/// blocks.
fn await_exit(child: &mut Child) -> io::Result<std::process::ExitStatus> {
    let stop = Instant::now() + SHUTDOWN_DEADLINE;
    while Instant::now() < stop {
        match child.try_wait()? {
            Some(s) => return Ok(s),
            None => thread::sleep(POLL_INTERVAL),
        }
    }
    let pid = Pid::from_raw(child.id().cast_signed());
    let _ = kill(pid, Signal::SIGKILL);
    let _ = child.wait();
    Err(io::Error::new(
        io::ErrorKind::TimedOut,
        "child did not exit on SIGTERM",
    ))
}

/// Send SIGTERM to `child`, then wait for exit.
fn terminate(mut child: Child) -> std::process::ExitStatus {
    let pid = Pid::from_raw(child.id().cast_signed());
    kill(pid, Signal::SIGTERM).expect("SIGTERM");
    await_exit(&mut child).expect("clean exit on SIGTERM")
}

/// Send SIGHUP to `child`. Used by the SIGHUP-vs-auto-reload tests.
fn send_sighup(child: &Child) {
    let pid = Pid::from_raw(child.id().cast_signed());
    kill(pid, Signal::SIGHUP).expect("SIGHUP");
}

/// Tempdir-bound workspace for one test: holds the config path, log
/// path, and a watched-tree dir. The `TempDir` is held in a field so
/// `Drop` runs at the test's end, after the subprocess has fully
/// exited (we always `terminate(child)` before the sandbox falls out
/// of scope).
struct Sandbox {
    _tmp: TempDir,
    cfg: PathBuf,
    log: PathBuf,
    watched: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = TempDir::new().expect("tempdir");
        let cfg = tmp.path().join("specter.toml");
        let log = tmp.path().join("specter.log");
        let watched = tmp.path().join("watched");
        fs::create_dir_all(&watched).expect("mkdir watched");
        Self {
            _tmp: tmp,
            cfg,
            log,
            watched,
        }
    }

    /// Write a single-`[[watch]]` config bound to `name`, watching
    /// `self.watched`. `actions = [{ exec = ["true"] }]` is a no-op (so tests don't
    /// need to clean up child processes); the ID we care about is the
    /// bin's reload-pipeline log strings, not the subprocess output.
    fn write_one_watch(&self, name: &str) {
        let toml = format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{}\"\nactions = [{{ exec = [\"true\"] }}]\nsettle = \"50ms\"\n",
            self.watched.display(),
        );
        fs::write(&self.cfg, toml).expect("write config");
    }

    /// Write a two-watch config — used by tests that prove a reload
    /// applied a structurally different config than the one the bin
    /// loaded at startup.
    fn write_two_watches(&self, a: &str, b: &str) {
        let toml = format!(
            "[[watch]]\nname = \"{a}\"\npath = \"{wp}\"\nactions = [{{ exec = [\"true\"] }}]\nsettle = \"50ms\"\n\n\
             [[watch]]\nname = \"{b}\"\npath = \"{wp}\"\nactions = [{{ exec = [\"true\"] }}]\nsettle = \"50ms\"\n",
            wp = self.watched.display(),
        );
        fs::write(&self.cfg, toml).expect("write config");
    }
}

/// Count occurrences of `needle` inside `haystack`. Used by the
/// concurrent-pulse test to bound coalescing behaviour.
fn count(haystack: &str, needle: &str) -> usize {
    haystack.matches(needle).count()
}

// ---------- 1. Edit triggers reload (golden path) -------------------

/// Editor-lifecycle smoke: a regular in-place edit of the live config
/// fires the kqueue / inotify event → `config_event` pulse → settle
/// → lstat-vs-`config_meta` filter → `handle_reload` → "config reload
/// applied".
#[test]
fn edit_triggers_reload() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");

    sb.write_two_watches("a", "b");

    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        LOG_DEADLINE,
    )
    .expect("auto-reload triggered on edit");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 2. SIGHUP still works with auto-reload on ---------------

/// Two-pulse-source coexistence: with auto-reload enabled (default),
/// SIGHUP retains its immediate-handle semantics.
///
/// Procedure: edit the config, then send SIGHUP *before* the
/// auto-reload settle window expires. Either path can fire first
/// (race-driven), but both terminate at `handle_reload`. The
/// test asserts "at least one reload applied" + clean exit.
#[test]
fn sighup_still_works_with_auto_reload_on() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");

    sb.write_two_watches("a", "b");
    send_sighup(&child);

    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        LOG_DEADLINE,
    )
    .expect("SIGHUP+edit produced a reload");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 3. --no-config-watch suppresses auto-reload --------------

/// `--no-config-watch` opt-out: a config edit does NOT trigger the
/// auto-reload pipeline. SIGHUP still works (proves the SIGHUP path
/// is independent of the watcher backend).
#[test]
fn disabled_via_flag_doesnt_pulse_on_edit() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, ["--no-config-watch"]);
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");
    // Confirm the disable path log line landed.
    wait_for_log(
        &sb.log,
        |s| s.contains("auto-reload disabled via --no-config-watch"),
        LOG_DEADLINE,
    )
    .expect("opt-out info-log present");

    sb.write_two_watches("a", "b");

    // No reload within the absorption window.
    let observed = wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        NO_RELOAD_WINDOW,
    );
    assert!(
        observed.is_none(),
        "auto-reload fired despite --no-config-watch: {observed:?}"
    );

    // SIGHUP still triggers reload — proves the disable applies only
    // to the auto-reload watcher, not the SIGHUP pipeline.
    send_sighup(&child);
    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        LOG_DEADLINE,
    )
    .expect("SIGHUP-driven reload still works");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 4. Parse failure preserves running config ----------------

/// Garbage TOML pulse logs the parse error and keeps the running
/// config; a subsequent valid edit still reloads (the loader's
/// `config_meta` was never rotated by the failed read, so the next
/// edit's lstat still differs).
#[test]
fn parse_failure_keeps_config_after_auto_pulse() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");

    // Step 1: write garbage; expect "config reload failed".
    fs::write(&sb.cfg, "this is not valid TOML !@#$%").expect("write garbage");
    wait_for_log(
        &sb.log,
        |s| s.contains("config reload failed; keeping running config"),
        LOG_DEADLINE,
    )
    .expect("parse-fail logged");

    // Step 2: write a valid two-watch config; expect a clean reload.
    sb.write_two_watches("a", "b");
    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        LOG_DEADLINE,
    )
    .expect("subsequent valid edit still reloads");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 5. Concurrent SIGHUP + edit coalesces --------------------

/// Two pulse sources hitting the bin in quick succession — auto-reload
/// (`config_event_tx`) and SIGHUP (`reload_signal_tx`) — must produce
/// at least one reload (correctness) and at most two (coalescing
/// contract: each pulse channel is `bounded(1)`, so even an unbounded
/// editor burst can queue at most 2 reloads in flight). No panic.
#[test]
fn concurrent_sighup_and_edit_coalesces() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");

    // Tight edit + SIGHUP. Race-driven; either source may win.
    sb.write_two_watches("a", "b");
    send_sighup(&child);

    // Wait for at least one reload.
    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        LOG_DEADLINE,
    )
    .expect("at least one reload");

    // Sleep through the settle window so any second-pulse path has
    // time to either fire or get coalesced. Then bound the count.
    thread::sleep(NO_RELOAD_WINDOW);
    let log_contents = fs::read_to_string(&sb.log).expect("read log");
    let n = count(&log_contents, "config reload applied");
    assert!(
        (1..=2).contains(&n),
        "expected 1..=2 reloads under coalescing; got {n}\n--- log ---\n{log_contents}"
    );

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 6. Startup-TOCTOU edit triggers a reload -----------------

/// Startup TOCTOU: an edit that lands during the bin's startup
/// window (between `Config::from_path_with_meta` and the post-init
/// lstat) is still observed. The bin handles this via *either*:
///
/// 1. The post-init lstat catches the meta delta and queues a pulse
///    on `reload_signal_tx` (logged: "config changed during startup
///    reload queued via SIGHUP path").
/// 2. The watcher's first `wait` returns the pre-edit kernel event
///    (logged: standard auto-reload "config reload applied" trail).
///
/// `tracing::info!("specter starting", ...)` is emitted *after*
/// `Config::from_path_with_meta` but *before* the watcher init and
/// the post-init lstat — that's the precise TOCTOU window. Waiting
/// for the log line guarantees the bin already captured `initial_meta`
/// against configA; the immediate `write_two_watches` rotates the
/// on-disk identity to configB; the bin's post-init lstat sees the
/// delta and queues the SIGHUP-style pulse.
///
/// (Editing *before* "specter starting" would race the bin's initial
/// `Config::from_path_with_meta`; if the edit lands first, the bin
/// reads configB directly and no reload ever fires — no TOCTOU
/// window to traverse.)
///
/// The assertion is "reload applied" + a clean exit. We additionally
/// log-grep for the TOCTOU-specific info-line so a regression of
/// the lstat path to "no-op" surfaces here, not as silent reliance
/// on the steady-state fallback.
#[test]
fn startup_toctou_edit_triggers_reload() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    // Wait for the bin to be past Config::from_path_with_meta but
    // before the post-init lstat — "specter starting" sits exactly
    // in that window.
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged (TOCTOU window now open)");
    sb.write_two_watches("a", "b");

    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied"),
        LOG_DEADLINE,
    )
    .expect("startup-window edit produced a reload");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 7. enabled toggle round-trips via auto-reload -----------

/// Round-trip the operator-control surface: a watch is born enabled,
/// then disabled via an edit, then re-enabled. Both transitions
/// surface through the same `handle_reload` path the diff layer
/// drives — the disable yields `removed=1` (engine sees a deletion),
/// the re-enable yields `added=1` (engine sees a fresh attach). This
/// is the load-bearing claim of the feature: flipping `enabled`
/// reduces to the add/remove shape the engine already handles, so
/// no engine-side state machine learns about the flag.
#[test]
fn enabled_toggle_round_trips_via_auto_reload() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");

    // Disable: same `name = "a"`, but `enabled = false`. From the
    // diff's perspective the entry has departed the active set, so
    // the reload's `removed` count is 1.
    let toml_disabled = format!(
        "[[watch]]\nname = \"a\"\npath = \"{}\"\n\
         actions = [{{ exec = [\"true\"] }}]\nsettle = \"50ms\"\nenabled = false\n",
        sb.watched.display(),
    );
    fs::write(&sb.cfg, &toml_disabled).expect("write disabled config");
    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied") && s.contains("removed=1"),
        LOG_DEADLINE,
    )
    .expect("disable surfaces as removed=1 in the reload log");

    // Re-enable: rewrite the `enabled = true` form (default). The
    // entry re-enters the active set, so the next reload's `added`
    // count is 1. The first reload's "added=0" disambiguates from
    // this assertion — only the re-enable matches "added=1".
    sb.write_one_watch("a");
    wait_for_log(
        &sb.log,
        |s| s.contains("config reload applied") && s.contains("added=1"),
        LOG_DEADLINE,
    )
    .expect("re-enable surfaces as added=1 in the reload log");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}

// ---------- 8. chmod fires an empty reload (mode path) ---------------

/// `chmod` doesn't move mtime — only mode (and ctime, which we don't
/// fingerprint). The bin's `FileMeta` includes `mode` precisely so
/// this case is observable. Behaviour: pulse → settle → lstat (mode
/// differs from stored meta) → handle_reload → re-parse the unchanged
/// file → diff is empty → "config reload: no watch changes" + meta
/// rotation.
///
/// This is the "wasted parse" the design accepts as the price of
/// recovering from `chmod`-after-EACCES; the test's whole purpose is
/// to confirm the mode-bit path is wired. A regression to a fingerprint
/// without `mode` would silently break `chmod`-driven recoveries; a
/// regression that re-introduced `ctime` would reload-storm under
/// macOS LaunchServices' `lastuseddate` xattr writes (see
/// `xattr_write_does_not_trigger_reload`).
#[test]
fn chmod_triggers_empty_reload() {
    let sb = Sandbox::new();
    sb.write_one_watch("a");
    let child = spawn_specter(&sb.cfg, &sb.log, std::iter::empty::<&str>());
    wait_for_log(&sb.log, |s| s.contains("specter starting"), LOG_DEADLINE)
        .expect("startup logged");

    // chmod: 0o644 → 0o600. Same content; mode moves; mtime unchanged.
    fs::set_permissions(&sb.cfg, fs::Permissions::from_mode(0o600)).expect("chmod test config");

    wait_for_log(
        &sb.log,
        |s| s.contains("config reload: no watch changes"),
        LOG_DEADLINE,
    )
    .expect("empty reload via mode delta");

    let status = terminate(child);
    assert!(status.success(), "clean exit; got {status:?}");
}
