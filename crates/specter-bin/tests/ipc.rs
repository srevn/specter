//! Integration tests for the operator IPC surface.
//!
//! Each test spawns the `specter` binary as a subprocess against a
//! per-test sandbox tempdir, then drives client behaviour over real
//! `UnixStream` pairs. Concurrent nextest runs are isolated by
//! pointing `TMPDIR` / `XDG_RUNTIME_DIR` at the sandbox dir, so the
//! daemon's default socket path lands inside the per-test scratch
//! space.
//!
//! Tests mirror the discipline established in
//! `config_auto_reload.rs`: one process per test, log-file polling
//! for observation, SIGTERM-then-await for clean shutdown.

#![cfg(unix)]

use nix::sys::signal::{Signal, kill};
use nix::unistd::Pid;
use serde::Deserialize;
use std::ffi::OsStr;
use std::fs;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::fs::PermissionsExt;
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Wall-clock budget for "the daemon has come up and bound its socket."
const STARTUP_DEADLINE: Duration = Duration::from_secs(10);

/// Wall-clock budget for shutdown after SIGTERM. Matches the
/// established convention from `config_auto_reload.rs`.
const SHUTDOWN_DEADLINE: Duration = Duration::from_secs(8);

/// Poll cadence shared with `config_auto_reload.rs`.
const POLL_INTERVAL: Duration = Duration::from_millis(50);

/// Tempdir-bound sandbox for one IPC integration test. Holds the
/// config + log paths and exposes the synthesised socket path the
/// daemon resolves (`$TMPDIR/specter.sock`).
struct Sandbox {
    _tmp: TempDir,
    dir: PathBuf,
    cfg: PathBuf,
    log: PathBuf,
    socket: PathBuf,
}

impl Sandbox {
    fn new() -> Self {
        let tmp = TempDir::new().expect("tempdir");
        let dir = tmp.path().to_path_buf();
        let cfg = dir.join("specter.toml");
        let log = dir.join("specter.log");
        let socket = dir.join("specter.sock");
        // Minimal valid config — no watches, no promoters. Tests
        // that need watches write a richer TOML themselves.
        fs::write(&cfg, "").expect("write empty config");
        Self {
            _tmp: tmp,
            dir,
            cfg,
            log,
            socket,
        }
    }
}

/// Spawn the workspace's `specter` binary against `sb`. `TMPDIR` and
/// `XDG_RUNTIME_DIR` point at the sandbox so the daemon's default
/// socket lands at `sb.socket` — no `--socket` CLI flag yet, so the
/// per-platform default path resolution is what these tests drive.
fn spawn_specter<I, S>(sb: &Sandbox, extra: I) -> Child
where
    I: IntoIterator<Item = S>,
    S: AsRef<OsStr>,
{
    let bin = env!("CARGO_BIN_EXE_specter");
    Command::new(bin)
        .arg("run")
        .arg("--config")
        .arg(&sb.cfg)
        .args(["--log-destination", "file"])
        .arg("--log-path")
        .arg(&sb.log)
        .args(["--log-level", "info"])
        .args(extra)
        .env_remove("SPECTER_NO_CONFIG_WATCH")
        .env("TMPDIR", &sb.dir)
        .env("XDG_RUNTIME_DIR", &sb.dir)
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn specter: {e}"))
}

/// Wait until the daemon's socket file appears on disk. The bin's
/// init order guarantees the socket exists by the time the engine
/// driver starts the main loop — `bind_socket_atomic` runs before
/// `run_initial_attach` in `App::run`.
fn wait_for_socket(socket: &Path, deadline: Duration) -> bool {
    let stop = Instant::now() + deadline;
    while Instant::now() < stop {
        if socket.exists() {
            // Belt-and-braces: confirm we can also actually connect.
            // The daemon may have bound the listener but not yet
            // spawned the accept thread; one rapid connect-and-drop
            // proves both.
            if let Ok(_s) = UnixStream::connect(socket) {
                return true;
            }
        }
        thread::sleep(POLL_INTERVAL);
    }
    false
}

/// Wait for a log line to appear in the daemon's log file. Mirrors
/// the helper from `config_auto_reload.rs`.
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
/// [`SHUTDOWN_DEADLINE`].
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

/// SIGTERM the daemon and await clean exit.
fn terminate(mut child: Child) -> std::process::ExitStatus {
    let pid = Pid::from_raw(child.id().cast_signed());
    kill(pid, Signal::SIGTERM).expect("SIGTERM");
    await_exit(&mut child).expect("clean exit on SIGTERM")
}

/// Send one IPC request line + read one response line. The
/// connection is closed at the end of the call; the daemon's
/// per-connection thread terminates on EOF.
fn one_shot(socket: &Path, request: &str) -> io::Result<String> {
    let mut stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(Duration::from_secs(5)))?;
    stream.set_write_timeout(Some(Duration::from_secs(2)))?;
    stream.write_all(request.as_bytes())?;
    if !request.ends_with('\n') {
        stream.write_all(b"\n")?;
    }
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line)?;
    Ok(line)
}

/// Mirror of the bin's `StatusResponse` shape — defined here as a
/// minimal `Deserialize` so the test does not depend on the bin's
/// `pub(crate)` types. Field order mirrors the projection's
/// `protocol::StatusResponse`; a rename or removal on the daemon
/// side surfaces as a deserialization failure at the integration
/// boundary.
///
/// `last_reload` flattens on the wire — the daemon emits
/// `last_reload_at` and `last_reload_via` directly alongside the
/// peer fields on the `Some` side, and omits both entirely on the
/// `None` side. The snap mirrors that shape via
/// `#[serde(flatten, default)]` over an `Option<LastReloadSnap>` so
/// a partial wire form (one of the two keys present) fails the
/// integration deserialize loudly, just like the daemon-side wire
/// type.
#[derive(Debug, Deserialize)]
struct StatusResponseSnap {
    uptime_secs: u64,
    start_wall: String,
    reload_count: u64,
    #[serde(flatten, default)]
    last_reload: Option<LastReloadSnap>,
    sub_total: usize,
    sub_disabled_toml: usize,
    sub_disabled_runtime: usize,
    profile_active: usize,
    promoter_active: usize,
    config_path: PathBuf,
    socket_path: PathBuf,
}

/// Mirror of [`WireLastReload`] — the wall-clock + trigger pair the
/// daemon emits as flattened keys (`last_reload_at`,
/// `last_reload_via`) on the `Some` side. Defined here so the
/// integration test's wire-shape pin is independent of the
/// daemon-side type's `pub(crate)` visibility.
#[derive(Debug, Deserialize)]
struct LastReloadSnap {
    last_reload_at: String,
    last_reload_via: ReloadTriggerSnap,
}

/// Mirror of `WireReloadTrigger`. Typed mirror (rather than
/// `Option<String>`) so a rename on the wire side is a compile
/// error at the integration boundary rather than a silently-passing
/// equality on stale text.
#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum ReloadTriggerSnap {
    Sighup,
    Auto,
    Ipc,
}

/// Outer envelope mirror. Internally-tagged on `kind`; only the
/// variants the read verbs exercise are modelled.
#[derive(Debug, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum ResponseSnap {
    Ok,
    Status(StatusResponseSnap),
    List(ListResponseSnap),
    Show(ShowResponseSnap),
    Err {
        code: String,
        error: String,
    },
    #[serde(other)]
    Other,
}

/// Minimal `ListResponse` mirror — just the rows, decoded loosely.
/// Only the fields the integration tests assert against are
/// modelled; unknown wire fields are deserialized into `_` via
/// `serde`'s default deny-unknown-fields-off behaviour.
#[derive(Debug, Deserialize)]
struct ListResponseSnap {
    rows: Vec<ListRowSnap>,
}

#[derive(Debug, Deserialize)]
struct ListRowSnap {
    name: String,
    disabled: Option<DisabledSourceSnap>,
}

/// Mirror of `DisabledSource`. Names + `snake_case` rename match the
/// wire serialization.
#[derive(Debug, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
enum DisabledSourceSnap {
    Runtime,
    Toml,
}

/// Minimal `ShowResponse` mirror. Internally tagged on `status`;
/// the `name` field is read off the `Active` arm by the integration
/// test, and the other arms' fields are kept for `Debug` output on
/// panic — the `#[allow]` muffles dead-code warnings that the
/// deserializer-side read does not silence.
#[derive(Debug, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[allow(dead_code)]
enum ShowResponseSnap {
    Active {
        name: String,
    },
    Disabled {
        name: String,
        source: DisabledSourceSnap,
    },
    Unknown {
        name: String,
    },
}

// ---------- status round-trip ----------------------------------------

/// Headline test: spawn the daemon, send `{"op":"status"}` over the
/// socket, parse a `StatusResponse`. Proves the entire IPC topology
/// (`channels` → `ipc::hub` → `drain_ipc` → `project::status` →
/// reply channel → wire) end-to-end.
#[test]
fn status_round_trip() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let line = one_shot(&sb.socket, r#"{"op":"status"}"#).expect("status request");
    let resp: ResponseSnap = serde_json::from_str(line.trim_end()).expect("parse response");
    match resp {
        ResponseSnap::Status(s) => {
            assert_eq!(s.sub_total, 0, "empty config ⇒ zero attached");
            assert_eq!(s.sub_disabled_toml, 0);
            assert_eq!(s.sub_disabled_runtime, 0);
            assert_eq!(s.profile_active, 0);
            assert_eq!(s.promoter_active, 0);
            assert_eq!(s.reload_count, 0, "no reload triggered yet");
            assert!(
                s.last_reload.is_none(),
                "no reload yet ⇒ paired last_reload absent on the wire",
            );
            assert!(
                !s.start_wall.is_empty(),
                "start_wall projection writes a non-empty RFC 3339 token",
            );
            assert_eq!(s.config_path, sb.cfg, "projection reports config path");
            // uptime_secs can legitimately be 0 on a fast machine.
            // The test only proves the projection ran; the value's
            // monotonicity is covered by the project_tests fixture.
            let _: u64 = s.uptime_secs;
            assert_eq!(s.socket_path, sb.socket, "projection reports bound path");
        }
        other => panic!("expected Status, got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- stale socket recovery ------------------------------------

/// A pre-existing socket file at the bound path is recovered: the
/// daemon connects-then-unlinks (no live peer holds it) and binds
/// fresh. Tests the `sockpath::check_stale_or_remove` arm of the
/// startup path.
#[test]
fn stale_socket_recovery() {
    let sb = Sandbox::new();
    // Stage a regular file at the socket path so the daemon must
    // recover from its presence. A regular file is the easier
    // stand-in for a true stale socket (both look like a present-
    // but-unowned path to `check_stale_or_remove`).
    fs::write(&sb.socket, b"stale daemon footprint").expect("write orphan");
    assert!(sb.socket.exists());

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon failed to recover stale path",
    );

    // Connect to confirm the new listener is live (not the orphan).
    let line = one_shot(&sb.socket, r#"{"op":"status"}"#).expect("status against recovered socket");
    let resp: ResponseSnap = serde_json::from_str(line.trim_end()).expect("parse response");
    assert!(
        matches!(resp, ResponseSnap::Status(_)),
        "expected Status, got {resp:?}",
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- live socket conflict -------------------------------------

/// Two daemons on the same socket path: the second must fail
/// startup with ExitCode::from(1) (sockpath returns AddrInUse). The
/// first daemon's bind is observed via `wait_for_socket` so the race
/// is deterministic.
#[test]
fn live_socket_conflict() {
    let sb = Sandbox::new();
    let first = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "first daemon failed to come up",
    );

    // Spawn a second daemon on the same sandbox (same TMPDIR ⇒ same
    // socket path). It must exit non-zero.
    let mut second = spawn_specter(&sb, std::iter::empty::<&str>());
    let second_exit = await_exit(&mut second).expect("second daemon exited");
    assert!(
        !second_exit.success(),
        "second daemon must NOT have succeeded; got {second_exit:?}",
    );

    let first_exit = terminate(first);
    assert!(first_exit.success(), "first clean exit; got {first_exit:?}");
}

// ---------- socket mode is 0600 --------------------------------------

/// Atomic-rename binding sets the file's mode to 0o600 before the
/// well-known path becomes observable. The lstat-after-startup
/// confirms the chmod ran.
#[test]
fn socket_mode_is_0600() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let meta = fs::metadata(&sb.socket).expect("lstat bound socket");
    let mode = meta.permissions().mode() & 0o777;
    assert_eq!(
        mode, 0o600,
        "socket must be owner-only (0o600); got {mode:o}"
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- unlink guard clears on graceful shutdown -----------------

/// After a clean SIGTERM shutdown, the socket file must be removed
/// from disk — the unlink guard's disarm + Drop combination owns
/// this responsibility. Operators starting a fresh daemon shortly
/// after a clean shutdown should not see the old path persist.
#[test]
fn unlink_guard_clears_on_graceful_shutdown() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");

    assert!(
        !sb.socket.exists(),
        "socket file persisted after graceful shutdown",
    );
}

// ---------- malformed / unknown request ------------------------------

/// A non-JSON line returns `ResponsePayload::Err { code: "malformed", .. }`
/// without dropping the connection — the loop is ready for another
/// request line.
#[test]
fn malformed_request_returns_err() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let line = one_shot(&sb.socket, "not json").expect("send malformed");
    let resp: ResponseSnap = serde_json::from_str(line.trim_end()).expect("parse response");
    match resp {
        ResponseSnap::Err { code, error } => {
            assert_eq!(code, "malformed", "code matches ERR_MALFORMED");
            assert!(!error.is_empty(), "Err carries a non-empty error message");
        }
        other => panic!("expected Err, got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

/// An unknown `op` value is a serde parse failure (the daemon's
/// `WireRequest` deny-list catches typos) and surfaces as the same
/// `malformed` error — operators get a structural signal at the
/// boundary, not silent acceptance.
#[test]
fn unknown_op_returns_err() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let line = one_shot(&sb.socket, r#"{"op":"frobnicate"}"#).expect("send unknown op");
    let resp: ResponseSnap = serde_json::from_str(line.trim_end()).expect("parse response");
    match resp {
        ResponseSnap::Err { code, error } => {
            assert_eq!(code, "malformed", "unknown op surfaces as malformed");
            assert!(!error.is_empty(), "Err carries a non-empty error message");
        }
        other => panic!("expected Err, got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- reload via IPC -------------------------------------------

/// `{"op":"reload"}` routes through the IPC drain to `handle_reload`
/// and bumps `reload_count`. A follow-up `status` call carries the
/// updated counter and `last_reload_via = "ipc"`.
#[test]
fn reload_via_ipc_increments_counters() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    // First snapshot: reload_count == 0.
    let first = one_shot(&sb.socket, r#"{"op":"status"}"#).expect("first status");
    let first_snap: ResponseSnap = serde_json::from_str(first.trim_end()).expect("parse");
    let initial_reloads = match first_snap {
        ResponseSnap::Status(s) => s.reload_count,
        other => panic!("expected Status, got {other:?}"),
    };
    assert_eq!(initial_reloads, 0);

    // Trigger a reload via IPC.
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload");
    let reply_snap: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse reload");
    assert!(matches!(reply_snap, ResponseSnap::Ok), "got {reply_snap:?}");

    // Wait for the reload to be observed (the engine thread processes
    // the IPC drain on its tick, and `record_reload` bumps
    // `reload_count` inside `handle_reload`). One short retry loop
    // covers the tick latency; the polled status carries the full
    // post-reload attribution.
    let final_status = poll_status_until(
        &sb.socket,
        |s| s.reload_count > initial_reloads,
        Duration::from_secs(5),
    )
    .unwrap_or_else(|| panic!("reload_count never advanced beyond {initial_reloads}"));
    let lr = final_status
        .last_reload
        .as_ref()
        .expect("successful reload stamps `last_reload`");
    assert_eq!(
        lr.last_reload_via,
        ReloadTriggerSnap::Ipc,
        "IPC reload attributes the trigger to `ipc`",
    );
    assert!(
        !lr.last_reload_at.is_empty(),
        "successful reload stamps a non-empty RFC 3339 wall-clock token",
    );

    // Confirm the log carries the reload-pipeline line — the
    // `handle_reload` ran end-to-end. The sandbox's config is empty
    // (no `[[watch]]` blocks), so the diff is empty and the log line
    // is "config reload: no watch changes" rather than
    // "config reload applied". Both branches converge on the same
    // `record_reload` bump, so the assertion is for the
    // empty-diff side.
    let log_contents = wait_for_log(
        &sb.log,
        |s| s.contains("config reload: no watch changes"),
        Duration::from_secs(2),
    );
    assert!(
        log_contents.is_some(),
        "expected 'config reload: no watch changes' in the log after IPC reload",
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- list round-trip ------------------------------------------

/// End-to-end `list`: spawn the daemon against a config carrying
/// one enabled watch, one disabled watch, and assert the response
/// surfaces both rows alphabetically with the right `disabled` tag.
/// Pins the entire list path: `channels` → `ipc::hub` →
/// `drain_ipc` → `project::list` → reply channel → wire.
#[test]
fn list_round_trip_alphabetic_with_disabled_rows() {
    let sb = Sandbox::new();
    // Anchor at the sandbox dir so the engine resolves it
    // synchronously — eliminates the descent's probe-arming variance
    // when the operator stats the projection.
    let cfg = format!(
        r#"
[[watch]]
name      = "zebra"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]

[[watch]]
name      = "alpha_off"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
enabled   = false
"#,
        anchor = sb.dir.display(),
    );
    fs::write(&sb.cfg, &cfg).expect("write config");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let line = one_shot(&sb.socket, r#"{"op":"list"}"#).expect("list request");
    let resp: ResponseSnap = serde_json::from_str(line.trim_end()).expect("parse response");
    match resp {
        ResponseSnap::List(list) => {
            assert_eq!(list.rows.len(), 2, "two rows declared");
            // BTreeMap-backed projection ⇒ alphabetic order:
            // `alpha_off` (Toml-disabled) before `zebra`
            // (engine-attached).
            assert_eq!(list.rows[0].name, "alpha_off");
            assert_eq!(list.rows[0].disabled, Some(DisabledSourceSnap::Toml));
            assert_eq!(list.rows[1].name, "zebra");
            assert!(
                list.rows[1].disabled.is_none(),
                "attached row carries no disabled tag",
            );
        }
        other => panic!("expected List, got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- show round-trip (Active) ---------------------------------

/// End-to-end `show <name>` for an attached Sub. Returns
/// `Show { status: "active", name: ... }`.
#[test]
fn show_round_trip_active_path() {
    let sb = Sandbox::new();
    let cfg = format!(
        r#"
[[watch]]
name      = "watched"
path      = "{anchor}"
actions   = [{{ exec = ["true"] }}]
"#,
        anchor = sb.dir.display(),
    );
    fs::write(&sb.cfg, &cfg).expect("write config");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let line = one_shot(&sb.socket, r#"{"op":"show","name":"watched"}"#).expect("show request");
    let resp: ResponseSnap = serde_json::from_str(line.trim_end()).expect("parse response");
    match resp {
        ResponseSnap::Show(ShowResponseSnap::Active { name }) => {
            assert_eq!(name, "watched");
        }
        other => panic!("expected Show::Active, got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- show client exits 1 on Unknown ---------------------------

/// `specter show <unknown_name>` exits `1` so operator shell scripts
/// can chain `specter show foo && do-thing`. Spawn the daemon, then
/// spawn the client binary as a subprocess so we can observe its
/// exit code through `output()`. The daemon needs no special config
/// for this — the `Unknown` arm depends only on the in-memory state
/// rejecting the typo.
#[test]
fn show_round_trip_unknown_exit_code() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );

    let bin = env!("CARGO_BIN_EXE_specter");
    let client = Command::new(bin)
        .args(["show", "ghost", "--socket"])
        .arg(&sb.socket)
        .args(["-o", "human"])
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .output()
        .expect("spawn show client");

    assert_eq!(
        client.status.code(),
        Some(1),
        "Unknown must exit 1; got status {:?}",
        client.status,
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- Tail / wait integration helpers --------------------------
//
// `tail` and `wait` are subscribe-arm verbs — the client subscribes,
// the daemon streams events line-by-line over the connection's
// lifetime. The tests below need three streaming-specific helpers
// beyond the read-verb surface:
//
// 1. `watched_anchor` — a subdirectory of the sandbox the daemon
//    does NOT itself write to. The sandbox root carries the daemon's
//    own log + socket, so watching it directly would surface log
//    writes as fs events; isolating the watch anchor underneath
//    keeps the burst trigger crisp.
// 2. `spawn_client_stream` — spawns one client subprocess (`tail` /
//    `wait`) with its stdout piped through a `mpsc::Receiver<String>`
//    populated by a background reader thread. Lets the main test
//    thread `recv_timeout` for "the next streamed line" without
//    polling.
// 3. Reload-driven watch removal — used as one trigger for
//    `SubDetached` in wait-detach tests; an IPC `disable` is the
//    other (`I-disable-emits-detach` covers it).

/// Make a fresh subdirectory under the sandbox dir that the daemon
/// does not itself write to — the right anchor for a watch that
/// must observe operator-driven touches without noise from the
/// daemon's own logging or socket bookkeeping.
fn watched_anchor(sb: &Sandbox) -> PathBuf {
    let p = sb.dir.join("watched");
    fs::create_dir_all(&p).expect("mkdir watched anchor");
    p
}

/// Construct a TOML config string with one `[[watch]]` whose anchor
/// is `anchor` and whose settle is `settle_ms`. Single-source so the
/// per-test TOML strings stay consistent.
fn one_watch_config(name: &str, anchor: &Path, settle_ms: u32) -> String {
    format!(
        r#"
[[watch]]
name      = "{name}"
path      = "{anchor}"
settle    = "{settle_ms}ms"
actions   = [{{ exec = ["true"] }}]
"#,
        anchor = anchor.display(),
    )
}

/// Spawn a `specter <verb> [args...]` client with stdout/stderr
/// piped. Returns the [`Child`] plus an [`mpsc::Receiver`] that
/// yields each line of stdout (one [`String`] per line, with
/// trailing LF stripped) as it arrives.
///
/// A background reader thread pumps the child's stdout into the
/// channel; the test's main thread reads via `recv_timeout`. The
/// thread exits cleanly on child stdout EOF, which happens on the
/// child's clean exit or when the test calls [`Child::kill`].
fn spawn_client_stream(
    sb: &Sandbox,
    verb: &str,
    extra: &[&str],
) -> (Child, mpsc::Receiver<String>) {
    let bin = env!("CARGO_BIN_EXE_specter");
    let mut child = Command::new(bin)
        .arg(verb)
        .args(["--socket"])
        .arg(&sb.socket)
        .args(extra)
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()
        .unwrap_or_else(|e| panic!("spawn specter {verb}: {e}"));

    let stdout = child.stdout.take().expect("piped stdout");
    let (tx, rx) = mpsc::channel::<String>();
    thread::Builder::new()
        .name(format!("specter-{verb}-stdout"))
        .spawn(move || {
            let reader = BufReader::new(stdout);
            for line in reader.lines() {
                let Ok(text) = line else { return };
                if tx.send(text).is_err() {
                    return;
                }
            }
        })
        .expect("spawn stdout-reader thread");

    (child, rx)
}

/// Wait up to `deadline` for `child` to exit. Returns the
/// [`std::process::ExitStatus`] on clean exit, or kills the child
/// and returns `Err(io::ErrorKind::TimedOut)` on overrun.
///
/// Distinct from [`await_exit`] (which targets the daemon under
/// [`SHUTDOWN_DEADLINE`]) — client-side verbs have their own
/// per-test deadlines (small for `wait --timeout`, large for the
/// happy-path tests).
fn await_client_exit(
    child: &mut Child,
    deadline: Duration,
) -> io::Result<std::process::ExitStatus> {
    let stop = Instant::now() + deadline;
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
        format!("client did not exit within {deadline:?}"),
    ))
}

/// Touch (create + close) a fresh file under the watch anchor to
/// drive a burst. The path is unique per call so consecutive
/// touches inside the same test do not coalesce into one filesystem
/// event by accidental path reuse.
fn touch_unique(anchor: &Path, label: &str) {
    let p = anchor.join(format!("{label}-{}", std::process::id()));
    fs::write(&p, b"x").unwrap_or_else(|e| panic!("touch {}: {e}", p.display()));
}

/// Wait until `rx` yields a line matching `pred`, or `deadline` is
/// reached. Returns the matching line on success, `None` on
/// timeout.
fn wait_for_line<F: Fn(&str) -> bool>(
    rx: &mpsc::Receiver<String>,
    pred: F,
    deadline: Duration,
) -> Option<String> {
    let stop = Instant::now() + deadline;
    loop {
        let now = Instant::now();
        if now >= stop {
            return None;
        }
        match rx.recv_timeout(stop - now) {
            Ok(line) => {
                if pred(&line) {
                    return Some(line);
                }
            }
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => {
                return None;
            }
        }
    }
}

/// Collect up to `n` consecutive lines from `rx` in arrival order,
/// or fewer if `deadline` elapses first. Distinct from
/// [`wait_for_line`], which filters by predicate and returns the
/// first match — this preserves arrival order, the contract
/// ordering-dependent tests pin.
fn collect_lines(rx: &mpsc::Receiver<String>, n: usize, deadline: Duration) -> Vec<String> {
    let stop = Instant::now() + deadline;
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let now = Instant::now();
        if now >= stop {
            break;
        }
        match rx.recv_timeout(stop - now) {
            Ok(line) => out.push(line),
            Err(mpsc::RecvTimeoutError::Timeout | mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }
    out
}

// ---------- tail --filter unknown exits 2 (no daemon needed) --------

/// `tail --filter <unknown>` exits `2` before any connection
/// attempt. The handler validates the filter vocabulary against the
/// wire-side `KNOWN_WIRE_VARIANTS` list; an unknown tag fails fast
/// with the operator-visible suggestion list.
#[test]
fn tail_unknown_filter_exits_two() {
    // No daemon needed — pure client-side validation.
    let bin = env!("CARGO_BIN_EXE_specter");
    let out = Command::new(bin)
        .args(["tail", "--filter", "NotARealVariant"])
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn tail client");
    assert_eq!(
        out.status.code(),
        Some(2),
        "unknown --filter must exit 2; got {:?}",
        out.status,
    );
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        stderr.contains("unknown filter"),
        "stderr must explain the rejection: {stderr}",
    );
    assert!(
        stderr.contains("Known filters:"),
        "stderr must list the wire vocabulary: {stderr}",
    );
}

// ---------- tail sees a SubFired arrive end-to-end ------------------

/// `tail --filter sub_fired` over an attached watch + an operator
/// file touch streams one matching line to stdout. Proves the
/// streaming surface end-to-end: engine emits Diagnostic ⇒ broker
/// fans out ⇒ per-conn thread writes wire JSON ⇒ tail client reads
/// + renders.
#[test]
fn tail_streams_sub_fired_after_touch() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("watcher", &anchor, 50)).expect("write config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );
    // Engine attach pass ran ⇒ Sub is in the registry.
    assert!(
        wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some(),
        "daemon never logged the initial attach",
    );

    // tail must subscribe BEFORE the touch lands or the broker
    // dispatches into thin air. A short sleep covers the
    // spawn → write Subscribe → ack round trip; the wait_for_line
    // deadline is the structural backstop if subscribe slips.
    let (mut tail, rx) = spawn_client_stream(&sb, "tail", &["--filter", "sub_fired", "-o", "json"]);
    thread::sleep(Duration::from_millis(400));

    touch_unique(&anchor, "tail2");

    let line = wait_for_line(
        &rx,
        |l| l.contains(r#""diag":"sub_fired""#),
        Duration::from_secs(8),
    );
    assert!(
        line.is_some(),
        "tail never observed a SubFired line within the deadline",
    );

    // Kill the indefinite stream cleanly.
    let _ = tail.kill();
    let _ = tail.wait();

    let exit = terminate(daemon);
    assert!(exit.success(), "clean daemon exit; got {exit:?}");
}

// ---------- tail -o json line round-trips via serde -----------------

/// `tail -o json` emits the wire shape losslessly. The streamed
/// line parses as a JSON object carrying the `diag` tag and the
/// expected per-Sub fields (the wire's `From<(&Diagnostic, SystemTime)>`
/// projection survives the broker → wire → stdout round-trip).
#[test]
fn tail_json_output_round_trips_via_serde() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("jsonwatch", &anchor, 50)).expect("write config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let (mut tail, rx) = spawn_client_stream(&sb, "tail", &["--filter", "sub_fired", "-o", "json"]);
    thread::sleep(Duration::from_millis(400));
    touch_unique(&anchor, "tail3");

    let line = wait_for_line(
        &rx,
        |l| l.contains(r#""diag":"sub_fired""#),
        Duration::from_secs(8),
    )
    .expect("tail observed a SubFired line");

    // Parse as a generic JSON value so the test does not duplicate
    // the bin's pub(crate) WireDiagnostic shape.
    let v: serde_json::Value =
        serde_json::from_str(line.trim_end()).expect("streamed line must be valid JSON");
    assert_eq!(v.get("diag").and_then(|x| x.as_str()), Some("sub_fired"));
    assert!(
        v.get("sub").and_then(serde_json::Value::as_u64).is_some(),
        "SubFired line carries a numeric sub id: {line}",
    );
    assert!(
        v.get("profile")
            .and_then(serde_json::Value::as_u64)
            .is_some(),
        "SubFired line carries a numeric profile id: {line}",
    );
    assert!(
        v.get("count").and_then(serde_json::Value::as_u64).is_some(),
        "SubFired line carries a numeric count: {line}",
    );

    let _ = tail.kill();
    let _ = tail.wait();
    let exit = terminate(daemon);
    assert!(exit.success(), "clean daemon exit; got {exit:?}");
}

// ---------- ack-ordering: subscribe_ack precedes every diagnostic ----

/// `subscribe_ack` MUST be the first JSON line on the wire after a
/// Subscribe verb, even when a fire-triggering touch races
/// concurrently with the subscribe write. Pins the ack-before-fanout
/// ordering in the driver's Subscribe arm: the ack bytes are pushed
/// into the conn's `write_queue` while the conn is still in
/// [`ConnRole::Reqs`], and only then does
/// [`ConnState::transition_to_sub`] flip the role to `Sub`. The
/// diagnostic fan-out (`DriverHub::dispatch_to_subscribers`) skips
/// `Reqs` conns, so no diag can interleave between the ack enqueue
/// and the role flip.
///
/// **Fence, not fuzzer.** The invariant holds structurally — the
/// fan-out's `Reqs`-skip gate is the proof — but this test bounds it
/// against any future refactor that reorders the ack push and the
/// role flip, or that lets the fan-out path observe a partially-
/// transitioned conn.
///
/// Per-Sub Subscribe (`name = "orderwatch"`) so the post-ack wire
/// carries only events naming that Sub: ambient Profile-keyed
/// diagnostics (e.g. teardown `ProfileReaped`) cannot pollute the
/// assertion.
///
/// **Settle sized for parallel-test load.** The settle window
/// (300ms) is comfortably larger than the worst-case subscribe
/// handshake under heavy `nextest` parallelism. A smaller window
/// (e.g. the 50ms used by the per-touch happy-path tests) lets the
/// touch's burst fire before the conn has flipped role under
/// contention — the diagnostic is then dispatched to no one and the
/// post-ack assertion below would fail, masking the ack-ordering
/// fence the test is meant to express.
#[test]
fn subscribe_ack_precedes_first_diagnostic_on_wire() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("orderwatch", &anchor, 300)).expect("write config");
    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(
        wait_for_socket(&sb.socket, STARTUP_DEADLINE),
        "daemon never bound its socket",
    );
    assert!(
        wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some(),
        "daemon never logged the initial attach",
    );

    let stream = UnixStream::connect(&sb.socket).expect("connect");
    stream
        .set_read_timeout(Some(Duration::from_secs(8)))
        .expect("set_read_timeout");
    let mut writer = stream.try_clone().expect("try_clone for writer");
    let mut reader = BufReader::new(stream);

    // Spawn the race-window toucher BEFORE the subscribe write. The
    // thread is scheduled concurrently; the OS may run it before
    // the subscribe lands on the daemon, during its processing, or
    // after the ack returns. B3 says the ack must reach the wire
    // first in every case.
    let toucher = thread::spawn(move || touch_unique(&anchor, "race"));

    writer
        .write_all(br#"{"op":"subscribe","name":"orderwatch"}"#)
        .expect("write subscribe");
    writer.write_all(b"\n").expect("write newline");

    let mut first = String::new();
    reader.read_line(&mut first).expect("read first line");
    let first_v: serde_json::Value =
        serde_json::from_str(first.trim_end()).expect("first line is valid JSON");
    assert_eq!(
        first_v.get("kind").and_then(serde_json::Value::as_str),
        Some("subscribe_ack"),
        "B3 ack-ordering: first line on the wire MUST be subscribe_ack; got {first}",
    );

    // Next line MUST be SubFired. The per-Sub name filter on the
    // broker drops every other variant (Profile-keyed diagnostics
    // never match a name filter), and our setup emits no other
    // per-Sub event in the post-ack window (SubAttached fired
    // pre-connect; no detach/rebind/effect-complete races are in
    // play). A back-pressure `_missed` cannot appear either: the
    // channel is drained synchronously line-by-line.
    let mut second = String::new();
    reader
        .read_line(&mut second)
        .expect("read second line (SubFired from racing touch)");
    let second_v: serde_json::Value =
        serde_json::from_str(second.trim_end()).expect("second line is valid JSON");
    assert_eq!(
        second_v.get("diag").and_then(serde_json::Value::as_str),
        Some("sub_fired"),
        "second wire line must be SubFired (proves the subscriber \
         was registered ahead of the burst's emission — the \
         observable shadow of the B3 fence); got {second}",
    );

    let _ = toucher.join();
    drop(reader);
    drop(writer);

    let exit = terminate(daemon);
    assert!(exit.success(), "clean daemon exit; got {exit:?}");
}

// ---------- wait --kind fire happy path ------------------------------

/// `wait <name> --kind fire` against an attached watch + an
/// operator file touch exits `0` once the burst settles and
/// `SubFired` arrives.
#[test]
fn wait_kind_fire_happy_path_exits_zero() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("firetarget", &anchor, 50)).expect("write config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let (mut wait_child, _rx) = spawn_client_stream(&sb, "wait", &["firetarget", "--kind", "fire"]);
    thread::sleep(Duration::from_millis(400));
    touch_unique(&anchor, "wait1");

    let exit = await_client_exit(&mut wait_child, Duration::from_secs(8))
        .expect("wait client must exit within deadline");
    assert_eq!(
        exit.code(),
        Some(0),
        "wait --kind fire must exit 0 on a fire; got {exit:?}",
    );

    let daemon_exit = terminate(daemon);
    assert!(
        daemon_exit.success(),
        "clean daemon exit; got {daemon_exit:?}"
    );
}

// ---------- wait --timeout exits 124 ---------------------------------

/// `wait <name> --kind fire --timeout 300ms` against an attached
/// watch with no touch exits `124` (POSIX `timeout(1)` convention)
/// after the deadline elapses.
#[test]
fn wait_kind_fire_timeout_exits_124() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("idletarget", &anchor, 50)).expect("write config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let (mut wait_child, _rx) = spawn_client_stream(
        &sb,
        "wait",
        &["idletarget", "--kind", "fire", "--timeout", "300ms"],
    );

    let exit = await_client_exit(&mut wait_child, Duration::from_secs(5))
        .expect("wait --timeout must exit within the test deadline");
    assert_eq!(
        exit.code(),
        Some(124),
        "wait --timeout must exit 124 on deadline; got {exit:?}",
    );

    let daemon_exit = terminate(daemon);
    assert!(
        daemon_exit.success(),
        "clean daemon exit; got {daemon_exit:?}"
    );
}

// ---------- wait <unknown> exits 1 -----------------------------------

/// `wait <ghost>` against a daemon that has no such Sub exits `1`
/// (subscribe-side `ERR_UNKNOWN_SUB`). Server-side name resolution
/// is atomic with `add_subscriber` so the wait client cannot
/// silently wait forever on a typo.
#[test]
fn wait_unknown_sub_exits_one() {
    let sb = Sandbox::new();
    // Empty config — no Subs in the registry.
    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));

    let (mut wait_child, _rx) = spawn_client_stream(&sb, "wait", &["ghost"]);
    let exit = await_client_exit(&mut wait_child, Duration::from_secs(5))
        .expect("wait <ghost> must exit immediately");
    assert_eq!(
        exit.code(),
        Some(1),
        "wait <unknown> must exit 1; got {exit:?}",
    );

    let daemon_exit = terminate(daemon);
    assert!(
        daemon_exit.success(),
        "clean daemon exit; got {daemon_exit:?}"
    );
}

// ---------- wait race-window closed ----------------------------------

/// `specter wait <name>` invoked AFTER an IPC reload that drops the
/// watch from the TOML exits `1` immediately. Server-side name
/// resolution sees the empty registry and returns `ERR_UNKNOWN_SUB`;
/// no event window can keep the client waiting. The same structural
/// guarantee covers the disable race-window — `name → SubId` resolves
/// on the driver thread, atomic with `add_subscriber`.
#[test]
fn wait_race_window_closed_via_reload() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("gone", &anchor, 50)).expect("write initial config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    // Reload: empty config, no watches.
    fs::write(&sb.cfg, "").expect("rewrite to empty config");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload request");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse reload");
    assert!(matches!(resp, ResponseSnap::Ok), "got {resp:?}");
    // Wait for the daemon to surface the detach in the log so we
    // know the reload's prune pass ran before we subscribe.
    assert!(
        wait_for_log(
            &sb.log,
            |s| s.contains("sub detached"),
            Duration::from_secs(5)
        )
        .is_some(),
        "daemon never logged the detach following reload",
    );

    let (mut wait_child, _rx) = spawn_client_stream(&sb, "wait", &["gone"]);
    let exit = await_client_exit(&mut wait_child, Duration::from_secs(5))
        .expect("wait must exit immediately on unknown_sub");
    assert_eq!(
        exit.code(),
        Some(1),
        "wait on a removed name must exit 1; got {exit:?}",
    );

    let daemon_exit = terminate(daemon);
    assert!(
        daemon_exit.success(),
        "clean daemon exit; got {daemon_exit:?}"
    );
}

// ---------- wait --kind fire observes SubDetached exits 2 -----------

/// `wait <name> --kind fire` that observes a `SubDetached` for the
/// target before any fire exits `2` ("target detached before
/// fire"). Distinct from `124` (timeout) and `1` (subscribe error).
///
/// The detach is reached here via an IPC reload that drops the
/// watch from the TOML — the classify arm doesn't care WHICH reason
/// the engine attached to the detach; an IPC `disable` would land
/// the same outcome.
#[test]
fn wait_fire_observing_detach_exits_two() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("victim", &anchor, 50)).expect("write initial config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    // Subscribe BEFORE the reload so the SubDetached lands on a
    // live stream.
    let (mut wait_child, _rx) = spawn_client_stream(&sb, "wait", &["victim", "--kind", "fire"]);
    thread::sleep(Duration::from_millis(400));

    // Reload removes the watch ⇒ engine emits SubDetached.
    fs::write(&sb.cfg, "").expect("rewrite to empty config");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload request");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse reload");
    assert!(matches!(resp, ResponseSnap::Ok), "got {resp:?}");

    let exit = await_client_exit(&mut wait_child, Duration::from_secs(5))
        .expect("wait must exit on detach");
    assert_eq!(
        exit.code(),
        Some(2),
        "wait --kind fire on SubDetached must exit 2; got {exit:?}",
    );

    let daemon_exit = terminate(daemon);
    assert!(
        daemon_exit.success(),
        "clean daemon exit; got {daemon_exit:?}"
    );
}

// ---------- Disable / enable / reload integration ------------------
//
// Each test spawns the daemon against a config carrying one or two
// `[[watch]]` blocks, drives the operator-mutation verbs via raw
// IPC (`one_shot`) or as a subprocess client, and asserts the
// observable state changes through subsequent `status` snapshots.
//
// `one_shot` returns synchronously once the daemon acks: the
// daemon's IPC drain processes a request to completion before
// writing the reply line, so consecutive `one_shot` calls compose
// without inter-operation polling.

/// Helper: poll status until the response matches a predicate, or
/// timeout. Useful when an async operation (initial attach, reload
/// settle) needs to propagate through the daemon's tick before the
/// projection reflects it.
fn poll_status_until<F: Fn(&StatusResponseSnap) -> bool>(
    socket: &Path,
    pred: F,
    deadline: Duration,
) -> Option<StatusResponseSnap> {
    let stop = Instant::now() + deadline;
    while Instant::now() < stop {
        if let Ok(line) = one_shot(socket, r#"{"op":"status"}"#)
            && let Ok(ResponseSnap::Status(s)) =
                serde_json::from_str::<ResponseSnap>(line.trim_end())
            && pred(&s)
        {
            return Some(s);
        }
        thread::sleep(POLL_INTERVAL);
    }
    None
}

// ---------- disable + enable cycle -----------------------------------

/// `disable foo` flips `status.sub_disabled_runtime` 0→1; a follow-
/// up `enable foo` flips it back to 0. The engine attaches foo
/// initially (status.sub_total=1), the disable detaches it
/// (sub_total=0), the enable re-attaches it (sub_total=1).
#[test]
fn disable_then_enable_cycle_round_trips_through_status() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("foo", &anchor, 50)).expect("write config");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    // Initial state: foo attached, no runtime overrides.
    let initial = poll_status_until(&sb.socket, |s| s.sub_total == 1, Duration::from_secs(5))
        .expect("initial attach reflected in status");
    assert_eq!(initial.sub_disabled_runtime, 0);

    // Disable.
    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"foo"}"#).expect("disable");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse disable");
    assert!(matches!(resp, ResponseSnap::Ok), "got {resp:?}");

    let post_disable = poll_status_until(
        &sb.socket,
        |s| s.sub_disabled_runtime == 1,
        Duration::from_secs(5),
    )
    .expect("status reflects disable");
    assert_eq!(post_disable.sub_total, 0, "engine detached foo");

    // Enable.
    let reply = one_shot(&sb.socket, r#"{"op":"enable","name":"foo"}"#).expect("enable");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse enable");
    assert!(matches!(resp, ResponseSnap::Ok), "got {resp:?}");

    let post_enable = poll_status_until(
        &sb.socket,
        |s| s.sub_disabled_runtime == 0 && s.sub_total == 1,
        Duration::from_secs(5),
    )
    .expect("status reflects enable");
    assert_eq!(post_enable.sub_disabled_runtime, 0);
    assert_eq!(post_enable.sub_total, 1, "engine re-attached foo");

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- disable suppresses fires; enable restores them ----------

/// The `disable_runtime ↔ engine` invariant end-to-end: a watched
/// anchor touched while the Sub is in `disabled_runtime` produces no
/// `SubFired`; the same anchor touched after `enable` re-attaches the
/// Sub with a fresh baseline that fires on the next operator-driven
/// change.
///
/// Sibling to [`disable_then_enable_cycle_round_trips_through_status`]:
/// that test pins the counter shape (`sub_total`,
/// `sub_disabled_runtime`); this one pins the behavioural contract
/// the counters stand for.
#[test]
fn disable_suppresses_fires_enable_restores_them() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("cycler", &anchor, 50)).expect("write config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    // Subscribe once for the full cycle. A filter scoped to SubFired
    // means the test only observes the behavioural witness — Detach /
    // Reap / Rebound noise stays off the stream.
    let (mut tail, rx) = spawn_client_stream(&sb, "tail", &["--filter", "sub_fired", "-o", "json"]);
    thread::sleep(Duration::from_millis(400));

    // Disable. `one_shot` returns once the daemon's IPC drain has
    // detached the Sub (the reply rides on the same step the detach
    // ran on); the subsequent status-poll proves the engine-side
    // detach landed before the test proceeds.
    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"cycler"}"#).expect("disable");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));
    poll_status_until(
        &sb.socket,
        |s| s.sub_disabled_runtime == 1 && s.sub_total == 0,
        Duration::from_secs(5),
    )
    .expect("disable propagated through engine");

    // Touch while disabled. The Sub is gone from the engine; the
    // Profile is reaped; the kqueue / inotify watch FD is closed by
    // the dispatched `Unwatch` op. No fire path can reach the broker.
    touch_unique(&anchor, "while-disabled");
    // Settle window is 50ms; a fire — were one to occur — would
    // surface well inside one second. The deadline is the upper
    // bound on "we waited long enough for a real fire to land", not
    // a tight latency claim.
    assert!(
        wait_for_line(
            &rx,
            |l| l.contains(r#""diag":"sub_fired""#),
            Duration::from_secs(1),
        )
        .is_none(),
        "disabled Sub must not emit SubFired on a touch",
    );

    // Enable. The re-attach drives a fresh `Input::AttachSub` through
    // the engine; the new Sub starts with `has_fired = false` and a
    // fresh seed baseline.
    let reply = one_shot(&sb.socket, r#"{"op":"enable","name":"cycler"}"#).expect("enable");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));
    poll_status_until(
        &sb.socket,
        |s| s.sub_disabled_runtime == 0 && s.sub_total == 1,
        Duration::from_secs(5),
    )
    .expect("enable propagated through engine");
    // Let the seed pass complete before the touch — without this,
    // the operator-driven event could fold into the Seed burst's
    // probe response rather than driving a fresh Standard burst.
    thread::sleep(Duration::from_millis(400));

    touch_unique(&anchor, "after-enable");
    assert!(
        wait_for_line(
            &rx,
            |l| l.contains(r#""diag":"sub_fired""#),
            Duration::from_secs(8),
        )
        .is_some(),
        "re-enabled Sub must fire on the next operator-driven change",
    );

    let _ = tail.kill();
    let _ = tail.wait();
    let exit = terminate(daemon);
    assert!(exit.success(), "clean daemon exit; got {exit:?}");
}

// ---------- IPC disable emission order: SubDetached then ProfileReaped ----------

/// IPC `disable` of a single-Sub Profile drives two diagnostics in
/// causal order: `SubDetached(IpcDisabled)` first, then
/// `ProfileReaped`. The engine emits them at adjacent sites
/// (`detach_sub_inner` → `reap_profile`) and the broker's dispatch
/// loop preserves insertion order; the contract holds end-to-end at
/// the streamed subscriber.
///
/// Distinct from the unit-level emission-order assertions in
/// `engine`: this pin closes the wire boundary, so a future refactor
/// that reordered the engine emissions, or one that re-sorted
/// diagnostics in the broker fan-out, would surface here.
#[test]
fn disable_streams_sub_detached_before_profile_reaped() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("reapme", &anchor, 50)).expect("write config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    // Filter is exhaustive over the expected emissions; any other
    // variant arriving on this stream would be a test-environment
    // artefact (the seed pass does not fire — a fresh Profile has no
    // prior `DedupKey::Subtree` hash to drift against).
    let (mut tail, rx) = spawn_client_stream(
        &sb,
        "tail",
        &[
            "--filter",
            "sub_detached",
            "--filter",
            "profile_reaped",
            "-o",
            "json",
        ],
    );
    thread::sleep(Duration::from_millis(400));

    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"reapme"}"#).expect("disable");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    let lines = collect_lines(&rx, 2, Duration::from_secs(5));
    assert_eq!(
        lines.len(),
        2,
        "expected SubDetached + ProfileReaped on the stream; got {lines:?}",
    );
    let first: serde_json::Value =
        serde_json::from_str(lines[0].trim_end()).expect("first line is valid JSON");
    let second: serde_json::Value =
        serde_json::from_str(lines[1].trim_end()).expect("second line is valid JSON");
    assert_eq!(
        first.get("diag").and_then(serde_json::Value::as_str),
        Some("sub_detached"),
        "first emission is SubDetached: {lines:?}",
    );
    assert_eq!(
        first.get("reason").and_then(serde_json::Value::as_str),
        Some("ipc_disabled"),
        "SubDetached carries the IpcDisabled reason: {lines:?}",
    );
    assert_eq!(
        second.get("diag").and_then(serde_json::Value::as_str),
        Some("profile_reaped"),
        "second emission is ProfileReaped: {lines:?}",
    );

    let _ = tail.kill();
    let _ = tail.wait();
    let exit = terminate(daemon);
    assert!(exit.success(), "clean daemon exit; got {exit:?}");
}

// ---------- disable client exits 1 on unknown name -----------------

/// `specter disable <ghost>` invoked as a subprocess exits `1` with
/// stderr carrying the structured `unknown_sub:` prefix. Pins the
/// client wiring (`lib.rs` dispatch → `ipc::client::disable::run`
/// → `connect::one_shot_unit`) end-to-end.
#[test]
fn disable_client_unknown_name_exits_one() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));

    let bin = env!("CARGO_BIN_EXE_specter");
    let output = Command::new(bin)
        .args(["disable", "ghost", "--socket"])
        .arg(&sb.socket)
        .stdout(Stdio::null())
        .stderr(Stdio::piped())
        .output()
        .expect("spawn disable client");
    assert_eq!(
        output.status.code(),
        Some(1),
        "unknown name must exit 1; got {:?}",
        output.status,
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("unknown_sub"),
        "stderr must carry the structured code: {stderr}",
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- disable typo with @-bearing name returns unknown_sub ----

/// `disable foo@/some/path` against a daemon with no such Sub
/// returns `Err { code: "unknown_sub" }` — a typo (an `@`-bearing
/// name the registry doesn't index) reports the structural truth
/// (the name does not resolve), not a misleading dynamic-sub
/// classification. The dynamic-vs-static discrimination is a
/// property of the resolved Sub; this case never reaches that gate
/// because the lookup is empty. The genuine dynamic-Sub gate is
/// exercised at the driver-handler layer.
#[test]
fn disable_unknown_dynamic_shape_name_returns_unknown_sub() {
    let sb = Sandbox::new();
    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));

    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"promoter@/tmp/x"}"#)
        .expect("disable request");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse response");
    match resp {
        ResponseSnap::Err { code, error } => {
            assert_eq!(code, "unknown_sub", "typo reports structural truth");
            assert!(!error.is_empty(), "Err carries a non-empty error message");
        }
        other => panic!("expected Err, got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- enable TOML-disabled clears override + errs --------------

/// Sequence:
/// 1. `disable foo` records a runtime override + detaches.
/// 2. Edit TOML to set `enabled = false` for foo, then reload.
///    The reload-pipeline prune retains the override (TOML still
///    carries foo, just disabled).
/// 3. `enable foo` clears the override but returns
///    `Err { code: "toml_disabled" }` — the runtime override is
///    gone, but the TOML keeps foo inactive.
/// 4. A second `enable foo` returns `Err { code: "not_disabled" }`
///    — proves step 3 cleared the override.
#[test]
fn enable_toml_disabled_clears_override_then_errs() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("foo", &anchor, 50)).expect("write config v1");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    // Step 1: disable.
    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"foo"}"#).expect("disable");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse disable");
    assert!(matches!(resp, ResponseSnap::Ok));

    // Step 2: edit TOML to mark foo as enabled=false, then reload.
    let v2 = format!(
        r#"
[[watch]]
name      = "foo"
path      = "{}"
settle    = "50ms"
actions   = [{{ exec = ["true"] }}]
enabled   = false
"#,
        anchor.display(),
    );
    fs::write(&sb.cfg, &v2).expect("write config v2");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse reload");
    assert!(matches!(resp, ResponseSnap::Ok));

    // After reload, override still recorded (TOML-disabled retention).
    let post_reload =
        poll_status_until(&sb.socket, |s| s.reload_count >= 1, Duration::from_secs(5))
            .expect("reload propagated");
    assert_eq!(
        post_reload.sub_disabled_runtime, 1,
        "TOML-disabled retention kept the override",
    );

    // Step 3: enable returns toml_disabled.
    let reply = one_shot(&sb.socket, r#"{"op":"enable","name":"foo"}"#).expect("enable 1");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse enable 1");
    match resp {
        ResponseSnap::Err { code, error } => {
            assert_eq!(code, "toml_disabled");
            assert!(!error.is_empty(), "Err carries a non-empty error message");
        }
        other => panic!("expected Err(toml_disabled), got {other:?}"),
    }

    // Step 4: second enable returns not_disabled — override was
    // cleared on step 3's failure path.
    let reply = one_shot(&sb.socket, r#"{"op":"enable","name":"foo"}"#).expect("enable 2");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse enable 2");
    match resp {
        ResponseSnap::Err { code, error } => {
            assert_eq!(code, "not_disabled");
            assert!(!error.is_empty(), "Err carries a non-empty error message");
        }
        other => panic!("expected Err(not_disabled), got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- reload prune drops names absent from TOML ----------------

/// Operator runs `disable foo`, then edits the TOML to remove foo
/// entirely. After the next reload, the prune drops the override
/// (no TOML row to anchor it against). A subsequent `enable foo`
/// returns `not_disabled`.
#[test]
fn reload_prune_drops_disabled_runtime_when_name_leaves_toml() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("foo", &anchor, 50)).expect("write config v1");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"foo"}"#).expect("disable");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    // Reload against a TOML that doesn't carry foo at all.
    fs::write(&sb.cfg, "").expect("write empty config");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    let post = poll_status_until(&sb.socket, |s| s.reload_count >= 1, Duration::from_secs(5))
        .expect("reload propagated");
    assert_eq!(
        post.sub_disabled_runtime, 0,
        "prune dropped the override whose TOML row vanished",
    );

    // Sanity: subsequent enable returns not_disabled (proves the
    // override is gone, not merely hidden from the projection).
    let reply = one_shot(&sb.socket, r#"{"op":"enable","name":"foo"}"#).expect("enable");
    match serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse") {
        ResponseSnap::Err { code, error } => {
            assert_eq!(code, "not_disabled");
            assert!(!error.is_empty(), "Err carries a non-empty error message");
        }
        other => panic!("expected Err(not_disabled), got {other:?}"),
    }

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- reload retains override over TOML-disabled row ----------

/// Operator's "off twice" preference survives: when the TOML
/// carries the entry as `enabled = false` (still operator-declared,
/// just disabled), a runtime override stacked over it is preserved
/// across the reload's prune. Only a complete removal from the
/// TOML evaporates the override (covered by
/// `reload_prune_drops_disabled_runtime_when_name_leaves_toml`).
#[test]
fn reload_retains_disabled_runtime_when_toml_keeps_row_disabled() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("foo", &anchor, 50)).expect("write config v1");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"foo"}"#).expect("disable");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    let v2 = format!(
        r#"
[[watch]]
name      = "foo"
path      = "{}"
settle    = "50ms"
actions   = [{{ exec = ["true"] }}]
enabled   = false
"#,
        anchor.display(),
    );
    fs::write(&sb.cfg, &v2).expect("write config v2");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    let post = poll_status_until(&sb.socket, |s| s.reload_count >= 1, Duration::from_secs(5))
        .expect("reload propagated");
    assert_eq!(
        post.sub_disabled_runtime, 1,
        "TOML-disabled row anchors the override; prune retains it",
    );
    assert_eq!(
        post.sub_disabled_toml, 1,
        "TOML carries foo as enabled=false"
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- reload filter blocks re-attach of disabled sub ----------

/// A TOML edit that produces a `modified_params` diff entry for a
/// runtime-disabled Sub gets filtered out by `compute_watch_diff`:
/// the engine never sees the rebind, so the Sub stays detached.
/// Pins the `compute_watch_diff` filter at the integration level.
#[test]
fn reload_filter_blocks_reattach_for_runtime_disabled_sub() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("foo", &anchor, 50)).expect("write config v1");

    let child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let reply = one_shot(&sb.socket, r#"{"op":"disable","name":"foo"}"#).expect("disable");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    // Edit the settle window. `settle` is a per-Sub param, so the
    // unfiltered diff would surface `modified_params` for foo. The
    // filter must strip it so the engine never re-attaches.
    fs::write(&sb.cfg, one_watch_config("foo", &anchor, 200)).expect("write config v2");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload");
    assert!(matches!(
        serde_json::from_str::<ResponseSnap>(reply.trim_end()).expect("parse"),
        ResponseSnap::Ok,
    ));

    let post = poll_status_until(&sb.socket, |s| s.reload_count >= 1, Duration::from_secs(5))
        .expect("reload propagated");
    assert_eq!(post.sub_total, 0, "foo stays detached across the reload");
    assert_eq!(
        post.sub_disabled_runtime, 1,
        "override survives — TOML still carries foo as active",
    );

    let exit = terminate(child);
    assert!(exit.success(), "clean exit; got {exit:?}");
}

// ---------- wait --kind detach happy path ----------------------------

/// `wait <name> --kind detach` matches when the engine reaps the
/// Sub. Exit `0`, same matched-render semantics as `--kind fire`.
#[test]
fn wait_kind_detach_happy_path_exits_zero() {
    let sb = Sandbox::new();
    let anchor = watched_anchor(&sb);
    fs::write(&sb.cfg, one_watch_config("detachtarget", &anchor, 50))
        .expect("write initial config");

    let daemon = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));
    assert!(wait_for_log(&sb.log, |s| s.contains("sub attached"), STARTUP_DEADLINE).is_some());

    let (mut wait_child, _rx) =
        spawn_client_stream(&sb, "wait", &["detachtarget", "--kind", "detach"]);
    thread::sleep(Duration::from_millis(400));

    fs::write(&sb.cfg, "").expect("rewrite to empty config");
    let reply = one_shot(&sb.socket, r#"{"op":"reload"}"#).expect("reload request");
    let resp: ResponseSnap = serde_json::from_str(reply.trim_end()).expect("parse reload");
    assert!(matches!(resp, ResponseSnap::Ok), "got {resp:?}");

    let exit = await_client_exit(&mut wait_child, Duration::from_secs(5))
        .expect("wait must exit on detach");
    assert_eq!(
        exit.code(),
        Some(0),
        "wait --kind detach on SubDetached must exit 0; got {exit:?}",
    );

    let daemon_exit = terminate(daemon);
    assert!(
        daemon_exit.success(),
        "clean daemon exit; got {daemon_exit:?}"
    );
}

// ---------- bounded shutdown under wedged subscribers ----------------

/// SIGTERM-driven shutdown completes within a bounded window even
/// when every IPC connection slot is occupied by a wedged
/// subscriber — a client that finished its Subscribe handshake and
/// then stopped reading the socket. Pins two structural exits the
/// teardown path depends on:
///
/// - Per-conn worker threads are detached. The accept loop is the
///   only IPC thread the daemon joins; it polls `shutdown_flag`
///   between non-blocking accepts and exits within
///   `ACCEPT_IDLE_SLEEP` of the flag store.
/// - The broker drops with the engine driver. Every per-subscriber
///   `event_rx` disconnects in lockstep, releasing every blocked
///   worker `recv` cleanly without dependence on event traffic.
///
/// The chosen deadline (`MAX_IPC_CONNS × PER_CONN_WRITE_TIMEOUT +
/// 4s headroom`, mirroring the bin's `server.rs` constants) is the
/// conservative structural ceiling assuming a hypothetical
/// sequential per-worker write-block; the observed latency under
/// the broker-drop path is sub-second.
#[test]
fn shutdown_with_wedged_subscribers_is_bounded() {
    // Mirror of `ipc::server::MAX_IPC_CONNS`. The bin's constant is
    // `pub(crate)`; this test lives at the integration boundary and
    // pins the contract from the outside, so duplication is the
    // honest seam.
    const WEDGED_CLIENTS: usize = 8;
    // Mirror of `MAX_IPC_CONNS × PER_CONN_WRITE_TIMEOUT` plus 4s
    // headroom. Same reasoning as `WEDGED_CLIENTS`.
    const SHUTDOWN_BUDGET: Duration = Duration::from_secs(8 * 2 + 4);

    let sb = Sandbox::new();
    let mut child = spawn_specter(&sb, std::iter::empty::<&str>());
    assert!(wait_for_socket(&sb.socket, STARTUP_DEADLINE));

    // Saturate the connection cap with subscribers that finish their
    // handshake and then stop reading. The `Vec` keeps every stream
    // alive across the SIGTERM: dropping one would close the socket
    // and unblock the daemon's worker on EOF, defeating the
    // wedged-subscriber premise.
    let mut wedged: Vec<UnixStream> = Vec::with_capacity(WEDGED_CLIENTS);
    for i in 0..WEDGED_CLIENTS {
        let stream = UnixStream::connect(&sb.socket)
            .unwrap_or_else(|e| panic!("connect subscriber {i}: {e}"));
        stream
            .set_read_timeout(Some(Duration::from_secs(2)))
            .expect("set_read_timeout");
        let mut writer = stream.try_clone().expect("try_clone for writer");
        writer
            .write_all(b"{\"op\":\"subscribe\"}\n")
            .unwrap_or_else(|e| panic!("send subscribe {i}: {e}"));
        // Read exactly the SubscribeAck line on a cloned reader; the
        // original stream stays in `wedged` for the rest of the test
        // without ever consuming another byte off the socket.
        let mut reader = BufReader::new(stream.try_clone().expect("try_clone for reader"));
        let mut ack = String::new();
        reader
            .read_line(&mut ack)
            .unwrap_or_else(|e| panic!("read ack {i}: {e}"));
        let v: serde_json::Value = serde_json::from_str(ack.trim_end()).expect("ack is valid JSON");
        assert_eq!(
            v.get("kind").and_then(serde_json::Value::as_str),
            Some("subscribe_ack"),
            "subscriber {i} expected SubscribeAck; got {ack}",
        );
        wedged.push(stream);
    }

    let pid = Pid::from_raw(child.id().cast_signed());
    let start = Instant::now();
    kill(pid, Signal::SIGTERM).expect("SIGTERM");

    let exit = loop {
        match child.try_wait().expect("try_wait") {
            Some(s) => break s,
            None => {
                if start.elapsed() >= SHUTDOWN_BUDGET {
                    let _ = kill(pid, Signal::SIGKILL);
                    let _ = child.wait();
                    panic!(
                        "daemon did not exit within {SHUTDOWN_BUDGET:?} with \
                         {WEDGED_CLIENTS} wedged subscribers — bounded \
                         shutdown contract violated",
                    );
                }
                thread::sleep(POLL_INTERVAL);
            }
        }
    };
    assert!(exit.success(), "clean daemon exit; got {exit:?}");

    // Holding `wedged` until here guarantees the subscriber sockets
    // were live across the entire SIGTERM teardown; explicit drop
    // documents the lifetime contract.
    drop(wedged);
}
