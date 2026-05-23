//! Operator-IPC server thread — accept loop + per-connection handler.
//!
//! Owns the bound [`UnixListener`] for the daemon's lifetime; spawns
//! one short-lived worker thread per accepted connection. The
//! per-connection thread does JSON framing in both directions and
//! routes the parsed request through
//! [`crate::channels::IpcServerSide::ipc_request_tx`] to the driver,
//! awaiting the reply on a per-request `bounded(1)` channel.
//!
//! # Wake-vs-shutdown discipline
//!
//! The accept loop is non-blocking ([`UnixListener::set_nonblocking`]);
//! each iteration checks `shutdown_flag` before re-arming `accept()`.
//! The [`ACCEPT_IDLE_SLEEP`] sleep between empty accepts keeps the
//! thread's CPU cost negligible without adding select/poll machinery
//! for a single fd.
//!
//! # Connection cap + back-pressure
//!
//! [`MAX_IPC_CONNS`] bounds concurrent operator clients. Overshoot
//! gets an `ERR_BUSY` reply then EOF; the cap is sized for the
//! worst-case "operator dashboard + ad-hoc shell + on-call wait +
//! automation" fan-in without inviting accidental DoS. The
//! fetch-add-then-check-then-fetch-sub pattern is intentional: a
//! concurrent close-path decrement cannot race us under-cap.
//!
//! # Per-connection write timeout
//!
//! [`PER_CONN_WRITE_TIMEOUT`] bounds the worst-case shutdown latency
//! contributed by a wedged client (a peer that stopped reading
//! mid-stream). Without this, kernel-keepalive times in the minutes
//! would dominate teardown.
//!
//! # Verb dispatch
//!
//! Every parsed [`WireRequest`] falls into one of two shapes:
//!
//! - **Subscribe** — terminal for the connection. The per-conn thread
//!   builds a per-subscriber [`bounded(SUBSCRIBE_QUEUE)`](SUBSCRIBE_QUEUE)
//!   event channel, ships the [`crossbeam::channel::Sender`] half on
//!   the request channel for the broker to fan into, awaits a
//!   [`ResponsePayload::SubscribeAck`] on its reply channel, and then
//!   pumps every [`BrokerEvent`] through the wire projection until the
//!   broker GCs it or the client disconnects.
//! - **Every other verb** — one request → one response. The per-conn
//!   thread ships an [`IpcRequest`], waits up to [`REPLY_TIMEOUT`] for
//!   the driver's response, writes it, and loops to read the next
//!   line.

use compact_str::CompactString;
use crossbeam::channel::{RecvTimeoutError, SendTimeoutError, Sender, bounded};
use serde::Serialize;
use std::borrow::Cow;
use std::io::{self, BufRead, BufReader, ErrorKind, Write};
use std::ops::ControlFlow;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::thread;
use std::time::Duration;

use crate::channels::IpcServerSide;
use crate::ipc::protocol::{
    ERR_BUSY, ERR_MALFORMED, ERR_SHUTDOWN, IpcRequest, RequestPayload, ResponsePayload, WireRequest,
};
use crate::ipc::wire::{BrokerEvent, WireDiagnostic};

/// Max concurrent operator client connections. Operator usage is
/// human-paced (`status`, `list`, `wait`); 8 covers the worst-case
/// "operator dashboard + ad-hoc shell + on-call wait + automation"
/// fan-in without inviting accidental DoS.
pub(crate) const MAX_IPC_CONNS: usize = 8;

/// Per-connection socket write timeout. Bounds shutdown latency: the
/// bin's teardown path can `shutdown_flag = true` and the longest a
/// stuck client wedges the per-conn thread is this many seconds per
/// write call. Without this, kernel-keepalive times in the minutes
/// would dominate teardown.
pub(crate) const PER_CONN_WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// How long the per-conn thread waits on `reply_rx.recv_timeout`
/// before returning [`ERR_SHUTDOWN`]. Bounded by a healthy driver
/// tick's duration (sub-ms to ms) plus headroom; 5s captures
/// pathological deferred-attach contention without becoming an
/// operator-visible "hung" feel.
pub(crate) const REPLY_TIMEOUT: Duration = Duration::from_secs(5);

/// Empty-accept sleep — back-pressure against busy-looping the
/// accept loop when the listener has nothing pending. 50ms is
/// indistinguishable from human-paced operator latency.
const ACCEPT_IDLE_SLEEP: Duration = Duration::from_millis(50);

/// Per-subscriber event-queue capacity. ~1s of headroom under heavy
/// load (one diagnostic per ms is the engine's pathological burst
/// rate). Saturation triggers `_missed` markers via the broker's
/// per-subscriber accumulator (see [`crate::driver::broker`]).
pub(crate) const SUBSCRIBE_QUEUE: usize = 256;

/// Run the IPC server thread to completion.
///
/// The thread exits when `shutdown_flag` flips to `true`. Accepted
/// per-connection workers are detached; they observe shutdown via
/// their own `shutdown_flag` clone, the [`PER_CONN_WRITE_TIMEOUT`]
/// on writes, and EOF from clients. The listener fd closes when this
/// function returns and the bound [`UnixListener`] drops — the
/// thread-entry shape is by-value, the inverse of the loop-body
/// pattern `watcher_loop` and `config_watcher_loop` use.
///
/// The signature takes [`UnixListener`], [`IpcServerSide`], and the
/// shutdown flag's [`Arc`] **by value** because this is a thread entry
/// point — the function owns the resources for its lifetime. The
/// `clippy::needless_pass_by_value` lint would prefer references for
/// the [`Arc`], but for thread-entry semantics that's the wrong
/// shape: a reference would force the caller to outlive the spawned
/// thread, which is exactly the lifetime story we want to refuse.
#[allow(clippy::needless_pass_by_value)]
pub(crate) fn ipc_server_run(
    listener: UnixListener,
    side: IpcServerSide,
    shutdown_flag: Arc<AtomicBool>,
) {
    // Non-blocking accept lets the loop poll `shutdown_flag` between
    // accept attempts. Failure here is unrecoverable for the IPC
    // surface — without it shutdown would have to rely on closing the
    // listener fd from outside, which races in-flight connections.
    if let Err(e) = listener.set_nonblocking(true) {
        tracing::error!(
            ?e,
            "ipc: failed to set listener non-blocking; thread exiting"
        );
        return;
    }

    // Per-accept-loop counter — scoped to this function's stack, so a
    // panic on the way out drops it cleanly. Each spawned worker owns
    // its own [`ConnGuard`] clone; the decrement on every worker exit
    // is RAII via Drop, independent of the exit path.
    let conn_cap = Arc::new(AtomicUsize::new(0));
    let mut conn_id: u64 = 0;

    loop {
        if shutdown_flag.load(Ordering::Acquire) {
            return;
        }
        match listener.accept() {
            Ok((stream, _addr)) => {
                // Reserve the slot up front. A concurrent decrement
                // path cannot land us under-cap (it can only ever go
                // down to 0), so a fetch-add-then-check-then-decrement
                // never rejects a slot that would otherwise be free.
                let prev = conn_cap.fetch_add(1, Ordering::AcqRel);
                if prev >= MAX_IPC_CONNS {
                    conn_cap.fetch_sub(1, Ordering::AcqRel);
                    write_err_then_drop(
                        &stream,
                        ERR_BUSY,
                        format!("at most {MAX_IPC_CONNS} concurrent operator connections"),
                    );
                    continue;
                }
                if let Err(e) = stream.set_write_timeout(Some(PER_CONN_WRITE_TIMEOUT)) {
                    tracing::warn!(?e, "ipc: set_write_timeout failed; refusing conn");
                    conn_cap.fetch_sub(1, Ordering::AcqRel);
                    write_err_then_drop(
                        &stream,
                        ERR_BUSY,
                        "daemon could not configure connection write timeout".to_string(),
                    );
                    continue;
                }
                conn_id = conn_id.wrapping_add(1);

                let req_tx = side.ipc_request_tx.clone();
                let cap = Arc::clone(&conn_cap);
                let flag = Arc::clone(&shutdown_flag);
                let spawn_res = thread::Builder::new()
                    .name(format!("specter-ipc-conn-{conn_id}"))
                    .spawn(move || {
                        let _guard = ConnGuard(cap);
                        handle_connection(stream, req_tx, flag);
                    });
                if let Err(e) = spawn_res {
                    conn_cap.fetch_sub(1, Ordering::AcqRel);
                    tracing::warn!(?e, "ipc: connection thread spawn failed");
                }
            }
            Err(e) if e.kind() == ErrorKind::WouldBlock => {
                thread::sleep(ACCEPT_IDLE_SLEEP);
            }
            Err(e) => {
                // EMFILE, ENFILE, and other transient pressures
                // recover on their own; we don't kill the thread. A
                // persistent failure floods the log at one line per
                // `ACCEPT_IDLE_SLEEP`, which is itself a structural
                // signal an operator wants to see.
                tracing::warn!(?e, "ipc: accept failed");
                thread::sleep(ACCEPT_IDLE_SLEEP);
            }
        }
    }
}

/// RAII decrement for the per-accept-loop connection counter. The
/// per-conn worker thread holds one; the counter falls when the
/// worker exits (EOF, write failure, shutdown). Independent of the
/// thread's exit path — a panic still runs `Drop`.
struct ConnGuard(Arc<AtomicUsize>);

impl Drop for ConnGuard {
    fn drop(&mut self) {
        self.0.fetch_sub(1, Ordering::AcqRel);
    }
}

/// Per-connection event loop. Reads line-delimited [`WireRequest`]s
/// off the stream and dispatches each to a one-shot or
/// stream-Subscribe handler.
///
/// Owned arguments (`stream`, `request_tx`, `shutdown_flag`) mirror
/// the thread-entry shape: the spawned closure moves them in, this
/// function owns them for its lifetime, and they drop on return. The
/// helper functions called below ([`handle_subscribe`] /
/// [`handle_oneshot`]) take references — they don't need ownership,
/// and the references stay valid for the duration of each call. The
/// `clippy::needless_pass_by_value` lint suggests reverting to
/// references, but for per-connection-thread semantics that's wrong:
/// the spawned closure's lifetime is the function call's lifetime,
/// and the by-value shape is what makes that ownership transfer
/// explicit at the call site.
///
/// The Subscribe arm is terminal for the connection: once streaming
/// begins, the connection's only output is diagnostics. Every other
/// verb is one request → one response; the loop continues reading
/// the next line until EOF, parse failure routes back as
/// [`ERR_MALFORMED`], or `shutdown_flag` flips.
#[allow(clippy::needless_pass_by_value)]
fn handle_connection(
    stream: UnixStream,
    request_tx: Sender<IpcRequest>,
    shutdown_flag: Arc<AtomicBool>,
) {
    // `try_clone` separates the read and write halves so the per-conn
    // thread can hold a buffered reader against the same socket while
    // writing through the original handle. Failure here is rare (only
    // `EMFILE` / `ENFILE`) and unrecoverable for this connection.
    let read_clone = match stream.try_clone() {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!(?e, "ipc: stream clone failed; dropping conn");
            return;
        }
    };
    let mut reader = BufReader::new(read_clone);
    let mut writer = stream;
    let mut line = String::new();

    while !shutdown_flag.load(Ordering::Acquire) {
        line.clear();
        match reader.read_line(&mut line) {
            Ok(0) => return, // EOF — client closed.
            Ok(_) => {}
            Err(e) => {
                tracing::debug!(?e, "ipc: read_line failed; dropping conn");
                return;
            }
        }

        let parsed = match parse_wire_request(&line) {
            Ok(p) => p,
            Err(e) => {
                let _ = write_response(
                    &mut writer,
                    &ResponsePayload::Err {
                        code: Cow::Borrowed(ERR_MALFORMED),
                        error: format!("json parse: {e}"),
                    },
                );
                continue;
            }
        };

        match parsed {
            WireRequest::Subscribe { name } => {
                // Subscribe is terminal for this connection. Once the
                // stream loop starts the only output is diagnostics
                // until disconnect, so we return whether the ack was
                // success or an error response.
                let _ = handle_subscribe(&mut writer, &request_tx, name);
                return;
            }
            other => {
                if handle_oneshot(&mut writer, &request_tx, other).is_break() {
                    return;
                }
            }
        }
    }
}

/// Subscribe arm: send the request with a per-subscriber channel,
/// wait for the [`ResponsePayload::SubscribeAck`], write it to the
/// client, then stream every event until the broker GCs the
/// subscriber (channel close) or the client disconnects (write
/// failure).
///
/// # Channel-edge discipline
///
/// Both `request_tx.send_timeout` and `reply_rx.recv_timeout` carry
/// the [`REPLY_TIMEOUT`] deadline; the four resulting variants map
/// to operator-actionable codes:
///
/// - send Timeout → [`ERR_BUSY`]: driver is wedged, the operator's
///   action is "retry" (likely a fresh connection — this one
///   breaks).
/// - send Disconnected → break with no write: the driver is gone,
///   writing a fresh frame would race the same disconnect.
/// - recv Timeout → [`ERR_BUSY`]: request was accepted but the
///   driver could not produce a reply in time.
/// - recv Disconnected → [`ERR_SHUTDOWN`]: the driver exited
///   between accept and reply.
///
/// # Shutdown signal
///
/// The stream loop's canonical shutdown signal is `event_rx`
/// disconnection. When the engine driver is dropped on its way
/// down, the broker drops with it; every `Sender<BrokerEvent>`
/// clone the broker held releases, every blocked `recv` returns
/// `Disconnected`, and the `for ev in event_rx` loop exits
/// cleanly. The broker-drop arrives without dependence on event
/// traffic, so a quiet daemon's per-conn worker exits with the
/// same latency a busy one does — no `shutdown_flag` poll is
/// needed mid-stream.
fn handle_subscribe(
    writer: &mut UnixStream,
    request_tx: &Sender<IpcRequest>,
    name: Option<CompactString>,
) -> ControlFlow<()> {
    let (event_tx, event_rx) = bounded::<BrokerEvent>(SUBSCRIBE_QUEUE);
    let (reply_tx, reply_rx) = bounded::<ResponsePayload>(1);

    let req = IpcRequest {
        payload: RequestPayload::Subscribe { tx: event_tx, name },
        reply_tx,
    };
    match request_tx.send_timeout(req, REPLY_TIMEOUT) {
        Ok(()) => {}
        Err(SendTimeoutError::Timeout(_)) => {
            return write_err_then_break(
                writer,
                ERR_BUSY,
                "driver did not accept request within REPLY_TIMEOUT".into(),
            );
        }
        Err(SendTimeoutError::Disconnected(_)) => return ControlFlow::Break(()),
    }

    let ack = match reply_rx.recv_timeout(REPLY_TIMEOUT) {
        Ok(r) => r,
        Err(RecvTimeoutError::Timeout) => ResponsePayload::Err {
            code: Cow::Borrowed(ERR_BUSY),
            error: "driver did not reply within REPLY_TIMEOUT".into(),
        },
        Err(RecvTimeoutError::Disconnected) => ResponsePayload::Err {
            code: Cow::Borrowed(ERR_SHUTDOWN),
            error: "driver exited mid-request".into(),
        },
    };
    let ack_is_success = matches!(ack, ResponsePayload::SubscribeAck { .. });
    if write_response(writer, &ack).is_err() {
        return ControlFlow::Break(());
    }
    if !ack_is_success {
        // Ack-side error: the broker holds no subscriber for this
        // conn (the handle_ipc Subscribe arm early-returned before
        // add_subscriber). The Err line was the connection's last
        // output; the client closes its read.
        return ControlFlow::Continue(());
    }

    // Stream loop. Iterating the receiver blocks per-event; the
    // broker-drop path closes `event_rx` and the for-loop exits.
    for ev in event_rx {
        let wire = WireDiagnostic::from(&ev);
        if write_json_line(writer, &wire).is_err() {
            return ControlFlow::Break(());
        }
    }
    // `event_rx` closed — the broker GC'd us (driver shutdown or
    // sender dropped). Clean exit.
    ControlFlow::Continue(())
}

/// One-shot verb arm: send a request, wait for the response, write
/// it, return [`ControlFlow::Continue`] so the connection loop reads
/// the next line.
///
/// Channel-edge discipline mirrors [`handle_subscribe`]: a send
/// Timeout breaks the connection with a structured [`ERR_BUSY`]
/// frame, a send Disconnected breaks without writing, and the
/// recv-side Timeout / Disconnected map to [`ERR_BUSY`] /
/// [`ERR_SHUTDOWN`] so the operator's mental model distinguishes
/// "retry, daemon slow" from "daemon gone".
fn handle_oneshot(
    writer: &mut UnixStream,
    request_tx: &Sender<IpcRequest>,
    wire: WireRequest,
) -> ControlFlow<()> {
    let payload = wire_to_request_payload(wire);
    let (reply_tx, reply_rx) = bounded::<ResponsePayload>(1);
    let req = IpcRequest { payload, reply_tx };
    match request_tx.send_timeout(req, REPLY_TIMEOUT) {
        Ok(()) => {}
        Err(SendTimeoutError::Timeout(_)) => {
            return write_err_then_break(
                writer,
                ERR_BUSY,
                "driver did not accept request within REPLY_TIMEOUT".into(),
            );
        }
        Err(SendTimeoutError::Disconnected(_)) => return ControlFlow::Break(()),
    }
    let response = match reply_rx.recv_timeout(REPLY_TIMEOUT) {
        Ok(r) => r,
        Err(RecvTimeoutError::Timeout) => ResponsePayload::Err {
            code: Cow::Borrowed(ERR_BUSY),
            error: "driver did not reply within REPLY_TIMEOUT".into(),
        },
        Err(RecvTimeoutError::Disconnected) => ResponsePayload::Err {
            code: Cow::Borrowed(ERR_SHUTDOWN),
            error: "driver exited mid-request".into(),
        },
    };
    match write_response(writer, &response) {
        Ok(()) => ControlFlow::Continue(()),
        Err(_) => ControlFlow::Break(()),
    }
}

/// Map every non-`Subscribe` [`WireRequest`] variant to its
/// [`RequestPayload`] counterpart.
///
/// Subscribe is excluded by construction: it carries a
/// `Sender<BrokerEvent>` the per-conn thread builds at dispatch
/// time, which this free function has no way to synthesise.
/// [`handle_subscribe`] owns the Subscribe arm directly.
fn wire_to_request_payload(w: WireRequest) -> RequestPayload {
    match w {
        WireRequest::Status => RequestPayload::Status,
        WireRequest::List => RequestPayload::List,
        WireRequest::Show { name } => RequestPayload::Show { name },
        WireRequest::Disable { name } => RequestPayload::Disable { name },
        WireRequest::Enable { name } => RequestPayload::Enable { name },
        WireRequest::Reload => RequestPayload::Reload,
        WireRequest::Subscribe { .. } => unreachable!(
            "handle_subscribe owns the Subscribe arm; this adapter is unreachable for it",
        ),
    }
}

/// Parse one line from a per-conn read into a [`WireRequest`]. The
/// trailing newline produced by [`BufRead::read_line`] is trimmed
/// once before serde sees the bytes — a stray newline inside the
/// JSON object would still surface as a parse error, but the common
/// "BufRead leaves the \n in place" case is handled here so callers
/// don't repeat the trim.
fn parse_wire_request(line: &str) -> serde_json::Result<WireRequest> {
    serde_json::from_str(line.trim_end_matches('\n'))
}

/// Write a [`ResponsePayload`] as one JSON line.
fn write_response(stream: &mut UnixStream, resp: &ResponsePayload) -> io::Result<()> {
    write_json_line(stream, resp)
}

/// Serialize `val` to JSON, append a single newline, and write the
/// whole thing in one `write_all`. The serialize-then-write order
/// matters: a serialiser failure must not write a partial line
/// onto the wire (clients rely on line framing).
fn write_json_line<T: Serialize>(stream: &mut UnixStream, val: &T) -> io::Result<()> {
    let mut buf =
        serde_json::to_vec(val).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    buf.push(b'\n');
    stream.write_all(&buf)
}

/// Best-effort write of a structured error response on a connection
/// the accept loop is about to drop. Honours the module-wide
/// "structured error before EOF" contract on every refusal path:
/// `MAX_IPC_CONNS` overflow, `set_write_timeout` configuration
/// failure, and any future variant of "we accepted the socket but
/// cannot safely serve this client".
///
/// `try_clone` is the path to a `&mut`-able handle for
/// [`write_json_line`]; failure here (rare) collapses to "client
/// sees EOF instead of the structured error", which is acceptable.
fn write_err_then_drop(stream: &UnixStream, code: &'static str, error: String) {
    let resp = ResponsePayload::Err {
        code: Cow::Borrowed(code),
        error,
    };
    if let Ok(mut s) = stream.try_clone() {
        let _ = write_response(&mut s, &resp);
    }
}

/// Best-effort write of a structured error response on the per-conn
/// writer, then break the connection. Sole use is the
/// `SendTimeoutError::Timeout` arm of an [`IpcRequest`] dispatch:
/// the driver is wedged, this connection cannot make further
/// progress, but the operator deserves a structured frame before
/// EOF.
///
/// A write failure here is silently dropped — the connection is
/// already on the break path, and a failed write means the peer
/// is gone anyway.
fn write_err_then_break(
    writer: &mut UnixStream,
    code: &'static str,
    error: String,
) -> ControlFlow<()> {
    let resp = ResponsePayload::Err {
        code: Cow::Borrowed(code),
        error,
    };
    let _ = write_response(writer, &resp);
    ControlFlow::Break(())
}

#[cfg(test)]
mod tests {
    use super::{parse_wire_request, wire_to_request_payload, write_json_line};
    use crate::ipc::protocol::{RequestPayload, ResponsePayload, WireRequest};
    use compact_str::CompactString;
    use std::io::{BufRead, BufReader};
    use std::os::unix::net::UnixStream;

    /// `write_json_line` appends exactly one LF terminator and no
    /// internal pretty-print newlines. Operators (and the client side)
    /// read line-by-line; a serialiser tweak that switched to
    /// `to_string_pretty` would break framing — this test fails.
    ///
    /// Uses [`ResponsePayload::Ok`] as the witness because it is both
    /// `Serialize` (server writes responses) and the simplest variant on
    /// the wire — `{"kind":"ok"}\n`. [`WireRequest`] is `Deserialize`-only
    /// on purpose (the daemon never round-trips its own request shape),
    /// so the response side is the right write-path witness.
    #[test]
    fn write_json_line_appends_newline() {
        let (mut writer, reader) = UnixStream::pair().expect("socketpair");
        write_json_line(&mut writer, &ResponsePayload::Ok).expect("write succeeds");
        drop(writer);
        let mut reader = BufReader::new(reader);
        let mut line = String::new();
        reader.read_line(&mut line).expect("read_line");
        assert!(line.ends_with('\n'), "must terminate with LF; got {line:?}");
        assert_eq!(line.matches('\n').count(), 1, "exactly one newline");
        // Sanity: the line is the canonical Ok shape.
        assert_eq!(line.trim_end(), r#"{"kind":"ok"}"#);
    }

    /// `read_line` from `BufRead` leaves the trailing `\n` in the
    /// buffer; `parse_wire_request` strips it once before serde sees the
    /// bytes. A client that includes the newline in its request must
    /// still parse cleanly — the daemon refuses to silently accept a
    /// double-newline (that's still a serde error), but a single
    /// trailing newline is the canonical line-framed shape.
    #[test]
    fn parse_wire_request_handles_trailing_newline() {
        let parsed = parse_wire_request("{\"op\":\"status\"}\n").expect("parse");
        assert!(matches!(parsed, WireRequest::Status));
    }

    /// Every non-Subscribe verb maps to its `RequestPayload` counterpart.
    /// A regression that swapped the variants (e.g. `WireRequest::List`
    /// → `RequestPayload::Show`) would fail here — the table is
    /// load-bearing because clients address by verb name and the daemon
    /// dispatches by enum tag.
    #[test]
    fn wire_to_request_payload_round_trips_non_subscribe() {
        assert!(matches!(
            wire_to_request_payload(WireRequest::Status),
            RequestPayload::Status,
        ));
        assert!(matches!(
            wire_to_request_payload(WireRequest::List),
            RequestPayload::List,
        ));
        let show = wire_to_request_payload(WireRequest::Show {
            name: CompactString::const_new("foo"),
        });
        match show {
            RequestPayload::Show { name } => assert_eq!(name.as_str(), "foo"),
            other => panic!("expected Show, got {other:?}"),
        }
        let disable = wire_to_request_payload(WireRequest::Disable {
            name: CompactString::const_new("bar"),
        });
        match disable {
            RequestPayload::Disable { name } => assert_eq!(name.as_str(), "bar"),
            other => panic!("expected Disable, got {other:?}"),
        }
        let enable = wire_to_request_payload(WireRequest::Enable {
            name: CompactString::const_new("baz"),
        });
        match enable {
            RequestPayload::Enable { name } => assert_eq!(name.as_str(), "baz"),
            other => panic!("expected Enable, got {other:?}"),
        }
        assert!(matches!(
            wire_to_request_payload(WireRequest::Reload),
            RequestPayload::Reload,
        ));
    }

    /// `handle_subscribe` owns the Subscribe arm; feeding Subscribe into
    /// the one-shot adapter is a construction error. The `unreachable!`
    /// is structural — a future refactor that started feeding Subscribe
    /// through this path needs a deliberate rework, not a silent
    /// fall-through.
    #[test]
    #[should_panic(expected = "handle_subscribe owns the Subscribe arm")]
    fn wire_to_request_payload_panics_on_subscribe() {
        let _ = wire_to_request_payload(WireRequest::Subscribe { name: None });
    }
}
