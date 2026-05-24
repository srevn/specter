//! Subscribe-arm scaffold shared by `specter tail` and
//! `specter wait`.
//!
//! Both verbs open a connection, write a [`WireRequest::Subscribe`],
//! read + validate the [`ResponsePayload::SubscribeAck`] line, and
//! then loop reading line-delimited [`WireDiagnostic`]s. The pre-loop
//! work is identical; this module owns it.
//!
//! The post-ack stream loop diverges between the two verbs:
//!
//! - `tail` reads indefinitely until EOF or I/O error; the deadline
//!   is cleared once after open via [`Subscription::set_read_timeout`].
//! - `wait` re-applies the deadline before every read so the
//!   `--timeout` budget is honoured; EOF without a match is a
//!   distinct outcome from a deadline-fired timeout (the consumer
//!   sees `Ok(None)` versus a `WouldBlock`/`TimedOut` error).
//!
//! Both behaviours land on the [`Subscription::read_next`] +
//! [`Subscription::set_read_timeout`] surface here.

use compact_str::CompactString;
use specter_config::ClientArgs;
use std::io::{self, BufRead, BufReader};
use std::os::unix::net::UnixStream;
use std::process::ExitCode;
use std::time::Duration;

use crate::ipc::client::connect;
use crate::ipc::protocol::{ResponsePayload, WireRequest};
use crate::ipc::wire::WireDiagnostic;

/// Live subscription over a daemon connection: post-ack, ready for
/// streamed [`WireDiagnostic`] reads.
///
/// Drops the underlying [`UnixStream`] on `drop`. The daemon's mio
/// reactor observes the peer-side close on the next read drain and
/// terminates the corresponding per-conn entry, dropping the
/// subscriber from the fan-out map on the same tick. No explicit
/// "unsubscribe" verb is required.
pub(crate) struct Subscription {
    reader: BufReader<UnixStream>,
}

/// Open a daemon connection, ship the Subscribe request, validate
/// the ack, return a [`Subscription`] ready for the stream loop.
///
/// `verb` is the operator-facing command name (`"tail"` / `"wait"`),
/// used as the `eprintln!` prefix on every failure path so the
/// caller's `match` arms stay minimal — call sites `return code`
/// directly without rewriting the message.
///
/// `name = None` ⇒ unfiltered subscription (the `tail` shape).
/// `name = Some(_)` ⇒ per-Sub filter, server-resolved atomically
/// inside the Subscribe handler (closes the historical
/// resolve-then-subscribe race window — `disable` either lands
/// before, producing `ERR_UNKNOWN_SUB`, or after, surfacing as
/// `SubDetached` on the stream).
pub(crate) fn open(
    client: &ClientArgs,
    verb: &'static str,
    name: Option<CompactString>,
) -> Result<Subscription, ExitCode> {
    let socket = connect::resolve_socket(client);
    let mut stream = connect::open(&socket).map_err(|e| {
        eprintln!(
            "specter {verb}: cannot connect to {}: {e}",
            socket.display(),
        );
        ExitCode::from(1)
    })?;

    // Ship the Subscribe through the same write helper status/list/show
    // use — JSON line + LF in one `write_all`. Symmetric with the
    // one-shot verbs so a daemon-side parse-error path is identical.
    connect::write_request(&mut stream, &WireRequest::Subscribe { name }).map_err(|e| {
        eprintln!("specter {verb}: send failed: {e}");
        ExitCode::from(1)
    })?;

    // BufReader wraps the stream by ownership. The ack uses the
    // connect-time 5s read deadline (mirrors `connect::round_trip`);
    // the caller overrides it after `open` returns — `tail` clears
    // to indefinite, `wait` re-applies the remaining `--timeout`
    // budget before each read.
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let n = reader.read_line(&mut line).map_err(|e| {
        eprintln!("specter {verb}: receive failed: {e}");
        ExitCode::from(1)
    })?;
    if n == 0 {
        eprintln!("specter {verb}: daemon closed connection before ack");
        return Err(ExitCode::from(1));
    }
    let ack: ResponsePayload = serde_json::from_str(line.trim_end_matches('\n')).map_err(|e| {
        eprintln!("specter {verb}: parse ack failed: {e}");
        ExitCode::from(1)
    })?;
    match ack {
        ResponsePayload::SubscribeAck { .. } => Ok(Subscription { reader }),
        ResponsePayload::Err { code, error } => {
            eprintln!("specter {verb}: {code}: {error}");
            Err(ExitCode::from(1))
        }
        other => {
            eprintln!("specter {verb}: unexpected ack: {other:?}");
            Err(ExitCode::from(1))
        }
    }
}

impl Subscription {
    /// Set the read deadline on the underlying socket. `None` clears
    /// any prior deadline (used by `tail` once after open). `wait`
    /// calls this before every iteration with the remaining
    /// `--timeout` budget.
    ///
    /// Goes through [`BufReader::get_mut`] because
    /// [`UnixStream::set_read_timeout`] is a syscall on the
    /// underlying fd; the buffered-reader layer has no timeout knob
    /// of its own. The reader's internal buffer is preserved across
    /// this call — a partial line in the buffer (the daemon's last
    /// dispatch landed before a deadline fired) is still readable on
    /// the next successful `read_next`.
    pub(crate) fn set_read_timeout(&mut self, t: Option<Duration>) -> io::Result<()> {
        self.reader.get_mut().set_read_timeout(t)
    }

    /// Read the next streamed line, parse to [`WireDiagnostic`].
    ///
    /// - `Ok(Some(wire))` — a complete line parsed cleanly.
    /// - `Ok(None)` — EOF; the daemon closed the connection (per-conn
    ///   teardown on driver shutdown, or peer-initiated close).
    /// - `Err(io::Error)` — any other failure. A JSON parse failure
    ///   maps to `ErrorKind::InvalidData`, mirroring
    ///   [`connect::read_response`]'s discipline so callers can
    ///   distinguish "transport broken" from "daemon sent malformed
    ///   JSON" with one error type.
    pub(crate) fn read_next(&mut self) -> io::Result<Option<WireDiagnostic>> {
        let mut line = String::new();
        let n = self.reader.read_line(&mut line)?;
        if n == 0 {
            return Ok(None);
        }
        serde_json::from_str::<WireDiagnostic>(line.trim_end_matches('\n'))
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
