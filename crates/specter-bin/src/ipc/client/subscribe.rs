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
use crate::ipc::framing::parse_strict;
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
///
/// `line_buf` is the reused inbound line buffer — [`Self::read_next`]
/// clears it before each [`BufRead::read_line`] so the per-event
/// allocation collapses to zero after the first read sizes the heap.
/// Symmetric with [`crate::ipc::client::tail`]'s reused render buffer
/// on the outbound side.
pub(crate) struct Subscription {
    reader: BufReader<UnixStream>,
    line_buf: String,
}

/// Open a daemon connection, ship the Subscribe request, validate
/// the ack, return a [`Subscription`] ready for the stream loop.
///
/// `verb` is the operator-facing command name (`"tail"` / `"wait"`),
/// the `specter <verb>:` prefix on every failure path: transport
/// stages route through [`connect::emit_error`], the ack tail through
/// [`connect::fail_response`]. Call sites `return code` directly.
///
/// `name = None` ⇒ unfiltered subscription (the `tail` shape).
/// `name = Some(_)` ⇒ per-Sub filter, server-resolved atomically
/// inside the Subscribe handler (closes the historical
/// resolve-then-subscribe race window — `disable` either lands
/// before, producing `WireErrorCode::UnknownSub`, or after,
/// surfacing as `SubDetached` on the stream).
pub(crate) fn open(
    client: &ClientArgs,
    verb: &'static str,
    name: Option<CompactString>,
) -> Result<Subscription, ExitCode> {
    let socket = connect::resolve_socket(client);
    let mut stream = connect::open(&socket).map_err(|e| {
        connect::emit_error(
            client,
            format_args!(
                "specter {verb}: cannot connect to {}: {e}",
                socket.display()
            ),
        );
        ExitCode::from(1)
    })?;

    // Ship the Subscribe through the same write helper status/list/show
    // use — JSON line + LF in one `write_all`. Symmetric with the
    // one-shot verbs so a daemon-side parse-error path is identical.
    connect::write_request(&mut stream, &WireRequest::Subscribe { name }).map_err(|e| {
        connect::emit_error(client, format_args!("specter {verb}: send failed: {e}"));
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
        connect::emit_error(client, format_args!("specter {verb}: receive failed: {e}"));
        ExitCode::from(1)
    })?;
    if n == 0 {
        connect::emit_error(
            client,
            format_args!("specter {verb}: daemon closed connection before ack"),
        );
        return Err(ExitCode::from(1));
    }
    let ack: ResponsePayload =
        parse_strict(line.trim_end_matches('\n').as_bytes()).map_err(|e| {
            connect::emit_error(
                client,
                format_args!("specter {verb}: parse ack failed: {e}"),
            );
            ExitCode::from(1)
        })?;
    match ack {
        ResponsePayload::SubscribeAck { .. } => Ok(Subscription {
            reader,
            line_buf: String::new(),
        }),
        other => Err(connect::fail_response(client, verb, other)),
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
    ///
    /// The inbound line buffer is reused across calls — `clear`
    /// preserves the previously-grown heap so subsequent lines pay no
    /// allocation. A partial read on `Err` discards the bytes (the
    /// next call clears before reading), matching the
    /// fresh-`String`-per-call behaviour the prior shape carried.
    pub(crate) fn read_next(&mut self) -> io::Result<Option<WireDiagnostic>> {
        self.line_buf.clear();
        let n = self.reader.read_line(&mut self.line_buf)?;
        if n == 0 {
            return Ok(None);
        }
        serde_json::from_str::<WireDiagnostic>(self.line_buf.trim_end_matches('\n'))
            .map(Some)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
    }
}
