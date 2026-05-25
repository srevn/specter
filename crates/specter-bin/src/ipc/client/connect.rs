//! Client-side connect + framing helpers.
//!
//! Single source of socket-timeout policy: every verb handler
//! reaches the daemon through [`open`], which applies a 5s read
//! deadline and a 2s write deadline. A daemon that takes longer than
//! either to respond is operator-visibly hung; surfacing the deadline
//! is better than a silent indefinite park.
//!
//! # Framing
//!
//! Requests and responses are line-delimited JSON, one object per
//! line — the shared framing contract owned by
//! [`crate::ipc::framing::encode_line`]. [`write_request`]
//! serialises + appends LF + writes in one `write_all`.
//! [`read_response`] reads through a [`BufReader`] until a newline,
//! then strict-parses via [`crate::ipc::framing::parse_strict`].

use specter_config::ClientArgs;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::time::Duration;

use crate::ipc::framing::{encode_line, parse_strict};
use crate::ipc::protocol::{ResponsePayload, WireRequest};
use crate::ipc::sockpath;

/// Read deadline — the daemon's mio-reactor tick runs in sub-ms to
/// ms under healthy load; 5s covers mio-tick contention under a
/// heavily-loaded reactor without becoming an operator-visible "hung"
/// feel. The daemon answers inline on the reactor thread, so every
/// operator verb has the same horizon regardless of server load.
const READ_TIMEOUT: Duration = Duration::from_secs(5);

/// Write deadline — the client's outgoing request is small (sub-KB),
/// so the timeout's primary role is symmetry: a daemon that refuses
/// to drain its accept queue surfaces as a write timeout rather than
/// a hung `write_all`. 2s rides out scheduler contention on a busy
/// reactor without masking a hung socket.
const WRITE_TIMEOUT: Duration = Duration::from_secs(2);

/// Connect to the daemon's IPC socket and apply read/write
/// timeouts. The two timeouts are non-overlapping: read covers the
/// daemon-to-client direction (response latency), write covers the
/// client-to-daemon direction (request shipping). Errors propagate
/// the underlying `io::ErrorKind` so the verb handler can render the
/// operator-visible cause precisely.
pub(crate) fn open(socket: &Path) -> io::Result<UnixStream> {
    let stream = UnixStream::connect(socket)?;
    stream.set_read_timeout(Some(READ_TIMEOUT))?;
    stream.set_write_timeout(Some(WRITE_TIMEOUT))?;
    Ok(stream)
}

/// Serialise a [`WireRequest`] to a single LF-delimited line and
/// write it in one [`Write::write_all`] call.
///
/// The atomic single-write matters for framing correctness: a
/// partial write would leave the daemon parsing half a JSON object
/// next time the conn is readable, and serde's compact serializer
/// (used by [`encode_line`]) emits the object without
/// internal newlines so the daemon's LF-splitter only ever sees the
/// trailing frame delimiter. The wrapper's
/// [`crate::ipc::framing::InfallibleSerialize`] bound asserts the
/// `Vec<u8>`-build cannot fail for [`WireRequest`] (audited at the
/// impl site in [`crate::ipc::protocol`]), so the only fallible
/// step here is [`Write::write_all`].
pub(crate) fn write_request(stream: &mut UnixStream, req: &WireRequest) -> io::Result<()> {
    stream.write_all(&encode_line(req))
}

/// Read the daemon's next JSON line and parse it as a
/// [`ResponsePayload`]. Trailing newlines are tolerated (the daemon
/// always emits one; future framing tweaks shouldn't break clients).
///
/// Strict parse via [`parse_strict`]: a daemon-bug response carrying
/// a stale field name surfaces as
/// [`io::ErrorKind::InvalidData`] (the boundary's uniform parse-
/// failure shape) rather than silently dropping the unknown key.
/// Symmetric with the daemon-side request gate so an unknown field
/// is a deterministic, debuggable failure in either direction.
pub(crate) fn read_response(stream: &mut UnixStream) -> io::Result<ResponsePayload> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    let n = reader.read_line(&mut line)?;
    if n == 0 {
        return Err(io::Error::new(
            io::ErrorKind::UnexpectedEof,
            "daemon closed connection before responding",
        ));
    }
    parse_strict(line.trim_end_matches('\n').as_bytes())
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
}

/// Resolve the operator-facing socket path. CLI override wins; the
/// per-platform default backs it. Shared across every client verb so
/// the resolution stays single-source.
///
/// `client.socket: Option<PathBuf>` carries `--socket`;
/// [`sockpath::default_socket_path`] mirrors the daemon's bind-time
/// resolution. The override is taken by `Path` reference + `to_owned`
/// so the default branch never pays the allocation it isn't using.
pub(crate) fn resolve_socket(client: &ClientArgs) -> PathBuf {
    client
        .socket
        .as_deref()
        .map_or_else(sockpath::default_socket_path, Path::to_path_buf)
}

/// Stereotyped one-shot round trip — resolve socket, open, write
/// request, read response.
///
/// Every verb's network-side work has the same four-step shape;
/// centralising it keeps the operator-visible error surface uniform
/// (`specter <verb>: <stage>: <io::Error>`) and the per-stage prefixes
/// single-source. The mapping from io::Error stage to operator
/// vocabulary is hand-rolled per stage because each error path needs a
/// distinct prefix — a single template doesn't fit.
///
/// `verb` is the operator-facing command name (`"status"`, `"list"`,
/// …) and is constrained to `&'static str` so callers cannot
/// accidentally pass a borrowed runtime string.
///
/// Returns `Ok(response)` on a successful round trip; `Err(code)`
/// after eprinting the structured cause — callers `return code`
/// directly.
pub(crate) fn round_trip(
    client: &ClientArgs,
    verb: &'static str,
    request: &WireRequest,
) -> Result<ResponsePayload, ExitCode> {
    let socket = resolve_socket(client);
    let mut stream = open(&socket).map_err(|e| {
        eprintln!(
            "specter {verb}: cannot connect to {}: {e}",
            socket.display(),
        );
        ExitCode::from(1)
    })?;
    write_request(&mut stream, request).map_err(|e| {
        eprintln!("specter {verb}: send failed: {e}");
        ExitCode::from(1)
    })?;
    read_response(&mut stream).map_err(|e| {
        eprintln!("specter {verb}: receive failed: {e}");
        ExitCode::from(1)
    })
}

/// One-shot round-trip for a verb whose successful response is the
/// unit-shaped [`ResponsePayload::Ok`] — `disable`, `enable`,
/// `reload`, and any future verb whose ack carries no payload.
///
/// Maps each response arm to the operator-visible outcome:
/// - [`ResponsePayload::Ok`] → exit `0`.
/// - [`ResponsePayload::Err`] → render `specter <verb>: <code>:
///   <error>` on stderr and exit `1`. The structured `code` lets
///   operators branch in shell scripts; the human-readable `error`
///   carries the amplification.
/// - Any other variant → render `specter <verb>: unexpected
///   response: <debug>` on stderr and exit `1`. This is a daemon-
///   bug signal an operator wants to see, not silently coerce.
///
/// Centralises the Ok/Err/other dispatch the unit-ack verbs would
/// otherwise duplicate, so the operator-visible error surface stays
/// uniform across `disable` / `enable` / `reload`.
pub(crate) fn one_shot_unit(
    client: &ClientArgs,
    verb: &'static str,
    request: &WireRequest,
) -> ExitCode {
    let resp = match round_trip(client, verb, request) {
        Ok(r) => r,
        Err(code) => return code,
    };
    match resp {
        ResponsePayload::Ok => ExitCode::SUCCESS,
        ResponsePayload::Err { code, error } => {
            eprintln!("specter {verb}: {code}: {error}");
            ExitCode::from(1)
        }
        other => {
            eprintln!("specter {verb}: unexpected response: {other:?}");
            ExitCode::from(1)
        }
    }
}
