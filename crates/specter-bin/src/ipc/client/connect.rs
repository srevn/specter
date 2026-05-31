//! Client-side connect + framing helpers.
//!
//! Single source of socket-timeout policy: every verb handler reaches
//! the daemon through [`dial`], which resolves the socket-path policy
//! ([`crate::ipc::resolve`]) and connects — pinning the one explicit
//! override, or probing the per-platform convention cascade and taking
//! the first path that answers. The chosen connection is armed with a
//! 5s read deadline and a 2s write deadline. A daemon that takes longer
//! than either to respond is operator-visibly hung; surfacing the
//! deadline is better than a silent indefinite park.
//!
//! # Framing
//!
//! Requests and responses are line-delimited JSON, one object per
//! line — the shared framing contract owned by
//! [`crate::ipc::framing::encode_line`]. [`write_request`]
//! serialises + appends LF + writes in one `write_all`.
//! [`read_response`] reads through a [`BufReader`] until a newline,
//! then strict-parses via [`crate::ipc::framing::parse_strict`].

use anstyle::Style;
use serde::Serialize;
use specter_config::{ClientArgs, OutputFormat};
use std::fmt;
use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::Path;
use std::process::ExitCode;
use std::time::Duration;

use crate::ipc::framing::{encode_line, parse_strict};
use crate::ipc::protocol::{ResponsePayload, WireRequest};
use crate::ipc::render::style::{self, Stream, Styler};
use crate::ipc::resolve::{self, Resolution, SocketCandidate};

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
fn open(socket: &Path) -> io::Result<UnixStream> {
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

/// Why the client could not reach the daemon. Mirrors the two
/// [`Resolution`] shapes so the operator message matches what they
/// asked for: an explicit override that didn't answer, versus a
/// convention cascade where no candidate did. Renders through the
/// shared [`crate::ipc::resolve::SocketSource`] vocab — the same
/// labels the daemon's `BindFailure` uses, so both sides name a path
/// the same way. Never escapes [`dial`], which folds it to an
/// [`ExitCode`] after rendering; module-private accordingly.
#[derive(Debug)]
enum ConnectFailure {
    /// An explicit `--socket` / `$SPECTER_SOCK` path that refused or was
    /// absent. There is no fall-through — a stale override is the
    /// operator's to fix, never silently retargeted.
    Pinned {
        candidate: SocketCandidate,
        error: io::Error,
    },
    /// Every convention candidate was probed and none answered. `tried`
    /// is non-empty (a [`Resolution::Cascade`] always carries a head),
    /// in probe order.
    Exhausted {
        tried: Vec<(SocketCandidate, io::Error)>,
    },
}

impl fmt::Display for ConnectFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Pinned { candidate, error } => write!(
                f,
                "cannot connect to {} ({}): {error}",
                candidate.path.display(),
                candidate.source.label(),
            ),
            // One indented `<path> (<label>): <errno>` line per probed
            // candidate, so the operator sees every path that was tried
            // and why each refused.
            Self::Exhausted { tried } => {
                f.write_str("cannot reach the daemon; tried:")?;
                for (candidate, error) in tried {
                    write!(
                        f,
                        "\n    {} ({}): {error}",
                        candidate.path.display(),
                        candidate.source.label(),
                    )?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for ConnectFailure {}

/// Turn a resolved [`Resolution`] into a live daemon connection.
///
/// The winning candidate *moves* into the returned stream's lifetime and
/// the losers move into the [`ConnectFailure::Exhausted`] `tried` list —
/// no path is cloned. Variant-directed:
///
/// - [`Resolution::Pinned`] — one [`open`] attempt; its failure is the
///   operator's own path not answering, never a fall-through.
/// - [`Resolution::Cascade`] — probe head-then-tail, returning the first
///   candidate whose [`open`] succeeds. Falls through on a connect
///   failure of *any* kind (refused, absent, `EACCES`) but never on
///   success: a live-but-hung session daemon wins the probe and later
///   trips the read deadline rather than being silently bypassed for a
///   different daemon. Connecting *is* the liveness test — a stale
///   socket whose daemon crashed refuses and the probe transparently
///   reaches the next candidate, with no probe-then-reconnect TOCTOU.
fn connect_daemon(resolution: Resolution) -> Result<UnixStream, ConnectFailure> {
    match resolution {
        Resolution::Pinned(candidate) => {
            open(&candidate.path).map_err(|error| ConnectFailure::Pinned { candidate, error })
        }
        Resolution::Cascade(candidates) => {
            let mut tried = Vec::new();
            for candidate in candidates {
                match open(&candidate.path) {
                    Ok(stream) => return Ok(stream),
                    Err(error) => tried.push((candidate, error)),
                }
            }
            Err(ConnectFailure::Exhausted { tried })
        }
    }
}

/// The single resolve-and-paint site for client-side stderr.
///
/// Stays content-agnostic: the caller forms the whole line with
/// `format_args!` — including any `specter <verb>:` prefix — so every
/// stderr diagnostic flows through one place without special-casing.
/// [`fmt::Arguments`] keeps the call zero-allocation. Resolves the
/// stderr [`Styler`] from the operator's `--color` choice + the stderr
/// TTY / environment gate, then paints the semantic `role` over the
/// whole line; under a plain Styler the painted adapter is a
/// byte-identical passthrough. The caller still owns the exit code.
///
/// Two roles ride this core — [`emit_error`] ([`style::ERR`]) for
/// operator-error text and [`emit_hint`] ([`style::SECONDARY`]) for a
/// help continuation that is not itself an error.
fn emit_stderr(client: &ClientArgs, role: Style, msg: fmt::Arguments<'_>) {
    let sty = style::resolve(client.color, Stream::Stderr);
    eprintln!("{}", sty.paint(role, msg));
}

/// Operator-error text on stderr — the `specter <verb>:` line, a
/// transport-stage failure, an unknown-name report. Painted
/// [`style::ERR`] (red).
pub(crate) fn emit_error(client: &ClientArgs, msg: fmt::Arguments<'_>) {
    emit_stderr(client, style::ERR, msg);
}

/// A stderr *hint* — a help continuation that follows an error line
/// but is not itself one (e.g. `tail`'s `Known filters: …` listing the
/// wire vocabulary). Painted [`style::SECONDARY`] (dimmed) so it reads
/// as guidance, not alarm, and stays visually distinct from the
/// [`style::ERR`] line it trails.
pub(crate) fn emit_hint(client: &ClientArgs, msg: fmt::Arguments<'_>) {
    emit_stderr(client, style::SECONDARY, msg);
}

/// Render a [`ResponsePayload`] a verb cannot use — the daemon's
/// structured [`ResponsePayload::Err`], or any variant other than the
/// one expected — and yield the failure exit code (`1`). Callers
/// `return fail_response(…)` directly.
///
/// The single source of the response-tail shape that `status`,
/// `list`, `show`, [`one_shot_unit`], and `subscribe`'s ack
/// validation would otherwise each restate:
///
/// - [`ResponsePayload::Err`] → `specter <verb>: <code>: <error>`.
///   The closed-set `code` renders its stable wire token (scripts
///   branch on it); `error` is the human amplification.
/// - any other variant → `specter <verb>: unexpected response:
///   <debug>` — a daemon-bug signal surfaced loudly, not coerced.
///
/// `verb` is `&'static str` so a borrowed runtime string cannot leak
/// into the prefix. `client` resolves the stderr [`Styler`](style::Styler):
/// the `code` paints [`style::ERR_CODE`] (bold) so it stands out from
/// the surrounding [`style::ERR`] amplification the operator scripts
/// against; the three painted spans are siblings, never nested. Under a
/// plain Styler the line is byte-identical to the pre-color form.
#[must_use]
pub(crate) fn fail_response(
    client: &ClientArgs,
    verb: &'static str,
    resp: ResponsePayload,
) -> ExitCode {
    let sty = style::resolve(client.color, Stream::Stderr);
    match resp {
        ResponsePayload::Err { code, error } => {
            eprintln!(
                "{}{}{}",
                sty.paint(style::ERR, format_args!("specter {verb}: ")),
                sty.paint(style::ERR_CODE, code),
                sty.paint(style::ERR, format_args!(": {error}")),
            );
        }
        other => {
            eprintln!(
                "{}",
                sty.paint(
                    style::ERR,
                    format_args!("specter {verb}: unexpected response: {other:?}"),
                ),
            );
        }
    }
    ExitCode::from(1)
}

/// The single resolve-and-connect seam both client verb families
/// route through — the one-shot family ([`round_trip`]) and the
/// streaming family ([`crate::ipc::client::subscribe::open`]).
///
/// Resolve the socket-path policy ([`resolve::resolve`], reading
/// `--socket` then `$SPECTER_SOCK` then the per-platform convention
/// through the one [`resolve::env_os`] touchpoint the daemon shares),
/// then [`connect_daemon`] it (pinning the override or probing the
/// cascade, applying the read/write deadlines). Both stages render
/// their own diagnostic through [`emit_error`] under one
/// `specter <verb>: ` prefix — a [`resolve::ResolveError`] (a relative
/// or over-long override) or a [`ConnectFailure`] (the override / every
/// cascade candidate refusing) — before yielding the failure exit code
/// (`1`). The error types own their messages, so no per-stage render
/// helper is needed.
///
/// Centralising the connect keeps that surface single-source, so
/// neither verb family can grow a divergent connect path. `verb` is
/// `&'static str` so a borrowed runtime string cannot leak into the
/// prefix. Callers `return code` directly on `Err`.
pub(crate) fn dial(client: &ClientArgs, verb: &'static str) -> Result<UnixStream, ExitCode> {
    let resolution = resolve::resolve(client.socket.as_deref(), resolve::env_os).map_err(|e| {
        emit_error(client, format_args!("specter {verb}: {e}"));
        ExitCode::from(1)
    })?;
    connect_daemon(resolution).map_err(|fail| {
        emit_error(client, format_args!("specter {verb}: {fail}"));
        ExitCode::from(1)
    })
}

/// Stereotyped one-shot round trip — [`dial`] the daemon, write the
/// request, read the response.
///
/// Every one-shot verb's network-side work has this shape; centralising
/// it keeps the operator-visible error surface uniform
/// (`specter <verb>: <stage>: <io::Error>`) and the per-stage prefixes
/// single-source. The connect stage rides the shared [`dial`] seam; the
/// send/receive stages hand-roll their `io::Error`→operator-prefix
/// mapping because each path needs a distinct prefix — a single
/// template doesn't fit.
///
/// `verb` is the operator-facing command name (`"status"`, `"list"`,
/// …) and is constrained to `&'static str` so callers cannot
/// accidentally pass a borrowed runtime string.
///
/// Returns `Ok(response)` on a successful round trip; `Err(code)`
/// after reporting the cause through [`emit_error`] — callers
/// `return code` directly.
pub(crate) fn round_trip(
    client: &ClientArgs,
    verb: &'static str,
    request: &WireRequest,
) -> Result<ResponsePayload, ExitCode> {
    let mut stream = dial(client, verb)?;
    write_request(&mut stream, request).map_err(|e| {
        emit_error(client, format_args!("specter {verb}: send failed: {e}"));
        ExitCode::from(1)
    })?;
    read_response(&mut stream).map_err(|e| {
        emit_error(client, format_args!("specter {verb}: receive failed: {e}"));
        ExitCode::from(1)
    })
}

/// One-shot round-trip for a verb whose successful response is the
/// unit-shaped [`ResponsePayload::Ok`] — `disable`, `enable`,
/// `reload`, `absorb`, and any future verb whose ack carries no
/// payload.
///
/// [`ResponsePayload::Ok`] is exit `0`; every other variant — the
/// structured [`ResponsePayload::Err`] or an unexpected shape —
/// routes through [`fail_response`], which renders the tail and
/// yields exit `1`.
///
/// Centralises the Ok/non-Ok dispatch the unit-ack verbs would
/// otherwise duplicate, so the operator-visible error surface stays
/// uniform across `disable` / `enable` / `reload` / `absorb`.
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
        other => fail_response(client, verb, other),
    }
}

/// Render `value` to stdout in the operator's chosen format — the
/// single source of the Human/Json + Styler-resolution + buffered-write
/// triad every data verb (`status` / `list` / `show`) shares.
///
/// `Human` resolves the stdout [`Styler`] once (the `--color` choice +
/// the stdout TTY / environment gate) and threads it into `render` — a
/// pure writer that appends into a fresh buffer — then writes the
/// buffer to a locked stdout in one pass. `Json` re-serialises `value`
/// through its wire carrier so the bytes match the daemon's own
/// emission, and never consults the Styler.
///
/// Returns `Ok(())` once the response is delivered — *or* once a
/// downstream consumer closes the pipe (`head -1`, `grep -q`, …): a
/// broken pipe is the operator's reader stopping, not a daemon fault,
/// so it is the same graceful success the streaming verbs (`tail` /
/// `wait`) already grant. Any other write failure reports through
/// [`emit_error`] (`specter <verb>: write failed: …`) and yields
/// `Err(exit 1)`. The caller owns the *success* exit code (`show`
/// derives `Unknown → 1` from the response arm), so this never
/// fabricates one.
///
/// Locking stdout once and writing through [`Write::write_all`] +
/// [`Write::flush`] is deliberate: `print!` / `println!` panic on a
/// broken pipe (Rust ignores `SIGPIPE`), whereas the explicit
/// write/flush surfaces it as an `io::Error` this maps to a clean exit.
pub(crate) fn emit_human_or_json<T: Serialize>(
    client: &ClientArgs,
    verb: &'static str,
    output: OutputFormat,
    value: &T,
    render: impl FnOnce(&mut String, &T, Styler),
) -> Result<(), ExitCode> {
    let mut out = io::stdout().lock();
    let written = match output {
        OutputFormat::Human => {
            let sty = style::resolve(client.color, Stream::Stdout);
            let mut buf = String::new();
            render(&mut buf, value, sty);
            out.write_all(buf.as_bytes())
        }
        OutputFormat::Json => {
            let json = serde_json::to_string(value).expect("response always serializes");
            out.write_all(json.as_bytes())
                .and_then(|()| out.write_all(b"\n"))
        }
    };
    match written.and_then(|()| out.flush()) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::BrokenPipe => Ok(()),
        Err(e) => {
            emit_error(client, format_args!("specter {verb}: write failed: {e}"));
            Err(ExitCode::from(1))
        }
    }
}

/// [`emit_human_or_json`] for a verb whose successful render is always
/// exit `0` — `status` and `list`. Collapses the `Ok → SUCCESS,
/// Err → code` tail those two would otherwise restate; the broken-pipe
/// and write-failure policy already lives in the core.
///
/// `show` does not use this wrapper: its exit code derives from the
/// response arm (`Unknown → 1`), so it calls [`emit_human_or_json`]
/// directly and falls through to the arm match after a delivered (or
/// pipe-closed) render. Mirrors [`one_shot_unit`]'s relationship to
/// [`round_trip`].
#[must_use]
pub(crate) fn render_response<T: Serialize>(
    client: &ClientArgs,
    verb: &'static str,
    output: OutputFormat,
    value: &T,
    render: impl FnOnce(&mut String, &T, Styler),
) -> ExitCode {
    match emit_human_or_json(client, verb, output, value, render) {
        Ok(()) => ExitCode::SUCCESS,
        Err(code) => code,
    }
}
