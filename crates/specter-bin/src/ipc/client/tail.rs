//! `specter tail` client handler.
//!
//! Subscribes unfiltered, optionally restricts the stream to a set
//! of `--filter <variant>` tags (validated against
//! [`KNOWN_WIRE_VARIANTS`] at handler entry), and emits each surviving
//! event through [`diag`] (default `-o human`) or the wire's
//! own JSON line (`-o json`).
//!
//! # Exit codes
//!
//! - `0` — graceful EOF (the daemon closed the per-conn socket on
//!   shutdown ⇒ the next read returns EOF) or a downstream pipe
//!   consumer closed (`BrokenPipe`).
//! - `1` — connect / subscribe / read I/O failure (a parse failure
//!   on one streamed line is logged to stderr but the loop
//!   continues).
//! - `2` — `--filter <unknown>`: the operator's flag carries a tag
//!   that is not part of the wire vocabulary. Matches clap's
//!   "argument error" exit-code shape.
//!
//! # Filter semantics
//!
//! `--filter` is OR across variants — `--filter SubFired --filter
//! SubDetached` admits either one. Empty filter (the default) admits
//! every variant. Filter validation lives in the handler, not in
//! clap's `value_parser`, because the vocabulary is owned by the
//! wire crate ([`KNOWN_WIRE_VARIANTS`]) and `specter-config` should
//! not depend on `specter-bin`'s wire shape. The handler's
//! `eprintln!` shape ("specter tail: unknown filter …") matches the
//! other verbs' inline error messages.
//!
//! # Output buffering
//!
//! Stdout is locked once for the loop's duration so per-line writes
//! reach the kernel pipe immediately on each `flush()`. Operators
//! piping to `jq` / `grep` see one line at a time without depending
//! on Rust's per-process stdout buffering policy (which is
//! block-buffered when piped, line-buffered on a tty).

use specter_config::{OutputFormat, TailArgs};
use std::io::{self, Write};
use std::process::ExitCode;

use crate::ipc::client::subscribe;
use crate::ipc::framing::encode_line;
use crate::ipc::render::diag;
use crate::ipc::wire::{KNOWN_WIRE_VARIANTS, WireDiagnostic};

/// Run the `specter tail` stream loop.
pub(crate) fn run(args: &TailArgs) -> ExitCode {
    if let Err(code) = validate_filter(&args.filter) {
        return code;
    }

    let mut sub = match subscribe::open(&args.client, "tail", None) {
        Ok(s) => s,
        Err(code) => return code,
    };

    // Indefinite tail: clear the connect-time 5s deadline so the
    // read blocks until the next event arrives (or EOF when the
    // daemon closes the conn on shutdown).
    if let Err(e) = sub.set_read_timeout(None) {
        eprintln!("specter tail: clear read deadline failed: {e}");
        return ExitCode::from(1);
    }

    let mut stdout = io::stdout().lock();
    // One render buffer reused for the lifetime of the stream loop —
    // amortizes the per-event allocation the previous owned-String
    // `render` shape paid. Symmetric with
    // [`crate::ipc::client::subscribe::Subscription`]'s reused
    // inbound `line_buf`. The 256-byte initial capacity covers the
    // common diag line (~120 bytes) without growing on the first hit.
    let mut buf = String::with_capacity(256);
    loop {
        match sub.read_next() {
            Ok(Some(wire)) => {
                if !should_emit(&wire, &args.filter) {
                    continue;
                }
                if let Err(e) = emit(&mut stdout, &wire, args.output, &mut buf) {
                    // Downstream pipe consumer closed (`BrokenPipe`)
                    // or any other stdout failure: graceful exit.
                    // The daemon's stream is healthy; the operator's
                    // consumer (`head -1`, etc.) just stopped reading.
                    if e.kind() == io::ErrorKind::BrokenPipe {
                        return ExitCode::SUCCESS;
                    }
                    eprintln!("specter tail: write failed: {e}");
                    return ExitCode::from(1);
                }
            }
            Ok(None) => return ExitCode::SUCCESS,
            Err(e) if e.kind() == io::ErrorKind::InvalidData => {
                eprintln!("specter tail: malformed diagnostic line: {e}");
            }
            Err(e) => {
                eprintln!("specter tail: read failed: {e}");
                return ExitCode::from(1);
            }
        }
    }
}

/// Validate every `--filter <tag>` against [`KNOWN_WIRE_VARIANTS`].
/// On the first unknown tag, prints the operator-visible suggestion
/// list and returns exit code `2` (matches clap's argument-error
/// shape). Returns `Ok(())` on an empty or fully-valid filter list.
fn validate_filter(filter: &[String]) -> Result<(), ExitCode> {
    if let Some(bad) = filter
        .iter()
        .find(|f| !KNOWN_WIRE_VARIANTS.contains(&f.as_str()))
    {
        eprintln!("specter tail: unknown filter '{bad}'");
        eprintln!("Known filters: {}", KNOWN_WIRE_VARIANTS.join(", "));
        return Err(ExitCode::from(2));
    }
    Ok(())
}

/// Filter predicate — `true` iff `wire` should reach stdout.
///
/// Empty filter ⇒ admit every variant. Non-empty filter ⇒ admit
/// only when `wire.variant_name()` matches one of the entries (OR
/// across the list). Pure function so the filter rule pins as a
/// unit test independent of the surrounding I/O.
fn should_emit(wire: &WireDiagnostic, filter: &[String]) -> bool {
    filter.is_empty() || filter.iter().any(|f| f == wire.variant_name())
}

/// Write one event to `out` and flush. Returns the underlying
/// `io::Error` on any failure so the caller can distinguish
/// `BrokenPipe` (graceful exit) from genuine write failures.
///
/// `buf` is the caller's reused render buffer — the Human branch
/// clears and refills it through [`diag::render`]'s writer-shape
/// surface; the Json branch is untouched and goes through
/// [`encode_line`]'s owned [`Vec<u8>`] path. Threading the buffer
/// keeps the per-event amortization legible at the call site without
/// inlining the I/O error matching into the stream loop.
///
/// JSON output re-serializes the parsed [`WireDiagnostic`] rather
/// than passing through the daemon's original bytes. Serde derive is
/// symmetric over the wire's `#[serde]` tags, so the re-serialized
/// line is byte-equivalent to the daemon's emission for every
/// variant — the witness-fixture round-trip test
/// (`wire_diagnostic_round_trips_via_serde`) pins this contract.
/// [`encode_line`]'s
/// [`crate::ipc::framing::InfallibleSerialize`] bound asserts the
/// re-emit's `Vec<u8>`-build cannot fail (audited at the marker
/// impl in [`crate::ipc::wire`]).
fn emit<W: Write>(
    out: &mut W,
    wire: &WireDiagnostic,
    output: OutputFormat,
    buf: &mut String,
) -> io::Result<()> {
    match output {
        OutputFormat::Human => {
            buf.clear();
            diag::render(buf, wire);
            out.write_all(buf.as_bytes())?;
        }
        OutputFormat::Json => {
            out.write_all(&encode_line(wire))?;
        }
    }
    out.flush()
}

#[cfg(test)]
mod tests {
    use super::{should_emit, validate_filter};
    use crate::ipc::protocol::WireId;
    use crate::ipc::wire::{WireDiagnostic, WireTime};
    use std::time::UNIX_EPOCH;

    fn sub_fired() -> WireDiagnostic {
        WireDiagnostic::SubFired {
            at: WireTime::from(UNIX_EPOCH),
            sub: WireId(1),
            profile: WireId(2),
            count: 1,
        }
    }

    fn missed() -> WireDiagnostic {
        WireDiagnostic::Missed {
            at: WireTime::from(UNIX_EPOCH),
            count: 3,
        }
    }

    /// Empty filter admits every variant — the default `tail`
    /// behaviour.
    #[test]
    fn should_emit_empty_filter_admits_everything() {
        let filter: Vec<String> = vec![];
        assert!(should_emit(&sub_fired(), &filter));
        assert!(should_emit(&missed(), &filter));
    }

    /// Filter matching the variant's tag admits the event.
    #[test]
    fn should_emit_matching_filter_admits_the_variant() {
        let filter = vec!["SubFired".to_string()];
        assert!(should_emit(&sub_fired(), &filter));
    }

    /// Filter naming a different variant rejects this one.
    /// Mirrors the operator's `--filter SubDetached` against an
    /// incoming `SubFired` line.
    #[test]
    fn should_emit_non_matching_filter_rejects() {
        let filter = vec!["SubDetached".to_string()];
        assert!(!should_emit(&sub_fired(), &filter));
    }

    /// Multiple `--filter` entries are OR — either matches admits
    /// the event. Catches a future bug that re-interpreted the list
    /// as AND.
    #[test]
    fn should_emit_or_across_multiple_filters() {
        let filter = vec!["SubFired".to_string(), "SubDetached".to_string()];
        assert!(should_emit(&sub_fired(), &filter));
        assert!(!should_emit(&missed(), &filter));
    }

    /// The back-pressure marker uses the `_missed` tag — operators
    /// filtering for it use the underscore-prefixed name (the only
    /// `#[serde(rename)]` override on the enum).
    #[test]
    fn should_emit_missed_marker_matches_underscore_tag() {
        let filter = vec!["_missed".to_string()];
        assert!(should_emit(&missed(), &filter));
        assert!(!should_emit(&sub_fired(), &filter));
    }

    /// `validate_filter` accepts an empty list and any list of
    /// valid wire-vocabulary tags. Catches a future regression that
    /// rejected the empty default.
    #[test]
    fn validate_filter_accepts_known_tags() {
        assert!(validate_filter(&[]).is_ok());
        assert!(validate_filter(&["SubFired".to_string()]).is_ok());
        assert!(
            validate_filter(&["SubFired".to_string(), "_missed".to_string()]).is_ok(),
            "the underscore-prefixed back-pressure marker is in the vocabulary",
        );
    }

    /// `validate_filter` rejects on the first unknown tag — exit
    /// code `2`, matches clap's argument-error shape.
    #[test]
    fn validate_filter_rejects_unknown_tag() {
        let bad = vec!["NotAVariant".to_string()];
        let err = validate_filter(&bad).expect_err("must reject unknown tag");
        // ExitCode does not derive PartialEq; compare its Debug shape
        // (the only operator-visible discriminator).
        assert_eq!(
            format!("{err:?}"),
            format!("{:?}", std::process::ExitCode::from(2))
        );
    }
}
