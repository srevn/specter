//! Operator duration grammar — the one vocabulary for every duration Specter reads (TOML `settle` /
//! `max_settle` / `timeout`, CLI `--for` / `--timeout`) and every duration it renders back
//! (validation messages, the IPC program projection).
//!
//! The grammar is jiff's "friendly" format over absolute units: `500ms`, `30s`, `1m30s`, `1m 30s`,
//! `1.5s`, `2h`, `2 hours 30 minutes`. Calendar units (`d`, `w`, `mo`, `y`) are rejected: every
//! Specter duration is a fixed elapsed-time window, and a calendar day has no fixed length, so
//! `24h` says what is meant. The printer never emits a unit above `h`, so anything Specter renders
//! parses back to the same value.

use jiff::fmt::StdFmtWrite;
use jiff::fmt::friendly::{SpanParser, SpanPrinter};
use std::fmt;
use std::time::Duration;

static PARSER: SpanParser = SpanParser::new();
static PRINTER: SpanPrinter = SpanPrinter::new();

/// Parse one duration token. The error's `Display` is operator-facing — clap and the TOML layer
/// both surface it verbatim.
pub(crate) fn parse(s: &str) -> Result<Duration, jiff::Error> {
    PARSER.parse_unsigned_duration(s)
}

/// Renders a [`Duration`] in the grammar [`parse`] accepts (`5s`, `1m 30s`, `1s 500ms`), writing
/// straight into the caller's formatter rather than through an intermediate `String`.
#[derive(Clone, Copy, Debug)]
pub struct DisplayDuration(pub Duration);

impl fmt::Display for DisplayDuration {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        PRINTER
            .print_unsigned_duration(&self.0, StdFmtWrite(f))
            .map_err(|_| fmt::Error)
    }
}
