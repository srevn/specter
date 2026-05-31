//! `specter status` client handler.
//!
//! Resolves the daemon's socket path (CLI override or per-platform
//! default), connects, ships a [`WireRequest::Status`], parses the
//! [`ResponsePayload::Status`], and dispatches to
//! [`crate::ipc::render::status`] (default `-o human`) or
//! serialises the response verbatim (`-o json`).
//!
//! Exit codes mirror the daemon convention: `0` success, `1`
//! connect/protocol failure, `2` reserved for the stub (unreachable
//! once this verb is dispatched here).

use specter_config::{OutputFormat, StatusArgs};
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::{ResponsePayload, WireRequest};
use crate::ipc::render::status;

/// Run the `specter status` round-trip.
///
/// Operator behaviour:
///
/// - Connect failure prints `cannot connect to <path>: <error>` on
///   stderr and exits `1`. Common causes are surfaced verbatim by
///   the underlying `io::Error` (file not found, permission denied,
///   connection refused).
/// - Send / receive failure prints `send failed: <error>` or
///   `receive failed: <error>` and exits `1`.
/// - A daemon-side error response (`ResponsePayload::Err`) prints
///   `<code>: <error>` and exits `1` — the structured code is
///   surfaced for scripting.
/// - An unexpected response shape (anything other than Status / Err)
///   exits `1` with a diagnostic; this is a daemon-bug signal that
///   an operator wants to see immediately, not silently coerce.
pub(crate) fn run(args: &StatusArgs) -> ExitCode {
    let resp = match connect::round_trip(&args.client, "status", &WireRequest::Status) {
        Ok(r) => r,
        Err(code) => return code,
    };

    let ResponsePayload::Status(status) = resp else {
        return connect::fail_response(&args.client, "status", resp);
    };

    match args.output {
        OutputFormat::Human => {
            print!("{}", status::render(&status, args.wide));
            ExitCode::SUCCESS
        }
        OutputFormat::Json => {
            // Re-serialise through the wire-side carrier so the JSON
            // shape matches the daemon's emission byte-for-byte.
            // `expect` is safe: every field on `StatusResponse` is
            // Serialize.
            let s = serde_json::to_string(&status).expect("StatusResponse always serializes");
            println!("{s}");
            ExitCode::SUCCESS
        }
    }
}
