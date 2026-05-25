//! `specter show <name>` client handler.
//!
//! Distinct exit-code discipline: `Unknown` → `1` (typo / stale
//! name), `Active` / `Disabled` → `0` (the operator-declared name
//! resolved). Operators chain `specter show foo && do-thing` to gate
//! on existence in the operator's config.
//!
//! The exit-code derivation lives at the bottom of the response
//! match, after rendering — that keeps the `Unknown → 1` rule
//! single-source across `-o human` and `-o json`.

use compact_str::CompactString;
use specter_config::{OutputFormat, ShowArgs};
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::{ResponsePayload, ShowResponse, WireRequest};
use crate::ipc::render::show;

/// Run the `specter show` round-trip.
///
/// `name` is converted into [`CompactString`] at the call site so the
/// wire request reuses the same string shape the server's
/// [`crate::ipc::protocol::WireRequest::Show`] holds.
pub(crate) fn run(args: &ShowArgs) -> ExitCode {
    let req = WireRequest::Show {
        name: CompactString::from(args.name.as_str()),
    };
    let resp = match connect::round_trip(&args.client, "show", &req) {
        Ok(r) => r,
        Err(code) => return code,
    };

    match resp {
        ResponsePayload::Show(show) => render_show(args.output, &show),
        ResponsePayload::Err { code, error } => {
            eprintln!("specter show: {code}: {error}");
            ExitCode::from(1)
        }
        other => {
            eprintln!("specter show: unexpected response: {other:?}");
            ExitCode::from(1)
        }
    }
}

/// Render the [`ShowResponse`] and derive the exit code from its
/// arm. Lifting the derivation above the format match keeps the
/// rule single-source across `-o human` and `-o json`.
fn render_show(output: OutputFormat, show: &ShowResponse) -> ExitCode {
    match output {
        OutputFormat::Human => print!("{}", show::render(show)),
        OutputFormat::Json => {
            let s = serde_json::to_string(show).expect("ShowResponse always serializes");
            println!("{s}");
        }
    }
    match show {
        ShowResponse::Unknown { .. } => ExitCode::from(1),
        ShowResponse::Active(_) | ShowResponse::Disabled { .. } => ExitCode::SUCCESS,
    }
}
