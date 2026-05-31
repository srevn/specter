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
use specter_config::{ClientArgs, OutputFormat, ShowArgs};
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

    let ResponsePayload::Show(show) = resp else {
        return connect::fail_response(&args.client, "show", resp);
    };
    render_show(args.output, &show, &args.client)
}

/// Render the [`ShowResponse`] and derive the exit code from its
/// arm. Lifting the derivation above the format match keeps the
/// rule single-source across `-o human` and `-o json`. The stdout
/// [`Styler`](style::Styler) resolves only on the `-o human` path —
/// `-o json` bypasses color entirely.
fn render_show(output: OutputFormat, show: &ShowResponse, client: &ClientArgs) -> ExitCode {
    if let Err(code) = connect::emit_human_or_json(client, "show", output, show, show::render) {
        return code;
    }
    // Exit code derives from the response arm, not the output format —
    // a delivered (or pipe-closed) render falls through here — so the
    // `Unknown → 1` rule stays single-source across `-o human` and
    // `-o json`.
    match show {
        ShowResponse::Unknown { .. } => ExitCode::from(1),
        ShowResponse::Active(_) | ShowResponse::Disabled { .. } => ExitCode::SUCCESS,
    }
}
