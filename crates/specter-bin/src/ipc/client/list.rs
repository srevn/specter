//! `specter list` client handler.
//!
//! Same shape as [`super::status::run`]: round-trip through [`super::connect::round_trip`],
//! dispatch on the output format, render through [`crate::ipc::render::list`] or emit the
//! deserialised JSON verbatim.
//!
//! Exit codes match the rest of the client surface: `0` success, `1` connect / protocol /
//! unexpected-response failure.

use specter_config::ListArgs;
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::{ResponsePayload, WireRequest};
use crate::ipc::render::list;

/// Run the `specter list` round-trip.
///
/// `-o json` re-serialises the deserialised carrier so the JSON on stdout matches the daemon's
/// emission byte-for-byte (the wire shape is the single canonical form). `-o human` defers column
/// gating to [`list::render`] — the server's response is lossless regardless of `args.wide`.
pub(crate) fn run(args: &ListArgs) -> ExitCode {
    let resp = match connect::round_trip(&args.client, "list", &WireRequest::List) {
        Ok(r) => r,
        Err(code) => return code,
    };

    let ResponsePayload::List(list) = resp else {
        return connect::fail_response(&args.client, "list", resp);
    };

    connect::render_response(
        &args.client,
        "list",
        args.output,
        &list,
        |buf, resp, sty| {
            list::render(buf, resp, args.wide, sty);
        },
    )
}
