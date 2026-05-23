//! `specter list` client handler.
//!
//! Same shape as [`super::status::run`]: round-trip through
//! [`super::connect::round_trip`], dispatch on the output format,
//! render through [`crate::ipc::render::list_table`] or emit the
//! deserialised JSON verbatim.
//!
//! Exit codes match the rest of the client surface: `0` success,
//! `1` connect / protocol / unexpected-response failure.

use specter_config::{ListArgs, OutputFormat};
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::{ResponsePayload, WireRequest};
use crate::ipc::render::list_table;

/// Run the `specter list` round-trip.
///
/// `-o json` re-serialises the deserialised carrier so the JSON on
/// stdout matches the daemon's emission byte-for-byte (the wire
/// shape is the single canonical form). `-o human` defers column
/// gating to [`list_table::render`] — the server's response is
/// lossless regardless of `args.wide`.
pub(crate) fn run(args: &ListArgs) -> ExitCode {
    let resp = match connect::round_trip(&args.client, "list", &WireRequest::List) {
        Ok(r) => r,
        Err(code) => return code,
    };

    match resp {
        ResponsePayload::List(list) => match args.output {
            OutputFormat::Human => {
                print!("{}", list_table::render(&list, args.wide));
                ExitCode::SUCCESS
            }
            OutputFormat::Json => {
                let s = serde_json::to_string(&list).expect("ListResponse always serializes");
                println!("{s}");
                ExitCode::SUCCESS
            }
        },
        ResponsePayload::Err { code, error } => {
            eprintln!("specter list: {code}: {error}");
            ExitCode::from(1)
        }
        other => {
            eprintln!("specter list: unexpected response: {other:?}");
            ExitCode::from(1)
        }
    }
}
