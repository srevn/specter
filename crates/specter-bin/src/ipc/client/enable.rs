//! `specter enable <name>` client handler — clears a runtime disable override and (when the TOML
//! entry is active) drives a fresh attach of the static Sub.
//!
//! The override is cleared even on the [`crate::ipc::protocol::WireErrorCode::TomlDisabled`] path:
//! the operator's "no longer want this suppressed" intent is honoured regardless of whether the
//! daemon can immediately re-attach. See `EngineDriver::handle_enable` for the server-side ordering.
//!
//! Exit-code discipline matches the other unit-ack verbs.

use compact_str::CompactString;
use specter_config::NameTargetArgs;
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::WireRequest;

pub(crate) fn run(args: &NameTargetArgs) -> ExitCode {
    let req = WireRequest::Enable {
        name: CompactString::from(args.name.as_str()),
    };
    connect::one_shot_unit(&args.client, "enable", &req)
}
