//! `specter disable <name>` client handler — runtime override that detaches a static Sub by name.
//! The override survives until cleared by `enable`, or until the operator removes the `[[watch]]`
//! entry and reloads (a subsequent reload's prune drops the override).
//!
//! Exit-code discipline matches the other unit-ack verbs (`enable` / `reload`): `0` on `Ok`, `1` on
//! any structured failure (connect / send / receive / daemon-side `Err`). The structured `code:`
//! prefix on stderr (e.g. `unknown_sub`, `dynamic_sub_no_op`, `not_disabled`) lets operator scripts
//! distinguish failure modes without a per-mode exit code — the wire-stable codes carry the
//! categorisation.

use compact_str::CompactString;
use specter_config::NameTargetArgs;
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::WireRequest;

pub(crate) fn run(args: &NameTargetArgs) -> ExitCode {
    let req = WireRequest::Disable {
        name: CompactString::from(args.name.as_str()),
    };
    connect::one_shot_unit(&args.client, "disable", &req)
}
