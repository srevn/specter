//! `specter reload` client handler — operator-requested config reload over IPC. Equivalent in
//! effect to a SIGHUP, but with the per-operator attribution surfaced via
//! [`crate::driver::ReloadTrigger::Ipc`].
//!
//! The ack semantics on the server are "pulse accepted", not "reload succeeded": a parse failure on
//! the daemon logs at `error!` and retains the running config, but still returns
//! [`crate::ipc::protocol::ResponsePayload::Ok`] here. This matches the SIGHUP / auto-reload
//! contract — one apply path, one ack shape.
//!
//! Exit-code discipline matches the other unit-ack verbs.

use specter_config::ClientArgs;
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::WireRequest;

pub(crate) fn run(args: &ClientArgs) -> ExitCode {
    connect::one_shot_unit(args, "reload", &WireRequest::Reload)
}
