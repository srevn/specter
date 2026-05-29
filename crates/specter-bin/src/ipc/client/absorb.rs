//! `specter absorb <name> [--for <dur>]` client handler — arm a
//! fold-without-fire window on the named watch's Profile. The next
//! fireable burst (or an in-flight one) advances the baseline silently
//! instead of firing, folding an expected replication rather than
//! echoing it.
//!
//! Exit-code discipline matches the other unit-ack verbs (`disable` /
//! `enable` / `reload`): `0` on `Ok`, `1` on any structured failure
//! (connect / send / receive / daemon-side `Err`). The structured
//! `code:` prefix on stderr (e.g. `unknown_sub`, `dynamic_sub_no_op`,
//! `shutting_down`) lets operator scripts distinguish failure modes
//! without a per-mode exit code.

use compact_str::CompactString;
use specter_config::AbsorbArgs;
use std::process::ExitCode;

use crate::ipc::client::connect;
use crate::ipc::protocol::WireRequest;

pub(crate) fn run(args: &AbsorbArgs) -> ExitCode {
    let req = WireRequest::Absorb {
        name: CompactString::from(args.name.as_str()),
        // Saturate the u128 → u64 millisecond projection rather than
        // truncate with `as`: an absurd `--for` (centuries) pins to
        // `u64::MAX` ms, which the driver then clamps to its window
        // ceiling. The wire stays a scalar; no silent wrap reaches it.
        duration_ms: args
            .for_
            .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
    };
    connect::one_shot_unit(&args.client, "absorb", &req)
}
