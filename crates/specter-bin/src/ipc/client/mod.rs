//! Operator-facing IPC clients — connect to the daemon over the UNIX socket, ship one
//! [`crate::ipc::protocol::WireRequest`], parse the [`crate::ipc::protocol::ResponsePayload`] (or
//! stream [`crate::ipc::wire::WireDiagnostic`]s, for the subscribe-arm verbs), and render through
//! [`crate::ipc::render`].
//!
//! # Submodules
//!
//! - [`connect`] — the socket connect seam (`dial`) + line framing helpers (`write_request`,
//!   `read_response`, `round_trip`, `one_shot_unit`), the stdout render dispatch (`render_response`
//!   / `emit_human_or_json` — the Human/Json + Styler-resolve + buffered-write triad the data verbs
//!   share), and the client stderr vocabulary (`emit_error` / `emit_hint` / `fail_response`).
//!   Single source of timeout, the `specter <verb>:` prefix, the Ok/non-Ok dispatch the unit-ack
//!   verbs share, and the response-tail rendering every data verb reuses.
//! - [`status`] — `specter status` round-trip.
//! - [`list`] — `specter list` round-trip.
//! - [`show`] — `specter show <name>` round-trip.
//! - [`disable`] — `specter disable <name>` round-trip.
//! - [`enable`] — `specter enable <name>` round-trip.
//! - [`absorb`] — `specter absorb <name> [--for <dur>]` round-trip.
//! - [`reload`] — `specter reload` round-trip.
//! - [`subscribe`] — shared Subscribe + ack + line-read scaffold for the streaming verbs.
//! - [`tail`] — `specter tail` (indefinite stream).
//! - [`wait`] — `specter wait <name>` (block-until-event).
//!
//! Every one-shot handler follows the same shape: `connect::round_trip(verb, request)` → match the
//! response → emit. The data verbs (`status`, `list`, `show`) collapse the human/JSON emission into
//! [`connect::emit_human_or_json`] (`status` / `list` via the exit-0 [`connect::render_response`]
//! wrapper); the unit-ack verbs (`disable`, `enable`, `reload`, `absorb`) collapse the Ok/non-Ok arm
//! into [`connect::one_shot_unit`]. The two streaming handlers reach [`subscribe::Subscription`]
//! instead of `round_trip` because the post-ack horizon is per-event, not per-call.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. The verb entry points (`status::run`, etc.) are reached by
//! [`crate::run`] via the top-level subcommand dispatcher in `lib.rs`.

pub(crate) mod absorb;
pub(crate) mod connect;
pub(crate) mod disable;
pub(crate) mod enable;
pub(crate) mod list;
pub(crate) mod reload;
pub(crate) mod show;
pub(crate) mod status;
pub(crate) mod subscribe;
pub(crate) mod tail;
pub(crate) mod wait;
