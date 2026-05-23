//! Operator-facing IPC clients — connect to the daemon over the
//! UNIX socket, ship one [`crate::ipc::protocol::WireRequest`], parse
//! the [`crate::ipc::protocol::ResponsePayload`] (or stream
//! [`crate::ipc::wire::WireDiagnostic`]s, for the subscribe-arm
//! verbs), and render through [`crate::ipc::render`].
//!
//! # Submodules
//!
//! - [`connect`] — socket open + line framing helpers (`open`,
//!   `write_request`, `read_response`, `resolve_socket`,
//!   `round_trip`, `one_shot_unit`). Single source of timeout,
//!   connect-prefix policy, and the Ok/Err/other dispatch the
//!   unit-ack verbs share.
//! - [`status`] — `specter status` round-trip.
//! - [`list`] — `specter list` round-trip.
//! - [`show`] — `specter show <name>` round-trip.
//! - [`disable`] — `specter disable <name>` round-trip.
//! - [`enable`] — `specter enable <name>` round-trip.
//! - [`reload`] — `specter reload` round-trip.
//! - [`subscribe`] — shared Subscribe + ack + line-read scaffold for
//!   the streaming verbs.
//! - [`tail`] — `specter tail` (indefinite stream).
//! - [`wait`] — `specter wait <name>` (block-until-event).
//!
//! Every one-shot handler follows the same shape:
//! `connect::round_trip(verb, request)` → match the response →
//! render. The unit-ack verbs (`disable`, `enable`, `reload`)
//! collapse the match arm into [`connect::one_shot_unit`]. The two
//! streaming handlers reach [`subscribe::Subscription`] instead of
//! `round_trip` because the post-ack horizon is per-event, not
//! per-call.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. The verb entry points (`status::run`,
//! etc.) are reached by [`crate::run`] via the top-level subcommand
//! dispatcher in `lib.rs`.

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
