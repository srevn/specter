//! Operator IPC scaffold — request/response shapes, diagnostic
//! projection, socket lifecycle.
//!
//! This module is the bin's fourth stable seam, alongside engine
//! state, the reload pipeline, and diagnostic fan-out. It owns every
//! type that escapes the daemon as a JSON line on the operator
//! socket; nothing in `specter-core` or the actor crates carries a
//! `serde` derive on its behalf.
//!
//! # Submodules
//!
//! - [`protocol`] — request layering (`WireRequest` →
//!   `RequestPayload` → `IpcRequest`), response carriers
//!   (`ResponsePayload`, `StatusResponse`, `ListResponse`,
//!   `ShowResponse`), the `WireId` newtype, and `ERR_*` constants.
//! - [`wire`] — `WireDiagnostic` (the exhaustive
//!   projection of every `specter_core::Diagnostic` variant),
//!   `BrokerEvent`, `WireTime`, and the per-core-type `Wire*`
//!   projections every variant transitively reaches.
//! - [`sockpath`] — UNIX-socket path resolution, atomic-rename bind
//!   with 0600 permissions, stale-socket recovery, and the
//!   drop-guard that unlinks the socket on graceful shutdown or
//!   panic.
//! - [`project`] — pure projection of engine + driver state into
//!   the response carriers. No I/O, no engine mutation.
//! - [`server`] — accept loop + per-connection handler. Owns the
//!   bound listener for the daemon's lifetime; spawns one
//!   short-lived worker thread per connection.
//! - [`client`] — operator-facing client verbs.
//! - [`render`] — human-readable rendering for `-o human` output.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. Nothing here is `pub` — operator
//! clients ship inside the same binary, so the wire surface is an
//! implementation detail of the bin, not a published library
//! interface.

pub(crate) mod client;
pub(crate) mod project;
pub(crate) mod protocol;
pub(crate) mod render;
pub(crate) mod server;
pub(crate) mod sockpath;
pub(crate) mod wire;
