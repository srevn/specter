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
//! - [`protocol`] — wire-side request shape (`WireRequest`), response
//!   carriers (`ResponsePayload`, `StatusResponse`, `ListResponse`,
//!   `ShowResponse`), the `WireId` newtype, and `ERR_*` constants.
//! - [`wire`] — `WireDiagnostic` (the exhaustive
//!   projection of every `specter_core::Diagnostic` variant),
//!   `WireTime`, and the per-core-type `Wire*` projections every
//!   variant transitively reaches.
//! - [`framing`] — `serialize_line<T>`, the shared "build the
//!   wire-ready bytes" step every send path on both client and
//!   server converges through. Owns the LF-delimited framing
//!   contract single-source.
//! - [`sockpath`] — UNIX-socket path resolution, atomic-rename bind
//!   with 0600 permissions, stale-socket recovery, and the
//!   drop-guard that unlinks the socket on graceful shutdown or
//!   panic.
//! - [`project`] — pure projection of engine + driver state into
//!   the response carriers. No I/O, no engine mutation.
//! - [`client`] — operator-facing client verbs.
//! - [`render`] — human-readable rendering for `-o human` output.
//!
//! There is no dedicated `server` module: the driver's mio reactor
//! accepts each connection directly via
//! [`crate::driver::hub::DriverHub`] and dispatches `WireRequest`
//! lines inline through [`crate::driver`]'s IPC handler.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. Nothing here is `pub` — operator
//! clients ship inside the same binary, so the wire surface is an
//! implementation detail of the bin, not a published library
//! interface.

pub(crate) mod client;
pub(crate) mod framing;
pub(crate) mod project;
pub(crate) mod protocol;
pub(crate) mod render;
pub(crate) mod sockpath;
pub(crate) mod wire;
