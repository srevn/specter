//! Operator IPC vocabulary — request/response shapes, diagnostic projection, socket lifecycle,
//! client verbs, human rendering.
//!
//! True leaf: nothing here imports [`crate::driver`]. The wire surface lives here so an external
//! CLI (today's in-bin client, a hypothetical out-of-process tool tomorrow) could reuse the same
//! vocabulary without dragging in the daemon's runtime.
//!
//! Daemon-side IPC code — the kernel-fd owner ([`crate::driver::Hub`]), per-conn state, verb
//! dispatch, and engine-state projection — lives under `crate::driver::ipc`. Direction is one-way:
//! `driver::ipc` consumes this module, never the reverse.
//!
//! # Submodules
//!
//! - [`protocol`] — wire-side request shape (`WireRequest`), response carriers (`ResponsePayload`,
//!   `StatusResponse`, `ListResponse`, `ShowResponse`), the `WireId` newtype, and the
//!   `WireErrorCode` closed-vocabulary enum.
//! - [`wire`] — `WireDiagnostic` (the exhaustive projection of every `specter_core::Diagnostic`
//!   variant), `WireTime`, and the per-core-type `Wire*` projections every variant transitively
//!   reaches.
//! - [`framing`] — `encode_line<T: InfallibleSerialize>` (the marker-bounded "build the wire-ready
//!   bytes" wrapper every production send path converges through) and `parse_strict<T>` (the
//!   round-trip unknown-field gate every incoming request / response is admitted through). Owns the
//!   LF-delimited framing contract single-source.
//! - [`resolve`] — socket-path resolution policy: one `resolve` → `Resolution` consumed two ways
//!   (daemon commits a path, client probes a cascade), the per-platform convention, explicit-override
//!   validation, and the `SocketSource` diagnostic vocabulary. Pure, env-injected.
//! - [`sockpath`] — UNIX-socket atomic-rename bind with 0600 permissions, stale-socket recovery,
//!   and the drop-guard that unlinks the socket on graceful shutdown or panic.
//! - [`client`] — operator-facing client verbs.
//! - [`render`] — human-readable rendering for `-o human` output.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. Nothing here is `pub` — operator clients ship inside the same
//! binary, so the wire surface is an implementation detail of the bin, not a published library
//! interface.

pub(crate) mod client;
pub(crate) mod framing;
pub(crate) mod protocol;
pub(crate) mod render;
pub(crate) mod resolve;
pub(crate) mod sockpath;
pub(crate) mod wire;
