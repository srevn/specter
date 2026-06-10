//! Daemon-side operator-IPC concern — transport, per-conn state, verb dispatch, and engine-state
//! projection.
//!
//! Three siblings own the runtime split; one sub-tree holds the pure projections:
//!
//! - [`hub`] — owner of the operator-IPC kernel surface (listener, per-conn map, per-conn Token
//!   allocator) plus the fan-out path that ships diagnostics to live subscribers. Registers against
//!   the same Poll selector as [`super::reactor::Reactor`] via a [`mio::Registry::try_clone()`]
//!   handle.
//! - [`conns`] — per-conn state (`ConnState` + `ConnRole` + `MissedWindow`) stored in the
//!   [`hub::Hub`]'s conn map.
//! - [`dispatch`] — drains the per-conn read events the tick collected, parses each LF-delimited
//!   [`crate::ipc::protocol::WireRequest`], and routes through the [`project`] helpers, the reload
//!   pipeline, or the per-conn role flip.
//! - [`project`] — pure free-function projections of [`specter_engine::Engine`] +
//!   [`super::DriverState`] into the wire response carriers (`StatusResponse`, `ListResponse`,
//!   `ShowResponse`). No I/O, no engine mutation; called from [`dispatch`].
//!
//! Lives at `crate::driver::ipc` (next to `reactor`, `tick`, `forward`) because every concern here
//! holds or reaches `&mut Engine` / `&mut DriverState`. The wire vocabulary itself (`framing`,
//! `protocol`, `wire`, `sockpath`) is a true leaf under [`crate::ipc`]; the direction is one-way
//! (`driver::ipc` consumes `crate::ipc`, never the reverse).
//!
//! # Visibility
//!
//! `pub(super)` for the submodules: only [`super::EngineDriver`] and [`super::tick`] reach in.
//! [`hub::Hub`] is re-exported from [`super`] (`crate::driver`) so `app.rs` can name the type
//! without threading the full path.

pub(super) mod conns;
pub(super) mod dispatch;
pub(super) mod hub;
pub(super) mod project;

pub(crate) use hub::Hub;
