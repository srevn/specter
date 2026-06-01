//! Free-function projections of engine + driver state into
//! [`ResponsePayload`](crate::ipc::protocol::ResponsePayload)-bound
//! carriers.
//!
//! Pure functions: every parameter is `&_`. The carriers clone what
//! they own (`PathBuf` for `socket_path` / `config_path` — they rotate
//! out to a fresh client per request, so sharing references is not an
//! option). No I/O, no `Arc<Mutex>`, no engine mutation.
//!
//! Lives on the daemon side ([`crate::driver::ipc::project`]) because
//! the source data ([`specter_engine::Engine`] +
//! [`crate::driver::DriverState`]) is driver-owned. The "projection
//! lives at the source" rule the codebase encodes for
//! [`crate::driver::state`]'s `From<ReloadTrigger> for WireReloadTrigger`
//! applies here too — the wire-side vocabulary at [`crate::ipc`] stays
//! a true leaf with no `crate::driver` import.
//!
//! # Visibility
//!
//! `pub(super)` — the verb dispatcher
//! ([`crate::driver::ipc::dispatch`]) is the sole caller;
//! client-side rendering reads through the response carriers, not
//! through the helpers.
//!
//! # Projections
//!
//! - [`status()`] — `specter status` summary.
//! - [`list()`] — `specter list` row union over engine + disabled
//!   sources.
//! - [`show()`] — `specter show <name>` detail block.
//!
//! `program` lowers each [`specter_core::ActionProgram`] op to
//! operator-readable lines for `show`. Every projection reuses the
//! same `&Engine`/`&DriverState`/`&BTreeSet`/`&Config` parameter
//! pattern.

mod list;
mod program;
mod show;
mod status;

pub(super) use list::list;
pub(super) use show::show;
pub(super) use status::status;

use std::time::{Instant, SystemTime};

/// Project an engine-monotonic [`Instant`] onto a wall-clock
/// [`SystemTime`] using the driver's startup anchor pair.
///
/// `Instant` is monotonic and not directly displayable; the wire emits
/// RFC3339 strings. The anchor pair (`start_wall`, `start_instant`) is
/// captured atomically in [`crate::driver::DriverState::new`], so the
/// projection agrees with the operator's wall clock at boot.
///
/// `t < start_instant` is structurally impossible (the engine records
/// fires after the driver starts); `saturating_duration_since` guards
/// the type-system corner anyway.
///
/// `start_wall + duration` overflow is unreachable in any realistic
/// operating window (`i64::MAX` seconds past the epoch); no guard.
pub(super) fn project_wall(
    start_wall: SystemTime,
    start_instant: Instant,
    t: Instant,
) -> SystemTime {
    start_wall + t.saturating_duration_since(start_instant)
}
