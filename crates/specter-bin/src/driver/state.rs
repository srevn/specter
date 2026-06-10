//! `DriverState` — process-level facts the engine driver carries for the lifetime of the daemon.
//! Owned by [`super::EngineDriver`]; mutated only through edge methods on this type.
//!
//! The struct collects load-bearing process metadata that has no natural home on `Engine`,
//! `Loader`, or any channel bundle:
//!
//! - **Start instants** (`start_instant`, `start_wall`) — captured once at [`DriverState::new`],
//!   invariant for the process lifetime. `start_instant` is monotonic (`Instant`) for
//!   elapsed-since-boot arithmetic; `start_wall` is wall-clock (`SystemTime`) for
//!   operator-meaningful boot display. Both are sampled inside the constructor so wall and
//!   monotonic agree to within their own nanosecond resolution.
//! - **Reload counter** (`reload_count`) — bumped by [`DriverState::record_reload`] on every
//!   successful reload (i.e., after `read_and_parse_config` returns `Some`; covers both empty-diff
//!   and apply-diff branches). A parse-fail reload short-circuits upstream of the bump site and
//!   never reaches the record.
//! - **Most-recent reload attribution** (`last_reload`) — wall-clock and trigger lifted into a
//!   single [`LastReload`] observable; `None` before the first reload fires, `Some` after.
//! - **Socket path** (`socket_path`) — the UNIX-socket path the IPC server bound to. Set once in
//!   [`DriverState::new`] from the path `App::run` passed to `sockpath::bind_socket_atomic`;
//!   invariant for the daemon's lifetime (no setter). Read by the IPC `status` projection so
//!   operators see the exact path the listener is serving.
//!
//! **Sole writer:** [`DriverState::record_reload`]. The counter and the [`LastReload`] pair move
//! together as one observable transition; the edge method captures the wall-clock internally rather
//! than taking it as a parameter. The [`LastReload`] sum type makes the "both attribution fields
//! present together" invariant structural rather than a convention enforced across two `Option`
//! fields.
//!
//! # Visibility
//!
//! `pub(crate)` so [`crate::driver::ipc::project`] can project the recorded facts into the
//! wire-side `StatusResponse`. The fields are `pub(crate)` for the same reason — projection reads
//! them directly. The write-once-via-`record_reload` invariant for the counters survives because
//! the *driver* owns the only `&mut DriverState` (it is a field of [`super::EngineDriver`]); a
//! `&DriverState` borrow handed out cross-module cannot mutate.

use crate::ipc::protocol::WireLastReload;
use crate::ipc::wire::{WireReloadTrigger, WireTime};
use std::path::PathBuf;
use std::time::{Instant, SystemTime};

/// Driver-owned process facts. See module rustdoc.
#[derive(Debug)]
pub(crate) struct DriverState {
    /// Monotonic startup instant — `Instant::now()` at [`Self::new`]. Elapsed-since-boot arithmetic
    /// reads off this via `Instant::elapsed()`.
    pub(crate) start_instant: Instant,
    /// Wall-clock startup time — `SystemTime::now()` at [`Self::new`]. Operator-meaningful boot
    /// display reads off this; sampled at the same physical moment as `start_instant`.
    pub(crate) start_wall: SystemTime,
    /// Successful-reload counter. Bumped by [`Self::record_reload`] — covers SIGHUP, auto-reload
    /// settle-expiry, and IPC reload. Parse-fail does NOT bump (the helper short-circuits before
    /// the record call).
    pub(crate) reload_count: u64,
    /// Most recent successful reload — wall-clock + trigger as one observable. `None` before the
    /// first reload fires. Packing both into one struct makes the impossible product `(Some, None)`
    /// / `(None, Some)` unconstructable at the type level — see [`LastReload`].
    pub(crate) last_reload: Option<LastReload>,
    /// UNIX-socket path the IPC server bound to. Set once in [`Self::new`] from `App::run`'s resolved
    /// path (which it also hands to `sockpath::bind_socket_atomic`); the projection's `socket_path`
    /// therefore exactly matches the bound listener. Invariant for the daemon's lifetime — no setter.
    pub(crate) socket_path: PathBuf,
}

impl DriverState {
    /// Construct at process boot. Captures `start_instant` / `start_wall` from a single physical
    /// moment — both `now()` calls happen in this constructor, so the wall-clock and the monotonic
    /// instant agree to within their own nanosecond resolution. Reload counters initialise to a
    /// fresh-process zero state. `socket_path` is the path the IPC server is bound to (resolved by
    /// `App::run` and threaded through `EngineDriver::new`).
    pub(crate) fn new(socket_path: PathBuf) -> Self {
        Self {
            start_instant: Instant::now(),
            start_wall: SystemTime::now(),
            reload_count: 0,
            last_reload: None,
            socket_path,
        }
    }

    /// Record a successful reload — counter + attribution move together. Bumps
    /// [`Self::reload_count`] (with `saturating_add` guarding the practically-unreachable
    /// `u64`-overflow case) and stamps [`Self::last_reload`] with the wall-clock at *this call*
    /// plus the caller's `trigger`, packed as one [`LastReload`].
    ///
    /// Sole call site is `EngineDriver::dispatch_reload`, immediately after `read_and_parse_config`
    /// returns `Some`. Both the empty-diff branch (operator re-saved unchanged bytes; pulse still
    /// honoured) and the apply-diff branch reach the bump. Parse-fail is upstream of this site and
    /// never reaches the record — the discipline lives at the call site, not as a branch here.
    pub(crate) fn record_reload(&mut self, trigger: ReloadTrigger) {
        self.reload_count = self.reload_count.saturating_add(1);
        self.last_reload = Some(LastReload {
            at: SystemTime::now(),
            via: trigger,
        });
    }
}

/// Most-recent successful reload — wall-clock + attribution as one observable. The struct carries the
/// invariant "both fields present iff at least one reload has fired"; packing both fields into one
/// struct makes the invariant structural (the impossible `(Some(at), None)` / `(None, Some(via))`
/// products are unconstructable). [`DriverState::record_reload`] is the sole constructor; the
/// wire-side mirror [`WireLastReload`] keeps the JSON shape flat via `#[serde(flatten)]`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct LastReload {
    /// Wall-clock at the reload-record call site. Sampled with `SystemTime::now()` inside
    /// [`DriverState::record_reload`].
    pub(crate) at: SystemTime,
    /// What drove this reload. See [`ReloadTrigger`].
    pub(crate) via: ReloadTrigger,
}

/// Projection lives at the source so the wire layer stays a leaf (no `crate::driver` import in
/// [`crate::ipc::protocol`]) and a new field on [`LastReload`] fails compilation here, at the
/// struct's declaration site.
impl From<LastReload> for WireLastReload {
    fn from(lr: LastReload) -> Self {
        Self {
            at: WireTime::from(lr.at),
            via: WireReloadTrigger::from(lr.via),
        }
    }
}

/// What drove a reload. Three sources converge on the same `EngineDriver::dispatch_reload` body;
/// this enum carries the per-caller attribution into [`DriverState::record_reload`].
///
/// `pub(crate)` so the IPC layer (`crate::driver::ipc::project`) can project the recorded trigger
/// into the wire-side `status` response. The enum is constructed at the call site that knows the
/// trigger (SIGHUP arm in `tick`, settle-expiry arm in
/// [`super::EngineDriver::apply_config_settle_expiry`], or the IPC `Reload` arm in the driver IPC
/// drain), keeping attribution exact rather than inferred.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ReloadTrigger {
    /// SIGHUP from the signal thread reached the reload-pulse drain in `EngineDriver::tick`.
    Sighup,
    /// Auto-reload settle expiry observed `FileMeta` drift against `loader.config_meta`
    /// (config-watcher pulse → settle window → lstat diff → `dispatch_reload`).
    AutoReload,
    /// IPC `Reload` request arrived through the driver's IPC drain (per-conn `WireRequest::Reload`
    /// line → driver). Single- source attribution: constructed at the IPC drain's `Reload` arm, not
    /// inferred from a peer pulse — operators reading `status.last_reload_via` after a `specter
    /// reload` round-trip see the exact trigger that drove the reload they observed.
    Ipc,
    /// Startup-TOCTOU drift detected by `App::run`: the on-disk `FileMeta` changed between the
    /// initial config read and the config-watcher's registration window. The driver runs the same
    /// reload pipeline as SIGHUP, with `Startup` attribution so operators can distinguish
    /// "boot-time drift caught and applied" from a subsequent operator-driven reload.
    Startup,
}

/// Projection lives at the source so the wire layer stays a leaf (no `crate::driver` import in
/// [`crate::ipc::wire`]) and a new [`ReloadTrigger`] variant fails compilation here, at the
/// variant's declaration site.
impl From<ReloadTrigger> for WireReloadTrigger {
    fn from(t: ReloadTrigger) -> Self {
        match t {
            ReloadTrigger::Sighup => Self::Sighup,
            ReloadTrigger::AutoReload => Self::Auto,
            ReloadTrigger::Ipc => Self::Ipc,
            ReloadTrigger::Startup => Self::Startup,
        }
    }
}
