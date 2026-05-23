//! `DriverState` — process-level facts the engine driver carries for
//! the lifetime of the daemon. Owned by [`super::EngineDriver`];
//! mutated only through edge methods on this type.
//!
//! The struct collects load-bearing process metadata that has no
//! natural home on `Engine`, `Loader`, or any channel bundle:
//!
//! - **Start instants** (`start_instant`, `start_wall`) — captured
//!   once at [`Self::new`], invariant for the process lifetime.
//!   `start_instant` is monotonic (`Instant`) for elapsed-since-boot
//!   arithmetic; `start_wall` is wall-clock (`SystemTime`) for
//!   operator-meaningful boot display. Both are sampled inside the
//!   constructor so wall and monotonic agree to within their own
//!   nanosecond resolution.
//! - **Reload counters** — bumped by [`Self::record_reload`] on every
//!   successful reload (i.e., after `read_and_parse_config` returns
//!   `Some`; covers both empty-diff and apply-diff branches). A
//!   parse-fail reload short-circuits upstream of the bump site and
//!   never reaches the record.
//!
//! **Sole writer:** [`Self::record_reload`]. The three counter fields
//! (`reload_count` / `last_reload_at` / `last_reload_via`) move
//! together as one observable transition, so the edge method captures
//! the wall-clock internally rather than taking it as a parameter —
//! the three fields cannot diverge.

use std::time::{Instant, SystemTime};

/// Driver-owned process facts. See module rustdoc.
///
/// `#[allow(dead_code)]` silences the dead-code lint on
/// `start_instant` / `start_wall` — both are written once in
/// [`Self::new`] and have no in-module reader. The counter fields
/// escape the lint via [`Self::record_reload`]'s in-place mutation
/// (which the lint counts as use), so the allowance is functionally
/// scoped to the two startup-instant fields even though the
/// attribute sits at the struct.
#[allow(dead_code)]
#[derive(Debug)]
pub(super) struct DriverState {
    /// Monotonic startup instant — `Instant::now()` at [`Self::new`].
    /// Elapsed-since-boot arithmetic reads off this via
    /// `Instant::elapsed()`.
    pub(super) start_instant: Instant,
    /// Wall-clock startup time — `SystemTime::now()` at [`Self::new`].
    /// Operator-meaningful boot display reads off this; sampled at
    /// the same physical moment as `start_instant`.
    pub(super) start_wall: SystemTime,
    /// Successful-reload counter. Bumped by [`Self::record_reload`]
    /// — covers SIGHUP and auto-reload settle-expiry. Parse-fail does
    /// NOT bump (the helper short-circuits before the record call).
    pub(super) reload_count: u64,
    /// Wall-clock of the most recent successful reload, `None` before
    /// the first one fires.
    pub(super) last_reload_at: Option<SystemTime>,
    /// Trigger of the most recent successful reload, `None` before
    /// the first one fires.
    pub(super) last_reload_via: Option<ReloadTrigger>,
}

impl DriverState {
    /// Construct at process boot. Captures `start_instant` /
    /// `start_wall` from a single physical moment — both `now()`
    /// calls happen in this constructor, so the wall-clock and the
    /// monotonic instant agree to within their own nanosecond
    /// resolution. Reload counters initialise to a fresh-process
    /// zero state.
    pub(super) fn new() -> Self {
        Self {
            start_instant: Instant::now(),
            start_wall: SystemTime::now(),
            reload_count: 0,
            last_reload_at: None,
            last_reload_via: None,
        }
    }

    /// Record a successful reload — the three counter fields move
    /// together. Bumps `reload_count`, stamps `last_reload_at` from
    /// the wall-clock at *this call*, and overwrites `last_reload_via`
    /// with `trigger`. `saturating_add` guards the (practically
    /// unreachable) `u64`-overflow case.
    ///
    /// Sole call site is `EngineDriver::handle_reload`, immediately
    /// after `read_and_parse_config` returns `Some`. Both the
    /// empty-diff branch (operator re-saved unchanged bytes; pulse
    /// still honoured) and the apply-diff branch reach the bump.
    /// Parse-fail is upstream of this site and never reaches the
    /// record — the discipline lives at the call site, not as a
    /// branch here.
    pub(super) fn record_reload(&mut self, trigger: ReloadTrigger) {
        self.reload_count = self.reload_count.saturating_add(1);
        self.last_reload_at = Some(SystemTime::now());
        self.last_reload_via = Some(trigger);
    }
}

/// What drove a reload. Two sources converge on the same
/// `EngineDriver::handle_reload` body; this enum carries the
/// per-caller attribution into [`DriverState::record_reload`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum ReloadTrigger {
    /// SIGHUP from the signal thread reached the reload-pulse drain
    /// in `EngineDriver::tick`.
    Sighup,
    /// Auto-reload settle expiry observed `FileMeta` drift against
    /// `loader.config_meta` (config-watcher pulse → settle window →
    /// lstat diff → `handle_reload`).
    AutoReload,
}
