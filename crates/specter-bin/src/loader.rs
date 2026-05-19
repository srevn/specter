//! `Loader` — the bin's persistent reload state.
//!
//! Holds the most recently applied [`Config`] (the snapshot the next
//! reload diffs against) plus its resolved `[log]` block and on-disk
//! [`FileMeta`] identity. It carries **no** `name → id` map: hot-reload
//! diffs are name-keyed and the engine resolves names to ids through
//! its own authoritative registries, so the bin never mirrors engine
//! identity.
//!
//! Lives on the engine driver thread — file I/O on SIGHUP runs there
//! too, eliminating the Mutex an early design sketch anticipated.
//! Sole writer: `EngineDriver::handle_reload` and
//! `EngineDriver::run_initial_attach`.

use specter_config::{Config, FileMeta, LogConfig};
use std::time::Duration;

/// Fixed trailing-latency window for the watcher's deferred-drain pass.
///
/// **Not an inbound-volume lever.** Inbound volume is owned by driver
/// same-tick coalescing (accumulate regime) and per-event engine cost
/// (keeps-up regime); one watcher-side scalar provably cannot serve a
/// per-Profile volume constraint, so this knob no longer scales with
/// `settle`. It is purely the latency budget the watcher trades for
/// batch granularity on its second drain pass — see
/// [`specter_sensor::DrainWindow`] for the deferred-drain semantics.
///
/// `50ms` is the top of the historical `[10ms, 50ms]` band — the value
/// default-`settle` configs already resolved to. The watcher's recency
/// gate skips the second drain for single touches in quiet periods, so
/// a quiet-period edit pays none of this. Fixed in v1 per the
/// project's "minimal config surface" alpha rule.
pub(crate) const WATCHER_DRAIN_WINDOW: Duration = Duration::from_millis(50);

/// Bin-side reload state. See module rustdoc.
#[derive(Debug)]
pub struct Loader {
    pub current_config: Config,
    /// `[log]` block as resolved at startup or after the last successful
    /// reload — *with* CLI overrides folded in. Compared against the
    /// next reload's resolved value to decide whether to call
    /// `obs_handle.set_level` and / or fail-with-error on a destination
    /// change.
    pub current_log: LogConfig,
    /// Inode-level identity of `current_config`'s on-disk source —
    /// captured atomically with the content read via
    /// [`Config::from_path_with_meta`] (the `f.metadata()` call binds to
    /// the same `File` handle that produced the bytes, so a concurrent
    /// `rename(2)` cannot rotate the meta out from under the bytes).
    /// Rotated alongside `current_config` on every successful reload —
    /// **including the empty-diff branch**, so a re-saved-but-identical
    /// file still updates the stored identity. Without that rotation,
    /// the auto-reload settle-expiry filter would observe a fresh
    /// lstat that differs from the stored value forever, looping
    /// `handle_reload` against the same content.
    pub config_meta: FileMeta,
}

impl Loader {
    /// Fresh loader starting from `current_config`. `current_log` is
    /// the resolved log config — the bin computes it once at startup
    /// (config + CLI merge) and hands it in. `config_meta` is the
    /// inode-level identity captured atomically with `current_config`
    /// (see [`Self::config_meta`]).
    #[must_use]
    pub const fn new(
        current_config: Config,
        current_log: LogConfig,
        config_meta: FileMeta,
    ) -> Self {
        Self {
            current_config,
            current_log,
            config_meta,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::WATCHER_DRAIN_WINDOW;
    use std::time::Duration;

    /// The watcher's deferred-drain window is a fixed trailing-latency
    /// constant — no longer config-derived. Pins the value and its
    /// in-band placement so a future change is a conscious latency
    /// decision, not accidental drift. The historical band was
    /// `[10ms, 50ms]`; `50ms` is the prior default-`settle` resolution.
    #[test]
    fn watcher_drain_window_is_fixed_at_band_ceiling() {
        assert_eq!(WATCHER_DRAIN_WINDOW, Duration::from_millis(50));
        assert!(
            WATCHER_DRAIN_WINDOW >= Duration::from_millis(10)
                && WATCHER_DRAIN_WINDOW <= Duration::from_millis(50),
            "constant must stay within the historical trailing-latency band",
        );
    }
}
