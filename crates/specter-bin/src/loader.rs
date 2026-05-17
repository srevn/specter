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

    /// Derive the watcher's deferred-drain window from `current_config`.
    ///
    /// Formula: `min(settle for every active Sub *and* Promoter) / 4`,
    /// clamped to the `[10ms, 50ms]` band. The floor (`10ms`) is below
    /// scheduler granularity on every supported platform — a
    /// 1ms-settle Profile pays at most ~9ms latency on the second
    /// drain of a sustained burst (the recency gate skips phase 2
    /// entirely for single touches in quiet periods); the ceiling
    /// (`50ms`) is the cap.
    ///
    /// **Disabled entries don't contribute.** Iterating
    /// [`Config::active_watches`] / [`Config::active_promoters`]
    /// strips the suppressed entries — a disabled `[[watch]]` with
    /// `settle = "1ms"` no longer shrinks the drain window for an
    /// engine that has no Profile against it.
    ///
    /// **All entries disabled / empty config** returns the floor —
    /// the watcher has no FDs so the value is moot, but
    /// `Duration::ZERO` would disable deferred drain permanently
    /// and miss the next re-enable's first burst.
    ///
    /// `settle > 0` is enforced at config-load
    /// (`specter-config::config`) for both static and dynamic entries,
    /// so `min_settle / 4` never divides by zero.
    ///
    /// Not `const fn` — `Duration::clamp` is not const-stable on every
    /// supported toolchain. `Loader::new` stays const.
    #[must_use]
    pub fn derive_drain_window(&self) -> Duration {
        let static_min = self.current_config.active_watches().map(|w| w.settle).min();
        let dynamic_min = self
            .current_config
            .active_promoters()
            .map(|p| p.settle)
            .min();
        let min_settle = match (static_min, dynamic_min) {
            (Some(a), Some(b)) => a.min(b),
            (Some(a), None) | (None, Some(a)) => a,
            (None, None) => return Duration::from_millis(10),
        };
        let raw = min_settle / 4;
        raw.clamp(Duration::from_millis(10), Duration::from_millis(50))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use specter_config::Config;

    /// Sentinel meta used in fixtures that don't exercise the
    /// auto-reload meta-comparison path. Inode 0 is reserved by every
    /// supported kernel and `mode = 0` cannot occur in a real lstat
    /// (the kernel always sets file-type bits); this value never
    /// compares equal to a real `FileMeta::from_path` capture.
    fn dummy_meta() -> FileMeta {
        FileMeta {
            inode: 0,
            device: 0,
            mtime_sec: 0,
            mtime_nsec: 0,
            size: 0,
            mode: 0,
            uid: 0,
            gid: 0,
        }
    }

    fn loader_with_toml(toml: &str) -> Loader {
        let cfg = Config::from_str(toml).expect("fixture parses");
        let log = cfg.log.clone();
        Loader::new(cfg, log, dummy_meta())
    }

    /// Empty config (no static, no dynamic) returns the floor.
    #[test]
    fn derive_drain_window_empty_config_returns_floor() {
        let loader = loader_with_toml("");
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(10));
    }

    /// Static-only with default settle (200ms) → 200/4 = 50ms (the
    /// ceiling).
    #[test]
    fn derive_drain_window_static_only_uses_static_min() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\nactions = [{ exec = [\"echo\"] }]",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(50));
    }

    /// Dynamic-only — same default settle (200ms) → 50ms. Promoter
    /// settle now folds into the min computation.
    #[test]
    fn derive_drain_window_dynamic_only_uses_dynamic_min() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"d\"\npath = \"/srv/*\"\nactions = [{ exec = [\"echo\"] }]",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(50));
    }

    /// Mixed: static settle 1000ms (1000/4 = 250 → clamped to 50);
    /// dynamic settle 100ms (100/4 = 25, in the band). The min is
    /// the dynamic one — drain window 25ms.
    #[test]
    fn derive_drain_window_mixed_uses_overall_min() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"1000ms\"\n\
             [[watch]]\nname = \"d\"\npath = \"/srv/*\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"100ms\"\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(25));
    }

    /// Mixed flipped: dynamic settle is larger; static is the min.
    /// Confirms the symmetry of the (Some, Some) match arm.
    #[test]
    fn derive_drain_window_mixed_static_smaller() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"100ms\"\n\
             [[watch]]\nname = \"d\"\npath = \"/srv/*\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"1000ms\"\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(25));
    }

    /// Tiny settle (40ms) clamps to floor. 40/4 = 10ms, exactly the
    /// floor — confirms inclusive boundary.
    #[test]
    fn derive_drain_window_tiny_settle_clamps_to_floor() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"d\"\npath = \"/srv/*\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"40ms\"\nmax_settle = \"200ms\"\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(10));
    }

    /// Sub-floor settle (1ms) clamps to floor.
    #[test]
    fn derive_drain_window_sub_floor_dynamic_settle_clamps_to_floor() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"d\"\npath = \"/srv/*\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"1ms\"\nmax_settle = \"60ms\"\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(10));
    }

    /// A disabled entry with a tiny settle is filtered out by
    /// `active_watches`, so it does not shrink the window. Two
    /// entries: an enabled 1000ms (1000/4 = 250 → clamped to 50ms)
    /// and a disabled 1ms (which would otherwise force the floor).
    /// Result: 50ms — proves the disabled entry is not in the min
    /// computation.
    #[test]
    fn derive_drain_window_ignores_disabled_settle() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"1000ms\"\n\
             [[watch]]\nname = \"b\"\npath = \"/tmp\"\nactions = [{ exec = [\"echo\"] }]\n\
             settle = \"1ms\"\nmax_settle = \"60ms\"\nenabled = false\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(50));
    }

    /// All-disabled config returns the floor (same fallback as the
    /// empty-config case). The `(None, None)` arm of the helper
    /// fires when both filtered iterators are empty.
    #[test]
    fn derive_drain_window_all_disabled_returns_floor() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\nactions = [{ exec = [\"echo\"] }]\nenabled = false\n\
             [[watch]]\nname = \"b\"\npath = \"/srv/*\"\nactions = [{ exec = [\"echo\"] }]\nenabled = false\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(10));
    }
}
