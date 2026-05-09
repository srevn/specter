//! `Loader` — the bin's persistent reload state.
//!
//! Holds the most recently applied [`Config`] (the snapshot the engine
//! was last reconciled against) and the `name → SubId` map the bin
//! needs to compute hot-reload diffs (`specter_config::diff` reads
//! `ids` to look up the [`SubId`] for each "removed" or "modified" Sub
//! by name).
//!
//! Lives on the engine driver thread — file I/O on SIGHUP runs there
//! too, eliminating the Mutex an early design sketch anticipated.
//! Sole writer: `EngineDriver::handle_reload` and
//! `EngineDriver::run_initial_attach`.
//!
//! `BTreeMap` (not `HashMap`) for `ids` so iteration order is stable
//! (I7 discipline is uniform across the workspace, even where the bin
//! isn't directly subject to it).

use compact_str::CompactString;
use specter_config::{Config, LogConfig};
use specter_core::{PromoterId, SubId};
use std::collections::BTreeMap;
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
    /// `name → SubId` map for currently-attached static Subs. Threaded
    /// into `specter_config::diff` so the diff function can populate
    /// `removed` / `modified` with the live `SubId`s. Mutated only on
    /// the engine driver thread.
    pub ids: BTreeMap<CompactString, SubId>,
    /// `name → PromoterId` map for currently-attached Promoters —
    /// the dynamic-watch analogue of [`Self::ids`]. Threaded into
    /// `specter_config::diff` so the Promoter half of
    /// [`specter_core::WatchRegistryDiff`] can populate `removed` /
    /// `modified` with live ids.
    ///
    /// Population is deferred to the diagnostic-driven reconciliation
    /// landing with the Promoter initial-attach pass — until that
    /// lands, the field stays empty, and Promoter `removed` /
    /// `modified` entries silently no-op (which is safe because no
    /// Promoter is ever attached through the bin without that
    /// reconciliation also running).
    pub promoter_ids: BTreeMap<CompactString, PromoterId>,
}

impl Loader {
    /// Fresh loader starting from `current_config` with empty id maps.
    /// The static map fills as `EngineDriver::run_initial_attach`
    /// walks `current_config.watches`; the Promoter map fills via the
    /// engine's `Diagnostic::PromoterAttached` once the corresponding
    /// reconciliation lands. `current_log` is the resolved log config —
    /// the bin computes it once at startup (config + CLI merge) and
    /// hands it in.
    #[must_use]
    pub const fn new(current_config: Config, current_log: LogConfig) -> Self {
        Self {
            current_config,
            current_log,
            ids: BTreeMap::new(),
            promoter_ids: BTreeMap::new(),
        }
    }

    /// Derive the watcher's deferred-drain window from `current_config`.
    ///
    /// Formula: `min(settle for every static Sub *and* Promoter) / 4`,
    /// clamped to the `[10ms, 50ms]` band. The floor (`10ms`) is below
    /// scheduler granularity on every supported platform — a
    /// 1ms-settle Profile pays at most ~9ms latency on the second
    /// drain of a sustained burst (the recency gate skips phase 2
    /// entirely for single touches in quiet periods); the ceiling
    /// (`50ms`) is the cap.
    ///
    /// **Empty `watches` and `promoters`** returns the floor — the
    /// watcher has no FDs so the value is moot, but `Duration::ZERO`
    /// would disable deferred drain permanently and miss the next
    /// added watch's first burst.
    ///
    /// `settle_ms ≥ 1` is enforced at config-load
    /// (`specter-config::config`) for both static and dynamic entries,
    /// so `min_settle / 4` never divides by zero.
    ///
    /// Not `const fn` — `Duration::clamp` is not const-stable on every
    /// supported toolchain. `Loader::new` stays const.
    #[must_use]
    pub fn derive_drain_window(&self) -> Duration {
        let static_min = self.current_config.watches.iter().map(|w| w.settle).min();
        let dynamic_min = self.current_config.promoters.iter().map(|p| p.settle).min();
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

    fn config_with_one_watch() -> Config {
        let toml = r#"
    [[watch]]
    name    = "build"
    path    = "/tmp"
    command = ["true"]
    "#;
        Config::from_str(toml).expect("fixture parses")
    }

    fn fresh_loader() -> Loader {
        let cfg = config_with_one_watch();
        let log = cfg.log.clone();
        Loader::new(cfg, log)
    }

    #[test]
    fn new_starts_with_empty_ids() {
        let loader = fresh_loader();
        assert!(loader.ids.is_empty());
        assert!(loader.promoter_ids.is_empty());
        assert_eq!(loader.current_config.watches.len(), 1);
    }

    #[test]
    fn promoter_ids_insert_round_trip() {
        let mut loader = fresh_loader();
        let pid = PromoterId::default();
        loader.promoter_ids.insert("logs".into(), pid);
        assert_eq!(loader.promoter_ids.get("logs"), Some(&pid));
    }

    #[test]
    fn ids_insert_round_trip() {
        let mut loader = fresh_loader();
        let sid = SubId::default();
        loader.ids.insert("build".into(), sid);
        assert_eq!(loader.ids.get("build"), Some(&sid));
    }

    #[test]
    fn ids_retain_drops_target_id() {
        let mut loader = fresh_loader();
        let id_a = SubId::default();
        loader.ids.insert("a".into(), id_a);
        loader.ids.insert("b".into(), id_a);
        loader.ids.retain(|_, v| *v != id_a);
        assert!(loader.ids.is_empty());
    }

    #[test]
    fn ids_iteration_is_sorted_by_name() {
        let mut loader = fresh_loader();
        loader.ids.insert("c".into(), SubId::default());
        loader.ids.insert("a".into(), SubId::default());
        loader.ids.insert("b".into(), SubId::default());
        let names: Vec<&str> = loader
            .ids
            .keys()
            .map(compact_str::CompactString::as_str)
            .collect();
        assert_eq!(names, vec!["a", "b", "c"]);
    }

    fn loader_with_toml(toml: &str) -> Loader {
        let cfg = Config::from_str(toml).expect("fixture parses");
        let log = cfg.log.clone();
        Loader::new(cfg, log)
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
        let loader =
            loader_with_toml("[[watch]]\nname = \"a\"\npath = \"/tmp\"\ncommand = [\"echo\"]");
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(50));
    }

    /// Dynamic-only — same default settle (200ms) → 50ms. Promoter
    /// settle now folds into the min computation.
    #[test]
    fn derive_drain_window_dynamic_only_uses_dynamic_min() {
        let loader =
            loader_with_toml("[[watch]]\nname = \"d\"\npath = \"/srv/*\"\ncommand = [\"echo\"]");
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(50));
    }

    /// Mixed: static settle 1000ms (1000/4 = 250 → clamped to 50);
    /// dynamic settle 100ms (100/4 = 25, in the band). The min is
    /// the dynamic one — drain window 25ms.
    #[test]
    fn derive_drain_window_mixed_uses_overall_min() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\ncommand = [\"echo\"]\n\
             settle_ms = 1000\n\
             [[watch]]\nname = \"d\"\npath = \"/srv/*\"\ncommand = [\"echo\"]\n\
             settle_ms = 100\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(25));
    }

    /// Mixed flipped: dynamic settle is larger; static is the min.
    /// Confirms the symmetry of the (Some, Some) match arm.
    #[test]
    fn derive_drain_window_mixed_static_smaller() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"a\"\npath = \"/tmp\"\ncommand = [\"echo\"]\n\
             settle_ms = 100\n\
             [[watch]]\nname = \"d\"\npath = \"/srv/*\"\ncommand = [\"echo\"]\n\
             settle_ms = 1000\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(25));
    }

    /// Tiny settle (40ms) clamps to floor. 40/4 = 10ms, exactly the
    /// floor — confirms inclusive boundary.
    #[test]
    fn derive_drain_window_tiny_settle_clamps_to_floor() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"d\"\npath = \"/srv/*\"\ncommand = [\"echo\"]\n\
             settle_ms = 40\nmax_settle_ms = 200\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(10));
    }

    /// Sub-floor settle (1ms) clamps to floor.
    #[test]
    fn derive_drain_window_sub_floor_dynamic_settle_clamps_to_floor() {
        let loader = loader_with_toml(
            "[[watch]]\nname = \"d\"\npath = \"/srv/*\"\ncommand = [\"echo\"]\n\
             settle_ms = 1\nmax_settle_ms = 60\n",
        );
        assert_eq!(loader.derive_drain_window(), Duration::from_millis(10));
    }
}
