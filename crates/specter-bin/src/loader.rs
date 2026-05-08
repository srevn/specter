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
use specter_core::SubId;
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
    pub ids: BTreeMap<CompactString, SubId>,
}

impl Loader {
    /// Fresh loader starting from `current_config` with an empty id
    /// map. The map fills as `EngineDriver::run_initial_attach` walks
    /// `current_config.watches`. `current_log` is the resolved log
    /// config — the bin computes it once at startup (config + CLI
    /// merge) and hands it in.
    #[must_use]
    pub const fn new(current_config: Config, current_log: LogConfig) -> Self {
        Self {
            current_config,
            current_log,
            ids: BTreeMap::new(),
        }
    }

    /// Derive the watcher's deferred-drain window from `current_config`.
    ///
    /// Formula: `min(settle for every Profile) / 4`, clamped to the
    /// audit's `[10ms, 50ms]` band. The floor (`10ms`) is below
    /// scheduler granularity on every supported platform — a 1ms-settle
    /// Profile pays at most ~9ms latency on the second drain of a
    /// sustained burst (the recency gate skips phase 2 entirely for
    /// single touches in quiet periods); the ceiling (`50ms`) is the
    /// audit §3.7 cap.
    ///
    /// **Empty `watches`** returns the floor — the watcher has no FDs
    /// so the value is moot, but `Duration::ZERO` would disable
    /// deferred drain permanently and miss the next added watch's
    /// first burst.
    ///
    /// `settle_ms ≥ 1` is enforced at config-load
    /// (`specter-config::config`), so `min_settle / 4` never
    /// divides by zero.
    ///
    /// Not `const fn` — `Duration::clamp` is not const-stable on every
    /// supported toolchain. `Loader::new` stays const.
    #[must_use]
    pub fn derive_drain_window(&self) -> Duration {
        let Some(min_settle) = self.current_config.watches.iter().map(|w| w.settle).min() else {
            return Duration::from_millis(10);
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
        assert_eq!(loader.current_config.watches.len(), 1);
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
}
