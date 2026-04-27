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
use specter_config::Config;
use specter_core::SubId;
use std::collections::BTreeMap;

/// Bin-side reload state. See module rustdoc.
#[derive(Debug)]
pub struct Loader {
    pub current_config: Config,
    pub ids: BTreeMap<CompactString, SubId>,
}

impl Loader {
    /// Fresh loader starting from `current_config` with an empty id
    /// map. The map fills as `EngineDriver::run_initial_attach` walks
    /// `current_config.watches`.
    #[must_use]
    pub const fn new(current_config: Config) -> Self {
        Self {
            current_config,
            ids: BTreeMap::new(),
        }
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

    #[test]
    fn new_starts_with_empty_ids() {
        let loader = Loader::new(config_with_one_watch());
        assert!(loader.ids.is_empty());
        assert_eq!(loader.current_config.watches.len(), 1);
    }

    #[test]
    fn ids_insert_round_trip() {
        let mut loader = Loader::new(config_with_one_watch());
        let sid = SubId::default();
        loader.ids.insert("build".into(), sid);
        assert_eq!(loader.ids.get("build"), Some(&sid));
    }

    #[test]
    fn ids_retain_drops_target_id() {
        let mut loader = Loader::new(config_with_one_watch());
        let id_a = SubId::default();
        loader.ids.insert("a".into(), id_a);
        loader.ids.insert("b".into(), id_a);
        loader.ids.retain(|_, v| *v != id_a);
        assert!(loader.ids.is_empty());
    }

    #[test]
    fn ids_iteration_is_sorted_by_name() {
        let mut loader = Loader::new(config_with_one_watch());
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
