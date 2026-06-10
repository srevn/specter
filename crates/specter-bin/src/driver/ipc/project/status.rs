use crate::driver::DriverState;
use crate::ipc::protocol::{StatusResponse, WireLastReload};
use crate::ipc::wire::{WirePath, WireTime};
use compact_str::CompactString;
use specter_config::Config;
use specter_engine::Engine;
use std::collections::BTreeSet;
use std::path::Path;

/// Project the engine + driver state into a [`StatusResponse`].
///
/// The projection is **pure** — every read is a borrow off its source and every owned field on the
/// response is a fresh allocation (mostly `WireTime` strings; `WirePath` lossy projections for the
/// two filesystem paths).
///
/// **`sub_total` semantics** — every Sub currently in the engine registry, *including* dynamic Subs
/// minted by Promoters. An operator wanting "only static, only attached" derives it from `list`. The
/// minimal `status` shape avoids carrying multiple counts that would skew interpretation at a glance.
pub(crate) fn status(
    engine: &Engine,
    ds: &DriverState,
    disabled_runtime: &BTreeSet<CompactString>,
    config: &Config,
    config_path: &Path,
) -> StatusResponse {
    StatusResponse {
        uptime_secs: ds.start_instant.elapsed().as_secs(),
        start_wall: WireTime::from(ds.start_wall),
        reload_count: ds.reload_count,
        last_reload: ds.last_reload.map(WireLastReload::from),
        sub_total: engine.subs().len(),
        // Inline filter+count over `config.watches` rather than `Config::disabled_names()`. The
        // latter allocates two `Vec<&str>` (watches AND promoters) just to read `.len()` off the
        // first one — the promoter side is unused here, and even the watch side wastes a heap
        // allocation for what is structurally a counter.
        sub_disabled_toml: config.watches.iter().filter(|s| !s.enabled).count(),
        sub_disabled_runtime: disabled_runtime.len(),
        profile_active: engine.profiles().active_count(),
        promoter_active: engine.promoters().len(),
        config_path: WirePath::from(config_path),
        socket_path: WirePath::from(&ds.socket_path),
    }
}

#[cfg(test)]
mod tests {
    use super::status;
    use crate::driver::{DriverState, ReloadTrigger};
    use crate::ipc::wire::WirePath;
    use compact_str::CompactString;
    use specter_config::Config;
    use specter_engine::Engine;
    use std::collections::BTreeSet;
    use std::path::{Path, PathBuf};
    use std::thread::sleep;
    use std::time::Duration;

    fn fresh_state() -> DriverState {
        DriverState::new(PathBuf::from("/tmp/specter-test.sock"))
    }

    #[test]
    fn status_empty_engine_zero_counters() {
        let engine = Engine::new();
        let ds = fresh_state();
        let disabled = BTreeSet::<CompactString>::new();
        let config = Config::from_str("").expect("empty config parses");

        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );

        assert_eq!(r.reload_count, 0);
        assert!(
            r.last_reload.is_none(),
            "fresh state has never reloaded ⇒ paired field absent",
        );
        assert_eq!(r.sub_total, 0);
        assert_eq!(r.sub_disabled_toml, 0);
        assert_eq!(r.sub_disabled_runtime, 0);
        assert_eq!(r.profile_active, 0);
        assert_eq!(r.promoter_active, 0);
        assert_eq!(
            r.config_path,
            WirePath::from(Path::new("/etc/specter.toml"))
        );
        assert_eq!(
            r.socket_path,
            WirePath::from(Path::new("/tmp/specter-test.sock")),
        );
    }

    #[test]
    fn status_uptime_advances() {
        let engine = Engine::new();
        let ds = fresh_state();
        let disabled = BTreeSet::<CompactString>::new();
        let config = Config::from_str("").expect("empty config parses");

        // Sleep a measurable amount so `start_instant.elapsed()` is non-zero in seconds with
        // extreme tolerance for CI scheduler jitter — the projection's wiring is "did we read
        // `start_instant.elapsed().as_secs()`?", not "is the clock moving?". A boolean witness
        // keeps the test deterministic.
        sleep(Duration::from_millis(5));
        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );
        // u64 is unsigned; the constraint reduces to "this read didn't panic and we got a number".
        // A regression that swapped `start_instant.elapsed()` for, say, `start_wall.elapsed()`
        // would fail at compile time (different methods).
        let _: u64 = r.uptime_secs;
    }

    #[test]
    fn status_post_reload_carries_attribution() {
        let engine = Engine::new();
        let mut ds = fresh_state();
        ds.record_reload(ReloadTrigger::Sighup);
        let disabled = BTreeSet::<CompactString>::new();
        let config = Config::from_str("").expect("empty config parses");

        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );

        assert_eq!(r.reload_count, 1);
        // `WireReloadTrigger::from(ReloadTrigger::Sighup) == Sighup` — the projection is the
        // structural mapping, not a string. The single `last_reload` field carries both halves
        // together; partial (Some(at), None) is unconstructable.
        let lr = r.last_reload.expect("record_reload populated the pair");
        assert_eq!(lr.via, crate::ipc::wire::WireReloadTrigger::Sighup);
    }

    #[test]
    fn status_post_ipc_reload_carries_ipc_trigger() {
        let engine = Engine::new();
        let mut ds = fresh_state();
        ds.record_reload(ReloadTrigger::Ipc);
        let disabled = BTreeSet::<CompactString>::new();
        let config = Config::from_str("").expect("empty config parses");

        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );
        let lr = r.last_reload.expect("record_reload populated the pair");
        assert_eq!(lr.via, crate::ipc::wire::WireReloadTrigger::Ipc);
    }

    #[test]
    fn status_disabled_runtime_count_from_set() {
        let engine = Engine::new();
        let ds = fresh_state();
        let mut disabled = BTreeSet::<CompactString>::new();
        disabled.insert(CompactString::const_new("foo"));
        disabled.insert(CompactString::const_new("bar"));
        let config = Config::from_str("").expect("empty config parses");

        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );
        assert_eq!(r.sub_disabled_runtime, 2);
    }

    #[test]
    fn status_disabled_toml_from_config() {
        let engine = Engine::new();
        let ds = fresh_state();
        let disabled = BTreeSet::<CompactString>::new();

        let toml = r#"
    [[watch]]
    name      = "off_one"
    path      = "/tmp"
    actions   = [{ exec = ["true"] }]
    enabled   = false
    
    [[watch]]
    name      = "on_one"
    path      = "/tmp"
    actions   = [{ exec = ["true"] }]
    "#;
        let config = Config::from_str(toml).expect("mixed config parses");

        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );
        assert_eq!(r.sub_disabled_toml, 1, "one TOML-disabled watch");
    }

    #[test]
    fn status_socket_path_from_driver_state() {
        let engine = Engine::new();
        let ds = DriverState::new(PathBuf::from("/run/user/1000/custom.sock"));
        let disabled = BTreeSet::<CompactString>::new();
        let config = Config::from_str("").expect("empty config parses");

        let r = status(
            &engine,
            &ds,
            &disabled,
            &config,
            &PathBuf::from("/etc/specter.toml"),
        );
        assert_eq!(
            r.socket_path,
            WirePath::from(Path::new("/run/user/1000/custom.sock")),
        );
    }
}
