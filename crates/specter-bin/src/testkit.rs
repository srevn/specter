//! Fixtures the bin's unit tests share, behind the `testkit` Cargo feature.

use specter_config::Config;
use specter_core::testkit::dir_snap;
use specter_core::{EntryKind, Input, SubId};
use specter_engine::Engine;
use specter_engine::testkit::{attach_discovery, pre_place_dir, seed_to_idle};
use std::fmt::Write as _;
use std::time::Instant;

/// A fresh daemon's driver state; nothing binds its socket path. Only the crate's own tests build
/// one: its type is crate-private, so the fixture cannot join the public ones below.
#[cfg(test)]
pub(crate) fn fresh_state() -> crate::driver::DriverState {
    crate::driver::DriverState::new("/tmp/specter-test.sock".into())
}

/// One `[[watch]]` per `(name, path, enabled)`, each running `true`. A path holding a glob lowers
/// to a dynamic watch, as in any config.
#[must_use]
pub fn config_from_watches(watches: &[(&str, &str, bool)]) -> Config {
    let mut s = String::new();
    for (name, path, enabled) in watches {
        let _ = write!(
            s,
            r#"
    [[watch]]
    name      = "{name}"
    path      = "{path}"
    actions   = [{{ exec = ["true"] }}]
    enabled   = {enabled}
    "#,
        );
    }
    Config::from_str(&s).expect("test config parses")
}

/// An engine with every active watch in `config` attached, in source order. The attach outputs are
/// dropped: the projections read the registries.
#[must_use]
pub fn engine_with(config: &Config) -> Engine {
    let mut engine = Engine::new();
    let now = Instant::now();
    for spec in config.active_watches() {
        let _ = engine.step(Input::AttachSub(spec.to_attach_request()), now);
    }
    engine
}

/// Owns an engine and drains its in-flight probes before it drops: every attach arms one, and
/// `ProbeSlot`'s tripwire (`specter_core::probe`) panics when an armed slot is destroyed.
#[derive(Debug)]
pub struct EngineGuard(Engine);

impl EngineGuard {
    #[must_use]
    pub const fn wrap(engine: Engine) -> Self {
        Self(engine)
    }

    #[must_use]
    pub const fn engine(&self) -> &Engine {
        &self.0
    }
}

impl Drop for EngineGuard {
    fn drop(&mut self) {
        let _ = self.0.cancel_all_in_flight_probes();
    }
}

/// Mint a Sub per match the way the daemon does.
///
/// A template `disc` on `/srv/*` attaches at a pre-placed `/srv`, and its Seed's walk lists
/// `matches`, each of which the pass mints as `disc@/srv/<match>`. Returns the minted Subs in
/// `matches`' order; their own Seeds are left in flight.
pub fn mint<const N: usize>(engine: &mut Engine, matches: [&str; N]) -> [SubId; N] {
    let now = Instant::now();
    pre_place_dir(engine, &["srv"]);
    let (_, discovery) = attach_discovery(engine, "disc", "/srv/*", now);
    let listing: Vec<_> =
        matches.iter().zip(1..).map(|(&name, inode)| (name, EntryKind::Dir, inode)).collect();
    let _ = seed_to_idle(engine, discovery, &dir_snap(&listing), now);
    matches.map(|name| {
        engine
            .subs()
            .find_by_name(&format!("disc@/srv/{name}"))
            .expect("the pass mints every match")
    })
}
