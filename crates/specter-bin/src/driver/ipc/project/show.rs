//! `specter show <name>` projection — four-way discrimination across engine-attached,
//! runtime-disabled, TOML-disabled, and Unknown.
//!
//! Resolution order matches the `disable` resolver in [`crate::driver::ipc`] so an operator who can
//! `show foo` can also `disable foo`. Dynamic Subs are addressed through `list -o json`, not
//! `show`: a synthesised name resolves through `SubRegistry::find_by_name` to a live id, but a
//! local guard at the lookup site returns `Unknown` for any discovery-minted Sub. The verb's
//! contract lives at its own callsite, not inside the index.

use std::collections::BTreeSet;
use std::time::Instant;

use compact_str::CompactString;
use specter_config::Config;
use specter_core::{Profile, Reaction, Sub, SubId};
use specter_engine::Engine;

use crate::driver::DriverState;
use crate::ipc::protocol::{DisabledSource, ShowResponse, SubDetails, WireId, WireReaction};
use crate::ipc::wire::{
    WireAbsorbMode, WireAbsorbWindow, WireEffectScope, WirePath, WireStateLabel, WireTime,
};

use super::{program, project_wall, settle_ms};

/// Resolve `name` and emit the matching [`ShowResponse`] arm.
///
/// Resolution is total:
/// 1. `engine.subs().find_by_name(name)` resolves to a Sub. If it is operator-declared
///    (`!is_dynamic()`, discovery templates included) → `Active`; if minted → `Unknown` (the verb's
///    contract: minted Subs are reached through `list`, not `show`). A minted-Sub hit
///    short-circuits to `Unknown` rather than falling through, because by the `@`-byte reservation
///    a synthesised name never appears in `disabled_runtime` or `config.watches`.
/// 2. `disabled_runtime.contains(name)?`  → `Disabled { Runtime }`
/// 3. `config.watches[*].name == name && !enabled` → `Disabled { Toml }`
/// 4. otherwise → `Unknown`
pub(crate) fn show(
    engine: &Engine,
    ds: &DriverState,
    disabled_runtime: &BTreeSet<CompactString>,
    config: &Config,
    name: &str,
    now: Instant,
) -> ShowResponse {
    if let Some(sid) = engine.subs().find_by_name(name) {
        let sub = engine
            .subs()
            .get(sid)
            .expect("by_name resolves to live SubId — registry lockstep invariant");
        if !sub.is_dynamic() {
            return ShowResponse::Active(project_details(sid, sub, engine, ds, now));
        }
        return ShowResponse::Unknown {
            name: name.to_string(),
        };
    }

    if disabled_runtime.contains(name) {
        return ShowResponse::Disabled {
            name: name.to_string(),
            source: DisabledSource::Runtime,
        };
    }

    if config.watches.iter().any(|s| s.name == name && !s.enabled) {
        return ShowResponse::Disabled {
            name: name.to_string(),
            source: DisabledSource::Toml,
        };
    }

    ShowResponse::Unknown {
        name: name.to_string(),
    }
}

/// Project a registry-attached `(SubId, &Sub)` into a [`SubDetails`].
///
/// Profile-lookup verdict: graceful, mirroring `super::list::project_attached`. Every attached Sub
/// has a Profile by engine construction (`SubRegistry::insert` runs in the same engine step as
/// `ProfileMap::attach`), but the IPC projection layer never panics on engine-invariant breach — it
/// surfaces `state: None` / `anchor: None` so the operator's introspection lifeline keeps serving
/// every other concurrent session during incidents. The engine's own panic discipline (the
/// `ProbeSlot` tripwire, the registry `debug_assert!`s) catches the breach loudly engine-side.
///
/// `anchor: Option<WirePath>`: `None` signals "anchor vanished / not yet resolved" (a Pending
/// descent in flight, an Unwatch event that hasn't reconciled). The shape mirrors
/// [`crate::ipc::protocol::ListRow::anchor`].
fn project_details(
    sid: SubId,
    sub: &Sub,
    engine: &Engine,
    ds: &DriverState,
    now: Instant,
) -> SubDetails {
    let profile = engine.profiles().get(sub.profile());
    let state = profile.map(|p| WireStateLabel::from(p.state().label()));
    let anchor = profile
        .and_then(|p| engine.tree().path_of(p.resource()))
        .map(|arc| WirePath::from(&arc));
    // Live-gate via the engine's own predicate: lazy expiry leaves an inert window resident in
    // Profile state, and `absorb_window_if_live` is the single owner of the `now < expiry` rule, so
    // the projection drops an inert window without re-stating the liveness test. The wall-clock
    // projection runs only for the live `Some` the operator sees.
    let absorb = profile
        .and_then(|p| p.absorb_window_if_live(now))
        .map(|w| WireAbsorbWindow {
            expiry: WireTime::from(project_wall(ds.start_wall, ds.start_instant, w.expiry)),
            mode: WireAbsorbMode::from(w.mode),
        });
    // Honest render across the reaction sum: a Spawn Sub reports its own payload and history; a
    // Mint Sub (discovery template) reports the knobs its minted Subs inherit plus the live minted
    // count — never fire stats, which a template has none of.
    let reaction = match sub.reaction() {
        Reaction::Spawn { spec, history, .. } => WireReaction::Spawn {
            scope: WireEffectScope::from(spec.scope()),
            program: program::render(spec.program()),
            last_fired_at: history
                .last_fired_at
                .map(|t| WireTime::from(project_wall(ds.start_wall, ds.start_instant, t))),
            fire_count: history.fire_count,
            dedup_suppressed_count: history.dedup_suppressed_count,
        },
        Reaction::Mint(t) => WireReaction::Mint {
            minted_settle_ms: settle_ms(t.spec.settle),
            minted_scope: WireEffectScope::from(t.spec.spawn.scope()),
            minted_program: program::render(t.spec.spawn.program()),
            // The discovery fan-out warning's scan, run at IPC cadence: O(total Subs).
            minted_live: engine
                .subs()
                .iter()
                .filter(|(_, s)| s.minted_by() == Some(sid))
                .count(),
        },
    };
    SubDetails {
        name: sub.name.to_string(),
        sub: WireId::from(sid),
        profile: WireId::from(sub.profile()),
        state,
        anchor,
        absorb,
        absorb_count: profile.map_or(0, Profile::absorb_count),
        settle_ms: settle_ms(sub.settle),
        minted_by: sub.minted_by().map(WireId::from),
        reaction,
    }
}

#[cfg(test)]
mod tests {
    use super::show;
    use crate::driver::DriverState;
    use crate::ipc::protocol::{DisabledSource, ShowResponse, WireReaction};
    use compact_str::CompactString;
    use specter_config::Config;
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, ClassSet, EffectScope, ExecAction, Input,
        ProfileIdentity, ScanConfig, SubAttachAnchor, SubAttachRequest, SubId, SubParams,
    };
    use specter_engine::Engine;
    use std::collections::BTreeSet;
    use std::fmt::Write as _;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    fn fresh_state() -> DriverState {
        DriverState::new(PathBuf::from("/tmp/specter-test.sock"))
    }

    fn config_from_watches(watches: &[(&str, &str, bool)]) -> Config {
        let mut s = String::new();
        for (name, path, enabled) in watches {
            let enabled = if *enabled { "true" } else { "false" };
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

    fn engine_with(config: &Config) -> Engine {
        let mut engine = Engine::new();
        let now = Instant::now();
        for spec in config.active_watches() {
            let _ = engine.step(Input::AttachSub(spec.to_attach_request()), now);
        }
        engine
    }

    /// RAII guard — see [`crate::driver::ipc::project::list::tests::EngineGuard`] for the
    /// linear-edge `ProbeSlot` rationale. Inlined here rather than reaching across the inline tests
    /// to keep the test modules self-contained.
    struct EngineGuard {
        engine: Option<Engine>,
    }

    impl EngineGuard {
        fn wrap(engine: Engine) -> Self {
            Self {
                engine: Some(engine),
            }
        }

        fn engine(&self) -> &Engine {
            self.engine.as_ref().expect("engine present until drop")
        }
    }

    impl Drop for EngineGuard {
        fn drop(&mut self) {
            if let Some(mut e) = self.engine.take() {
                let _ = e.cancel_all_in_flight_probes();
            }
        }
    }

    fn trivial_program() -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal("/bin/true")])],
            None,
        )));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// An engine-attached name returns `Active` with a populated `SubDetails`. Anchor is
    /// `Some(path)` for a freshly attached Sub (the engine descends synchronously when the path
    /// exists); `last_fired_at` is `None` until first fire.
    #[test]
    fn show_active_returns_full_details() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_from_watches(&[("watched", tmp.path().to_str().unwrap(), true)]);
        let guard = EngineGuard::wrap(engine_with(&config));

        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "watched",
            Instant::now(),
        );
        match r {
            ShowResponse::Active(d) => {
                assert_eq!(d.name, "watched");
                assert!(
                    d.state.is_some(),
                    "attached Sub's Profile lookup populates state",
                );
                assert!(d.anchor.is_some(), "attached Sub has resolved anchor path");
                assert!(d.minted_by.is_none(), "static Sub has no minted_by");
                match d.reaction {
                    WireReaction::Spawn {
                        program,
                        last_fired_at,
                        fire_count,
                        ..
                    } => {
                        assert_eq!(fire_count, 0, "never fired");
                        assert!(last_fired_at.is_none(), "never fired ⇒ None");
                        assert!(!program.is_empty(), "program renders ≥1 line");
                    }
                    other @ WireReaction::Mint { .. } => {
                        panic!("static watch projects Spawn, got {other:?}")
                    }
                }
            }
            other => panic!("expected Active, got {other:?}"),
        }
    }

    /// A name in the runtime-disabled set returns `Disabled { Runtime }`. The Sub is not in the
    /// engine; the config may or may not also list it (this test omits the TOML entry to pin the
    /// runtime branch in isolation).
    #[test]
    fn show_runtime_disabled() {
        let engine = Engine::new();
        let mut disabled = BTreeSet::new();
        disabled.insert(CompactString::const_new("paused"));
        let r = show(
            &engine,
            &fresh_state(),
            &disabled,
            &Config::from_str("").expect("empty"),
            "paused",
            Instant::now(),
        );
        match r {
            ShowResponse::Disabled { name, source } => {
                assert_eq!(name, "paused");
                assert_eq!(source, DisabledSource::Runtime);
            }
            other => panic!("expected Disabled(Runtime), got {other:?}"),
        }
    }

    /// A name with `enabled = false` in the TOML returns `Disabled { Toml }`. The runtime set is
    /// empty here so the TOML branch is reached in isolation.
    #[test]
    fn show_toml_disabled() {
        let config = config_from_watches(&[("off_by_toml", "/tmp/foo", false)]);
        let guard = EngineGuard::wrap(engine_with(&config)); // disabled ⇒ not attached
        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "off_by_toml",
            Instant::now(),
        );
        match r {
            ShowResponse::Disabled { name, source } => {
                assert_eq!(name, "off_by_toml");
                assert_eq!(source, DisabledSource::Toml);
            }
            other => panic!("expected Disabled(Toml), got {other:?}"),
        }
    }

    /// A name that appears nowhere returns `Unknown`. Operators chain `specter show foo &&
    /// do-thing` to gate on this.
    #[test]
    fn show_unknown_for_typo() {
        let engine = Engine::new();
        let r = show(
            &engine,
            &fresh_state(),
            &BTreeSet::new(),
            &Config::from_str("").expect("empty"),
            "ghost",
            Instant::now(),
        );
        match r {
            ShowResponse::Unknown { name } => assert_eq!(name, "ghost"),
            other => panic!("expected Unknown, got {other:?}"),
        }
    }

    /// With a name present in the engine AND the runtime-disabled set AND TOML-disabled, resolution
    /// returns Active — engine wins the precedence ladder. Pins the resolution order; the `disable`
    /// handler reuses the same ladder.
    #[test]
    fn show_engine_wins_over_disabled_sources_on_name_conflict() {
        let tmp = tempfile::TempDir::new().unwrap();
        // The TOML carries `engaged` enabled, so engine attaches it.
        let config = config_from_watches(&[("engaged", tmp.path().to_str().unwrap(), true)]);
        let guard = EngineGuard::wrap(engine_with(&config));
        let mut disabled = BTreeSet::new();
        // Hypothetical race: operator pushed `disable engaged` and the projection raced ahead of
        // the engine's detach.
        disabled.insert(CompactString::const_new("engaged"));

        let r = show(
            guard.engine(),
            &fresh_state(),
            &disabled,
            &config,
            "engaged",
            Instant::now(),
        );
        assert!(
            matches!(r, ShowResponse::Active(_)),
            "engine row wins precedence over the runtime-disabled set",
        );
    }

    /// An armed-and-live `absorb` window projects to `SubDetails.absorb = Some(..)` with the
    /// matching mode, and an inert (expired) window projects to `None` — the projection's `expiry >
    /// now` live-gate. `absorb_count` projects independently (0 here — arming does not fold).
    /// Arming with `Some(duration)` yields `PersistUntil`; the window's `expiry` is `arm_now +
    /// duration`, so a `now` before it is live and a `now` past it is inert.
    #[test]
    fn show_absorb_window_live_gated_by_now() {
        let tmp = tempfile::TempDir::new().unwrap();
        let config = config_from_watches(&[("watched", tmp.path().to_str().unwrap(), true)]);
        let mut engine = engine_with(&config);
        let sid = engine
            .subs()
            .find_by_name("watched")
            .expect("watched attached");
        let profile = engine.subs().get(sid).expect("live sub").profile();

        let arm_now = Instant::now();
        let window = Duration::from_mins(1);
        let _ = engine.step(
            Input::ArmAbsorb {
                profile,
                duration: Some(window),
            },
            arm_now,
        );

        let guard = EngineGuard::wrap(engine);

        // Before expiry ⇒ Some(PersistUntil).
        let live = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "watched",
            arm_now,
        );
        match live {
            ShowResponse::Active(d) => {
                let w = d.absorb.as_ref().expect("live window projects Some");
                assert_eq!(
                    w.mode,
                    crate::ipc::wire::WireAbsorbMode::PersistUntil,
                    "Some(duration) ⇒ PersistUntil",
                );
                assert_eq!(d.absorb_count, 0, "arming folds nothing");
            }
            other => panic!("expected Active, got {other:?}"),
        }

        // Past expiry ⇒ None (the projection drops the inert window).
        let inert = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "watched",
            arm_now + window + Duration::from_secs(1),
        );
        match inert {
            ShowResponse::Active(d) => assert!(
                d.absorb.is_none(),
                "inert window (expiry <= now) projects None: {:?}",
                d.absorb,
            ),
            other => panic!("expected Active, got {other:?}"),
        }
    }

    /// A minted Sub's synthesised name resolves through `SubRegistry::find_by_name` to a live id,
    /// but the verb's local guard maps `is_dynamic()` back to `Unknown` — preserving the operator
    /// contract that minted Subs are addressed through `list`, not `show`.
    #[test]
    fn show_dynamic_sub_name_resolves_unknown() {
        let req = SubAttachRequest::from_parts(
            SubAttachAnchor::Path(PathBuf::from("/tmp/dyn_anchor")),
            ProfileIdentity::new(
                ScanConfig::builder().build(),
                Duration::from_hours(1),
                ClassSet::DEFAULT_SUBTREE_ROOT,
            ),
            SubParams::minted(
                CompactString::const_new("template@/tmp/dyn_anchor"),
                trivial_program(),
                EffectScope::SubtreeRoot,
                Duration::from_millis(100),
                false,
                SubId::default(),
            ),
        );
        let mut engine = Engine::new();
        let _ = engine.step(Input::AttachSub(req), Instant::now());

        // The dynamic Sub IS in the registry — and now `by_name` indexes it. The Unknown verdict
        // comes from the guard at the show callsite, not from a `None` lookup.
        assert_eq!(engine.subs().len(), 1, "Sub did land in the registry");
        assert!(
            engine
                .subs()
                .find_by_name("template@/tmp/dyn_anchor")
                .is_some(),
            "registry indexes the dynamic name (load-bearing precondition for the guard test)",
        );
        let guard = EngineGuard::wrap(engine);
        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &Config::from_str("").expect("empty"),
            "template@/tmp/dyn_anchor",
            Instant::now(),
        );
        match r {
            ShowResponse::Unknown { name } => {
                assert_eq!(name, "template@/tmp/dyn_anchor");
            }
            other => panic!("expected Unknown for dynamic Sub name, got {other:?}"),
        }
    }

    /// A discovery **template** is operator-declared (`!is_dynamic()`), so `show` resolves it
    /// `Active` like any static watch — only the minted set maps to `Unknown`. The fixture goes
    /// through the real config lowering: a glob-bearing path lowers to a template-bearing Sub.
    #[test]
    fn show_discovery_template_returns_active() {
        let tmp = tempfile::TempDir::new().unwrap();
        let pattern = format!("{}/{{a,b}}/access.log", tmp.path().display());
        let config = config_from_watches(&[("disc", &pattern, true)]);
        let guard = EngineGuard::wrap(engine_with(&config));
        let sid = guard
            .engine()
            .subs()
            .find_by_name("disc")
            .expect("template attached");
        assert!(
            guard
                .engine()
                .subs()
                .get(sid)
                .expect("live sub")
                .is_template(),
            "fixture lowered to a template (load-bearing precondition)",
        );

        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "disc",
            Instant::now(),
        );
        match r {
            ShowResponse::Active(d) => {
                assert_eq!(d.name, "disc");
                assert!(
                    d.minted_by.is_none(),
                    "a template is operator-declared, not minted",
                );
                match d.reaction {
                    WireReaction::Mint {
                        minted_live,
                        minted_program,
                        ..
                    } => {
                        assert_eq!(minted_live, 0, "nothing minted yet");
                        assert!(
                            !minted_program.is_empty(),
                            "template's minted program renders ≥1 line",
                        );
                    }
                    other @ WireReaction::Spawn { .. } => {
                        panic!("template projects Mint, got {other:?}")
                    }
                }
            }
            other => panic!("expected Active for a discovery template, got {other:?}"),
        }
    }

    /// `minted_live` counts the template's live minted Subs — the registry scan keyed on
    /// `minted_by`, so a hand-attached minted Sub (the projection-side stand-in for a discovery
    /// reconcile's mint) moves the counter while an unrelated template stays at zero.
    #[test]
    fn show_template_minted_live_counts_its_minted_subs() {
        let tmp = tempfile::TempDir::new().unwrap();
        let pattern = format!("{}/{{a,b}}/access.log", tmp.path().display());
        let config = config_from_watches(&[("disc", &pattern, true)]);
        let mut engine = engine_with(&config);
        let template_sid = engine
            .subs()
            .find_by_name("disc")
            .expect("template attached");

        let req = SubAttachRequest::from_parts(
            SubAttachAnchor::Path(PathBuf::from("/tmp/dyn_anchor")),
            ProfileIdentity::new(
                ScanConfig::builder().build(),
                Duration::from_hours(1),
                ClassSet::DEFAULT_SUBTREE_ROOT,
            ),
            SubParams::minted(
                CompactString::const_new("disc@/tmp/dyn_anchor"),
                trivial_program(),
                EffectScope::SubtreeRoot,
                Duration::from_millis(100),
                false,
                template_sid,
            ),
        );
        let _ = engine.step(Input::AttachSub(req), Instant::now());
        let guard = EngineGuard::wrap(engine);

        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "disc",
            Instant::now(),
        );
        match r {
            ShowResponse::Active(d) => match d.reaction {
                WireReaction::Mint { minted_live, .. } => {
                    assert_eq!(minted_live, 1, "one live minted Sub attributes to disc");
                }
                other @ WireReaction::Spawn { .. } => {
                    panic!("template projects Mint, got {other:?}")
                }
            },
            other => panic!("expected Active for a discovery template, got {other:?}"),
        }
    }
}
