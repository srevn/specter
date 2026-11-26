//! `specter show <name>` projection — four-way discrimination across engine-attached,
//! runtime-disabled, TOML-disabled, and Unknown.
//!
//! An `Active` answer names a watch the engine holds, which `disable` admits as well
//! ([`crate::driver::ipc`]), so an operator who can `show foo` can also `disable foo`. Minted Subs
//! are addressed through `list -o json`, not `show`: a synthesised name resolves through
//! `SubRegistry::find_by_name` to a live id, but a local guard at the lookup site returns `Unknown`
//! for any minted Sub. The verb's contract lives at its own callsite, not inside the index.

use std::collections::BTreeSet;
use std::time::Instant;

use compact_str::CompactString;
use specter_config::Config;
use specter_core::{AbsorbWindow, Profile, Reaction, Sub, SubId, SubParams};
use specter_engine::Engine;

use crate::driver::DriverState;
use crate::ipc::protocol::{DisabledSource, ShowResponse, SubDetails, WireId, WireReaction};
use crate::ipc::wire::{
    WireAbsorbMode, WireAbsorbWindow, WireEffectScope, WireMintedAbsorbing, WirePath,
    WireStateLabel,
};

use super::{program, project_wall, settle_ms};

/// Resolve `name` and emit the matching [`ShowResponse`] arm.
///
/// Resolution is total:
/// 1. `engine.subs().find_by_name(name)` resolves to a Sub. If a request declared it (no `minted_by`,
///    templates included) → `Active`; if a template minted it → `Unknown` (the verb's contract:
///    minted Subs are reached through `list`, not `show`). A minted-Sub hit short-circuits to
///    `Unknown` rather than falling through, because by the `@`-byte reservation a synthesised name
///    never appears in `disabled_runtime` or `config.watches`.
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
        if sub.minted_by().is_none() {
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
/// `anchor: Option<WirePath>`: `None` signals "anchor vanished / not yet resolved" (a Pending descent
/// in flight, an Unwatch event that hasn't reconciled). The shape mirrors
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
    let reaction = match sub.reaction() {
        Reaction::Spawn {
            params: SubParams { spawn, .. },
            history,
            ..
        } => WireReaction::Spawn {
            scope: WireEffectScope::from(spawn.scope()),
            program: program::render(spawn.program()),
            last_fired_at: history.last_fired_at.map(|t| project_wall(ds, t)),
            fire_count: history.fire_count,
            dedup_suppressed_count: history.dedup_suppressed_count,
            absorb: profile
                .and_then(|p| p.absorb_window_if_live(now))
                .map(|w| wire_window(ds, w)),
            absorb_count: profile.map_or(0, Profile::absorb_count),
        },
        Reaction::Mint { spec, .. } => WireReaction::Mint {
            minted_settle_ms: settle_ms(spec.params().settle),
            minted_scope: WireEffectScope::from(spec.params().spawn.scope()),
            minted_program: program::render(spec.params().spawn.program()),
            minted_live: engine.subs().minted(sid).len(),
            minted_absorbing: minted_absorbing(engine, sid, ds, now),
        },
    };
    SubDetails {
        name: sub.name.to_string(),
        sub: WireId::from(sid),
        profile: WireId::from(sub.profile()),
        state,
        anchor,
        settle_ms: settle_ms(sub.settle()),
        minted_by: sub.minted_by().map(WireId::from),
        reaction,
    }
}

/// How many of `template`'s matches hold a live `absorb` window
/// ([`specter_core::SubRegistry::firing_profiles`]), and the window that lapses last; `None` while
/// none does.
fn minted_absorbing(
    engine: &Engine,
    template: SubId,
    ds: &DriverState,
    now: Instant,
) -> Option<WireMintedAbsorbing> {
    let live = || {
        engine
            .subs()
            .firing_profiles(template)
            .filter_map(|p| engine.profiles().get(p)?.absorb_window_if_live(now))
    };
    let last = live().max_by_key(|w| w.expiry)?;
    Some(WireMintedAbsorbing {
        armed: live().count(),
        last: wire_window(ds, last),
    })
}

fn wire_window(ds: &DriverState, window: &AbsorbWindow) -> WireAbsorbWindow {
    WireAbsorbWindow {
        expiry: project_wall(ds, window.expiry),
        mode: WireAbsorbMode::from(window.mode),
    }
}

#[cfg(test)]
mod tests {
    use super::{project_wall, show};
    use crate::ipc::protocol::{DisabledSource, ShowResponse, SubDetails, WireReaction};
    use crate::ipc::wire::{WireAbsorbMode, WireAbsorbWindow};
    use crate::testkit::{EngineGuard, config_from_watches, engine_with, fresh_state, mint};
    use compact_str::CompactString;
    use specter_config::Config;
    use specter_core::Input;
    use specter_engine::Engine;
    use std::collections::BTreeSet;
    use std::time::{Duration, Instant};

    /// An engine-attached name returns `Active` with a populated `SubDetails`. Anchor is `Some(path)`
    /// for a freshly attached Sub (the engine descends synchronously when the path exists);
    /// `last_fired_at` is `None` until first fire.
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

    /// A name that appears nowhere returns `Unknown`. Operators chain `specter show foo && do-thing`
    /// to gate on this.
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
    /// returns Active — engine wins the precedence ladder. Pins the resolution order.
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

    /// An armed-and-live `absorb` window projects to the spawn payload's `absorb = Some(..)` with
    /// the matching mode, and an inert (expired) window projects to `None` — the projection's
    /// `expiry > now` live-gate. `absorb_count` projects independently (0 here — arming does not
    /// fold). Arming with `Some(duration)` yields `PersistUntil`; the window's `expiry` is `arm_now +
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

        let arm_now = Instant::now();
        let window = Duration::from_mins(1);
        let _ = engine.step(
            Input::ArmAbsorb {
                sub: sid,
                duration: Some(window),
            },
            arm_now,
        );

        let guard = EngineGuard::wrap(engine);
        let spawn_absorb = |now| match show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &config,
            "watched",
            now,
        ) {
            ShowResponse::Active(SubDetails {
                reaction:
                    WireReaction::Spawn {
                        absorb,
                        absorb_count,
                        ..
                    },
                ..
            }) => (absorb, absorb_count),
            other => panic!("expected a spawn row, got {other:?}"),
        };

        // Before expiry ⇒ Some(PersistUntil).
        let (absorb, absorb_count) = spawn_absorb(arm_now);
        assert_eq!(
            absorb.expect("live window projects Some").mode,
            WireAbsorbMode::PersistUntil,
            "Some(duration) ⇒ PersistUntil",
        );
        assert_eq!(absorb_count, 0, "arming folds nothing");

        // Past expiry ⇒ None (the projection drops the inert window).
        let (inert, _) = spawn_absorb(arm_now + window + Duration::from_secs(1));
        assert!(
            inert.is_none(),
            "inert window (expiry <= now) projects None: {inert:?}"
        );
    }

    /// A template's row sums up the live windows across its matches: of three windows, one past its
    /// expiry, two are counted and the one that lapses last is shown; past every expiry there is no
    /// summary.
    #[test]
    fn show_template_sums_up_its_matches_live_windows() {
        let mut engine = Engine::new();
        let [a, b, c] = mint(&mut engine, ["a", "b", "c"]);
        let armed_at = Instant::now();
        for (sub, window) in [
            (a, Duration::from_mins(1)),
            (b, Duration::from_mins(2)),
            (c, Duration::from_secs(1)),
        ] {
            let _ = engine.step(
                Input::ArmAbsorb {
                    sub,
                    duration: Some(window),
                },
                armed_at,
            );
        }
        let guard = EngineGuard::wrap(engine);
        let ds = fresh_state();
        let empty = Config::from_str("").expect("empty");
        let minted_absorbing =
            |now| match show(guard.engine(), &ds, &BTreeSet::new(), &empty, "disc", now) {
                ShowResponse::Active(SubDetails {
                    reaction:
                        WireReaction::Mint {
                            minted_absorbing, ..
                        },
                    ..
                }) => minted_absorbing,
                other => panic!("expected a mint row, got {other:?}"),
            };

        let summary = minted_absorbing(armed_at + Duration::from_secs(2)).expect("two are live");
        assert_eq!(summary.armed, 2, "a window past its expiry is not counted");
        assert_eq!(
            summary.last,
            WireAbsorbWindow {
                expiry: project_wall(&ds, armed_at + Duration::from_mins(2)),
                mode: WireAbsorbMode::PersistUntil,
            },
            "the window that lapses last",
        );
        assert!(
            minted_absorbing(armed_at + Duration::from_mins(3)).is_none(),
            "no summary once every window has lapsed",
        );
    }

    /// A minted Sub's synthesised name resolves through `SubRegistry::find_by_name` to a live id,
    /// but the verb's local guard maps its `minted_by` back to `Unknown` — preserving the operator
    /// contract that minted Subs are addressed through `list`, not `show`.
    #[test]
    fn show_minted_sub_name_resolves_unknown() {
        let mut engine = Engine::new();
        let [minted] = mint(&mut engine, ["a"]);
        let name = engine.subs().get(minted).expect("live").name.to_string();
        let guard = EngineGuard::wrap(engine);

        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &Config::from_str("").expect("empty"),
            &name,
            Instant::now(),
        );
        match r {
            ShowResponse::Unknown { name: unknown } => assert_eq!(unknown, name),
            other => panic!("expected Unknown for a minted Sub's name, got {other:?}"),
        }
    }

    /// A discovery **template** is operator-declared (no `minted_by`), so `show` resolves it `Active`
    /// like any static watch — only the minted set maps to `Unknown`. The fixture goes through the
    /// real config lowering: a glob-bearing path lowers to a template.
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

    /// `minted_live` is the length of the template's minted set.
    #[test]
    fn show_template_minted_live_counts_its_minted_subs() {
        let mut engine = Engine::new();
        let _ = mint(&mut engine, ["a"]);
        let guard = EngineGuard::wrap(engine);

        let r = show(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &Config::from_str("").expect("empty"),
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
