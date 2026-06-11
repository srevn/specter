//! `specter list` projection — union three populations (engine- attached, runtime-disabled,
//! TOML-disabled) into one alphabetically-sorted row set.
//!
//! [`BTreeMap`]-keyed by owned `String`: insertion-time conflict-resolution (engine wins) AND
//! alphabetic iteration are both structural to the data structure, not the algorithm. The map
//! allocates once per request and drops on send; the cost is operator-paced (single requests,
//! human-latency tolerance), not per-tick.

use std::collections::{BTreeMap, BTreeSet};

use compact_str::CompactString;
use specter_config::Config;
use specter_core::{Sub, SubId};
use specter_engine::Engine;

use crate::driver::DriverState;
use crate::ipc::protocol::{DisabledSource, ListResponse, ListRow, WireId};
use crate::ipc::wire::{WirePath, WireReactionKind, WireStateLabel, WireTime};

use super::{project_wall, settle_ms};

/// Project the three populations into a [`ListResponse`].
///
/// Order of insertion mirrors the engine-wins conflict-resolution rule (a name in the engine *and*
/// the runtime-disabled set can only race during an in-flight IPC `disable`):
/// 1. Engine-attached rows first (every Sub in the registry — static AND dynamic).
///    [`BTreeMap::insert`] semantics overwrite by key, but no other insertion follows for the same
///    name; the engine row is the authoritative one.
/// 2. Runtime-disabled rows — <code>[BTreeMap::entry].or_insert_with</code> skips when the engine
///    already claimed the name.
/// 3. TOML-disabled rows — same [`BTreeMap::entry`] rule.
///
/// Owned `String` keys (not `&str`): each source has a different borrow lifetime (engine's
/// `&Sub.name`, the runtime set's `&CompactString`, the config's `&SubSpec.name`). An owned key
/// outlives every borrow and round-trips into [`ListRow::name`] without a second allocation —
/// [`BTreeMap::into_values`] consumes the map.
pub(crate) fn list(
    engine: &Engine,
    ds: &DriverState,
    disabled_runtime: &BTreeSet<CompactString>,
    config: &Config,
) -> ListResponse {
    let mut rows: BTreeMap<String, ListRow> = BTreeMap::new();

    for (sid, sub) in engine.subs().iter() {
        rows.insert(sub.name.to_string(), project_attached(sid, sub, engine, ds));
    }

    for name in disabled_runtime {
        rows.entry(name.to_string())
            .or_insert_with(|| disabled_row(name.as_str(), DisabledSource::Runtime));
    }

    for spec in config.watches.iter().filter(|s| !s.enabled) {
        rows.entry(spec.name.to_string())
            .or_insert_with(|| disabled_row(spec.name.as_str(), DisabledSource::Toml));
    }

    ListResponse {
        rows: rows.into_values().collect(),
    }
}

/// Project a registry-attached `(SubId, &Sub)` into a [`ListRow`].
///
/// Looks up the hosting Profile and anchor path. Both lookups must succeed for an attached Sub: a
/// missing Profile or unresolved anchor is an engine invariant breach. The projection surfaces the
/// partial row (`state: None` / `anchor: None`) rather than panicking — the breach is
/// operator-visible, not hidden behind a crash.
fn project_attached(sid: SubId, sub: &Sub, engine: &Engine, ds: &DriverState) -> ListRow {
    let profile = engine.profiles().get(sub.profile());
    let state = profile.map(|p| WireStateLabel::from(p.state().label()));
    let anchor = profile
        .and_then(|p| engine.tree().path_of(p.resource()))
        .map(|arc| WirePath::from(&arc));
    // Honest render across the reaction sum: a Mint Sub (discovery template) has no fire history
    // — its stat columns stay `None`, attributed by the `reaction` discriminator.
    let history = sub.fire_history();
    let last_fired_at = history
        .and_then(|h| h.last_fired_at)
        .map(|t| WireTime::from(project_wall(ds.start_wall, ds.start_instant, t)));
    ListRow {
        name: sub.name.to_string(),
        state,
        anchor,
        last_fired_at,
        fire_count: history.map(|h| h.fire_count),
        dedup_suppressed_count: history.map(|h| h.dedup_suppressed_count),
        settle_ms: Some(settle_ms(sub.settle)),
        reaction: Some(WireReactionKind::from(sub.reaction())),
        disabled: None,
        sub: Some(WireId::from(sid)),
        profile: Some(WireId::from(sub.profile())),
        minted_by: sub.minted_by().map(WireId::from),
    }
}

/// Construct a row for a non-attached Sub (runtime-disabled or TOML-disabled). Every engine-derived
/// field is `None`; only `name` and `disabled` are set.
fn disabled_row(name: &str, source: DisabledSource) -> ListRow {
    ListRow {
        name: name.to_string(),
        state: None,
        anchor: None,
        last_fired_at: None,
        fire_count: None,
        dedup_suppressed_count: None,
        settle_ms: None,
        reaction: None,
        disabled: Some(source),
        sub: None,
        profile: None,
        minted_by: None,
    }
}

#[cfg(test)]
mod tests {
    use super::list;
    use crate::driver::DriverState;
    use crate::ipc::protocol::DisabledSource;
    use crate::ipc::wire::WireReactionKind;
    use compact_str::CompactString;
    use specter_config::Config;
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::{ActionProgram, ArgPart, ArgTemplate, ExecAction, Input};
    use specter_engine::Engine;
    use std::collections::BTreeSet;
    use std::fmt::Write as _;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::time::Instant;

    fn fresh_state() -> DriverState {
        DriverState::new(PathBuf::from("/tmp/specter-test.sock"))
    }

    /// Build a Config from a TOML string keyed by anchor path so the tests can stage many watches
    /// at different paths. The path itself is irrelevant to the projection — the engine descends
    /// but never resolves it in these tests.
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

    /// Attach every active watch in `config` through `engine.step`. The returned engine carries one
    /// Sub per active watch, in source order. `StepOutput`s are discarded — the projection reads
    /// `engine.subs()` directly.
    fn engine_with(config: &Config) -> Engine {
        let mut engine = Engine::new();
        let now = Instant::now();
        for spec in config.active_watches() {
            let _ = engine.step(Input::AttachSub(spec.to_attach_request()), now);
        }
        engine
    }

    /// RAII guard that drains every in-flight probe before the wrapped `Engine` drops. Without it
    /// `ProbeSlot`'s linear-edge tripwire (`specter_core::probe`) panics on test teardown — every
    /// attach arms a probe on the descent / Seed path.
    ///
    /// Tests construct the engine, hand it to [`Self::wrap`], project off [`Self::engine`], and
    /// `drop(guard)` cleans up.
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

    /// All three sources contribute rows; output is name-keyed and alphabetic.
    #[test]
    fn list_unions_engine_disabled_runtime_toml() {
        let config = config_from_watches(&[
            ("attached_a", "/tmp/a", true),
            ("toml_off", "/tmp/b", false),
        ]);
        let guard = EngineGuard::wrap(engine_with(&config));
        let mut disabled = BTreeSet::new();
        disabled.insert(CompactString::const_new("runtime_off"));

        let resp = list(guard.engine(), &fresh_state(), &disabled, &config);

        assert_eq!(
            resp.rows.len(),
            3,
            "one attached, one runtime-disabled, one toml-disabled — got {}",
            resp.rows.len(),
        );
        // Alphabetic order is structural via BTreeMap.
        let names: Vec<_> = resp.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(names, vec!["attached_a", "runtime_off", "toml_off"]);

        // Field discrimination by source.
        let attached = &resp.rows[0];
        assert!(
            attached.disabled.is_none(),
            "engine row carries no disabled"
        );
        assert!(attached.state.is_some(), "engine row carries state");
        assert!(attached.sub.is_some(), "engine row carries SubId");

        let runtime = &resp.rows[1];
        assert_eq!(runtime.disabled, Some(DisabledSource::Runtime));
        assert!(runtime.state.is_none());
        assert!(runtime.sub.is_none());

        let toml = &resp.rows[2];
        assert_eq!(toml.disabled, Some(DisabledSource::Toml));
        assert!(toml.state.is_none());
        assert!(toml.sub.is_none());
    }

    /// On name conflict between engine and the runtime-disabled set (the operator disable IPC raced
    /// with the projection), the engine row wins — its insertion comes first and the runtime row's
    /// `or_insert_with` is a no-op.
    #[test]
    fn list_engine_wins_on_name_conflict_with_runtime_set() {
        let config = config_from_watches(&[("alpha", "/tmp/a", true)]);
        let guard = EngineGuard::wrap(engine_with(&config));
        let mut disabled = BTreeSet::new();
        // Same name in the runtime-disabled set — should NOT shadow the engine row.
        disabled.insert(CompactString::const_new("alpha"));

        let resp = list(guard.engine(), &fresh_state(), &disabled, &config);

        assert_eq!(resp.rows.len(), 1, "engine row absorbs the conflict");
        let row = &resp.rows[0];
        assert_eq!(row.name, "alpha");
        assert!(
            row.disabled.is_none(),
            "engine row wins; disabled stays None"
        );
        assert!(row.sub.is_some(), "engine row carries SubId");
    }

    /// Insertion order across sources does not break sort — the BTreeMap reorders
    /// deterministically. Construct in reverse and observe alphabetic output.
    #[test]
    fn list_alphabetic_order_across_sources() {
        // Source order: TOML first (disabled), then runtime, then engine.
        let config = config_from_watches(&[
            ("zebra_toml", "/tmp/z", false),
            ("alpha_attached", "/tmp/a", true),
        ]);
        let guard = EngineGuard::wrap(engine_with(&config));
        let mut disabled = BTreeSet::new();
        disabled.insert(CompactString::const_new("middle_runtime"));

        let resp = list(guard.engine(), &fresh_state(), &disabled, &config);
        let names: Vec<_> = resp.rows.iter().map(|r| r.name.as_str()).collect();
        assert_eq!(
            names,
            vec!["alpha_attached", "middle_runtime", "zebra_toml"],
            "alphabetic regardless of source order",
        );
    }

    /// Attached row carries every engine-derived field: state, sub id, profile id, fire counters
    /// (`Some(0)` for a never-fired Sub), settle ms, and a null `minted_by` for static Subs.
    #[test]
    fn list_attached_row_carries_every_engine_field() {
        let config = config_from_watches(&[("only", "/tmp/foo", true)]);
        let guard = EngineGuard::wrap(engine_with(&config));
        let resp = list(guard.engine(), &fresh_state(), &BTreeSet::new(), &config);
        let row = &resp.rows[0];

        assert_eq!(row.name, "only");
        assert!(row.state.is_some(), "attached row carries state");
        assert!(row.sub.is_some(), "attached row carries SubId");
        assert!(row.profile.is_some(), "attached row carries ProfileId");
        assert_eq!(
            row.fire_count,
            Some(0),
            "never-fired Sub carries Some(0), not None",
        );
        assert_eq!(row.dedup_suppressed_count, Some(0));
        assert!(row.settle_ms.is_some(), "attached row carries settle_ms");
        assert_eq!(
            row.reaction,
            Some(WireReactionKind::Spawn),
            "static Sub discriminates as spawn",
        );
        assert!(row.minted_by.is_none(), "static Sub has no minted_by");
        assert!(row.disabled.is_none(), "attached row's disabled is None");
    }

    /// A discovery template's row discriminates as `mint` and carries `None` fire stats — a
    /// template never fires, so there is no history whose counters could move; the `reaction`
    /// column attributes the n/a (distinguishing it from a non-attached row's `None`s).
    #[test]
    fn list_template_row_discriminates_mint_with_no_fire_stats() {
        let tmp = tempfile::TempDir::new().unwrap();
        let pattern = format!("{}/{{a,b}}/access.log", tmp.path().display());
        let config = config_from_watches(&[("disc", &pattern, true)]);
        let guard = EngineGuard::wrap(engine_with(&config));

        let resp = list(guard.engine(), &fresh_state(), &BTreeSet::new(), &config);
        assert_eq!(resp.rows.len(), 1);
        let row = &resp.rows[0];
        assert_eq!(row.name, "disc");
        assert_eq!(row.reaction, Some(WireReactionKind::Mint));
        assert_eq!(row.fire_count, None, "a template has no fire count");
        assert_eq!(row.dedup_suppressed_count, None);
        assert!(row.last_fired_at.is_none());
        assert!(row.state.is_some(), "the template row is attached");
        assert!(row.sub.is_some(), "attached ⇒ SubId present");
    }

    /// A dynamic Sub (`minted_by: Some(_)`) projects with the discriminator populated.
    /// Construct it by directly attaching a `SubAttachRequest` whose `minted_by` is
    /// `Some(_)` — the engine's `attach_sub` does not require the source template to exist for this
    /// projection-side test.
    #[test]
    fn list_dynamic_sub_carries_minted_by() {
        use specter_core::{ClassSet, ProfileIdentity};
        use specter_core::{
            EffectScope, ScanConfig, SubAttachAnchor, SubAttachRequest, SubId, SubParams,
        };
        use std::time::Duration;

        let program = trivial_program();
        let req = SubAttachRequest::from_parts(
            SubAttachAnchor::Path(PathBuf::from("/tmp/dyn_anchor")),
            ProfileIdentity::new(
                ScanConfig::builder().build(),
                Duration::from_hours(1),
                ClassSet::DEFAULT_SUBTREE_ROOT,
            ),
            SubParams::minted(
                CompactString::const_new("template@/tmp/dyn_anchor"),
                program,
                EffectScope::SubtreeRoot,
                Duration::from_millis(100),
                false,
                SubId::default(),
            ),
        );
        let mut engine = Engine::new();
        let _ = engine.step(Input::AttachSub(req), Instant::now());
        let guard = EngineGuard::wrap(engine);

        let resp = list(
            guard.engine(),
            &fresh_state(),
            &BTreeSet::new(),
            &Config::from_str("").expect("empty config"),
        );
        assert_eq!(resp.rows.len(), 1);
        let row = &resp.rows[0];
        assert!(row.minted_by.is_some(), "dynamic Sub must carry minted_by");
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
}
