//! Per-input dispatch tests. Each `(state, input)` cell of the transition
//! table gets a focused test. Goes hand-in-hand with the integration suite
//! at `tests/integration.rs` which covers full-lifecycle flows.

// Tests prioritize readability over the workspace's pedantic style budget.
#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::single_match_else,
    clippy::too_many_lines,
    clippy::unnecessary_wraps
)]

use crate::Engine;
use compact_str::CompactString;
use specter_core::program::SpawnBody;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ActiveBurst, AnchorClaim, ArgPart, ArgTemplate, BurstFinish, BurstIntent,
    ChildEntry, ClaimKind, ClassSet, DedupKey, Diagnostic, DirChild, DirMeta, DirSnapshot,
    EffectOutcome, EffectScope, EntryKind, FS_ROOT_SEGMENT, FsEvent, FsIdentity, Input, LeafEntry,
    OverflowScope, PatternSpec, Placeholder, PostFireBurst, PostFirePhase, PreFireBurst,
    PreFirePhase, ProbeOp, ProbeOutcome, ProbeOwner, ProbeRequest, ProbeResponse, ProfileState,
    PromoterAttachRequest, PromoterId, PromoterRegistryDiff, PromoterState, ResourceId,
    ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachRequest, SubId, TimerKind,
    TreeSnapshot, WatchOp, WatchRegistryDiff,
};
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
/// Default events mask for transition tests. Empty mask gives a Profile
/// with `has_per_file_fds = false`, matching the prior tests' assumption
/// that per-file FDs are out of scope unless the test specifically opts
/// into PerStableFile + a CONTENT/METADATA mask. Events fold into
/// `config_hash` so two test fixtures differing only on `events` fork
/// separate Profiles (intentional partition).
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn diff_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([
        ArgPart::literal("fmt"),
        ArgPart::Placeholder(Placeholder::Created),
    ])])
}

/// Engine + Sub attached at `/anchor` (Dir, recursive). Returns the
/// engine, `ProfileId`, `SubId`.
fn engine_with_attached_sub() -> (
    Engine,
    specter_core::ProfileId,
    specter_core::SubId,
    ResourceId,
    Instant,
) {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    (e, pid, sid, r, now)
}

/// V5-native test helper: build a `TreeSnapshot::Dir` with the supplied
/// single-component children. Each child is `(name, EntryKind, inode)`;
/// Dirs are emitted with `subtree: None` (uncovered). Tests that need
/// nested subtrees should use `dir_with_subtree`. Returns
/// `Arc<DirSnapshot>` directly — the typed `ProbeOutcome::SubtreeOk`
/// variant carries an `Arc<DirSnapshot>`, not a wrapping `TreeSnapshot`.
fn dir_tree_snap(children: Vec<(&str, EntryKind, u64)>) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild {
                fs_id: FsIdentity { inode, device: 0 },
                subtree: None,
            }),
            _ => ChildEntry::Leaf(LeafEntry::new(
                kind,
                0,
                UNIX_EPOCH,
                FsIdentity { inode, device: 0 },
            )),
        };
        map.insert(CompactString::new(name), child);
    }
    Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: UNIX_EPOCH,
            fs_id: FsIdentity {
                inode: 0,
                device: 0,
            },
        },
        0,
        map,
    ))
}

/// `LeafEntry` for File-anchored Profiles. Consumed directly by
/// `ProbeOutcome::AnchorOk`; the wrapping `TreeSnapshot::File` lives on
/// the engine-internal `Profile.current`, not the wire response.
#[allow(dead_code)]
fn file_tree_snap(kind: EntryKind, size: u64, mtime: SystemTime, inode: u64) -> LeafEntry {
    LeafEntry::new(kind, size, mtime, FsIdentity { inode, device: 0 })
}

/// Drive the Profile from fresh-attach through Seed-Ok → Idle (post-Seed
/// state). After this, Profile.current and Profile.baseline are set.
fn complete_seed_burst(e: &mut Engine, pid: specter_core::ProfileId) {
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let snap = dir_tree_snap(vec![]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
}

// ---- attach_sub ----

#[test]
fn attach_sub_fresh_profile_emits_watch_suppress_probe() {
    let (e, _pid, _sid, r, _now) = engine_with_attached_sub();
    // After attach: anchor watch_demand=1, suppress_count=1, Profile is
    // Active(Seed Verifying).
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);
    assert_eq!(e.tree.get(r).unwrap().suppress_count(), 1);
}

/// `Profile.kind` is the cached witness of the anchor's classification:
/// `transition_to_verifying`'s probe-target dispatch and
/// `emit_effects`'s `compute_cwd` dispatch read this rather than
/// re-deriving the kind from the Tree on every call. A resource-based
/// attach against a kind-classified slot must populate the field at the
/// `attach_sub_inner` post-`Profile::new` write.
#[test]
fn attach_sub_caches_anchor_kind_for_classified_resource() {
    let (e, pid, _sid, _r, _now) = engine_with_attached_sub();
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        Some(ResourceKind::Dir),
        "resource-based attach reads the classified anchor's kind into Profile.kind",
    );
}

/// Resource-based attach against an `Unknown` slot leaves `Profile.kind
/// = None` until the first probe response classifies the anchor. The
/// `dispatch_seed_ok` fallback writes the field from the response shape
/// — the rare unprobed-attach path's only signal of the anchor's
/// classification.
#[test]
fn attach_sub_unprobed_anchor_seeds_kind_on_first_response() {
    let mut e = Engine::new();
    // Resource exists but kind is left Unknown — the rare path where a
    // caller passes a resource-based attach against a freshly-`ensure`'d
    // slot whose kind hasn't been classified by any prior probe.
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;

    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    // Drive the Seed-Ok with a Dir-shaped response. The fallback in
    // `dispatch_seed_ok` should pick the kind off the response shape.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight");
    let snap = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        Some(ResourceKind::Dir),
        "Seed-Ok fallback caches the anchor kind from the response shape",
    );
}

/// `dispatch_burst_outcome` is the unified fan-out for both Seed and
/// Standard intents, so the kind-classification fallback fires from every
/// burst arm — not just Seed. Companion to
/// `attach_sub_unprobed_anchor_seeds_kind_on_first_response`: that test
/// pins the Seed-Ok / SubtreeOk path; this one pins it explicitly through
/// the same outcome shape and asserts the Profile reaches its first
/// classification before any subsequent dispatcher work runs.
#[test]
fn dispatch_burst_outcome_classifies_kind_on_first_seed_subtree() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    // Leave the Resource Unknown — anchor_kind from `Resource::kind()`
    // collapses Unknown to None, so Profile.kind starts as None.
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight");
    let snap = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        Some(ResourceKind::Dir),
        "SubtreeOk on a kind-None Profile classifies as Dir at dispatch_burst_outcome",
    );
}

/// Mirror of the SubtreeOk test for the AnchorOk arm: an `AnchorOk(leaf)`
/// response on a Profile whose `kind` was None classifies the anchor as
/// `File`. The walker's response variant is the canonical witness, so the
/// fallback cannot be specialised to one shape.
#[test]
fn dispatch_burst_outcome_classifies_kind_on_first_seed_anchor() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    // Resource is Unknown ⇒ Profile.kind starts as None. The Seed burst
    // emits `ProbeRequest::Subtree` per the unified fallback (Subtree
    // is the safe default for unclassified anchors). The walker, finding a
    // regular file at the path, replies with `Vanished` in production
    // (kind mismatch). For this test we synthesise an `AnchorOk(leaf)`
    // response — a deliberate deviation that exercises the
    // dispatch_burst_outcome classification path for AnchorOk; the walker
    // never produces this response shape against a Subtree request, but
    // the engine's classification logic must still fall out correctly if
    // it ever does (defense-in-depth + symmetry with the SubtreeOk arm).
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        None,
        "unprobed anchor → Profile.kind starts as None",
    );

    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight");
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        Some(ResourceKind::File),
        "AnchorOk on a kind-None Profile classifies as File at dispatch_burst_outcome",
    );
}

/// Walker contract: a `Pending` Profile (descent state) probes a Dir
/// prefix with `ProbeRequest::Descent`; the only valid responses are
/// `SubtreeOk`, `Vanished`, or `Failed`. An `AnchorOk` in this slot is a
/// walker-side bug — descent never queries an anchor's `lstat` shape. The
/// `(DispatchTarget::Descent, ProbeOutcome::AnchorOk(_))` arm fires a
/// `debug_assert!` in dev/CI and falls through to `StaleProbeResponse` in
/// release. The test pins the dev/CI behaviour.
///
/// Disabled in release builds via the standard `cfg_attr` discipline,
/// mirroring `mint_probe_correlation_panics_on_double_open`.
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "walker contract violated")]
fn dispatch_descent_with_anchor_outcome_is_walker_contract_violation() {
    let mut e = Engine::new();
    let foo = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User);
    e.tree.set_kind(foo, ResourceKind::Dir);
    let req = SubAttachRequest::for_path(
        "guard".into(),
        std::path::PathBuf::from("/foo/bar"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(
        matches!(
            e.profiles.get(pid).map(|p| &p.state),
            Some(ProfileState::Pending(_)),
        ),
        "path-based attach against an absent leaf goes Pending",
    );
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("descent probe in flight at the prefix");

    // `AnchorOk` from a Descent probe is structurally impossible from the
    // production walker — `probe_descent` calls `probe_subtree`, whose
    // root-`lstat` rejects non-Dir paths via `Vanished`. We synthesise the
    // breach to exercise the walker-contract debug_assert in
    // `on_probe_response`.
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
}

/// `Engine::kind_agrees_or_finalize` boundary check: a `Profile.kind =
/// Some(File)` receiving a Dir-shaped response is a structurally
/// unreachable walker-contract violation (the typed `ProbeRequest`
/// chain emits `AnchorFile` for File-kinded Profiles, and the walker's
/// `ProbeOutcome` variant matches the request by construction). The
/// boundary catches the case at dispatch time and routes through
/// [`Engine::finalize_anchor_lost`] rather than misroute the Dir
/// snapshot onto a File-kinded Profile (which would leak watch
/// contributions and break the cross-field invariant).
#[test]
#[cfg_attr(
    not(debug_assertions),
    ignore = "debug_assert! is compiled out in release"
)]
#[should_panic(expected = "walker contract violated")]
fn dispatch_standard_ok_with_kind_mismatched_response_routes_through_finalize_anchor_lost_debug() {
    // Set up a File-kinded Profile in Active(Verifying) and inject a
    // SubtreeOk (Dir) response.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::File);
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    // Complete the Seed burst with an AnchorOk so the Profile lands at
    // kind = Some(File).
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight");
    let leaf = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::AnchorOk(leaf),
        }),
        now,
    );
    assert_eq!(
        e.profiles.get(pid).and_then(|p| p.kind),
        Some(ResourceKind::File),
    );

    // Drive a Standard burst (FsEvent at the anchor) and let the settle
    // timer fire so a Verifying probe is in flight.
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::MetadataChanged,
        },
        now,
    );
    while let Some(entry) = e.pop_expired(now + SETTLE) {
        e.step(
            Input::TimerExpired {
                id: entry.id,
                kind: entry.kind,
                profile: entry.profile,
            },
            now + SETTLE,
        );
    }
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Inject the kind-mismatched response: a SubtreeOk (Dir) for a
    // File-kinded Profile. The boundary check fires the debug_assert.
    let dir = dir_tree_snap(vec![]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(dir),
        }),
        now + SETTLE,
    );
}

/// `Profile.kind = None` is the `(Some(Dir | Unknown) | None)` fallback
/// arm in `transition_to_verifying`'s match: an unclassified anchor probes
/// as `Subtree`, never as `AnchorFile`. The unified fallback collapses
/// the prior two-layer defensive defaults (File at the burst site,
/// Directory at the probe-shape site) into one rule applied at one site.
/// This test pins the rule for the Standard-burst path; the Seed-burst
/// path is covered by
/// `dispatch_burst_outcome_classifies_kind_on_first_seed_subtree` (whose
/// initial probe shape is Subtree).
#[test]
fn standard_burst_on_unknown_anchor_emits_subtree_probe() {
    let (mut e, pid, _sid, r, now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // After Seed completes, Profile.kind = Some(Dir). Reset to None to
    // simulate the corner case where a Standard burst runs before any
    // probe has classified the anchor (e.g., a future code path that
    // attaches a Profile in Idle without driving a Seed first).
    if let Some(p) = e.profiles.get_mut(pid) {
        p.kind = None;
    }

    // Drive a Standard burst from an FsEvent at the anchor; advance the
    // settle timer to reach Verifying.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    let mut probe_request: Option<ProbeRequest> = None;
    while let Some(entry) = e.pop_expired(t2) {
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
        for op in &out.probe_ops {
            if let ProbeOp::Probe { request } = op {
                probe_request = Some(request.clone());
            }
        }
    }

    match probe_request {
        Some(ProbeRequest::Subtree { .. }) => {}
        other => panic!(
            "Profile.kind = None on a Standard burst must emit ProbeRequest::Subtree \
             (unified fallback); got {other:?}",
        ),
    }
}

#[test]
fn attach_sub_existing_profile_bumps_refcount() {
    let (mut e, pid, _sid, r, now) = engine_with_attached_sub();
    let pre_state = matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Active(_, _)
    );
    assert!(pre_state, "first attach went Active");
    let pre_refcount = e.profiles.get(pid).unwrap().sub_refcount;

    // Second attach with the same config_hash.
    let req = SubAttachRequest {
        name: String::from("second"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid2 = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    assert_eq!(e.profiles.get(pid).unwrap().sub_refcount, pre_refcount + 1);
    assert_eq!(e.subs.get(sid2).unwrap().profile, pid, "shared Profile");
    // No fresh watch/probe/suppress emitted: existing Profile already has
    // them.
    assert!(out.watch_ops.is_empty());
    assert!(out.probe_ops.is_empty());
}

// ---- ProbeResponse dispatch ----

/// Smoke test: a `TreeSnapshot::Dir(...)` with one Leaf entry
/// lands as a Seed-Ok on the Profile (no Effect, baseline set). Pins
/// the dispatch wiring; the rest of the engine test suite is the broad
/// coverage.
#[test]
fn engine_dispatch_through_shim_matches_v4_behaviour() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    // One-Leaf TreeSnapshot — the shim flattens to one V4 Entry.
    let snap = dir_tree_snap(vec![("main.rs", EntryKind::File, 100)]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.current.is_some(), "Seed-Ok sets current via shim");
    assert!(out.effects.is_empty(), "Seed bursts never fire Effects");
}

#[test]
fn probe_response_seed_ok_sets_baseline_and_idles_no_effect() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_some());
    assert!(p.current.is_some());
    assert!(out.effects.is_empty(), "Seed bursts never fire Effects");
    // Anchor unsuppressed.
    let unsuppress = out
        .watch_ops
        .iter()
        .any(|op| matches!(op, WatchOp::Unsuppress { .. }));
    assert!(unsuppress);
}

#[test]
fn probe_response_seed_vanished_clears_baseline_and_diagnoses() {
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::ProbeVanished {
                intent: BurstIntent::Seed,
                ..
            }
        )
    });
    assert!(has_diag);
}

#[test]
fn probe_response_seed_failed_clears_baseline_and_diagnoses() {
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 13 },
        }),
        Instant::now(),
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::ProbeFailed {
                intent: BurstIntent::Seed,
                errno: 13,
                ..
            },
        )
    });
    assert!(has_diag);
}

#[test]
fn probe_response_correlation_mismatch_drops_with_diagnostic() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    // Inject a response with the wrong correlation.
    let bogus = specter_core::ProbeCorrelation::from(99_999);
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: bogus,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }));
    assert!(stale);
    // State unchanged: still Active(Seed Verifying).
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Active(_, _),
    ));
}

#[test]
fn probe_response_for_idle_profile_drops_with_diagnostic() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Profile is Idle; injecting a ProbeResponse drops with diagnostic.
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: specter_core::ProbeCorrelation::from(1),
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }));
    assert!(stale);
}

// I5-breach panic/diagnostic tests deleted: the forged "channel-open in
// non-mint phase" shape they exercised is structurally unrepresentable
// post-ProbeChannel — `ProbeChannel::open` panics on double-open and
// dispatchers route on `OpenKind`, so the (state, phase)-mismatch arm
// cannot be reached without bypassing the channel's mint API. Structural
// property tests live in `probe_channel.rs`'s `#[cfg(test)] mod tests`.

// ---- Standard burst dispatch ----

#[test]
fn standard_burst_stable_emits_effect_and_awaits() {
    // Stable verdict emits the Effect and transitions to
    // `PostFirePhase::Awaiting`; the engine waits for the completion before
    // returning to Idle. Idle means "nothing in flight" — outstanding
    // Effects keep the burst Active until they report back.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    // FsEvent at anchor → Standard Settling.
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        now,
    );
    // Settle fires.
    while let Some(entry) = e.pop_expired(now + SETTLE) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            now + SETTLE,
        );
    }
    // We're in Verifying; pick up the correlation.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    // Reproduce the seed-burst's empty snapshot — both shim to the same
    // V4 content_hash (entries.len() == 0), so `stable_against` holds.
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        now + SETTLE + Duration::from_millis(1),
    );
    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        _ => panic!("expected Active(Awaiting) after firing"),
    };
    assert_eq!(burst.intent, BurstIntent::Standard);
    assert!(
        matches!(burst.phase, PostFirePhase::Awaiting { outstanding: 1, .. }),
        "stable verdict transitions to Awaiting with one outstanding Effect; got {:?}",
        burst.phase,
    );
    assert_eq!(out.effects.len(), 1, "one Effect emitted at stable verdict");
    let eff = &out.effects[0];
    assert!(!eff.forced);
    // Engine carries the lowered ActionProgram; the actuator resolves
    // argv at spawn time. Assert on the template's literal-only first arg
    // instead of the resolved argv. (`/bin/true` is the test's stub
    // command — see `empty_program()`.)
    let SpawnBody::Exec(exec) = &eff.program.ops()[0].body else {
        panic!("expected SpawnBody::Exec");
    };
    assert_eq!(exec.argv.len(), 1);
    assert!(matches!(
        exec.argv[0].parts.as_slice(),
        [specter_core::ArgPart::Literal(s)] if s.as_str() == "/bin/true"
    ));
    // Substitution-domain inputs that the actuator-side resolver renders
    // to SPECTER_PATH / SPECTER_WATCH / SPECTER_FORCED / SPECTER_EVENT_KIND.
    assert!(
        !eff.anchor_path.as_os_str().is_empty(),
        "anchor_path populated; resolver derives target_path from anchor + relative"
    );
    assert!(
        !eff.sub_name.is_empty(),
        "sub_name populated for ${{specter.watch}} / SPECTER_WATCH"
    );
    assert!(
        !eff.forced,
        "SPECTER_FORCED == \"0\" derives from forced=false"
    );
    assert!(
        matches!(eff.key, specter_core::DedupKey::Subtree { .. }),
        "EVENT_KIND=dir-subtree derives from DedupKey::Subtree variant",
    );
    // SPECTER_DIFF_PATH is an actuator-side augmentation; engine's Effect
    // doesn't carry it. The structural witness is `eff.diff`:
    assert!(
        eff.diff.is_none(),
        "engine doesn't include diff for non-needs_diff Sub"
    );
    // cwd derives from (anchor_path, anchor_kind) at spawn time. Pin both:
    assert_eq!(eff.anchor_path.as_os_str(), "anchor");
    assert_eq!(eff.anchor_kind, specter_core::ResourceKind::Dir);
}

/// The Subtree suppress decision is `nothing_changed &&
/// fired_subs.contains(&dk)` — two gates in conjunction. The
/// per-Profile half (`baseline.hash() == current.hash()`) covers
/// the "fired then noop" arm; the per-Sub half (`fired_subs`
/// existence) is the "Sub has fired before" discriminator that
/// distinguishes a fresh Sub (must fire even on an unchanged tree —
/// first emission) from a repeat fire (suppress on an unchanged
/// tree).
///
/// A noise FsEvent on a fresh Profile drives a phantom Standard
/// burst whose stable verdict observes `baseline.hash() ==
/// current.hash()`. Without the `fired_subs.contains` gate, the
/// phantom would suppress the very first Effect — the Sub would
/// never fire and the user's command would never run. This test
/// pins the gate so any future flattening of the suppress
/// derivation (e.g., dropping back to the per-Profile signal
/// alone) fails here, not in production.
#[test]
fn b1_dedup_fresh_sub_fires_on_phantom_standard_burst() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Precondition: post-Seed Profile has baseline = current and no
    // fire history. The phantom condition (`baseline.hash() ==
    // current.hash()`) is structurally satisfied; the fresh-Sub
    // condition (`fired_subs.is_empty()`) is by construction.
    let baseline_hash = match e.profiles.get(pid).unwrap().baseline.as_ref() {
        Some(TreeSnapshot::Dir(arc)) => arc.dir_hash(),
        _ => panic!("post-Seed baseline must be Some(Dir)"),
    };
    assert!(
        e.profiles.get(pid).unwrap().fired_subs.is_empty(),
        "fresh Sub: no fire history",
    );

    // Drive a Standard burst whose probe response equals the Seed
    // baseline byte-for-byte — a phantom (noise FsEvent, no actual
    // disk change). One probe suffices because the stability verdict
    // compares the response against `current.subtree_at(target)`,
    // which equals baseline immediately post-Seed.
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        now,
    );
    while let Some(entry) = e.pop_expired(now + SETTLE) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            now + SETTLE,
        );
    }
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let phantom_snap = dir_tree_snap(vec![]);
    assert_eq!(
        phantom_snap.dir_hash(),
        baseline_hash,
        "test setup: probe response must hash-match baseline (phantom)",
    );
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(phantom_snap),
        }),
        now + SETTLE + Duration::from_millis(1),
    );

    assert_eq!(
        out.effects.len(),
        1,
        "fresh Sub must fire on first stable verdict, even on a phantom \
         (`baseline.hash() == current.hash()`); `fired_subs.contains` \
         is the discriminator that prevents the noop-suppress path",
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().fired_subs.len(),
        1,
        "post-emit: fire history populated",
    );
}

#[test]
fn emit_effects_subtree_root_uses_parent_dir_for_file_profile() {
    // Contract: SubtreeRoot Sub anchored at a File-kind Profile derives
    // the Effect's `cwd` from the file's parent dir (not the file
    // itself — `Command::current_dir` requires a directory). The
    // surrounding burst flow (probe target, current-shape preservation,
    // graft path) is exercised by
    // `standard_burst_on_file_anchor_targets_anchor_not_parent_dir`;
    // this test asserts only the cwd / env-var contract.
    let mut e = Engine::new();
    let parent = e.tree.ensure(None, "parentdir", ResourceRole::User);
    e.tree.set_kind(parent, ResourceKind::Dir);
    let file_anchor = e.tree.ensure(Some(parent), "main.rs", ResourceRole::User);
    e.tree.set_kind(file_anchor, ResourceKind::File);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "build".into(),
        resource: file_anchor,
        path: None,
        config: ScanConfig::builder().recursive(false).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    // Seed → Idle.
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let snap = file_tree_snap(EntryKind::File, 0, std::time::UNIX_EPOCH, 1);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(snap.clone()),
        }),
        now,
    );
    // Standard burst with the same snap (stable).
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: file_anchor,
            event: FsEvent::Modified,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::AnchorOk(snap),
        }),
        t2,
    );
    assert_eq!(out.effects.len(), 1);
    let eff = &out.effects[0];
    // File-kind anchor: actuator's `compute_cwd` returns parent dir.
    // The engine's job here is to pin (anchor_path, anchor_kind) so the
    // actuator's compute_cwd reaches "parentdir". The original cwd
    // assertion ("File-kind anchor uses parent dir as cwd") is now
    // structural: anchor_path is the file, anchor_kind is File ⇒
    // compute_cwd returns parent.
    assert_eq!(eff.anchor_path.as_os_str(), "parentdir/main.rs");
    assert_eq!(eff.anchor_kind, specter_core::ResourceKind::File);
    // SPECTER_PATH and SPECTER_ANCHOR both derive from `anchor_path`
    // for a File-anchor Subtree Effect: the resolver returns
    // `Cow::Borrowed(&anchor_path)` when `target_relative` is empty,
    // so both env values share the same byte sequence.
    assert_eq!(eff.target_relative.as_str(), "");
}

#[test]
fn standard_burst_on_file_anchor_targets_anchor_not_parent_dir() {
    // Realistic Standard-burst-on-File-anchor flow. A real Sensor
    // probing a File anchor returns `TreeSnapshot::File(leaf)`; the
    // engine must (1) probe the anchor itself rather than the parent
    // dir and (2) preserve `Profile.current` as `TreeSnapshot::File(_)`
    // post-dispatch — the snapshot navigation invariant
    // `current` is anchor-shaped breaks if a Standard burst graft
    // wholesale-replaces with a Dir snapshot rooted at the parent.
    let mut e = Engine::new();
    let parent = e.tree.ensure(None, "parentdir", ResourceRole::User);
    e.tree.set_kind(parent, ResourceKind::Dir);
    let file_anchor = e.tree.ensure(Some(parent), "main.rs", ResourceRole::User);
    e.tree.set_kind(file_anchor, ResourceKind::File);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("build"),
        resource: file_anchor,
        path: None,
        config: ScanConfig::builder().recursive(false).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;

    // Seed → leaf v1; Standard injects the same leaf so the verdict
    // is stable (matching the conventional pattern across this file).
    let snap = file_tree_snap(EntryKind::File, 0, UNIX_EPOCH, 1);
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed verify probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::AnchorOk(snap.clone()),
        }),
        now,
    );

    // Drive a Standard burst from an FsEvent at the file. Capture the
    // probe request emitted on the settle-timer expiry step.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: file_anchor,
            event: FsEvent::Modified,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    let mut probe_request: Option<ProbeRequest> = None;
    while let Some(entry) = e.pop_expired(t2) {
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
        for op in &out.probe_ops {
            if let ProbeOp::Probe { request } = op {
                probe_request = Some(request.clone());
            }
        }
    }

    // (1) The Standard probe is `AnchorFile` and its `target_path` is
    // the anchor's filesystem path. The two assertions are the structural
    // witnesses for the v1 design: File anchors take the typed
    // `AnchorFile` arm (single-`lstat` walker dispatch) and never promote
    // past the anchor to the parent dir.
    let anchor_path = e.tree.path_of(file_anchor).expect("anchor path resolves");
    match probe_request.as_ref() {
        Some(ProbeRequest::AnchorFile { target_path, .. }) => {
            assert_eq!(
                *target_path, anchor_path,
                "AnchorFile target_path is the anchor's filesystem path",
            );
        }
        other => panic!(
            "Standard burst on a File-anchored Profile must emit ProbeRequest::AnchorFile; \
             got {other:?}",
        ),
    }

    // (2) Inject a realistic File response (kqueue per-file FD path).
    // After dispatch, Profile.current must remain File-shaped.
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Standard verify probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::AnchorOk(snap),
        }),
        t2,
    );

    let p = e.profiles.get(pid).expect("Profile alive");
    match &p.current {
        Some(TreeSnapshot::File(_)) => {} // navigation invariant preserved
        Some(TreeSnapshot::Dir(arc)) => panic!(
            "Profile.current corrupted to Dir(root_meta={:?}); expected File(leaf)",
            arc.root_meta,
        ),
        None => panic!("Profile.current must be Some(File(leaf)) post-Standard"),
    }

    // (3) Stable verdict (same leaf hash) + dirty=0 ⇒ exactly one Effect fires.
    assert_eq!(
        out.effects.len(),
        1,
        "stable verdict + dirty=0 ⇒ exactly one Effect fires",
    );

    // (4) The anchor's `watch_demand` is exactly 1 (Profile claim only).
    assert_eq!(
        e.tree
            .get(file_anchor)
            .map(specter_core::Resource::watch_demand),
        Some(1),
        "no spurious watch_demand bump on the anchor",
    );
}

#[test]
fn standard_burst_force_fires_on_max_settle() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        now,
    );
    // Burst deadline fires before settle.
    let deadline = now + MAX_SETTLE + Duration::from_millis(1);
    while let Some(entry) = e.pop_expired(deadline) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            deadline,
        );
    }
    // After force-fire, we're either in Verifying (forced=true) or already
    // Awaiting if the deadline race resolved both timers. Drive the
    // response back if needed.
    if let Some(correlation) = e.pending_probe_for(ProbeOwner::Profile(pid)) {
        // Inject a not-stable response to test the forced effect emission.
        let snap = dir_tree_snap(vec![("new.rs", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation,
                outcome: ProbeOutcome::SubtreeOk(snap),
            }),
            deadline,
        );
        // Forced fire transitions to Awaiting (Effect in flight). The
        // post-fire rebase happens when the eventual EffectComplete
        // drives the Awaiting → Rebasing transition.
        let phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
            _ => panic!("expected Active(Awaiting)"),
        };
        assert!(
            matches!(phase, PostFirePhase::Awaiting { outstanding: 1, .. }),
            "force-fired stable verdict transitions to Awaiting; got {phase:?}",
        );
        assert_eq!(out.effects.len(), 1);
        assert!(
            out.effects[0].forced,
            "force-fired Effect must carry forced=true",
        );
    }
}

#[test]
fn fs_event_modified_during_seed_probing_preserves_intent() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    // Profile is in Active(Seed Verifying) right after attach. Inject an
    // FsEvent — should transition to Active(Seed Batching), emit Cancel.
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );
    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        _ => panic!(),
    };
    assert_eq!(
        burst.intent,
        BurstIntent::Seed,
        "intent preserved across Verifying → Batching",
    );
    assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));
    let cancels = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(cancels, 1);
}

/// Field-discipline pin for `event_drives_batching`: an FsEvent during
/// Verifying closes the probe channel atomically with the Cancel emission.
/// Pre-refactor the close was implicit in the `PreFirePhase::Verifying { ... }
/// → Batching { ... }` variant rewrite; post-refactor it must clear the
/// per-Profile `pending_probe` slot explicitly.
#[test]
fn event_drives_batching_clears_pending_probe() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "Seed probe in flight after attach",
    );

    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_none(),
        "channel closed atomically with Verifying → Batching transition",
    );
}

/// Field-discipline pin for `finalize_anchor_lost`: an anchor terminal
/// event during Verifying cancels the in-flight probe and clears the
/// channel. Replaces the pre-refactor `was_verifying` snapshot's role.
#[test]
fn finalize_anchor_lost_during_verifying_clears_pending_probe() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "Seed probe in flight after attach",
    );

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_none(),
        "anchor terminal during Verifying closes the channel",
    );
    let cancels = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid))
        .count();
    assert_eq!(
        cancels, 1,
        "exactly one Cancel emitted; got {:?}",
        out.probe_ops
    );
}

/// Single-diagnostic guarantee for stale `ProbeResponse`. Pre-refactor
/// the dispatch had two stale-detection layers (state-shape mismatch and
/// inner-correlation mismatch) that could both fire on degenerate inputs.
/// Post-refactor the top-level `pending_probe == Some(received)` check is
/// the sole gate — exactly one diagnostic per stale response.
#[test]
fn stale_probe_response_emits_exactly_one_diagnostic() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let bogus = specter_core::ProbeCorrelation::from(99_999);
    let snap = dir_tree_snap(vec![]);

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: bogus,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    let stale_count = out
        .diagnostics
        .iter()
        .filter(|d| matches!(d, Diagnostic::StaleProbeResponse { owner: ProbeOwner::Profile(profile), .. } if *profile == pid))
        .count();
    assert_eq!(
        stale_count, 1,
        "exactly one StaleProbeResponse diagnostic; got {:?}",
        out.diagnostics,
    );
    // Live channel untouched: the legitimate Seed probe is still in flight.
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "live channel untouched by stale response",
    );
}

/// Anchor events bypass the class filter unconditionally.
/// Profile has events = EMPTY (nothing in the mask); a `MetadataChanged`
/// at the anchor still drives the lifecycle path (burst start), and no
/// `EventClassDropped` is emitted. This guards the lifecycle-continuity
/// invariant: anchor events never get filtered out by user mask choice.
#[test]
fn fs_event_metadatachanged_at_anchor_bypasses_class_filter() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::MetadataChanged,
        },
        Instant::now(),
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EventClassDropped { .. })),
        "anchor events bypass the class filter",
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state,
            ProfileState::Active(_, _),
        ),
        "MetadataChanged at the anchor drives a burst even on EMPTY mask",
    );
}

/// Descendant events whose class is not in the covering Profile's
/// `events_union` drop with `EventClassDropped` BEFORE driving the burst.
/// Profile has events = EMPTY ⇒ `intersects(any_class) == false`, so a
/// `MetadataChanged` on a covered descendant drops cleanly without state
/// mutation.
#[test]
fn fs_event_metadatachanged_at_descendant_drops_with_event_class_dropped() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Materialize a covered descendant. Bump `watch_demand` so the event
    // passes the `EventOnUnwatchedResource` head guard. The Profile's
    // ScanConfig has `recursive(true)` so `covers(profile, child, tree)`
    // is satisfied.
    let child = e.tree.ensure(Some(root), "child.txt", ResourceRole::User);
    e.tree.set_kind(child, ResourceKind::File);
    e.tree.get_mut(child).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileDescendant(pid),
        ClassSet::CONTENT,
    );

    let out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::MetadataChanged,
        },
        Instant::now(),
    );
    let has_class_drop = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EventClassDropped {
                resource,
                event: FsEvent::MetadataChanged,
                profile,
            } if *resource == child && *profile == pid,
        )
    });
    assert!(
        has_class_drop,
        "descendant MetadataChanged drops with EventClassDropped on EMPTY mask",
    );
    // No `MetadataChangedIgnored` lingers — the variant was deleted.
    // No state mutation: the filter `continue`s before drive_burst.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
}

/// Identity events on a *descendant File* fold into the CONTENT
/// class. A Profile excluding CONTENT (here: STRUCTURE-only on a Dir
/// anchor) drops the descendant File `Removed` with `EventClassDropped`.
/// The dropped event is not routed through `on_anchor_terminal_event`
/// — that routing is anchor-only.
#[test]
fn fs_event_terminal_on_descendant_file_folds_to_content_and_drops() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::STRUCTURE,
        log_output: false,
        source_promoter: None,
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    complete_seed_burst(&mut e, pid);

    let child = e.tree.ensure(Some(r), "f.txt", ResourceRole::User);
    e.tree.set_kind(child, ResourceKind::File);
    e.tree.get_mut(child).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileDescendant(pid),
        ClassSet::STRUCTURE,
    );

    let out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    let has_class_drop = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EventClassDropped {
                event: FsEvent::Removed,
                ..
            },
        )
    });
    assert!(
        has_class_drop,
        "Removed on a descendant File folds to CONTENT and drops on STRUCTURE-only mask",
    );
    // Profile remains Idle: dropped events do not extend dirty sets.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // Sanity: anchor's contribution is intact (we did NOT terminate).
    assert_eq!(e.profiles.get(pid).unwrap().anchor_claim, AnchorClaim::Held,);
    let _ = sid;
}

/// Terminal events on the anchor route through
/// `on_anchor_terminal_event` regardless of the Profile's `events_union`.
/// Anchor is a Dir, events = EMPTY: the kqexec class for `Removed` on a
/// Dir is STRUCTURE — not in the EMPTY mask — but anchor events bypass
/// the filter. After the call, `anchor_claim` is cleared to `None` and
/// `baseline` / `current` are dropped.
#[test]
fn fs_event_anchor_terminal_bypasses_class_filter() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim,
        AnchorClaim::Held,
        "anchor_claim set to Held after attach_sub_inner",
    );

    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EventClassDropped { .. })),
        "anchor terminal events bypass the class filter",
    );

    let p = e.profiles.get(pid).unwrap();
    assert_eq!(
        p.anchor_claim,
        AnchorClaim::None,
        "anchor_claim cleared by on_anchor_terminal_event",
    );
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
    assert!(matches!(p.state, ProfileState::Idle));
}

#[test]
fn fs_event_for_unwatched_resource_emits_diagnostic() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "ghost", ResourceRole::User);
    let out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );
    let has_diag = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::EventOnUnwatchedResource { .. }));
    assert!(has_diag);
}

#[test]
fn fs_event_at_watched_resource_with_no_consumer_emits_event_no_consumer_not_unwatched() {
    // A WatchRootParent fires
    // `StructureChanged` (e.g., a sibling directory was created /
    // renamed) and no Profile in the engine cares. The event must NOT
    // be diagnosed as "unwatched resource" — the Resource IS Watched.
    // The new `EventNoConsumer` variant signals this benign case so the
    // bin can log it at TRACE rather than WARN.
    let mut e = Engine::new();
    // Materialize an unrelated Watched resource (e.g., a parent that
    // someone else holds open). watch_demand > 0 ensures the event isn't
    // routed through the `EventOnUnwatchedResource` path.
    let r = e.tree.ensure(None, "lonely", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    e.tree.get_mut(r).unwrap().insert_contribution(
        specter_core::ContribKey::ProfileAnchor(specter_core::ProfileId::default()),
        ClassSet::STRUCTURE,
    );

    let out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    let no_consumer = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::EventNoConsumer { .. }));
    let unwatched = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::EventOnUnwatchedResource { .. }));
    assert!(no_consumer, "should emit EventNoConsumer");
    assert!(
        !unwatched,
        "should NOT emit EventOnUnwatchedResource — the resource IS Watched"
    );
}

#[test]
fn fs_event_removed_at_anchor_active_terminates() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Drive a Standard burst.
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        now,
    );
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Active(_, _),
    ));
    // Now Removed at anchor → terminate.
    let out = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        now,
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
    // watch_demand on anchor → 0; one Unwatch op emitted.
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 0);
    let unwatches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
        .count();
    assert!(unwatches >= 1);
}

#[test]
fn fs_event_removed_at_anchor_idle_releases_watch_and_clears_baseline() {
    // FsEvent: Removed/Renamed/Revoked on an Idle profile transitions
    // idempotently. We additionally release the watch contribution and
    // drop baseline/current — they refer to a now-vanished slot, and
    // clearing them lets the watch-root-parent recovery path
    // (`on_fs_event`'s `start_pending_recovery`) detect "anchor is gone"
    // via `current.is_none()`.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 1);
    assert!(e.profiles.get(pid).unwrap().current.is_some());

    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // Profile state stays Idle (no Active transition).
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // watch_demand released; baseline/current cleared.
    assert_eq!(e.tree.get(root).unwrap().watch_demand(), 0);
    let p = e.profiles.get(pid).unwrap();
    assert!(p.baseline.is_none());
    assert!(p.current.is_none());
}

// ---- TimerExpired dispatch ----

#[test]
fn timer_expired_settle_in_settling_transitions_to_probing() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        now,
    );
    // Pop the settle timer.
    let entry = e.pop_expired(now + SETTLE).expect("settle timer ready");
    let out = e.step(
        Input::TimerExpired {
            profile: entry.profile,
            kind: entry.kind,
            id: entry.id,
        },
        now + SETTLE,
    );
    let p = e.profiles.get(pid).unwrap();
    assert!(matches!(
        p.state,
        ProfileState::Active(_, _) // Verifying
    ));
    let probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(probes, 1);
}

#[test]
fn timer_expired_stale_id_emits_diagnostic() {
    let mut e = Engine::new();
    let bogus = specter_core::TimerId::from(99_999);
    let out = e.step(
        Input::TimerExpired {
            profile: specter_core::ProfileId::default(),
            kind: specter_core::TimerKind::Settle,
            id: bogus,
        },
        Instant::now(),
    );
    let stale = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::StaleTimer { .. }));
    assert!(stale);
}

// ---- EffectComplete dispatch ----
//
// The engine does not return to Idle after firing Effects: the burst
// stays `Active(Awaiting)` until each completion reports back, and the
// post-Effect rebase happens in `PostFirePhase::Rebasing` as a phase of
// the same burst. `EffectComplete` arrivals route by phase: Awaiting
// decrements / transitions; non-Awaiting emits
// `EffectCompleteOutsideAwaiting`.

#[test]
fn effect_complete_ok_in_idle_diagnoses_outside_awaiting() {
    // No path leaves Idle with an outstanding EffectComplete: the burst
    // stays Active(Awaiting) until completions arrive. A completion
    // landing in Idle is therefore unexpected (gate-deadline force-
    // transition or anchor-loss) — emit `EffectCompleteOutsideAwaiting`
    // and drop without state change.
    let (mut e, pid, sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            result: EffectOutcome::Ok,
        },
        Instant::now(),
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }
                if *s == sid && *p == pid,
        )
    });
    assert!(
        has_diag,
        "EffectComplete::Ok in Idle is a late completion and diagnoses",
    );
    // No probe emitted — the Profile stays Idle.
    assert!(out.probe_ops.is_empty());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
}

#[test]
fn effect_complete_failed_in_idle_clears_hash_and_diagnoses() {
    // Failed always clears `fired_subs[key]` regardless of
    // phase — a failed Effect leaves no observable state to dedupe
    // against. In Idle the completion is also "late" (the engine isn't
    // tracking it), so it diagnoses with EffectCompleteOutsideAwaiting.
    let (mut e, pid, sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let pre_baseline = e.profiles.get(pid).unwrap().baseline.is_some();
    let out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            result: EffectOutcome::Failed {
                exit_code: Some(1),
                signal: None,
            },
        },
        Instant::now(),
    );
    let has_diag = out.diagnostics.iter().any(|d| {
        matches!(
            d,
            Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }
                if *s == sid && *p == pid,
        )
    });
    assert!(has_diag, "Failed in Idle diagnoses as late completion");
    assert!(out.effects.is_empty());
    assert!(out.probe_ops.is_empty());
    assert_eq!(
        e.profiles.get(pid).unwrap().baseline.is_some(),
        pre_baseline,
        "baseline unchanged on Failed",
    );
}

// ---- Effect needs_diff carries Diff ----

#[test]
fn effect_emission_carries_diff_when_needs_diff() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("fmt"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: diff_program(), // references ${specter.created}
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(e.subs.get(sid).unwrap().needs_diff);

    // Seed burst → baseline = empty snapshot.
    complete_seed_burst(&mut e, pid);

    // Standard burst, first round: FsEvent → settle → probe → snapshot with
    // a new entry. The first response is *not stable* (current was empty),
    // so the Engine reschedules another settle cycle.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now,
    );
    let snap_with_entry = dir_tree_snap(vec![("new.rs", EntryKind::File, 5)]);

    // Iteratively drain settle timers and inject probe responses until the
    // burst stabilizes (one Effect emitted).
    let mut t = now;
    let mut effect_out = None;
    for _ in 0..6 {
        t += SETTLE * 4; // big enough to cover backoff
        while let Some(entry) = e.pop_expired(t) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t,
            );
        }
        let correlation = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation,
                outcome: ProbeOutcome::SubtreeOk(snap_with_entry.clone()),
            }),
            t,
        );
        if !out.effects.is_empty() {
            effect_out = Some(out);
            break;
        }
    }
    let out = effect_out.expect("burst stabilized and emitted an Effect");
    assert_eq!(out.effects.len(), 1);
    let effect = &out.effects[0];
    assert!(effect.diff.is_some(), "needs_diff Effect carries the Diff");
    let diff = effect.diff.as_ref().unwrap();
    assert_eq!(diff.created.len(), 1);
    assert_eq!(diff.created[0].segment.as_str(), "new.rs");
}

// ---- Descent integration ----

#[test]
fn seed_burst_descendants_watched_via_first_probe() {
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    // First-probe response with one File and one Dir descendant.
    // Only the Dir gets a Watch op; the File materializes without an FD
    // contribution.
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("subdir", EntryKind::Dir, 2),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    // 1 Watch op (subdir Dir) + 1 Unsuppress for the anchor. File doesn't
    // contribute Watch.
    let watches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    assert_eq!(watches, 1);
}

// ---- Probe kind selection ----

#[test]
fn probe_op_for_file_anchor_is_file_kind() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "log.txt", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::File);
    let req = SubAttachRequest {
        name: String::from("file-sub"),
        resource: r,
        path: None,
        config: ScanConfig::builder().build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let out = e.step(Input::AttachSub(req), Instant::now());
    let probe_request = out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.clone()),
        _ => None,
    });
    assert!(
        matches!(probe_request, Some(ProbeRequest::AnchorFile { .. })),
        "File-anchored Profile's seed burst must emit ProbeRequest::AnchorFile",
    );
}

// ---- on_watch_op_rejected ----

#[test]
fn watch_op_rejected_clamps_watch_demand_to_zero() {
    // Build a Resource with watch_demand=2 (multi-Profile co-located).
    // Inject WatchOpRejected. Expect watch_demand → 0, Unwatch emitted,
    // Diagnostic.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "x", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let mut out = StepOutput::default();
    let pid = specter_core::ProfileId::default();
    crate::refcounts::add_watch(
        &mut e.tree,
        r,
        specter_core::ContribKey::ProfileAnchor(pid),
        NO_EVENTS,
        &mut out,
    );
    crate::refcounts::add_watch(
        &mut e.tree,
        r,
        specter_core::ContribKey::ProfileParent(pid),
        NO_EVENTS,
        &mut out,
    );
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            op: WatchOp::Watch {
                resource: r,
                path: std::path::PathBuf::new(),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 0);
    assert!(
        result
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { .. }))
    );
    assert!(result.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::WatchOpRejected {
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
            ..
        }
    )),);
}

#[test]
fn watch_op_rejected_already_unwatched_emits_diagnostic_only() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "x", ResourceRole::User);
    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            op: WatchOp::Unwatch { resource: r },
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );
    assert!(result.watch_ops.is_empty());
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::WatchOpRejected { .. }))
    );
}

// ---- P11.0: WatchOpRejected purges descents at the rejected resource ----

#[test]
fn watch_op_rejected_purges_pending_descent_at_rejected_prefix() {
    // Set up: pre-existing /foo, attach `/foo/bar` (descent at /foo).
    let mut e = Engine::new();
    let foo = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User);
    e.tree.set_kind(foo, ResourceKind::Dir);
    let req = SubAttachRequest::for_path(
        "guard".into(),
        std::path::PathBuf::from("/foo/bar"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let _ = e.step(Input::AttachSub(req), Instant::now());
    let pid = {
        let mut iter = e.profiles.iter();
        iter.next().expect("profile exists").0
    };
    assert!(e.descent_state(ProbeOwner::Profile(pid)).is_some());
    let initial_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("first probe in flight");
    let initial_demand = e.tree.get(foo).unwrap().watch_demand();
    assert_eq!(initial_demand, 1);

    // Inject WatchOpRejected (e.g., EMFILE) for the descent prefix.
    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: std::path::PathBuf::from("foo"),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // The clamp zeroed watch_demand; the descent has been purged.
    assert_eq!(e.tree.get(foo).unwrap().watch_demand(), 0);
    assert!(
        e.descent_state(ProbeOwner::Profile(pid)).is_none(),
        "descent purged on rejection",
    );

    // A Cancel for the in-flight probe was emitted.
    assert!(
        result
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid)),
        "ProbeOp::Cancel emitted for the in-flight descent probe",
    );

    // ProfileClaimPurged{DescentPrefix} surfaces (in addition to WatchOpRejected).
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
                profile, claim, resource, failure
            } if *profile == pid
                && *claim == ClaimKind::DescentPrefix
                && *resource == foo
                && *failure == specter_core::WatchFailure::Pressure { errno: 24 })),
        "ProfileClaimPurged{{DescentPrefix}} diagnostic emitted",
    );

    // Late `ProbeResponse` for the cancelled correlation arrives — must
    // be silently discarded (descent removed, correlation no longer
    // matches anything).
    let late = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: initial_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    // Stale-response diagnostic, no panic.
    assert!(
        late.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. })),
    );
}

#[test]
fn watch_op_rejected_for_anchored_profile_emits_anchor_claim_purged() {
    // Materialized Profile, WatchOpRejected at its anchor — emits
    // ProfileClaimPurged{Anchor} + WatchOpRejected.
    let (mut e, pid, _sid, r, _now) = engine_with_attached_sub();
    let result = e.step(
        Input::WatchOpRejected {
            resource: r,
            op: WatchOp::Watch {
                resource: r,
                path: std::path::PathBuf::new(),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
            profile, claim, resource, ..
        } if *profile == pid && *claim == ClaimKind::Anchor && *resource == r)),
        "ProfileClaimPurged{{Anchor}} emitted for anchored Profile",
    );
    assert!(
        result
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::WatchOpRejected { .. })),
    );
    // Anchor claim cleared.
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim,
        AnchorClaim::None,
        "anchor_claim cleared by purge",
    );
}

#[test]
fn watch_op_rejected_purges_multiple_descents_at_same_prefix() {
    // Two Profiles share a descent prefix (e.g., two Subs anchored at
    // siblings under the same scaffold). WatchOpRejected purges both.
    let mut e = Engine::new();
    let foo = e
        .tree
        .ensure_path(&[FS_ROOT_SEGMENT, "foo"], ResourceRole::User);
    e.tree.set_kind(foo, ResourceKind::Dir);
    let req_a = SubAttachRequest::for_path(
        "a".into(),
        std::path::PathBuf::from("/foo/sib_a"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let req_b = SubAttachRequest::for_path(
        "b".into(),
        std::path::PathBuf::from("/foo/sib_b"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req_a), Instant::now());
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let attach_out = e.step(Input::AttachSub(req_b), Instant::now());
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_a = e.subs.get(sid_a).unwrap().profile;
    let pid_b = e.subs.get(sid_b).unwrap().profile;
    // Both descents at /foo (different anchors).
    assert!(e.descent_state(ProbeOwner::Profile(pid_a)).is_some());
    assert!(e.descent_state(ProbeOwner::Profile(pid_b)).is_some());
    assert_eq!(e.tree.get(foo).unwrap().watch_demand(), 2);

    let result = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: std::path::PathBuf::from("foo"),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            },
            failure: specter_core::WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    assert!(e.descent_state(ProbeOwner::Profile(pid_a)).is_none());
    assert!(e.descent_state(ProbeOwner::Profile(pid_b)).is_none());
    let purged_count = result
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(
                d,
                Diagnostic::ProfileClaimPurged {
                    claim: ClaimKind::DescentPrefix,
                    ..
                }
            )
        })
        .count();
    assert_eq!(
        purged_count, 2,
        "one ProfileClaimPurged{{DescentPrefix}} per descent",
    );
}

// ---- SensorOverflow reseeds in-scope Profiles ----

#[test]
fn sensor_overflow_global_idle_reseeds_to_active_seed() {
    // Idle Profile (post-`complete_seed_burst`): an overflow drives a
    // direct `start_seed_burst` call; the Profile transitions to
    // `Active(Seed)` and a fresh probe is in flight.
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle
    ));

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );

    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Seed);
    assert!(matches!(burst.phase, PreFirePhase::Verifying));
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
        "seed burst armed a fresh verify probe",
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::SensorOverflow {
                scope: OverflowScope::Global
            }
        )),
        "Diagnostic::SensorOverflow{{Global}} emitted exactly once per overflow input",
    );
}

#[test]
fn sensor_overflow_active_standard_transitions_to_active_seed() {
    // Active(Standard) Profile: an overflow `finish_burst_to_idle` +
    // `start_seed_burst` round-trip transitions the burst to
    // `Active(Seed)`. The Standard burst's `dirty_resources` /
    // `force_walk_resources` are discarded — the seed re-baselines.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    let now = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        now,
    );
    // Now in Active(Standard) Batching.
    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Standard) after FsEvent; got {s:?}"),
    };
    assert_eq!(burst.intent, BurstIntent::Standard);
    assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        now,
    );

    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
        s => panic!("expected Active(Seed) after overflow; got {s:?}"),
    };
    assert_eq!(
        burst.intent,
        BurstIntent::Seed,
        "overflow abandoned the Standard burst and re-seeded",
    );
    assert!(
        burst.dirty_resources.is_empty() && burst.force_walk_resources.is_empty(),
        "seed burst starts with empty dirty / force_walk sets — Standard's accumulators discarded",
    );
    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::SensorOverflow {
            scope: OverflowScope::Global
        }
    )),);
}

#[test]
fn sensor_overflow_pending_profile_is_skipped() {
    // Pending(_) Profile: descent in flight; no baseline to drift-test.
    // Overflow is a no-op for the Profile state but still emits the
    // diagnostic.
    let mut e = Engine::new();
    let req = SubAttachRequest::for_path(
        "guard".into(),
        std::path::PathBuf::from("/missing/anchor"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let _ = e.step(Input::AttachSub(req), Instant::now());
    let pid = {
        let mut iter = e.profiles.iter();
        iter.next().expect("profile exists").0
    };
    assert!(
        e.descent_state(ProbeOwner::Profile(pid)).is_some(),
        "fixture: profile is in Pending(_)",
    );

    let pre_state = format!("{:?}", e.profiles.get(pid).unwrap().state);
    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        Instant::now(),
    );
    let post_state = format!("{:?}", e.profiles.get(pid).unwrap().state);

    assert_eq!(
        pre_state, post_state,
        "Pending Profile state preserved across overflow",
    );
    assert!(
        e.descent_state(ProbeOwner::Profile(pid)).is_some(),
        "descent still in flight after overflow",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::SensorOverflow { .. })),
        "diagnostic still emitted regardless of per-Profile dispatch",
    );
}

#[test]
fn sensor_overflow_resource_scope_filters_profiles() {
    // OverflowScope::Resource(r) reseeds only Profiles whose anchor
    // lies in the subtree rooted at r — the FSEvents per-stream signal.
    // Set up two siblings under one root; overflow at the first
    // sibling's resource reseeds only the first.
    let mut e = Engine::new();
    let parent = e.tree.ensure(None, "parent", ResourceRole::User);
    e.tree.set_kind(parent, ResourceKind::Dir);
    let a = e.tree.ensure(Some(parent), "a", ResourceRole::User);
    e.tree.set_kind(a, ResourceKind::Dir);
    let b = e.tree.ensure(Some(parent), "b", ResourceRole::User);
    e.tree.set_kind(b, ResourceKind::Dir);
    let now = Instant::now();
    let req_a = SubAttachRequest {
        name: "sub-a".into(),
        resource: a,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let req_b = SubAttachRequest {
        name: "sub-b".into(),
        resource: b,
        ..req_a.clone()
    };
    let attach_out = e.step(Input::AttachSub(req_a), now);
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let attach_out = e.step(Input::AttachSub(req_b), now);
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_a = e.subs.get(sid_a).unwrap().profile;
    let pid_b = e.subs.get(sid_b).unwrap().profile;
    complete_seed_burst(&mut e, pid_a);
    complete_seed_burst(&mut e, pid_b);
    assert!(matches!(
        e.profiles.get(pid_a).unwrap().state,
        ProfileState::Idle
    ));
    assert!(matches!(
        e.profiles.get(pid_b).unwrap().state,
        ProfileState::Idle
    ));

    // Overflow scoped to `a` — only Profile A reseeds.
    let _ = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Resource(a),
        },
        Instant::now(),
    );

    assert!(
        matches!(
            &e.profiles.get(pid_a).unwrap().state,
            ProfileState::Active(ActiveBurst::PreFire(pre), _) if pre.intent == BurstIntent::Seed
        ),
        "Profile A (anchor at a) reseeded",
    );
    assert!(
        matches!(&e.profiles.get(pid_b).unwrap().state, ProfileState::Idle),
        "Profile B (anchor at b, sibling of a) untouched",
    );
}

// ---- P11.0: anchor_claim drives reap correctness ----

#[test]
fn seed_vanished_then_reap_releases_anchor_via_claim() {
    let (mut e, pid, sid, r, _now) = engine_with_attached_sub();
    // Anchor watch_demand is 1, anchor_claim is Held.
    assert_eq!(e.tree.get(r).unwrap().watch_demand(), 1);
    assert_eq!(e.profiles.get(pid).unwrap().anchor_claim, AnchorClaim::Held,);

    // Detach the Sub mid-burst → reap_pending = true.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state.burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drive Seed Vanished to fire the reap.
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );

    // Profile reaped; anchor's contribution released → Unwatch emitted.
    assert!(e.profiles.get(pid).is_none(), "Profile reaped");
    let saw_unwatch = out
        .watch_ops
        .iter()
        .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == r));
    assert!(
        saw_unwatch,
        "Unwatch for the anchor emitted on reap (anchor_claim drove the release)",
    );
}

#[test]
fn anchor_terminal_event_clears_anchor_claim() {
    // After on_anchor_terminal_event releases the anchor, a subsequent
    // reap must NOT double-release it (the claim is cleared to None).
    let (mut e, pid, _sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(e.profiles.get(pid).unwrap().anchor_claim, AnchorClaim::Held,);

    // Inject a Removed event at the anchor: the terminal event releases
    // the anchor's contribution and clears the claim.
    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().anchor_claim,
        AnchorClaim::None,
        "anchor_claim cleared by terminal event",
    );
    assert_eq!(
        e.tree.get(r).unwrap().watch_demand(),
        0,
        "anchor's watch_demand released",
    );
}

// ---- detach_sub ----

#[test]
fn detach_sub_idle_profile_reaps_immediately() {
    let (mut e, pid, sid, r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Profile is now Idle.
    assert!(matches!(
        e.profiles.get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    let out = e.step(Input::DetachSub(sid), Instant::now());
    // Profile reaped; anchor unwatched.
    assert!(e.profiles.get(pid).is_none());
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Unwatch { resource } if *resource == r))
    );
    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::ProfileReaped {
            via: specter_core::ReapTrigger::Immediate,
            ..
        }
    )));
}

#[test]
fn detach_sub_active_profile_marks_reap_pending() {
    let (mut e, pid, sid, _r, _now) = engine_with_attached_sub();
    // Profile is Active(Seed Verifying) — Seed-burst still in flight.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    let p = e.profiles.get(pid).expect("profile alive until burst ends");
    assert!(matches!(p.state.burst_finish(), Some(BurstFinish::Reap)));
    assert_eq!(p.sub_refcount, 0);
}

#[test]
fn reap_pending_burst_completion_skips_effects_and_reaps() {
    // Sub on Active(Standard, stable) Profile; detach mid-burst; finish
    // burst — no Effect emitted; Profile reaped.
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);

    // Drive Standard burst.
    let t1 = Instant::now();
    e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Detach the Sub mid-burst.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(matches!(
        e.profiles.get(pid).unwrap().state.burst_finish(),
        Some(BurstFinish::Reap)
    ));

    // Drain the settle timer to advance to Verifying.
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Inject stable response. Profile should reap; no Effect emitted.
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        t2,
    );
    assert!(out.effects.is_empty(), "reap_pending suppresses Effect");
    assert!(e.profiles.get(pid).is_none(), "Profile reaped at burst end");
}

#[test]
fn detach_sub_settle_recomputed_when_subs_remain() {
    // Profile with two Subs of different settle; detach the faster one;
    // remaining Sub's settle wins.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let cfg = ScanConfig::builder().recursive(true).build();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest {
            name: "fast".into(),
            resource: r,
            path: None,
            config: cfg.clone(),
            max_settle: MAX_SETTLE,
            settle: Duration::from_millis(50),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            events: NO_EVENTS,
            log_output: false,
            source_promoter: None,
        }),
        now,
    );
    let sid_fast =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid_fast).unwrap().profile;
    let _ = e.step(
        Input::AttachSub(SubAttachRequest {
            name: "slow".into(),
            resource: r,
            path: None,
            config: cfg,
            max_settle: MAX_SETTLE,
            settle: Duration::from_millis(200),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            events: NO_EVENTS,
            log_output: false,
            source_promoter: None,
        }),
        now,
    );
    // Fast Sub's settle wins on attach.
    assert_eq!(
        e.profiles.get(pid).unwrap().settle,
        Duration::from_millis(50)
    );

    // Detach the fast Sub. Remaining settle is the slow one's.
    let _ = e.step(Input::DetachSub(sid_fast), Instant::now());
    assert_eq!(
        e.profiles.get(pid).unwrap().settle,
        Duration::from_millis(200)
    );
}

// ---- on_config_diff ----

#[test]
fn config_diff_added_only_attaches_subs() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);

    let req = SubAttachRequest {
        name: "added".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.added.push(req);

    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        Instant::now(),
    );
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    assert!(
        out.probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. }))
    );
    assert_eq!(e.subs().len(), 1);
}

#[test]
fn config_diff_removed_then_added_atomic() {
    // Engine has Sub A at /anchor; ConfigDiff removes A and adds B
    // (path-based, anchored at /anchor — re-creates the slot if A's
    // detach reaped it).
    let (mut e, pid_a, sid_a, _r, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid_a);

    // Path-based add — the engine re-materializes if needed.
    let req_b = SubAttachRequest::for_path(
        "B".into(),
        std::path::PathBuf::from("/anchor"),
        ScanConfig::builder().build(), // different config_hash (non-recursive)
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let mut diff = specter_core::SubRegistryDiff::default();
    diff.removed.push(sid_a);
    diff.added.push(req_b);

    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        Instant::now(),
    );
    // A reaped (sub registry no longer has it); B added.
    assert!(e.subs().get(sid_a).is_none());
    assert_eq!(e.subs().len(), 1);
    // Single sorted StepOutput; multiple watch_ops merged.
    assert!(!out.watch_ops.is_empty());
}

// ---- on_config_diff: promoter half ----

fn promoter_req(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.to_owned(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::EMPTY,
        log_output: false,
    }
}

/// `promoters.added` runs `attach_promoter_inner` for each request,
/// registering the Promoter and emitting `PromoterAttached`.
#[test]
fn config_diff_promoter_added_attaches_promoter() {
    let mut e = Engine::new();
    // Pre-place the literal-prefix dir so the Promoter lands in
    // immediate-Active mode (no descent state to inspect for this
    // test).
    let _var_log = {
        let r = e
            .tree_mut()
            .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User);
        e.tree_mut().set_kind(r, ResourceKind::Dir);
        r
    };

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: vec![promoter_req("logs", "/var/log/*.log")],
            removed: Vec::new(),
            modified: Vec::new(),
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert_eq!(e.promoters.len(), 1, "Promoter registered");
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterAttached { name, .. } if name == "logs"
        )),
        "PromoterAttached diagnostic emitted; got {:?}",
        out.diagnostics,
    );
}

/// `promoters.removed` runs `reap_promoter_inner` for each id.
/// Cancels any in-flight probe, drops the registry entry, emits
/// `PromoterReaped`.
#[test]
fn config_diff_promoter_removed_reaps_promoter() {
    let mut e = Engine::new();
    let var_log = e
        .tree_mut()
        .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User);
    e.tree_mut().set_kind(var_log, ResourceKind::Dir);

    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    assert_eq!(e.promoters.len(), 1);

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: vec![pid],
            modified: Vec::new(),
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert!(e.promoters.get(pid).is_none(), "Promoter reaped");
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterReaped { promoter } if *promoter == pid
        )),
        "PromoterReaped diagnostic emitted; got {:?}",
        out.diagnostics,
    );
}

/// `promoters.modified` is wholesale: reap then attach. The new
/// PromoterId differs from the old (the registry mints a fresh slot
/// on attach).
#[test]
fn config_diff_promoter_modified_reaps_and_attaches() {
    let mut e = Engine::new();
    let var_log = e
        .tree_mut()
        .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User);
    e.tree_mut().set_kind(var_log, ResourceKind::Dir);

    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let old_pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: Vec::new(),
            modified: vec![(old_pid, promoter_req("logs", "/var/log/*.json"))],
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert!(e.promoters.get(old_pid).is_none(), "old Promoter reaped");
    assert_eq!(e.promoters.len(), 1, "fresh Promoter attached");
    let new_pid = e
        .promoters
        .find_by_name("logs")
        .expect("name re-registered");
    assert_ne!(new_pid, old_pid, "PromoterId minted fresh on modify");

    // Both diagnostics emitted in order: reap then attach.
    let mut saw_reap = false;
    let mut saw_attach_after_reap = false;
    for d in &out.diagnostics {
        match d {
            Diagnostic::PromoterReaped { promoter } if *promoter == old_pid => {
                saw_reap = true;
            }
            Diagnostic::PromoterAttached { promoter, name } if name == "logs" => {
                assert!(saw_reap, "attach must come after reap");
                assert_eq!(*promoter, new_pid);
                saw_attach_after_reap = true;
            }
            _ => {}
        }
    }
    assert!(
        saw_reap,
        "PromoterReaped emitted; got {:?}",
        out.diagnostics
    );
    assert!(
        saw_attach_after_reap,
        "PromoterAttached emitted after reap; got {:?}",
        out.diagnostics,
    );
}

/// Sub side runs before Promoter side. Confirms ordering by combining
/// a Sub add with a Promoter add and asserting both land in one step.
/// Also exercises that promoter and sub diagnostics merge into the
/// same `StepOutput`.
#[test]
fn config_diff_applies_both_halves_in_one_step() {
    let mut e = Engine::new();
    // Anchor for the static Sub.
    let r = e.tree_mut().ensure(None, "anchor", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    // Literal prefix for the Promoter.
    let var_log = e
        .tree_mut()
        .ensure_path(&[FS_ROOT_SEGMENT, "var", "log"], ResourceRole::User);
    e.tree_mut().set_kind(var_log, ResourceKind::Dir);

    let sub_req = SubAttachRequest {
        name: "static_a".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };

    let mut sub_diff = specter_core::SubRegistryDiff::default();
    sub_diff.added.push(sub_req);

    let diff = WatchRegistryDiff {
        subs: sub_diff,
        promoters: PromoterRegistryDiff {
            added: vec![promoter_req("dyn_a", "/var/log/*.log")],
            removed: Vec::new(),
            modified: Vec::new(),
        },
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert_eq!(e.subs().len(), 1, "Sub attached");
    assert_eq!(e.promoters.len(), 1, "Promoter attached");
    assert!(
        e.subs().find_by_name("static_a").is_some(),
        "static_a registered by name",
    );
    let promoter_attached = out
        .diagnostics
        .iter()
        .any(|d| matches!(d, Diagnostic::PromoterAttached { name, .. } if name == "dyn_a"));
    assert!(
        promoter_attached,
        "PromoterAttached emitted; got {:?}",
        out.diagnostics,
    );
}

/// Stale `PromoterId` in `removed` is a silent no-op (mirrors
/// `reap_promoter_inner`'s defensive idempotence). The bin can race
/// a ConfigDiff against an in-flight reap; this test confirms the
/// engine doesn't panic on it.
#[test]
fn config_diff_promoter_removed_with_stale_id_is_silent_noop() {
    let mut e = Engine::new();
    let stale = PromoterId::default();

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: vec![stale],
            modified: Vec::new(),
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    // No PromoterReaped diagnostic — the stale id branch returns
    // before the lifecycle emit.
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PromoterReaped { .. })),
        "stale id must not emit PromoterReaped; got {:?}",
        out.diagnostics,
    );
}

/// PrefixPending Promoter modify: v1 in PrefixPending (literal
/// prefix doesn't exist); reload modifies. Reap unwinds the
/// PrefixPending state, attach mints a fresh Promoter against the
/// new pattern (which may or may not be PrefixPending depending on
/// disk reality). The pre-flip state-branch in
/// `reap_promoter_inner` handles the PrefixPending arm cleanly.
#[test]
fn config_diff_promoter_modify_during_prefix_pending() {
    let mut e = Engine::new();
    // Don't pre-create the literal prefix → Promoter lands in
    // PrefixPending.
    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/missing/dir/*.log")),
        Instant::now(),
    );
    let old_pid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    assert!(matches!(
        e.promoters.get(old_pid).map(|q| &q.state),
        Some(PromoterState::PrefixPending(_)),
    ));
    // Descent probe was emitted.
    assert!(
        attach_out
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "attach in PrefixPending must emit a descent probe",
    );

    let diff = WatchRegistryDiff {
        promoters: PromoterRegistryDiff {
            added: Vec::new(),
            removed: Vec::new(),
            modified: vec![(old_pid, promoter_req("logs", "/different/dir/*.log"))],
        },
        ..Default::default()
    };

    let out = e.step(Input::ConfigDiff(diff), Instant::now());

    assert!(e.promoters.get(old_pid).is_none(), "v1 reaped");
    assert_eq!(e.promoters.len(), 1, "v2 attached");
    // Reap of an in-flight descent emits a Cancel.
    assert!(
        out.probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { .. })),
        "PrefixPending reap must Cancel the in-flight descent probe",
    );
}

// ---- emit_effects PerStableFile ----

#[test]
fn per_stable_file_fires_one_effect_per_created_entry() {
    // Profile with PerStableFile Sub; burst stabilizes with 2 created
    // file entries.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "fmt".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: diff_program(),
        scope: EffectScope::PerStableFile,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;

    // Complete Seed with empty baseline.
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_tree_snap(vec![])),
        }),
        now,
    );

    // FsEvent → Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Drain settle.
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Inject stable response with 2 file entries.
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    // Send same snap twice via state cycling: first probe sees the
    // change; second confirms stable. Simplify by re-routing — the
    // engine's Standard dispatch needs `current` set to the same snap
    // for stability. For this test we drive: snap1 (not stable), then
    // snap1 again (stable).
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::SubtreeOk(snap.clone()),
        }),
        t2,
    );
    // Now drain the rescheduled settle and inject the same snapshot.
    let t3 = t2 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t3) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t3,
        );
    }
    let std_corr2 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr2,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        t3,
    );
    // Stable; baseline is empty, current has 2 files → diff.created has 2.
    let per_file_effects: Vec<&specter_core::Effect> = out
        .effects
        .iter()
        .filter(|e| matches!(&e.key, DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(per_file_effects.len(), 2, "one Effect per created file");
    for eff in &per_file_effects {
        // Engine carries the unresolved ActionProgram; the resolver
        // runs in the actuator. Assert the template references the
        // diff-derived `${specter.created}` placeholder (the test fixture's
        // `diff_program()`).
        let SpawnBody::Exec(exec) = &eff.program.ops()[0].body else {
            panic!("expected SpawnBody::Exec");
        };
        assert!(
            exec.argv.iter().any(|a| a.parts.iter().any(|p| matches!(
                p,
                specter_core::ArgPart::Placeholder(specter_core::Placeholder::Created)
            ))),
            "diff_program's template references ${{specter.created}}"
        );
        // anchor_path + anchor_kind ⇒ actuator's compute_cwd("anchor", Dir) = "anchor".
        assert_eq!(eff.anchor_path.as_os_str(), "anchor");
        assert_eq!(eff.anchor_kind, specter_core::ResourceKind::Dir);
        // target_relative ⇒ SPECTER_RELATIVE_PATH source. The resolver
        // derives SPECTER_PATH = anchor_path.join(target_relative) at
        // spawn time; this assertion implicitly pins target_path to
        // "anchor/a.rs" or "anchor/b.rs".
        assert!(
            eff.target_relative == "a.rs" || eff.target_relative == "b.rs",
            "target_relative = {:?}",
            eff.target_relative,
        );
        // SPECTER_EVENT_KIND="file" derives from the DedupKey::PerFile
        // variant.
        assert!(matches!(eff.key, specter_core::DedupKey::PerFile { .. }));
    }
}

#[test]
fn per_stable_file_skips_dir_entries() {
    // Mixed Diff: 1 created File, 1 created Dir, 1 modified Dir.
    // PerStableFile must fire ONE Effect (the File), not three.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "fmt".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: diff_program(),
        scope: EffectScope::PerStableFile,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;

    // Seed completes against a snapshot already containing one Dir
    // (`subdir`). After Seed, `subdir` is in the baseline and won't
    // re-appear as `created` later.
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let seed_snap = dir_tree_snap(vec![("subdir", EntryKind::Dir, 10)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(seed_snap),
        }),
        now,
    );

    // FsEvent → Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Mixed snapshot: subdir (modified — different mtime), newdir (new
    // Dir), main.rs (new File). Diff = created=[newdir, main.rs],
    // modified=[subdir]. Only main.rs should fire.
    let mixed_snap = dir_tree_snap(vec![
        ("main.rs", EntryKind::File, 1),
        ("newdir", EntryKind::Dir, 11),
        // subdir has different mtime ⇒ counted as Modified.
        ("subdir", EntryKind::Dir, 10),
    ]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::SubtreeOk(mixed_snap.clone()),
        }),
        t2,
    );
    let t3 = t2 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t3) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t3,
        );
    }
    let std_corr2 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr2,
            outcome: ProbeOutcome::SubtreeOk(mixed_snap),
        }),
        t3,
    );

    let per_file_effects: Vec<&specter_core::Effect> = out
        .effects
        .iter()
        .filter(|e| matches!(&e.key, DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(
        per_file_effects.len(),
        1,
        "exactly ONE Effect for the File entry; Dir entries skipped"
    );
    // SPECTER_RELATIVE_PATH source.
    assert_eq!(per_file_effects[0].target_relative, "main.rs");
}

// ---------- Dedup-hash + drift suppression ----------

/// Drive a complete attach + Seed-Ok + FsEvent + two Standard-Ok responses
/// (the second confirming stability) and return the StepOutput that contains
/// the Effect emission. Common harness for SubtreeRoot dedup-hash tests.
///
/// Stability requires `current.subtree_at(target).dir_hash() ==
/// response.dir_hash()`. The first Standard probe diffs against the Seed
/// baseline (different ⇒ Settling); the second probe lands the same
/// response post-graft ⇒ stable ⇒ Effect fires.
fn drive_to_first_effect(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    root: ResourceId,
    now: Instant,
) -> StepOutput {
    // Complete Seed.
    complete_seed_burst(e, pid);
    // Inject FsEvent → Standard burst at root.
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    // Drain settle timer → Verifying.
    let settle_timer = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Batching"),
        },
        _ => panic!("expected Active"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        now + SETTLE,
    );
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    // First probe — response differs from Seed baseline ⇒ not-stable ⇒
    // Batching.
    let snap1 = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap1),
        }),
        now + SETTLE,
    );
    // Drain settle timer → Verifying again.
    let settle_timer2 = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Batching"),
        },
        _ => panic!("expected Active(Batching)"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer2,
        },
        now + SETTLE + SETTLE,
    );
    let correlation2 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    // Second probe — same content ⇒ stable ⇒ Effect.
    let snap2 = dir_tree_snap(vec![("a.rs", EntryKind::File, 1)]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: correlation2,
            outcome: ProbeOutcome::SubtreeOk(snap2),
        }),
        now + SETTLE + SETTLE,
    )
}

#[test]
fn records_fired_subs_after_subtree_effect() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let out = drive_to_first_effect(&mut e, pid, root, now);

    // First Effect fires (no prior emission).
    assert_eq!(out.effects.len(), 1, "first Standard-Ok fires Effect");
    // fired_subs now has one entry for the SubtreeRoot key.
    let p = e.profiles.get(pid).unwrap();
    assert_eq!(p.fired_subs.len(), 1);
    let key = p.fired_subs.iter().next().unwrap();
    assert!(matches!(
        key,
        DedupKey::Subtree {
            profile,
            ..
        } if *profile == pid,
    ));
}

/// `Effect.target` for a `Subtree`-keyed Effect is the Profile anchor
/// — captured from `Profile.resource` at emit time. The sort-key
/// extractor pulls `target` directly without a `&Engine` lookup; this
/// pins the emission-side capture so a future refactor that drops the
/// `target` assignment surfaces here.
#[test]
fn subtree_effect_target_is_anchor_at_emission() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let out = drive_to_first_effect(&mut e, pid, root, now);

    assert_eq!(out.effects.len(), 1, "Subtree-Ok fires one Effect");
    assert!(
        matches!(&out.effects[0].key, DedupKey::Subtree { profile, .. } if *profile == pid),
        "Effect is keyed Subtree at the burst's Profile",
    );
    assert_eq!(
        out.effects[0].target, root,
        "Subtree.target is the Profile anchor at emission time",
    );
    assert_eq!(
        e.profiles.get(pid).unwrap().resource,
        root,
        "anchor is unchanged post-emit (sanity)",
    );
}

/// `Effect.target` for a `PerFile`-keyed Effect is the file resource
/// — same value as `DedupKey::PerFile.resource` by construction. Pins
/// the redundancy: if `target` and `key.resource` ever diverge for
/// PerFile, sort and coalescing-identity would tell different stories.
#[test]
fn per_file_effect_target_matches_dedup_key_resource() {
    // Reuse the standard PerStableFile fixture: empty baseline, two
    // created Files, stable response → two PerFile Effects.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: "fmt".into(),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: diff_program(),
        scope: EffectScope::PerStableFile,
        events: NO_EVENTS,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;

    // Complete Seed with empty baseline.
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_tree_snap(vec![])),
        }),
        now,
    );

    // FsEvent → Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Drain settle.
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }

    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let snap = dir_tree_snap(vec![
        ("a.rs", EntryKind::File, 1),
        ("b.rs", EntryKind::File, 2),
    ]);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::SubtreeOk(snap.clone()),
        }),
        t2,
    );

    // Drain rescheduled settle, send same snapshot for stability.
    let t3 = t2 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t3) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t3,
        );
    }
    let std_corr2 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr2,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        t3,
    );

    let per_file: Vec<&specter_core::Effect> = out
        .effects
        .iter()
        .filter(|e| matches!(&e.key, DedupKey::PerFile { sub, .. } if *sub == sid))
        .collect();
    assert_eq!(per_file.len(), 2, "two created files ⇒ two PerFile Effects");
    for eff in &per_file {
        match &eff.key {
            DedupKey::PerFile { resource, .. } => {
                assert_eq!(
                    eff.target, *resource,
                    "PerFile.target == DedupKey::PerFile.resource by construction",
                );
            }
            DedupKey::Subtree { .. } => unreachable!("filtered above"),
        }
    }
}

#[test]
fn clears_fired_subs_on_effect_complete_failed() {
    let (mut e, pid, sid, root, _now) = engine_with_attached_sub();
    let now = Instant::now();
    let _ = drive_to_first_effect(&mut e, pid, root, now);

    // Confirm the dedup-hash entry was written.
    assert!(!e.profiles.get(pid).unwrap().fired_subs.is_empty());

    // EffectComplete::Failed clears the dedup-hash entry for this DedupKey.
    let dk = DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };
    let _ = e.step(
        Input::EffectComplete {
            sub: sid,
            key: dk,
            result: EffectOutcome::Failed {
                exit_code: Some(1),
                signal: None,
            },
        },
        now,
    );
    assert!(
        e.profiles.get(pid).unwrap().fired_subs.is_empty(),
        "Failed Effect clears the suppression entry",
    );
}

#[test]
fn recovery_seed_no_prior_emit_does_not_fire() {
    // Fresh attach → Seed-Ok → no prior `fired_subs` ⇒
    // seed_drift_observed returns an empty key set ⇒ no Effect
    // (preserves "fresh Seed never fires Effect").
    let (mut e, pid, _sid, _root, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");
    let snap = dir_tree_snap(vec![]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    assert!(out.effects.is_empty(), "fresh-Profile Seed fires no Effect");
}

/// Standard burst with a per-stable-file Sub: drift filter is `None`,
/// PerFile keys still emit per matching diff entry. This pins that the
/// SeedDrift-path narrowing (PerFile Subs skipped on `EmitMode::SeedDrift`)
/// doesn't accidentally skip PerFile emission on the unrelated Standard
/// burst path.
#[test]
fn b3_per_key_filter_does_not_affect_standard_burst_perfile_emission() {
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("fmt"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::PerStableFile,
        events: ClassSet::CONTENT,
        log_output: false,
        source_promoter: None,
    };
    let _ = e.step(Input::AttachSub(req), now);
    let pid = e.profiles.iter().next().unwrap().0;
    // Seed → Idle.
    let seed_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_tree_snap(vec![])),
        }),
        now,
    );

    // Standard burst with a created file → PerFile Effect emits.
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        now,
    );
    let mut t = now;
    let mut effect_out = None;
    let snap_with_file = dir_tree_snap(vec![("new.rs", EntryKind::File, 5)]);
    for _ in 0..6 {
        t += SETTLE * 4;
        while let Some(entry) = e.pop_expired(t) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t,
            );
        }
        let correlation = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation,
                outcome: ProbeOutcome::SubtreeOk(snap_with_file.clone()),
            }),
            t,
        );
        if !out.effects.is_empty() {
            effect_out = Some(out);
            break;
        }
    }
    let out = effect_out.expect("Standard burst stabilised and emitted");
    assert_eq!(
        out.effects.len(),
        1,
        "Standard burst with PerFile Sub fires one Effect for the new file",
    );
}

#[test]
fn has_per_file_fds_is_invariant_for_profile_lifetime() {
    // The events mask folds into `config_hash`, so every Sub on a Profile
    // shares the same events by construction. `has_per_file_fds` is
    // derived once at `Profile::new` from the events mask and never flips
    // during the Profile's lifetime.
    //
    // This test pins the new invariant: a Profile constructed with a
    // mask containing CONTENT (or METADATA) starts with the flag set,
    // and a Sub attaching via the same `(resource, config_hash)` does
    // not change it.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        name: String::from("formatter"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::PerStableFile,
        events: ClassSet::CONTENT,
        log_output: false,
        source_promoter: None,
    };
    let out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(
        e.profiles.get(pid).unwrap().has_per_file_fds,
        "CONTENT-mask Profile has has_per_file_fds = true at construction",
    );

    // A Sub with the same `(resource, max_settle, scan, events)` shares
    // the existing Profile; the flag stays true.
    let req2 = SubAttachRequest {
        name: String::from("formatter-2"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::PerStableFile,
        events: ClassSet::CONTENT,
        log_output: false,
        source_promoter: None,
    };
    let _ = e.step(Input::AttachSub(req2), Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds);

    // Detaching the second Sub leaves the Profile alive (sub_refcount > 0
    // before detach); the flag still doesn't flip because the Profile's
    // events mask is invariant.
    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(e.profiles.get(pid).unwrap().has_per_file_fds);
}

#[test]
fn structure_only_profile_has_per_file_fds_false() {
    // Inverse case: a STRUCTURE-only mask leaves `has_per_file_fds`
    // false. walk_pair then doesn't bump per-leaf watch_demand for
    // covered files.
    let mut e = Engine::new();
    let r = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        name: String::from("ls-only"),
        resource: r,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::STRUCTURE,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;
    assert!(!e.profiles.get(pid).unwrap().has_per_file_fds);
}

// ---------- Anchor-loss kind-cache invalidation ----------
//
// Per-site assertions that every dispatch path through
// `Engine::discard_anchor_state` clears the cached `Profile.kind`. The
// helper unit tests in `claims_tests.rs` pin the contract in
// isolation; these tests pin the integration at the seven production
// call sites so the kind-clear cannot regress at any one of them
// without a test failure.

/// Drive a Profile from fresh-attach into `Active(Standard, Verifying)`
/// with `pending_probe.is_some()`. Returns the live correlation.
fn drive_to_standard_verifying(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    root: ResourceId,
    now: Instant,
) -> specter_core::ProbeCorrelation {
    complete_seed_burst(e, pid);
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    let settle_timer = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            _ => panic!("expected Standard Batching"),
        },
        _ => panic!("expected Active"),
    };
    let _ = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_timer,
        },
        now + SETTLE,
    );
    e.pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight")
}

/// Drive into `Active(_, Rebasing)` by completing a Standard burst's
/// stable verdict + Effect → EffectComplete::Ok. Returns
/// `(correlation, sub_id)` so the caller can drive the rebase response.
fn drive_to_rebasing(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    sid: SubId,
    root: ResourceId,
    now: Instant,
) -> specter_core::ProbeCorrelation {
    let stable_out = drive_to_first_effect(e, pid, root, now);
    assert_eq!(
        stable_out.effects.len(),
        1,
        "Standard stable verdict fires one Effect; got {:?}",
        stable_out.effects,
    );
    let key = stable_out.effects[0].key.clone();
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key,
            result: EffectOutcome::Ok,
        },
        now + SETTLE * 3,
    );
    rebase_out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on EffectComplete::Ok")
}

#[test]
fn dispatch_seed_vanished_clears_profile_kind() {
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    assert_eq!(
        e.profiles.get(pid).unwrap().kind,
        Some(ResourceKind::Dir),
        "fresh attach caches anchor's classified kind",
    );
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "Seed-Vanished must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_seed_failed_clears_profile_kind() {
    let (mut e, pid, _sid, _r, _now) = engine_with_attached_sub();
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 5 },
        }),
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "Seed-Failed must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_standard_vanished_clears_profile_kind() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind,
        Some(ResourceKind::Dir),
        "kind cached pre-dispatch",
    );
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        now + SETTLE,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "Standard-Vanished must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_standard_failed_clears_profile_kind() {
    let (mut e, pid, _sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_standard_verifying(&mut e, pid, root, now);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 13 },
        }),
        now + SETTLE,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "Standard-Failed must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_rebase_vanished_clears_profile_kind() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_rebasing(&mut e, pid, sid, root, now);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind,
        Some(ResourceKind::Dir),
        "kind cached pre-rebase",
    );
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        now + SETTLE * 4,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "Rebase-Vanished must clear the cached anchor kind",
    );
}

#[test]
fn dispatch_rebase_failed_clears_profile_kind() {
    let (mut e, pid, sid, root, now) = engine_with_attached_sub();
    let correlation = drive_to_rebasing(&mut e, pid, sid, root, now);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Failed { errno: 5 },
        }),
        now + SETTLE * 4,
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "Rebase-Failed must clear the cached anchor kind",
    );
}

#[test]
fn finalize_anchor_lost_clears_profile_kind() {
    // Anchor terminal event during a materialised burst.
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    assert_eq!(
        e.profiles.get(pid).unwrap().kind,
        Some(ResourceKind::Dir),
        "kind cached post-Seed-Ok",
    );
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert!(
        e.profiles.get(pid).unwrap().kind.is_none(),
        "anchor terminal event must clear the cached anchor kind",
    );
}

/// Pin `finalize_anchor_lost`'s ordering invariant: `was_active` is
/// captured BEFORE `discard_anchor_state` runs. Exercises the
/// Active-burst path and asserts the burst is finished to Idle (i.e.
/// the `was_active = true` branch ran). A future helper change that
/// flips `state` mid-helper would otherwise silently break the
/// burst-end pathway.
#[test]
fn finalize_anchor_lost_was_active_pre_helper_ordering() {
    let (mut e, pid, _sid, root, _now) = engine_with_attached_sub();
    complete_seed_burst(&mut e, pid);
    // Re-enter Active by injecting an FsEvent → Standard Batching.
    let _ = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    assert!(
        matches!(
            e.profiles.get(pid).unwrap().state,
            ProfileState::Active(_, _)
        ),
        "harness pre-condition: Profile is Active",
    );

    let mut out = StepOutput::default();
    e.finalize_anchor_lost(pid, &mut out);

    let p = e.profiles.get(pid).expect("Profile lives");
    assert!(
        matches!(p.state, ProfileState::Idle),
        "was_active=true ⇒ finish_burst_to_idle ran ⇒ state is Idle; got {:?}",
        p.state,
    );
    assert!(p.kind.is_none(), "kind cleared by discard_anchor_state");
    assert_eq!(
        p.anchor_claim,
        AnchorClaim::None,
        "anchor claim released by discard_anchor_state",
    );
}

// ---------- Awaiting-absorbed events fold into rebase force_walk ----------

/// `transition_to_rebasing` consumes Awaiting-absorbed events as the
/// rebase walker's `force_walk`, defeating the POSIX content-edit
/// mtime-skip carve-out: a command that edits a descendant in place
/// (no parent-dir mtime bump) would otherwise leave the rebased
/// baseline with stale leaves. Field is cleared post-emit, mirroring
/// `transition_to_verifying`'s hygiene.
///
/// Sub uses `ClassSet::CONTENT` so the descendant `Modified` event
/// passes both gates: (1) a per-file FD is wired up by the standard
/// burst's reconcile (`has_per_file_fds = true`), bumping the leaf's
/// `watch_demand` past `on_fs_event`'s zero-gate, and (2) the
/// per-Profile class filter (which sits BEFORE `drive_burst`'s absorb
/// arm) admits the CONTENT-classed event.
#[test]
fn rebasing_ships_awaiting_absorbed_resources_as_force_walk() {
    let mut e = Engine::new();
    let root = e.tree.ensure(None, "anchor", ResourceRole::User);
    e.tree.set_kind(root, ResourceKind::Dir);
    let now = Instant::now();
    let req = SubAttachRequest {
        name: String::from("test-sub"),
        resource: root,
        path: None,
        config: ScanConfig::builder().recursive(true).build(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::CONTENT,
        log_output: false,
        source_promoter: None,
    };
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs.get(sid).unwrap().profile;

    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    assert_eq!(stable_out.effects.len(), 1, "stable verdict fires Effect");
    let key = stable_out.effects[0].key.clone();

    // Look up the descendant the standard burst's reconcile created.
    // `drive_to_first_effect` ships `[("a.rs", File, 1)]` as the
    // probe response; the engine's graft creates an `a.rs` Resource
    // under root and bumps its watch_demand (per-file FD) because the
    // Profile carries CONTENT in its events_union.
    let descendant = e
        .tree
        .lookup(Some(root), "a.rs")
        .expect("standard burst's reconcile created a.rs");
    assert!(
        e.tree.get(descendant).is_some_and(|r| r.watch_demand() > 0),
        "per-file FD must be wired up for the descendant — otherwise \
         the Modified event drops at on_fs_event's watch_demand gate \
         before reaching the absorb arm",
    );

    // Inject an FsEvent during Awaiting → absorb arm. `Modified` is
    // the in-place content-edit class — the same FsEvent kqueue emits
    // for a `write(2)` against a per-file FD, which is the carve-out
    // scenario this test pins (the parent dir's mtime is unchanged).
    let absorb_out = e.step(
        Input::FsEvent {
            resource: descendant,
            event: FsEvent::Modified,
        },
        now + SETTLE * 2,
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { resource, .. } if *resource == descendant,
        )),
        "Awaiting absorb must emit EventAbsorbedByFireTail",
    );
    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        _ => panic!("expected Active(Awaiting)"),
    };
    assert!(
        burst.force_walk_resources.contains(&descendant),
        "Awaiting absorb must accumulate event_resource into \
         force_walk_resources for the next Rebasing probe; got {:?}",
        burst.force_walk_resources,
    );

    // EffectComplete::Ok → transition_to_rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key,
            result: EffectOutcome::Ok,
        },
        now + SETTLE * 3,
    );

    let descendant_path = e.tree.path_of(descendant).expect("path resolves");
    let req = rebase_out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on EffectComplete::Ok");
    match req {
        ProbeRequest::Subtree { force_walk, .. } => {
            assert!(
                force_walk.contains(&descendant_path),
                "Rebasing probe must ship absorbed resource's path in \
                 force_walk; got {force_walk:?}",
            );
        }
        other => panic!("Rebasing on Dir-anchored Profile must emit Subtree probe; got {other:?}"),
    }

    let burst = match &e.profiles.get(pid).unwrap().state {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => post,
        _ => panic!("expected Active(Rebasing)"),
    };
    assert!(matches!(burst.phase, PostFirePhase::Rebasing));
    assert!(
        burst.force_walk_resources.is_empty(),
        "transition_to_rebasing clears force_walk_resources after \
         consuming them",
    );
}

/// Idempotent fire-tail: no FsEvent absorbs during Awaiting →
/// Rebasing probe ships empty `force_walk` so the walker mtime-skips
/// at every level. Pins the optimization against a future regression
/// where someone unconditionally extends `force_walk_resources`.
#[test]
fn rebasing_without_absorbed_events_ships_empty_force_walk() {
    let (mut e, pid, sid, root, _now0) = engine_with_attached_sub();
    let now = Instant::now();
    let stable_out = drive_to_first_effect(&mut e, pid, root, now);
    let key = stable_out.effects[0].key.clone();

    // No FsEvent during Awaiting — drive directly to EffectComplete.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key,
            result: EffectOutcome::Ok,
        },
        now + SETTLE * 3,
    );

    let req = rebase_out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request),
            ProbeOp::Cancel { .. } => None,
        })
        .expect("Rebase probe minted on EffectComplete::Ok");
    match req {
        ProbeRequest::Subtree {
            force_walk, forced, ..
        } => {
            assert!(
                force_walk.is_empty(),
                "Rebasing without absorbs ships empty force_walk \
                 (preserves walker mtime-skip); got {force_walk:?}",
            );
            assert!(!forced, "Rebasing is never forced");
        }
        other => panic!("expected Subtree probe; got {other:?}"),
    }
}

// ---------- rebase_baseline witness clears at every site ----------

/// Construct an `Active(PreFire)` state populated with default empty
/// per-burst sets and the supplied phase / intent / probe target. Used
/// by witness-clear tests that drive `dispatch_*_ok` directly with a
/// pre-staged Profile.
fn active_pre_fire_burst(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    phase: PreFirePhase,
    intent: BurstIntent,
    probe_target: ResourceId,
    now: Instant,
) -> ProfileState {
    let burst_deadline = e
        .timers
        .schedule(now + MAX_SETTLE, pid, TimerKind::BurstDeadline);
    ProfileState::Active(
        ActiveBurst::PreFire(PreFireBurst {
            burst_deadline,
            phase,
            intent,
            forced: false,
            dirty_resources: BTreeSet::new(),
            force_walk_resources: BTreeSet::new(),
            probe_target,
            suppressed_resources: BTreeSet::new(),
            last_event_time: None,
        }),
        BurstFinish::ReturnToIdle,
    )
}

/// Construct an `Active(PostFire)` state with the supplied phase /
/// intent. Used by witness-clear tests that drive `dispatch_rebase_*`
/// directly with a pre-staged Profile.
fn active_post_fire_burst(
    _e: &mut Engine,
    _pid: specter_core::ProfileId,
    phase: PostFirePhase,
    intent: BurstIntent,
    _probe_target: ResourceId,
    _now: Instant,
) -> ProfileState {
    ProfileState::Active(
        ActiveBurst::PostFire(PostFireBurst {
            intent,
            phase,
            force_walk_resources: BTreeSet::new(),
        }),
        BurstFinish::ReturnToIdle,
    )
}

#[test]
fn dispatch_rebase_ok_clears_last_settled_hash_at_loss() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();

    let state = active_post_fire_burst(
        &mut e,
        pid,
        PostFirePhase::Rebasing,
        BurstIntent::Standard,
        anchor,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.last_settled_hash_at_loss = Some(0xdead_beef);
        p.baseline = Some(TreeSnapshot::Dir(dir_tree_snap(vec![])));
        p.current = Some(TreeSnapshot::Dir(dir_tree_snap(vec![])));
        p.state = state;
    }

    let mut out = StepOutput::default();
    e.dispatch_rebase_ok(pid, TreeSnapshot::Dir(dir_tree_snap(vec![])), &mut out);

    let p = e.profiles.get(pid).expect("Profile lives");
    assert!(
        p.last_settled_hash_at_loss.is_none(),
        "witness cleared on rebase",
    );
    assert!(p.baseline.is_some(), "baseline set to current");
}

#[test]
fn dispatch_seed_ok_no_drift_branch_clears_last_settled_hash_at_loss() {
    let (mut e, pid, _sid, anchor, now) = engine_with_attached_sub();

    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying,
        BurstIntent::Seed,
        anchor,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        p.last_settled_hash_at_loss = Some(0xdead_beef);
        // Survival mode at entry: baseline = None, witness populated.
        // Empty dedup map and empty fired_subs ⇒ no drift, no fire.
        p.baseline = None;
        p.current = None;
        p.state = state;
    }

    let mut out = StepOutput::default();
    e.dispatch_seed_ok(pid, TreeSnapshot::Dir(dir_tree_snap(vec![])), now, &mut out);

    assert!(
        e.profiles
            .get(pid)
            .unwrap()
            .last_settled_hash_at_loss
            .is_none(),
        "no-drift Seed-Ok cleared witness",
    );
}

#[test]
fn dispatch_seed_ok_drift_branch_clears_last_settled_hash_at_loss_eagerly() {
    // Sub fired pre-loss; anchor lost; recovery Seed-Ok with drift.
    // The eager clear (on the drift branch, before transition_to_awaiting)
    // is what gives us the strict cross-field invariant at every step
    // boundary — not just at consumption sites.
    let (mut e, pid, sid, anchor, now) = engine_with_attached_sub();

    let dk = DedupKey::Subtree {
        sub: sid,
        profile: pid,
    };

    let state = active_pre_fire_burst(
        &mut e,
        pid,
        PreFirePhase::Verifying,
        BurstIntent::Seed,
        anchor,
        now,
    );
    if let Some(p) = e.profiles.get_mut(pid) {
        // Survival-mode drift setup. Pre-loss fire history in
        // `fired_subs` plus a witness hash that won't match the
        // post-graft current.hash() — the bool drift signal triggers
        // and the SeedDrift filter (Subtree subset of fired_subs)
        // narrows to `dk`.
        p.fired_subs.insert(dk.clone());
        p.fired_subs.insert(dk);
        p.last_settled_hash_at_loss = Some(0xdead_beef);
        p.baseline = None;
        p.current = None;
        p.state = state;
    }

    let mut out = StepOutput::default();
    e.dispatch_seed_ok(pid, TreeSnapshot::Dir(dir_tree_snap(vec![])), now, &mut out);

    // Drift branch must have fired one Effect.
    assert_eq!(out.effects.len(), 1, "drift branch emitted one Effect");

    let p = e.profiles.get(pid).unwrap();
    assert!(
        p.last_settled_hash_at_loss.is_none(),
        "drift Seed-Ok cleared witness eagerly",
    );
    assert!(
        p.baseline.is_some(),
        "drift branch rebased baseline = current",
    );
}

// ---- seed_drift_observed: baseline-or-witness contract ----

/// Fresh Profile reports no drift: `fired_subs` is empty by
/// construction, baseline/witness are both `None`. Pins the
/// "fresh Seed never fires Effect" contract — without the
/// `fired_subs.is_empty()` short-circuit, a Profile with a None
/// baseline AND a None witness AND a Some current would still
/// fall through to the `match p.baseline` arm; the short-circuit
/// is the load-bearing guard for the fresh-attach case.
#[test]
fn seed_drift_observed_returns_false_for_fresh_profile() {
    let (e, pid, _sid, _anchor, _now) = engine_with_attached_sub();
    let p = e.profiles.get(pid).expect("Profile lives post-attach");
    assert!(p.fired_subs.is_empty(), "precondition: no fire history");
    assert!(
        p.last_settled_hash_at_loss.is_none(),
        "precondition: no witness"
    );
    assert!(
        p.baseline.is_none(),
        "precondition: baseline cleared post-attach"
    );

    assert!(
        !e.seed_drift_observed(pid),
        "fresh Profile (no fire history, no settled state) reports no drift",
    );
}

/// Survival-mode drift: `discard_anchor_state` stashed the pre-loss
/// `baseline.hash()` into `last_settled_hash_at_loss`. The recovery
/// Seed-Ok lands a new `current` whose hash differs from the
/// witness — drift detected, conservative re-fire required. Pins
/// the witness-consultation arm.
#[test]
fn seed_drift_observed_returns_true_on_post_recovery_drift() {
    let (mut e, pid, sid, _anchor, _now) = engine_with_attached_sub();
    let snap = dir_tree_snap(vec![("file", EntryKind::File, 1)]);
    let curr_hash = snap.dir_hash();

    if let Some(p) = e.profiles.get_mut(pid) {
        p.fired_subs.insert(DedupKey::Subtree {
            sub: sid,
            profile: pid,
        });
        // Survival mode: baseline cleared by anchor loss; witness carries
        // the pre-loss baseline.hash(). Use a witness that intentionally
        // disagrees with `current` so drift is detected.
        p.last_settled_hash_at_loss = Some(curr_hash.wrapping_add(1));
        p.baseline = None;
        p.current = Some(TreeSnapshot::Dir(snap));
    }

    assert!(
        e.seed_drift_observed(pid),
        "survival mode: witness != current.hash() ⇒ drift detected",
    );
}

/// Active-mode drift: `baseline.is_some()` (no anchor loss has
/// occurred), `last_settled_hash_at_loss` is `None` per the
/// cross-field invariant. Drift derives from `baseline.hash() !=
/// current.hash()`. Covers the `on_sensor_overflow` reseed path —
/// the witness-only design would have lost this case (overflow
/// doesn't go through `discard_anchor_state`, so the witness stays
/// `None`); the baseline-or-witness arm preserves the conservative
/// re-fire contract that the original `fired_subs`-
/// based drift performed.
#[test]
fn seed_drift_observed_returns_true_on_active_mode_drift() {
    let (mut e, pid, sid, _anchor, _now) = engine_with_attached_sub();
    let baseline_snap = dir_tree_snap(vec![("a", EntryKind::File, 1)]);
    let current_snap = dir_tree_snap(vec![("b", EntryKind::File, 1)]);
    assert_ne!(
        baseline_snap.dir_hash(),
        current_snap.dir_hash(),
        "test setup: baseline and current must have distinct hashes",
    );

    if let Some(p) = e.profiles.get_mut(pid) {
        p.fired_subs.insert(DedupKey::Subtree {
            sub: sid,
            profile: pid,
        });
        // Active mode: baseline persists across the overflow, witness
        // remains `None` per the cross-field invariant.
        p.last_settled_hash_at_loss = None;
        p.baseline = Some(TreeSnapshot::Dir(baseline_snap));
        p.current = Some(TreeSnapshot::Dir(current_snap));
    }

    assert!(
        e.seed_drift_observed(pid),
        "active-mode (overflow) drift: baseline.hash() != current.hash() ⇒ \
         drift detected ⇒ conservative re-fire",
    );
}

/// Active-mode no-drift: `baseline.hash() == current.hash()` —
/// overflow happened to coincide with no actual disk change. The
/// witness is `None`, the baseline arm runs and reports no drift.
/// No conservative re-fire — `baseline` still represents reality.
#[test]
fn seed_drift_observed_returns_false_when_active_mode_baseline_matches_current() {
    let (mut e, pid, sid, _anchor, _now) = engine_with_attached_sub();
    let snap = dir_tree_snap(vec![("file", EntryKind::File, 1)]);

    if let Some(p) = e.profiles.get_mut(pid) {
        p.fired_subs.insert(DedupKey::Subtree {
            sub: sid,
            profile: pid,
        });
        p.last_settled_hash_at_loss = None;
        p.baseline = Some(TreeSnapshot::Dir(snap.clone()));
        p.current = Some(TreeSnapshot::Dir(snap));
    }

    assert!(
        !e.seed_drift_observed(pid),
        "active mode, baseline.hash() == current.hash() ⇒ no drift \
         (overflow without disk change)",
    );
}

// ---------- Property tests ----------

mod props {
    use super::*;
    use proptest::prelude::*;

    /// Each property generates a sequence of opaque "actions" for the
    /// engine to dispatch, then asserts a global invariant. We don't
    /// generate random `ResourceIds` — the resource always exists from a
    /// fresh attach.
    #[derive(Clone, Debug)]
    enum Action {
        FsEvent(FsEvent),
        AdvanceTime(u32), // ms to advance
        Probe,            // accept whatever probe is in flight, return Ok
        ProbeVanished,
        ProbeFailed(i32),
        EffectComplete,
    }

    fn arb_fsevent() -> impl Strategy<Value = FsEvent> {
        prop_oneof![
            Just(FsEvent::Modified),
            Just(FsEvent::StructureChanged),
            Just(FsEvent::Removed),
            Just(FsEvent::Renamed),
        ]
    }

    fn arb_action() -> impl Strategy<Value = Action> {
        prop_oneof![
            arb_fsevent().prop_map(Action::FsEvent),
            (1u32..200).prop_map(Action::AdvanceTime),
            Just(Action::Probe),
            Just(Action::ProbeVanished),
            (1i32..5).prop_map(Action::ProbeFailed),
            Just(Action::EffectComplete),
        ]
    }

    /// Apply `action` to a freshly-attached single-Profile engine; collect
    /// the `StepOutput` from each step. Returns the latest correlation seen
    /// so the next `Probe` action can target it.
    fn run_action(
        e: &mut Engine,
        sid: specter_core::SubId,
        r: ResourceId,
        action: Action,
        t: &mut Instant,
        last_correlation: &mut Option<specter_core::ProbeCorrelation>,
    ) -> StepOutput {
        let pid = e.subs.get(sid).unwrap().profile;
        let out = match action {
            Action::FsEvent(event) => e.step(Input::FsEvent { resource: r, event }, *t),
            Action::AdvanceTime(ms) => {
                *t += Duration::from_millis(u64::from(ms));
                let mut combined = StepOutput::default();
                while let Some(entry) = e.pop_expired(*t) {
                    let s = e.step(
                        Input::TimerExpired {
                            profile: entry.profile,
                            kind: entry.kind,
                            id: entry.id,
                        },
                        *t,
                    );
                    for c in s.probe_ops.iter().filter_map(|op| match op {
                        ProbeOp::Probe { request } => Some(request.correlation()),
                        _ => None,
                    }) {
                        *last_correlation = Some(c);
                    }
                    extend_step_output(&mut combined, s);
                }
                combined
            }
            Action::Probe => {
                let snap = dir_tree_snap(vec![]);
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation::from(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        owner: ProbeOwner::Profile(pid),
                        correlation: corr,
                        outcome: ProbeOutcome::SubtreeOk(snap),
                    }),
                    *t,
                )
            }
            Action::ProbeVanished => {
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation::from(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        owner: ProbeOwner::Profile(pid),
                        correlation: corr,
                        outcome: ProbeOutcome::Vanished,
                    }),
                    *t,
                )
            }
            Action::ProbeFailed(errno) => {
                let corr = last_correlation.unwrap_or(specter_core::ProbeCorrelation::from(0));
                e.step(
                    Input::ProbeResponse(ProbeResponse {
                        owner: ProbeOwner::Profile(pid),
                        correlation: corr,
                        outcome: ProbeOutcome::Failed { errno },
                    }),
                    *t,
                )
            }
            Action::EffectComplete => e.step(
                Input::EffectComplete {
                    sub: sid,
                    key: DedupKey::Subtree {
                        sub: sid,
                        profile: pid,
                    },
                    result: EffectOutcome::Ok,
                },
                *t,
            ),
        };

        // Update last_correlation from any Probe in the output.
        for c in out.probe_ops.iter().filter_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        }) {
            *last_correlation = Some(c);
        }

        out
    }

    fn extend_step_output(dst: &mut StepOutput, src: StepOutput) {
        for op in src.watch_ops {
            dst.watch_ops.push(op);
        }
        for op in src.probe_ops {
            dst.probe_ops.push(op);
        }
        for ef in src.effects {
            dst.effects.push(ef);
        }
        for d in src.diagnostics {
            dst.diagnostics.push(d);
        }
    }

    fn fresh_engine_with_sub() -> (
        Engine,
        specter_core::SubId,
        ResourceId,
        Instant,
        Option<specter_core::ProbeCorrelation>,
    ) {
        let mut e = Engine::new();
        let r = e.tree.ensure(None, "anchor", ResourceRole::User);
        e.tree.set_kind(r, ResourceKind::Dir);
        let now = Instant::now();
        let req = SubAttachRequest {
            name: String::from("test"),
            resource: r,
            path: None,
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            settle: SETTLE,
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            events: NO_EVENTS,
            log_output: false,
            source_promoter: None,
        };
        let out = e.step(Input::AttachSub(req), now);
        let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
        let last_correlation = out.probe_ops.iter().find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        });
        (e, sid, r, now, last_correlation)
    }

    proptest! {
        #![proptest_config(ProptestConfig {
            cases: 64,
            ..ProptestConfig::default()
        })]

        /// Every StepOutput is sorted canonically. Run a random sequence of
        /// inputs and verify after each step.
        #[test]
        fn prop_step_output_sorted_after_every_step(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();

            for action in actions {
                let out = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);

                // watch_ops sorted by ResourceId.
                let watch_keys: Vec<_> = out
                    .watch_ops
                    .iter()
                    .map(WatchOp::resource)
                    .collect();
                let mut sorted = watch_keys.clone();
                sorted.sort();
                prop_assert_eq!(watch_keys, sorted);

                // probe_ops sorted by ProbeOwner.
                let probe_keys: Vec<_> = out
                    .probe_ops
                    .iter()
                    .map(ProbeOp::owner)
                    .collect();
                let mut sorted_p = probe_keys.clone();
                sorted_p.sort();
                prop_assert_eq!(probe_keys, sorted_p);
            }
        }

        /// I5: at most one outstanding ProbeRequest per Profile. Track
        /// outstanding probes via emit/cancel/respond; assert ≤ 1.
        #[test]
        fn prop_at_most_one_outstanding_probe(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile;

            // attach_sub emits the initial Seed probe; outstanding = 1.
            let mut outstanding: u32 = 1;

            for action in actions {
                let was_probe = matches!(action, Action::Probe | Action::ProbeVanished | Action::ProbeFailed(_));
                let out = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);

                // Each Probe op increments; each Cancel and each accepted
                // ProbeResponse decrements. (We treat "any Probe op
                // emitted" as +1 and "any Cancel emitted" as -1; the test
                // doesn't care about the difference, only that the running
                // count stays ≤ 1.)
                let probes_emitted = out
                    .probe_ops
                    .iter()
                    .filter(|op| matches!(op, ProbeOp::Probe { .. }))
                    .count();
                let cancels_emitted = out
                    .probe_ops
                    .iter()
                    .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
                    .count();

                outstanding = outstanding
                    .saturating_add(u32::try_from(probes_emitted).unwrap_or(0))
                    .saturating_sub(u32::try_from(cancels_emitted).unwrap_or(0));

                // If a ProbeResponse action was injected and didn't cause
                // a stale-diagnostic, the outstanding probe is consumed.
                if was_probe {
                    let stale = out.diagnostics.iter().any(|d| {
                        matches!(d, Diagnostic::StaleProbeResponse { .. })
                    });
                    if !stale {
                        outstanding = outstanding.saturating_sub(1);
                    }
                }

                prop_assert!(
                    outstanding <= 1,
                    "I5: outstanding probes per profile = {} > 1",
                    outstanding,
                );

                // Channel-discipline I5: at most one outstanding probe
                // per Profile, expressed as a single `ProbeChannel` entry
                // keyed by `ProbeOwner::Profile(pid)`. `correlation_for`
                // returns `Option<ProbeCorrelation>` so `<= 1` is
                // trivially true; the assertion is a regression guard
                // against a future widening of the channel's per-owner
                // slot shape.
                let probing_count =
                    u32::from(e.pending_probe_for(ProbeOwner::Profile(pid)).is_some());
                prop_assert!(
                    probing_count <= 1,
                    "I5 channel-discipline: probe channel carries at most one outstanding probe per Profile",
                );
            }
        }

        /// `prop_seed_burst_emits_no_effects`: from a fresh attach, the
        /// Seed-burst's eventual ProbeResponse path never produces an
        /// Effect (fresh Seed bursts never emit Effects).
        #[test]
        fn prop_seed_burst_emits_no_effects(
            seed_outcome in prop_oneof![
                Just(0),  // Ok
                Just(1),  // Vanished
                Just(2),  // Failed
            ],
        ) {
            let (mut e, sid, _r, now, last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile;
            let corr = last_correlation.expect("seed probe correlation");
            let outcome = match seed_outcome {
                0 => ProbeOutcome::SubtreeOk(dir_tree_snap(vec![])),
                1 => ProbeOutcome::Vanished,
                _ => ProbeOutcome::Failed { errno: 13 },
            };
            let _ = now;
            let out = e.step(
                Input::ProbeResponse(ProbeResponse { owner: ProbeOwner::Profile(pid),
                    correlation: corr,
                    outcome,
                }),
                now,
            );
            prop_assert!(out.effects.is_empty(), "Seed bursts never emit Effects");
        }

        /// `prop_dirty_descendants_clamps_at_zero` — I4 on a single-Profile
        /// engine. After any sequence of inputs, the Profile's
        /// `dirty_descendants` is ≥ 0 (always 0 for a single Profile).
        #[test]
        fn prop_dirty_descendants_clamps_at_zero(
            actions in prop::collection::vec(arb_action(), 0..16),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            let pid = e.subs.get(sid).unwrap().profile;
            for action in actions {
                let _ = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);
                if let Some(p) = e.profiles.get(pid) {
                    // u32 is ≥ 0 by type. Confirm anyway and tighten:
                    // single Profile with no parent edges → dirty_descendants
                    // is always 0.
                    prop_assert_eq!(p.dirty_descendants, 0);
                }
            }
        }

        /// `prop_step_is_total` — for any input sequence on a fresh engine,
        /// no panic in release. Implicit by reaching this assertion. Keep
        /// as a smoke test for the random-input fuzzer.
        #[test]
        fn prop_step_is_total(
            actions in prop::collection::vec(arb_action(), 0..32),
        ) {
            let (mut e, sid, r, mut t, mut last_correlation) =
                fresh_engine_with_sub();
            for action in actions {
                let _ = run_action(&mut e, sid, r, action, &mut t, &mut last_correlation);
            }
            prop_assert!(true);
        }
    }

    /// Reference-only: avoid an "unused field" warning for `BurstIntent`.
    const _: BurstIntent = BurstIntent::Standard;
}
