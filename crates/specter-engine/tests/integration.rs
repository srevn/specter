//! Cross-module integration tests for `specter-engine`.
//!
//! Two suites:
//! - **P3-era primitives**: the `covers` predicate against a real
//!   `Tree` + `ProfileMap`. Its transitive derivation
//!   (`nearest_covering_ancestor`) and the reconfirm query built on it
//!   (`has_active_standard_descendant`) are engine-internal, so their
//!   units live inline in `coverage::tests`; the `Draining → Verifying`
//!   reconfirm is exercised end-to-end through the burst lifecycle in
//!   `tests/multi_profile.rs`.
//! - **P4 lifecycle**: full `Idle ↔ Active(Burst)` flows driven through
//!   `Engine::attach_sub` and `Engine::step` against a `MockSensor`-style
//!   harness (assertions read from `StepOutput`).

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
    dead_code
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ActiveBurst, ArgPart, ArgTemplate, BurstIntent, ChildEntry, ClassSet,
    Diagnostic, DirChild, DirMeta, DirSnapshot, EffectOutcome, EffectScope, EntryKind, FsEvent,
    FsIdentity, Input, LeafEntry, Placeholder, PostFireBurst, PostFirePhase, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, Profile, ProfileIdentity, ProfileMap,
    ProfileState, ProofAuthority, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput,
    SubAttachAnchor, SubAttachRequest, SubParams, TimerKind, Tree, WatchOp,
};
use specter_engine::{Engine, covers};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn cfg_recursive() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

fn mark_dir(tree: &mut Tree, id: ResourceId) {
    tree.set_kind(id, ResourceKind::Dir);
}

#[test]
fn engine_default_constructible() {
    let e = Engine::new();
    assert!(e.next_deadline().is_none());
}

#[test]
fn covers_handles_pattern_with_dir_bypass_in_engine_context() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let root = tree.ensure_root("root", ResourceRole::User);
    let src = tree
        .ensure_child(root, "src", ResourceRole::User)
        .expect("test live parent");
    let lib_rs = tree
        .ensure_child(src, "lib.rs", ResourceRole::User)
        .expect("test live parent");
    let lib_c = tree
        .ensure_child(src, "lib.c", ResourceRole::User)
        .expect("test live parent");
    mark_dir(&mut tree, root);
    mark_dir(&mut tree, src);
    tree.set_kind(lib_rs, ResourceKind::File);
    tree.set_kind(lib_c, ResourceKind::File);

    let p = profiles.attach(
        &mut tree,
        Profile::new(
            root,
            ProfileIdentity {
                config: ScanConfig::builder()
                    .recursive(true)
                    .pattern(specter_core::GlobPattern::compile("*.rs").unwrap())
                    .build(),
                max_settle: MAX_SETTLE,
                events: NO_EVENTS,
            },
            SETTLE,
            None,
        ),
    );
    let profile = profiles.get(p).unwrap();

    assert!(covers(profile, src, &tree), "Dir bypasses pattern");
    assert!(covers(profile, lib_rs, &tree), "matching File covered");
    assert!(
        !covers(profile, lib_c, &tree),
        "non-matching File uncovered"
    );
}

// ---------- P4 single-Profile lifecycle scenarios ----------

/// Pluck the correlation from the Probe (if any) in a `StepOutput`.
fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn diff_aware_command() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([
        ArgPart::literal("fmt"),
        ArgPart::Placeholder(Placeholder::Created),
    ])])
}

/// V5-native helper: build a `TreeSnapshot::Dir` from a list of
/// `(name, kind, inode)` triples. Multi-segment names (e.g. "sub/foo.rs")
/// are *not* supported — tests in this file use leaf-name segments only.
fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> std::sync::Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        debug_assert!(
            !name.contains('/'),
            "dir_snap takes single-component children; nested paths must be built explicitly",
        );
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild::Uncovered(FsIdentity::synthetic(inode, 0))),
            _ => ChildEntry::Leaf(LeafEntry::synthetic(
                kind,
                0,
                UNIX_EPOCH,
                FsIdentity::synthetic(inode, 0),
            )),
        };
        map.insert(CompactString::new(name), child);
    }
    Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        map,
    ))
}

/// Walk through a Standard burst — drains settle timers, injects probe
/// responses with `snap` until the burst stabilizes and emits an Effect.
/// Returns the `StepOutput` containing the Effect.
///
/// `t0` is the moment the `FsEvent` fired. The walker advances time by
/// large strides each iteration so backoff catches up.
fn drive_standard_burst_to_stable(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: std::sync::Arc<DirSnapshot>,
    t0: Instant,
) -> StepOutput {
    let mut t = t0;
    for _ in 0..8 {
        t += SETTLE * 4;
        let correlation = drain_to_probe_correlation(e, t);
        if let Some(c) = correlation {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: ProbeOwner::Profile(pid),
                    correlation: c,
                    outcome: ProbeOutcome::SubtreeProven {
                        snapshot: snap.clone(),
                        authority: ProofAuthority::Authoritative,
                    },
                }),
                t,
            );
            if !out.effects().is_empty() {
                return out;
            }
        }
    }
    panic!("Standard burst failed to stabilize within drive iterations");
}

/// Drain timers and return the most recent probe's correlation, if any
/// fired in the process.
fn drain_to_probe_correlation(e: &mut Engine, t: Instant) -> Option<ProbeCorrelation> {
    let mut last_correlation = None;
    while let Some(entry) = e.pop_expired(t) {
        let out = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t,
        );
        if let Some(c) = first_probe_correlation(&out) {
            last_correlation = Some(c);
        }
    }
    last_correlation
}

/// Expire the Seed burst's first `Settle` window and return the first
/// Seed probe's correlation. A Seed is Batching-first: no probe
/// fires at attach; the first probe materializes only after the initial
/// settle timer (`attach_now + SETTLE`) expires and the burst moves
/// `Batching → Verifying`. Lighter than [`complete_seed_burst`] — for
/// scenarios that terminate the Seed on its *first* response (Vanished,
/// a stale/late reply) and never reach the second N=2 cycle.
fn first_seed_probe(e: &mut Engine, pid: specter_core::ProfileId, t0: Instant) -> ProbeCorrelation {
    let at = t0 + SETTLE;
    while let Some(en) = e.pop_expired(at) {
        e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            at,
        );
    }
    e.pending_probe_for(ProbeOwner::Profile(pid))
        .expect("first Seed probe in flight after the initial settle expiry")
}

/// Drive a Batching-first Seed burst through its full N=2 quiescence
/// proof to `Idle`. A Seed runs the same two-settle-spaced
/// equal-sample proof as a Standard burst:
///
/// 1. expire settle #1 (`t0 + SETTLE`) → first Seed probe; respond with
///    `seed_snap`. Prior `certified` is `None`, so the verdict
///    is `Unstable` by construction → graft + re-batch.
/// 2. expire settle #2 (`t0 + SETTLE*2`) → second Seed probe; respond
///    with the hash-equal `seed_snap` → `Stable` → seed pin + rebase →
///    `Idle`.
///
/// Hash-equal both responses and within `MAX_SETTLE`, so the burst
/// reaches a clean `Stable` and is never forced. A fresh Seed emits no
/// Effects.
fn complete_seed_burst(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    seed_snap: std::sync::Arc<DirSnapshot>,
    t0: Instant,
) {
    for at in [t0 + SETTLE, t0 + SETTLE * 2] {
        while let Some(en) = e.pop_expired(at) {
            e.step(
                Input::TimerExpired {
                    profile: en.profile,
                    kind: en.kind,
                    id: en.id,
                },
                at,
            );
        }
        let c = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Seed Verifying probe in flight after settle expiry");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: c,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: Arc::clone(&seed_snap),
                    authority: ProofAuthority::Authoritative,
                },
            }),
            at,
        );
        assert!(out.effects().is_empty(), "a fresh Seed never emits Effects");
    }
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Idle,
        ),
        "Seed burst completes its N=2 proof and returns to Idle",
    );
}

/// Drive the post-fire rebase loop to Idle — the structural mirror of
/// [`complete_seed_burst`]'s Batching-first Seed N=2 proof.
///
/// The caller has driven `EffectComplete::Ok` so the burst is
/// `Active(PostFire(Rebasing))` with the first rebase probe in flight;
/// `first_rebase_corr` is that probe's correlation. Both reads carry
/// `Arc::clone(&snap)` (an idempotent command): sample 1 is `Unstable`
/// by construction (the post-fire `certified` prior is `None`) →
/// `RebaseSettling`; the `RebaseSettle` spacing timer expires →
/// `Rebasing` again; sample 2 hashes equal → `Stable` →
/// `rebase_baseline` + finish to Idle. The spacing wait is `SETTLE`,
/// far inside `max_settle`, so the loop never reaches the
/// `RebaseCeiling`. Returns the final (`Stable`-step) `StepOutput` and
/// the instant it was produced.
fn complete_rebase_loop(
    e: &mut Engine,
    pid: specter_core::ProfileId,
    snap: std::sync::Arc<DirSnapshot>,
    first_rebase_corr: ProbeCorrelation,
    t0: Instant,
) -> (StepOutput, Instant) {
    // Sample 1: prior `None` ⇒ Unstable ⇒ RebaseSettling.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: first_rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t0,
    );
    let spacing_timer = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::RebaseSettling { spacing_timer },
                ..
            }),
            _,
        ) => *spacing_timer,
        other => {
            panic!("rebase sample 1 must loop to Active(PostFire(RebaseSettling)); got {other:?}")
        }
    };

    // The `RebaseSettle` spacing timer expires → re-arm `Rebasing`.
    let t1 = t0 + SETTLE;
    let rearm_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseSettle,
            id: spacing_timer,
        },
        t1,
    );
    let corr2 = first_probe_correlation(&rearm_out)
        .expect("RebaseSettle expiry re-arms the Rebasing probe");

    // Sample 2: hash-equal ⇒ Stable ⇒ rebase_baseline + finish.
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr2,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: Arc::clone(&snap),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t1,
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle,),
        "idempotent rebase loop closes Stable → Idle",
    );
    (stable_out, t1)
}

#[test]
fn golden_path_full_lifecycle() {
    // The whole V4 spine: attach_sub → Seed → Idle → FsEvent → Standard →
    // Effect → EffectComplete → Seed → Idle. Each transition observable
    // in the StepOutputs.
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg_recursive(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = now;
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // attach_sub emits Watch (anchor) but NO Probe — a Seed is
    // Batching-first; the first Seed probe fires only on settle expiry.
    assert!(
        attach_out
            .watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { .. }))
    );
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    assert!(attach_out.effects().is_empty());

    // Seed N=2 quiescence proof → baseline = current = empty snapshot;
    // → Idle. Never emits Effects (fresh Profile).
    let snap_seed = dir_snap(vec![]);
    complete_seed_burst(&mut e, pid, snap_seed.clone(), t0);

    // FsEvent on anchor → Standard Settling. The Seed consumed two
    // settle windows (`t0 + SETTLE*2`); keep instants monotonic by
    // opening the Standard burst strictly after.
    let t1 = t0 + SETTLE * 2 + Duration::from_millis(10);
    let fs_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    assert!(fs_out.effects().is_empty());

    // Drive the Standard burst to a stable verdict (probing against the
    // empty snapshot; the burst stabilizes when current matches the
    // response). The walker advances time and injects probe responses.
    let stable_out = drive_standard_burst_to_stable(&mut e, pid, snap_seed.clone(), t1);
    assert_eq!(stable_out.effects().len(), 1);
    assert!(!stable_out.effects()[0].forced);

    // EffectComplete::Ok drives the burst out of Awaiting and into
    // Rebasing — a fresh probe is emitted at the anchor to capture the
    // post-command tree.
    let post_effect = e.step(
        Input::EffectComplete {
            sub: sid,
            key: stable_out.effects()[0].key(),
            result: EffectOutcome::Ok,
        },
        t1 + SETTLE * 16,
    );
    let rebase_correlation =
        first_probe_correlation(&post_effect).expect("post-Effect rebase probe");

    // Post-fire N=2 rebase loop (idempotent — same snapshot) → Idle;
    // baseline := current (the post-command tree).
    let (_final_out, _) = complete_rebase_loop(
        &mut e,
        pid,
        snap_seed,
        rebase_correlation,
        t1 + SETTLE * 16 + Duration::from_millis(1),
    );
}

#[test]
fn trailing_latched_anchor_event_does_not_double_fire() {
    // After a fire cycle completes (Profile → Idle, baseline := the
    // post-command tree), a trailing latched anchor FsEvent — the late
    // event the watcher now delivers post-burst — opens a spurious
    // Standard burst. Its verdict is `baseline == current` (the rebase
    // just set it) and the Sub has already fired, so B1 dedup
    // suppresses: the burst finishes to Idle with no second Effect.
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg_recursive(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = now;
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // Seed → Idle with an empty baseline. The Seed is
    // Batching-first (no probe at attach) and runs an N=2 quiescence
    // proof before pinning the empty baseline.
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let snap = dir_snap(vec![]);
    complete_seed_burst(&mut e, pid, snap.clone(), t0);

    // First fire cycle: FsEvent → Standard → Effect → EffectComplete →
    // Rebase Ok → Idle. The Sub's has_fired is set by the emission.
    // Open the Standard burst after the Seed's two settle windows so
    // instants stay monotonic.
    let t1 = t0 + SETTLE * 2 + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    let stable_out = drive_standard_burst_to_stable(&mut e, pid, snap.clone(), t1);
    assert_eq!(stable_out.effects().len(), 1, "first fire emits one Effect");
    let post_effect = e.step(
        Input::EffectComplete {
            sub: sid,
            key: stable_out.effects()[0].key(),
            result: EffectOutcome::Ok,
        },
        t1 + SETTLE * 16,
    );
    let rebase_correlation =
        first_probe_correlation(&post_effect).expect("post-Effect rebase probe");
    // Post-fire N=2 rebase loop (idempotent) → Idle; baseline := the
    // post-command tree. The helper asserts the loop closes to Idle
    // before the trailing event below.
    let _ = complete_rebase_loop(
        &mut e,
        pid,
        snap.clone(),
        rebase_correlation,
        t1 + SETTLE * 16 + Duration::from_millis(1),
    );

    // Trailing latched anchor event opens a spurious Standard burst.
    let t2 = t1 + SETTLE * 20;
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t2,
    );

    // Drive the spurious burst: every verify responds with the same
    // unchanged tree. B1 dedup must suppress — no Effect ever — and the
    // burst must finish cleanly back to Idle.
    let mut t = t2;
    let mut returned_to_idle = false;
    for _ in 0..8 {
        t += SETTLE * 4;
        if let Some(c) = drain_to_probe_correlation(&mut e, t) {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: ProbeOwner::Profile(pid),
                    correlation: c,
                    outcome: ProbeOutcome::SubtreeProven {
                        snapshot: snap.clone(),
                        authority: ProofAuthority::Authoritative,
                    },
                }),
                t,
            );
            assert!(
                out.effects().is_empty(),
                "trailing latched anchor event must not re-fire (B1 dedup); got {:?}",
                out.effects(),
            );
        }
        if matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Idle,
        ) {
            returned_to_idle = true;
            break;
        }
    }
    assert!(
        returned_to_idle,
        "spurious burst from the trailing anchor event finished cleanly to Idle — no double-fire",
    );
}

#[test]
fn vanished_during_seed_clears_baseline_and_diagnoses() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "log.txt");
    e.tree_mut().set_kind(r, ResourceKind::File);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: ScanConfig::builder().build(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "fmt".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = Instant::now();
    let out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // The Seed is Batching-first; the first Seed probe fires
    // only after the initial settle expiry. The Vanished arrives on
    // that first probe — terminal, no second N=2 cycle.
    assert!(
        first_probe_correlation(&out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let correlation = first_seed_probe(&mut e, pid, t0);

    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::Vanished,
        }),
        t0 + SETTLE + Duration::from_millis(1),
    );
    assert!(resp_out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::ProbeVanished {
            intent: BurstIntent::Seed,
            ..
        }
    )));
}

#[test]
fn pending_event_race_late_probe_response_discarded() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg_recursive(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = now;
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // The Seed is Batching-first (no probe at attach). The
    // first Seed probe materializes only after the initial settle
    // expiry; its correlation is the one the intervening FsEvent will
    // render stale.
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let stale_correlation = first_seed_probe(&mut e, pid, t0);

    // Inject FsEvent while the first Seed probe is in flight
    // (Verifying). `event_drives_batching` Cancels + disarms that
    // verify and re-arms Batching, preserving Seed intent — the
    // correlation above is now stale.
    let evt_t = t0 + SETTLE + Duration::from_millis(1);
    let _evt_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        evt_t,
    );

    // Late ProbeResponse with the now-stale correlation arrives.
    let late_resp = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: stale_correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_snap(vec![]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        evt_t + Duration::from_millis(1),
    );
    assert!(
        late_resp
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. }))
    );
    // No baseline change; Profile still Active (re-batched as Seed).
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            specter_core::ProfileState::Active(..),
        ),
        "Profile remains Active after the stale late response is discarded",
    );
    assert!(
        e.profiles().get(pid).unwrap().baseline().is_none(),
        "stale late response must not commit a baseline",
    );
}

#[test]
fn seed_burst_descendants_watched_via_first_probe() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg_recursive(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // The Seed is Batching-first: descendants are watched via
    // the *first* Seed probe, which materializes only after the initial
    // settle expiry — not at attach. The first sample's verdict is
    // Unstable (no prior), but `apply_snapshot` still runs the
    // reconcile / Watch side effects on that first response.
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let correlation = first_seed_probe(&mut e, pid, t0);

    let snap = dir_snap(vec![
        ("foo.rs", EntryKind::File, 1),
        ("bar", EntryKind::Dir, 2),
    ]);
    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        t0 + SETTLE + Duration::from_millis(1),
    );
    let watches = resp_out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    // Files don't get Watch ops; only the Dir descendant
    // contributes. The File still materializes as a Resource (for
    // PerStableFile DedupKey support), no FD.
    assert_eq!(watches, 1, "one Watch for Dir descendant only");
}

#[test]
fn force_fire_emits_effect_with_forced_true() {
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "src");
    let now = Instant::now();
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg_recursive(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = now;
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // Complete the Seed burst with an empty baseline. The Seed
    // is Batching-first (no probe at attach) and runs an N=2 quiescence
    // proof before pinning.
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    complete_seed_burst(&mut e, pid, dir_snap(vec![]), t0);

    // FsEvent → Standard Settling. Open it after the Seed's two settle
    // windows so instants stay monotonic.
    let t1 = t0 + SETTLE * 2 + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Advance past max_settle so burst_deadline fires.
    let deadline_t = t1 + MAX_SETTLE + Duration::from_millis(1);
    let probe_corr = drain_to_probe_correlation(&mut e, deadline_t);

    if let Some(corr) = probe_corr {
        // Inject a not-stable response — different snapshot.
        let snap = dir_snap(vec![("x", EntryKind::File, 99)]);
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: corr,
                outcome: ProbeOutcome::SubtreeProven {
                    snapshot: snap,
                    authority: ProofAuthority::Authoritative,
                },
            }),
            deadline_t,
        );
        assert_eq!(out.effects().len(), 1);
        assert!(
            out.effects()[0].forced,
            "force-fired Effect carries forced=true"
        );
    } else {
        panic!("burst_deadline did not produce a probe");
    }
}

#[test]
fn step_output_is_sorted() {
    // Build a multi-Watch scenario (descendants reconciled on first probe)
    // and confirm StepOutput.watch_ops is sorted by ResourceId.
    let mut e = Engine::new();
    let r = e_anchor(&mut e, "root");
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg_recursive(),
            max_settle: MAX_SETTLE,
            events: NO_EVENTS,
        },
        params: SubParams {
            name: "build".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let t0 = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), t0);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);

    // The attach output is probe-less (Batching-first Seed) and
    // carries no reconciled descendants. The multi-Watch StepOutput is
    // the *first* Seed probe response (post initial settle expiry),
    // where descendants reconcile — assert sortedness there.
    assert!(
        first_probe_correlation(&attach_out).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let correlation = first_seed_probe(&mut e, pid, t0);
    let leaves: Vec<(String, EntryKind, u64)> = (0..5)
        .map(|i| (format!("dir-{i}"), EntryKind::Dir, 100 + i))
        .collect();
    let snap = dir_snap(
        leaves
            .iter()
            .map(|(s, k, i)| (s.as_str(), *k, *i))
            .collect(),
    );
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: snap,
                authority: ProofAuthority::Authoritative,
            },
        }),
        t0 + SETTLE + Duration::from_millis(1),
    );
    let resources: Vec<ResourceId> = out
        .watch_ops
        .iter()
        .map(|op| match op {
            WatchOp::Watch { resource, .. } => *resource,
            WatchOp::Unwatch { resource } => *resource,
        })
        .collect();
    let mut sorted = resources.clone();
    sorted.sort();
    assert_eq!(resources, sorted, "watch_ops sorted by ResourceId");
}

// ---------- helpers ----------

fn e_anchor(e: &mut Engine, name: &str) -> ResourceId {
    let r = e.tree_mut().ensure_root(name, ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

fn pid_of(e: &Engine, sid: specter_core::SubId) -> specter_core::ProfileId {
    e.subs().get(sid).expect("sub exists").profile
}
