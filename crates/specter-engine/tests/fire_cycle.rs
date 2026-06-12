//! Fire-cycle integration tests. The fire-cycle unifies the observe → fire → rebase loop into a
//! single Burst whose phase walks Batching → Verifying → Awaiting → Rebasing → Idle. Tests in this
//! file pin the structural invariants:
//!
//! - The cycle terminates in one run for an idempotent command.
//! - Concurrent FsEvents during Awaiting / Rebasing are absorbed and folded into the post-fire
//!   baseline.
//! - The Awaiting counter decrements correctly across multi-Effect bursts and mixed Ok/Failed
//!   outcomes.
//! - The `gate_deadline` recovery path force-transitions to Rebasing when the actuator hangs; late
//!   completions diagnose.
//! - `reap_pending` mid-Awaiting reaps without re-probing.
//! - Anchor loss during Awaiting / Rebasing finishes the burst cleanly.
//! - The Seed-side drift path that produces zero effects skips Awaiting; the Standard-side
//!   hash-dedup suppression skips Awaiting too.

use compact_str::CompactString;
use specter_core::testkit::{covered, dir_snap, dir_snap_nested, empty_program, proven};
use specter_core::{
    ActiveBurst, BurstFinish, BurstIntent, ChildEntry, ClassSet, DedupKey, Diagnostic, DirMeta,
    DirSnapshot, EffectCompletion, EffectOutcome, EffectScope, EntryKind, FsEvent, FsIdentity,
    Input, LeafEntry, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeResponse, ProfileId, ProfileState, ProofAuthority, ResourceId,
    ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId,
    Termination, TimerKind, TreeSnapshot,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    anchor_dir, attach_structure_only, complete_effect_to_rebasing, drain_due,
    first_probe_correlation, pid_of, rebase_post_fire_to_idle, seed_to_idle,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
/// Production-realistic `EffectScope::SubtreeRoot` events mask — CONTENT in the mask sets
/// `events_witness_quiescence == true`, so a single Authoritative sample closes the verdict floor's
/// hash-equality channel.
const DEFAULT_EVENTS: ClassSet = ClassSet::DEFAULT_SUBTREE_ROOT;

/// Single-file directory snapshot with an explicit `size`, so a post-rebase read can carry a
/// different leaf hash for the same `inode` (an in-place formatter rewrite). The canonical
/// `dir_snap` bakes `size = 0` and offers no override, so this distinct sized fixture stays
/// file-local — two consumers is not a shared pattern.
fn sized_file_snap(
    name: &str,
    kind: EntryKind,
    inode: u64,
    size: u64,
) -> std::sync::Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    map.insert(
        CompactString::new(name),
        ChildEntry::Leaf(LeafEntry::synthetic(
            kind,
            size,
            UNIX_EPOCH,
            FsIdentity::synthetic(inode, 0),
        )),
    );
    Arc::new(DirSnapshot::new(
        DirMeta::synthetic(UNIX_EPOCH, FsIdentity::synthetic(0, 0)),
        0,
        map,
    ))
}

/// Subtree-root attach request returning a recursive Sub with `/bin/true`. Uses [`DEFAULT_EVENTS`]
/// so a single Authoritative sample closes the verdict floor's hash-equality channel — the
/// canonical shape every test in this file relies on. Tests that need a different mask construct
/// their own request inline.
fn subtree_request(name: &str, r: ResourceId) -> SubAttachRequest {
    SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        DEFAULT_EVENTS,
        false,
    )
}

/// Same as `subtree_request` but with `CONTENT` in the events mask so descendant `ContentChanged`
/// events pass the class filter.
fn subtree_request_with_content(name: &str, r: ResourceId) -> SubAttachRequest {
    SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    )
}

/// Drive a fresh attach (with the supplied request) through the cold-arm Seed proof → Idle. Asserts
/// the attach `StepOutput` emits the cold-walk probe (cold-arm Verifying-first: the probe is armed
/// at burst construction). Returns the `SubId`, `ProfileId`, and the instant the Seed completed.
fn attach_and_complete_seed_with(
    e: &mut Engine,
    req: SubAttachRequest,
    snap: &std::sync::Arc<DirSnapshot>,
    now: Instant,
) -> (SubId, ProfileId, Instant) {
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(e, sid);
    assert!(
        first_probe_correlation(&out).is_some(),
        "cold-arm Seed: attach emits the cold-walk probe at burst construction",
    );
    let done = seed_to_idle(e, pid, snap, now);
    (sid, pid, done)
}

/// Drive a fresh subtree-root attach through the cold-arm Seed proof → Idle. Returns the `SubId`,
/// `ProfileId`, and the instant the Seed completed.
fn attach_and_complete_seed(
    e: &mut Engine,
    r: ResourceId,
    snap: &std::sync::Arc<DirSnapshot>,
    now: Instant,
) -> (SubId, ProfileId, Instant) {
    attach_and_complete_seed_with(e, subtree_request("test", r), snap, now)
}

/// Drain timers and inject probe responses until the Standard burst reaches a stable verdict and
/// emits Effects (transitioning to Awaiting) — or exits the cycle (hash-dedup-suppressed, no Subs
/// match) and finishes to Idle. Returns the StepOutput from the verdict step.
///
/// A Standard burst's first probe diffs against the seed baseline; if the response carries a
/// different snapshot, the verdict is unstable and the burst re-arms `Batching`. The second probe
/// (with the same response) should match the just-grafted `current` and stabilise. This helper
/// drives the loop until either an Effect fires or the burst self-terminates.
fn drive_to_awaiting(
    e: &mut Engine,
    pid: ProfileId,
    r: ResourceId,
    snap: &std::sync::Arc<DirSnapshot>,
    t: Instant,
) -> StepOutput {
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::ContentChanged,
        },
        t,
    );
    let mut t_drain = t;
    let mut last_out = StepOutput::default();
    for _ in 0..8 {
        t_drain += SETTLE * 4;
        // Drain settle / burst-deadline timers to advance to Verifying.
        let mut probe_corr: Option<ProbeCorrelation> = None;
        while let Some(entry) = e.pop_expired(t_drain) {
            let s = e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                t_drain,
            );
            if let Some(c) = first_probe_correlation(&s) {
                probe_corr = Some(c);
            }
        }
        if let Some(c) = probe_corr {
            let out = e.step(
                Input::ProbeResponse(ProbeResponse {
                    owner: pid,
                    correlation: c,
                    outcome: proven(std::sync::Arc::clone(snap)),
                }),
                t_drain,
            );
            // Done when an Effect fires OR the burst returned to Idle.
            let is_idle = matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle);
            if !out.effects().is_empty() || is_idle {
                return out;
            }
            last_out = out;
        }
    }
    panic!(
        "drive_to_awaiting: burst did not stabilise within drain iterations; last_out={last_out:?}"
    );
}

#[test]
fn fire_cycle_terminates_in_one_run_for_idempotent_command() {
    // Subtree-root Sub on /src; baseline = empty. FsEvent → Standard burst → stable verdict (response
    // == seed snap) → Awaiting (one Effect). EffectComplete::Ok → Rebasing directly (probe-first; the
    // rebase probe is already in flight). The post-fire rebase closes on the Authoritative response
    // (idempotent command) → Idle, baseline == current. A fresh FsEvent identical to the first must
    // NOT re-fire — hash dedup catches it because fired_subs matches the current view.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);

    // Standard burst → Awaiting.
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    assert_eq!(
        stable_out.effects().len(),
        1,
        "one Effect emitted at stable verdict"
    );
    let effect_key = stable_out.effects()[0].key();
    let phase = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
        _ => panic!("expected Active(Awaiting)"),
    };
    assert!(matches!(
        phase,
        PostFirePhase::Awaiting { outstanding: 1, .. }
    ));

    // EffectComplete::Ok goes probe-first: Awaiting → Rebasing directly, with the WholeSubtree
    // rebase probe already in flight in this step.
    let _ = complete_effect_to_rebasing(
        &mut e,
        sid,
        effect_key,
        seed_done + Duration::from_millis(20),
    );
    let phase = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
        _ => panic!("expected Active(Rebasing)"),
    };
    assert!(matches!(phase, PostFirePhase::Rebasing(_)));

    // Post-fire rebase (answer the in-flight probe → commit) → Idle, baseline rebased.
    let _ = rebase_post_fire_to_idle(&mut e, pid, &snap, seed_done + Duration::from_millis(30));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );
    assert!(e.profiles().get(pid).unwrap().baseline().is_some());

    // Fresh FsEvent identical to the first → Standard burst starts but hash dedup suppresses the
    // Effect (current == fired_subs).
    let later_out = drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(40));
    assert!(
        later_out.effects().is_empty(),
        "hash dedup suppresses idempotent re-fire — fire-cycle terminated cleanly",
    );
    // Burst returned to Idle directly (no Awaiting because count==0).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}

#[test]
fn fire_cycle_absorbs_descendant_event_during_awaiting() {
    // Drive to Awaiting; inject an FsEvent at a covered descendant; assert EventAbsorbedByFireTail;
    // assert phase still Awaiting and outstanding unchanged.
    //
    // The Sub uses a `CONTENT` events mask so the descendant ContentChanged event passes the class
    // filter (which sits BEFORE drive_burst's absorb path). With the EMPTY default mask the event
    // would drop as `EventClassDropped` and never reach the fire-tail.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap_with_child = dir_snap(&[("child", EntryKind::Dir, 7)]);
    let (_sid, pid, seed_done) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        &snap_with_child,
        now,
    );

    // Confirm the child has watch_demand > 0 (Seed reconciler bumped it).
    assert!(
        e.tree().get(child).unwrap().watch_demand() > 0,
        "Seed reconciler watched the descendant Dir",
    );

    // Drive to Awaiting using the same snap → stable.
    let _ = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &snap_with_child,
        seed_done + Duration::from_millis(10),
    );
    let phase_before = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => format!("{:?}", post.phase),
        _ => panic!("expected Active(Awaiting)"),
    };
    assert!(phase_before.contains("Awaiting"));

    // Inject FsEvent at the covered descendant. The descendant has a watch_demand bumped via the
    // Seed's reconcile, so the event isn't dropped as "unwatched".
    let descendant_event_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        seed_done + Duration::from_millis(50),
    );
    assert!(
        descendant_event_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { profile, resource, .. }
                if *profile == pid && *resource == child,
        )),
        "descendant FsEvent absorbed during Awaiting; got diagnostics: {:?}",
        descendant_event_out.diagnostics,
    );
    assert!(
        descendant_event_out.probe_ops().is_empty(),
        "no probe emitted for absorbed event",
    );

    // Phase is unchanged.
    let phase_after = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => format!("{:?}", post.phase),
        _ => panic!("expected Active(Awaiting) post-absorb"),
    };
    assert_eq!(phase_after, phase_before, "phase unchanged after absorb");
}

#[test]
fn fire_cycle_post_rebase_residual_restarts_debounced_burst() {
    // Drive a Standard burst through the post-fire loop. An FsEvent absorbed during the Rebasing
    // round-trip (between the Awaiting → Rebasing transition and the Authoritative response) is the
    // genuine final-window residual — `transition_to_rebasing` clears `dirty` at the loop entry, so
    // only the Rebasing round-trip's absorbs survive to the Authoritative verdict. A non-empty
    // residual there restarts a fresh debounced Standard burst seeded from the residual via a typed
    // PostFire→PreFire move that preserves the watched anchor — no refcount edge changes (no
    // Unwatch/re-Watch flicker).
    //
    // CONTENT events mask: descendants must pass the class filter to reach drive_burst's absorb arm.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap = dir_snap(&[("child", EntryKind::Dir, 7)]);
    let (sid, pid, seed_done) =
        attach_and_complete_seed_with(&mut e, subtree_request_with_content("test", r), &snap, now);

    // Drive to Awaiting (a Standard burst — the FsEvent path).
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok goes probe-first: Awaiting → Rebasing directly. `transition_to_rebasing`
    // clears `dirty` at the loop entry and arms the rebase probe, so the WholeSubtree probe is
    // already in flight in this step's output.
    let rebasing_at = seed_done + Duration::from_millis(20);
    let rearm_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: effect_key,
            outcome: EffectOutcome::Ok,
        }),
        rebasing_at,
    );
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing(_),
                ..
            }),
            BurstFinish::ReturnToIdle,
        ) => {}
        other => panic!("EffectComplete::Ok must go probe-first to Rebasing; got {other:?}"),
    }
    let rebase_corr = first_probe_correlation(&rearm_out)
        .expect("EffectComplete drives Awaiting → Rebasing with the rebase probe in flight");

    // FsEvent during the Rebasing round-trip → absorbed. The Authoritative response that follows
    // pins this absorb as the final-window residual.
    let absorb_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        rebasing_at + Duration::from_millis(2),
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { profile, resource, .. }
                if *profile == pid && *resource == child,
        )),
        "FsEvent during the Rebasing round-trip absorbed",
    );

    // The anchor's kernel watch taken at start_standard_burst is held through the loop (the
    // surviving refcount).
    let watch_before = e.tree().get(r).unwrap().watch_demand();
    assert_eq!(watch_before, 1, "anchor watched for the in-flight burst");

    // Authoritative response; non-empty final-window residual ⇒ restart, NOT Idle.
    let t_restart = rebasing_at + Duration::from_millis(5);
    let restart_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: proven(snap),
        }),
        t_restart,
    );

    // A fresh debounced Standard burst is armed, carrying the residual as `dirty` provenance — the
    // LCA basis and the source of the mtime-skip-defeating obligation. ReturnToIdle is preserved
    // across the typed move.
    let child_path = Arc::clone(e.tree().get(child).unwrap().path());
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Batching { .. },
                intent: BurstIntent::Standard,
                forced: false,
                dirty,
                last_event_time,
                ..
            }),
            BurstFinish::ReturnToIdle,
        ) => {
            assert!(
                dirty.chains().contains(&child_path),
                "residual seeds the next probe's obligation chains",
            );
            assert_eq!(
                dirty.lca_path(),
                Some(child_path),
                "the lone residual resource is the component-LCA basis",
            );
            assert_eq!(
                *last_event_time,
                Some(t_restart),
                "settle window reckons from the rebase-response instant",
            );
        }
        other => panic!("expected a restarted Batching burst, got {other:?}"),
    }

    // No immediate re-probe — the restart re-enters the settle debounce, so it cannot livelock.
    assert!(
        first_probe_correlation(&restart_out).is_none(),
        "restart re-enters Batching, emits no probe",
    );

    // The kernel watch did NOT flicker: the typed PostFire→PreFire move never finished the burst,
    // so the watch is still held (not released-then-reacquired) — no refcount edge changes.
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        watch_before,
        "anchor watch held across the restart, no finish-then-start flicker",
    );
}

#[test]
fn fire_cycle_forced_rebase_terminal_restarts_from_residual() {
    // The Forced rebase terminal runs the identical final-round-trip race window as the Natural
    // one, so it takes the identical restart exit. Drive a hung actuator to the gate deadline
    // (which latches `CeilingState::Reached` and forces the FINAL rebase walk), absorb a
    // kernel-delivered in-mask event while that walk is in flight, and answer Authoritative:
    // `Stable(Forced)` must narrate the ceiling AND restart a debounced Standard burst from the
    // residual — the companion pin to
    // `fire_cycle_post_rebase_residual_restarts_debounced_burst` (the Natural terminal). The
    // restarted burst then fires the absorbed change.
    //
    // The seed snapshot carries a `Covered` child Dir (not `dir_snap`'s `Uncovered` shape): the
    // restarted burst's verify targets the dirty-LCA — the child — and the response graft must
    // splice through the baseline's child hop.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap = dir_snap_nested(&[("child", covered(dir_snap(&[("f", EntryKind::File, 10)])))]);
    let (_sid, pid, seed_done) =
        attach_and_complete_seed_with(&mut e, subtree_request_with_content("test", r), &snap, now);

    // Drive to Awaiting; the actuator hangs (the effect never completes). The gate deadline
    // (4 × max_settle) expires → handle_gate_deadline cancels the effect, latches the forced
    // ceiling, and drives the final rebase walk.
    let t_fire = seed_done + Duration::from_millis(10);
    let _ = drive_to_awaiting(&mut e, pid, r, &snap, t_fire);
    let t_gate = t_fire + MAX_SETTLE * 8;
    let gate_diags = drain_due(&mut e, t_gate);
    assert!(
        gate_diags
            .iter()
            .any(|d| matches!(d, Diagnostic::AwaitGateDeadlineForceRebasing { .. })),
        "gate deadline forces Awaiting → Rebasing; got {gate_diags:?}",
    );
    let rebase_corr = e
        .pending_probe_for(pid)
        .expect("forced rebase probe in flight");

    // A kernel-delivered, in-mask change lands during the FINAL probe round-trip — the residual
    // reset at the Rebasing entry means this absorb is exactly the final-window race the residual
    // exists to save.
    let absorb_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        t_gate + Duration::from_millis(2),
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { profile, resource, .. }
                if *profile == pid && *resource == child,
        )),
        "the final-window event is absorbed into the residual",
    );

    // The forced walk responds Authoritative (it raced the write — same snapshot). The gate
    // projects forced = true off `CeilingState::Reached` ⇒ Stable(Forced): the ceiling narrates
    // BEFORE the exit, then the non-empty residual restarts.
    let t_term = t_gate + Duration::from_millis(5);
    let term_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: proven(Arc::clone(&snap)),
        }),
        t_term,
    );
    assert!(
        term_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::RebaseCeilingForced { .. })),
        "the forced terminal still diagnoses; got {:?}",
        term_out.diagnostics,
    );
    let child_path = Arc::clone(e.tree().get(child).unwrap().path());
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Batching { .. },
                intent: BurstIntent::Standard,
                dirty,
                ..
            }),
            BurstFinish::ReturnToIdle,
        ) => {
            assert!(
                dirty.chains().contains(&child_path),
                "the residual seeds the restarted burst's obligation chains",
            );
        }
        other => {
            panic!("expected a restarted Batching burst after the Forced terminal, got {other:?}")
        }
    }
    assert!(
        first_probe_correlation(&term_out).is_none(),
        "restart re-enters Batching, emits no probe — settle-debounced, no livelock",
    );

    // The restarted burst fires the absorbed change: the settle expiry drives the verify at the
    // dirty-LCA (the child), and the fresh read observes the changed subtree (a different file
    // identity than the rebased baseline), so the Stable verdict fires rather than
    // dedup-suppressing.
    let t_settle = t_term + SETTLE * 2;
    let _ = drain_due(&mut e, t_settle);
    let restart_corr = e
        .pending_probe_for(pid)
        .expect("restarted burst's settle expiry drives Verifying");
    let fire_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: restart_corr,
            outcome: proven(dir_snap(&[("f", EntryKind::File, 11)])),
        }),
        t_settle,
    );
    assert!(
        !fire_out.effects().is_empty(),
        "the restarted burst fires the absorbed change",
    );
}

#[test]
fn fire_cycle_gate_deadline_force_transitions_to_rebasing() {
    // Drive to Awaiting; advance clock past gate_deadline; pop_expired returns the AwaitGateDeadline
    // timer; on_timer_expired runs handle_gate_deadline → AwaitGateDeadlineForceRebasing diagnostic +
    // EffectOp::Cancel emission for the profile; phase == Rebasing; rebase probe emitted.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (_sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let _stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));

    // Advance clock past gate_deadline (4 * MAX_SETTLE).
    let gate_t = seed_done + Duration::from_millis(10) + MAX_SETTLE * 8;
    let mut combined = StepOutput::default();
    while let Some(entry) = e.pop_expired(gate_t) {
        let s = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            gate_t,
        );
        let (_, probe_ops, _, cancel_effects, diagnostics) = s.into_parts();
        for d in diagnostics {
            combined.diagnostics.push(d);
        }
        for op in probe_ops.into_values() {
            combined.push_probe_op(op);
        }
        for profile in cancel_effects {
            combined.push_cancel_effect(profile);
        }
    }
    assert!(
        combined.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::AwaitGateDeadlineForceRebasing { profile, outstanding: 1 }
                if *profile == pid,
        )),
        "gate-deadline force-rebasing diagnostic emitted",
    );
    // The engine tells the actuator to abandon in-flight effects on the same edge it gives up
    // waiting on them; without this, orphaned children would hold permits, FDs, and diff-tmp files
    // until process exit.
    let cancels: Vec<_> = combined.cancel_effects().iter().collect();
    assert_eq!(
        cancels,
        vec![pid],
        "gate-deadline emits exactly one EffectOp::Cancel for the profile",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing(_),
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    let rebase_emitted = combined
        .probe_ops()
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid));
    assert!(
        rebase_emitted,
        "rebase probe emitted on gate-deadline force-transition"
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn fire_cycle_gate_deadline_on_zombie_burst_reaps_profile() {
    // Detach the only Sub mid-Awaiting → BurstFinish::Reap (the zombie burst), then let
    // gate_deadline expire. handle_gate_deadline emits AwaitGateDeadlineReap (not ForceRebasing)
    // and routes through finish_burst_to_idle → reap_profile, eliding the rebase probe a dying
    // Profile has no consumer for.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let _stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));

    let _detach_out = e.step(Input::DetachSub(sid), seed_done + Duration::from_millis(15));
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap)
        ),
        "detach during Awaiting flips the burst's finish directive to Reap",
    );

    let gate_t = seed_done + Duration::from_millis(10) + MAX_SETTLE * 8;
    let mut combined = StepOutput::default();
    while let Some(entry) = e.pop_expired(gate_t) {
        let s = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            gate_t,
        );
        let (_, probe_ops, _, cancel_effects, diagnostics) = s.into_parts();
        for d in diagnostics {
            combined.diagnostics.push(d);
        }
        for op in probe_ops.into_values() {
            combined.push_probe_op(op);
        }
        for profile in cancel_effects {
            combined.push_cancel_effect(profile);
        }
    }
    assert!(
        combined.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::AwaitGateDeadlineReap { profile, outstanding: 1 }
                if *profile == pid,
        )),
        "gate-deadline reap diagnostic emitted on zombie burst",
    );
    // The zombie burst still emits Cancel — the actuator must SIGTERM the orphaned children even
    // when the Profile is dying (the engine has already let go of any consumer for the rebased
    // baseline, but the children still hold OS resources).
    let cancels: Vec<_> = combined.cancel_effects().iter().collect();
    assert_eq!(
        cancels,
        vec![pid],
        "zombie gate-deadline emits Cancel for the dying profile",
    );
    assert!(
        e.profiles().get(pid).is_none(),
        "zombie burst reaped; Profile gone from registry",
    );
    assert!(
        combined.probe_ops().iter().all(|op| !matches!(
            op,
            ProbeOp::Probe { request } if request.owner() == pid,
        )),
        "no rebase probe emitted — the wasted round-trip on a dying Profile is elided",
    );
}

#[test]
fn fire_cycle_late_effect_complete_after_gate_deadline_diagnoses() {
    // Drive to Awaiting; force gate-deadline to Rebasing; inject EffectComplete::Ok; assert
    // EffectCompleteOutsideAwaiting.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // Force gate-deadline.
    let gate_t = seed_done + Duration::from_millis(10) + MAX_SETTLE * 8;
    while let Some(entry) = e.pop_expired(gate_t) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            gate_t,
        );
    }
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing(_),
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));

    // Late EffectComplete::Ok arrives in Rebasing → diagnoses.
    let late_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: effect_key,
            outcome: EffectOutcome::Ok,
        }),
        gate_t + Duration::from_millis(1),
    );
    assert!(
        late_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }
                if *s == sid && *p == pid,
        )),
        "late completion in Rebasing diagnoses",
    );
    // Phase unchanged (still Rebasing).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing(_),
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn fire_cycle_anchor_loss_during_awaiting_drops_burst() {
    // Drive to Awaiting; inject anchor terminal event; finalize_anchor_lost releases anchor,
    // finishes burst → Parked (root anchor — no recovery parent). Inject late EffectComplete →
    // diagnoses outside Awaiting.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // Anchor terminal event → finalize_anchor_lost → finish_burst_to_idle.
    let lost_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        seed_done + Duration::from_millis(15),
    );
    // No probe Cancel emitted (Awaiting has no probe in flight).
    let cancels = lost_out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
        .count();
    assert_eq!(
        cancels, 0,
        "no probe in flight during Awaiting; nothing to cancel"
    );
    // Root anchor — no recovery parent, so the loss wrapper's fallback parks; baseline cleared.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Parked
    ));
    assert!(e.profiles().get(pid).unwrap().baseline().is_none());

    // Late EffectComplete → diagnoses (Profile is Parked, not Awaiting).
    let late_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: effect_key,
            outcome: EffectOutcome::Ok,
        }),
        seed_done + Duration::from_millis(20),
    );
    assert!(
        late_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }
                if *s == sid && *p == pid,
        )),
        "late completion after anchor loss diagnoses",
    );
}

#[test]
fn fire_cycle_anchor_loss_during_rebasing_cancels_probe() {
    // Drive to Rebasing; inject anchor terminal event; cancel_pending_probe emits ProbeOp::Cancel;
    // the burst finishes and the root anchor parks (no recovery parent).
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok goes probe-first: Awaiting → Rebasing directly, rebase probe already in
    // flight.
    let rebasing_at = seed_done + Duration::from_millis(20);
    e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: effect_key,
            outcome: EffectOutcome::Ok,
        }),
        rebasing_at,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing(_),
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));

    // Anchor terminal event during Rebasing.
    let lost_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        rebasing_at + Duration::from_millis(1),
    );
    // Probe Cancel emitted (Rebasing's probe in flight).
    let cancels = lost_out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { owner: profile} if *profile == pid))
        .count();
    assert_eq!(cancels, 1, "Rebasing probe cancelled on anchor loss");
    // Root anchor — no recovery parent, so the loss wrapper's fallback parks.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Parked
    ));
}

#[test]
fn fire_cycle_fresh_seed_skips_awaiting() {
    // Covers the **no-activity** fresh Seed: a fresh attach with NO FsEvents injected. With an
    // empty `dirty` provenance, `seed_owes_first_fire` is false and `seed_drift_observed` is false
    // (never-fired) ⇒ `classify_consequence` yields the silent `SilentPin` ⇒ finish_to_idle
    // directly, no Awaiting tail. Probe 1 (Retry, prior None) re-batches into PreFire(Batching);
    // probe 2 (Stable, hash-equal) pins straight to Idle. The witnessed-activity case (a fresh Seed
    // that *did* see events fires one Effect and *does* enter Awaiting) is covered by the
    // `fresh_seed_fires::*` reproduction tests — this test deliberately exercises only the
    // silent-pin path.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let out = e.step(Input::AttachSub(subtree_request("test", r)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);
    assert!(
        first_probe_correlation(&out).is_some(),
        "cold-arm Seed: attach emits the cold-walk probe at burst construction",
    );

    let snap = dir_snap(&[]);
    // The cold-arm Seed burst pins on the first Authoritative sample: dispatch reaches `SilentPin`
    // (no fired Subs, no drift) and finishes to Idle. A fresh Seed never fires an Effect and never
    // lands in a post-fire Awaiting tail.
    let corr = e
        .pending_probe_for(pid)
        .expect("cold-arm Seed Verifying probe in flight at burst construction");
    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(Arc::clone(&snap)),
        }),
        now,
    );
    assert!(
        resp_out.effects().is_empty(),
        "fresh Seed never fires Effects on the Authoritative pin",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(
        !e.subs().any_fired(pid),
        "fresh Seed leaves all Subs unfired",
    );
}

#[test]
fn fire_cycle_mixed_ok_failed_decrements_uniformly() {
    // Per-stable-file Sub on /src; baseline = empty. FsEvent batch creates 2 files (driven via the
    // test by injecting a snapshot with 2 leaves). Standard burst → 2 PerFile Effects emitted;
    // Awaiting outstanding=2. Inject one EffectComplete::Ok then one EffectComplete::Failed; the
    // counter decrements uniformly to 0; transition to Rebasing.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();

    // Per-stable-file requires CONTENT in the events mask.
    let req = SubAttachRequest::for_anchor(
        "fmt".into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::PerStableFile,
        ClassSet::CONTENT,
        false,
    );
    let (sid, pid, seed_done) = attach_and_complete_seed_with(&mut e, req, &dir_snap(&[]), now);

    // Standard burst with two files in the response.
    let snap_with_files = dir_snap(&[("a.txt", EntryKind::File, 1), ("b.txt", EntryKind::File, 2)]);
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &snap_with_files,
        seed_done + Duration::from_millis(10),
    );
    assert_eq!(
        stable_out.effects().len(),
        2,
        "two PerStableFile Effects emitted",
    );
    let phase = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
        _ => panic!(),
    };
    assert!(matches!(
        phase,
        PostFirePhase::Awaiting { outstanding: 2, .. }
    ));
    let key_a = stable_out.effects()[0].key();
    let key_b = stable_out.effects()[1].key();

    // First completion: Ok → outstanding=1.
    e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: key_a,
            outcome: EffectOutcome::Ok,
        }),
        seed_done + Duration::from_millis(20),
    );
    let phase = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
        _ => panic!(),
    };
    assert!(matches!(
        phase,
        PostFirePhase::Awaiting { outstanding: 1, .. }
    ));

    // Second completion: Failed → outstanding=0 → LastReached. The last completion goes probe-first
    // to Rebasing directly (the Failed outcome decrements the counter uniformly, same as Ok), with
    // the rebase probe already in flight in this step's output.
    let second_complete_at = seed_done + Duration::from_millis(30);
    let rebase_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: key_b,
            outcome: EffectOutcome::Failed(Termination::Exit(1)),
        }),
        second_complete_at,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing(_),
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    assert!(
        first_probe_correlation(&rebase_out).is_some(),
        "rebase probe emitted probe-first on the last EffectComplete"
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn fire_cycle_reap_pending_during_awaiting_reaps_at_gate_close() {
    // Drive to Awaiting; detach the only Sub → reap_pending=true, phase still Awaiting. Inject
    // EffectComplete::Ok → last completion (LastReached) + BurstFinish::Reap → finish_burst_to_idle
    // → reap_profile (deferred). Profile gone from registry; ProfileReaped(DeferredFromBurst)
    // diagnostic.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // Detach the only Sub. Profile is Active(Awaiting) → reap_pending=true.
    let _detach_out = e.step(Input::DetachSub(sid), seed_done + Duration::from_millis(15));
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap)
        ),
        "reap_pending set on Active profile detach",
    );

    // EffectComplete::Ok → LastReached + BurstFinish::Reap → finish_burst_to_idle → reap_profile.
    let reap_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: effect_key,
            outcome: EffectOutcome::Ok,
        }),
        seed_done + Duration::from_millis(20),
    );
    assert!(
        reap_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProfileReaped {
                profile,
                via: specter_core::ReapTrigger::DeferredFromBurst,
            } if *profile == pid,
        )),
        "ProfileReaped(DeferredFromBurst) diagnostic on reap-during-Awaiting",
    );
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped from registry",
    );
}

#[test]
fn fire_cycle_zombie_rebase_retry_short_circuits_to_reap() {
    // A burst whose last Sub detached mid-tail (`BurstFinish::Reap`) has no consumer for the
    // baseline its rebase loop would walk to certify. The next Retry-folding response must finish
    // the burst (and reap the Profile) instead of walking settle-spaced `WholeSubtree` probes to
    // the ceiling — mirroring `handle_gate_deadline`'s zombie route.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);

    // Standard fire → Awaiting → EffectComplete::Ok → Rebasing (probe in flight).
    let t_fire = seed_done + Duration::from_millis(10);
    let stable_out = drive_to_awaiting(&mut e, pid, r, &snap, t_fire);
    let key = stable_out.effects()[0].key();
    let t_rebase = t_fire + SETTLE * 8;
    let _ = complete_effect_to_rebasing(&mut e, sid, key, t_rebase);
    let corr = e.pending_probe_for(pid).expect("rebase probe in flight");

    // First refusal (unforced `Undischarged`) folds Retry: the live burst loops to Settling.
    let unread: Arc<std::path::Path> = Arc::from(std::path::Path::new("/src/opaque"));
    let undischarged = || ProbeOutcome::SubtreeProven {
        snapshot: Arc::clone(&snap),
        authority: ProofAuthority::Undischarged {
            first_unread: Arc::clone(&unread),
        },
    };
    let t1 = t_rebase + Duration::from_millis(5);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: undischarged(),
        }),
        t1,
    );
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), BurstFinish::ReturnToIdle)
            if matches!(post.phase, PostFirePhase::Settling { .. }) => {}
        other => panic!("unforced Undischarged rebase response loops to Settling; got {other:?}"),
    }

    // The last Sub detaches mid-Settling — the burst is now a zombie.
    let _ = e.step(Input::DetachSub(sid), t1 + Duration::from_millis(1));
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap),
        ),
        "last-Sub detach mid-burst defers the reap to burst end",
    );

    // The settle expiry drives the already-armed loop once more (Settling → Rebasing probe); the
    // Retry-folding response then short-circuits: finish + reap in this step, zero further probes.
    let t2 = t1 + SETTLE * 2;
    let _ = drain_due(&mut e, t2);
    let corr = e
        .pending_probe_for(pid)
        .expect("loop probe in flight after the settle expiry");
    let term_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: undischarged(),
        }),
        t2 + Duration::from_millis(1),
    );
    assert!(
        term_out.probe_ops().is_empty(),
        "the zombie short-circuit emits no further probe",
    );
    assert!(
        e.profiles().get(pid).is_none(),
        "the zombie burst finished and the Profile reaped — not another Settling window",
    );
    let _ = drain_due(&mut e, t2 + Duration::from_hours(1));
    assert!(
        e.profiles().get(pid).is_none(),
        "an hour of timer drains produces nothing further",
    );
}

#[test]
fn fire_cycle_burst_deadline_during_awaiting_dropped_silently() {
    // The pre-fire BurstDeadline timer scheduled at start_standard_burst remains in the heap when
    // the burst transitions to Awaiting. Once the burst is post-fire, is_timer_referenced filters
    // BurstDeadline out of Awaiting — pop_expired drops the stale entry without dispatching
    // handle_burst_deadline (which would otherwise try to re-emit a probe).
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (_sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let _ = drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let pending_probe_before = e.pending_probe_for(pid);

    // Advance well past max_settle (the BurstDeadline) but stop short of the gate_deadline (4 *
    // max_settle).
    let post_burst_deadline = seed_done + Duration::from_millis(10) + MAX_SETTLE * 2;
    let mut combined = StepOutput::default();
    while let Some(entry) = e.pop_expired(post_burst_deadline) {
        let s = e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            post_burst_deadline,
        );
        let (_, probe_ops, _, _, _) = s.into_parts();
        for op in probe_ops.into_values() {
            combined.push_probe_op(op);
        }
    }
    // No probe emitted — BurstDeadline filtered out, gate_deadline not yet expired (4× max_settle
    // vs 2×).
    assert!(
        combined.probe_ops().is_empty(),
        "stale BurstDeadline in Awaiting does not emit a probe",
    );
    // Phase still Awaiting.
    let phase = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
        _ => panic!(),
    };
    assert!(matches!(phase, PostFirePhase::Awaiting { .. }));
    assert_eq!(
        e.pending_probe_for(pid),
        pending_probe_before,
        "no probe minted"
    );
    // Use the imported types so dead_code rules don't trip on tests.
    let _ = (DedupKey::default(), TimerKind::Settle);
}

#[test]
fn fire_cycle_concurrent_user_edit_during_awaiting_folds_into_baseline() {
    // Concurrent user edit during Awaiting on a covered descendant: absorbed into the fire-tail. The
    // post-fire rebase captures the post-edit state via its WholeSubtree read; the user's edit folds
    // into the new baseline; it does not fire its own Effect (v1 documented loss-of-fidelity).
    //
    // CONTENT events mask so the ContentChanged event passes the class filter.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap_initial = dir_snap(&[("child", EntryKind::Dir, 7)]);
    let (sid, pid, seed_done) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        &snap_initial,
        now,
    );

    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &snap_initial,
        seed_done + Duration::from_millis(10),
    );
    let effect_key = stable_out.effects()[0].key();

    // User edits the child (concurrent with the in-flight Effect). The event is absorbed into the
    // fire-tail during Awaiting.
    e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::ContentChanged,
        },
        seed_done + Duration::from_millis(15),
    );
    // Effect completes — probe-first: Awaiting → Rebasing directly. The absorbed edit is cleared
    // from `dirty` at the Rebasing entry (`reset_residual`); the WholeSubtree rebase walk
    // re-observes it regardless via the post-edit response below.
    let _ = complete_effect_to_rebasing(
        &mut e,
        sid,
        effect_key,
        seed_done + Duration::from_millis(20),
    );

    // The rebase read carries the post-edit snapshot (the user's edit changed the directory; the
    // post-command tree is now quiescent at that state). The rebase settles on it and the
    // post-rebase baseline reflects the new state.
    let snap_after_edit = dir_snap(&[
        ("child", EntryKind::Dir, 7),
        ("user_edit.txt", EntryKind::File, 99),
    ]);
    let r = rebase_post_fire_to_idle(
        &mut e,
        pid,
        &snap_after_edit,
        seed_done + Duration::from_millis(30),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );
    // No second Effect — the rebase path never emits; the user's edit folded into baseline silently.
    assert!(
        r.finish.effects().is_empty(),
        "v1 loss-of-fidelity: user edit during fire-tail does not fire its own Effect",
    );
    // baseline reflects the post-edit tree.
    let baseline = e.profiles().get(pid).unwrap().baseline().unwrap();
    match baseline {
        TreeSnapshot::Dir(arc) => {
            assert!(
                arc.entries().contains_key("user_edit.txt"),
                "baseline includes the user's edit",
            );
        }
        TreeSnapshot::File(_) => panic!("expected Dir baseline"),
    }
}

#[test]
fn fire_cycle_standard_b1_suppresses_post_rebase_phantom_for_non_idempotent_command() {
    // A non-idempotent command rewrites the watched tree mid-burst. The post-fire rebase sets
    // baseline := current := the post-Effect tree. The next Standard burst probes that same
    // post-Effect tree, so structural B1 (`baseline.hash() == current.hash()` AND the Sub already
    // fired) suppresses the phantom — no second Effect for the same intent.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();

    let pre_emit = dir_snap(&[]);
    let post_effect = dir_snap(&[("post.rs", EntryKind::File, 42)]);
    assert_ne!(
        pre_emit.dir_hash(),
        post_effect.dir_hash(),
        "test sanity: pre/post-Effect hashes differ",
    );

    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &pre_emit, now);

    // Burst 1 — verify response = pre_emit. The Standard verify folds against the seed baseline to
    // `Stable`; emit_effects fires one Effect and records the Sub's fire history (the B1 gate for
    // burst 2).
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &pre_emit,
        seed_done + Duration::from_millis(10),
    );
    assert_eq!(stable_out.effects().len(), 1, "burst 1 fires one Effect");
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok goes probe-first: Awaiting → Rebasing directly, rebase probe already in
    // flight (answered inside rebase_post_fire_to_idle below).
    let _ = complete_effect_to_rebasing(
        &mut e,
        sid,
        effect_key,
        seed_done + Duration::from_millis(20),
    );

    // The rebase read = post_effect (non-idempotent — the command rewrote the tree, which is now
    // quiescent at the post-Effect state). The rebase settles Stable: dispatch_rebase_ok grafts and
    // rebases baseline := post_effect.
    let _ = rebase_post_fire_to_idle(
        &mut e,
        pid,
        &post_effect,
        seed_done + Duration::from_millis(30),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );

    // Post-rebase: baseline := current (= post_effect). The fire history records the Sub's Subtree
    // key — used to gate the B1 suppress in the phantom burst below.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert_eq!(
        p.baseline().unwrap().hash(),
        post_effect.dir_hash(),
        "rebase aligned baseline with the post-Effect tree",
    );

    // Burst 2 — phantom event. The verify probe responds with post_effect (the tree the user
    // actually has now). B1 dedup derives suppress from `baseline.hash() == current.hash()` AND
    // `fired_subs.contains(dk)` — both true here, so the phantom is suppressed.
    let phantom_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &post_effect,
        seed_done + Duration::from_millis(40),
    );
    assert!(
        phantom_out.effects().is_empty(),
        "B1 dedup suppresses post-rebase phantom for non-idempotent command",
    );
    // Burst returned to Idle (no Awaiting because count==0).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}

#[test]
fn fire_cycle_perfile_suppresses_post_rebase_phantom_for_non_idempotent_format() {
    // PerFile mirror of the Subtree test. A formatter-style non-idempotent command rewrites
    // foo.rs's content **in place** (same inode, different leaf-hash inputs — `size` here, the same
    // shape as a real formatter's `mtime`/`size` change). The slot survives `graft` (same
    // inode/device → identity match), so the PerFile dedup entry survives the purge. Post-rebase
    // the baseline carries the post-Effect leaf hash, so a phantom event at the same file diffs
    // empty against the rebased baseline — no re-fire.
    //
    // `sized_file_snap` builds a `foo.rs` LeafEntry with an explicit `size` so post-rebase carries
    // a different leaf hash for the same `inode` (the canonical `dir_snap` bakes `size = 0`).
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();

    // PerStableFile Sub on the anchor; CONTENT events so per-leaf FDs are issued. Seed baseline
    // empty.
    let req = SubAttachRequest::for_anchor(
        "fmt".into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::PerStableFile,
        ClassSet::CONTENT,
        false,
    );
    let (sid, pid, seed_done) = attach_and_complete_seed_with(&mut e, req, &dir_snap(&[]), now);

    // Burst 1 — verify response = pre_emit (foo.rs at inode 42, size 0). The Seed → Standard diff
    // (created foo.rs) drives one PerFile Effect.
    let pre_emit = sized_file_snap("foo.rs", EntryKind::File, 42, 0);
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &pre_emit,
        seed_done + Duration::from_millis(10),
    );
    assert_eq!(
        stable_out.effects().len(),
        1,
        "one PerFile Effect for foo.rs"
    );
    let effect_key = stable_out.effects()[0].key();
    assert!(
        matches!(effect_key, DedupKey::PerFile { .. }),
        "expected PerFile key for foo.rs",
    );

    // EffectComplete::Ok goes probe-first: Awaiting → Rebasing directly, minting the WholeSubtree
    // rebase probe in this step's output.
    let effect_complete_at = seed_done + Duration::from_millis(20);
    let rebase_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key: effect_key,
            outcome: EffectOutcome::Ok,
        }),
        effect_complete_at,
    );
    let rebase_corr = first_probe_correlation(&rebase_out)
        .expect("EffectComplete drives Awaiting → Rebasing with the rebase probe in flight");

    // Rebase response: foo.rs at the **same inode 42** (in-place formatter rewrite, slot identity
    // preserved) but `size = 1` — changes the leaf hash without triggering a delete/create cycle.
    let post_effect = sized_file_snap("foo.rs", EntryKind::File, 42, 1);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: rebase_corr,
            outcome: proven(post_effect.clone()),
        }),
        effect_complete_at + Duration::from_millis(5),
    );

    // Post-rebase: baseline := current carries the post-Effect leaf hash; the fire history records a
    // PerFile key keyed at the file resource (slot survived graft via inode identity). Both signals
    // gate the phantom-suppress path below — validated behaviourally by that burst producing no fire.

    // Burst 2 — phantom event. The verify probe responds with post_effect (foo.rs at inode 42, size
    // 1 — the "formatted" content). The diff is empty (baseline == response), so
    // `emit_effects_per_stable_file` walks zero entries — no fire. The Subtree-arm B1 suppress
    // (`baseline.hash() == current.hash()` AND `fired_subs.contains(dk)`) holds for the SubtreeRoot
    // key implicitly recorded alongside the PerFile one — so the burst returns to Idle without
    // entering Awaiting.
    let phantom_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &post_effect,
        seed_done + Duration::from_millis(40),
    );
    assert!(
        phantom_out.effects().is_empty(),
        "B1 dedup suppresses PerFile phantom for non-idempotent format",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}

/// PerStableFile contract regression: a `PerStableFile` Sub's Effect fires iff its file is in the
/// diff, re-fires on a *subsequent real change* to that file, and is deduped by **nothing but diff
/// membership** — in particular it is NOT gated by the per-Sub `Sub.has_fired` flag (which the
/// relocation introduced for the Subtree B1 path only).
///
/// The load-bearing step is Burst 2: `Sub.has_fired` is already `true` from Burst 1, yet a real
/// `foo.rs` content change must still fire a fresh PerFile Effect. If a future maintainer
/// re-introduces a spurious PerFile suppression gate keyed on fire history, Burst 2 emits zero
/// effects and this test fails.
#[test]
fn fire_cycle_perfile_refires_on_real_change_not_gated_by_fire_history() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();

    // PerStableFile Sub on the anchor; CONTENT events so per-leaf FDs are issued. Seed baseline
    // empty.
    let req = SubAttachRequest::for_anchor(
        "fmt".into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::PerStableFile,
        ClassSet::CONTENT,
        false,
    );
    let (sid, pid, seed_done) = attach_and_complete_seed_with(&mut e, req, &dir_snap(&[]), now);

    // Burst 1 — foo.rs created (inode 42, size 0). Seed → Standard diff (created foo.rs) drives
    // exactly one PerFile Effect.
    let v1 = sized_file_snap("foo.rs", EntryKind::File, 42, 0);
    let out1 = drive_to_awaiting(&mut e, pid, r, &v1, seed_done + Duration::from_millis(10));
    let perfile1: Vec<_> = out1
        .effects()
        .iter()
        .filter(|ef| matches!(ef.key(), DedupKey::PerFile { sub, .. } if sub == sid))
        .collect();
    assert_eq!(
        perfile1.len(),
        1,
        "Burst 1: PerFile Effect fires for the created foo.rs",
    );
    let key1 = perfile1[0].key();

    // EffectComplete::Ok goes probe-first to Rebasing. Idempotent command: rebase response leaves
    // foo.rs unchanged (inode 42, size 0). baseline := current carries foo.rs.
    let _ = complete_effect_to_rebasing(&mut e, sid, key1, seed_done + Duration::from_millis(20));
    // The rebase read leaves foo.rs unchanged (inode 42, size 0) → Stable, baseline := current
    // carries foo.rs.
    let _ = rebase_post_fire_to_idle(&mut e, pid, &v1, seed_done + Duration::from_millis(30));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );
    // A PerStableFile Sub's fire-history flag is NEVER set: `mark_fired` is called only by the
    // SubtreeRoot emit arm. PerFile has no B1 fire-history suppression — it is
    // diff-membership-gated only, so there is no flag to set and none to dedup against.
    assert!(
        !e.subs().get(sid).unwrap().has_fired(),
        "PerStableFile Sub is never fire-history-marked (mark_fired is SubtreeRoot-only)",
    );

    // Burst 2 — a *real* change: foo.rs rewritten in place (same inode 42, size 0 → 1). The diff
    // carries foo.rs as modified, so the PerFile Effect MUST re-fire. PerFile emission is gated by
    // diff membership alone, never by any fire-history suppression.
    let v2 = sized_file_snap("foo.rs", EntryKind::File, 42, 1);
    let out2 = drive_to_awaiting(&mut e, pid, r, &v2, seed_done + Duration::from_millis(40));
    let perfile2 = out2
        .effects()
        .iter()
        .filter(|ef| matches!(ef.key(), DedupKey::PerFile { sub, .. } if sub == sid))
        .count();
    assert_eq!(
        perfile2, 1,
        "Burst 2: PerFile Effect RE-FIRES on a real foo.rs change; \
         PerFile is gated by diff membership alone, never fire history",
    );
}

/// The user-reported scp regression, reduced to its Standard-burst shape. A structure-only Profile
/// attached to a Dir; scp creates the destination file (a `StructureChanged` event at the anchor)
/// then streams data into it across many settle windows. Pre-Layer-C the verdict's
/// `Stable(Natural)` folded to `Stable` on the first sample regardless of mask, firing seconds into
/// a multi-minute transfer (kernel-silent without tree-quiescent — no per-file FDs wired, no
/// `CONTENT` subscription to catch `NOTE_WRITE`).
///
/// Layer-C: the hash channel is active (events-incomplete + fire-bearing burst), so the carrier
/// holds the fire until two consecutive samples observe equal `dir_hash`. Two settle-spaced
/// still-moving samples (`size = 10` → `size = 4096`) fold to `Retry`; the third sample (file
/// stabilised) closes `Stable` and the burst fires exactly once.
#[test]
fn scp_into_structure_only_does_not_fire_during_growing_file() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "dest");
    let now = Instant::now();
    let (_sid, pid) = attach_structure_only(&mut e, r, now);

    // Cold-Seed bypass: an empty-dir baseline pins on one Authoritative sample (no events, no fires
    // ⇒ `owes_proof_from` is false ⇒ `EventsReliable` witness even on a structure-only mask).
    let seed_done = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Open the Standard burst — the `scp` create at the anchor.
    let burst_start = seed_done + Duration::from_millis(1);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        burst_start,
    );

    // Two settle-spaced still-moving samples (file growing in place). The carrier observes two
    // distinct hashes; both fold to Retry. **No fire** — the regression-guarded contract.
    let s1 = sized_file_snap("scp.bin", EntryKind::File, 21, 10);
    let s2 = sized_file_snap("scp.bin", EntryKind::File, 21, 4096);
    assert_ne!(
        s1.dir_hash(),
        s2.dir_hash(),
        "the growing-leaf samples must hash distinctly so the carrier observes disagreement",
    );
    let mut at = burst_start;
    for sample in [&s1, &s2] {
        at += SETTLE * 2;
        drain_due(&mut e, at);
        let corr = e
            .pending_probe_for(pid)
            .expect("Verifying probe in flight after settle expiry");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: pid,
                correlation: corr,
                outcome: proven(Arc::clone(sample)),
            }),
            at,
        );
        assert!(
            out.effects().is_empty(),
            "Layer-C hash channel holds the fire — a still-moving sample must NOT fire",
        );
    }

    // Third sample: the file is now stable. carrier prior == response ⇒ Stable ⇒ fire.
    at += SETTLE * 2;
    drain_due(&mut e, at);
    let corr = e
        .pending_probe_for(pid)
        .expect("Verifying probe in flight for the stabilised sample");
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: proven(Arc::clone(&s2)),
        }),
        at,
    );
    assert_eq!(
        stable_out.effects().len(),
        1,
        "the third sample (carrier prior == response) fires exactly one Effect",
    );
}
