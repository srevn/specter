//! Fire-cycle integration tests. The fire-cycle unifies the
//! observe → fire → rebase loop into a single Burst whose phase walks
//! Batching → Verifying → Awaiting → Rebasing → Idle. Tests in this file
//! pin the structural invariants:
//!
//! - The cycle terminates in one run for an idempotent command.
//! - Concurrent FsEvents during Awaiting / Rebasing are absorbed and
//!   folded into the post-fire baseline.
//! - The Awaiting counter decrements correctly across multi-Effect
//!   bursts and mixed Ok/Failed outcomes.
//! - The `gate_deadline` recovery path force-transitions to Rebasing
//!   when the actuator hangs; late completions diagnose.
//! - `reap_pending` mid-Awaiting reaps without re-probing.
//! - Anchor loss during Awaiting / Rebasing finishes the burst cleanly.
//! - The Seed-side drift path that produces zero effects skips
//!   Awaiting; the Standard-side hash-dedup suppression skips Awaiting
//!   too.

use compact_str::CompactString;
use specter_core::testkit::{dir_snap, empty_program, proven};
use specter_core::{
    ActiveBurst, BurstFinish, BurstIntent, ChildEntry, ClassSet, DedupKey, Diagnostic, DirMeta,
    DirSnapshot, EffectOutcome, EffectScope, EntryKind, FsEvent, FsIdentity, Input, LeafEntry,
    PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase, ProbeCorrelation, ProbeOp,
    ProbeOwner, ProbeResponse, ProfileId, ProfileState, ResourceId, ResourceKind, ResourceRole,
    ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId, Termination, TimerKind,
    TreeSnapshot,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    anchor_dir, complete_effect_to_rebasing, first_probe_correlation, pid_of, rebase_loop_to_idle,
    seed_to_idle,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

/// Single-file directory snapshot with an explicit `size`, so a
/// post-rebase read can carry a different leaf hash for the same
/// `inode` (an in-place formatter rewrite). The canonical `dir_snap`
/// bakes `size = 0` and offers no override, so this distinct sized
/// fixture stays file-local — two consumers is not a shared pattern.
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

/// Subtree-root attach request returning a recursive Sub with `/bin/true`.
fn subtree_request(name: &str, r: ResourceId) -> SubAttachRequest {
    SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    )
}

/// Same as `subtree_request` but with `CONTENT` in the events mask so
/// descendant `Modified` events pass the class filter.
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

/// Drive a fresh attach (with the supplied request) through the
/// Batching-first Seed N=2 proof → Idle. Asserts the attach
/// StepOutput emits **no** probe (the Seed probe now fires only on
/// settle expiry, not at attach). Returns the `SubId`, `ProfileId`,
/// and the instant the Seed completed.
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
        first_probe_correlation(&out).is_none(),
        "Seed is Batching-first: attach emits no probe",
    );
    let done = seed_to_idle(e, pid, snap, now);
    (sid, pid, done)
}

/// Drive a fresh subtree-root attach through the
/// Batching-first Seed N=2 proof → Idle. Returns the `SubId`,
/// `ProfileId`, and the instant the Seed completed.
fn attach_and_complete_seed(
    e: &mut Engine,
    r: ResourceId,
    snap: &std::sync::Arc<DirSnapshot>,
    now: Instant,
) -> (SubId, ProfileId, Instant) {
    attach_and_complete_seed_with(e, subtree_request("test", r), snap, now)
}

/// Drain timers and inject probe responses until the Standard burst
/// reaches a stable verdict and emits Effects (transitioning to
/// Awaiting) — or exits the cycle (hash-dedup-suppressed, no Subs match)
/// and finishes to Idle. Returns the StepOutput from the verdict step.
///
/// A Standard burst's first probe diffs against the seed baseline; if
/// the response carries a different snapshot, the verdict is unstable
/// and the burst re-arms `Batching`. The second probe (with the same
/// response) should match the just-grafted `current` and stabilise.
/// This helper drives the loop until either an Effect fires or the
/// burst self-terminates.
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
            event: FsEvent::Modified,
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
                    owner: ProbeOwner::Profile(pid),
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
    // Subtree-root Sub on /src; baseline = empty. FsEvent → Standard burst
    // → stable verdict (response == seed snap) → Awaiting (one Effect).
    // EffectComplete::Ok → Rebasing. The post-fire N=2 loop closes on
    // two settle-spaced hash-equal WholeSubtree reads (idempotent
    // command) → Idle, baseline == current. A fresh FsEvent identical
    // to the first must NOT re-fire — hash dedup catches it because
    // fired_subs matches the current view.
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

    // EffectComplete::Ok → Rebasing.
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

    // Post-fire N=2 loop (idempotent — same snap each read) → Idle,
    // baseline rebased.
    let _ = rebase_loop_to_idle(&mut e, pid, &snap, seed_done + Duration::from_millis(30));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );
    assert!(e.profiles().get(pid).unwrap().baseline().is_some());

    // Fresh FsEvent identical to the first → Standard burst starts but
    // hash dedup suppresses the Effect (current == fired_subs).
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
    // Drive to Awaiting; inject an FsEvent at a covered descendant;
    // assert EventAbsorbedByFireTail; assert phase still Awaiting and
    // outstanding unchanged.
    //
    // The Sub uses a `CONTENT` events mask so the descendant Modified
    // event passes the class filter (which sits BEFORE drive_burst's
    // absorb path). With the EMPTY default mask the event would drop
    // as `EventClassDropped` and never reach the fire-tail.
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

    // Inject FsEvent at the covered descendant. The descendant has a
    // watch_demand bumped via the Seed's reconcile, so the event isn't
    // dropped as "unwatched".
    let descendant_event_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
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
    // Drive a Standard burst through the post-fire N=2 loop. An FsEvent
    // absorbed during the *final* rebase round-trip (after the last
    // `transition_to_rebasing` re-arm, before the Stable response) is
    // the genuine final-window residual — `transition_to_rebasing`
    // clears `dirty` at every loop entry, so only the final
    // round-trip's absorbs survive to the `Stable` verdict. A non-empty
    // residual there restarts a fresh debounced Standard burst seeded
    // from the residual via a typed PostFire→PreFire move that
    // preserves the watched anchor — no refcount edge changes (no
    // Unwatch/re-Watch flicker).
    //
    // CONTENT events mask: descendants must pass the class filter to
    // reach drive_burst's absorb arm.
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

    // EffectComplete::Ok → transition_to_rebasing(First): the (empty)
    // residual is cleared, rebase probe #1 armed.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        seed_done + Duration::from_millis(20),
    );
    let corr1 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("rebase probe #1 correlation");
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

    // Sample 1 (prior `None`) ⇒ Unstable ⇒ RebaseSettling. No absorb
    // yet — a residual accumulated here would be cleared by the next
    // `transition_to_rebasing` re-arm, so it must NOT drive the restart.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr1,
            outcome: proven(snap.clone()),
        }),
        seed_done + Duration::from_millis(22),
    );
    let spacing_timer = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::RebaseSettling { spacing_timer },
                ..
            }),
            _,
        ) => *spacing_timer,
        other => panic!("sample 1 must loop to Active(PostFire(RebaseSettling)); got {other:?}"),
    };

    // RebaseSettle expiry → transition_to_rebasing(LoopReArm): clears
    // `dirty` again and arms rebase probe #2 — the FINAL
    // round-trip.
    let rearm_out = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseSettle,
            id: spacing_timer,
        },
        seed_done + Duration::from_millis(25),
    );
    let corr2 =
        first_probe_correlation(&rearm_out).expect("RebaseSettle expiry re-arms rebase probe #2");

    // FsEvent during the FINAL Rebasing round-trip → absorbed. No
    // further loop entry follows (sample 2 is Stable), so this residual
    // survives to the verdict — the genuine final-window race.
    let absorb_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
        },
        seed_done + Duration::from_millis(27),
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { profile, resource, .. }
                if *profile == pid && *resource == child,
        )),
        "FsEvent during the final Rebasing round-trip absorbed",
    );

    // The anchor's kernel watch taken at start_standard_burst is held
    // through the whole loop (the surviving refcount).
    let watch_before = e.tree().get(r).unwrap().watch_demand();
    assert_eq!(watch_before, 1, "anchor watched for the in-flight burst");

    // Sample 2 hash-equal ⇒ Stable; non-empty final-window residual ⇒
    // restart, NOT Idle.
    let t_restart = seed_done + Duration::from_millis(30);
    let restart_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr2,
            outcome: proven(snap),
        }),
        t_restart,
    );

    // A fresh debounced Standard burst is armed, carrying the residual
    // as `dirty` provenance — the LCA basis and the source of the
    // mtime-skip-defeating obligation. ReturnToIdle is preserved across
    // the typed move.
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

    // No immediate re-probe — the restart re-enters the settle debounce,
    // so it cannot livelock.
    assert!(
        first_probe_correlation(&restart_out).is_none(),
        "restart re-enters Batching, emits no probe",
    );

    // The kernel watch did NOT flicker: the typed PostFire→PreFire move
    // never finished the burst, so the watch is still held (not
    // released-then-reacquired) — no refcount edge changes.
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        watch_before,
        "anchor watch held across the restart, no finish-then-start flicker",
    );
}

#[test]
fn fire_cycle_gate_deadline_force_transitions_to_rebasing() {
    // Drive to Awaiting; advance clock past gate_deadline; pop_expired
    // returns the AwaitGateDeadline timer; on_timer_expired runs
    // handle_gate_deadline → AwaitGateDeadlineElapsed diagnostic; phase
    // == Rebasing; rebase probe emitted.
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
        let (_, probe_ops, _, diagnostics) = s.into_parts();
        for d in diagnostics {
            combined.diagnostics.push(d);
        }
        for op in probe_ops.into_values() {
            combined.push_probe_op(op);
        }
    }
    assert!(
        combined.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::AwaitGateDeadlineElapsed { profile, outstanding: 1 }
                if *profile == pid,
        )),
        "gate-deadline elapsed diagnostic emitted",
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
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid)));
    assert!(
        rebase_emitted,
        "rebase probe emitted on gate-deadline force-transition"
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn fire_cycle_late_effect_complete_after_gate_deadline_diagnoses() {
    // Drive to Awaiting; force gate-deadline to Rebasing; inject
    // EffectComplete::Ok; assert EffectCompleteOutsideAwaiting.
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
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
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
    // Drive to Awaiting; inject anchor terminal event; finalize_anchor_lost
    // releases anchor, finishes burst → Idle. Inject late EffectComplete
    // → diagnoses outside Awaiting.
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
    // Profile is Idle, baseline cleared.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(e.profiles().get(pid).unwrap().baseline().is_none());

    // Late EffectComplete → diagnoses (Profile Idle now).
    let late_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
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
    // Drive to Rebasing; inject anchor terminal event; cancel_pending_probe
    // emits ProbeOp::Cancel; finish_burst_to_idle.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Rebasing.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        seed_done + Duration::from_millis(20),
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
        seed_done + Duration::from_millis(25),
    );
    // Probe Cancel emitted (Rebasing's probe in flight).
    let cancels = lost_out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid))
        .count();
    assert_eq!(cancels, 1, "Rebasing probe cancelled on anchor loss");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}

#[test]
fn fire_cycle_fresh_seed_skips_awaiting() {
    // Covers the **no-activity** fresh Seed: a fresh attach with NO
    // FsEvents injected. With an empty `dirty` provenance,
    // `seed_owes_first_fire` is false and `seed_drift_observed` is
    // false (never-fired) ⇒ `classify_consequence` yields the silent
    // `RecoverySeal` ⇒ finish_to_idle directly, no Awaiting tail.
    // Probe 1 (Unstable, prior None) re-batches into
    // PreFire(Batching); probe 2 (Stable, hash-equal) pins straight
    // to Idle. The witnessed-activity case
    // (a fresh Seed that *did* see events fires one Effect and *does*
    // enter Awaiting) is covered by the `fresh_seed_fires::*`
    // reproduction tests — this test deliberately exercises only the
    // silent-pin path.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let out = e.step(Input::AttachSub(subtree_request("test", r)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);
    assert!(
        first_probe_correlation(&out).is_none(),
        "Seed is Batching-first: attach emits no probe",
    );

    let snap = dir_snap(&[]);
    // Drive the N=2 proof one cycle at a time so we can assert the
    // fresh Seed never fires an Effect on *either* probe response and
    // never lands in a post-fire Awaiting tail.
    for (i, at) in [now + SETTLE, now + SETTLE * 2].into_iter().enumerate() {
        while let Some(entry) = e.pop_expired(at) {
            e.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                at,
            );
        }
        let corr = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Seed Verifying probe in flight after settle expiry");
        let resp_out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: corr,
                outcome: proven(Arc::clone(&snap)),
            }),
            at,
        );
        assert!(
            resp_out.effects().is_empty(),
            "fresh Seed never fires Effects (probe {})",
            i + 1,
        );
        let state = e.profiles().get(pid).unwrap().state();
        if i == 0 {
            // Probe 1 is Unstable by construction (prior
            // certified prior is None) → re-batch, NOT a fire.
            assert!(
                matches!(
                    state,
                    ProfileState::Active(
                        ActiveBurst::PreFire(PreFireBurst {
                            phase: PreFirePhase::Batching { .. },
                            intent: BurstIntent::Seed,
                            ..
                        }),
                        _
                    )
                ),
                "fresh Seed re-batches after the first (Unstable) sample, never enters Awaiting",
            );
        }
    }
    // Probe 2 (hash-equal) is Stable → `RecoverySeal` → finish_to_idle.
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
fn fire_cycle_standard_b1_suppressed_skips_awaiting() {
    // Drive a complete fire cycle once (sets the Sub's has_fired).
    // Then trigger an identical Standard burst whose stable verdict has
    // the same hash — emit_effects returns count == 0 → finish_to_idle.
    // Profile must NOT enter Awaiting.
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);

    // First fire cycle.
    let stable_out =
        drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();
    let _ = complete_effect_to_rebasing(
        &mut e,
        sid,
        effect_key,
        seed_done + Duration::from_millis(20),
    );
    let _ = rebase_loop_to_idle(&mut e, pid, &snap, seed_done + Duration::from_millis(30));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );

    // Second burst: identical event/probe; hash matches → no Effect.
    let later = seed_done + Duration::from_millis(40);
    let second_out = drive_to_awaiting(&mut e, pid, r, &snap, later);
    assert!(
        second_out.effects().is_empty(),
        "hash dedup suppresses the second fire — count == 0",
    );
    // Profile finished directly to Idle; no Awaiting.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}

#[test]
fn fire_cycle_mixed_ok_failed_decrements_uniformly() {
    // Per-stable-file Sub on /src; baseline = empty. FsEvent batch
    // creates 2 files (driven via the test by injecting a snapshot with
    // 2 leaves). Standard burst → 2 PerFile Effects emitted; Awaiting
    // outstanding=2. Inject one EffectComplete::Ok then one
    // EffectComplete::Failed; the counter decrements uniformly to 0;
    // transition to Rebasing.
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
        Input::EffectComplete {
            sub: sid,
            key: key_a,
            result: EffectOutcome::Ok,
        },
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

    // Second completion: Failed → outstanding=0 → Rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: key_b,
            result: EffectOutcome::Failed(Termination::Exit(1)),
        },
        seed_done + Duration::from_millis(30),
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
        "rebase probe emitted"
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn fire_cycle_reap_pending_during_awaiting_reaps_at_gate_close() {
    // Drive to Awaiting; detach the only Sub → reap_pending=true, phase
    // still Awaiting. Inject EffectComplete::Ok → last completion
    // (LastReached) + BurstFinish::Reap → finish_burst_to_idle →
    // reap_profile (deferred). Profile gone from registry;
    // ProfileReaped(DeferredFromBurst) diagnostic.
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

    // EffectComplete::Ok → LastReached + BurstFinish::Reap →
    // finish_burst_to_idle → reap_profile.
    let reap_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
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
fn fire_cycle_burst_deadline_during_awaiting_dropped_silently() {
    // The pre-fire BurstDeadline timer scheduled at start_standard_burst
    // remains in the heap when the burst transitions to Awaiting. Once
    // the burst is post-fire, is_timer_referenced filters BurstDeadline
    // out of Awaiting — pop_expired drops the stale entry without
    // dispatching handle_burst_deadline (which would otherwise try to
    // re-emit a probe).
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(&[]);
    let (_sid, pid, seed_done) = attach_and_complete_seed(&mut e, r, &snap, now);
    let _ = drive_to_awaiting(&mut e, pid, r, &snap, seed_done + Duration::from_millis(10));
    let pending_probe_before = e.pending_probe_for(ProbeOwner::Profile(pid));

    // Advance well past max_settle (the BurstDeadline) but stop short
    // of the gate_deadline (4 * max_settle).
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
        let (_, probe_ops, _, _) = s.into_parts();
        for op in probe_ops.into_values() {
            combined.push_probe_op(op);
        }
    }
    // No probe emitted — BurstDeadline filtered out, gate_deadline not
    // yet expired (4× max_settle vs 2×).
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
        e.pending_probe_for(ProbeOwner::Profile(pid)),
        pending_probe_before,
        "no probe minted"
    );
    // Use the imported types so dead_code rules don't trip on tests.
    let _ = (DedupKey::default(), TimerKind::Settle);
}

#[test]
fn fire_cycle_concurrent_user_edit_during_awaiting_folds_into_baseline() {
    // Concurrent user edit during Awaiting on a covered descendant:
    // absorbed into the fire-tail. The post-fire N=2 loop captures the
    // post-edit state (both settle-spaced WholeSubtree reads observe
    // it); the user's edit folds into the new baseline; it does not
    // fire its own Effect (v1 documented loss-of-fidelity).
    //
    // CONTENT events mask so the Modified event passes the class filter.
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

    // User edits the child (concurrent with the in-flight Effect).
    e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
        },
        seed_done + Duration::from_millis(15),
    );
    // Effect completes.
    let _ = complete_effect_to_rebasing(
        &mut e,
        sid,
        effect_key,
        seed_done + Duration::from_millis(20),
    );

    // Both rebase reads carry the post-edit snapshot (the user's edit
    // changed the directory; the post-command tree is now quiescent at
    // that state). The N=2 loop settles on it and the post-rebase
    // baseline reflects the new state.
    let snap_after_edit = dir_snap(&[
        ("child", EntryKind::Dir, 7),
        ("user_edit.txt", EntryKind::File, 99),
    ]);
    let r = rebase_loop_to_idle(
        &mut e,
        pid,
        &snap_after_edit,
        seed_done + Duration::from_millis(30),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );
    // No second Effect — the rebase path never emits; the user's edit
    // folded into baseline silently.
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
    // A non-idempotent command rewrites the watched tree mid-burst.
    // The post-fire rebase sets baseline := current := the post-Effect
    // tree. The next Standard burst probes that same post-Effect tree,
    // so structural B1 (`baseline.hash() == current.hash()` AND the Sub
    // already fired) suppresses the phantom — no second Effect for the
    // same intent.
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

    // Burst 1 — verify response = pre_emit. The N=2 Standard proof
    // stabilises against the seed baseline (probe 1 Unstable, probe 2
    // hash-equal → Stable); emit_effects fires one Effect and records
    // the Sub's fire history (the B1 gate for burst 2).
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        &pre_emit,
        seed_done + Duration::from_millis(10),
    );
    assert_eq!(stable_out.effects().len(), 1, "burst 1 fires one Effect");
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Rebasing → rebase probe in flight.
    let _ = complete_effect_to_rebasing(
        &mut e,
        sid,
        effect_key,
        seed_done + Duration::from_millis(20),
    );

    // Both rebase reads = post_effect (non-idempotent — the command
    // rewrote the tree, which is now quiescent at the post-Effect
    // state). The N=2 loop settles Stable: dispatch_rebase_ok grafts
    // and rebases baseline := post_effect.
    let _ = rebase_loop_to_idle(
        &mut e,
        pid,
        &post_effect,
        seed_done + Duration::from_millis(30),
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );

    // Post-rebase: baseline := current (= post_effect). The fire
    // history records the Sub's Subtree key — used to gate the B1
    // suppress in the phantom burst below.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert_eq!(
        p.baseline().unwrap().hash(),
        post_effect.dir_hash(),
        "rebase aligned baseline with the post-Effect tree",
    );

    // Burst 2 — phantom event. The verify probe responds with
    // post_effect (the tree the user actually has now). B1 dedup
    // derives suppress from `baseline.hash() == current.hash()` AND
    // `fired_subs.contains(dk)` — both true here, so the phantom is
    // suppressed.
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
    // PerFile mirror of the Subtree test. A formatter-style
    // non-idempotent command rewrites foo.rs's content **in place**
    // (same inode, different leaf-hash inputs — `size` here, the same
    // shape as a real formatter's `mtime`/`size` change). The slot
    // survives `graft` (same inode/device → identity match), so the
    // PerFile dedup entry survives the purge. Post-rebase the baseline
    // carries the post-Effect leaf hash, so a phantom event at the
    // same file diffs empty against the rebased baseline — no re-fire.
    //
    // `sized_file_snap` builds a `foo.rs` LeafEntry with an explicit
    // `size` so post-rebase carries a different leaf hash for the same
    // `inode` (the canonical `dir_snap` bakes `size = 0`).
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();

    // PerStableFile Sub on the anchor; CONTENT events so per-leaf FDs
    // are issued. Seed baseline empty.
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

    // Burst 1 — verify response = pre_emit (foo.rs at inode 42,
    // size 0). The Seed → Standard diff (created foo.rs) drives one
    // PerFile Effect.
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

    // EffectComplete::Ok → Rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        seed_done + Duration::from_millis(20),
    );
    let rebase_corr = first_probe_correlation(&rebase_out).expect("rebase probe");

    // Rebase response: foo.rs at the **same inode 42** (in-place
    // formatter rewrite, slot identity preserved) but `size = 1` —
    // changes the leaf hash without triggering a delete/create cycle.
    let post_effect = sized_file_snap("foo.rs", EntryKind::File, 42, 1);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: proven(post_effect.clone()),
        }),
        seed_done + Duration::from_millis(30),
    );

    // Post-rebase: baseline := current carries the post-Effect leaf
    // hash; the fire history records a PerFile key keyed at the file
    // resource (slot survived graft via inode identity). Both signals
    // gate the phantom-suppress path below — validated behaviourally
    // by that burst producing no fire.

    // Burst 2 — phantom event. The verify probe responds with
    // post_effect (foo.rs at inode 42, size 1 — the "formatted"
    // content). The diff is empty (baseline == response), so
    // `emit_effects_per_stable_file` walks zero entries — no fire.
    // The Subtree-arm B1 suppress (`baseline.hash() == current.hash()`
    // AND `fired_subs.contains(dk)`) holds for the SubtreeRoot key
    // implicitly recorded alongside the PerFile one — so the burst
    // returns to Idle without entering Awaiting.
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

/// PerStableFile contract regression: a `PerStableFile` Sub's Effect
/// fires iff its file is in the diff, re-fires on a *subsequent real
/// change* to that file, and is deduped by **nothing but diff
/// membership** — in particular it is NOT gated by the per-Sub
/// `Sub.has_fired` flag (which the relocation introduced for the
/// Subtree B1 path only).
///
/// The load-bearing step is Burst 2: `Sub.has_fired` is already `true`
/// from Burst 1, yet a real `foo.rs` content change must still fire a
/// fresh PerFile Effect. If a future maintainer re-introduces a
/// spurious PerFile suppression gate keyed on fire history, Burst 2
/// emits zero effects and this test fails.
#[test]
fn fire_cycle_perfile_refires_on_real_change_not_gated_by_fire_history() {
    let mut e = Engine::new();
    let r = anchor_dir(&mut e, "src");
    let now = Instant::now();

    // PerStableFile Sub on the anchor; CONTENT events so per-leaf FDs
    // are issued. Seed baseline empty.
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

    // Burst 1 — foo.rs created (inode 42, size 0). Seed → Standard
    // diff (created foo.rs) drives exactly one PerFile Effect.
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

    // EffectComplete::Ok → Rebasing. Idempotent command: rebase
    // response leaves foo.rs unchanged (inode 42, size 0). baseline
    // := current carries foo.rs.
    let _ = complete_effect_to_rebasing(&mut e, sid, key1, seed_done + Duration::from_millis(20));
    // Idempotent command: both N=2 reads leave foo.rs unchanged
    // (inode 42, size 0) → Stable, baseline := current carries foo.rs.
    let _ = rebase_loop_to_idle(&mut e, pid, &v1, seed_done + Duration::from_millis(30));
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "idempotent rebase loop closes Stable → Idle (empty residual ⇒ no restart)",
    );
    // A PerStableFile Sub's fire-history flag is NEVER set: `mark_fired`
    // is called only by the SubtreeRoot emit arm. PerFile has no B1
    // fire-history suppression — it is diff-membership-gated only, so
    // there is no flag to set and none to dedup against.
    assert!(
        !e.subs().get(sid).unwrap().has_fired,
        "PerStableFile Sub is never fire-history-marked (mark_fired is SubtreeRoot-only)",
    );

    // Burst 2 — a *real* change: foo.rs rewritten in place (same
    // inode 42, size 0 → 1). The diff carries foo.rs as modified, so
    // the PerFile Effect MUST re-fire. PerFile emission is gated by
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
