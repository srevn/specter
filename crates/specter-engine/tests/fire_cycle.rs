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
    ActionProgram, ActiveBurst, ArgPart, ArgTemplate, BurstFinish, ChildEntry, ClassSet, DedupKey,
    Diagnostic, DirChild, DirMeta, DirSnapshot, EffectOutcome, EffectScope, EntryKind, FsEvent,
    FsIdentity, Input, LeafEntry, PostFireBurst, PostFirePhase, ProbeCorrelation, ProbeOp,
    ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileState, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId, Termination,
    TimerKind, TreeSnapshot,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> std::sync::Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild::Uncovered(FsIdentity { inode, device: 0 })),
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

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

fn anchor(e: &mut Engine, name: &str) -> ResourceId {
    let r = e.tree_mut().ensure_root(name, ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

fn pid_of(e: &Engine, sid: SubId) -> ProfileId {
    e.subs().get(sid).expect("sub exists").profile
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

/// Drive a fresh attach (with the supplied request) through Seed-Ok →
/// Idle. Returns the `SubId` and `ProfileId`.
fn attach_and_complete_seed_with(
    e: &mut Engine,
    req: SubAttachRequest,
    pid_resource: ResourceId,
    snap: std::sync::Arc<DirSnapshot>,
    now: Instant,
) -> (SubId, ProfileId) {
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(e, sid);
    let seed_corr = first_probe_correlation(&out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        now,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    let _ = pid_resource;
    (sid, pid)
}

/// Drive a fresh attach through Seed-Ok → Idle. Returns the Profile id.
fn attach_and_complete_seed(
    e: &mut Engine,
    r: ResourceId,
    snap: std::sync::Arc<DirSnapshot>,
    now: Instant,
) -> (SubId, ProfileId) {
    let out = e.step(Input::AttachSub(subtree_request("test", r)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(e, sid);
    let seed_corr = first_probe_correlation(&out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        now,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    (sid, pid)
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
    snap: std::sync::Arc<DirSnapshot>,
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
                    outcome: ProbeOutcome::SubtreeOk(snap.clone()),
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
    // EffectComplete::Ok → Rebasing (probe at anchor). ProbeResponse Ok
    // with the SAME snapshot (idempotent command) → Idle, baseline ==
    // current. A fresh FsEvent identical to the first must NOT re-fire
    // — hash dedup catches it because fired_subs matches
    // the current view.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);

    // Standard burst → Awaiting.
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap.clone(),
        now + Duration::from_millis(10),
    );
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
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr = first_probe_correlation(&rebase_out).expect("rebase probe emitted");
    let phase = match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PostFire(post), _) => &post.phase,
        _ => panic!("expected Active(Rebasing)"),
    };
    assert!(matches!(phase, PostFirePhase::Rebasing));

    // ProbeResponse Ok (idempotent — same snap) → Idle, baseline rebased.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(snap.clone()),
        }),
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(e.profiles().get(pid).unwrap().baseline().is_some());

    // Fresh FsEvent identical to the first → Standard burst starts but
    // hash dedup suppresses the Effect (current == fired_subs).
    let later_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(40));
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
    let r = anchor(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap_with_child = dir_snap(vec![("child", EntryKind::Dir, 7)]);
    let (_sid, pid) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        r,
        snap_with_child.clone(),
        now,
    );

    // Drive to Awaiting using the same snap → stable.
    let _ = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap_with_child,
        now + Duration::from_millis(10),
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
        now + Duration::from_millis(50),
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
        descendant_event_out.probe_ops.is_empty(),
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
fn fire_cycle_absorbs_event_during_rebasing() {
    // Drive to Rebasing (via EffectComplete::Ok); inject an FsEvent;
    // absorb diagnostic; rebase response → Idle.
    //
    // CONTENT events mask: descendants must pass the class filter to
    // reach drive_burst's absorb arm.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap = dir_snap(vec![("child", EntryKind::Dir, 7)]);
    let (sid, pid) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        r,
        snap.clone(),
        now,
    );

    // Drive to Awaiting.
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap.clone(),
        now + Duration::from_millis(10),
    );
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Rebasing.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("rebase probe correlation");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing,
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));

    // FsEvent during Rebasing → absorbed.
    let absorb_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
        },
        now + Duration::from_millis(25),
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { profile, resource, .. }
                if *profile == pid && *resource == child,
        )),
        "FsEvent during Rebasing absorbed",
    );

    // Rebase response → Idle.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}

#[test]
fn fire_cycle_gate_deadline_force_transitions_to_rebasing() {
    // Drive to Awaiting; advance clock past gate_deadline; pop_expired
    // returns the AwaitGateDeadline timer; on_timer_expired runs
    // handle_gate_deadline → AwaitGateDeadlineElapsed diagnostic; phase
    // == Rebasing; rebase probe emitted.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (_sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let _stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));

    // Advance clock past gate_deadline (4 * MAX_SETTLE).
    let gate_t = now + Duration::from_millis(10) + MAX_SETTLE * 8;
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
        for d in s.diagnostics {
            combined.diagnostics.push(d);
        }
        for op in s.probe_ops {
            combined.probe_ops.push(op);
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
                phase: PostFirePhase::Rebasing,
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    let rebase_emitted = combined
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid)));
    assert!(
        rebase_emitted,
        "rebase probe emitted on gate-deadline force-transition"
    );
}

#[test]
fn fire_cycle_late_effect_complete_after_gate_deadline_diagnoses() {
    // Drive to Awaiting; force gate-deadline to Rebasing; inject
    // EffectComplete::Ok; assert EffectCompleteOutsideAwaiting.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // Force gate-deadline.
    let gate_t = now + Duration::from_millis(10) + MAX_SETTLE * 8;
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
                phase: PostFirePhase::Rebasing,
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
                phase: PostFirePhase::Rebasing,
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
}

#[test]
fn fire_cycle_anchor_loss_during_awaiting_drops_burst() {
    // Drive to Awaiting; inject anchor terminal event; finalize_anchor_lost
    // releases anchor, finishes burst → Idle. Inject late EffectComplete
    // → diagnoses outside Awaiting.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // Anchor terminal event → finalize_anchor_lost → finish_burst_to_idle.
    let lost_out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        now + Duration::from_millis(15),
    );
    // No probe Cancel emitted (Awaiting has no probe in flight).
    let cancels = lost_out
        .probe_ops
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
        now + Duration::from_millis(20),
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
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Rebasing.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing,
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
        now + Duration::from_millis(25),
    );
    // Probe Cancel emitted (Rebasing's probe in flight).
    let cancels = lost_out
        .probe_ops
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
    // Fresh attach → Seed-Ok → no prior `fired_subs` ⇒
    // seed_drift_observed returns false ⇒ no emit ⇒ finish_to_idle
    // directly. Verify no Awaiting state is entered.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let out = e.step(Input::AttachSub(subtree_request("test", r)), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);
    let seed_corr = first_probe_correlation(&out).expect("Seed probe");

    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now + Duration::from_millis(1),
    );
    assert!(
        resp_out.effects().is_empty(),
        "fresh Seed never fires Effects"
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(
        e.profiles().get(pid).unwrap().fired_subs.is_empty(),
        "fresh Seed leaves fired_subs empty",
    );
}

#[test]
fn fire_cycle_standard_b1_suppressed_skips_awaiting() {
    // Drive a complete fire cycle once (populates fired_subs).
    // Then trigger an identical Standard burst whose stable verdict has
    // the same hash — emit_effects returns count == 0 → finish_to_idle.
    // Profile must NOT enter Awaiting.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);

    // First fire cycle.
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap.clone(),
        now + Duration::from_millis(10),
    );
    let effect_key = stable_out.effects()[0].key();
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr = first_probe_correlation(&rebase_out).expect("rebase probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(snap.clone()),
        }),
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert_eq!(
        e.profiles().get(pid).unwrap().fired_subs.len(),
        1,
        "first fire cycle records the SubtreeRoot DedupKey hash",
    );

    // Second burst: identical event/probe; hash matches → no Effect.
    let later = now + Duration::from_millis(40);
    let second_out = drive_to_awaiting(&mut e, pid, r, snap, later);
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
    let r = anchor(&mut e, "src");
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
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);
    let seed_corr = first_probe_correlation(&attach_out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Standard burst with two files in the response.
    let snap_with_files = dir_snap(vec![
        ("a.txt", EntryKind::File, 1),
        ("b.txt", EntryKind::File, 2),
    ]);
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap_with_files.clone(),
        now + Duration::from_millis(10),
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
        now + Duration::from_millis(20),
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
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Rebasing,
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    assert!(
        first_probe_correlation(&rebase_out).is_some(),
        "rebase probe emitted"
    );
}

#[test]
fn fire_cycle_reap_pending_during_awaiting_reaps_at_gate_close() {
    // Drive to Awaiting; detach the only Sub → reap_pending=true, phase
    // still Awaiting. Inject EffectComplete::Ok → AwaitAction::Reap →
    // finish_burst_to_idle → reap_profile (deferred). Profile gone from
    // registry; ProfileReaped(DeferredFromBurst) diagnostic.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects()[0].key();

    // Detach the only Sub. Profile is Active(Awaiting) → reap_pending=true.
    let _detach_out = e.step(Input::DetachSub(sid), Instant::now());
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap)
        ),
        "reap_pending set on Active profile detach",
    );

    // EffectComplete::Ok → AwaitAction::Reap → finish_burst_to_idle →
    // reap_profile.
    let reap_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
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
fn fire_cycle_event_at_unsuppressed_descendant_during_awaiting_absorbs() {
    // Sanity: descendant FDs are NOT suppressed during the burst — the
    // anchor's add_suppress only silences anchor events. A descendant
    // event fires, reaches the engine, and absorbs into the fire-tail.
    //
    // CONTENT events mask so the Modified event passes the class filter.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap_with_child = dir_snap(vec![("child", EntryKind::Dir, 7)]);
    let (_sid, pid) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        r,
        snap_with_child.clone(),
        now,
    );

    // Confirm the child has watch_demand > 0 (Seed reconciler bumped it).
    assert!(
        e.tree().get(child).unwrap().watch_demand() > 0,
        "Seed reconciler watched the descendant Dir",
    );
    // Confirm the child is NOT suppressed.
    assert_eq!(
        e.tree().get(child).unwrap().suppress_count(),
        0,
        "descendants are not suppressed during burst",
    );

    let _ = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap_with_child,
        now + Duration::from_millis(10),
    );

    // Inject FsEvent on the (unsuppressed) descendant.
    let absorb_out = e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
        },
        now + Duration::from_millis(50),
    );
    assert!(
        absorb_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::EventAbsorbedByFireTail { profile, resource, .. }
                if *profile == pid && *resource == child,
        )),
        "descendant FsEvent absorbed despite unsuppressed FD",
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
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(vec![]);
    let (_sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let _ = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let pending_probe_before = e.pending_probe_for(ProbeOwner::Profile(pid));

    // Advance well past max_settle (the BurstDeadline) but stop short
    // of the gate_deadline (4 * max_settle).
    let post_burst_deadline = now + Duration::from_millis(10) + MAX_SETTLE * 2;
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
        for op in s.probe_ops {
            combined.probe_ops.push(op);
        }
    }
    // No probe emitted — BurstDeadline filtered out, gate_deadline not
    // yet expired (4× max_settle vs 2×).
    assert!(
        combined.probe_ops.is_empty(),
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
    // absorbed into the fire-tail. The Rebasing probe captures the
    // post-edit state; the user's edit folds into the new baseline; it
    // does not fire its own Effect (v1 documented loss-of-fidelity).
    //
    // CONTENT events mask so the Modified event passes the class filter.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let child = e
        .tree_mut()
        .ensure_child(r, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child, ResourceKind::Dir);
    let now = Instant::now();
    let snap_initial = dir_snap(vec![("child", EntryKind::Dir, 7)]);
    let (sid, pid) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        r,
        snap_initial.clone(),
        now,
    );

    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap_initial.clone(),
        now + Duration::from_millis(10),
    );
    let effect_key = stable_out.effects()[0].key();

    // User edits the child (concurrent with the in-flight Effect).
    e.step(
        Input::FsEvent {
            resource: child,
            event: FsEvent::Modified,
        },
        now + Duration::from_millis(15),
    );
    // Effect completes.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("rebase probe");

    // Rebase probe response carries a DIFFERENT snapshot (the user's
    // edit changed the directory). The post-rebase baseline reflects
    // the new state.
    let snap_after_edit = dir_snap(vec![
        ("child", EntryKind::Dir, 7),
        ("user_edit.txt", EntryKind::File, 99),
    ]);
    let final_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(snap_after_edit),
        }),
        now + Duration::from_millis(30),
    );
    // No second Effect — the user's edit folded into baseline silently.
    assert!(
        final_out.effects().is_empty(),
        "v1 loss-of-fidelity: user edit during fire-tail does not fire its own Effect",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    // baseline reflects the post-edit tree.
    let baseline = e.profiles().get(pid).unwrap().baseline().unwrap();
    match baseline {
        TreeSnapshot::Dir(arc) => {
            assert!(
                arc.entries.contains_key("user_edit.txt"),
                "baseline includes the user's edit",
            );
        }
        TreeSnapshot::File(_) => panic!("expected Dir baseline"),
    }
}

#[test]
fn fire_cycle_standard_b1_suppresses_post_rebase_phantom_for_non_idempotent_command() {
    // Concern B fix: a non-idempotent command rewrites the watched
    // tree mid-burst. Without the settle-time refresh,
    // `recorded[Subtree]` carries the **pre-Effect** stable hash; the
    // next Standard burst at the **post-Effect** state would
    // B1-mismatch and fire a phantom Effect for the same intent.
    //
    // The refresh inside `dispatch_rebase_ok` aligns
    // `recorded[Subtree]` with the post-rebase baseline-derived hash;
    // the next burst's verify probe at the post-Effect state matches
    // recorded → B1 suppress → no phantom.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();

    let pre_emit = dir_snap(vec![]);
    let post_effect = dir_snap(vec![("post.rs", EntryKind::File, 42)]);
    assert_ne!(
        pre_emit.dir_hash(),
        post_effect.dir_hash(),
        "test sanity: pre/post-Effect hashes differ",
    );

    let (sid, pid) = attach_and_complete_seed(&mut e, r, pre_emit.clone(), now);

    // Burst 1 — verify response = pre_emit (matches the seed
    // baseline; stable on first probe). emit_effects fires one Effect
    // and writes recorded[Subtree] = pre_emit.dir_hash() (the
    // emit-time defensive write).
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        pre_emit.clone(),
        now + Duration::from_millis(10),
    );
    assert_eq!(stable_out.effects().len(), 1, "burst 1 fires one Effect");
    let effect_key = stable_out.effects()[0].key();

    // EffectComplete::Ok → Rebasing → rebase probe in flight.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr =
        first_probe_correlation(&rebase_out).expect("rebase probe emitted on EffectComplete::Ok");

    // Rebase response = post_effect (non-idempotent — the command
    // rewrote the tree). dispatch_rebase_ok grafts, rebases baseline,
    // then refreshes recorded[Subtree] to post_effect.dir_hash().
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(post_effect.clone()),
        }),
        now + Duration::from_millis(30),
    );

    // Post-rebase: baseline := current (= post_effect). The fire
    // history records the Sub's Subtree key — used to gate the B1
    // suppress in the phantom burst below.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    let recorded_key = p.fired_subs.iter().next().expect("fire history recorded");
    assert!(
        matches!(recorded_key, DedupKey::Subtree { profile, .. } if *profile == pid),
        "fire history records the Subtree key for this Profile",
    );
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
    let phantom_out =
        drive_to_awaiting(&mut e, pid, r, post_effect, now + Duration::from_millis(40));
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
    // PerFile dedup entry survives the purge. Without the refresh,
    // `recorded[PerFile]` would still carry the pre-Effect leaf hash;
    // a phantom event at the same file would B1-mismatch and re-fire.
    // The refresh aligns `recorded[PerFile]` with the post-rebase
    // baseline's leaf hash; the next burst's leaf dedup matches and
    // suppresses.
    //
    // Local snapshot helper: lets us build a `foo.rs` LeafEntry with
    // an explicit `size` so post-rebase has a different leaf hash for
    // the same `inode`. `dir_snap` (file-level helper) bakes
    // `size = 0` and offers no override.
    fn dir_snap_one_file(
        name: &str,
        kind: EntryKind,
        inode: u64,
        size: u64,
    ) -> std::sync::Arc<DirSnapshot> {
        let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
        map.insert(
            CompactString::new(name),
            ChildEntry::Leaf(LeafEntry::new(
                kind,
                size,
                UNIX_EPOCH,
                FsIdentity { inode, device: 0 },
            )),
        );
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

    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
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
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = pid_of(&e, sid);
    let seed_corr = first_probe_correlation(&attach_out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Burst 1 — verify response = pre_emit (foo.rs at inode 42,
    // size 0). The Seed → Standard diff (created foo.rs) drives one
    // PerFile Effect.
    let pre_emit = dir_snap_one_file("foo.rs", EntryKind::File, 42, 0);
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        pre_emit.clone(),
        now + Duration::from_millis(10),
    );
    assert_eq!(
        stable_out.effects().len(),
        1,
        "one PerFile Effect for foo.rs"
    );
    let effect_key = stable_out.effects()[0].key();
    let foo_resource = match &effect_key {
        DedupKey::PerFile { resource, .. } => *resource,
        DedupKey::Subtree { .. } => panic!("expected PerFile key"),
    };

    // EffectComplete::Ok → Rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr = first_probe_correlation(&rebase_out).expect("rebase probe");

    // Rebase response: foo.rs at the **same inode 42** (in-place
    // formatter rewrite, slot identity preserved) but `size = 1` —
    // changes the leaf hash without triggering a delete/create cycle.
    let post_effect = dir_snap_one_file("foo.rs", EntryKind::File, 42, 1);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(post_effect.clone()),
        }),
        now + Duration::from_millis(30),
    );

    // Post-rebase: baseline := current carries the post-Effect leaf
    // hash; the fire history records a PerFile key keyed at the file
    // resource (slot survived graft via inode identity). Both signals
    // are required to gate the phantom-suppress path below.
    assert!(
        e.profiles()
            .get(pid)
            .unwrap()
            .fired_subs
            .iter()
            .any(|k| matches!(k, DedupKey::PerFile { resource, .. } if *resource == foo_resource)),
        "fire history records the PerFile key at foo.rs's resource id",
    );

    // Burst 2 — phantom event. The verify probe responds with
    // post_effect (foo.rs at inode 42, size 1 — the "formatted"
    // content). The diff is empty (baseline == response), so
    // `emit_effects_per_stable_file` walks zero entries — no fire.
    // The Subtree-arm B1 suppress (`baseline.hash() == current.hash()`
    // AND `fired_subs.contains(dk)`) holds for the SubtreeRoot key
    // implicitly recorded alongside the PerFile one — so the burst
    // returns to Idle without entering Awaiting.
    let phantom_out =
        drive_to_awaiting(&mut e, pid, r, post_effect, now + Duration::from_millis(40));
    assert!(
        phantom_out.effects().is_empty(),
        "B1 dedup suppresses PerFile phantom for non-idempotent format",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
}
