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
//! - The Seed-side B3 drift path that produces zero effects skips
//!   Awaiting; the Standard-side B1 suppression skips Awaiting too.

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
use specter_core::{
    ArgPart, ArgTemplate, BurstPhase, ChildEntry, ClassSet, CommandTemplate, DedupKey, Diagnostic,
    DirChild, DirMeta, DirSnapshot, EffectOutcome, EffectScope, EntryKind, FsEvent, Input,
    LeafEntry, ProbeCorrelation, ProbeOp, ProbeRequest, ProbeResponse, ProbeResult, ProfileId,
    ProfileState, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachRequest,
    SubId, TimerKind, TreeSnapshot,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(root: ResourceId, children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild {
                inode,
                device: 0,
                subtree: None,
            }),
            _ => ChildEntry::Leaf(LeafEntry::new(kind, 0, UNIX_EPOCH, inode, 0)),
        };
        map.insert(CompactString::new(name), child);
    }
    TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        root,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    )))
}

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

fn anchor(e: &mut Engine, name: &str) -> ResourceId {
    let r = e.tree_mut().ensure(None, name, ResourceRole::User);
    e.tree_mut().get_mut(r).unwrap().kind = ResourceKind::Dir;
    r
}

fn pid_of(e: &Engine, sid: SubId) -> ProfileId {
    e.subs().get(sid).expect("sub exists").profile
}

/// Subtree-root attach request returning a recursive Sub with `/bin/true`.
fn subtree_request(name: &str, r: ResourceId) -> SubAttachRequest {
    SubAttachRequest::for_resource(
        name.into(),
        r,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    )
}

/// Same as `subtree_request` but with `CONTENT` in the events mask so
/// descendant `Modified` events pass the L5 class filter.
fn subtree_request_with_content(name: &str, r: ResourceId) -> SubAttachRequest {
    SubAttachRequest::for_resource(
        name.into(),
        r,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
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
    snap: TreeSnapshot,
    now: Instant,
) -> (SubId, ProfileId) {
    let (sid, out) = e.attach_sub(req, now);
    let pid = pid_of(e, sid);
    let seed_corr = first_probe_correlation(&out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(snap),
        }),
        now,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    let _ = pid_resource;
    (sid, pid)
}

/// Drive a fresh attach through Seed-Ok → Idle. Returns the Profile id.
fn attach_and_complete_seed(
    e: &mut Engine,
    r: ResourceId,
    snap: TreeSnapshot,
    now: Instant,
) -> (SubId, ProfileId) {
    let (sid, out) = e.attach_sub(subtree_request("test", r), now);
    let pid = pid_of(e, sid);
    let seed_corr = first_probe_correlation(&out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(snap),
        }),
        now,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    (sid, pid)
}

/// Drain timers and inject probe responses until the Standard burst
/// reaches a stable verdict and emits Effects (transitioning to
/// Awaiting) — or exits the cycle (B1-suppressed, no Subs match) and
/// finishes to Idle. Returns the StepOutput from the verdict step.
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
    snap: TreeSnapshot,
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
                    profile: pid,
                    correlation: c,
                    result: ProbeResult::Ok(snap.clone()),
                }),
                t_drain,
            );
            // Done when an Effect fires OR the burst returned to Idle.
            let is_idle = matches!(e.profiles().get(pid).unwrap().state, ProfileState::Idle);
            if !out.effects.is_empty() || is_idle {
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
    // — B1 hash dedup catches it because last_emitted_dir_hash matches
    // the current view.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(r, vec![]);
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
        stable_out.effects.len(),
        1,
        "one Effect emitted at stable verdict"
    );
    let effect_key = stable_out.effects[0].key.clone();
    let phase = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => &b.phase,
        _ => panic!("expected Active(Awaiting)"),
    };
    assert!(matches!(phase, BurstPhase::Awaiting { outstanding: 1, .. }));

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
    let phase = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => &b.phase,
        _ => panic!("expected Active(Rebasing)"),
    };
    assert!(matches!(phase, BurstPhase::Rebasing));

    // ProbeResponse Ok (idempotent — same snap) → Idle, baseline rebased.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: rebase_corr,
            result: ProbeResult::Ok(snap.clone()),
        }),
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    assert!(e.profiles().get(pid).unwrap().baseline.is_some());

    // Fresh FsEvent identical to the first → Standard burst starts but
    // B1 hash dedup suppresses the Effect (current == last_emitted_dir_hash).
    let later_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(40));
    assert!(
        later_out.effects.is_empty(),
        "B1 dedup suppresses idempotent re-fire — fire-cycle terminated cleanly",
    );
    // Burst returned to Idle directly (no Awaiting because count==0).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
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
    // event passes the L5 class filter (which sits BEFORE drive_burst's
    // absorb path). With the EMPTY default mask the event would drop
    // as `EventClassDropped` and never reach the fire-tail.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let child = e.tree_mut().ensure(Some(r), "child", ResourceRole::User);
    e.tree_mut().get_mut(child).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let snap_with_child = dir_snap(r, vec![("child", EntryKind::Dir, 7)]);
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
    let phase_before = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => format!("{:?}", b.phase),
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
    let phase_after = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => format!("{:?}", b.phase),
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
    let child = e.tree_mut().ensure(Some(r), "child", ResourceRole::User);
    e.tree_mut().get_mut(child).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let snap = dir_snap(r, vec![("child", EntryKind::Dir, 7)]);
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
    let effect_key = stable_out.effects[0].key.clone();

    // EffectComplete::Ok → Rebasing.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: effect_key,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let rebase_corr = e.pending_probe(pid).expect("rebase probe correlation");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Rebasing,
            ..
        }),
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
            profile: pid,
            correlation: rebase_corr,
            result: ProbeResult::Ok(snap),
        }),
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
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
    let snap = dir_snap(r, vec![]);
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
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Rebasing,
            ..
        }),
    ));
    let rebase_emitted = combined
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid));
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
    let snap = dir_snap(r, vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects[0].key.clone();

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
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Rebasing,
            ..
        }),
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
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Rebasing,
            ..
        }),
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
    let snap = dir_snap(r, vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects[0].key.clone();

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
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    assert!(e.profiles().get(pid).unwrap().baseline.is_none());

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
    let snap = dir_snap(r, vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects[0].key.clone();

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
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Rebasing,
            ..
        }),
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
        .filter(|op| matches!(op, ProbeOp::Cancel { profile } if *profile == pid))
        .count();
    assert_eq!(cancels, 1, "Rebasing probe cancelled on anchor loss");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
}

#[test]
fn fire_cycle_b3_fresh_seed_skips_awaiting() {
    // Fresh attach → Seed-Ok → no prior `last_emitted_dir_hash` ⇒
    // b3_seed_drift_observed returns false ⇒ no emit ⇒ finish_to_idle
    // directly. Verify no Awaiting state is entered.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let (sid, out) = e.attach_sub(subtree_request("test", r), now);
    let pid = pid_of(&e, sid);
    let seed_corr = first_probe_correlation(&out).expect("Seed probe");

    let resp_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_snap(r, vec![])),
        }),
        now + Duration::from_millis(1),
    );
    assert!(
        resp_out.effects.is_empty(),
        "fresh Seed never fires Effects"
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    assert!(
        e.profiles()
            .get(pid)
            .unwrap()
            .last_emitted_dir_hash
            .is_empty(),
        "fresh Seed leaves last_emitted_dir_hash empty",
    );
}

#[test]
fn fire_cycle_standard_b1_suppressed_skips_awaiting() {
    // Drive a complete fire cycle once (populates last_emitted_dir_hash).
    // Then trigger an identical Standard burst whose stable verdict has
    // the same hash — emit_effects returns count == 0 → finish_to_idle.
    // Profile must NOT enter Awaiting.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(r, vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);

    // First fire cycle.
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap.clone(),
        now + Duration::from_millis(10),
    );
    let effect_key = stable_out.effects[0].key.clone();
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
            profile: pid,
            correlation: rebase_corr,
            result: ProbeResult::Ok(snap.clone()),
        }),
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    assert_eq!(
        e.profiles().get(pid).unwrap().last_emitted_dir_hash.len(),
        1,
        "first fire cycle records the SubtreeRoot DedupKey hash",
    );

    // Second burst: identical event/probe; B1 hash matches → no Effect.
    let later = now + Duration::from_millis(40);
    let second_out = drive_to_awaiting(&mut e, pid, r, snap, later);
    assert!(
        second_out.effects.is_empty(),
        "B1 hash dedup suppresses the second fire — count == 0",
    );
    // Profile finished directly to Idle; no Awaiting.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
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
    let req = SubAttachRequest::for_resource(
        "fmt".into(),
        r,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::PerStableFile,
        ClassSet::CONTENT,
        false,
    );
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = pid_of(&e, sid);
    let seed_corr = first_probe_correlation(&attach_out).expect("Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_snap(r, vec![])),
        }),
        now,
    );

    // Standard burst with two files in the response.
    let snap_with_files = dir_snap(
        r,
        vec![("a.txt", EntryKind::File, 1), ("b.txt", EntryKind::File, 2)],
    );
    let stable_out = drive_to_awaiting(
        &mut e,
        pid,
        r,
        snap_with_files.clone(),
        now + Duration::from_millis(10),
    );
    assert_eq!(
        stable_out.effects.len(),
        2,
        "two PerStableFile Effects emitted",
    );
    let phase = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => &b.phase,
        _ => panic!(),
    };
    assert!(matches!(phase, BurstPhase::Awaiting { outstanding: 2, .. }));
    let key_a = stable_out.effects[0].key.clone();
    let key_b = stable_out.effects[1].key.clone();

    // First completion: Ok → outstanding=1.
    e.step(
        Input::EffectComplete {
            sub: sid,
            key: key_a,
            result: EffectOutcome::Ok,
        },
        now + Duration::from_millis(20),
    );
    let phase = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => &b.phase,
        _ => panic!(),
    };
    assert!(matches!(phase, BurstPhase::Awaiting { outstanding: 1, .. }));

    // Second completion: Failed → outstanding=0 → Rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: key_b,
            result: EffectOutcome::Failed {
                exit_code: Some(1),
                signal: None,
            },
        },
        now + Duration::from_millis(30),
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Rebasing,
            ..
        }),
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
    // registry; ReapPendingResolved diagnostic.
    let mut e = Engine::new();
    let r = anchor(&mut e, "src");
    let now = Instant::now();
    let snap = dir_snap(r, vec![]);
    let (sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let stable_out = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let effect_key = stable_out.effects[0].key.clone();

    // Detach the only Sub. Profile is Active(Awaiting) → reap_pending=true.
    let _detach_out = e.detach_sub(sid, now + Duration::from_millis(15));
    assert!(
        e.profiles().get(pid).unwrap().reap_pending,
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
            Diagnostic::ReapPendingResolved { profile } if *profile == pid,
        )),
        "ReapPendingResolved diagnostic on reap-during-Awaiting",
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
    let child = e.tree_mut().ensure(Some(r), "child", ResourceRole::User);
    e.tree_mut().get_mut(child).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let snap_with_child = dir_snap(r, vec![("child", EntryKind::Dir, 7)]);
    let (_sid, pid) = attach_and_complete_seed_with(
        &mut e,
        subtree_request_with_content("test", r),
        r,
        snap_with_child.clone(),
        now,
    );

    // Confirm the child has watch_demand > 0 (Seed reconciler bumped it).
    assert!(
        e.tree().get(child).unwrap().watch_demand > 0,
        "Seed reconciler watched the descendant Dir",
    );
    // Confirm the child is NOT suppressed.
    assert_eq!(
        e.tree().get(child).unwrap().suppress_count,
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
    let snap = dir_snap(r, vec![]);
    let (_sid, pid) = attach_and_complete_seed(&mut e, r, snap.clone(), now);
    let _ = drive_to_awaiting(&mut e, pid, r, snap, now + Duration::from_millis(10));
    let pending_probe_before = e.pending_probe(pid);

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
    let phase = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => &b.phase,
        _ => panic!(),
    };
    assert!(matches!(phase, BurstPhase::Awaiting { .. }));
    assert_eq!(
        e.pending_probe(pid),
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
    let child = e.tree_mut().ensure(Some(r), "child", ResourceRole::User);
    e.tree_mut().get_mut(child).unwrap().kind = ResourceKind::Dir;
    let now = Instant::now();
    let snap_initial = dir_snap(r, vec![("child", EntryKind::Dir, 7)]);
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
    let effect_key = stable_out.effects[0].key.clone();

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
    let rebase_corr = e.pending_probe(pid).expect("rebase probe");

    // Rebase probe response carries a DIFFERENT snapshot (the user's
    // edit changed the directory). The post-rebase baseline reflects
    // the new state.
    let snap_after_edit = dir_snap(
        r,
        vec![
            ("child", EntryKind::Dir, 7),
            ("user_edit.txt", EntryKind::File, 99),
        ],
    );
    let final_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: rebase_corr,
            result: ProbeResult::Ok(snap_after_edit),
        }),
        now + Duration::from_millis(30),
    );
    // No second Effect — the user's edit folded into baseline silently.
    assert!(
        final_out.effects.is_empty(),
        "v1 loss-of-fidelity: user edit during fire-tail does not fire its own Effect",
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle
    ));
    // baseline reflects the post-edit tree.
    let baseline = e.profiles().get(pid).unwrap().baseline.as_ref().unwrap();
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
