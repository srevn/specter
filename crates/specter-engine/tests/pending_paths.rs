//! Pending-path descent end-to-end. Drives `Engine::attach_sub` with a path-based request, walks
//! descent through scaffolds, and confirms anchor materialization triggers a Seed burst. The
//! witnessed-appearance latch splits the materialization: a descent whose probes observed the
//! awaited segment absent and then present (an absence→presence pair) opens a *triggered* Seed that
//! owes a fire; a descent that found every segment on first observation stays cold and pins —
//! prefix events drive re-probes but never write the witness, so sibling churn at a shared prefix
//! can't fire a Sub whose anchor sat unchanged on disk.

use specter_core::testkit::{dir_snap, empty_program};
use specter_core::{
    ActiveBurst, ClassSet, Diagnostic, EffectScope, EntryKind, FsEvent, Input, ProbeFailure,
    ProbeOp, ProbeOutcome, ProbeResponse, ProfileState, ResourceKind, ResourceRole, ScanConfig,
    SubAttachAnchor, SubAttachRequest,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    MAX_SETTLE, NO_EVENTS, SETTLE, complete_effect_to_rebasing, descent_advance, drain_due,
    fire_standard_once, first_probe_correlation, pre_place_dir, respond_anchor_file, seed_to_idle,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

#[test]
fn attach_sub_path_pending_then_anchor_appears() {
    // Tree has /var only. attach_sub at path /var/log/myapp pending state: prefix=/var,
    // remaining=[log, myapp]. Inject probe responses showing log appears, then myapp appears.
    // Anchor materializes; Seed burst starts.
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/log/myapp")),
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
    let pid = e.subs().get(sid).unwrap().profile();

    // Initial pending state: intermediate scaffold in place; anchor already has role=User ("role =
    // User for the anchor, role = DescentScaffold for everything else"). Pending status lives on
    // `Profile.state == ProfileState::Pending(_)`, not on the anchor's role.
    let log = e.tree().lookup(Some(var), "log").expect("log scaffold");
    let myapp = e.tree().lookup(Some(log), "myapp").expect("anchor slot");
    assert!(matches!(
        e.tree().get(log).unwrap().role,
        ResourceRole::DescentScaffold,
    ));
    assert!(
        matches!(e.tree().get(myapp).unwrap().role, ResourceRole::User),
        "anchor's role is User even when pending",
    );
    let var_corr = first_probe_correlation(&attach_out).expect("descent probe at /var emitted");

    // Inject probe response showing `log` appears.
    let log_advance = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: var_corr,
            outcome: ProbeOutcome::SegmentObserved {
                kind: Some(EntryKind::Dir),
            },
        }),
        now,
    );
    let log_corr =
        first_probe_correlation(&log_advance).expect("descent probe at /var/log emitted");

    // Inject probe response showing `myapp` appears under /var/log.
    let myapp_materialize = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: log_corr,
            outcome: ProbeOutcome::SegmentObserved {
                kind: Some(EntryKind::Dir),
            },
        }),
        now,
    );

    // Anchor materialized: kind set from the snapshot's entry; role stays User (set at attach time).
    assert!(matches!(
        e.tree().get(myapp).unwrap().role,
        ResourceRole::User,
    ));
    assert_eq!(e.tree().get(myapp).unwrap().kind(), Some(ResourceKind::Dir));

    // Profile is now in Active(PreFire(Seed)) — the Seed burst was started at materialization.
    // Under the cold-arm Verifying-first contract, the materializing step opens the burst in
    // `Verifying(ProbeSlot::armed(corr))` and emits the cold-walk Probe directly.
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
            assert_eq!(pre.intent, specter_core::BurstIntent::Seed);
            assert!(
                matches!(pre.phase, specter_core::PreFirePhase::Verifying { .. }),
                "cold-arm Seed opens Verifying-first; got {:?}",
                pre.phase,
            );
        }
        s => panic!("expected Active(PreFire(Seed)), got {s:?}"),
    }
    assert!(
        first_probe_correlation(&myapp_materialize).is_some(),
        "cold-arm Seed: probe emitted at burst construction (materialization)",
    );

    // Drive the Seed proof. `t0` is the instant the Seed burst started — the step that materialized
    // the anchor (`now`), not the original attach.
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);

    // Profile should now be Idle with baseline established.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state(), ProfileState::Idle));
    assert!(p.baseline().is_some());
}

/// The watched file APPEARS after attach: the first descent probe observes the awaited segment absent
/// (the standing absence half), the parent's STRUCTURE event drives a re-probe, and the response that
/// finds the segment completes the absence→presence appearance witness — so materialization opens a
/// triggered Seed (Batching-first) that FIRES once the settle window passes (`FreshSeedFire`). The
/// witnessed counterpart of [`attach_sub_path_pending_then_anchor_appears`]'s cold pin.
#[test]
fn pending_path_witnessed_appearance_fires() {
    let mut e = Engine::new();
    let parent = pre_place_dir(&mut e, &["watch"]);
    let now = Instant::now();

    let req = SubAttachRequest::for_anchor(
        "pending".into(),
        SubAttachAnchor::Path(PathBuf::from("/watch/app.log")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_)
    ));

    // First descent probe: file not there yet.
    let out = descent_advance(&mut e, pid, None, now);
    assert!(out.effects().is_empty());
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_)
    ));

    // The file appears: the parent STRUCTURE event re-probes; the response finds the segment the
    // prior probe observed absent — the appearance witness — and materializes the anchor into a
    // TRIGGERED Seed: Batching-first, no probe until the settle window expires.
    let t1 = now + Duration::from_millis(50);
    let _ = e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    let out = descent_advance(&mut e, pid, Some(EntryKind::File), t1);
    assert!(
        out.effects().is_empty(),
        "materialization itself never fires"
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "triggered Seed opens Batching-first — no cold walk in flight",
    );

    // Settle expiry -> Verifying -> the Authoritative sample classifies FreshSeedFire.
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let out = respond_anchor_file(&mut e, pid, 1, t2);
    assert_eq!(
        out.effects().len(),
        1,
        "the witnessed appearance fires once it settles",
    );
    assert!(e.subs().get(sid).unwrap().has_fired());

    // Drain the fire cycle: effect Ok -> rebase -> Idle.
    let key = out.effects()[0].key();
    let _ = complete_effect_to_rebasing(&mut e, sid, key, t2);
    let _ = respond_anchor_file(&mut e, pid, 1, t2);
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));

    // And the watch is healthy thereafter: an in-place change fires as Standard.
    let anchor = e.profiles().get(pid).unwrap().resource();
    let _ = fire_standard_once(&mut e, sid, anchor, 2, t2 + SETTLE);
    let _ = e.cancel_all_in_flight_probes();
}

/// The W1 wedge, closed: a prefix event that *races* the in-flight descent probe — arriving before
/// its response is processed — must not be dropped. The walk behind that probe may predate the
/// event, so its response finds the awaited segment still absent and parks; historically that was a
/// permanent wedge, because the segment's own creation was the dropped event and nothing would
/// re-probe. The re-probe-owed latch repays it with a postdating probe that observes the segment,
/// completing the absence→presence appearance witness exactly as the non-racing path does — so the
/// terminus opens a TRIGGERED Seed that fires.
///
/// The racing counterpart of [`pending_path_witnessed_appearance_fires`], where the event arrives
/// *after* the absent response with the slot already disarmed (no latch involved). Both reach the
/// same fire, which is the point: the latch makes the wedge interleaving observationally identical
/// to the benign one.
#[test]
fn pending_path_event_racing_inflight_probe_repays_and_fires() {
    let mut e = Engine::new();
    let parent = pre_place_dir(&mut e, &["watch"]);
    let now = Instant::now();

    let req = SubAttachRequest::for_anchor(
        "pending".into(),
        SubAttachAnchor::Path(PathBuf::from("/watch/app.log")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    let corr1 = e
        .pending_probe_for(pid)
        .expect("descent probe in flight at /watch");

    // The file is being created: a STRUCTURE event lands at the prefix WHILE the descent probe is
    // still in flight. The walk behind corr1 may predate this event, so it cannot witness the
    // creation — the event is latched, not dropped, and no second probe is emitted yet.
    let t1 = now + Duration::from_millis(20);
    let out = e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    assert_eq!(
        e.pending_probe_for(pid),
        Some(corr1),
        "racing event leaves the in-flight probe untouched (latched, not superseded)",
    );
    assert!(
        !out.probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "racing event emits no probe — the debt is latched",
    );

    // corr1's stale, pre-creation response: app.log still absent → the descent parks and records the
    // absence half of the witness. The latch repays with a fresh probe that postdates the event.
    let out = descent_advance(&mut e, pid, None, t1);
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "stale absent response parks the descent",
    );
    let corr2 = e
        .pending_probe_for(pid)
        .expect("re-probe-owed debt repaid: a postdating descent probe is in flight");
    assert_ne!(corr1, corr2);
    assert!(
        out.probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "the repay probe is emitted in the park's dispatch step",
    );

    // The repay probe observes the now-present file → the absence→presence pair latches the
    // appearance witness → TRIGGERED Seed (Batching-first, no cold walk in flight).
    let out = descent_advance(&mut e, pid, Some(EntryKind::File), t1);
    assert!(
        out.effects().is_empty(),
        "materialization itself never fires"
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "triggered Seed opens Batching-first — the latch preserved the witness through the wedge",
    );

    // Settle expiry → Verifying → the Authoritative sample classifies FreshSeedFire.
    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let out = respond_anchor_file(&mut e, pid, 1, t2);
    assert_eq!(
        out.effects().len(),
        1,
        "the witnessed appearance fires once it settles — same outcome as the non-racing path",
    );
    assert!(e.subs().get(sid).unwrap().has_fired());
    let _ = e.cancel_all_in_flight_probes();
}

/// Sibling churn at a descent prefix during an attach over an existing tree pins silently — zero
/// effects. The flake shape this pins against: a daemon whose attach path crosses a busy shared
/// directory (`/tmp`, `$TMPDIR`, `/var/log`) receives STRUCTURE events at the live descent prefix
/// from churn entirely outside the Sub's scope; a directory event names no segment on either backend,
/// so treating it as a witness would false-first-fire a never-fired Sub whose anchor sat unchanged on
/// disk the whole descent. Every probe here finds its segment on first observation — no absence half
/// is ever observed — so the terminus Seed stays cold no matter how much churn the prefixes saw.
///
/// Accepted narrow miss (uniform with the recovery arm's): an anchor created between attach and the
/// first probe's response is found on first observation and is indistinguishable from having
/// existed all along — it pins. The window is one probe round-trip; the restart-safe doctrine
/// already pins the attach-over-existing side of that ambiguity.
#[test]
fn sibling_churn_during_attach_descent_pins_silently() {
    let mut e = Engine::new();
    let parent = pre_place_dir(&mut e, &["watch"]);
    let now = Instant::now();

    let req = SubAttachRequest::for_anchor(
        "churn".into(),
        SubAttachAnchor::Path(PathBuf::from("/watch/a/b")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    assert!(
        e.pending_probe_for(pid).is_some(),
        "attach-time descent probe in flight",
    );

    // Sibling churn at /watch while the attach-time probe is in flight: the I5 gate drops the
    // re-probe and nothing latches — the in-flight response already reflects the change.
    let t1 = now + Duration::from_millis(10);
    let out = e.step(
        Input::FsEvent {
            resource: parent,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    assert!(
        out.probe_ops().is_empty(),
        "I5: the in-flight probe absorbs the re-probe",
    );

    // The response finds `a` present — first observation, no absence half — and advances; the
    // descent's fresh probe at /watch/a is immediately in flight.
    let out = descent_advance(&mut e, pid, Some(EntryKind::Dir), t1);
    assert!(out.effects().is_empty());
    let a = e
        .tree()
        .lookup(Some(parent), "a")
        .expect("descent advanced into /watch/a");
    assert!(e.pending_probe_for(pid).is_some());

    // More sibling churn, now at the advanced prefix — same drop, same silence.
    let out = e.step(
        Input::FsEvent {
            resource: a,
            event: FsEvent::StructureChanged,
        },
        t1,
    );
    assert!(out.probe_ops().is_empty(), "I5 again at the deeper prefix");

    // The terminus finds `b` present on first observation: a COLD Seed — Verifying-first, the probe
    // emitted at burst construction — that pins silently.
    let out = descent_advance(&mut e, pid, Some(EntryKind::Dir), t1);
    assert!(
        out.effects().is_empty(),
        "materialization itself never fires"
    );
    assert!(
        e.pending_probe_for(pid).is_some(),
        "cold Seed opens Verifying-first — attach-over-existing, not an appearance",
    );
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), t1);

    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle
    ));
    assert!(
        !e.subs().get(sid).unwrap().has_fired(),
        "sibling churn at descent prefixes must not fire an attach over an existing tree",
    );
}

/// A descent prefix that VANISHES mid-descent is a first-hand absence observation for the whole
/// remaining chain — a path cannot complete through a vanished directory — so the rewound descent's
/// eventual re-completion is a witnessed appearance: the terminus opens a triggered Seed that
/// fires. The Vanished-arm counterpart of [`pending_path_witnessed_appearance_fires`]'s
/// park-observed absence (the two writers of the absence half).
#[test]
fn prefix_vanished_rewind_then_recompletion_fires() {
    let mut e = Engine::new();
    let _parent = pre_place_dir(&mut e, &["watch"]);
    let now = Instant::now();

    let req = SubAttachRequest::for_anchor(
        "vanish".into(),
        SubAttachAnchor::Path(PathBuf::from("/watch/a/b")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::DEFAULT_SUBTREE_ROOT,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach succeeded");
    let pid = e.subs().get(sid).unwrap().profile();

    // First probe finds `a` — first observation, descent advances to /watch/a.
    let _ = descent_advance(&mut e, pid, Some(EntryKind::Dir), now);

    // The advanced prefix vanishes out from under the descent (`rm -rf /watch/a` mid-attach): the
    // rewind re-injects `a` as the head and records the absence observation.
    let t1 = now + Duration::from_millis(10);
    let corr = e
        .pending_probe_for(pid)
        .expect("probe at /watch/a in flight");
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        t1,
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PendingPathProbeVanished { .. })),
        "rewind narrated",
    );

    // The path re-completes: `a` recreated, then `b` lands. The found-after-absent pair latches the
    // appearance witness, so the terminus opens a TRIGGERED Seed — Batching-first.
    let _ = descent_advance(&mut e, pid, Some(EntryKind::Dir), t1);
    let out = descent_advance(&mut e, pid, Some(EntryKind::File), t1);
    assert!(
        out.effects().is_empty(),
        "materialization itself never fires"
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "triggered Seed opens Batching-first",
    );

    let t2 = t1 + SETTLE * 2;
    drain_due(&mut e, t2);
    let out = respond_anchor_file(&mut e, pid, 4, t2);
    assert_eq!(
        out.effects().len(),
        1,
        "the observed delete-then-recreate of the anchor's path fires once it settles",
    );
    assert!(e.subs().get(sid).unwrap().has_fired());
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn pending_path_failed_probe_retains_state() {
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/missing")),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    let corr = first_probe_correlation(&attach_out).expect("descent probe");

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        Instant::now(),
    );

    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::PendingPathProbeFailed {
            failure: ProbeFailure::Anchor { errno: 13 },
            ..
        },
    )));
    // Profile still pending (descent state lives on `ProfileState::Pending`, not on a separate
    // SecondaryMap).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));
}

#[test]
fn pending_path_event_at_prefix_emits_fresh_probe() {
    // Pending descent waiting for /var/missing/. Drain in-flight probe with a no-progress response,
    // then inject FsEvent at /var (the prefix) to trigger a fresh probe (no settle).
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/missing")),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    let corr = first_probe_correlation(&attach_out).expect("descent probe");

    // No-progress response — descent stays pending. An attach-time (unwitnessed) park is silent:
    // the `PendingPathAwaitingSegment` narration is gated to witnessed descents, where parking is a
    // recovery anomaly rather than the steady state.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid,
            correlation: corr,
            outcome: ProbeOutcome::SegmentObserved { kind: None },
        }),
        Instant::now(),
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PendingPathAwaitingSegment { .. })),
        "unwitnessed park narrates nothing",
    );

    // FsEvent at /var triggers a fresh descent probe.
    let out = e.step(
        Input::FsEvent {
            resource: var,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    let probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid))
        .count();
    assert_eq!(probes, 1, "FsEvent at prefix triggers fresh descent probe");
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn anchor_disappears_re_enters_pending_via_watch_root_parent() {
    // "Watch root deletion": Sub at /src; / is the watch_root_parent. The anchor's Removed terminal
    // re-enters pending descent at the parent inside the loss step itself — no later parent event
    // is needed.
    let mut e = Engine::new();
    // Both / and /src exist; /src is the anchor.
    let root_dir = e.tree_mut().ensure_root("root", ResourceRole::User);
    e.tree_mut().set_kind(root_dir, ResourceKind::Dir);
    let src = e
        .tree_mut()
        .ensure_child(root_dir, "src", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(src),
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
    let pid = e.subs().get(sid).unwrap().profile();
    // The immediate Seed is Batching-first: no probe at attach.
    assert!(
        first_probe_correlation(&attach_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    // Drive the Seed proof → Idle (`t0` is the attach instant).
    let seed_done = seed_to_idle(&mut e, pid, &dir_snap(&[]), now);
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    assert!(e.profiles().get(pid).unwrap().watch_root_parent() == Some(root_dir));

    // The Seed proof consumed two settle windows; keep instants monotonic for the recovery sequence
    // that follows.
    let after_seed = seed_done + SETTLE;

    // Anchor gone (Removed event at /src): the loss step re-enters pending descent with prefix=/,
    // remaining=[src], and emits the descent probe at the watch_root_parent.
    let out = e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Removed,
        },
        after_seed,
    );
    let p = e.profiles().get(pid).unwrap();
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "observed loss re-enters descent in the loss step itself",
    );
    assert!(p.current().is_none());
    let recovery_probe = out
        .probe_ops()
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid));
    assert!(
        recovery_probe,
        "loss step emits the descent probe at watch_root_parent",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// Detach Pending Profile with in-flight descent probe
#[test]
fn detach_pending_profile_with_inflight_descent_emits_cancel() {
    let mut e = Engine::new();
    let var = e
        .tree_mut()
        .ensure_path(&["/", "var"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/var/log/myapp")),
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
    let pid = e.subs().get(sid).unwrap().profile();

    // Profile is Pending with an in-flight descent probe.
    let initial_corr = first_probe_correlation(&attach_out).expect("descent probe at attach");
    let is_pending = matches!(
        e.profiles().get(pid).expect("Profile attached").state(),
        ProfileState::Pending(_)
    );
    assert!(is_pending, "Profile is in Pending state");
    assert_eq!(
        e.pending_probe_for(pid),
        Some(initial_corr),
        "descent state carries the outstanding probe correlation",
    );

    // Detach without delivering a probe response.
    let detach_out = e.step(Input::DetachSub(sid), Instant::now());

    // Profile is reaped.
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped on detach (Pending + last Sub detached)",
    );
    // ProbeOp::Cancel emitted for the in-flight descent probe.
    let cancel_present = detach_out
        .probe_ops()
        .iter()
        .any(|op| matches!(op, ProbeOp::Cancel { owner: profile} if *profile == pid));
    assert!(
        cancel_present,
        "ProbeOp::Cancel emitted for in-flight descent probe; got {:?}",
        detach_out.probe_ops(),
    );
}

// Anchor terminal event on a Pending Profile pins the no-consumer routing. An absolute attach
// against an empty Tree puts the FS-root bootstrap between prefix and anchor: prefix is the
// synthetic `/`, anchor is the scaffolded `/foo`. The two are distinct slots, and the anchor's
// `watch_demand` is zero (descent hasn't materialized it yet), so a `Removed` at the anchor lands
// in `EventOnUnwatchedResource` rather than coercing the Pending Profile through
// `finalize_anchor_lost` / `finish_burst_to_idle`.
#[test]
fn pending_profile_event_at_anchor_lands_in_no_consumer_branch() {
    let mut e = Engine::new();
    // Absolute path against an empty Tree: bootstrap creates `/`, anchor `/foo` is scaffolded under
    // `/`. Profile lands Pending with current_prefix = `/`, anchor = /foo (different slots).
    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/foo")),
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
    let pid = e.subs().get(sid).unwrap().profile();

    let p = e.profiles().get(pid).expect("Profile attached");
    let anchor = p.resource();
    let prefix = match p.state() {
        ProfileState::Pending(d) => d.current_prefix(),
        s => panic!("expected Pending, got {s:?}"),
    };
    assert_ne!(
        prefix, anchor,
        "FS-root bootstrap separates prefix from anchor"
    );
    assert_eq!(
        e.tree().get(prefix).unwrap().watch_demand(),
        1,
        "descent prefix `/` carries the +1 STRUCTURE contribution",
    );
    assert_eq!(
        e.tree().get(anchor).unwrap().watch_demand(),
        0,
        "anchor scaffold is not yet bumped (descent hasn't materialized it)",
    );

    // Dispatch FsEvent::Removed at the anchor (/foo). The anchor's `watch_demand == 0` short-circuits
    // at the `EventOnUnwatchedResource` head guard in `on_fs_event` before any classifier work runs.
    // Earlier this same Profile shape (a degenerate `prefix == anchor` fixture from a relative-path
    // attach against an empty Tree) routed through `finish_burst_to_idle` and underflowed a Resource
    // refcount. The FS-root bootstrap rules out that degenerate shape entirely.
    let out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        now,
    );

    // Profile remains Pending (no covering-profile fan-out touched it).
    let still_pending = matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    );
    assert!(
        still_pending,
        "Pending Profile not coerced through anchor-terminal-event path",
    );
    // The head guard short-circuits before any classifier work, so it emits no watch op (no
    // spurious Unwatch/Watch on this path).
    assert!(
        out.watch_ops.is_empty(),
        "EventOnUnwatchedResource head guard emits no watch op; got {:?}",
        out.watch_ops,
    );
    // The event landed in the no-consumer head guard.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            specter_core::Diagnostic::EventOnUnwatchedResource { resource, .. } if *resource == anchor,
        )),
        "anchor terminal event on Pending Profile lands in EventOnUnwatchedResource diagnostic; got {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

// Behavioral parity: a single FsEvent at one Resource fans out to a Pending Profile (descent
// dispatch) AND an Idle Profile with absent anchor (recovery dispatch), without disturbing an
// unrelated Profile.
#[test]
#[allow(clippy::similar_names)]
fn classifier_routes_descent_and_recovery_in_single_pass() {
    // /root and /root/bar exist; /root/foo does not. /elsewhere exists.
    let mut e = Engine::new();
    let root_dir = e
        .tree_mut()
        .ensure_path(&["/", "root"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(root_dir, ResourceKind::Dir);
    let bar = e
        .tree_mut()
        .ensure_child(root_dir, "bar", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(bar, ResourceKind::Dir);
    let elsewhere = e
        .tree_mut()
        .ensure_path(&["/", "elsewhere"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(elsewhere, ResourceKind::Dir);

    // Profile A: Pending at /root, descending toward `foo` (does not exist). Drain its initial
    // descent probe with a no-progress response so its `pending_probe` slot is empty before the
    // test event — on a busy slot `on_descent_event` latches a re-probe-owed debt rather than
    // emitting, so the test event would produce no fresh probe to observe.
    let req_a = SubAttachRequest::for_anchor(
        "watch-a".into(),
        SubAttachAnchor::Path(PathBuf::from("/root/foo")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_a_out = e.step(Input::AttachSub(req_a), now);
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_a_out).expect("attach_sub succeeded");
    let pid_a = e.subs().get(sid_a).unwrap().profile();
    let a_corr = first_probe_correlation(&attach_a_out).expect("descent probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_a,
            correlation: a_corr,
            outcome: ProbeOutcome::SegmentObserved { kind: None },
        }),
        now,
    );
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Pending(_),
        ),
        "A still Pending after no-progress response",
    );

    // Profile B: anchor at /root/bar; drive Seed → Idle, then Removed at /root/bar → Idle with
    // current=None and watch_root_parent=/root.
    let req_b = SubAttachRequest::for_anchor(
        "watch-b".into(),
        SubAttachAnchor::Resource(bar),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_b_out = e.step(Input::AttachSub(req_b), now);
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_b_out).expect("attach_sub succeeded");
    let pid_b = e.subs().get(sid_b).unwrap().profile();
    assert!(
        first_probe_correlation(&attach_b_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    // Drive B's Seed proof → Idle (`t0` is B's attach instant). A is Pending with an empty descent
    // slot (no settle timer), so its timers do not interfere with B's settle drain.
    let b_seed_done = seed_to_idle(&mut e, pid_b, &dir_snap(&[]), now);
    assert_eq!(
        e.profiles().get(pid_b).unwrap().watch_root_parent(),
        Some(root_dir),
        "B watches its parent /root for anchor recovery",
    );
    // B's loss arrives via probe-`Failed` — the path that parks Idle-anchorless awaiting event-scan
    // recovery. (An observed loss — terminal / Vanished — re-enters descent inside the loss step
    // and would join the *descents* class instead; `Failed` is how the recoveries class is
    // populated.) Drive a Standard burst at the anchor, then fail its verify probe.
    let after_b_seed = b_seed_done + SETTLE;
    e.step(
        Input::FsEvent {
            resource: bar,
            event: FsEvent::ContentChanged,
        },
        after_b_seed,
    );
    let b_fail_at = after_b_seed + SETTLE * 2;
    drain_due(&mut e, b_fail_at);
    let b_corr = e
        .pending_probe_for(pid_b)
        .expect("B's verify probe in flight after settle expiry");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: pid_b,
            correlation: b_corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        b_fail_at,
    );
    let p_b = e.profiles().get(pid_b).unwrap();
    assert!(matches!(p_b.state(), ProfileState::Idle));
    assert!(p_b.current().is_none(), "B's anchor is gone");
    assert_eq!(p_b.watch_root_parent(), Some(root_dir));

    // Profile C: anchor at /elsewhere; Seed → Idle. Unrelated to /root.
    let req_c = SubAttachRequest::for_anchor(
        "watch-c".into(),
        SubAttachAnchor::Resource(elsewhere),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    // C attaches after B's loss; keep instants strictly monotonic.
    let c_attach = b_fail_at + SETTLE;
    let attach_c_out = e.step(Input::AttachSub(req_c), c_attach);
    let sid_c =
        specter_core::testkit::first_attached_sub(&attach_c_out).expect("attach_sub succeeded");
    let pid_c = e.subs().get(sid_c).unwrap().profile();
    assert!(
        first_probe_correlation(&attach_c_out).is_some(),
        "cold-arm Seed: probe emitted at burst construction",
    );
    // Drive C's Seed proof → Idle (`t0` is C's attach instant).
    let c_seed_done = seed_to_idle(&mut e, pid_c, &dir_snap(&[]), c_attach);
    assert!(matches!(
        e.profiles().get(pid_c).unwrap().state(),
        ProfileState::Idle,
    ));

    // The trigger: a single StructureChanged event at /root.
    // - A's `current_prefix == /root` ⇒ descent dispatch.
    // - B's `watch_root_parent == /root && current.is_none()` ⇒ recovery dispatch (Idle → Pending).
    // - C is anchored at /elsewhere ⇒ untouched. Strictly after both Seed proofs (B and C each
    //   consumed two settle windows since `now`).
    let trigger = c_seed_done + SETTLE;
    let out = e.step(
        Input::FsEvent {
            resource: root_dir,
            event: FsEvent::StructureChanged,
        },
        trigger,
    );

    // A: a fresh descent probe minted (slot was empty after drain).
    let a_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid_a))
        .count();
    assert_eq!(a_probes, 1, "A's descent advance emits one probe");
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Pending(_),
        ),
        "A remains Pending",
    );

    // B: re-entered Pending (recovery descent) and emitted a probe.
    let b_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid_b))
        .count();
    assert_eq!(b_probes, 1, "B's recovery emits one descent probe");
    assert!(
        matches!(
            e.profiles().get(pid_b).unwrap().state(),
            ProfileState::Pending(_),
        ),
        "B transitioned Idle → Pending",
    );

    // C: untouched. No probe; state still Idle.
    let c_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == pid_c))
        .count();
    assert_eq!(c_probes, 0, "C is unrelated to /root; no probe");
    assert!(matches!(
        e.profiles().get(pid_c).unwrap().state(),
        ProfileState::Idle,
    ));
    let _ = e.cancel_all_in_flight_probes();
}
