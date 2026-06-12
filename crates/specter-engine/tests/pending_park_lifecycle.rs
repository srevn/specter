//! Park-lifecycle regression suite — the `ProfileState::Parked` properties. A park is typed and
//! narrated at entry, exits only through a recovery descent (never a Seed), and each of its consumers
//! holds its door: burst routing filters `Parked` out of coverage, the event-carriers scan recovers
//! via the parent channel or the co-claimed anchor slot, an overflow attempts one Tree-derived
//! descent, a re-attach re-arms recovery, and a detach reaps. A signal-bearing descent probe that
//! fails transiently re-latches a bounded retry rather than dropping the consumed signal.

use specter_core::testkit::{MockSensor, dir_snap};
use specter_core::{
    AnchorClaim, ClassSet, Diagnostic, EntryKind, FsEvent, Input, OverflowScope, ProbeFailure,
    ProbeOutcome, ProbeResponse, ProfileState, ScanConfig, StateLabel, SubAttachAnchor,
    WatchFailure, WatchOp,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    MAX_SETTLE, SETTLE, attach, attach_returning, attach_seeded, descent_advance, drain_due,
    first_probe_correlation, pre_place_dir,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

fn cfg() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

/// An event at a co-watched parked anchor re-enters recovery descent and never fires unanchored. A
/// parked anchor's Tree slot can stay kernel-watched through a co-claimer (here R's
/// `watch_root_parent` claim on `/var/log`), so ordinary event traffic does reach it; the coverage
/// filter keeps `Parked` out of burst routing and the carriers scan's anchor-slot disjunct converts
/// the co-claimer's FD into a free recovery signal.
#[test]
fn third_door_event_at_co_watched_parked_anchor_must_not_fire_unanchored() {
    let mut e = Engine::new();
    let log = pre_place_dir(&mut e, &["var", "log"]);
    let app = pre_place_dir(&mut e, &["var", "log", "app"]);

    let t0 = Instant::now();
    // R anchored at /var/log/app — its watch_root_parent claim keeps /var/log kernel-watched after
    // P's park.
    let (_sid_r, _pid_r, t_r) = attach_seeded(
        &mut e,
        "R",
        SubAttachAnchor::Resource(app),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        &dir_snap(&[]),
        t0,
    );
    // P anchored at /var/log.
    let (_sid_p, pid_p, t_p) = attach_seeded(
        &mut e,
        "P",
        SubAttachAnchor::Resource(log),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        &dir_snap(&[]),
        t_r + SETTLE * 4,
    );

    // Park P via a path-fatal probe failure: Standard burst → Verifying → Failed(Anchor).
    let t2 = t_p + SETTLE * 4;
    let _ = e.step(MockSensor::fs_event(log, FsEvent::ContentChanged), t2);
    let _ = drain_due(&mut e, t2 + SETTLE * 2);
    let corr = e
        .pending_probe_for(pid_p)
        .expect("standard verify probe in flight");
    let _ = e.step(
        MockSensor::probe_response(ProbeResponse {
            owner: pid_p,
            correlation: corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        t2 + SETTLE * 3,
    );
    {
        let p = e.profiles().get(pid_p).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked), "parked");
        assert_eq!(p.anchor_claim(), AnchorClaim::None, "parked: claim gone");
        assert!(!p.current_is_some(), "parked: anchorless");
    }
    let log_demand = e.tree().get(log).unwrap().watch_demand();
    assert!(log_demand >= 1, "co-claimer keeps /var/log watched");

    // Ordinary STRUCTURE churn at the parked anchor (the slot is still kernel-watched via R): the
    // anchor-slot recovery channel re-enters descent at P's watch_root_parent.
    let t3 = t2 + SETTLE * 10;
    let _ = e.step(MockSensor::fs_event(log, FsEvent::StructureChanged), t3);
    assert!(
        matches!(
            e.profiles().get(pid_p).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "event at the parked anchor re-enters recovery descent",
    );

    // The descent finds the anchor present — the terminus re-installs the claim and opens a cold
    // (unwitnessed-recovery) Seed, which never fires.
    let out = descent_advance(&mut e, pid_p, Some(EntryKind::Dir), t3 + SETTLE);
    let fired = out.effects().len();

    let p = e.profiles().get(pid_p).unwrap();
    assert_eq!(
        p.anchor_claim(),
        AnchorClaim::Held,
        "terminus re-installed the anchor claim (state = {:?})",
        p.state().discriminant(),
    );
    assert_eq!(
        fired, 0,
        "a park exit is a descent, never an unanchored fire"
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// An overflow on a watch-rejection park re-enters recovery descent — never the Idle arm's cold
/// re-Seed, which would fabricate a baseline for a tree the Profile no longer observes and close the
/// recovery channel. Driving the descent to its terminus re-installs the claim and the kernel watch.
#[test]
fn overflow_reseed_of_park_reanchors_or_descends() {
    let mut e = Engine::new();
    let _watch = pre_place_dir(&mut e, &["watch"]);
    let src = pre_place_dir(&mut e, &["watch", "src"]);

    let t0 = Instant::now();
    let (_sid, pid, t_seeded) = attach_seeded(
        &mut e,
        "P",
        SubAttachAnchor::Resource(src),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        &dir_snap(&[]),
        t0,
    );

    // Park via kernel watch rejection on the anchor.
    let t1 = t_seeded + SETTLE;
    let _ = e.step(
        Input::WatchOpRejected {
            resource: src,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t1,
    );
    {
        let p = e.profiles().get(pid).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked), "parked");
        assert_eq!(p.anchor_claim(), AnchorClaim::None, "parked: claim gone");
        assert!(!p.current_is_some(), "parked: anchorless");
        assert!(p.watch_root_parent().is_some(), "recovery channel cached");
    }

    // Overflow: the Parked arm enters recovery descent at the cached channel.
    let t2 = t1 + SETTLE;
    let _ = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t2,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "overflow re-enters recovery descent",
    );

    // The descent finds the anchor present — claim and kernel watch re-installed at the terminus.
    let _ = descent_advance(&mut e, pid, Some(EntryKind::Dir), t2 + SETTLE);

    let p = e.profiles().get(pid).unwrap();
    let src_demand = e
        .tree()
        .get(src)
        .map_or(0, specter_core::Resource::watch_demand);
    assert_eq!(
        p.anchor_claim(),
        AnchorClaim::Held,
        "terminus re-installed the claim (state = {:?})",
        p.state().discriminant(),
    );
    assert!(src_demand >= 1, "terminus re-installed the kernel watch");
    let _ = e.cancel_all_in_flight_probes();
}

/// The recovery channel survives an overflow whose recovery descent finds the anchor still absent:
/// the descent parks on the standing absence observation, and the parent's next `StructureChanged`
/// drives a fresh probe — no recovery trigger ever seals the channel shut.
#[test]
fn parent_event_after_overflow_recovery_reprobes_descent() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);
    let src = pre_place_dir(&mut e, &["watch", "src"]);

    let t0 = Instant::now();
    let (_sid, pid, t_seeded) = attach_seeded(
        &mut e,
        "P",
        SubAttachAnchor::Resource(src),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        &dir_snap(&[]),
        t0,
    );
    let t1 = t_seeded + SETTLE;
    let _ = e.step(
        Input::WatchOpRejected {
            resource: src,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t1,
    );
    let t2 = t1 + SETTLE;
    let _ = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t2,
    );
    // The overflow's recovery descent observes the awaited segment absent — the descent parks
    // disarmed, awaiting the next prefix signal.
    let _ = descent_advance(&mut e, pid, None, t2 + SETTLE);
    assert!(
        e.pending_probe_for(pid).is_none(),
        "segment absent: descent awaits the next event",
    );

    // The parent slot still holds P's watch_root_parent claim → the event passes the head guard and
    // drives a fresh descent probe.
    let t3 = t2 + SETTLE * 2;
    let out = e.step(MockSensor::fs_event(watch, FsEvent::StructureChanged), t3);

    let p = e.profiles().get(pid).unwrap();
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "descent retained across the absent-segment park (state = {:?})",
        p.state().discriminant(),
    );
    assert!(
        first_probe_correlation(&out).is_some(),
        "parent StructureChanged re-probes the live descent",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// A fresh attach joining a parked Profile re-arms recovery (`ParkedRejoin`) instead of silently
/// inheriting the park — here the channel-less shape, whose recovery prefix re-derives from the Tree.
#[test]
fn attach_join_onto_parked_profile_rearms_recovery() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);

    // Sub A attaches at a not-yet-existing path → Pending descent at prefix /watch.
    let t0 = Instant::now();
    let (_sid_a, pid_a) = attach(
        &mut e,
        "A",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo/bar")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    assert!(matches!(
        e.profiles().get(pid_a).unwrap().state(),
        ProfileState::Pending(_)
    ));

    // Kernel rejects the descent-prefix watch → purge → channel-less park.
    let t1 = t0 + SETTLE;
    let _ = e.step(
        Input::WatchOpRejected {
            resource: watch,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t1,
    );
    {
        let p = e.profiles().get(pid_a).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked), "parked");
        assert!(p.watch_root_parent().is_none(), "channel-less park");
    }

    // Sub B: same path, same config → ParkedRejoin.
    let t2 = t1 + SETTLE;
    let (_sid_b, pid_b, join_out) = attach_returning(
        &mut e,
        "B",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo/bar")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t2,
    );
    assert_eq!(pid_a, pid_b, "B joins A's Profile");

    let p = e.profiles().get(pid_b).unwrap();
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "an operator attach onto a parked Profile re-enters descent (state = {:?})",
        p.state().discriminant(),
    );
    assert!(
        first_probe_correlation(&join_out).is_some(),
        "the rejoin's recovery descent emits its entry probe",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// A re-attach (different config) over a parked Profile's never-observed scaffold chain must
/// classify Pending (disk-honest), not Materialized-with-a-kernel-watch-on-a-nonexistent-path.
#[test]
fn reattach_over_never_observed_scaffold_classifies_pending() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);

    let t0 = Instant::now();
    let (_sid_a, pid_a) = attach(
        &mut e,
        "A",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo/bar")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    let t1 = t0 + SETTLE;
    let _ = e.step(
        Input::WatchOpRejected {
            resource: watch,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t1,
    );
    assert!(matches!(
        e.profiles().get(pid_a).unwrap().state(),
        ProfileState::Parked
    ));

    // Sub C: same path, different identity (max_settle folds into config_hash) → fresh Profile.
    let t2 = t1 + SETTLE;
    let (_sid_c, pid_c, out) = attach_returning(
        &mut e,
        "C",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo/bar")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE + Duration::from_secs(1),
        t2,
    );
    assert_ne!(pid_a, pid_c, "different config_hash forks a fresh Profile");

    let p = e.profiles().get(pid_c).unwrap();
    assert!(
        matches!(p.state(), ProfileState::Pending(_)),
        "attach over a never-observed chain classifies Pending; \
         got state={:?} watch_ops={:?} (a Watch for a disk-absent path would mean Materialized)",
        p.state().discriminant(),
        out.watch_ops,
    );
    // The disk-honest classification dropped C into a live recovery descent with an armed entry
    // probe; cancel it so the Engine's teardown doesn't trip `ProbeSlot`'s Drop tripwire.
    let _ = e.cancel_all_in_flight_probes();
}

/// A signal-bearing transient repay failure must not permanently drop the consumed signal. After
/// `latched signal → repay probe fails transiently`, the bounded re-latch arms a fresh postdating
/// probe so the descent observes again rather than wedging on a quiet prefix.
#[test]
fn signal_bearing_transient_repay_failure_eventually_reprobes() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);

    let t0 = Instant::now();
    let (_sid, pid, attach_out) = attach_returning(
        &mut e,
        "A",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    let corr0 = first_probe_correlation(&attach_out).expect("descent entry probe");

    // A prefix event races the in-flight probe → latches reprobe_owed.
    let t1 = t0 + SETTLE;
    let _ = e.step(MockSensor::fs_event(watch, FsEvent::StructureChanged), t1);

    // The in-flight probe fails transiently → repay hook emits the postdating probe.
    let t2 = t1 + SETTLE;
    let repay_out = e.step(
        MockSensor::probe_response(ProbeResponse {
            owner: pid,
            correlation: corr0,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        t2,
    );
    let corr1 = first_probe_correlation(&repay_out).expect("postdating repay probe emitted");

    // The repay probe ALSO fails transiently — the signal is now consumed and gone.
    let t3 = t2 + SETTLE;
    let _ = e.step(
        MockSensor::probe_response(ProbeResponse {
            owner: pid,
            correlation: corr1,
            outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
        }),
        t3,
    );

    // DESIRED: a probe in flight (bounded re-latch) — today: parked disarmed, no timer class exists
    // on a descent, the segment's creation event was already consumed.
    let in_flight = e.pending_probe_for(pid).is_some();
    let timer_due = e.pop_expired(t3 + MAX_SETTLE * 4).is_some();
    assert!(
        in_flight || timer_due,
        "DESIRED: a signal-bearing transient repay failure re-latches a bounded retry; \
         OBSERVED: probe_in_flight={in_flight} any_timer_due_within_4×max_settle={timer_due} \
         state={:?}",
        e.profiles().get(pid).unwrap().state().discriminant(),
    );
    // The bounded re-latch left a fresh postdating probe armed; cancel it so the Engine's teardown
    // doesn't trip `ProbeSlot`'s Drop tripwire.
    let _ = e.cancel_all_in_flight_probes();
}

/// A claim-Held Idle Profile is never kidnapped into recovery descent. A transient-forced Seed
/// finishes to Idle with `claim Held` + `current None` — anchored but never-grafted, **not** a park
/// — and the recovery arm selects on the `Parked` variant, so parent churn cannot route it into a
/// `Pending ∧ Held` breach of the reap trichotomy; its recovery is the anchor's own next event
/// (`drive_burst`'s triggered-Seed fork).
#[test]
fn transient_forced_seed_idle_is_not_kidnapped_into_descent() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);
    let src = pre_place_dir(&mut e, &["watch", "src"]);

    let t0 = Instant::now();
    let (_sid, pid) = attach(
        &mut e,
        "P",
        SubAttachAnchor::Resource(src),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );

    // Cold Seed under sustained transient pressure: answer Transient, drain timers, repeat until
    // the burst-deadline forces and the forced Transient finishes the burst to Idle.
    let mut t = t0;
    for i in 0..64 {
        if matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle) {
            break;
        }
        if let Some(corr) = e.pending_probe_for(pid) {
            t += SETTLE;
            let _ = e.step(
                MockSensor::probe_response(ProbeResponse {
                    owner: pid,
                    correlation: corr,
                    outcome: ProbeOutcome::Failed(ProbeFailure::Transient { errno: 24 }),
                }),
                t,
            );
        } else {
            t += SETTLE * 2;
            let _ = drain_due(&mut e, t);
        }
        assert!(i < 63, "did not converge to Idle under transient pressure");
    }
    {
        let p = e.profiles().get(pid).unwrap();
        assert!(matches!(p.state(), ProfileState::Idle), "forced finish");
        assert_eq!(
            p.anchor_claim(),
            AnchorClaim::Held,
            "transient posture retains the anchor claim"
        );
        assert!(!p.current_is_some(), "no graft ever landed");
        assert!(p.watch_root_parent().is_some());
    }

    // Parent churn at the watch_root_parent — the recoveries arm must not select this Profile.
    let t1 = t + SETTLE;
    let _ = e.step(MockSensor::fs_event(watch, FsEvent::StructureChanged), t1);

    let p = e.profiles().get(pid).unwrap();
    let kidnapped =
        matches!(p.state(), ProfileState::Pending(_)) && p.anchor_claim() == AnchorClaim::Held;
    assert!(
        !kidnapped,
        "a claim-Held Idle Profile must never route into recovery descent; \
         got state={:?} claim={:?} — Pending ∧ Held breaches the reap trichotomy \
         and materialize_anchor's precondition",
        p.state().discriminant(),
        p.anchor_claim(),
    );
}

/// The descent terminus' bundled `Pending → (Idle, Held, classified)` write is a carrier-count edge
/// under the pure-state `is_nonsteady` (`Pending` counted, `Idle` not);
/// `ProfileMap::materialize_anchor` reconciles it. The debug full-recount tripwire in the
/// event-carriers scan runs on the next delivered event, so a skipped reconcile fails loudly
/// exactly where the stale count would first mislead.
#[test]
fn descent_terminus_keeps_carrier_count_exact_on_next_event() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);
    let t0 = Instant::now();
    let (_sid, pid) = attach(
        &mut e,
        "A",
        SubAttachAnchor::Path(PathBuf::from("/watch/src")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );
    assert_eq!(
        e.profiles().nonsteady(),
        1,
        "a live descent is a counted carrier"
    );

    // Terminus: the awaited segment exists — materialize, then the cold Seed opens.
    let _ = descent_advance(&mut e, pid, Some(EntryKind::Dir), t0 + SETTLE);
    assert_eq!(
        e.profiles().nonsteady(),
        0,
        "materialization recorded the Pending → Idle carrier edge",
    );

    // The next delivered event runs the debug recount tripwire against the maintained count.
    let src = e.tree().lookup(Some(watch), "src").expect("materialized");
    let _ = e.step(
        MockSensor::fs_event(src, FsEvent::ContentChanged),
        t0 + SETTLE * 2,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// `ParkedRejoin` re-arms recovery for a channel-bearing park: a probe-`Failed { Anchor }` terminal
/// preserves `watch_root_parent`, and a same-identity re-attach enters descent at that cached
/// channel. (The channel-less shape is pinned by `attach_join_onto_parked_profile_rearms_recovery`.)
#[test]
fn parked_rejoin_rearms_recovery_for_channel_bearing_park() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);
    let src = pre_place_dir(&mut e, &["watch", "src"]);

    let t0 = Instant::now();
    let (_sid_a, pid_a, t_seeded) = attach_seeded(
        &mut e,
        "A",
        SubAttachAnchor::Resource(src),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        &dir_snap(&[]),
        t0,
    );

    // Park via a path-fatal probe failure: Standard burst → Verifying → Failed(Anchor).
    let t1 = t_seeded + SETTLE * 4;
    let _ = e.step(MockSensor::fs_event(src, FsEvent::ContentChanged), t1);
    let _ = drain_due(&mut e, t1 + SETTLE * 2);
    let corr = e.pending_probe_for(pid_a).expect("standard verify probe");
    let _ = e.step(
        MockSensor::probe_response(ProbeResponse {
            owner: pid_a,
            correlation: corr,
            outcome: ProbeOutcome::Failed(ProbeFailure::Anchor { errno: 13 }),
        }),
        t1 + SETTLE * 3,
    );
    {
        let p = e.profiles().get(pid_a).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked), "parked");
        assert_eq!(p.watch_root_parent(), Some(watch), "channel-bearing park");
    }

    // A same-(anchor, config) attach joins the parked Profile and re-arms recovery descent.
    let t2 = t1 + SETTLE * 8;
    let (_sid_b, pid_b, join_out) = attach_returning(
        &mut e,
        "B",
        SubAttachAnchor::Resource(src),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t2,
    );
    assert_eq!(pid_a, pid_b, "B joins the parked Profile");
    assert!(
        matches!(
            e.profiles().get(pid_b).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "rejoin re-enters recovery descent at the cached channel",
    );
    assert!(
        first_probe_correlation(&join_out).is_some(),
        "descent entry probe emitted on the attach step",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// An overflow gives a channel-less park exactly one recovery-descent attempt — re-deriving the
/// prefix from the Tree and re-trying the kernel watch the purge could not — and a re-rejected
/// watch re-parks: converges, no loop within a step.
#[test]
fn overflow_on_channel_less_park_descends_once_and_rejection_reparks() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);

    let t0 = Instant::now();
    let (_sid, pid) = attach(
        &mut e,
        "A",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );

    // Kernel rejects the descent-prefix watch → channel-less park.
    let t1 = t0 + SETTLE;
    let _ = e.step(
        Input::WatchOpRejected {
            resource: watch,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t1,
    );
    {
        let p = e.profiles().get(pid).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked), "parked");
        assert!(p.watch_root_parent().is_none(), "channel-less park");
    }

    // One descent attempt per overflow: prefix re-derived from the Tree, watch re-attempted.
    let t2 = t1 + SETTLE;
    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t2,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state(),
            ProfileState::Pending(_)
        ),
        "overflow re-enters recovery descent",
    );
    assert!(
        first_probe_correlation(&out).is_some(),
        "one descent probe per overflow",
    );
    assert!(
        out.watch_ops
            .iter()
            .any(|op| matches!(op, WatchOp::Watch { resource, .. } if *resource == watch)),
        "prefix watch re-attempted; got {:?}",
        out.watch_ops,
    );

    // The kernel rejects again — the purge re-parks, probe cancelled. Bounded by overflow rate.
    let t3 = t2 + SETTLE;
    let _ = e.step(
        Input::WatchOpRejected {
            resource: watch,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t3,
    );
    let p = e.profiles().get(pid).unwrap();
    assert!(
        matches!(p.state(), ProfileState::Parked),
        "re-rejection re-parks"
    );
    assert!(
        e.pending_probe_for(pid).is_none(),
        "descent probe cancelled by the purge",
    );
}

/// A park is narrated at entry, pins the carrier gate open while it lives, surfaces the honest
/// `Parked` operator row, and ends through detach: the last Sub's detach reaps the parked Profile
/// (`Parked ⇒ ReapNow`) and the nonsteady count returns to zero.
#[test]
fn park_narrates_counts_as_carrier_and_detach_reaps_to_zero() {
    let mut e = Engine::new();
    let watch = pre_place_dir(&mut e, &["watch"]);

    let t0 = Instant::now();
    let (sid, pid) = attach(
        &mut e,
        "A",
        SubAttachAnchor::Path(PathBuf::from("/watch/foo")),
        cfg(),
        ClassSet::CONTENT,
        MAX_SETTLE,
        t0,
    );

    let out = e.step(
        Input::WatchOpRejected {
            resource: watch,
            failure: WatchFailure::Pressure { errno: 24 },
        },
        t0 + SETTLE,
    );
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProfileParked {
                profile,
                recovery: None,
            } if *profile == pid
        )),
        "operational park narrates once, channel-less; got {:?}",
        out.diagnostics,
    );
    {
        let p = e.profiles().get(pid).unwrap();
        assert!(matches!(p.state(), ProfileState::Parked));
        assert_eq!(p.state().label(), StateLabel::Parked, "honest operator row");
    }
    assert_eq!(
        e.profiles().nonsteady(),
        1,
        "a park pins the carrier gate open"
    );

    // Detaching the last Sub reaps the parked Profile synchronously and ends the nonsteady pin.
    let out = e.step(Input::DetachSub(sid), t0 + SETTLE * 2);
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileReaped { profile, .. } if *profile == pid)),
        "Parked ⇒ ReapNow; got {:?}",
        out.diagnostics,
    );
    assert!(e.profiles().get(pid).is_none(), "Profile gone");
    assert_eq!(
        e.profiles().nonsteady(),
        0,
        "count returns to zero after the last parked Profile detaches",
    );
}
