//! Anchor-lifecycle integration tests. The fix-validation half pins
//! that post-anchor-loss recovery routes through the kind-agnostic
//! Subtree probe in both directions (File→Dir and Dir→File), bounding
//! recovery to one round-trip; the regression-prevention half pins the
//! same bound in the Dir→File direction, where the probe shape is
//! Subtree both pre-fix and post-fix.
//!
//! The bug surface: after anchor loss, `Profile.kind` was retained
//! across the lost-recovered cycle. A subsequent `start_seed_burst`
//! routed by stale `kind`, misrouting `Some(File)` against a
//! recreated-as-Dir slot as a `ProbeRequest::AnchorFile` and wasting a
//! round-trip. The fix clears `Profile.kind` inside
//! `discard_anchor_state`; the Subtree fallback in the post-loss
//! window is the new invariant.

use specter_core::testkit::{anchor_ok, dir_snap, empty_program, file_leaf};
use specter_core::{
    AnchorClaim, ClassSet, EffectScope, EntryKind, FsEvent, Input, ProbeOp, ProbeOutcome,
    ProbeOwner, ProbeRequest, ProbeResponse, ProfileId, ProfileState, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    first_probe_correlation, seed_settle_to_verifying, seed_to_idle, seed_to_idle_with,
};
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn first_probe_request(out: &StepOutput) -> Option<&ProbeRequest> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request),
        ProbeOp::Cancel { .. } => None,
    })
}

fn count_probes(out: &StepOutput) -> usize {
    out.probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count()
}

fn attach_at(
    e: &mut Engine,
    name: &str,
    anchor: ResourceId,
    events: ClassSet,
    max_settle: Duration,
    now: Instant,
) -> (SubId, ProfileId, StepOutput) {
    let req = SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        max_settle,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        events,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    (sid, pid, out)
}

#[test]
fn recovery_from_file_to_dir_anchor_uses_subtree_probe() {
    // Multi-Profile sharing a File-classified anchor. Profile P loses
    // its anchor via probe Vanished; `discard_anchor_state` clears
    // `kind`. With Q's anchor claim keeping the watch alive, a
    // subsequent FsEvent at the anchor routes through `drive_burst`
    // into `start_seed_burst` for P (Idle, current=None). Post-fix:
    // kind=None, start_seed_burst routes through the kind-agnostic
    // Subtree arm — recovery in one round-trip via descent regardless
    // of the recreated anchor's shape. Pre-fix the cached `Some(File)`
    // misrouted as a `ProbeRequest::AnchorFile` and wasted a round-trip.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::File);

    // Q first — completes its N=2 Seed (File anchor → AnchorOk both
    // samples) → Idle, kind=Some(File). The immediate Seed is
    // Batching-first: no probe at attach.
    let t_q = Instant::now();
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
        t_q,
    );
    assert!(
        first_probe_correlation(&out_q).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let _ = seed_to_idle_with(
        &mut e,
        pid_q,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        t_q,
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().kind(),
        Some(ResourceKind::File),
    );

    // P next — Active(PreFire(Seed Batching)) right after attach (no
    // probe yet). Attach strictly after Q's two settle windows.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_correlation(&out_p).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    // `Profile.kind` is pinned at construction from the anchor's
    // classified kind, independent of the Batching-first Seed.
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::File),
    );
    // Both Profiles claim the anchor → watch_demand = 2 (the claim is
    // bumped at attach by `bootstrap_immediate`, before the Seed).
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 2);

    // Expire P's first settle window → first Seed probe, then drive it
    // to Vanished. discard_anchor_state clears P.kind, P.current,
    // P.baseline, P.anchor_claim. Vanished terminates the Seed on its
    // first response — N=2's second cycle is never reached.
    let (p_corr, p_at) = seed_settle_to_verifying(&mut e, pid_p, t_p);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        p_at,
    );
    let p = e.profiles().get(pid_p).expect("P alive");
    assert!(p.kind().is_none(), "P.kind cleared by discard_anchor_state");
    assert!(p.current().is_none());
    assert!(p.baseline().is_none());
    assert_eq!(p.anchor_claim(), AnchorClaim::None);
    assert!(matches!(p.state(), ProfileState::Idle));
    // Q's claim keeps the anchor alive.
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 1);

    // Inject FsEvent at the anchor — Q is alive so the kernel watch
    // is still in place. drive_burst routes P (Idle, current=None) to
    // start_seed_burst, which opens Batching-first: no probe
    // at burst start.
    let recovery_t0 = p_at + SETTLE;
    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Modified,
        },
        recovery_t0,
    );
    assert_eq!(
        count_probes(&recovery_out),
        0,
        "Batching-first recovery Seed emits no probe at burst start",
    );
    assert!(
        matches!(
            e.profiles().get(pid_p).unwrap().state(),
            ProfileState::Active(_, _),
        ),
        "P re-entered an Active Seed burst on the recovery event",
    );

    // Expire the recovery Seed's first settle window → the recovery
    // Seed probe materializes. With kind=None post-fix, start_seed_burst
    // routes through the Subtree arm; pre-fix the cached `Some(File)`
    // would have emitted a `ProbeRequest::AnchorFile`.
    let probe_at = recovery_t0 + SETTLE;
    let mut p_probe_out = None;
    while let Some(en) = e.pop_expired(probe_at) {
        let o = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            probe_at,
        );
        if first_probe_request(&o).is_some() {
            p_probe_out = Some(o);
        }
    }
    let p_probe_out = p_probe_out.expect("recovery Seed probe after settle expiry");
    let p_probe = p_probe_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_p) => {
                Some(request)
            }
            _ => None,
        })
        .expect("P emits a recovery Seed probe");
    assert!(
        matches!(p_probe, ProbeRequest::Subtree { .. }),
        "post-fix: kind=None routes recovery through Subtree probe; got {p_probe:?}",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn recovery_from_dir_to_file_anchor_bounded_to_one_round_trip() {
    // Regression-prevention: post-fix recovery in the Dir→File
    // direction still bounds at one round-trip. Both pre-fix and
    // post-fix ship Subtree (pre-fix kind=Some(Dir) → Subtree;
    // post-fix kind=None → Subtree) so this test does NOT
    // discriminate the fix; it pins the bound against future
    // regressions where the recovery path could unintentionally
    // multi-probe.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "build", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    // Q's N=2 Seed (Dir anchor → SubtreeProven both samples) → Idle.
    // The immediate Seed is Batching-first: no probe at attach.
    let t_q = Instant::now();
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
        t_q,
    );
    assert!(
        first_probe_correlation(&out_q).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let _ = seed_to_idle(&mut e, pid_q, &dir_snap(&[]), t_q);

    // P attaches strictly after Q's two settle windows; Batching-first
    // (no probe at attach). `Profile.kind` is pinned at construction
    // from the anchor's classified kind, independent of the Seed shape.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_correlation(&out_p).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::Dir),
    );

    // Expire P's first settle window → first Seed probe; drive it to
    // Vanished (terminates the Seed on its first response).
    let (p_corr, p_at) = seed_settle_to_verifying(&mut e, pid_p, t_p);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        p_at,
    );
    assert!(e.profiles().get(pid_p).unwrap().kind().is_none());

    // Recovery FsEvent → Batching-first Seed (no probe at burst start).
    let recovery_t0 = p_at + SETTLE;
    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Modified,
        },
        recovery_t0,
    );
    assert_eq!(
        count_probes(&recovery_out),
        0,
        "Batching-first recovery Seed emits no probe at burst start",
    );

    // Expire the recovery Seed's first settle window. The bound: P's
    // recovery emits at most one probe (across burst start + the settle
    // expiry that surfaces it), and it is Subtree-shaped.
    let probe_at = recovery_t0 + SETTLE;
    let mut settle_out = None;
    while let Some(en) = e.pop_expired(probe_at) {
        let o = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            probe_at,
        );
        if first_probe_request(&o).is_some() {
            settle_out = Some(o);
        }
    }
    let settle_out = settle_out.expect("recovery Seed probe after settle expiry");
    let p_probe_count = recovery_out
        .probe_ops()
        .iter()
        .chain(settle_out.probe_ops().iter())
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_p)))
        .count();
    assert!(
        p_probe_count <= 1,
        "post-fix: at most one probe emitted for P during recovery; got {p_probe_count}",
    );
    let p_probe = first_probe_request(&settle_out).expect("recovery probe emitted");
    assert!(
        matches!(p_probe, ProbeRequest::Subtree { .. }),
        "Dir→File direction emits Subtree both pre-fix and post-fix",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn anchor_loss_via_probe_failed_clears_kind_and_recovers_via_subtree() {
    // Mirror of `recovery_from_file_to_dir_anchor_uses_subtree_probe`
    // for the Failed dispatch path. dispatch_seed_failed shares the
    // helper; the post-recovery probe must be Subtree.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::File);

    // Q's N=2 Seed (File anchor → AnchorOk both samples) → Idle.
    // The immediate Seed is Batching-first: no probe at attach.
    let t_q = Instant::now();
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
        t_q,
    );
    assert!(
        first_probe_correlation(&out_q).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    let _ = seed_to_idle_with(
        &mut e,
        pid_q,
        || anchor_ok(file_leaf(EntryKind::File, 1)),
        t_q,
    );

    // P attaches strictly after Q's two settle windows; Batching-first.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_correlation(&out_p).is_none(),
        "Batching-first Seed emits no probe at attach",
    );

    // Expire P's first settle window → first Seed probe; drive it to
    // Failed. dispatch_seed_failed clears P.kind and terminates the
    // Seed on its first response (the N=2 second cycle is never
    // reached).
    let (p_corr, p_at) = seed_settle_to_verifying(&mut e, pid_p, t_p);
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::Failed { errno: 5 },
        }),
        p_at,
    );
    assert!(e.profiles().get(pid_p).unwrap().kind().is_none());

    // Recovery FsEvent → Batching-first Seed (no probe at burst start).
    let recovery_t0 = p_at + SETTLE;
    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Modified,
        },
        recovery_t0,
    );
    assert_eq!(
        count_probes(&recovery_out),
        0,
        "Batching-first recovery Seed emits no probe at burst start",
    );

    // Expire the recovery Seed's first settle window → the recovery
    // Seed probe materializes; it must be Subtree-shaped (kind=None
    // after the Failed-driven discard).
    let probe_at = recovery_t0 + SETTLE;
    let mut settle_out = None;
    while let Some(en) = e.pop_expired(probe_at) {
        let o = e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            probe_at,
        );
        if first_probe_request(&o).is_some() {
            settle_out = Some(o);
        }
    }
    let settle_out = settle_out.expect("recovery Seed probe after settle expiry");
    let p_probe = settle_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_p) => {
                Some(request)
            }
            _ => None,
        })
        .expect("P emits a recovery Seed probe");
    assert!(matches!(p_probe, ProbeRequest::Subtree { .. }));
    let _ = e.cancel_all_in_flight_probes();
}
