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

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, AnchorClaim, ArgPart, ArgTemplate, ChildEntry, ClassSet, DirChild, DirMeta,
    DirSnapshot, EffectScope, EntryKind, FsEvent, FsIdentity, Input, LeafEntry, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeRequest, ProbeResponse, ProfileId, ProfileState,
    ProofAuthority, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput,
    SubAttachAnchor, SubAttachRequest, SubId,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
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

fn file_leaf() -> LeafEntry {
    LeafEntry::synthetic(EntryKind::File, 0, UNIX_EPOCH, FsIdentity::synthetic(1, 0))
}

fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

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

/// Expire a Batching-first Seed burst's first `Settle` window and
/// return the first Seed probe's correlation. A Seed emits no
/// probe at attach; the first probe materializes only after the initial
/// settle timer (`t0 + SETTLE`) expires and `Batching → Verifying`.
/// Lighter than [`complete_seed_burst`] — for scenarios that terminate
/// the Seed on its *first* response (Vanished / Failed) and never reach
/// the second N=2 cycle. `t0` is the instant the Seed burst started
/// (the attach instant for a live `Resource` anchor).
fn first_seed_probe(e: &mut Engine, pid: ProfileId, t0: Instant) -> (ProbeCorrelation, Instant) {
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
    let c = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("first Seed probe in flight after the initial settle expiry");
    (c, at)
}

/// Drive a Batching-first Seed burst through its full N=2 quiescence
/// proof to `Idle`. A Seed runs the same two-settle-spaced
/// equal-sample proof as a Standard burst: no probe at burst start; the
/// first Seed probe materializes only after the initial settle window
/// (`t0 + SETTLE`) expires.
///
/// 1. expire settle #1 (`t0 + SETTLE`) → first Seed probe; respond with
///    `outcome()`. The carrier's prior `certified` is `None`,
///    so the verdict is `Unstable` by construction → graft + re-batch.
/// 2. expire settle #2 (`t0 + SETTLE*2`) → second Seed probe; respond
///    with the hash-equal `outcome()` → `Stable` → seed pin + rebase →
///    `Idle`.
///
/// `outcome` is re-invoked per probe so the same hash is presented
/// twice (`AnchorOk(leaf)` for a File anchor, `SubtreeProven` for a
/// Dir anchor — both fold through `CertifiedPrior::advance` identically).
/// A fresh Seed emits no Effects. `t0` is the instant the Seed burst
/// started.
fn complete_seed_burst(
    e: &mut Engine,
    pid: ProfileId,
    t0: Instant,
    outcome: impl Fn() -> ProbeOutcome,
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
                outcome: outcome(),
            }),
            at,
        );
        assert!(out.effects().is_empty(), "a fresh Seed never emits Effects");
    }
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle,),
        "Seed burst completes its N=2 proof and returns to Idle",
    );
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
        first_probe_corr(&out_q).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    complete_seed_burst(&mut e, pid_q, t_q, || ProbeOutcome::AnchorOk(file_leaf()));
    assert_eq!(
        e.profiles().get(pid_q).unwrap().kind(),
        Some(ResourceKind::File),
    );

    // P next — Active(PreFire(Seed Batching)) right after attach (no
    // probe yet). Attach strictly after Q's two settle windows.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_corr(&out_p).is_none(),
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
    let (p_corr, p_at) = first_seed_probe(&mut e, pid_p, t_p);
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
        first_probe_corr(&out_q).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    complete_seed_burst(&mut e, pid_q, t_q, || ProbeOutcome::SubtreeProven {
        snapshot: dir_snap(vec![]),
        authority: ProofAuthority::Authoritative,
    });

    // P attaches strictly after Q's two settle windows; Batching-first
    // (no probe at attach). `Profile.kind` is pinned at construction
    // from the anchor's classified kind, independent of the Seed shape.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_corr(&out_p).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::Dir),
    );

    // Expire P's first settle window → first Seed probe; drive it to
    // Vanished (terminates the Seed on its first response).
    let (p_corr, p_at) = first_seed_probe(&mut e, pid_p, t_p);
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
        first_probe_corr(&out_q).is_none(),
        "Batching-first Seed emits no probe at attach",
    );
    complete_seed_burst(&mut e, pid_q, t_q, || ProbeOutcome::AnchorOk(file_leaf()));

    // P attaches strictly after Q's two settle windows; Batching-first.
    let t_p = t_q + SETTLE * 3;
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE, t_p);
    assert!(
        first_probe_corr(&out_p).is_none(),
        "Batching-first Seed emits no probe at attach",
    );

    // Expire P's first settle window → first Seed probe; drive it to
    // Failed. dispatch_seed_failed clears P.kind and terminates the
    // Seed on its first response (the N=2 second cycle is never
    // reached).
    let (p_corr, p_at) = first_seed_probe(&mut e, pid_p, t_p);
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
