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
//! recreated-as-Dir slot through `emit_anchor_probe` and wasting a
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
    ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachRequest, SubId,
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

fn file_leaf() -> LeafEntry {
    LeafEntry::new(
        EntryKind::File,
        0,
        UNIX_EPOCH,
        FsIdentity {
            inode: 1,
            device: 0,
        },
    )
}

fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

fn first_probe_request(out: &StepOutput) -> Option<&ProbeRequest> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request),
        ProbeOp::Cancel { .. } => None,
    })
}

fn count_probes(out: &StepOutput) -> usize {
    out.probe_ops
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
) -> (SubId, ProfileId, StepOutput) {
    let req = SubAttachRequest::for_resource(
        name.into(),
        anchor,
        ScanConfig::builder().recursive(true).build(),
        max_settle,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        events,
        false,
    );
    let out = e.step(Input::AttachSub(req), Instant::now());
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
    // misrouted through `emit_anchor_probe` and wasted a round-trip.
    let mut e = Engine::new();
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::File);

    // Q first — completes Seed AnchorOk → Idle, kind=Some(File).
    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
    );
    let q_corr = first_probe_corr(&out_q).expect("Q's Seed probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_q),
            correlation: q_corr,
            outcome: ProbeOutcome::AnchorOk(file_leaf()),
        }),
        Instant::now(),
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().kind(),
        Some(ResourceKind::File),
    );

    // P next — Active(Seed Verifying) right after attach.
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE);
    let p_corr = first_probe_corr(&out_p).expect("P's Seed probe at attach");
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::File),
    );
    // Both Profiles claim the anchor → watch_demand = 2.
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 2);

    // Drive P's probe to Vanished. discard_anchor_state clears
    // P.kind, P.current, P.baseline, P.anchor_claim.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
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
    // start_seed_burst.
    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );

    // Find P's freshly emitted Seed probe. With kind=None
    // post-fix, start_seed_burst routes through the Subtree arm;
    // pre-fix the cached `Some(File)` would have routed through
    // `emit_anchor_probe` (`ProbeRequest::Anchor`).
    let p_probe = recovery_out
        .probe_ops
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

    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
    );
    let q_corr = first_probe_corr(&out_q).expect("Q's Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_q),
            correlation: q_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        Instant::now(),
    );
    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE);
    let p_corr = first_probe_corr(&out_p).expect("P's Seed probe");
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::Dir),
    );

    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    assert!(e.profiles().get(pid_p).unwrap().kind().is_none());

    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );

    let p_probe_count = recovery_out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_p)))
        .count();
    assert!(
        p_probe_count <= 1,
        "post-fix: at most one probe emitted for P during recovery; got {p_probe_count}",
    );
    let p_probe = first_probe_request(&recovery_out).expect("recovery probe emitted");
    assert!(
        matches!(p_probe, ProbeRequest::Subtree { .. }),
        "Dir→File direction emits Subtree both pre-fix and post-fix",
    );
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

    let (_sid_q, pid_q, out_q) = attach_at(
        &mut e,
        "Q",
        anchor,
        ClassSet::EMPTY,
        MAX_SETTLE + Duration::from_secs(1),
    );
    let q_corr = first_probe_corr(&out_q).expect("Q's Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_q),
            correlation: q_corr,
            outcome: ProbeOutcome::AnchorOk(file_leaf()),
        }),
        Instant::now(),
    );

    let (_sid_p, pid_p, out_p) = attach_at(&mut e, "P", anchor, ClassSet::EMPTY, MAX_SETTLE);
    let p_corr = first_probe_corr(&out_p).expect("P's Seed probe");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::Failed { errno: 5 },
        }),
        Instant::now(),
    );
    assert!(e.profiles().get(pid_p).unwrap().kind().is_none());

    let recovery_out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Modified,
        },
        Instant::now(),
    );
    let p_probe = recovery_out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_p) => {
                Some(request)
            }
            _ => None,
        })
        .expect("P emits a recovery Seed probe");
    assert!(matches!(p_probe, ProbeRequest::Subtree { .. }));
    let _ = count_probes(&recovery_out);
}
