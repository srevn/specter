#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::similar_names,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, AnchorClaim, ArgPart, ArgTemplate, ChildEntry, ClaimKind, ClassSet, Diagnostic,
    DirChild, DirMeta, DirSnapshot, EffectScope, EntryKind, FS_ROOT_SEGMENT, FsEvent, FsIdentity,
    Input, LeafEntry, PatternSpec, ProbeCorrelation, ProbeOp, ProbeOutcome, ProbeOwner,
    ProbeResponse, ProfileId, ProfileIdentity, ProfileState, PromoterAttachRequest,
    PromoterClaimKind, PromoterState, ResourceId, ResourceKind, ResourceRole, ScanConfig,
    StepOutput, SubAttachAnchor, SubAttachRequest, SubId, WatchFailure, WatchOp,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

// ───────────────────────────────────────────────────────────────────────
// Fixtures
// ───────────────────────────────────────────────────────────────────────

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> std::sync::Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
        let child = match kind {
            EntryKind::Dir => ChildEntry::Dir(DirChild::Uncovered(FsIdentity::synthetic(inode, 0))),
            _ => ChildEntry::Leaf(LeafEntry::new(
                kind,
                0,
                UNIX_EPOCH,
                FsIdentity::synthetic(inode, 0),
            )),
        };
        map.insert(CompactString::new(name), child);
    }
    Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: UNIX_EPOCH,
            fs_id: FsIdentity::synthetic(0, 0),
        },
        0,
        map,
    ))
}

fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

fn complete_seed_burst(
    e: &mut Engine,
    pid: ProfileId,
    attach_out: &StepOutput,
    seed_snap: std::sync::Arc<DirSnapshot>,
) {
    let corr = first_probe_corr(attach_out).expect("Seed probe fires at attach");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(seed_snap),
        }),
        Instant::now(),
    );
}

fn attach_subtree_root(
    e: &mut Engine,
    name: &str,
    resource: ResourceId,
    max_settle: Duration,
) -> (SubId, ProfileId, StepOutput) {
    let req = SubAttachRequest::for_anchor(
        name.to_string(),
        SubAttachAnchor::Resource(resource),
        ScanConfig::builder().recursive(true).build(),
        max_settle,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    (sid, pid, out)
}

// ───────────────────────────────────────────────────────────────────────
// Anchor case — single Profile
// ───────────────────────────────────────────────────────────────────────

#[test]
fn anchor_claim_purged_then_detach_no_panic() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);

    let (sid, pid, attach_out) = attach_subtree_root(&mut e, "build", root, MAX_SETTLE);
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(vec![]));
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 1);

    // Reject the kernel watch on the anchor.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: root,
            op: WatchOp::Watch {
                resource: root,
                path: Arc::from(Path::new("src")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            failure: WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // Anchor claim cleared; counter zeroed.
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::None,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 0);
    // ProfileClaimPurged{Anchor} surfaces.
    assert!(
        purge_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
                profile, claim, resource, ..
            } if *profile == pid
                && *claim == ClaimKind::Anchor
                && *resource == root)),
        "ProfileClaimPurged{{Anchor}} emitted",
    );

    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(e.profiles().get(pid).is_none(), "Profile reaped cleanly");
}

// ───────────────────────────────────────────────────────────────────────
// Anchor case — multi-Profile (transitive bug surface)
// ───────────────────────────────────────────────────────────────────────

#[test]
fn anchor_claim_purged_for_two_profiles_clears_kind_on_both() {
    // Two Profiles share an anchor classified as Dir. WatchOpRejected
    // on the anchor purges the kernel watch and runs
    // `finalize_anchor_lost` for each anchor claimer; the helper's
    // `discard_anchor_state` must clear `Profile.kind` on each so any
    // subsequent recovery uses the safe Subtree fallback rather than
    // misrouting against a recreated anchor of a different shape.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);

    let (_sid_p, pid_p, out_p) = attach_subtree_root(&mut e, "P", root, MAX_SETTLE);
    let (_sid_q, pid_q, out_q) =
        attach_subtree_root(&mut e, "Q", root, MAX_SETTLE + Duration::from_secs(1));
    complete_seed_burst(&mut e, pid_p, &out_p, dir_snap(vec![]));
    complete_seed_burst(&mut e, pid_q, &out_q, dir_snap(vec![]));

    // Pre-condition: both Profiles cache the anchor's kind.
    assert_eq!(
        e.profiles().get(pid_p).unwrap().kind(),
        Some(ResourceKind::Dir),
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().kind(),
        Some(ResourceKind::Dir),
    );

    let _ = e.step(
        Input::WatchOpRejected {
            resource: root,
            op: WatchOp::Watch {
                resource: root,
                path: Arc::from(Path::new("src")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            failure: WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    assert!(
        e.profiles().get(pid_p).unwrap().kind().is_none(),
        "P's kind cleared by WatchOpRejected anchor purge",
    );
    assert!(
        e.profiles().get(pid_q).unwrap().kind().is_none(),
        "Q's kind cleared by WatchOpRejected anchor purge",
    );
    // Sibling assertion that the existing claim-side discipline is
    // also intact — both anchor claims released, counter zeroed.
    assert_eq!(
        e.profiles().get(pid_p).unwrap().anchor_claim(),
        AnchorClaim::None,
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().anchor_claim(),
        AnchorClaim::None,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 0);
}

#[test]
fn anchor_claim_purged_for_two_profiles_each_no_panic() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);

    let (sid_p, pid_p, out_p) = attach_subtree_root(&mut e, "P", root, MAX_SETTLE);
    let (_sid_q, pid_q, out_q) =
        attach_subtree_root(&mut e, "Q", root, MAX_SETTLE + Duration::from_secs(1));
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 2);
    complete_seed_burst(&mut e, pid_p, &out_p, dir_snap(vec![]));
    complete_seed_burst(&mut e, pid_q, &out_q, dir_snap(vec![]));

    // Reject the kernel watch.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: root,
            op: WatchOp::Watch {
                resource: root,
                path: Arc::from(Path::new("src")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            failure: WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // Both claims cleared.
    assert_eq!(
        e.profiles().get(pid_p).unwrap().anchor_claim(),
        AnchorClaim::None,
    );
    assert_eq!(
        e.profiles().get(pid_q).unwrap().anchor_claim(),
        AnchorClaim::None,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 0);

    // Two anchor purge diagnostics surface.
    let anchor_purge_count = purge_out
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(d, Diagnostic::ProfileClaimPurged {
            claim: ClaimKind::Anchor, resource, ..
        } if *resource == root)
        })
        .count();
    assert_eq!(anchor_purge_count, 2, "one purge per Profile");

    let p_seed_vanished = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    // No panic. Profile P (Idle, anchor_claim=None) is left alone by
    // covering_profiles' filter (P.state is Idle, not Pending, so it
    // would normally route through finalize_anchor_lost). But wait —
    // covering_profiles still includes Idle Profiles. The route is
    // finalize_anchor_lost(P) which is a no-op because anchor_claim is
    // already None (post-purge) and was_active is false (state is Idle,
    // not Active). So it's a clean no-op.
    let _ = p_seed_vanished;

    // Detach P; assert clean reap.
    let _ = e.step(Input::DetachSub(sid_p), Instant::now());
    assert!(e.profiles().get(pid_p).is_none());
    // Q remains alive.
    assert!(e.profiles().get(pid_q).is_some());
}

// ───────────────────────────────────────────────────────────────────────
// Watch-root parent case
// ───────────────────────────────────────────────────────────────────────

#[test]
fn watch_root_parent_claim_purged_then_reap_no_panic() {
    let mut e = Engine::new();
    // Build a parent / anchor pair.
    let parent = e.tree_mut().ensure_root("var", ResourceRole::User);
    e.tree_mut().set_kind(parent, ResourceKind::Dir);
    let anchor = e
        .tree_mut()
        .ensure_child(parent, "log", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(anchor, ResourceKind::Dir);

    let (sid, pid, attach_out) = attach_subtree_root(&mut e, "watch", anchor, MAX_SETTLE);
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(vec![]));
    // `set_watch_root_parent` ran at attach; parent has +1 STRUCTURE.
    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent(),
        Some(parent),
    );
    assert_eq!(e.tree().get(parent).unwrap().watch_demand(), 1);
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 1);

    // Reject the kernel watch on the parent (not the anchor).
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: parent,
            op: WatchOp::Watch {
                resource: parent,
                path: Arc::from(Path::new("var")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            failure: WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // Parent's flag cleared on the Profile; anchor stays watched.
    assert_eq!(e.profiles().get(pid).unwrap().watch_root_parent(), None);
    assert_eq!(
        e.tree()
            .get(parent)
            .map_or(0, specter_core::Resource::watch_demand),
        0
    );
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand(), 1);

    // Diagnostic surfaces.
    assert!(
        purge_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
                profile, claim, resource, ..
            } if *profile == pid
                && *claim == ClaimKind::WatchRootParent
                && *resource == parent)),
        "ProfileClaimPurged{{WatchRootParent}} emitted",
    );

    let _ = e.step(Input::DetachSub(sid), Instant::now());
    assert!(e.profiles().get(pid).is_none(), "Profile reaped cleanly");
}

// ───────────────────────────────────────────────────────────────────────
// Descent prefix case (regression-pinning the existing path)
// ───────────────────────────────────────────────────────────────────────

#[test]
fn descent_prefix_claim_purged_then_anchor_appears_no_recovery() {
    // Pending Profile with prefix=/foo. WatchOpRejected at /foo purges
    // the descent. Profile transitions to Idle without an anchor;
    // operator restart is required to recover (no automatic recovery
    // via parent's StructureChanged because the parent watch failed).
    let mut e = Engine::new();
    let foo = e
        .tree_mut()
        .ensure_path(&["/", "foo"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/foo/bar")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    let initial_corr = first_probe_corr(&attach_out).expect("descent probe");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));

    // Reject the kernel watch on the descent prefix.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: Arc::from(Path::new("foo")),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            failure: WatchFailure::Pressure { errno: 24 },
        },
        Instant::now(),
    );

    // Descent vacated.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    // Cancel + ProfileClaimPurged{DescentPrefix} surface.
    assert!(
        purge_out
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Profile(profile)} if *profile == pid)),
        "in-flight descent probe cancelled",
    );
    assert!(
        purge_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ProfileClaimPurged {
                profile, claim, resource, ..
            } if *profile == pid
                && *claim == ClaimKind::DescentPrefix
                && *resource == foo)),
        "ProfileClaimPurged{{DescentPrefix}} emitted",
    );

    // Late ProbeResponse for the cancelled correlation drops cleanly.
    let late = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: initial_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    assert!(
        late.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. })),
    );
}

// ───────────────────────────────────────────────────────────────────────
// Promoter claim purge
// ───────────────────────────────────────────────────────────────────────

fn promoter_req(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.to_owned(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        log_output: false,
    }
}

fn pre_place_dir(e: &mut Engine, segments: &[&str]) -> ResourceId {
    let mut comps = Vec::with_capacity(segments.len() + 1);
    comps.push(FS_ROOT_SEGMENT);
    comps.extend_from_slice(segments);
    let r = e
        .tree_mut()
        .ensure_path(&comps, ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

fn watch_op_rejected_input(resource: ResourceId, path: &str) -> Input {
    Input::WatchOpRejected {
        resource,
        op: WatchOp::Watch {
            resource,
            path: Arc::from(Path::new(path)),
            kind: ResourceKind::Unknown,
            events: ClassSet::EMPTY,
        },
        failure: WatchFailure::Pressure { errno: 24 },
    }
}

/// `WatchOpRejected` on a Promoter's literal-prefix descent watch
/// purges the descent claim: cancels any in-flight descent probe,
/// transitions the Promoter to `Active{empty}`, and emits a
/// `PromoterClaimPurged{DescentPrefix}` diagnostic. Mirrors the
/// existing Profile-side `descent_prefix_claim_purged_*` test.
#[test]
fn watch_op_rejected_purges_promoter_descent_prefix() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a; attach a Promoter with literal prefix /a/b.
    // /a/b doesn't exist, so the Promoter starts in PrefixPending(/a, [b])
    // and emits a descent probe at /a.
    let a = pre_place_dir(&mut e, &["a"]);
    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/a/b/*.log")),
        now,
    );
    let qid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    assert!(matches!(
        e.promoters().get(qid).unwrap().state,
        PromoterState::PrefixPending(_),
    ));
    let descent_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("descent probe in flight");

    // Reject the kernel watch on /a (the descent prefix).
    let purge_out = e.step(watch_op_rejected_input(a, "/a"), now);

    // Promoter transitioned out of PrefixPending; channel closed.
    assert!(matches!(
        e.promoters().get(qid).unwrap().state,
        PromoterState::Active { .. },
    ));
    assert!(e.pending_probe_for(ProbeOwner::Promoter(qid)).is_none());
    assert_eq!(e.tree().get(a).unwrap().watch_demand(), 0);

    // Cancel emitted for the in-flight descent probe.
    let cancel_emitted = purge_out
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Cancel { owner: ProbeOwner::Promoter(q) } if *q == qid));
    assert!(cancel_emitted, "Cancel emitted for in-flight descent probe");

    // PromoterClaimPurged{DescentPrefix} surfaces.
    let purged = purge_out.diagnostics.iter().any(|d| {
        matches!(d, Diagnostic::PromoterClaimPurged {
            promoter, claim, resource, ..
        } if *promoter == qid
            && *claim == PromoterClaimKind::DescentPrefix
            && *resource == a)
    });
    assert!(purged, "PromoterClaimPurged{{DescentPrefix}} emitted");

    // Late ProbeResponse for the cancelled correlation drops as Stale.
    let late = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: descent_corr,
            outcome: ProbeOutcome::Vanished,
        }),
        now,
    );
    assert!(
        late.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. })),
    );
}

/// `WatchOpRejected` on a Promoter's `Active` proxy purges the proxy
/// claim: clears the proxies map entry, drops the back-ref, and emits
/// a `PromoterClaimPurged{ActiveProxy}` diagnostic. The Promoter
/// remains `Active` (other proxies of the same Promoter, if any,
/// stay; here the single proxy goes empty).
#[test]
fn watch_op_rejected_purges_promoter_active_proxy() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a; attach a Promoter at `/a/*.log`. literal prefix
    // is /a; first proxy registers at /a (immediate-Active mode);
    // an enumeration probe is in flight.
    let a = pre_place_dir(&mut e, &["a"]);
    let attach_out = e.step(Input::AttachPromoter(promoter_req("logs", "/a/*.log")), now);
    let qid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies, .. } => assert!(proxies.contains_key(&a)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active at /a, got {s:?}"),
    }
    assert_eq!(e.tree().get(a).unwrap().watch_demand(), 1);

    // Drain the initial enumeration probe so `pending_enumeration_target`
    // is None — keeps the WatchOpRejected purge's cancel-first
    // contract test focused on the proxy-claim path. Empty entries:
    // no promotions, no sub-proxies.
    let enum_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("enumeration probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: enum_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );
    assert!(e.pending_probe_for(ProbeOwner::Promoter(qid)).is_none());

    // Reject the kernel watch on /a (the proxy).
    let purge_out = e.step(watch_op_rejected_input(a, "/a"), now);

    // Proxy unregistered. Counter zeroed by clamp; back-ref cleared.
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies, .. } => assert!(!proxies.contains_key(&a)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active, got {s:?}"),
    }
    assert_eq!(
        e.tree()
            .get(a)
            .map_or(0, specter_core::Resource::watch_demand),
        0
    );
    let still_back_refed = e
        .tree()
        .get(a)
        .is_some_and(|r| r.proxy_promoters.contains(&qid));
    assert!(!still_back_refed, "back-ref cleared");

    // PromoterClaimPurged{ActiveProxy} surfaces.
    let purged = purge_out.diagnostics.iter().any(|d| {
        matches!(d, Diagnostic::PromoterClaimPurged {
            promoter, claim, resource, ..
        } if *promoter == qid
            && *claim == PromoterClaimKind::ActiveProxy
            && *resource == a)
    });
    assert!(purged, "PromoterClaimPurged{{ActiveProxy}} emitted");
}

/// `WatchOpRejected` on a resource co-claimed by both a Profile
/// (descent prefix) and a Promoter (Active proxy): both purge loops
/// run; both `ProfileClaimPurged{DescentPrefix}` and
/// `PromoterClaimPurged{ActiveProxy}` diagnostics emit; the clamp
/// runs once. Pinning this composition closes the
/// "anchor of P, watch-root-parent of Q, descent prefix of R, proxy
/// of S" multi-actor co-claim story for the Promoter half of the
/// fan-out.
#[test]
fn watch_op_rejected_purges_co_claimed_resource() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a so the Promoter goes immediate-Active with a proxy
    // at /a. The proxy contributes +1 STRUCTURE.
    let a = pre_place_dir(&mut e, &["a"]);
    let attach_q_out = e.step(Input::AttachPromoter(promoter_req("logs", "/a/*.log")), now);
    let qid = specter_core::testkit::first_attached_promoter(&attach_q_out)
        .expect("attach_promoter succeeded");
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies, .. } => assert!(proxies.contains_key(&a)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active, got {s:?}"),
    }
    // Drain the enumeration so `pending_enumeration_target` is None.
    let enum_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("enumeration probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: enum_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Attach a Profile at /a/foo. /a exists; /a/foo does not. The
    // Profile starts in Pending(/a, ["foo"]) and bumps /a's STRUCTURE
    // contribution to 2.
    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Path(PathBuf::from("/a/foo")),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let attach_p_out = e.step(Input::AttachSub(req), now);
    let sid_p =
        specter_core::testkit::first_attached_sub(&attach_p_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_p).unwrap().profile;
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));
    assert_eq!(e.tree().get(a).unwrap().watch_demand(), 2);

    // Reject the kernel watch at /a — co-claimed by Profile descent
    // (ClaimKind::DescentPrefix) and Promoter active proxy
    // (PromoterClaimKind::ActiveProxy).
    let purge_out = e.step(watch_op_rejected_input(a, "/a"), now);

    // Counter zeroed; both claims released.
    assert_eq!(
        e.tree()
            .get(a)
            .map_or(0, specter_core::Resource::watch_demand),
        0
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Idle,
    ));
    match &e.promoters().get(qid).unwrap().state {
        PromoterState::Active { proxies, .. } => assert!(!proxies.contains_key(&a)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active, got {s:?}"),
    }

    // Both diagnostics emit, exactly once each. The umbrella
    // `WatchOpRejected` diagnostic also fires once.
    let profile_purge = purge_out
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(d, Diagnostic::ProfileClaimPurged {
            profile: p, claim: ClaimKind::DescentPrefix, resource, ..
        } if *p == pid && *resource == a)
        })
        .count();
    let promoter_purge = purge_out
        .diagnostics
        .iter()
        .filter(|d| {
            matches!(d, Diagnostic::PromoterClaimPurged {
            promoter: q, claim: PromoterClaimKind::ActiveProxy, resource, ..
        } if *q == qid && *resource == a)
        })
        .count();
    let watch_op_rejected = purge_out
        .diagnostics
        .iter()
        .filter(|d| matches!(d, Diagnostic::WatchOpRejected { resource, .. } if *resource == a))
        .count();

    assert_eq!(
        profile_purge, 1,
        "ProfileClaimPurged{{DescentPrefix}} emitted"
    );
    assert_eq!(
        promoter_purge, 1,
        "PromoterClaimPurged{{ActiveProxy}} emitted"
    );
    assert_eq!(watch_op_rejected, 1, "WatchOpRejected emitted exactly once");
}
