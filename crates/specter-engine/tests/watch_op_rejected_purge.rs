use specter_core::testkit::{dir_snap, empty_program};
use specter_core::{
    AnchorClaim, ClaimKind, ClassSet, Diagnostic, EffectScope, FS_ROOT_SEGMENT, FsEvent, Input,
    PatternSpec, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileIdentity,
    ProfileState, PromoterAttachRequest, PromoterClaimKind, PromoterState, ResourceId,
    ResourceKind, ResourceRole, ScanConfig, SubAttachAnchor, SubAttachRequest, SubId, WatchFailure,
};
use specter_engine::Engine;
use specter_engine::testkit::{SETTLE, first_probe_correlation, seed_to_idle};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const MAX_SETTLE: Duration = Duration::from_secs(6);

// ───────────────────────────────────────────────────────────────────────
// Fixtures
// ───────────────────────────────────────────────────────────────────────

fn attach_subtree_root(
    e: &mut Engine,
    name: &str,
    resource: ResourceId,
    max_settle: Duration,
    now: Instant,
) -> (SubId, ProfileId) {
    let req = SubAttachRequest::for_anchor(
        name.into(),
        SubAttachAnchor::Resource(resource),
        ScanConfig::builder().recursive(true).build(),
        max_settle,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile();
    (sid, pid)
}

// ───────────────────────────────────────────────────────────────────────
// Anchor case — single Profile
// ───────────────────────────────────────────────────────────────────────

#[test]
fn anchor_claim_purged_then_detach_no_panic() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);

    let t0 = Instant::now();
    let (sid, pid) = attach_subtree_root(&mut e, "build", root, MAX_SETTLE, t0);
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), t0);
    assert_eq!(
        e.profiles().get(pid).unwrap().anchor_claim(),
        AnchorClaim::Held,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 1);

    // Reject the kernel watch on the anchor.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: root,
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
    // Two Profiles share an anchor classified as Dir. WatchOpRejected on the anchor purges the
    // kernel watch and runs `finalize_anchor_lost` for each anchor claimer; the helper's
    // `discard_anchor_state` must clear `Profile.kind` on each so any subsequent recovery uses the
    // safe Subtree fallback rather than misrouting against a recreated anchor of a different shape.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);

    let t_p = Instant::now();
    let (_sid_p, pid_p) = attach_subtree_root(&mut e, "P", root, MAX_SETTLE, t_p);
    let t_q = t_p + SETTLE * 4;
    let (_sid_q, pid_q) =
        attach_subtree_root(&mut e, "Q", root, MAX_SETTLE + Duration::from_secs(1), t_q);
    // Each Seed burst is driven on its own settle timer (the helper steps only the named Profile's
    // Batching id), so P and Q never cross-fire despite sharing the anchor.
    let _ = seed_to_idle(&mut e, pid_p, &dir_snap(&[]), t_p);
    let _ = seed_to_idle(&mut e, pid_q, &dir_snap(&[]), t_q);

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
    // Sibling assertion that the existing claim-side discipline is also intact — both anchor claims
    // released, counter zeroed.
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

    let t_p = Instant::now();
    let (sid_p, pid_p) = attach_subtree_root(&mut e, "P", root, MAX_SETTLE, t_p);
    let t_q = t_p + SETTLE * 4;
    let (_sid_q, pid_q) =
        attach_subtree_root(&mut e, "Q", root, MAX_SETTLE + Duration::from_secs(1), t_q);
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 2);
    // Per-Profile settle stepping keeps P and Q from cross-firing.
    let _ = seed_to_idle(&mut e, pid_p, &dir_snap(&[]), t_p);
    let _ = seed_to_idle(&mut e, pid_q, &dir_snap(&[]), t_q);

    // Reject the kernel watch.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: root,
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
    // No panic. Profile P (Idle, anchor_claim=None) is left alone by covering_profiles' filter
    // (P.state is Idle, not Pending, so it would normally route through finalize_anchor_lost). But
    // wait — covering_profiles still includes Idle Profiles. The route is finalize_anchor_lost(P)
    // which is a no-op because anchor_claim is already None (post-purge) and was_active is false
    // (state is Idle, not Active). So it's a clean no-op.
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

    let t0 = Instant::now();
    let (sid, pid) = attach_subtree_root(&mut e, "watch", anchor, MAX_SETTLE, t0);
    let _ = seed_to_idle(&mut e, pid, &dir_snap(&[]), t0);
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
    // Pending Profile with prefix=/foo. WatchOpRejected at /foo purges the descent. Profile
    // transitions to Idle without an anchor; operator restart is required to recover (no automatic
    // recovery via parent's StructureChanged because the parent watch failed).
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
    let pid = e.subs().get(sid).unwrap().profile();
    let initial_corr = first_probe_correlation(&attach_out).expect("descent probe");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));

    // Reject the kernel watch on the descent prefix.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: foo,
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
            .probe_ops()
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
        name: name.into(),
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

const fn watch_op_rejected_input(resource: ResourceId) -> Input {
    Input::WatchOpRejected {
        resource,
        failure: WatchFailure::Pressure { errno: 24 },
    }
}

/// `WatchOpRejected` on a Promoter's literal-prefix descent watch purges the descent claim: cancels
/// any in-flight descent probe, transitions the Promoter to `Active{empty}`, and emits a
/// `PromoterClaimPurged{DescentPrefix}` diagnostic. Mirrors the existing Profile-side
/// `descent_prefix_claim_purged_*` test.
#[test]
fn watch_op_rejected_purges_promoter_descent_prefix() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a; attach a Promoter with literal prefix /a/b. /a/b doesn't exist, so the Promoter
    // starts in PrefixPending(/a, [b]) and emits a descent probe at /a.
    let a = pre_place_dir(&mut e, &["a"]);
    let attach_out = e.step(
        Input::AttachPromoter(promoter_req("logs", "/a/b/*.log")),
        now,
    );
    let qid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    assert!(matches!(
        e.promoters().get(qid).unwrap().state(),
        PromoterState::PrefixPending(_),
    ));
    let descent_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("descent probe in flight");

    // Reject the kernel watch on /a (the descent prefix).
    let purge_out = e.step(watch_op_rejected_input(a), now);

    // Promoter transitioned out of PrefixPending; channel closed.
    assert!(matches!(
        e.promoters().get(qid).unwrap().state(),
        PromoterState::Active { .. },
    ));
    assert!(e.pending_probe_for(ProbeOwner::Promoter(qid)).is_none());
    assert_eq!(e.tree().get(a).unwrap().watch_demand(), 0);

    // Cancel emitted for the in-flight descent probe.
    let cancel_emitted = purge_out
        .probe_ops()
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

/// `WatchOpRejected` on a Promoter's `Active` proxy purges the proxy claim: clears the proxies map
/// entry, drops the back-ref, and emits a `PromoterClaimPurged{ActiveProxy}` diagnostic. The
/// Promoter remains `Active` (other proxies of the same Promoter, if any, stay; here the single
/// proxy goes empty).
#[test]
fn watch_op_rejected_purges_promoter_active_proxy() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a; attach a Promoter at `/a/*.log`. literal prefix is /a; first proxy registers at
    // /a (immediate-Active mode); an enumeration probe is in flight.
    let a = pre_place_dir(&mut e, &["a"]);
    let attach_out = e.step(Input::AttachPromoter(promoter_req("logs", "/a/*.log")), now);
    let qid = specter_core::testkit::first_attached_promoter(&attach_out)
        .expect("attach_promoter succeeded");
    match e.promoters().get(qid).unwrap().state() {
        PromoterState::Active { proxies, .. } => assert!(proxies.contains_key(&a)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active at /a, got {s:?}"),
    }
    assert_eq!(e.tree().get(a).unwrap().watch_demand(), 1);

    // Drain the initial enumeration probe so `pending_enumeration_target` is None — keeps the
    // WatchOpRejected purge's cancel-first contract test focused on the proxy-claim path. Empty
    // entries: no promotions, no sub-proxies.
    let enum_corr = e
        .pending_probe_for(ProbeOwner::Promoter(qid))
        .expect("enumeration probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(qid),
            correlation: enum_corr,
            outcome: ProbeOutcome::DirEnumerated(dir_snap(&[])),
        }),
        now,
    );
    assert!(e.pending_probe_for(ProbeOwner::Promoter(qid)).is_none());

    // Reject the kernel watch on /a (the proxy).
    let purge_out = e.step(watch_op_rejected_input(a), now);

    // Proxy unregistered. Counter zeroed by clamp; back-ref cleared.
    match e.promoters().get(qid).unwrap().state() {
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
        .is_some_and(|r| r.proxy_promoters().contains(&qid));
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

/// `WatchOpRejected` on a resource co-claimed by both a Profile (descent prefix) and a Promoter
/// (Active proxy): both purge loops run; both `ProfileClaimPurged{DescentPrefix}` and
/// `PromoterClaimPurged{ActiveProxy}` diagnostics emit; the clamp runs once. Pinning this
/// composition closes the "anchor of P, watch-root-parent of Q, descent prefix of R, proxy of S"
/// multi-actor co-claim story for the Promoter half of the fan-out.
#[test]
fn watch_op_rejected_purges_co_claimed_resource() {
    let mut e = Engine::new();
    let now = Instant::now();

    // Pre-place /a so the Promoter goes immediate-Active with a proxy at /a. The proxy contributes
    // +1 STRUCTURE.
    let a = pre_place_dir(&mut e, &["a"]);
    let promoter_attach_out = e.step(Input::AttachPromoter(promoter_req("logs", "/a/*.log")), now);
    let qid = specter_core::testkit::first_attached_promoter(&promoter_attach_out)
        .expect("attach_promoter succeeded");
    match e.promoters().get(qid).unwrap().state() {
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
            outcome: ProbeOutcome::DirEnumerated(dir_snap(&[])),
        }),
        now,
    );

    // Attach a Profile at /a/foo. /a exists; /a/foo does not. The Profile starts in Pending(/a,
    // ["foo"]) and bumps /a's STRUCTURE contribution to 2.
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
    let sub_attach_out = e.step(Input::AttachSub(req), now);
    let sid_p =
        specter_core::testkit::first_attached_sub(&sub_attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_p).unwrap().profile();
    assert!(matches!(
        e.profiles().get(pid).unwrap().state(),
        ProfileState::Pending(_),
    ));
    assert_eq!(e.tree().get(a).unwrap().watch_demand(), 2);

    // Reject the kernel watch at /a — co-claimed by Profile descent (ClaimKind::DescentPrefix) and
    // Promoter active proxy (PromoterClaimKind::ActiveProxy).
    let purge_out = e.step(watch_op_rejected_input(a), now);

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
    match e.promoters().get(qid).unwrap().state() {
        PromoterState::Active { proxies, .. } => assert!(!proxies.contains_key(&a)),
        s @ PromoterState::PrefixPending(_) => panic!("expected Active, got {s:?}"),
    }

    // Both diagnostics emit, exactly once each. The umbrella `WatchOpRejected` diagnostic also
    // fires once.
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
