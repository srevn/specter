//! Promoter terminus-loss recovery (the structural mirror of a Profile's `watch_root_parent`
//! anchor-loss recovery).
//!
//! These cross-cutting tests drive one [`Engine`] through the full recovery composition with
//! synthetic [`ProbeResponse`] / [`Input`] injection: an `Active` Promoter loses its materialised
//! literal prefix (`terminus`), collapses to `Active { proxies: ∅ }` while the preserved
//! `prefix_parent` edge survives, and re-enters the shared descent machine on the parent's *next*
//! structural event — never via an idle self-probe. The inline `promoter_tests.rs` pins the
//! individual transitions; this file pins their composition and the refcount / dedup invariants
//! that only emerge end-to-end.
//!
//! Pattern under test is `/srv/app/*`: literal prefix `/srv/app` (`literal_prefix_len == 3`),
//! terminus `/srv/app`, recovery parent edge at `/srv`.

use compact_str::CompactString;
use specter_core::testkit::{dir_snap, empty_program};
use specter_core::{
    ClassSet, ContribKey, Diagnostic, EffectScope, EntryKind, FS_ROOT_SEGMENT, FsEvent, Input,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileIdentity, PromoterId, PromoterState,
    ResourceId, ScanConfig, SubAttachAnchor, SubAttachRequest, SubParams,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    attach_promoter, descent_advance, dynamic_subs_of, last_probe_path, pre_place_dir, promoter_req,
};
use std::time::{Duration, Instant};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

/// `true` iff the Promoter is `Active { proxies: ∅ }` — the exact terminus-lost discriminant Design
/// W keys recovery on.
fn active_empty(e: &Engine, pid: PromoterId) -> bool {
    matches!(
        e.promoters().get(pid).map(specter_core::Promoter::state),
        Some(PromoterState::Active { proxies, .. }) if proxies.is_empty()
    )
}

/// The proxy resource ids of an `Active` Promoter (panics if `PrefixPending` — the caller asserts
/// `Active` shape first).
fn proxy_ids(e: &Engine, pid: PromoterId) -> Vec<ResourceId> {
    let other = e.promoters().get(pid).map(specter_core::Promoter::state);
    let Some(PromoterState::Active { proxies, .. }) = other else {
        panic!("expected Active, got {other:?}");
    };
    proxies.keys().copied().collect()
}

/// Attach `/srv/app/*` with `/srv/app` pre-placed ⇒ immediate `Active` with the proxy at the
/// materialised terminus. Returns `(pid, srv, srv_app)`.
fn attach_active(e: &mut Engine) -> (PromoterId, ResourceId, ResourceId) {
    let srv = pre_place_dir(e, &["srv"]);
    let srv_app = pre_place_dir(e, &["srv", "app"]);
    let pid = attach_promoter(e, "recover", "/srv/app/*", Instant::now());
    (pid, srv, srv_app)
}

/// Respond the in-flight Promoter probe with `DirEnumerated(children)`.
fn respond_ok(e: &mut Engine, pid: PromoterId, children: &[(&str, EntryKind, u64)]) {
    let _ = descent_advance(
        e,
        ProbeOwner::Promoter(pid),
        &dir_snap(children),
        Instant::now(),
    );
}

/// Drive `Active`-with-proxy → terminus loss: an `FsEvent` at the proxy arms a fresh enumeration
/// probe, which we answer `Vanished`. Returns the `Vanished`-step output (for the no-idle-probe
/// assertion).
fn lose_terminus(
    e: &mut Engine,
    pid: PromoterId,
    terminus: ResourceId,
) -> specter_core::StepOutput {
    let _ = e.step(
        Input::FsEvent {
            resource: terminus,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    let corr = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("enumeration probe armed by the proxy event");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    )
}

// ---------------------------------------------------------------------
// Scenario 1 + 2 — terminus reappearance is event-gated; no idle probe.
// ---------------------------------------------------------------------

#[test]
fn terminus_loss_then_parent_event_recovers_and_remints_with_no_idle_probe() {
    let mut e = Engine::new();
    let (pid, srv, srv_app) = attach_active(&mut e);

    // Post-attach: proxy at the terminus; the preserved parent edge is installed at /srv.
    assert_eq!(
        proxy_ids(&e, pid),
        vec![srv_app],
        "attach lands Active with the proxy at the materialised terminus",
    );
    assert_eq!(
        e.promoters().get(pid).unwrap().prefix_parent(),
        Some(srv),
        "prefix_parent edge cached at the terminus's parent",
    );
    assert!(
        e.tree()
            .get(srv)
            .unwrap()
            .contributions()
            .contains_key(&ContribKey::PromoterPrefixParent(pid)),
        "PromoterPrefixParent STRUCTURE contribution installed at /srv",
    );
    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        1,
        "/srv carries exactly the parent-edge contribution",
    );

    // Initial enumeration is empty — no dynamic Sub yet (keeps the recovery mechanism isolated from
    // the dynamic-Sub lifecycle).
    respond_ok(&mut e, pid, &[]);
    assert!(
        dynamic_subs_of(&e, pid).is_empty(),
        "empty enumeration mints nothing",
    );

    // ---- terminus loss ----
    let loss_out = lose_terminus(&mut e, pid, srv_app);

    assert!(
        loss_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterEnumerationVanished { promoter, .. } if *promoter == pid,
        )),
        "terminus loss observed as PromoterEnumerationVanished",
    );
    assert!(
        active_empty(&e, pid),
        "terminus loss collapses to Active {{ proxies: ∅ }}",
    );
    assert_eq!(
        e.promoters().get(pid).unwrap().prefix_parent(),
        Some(srv),
        "prefix_parent edge survives terminus loss (downward-only unreg)",
    );
    assert!(
        e.tree()
            .get(srv)
            .unwrap()
            .contributions()
            .contains_key(&ContribKey::PromoterPrefixParent(pid)),
        "parent-edge contribution preserved across the loss",
    );

    // Scenario 2 — between loss and the parent event the Promoter emits ZERO probes. This is the
    // property that distinguishes Design W (event-gated) from a self-triggered idle-probe recovery.
    assert!(
        !loss_out
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { .. })),
        "no idle recovery probe is emitted on terminus loss; got {:?}",
        loss_out.probe_ops(),
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "no Promoter probe in flight while awaiting the parent's event",
    );

    // ---- recovery: the parent's next StructureChanged ----
    let rec_out = e.step(
        Input::FsEvent {
            resource: srv,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    {
        let q = e.promoters().get(pid).expect("promoter alive");
        let PromoterState::PrefixPending(d) = q.state() else {
            panic!("expected PrefixPending after recovery, got {:?}", q.state());
        };
        assert_eq!(
            d.current_prefix(),
            srv,
            "recovery descent roots at the preserved parent edge",
        );
        assert_eq!(
            d.remaining_components().iter().cloned().collect::<Vec<_>>(),
            vec![CompactString::from("app")],
            "recovery segment is the STATIC pattern literal, not tree.name(terminus)",
        );
        assert_eq!(
            q.prefix_parent(),
            Some(srv),
            "prefix_parent unchanged by the recovery re-entry",
        );
    }
    assert_eq!(
        last_probe_path(&rec_out).as_deref(),
        e.tree().path_of(srv).as_deref(),
        "recovery descent probe targets the parent",
    );
    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        2,
        "the documented +2 overlap: PromoterPrefixParent + PromoterPrefix",
    );

    // ---- descent re-materialises the terminus ----
    respond_ok(&mut e, pid, &[("app", EntryKind::Dir, 5)]);
    {
        let q = e.promoters().get(pid).expect("promoter alive");
        let PromoterState::Active { proxies, .. } = q.state() else {
            panic!(
                "expected Active after re-materialisation, got {:?}",
                q.state()
            );
        };
        let proxy = *proxies.keys().next().expect("one proxy re-registered");
        assert_eq!(
            e.tree().path_of(proxy).as_deref(),
            std::path::Path::new("/srv/app").into(),
            "proxy re-registered at the re-materialised terminus",
        );
        assert_eq!(
            q.prefix_parent(),
            Some(srv),
            "prefix_parent still cached (idempotent set_promoter_prefix_parent)",
        );
    }
    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        1,
        "descent contribution handed off; only the parent edge remains (no +1 leak)",
    );

    // ---- recovered enumeration mints a fresh dynamic Sub ----
    respond_ok(&mut e, pid, &[("bar", EntryKind::Dir, 6)]);
    let promoted = dynamic_subs_of(&e, pid);
    assert_eq!(promoted.len(), 1, "recovery re-enumeration mints one Sub");
    let anchor = *promoted.keys().next().unwrap();
    assert_eq!(
        e.tree().path_of(anchor).as_deref(),
        std::path::Path::new("/srv/app/bar").into(),
        "fresh dynamic Sub anchored at the recovered match",
    );

    let _ = e.cancel_all_in_flight_probes();
}

// ---------------------------------------------------------------------
// Scenario 3 — cascade: rm -rf terminus AND its parent. Recovery rewinds through the shared
// owner-polymorphic descent machine; the +2 overlap pins /srv alive while the descent climbs to
// FS-root and back.
// ---------------------------------------------------------------------

#[test]
fn recovery_cascade_rewinds_through_parent_to_fs_root() {
    let mut e = Engine::new();
    let (pid, srv, srv_app) = attach_active(&mut e);
    respond_ok(&mut e, pid, &[]); // empty initial enumeration

    let _ = lose_terminus(&mut e, pid, srv_app);
    assert!(active_empty(&e, pid));

    // Parent event re-enters recovery descent rooted at /srv.
    let _ = e.step(
        Input::FsEvent {
            resource: srv,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        2,
        "+2 overlap during recovery PrefixPending",
    );

    // /srv itself is now gone too — answer the descent Vanished.
    let corr = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("recovery descent probe in flight");
    let vanish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    assert!(
        vanish_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterDescentVanished { promoter, .. } if *promoter == pid,
        )),
        "cascade routed through the shared descent-vanished rewind",
    );

    let fs_root = e.tree().lookup(None, FS_ROOT_SEGMENT).expect("FS-root");
    {
        let q = e.promoters().get(pid).expect("promoter alive");
        let PromoterState::PrefixPending(d) = q.state() else {
            panic!("expected PrefixPending after rewind, got {:?}", q.state());
        };
        assert_eq!(d.current_prefix(), fs_root, "rewound to FS-root");
        assert_eq!(
            d.remaining_components().iter().cloned().collect::<Vec<_>>(),
            vec![CompactString::from("srv"), CompactString::from("app")],
            "vanished prefix segment prepended; static terminus segment preserved",
        );
        assert_eq!(
            q.prefix_parent(),
            Some(srv),
            "prefix_parent NEVER touched by the descent rewind (only PromoterPrefix moves)",
        );
    }
    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        1,
        "rewind dropped PromoterPrefix; PromoterPrefixParent pins /srv alive (+2 → +1)",
    );

    // Reappearance: descend FS-root → /srv → /srv/app, back to Active.
    respond_ok(&mut e, pid, &[("srv", EntryKind::Dir, 2)]); // FS-root has srv
    respond_ok(&mut e, pid, &[("app", EntryKind::Dir, 3)]); // /srv has app → enter_active

    {
        let q = e.promoters().get(pid).expect("promoter alive");
        let PromoterState::Active { proxies, .. } = q.state() else {
            panic!(
                "expected Active after cascade recovery, got {:?}",
                q.state()
            );
        };
        assert_eq!(proxies.len(), 1, "terminus re-materialised");
        assert_eq!(
            q.prefix_parent(),
            Some(srv),
            "prefix_parent stable across the whole cascade",
        );
    }
    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        1,
        "post-recovery /srv carries exactly the parent edge — no leak across the cascade",
    );

    let _ = e.cancel_all_in_flight_probes();
}

// ---------------------------------------------------------------------
// Scenario 4 — terminus == "/" (literal_prefix_len == 1): no parent, so no PromoterPrefixParent,
// the recovery carrier never classifies, and start_promoter_prefix_recovery (with its
// components[lpl-1] read) is structurally unreachable — no from_vec panic path exists.
// ---------------------------------------------------------------------

#[test]
fn root_terminus_installs_no_parent_edge_and_never_recovers() {
    let mut e = Engine::new();
    // `/*/data`: components [Literal("/"), Glob("*"), Literal("data")]; literal_prefix_len == 1 ⇒
    // terminus is "/" (FS-root, always present ⇒ immediate Active).
    let out = e.step(
        Input::AttachPromoter(promoter_req("rooted", "/*/data")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let fs_root = e.tree().lookup(None, FS_ROOT_SEGMENT).expect("FS-root");

    assert!(
        e.promoters().get(pid).unwrap().prefix_parent().is_none(),
        "terminus == / has no parent ⇒ no prefix_parent edge",
    );
    assert!(
        e.tree().get(fs_root).is_none_or(|r| !r
            .contributions()
            .contains_key(&ContribKey::PromoterPrefixParent(pid))),
        "no PromoterPrefixParent contribution anywhere",
    );

    // Force Active { proxies: ∅ } via a Vanished enumeration.
    let corr = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("enumeration probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );
    assert!(active_empty(&e, pid));

    // An event at FS-root must NOT classify as a Promoter recovery (prefix_parent is None) — no
    // PrefixPending re-entry, no panic.
    let ev_out = e.step(
        Input::FsEvent {
            resource: fs_root,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    assert!(
        active_empty(&e, pid),
        "no recovery: stays Active {{ proxies: ∅ }} (carrier never classifies)",
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "no recovery descent probe minted for a root-terminus Promoter",
    );
    assert!(
        e.promoters().get(pid).unwrap().prefix_parent().is_none(),
        "prefix_parent stays None — start_promoter_prefix_recovery is unreachable",
    );
    // The event went nowhere (benign no-op: FS-root carries no contribution once the synthetic
    // Vanish removed the sole proxy, so it is reported unwatched / no-consumer — never a recovery).
    // The exact benign variant is incidental; what matters is that NO recovery diagnostic and NO
    // PromoterPrefixParent purge fired.
    assert!(
        ev_out.diagnostics.iter().all(|d| matches!(
            d,
            Diagnostic::EventNoConsumer { .. } | Diagnostic::EventOnUnwatchedResource { .. },
        )),
        "root-terminus parent event is a benign no-op, never recovery; got {:?}",
        ev_out.diagnostics,
    );

    let _ = e.cancel_all_in_flight_probes();
}

// ---------------------------------------------------------------------
// Scenario 7 — repeated loss → recovery cycles do not leak the PromoterPrefixParent refcount (the
// recovery-idempotence guard).
// ---------------------------------------------------------------------

#[test]
fn repeated_loss_recovery_cycles_keep_prefix_parent_refcount_invariant() {
    let mut e = Engine::new();
    let (pid, srv, mut terminus) = attach_active(&mut e);
    respond_ok(&mut e, pid, &[]); // empty initial enumeration

    for cycle in 0..3 {
        assert_eq!(
            e.tree().get(srv).unwrap().watch_demand(),
            1,
            "cycle {cycle}: steady state is exactly the parent edge (+1)",
        );

        let _ = lose_terminus(&mut e, pid, terminus);
        assert!(active_empty(&e, pid), "cycle {cycle}: terminus lost");

        // Recover.
        let _ = e.step(
            Input::FsEvent {
                resource: srv,
                event: FsEvent::StructureChanged,
            },
            Instant::now(),
        );
        assert_eq!(
            e.tree().get(srv).unwrap().watch_demand(),
            2,
            "cycle {cycle}: +2 overlap during recovery, never +3",
        );
        respond_ok(&mut e, pid, &[("app", EntryKind::Dir, 10 + cycle)]);

        let q = e.promoters().get(pid).expect("promoter alive");
        let PromoterState::Active { proxies, .. } = q.state() else {
            panic!("cycle {cycle}: expected Active, got {:?}", q.state());
        };
        terminus = *proxies.keys().next().expect("terminus re-materialised");
        assert_eq!(
            q.prefix_parent(),
            Some(srv),
            "cycle {cycle}: prefix_parent stable",
        );
    }

    assert_eq!(
        e.tree().get(srv).unwrap().watch_demand(),
        1,
        "after 3 full cycles the parent-edge refcount is still exactly +1",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---------------------------------------------------------------------
// Scenario 8 — reap_promoter releases the preserved prefix_parent contribution; the parent slot
// reaps if otherwise unheld.
// ---------------------------------------------------------------------

#[test]
fn reap_promoter_releases_prefix_parent_contribution() {
    let mut e = Engine::new();
    let (pid, srv, _srv_app) = attach_active(&mut e);
    respond_ok(&mut e, pid, &[]);

    assert!(
        e.tree()
            .get(srv)
            .unwrap()
            .contributions()
            .contains_key(&ContribKey::PromoterPrefixParent(pid)),
        "parent-edge contribution present pre-reap",
    );

    let reap_diff = specter_core::WatchRegistryDiff {
        promoters: specter_core::PromoterRegistryDiff {
            removed: vec![CompactString::from("recover")],
            ..Default::default()
        },
        ..Default::default()
    };
    let reap_out = e.step(Input::ConfigDiff(reap_diff), Instant::now());

    assert!(
        e.promoters().get(pid).is_none(),
        "Promoter removed from the registry",
    );
    assert!(
        reap_out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterReaped { promoter } if *promoter == pid,
        )),
        "PromoterReaped emitted",
    );
    assert!(
        e.tree().get(srv).is_none_or(|r| !r
            .contributions()
            .contains_key(&ContribKey::PromoterPrefixParent(pid))
            && r.watch_demand() == 0),
        "PromoterPrefixParent released; /srv reaped or back to zero watch_demand",
    );
}

// ---------------------------------------------------------------------
// Scenario 6 — all-dynamic: terminus loss + the promoted anchor's own anchor-terminal reaps the
// dynamic Sub's Profile; recovery then mints a genuinely fresh Sub (the derived gate is false —
// nothing attached).
// ---------------------------------------------------------------------

#[test]
fn all_dynamic_recovery_remints_after_profile_reap() {
    let mut e = Engine::new();
    let (pid, srv, srv_app) = attach_active(&mut e);

    // Promote a dynamic Sub at /srv/app/foo.
    respond_ok(&mut e, pid, &[("foo", EntryKind::Dir, 20)]);
    let first = dynamic_subs_of(&e, pid);
    assert_eq!(first.len(), 1, "one dynamic Sub minted");
    let (foo_anchor, first_sid) = first.into_iter().next().unwrap();

    // Anchor-terminal at /srv/app/foo: all-dynamic ⇒ Profile reaped, dynamic Sub detached
    // (DynamicSubReaped).
    let at_out = e.step(
        Input::FsEvent {
            resource: foo_anchor,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    assert!(
        e.subs().get(first_sid).is_none(),
        "all-dynamic anchor-terminal reaps the dynamic Sub",
    );
    assert!(
        at_out
            .diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::DynamicSubReaped { .. },)),
        "DynamicSubReaped emitted on the all-dynamic teardown",
    );

    // Terminus loss + event-gated recovery.
    let _ = lose_terminus(&mut e, pid, srv_app);
    assert!(active_empty(&e, pid));
    assert_eq!(
        e.promoters().get(pid).unwrap().prefix_parent(),
        Some(srv),
        "prefix_parent preserved through the all-dynamic teardown + terminus loss",
    );

    let _ = e.step(
        Input::FsEvent {
            resource: srv,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    respond_ok(&mut e, pid, &[("app", EntryKind::Dir, 21)]); // descent → enter_active
    respond_ok(&mut e, pid, &[("foo", EntryKind::Dir, 22)]); // re-enumerate

    let again = dynamic_subs_of(&e, pid);
    assert_eq!(
        again.len(),
        1,
        "recovery re-enumeration mints exactly one Sub",
    );
    let new_sid = *again.values().next().unwrap();
    assert_ne!(
        new_sid, first_sid,
        "the Profile was reaped, so this is a genuinely fresh dynamic Sub",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---------------------------------------------------------------------
// Scenario 5 — mixed co-resident: a static Sub keeps the promoted anchor's Profile alive across
// terminus loss, so recovery re-enumeration's derived gate finds the still-attached dynamic Sub and
// does NOT mint a duplicate.
// ---------------------------------------------------------------------

#[test]
fn mixed_resident_recovery_does_not_duplicate_dynamic_sub() {
    let mut e = Engine::new();
    let (pid, srv, srv_app) = attach_active(&mut e);

    respond_ok(&mut e, pid, &[("foo", EntryKind::Dir, 30)]);
    let promoted = dynamic_subs_of(&e, pid);
    assert_eq!(promoted.len(), 1);
    let (foo_anchor, dyn_sid) = promoted.into_iter().next().unwrap();
    let profile = e.subs().get(dyn_sid).expect("dynamic Sub alive").profile();

    // A static Sub co-resident at the same anchor joins the Profile via dedup — it keeps the
    // Profile (and hence the dynamic Sub) alive across terminus loss.
    let static_req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(foo_anchor),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        params: SubParams {
            name: "static-coresident".into(),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
            template: None,
            source_discovery: None,
        },
    };
    let s_out = e.step(Input::AttachSub(static_req), Instant::now());
    let static_sid =
        specter_core::testkit::first_attached_sub(&s_out).expect("static attach succeeded");
    assert_eq!(
        e.subs().get(static_sid).unwrap().profile(),
        profile,
        "static Sub dedups onto the dynamic Sub's Profile",
    );

    // Terminus loss — the dynamic Sub is NOT reaped here (it reaps only via its own
    // anchor-terminal, and the static Sub pins the Profile).
    let _ = lose_terminus(&mut e, pid, srv_app);
    assert!(active_empty(&e, pid));
    assert!(
        e.subs().get(dyn_sid).is_some(),
        "dynamic Sub still attached across terminus loss (Profile pinned by static Sub)",
    );

    // Recover + re-enumerate the same match.
    let _ = e.step(
        Input::FsEvent {
            resource: srv,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    respond_ok(&mut e, pid, &[("app", EntryKind::Dir, 31)]); // descent → enter_active
    respond_ok(&mut e, pid, &[("foo", EntryKind::Dir, 32)]); // re-enumerate same match

    let after = dynamic_subs_of(&e, pid);
    assert_eq!(
        after.len(),
        1,
        "derived gate sees the still-attached dynamic Sub ⇒ no duplicate mint",
    );
    assert_eq!(
        *after.values().next().unwrap(),
        dyn_sid,
        "the original dynamic Sub id is unchanged (idempotent re-enumeration)",
    );
    let _ = e.cancel_all_in_flight_probes();
}
