//! Inline tests for `engine::promoter`. Compose `Engine` with synthetic
//! `Input::ProbeResponse` injection — the same pattern `descent_tests.rs`
//! uses to exercise descent state machines without involving a real
//! Sensor.
//!
//! Coverage focuses on the load-bearing transitions:
//! - `attach_promoter_inner`'s two materialisation arms (immediate
//!   `Active` vs `PrefixPending`).
//! - `enter_active`'s 5a → 5b carrier-preservation walkthrough.
//! - `register_proxy` idempotence ([H-5]).
//! - `dispatch_descent_ok` advance + materialise (Promoter owner).
//! - `dispatch_promoter_enumeration_ok` forward pass (sub-proxy
//!   registration + final-position promotion).
//! - `try_promote` dedup.
//! - Event routing: `on_promoter_proxy_event` enqueue + dispatch +
//!   stale-event diagnostic.
//! - `dispatch_descent_vanished` rewind (Promoter owner).

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::single_match_else,
    clippy::too_many_lines
)]

use crate::Engine;
use compact_str::CompactString;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, AnchorClaim, ChildEntry, ClassSet, Diagnostic, DirChild, DirMeta, DirSnapshot,
    EffectScope, EntryKind, FS_ROOT_SEGMENT, FsEvent, FsIdentity, Input, LeafEntry, PatternSpec,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileIdentity, PromoterAttachRequest,
    PromoterId, PromoterState, ResourceId, ResourceKind, ResourceRole, ScanConfig, SubAttachAnchor,
    SubAttachRequest, SubId, SubParams,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn cfg() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

/// Build a `PromoterAttachRequest` with a freshly parsed `PatternSpec`.
fn req_for(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.to_owned(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        identity: ProfileIdentity {
            config: cfg(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        settle: SETTLE,
        program: empty_program(),
        scope: EffectScope::SubtreeRoot,
        log_output: false,
    }
}

/// Build an `Arc<DirSnapshot>` with the supplied children. The walker
/// speaks paths; engine identity stays engine-side, so the snapshot
/// carries pure content.
fn dir_snap_at(children: &[(&str, EntryKind, u64)]) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children.iter().copied() {
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

/// Resolve a `/var/log` chain (or any existing path) under the FS-root.
/// Helper for setup blocks that pre-place a Dir on the Tree.
fn ensure_dir(e: &mut Engine, segments: &[&str]) -> ResourceId {
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

/// Read the descent probe target-path from a probe-ops list (the latest
/// outstanding descent/subtree probe in emission order). The wire is
/// path-only; tests compare via `e.tree().path_of(<id>)`.
fn last_probe_path(out: &specter_core::StepOutput) -> Option<std::path::PathBuf> {
    out.probe_ops.iter().rev().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.target_path().to_path_buf()),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Read the `pending_enumerations` BTreeSet from a Promoter — convenience
/// for tests asserting on queue contents.
fn pending_enumerations(e: &Engine, pid: PromoterId) -> Vec<ResourceId> {
    e.promoters
        .get(pid)
        .map(|q| q.pending_enumerations.iter().copied().collect())
        .unwrap_or_default()
}

/// Read the `Active { proxies }` map from a Promoter (panicking if the
/// state is `PrefixPending`).
fn active_proxies(e: &Engine, pid: PromoterId) -> BTreeMap<ResourceId, specter_core::ProxyState> {
    match e.promoters.get(pid).map(|q| &q.state) {
        Some(PromoterState::Active { proxies, .. }) => proxies.clone(),
        s => panic!("expected Active state, got {s:?}"),
    }
}

#[test]
fn attach_immediate_active_at_existing_prefix() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    assert_ne!(pid, PromoterId::default(), "promoter id minted");

    // State: Active{proxies: {/var/log → idx=lpl}} where lpl=3.
    let proxies = active_proxies(&e, pid);
    assert_eq!(proxies.len(), 1, "single proxy registered at prefix");
    let (proxy, ps) = proxies.iter().next().unwrap();
    assert_eq!(*proxy, var_log, "proxy at the materialised prefix");
    assert_eq!(
        ps.pattern_component_index, 3,
        "first proxy carries pattern.literal_prefix_len()",
    );

    // Watch_demand on /var/log: User Profile-less Dir starts at 0; the
    // proxy registration bumps to 1 with STRUCTURE.
    assert_eq!(e.tree().get(var_log).unwrap().watch_demand(), 1);
    assert!(
        e.tree()
            .get(var_log)
            .unwrap()
            .events_union()
            .intersects(ClassSet::STRUCTURE),
        "STRUCTURE bit set on the proxy slot",
    );

    // Back-ref: tree.get(/var/log).proxy_promoters contains pid.
    assert_eq!(
        e.tree().get(var_log).unwrap().proxy_promoters(),
        &[pid],
        "back-ref points at the Promoter",
    );

    // Lifecycle diagnostic emitted.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterAttached { promoter, .. } if *promoter == pid,
        )),
        "PromoterAttached emitted",
    );

    // Initial enumeration probe in flight at the proxy.
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some(),
        "enumeration probe in flight",
    );
    assert_eq!(
        last_probe_path(&out).as_deref(),
        e.tree().path_of(var_log).as_deref()
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn attach_pending_when_literal_prefix_missing() {
    let mut e = Engine::new();
    // No /var/log on disk; only the FS-root bootstrap will create /.
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    assert_ne!(pid, PromoterId::default());

    // State: PrefixPending(d). d.current_prefix() == FS-root slot;
    // remaining_components = ["var", "log"].
    let q = e.promoters.get(pid).expect("promoter registered");
    let PromoterState::PrefixPending(d) = &q.state else {
        panic!("expected PrefixPending, got {:?}", q.state);
    };
    let fs_root = e
        .tree()
        .lookup(None, FS_ROOT_SEGMENT)
        .expect("FS-root exists");
    assert_eq!(d.current_prefix(), fs_root, "descent at FS-root");
    assert_eq!(
        d.remaining_components().iter().cloned().collect::<Vec<_>>(),
        vec![CompactString::from("var"), CompactString::from("log")],
        "two literal segments to descend",
    );

    // Watch_demand on FS-root bumped (STRUCTURE).
    assert_eq!(e.tree().get(fs_root).unwrap().watch_demand(), 1);

    // Descent probe in flight at FS-root.
    assert_eq!(
        last_probe_path(&out).as_deref(),
        e.tree().path_of(fs_root).as_deref()
    );
    assert!(e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some());
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn descent_advances_one_segment_on_partial_response() {
    let mut e = Engine::new();
    // /var exists; /var/log doesn't.
    let var = ensure_dir(&mut e, &["var"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    assert_eq!(
        last_probe_path(&out).as_deref(),
        e.tree().path_of(var).as_deref(),
        "first probe at /var"
    );

    // Inject SubtreeOk: /var contains "log" as a Dir. Descent should
    // advance to /var/log; remaining = [].
    let snap = dir_snap_at(&[("log", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // Last literal segment ("log") materialised → enter_active. State
    // is Active, with the new /var/log slot as the first proxy.
    let proxies = active_proxies(&e, pid);
    assert_eq!(proxies.len(), 1);
    let new_proxy = *proxies.keys().next().unwrap();
    assert_eq!(
        e.tree().name(new_proxy),
        Some("log"),
        "proxy at the freshly-materialised /var/log slot",
    );
    assert_eq!(
        e.tree().get(new_proxy).unwrap().role,
        ResourceRole::User,
        "[S-8] proxy slot is User-roled",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn enumeration_ok_promotes_final_match() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    assert_eq!(
        last_probe_path(&out).as_deref(),
        e.tree().path_of(var_log).as_deref()
    );

    // Inject enumeration response listing two files: one matches *.log,
    // one doesn't.
    let snap = dir_snap_at(&[
        ("foo.log", EntryKind::File, 1),
        ("bar.txt", EntryKind::File, 2),
    ]);
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // dynamic_subs contains the matched anchor; bar.txt is absent.
    let promoted_resources: Vec<ResourceId> = e
        .promoters
        .get(pid)
        .unwrap()
        .dynamic_subs
        .keys()
        .copied()
        .collect();
    assert_eq!(promoted_resources.len(), 1);
    let promoted_path = e
        .tree()
        .path_of(promoted_resources[0])
        .expect("anchor path resolves");
    assert_eq!(
        promoted_path.to_string_lossy(),
        "/var/log/foo.log",
        "single dynamic Sub at the matched path",
    );

    // PromotionKindObserved diagnostic emitted.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromotionKindObserved { promoter, .. } if *promoter == pid,
        )),
        "PromotionKindObserved diagnostic emitted",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn enumeration_ok_registers_subproxy_for_intermediate_glob() {
    let mut e = Engine::new();
    // Pattern /srv/*/site — first proxy at /srv (lpl=2).
    let srv = ensure_dir(&mut e, &["srv"]);

    let out = e.step(
        Input::AttachPromoter(req_for("sites", "/srv/*/site")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Inject /srv listing: two child Dirs ("alpha", "beta") and a stray
    // File ("noisy.cfg") that the glob would match but should be
    // skipped at non-final position (only Dir matches descend).
    let snap = dir_snap_at(&[
        ("alpha", EntryKind::Dir, 1),
        ("beta", EntryKind::Dir, 2),
        ("noisy.cfg", EntryKind::File, 3),
    ]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // proxies should now include the original /srv plus two sub-proxies
    // at /srv/alpha and /srv/beta. Each sub-proxy carries
    // pattern_component_index = 3 (idx of `site` literal).
    let proxies = active_proxies(&e, pid);
    assert_eq!(proxies.len(), 3, "/srv + alpha + beta = 3 proxies");
    let alpha = e
        .tree()
        .lookup(Some(srv), "alpha")
        .expect("alpha materialised");
    let beta = e
        .tree()
        .lookup(Some(srv), "beta")
        .expect("beta materialised");
    let alpha_state = proxies.get(&alpha).expect("alpha registered as proxy");
    let beta_state = proxies.get(&beta).expect("beta registered as proxy");
    assert_eq!(alpha_state.pattern_component_index, 3);
    assert_eq!(beta_state.pattern_component_index, 3);

    // No promotion: /srv/alpha/site and /srv/beta/site enumerations are
    // queued, not yet probed. dynamic_subs is empty.
    assert!(
        e.promoters.get(pid).unwrap().dynamic_subs.is_empty(),
        "no promotions at intermediate level",
    );

    // The non-Dir File "noisy.cfg" was skipped at non-final position.
    assert!(
        e.tree().lookup(Some(srv), "noisy.cfg").is_none()
            || !proxies.contains_key(&e.tree().lookup(Some(srv), "noisy.cfg").unwrap()),
        "Leaf children are not registered as sub-proxies at non-final position",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn try_promote_is_idempotent_on_repeated_match() {
    // Two enumeration cycles (e.g., a parent's StructureChanged refire)
    // for the same path do NOT mint two dynamic Subs. The contains-check
    // dedup gate fires at try_promote.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Cycle 1: foo.log matches. Promotion mints SubA.
    let snap = dir_snap_at(&[("foo.log", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let dynamic_count_after_cycle_1 = e.promoters.get(pid).unwrap().dynamic_subs.len();
    assert_eq!(dynamic_count_after_cycle_1, 1);

    // Re-trigger enumeration via FsEvent at the proxy. Inject the same
    // snapshot — same entry, same path. dynamic_subs should still have
    // exactly one entry (no duplicate Sub minted).
    let _ = e.step(
        Input::FsEvent {
            resource: var_log,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    let corr2 = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap2 = dir_snap_at(&[("foo.log", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr2,
            outcome: ProbeOutcome::SubtreeOk(snap2),
        }),
        Instant::now(),
    );
    assert_eq!(
        e.promoters.get(pid).unwrap().dynamic_subs.len(),
        1,
        "dedup gate prevents re-promotion of the same path",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn proxy_event_enqueues_and_dispatches() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    // Drain the initial enumeration: respond Ok with empty entries.
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap = dir_snap_at(&[]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "probe slot disarmed after empty enumeration",
    );
    assert!(
        pending_enumerations(&e, pid).is_empty(),
        "queue empty after dispatch_next_enumeration drains the initial entry",
    );

    // FsEvent at the proxy slot triggers a fresh enumeration probe.
    let out = e.step(
        Input::FsEvent {
            resource: var_log,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some(),
        "fresh enumeration probe in flight",
    );
    assert_eq!(
        last_probe_path(&out).as_deref(),
        e.tree().path_of(var_log).as_deref(),
        "probe target = the proxy that received the event",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn proxy_event_for_unregistered_promoter_emits_stale_diagnostic() {
    // Register a phantom back-ref via the typed `insert_proxy_promoter`
    // mutator for a Promoter id that never registered at that slot
    // through the engine — the on_promoter_proxy_event safety check
    // must catch this and emit PromoterProxyStaleEvent rather than
    // enqueue garbage.
    use specter_core::PromoterId;
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("phantom", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    // Synthesise a Promoter id that's NOT in the registry; register it
    // onto the back-ref via the typed mutator.
    let phantom = PromoterId::default();

    // Bump watch_demand so the on_fs_event head guard doesn't drop the
    // event with EventOnUnwatchedResource.
    let mut out = specter_core::StepOutput::default();
    crate::refcounts::add_watch(
        e.tree_mut(),
        r,
        specter_core::ContribKey::PromoterProxy(phantom),
        ClassSet::STRUCTURE,
        &mut out,
    );
    e.tree_mut()
        .get_mut(r)
        .unwrap()
        .insert_proxy_promoter(phantom);

    let out = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterProxyStaleEvent { promoter, resource } if *promoter == phantom && *resource == r,
        )),
        "PromoterProxyStaleEvent emitted for back-ref that doesn't resolve to a live proxy",
    );
}

#[test]
fn descent_vanished_rewinds_to_parent() {
    let mut e = Engine::new();
    // /var exists; descent target /var/log doesn't.
    let var = ensure_dir(&mut e, &["var"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr1 = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    // Sanity: descent is at /var with remaining=["log"].
    let q = e.promoters.get(pid).unwrap();
    let PromoterState::PrefixPending(d) = &q.state else {
        panic!("expected PrefixPending pre-vanish");
    };
    assert_eq!(d.current_prefix(), var, "descent at /var pre-vanish");

    // Inject Vanished — /var has been removed mid-descent.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr1,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );

    // PromoterDescentVanished diagnostic emitted.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterDescentVanished { promoter, .. } if *promoter == pid,
        )),
        "vanished diagnostic emitted",
    );

    // Descent rewinds to /var's parent (the FS-root). State stays
    // PrefixPending; current_prefix is now FS-root; remaining starts
    // with "var" (prepended) and ends with "log" (the original
    // remaining segment).
    let q = e.promoters.get(pid).expect("Promoter still alive");
    let PromoterState::PrefixPending(d) = &q.state else {
        panic!("expected PrefixPending after rewind");
    };
    let fs_root = e.tree().lookup(None, FS_ROOT_SEGMENT).unwrap();
    assert_eq!(d.current_prefix(), fs_root, "rewind landed at FS-root");
    assert_eq!(
        d.remaining_components().iter().cloned().collect::<Vec<_>>(),
        vec![CompactString::from("var"), CompactString::from("log")],
        "vanished prefix's segment prepended; original remaining preserved",
    );

    // Fresh probe in flight at the parent.
    let corr2 = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("fresh probe minted");
    assert_ne!(corr1, corr2, "post-rewind correlation differs from pre");
    assert_eq!(
        last_probe_path(&out).as_deref(),
        e.tree().path_of(fs_root).as_deref()
    );

    // The vanished /var slot was vacated. Vacate clears
    // watch_demand and kind; the slot itself is retained while it
    // still has children (the still-existing /var/log
    // DescentScaffold the original descent created). Asserting on
    // vacate's observable effect:
    assert_eq!(
        e.tree().get(var).unwrap().watch_demand(),
        0,
        "vanished prefix had its watch_demand zeroed (vacate)",
    );
    let _ = e.cancel_all_in_flight_probes();
}

// ---- §A regression: PrefixPending events at the prefix re-trigger descent.
//
// Pre-unification, `classify_event_carriers` only walked Profiles —
// Promoter `PrefixPending` descents were invisible to event dispatch, so
// any FsEvent at the prefix routed to `EventNoConsumer` and the Promoter
// could be permanently stuck waiting for a segment it never re-probed.
// These four tests pin the post-unification dispatch — `on_descent_event`
// is now owner-polymorphic and `EventCarriers.descents` carries
// `ProbeOwner::Promoter(_)`s alongside Profiles.

/// Drained-probe + StructureChanged at the prefix → fresh descent probe.
/// Mirror of `descent_tests.rs::descent_event_at_prefix_emits_fresh_probe`
/// for the Promoter side. The §A surface — without this dispatch the
/// Promoter would never re-probe after the next segment first appeared.
#[test]
fn prefix_pending_event_at_prefix_emits_fresh_descent_probe() {
    let mut e = Engine::new();
    let var = ensure_dir(&mut e, &["var"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    // Sanity: descent at /var with remaining=["log"], probe in flight.
    let corr = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("descent probe in flight after attach");

    // Drain the in-flight probe with an empty response (segment not
    // yet present; descent stays PrefixPending awaiting the next event).
    let snap = dir_snap_at(&[]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "probe slot disarmed after empty response",
    );
    assert!(
        matches!(
            e.promoters.get(pid).map(|q| &q.state),
            Some(PromoterState::PrefixPending(_)),
        ),
        "still PrefixPending — segment not yet present",
    );

    // FsEvent at /var (the prefix) → fresh descent probe via
    // `on_descent_event(ProbeOwner::Promoter(pid))`.
    let out = e.step(
        Input::FsEvent {
            resource: var,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    let probe_for_pid = out.probe_ops.iter().any(|op| {
        matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Promoter(pid))
    });
    assert!(probe_for_pid, "fresh descent probe minted for Promoter");
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some(),
        "probe slot re-armed",
    );
    // No EventNoConsumer diagnostic — the §A bug surface.
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EventNoConsumer { .. })),
        "Promoter PrefixPending consumed the event (no EventNoConsumer): {:?}",
        out.diagnostics,
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// I5 guard: while a descent probe is in flight, an FsEvent at the
/// prefix drops without minting a second probe (the in-flight probe
/// will pick up the new segment in its response). Mirror of
/// `descent_tests.rs::descent_event_during_in_flight_probe_drops`.
#[test]
fn prefix_pending_event_during_in_flight_probe_drops() {
    let mut e = Engine::new();
    let var = ensure_dir(&mut e, &["var"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    // Probe in flight from setup.
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some(),
        "descent probe in flight (precondition for I5 guard)",
    );

    let out = e.step(
        Input::FsEvent {
            resource: var,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    let descent_probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Promoter(pid)))
        .count();
    assert_eq!(
        descent_probes, 0,
        "I5: no second probe minted while one is in flight",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Terminal events at the prefix (Removed / Renamed / Revoked) must
/// also re-trigger descent — the §A bug surface is broader than the
/// non-terminal `StructureChanged` case the original report focused
/// on. Without dispatch, a `mv /var /var.old` mid-descent would
/// strand the Promoter with a dangling watch on a non-existent path
/// (the next probe response would clean up via `Vanished` rewind, but
/// only if the dispatch fires).
#[test]
fn prefix_pending_terminal_event_at_prefix_emits_fresh_descent_probe() {
    let mut e = Engine::new();
    let var = ensure_dir(&mut e, &["var"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    // Drain the in-flight probe.
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap = dir_snap_at(&[]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // Inject a terminal event at the prefix — `Removed` (e.g.,
    // `mv /var /var.old`).
    let out = e.step(
        Input::FsEvent {
            resource: var,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    let probe_for_pid = out.probe_ops.iter().any(|op| {
        matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Promoter(pid))
    });
    assert!(
        probe_for_pid,
        "fresh descent probe minted for terminal event at prefix",
    );
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EventNoConsumer { .. })),
        "terminal event consumed by descent dispatch (no EventNoConsumer)",
    );
    let _ = e.cancel_all_in_flight_probes();
}

/// Failed descent + event-driven retry: after a `Failed { errno }`
/// response retains `PrefixPending` state, the next event at the
/// prefix must re-trigger descent. Without unified dispatch this
/// retry path was unreachable — `Failed` is documented as "await
/// next event at prefix" (Promoter-side `dispatch_descent_failed`
/// arm) but no event dispatch existed for the Promoter side
/// pre-unification, so failed Promoter descents would stall
/// permanently.
#[test]
fn prefix_pending_event_after_failed_descent_emits_fresh_descent_probe() {
    let mut e = Engine::new();
    let var = ensure_dir(&mut e, &["var"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Inject Failed (e.g. transient EACCES).
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::Failed { errno: 13 },
        }),
        Instant::now(),
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "probe slot disarmed after Failed",
    );
    assert!(
        matches!(
            e.promoters.get(pid).map(|q| &q.state),
            Some(PromoterState::PrefixPending(_)),
        ),
        "PrefixPending retained after Failed",
    );

    // Event at the prefix → retry descent.
    let out = e.step(
        Input::FsEvent {
            resource: var,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );

    let probe_for_pid = out.probe_ops.iter().any(|op| {
        matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Promoter(pid))
    });
    assert!(
        probe_for_pid,
        "fresh descent probe minted post-Failed via event dispatch",
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some(),
        "probe slot re-armed",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn register_proxy_is_idempotent_on_re_registration() {
    // Re-registering the same proxy at the same idx must NOT
    // double-bump watch_demand or duplicate the queue entry. [H-5]
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let watch_demand_after_attach = e.tree().get(var_log).unwrap().watch_demand();
    let queue_after_attach = pending_enumerations(&e, pid).len();

    // Re-register: directly invoke the (pub(crate)) helper.
    let mut out = specter_core::StepOutput::default();
    e.register_proxy(pid, var_log, 3, &mut out);

    assert_eq!(
        e.tree().get(var_log).unwrap().watch_demand(),
        watch_demand_after_attach,
        "watch_demand unchanged on re-registration",
    );
    assert_eq!(
        pending_enumerations(&e, pid).len(),
        queue_after_attach,
        "pending_enumerations unchanged on re-registration",
    );
    assert_eq!(
        e.tree().get(var_log).unwrap().proxy_promoters(),
        &[pid],
        "back-ref unchanged (single entry) on re-registration",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn register_then_release_proxy_clears_both_halves_of_the_join() {
    // The proxy join is two-sided: `Resource.proxy_promoters` holds
    // the back-ref to the Promoter, and `Promoter.state.proxies`
    // holds the forward entry to the Resource. Releasing a registered
    // proxy must tear down BOTH halves — neither a dangling back-ref
    // on the slot nor a stale forward entry on the Promoter may
    // survive the round-trip.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");

    // `attach` materialised an immediate proxy at the existing prefix:
    // both halves of the join are present.
    assert!(
        active_proxies(&e, pid).contains_key(&var_log),
        "forward entry present in Promoter.proxies pre-release",
    );
    assert!(
        e.tree()
            .get(var_log)
            .unwrap()
            .proxy_promoters()
            .contains(&pid),
        "back-ref present on the slot pre-release",
    );

    // `attach` left an enumeration probe in flight at the proxy.
    // `unregister_proxy`'s cancel-first contract requires the
    // owner probe be disarmed before release — the same ordering
    // every production release path observes.
    let mut out = specter_core::StepOutput::default();
    e.cancel_owner_probe(ProbeOwner::Promoter(pid), &mut out);

    // Release the proxy via the documented inverse of `register_proxy`.
    let mut out = specter_core::StepOutput::default();
    e.unregister_proxy(pid, var_log, &mut out);

    // Forward half: the Promoter's `proxies` map no longer carries the
    // resource.
    assert!(
        !active_proxies(&e, pid).contains_key(&var_log),
        "forward entry cleared from Promoter.proxies after release",
    );
    // Back-ref half: the slot's `proxy_promoters` no longer points at
    // the Promoter. The User-roled, promoter-only slot reaps once the
    // back-ref drops, so the slot may be absent entirely — either way
    // no dangling back-ref survives.
    assert!(
        e.tree()
            .get(var_log)
            .is_none_or(|r| !r.proxy_promoters().contains(&pid)),
        "back-ref cleared from the slot after release (or slot reaped)",
    );
}

// ---- promoter enumeration slot lifecycle ----

#[test]
fn dispatch_next_enumeration_records_pending_target() {
    // The in-flight proxy resource is tagged on the state-resident
    // enumeration slot at probe-emit time so `Vanished` / `Failed`
    // responses (which carry no payload) can identify the proxy.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    // Initial enumeration was dispatched by `enter_active`; the
    // enumeration slot's tag carries the in-flight proxy resource.
    let target = e
        .promoters
        .get(pid)
        .expect("promoter alive")
        .state
        .enumeration_target();
    assert_eq!(
        target,
        Some(var_log),
        "enumeration slot tag carries the in-flight proxy target (got {target:?})",
    );
    // The probe correlation is also live.
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some(),
        "probe correlation in flight alongside the enumeration slot tag",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn pending_enumeration_target_clears_on_response() {
    // The enumeration slot disarms on response — both the correlation
    // and the proxy-target tag go away atomically (single `take_probe`
    // / `disarm` on the state-resident slot).
    let mut e = Engine::new();
    let _var_log = ensure_dir(&mut e, &["var", "log"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    let snap = dir_snap_at(&[]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    assert!(
        e.promoters
            .get(pid)
            .expect("promoter alive")
            .state
            .enumeration_target()
            .is_none(),
        "enumeration slot disarmed after response",
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "probe slot cleared after response",
    );
}

#[test]
fn cancel_owner_probe_clears_promoter_enumeration_slot() {
    // `cancel_owner_probe` is the canonical disarm-on-cancel path. For
    // a Promoter owner with an armed enumeration slot it disarms the
    // slot and emits a Cancel op.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let pre_target = e
        .promoters
        .get(pid)
        .expect("promoter alive")
        .state
        .enumeration_target();
    assert_eq!(
        pre_target,
        Some(var_log),
        "pre-cancel enumeration slot targets /var/log (got {pre_target:?})",
    );

    let mut out = specter_core::StepOutput::default();
    e.cancel_owner_probe(ProbeOwner::Promoter(pid), &mut out);

    assert!(
        e.promoters
            .get(pid)
            .expect("promoter alive")
            .state
            .enumeration_target()
            .is_none(),
        "enumeration slot disarmed by cancel_owner_probe",
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "probe slot cleared by cancel_owner_probe",
    );
    assert!(
        out.probe_ops.iter().any(|op| matches!(
            op,
            ProbeOp::Cancel { owner: ProbeOwner::Promoter(p) } if *p == pid,
        )),
        "Cancel op emitted for the armed probe slot",
    );
}

// ---- dispatch_promoter_enumeration_vanished cascade ----

#[test]
fn enumeration_vanished_unregisters_proxy_and_emits_diagnostic() {
    // Pattern /var/log/*.log; proxy at /var/log. Inject a Vanished
    // response — the proxy's directory is gone. The dispatcher
    // emits PromoterEnumerationVanished {proxy: /var/log} and
    // cascades unregister_proxy_subtree(/var/log) which clears the
    // proxy's back-ref, watch_demand contribution, and removes it
    // from `Promoter.state.proxies`.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    assert!(
        e.tree()
            .get(var_log)
            .unwrap()
            .proxy_promoters()
            .contains(&pid),
        "back-ref present pre-vanish",
    );
    assert!(
        e.tree().get(var_log).unwrap().watch_demand() >= 1,
        "watch_demand carries the proxy contribution pre-vanish",
    );

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );

    // Diagnostic carries the vanished proxy.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterEnumerationVanished { promoter, proxy }
                if *promoter == pid && *proxy == var_log,
        )),
        "PromoterEnumerationVanished emitted with the proxy: {:?}",
        out.diagnostics,
    );
    // Old descent-side variant is NOT emitted by the enumeration arm.
    assert!(
        !out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PromoterDescentVanished { .. },)),
        "old descent variant must not fire on enumeration vanish",
    );

    // Proxy unregistered: removed from Active.proxies. Whether the
    // Tree slot survives depends on its retention signals — for a
    // User-roled slot with no children/profiles/other-promoters,
    // `try_reap` collects it; we only assert the Promoter-side
    // invariant (back-ref absent, proxies map empty).
    let q = e.promoters.get(pid).expect("promoter alive");
    let proxies = match &q.state {
        PromoterState::Active { proxies, .. } => proxies,
        PromoterState::PrefixPending(_) => panic!("expected Active, got PrefixPending"),
    };
    assert!(
        !proxies.contains_key(&var_log),
        "proxy removed from Active.proxies",
    );
    let back_ref_intact = e
        .tree()
        .get(var_log)
        .is_some_and(|r| r.proxy_promoters().contains(&pid));
    assert!(!back_ref_intact, "back-ref cleared (or slot reaped)");
}

#[test]
fn enumeration_vanished_cascades_subproxies() {
    // Pattern /srv/*/site — sub-proxy at /srv/alpha after a forward
    // pass. A Vanished at /srv (impossible in production but lets us
    // unit-test the cascade scope) clears /srv AND /srv/alpha.
    let mut e = Engine::new();
    let srv = ensure_dir(&mut e, &["srv"]);
    let out = e.step(
        Input::AttachPromoter(req_for("sites", "/srv/*/site")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr1 = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Forward pass at /srv: alpha (Dir) → register sub-proxy at /srv/alpha.
    let snap = dir_snap_at(&[("alpha", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr1,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // Two proxies registered: /srv and /srv/alpha.
    assert_eq!(active_proxies(&e, pid).len(), 2);

    // Drain the queued enumeration at /srv/alpha by responding empty.
    let alpha = e.tree().lookup(Some(srv), "alpha").expect("alpha present");
    let corr2 = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap = dir_snap_at(&[]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr2,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // Trigger an enumeration at /srv via FsEvent and Vanish it.
    let _ = e.step(
        Input::FsEvent {
            resource: srv,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    let corr3 = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("re-enumeration probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr3,
            outcome: ProbeOutcome::Vanished,
        }),
        Instant::now(),
    );

    // Both /srv and /srv/alpha are unregistered.
    let proxies = active_proxies(&e, pid);
    assert!(!proxies.contains_key(&srv), "/srv unregistered");
    assert!(
        !proxies.contains_key(&alpha),
        "/srv/alpha (descendant proxy) cascaded",
    );
}

// ---- dispatch_promoter_enumeration_failed retains state ----

#[test]
fn enumeration_failed_retains_proxy_state_with_diagnostic() {
    // Failed responses preserve proxy state (next event re-triggers
    // enumeration); only the diagnostic emits.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::Failed {
                errno: 13, /* EACCES */
            },
        }),
        Instant::now(),
    );

    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterEnumerationFailed { promoter, proxy, errno }
                if *promoter == pid && *proxy == var_log && *errno == 13,
        )),
        "PromoterEnumerationFailed carries promoter, proxy, and errno: {:?}",
        out.diagnostics,
    );

    // Proxy still registered and watch_demand intact.
    let proxies = active_proxies(&e, pid);
    assert!(proxies.contains_key(&var_log), "proxy retained");
    assert_eq!(
        e.tree().get(var_log).unwrap().proxy_promoters(),
        &[pid],
        "back-ref intact",
    );
}

// ---- reap_promoter ----

#[test]
fn reap_promoter_active_with_proxy_unregisters_and_removes() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    assert_eq!(e.tree().get(var_log).unwrap().watch_demand(), 1);

    let out = e.reap_promoter(pid);

    // PromoterReaped emitted.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterReaped { promoter } if *promoter == pid,
        )),
        "PromoterReaped emitted: {:?}",
        out.diagnostics,
    );

    // Promoter removed from registry.
    assert!(e.promoters.get(pid).is_none(), "Promoter removed");

    // The proxy slot's contribution was released. The slot itself
    // may have been reaped (User-roled with no other anchors) — read
    // both the surviving and reaped cases through `Option`.
    let post_reap_demand = e
        .tree()
        .get(var_log)
        .map_or(0, specter_core::Resource::watch_demand);
    assert_eq!(
        post_reap_demand, 0,
        "watch_demand dropped after unregister_proxy",
    );
    let back_ref_intact = e
        .tree()
        .get(var_log)
        .is_some_and(|r| r.proxy_promoters().contains(&pid));
    assert!(!back_ref_intact, "back-ref cleared (or slot reaped)");

    // In-flight initial enumeration cancelled.
    assert!(
        out.probe_ops.iter().any(|op| matches!(
            op,
            ProbeOp::Cancel { owner: ProbeOwner::Promoter(p) } if *p == pid,
        )),
        "Cancel op emitted for in-flight enumeration",
    );
}

#[test]
fn reap_promoter_prefix_pending_releases_prefix() {
    // Pattern /var/log/*.log with /var/log absent → PrefixPending at FS-root.
    let mut e = Engine::new();
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let fs_root = e.tree().lookup(None, FS_ROOT_SEGMENT).unwrap();
    assert!(matches!(
        e.promoters.get(pid).unwrap().state,
        PromoterState::PrefixPending(_),
    ));
    assert_eq!(
        e.tree().get(fs_root).unwrap().watch_demand(),
        1,
        "FS-root carries the prefix's STRUCTURE contribution",
    );

    let out = e.reap_promoter(pid);

    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::PromoterReaped { promoter } if *promoter == pid,
        )),
        "PromoterReaped emitted",
    );
    assert!(e.promoters.get(pid).is_none(), "Promoter removed");
    assert_eq!(
        e.tree().get(fs_root).unwrap().watch_demand(),
        0,
        "FS-root contribution released",
    );
    // In-flight descent probe cancelled.
    assert!(
        out.probe_ops.iter().any(|op| matches!(
            op,
            ProbeOp::Cancel { owner: ProbeOwner::Promoter(p) } if *p == pid,
        )),
        "Cancel op emitted for in-flight descent probe",
    );
}

#[test]
fn reap_promoter_drains_dynamic_subs() {
    // Promoter with one dynamic Sub. `reap_promoter` detaches the
    // Sub and removes the (path, sub) entry from `dynamic_subs`.
    let mut e = Engine::new();
    let _var_log = ensure_dir(&mut e, &["var", "log"]);
    let out = e.step(
        Input::AttachPromoter(req_for("logs", "/var/log/*.log")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Mint a dynamic Sub at /var/log/foo.log.
    let snap = dir_snap_at(&[("foo.log", EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let dynamic_count_after_promotion = e.promoters.get(pid).unwrap().dynamic_subs.len();
    assert_eq!(dynamic_count_after_promotion, 1, "dynamic Sub minted");
    let sub_id = *e
        .promoters
        .get(pid)
        .unwrap()
        .dynamic_subs
        .values()
        .next()
        .unwrap();
    assert!(e.subs().get(sub_id).is_some(), "Sub registered");

    let _ = e.reap_promoter(pid);

    assert!(e.promoters.get(pid).is_none(), "Promoter removed");
    assert!(
        e.subs().get(sub_id).is_none(),
        "dynamic Sub detached from registry",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn reap_promoter_stale_id_is_silent_noop() {
    let mut e = Engine::new();
    let stale = PromoterId::default();
    let out = e.reap_promoter(stale);
    assert!(
        out.diagnostics.is_empty(),
        "no diagnostic on stale id: {:?}",
        out.diagnostics,
    );
    assert!(out.probe_ops.is_empty(), "no probe ops on stale id");
    assert!(out.watch_ops.is_empty(), "no watch ops on stale id");
}

#[test]
fn reap_promoter_active_with_subproxies_clears_all() {
    // Pattern /srv/*/site with one sub-proxy registered. reap_promoter
    // unregisters every proxy (including sub-proxies) and clears all
    // back-refs.
    let mut e = Engine::new();
    let srv = ensure_dir(&mut e, &["srv"]);
    let out = e.step(
        Input::AttachPromoter(req_for("sites", "/srv/*/site")),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Forward pass: alpha → sub-proxy at /srv/alpha.
    let snap = dir_snap_at(&[("alpha", EntryKind::Dir, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let alpha = e.tree().lookup(Some(srv), "alpha").expect("alpha present");
    assert_eq!(active_proxies(&e, pid).len(), 2);

    let _ = e.reap_promoter(pid);

    assert!(e.promoters.get(pid).is_none(), "Promoter removed");
    // With the Promoter as their sole claim, both `/srv` and
    // `/srv/alpha` cascade-reap when `unregister_proxy` drops the last
    // proxy contribution and back-ref at each slot. Assert the
    // back-ref is cleared (or the slot is gone) at every involved
    // resource — the production contract is "no live `proxy_promoters`
    // entry naming this Promoter survives the reap."
    for r in [srv, alpha] {
        assert!(
            e.tree()
                .get(r)
                .is_none_or(|node| !node.proxy_promoters().contains(&pid)),
            "{r:?} back-ref cleared (or slot reaped)",
        );
    }
}

// ---- recovery split (on_anchor_terminal_event dispatcher) ----

/// Pre-materialise the leaf File slot at `parent_segs/leaf` so the
/// Promoter's enumeration mints a Sub against an *existing* anchor
/// (the dynamic Sub's Profile attaches in `Idle`, not `Pending` —
/// the anchor's `watch_demand` is bumped, and a subsequent
/// FsEvent::Removed reaches `on_anchor_terminal_event`).
fn ensure_file(e: &mut Engine, parent_segs: &[&str], leaf: &str) -> ResourceId {
    let mut comps: Vec<&str> = Vec::with_capacity(parent_segs.len() + 2);
    comps.push(FS_ROOT_SEGMENT);
    comps.extend_from_slice(parent_segs);
    comps.push(leaf);
    let r = e
        .tree_mut()
        .ensure_path(&comps, ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(r, ResourceKind::File);
    r
}

/// Promote one match into a dynamic Sub against a pre-materialised
/// leaf. Returns `(promoter_id, sub_id, anchor_resource)`.
fn promote_one(
    e: &mut Engine,
    pattern: &str,
    parent_segs: &[&str],
    leaf: &str,
) -> (PromoterId, SubId, ResourceId) {
    let _parent = ensure_dir(e, parent_segs);
    let anchor_resource = ensure_file(e, parent_segs, leaf);
    let out = e.step(
        Input::AttachPromoter(req_for("test", pattern)),
        Instant::now(),
    );
    let pid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap = dir_snap_at(&[(leaf, EntryKind::File, 1)]);
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    let q = e.promoters.get(pid).expect("promoter alive");
    let sid = *q.dynamic_subs.values().next().expect("Sub minted");
    (pid, sid, anchor_resource)
}

#[test]
fn anchor_terminal_all_dynamic_reaps_profile_and_notifies_promoter() {
    // Promoter `/var/log/*.log` mints a dynamic Sub at /var/log/foo.log.
    // The Sub is the only attachment on the Profile (all_dynamic=true).
    // FsEvent::Removed at the anchor → reap Profile, notify Promoter
    // (DynamicSubReaped + PromoterReap-like teardown sequence).
    let mut e = Engine::new();
    let (pid, sid, anchor) = promote_one(&mut e, "/var/log/*.log", &["var", "log"], "foo.log");
    assert_eq!(
        e.promoters.get(pid).unwrap().dynamic_subs.len(),
        1,
        "exactly one dynamic Sub minted",
    );
    let profile_id = e.subs().get(sid).expect("Sub alive").profile;
    assert!(e.profiles.get(profile_id).is_some(), "Profile attached");
    // Sanity: the Sub carries source_promoter.
    assert_eq!(e.subs().get(sid).and_then(|s| s.source_promoter), Some(pid),);

    let out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // DynamicSubReaped emitted with the (promoter, sub, path) triple.
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::DynamicSubReaped { promoter, sub, .. }
                if *promoter == pid && *sub == sid,
        )),
        "DynamicSubReaped emitted: {:?}",
        out.diagnostics,
    );
    // ProfileReaped emitted (the all-dynamic teardown reaps the Profile).
    assert!(
        out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::ProfileReaped { profile, via: _ } if *profile == profile_id,
        )),
        "ProfileReaped emitted for the dynamic-only Profile: {:?}",
        out.diagnostics,
    );
    // Profile gone, Sub gone, Promoter dynamic_subs entry cleared.
    assert!(e.profiles.get(profile_id).is_none(), "Profile reaped");
    assert!(
        e.subs().get(sid).is_none(),
        "dynamic Sub removed from registry"
    );
    assert!(
        e.promoters.get(pid).unwrap().dynamic_subs.is_empty(),
        "dynamic_subs entry dropped",
    );
}

#[test]
fn anchor_terminal_mixed_profile_preserves_recovery() {
    // Promoter `/var/log/*.log` mints a dynamic Sub at /var/log/foo.log.
    // A static Sub at the same anchor joins via Profile dedup.
    // FsEvent::Removed at the anchor: NOT all_dynamic ⇒ falls to
    // finalize_anchor_lost (Profile lives, watch_root_parent retained).
    let mut e = Engine::new();
    let (pid, dyn_sid, anchor) = promote_one(&mut e, "/var/log/*.log", &["var", "log"], "foo.log");
    let profile_id = e.subs().get(dyn_sid).expect("Sub alive").profile;
    let static_req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(anchor),
        identity: ProfileIdentity {
            config: cfg(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        params: SubParams {
            name: String::from("static-foo"),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let attach_out = e.step(Input::AttachSub(static_req), Instant::now());
    let static_sid =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    assert_eq!(
        e.subs().get(static_sid).unwrap().profile,
        profile_id,
        "static Sub joins the dynamic Sub's Profile via dedup",
    );

    let out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // Profile is NOT reaped — the recovery channel stays open.
    assert!(
        e.profiles.get(profile_id).is_some(),
        "Profile preserved for the static Sub's recovery channel",
    );
    // anchor_claim cleared (finalize_anchor_lost path).
    assert!(
        matches!(
            e.profiles.get(profile_id).unwrap().anchor_claim(),
            AnchorClaim::None,
        ),
        "anchor_claim cleared by finalize_anchor_lost",
    );
    // No DynamicSubReaped — the dynamic Sub remains attached (Promoter's
    // dynamic_subs entry is intact; on path reappearance, try_promote's
    // contains_check is the dedup gate).
    assert!(
        !out.diagnostics.iter().any(|d| matches!(
            d,
            Diagnostic::DynamicSubReaped { promoter, .. } if *promoter == pid,
        )),
        "no DynamicSubReaped on mixed Profile teardown: {:?}",
        out.diagnostics,
    );
    assert!(e.subs().get(dyn_sid).is_some(), "dynamic Sub retained");
    assert_eq!(
        e.promoters.get(pid).unwrap().dynamic_subs.len(),
        1,
        "Promoter.dynamic_subs entry retained",
    );
}

#[test]
fn anchor_terminal_no_subs_falls_back_to_finalize_anchor_lost() {
    // Defense-in-depth: a Profile with empty subs (structurally
    // unreachable in production) routes to finalize_anchor_lost.
    // We can't construct an empty-subs Profile cleanly via public
    // API, so this test exercises the predicate via a static-only
    // Profile (which also picks the finalize_anchor_lost branch
    // because all_dynamic is false).
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("anchor", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(r),
        identity: ProfileIdentity {
            config: cfg(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        params: SubParams {
            name: String::from("static"),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let profile_id = e.subs().get(sid).unwrap().profile;

    let _ = e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );

    // Static Sub: finalize_anchor_lost path. Profile lives.
    assert!(
        e.profiles.get(profile_id).is_some(),
        "static Profile preserved (finalize_anchor_lost retains Profile)",
    );
    assert!(
        matches!(
            e.profiles.get(profile_id).unwrap().anchor_claim(),
            AnchorClaim::None,
        ),
        "anchor_claim cleared",
    );
}

/// Pin the predicate: `subs.iter().all(s.source_promoter.is_some())` —
/// a single static Sub (source_promoter=None) flips all_dynamic to
/// false.
#[test]
fn anchor_terminal_predicate_static_sub_makes_mixed() {
    let mut e = Engine::new();
    let (pid, dyn_sid, anchor) = promote_one(&mut e, "/var/log/*.log", &["var", "log"], "foo.log");
    let profile_id = e.subs().get(dyn_sid).expect("Sub alive").profile;
    let req = SubAttachRequest {
        anchor: SubAttachAnchor::Resource(anchor),
        identity: ProfileIdentity {
            config: cfg(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        params: SubParams {
            name: String::from("static"),
            program: empty_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            source_promoter: None,
        },
    };
    let _ = e.step(Input::AttachSub(req), Instant::now());

    // Two Subs on this Profile: one static, one dynamic.
    let subs_on_profile = e.subs().at(profile_id).len();
    assert_eq!(subs_on_profile, 2);
    let all_dynamic = e.subs().at(profile_id).iter().all(|sid| {
        e.subs()
            .get(*sid)
            .is_some_and(|s| s.source_promoter.is_some())
    });
    assert!(!all_dynamic, "mixed Profile must not be all_dynamic");
    let _ = pid; // pid only needed to anchor the Promoter alive
    let _ = e.cancel_all_in_flight_probes();
}
