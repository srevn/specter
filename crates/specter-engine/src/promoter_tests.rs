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
//! - `dispatch_promoter_descent_ok` advance + materialise.
//! - `dispatch_promoter_enumeration_ok` forward pass (sub-proxy
//!   registration + final-position promotion).
//! - `try_promote` dedup ([I-Promoter-5]).
//! - Event routing: `on_promoter_proxy_event` enqueue + dispatch +
//!   stale-event diagnostic.
//! - `dispatch_promoter_descent_vanished` rewind.

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
use crate::engine::FS_ROOT_SEG;
use compact_str::CompactString;
use specter_core::{
    ChildEntry, ClassSet, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectScope, EntryKind,
    FsEvent, Input, LeafEntry, PatternSpec, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse,
    PromoterAttachRequest, PromoterId, PromoterState, ResourceId, ResourceKind, ResourceRole,
    ScanConfig,
};
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

fn cfg() -> ScanConfig {
    ScanConfig::builder().recursive(true).build()
}

fn empty_command() -> specter_core::CommandTemplate {
    specter_core::CommandTemplate::new([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

/// Build a `PromoterAttachRequest` with a freshly parsed `PatternSpec`.
fn req_for(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.to_owned(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        config: cfg(),
        max_settle: MAX_SETTLE,
        settle: SETTLE,
        command: empty_command(),
        scope: EffectScope::SubtreeRoot,
        events: ClassSet::EMPTY,
        log_output: false,
    }
}

/// Build an `Arc<DirSnapshot>` whose root-resource is `target` (the proxy
/// or descent prefix the request named) and whose entries match the
/// supplied list. Mirrors `descent_tests.rs::dir_snap_with`'s shape.
fn dir_snap_at(target: ResourceId, children: &[(&str, EntryKind, u64)]) -> Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children.iter().copied() {
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
    Arc::new(DirSnapshot::new(
        target,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    ))
}

/// Resolve a `/var/log` chain (or any existing path) under the FS-root.
/// Helper for setup blocks that pre-place a Dir on the Tree.
fn ensure_dir(e: &mut Engine, segments: &[&str]) -> ResourceId {
    let mut comps = Vec::with_capacity(segments.len() + 1);
    comps.push(FS_ROOT_SEG);
    comps.extend_from_slice(segments);
    let r = e.tree_mut().ensure_path(&comps, ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

/// Read the descent probe target from a probe-ops list (the latest
/// outstanding descent/subtree probe in emission order).
fn last_probe_target(out: &specter_core::StepOutput) -> Option<ResourceId> {
    out.probe_ops.iter().rev().find_map(|op| match op {
        ProbeOp::Probe { request } => request.target_resource(),
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
        Some(PromoterState::Active { proxies }) => proxies.clone(),
        s => panic!("expected Active state, got {s:?}"),
    }
}

#[test]
fn attach_immediate_active_at_existing_prefix() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let (pid, out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
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
    assert_eq!(e.tree().get(var_log).unwrap().watch_demand, 1);
    assert!(
        e.tree()
            .get(var_log)
            .unwrap()
            .events_union
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
    assert_eq!(last_probe_target(&out), Some(var_log));
}

#[test]
fn attach_pending_when_literal_prefix_missing() {
    let mut e = Engine::new();
    // No /var/log on disk; only the FS-root bootstrap will create /.
    let (pid, out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    assert_ne!(pid, PromoterId::default());

    // State: PrefixPending(d). d.current_prefix == FS-root slot;
    // remaining_components = ["var", "log"].
    let q = e.promoters.get(pid).expect("promoter registered");
    let PromoterState::PrefixPending(d) = &q.state else {
        panic!("expected PrefixPending, got {:?}", q.state);
    };
    let fs_root = e.tree().lookup(None, FS_ROOT_SEG).expect("FS-root exists");
    assert_eq!(d.current_prefix, fs_root, "descent at FS-root");
    assert_eq!(
        d.remaining_components,
        vec![CompactString::from("var"), CompactString::from("log")],
        "two literal segments to descend",
    );

    // Watch_demand on FS-root bumped (STRUCTURE).
    assert_eq!(e.tree().get(fs_root).unwrap().watch_demand, 1);

    // Descent probe in flight at FS-root.
    assert_eq!(last_probe_target(&out), Some(fs_root));
    assert!(e.pending_probe_for(ProbeOwner::Promoter(pid)).is_some());
}

#[test]
fn descent_advances_one_segment_on_partial_response() {
    let mut e = Engine::new();
    // /var exists; /var/log doesn't.
    let var = ensure_dir(&mut e, &["var"]);

    let (pid, out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    assert_eq!(last_probe_target(&out), Some(var), "first probe at /var");

    // Inject SubtreeOk: /var contains "log" as a Dir. Descent should
    // advance to /var/log; remaining = [].
    let snap = dir_snap_at(var, &[("log", EntryKind::Dir, 1)]);
    let _out = e.step(
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
}

#[test]
fn enumeration_ok_promotes_final_match() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let (pid, out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    assert_eq!(last_probe_target(&out), Some(var_log));

    // Inject enumeration response listing two files: one matches *.log,
    // one doesn't.
    let snap = dir_snap_at(
        var_log,
        &[
            ("foo.log", EntryKind::File, 1),
            ("bar.txt", EntryKind::File, 2),
        ],
    );
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );

    // dynamic_subs contains the matched path; bar.txt is absent.
    let q = e.promoters.get(pid).unwrap();
    let promoted_paths: Vec<_> = q.dynamic_subs.keys().cloned().collect();
    assert_eq!(promoted_paths.len(), 1);
    assert_eq!(
        promoted_paths[0].to_string_lossy(),
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
}

#[test]
fn enumeration_ok_registers_subproxy_for_intermediate_glob() {
    let mut e = Engine::new();
    // Pattern /srv/*/site — first proxy at /srv (lpl=2).
    let srv = ensure_dir(&mut e, &["srv"]);

    let (pid, _out) = e.attach_promoter(req_for("sites", "/srv/*/site"), Instant::now());
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Inject /srv listing: two child Dirs ("alpha", "beta") and a stray
    // File ("noisy.cfg") that the glob would match but should be
    // skipped at non-final position (only Dir matches descend).
    let snap = dir_snap_at(
        srv,
        &[
            ("alpha", EntryKind::Dir, 1),
            ("beta", EntryKind::Dir, 2),
            ("noisy.cfg", EntryKind::File, 3),
        ],
    );
    let _out = e.step(
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
}

#[test]
fn try_promote_is_idempotent_on_repeated_match() {
    // Two enumeration cycles (e.g., a parent's StructureChanged refire)
    // for the same path do NOT mint two dynamic Subs. I-Promoter-5
    // contains-check gates at try_promote.
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let (pid, _out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();

    // Cycle 1: foo.log matches. Promotion mints SubA.
    let snap = dir_snap_at(var_log, &[("foo.log", EntryKind::File, 1)]);
    let _out = e.step(
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
    let _out = e.step(
        Input::FsEvent {
            resource: var_log,
            event: FsEvent::StructureChanged,
        },
        Instant::now(),
    );
    let corr2 = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap2 = dir_snap_at(var_log, &[("foo.log", EntryKind::File, 1)]);
    let _out = e.step(
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
}

#[test]
fn proxy_event_enqueues_and_dispatches() {
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let (pid, _out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    // Drain the initial enumeration: respond Ok with empty entries.
    let corr = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    let snap = dir_snap_at(var_log, &[]);
    let _out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Promoter(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(snap),
        }),
        Instant::now(),
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Promoter(pid)).is_none(),
        "channel closed after empty enumeration",
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
        last_probe_target(&out),
        Some(var_log),
        "probe target = the proxy that received the event",
    );
}

#[test]
fn proxy_event_for_unregistered_promoter_emits_stale_diagnostic() {
    // Manually populate `proxy_promoters` on a slot for a Promoter id
    // that never registered at that slot — the on_promoter_proxy_event
    // safety check must catch this and emit PromoterProxyStaleEvent
    // rather than enqueue garbage.
    use specter_core::PromoterId;
    let mut e = Engine::new();
    let r = e.tree_mut().ensure(None, "phantom", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    // Bump watch_demand so the on_fs_event head guard doesn't drop the
    // event with EventOnUnwatchedResource.
    let mut out = specter_core::StepOutput::default();
    crate::refcounts::add_watch_demand(e.tree_mut(), r, ClassSet::STRUCTURE, &mut out);

    // Synthesise a Promoter id that's NOT in the registry; push it
    // onto the back-ref.
    let phantom = PromoterId::default();
    e.tree_mut()
        .get_mut(r)
        .unwrap()
        .proxy_promoters
        .push(phantom);

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

    let (pid, _out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    let corr1 = e.pending_probe_for(ProbeOwner::Promoter(pid)).unwrap();
    // Sanity: descent is at /var with remaining=["log"].
    let q = e.promoters.get(pid).unwrap();
    let PromoterState::PrefixPending(d) = &q.state else {
        panic!("expected PrefixPending pre-vanish");
    };
    assert_eq!(d.current_prefix, var, "descent at /var pre-vanish");

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
    let fs_root = e.tree().lookup(None, FS_ROOT_SEG).unwrap();
    assert_eq!(d.current_prefix, fs_root, "rewind landed at FS-root");
    assert_eq!(
        d.remaining_components,
        vec![CompactString::from("var"), CompactString::from("log")],
        "vanished prefix's segment prepended; original remaining preserved",
    );

    // Fresh probe in flight at the parent.
    let corr2 = e
        .pending_probe_for(ProbeOwner::Promoter(pid))
        .expect("fresh probe minted");
    assert_ne!(corr1, corr2, "post-rewind correlation differs from pre");
    assert_eq!(last_probe_target(&out), Some(fs_root));

    // The vanished /var slot was vacated. Vacate clears
    // watch_demand and kind; the slot itself is retained while it
    // still has children (the still-existing /var/log
    // DescentScaffold the original descent created). Asserting on
    // vacate's observable effect:
    assert_eq!(
        e.tree().get(var).unwrap().watch_demand,
        0,
        "vanished prefix had its watch_demand zeroed (vacate)",
    );
}

#[test]
fn register_proxy_is_idempotent_on_re_registration() {
    // Re-registering the same proxy at the same idx must NOT
    // double-bump watch_demand or duplicate the queue entry. [H-5]
    let mut e = Engine::new();
    let var_log = ensure_dir(&mut e, &["var", "log"]);

    let (pid, _out) = e.attach_promoter(req_for("logs", "/var/log/*.log"), Instant::now());
    let watch_demand_after_attach = e.tree().get(var_log).unwrap().watch_demand;
    let queue_after_attach = pending_enumerations(&e, pid).len();

    // Re-register: directly invoke the (pub(crate)) helper.
    let mut out = specter_core::StepOutput::default();
    e.register_proxy(pid, var_log, 3, &mut out);

    assert_eq!(
        e.tree().get(var_log).unwrap().watch_demand,
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
}
