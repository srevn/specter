//! Pending-path descent end-to-end. Drives `Engine::attach_sub` with a
//! path-based request, walks descent through scaffolds, and confirms
//! anchor materialization triggers a Seed burst.

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::single_match_else,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::{
    ChildEntry, ClassSet, CommandTemplate, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectScope,
    EntryKind, FsEvent, Input, LeafEntry, ProbeCorrelation, ProbeOp, ProbeRequest, ProbeResponse,
    ProbeResult, ProfileState, ResourceId, ResourceKind, ResourceRole, ScanConfig, StepOutput,
    SubAttachRequest, TreeSnapshot,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

/// V5-native helper: build a `TreeSnapshot::Dir` rooted at the default
/// `ResourceId` with single-component children. Tests in this file use
/// leaf-name segments only.
fn dir_snap_with(children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
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
    TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        ResourceId::default(),
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    )))
}

/// Pluck the correlation from the (single) Probe in `out`.
fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

#[test]
fn attach_sub_path_pending_then_anchor_appears() {
    // Tree has /var only. attach_sub at path /var/log/myapp pending state:
    // prefix=/var, remaining=[log, myapp]. Inject probe responses showing
    // log appears, then myapp appears. Anchor materializes; Seed burst
    // starts.
    let mut e = Engine::new();
    let var = e.tree_mut().ensure(None, "var", ResourceRole::User);
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/log/myapp"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).unwrap().profile;

    // Initial pending state: intermediate scaffold in place; anchor
    // already has role=User ("role = User for the
    // anchor, role = DescentScaffold for everything else"). Pending
    // status lives on `Profile.state == ProfileState::Pending(_)`, not
    // on the anchor's role.
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
    let var_corr = first_probe_corr(&attach_out).expect("descent probe at /var emitted");

    // Inject probe response showing `log` appears.
    let log_advance = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: var_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![("log", EntryKind::Dir, 1)])),
        }),
        now,
    );
    let log_corr = first_probe_corr(&log_advance).expect("descent probe at /var/log emitted");

    // Inject probe response showing `myapp` appears under /var/log.
    let myapp_materialize = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: log_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![("myapp", EntryKind::Dir, 2)])),
        }),
        now,
    );

    // Anchor materialized: kind set from the snapshot's entry; role
    // stays User (set at attach time).
    assert!(matches!(
        e.tree().get(myapp).unwrap().role,
        ResourceRole::User,
    ));
    assert_eq!(e.tree().get(myapp).unwrap().kind(), Some(ResourceKind::Dir));

    // Profile is now in Active(Seed Probing) — the Seed burst was
    // started at materialization.
    let burst_intent = match &e.profiles().get(pid).unwrap().state {
        ProfileState::Active(b) => b.intent,
        _ => panic!("expected Active"),
    };
    assert_eq!(burst_intent, specter_core::BurstIntent::Seed);
    let seed_corr = first_probe_corr(&myapp_materialize).expect("seed probe emitted");

    // Complete the Seed burst.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![])),
        }),
        now,
    );

    // Profile should now be Idle with baseline established.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.baseline.is_some());
}

#[test]
fn pending_path_failed_probe_retains_state() {
    let mut e = Engine::new();
    let var = e.tree_mut().ensure(None, "var", ResourceRole::User);
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/missing"),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (sid, attach_out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    let corr = first_probe_corr(&attach_out).expect("descent probe");

    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Failed { errno: 13 },
        }),
        Instant::now(),
    );

    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::PendingPathProbeFailed { errno: 13, .. }))
    );
    // Profile still pending (descent state lives on
    // `ProfileState::Pending`, not on a separate SecondaryMap).
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_),
    ));
}

#[test]
fn pending_path_event_at_prefix_emits_fresh_probe() {
    // Pending descent waiting for /var/missing/. Drain in-flight probe
    // with a no-progress response, then inject FsEvent at /var (the
    // prefix) to trigger a fresh probe (no settle).
    let mut e = Engine::new();
    let var = e.tree_mut().ensure(None, "var", ResourceRole::User);
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/missing"),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (sid, attach_out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    let corr = first_probe_corr(&attach_out).expect("descent probe");

    // No-progress response — descent stays pending.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(dir_snap_with(vec![("other", EntryKind::File, 99)])),
        }),
        Instant::now(),
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
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid))
        .count();
    assert_eq!(probes, 1, "FsEvent at prefix triggers fresh descent probe");
}

#[test]
fn anchor_disappears_re_enters_pending_via_watch_root_parent() {
    // "Watch root deletion": Sub at /src; / is the
    // watch_root_parent. Anchor is removed; Profile → Idle with
    // current=None. Then a StructureChanged at / triggers the recovery
    // path which re-enters pending descent.
    let mut e = Engine::new();
    // Both / and /src exist; /src is the anchor.
    let root_dir = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().set_kind(root_dir, ResourceKind::Dir);
    let src = e
        .tree_mut()
        .ensure(Some(root_dir), "src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);

    let req = SubAttachRequest::for_resource(
        "watch".into(),
        src,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).unwrap().profile;
    // Drive Seed → Idle.
    let seed_corr = first_probe_corr(&attach_out).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: seed_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![])),
        }),
        now,
    );
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    assert!(e.profiles().get(pid).unwrap().watch_root_parent == Some(root_dir));

    // Anchor gone (Removed event at /src).
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Removed,
        },
        now,
    );
    // Profile is Idle with current=None now.
    let p = e.profiles().get(pid).unwrap();
    assert!(matches!(p.state, ProfileState::Idle));
    assert!(p.current.is_none());

    // StructureChanged at / triggers recovery: Profile re-enters pending
    // descent with prefix=/, remaining=[src].
    let out = e.step(
        Input::FsEvent {
            resource: root_dir,
            event: FsEvent::StructureChanged,
        },
        now,
    );
    let recovery_probe = out
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid));
    assert!(
        recovery_probe,
        "recovery emits descent probe at watch_root_parent",
    );
}

// ───────────────────────────────────────────────────────────────────────
// F-MED-1 regression: detach Pending Profile with in-flight descent probe
//
// Pre-fix `reap_profile`'s Pending branch released the prefix's
// `watch_demand` and detached the Profile, but did NOT emit
// `ProbeOp::Cancel`. The prober finishes the syscall and ships a
// `ProbeResponse` for a now-detached Profile, which the engine drops as
// `StaleProbeResponse` — wasted prober capacity and I/O.
//
// Post-fix `reap_profile` invokes `cancel_pending_probe` (which checks
// `Profile.pending_probe` and emits `ProbeOp::Cancel` if the channel is
// open) before `release_descent_prefix_claim` (which transitions the
// Profile to Idle).
// ───────────────────────────────────────────────────────────────────────
#[test]
fn detach_pending_profile_with_inflight_descent_emits_cancel() {
    let mut e = Engine::new();
    let var = e.tree_mut().ensure(None, "var", ResourceRole::User);
    e.tree_mut().set_kind(var, ResourceKind::Dir);

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/log/myapp"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).unwrap().profile;

    // Profile is Pending with an in-flight descent probe.
    let initial_corr = first_probe_corr(&attach_out).expect("descent probe at attach");
    let is_pending = matches!(
        &e.profiles().get(pid).expect("Profile attached").state,
        ProfileState::Pending(_)
    );
    assert!(is_pending, "Profile is in Pending state");
    assert_eq!(
        e.pending_probe(pid),
        Some(initial_corr),
        "descent state carries the outstanding probe correlation",
    );

    // Detach without delivering a probe response.
    let detach_out = e.detach_sub(sid, now);

    // Profile is reaped.
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped on detach (Pending+sub_refcount==0)",
    );
    // ProbeOp::Cancel emitted for the in-flight descent probe.
    let cancel_present = detach_out
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Cancel { profile } if *profile == pid));
    assert!(
        cancel_present,
        "ProbeOp::Cancel emitted for in-flight descent probe; got {:?}",
        detach_out.probe_ops,
    );
}

// ───────────────────────────────────────────────────────────────────────
// F-HIGH-1 regression: anchor terminal event on a Pending Profile
//
// `was_active = !matches!(state, Idle)` historically included Pending,
// so a terminal event at a Pending Profile's anchor (degenerate path:
// `prefix == anchor` from `materialize_path_or_pending`'s None branch,
// reachable only via test-fixture relative-path attaches against an
// empty Tree) routed through `finish_burst_to_idle` and underflowed
// `sub_suppress` (Pending never bumped suppress_count).
//
// Production reach is sealed by the FS_ROOT_SEG bootstrap: absolute
// paths always have at least the bootstrapped root pre-existing, so
// `prefix_idx >= Some(0)` and the None branch never fires. This test
// uses the relative-path test fixture to construct the degenerate state
// and asserts no panic.
//
// Post-fix:
//   - `covering_profiles` filters Pending at the source.
//   - `finalize_anchor_lost` early-returns on Pending defensively.
//   - `finish_burst_to_idle` tightens `was_active` to `Active(_)`.
// ───────────────────────────────────────────────────────────────────────
#[test]
fn pending_profile_anchor_terminal_event_does_not_underflow_suppress() {
    let mut e = Engine::new();
    // Relative-path attach against an empty Tree triggers
    // `materialize_path_or_pending`'s None branch — anchor and prefix
    // are the same Resource (degenerate fixture).
    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("foo"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let (sid, _attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).unwrap().profile;

    // Confirm the degenerate setup: prefix == anchor.
    let p = e.profiles().get(pid).expect("Profile attached");
    let anchor = p.resource;
    let prefix = match &p.state {
        ProfileState::Pending(d) => d.current_prefix,
        s => panic!("expected Pending, got {s:?}"),
    };
    assert_eq!(prefix, anchor, "fixture: prefix == anchor");
    assert!(
        e.tree().get(anchor).unwrap().watch_demand >= 1,
        "prefix bumped its STRUCTURE contribution",
    );

    // Dispatch FsEvent::Removed at the anchor (== prefix). Routing:
    //   - classify_event_carriers(anchor) finds P in `descents`;
    //     on_descent_event short-circuits (probe still in flight from
    //     attach — I5).
    //   - covering_profiles filters Pending → no per-Profile dispatch.
    //   - finalize_anchor_lost is NOT called for P.
    // Pre-fix this routed through finish_burst_to_idle → sub_suppress
    // (underflow because Pending never bumped suppress_count). Post-fix:
    // no panic. The Profile remains Pending; no Unsuppress is emitted.
    let out = e.step(
        Input::FsEvent {
            resource: anchor,
            event: FsEvent::Removed,
        },
        now,
    );

    // Profile still Pending — covering-Profile fan-out skipped P.
    let still_pending = matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_),
    );
    assert!(
        still_pending,
        "Pending Profile not coerced through anchor-terminal-event path",
    );
    // No suppress underflow ⇒ no Unsuppress emitted (suppress_count
    // was never bumped on this Resource).
    let unsuppress_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, specter_core::WatchOp::Unsuppress { .. }))
        .count();
    assert_eq!(unsuppress_count, 0);
    // suppress_count remains untouched — Pending never bumped it.
    assert_eq!(
        e.tree().get(anchor).unwrap().suppress_count,
        0,
        "suppress_count untouched (Pending never bumped it)",
    );
}

// ───────────────────────────────────────────────────────────────────────
// Behavioral parity: a single FsEvent at one Resource fans out to a
// Pending Profile (descent dispatch) AND an Idle Profile with absent
// anchor (recovery dispatch), without disturbing an unrelated Profile.
//
// Pre-fix: `on_fs_event` walks `self.profiles` twice — once to collect
// `descents_at_prefix(resource)`, then again to filter recovery
// candidates (`watch_root_parent == Some(resource) && Idle &&
// current.is_none()`). Both passes produce disjoint Profile sets.
//
// Post-fix: `classify_event_carriers` collects both classes in a single
// pass over `self.profiles`, returning an `EventCarriers { descents,
// recoveries }`. Mutual exclusivity holds structurally — `Pending`
// excludes `Idle` at the `ProfileState` level.
//
// This test asserts the routing observable: A receives a fresh descent
// probe; B transitions Idle → Pending and emits a recovery descent
// probe; C is untouched. It passes both pre- and post-refactor — the
// refactor's purpose is performance, not behavior.
// ───────────────────────────────────────────────────────────────────────
#[test]
#[allow(clippy::similar_names)]
fn classifier_routes_descent_and_recovery_in_single_pass() {
    // /root and /root/bar exist; /root/foo does not. /elsewhere exists.
    let mut e = Engine::new();
    let root_dir = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().set_kind(root_dir, ResourceKind::Dir);
    let bar = e
        .tree_mut()
        .ensure(Some(root_dir), "bar", ResourceRole::User);
    e.tree_mut().set_kind(bar, ResourceKind::Dir);
    let elsewhere = e.tree_mut().ensure(None, "elsewhere", ResourceRole::User);
    e.tree_mut().set_kind(elsewhere, ResourceKind::Dir);

    // Profile A: Pending at /root, descending toward `foo` (does not
    // exist). Drain its initial descent probe with a no-progress
    // response so its `pending_probe` slot is empty before the test
    // event — `on_descent_event` short-circuits on a busy slot.
    let req_a = SubAttachRequest::for_path(
        "watch-a".into(),
        PathBuf::from("root/foo"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let (sid_a, attach_a_out) = e.attach_sub(req_a, now);
    let pid_a = e.subs().get(sid_a).unwrap().profile;
    let a_corr = first_probe_corr(&attach_a_out).expect("descent probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_a,
            correlation: a_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![("bar", EntryKind::Dir, 1)])),
        }),
        now,
    );
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state,
            ProfileState::Pending(_),
        ),
        "A still Pending after no-progress response",
    );

    // Profile B: anchor at /root/bar; drive Seed → Idle, then Removed
    // at /root/bar → Idle with current=None and watch_root_parent=/root.
    let req_b = SubAttachRequest::for_resource(
        "watch-b".into(),
        bar,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (sid_b, attach_b_out) = e.attach_sub(req_b, now);
    let pid_b = e.subs().get(sid_b).unwrap().profile;
    let b_corr = first_probe_corr(&attach_b_out).expect("Seed probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_b,
            correlation: b_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![])),
        }),
        now,
    );
    assert_eq!(
        e.profiles().get(pid_b).unwrap().watch_root_parent,
        Some(root_dir),
        "B watches its parent /root for anchor recovery",
    );
    e.step(
        Input::FsEvent {
            resource: bar,
            event: FsEvent::Removed,
        },
        now,
    );
    let p_b = e.profiles().get(pid_b).unwrap();
    assert!(matches!(p_b.state, ProfileState::Idle));
    assert!(p_b.current.is_none(), "B's anchor is gone");
    assert_eq!(p_b.watch_root_parent, Some(root_dir));

    // Profile C: anchor at /elsewhere; Seed → Idle. Unrelated to /root.
    let req_c = SubAttachRequest::for_resource(
        "watch-c".into(),
        elsewhere,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (sid_c, attach_c_out) = e.attach_sub(req_c, now);
    let pid_c = e.subs().get(sid_c).unwrap().profile;
    let c_corr = first_probe_corr(&attach_c_out).expect("Seed probe at attach");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_c,
            correlation: c_corr,
            result: ProbeResult::Ok(dir_snap_with(vec![])),
        }),
        now,
    );
    assert!(matches!(
        e.profiles().get(pid_c).unwrap().state,
        ProfileState::Idle,
    ));

    // The trigger: a single StructureChanged event at /root.
    // - A's `current_prefix == /root` ⇒ descent dispatch.
    // - B's `watch_root_parent == /root && current.is_none()` ⇒ recovery
    //   dispatch (Idle → Pending).
    // - C is anchored at /elsewhere ⇒ untouched.
    let out = e.step(
        Input::FsEvent {
            resource: root_dir,
            event: FsEvent::StructureChanged,
        },
        now,
    );

    // A: a fresh descent probe minted (slot was empty after drain).
    let a_probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid_a))
        .count();
    assert_eq!(a_probes, 1, "A's descent advance emits one probe");
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state,
            ProfileState::Pending(_),
        ),
        "A remains Pending",
    );

    // B: re-entered Pending (recovery descent) and emitted a probe.
    let b_probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid_b))
        .count();
    assert_eq!(b_probes, 1, "B's recovery emits one descent probe");
    assert!(
        matches!(
            e.profiles().get(pid_b).unwrap().state,
            ProfileState::Pending(_),
        ),
        "B transitioned Idle → Pending",
    );

    // C: untouched. No probe; state still Idle.
    let c_probes = out
        .probe_ops
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid_c))
        .count();
    assert_eq!(c_probes, 0, "C is unrelated to /root; no probe");
    assert!(matches!(
        e.profiles().get(pid_c).unwrap().state,
        ProfileState::Idle,
    ));
}
