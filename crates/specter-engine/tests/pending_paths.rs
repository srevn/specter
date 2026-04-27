//! Pending-path descent end-to-end. Drives `Engine::attach_sub` with a
//! path-based request, walks descent through scaffolds, and confirms
//! anchor materialization triggers a Seed burst.

#![allow(
    clippy::doc_markdown,
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
    ChildEntry, CommandTemplate, Diagnostic, DirChild, DirMeta, DirSnapshot, EffectScope,
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
        Instant::now(),
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
    e.tree_mut().get_mut(var).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/log/myapp"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
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
    assert_eq!(e.tree().get(myapp).unwrap().kind, ResourceKind::Dir);

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
    e.tree_mut().get_mut(var).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/missing"),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
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
    e.tree_mut().get_mut(var).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("var/missing"),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
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
    e.tree_mut().get_mut(root_dir).unwrap().kind = ResourceKind::Dir;
    let src = e
        .tree_mut()
        .ensure(Some(root_dir), "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_resource(
        "watch".into(),
        src,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
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
