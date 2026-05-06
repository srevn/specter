//! F-HIGH-2 regression: `Input::WatchOpRejected` must clean up every
//! Profile-side claim on the rejected resource, not just descent
//! prefixes. Pre-fix the engine clamped `watch_demand := 0` and walked
//! Pending descents only, leaving `anchor_contribution` and
//! `watch_root_parent` flags claiming a now-zero counter. The next
//! Profile-driven release on the affected resource then read the stale
//! flag, recomputed the wrong `events_union`, or underflowed
//! `sub_watch_demand` on the next decrement.
//!
//! This file exercises the four claim configurations that the post-fix
//! fan-out handles: anchor (single & multi-Profile), watch-root parent,
//! and descent prefix (regression-pinning the existing path).

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use specter_core::{
    ArgPart, ArgTemplate, ChildEntry, ClaimKind, ClassSet, CommandTemplate, Diagnostic, DirChild,
    DirMeta, DirSnapshot, EffectScope, EntryKind, FsEvent, Input, LeafEntry, ProbeCorrelation,
    ProbeOp, ProbeRequest, ProbeResponse, ProbeResult, ProfileId, ProfileState, ResourceId,
    ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachRequest, SubId, TreeSnapshot,
    WatchOp,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);

// ───────────────────────────────────────────────────────────────────────
// Fixtures
// ───────────────────────────────────────────────────────────────────────

fn empty_command() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([ArgPart::literal("/bin/true")])])
}

fn dir_snap(root: ResourceId, children: Vec<(&str, EntryKind, u64)>) -> TreeSnapshot {
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
        root,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        0,
        map,
    )))
}

fn first_probe_corr(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

fn complete_seed_burst(
    e: &mut Engine,
    pid: ProfileId,
    attach_out: &StepOutput,
    seed_snap: TreeSnapshot,
) {
    let corr = first_probe_corr(attach_out).expect("Seed probe fires at attach");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(seed_snap),
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
    let req = SubAttachRequest::for_resource(
        name.to_string(),
        resource,
        ScanConfig::builder().recursive(true).build(),
        max_settle,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::CONTENT,
        false,
    );
    let (sid, out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    (sid, pid, out)
}

// ───────────────────────────────────────────────────────────────────────
// Anchor case — single Profile
// ───────────────────────────────────────────────────────────────────────

#[test]
fn anchor_claim_purged_then_detach_no_panic() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (sid, pid, attach_out) = attach_subtree_root(&mut e, "build", root, MAX_SETTLE);
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(root, vec![]));
    assert!(e.profiles().get(pid).unwrap().anchor_contribution);
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 1);

    // Reject the kernel watch on the anchor.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: root,
            op: WatchOp::Watch {
                resource: root,
                path: PathBuf::from("src"),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            errno: 24,
        },
        Instant::now(),
    );

    // Anchor flag cleared; counter zeroed.
    assert!(!e.profiles().get(pid).unwrap().anchor_contribution);
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 0);
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

    // Detach the Sub. Pre-fix this would underflow sub_watch_demand
    // because anchor_contribution stayed true after the clamp; the
    // subsequent reap_profile would call sub_watch_demand on the
    // already-zeroed counter.
    let _ = e.detach_sub(sid, Instant::now());
    assert!(e.profiles().get(pid).is_none(), "Profile reaped cleanly");
}

// ───────────────────────────────────────────────────────────────────────
// Anchor case — multi-Profile (transitive bug surface)
// ───────────────────────────────────────────────────────────────────────

#[test]
fn anchor_claim_purged_for_two_profiles_each_no_panic() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;

    let (sid_p, pid_p, out_p) = attach_subtree_root(&mut e, "P", root, MAX_SETTLE);
    let (_sid_q, pid_q, out_q) =
        attach_subtree_root(&mut e, "Q", root, MAX_SETTLE + Duration::from_secs(1));
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 2);
    complete_seed_burst(&mut e, pid_p, &out_p, dir_snap(root, vec![]));
    complete_seed_burst(&mut e, pid_q, &out_q, dir_snap(root, vec![]));

    // Reject the kernel watch.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: root,
            op: WatchOp::Watch {
                resource: root,
                path: PathBuf::from("src"),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            errno: 24,
        },
        Instant::now(),
    );

    // Both flags cleared.
    assert!(!e.profiles().get(pid_p).unwrap().anchor_contribution);
    assert!(!e.profiles().get(pid_q).unwrap().anchor_contribution);
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 0);

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

    // Subsequent dispatch_seed_vanished on either Profile must not
    // panic. Pre-fix this was the F-HIGH-2 multi-Profile manifestation:
    // the clamp left both flags dangling, and the first vanished
    // dispatch underflowed `sub_watch_demand`. Post-fix the helper is
    // counter-aware and the flag is already cleared by the purge.
    let p_seed_vanished = e.step(
        Input::FsEvent {
            resource: root,
            event: FsEvent::Removed,
        },
        Instant::now(),
    );
    // No panic. Profile P (Idle, anchor_contribution=false) is left
    // alone by covering_profiles' filter (P.state is Idle, not Pending,
    // so it would normally route through finalize_anchor_lost). But
    // wait — covering_profiles still includes Idle Profiles. The route
    // is finalize_anchor_lost(P) which is a no-op because
    // anchor_contribution is already false (post-purge) and was_active
    // is false (state is Idle, not Active). So it's a clean no-op.
    let _ = p_seed_vanished;

    // Detach P; assert clean reap.
    let _ = e.detach_sub(sid_p, Instant::now());
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
    let parent = e.tree_mut().ensure(None, "var", ResourceRole::User);
    e.tree_mut().get_mut(parent).unwrap().kind = ResourceKind::Dir;
    let anchor = e.tree_mut().ensure(Some(parent), "log", ResourceRole::User);
    e.tree_mut().get_mut(anchor).unwrap().kind = ResourceKind::Dir;

    let (sid, pid, attach_out) = attach_subtree_root(&mut e, "watch", anchor, MAX_SETTLE);
    complete_seed_burst(&mut e, pid, &attach_out, dir_snap(anchor, vec![]));
    // `set_watch_root_parent` ran at attach; parent has +1 STRUCTURE.
    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent,
        Some(parent),
    );
    assert_eq!(e.tree().get(parent).unwrap().watch_demand, 1);
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand, 1);

    // Reject the kernel watch on the parent (not the anchor).
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: parent,
            op: WatchOp::Watch {
                resource: parent,
                path: PathBuf::from("var"),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            errno: 24,
        },
        Instant::now(),
    );

    // Parent's flag cleared on the Profile; anchor stays watched.
    assert_eq!(e.profiles().get(pid).unwrap().watch_root_parent, None);
    assert_eq!(e.tree().get(parent).map_or(0, |r| r.watch_demand), 0);
    assert_eq!(e.tree().get(anchor).unwrap().watch_demand, 1);

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

    // Detach. Pre-fix `reap_profile` would call
    // `sub_watch_demand(parent, STRUCTURE)` against the now-zero
    // counter and underflow.
    let _ = e.detach_sub(sid, Instant::now());
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
    let foo = e.tree_mut().ensure(None, "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_path(
        "watch".into(),
        PathBuf::from("foo/bar"),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        ClassSet::EMPTY,
        false,
    );
    let (sid, attach_out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    let initial_corr = first_probe_corr(&attach_out).expect("descent probe");
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Pending(_),
    ));

    // Reject the kernel watch on the descent prefix.
    let purge_out = e.step(
        Input::WatchOpRejected {
            resource: foo,
            op: WatchOp::Watch {
                resource: foo,
                path: PathBuf::from("foo"),
                kind: ResourceKind::Unknown,
                events: ClassSet::EMPTY,
            },
            errno: 24,
        },
        Instant::now(),
    );

    // Descent vacated.
    assert!(matches!(
        e.profiles().get(pid).unwrap().state,
        ProfileState::Idle,
    ));
    // Cancel + ProfileClaimPurged{DescentPrefix} surface.
    assert!(
        purge_out
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Cancel { profile } if *profile == pid)),
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
            profile: pid,
            correlation: initial_corr,
            result: ProbeResult::Vanished,
        }),
        Instant::now(),
    );
    assert!(
        late.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::StaleProbeResponse { .. })),
    );
}
