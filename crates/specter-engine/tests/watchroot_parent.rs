//! Watch-root-parent infrastructure. Each User Profile contributes a `+1`
//! watch_demand to its parent Resource so the engine can detect
//! rename/delete of the anchor itself; the contribution is released on
//! `detach_sub` reap.

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

use specter_core::{
    ClassSet, CommandTemplate, DirMeta, DirSnapshot, EffectScope, Input, ProbeOp, ProbeResponse,
    ProbeResult, ResourceId, ResourceKind, ResourceRole, ScanConfig, SubAttachRequest, TreeSnapshot,
    WatchOp,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
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

/// Empty `TreeSnapshot::Dir` rooted at `root`.
fn dir_snap(root: ResourceId) -> TreeSnapshot {
    TreeSnapshot::Dir(Arc::new(DirSnapshot::new(
        root,
        DirMeta {
            mtime: UNIX_EPOCH,
            inode: 0,
            device: 0,
        },
        Instant::now(),
        0,
        BTreeMap::new(),
    )))
}

#[test]
fn attach_sub_creates_watch_root_parent_contribution() {
    // Tree has /root and /root/src. attach_sub at /root/src; /root's
    // watch_demand bumps; /root/src watch_demand bumps; Profile records
    // /root as its watch_root_parent.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let src = e.tree_mut().ensure(Some(root), "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;

    let req = SubAttachRequest::for_resource(
        "watch".into(),
        src,
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _out) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;

    assert_eq!(e.tree().get(src).unwrap().watch_demand, 1, "anchor watched");
    assert_eq!(
        e.tree().get(root).unwrap().watch_demand,
        1,
        "watch_root_parent contributes",
    );
    assert_eq!(
        e.profiles().get(pid).unwrap().watch_root_parent,
        Some(root),
        "Profile caches its watch_root_parent",
    );
}

#[test]
fn root_anchor_has_no_watch_root_parent() {
    // attach_sub at /src directly (no parent in Tree). watch_root_parent
    // stays None — root rename detection is unavailable.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;
    let req = SubAttachRequest::for_resource(
        "watch".into(),
        src,
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, _) = e.attach_sub(req, Instant::now());
    let pid = e.subs().get(sid).unwrap().profile;
    assert!(e.profiles().get(pid).unwrap().watch_root_parent.is_none());
}

#[test]
fn detach_sub_releases_watch_root_parent_contribution() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let src = e.tree_mut().ensure(Some(root), "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;

    let now = Instant::now();
    let req = SubAttachRequest::for_resource(
        "watch".into(),
        src,
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
    );
    let (sid, attach_out) = e.attach_sub(req, now);
    let pid = e.subs().get(sid).unwrap().profile;

    // Drive Seed → Idle.
    let corr = attach_out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid,
            correlation: corr,
            result: ProbeResult::Ok(dir_snap(src)),
        }),
        now,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 1);

    // Detach.
    let out = e.detach_sub(sid, now);
    // /root's watch_demand back to 0; Unwatch emitted.
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 0);
    let unwatch_count = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
        .count();
    // One Unwatch each for anchor and watch_root_parent (sorted by id).
    assert!(unwatch_count >= 2, "anchor + parent both unwatched");
}

#[test]
fn multiple_profiles_share_one_watch_root_parent() {
    // Sub A at /root/srcA, Sub B at /root/srcB. Both register /root as
    // watch_root_parent. /root's watch_demand = 2.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let src_a = e.tree_mut().ensure(Some(root), "srcA", ResourceRole::User);
    e.tree_mut().get_mut(src_a).unwrap().kind = ResourceKind::Dir;
    let src_b = e.tree_mut().ensure(Some(root), "srcB", ResourceRole::User);
    e.tree_mut().get_mut(src_b).unwrap().kind = ResourceKind::Dir;

    let now = Instant::now();
    let _ = e.attach_sub(
        SubAttachRequest::for_resource(
            "A".into(),
            src_a,
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
        ),
        now,
    );
    let _ = e.attach_sub(
        SubAttachRequest::for_resource(
            "B".into(),
            src_b,
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
        ),
        now,
    );
    assert_eq!(
        e.tree().get(root).unwrap().watch_demand,
        2,
        "both Profiles contribute to /root's watch_demand",
    );
}

#[test]
fn watch_root_parent_role_stays_user_when_already_user() {
    // /root is User-anchored by another Profile. Adding a Sub at
    // /root/src registers /root as watch_root_parent but does NOT
    // demote its role.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure(None, "root", ResourceRole::User);
    e.tree_mut().get_mut(root).unwrap().kind = ResourceKind::Dir;
    let src = e.tree_mut().ensure(Some(root), "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;

    let now = Instant::now();
    // Sub at /root.
    let _ = e.attach_sub(
        SubAttachRequest::for_resource(
            "outer".into(),
            root,
            ScanConfig::builder().recursive(false).build(),
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
        ),
        now,
    );
    assert!(matches!(
        e.tree().get(root).unwrap().role,
        ResourceRole::User,
    ));

    // Sub at /root/src — /root becomes its watch_root_parent.
    let _ = e.attach_sub(
        SubAttachRequest::for_resource(
            "inner".into(),
            src,
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
        ),
        now,
    );

    // /root's role stays User (never demote User).
    assert!(matches!(
        e.tree().get(root).unwrap().role,
        ResourceRole::User,
    ));
    // watch_demand has both contributions (root's own + inner's parent).
    assert_eq!(e.tree().get(root).unwrap().watch_demand, 2);
}
