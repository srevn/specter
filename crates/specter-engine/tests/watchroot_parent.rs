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

use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ClassSet, DirMeta, DirSnapshot, EffectScope, FsIdentity, Input, ProbeOp,
    ProbeOutcome, ProbeOwner, ProbeResponse, ResourceKind, ResourceRole, ScanConfig,
    SubAttachAnchor, SubAttachRequest, WatchOp,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::{Duration, Instant, UNIX_EPOCH};

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn empty_program() -> Arc<ActionProgram> {
    single_exec_program([specter_core::ArgTemplate::new([
        specter_core::ArgPart::literal("/bin/true"),
    ])])
}

/// Empty `TreeSnapshot::Dir`.
fn dir_snap() -> std::sync::Arc<DirSnapshot> {
    Arc::new(DirSnapshot::new(
        DirMeta {
            mtime: UNIX_EPOCH,
            fs_id: FsIdentity {
                inode: 0,
                device: 0,
            },
        },
        0,
        BTreeMap::new(),
    ))
}

#[test]
fn attach_sub_creates_watch_root_parent_contribution() {
    // Tree has /root and /root/src. attach_sub at /root/src; /root's
    // watch_demand bumps; /root/src watch_demand bumps; Profile records
    // /root as its watch_root_parent.
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("root", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);
    let src = e
        .tree_mut()
        .ensure_child(root, "src", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(src),
        ScanConfig::builder().recursive(true).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    assert_eq!(
        e.tree().get(src).unwrap().watch_demand(),
        1,
        "anchor watched"
    );
    assert_eq!(
        e.tree().get(root).unwrap().watch_demand(),
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
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(src),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), Instant::now());
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    assert!(e.profiles().get(pid).unwrap().watch_root_parent.is_none());
}

#[test]
fn detach_sub_releases_watch_root_parent_contribution() {
    let mut e = Engine::new();
    let root = e.tree_mut().ensure_root("root", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);
    let src = e
        .tree_mut()
        .ensure_child(root, "src", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src, ResourceKind::Dir);

    let now = Instant::now();
    let req = SubAttachRequest::for_anchor(
        "watch".into(),
        SubAttachAnchor::Resource(src),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;

    // Drive Seed → Idle.
    let corr = attach_out
        .probe_ops
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap()),
        }),
        now,
    );
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 1);

    // Detach. The detach releases the anchor's contribution and the
    // watch-root parent's contribution; `Tree::try_reap` cascades up
    // from the now-orphaned anchor and reaps `/root` in the same step
    // (no other claims). Both slots emit `Unwatch` on the way out.
    let out = e.step(Input::DetachSub(sid), Instant::now());
    assert!(
        e.tree().get(root).is_none_or(|r| r.watch_demand() == 0),
        "/root's watch_demand back to 0 (or slot reaped by the cascade)",
    );
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
    let root = e.tree_mut().ensure_root("root", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);
    let src_a = e
        .tree_mut()
        .ensure_child(root, "srcA", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src_a, ResourceKind::Dir);
    let src_b = e
        .tree_mut()
        .ensure_child(root, "srcB", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src_b, ResourceKind::Dir);

    let now = Instant::now();
    let _ = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(src_a),
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let _ = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "B".into(),
            SubAttachAnchor::Resource(src_b),
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    assert_eq!(
        e.tree().get(root).unwrap().watch_demand(),
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
    let root = e.tree_mut().ensure_root("root", ResourceRole::User);
    e.tree_mut().set_kind(root, ResourceKind::Dir);
    let src = e
        .tree_mut()
        .ensure_child(root, "src", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(src, ResourceKind::Dir);

    let now = Instant::now();
    // Sub at /root.
    let _ = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "outer".into(),
            SubAttachAnchor::Resource(root),
            ScanConfig::builder().recursive(false).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    assert!(matches!(
        e.tree().get(root).unwrap().role,
        ResourceRole::User,
    ));

    // Sub at /root/src — /root becomes its watch_root_parent.
    let _ = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "inner".into(),
            SubAttachAnchor::Resource(src),
            ScanConfig::builder().build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );

    // /root's role stays User (never demote User).
    assert!(matches!(
        e.tree().get(root).unwrap().role,
        ResourceRole::User,
    ));
    // watch_demand has both contributions (root's own + inner's parent).
    assert_eq!(e.tree().get(root).unwrap().watch_demand(), 2);
}
