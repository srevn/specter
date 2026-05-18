//! Hot-reload via `Input::ConfigDiff`. Atomic apply of
//! `removed → modified → added`; reap-pending mid-burst handling;
//! in-flight Effect race after detach.

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
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, BurstFinish, ChildEntry, ClassSet, DedupKey, Diagnostic, DirChild, DirMeta,
    DirSnapshot, EffectOutcome, EffectScope, EntryKind, FsEvent, FsIdentity, Input, LeafEntry,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ResourceKind, ResourceRole, ScanConfig,
    SubAttachAnchor, SubAttachRequest, SubRegistryDiff, WatchOp, WatchRegistryDiff,
};
use specter_engine::Engine;
use std::collections::BTreeMap;
use std::path::PathBuf;
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

/// V5-native helper: build a `TreeSnapshot::Dir` with single-component
/// children. Tests in this file use leaf-name segments only (no `/`).
fn dir_snap(children: Vec<(&str, EntryKind, u64)>) -> std::sync::Arc<DirSnapshot> {
    let mut map: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    for (name, kind, inode) in children {
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

#[test]
fn config_diff_add_sub_to_existing_profile() {
    // Engine has Sub A; ConfigDiff adds Sub B at the same anchor with the
    // same config — both share one Profile. The Profile's Sub count goes 1 → 2; no
    // new Watch/Probe/Suppress.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();
    let attach = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a = specter_core::testkit::first_attached_sub(&attach).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile;
    assert_eq!(e.subs().at(pid).len(), 1);

    // ConfigDiff with one added Sub at the same anchor + same cfg.
    let mut diff = SubRegistryDiff::default();
    diff.added.push(SubAttachRequest::for_anchor(
        "B".into(),
        SubAttachAnchor::Resource(r),
        cfg,
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        now,
    );

    assert_eq!(e.subs().at(pid).len(), 2);
    let new_watches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(new_watches, 0, "no fresh Watch on existing Profile");
    assert_eq!(new_probes, 0, "no fresh Probe on existing Profile");
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn config_diff_remove_sole_sub_reaps_profile() {
    // Engine has Sub A on its own Profile, post-Seed Idle. ConfigDiff
    // removes A. Profile reaped immediately (no Subs remain, Idle);
    // anchor unwatched.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let req = SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Resource(r),
        ScanConfig::builder().build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let now = Instant::now();
    let attach_out = e.step(Input::AttachSub(req), now);
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile;
    let seed_corr = attach_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Profile is Idle. Remove via ConfigDiff (by operator watch name).
    let mut diff = SubRegistryDiff::default();
    diff.removed.push(CompactString::from("A"));
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        now,
    );

    assert!(e.profiles().get(pid).is_none(), "Profile reaped");
    assert!(out.diagnostics.iter().any(|d| matches!(
        d,
        Diagnostic::ProfileReaped {
            via: specter_core::ReapTrigger::Immediate,
            ..
        }
    )));
    let unwatches = out
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unwatch { .. }))
        .count();
    assert!(unwatches >= 1, "anchor unwatched");
}

#[test]
fn config_diff_mid_burst_remove_defers_reap() {
    // Engine has Sub A; Standard burst in flight; ConfigDiff removes A.
    // reap_pending=true; on burst-end, no Effect; Profile reaped.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
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
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile;
    let seed_corr = attach_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Drive a Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Mid-burst ConfigDiff: remove A (by operator watch name).
    let mut diff = SubRegistryDiff::default();
    diff.removed.push(CompactString::from("A"));
    let _ = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        t1,
    );
    assert!(
        matches!(
            e.profiles().get(pid).unwrap().state().burst_finish(),
            Some(BurstFinish::Reap)
        ),
        "reap deferred to burst end",
    );

    // Drain settle to enter Probing.
    let t2 = t1 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t2) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t2,
        );
    }
    let std_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight");

    // Inject stable response. Profile reaps; no Effect.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: std_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(out.effects().is_empty(), "reap_pending suppresses Effect");
    assert!(
        e.profiles().get(pid).is_none(),
        "Profile reaped at burst end"
    );
}

#[test]
fn config_diff_mid_burst_modify_revives_profile() {
    // Engine has Sub A; Standard burst in flight; ConfigDiff modifies
    // the watch named "A" in place with the SAME `config_hash`
    // (different command, same anchor / max_settle / events). The
    // name-keyed shim resolves "A" → old SubId, runs `detach_sub_inner`
    // → `attach_sub_inner`, triggering the zombie-revival branch.
    // Production path that the user-API tests in `engine.rs` cannot
    // exercise on their own.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let cfg = ScanConfig::builder().build();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid_a).unwrap().profile;
    let seed_corr = attach_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Drive a Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    let watch_demand_before = e.tree().get(r).unwrap().watch_demand();

    // Mid-burst ConfigDiff: modify the watch "A" in place (same
    // config_hash; different command). The shim resolves "A" → old
    // SubId: detach A (refcount→0, reap_pending), then attach the
    // fresh "A" (zombie revival).
    let mut diff = SubRegistryDiff::default();
    diff.modified.push(SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Resource(r),
        cfg,
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        t1,
    );

    let sid_b = e.subs().find_by_name("A").expect("A re-attached");
    let pid_b = e.subs().get(sid_b).unwrap().profile;
    assert_eq!(
        pid_b, pid,
        "re-attached A revives its Profile (same config_hash)"
    );
    let p = e.profiles().get(pid).unwrap();
    assert!(
        !matches!(p.state().burst_finish(), Some(BurstFinish::Reap)),
        "reap_pending cleared by revival"
    );
    assert_eq!(e.subs().at(pid).len(), 1, "exactly one live Sub (A)");
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        watch_demand_before,
        "anchor watch_demand unchanged on hot-reload modify (no double-bump)",
    );
    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::ReapPendingCancelled { profile } if *profile == pid)),
        "ReapPendingCancelled emitted",
    );
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(
        new_probes, 0,
        "no fresh Probe — existing Standard burst's settle timer still owns the lifecycle",
    );
}

#[test]
fn effect_complete_after_detach_drops_silently() {
    // Engine has Sub on Idle Profile; an Effect was previously emitted
    // (we mock the EffectComplete path manually). Detach the Sub; then
    // inject EffectComplete for the now-removed Sub. Engine drops with
    // a Diagnostic — no panic, no reseed.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
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
    let sid = specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid = e.subs().get(sid).unwrap().profile;
    let seed_corr = attach_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Detach via ConfigDiff (by operator watch name).
    let mut diff = SubRegistryDiff::default();
    diff.removed.push(CompactString::from("A"));
    e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        now,
    );

    // Inject EffectComplete for the now-removed Sub.
    let out = e.step(
        Input::EffectComplete {
            sub: sid,
            key: DedupKey::Subtree {
                sub: sid,
                profile: pid,
            },
            result: EffectOutcome::Ok,
        },
        now,
    );

    assert!(
        out.diagnostics
            .iter()
            .any(|d| matches!(d, Diagnostic::EffectCompleteForUnknownSub { .. }))
    );
    // No Probe re-emitted (no reseed).
    let new_probes = out
        .probe_ops()
        .iter()
        .filter(|op| matches!(op, ProbeOp::Probe { .. }))
        .count();
    assert_eq!(new_probes, 0);
}

#[test]
fn config_diff_modified_remove_then_add() {
    // Sub "A" at /src with recursive=true; ConfigDiff modifies the
    // watch "A" in place to recursive=false. The name-keyed shim
    // resolves "A" → old SubId and processes as detach + attach. The
    // new Sub gets a fresh Profile (different config_hash) anchored at
    // the same path (path-based add re-materializes if needed).
    let mut e = Engine::new();
    let r = e
        .tree_mut()
        .ensure_path(&["/", "src"], ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    let now = Instant::now();
    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "A".into(),
            SubAttachAnchor::Resource(r),
            ScanConfig::builder().recursive(true).build(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_a = e.subs().get(sid_a).unwrap().profile;
    let seed_corr = attach_out
        .probe_ops()
        .iter()
        .find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        })
        .unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_a),
            correlation: seed_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Modified entry: same watch name "A"; new request with a
    // different config_hash. Path-based to handle anchor
    // re-materialization safely.
    let mut diff = SubRegistryDiff::default();
    diff.modified.push(SubAttachRequest::for_anchor(
        "A".into(),
        SubAttachAnchor::Path(PathBuf::from("/src")),
        ScanConfig::builder().recursive(false).build(),
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    ));
    let _out = e.step(
        Input::ConfigDiff(WatchRegistryDiff {
            subs: diff,
            ..Default::default()
        }),
        now,
    );

    // Old Profile reaped; new Profile attached with different
    // config_hash. Old SubId no longer in registry; a fresh one was
    // minted by attach_sub_inner.
    assert!(e.subs().get(sid_a).is_none(), "old Sub removed");
    assert_eq!(e.subs().len(), 1, "exactly one Sub remains");
    let _ = e.cancel_all_in_flight_probes();
}
