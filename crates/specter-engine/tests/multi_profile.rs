//! Multi-Profile composition end-to-end. Two Profiles co-located on one
//! Resource share `watch_demand`/`suppress_count` via refcount aggregation;
//! parent–child Profiles propagate `dirty_descendants` through the
//! StabilityIndex; the `Active(Draining)` exit row drives the reconfirm
//! probe.

#![allow(
    clippy::items_after_statements,
    clippy::manual_let_else,
    clippy::match_wildcard_for_single_variants,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::option_if_let_else,
    clippy::redundant_clone,
    clippy::single_match_else,
    clippy::too_many_lines,
    dead_code
)]

use compact_str::CompactString;
use specter_core::{
    BurstPhase, ChildEntry, ClassSet, CommandTemplate, Diff, DirChild, DirMeta, DirSnapshot,
    EffectScope, EntryKind, FsEvent, Input, LeafEntry, ProbeCorrelation, ProbeOp, ProbeRequest,
    ProbeResponse, ProbeResult, ProfileState, ResourceId, ResourceKind, ResourceRole, ScanConfig,
    StepOutput, SubAttachRequest, TreeSnapshot, WatchOp,
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

/// V5-native helper: build a `TreeSnapshot::Dir` with flat single-component
/// children. Tests in this file use leaf-name segments only (no `/`).
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

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops.iter().find_map(|op| match op {
        ProbeOp::Probe {
            request: ProbeRequest { correlation, .. },
        } => Some(*correlation),
        ProbeOp::Cancel { .. } => None,
    })
}

#[test]
fn two_profiles_one_resource_share_watch_demand() {
    // Two Profiles at the same anchor (different config_hash). After
    // both attaches: anchor.watch_demand == 2; only one Watch op was
    // emitted (the 0→1 edge).
    let mut e = Engine::new();
    let r = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(r).unwrap().kind = ResourceKind::Dir;

    let cfg_a = ScanConfig::builder().recursive(true).build();
    let cfg_b = ScanConfig::builder().recursive(false).build();

    let req_a = SubAttachRequest::for_resource(
        "build".into(),
        r,
        cfg_a,
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (_sid_a, out_a) = e.attach_sub(req_a, Instant::now());
    let watch_count_a = out_a
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    assert_eq!(watch_count_a, 1, "0→1 edge emits one Watch");

    let req_b = SubAttachRequest::for_resource(
        "lint".into(),
        r,
        cfg_b,
        MAX_SETTLE,
        SETTLE,
        empty_command(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let (_sid_b, out_b) = e.attach_sub(req_b, Instant::now());
    let watch_count_b = out_b
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    assert_eq!(watch_count_b, 0, "1→2 edge emits no Watch");

    assert_eq!(e.tree().get(r).unwrap().watch_demand, 2);
}

#[test]
fn parent_child_standard_burst_propagates_dirty_descendants() {
    // Parent Profile at /src (recursive); child Profile at /src/foo.
    // only Standard bursts propagate (+1 at start, -1 at end).
    // Seed bursts establish the baseline and don't count as "dirty" for
    // ancestors.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;
    let foo = e.tree_mut().ensure(Some(src), "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;

    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();
    let (sid_p, out_p) = e.attach_sub(
        SubAttachRequest::for_resource(
            "parent".into(),
            src,
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        ),
        now,
    );
    let pid_parent = e.subs().get(sid_p).unwrap().profile;

    // Drive parent through Seed → Idle.
    let parent_seed = first_probe_correlation(&out_p).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_parent,
            correlation: parent_seed,
            result: ProbeResult::Ok(dir_snap(src, vec![])),
        }),
        now,
    );

    let (sid_c, out_c) = e.attach_sub(
        SubAttachRequest::for_resource(
            "child".into(),
            foo,
            cfg,
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        ),
        now,
    );
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    let child_seed = first_probe_correlation(&out_c).unwrap();

    // Parent's dirty_descendants does NOT bump on child's Seed burst.
    assert_eq!(
        e.profiles().get(pid_parent).unwrap().dirty_descendants,
        0,
        "Seed bursts don't propagate",
    );

    // Drive child through Seed → Idle.
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_child,
            correlation: child_seed,
            result: ProbeResult::Ok(dir_snap(foo, vec![])),
        }),
        now,
    );
    assert_eq!(
        e.profiles().get(pid_parent).unwrap().dirty_descendants,
        0,
        "Seed end doesn't propagate either",
    );

    // Now trigger a Standard burst on the child via FsEvent at /src/foo.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t1,
    );
    // Standard burst on child propagates +1 to parent.
    assert_eq!(
        e.profiles().get(pid_parent).unwrap().dirty_descendants,
        1,
        "child Standard burst start bumps parent",
    );
}

#[test]
fn parent_in_draining_reconfirms_after_child_settles() {
    // Build the topology, get parent into Draining (stable + dirty>0),
    // then complete the child — propagate(-1) returns parent's id and
    // the engine emits a reconfirm probe in the same step.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(src).unwrap().kind = ResourceKind::Dir;
    let foo = e.tree_mut().ensure(Some(src), "foo", ResourceRole::User);
    e.tree_mut().get_mut(foo).unwrap().kind = ResourceKind::Dir;

    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();
    let (sid_p, out_p) = e.attach_sub(
        SubAttachRequest::for_resource(
            "parent".into(),
            src,
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        ),
        now,
    );
    let pid_parent = e.subs().get(sid_p).unwrap().profile;
    let parent_seed = first_probe_correlation(&out_p).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_parent,
            correlation: parent_seed,
            result: ProbeResult::Ok(dir_snap(src, vec![])),
        }),
        now,
    );

    let (sid_c, out_c) = e.attach_sub(
        SubAttachRequest::for_resource(
            "child".into(),
            foo,
            cfg,
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        ),
        now,
    );
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    let child_seed = first_probe_correlation(&out_c).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_child,
            correlation: child_seed,
            result: ProbeResult::Ok(dir_snap(foo, vec![])),
        }),
        now,
    );

    // Both Profiles Idle. Trigger child's Standard burst FIRST so it
    // bumps parent.dirty_descendants to 1 before parent's Standard
    // starts; ordering matters because we need parent to enter Draining
    // (stable + dirty>0).
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t1,
    );
    // Now parent.dirty_descendants == 1. Trigger parent's Standard.
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Drain timers; both Profiles transition Settling → Probing.
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

    let parent_probe_corr = e
        .pending_probe(pid_parent)
        .expect("Verifying probe in flight");
    assert!(
        e.profiles().get(pid_parent).unwrap().dirty_descendants >= 1,
        "child burst contributes dirty before parent stabilizes",
    );

    // Inject parent's stable response while dirty>0 → Draining.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_parent,
            correlation: parent_probe_corr,
            result: ProbeResult::Ok(dir_snap(src, vec![])),
        }),
        t2,
    );
    assert!(matches!(
        e.profiles().get(pid_parent).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Draining,
            ..
        }),
    ));

    // Drive child's stable verdict — the post-stable path routes through
    // Awaiting (effect emitted) → Rebasing (post-fire probe) → Idle.
    // propagate(-1) runs at finish_burst_to_idle, i.e. at the end of
    // the *full* fire-cycle, not at the stable verdict. That keeps the
    // parent from reconfirming while the child's Effect is still
    // mutating the disk.
    let child_probe_corr = e
        .pending_probe(pid_child)
        .expect("Verifying probe in flight");
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_child,
            correlation: child_probe_corr,
            result: ProbeResult::Ok(dir_snap(foo, vec![])),
        }),
        t2,
    );
    // Stable verdict transitions to Awaiting; no reconfirm yet.
    assert!(matches!(
        e.profiles().get(pid_child).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Awaiting { outstanding: 1, .. },
            ..
        }),
    ));
    assert!(
        !stable_out
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid_parent),),
        "parent does NOT reconfirm at child stable — child's Effect still in flight",
    );
    let child_effect = stable_out
        .effects
        .first()
        .cloned()
        .expect("child fired one Effect at stable verdict");
    assert!(matches!(
        e.profiles().get(pid_parent).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Draining,
            ..
        }),
    ));

    // Inject EffectComplete::Ok → child Awaiting → Rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid_c,
            key: child_effect.key.clone(),
            result: specter_core::EffectOutcome::Ok,
        },
        t2,
    );
    let rebase_corr = e
        .pending_probe(pid_child)
        .expect("rebase probe in flight after EffectComplete");
    // The rebase probe is on the child; parent still hasn't reconfirmed.
    assert!(
        !rebase_out
            .probe_ops
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid_parent),),
        "parent does NOT reconfirm during child Rebasing — burst not yet finished",
    );

    // Inject the rebase probe response → child finish_burst_to_idle →
    // propagate(-1) → parent's reconfirm transition_to_verifying.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_child,
            correlation: rebase_corr,
            result: ProbeResult::Ok(dir_snap(foo, vec![])),
        }),
        t2,
    );

    // Parent emitted a reconfirm probe in the same step.
    let reconfirm_emitted = out
        .probe_ops
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.profile == pid_parent));
    assert!(
        reconfirm_emitted,
        "parent's reconfirm probe fires in the same step as child finish_burst_to_idle",
    );

    // Parent's state is now Probing again (the reconfirm).
    assert!(matches!(
        e.profiles().get(pid_parent).unwrap().state,
        ProfileState::Active(specter_core::Burst {
            phase: BurstPhase::Verifying,
            ..
        }),
    ));
}

#[test]
fn co_located_profiles_share_suppress_count() {
    // Two Profiles at /src; both Standard-burst. Anchor's suppress_count
    // accumulates across the two bursts; Unsuppress emits only on the
    // last-finishing burst's burst-end.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure(None, "src", ResourceRole::User);
    e.tree_mut().get_mut(r).unwrap().kind = ResourceKind::Dir;

    let cfg_a = ScanConfig::builder().recursive(true).build();
    let cfg_b = ScanConfig::builder().recursive(false).build();
    let now = Instant::now();

    let (sid_a, _) = e.attach_sub(
        SubAttachRequest::for_resource(
            "a".into(),
            r,
            cfg_a,
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        ),
        now,
    );
    let pid_a = e.subs().get(sid_a).unwrap().profile;

    let (sid_b, _) = e.attach_sub(
        SubAttachRequest::for_resource(
            "b".into(),
            r,
            cfg_b,
            MAX_SETTLE,
            SETTLE,
            empty_command(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        ),
        now,
    );
    let pid_b = e.subs().get(sid_b).unwrap().profile;

    // After both attach: suppress_count == 2 (both Profiles in Seed).
    assert_eq!(e.tree().get(r).unwrap().suppress_count, 2);

    // Drive both Seeds.
    let corr_a = e.pending_probe(pid_a).expect("Verifying probe in flight");
    let corr_b = e.pending_probe(pid_b).expect("Verifying probe in flight");

    // Finish A's Seed first; suppress goes 2→1, no Unsuppress.
    let out_a = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_a,
            correlation: corr_a,
            result: ProbeResult::Ok(dir_snap(r, vec![])),
        }),
        now,
    );
    assert_eq!(e.tree().get(r).unwrap().suppress_count, 1);
    let unsuppresses_a = out_a
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unsuppress { .. }))
        .count();
    assert_eq!(unsuppresses_a, 0, "no Unsuppress on 2→1 edge");

    // Finish B's Seed; suppress goes 1→0, Unsuppress emitted.
    let out_b = e.step(
        Input::ProbeResponse(ProbeResponse {
            profile: pid_b,
            correlation: corr_b,
            result: ProbeResult::Ok(dir_snap(r, vec![])),
        }),
        now,
    );
    assert_eq!(e.tree().get(r).unwrap().suppress_count, 0);
    let unsuppresses_b = out_b
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unsuppress { .. }))
        .count();
    assert_eq!(unsuppresses_b, 1, "Unsuppress on 1→0 edge");
}

#[test]
fn diff_unused_signals_reachable() {
    // Reference-only: keep the `Diff` symbol exercised so the import
    // doesn't go stale.
    let _: Option<&Diff> = None;
}
