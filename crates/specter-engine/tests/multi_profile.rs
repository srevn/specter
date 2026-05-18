//! Multi-Profile composition end-to-end. Two Profiles co-located on one
//! Resource share `watch_demand`/`suppress_count` via refcount
//! aggregation. The `Draining → Verifying` reconfirm is exercised
//! through the burst lifecycle: a parent that stabilises while a
//! covered descendant is mid-Standard-burst enters `Draining`, and the
//! `finish_burst_to_idle` sweep re-evaluates the fresh
//! covered-descendant query for every Draining Profile — reconfirming
//! exactly when no covered descendant remains in an Active Standard
//! burst, robust under mid-burst topology moves (interpose / reap) and
//! across a fire-tail residual restart.

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
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ActiveBurst, BurstFinish, BurstIntent, ChildEntry, ClassSet, Diff, DirChild,
    DirMeta, DirSnapshot, EffectScope, EntryKind, FsEvent, FsIdentity, Input, LeafEntry,
    OverflowScope, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase, ProbeCorrelation,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileState, ResourceId,
    ResourceKind, ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId,
    WatchOp,
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

/// V5-native helper: build a `TreeSnapshot::Dir` with flat single-component
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

fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

#[test]
fn two_profiles_one_resource_share_watch_demand() {
    // Two Profiles at the same anchor (different config_hash). After
    // both attaches: anchor.watch_demand == 2; only one Watch op was
    // emitted (the 0→1 edge).
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let cfg_a = ScanConfig::builder().recursive(true).build();
    let cfg_b = ScanConfig::builder().recursive(false).build();

    let req_a = SubAttachRequest::for_anchor(
        "build".into(),
        SubAttachAnchor::Resource(r),
        cfg_a,
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let out_a = e.step(Input::AttachSub(req_a), Instant::now());
    let watch_count_a = out_a
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    assert_eq!(watch_count_a, 1, "0→1 edge emits one Watch");

    let req_b = SubAttachRequest::for_anchor(
        "lint".into(),
        SubAttachAnchor::Resource(r),
        cfg_b,
        MAX_SETTLE,
        SETTLE,
        empty_program(),
        EffectScope::SubtreeRoot,
        NO_EVENTS,
        false,
    );
    let out_b = e.step(Input::AttachSub(req_b), Instant::now());
    let watch_count_b = out_b
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Watch { .. }))
        .count();
    assert_eq!(watch_count_b, 0, "1→2 edge emits no Watch");

    assert_eq!(e.tree().get(r).unwrap().watch_demand(), 2);
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn parent_stays_gated_across_child_fire_tail_restart() {
    // Behavioral re-expression of the deleted "ancestor refcount held
    // across the fire-tail restart". With the refcount gone, the
    // guarantee is observable as the parent's `Draining` dwell: a parent
    // that stabilised while a covered child was mid-Standard-burst sits
    // in `Draining` and must NOT reconfirm while the child cycles
    // Verifying → Awaiting → Rebasing → (residual restart) → Batching →
    // Verifying. The child never leaves the Active Standard burst — the
    // restart is an in-place move, so there is no finish-then-start
    // flicker — and the parent reconfirms exactly once, at the restarted
    // burst's single `finish_burst_to_idle`.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let foo = e
        .tree_mut()
        .ensure_child(src, "foo", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);
    let bar = e
        .tree_mut()
        .ensure_child(foo, "bar", ResourceRole::User)
        .expect("test live parent");
    // A File leaf: the restarted burst's LCA of the residual `{bar}`
    // promotes the leaf to its parent Dir, so the restart re-probes the
    // anchor `foo` — a deterministic stable + B1-dedup finish.
    e.tree_mut().set_kind(bar, ResourceKind::File);

    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();
    // The child's view of /src/foo carries `bar` so the engine covers it
    // as a descendant — an FsEvent there can then absorb. Reused for
    // every child response so all hashes match (stable verdicts + B1
    // dedup on the restarted burst).
    let child_snap = dir_snap(vec![("bar", EntryKind::File, 9)]);

    // Parent: recursive @ /src, NO_EVENTS. It covers /src/foo, so a
    // child mid-Standard-burst gates it; it bursts only from the
    // explicit FsEvent at its own anchor below.
    let out_p = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "parent".into(),
            SubAttachAnchor::Resource(src),
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
    let sid_p = specter_core::testkit::first_attached_sub(&out_p).expect("attach_sub succeeded");
    let pid_parent = e.subs().get(sid_p).unwrap().profile;
    let parent_seed = first_probe_correlation(&out_p).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Child: recursive @ /src/foo, CONTENT mask so a Modified at
    // /src/foo/bar reaches the post-fire absorb arm.
    let out_c = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "child".into(),
            SubAttachAnchor::Resource(foo),
            cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            ClassSet::CONTENT,
            false,
        )),
        now,
    );
    let sid_c = specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    let child_seed = first_probe_correlation(&out_c).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_seed,
            outcome: ProbeOutcome::SubtreeOk(child_snap.clone()),
        }),
        now,
    );

    // Child Standard burst FIRST (so it gates the parent), then the
    // parent's own Standard burst.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t1,
    );
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Drain settle timers → both Profiles reach their Verifying probe.
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

    // Parent stabilises while the child is mid-Standard-burst → Draining.
    let parent_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_parent))
        .expect("parent Verifying probe in flight");
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );

    // Closures: observe the parent's gate purely from the public surface.
    let parent_reconfirmed = |out: &StepOutput| {
        out.probe_ops().iter().any(|op| {
            matches!(op, ProbeOp::Probe { request }
                if request.owner() == ProbeOwner::Profile(pid_parent))
        })
    };
    let parent_is_draining = |eng: &Engine| {
        matches!(
            eng.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Draining,
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        )
    };
    assert!(
        parent_is_draining(&e),
        "parent enters Draining (stable verdict, child still gating)",
    );

    // Child Verifying stable → fires (Awaiting). No finish, no sweep.
    let child_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("child Verifying probe in flight");
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_corr,
            outcome: ProbeOutcome::SubtreeOk(child_snap.clone()),
        }),
        t2,
    );
    let child_effect = stable_out
        .effects()
        .first()
        .cloned()
        .expect("child fired one Effect at the stable verdict");
    assert!(
        !parent_reconfirmed(&stable_out) && parent_is_draining(&e),
        "parent does not reconfirm at the child's stable verdict",
    );

    // EffectComplete → child Rebasing. No finish, no sweep.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid_c,
            key: child_effect.key(),
            result: specter_core::EffectOutcome::Ok,
        },
        t2,
    );
    let rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("child rebase probe in flight");
    assert!(
        !parent_reconfirmed(&rebase_out) && parent_is_draining(&e),
        "parent does not reconfirm during child Rebasing",
    );

    // Descendant edit absorbed while the rebase probe is in flight — the
    // fire-tail residual.
    let t3 = t2 + Duration::from_millis(5);
    let absorb_out = e.step(
        Input::FsEvent {
            resource: bar,
            event: FsEvent::Modified,
        },
        t3,
    );
    assert!(
        !parent_reconfirmed(&absorb_out) && parent_is_draining(&e),
        "parent does not reconfirm while the residual is absorbed",
    );

    // Rebase response with a non-empty residual → child restarts
    // in-place. THE key assertion: the child never leaves the Active
    // Standard burst (no finish-then-start flicker), so the parent must
    // not reconfirm in this step and stays Draining.
    let t4 = t3 + Duration::from_millis(5);
    let restart_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(child_snap.clone()),
        }),
        t4,
    );
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Batching { .. },
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "child restarted a fresh debounced burst from the residual",
    );
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "the restarted burst is still an Active Standard burst (stays counted)",
    );
    assert!(
        !parent_reconfirmed(&restart_out) && parent_is_draining(&e),
        "parent stays gated across the in-place restart — no flicker",
    );

    // Drive the restarted burst to its single finish: Batching →
    // Verifying → stable (baseline == current, Sub already fired ⇒ B1
    // dedup, zero effects) → finish_burst_to_idle. The sweep runs here
    // for the first time since the parent entered Draining; the child is
    // now Idle, so the parent's fresh query is false → reconfirm.
    let t5 = t4 + SETTLE * 2;
    while let Some(entry) = e.pop_expired(t5) {
        e.step(
            Input::TimerExpired {
                profile: entry.profile,
                kind: entry.kind,
                id: entry.id,
            },
            t5,
        );
    }
    let restart_verify_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("restarted burst's Verifying probe in flight");
    let finish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: restart_verify_corr,
            outcome: ProbeOutcome::SubtreeOk(child_snap.clone()),
        }),
        t5,
    );
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Idle
        ),
        "restarted burst dedup-suppressed (baseline == current) and finished",
    );
    assert!(
        parent_reconfirmed(&finish_out),
        "parent reconfirms exactly once — at the restarted burst's single finish",
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying(_),
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "parent transitioned Draining → Verifying on the reconfirm",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn parent_in_draining_reconfirms_after_child_settles() {
    // Build the topology, get parent into Draining (stable while a
    // covered descendant is mid-Standard-burst), then run the child's
    // full fire cycle to completion — the `finish_burst_to_idle` sweep
    // re-evaluates the parent's fresh covered-descendant query, finds it
    // false, and emits the reconfirm probe in the same step.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let foo = e
        .tree_mut()
        .ensure_child(src, "foo", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);

    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();
    let out_p = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "parent".into(),
            SubAttachAnchor::Resource(src),
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
    let sid_p = specter_core::testkit::first_attached_sub(&out_p).expect("attach_sub succeeded");
    let pid_parent = e.subs().get(sid_p).unwrap().profile;
    let parent_seed = first_probe_correlation(&out_p).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    let out_c = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "child".into(),
            SubAttachAnchor::Resource(foo),
            cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_c = specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    let child_seed = first_probe_correlation(&out_c).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Both Profiles Idle. Trigger child's Standard burst FIRST so the
    // child is in an Active Standard burst before parent's Standard
    // starts; ordering matters because we need parent to enter Draining
    // (stable verdict while a covered descendant still gates it).
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t1,
    );
    // The child now gates the parent. Trigger parent's Standard.
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
        .pending_probe_for(ProbeOwner::Profile(pid_parent))
        .expect("Verifying probe in flight");
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child is mid-Standard-burst — this is what gates the parent into Draining",
    );

    // Inject parent's stable response while dirty>0 → Draining.
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_probe_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(matches!(
        e.profiles().get(pid_parent).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Draining,
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));

    // Drive child's stable verdict — the post-stable path routes through
    // Awaiting (effect emitted) → Rebasing (post-fire probe) → Idle. The
    // sweep only runs at finish_burst_to_idle, i.e. at the end of the
    // *full* fire-cycle, not at the stable verdict — and the child stays
    // in an Active Standard burst (post-fire counts) until then. That
    // keeps the parent from reconfirming while the child's Effect is
    // still mutating the disk.
    let child_probe_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("Verifying probe in flight");
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_probe_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    // Stable verdict transitions to Awaiting; no reconfirm yet.
    assert!(matches!(
        e.profiles().get(pid_child).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Awaiting { outstanding: 1, .. },
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    assert!(
        !stable_out
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_parent)),),
        "parent does NOT reconfirm at child stable — child's Effect still in flight",
    );
    let child_effect = stable_out
        .effects()
        .first()
        .cloned()
        .expect("child fired one Effect at stable verdict");
    assert!(matches!(
        e.profiles().get(pid_parent).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Draining,
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));

    // Inject EffectComplete::Ok → child Awaiting → Rebasing.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid_c,
            key: child_effect.key(),
            result: specter_core::EffectOutcome::Ok,
        },
        t2,
    );
    let rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("rebase probe in flight after EffectComplete");
    // The rebase probe is on the child; parent still hasn't reconfirmed.
    assert!(
        !rebase_out
            .probe_ops()
            .iter()
            .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_parent)),),
        "parent does NOT reconfirm during child Rebasing — burst not yet finished",
    );

    // Inject the rebase probe response → child finish_burst_to_idle →
    // the Draining sweep finds the parent's covered-descendant query now
    // false → parent's reconfirm transition_to_verifying.
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );

    // Parent emitted a reconfirm probe in the same step.
    let reconfirm_emitted = out
        .probe_ops()
        .iter()
        .any(|op| matches!(op, ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_parent)));
    assert!(
        reconfirm_emitted,
        "parent's reconfirm probe fires in the same step as child finish_burst_to_idle",
    );

    // Parent's state is now Probing again (the reconfirm).
    assert!(matches!(
        e.profiles().get(pid_parent).unwrap().state(),
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Verifying(_),
                ..
            }),
            BurstFinish::ReturnToIdle
        ),
    ));
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn interposing_covering_profile_mid_burst_does_not_strand_draining_ancestor() {
    // F-CRIT-1 regression. A covered child is mid-Standard-burst; its
    // Draining ancestor's covering chain is rewritten by a hot-reload
    // attach that interposes a new covering Profile between them. The
    // deleted refcount took its `+1` against the *old* chain and its
    // `-1` against the *new* one — a dev `dirty_descendants underflow`
    // panic on the interposed Profile, or a release strand of the
    // ancestor's count `> 0` forever. With the refcount gone and the
    // exit driven by a fresh sweep query, the topology move is inert:
    // the ancestor reconfirms when the child finishes, no panic, no
    // strand.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let mid = e
        .tree_mut()
        .ensure_child(src, "mid", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(mid, ResourceKind::Dir);
    let foo = e
        .tree_mut()
        .ensure_child(mid, "foo", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);

    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();

    // Parent @ /src (recursive) — covers /src/mid/foo, no Profile at
    // /src/mid yet, so the child's covering chain is child → parent.
    let out_p = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "parent".into(),
            SubAttachAnchor::Resource(src),
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
    let sid_p = specter_core::testkit::first_attached_sub(&out_p).expect("attach_sub succeeded");
    let pid_parent = e.subs().get(sid_p).unwrap().profile;
    let parent_seed = first_probe_correlation(&out_p).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    let out_c = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "child".into(),
            SubAttachAnchor::Resource(foo),
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
    let sid_c = specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    let child_seed = first_probe_correlation(&out_c).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Child Standard burst FIRST (so it gates the parent), then parent's.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t1,
    );
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Modified,
        },
        t1,
    );

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

    // Parent stabilises while the child gates it → Draining.
    let parent_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_parent))
        .expect("parent Verifying probe in flight");
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Draining,
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "parent enters Draining",
    );

    // HOT-RELOAD INTERPOSE: attach a new covering Profile at /src/mid
    // while the child is mid-burst and the parent is Draining. This is
    // the exact mid-burst topology move that desynced the old refcount's
    // `+1` / `-1` chain walks.
    let out_m = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "mid".into(),
            SubAttachAnchor::Resource(mid),
            cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        t2,
    );
    let sid_m = specter_core::testkit::first_attached_sub(&out_m).expect("attach_sub succeeded");
    let pid_mid = e.subs().get(sid_m).unwrap().profile;
    assert!(
        !out_m.probe_ops().iter().any(|op| matches!(
            op,
            ProbeOp::Probe { request } if request.owner() == ProbeOwner::Profile(pid_parent)
        )),
        "the interpose itself must not reconfirm the parent",
    );
    // Interposed Profile is in its own Seed burst — not Standard, so it
    // does not itself gate the parent.
    assert!(
        !e.profiles()
            .get(pid_mid)
            .unwrap()
            .state()
            .in_active_standard_burst(),
    );

    // Drive the child through its full fire cycle to completion. No
    // reconfirm until the child actually finishes.
    let child_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("child Verifying probe in flight");
    let stable_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    let child_effect = stable_out
        .effects()
        .first()
        .cloned()
        .expect("child fired one Effect at the stable verdict");
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid_c,
            key: child_effect.key(),
            result: specter_core::EffectOutcome::Ok,
        },
        t2,
    );
    let rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("child rebase probe in flight");
    let parent_reconfirmed = |out: &StepOutput| {
        out.probe_ops().iter().any(|op| {
            matches!(op, ProbeOp::Probe { request }
                if request.owner() == ProbeOwner::Profile(pid_parent))
        })
    };
    assert!(
        !parent_reconfirmed(&stable_out) && !parent_reconfirmed(&rebase_out),
        "parent does not reconfirm until the child's burst finishes",
    );

    // Child rebase response → child finish_burst_to_idle. The sweep
    // re-evaluates the parent's fresh query (child now Idle, interposed
    // Profile only Seed) → false → parent reconfirms. No panic (the old
    // `dirty_descendants underflow` debug_assert is gone), no strand.
    let finish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(
        parent_reconfirmed(&finish_out),
        "parent reconfirms at the child's finish despite the interposed Profile",
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying(_),
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "parent is not stranded — it transitioned Draining → Verifying",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn sweep_reconfirms_draining_ancestor_off_the_finishers_chain() {
    // The §3 deeper-layer guard: the Draining-exit trigger is a sweep of
    // *every* Draining Profile, not a walk of the finishing Profile's
    // covering chain. `A`(/src, max_depth=1) is gated into Draining by a
    // deep descendant `P`(/src/mid/foo) *via the intermediate broader*
    // `B`(/src/mid): A's chain to P is P → B → A. `B` is then reaped
    // (its Sub detached) while A is still Draining, so P's live chain
    // collapses to P → (nothing) — A no longer covers `foo` directly
    // (max_depth=1), so A is unreachable from the finisher. A
    // chain-coupled trigger would strand A forever; the sweep re-checks
    // A directly and reconfirms it.
    //
    // Event isolation is by the anchor-bypass / class-filter rule, not
    // by coverage tricks: a `Modified` event is CONTENT-class, and a
    // NO_EVENTS Profile only bursts from an event at its *own anchor*
    // (descendant events of an unmatched class are dropped). So the
    // `foo` event drives only P, and the `src` event only A.
    let mut e = Engine::new();
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let mid = e
        .tree_mut()
        .ensure_child(src, "mid", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(mid, ResourceKind::Dir);
    let foo = e
        .tree_mut()
        .ensure_child(mid, "foo", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(foo, ResourceKind::Dir);

    let now = Instant::now();
    // A covers /src/mid (depth 1) but NOT /src/mid/foo (depth 2): once
    // the intermediate B is gone, A is off P's chain.
    let a_cfg = ScanConfig::builder()
        .recursive(true)
        .max_depth(Some(1))
        .build();
    let unbounded = ScanConfig::builder().recursive(true).build();

    let out_a = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "a".into(),
            SubAttachAnchor::Resource(src),
            a_cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_a = specter_core::testkit::first_attached_sub(&out_a).expect("attach_sub succeeded");
    let pid_a = e.subs().get(sid_a).unwrap().profile;
    let a_seed = first_probe_correlation(&out_a).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_a),
            correlation: a_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // B @ /src/mid (recursive): covers /src/mid/foo, so it sits on P's
    // chain (P → B → A). It never bursts — the only event under it is
    // the `foo` Modified, a descendant CONTENT event its NO_EVENTS mask
    // drops — so DetachSub reaps it immediately.
    let out_b = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "b".into(),
            SubAttachAnchor::Resource(mid),
            unbounded.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_b = specter_core::testkit::first_attached_sub(&out_b).expect("attach_sub succeeded");
    let pid_b = e.subs().get(sid_b).unwrap().profile;
    let b_seed = first_probe_correlation(&out_b).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_b),
            correlation: b_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // P @ /src/mid/foo (recursive).
    let out_pp = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "p".into(),
            SubAttachAnchor::Resource(foo),
            unbounded,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_p = specter_core::testkit::first_attached_sub(&out_pp).expect("attach_sub succeeded");
    let pid_p = e.subs().get(sid_p).unwrap().profile;
    let p_seed = first_probe_correlation(&out_pp).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // P's Standard burst from its own anchor `foo` (drives only P), then
    // A's own Standard burst from its anchor `src`.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: foo,
            event: FsEvent::Modified,
        },
        t1,
    );
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Modified,
        },
        t1,
    );
    assert!(
        matches!(e.profiles().get(pid_b).unwrap().state(), ProfileState::Idle),
        "B stays Idle (the foo event is a descendant CONTENT event its \
         NO_EVENTS mask drops)",
    );

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

    // A stabilises while P gates it through the chain P → B → A → A
    // enters Draining.
    let a_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_a))
        .expect("A Verifying probe in flight");
    assert!(
        e.profiles()
            .get(pid_p)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "P gates A (via the intermediate B) before A stabilises",
    );
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_a),
            correlation: a_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Draining,
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "A enters Draining (gated through the intermediate B)",
    );

    // Reap the intermediate B (Idle ⇒ immediate reap). P's live chain
    // collapses to P → (nothing): A does not cover `foo` directly
    // (max_depth=1), so A is OFF the finisher's chain.
    let detach_out = e.step(Input::DetachSub(sid_b), t2);
    assert!(
        e.profiles().get(pid_b).is_none(),
        "B reaped immediately (it was Idle)",
    );
    let a_reconfirmed = |out: &StepOutput| {
        out.probe_ops().iter().any(|op| {
            matches!(op, ProbeOp::Probe { request }
                if request.owner() == ProbeOwner::Profile(pid_a))
        })
    };
    assert!(
        !a_reconfirmed(&detach_out) && e.profiles().get(pid_a).unwrap().state().is_draining(),
        "reaping B does not itself reconfirm A — A is still gated by the \
         still-bursting P, just no longer through a chain that reaches it",
    );

    // Drive P through its full fire cycle. At P's single
    // finish_burst_to_idle the sweep re-checks *every* Draining Profile;
    // A is found and reconfirmed even though P's chain no longer reaches
    // it. (A chain-coupled trigger would strand A here forever.)
    let p_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_p))
        .expect("P Verifying probe in flight");
    let p_stable = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    let p_effect = p_stable
        .effects()
        .first()
        .cloned()
        .expect("P fired one Effect at the stable verdict");
    let p_rebase_out = e.step(
        Input::EffectComplete {
            sub: sid_p,
            key: p_effect.key(),
            result: specter_core::EffectOutcome::Ok,
        },
        t2,
    );
    let p_rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_p))
        .expect("P rebase probe in flight");
    assert!(
        !a_reconfirmed(&p_stable) && !a_reconfirmed(&p_rebase_out),
        "A does not reconfirm until P's burst actually finishes",
    );
    let p_finish = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_p),
            correlation: p_rebase_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(
        a_reconfirmed(&p_finish),
        "the sweep reconfirms A even though P's chain no longer reaches it",
    );
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying(_),
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "A is not stranded — sweep drove Draining → Verifying",
    );
    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn co_located_profiles_share_suppress_count() {
    // Two Profiles at /src; both Standard-burst. Anchor's suppress_count
    // accumulates across the two bursts; Unsuppress emits only on the
    // last-finishing burst's burst-end.
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);

    let cfg_a = ScanConfig::builder().recursive(true).build();
    let cfg_b = ScanConfig::builder().recursive(false).build();
    let now = Instant::now();

    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "a".into(),
            SubAttachAnchor::Resource(r),
            cfg_a,
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

    let attach_out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "b".into(),
            SubAttachAnchor::Resource(r),
            cfg_b,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_b =
        specter_core::testkit::first_attached_sub(&attach_out).expect("attach_sub succeeded");
    let pid_b = e.subs().get(sid_b).unwrap().profile;

    // After both attach: suppress_count == 2 (both Profiles in Seed).
    assert_eq!(e.tree().get(r).unwrap().suppress_count(), 2);

    // Drive both Seeds.
    let corr_a = e
        .pending_probe_for(ProbeOwner::Profile(pid_a))
        .expect("Verifying probe in flight");
    let corr_b = e
        .pending_probe_for(ProbeOwner::Profile(pid_b))
        .expect("Verifying probe in flight");

    // Finish A's Seed first; suppress goes 2→1, no Unsuppress.
    let out_a = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_a),
            correlation: corr_a,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );
    assert_eq!(e.tree().get(r).unwrap().suppress_count(), 1);
    let unsuppresses_a = out_a
        .watch_ops
        .iter()
        .filter(|op| matches!(op, WatchOp::Unsuppress { .. }))
        .count();
    assert_eq!(unsuppresses_a, 0, "no Unsuppress on 2→1 edge");

    // Finish B's Seed; suppress goes 1→0, Unsuppress emitted.
    let out_b = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_b),
            correlation: corr_b,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );
    assert_eq!(e.tree().get(r).unwrap().suppress_count(), 0);
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

/// Fixture for the sensor-overflow × Draining-ancestor scenarios. A
/// recursive parent Profile `A` at `/src` is parked in `Draining`,
/// gated by its covered child Profile `D` at `/src/child` which is
/// mid-Standard-burst (its Verify probe still in flight).
struct DrainingFixture {
    e: Engine,
    src: ResourceId,
    child_dir: ResourceId,
    pid_parent: ProfileId,
    pid_child: ProfileId,
    sid_parent: SubId,
    sid_child: SubId,
    /// The fixture clock at the point the parent reached `Draining`;
    /// subsequent steps reuse it so the scenario stays deterministic.
    t2: Instant,
}

/// Build [`DrainingFixture`]. The child Profile is attached *first* so
/// its `ProfileId` takes the earlier slotmap slot —
/// `Engine::profiles().iter()` (the very iterator the Global overflow
/// snapshot consumes) then yields the descendant `D` before the
/// ancestor `A`. That descendant-before-ancestor order is the one that
/// exercises the sweep↔reseed-loop seam: `finish_burst_to_idle(D)`'s
/// Draining sweep flips `A` `Draining→Verifying` *before* a reseed loop
/// would otherwise reach `A`. The order is asserted, not assumed, so a
/// future slotmap change cannot silently make the fixture order-lucky.
fn draining_parent_gated_by_child() -> DrainingFixture {
    let mut e = Engine::new();
    let src = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(src, ResourceKind::Dir);
    let child_dir = e
        .tree_mut()
        .ensure_child(src, "child", ResourceRole::User)
        .expect("test live parent");
    e.tree_mut().set_kind(child_dir, ResourceKind::Dir);

    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();

    // Child `D` @ /src/child FIRST — earlier slot ⇒ iterates before `A`.
    let out_c = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "child".into(),
            SubAttachAnchor::Resource(child_dir),
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
    let sid_child =
        specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = e.subs().get(sid_child).unwrap().profile;
    let child_seed = first_probe_correlation(&out_c).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: child_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // Parent `A` @ /src — recursive, so it covers /src/child.
    let out_p = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "parent".into(),
            SubAttachAnchor::Resource(src),
            cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_parent =
        specter_core::testkit::first_attached_sub(&out_p).expect("attach_sub succeeded");
    let pid_parent = e.subs().get(sid_parent).unwrap().profile;
    let parent_seed = first_probe_correlation(&out_p).unwrap();
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_seed,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        now,
    );

    // The descendant-before-ancestor premise — asserted, not assumed.
    let order: Vec<ProfileId> = e.profiles().iter().map(|(id, _)| id).collect();
    let pos = |target: ProfileId| {
        order
            .iter()
            .position(|&id| id == target)
            .expect("profile present in iteration")
    };
    assert!(
        pos(pid_child) < pos(pid_parent),
        "fixture premise: covered child must iterate before its ancestor \
         (descendant-before-ancestor is the order that drives the \
         sweep↔reseed-loop seam)",
    );

    // Child Standard burst from its own anchor `child_dir`; parent
    // Standard burst from its own anchor `src`. A NO_EVENTS Profile
    // only bursts from an event at its own anchor, so the two events
    // stay isolated.
    let t1 = now + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: child_dir,
            event: FsEvent::Modified,
        },
        t1,
    );
    e.step(
        Input::FsEvent {
            resource: src,
            event: FsEvent::Modified,
        },
        t1,
    );

    // Drain settle timers → both Profiles reach their Verify probe.
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

    // Parent stabilises while the child still gates it → Draining.
    let parent_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_parent))
        .expect("parent Verifying probe in flight");
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: parent_corr,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Draining,
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "fixture: parent parked in Draining",
    );
    assert!(
        e.pending_probe_for(ProbeOwner::Profile(pid_child))
            .is_some(),
        "fixture: child's Verify probe still in flight (gating descendant)",
    );

    DrainingFixture {
        e,
        src,
        child_dir,
        pid_parent,
        pid_child,
        sid_parent,
        sid_child,
        t2,
    }
}

#[test]
fn global_overflow_excludes_draining_ancestor_keeps_reconfirm() {
    // F-CRIT-1 / F-HIGH-2 regression. A Global sensor overflow lands
    // while `A` is `Draining` and its covered child `D` is mid-burst,
    // with `D` iterating before `A`. The overflow loop processes `D`
    // first: `finish_burst_to_idle(D)`'s Draining sweep flips `A`
    // `Draining→Verifying` and arms exactly one reconfirm Probe. Without
    // the snapshot-time Draining exclusion the loop then reaches `A`
    // (now `Verifying`, so an iteration-time phase guard never sees
    // `Draining`), tears it down, and reseeds it — discarding `A`'s
    // verified-stable `current` and the descendant-driven reconfirm. The
    // snapshot-time exclusion removes `A` from the loop entirely, so the
    // sweep's single reconfirm stands.
    let DrainingFixture {
        mut e,
        pid_parent,
        pid_child,
        t2,
        ..
    } = draining_parent_gated_by_child();

    assert!(
        e.profiles().get(pid_parent).unwrap().current().is_some(),
        "fixture: Draining ancestor holds a verified-stable `current`",
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t2,
    );

    // At most one probe op for the ancestor, and it is the sweep's
    // reconfirm Probe — never a second same-owner emit.
    let owner = ProbeOwner::Profile(pid_parent);
    let parent_ops: Vec<&ProbeOp> = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner)
        .collect();
    assert_eq!(
        parent_ops.len(),
        1,
        "exactly one probe op for the Draining ancestor (≤1-per-owner)",
    );
    assert!(
        matches!(parent_ops[0], ProbeOp::Probe { .. }),
        "the single ancestor op is the sweep's reconfirm Probe, not a Cancel",
    );

    // DISCRIMINATOR. Post-fix `A` stays on the descendant-driven
    // Standard reconfirm the sweep armed. Pre-fix the reseed loop tears
    // it down and it returns as `Seed`. `intent == Standard` is the
    // assertion that fails without the snapshot-time exclusion.
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying(_),
                    intent: BurstIntent::Standard,
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "Draining ancestor reconfirms as Standard — NOT reseeded to Seed",
    );

    // The verified-stable snapshot survived: a reseed would have routed
    // `A` through `finish_burst_to_idle`, discarding it.
    assert!(
        e.profiles().get(pid_parent).unwrap().current().is_some(),
        "ancestor's verified `current` preserved across overflow",
    );

    // Surgical: the in-scope, non-Draining descendant IS still reseeded
    // — the exclusion is Draining-only, not a blanket overflow skip.
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    intent: BurstIntent::Seed,
                    ..
                }),
                _
            ),
        ),
        "non-Draining descendant still reseeded (exclusion is Draining-only)",
    );

    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn resource_overflow_excludes_draining_ancestor_keeps_reconfirm() {
    // The Resource-scoped arm of the same exclusion. A covered
    // descendant's anchor is always a strict tree-descendant of its
    // ancestor's anchor, and `OverflowScope::Resource(r)` selects the
    // whole subtree rooted at `r` — so a Draining ancestor and its
    // gating descendant are *necessarily* both in scope when `r` is the
    // ancestor's anchor; the descendant cannot be scoped out while the
    // ancestor is scoped in. This pins the `profiles_in_subtree`
    // snapshot path (distinct from the Global `profiles().iter()` path)
    // so the exclusion cannot silently regress to a Global-only filter.
    // The scope-independent properties (≤1-per-owner, `current`
    // preserved, descendant still reseeded) are proven by the Global
    // test and deliberately not repeated here.
    let DrainingFixture {
        mut e,
        src,
        pid_parent,
        t2,
        ..
    } = draining_parent_gated_by_child();

    e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Resource(src),
        },
        t2,
    );

    // The discriminator, reached through the Resource snapshot path:
    // the in-scope Draining ancestor is excluded from the reseed and
    // stays on the sweep's Standard reconfirm instead of returning Seed.
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying(_),
                    intent: BurstIntent::Standard,
                    ..
                }),
                BurstFinish::ReturnToIdle
            ),
        ),
        "in-scope Draining ancestor reconfirms as Standard — NOT reseeded",
    );

    let _ = e.cancel_all_in_flight_probes();
}

#[test]
fn overflow_on_draining_reap_ancestor_defers_reap_to_reconfirm() {
    // Reap sub-shape. The Draining ancestor's sole Sub is detached
    // mid-Draining, flipping its burst-finish directive to a deferred
    // `Reap`. A Global overflow then arrives. Without the snapshot-time
    // exclusion the reseed loop reaches the ancestor (the sweep already
    // moved it `Draining→Verifying`), takes its `will_reap` branch, and
    // `finish_burst_to_idle` honours the directive — the Profile is
    // reaped *inside the overflow step*, before its descendant-driven
    // reconfirm could run. With the exclusion the ancestor stays out of
    // the loop: the sweep arms its lone reconfirm Probe, the `Reap`
    // directive rides through unchanged, and the reap is correctly
    // deferred to that reconfirm's resolution (the standard
    // Draining-exit contract, covered elsewhere).
    let DrainingFixture {
        mut e,
        pid_parent,
        pid_child,
        sid_parent,
        t2,
        ..
    } = draining_parent_gated_by_child();

    // Detach the ancestor's sole Sub mid-Draining → deferred Reap.
    let _ = e.step(Input::DetachSub(sid_parent), t2);
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Draining,
                    ..
                }),
                BurstFinish::Reap
            ),
        ),
        "fixture: ancestor is Draining with a deferred Reap directive",
    );
    assert_eq!(
        e.profiles().get(pid_parent).unwrap().state().burst_finish(),
        Some(BurstFinish::Reap),
    );

    let out = e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Global,
        },
        t2,
    );

    // DISCRIMINATOR. Post-fix the ancestor is NOT reaped inside the
    // overflow step — it survives as the sweep's reconfirm with the
    // Reap directive intact. Pre-fix the loop's `will_reap` branch
    // reaps it here, so the Profile would already be gone.
    assert!(
        e.profiles().get(pid_parent).is_some(),
        "Draining+Reap ancestor NOT reaped inside the overflow step",
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying(_),
                    intent: BurstIntent::Standard,
                    ..
                }),
                BurstFinish::Reap
            ),
        ),
        "ancestor reconfirms as Standard with the Reap directive preserved",
    );

    let owner = ProbeOwner::Profile(pid_parent);
    let parent_ops: Vec<&ProbeOp> = out
        .probe_ops()
        .iter()
        .filter(|op| op.owner() == owner)
        .collect();
    assert_eq!(parent_ops.len(), 1, "one probe op for the ancestor");
    assert!(
        matches!(parent_ops[0], ProbeOp::Probe { .. }),
        "the single ancestor op is the sweep's reconfirm Probe (not a reap Cancel)",
    );

    // The deferred reap completes on the reconfirm's resolution: the
    // child reseed left it non-gating (Seed), so the reconfirm settles
    // and `finish_burst_to_idle` honours `Reap`.
    let reconfirm = e
        .pending_probe_for(ProbeOwner::Profile(pid_parent))
        .expect("ancestor reconfirm probe in flight");
    let _ = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_parent),
            correlation: reconfirm,
            outcome: ProbeOutcome::SubtreeOk(dir_snap(vec![])),
        }),
        t2,
    );
    assert!(
        e.profiles().get(pid_parent).is_none(),
        "deferred Reap honoured once the reconfirm resolved — ancestor reaped",
    );
    let _ = e.profiles().get(pid_child); // child untouched by the reap

    let _ = e.cancel_all_in_flight_probes();
}
