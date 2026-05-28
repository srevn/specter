//! Multi-Profile composition end-to-end. Two Profiles co-located on one
//! Resource share `watch_demand` via refcount aggregation. The
//! `Draining → Verifying` reconfirm is exercised
//! through the burst lifecycle: a parent that stabilises while a
//! covered descendant is mid-Standard-burst enters `Draining`, and the
//! `finish_burst_to_idle` sweep re-evaluates the fresh
//! covered-descendant query for every Draining Profile — reconfirming
//! exactly when no covered descendant remains in an Active Standard
//! burst, robust under mid-burst topology moves (interpose / reap) and
//! across a fire-tail residual restart.

use specter_core::testkit::{dir_snap, empty_program};
use specter_core::{
    ActiveBurst, BurstFinish, BurstIntent, ClassSet, EffectCompletion, EffectScope, EntryKind,
    FsEvent, Input, OverflowScope, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase,
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileState, ProofAuthority,
    ResourceId, ResourceKind, ResourceRole, ScanConfig, SubAttachAnchor, SubAttachRequest, SubId,
    TimerKind, WatchOp,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    DEFAULT_EVENTS, MAX_SETTLE, NO_EVENTS, SETTLE, drain_due, post_fire_settle_id,
    rebase_post_fire_to_idle, reconfirm_probed, seed_to_idle, verify,
};
use std::time::{Duration, Instant};

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
    // A parent that stabilised while a covered child was
    // mid-Standard-burst sits in `Draining`; its exit is the fresh
    // `coverage::has_active_standard_descendant` sweep run at every
    // `finish_burst_to_idle` — a query, not a cached counter. The parent
    // must NOT reconfirm while the child cycles Verifying → Awaiting →
    // Settling → Rebasing → (residual restart) → Batching → Verifying:
    // the child never leaves the Active Standard burst — the residual
    // restart is an in-place PostFire→PreFire move, so there is no
    // finish-then-start flicker for the sweep to misread as a gap — and
    // the parent reconfirms exactly once, at the restarted burst's single
    // `finish_burst_to_idle`.
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
    let child_snap = dir_snap(&[("bar", EntryKind::File, 9)]);

    // Parent: recursive @ /src, NO_EVENTS — must NOT receive descendant
    // events (the test's later residual-absorb step drives only the child).
    // events_witness_quiescence == false on this mask, so the parent's
    // Standard verify owes the N=2 hash-equality channel: two consecutive
    // Authoritative samples must agree. Driven inline below.
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
    let pid_parent = e.subs().get(sid_p).unwrap().profile();
    // The parent's cold-arm Seed pins on one Authoritative response —
    // drive it to a settled `Idle` baseline before the child attach so
    // the child's Seed has a clean settle window.
    let parent_seed_done = seed_to_idle(&mut e, pid_parent, &dir_snap(&[]), now);

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
    let pid_child = e.subs().get(sid_c).unwrap().profile();
    // Drive the child's cold-arm Seed to Idle past the parent's
    // consumed settle windows so the Standard timeline below is
    // quiet-window-clean.
    let child_seed_done = seed_to_idle(&mut e, pid_child, &child_snap, parent_seed_done);

    // Child Standard burst FIRST (so it gates the parent), then the
    // parent's own Standard burst. Both Seeds consumed their settle
    // windows above; rebase the Standard timeline past them.
    let t1 = child_seed_done + Duration::from_millis(10);
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
    drain_due(&mut e, t2);

    // Closures: observe the parent's gate purely from the public surface.
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

    // Parent verifies and stabilises while the child is still
    // mid-Standard-burst → `Draining` (gated by the covered child). The
    // parent is on a NO_EVENTS mask, so its Standard verify owes the N=2
    // hash channel: first sample → Retry (re-Batching, prior=None);
    // second sample (settle-spaced, same hash) → Stable; the gate then
    // diverts it to Draining because `has_active_standard_descendant`
    // is true.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    let v_p_first = verify(&mut e, pid_parent, &dir_snap(&[]), t2);
    // First sample re-batches; expire the freshly-armed settle timer
    // to advance back to Verifying for the second sample.
    let parent_parked_at = v_p_first.at + SETTLE * 2;
    drain_due(&mut e, parent_parked_at);
    let v_p = verify(&mut e, pid_parent, &dir_snap(&[]), parent_parked_at);
    let parent_parked_at = v_p.at;
    assert!(
        !reconfirm_probed(&v_p.out, pid_parent),
        "Draining-divert emits no reconfirm probe (parent went Verifying → Draining)",
    );
    assert!(
        parent_is_draining(&e),
        "parent enters Draining (stable verdict gated by the child)",
    );

    // Child verifies stable → fires (`Awaiting`). The parent is parked
    // in Draining holding only its `BurstDeadline`, so the child's
    // verify response leaves the parent untouched.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child still gates the parent before it stabilises",
    );
    let v_c = verify(&mut e, pid_child, &child_snap, parent_parked_at);
    let child_parked_at = v_c.at;
    let child_effect = v_c
        .out
        .effects()
        .first()
        .cloned()
        .expect("child fired one Effect at the stable verdict");
    assert!(
        !reconfirm_probed(&v_c.out, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm at the child's stable verdict",
    );

    // Rebase tail kept inline (irregular: final-window absorb + residual
    // restart — not the clean rebase_post_fire_to_idle shape).
    //
    // EffectComplete (last completion) routes Awaiting → Settling; the
    // rebase probe is minted on PostFireSettle expiry. No finish, no
    // sweep — the child stays Active(PostFire) for the whole loop.
    let settle_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid_c,
            key: child_effect.key(),
            outcome: specter_core::EffectOutcome::Ok,
        }),
        child_parked_at,
    );
    let settle_timer = post_fire_settle_id(&e, pid_child);
    assert!(
        !reconfirm_probed(&settle_out, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm at EffectComplete (Settling has no probe)",
    );

    // PostFireSettle expiry (by id — scoped to the child, parent
    // untouched) → child enters Rebasing with a fresh probe in flight.
    let t_rebase = child_parked_at + SETTLE;
    let rebase_out = e.step(
        Input::TimerExpired {
            profile: pid_child,
            kind: TimerKind::PostFireSettle,
            id: settle_timer,
        },
        t_rebase,
    );
    let rebase_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("PostFireSettle expiry mints the child's Rebasing probe");
    assert!(
        !reconfirm_probed(&rebase_out, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm across PostFireSettle → Rebasing",
    );
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PostFire(PostFireBurst {
                    phase: PostFirePhase::Rebasing(_),
                    ..
                }),
                _,
            ),
        ),
        "child entered Active(PostFire(Rebasing)) on PostFireSettle expiry",
    );

    // Descendant edit absorbed during the Rebasing round-trip — the
    // final-window residual that seeds the restart. `last_event_time`
    // is also updated; no PostFireSettle is in flight while Rebasing,
    // so the absorb writes purely into the residual.
    let t_absorb = t_rebase + Duration::from_millis(5);
    let absorb_out = e.step(
        Input::FsEvent {
            resource: bar,
            event: FsEvent::Modified,
        },
        t_absorb,
    );
    assert!(
        !reconfirm_probed(&absorb_out, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm while the residual is absorbed",
    );

    // Authoritative ⇒ commit + rebase_baseline; non-empty residual +
    // ReturnToIdle ⇒ restart_burst_from_fire_tail_residual. THE key
    // assertion: the child never leaves the Active Standard burst (no
    // finish-then-start flicker), so the parent must not reconfirm in
    // this step and stays Draining.
    let t_restart = t_absorb + Duration::from_millis(5);
    let restart_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: rebase_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: child_snap.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_restart,
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
        !reconfirm_probed(&restart_out, pid_parent) && parent_is_draining(&e),
        "parent stays gated across the in-place restart — no flicker",
    );

    // Drive the restarted burst to its single finish. The restarted
    // Standard burst's first Authoritative response folds to a single
    // fire decision: baseline.hash() == current.hash() (the rebase
    // just synced them) AND Sub.has_fired == true ⇒ every Sub
    // suppresses via B1 dedup, emit_effects returns count == 0, and
    // fire_and_settle short-circuits to finish_burst_to_idle. One
    // walk, one response, then finish. The Draining sweep at that
    // finish flips the parent Draining → Verifying.
    let t_rdrain = t_restart + SETTLE;
    drain_due(&mut e, t_rdrain);
    let verify_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("restarted burst's Verifying probe in flight");
    let finish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: verify_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: child_snap.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_rdrain,
    );
    assert!(
        finish_out.effects().is_empty(),
        "restarted-burst Authoritative is B1-dedup-suppressed (baseline == current)",
    );
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Idle
        ),
        "restarted burst finished to Idle (zero effects ⇒ direct finish)",
    );
    assert!(
        reconfirm_probed(&finish_out, pid_parent),
        "parent reconfirms exactly once — at the restarted burst's single finish",
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying { .. },
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
fn interposing_covering_profile_mid_burst_does_not_strand_draining_ancestor() {
    // A covered child is mid-Standard-burst; its Draining ancestor's
    // covering chain is rewritten by a hot-reload attach that interposes
    // a new covering Profile between them. The ancestor's Draining exit
    // is the fresh `coverage::{has_active_standard_descendant,
    // chain_reaches}` query recomputed at every `finish_burst_to_idle`,
    // so the interpose changes the covering chain's *path* but not the
    // *truth* of the query: the ancestor reconfirms iff no covered
    // descendant is in an Active Standard burst at the child's finish —
    // topology-move-invariant by construction, with no cached counter to
    // desync. So the mid-burst interpose is inert: the ancestor
    // reconfirms when the child finishes, no panic, no strand.
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
    // DEFAULT_EVENTS so the parent's single Authoritative verify folds to
    // Stable (the gated-Draining path the test pins).
    let out_p = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "parent".into(),
            SubAttachAnchor::Resource(src),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            DEFAULT_EVENTS,
            false,
        )),
        now,
    );
    let sid_p = specter_core::testkit::first_attached_sub(&out_p).expect("attach_sub succeeded");
    let pid_parent = e.subs().get(sid_p).unwrap().profile();
    // Drive the parent's Seed to a pinned `Idle` baseline (per-Profile,
    // by-id timer steps so the child Seed below stays untouched).
    let parent_seed_done = seed_to_idle(&mut e, pid_parent, &dir_snap(&[]), now);

    let out_c = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "child".into(),
            SubAttachAnchor::Resource(foo),
            cfg.clone(),
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            DEFAULT_EVENTS,
            false,
        )),
        now,
    );
    let sid_c = specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = e.subs().get(sid_c).unwrap().profile();
    // Drive the child's Seed, rebased strictly past the parent
    // Seed's consumed settle window.
    let child_seed_done = seed_to_idle(&mut e, pid_child, &dir_snap(&[]), parent_seed_done);

    // Child Standard burst FIRST (so it gates the parent), then parent's.
    // The Seed drives pushed the clock past `now`; rebase the Standard
    // timeline past them.
    let t1 = child_seed_done + Duration::from_millis(10);
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
    drain_due(&mut e, t2);

    // Parent stabilises while the child gates it → Draining. The parent's
    // Standard burst folds its single Authoritative verify response to
    // `Stable` directly, and the child still being mid-burst routes the
    // parent through `transition_to_draining` instead of firing.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    let v_parent = verify(&mut e, pid_parent, &dir_snap(&[]), t2);
    let parent_parked_at = v_parent.at;
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
        parent_parked_at,
    );
    let sid_m = specter_core::testkit::first_attached_sub(&out_m).expect("attach_sub succeeded");
    let pid_mid = e.subs().get(sid_m).unwrap().profile();
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

    // Drive the child through its full fire cycle. The parent is parked
    // in Draining holding only its BurstDeadline, and the interposed
    // `mid` Profile sits in Seed with no expirable settle timer, so the
    // child's settle drain disturbs neither. No reconfirm until the
    // child finishes.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child still gates the Draining parent before it stabilises",
    );
    let v_child = verify(&mut e, pid_child, &dir_snap(&[]), parent_parked_at);
    let child_parked_at = v_child.at;
    let stable_out = v_child.out;
    let child_effect = stable_out
        .effects()
        .first()
        .cloned()
        .expect("child fired one Effect at the stable verdict");
    let rebase_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid_c,
            key: child_effect.key(),
            outcome: specter_core::EffectOutcome::Ok,
        }),
        child_parked_at,
    );
    assert!(
        !reconfirm_probed(&stable_out, pid_parent) && !reconfirm_probed(&rebase_out, pid_parent),
        "parent does not reconfirm until the child's burst finishes",
    );

    // Child post-fire rebase loop: PostFireSettle expiry → Rebasing →
    // response → commit. The child stays Active(PostFire) for the whole
    // loop, so the parent does not reconfirm until the child's single
    // finish_burst_to_idle.
    let r = rebase_post_fire_to_idle(&mut e, pid_child, &dir_snap(&[]), child_parked_at);
    assert!(
        !reconfirm_probed(&r.settle, pid_parent),
        "parent does not reconfirm across the child's PostFireSettle expiry \
         (Settling → Rebasing)",
    );

    // Response Authoritative ⇒ commit → child finish_burst_to_idle. The
    // sweep re-evaluates the parent's fresh query (child now Idle,
    // interposed Profile only Seed) → false → parent reconfirms. No
    // panic (the old `dirty_descendants underflow` debug_assert is
    // gone), no strand.
    assert!(
        reconfirm_probed(&r.finish, pid_parent),
        "parent reconfirms at the child's finish despite the interposed Profile",
    );
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying { .. },
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
    // Draining-exit soundness guard: the trigger is a sweep of
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

    // A: events-reliable (DEFAULT_EVENTS) so its single Authoritative
    // verify folds to Stable directly. max_depth=1 keeps A off P's
    // covering set anyway, so the descendant `foo` event never reaches A.
    let out_a = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "a".into(),
            SubAttachAnchor::Resource(src),
            a_cfg,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            DEFAULT_EVENTS,
            false,
        )),
        now,
    );
    let sid_a = specter_core::testkit::first_attached_sub(&out_a).expect("attach_sub succeeded");
    let pid_a = e.subs().get(sid_a).unwrap().profile();
    // Drive A's Seed to a pinned `Idle` baseline (per-Profile, by-id
    // timer steps so the B and P Seeds below stay untouched).
    let a_seed_done = seed_to_idle(&mut e, pid_a, &dir_snap(&[]), now);

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
    let pid_b = e.subs().get(sid_b).unwrap().profile();
    // Drive B's Seed, rebased strictly past A's consumed settle
    // window. B then sits pinned at `Idle` — exactly the state the
    // later immediate-reap detach requires.
    let b_seed_done = seed_to_idle(&mut e, pid_b, &dir_snap(&[]), a_seed_done);

    // P @ /src/mid/foo (recursive). DEFAULT_EVENTS so its single
    // Authoritative verify fires directly and the post-fire rebase loop
    // closes in one sample (events-reliable witness).
    let out_pp = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            "p".into(),
            SubAttachAnchor::Resource(foo),
            unbounded,
            MAX_SETTLE,
            SETTLE,
            empty_program(),
            EffectScope::SubtreeRoot,
            DEFAULT_EVENTS,
            false,
        )),
        now,
    );
    let sid_p = specter_core::testkit::first_attached_sub(&out_pp).expect("attach_sub succeeded");
    let pid_p = e.subs().get(sid_p).unwrap().profile();
    // Drive P's Seed, rebased strictly past B's consumed settle window.
    let p_seed_done = seed_to_idle(&mut e, pid_p, &dir_snap(&[]), b_seed_done);

    // P's Standard burst from its own anchor `foo` (drives only P), then
    // A's own Standard burst from its anchor `src`. The three Seed
    // drives pushed the clock past `now`; rebase the Standard timeline
    // past them.
    let t1 = p_seed_done + Duration::from_millis(10);
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
    drain_due(&mut e, t2);

    // A stabilises while P gates it through the chain P → B → A →
    // A enters Draining. A's single Authoritative verify response folds
    // to `Stable` directly (DEFAULT_EVENTS makes A events-reliable); P
    // still being mid-burst routes A through `transition_to_draining`
    // instead of firing.
    assert!(
        e.profiles()
            .get(pid_p)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "P gates A (via the intermediate B) before A stabilises",
    );
    let v_a = verify(&mut e, pid_a, &dir_snap(&[]), t2);
    let a_parked_at = v_a.at;
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
    let detach_out = e.step(Input::DetachSub(sid_b), a_parked_at);
    assert!(
        e.profiles().get(pid_b).is_none(),
        "B reaped immediately (it was Idle)",
    );
    assert!(
        !reconfirm_probed(&detach_out, pid_a)
            && e.profiles().get(pid_a).unwrap().state().is_draining(),
        "reaping B does not itself reconfirm A — A is still gated by the \
         still-bursting P, just no longer through a chain that reaches it",
    );

    // Drive P through its full fire cycle. A is parked in Draining
    // holding only its BurstDeadline, so P's settle drain does not
    // disturb it (and no `finish_burst_to_idle` runs in a drain, so
    // the sweep cannot fire early). At P's single finish the sweep
    // re-checks *every* Draining Profile; A is found and reconfirmed
    // even though P's chain no longer reaches it. (A chain-coupled
    // trigger would strand A here forever.)
    assert!(
        e.profiles()
            .get(pid_p)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "P still gates the Draining A before it stabilises",
    );
    let v_p = verify(&mut e, pid_p, &dir_snap(&[]), a_parked_at);
    let p_parked_at = v_p.at;
    let p_stable = v_p.out;
    let p_effect = p_stable
        .effects()
        .first()
        .cloned()
        .expect("P fired one Effect at the stable verdict");
    let p_rebase_out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid_p,
            key: p_effect.key(),
            outcome: specter_core::EffectOutcome::Ok,
        }),
        p_parked_at,
    );
    assert!(
        !reconfirm_probed(&p_stable, pid_a) && !reconfirm_probed(&p_rebase_out, pid_a),
        "A does not reconfirm until P's burst actually finishes",
    );
    // P's post-fire rebase loop: PostFireSettle expiry → Rebasing →
    // response → commit. P stays Active(PostFire) for the whole loop,
    // so A does not reconfirm until P's single finish_burst_to_idle.
    let p_r = rebase_post_fire_to_idle(&mut e, pid_p, &dir_snap(&[]), p_parked_at);
    assert!(
        !reconfirm_probed(&p_r.settle, pid_a),
        "A does not reconfirm across P's PostFireSettle expiry \
         (Settling → Rebasing)",
    );

    // Response Authoritative ⇒ commit → P finish_burst_to_idle → the
    // sweep re-checks every Draining Profile.
    assert!(
        reconfirm_probed(&p_r.finish, pid_a),
        "the sweep reconfirms A even though P's chain no longer reaches it",
    );
    assert!(
        matches!(
            e.profiles().get(pid_a).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying { .. },
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
fn co_located_profiles_independently_record_shared_resource_obligation() {
    // Two config_hash-distinct Profiles share one Resource /src. A
    // single FsEvent on the shared resource fans to *every* covering
    // Profile (on_fs_event iterates the covering set), and each Profile
    // records the resource's path in its *own* dirty provenance (the
    // obligation basis). There is no per-resource global filter that one Profile's
    // in-flight burst could use to blind another — the property that
    // keeps a co-resident Profile's probe from mtime-skipping a
    // genuinely changed directory.
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
    let pid_a = e.subs().get(sid_a).unwrap().profile();

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
    let pid_b = e.subs().get(sid_b).unwrap().profile();

    assert_ne!(pid_a, pid_b, "distinct config_hash ⇒ distinct Profiles");
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        2,
        "both Profiles contribute the shared resource's watch",
    );

    // Drive both Seeds → Idle with a baseline. Each Seed pins only on
    // its own Authoritative response. Driving A by *A's own* settle id
    // does not touch B's independent Seed (per-Profile correlation +
    // per-Profile timer id); B is then driven by its own fresh probe,
    // rebased strictly past A's consumed window — `seed_to_idle`
    // asserts each `pid` independently reaches `Idle`.
    let a_seed_done = seed_to_idle(&mut e, pid_a, &dir_snap(&[]), now);
    let b_seed_done = seed_to_idle(&mut e, pid_b, &dir_snap(&[]), a_seed_done);

    // One FsEvent on the shared anchor fans to both Profiles; each
    // opens its own Standard burst seeded with the resource. The Seed
    // drives pushed the clock past `now`; rebase past them.
    let t0 = b_seed_done + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t0,
    );
    let r_path = e.tree().get(r).unwrap().path().clone();
    for pid in [pid_a, pid_b] {
        let pre = match e.profiles().get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
            other => panic!("expected Active(PreFire) for {pid:?}, got {other:?}"),
        };
        assert!(
            pre.dirty.chains().contains(&r_path),
            "{pid:?} independently recorded the shared resource's path in \
             its own provenance (the obligation basis)",
        );
        assert_eq!(
            pre.dirty.lca_path().as_ref(),
            Some(&r_path),
            "{pid:?}: the lone recorded path is the component-LCA",
        );
    }

    // A second external event on the shared resource while both bursts
    // are in flight: again fanned to each Profile independently. No
    // Profile's in-flight burst removes the resource's path from
    // another's provenance (obligation basis).
    let t1 = t0 + SETTLE / 2;
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t1,
    );
    for pid in [pid_a, pid_b] {
        let pre = match e.profiles().get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
            other => panic!("expected Active(PreFire) for {pid:?}, got {other:?}"),
        };
        assert!(
            pre.dirty.chains().contains(&r_path),
            "{pid:?} still records the shared resource's path in its own \
             provenance after a mid-burst event (its own \
             ProofObligation::Chains basis) — no global filter poisons a \
             co-resident Profile",
        );
    }
}

/// Fixture for the sensor-overflow × Draining-ancestor scenarios. A
/// recursive parent Profile `A` at `/src` is parked in `Draining`,
/// gated by its covered child Profile `D` at `/src/child` which is
/// mid-Standard-burst (its Verify probe still in flight).
struct DrainingFixture {
    e: Engine,
    src: ResourceId,
    pid_parent: ProfileId,
    pid_child: ProfileId,
    sid_parent: SubId,
    /// The fixture clock at the point the parent reached `Draining`
    /// (after the quiescence confirm); subsequent steps reuse it so
    /// the scenario stays deterministic.
    parked_at: Instant,
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
    let pid_child = e.subs().get(sid_child).unwrap().profile();
    // Drive the child's Seed to a pinned `Idle` baseline (per-Profile,
    // by-id timer steps so the parent Seed below stays untouched).
    let child_seed_done = seed_to_idle(&mut e, pid_child, &dir_snap(&[]), now);

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
    let pid_parent = e.subs().get(sid_parent).unwrap().profile();
    // Drive the parent's Seed, rebased strictly past the child Seed's
    // consumed settle window.
    let parent_seed_done = seed_to_idle(&mut e, pid_parent, &dir_snap(&[]), child_seed_done);

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
    // stay isolated. The Seed drives pushed the clock past `now`;
    // rebase the Standard timeline past them.
    let t1 = parent_seed_done + Duration::from_millis(10);
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
    drain_due(&mut e, t2);

    // Parent stabilises while the child still gates it → Draining.
    // The parent is on a NO_EVENTS mask, so its Standard verify owes
    // the N=2 hash channel: first sample → Retry (prior=None,
    // re-batches); second sample (settle-spaced, same hash) → Stable;
    // the child still being mid-burst routes the parent through
    // `transition_to_draining` instead of firing.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    let v_first = verify(&mut e, pid_parent, &dir_snap(&[]), t2);
    let parked_at = v_first.at + SETTLE * 2;
    drain_due(&mut e, parked_at);
    let v = verify(&mut e, pid_parent, &dir_snap(&[]), parked_at);
    let parked_at = v.at;
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
        pid_parent,
        pid_child,
        sid_parent,
        parked_at,
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
        parked_at,
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
        parked_at,
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
                    phase: PreFirePhase::Verifying { .. },
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
        parked_at,
        ..
    } = draining_parent_gated_by_child();

    e.step(
        Input::SensorOverflow {
            scope: OverflowScope::Resource(src),
        },
        parked_at,
    );

    // The discriminator, reached through the Resource snapshot path:
    // the in-scope Draining ancestor is excluded from the reseed and
    // stays on the sweep's Standard reconfirm instead of returning Seed.
    assert!(
        matches!(
            e.profiles().get(pid_parent).unwrap().state(),
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    phase: PreFirePhase::Verifying { .. },
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
        parked_at,
        ..
    } = draining_parent_gated_by_child();

    // Detach the ancestor's sole Sub mid-Draining → deferred Reap.
    let _ = e.step(Input::DetachSub(sid_parent), parked_at);
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
        parked_at,
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
                    phase: PreFirePhase::Verifying { .. },
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
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: dir_snap(&[]),
                authority: ProofAuthority::Authoritative,
            },
        }),
        parked_at,
    );
    assert!(
        e.profiles().get(pid_parent).is_none(),
        "deferred Reap honoured once the reconfirm resolved — ancestor reaped",
    );
    let _ = e.profiles().get(pid_child); // child untouched by the reap

    let _ = e.cancel_all_in_flight_probes();
}
