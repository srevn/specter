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
    ActiveBurst, BurstFinish, BurstIntent, ClassSet, EffectScope, EntryKind, FsEvent, Input,
    OverflowScope, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase, ProbeOp, ProbeOutcome,
    ProbeOwner, ProbeResponse, ProfileId, ProfileState, ProofAuthority, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, SubAttachAnchor, SubAttachRequest, SubId, TimerKind, WatchOp,
};
use specter_engine::Engine;
use specter_engine::testkit::{
    MAX_SETTLE, NO_EVENTS, SETTLE, drain_due, rebase_loop_to_idle, reconfirm_probed, seed_to_idle,
    verify_n2,
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
    // Rebasing → (residual restart) → Batching → Verifying: the child
    // never leaves the Active Standard burst — the residual restart is
    // an in-place PostFire→PreFire move, so there is no finish-then-start
    // flicker for the sweep to misread as a gap — and the parent
    // reconfirms exactly once, at the restarted burst's single
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
    // The parent's Seed is Batching-first; drive its N=2
    // quiescence proof to a pinned `Idle` baseline (per-Profile, by-id
    // timer steps so the child Seed below is untouched).
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
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    // Drive the child's Seed N=2, rebased strictly past the parent
    // Seed's consumed settle windows. The child attached at `now` so
    // its Batching settle id is read fresh from its still-Batching
    // state; stepping it at a later instant is quiet-window-clean.
    let child_seed_done = seed_to_idle(&mut e, pid_child, &child_snap, parent_seed_done);

    // Child Standard burst FIRST (so it gates the parent), then the
    // parent's own Standard burst. The Seed N=2 drives pushed the clock
    // ~4·SETTLE past `now`; rebase the Standard timeline past them.
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

    // ── N=2 #1: parent stabilises while the child is mid-Standard-burst
    // → Draining. Layer B opens a fresh Standard burst with
    // a fresh `CertifiedPrior`, so the parent's first Verify sample
    // is `Unstable` by construction (prime ⇒ re-batch); only the
    // second settle-spaced hash-equal sample is `Stable`. The child is
    // parked in Verifying with no expirable settle timer, so draining
    // the parent's re-armed settle does not advance it — it keeps
    // gating.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    let n2_p = verify_n2(&mut e, pid_parent, &dir_snap(&[]), t2);
    assert!(
        n2_p.primed.effects().is_empty() && !reconfirm_probed(&n2_p.primed, pid_parent),
        "parent prime sample (prior == None ⇒ Unstable) must not fire",
    );
    let parent_parked_at = n2_p.confirm_at;
    assert!(
        parent_is_draining(&e),
        "parent enters Draining (stable verdict, child still gating)",
    );

    // ── N=2 #2: child Verifying stable → fires (Awaiting). The child's
    // burst is also fresh (its carrier opened `None` at the FsEvent),
    // so it too needs prime → drain → confirm. The parent is parked in
    // Draining holding only its 6 s BurstDeadline, so draining the
    // child's re-armed settle does not disturb it.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child still gates the parent before it stabilises",
    );
    let n2_c = verify_n2(&mut e, pid_child, &child_snap, parent_parked_at);
    assert!(
        n2_c.primed.effects().is_empty()
            && !reconfirm_probed(&n2_c.primed, pid_parent)
            && parent_is_draining(&e),
        "child prime sample must not fire and must not disturb the Draining parent",
    );
    let child_parked_at = n2_c.confirm_at;
    let stable_out = n2_c.confirmed;
    let child_effect = stable_out
        .effects()
        .first()
        .cloned()
        .expect("child fired one Effect at the stable verdict");
    assert!(
        !reconfirm_probed(&stable_out, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm at the child's stable verdict",
    );

    // Rebase tail kept inline (irregular: final-window absorb + residual
    // restart — not the clean rebase_loop_to_idle shape).
    // EffectComplete → child transition_to_rebasing(First). No finish,
    // no sweep — the child stays Active(PostFire) for the whole loop.
    let rebase_out = e.step(
        Input::EffectComplete {
            sub: sid_c,
            key: child_effect.key(),
            result: specter_core::EffectOutcome::Ok,
        },
        child_parked_at,
    );
    let rebase_corr1 = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("child rebase probe #1 in flight");
    assert!(
        !reconfirm_probed(&rebase_out, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm during child Rebasing #1",
    );

    // Sample 1 (prior `None`) ⇒ Unstable ⇒ RebaseSettling. An absorb
    // here would be cleared by the next transition_to_rebasing re-arm,
    // so the residual that drives the restart must land in the FINAL
    // round-trip.
    let s1 = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: rebase_corr1,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: child_snap.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        child_parked_at,
    );
    assert!(
        !reconfirm_probed(&s1, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm across the child's first rebase sample",
    );
    let spacing_timer = match e.profiles().get(pid_child).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::RebaseSettling { spacing_timer },
                ..
            }),
            _,
        ) => *spacing_timer,
        other => {
            panic!(
                "child rebase sample 1 must loop to Active(PostFire(RebaseSettling)); got {other:?}"
            )
        }
    };

    // RebaseSettle expiry (by id — scoped to the child, parent
    // untouched) → Rebasing #2, the FINAL round-trip.
    let t_respace = child_parked_at + SETTLE;
    let s2 = e.step(
        Input::TimerExpired {
            profile: pid_child,
            kind: TimerKind::RebaseSettle,
            id: spacing_timer,
        },
        t_respace,
    );
    let rebase_corr2 = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("RebaseSettle re-arms the child's Rebasing probe #2");
    assert!(
        !reconfirm_probed(&s2, pid_parent) && parent_is_draining(&e),
        "parent does not reconfirm across the child's settle-spaced re-arm",
    );

    // Descendant edit absorbed during the FINAL Rebasing round-trip —
    // the genuine final-window residual (no further loop entry clears
    // it).
    let t_absorb = t_respace + Duration::from_millis(5);
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

    // Sample 2 hash-equal ⇒ Stable; non-empty final-window residual ⇒
    // child restarts in-place. THE key assertion: the child never
    // leaves the Active Standard burst (no finish-then-start flicker),
    // so the parent must not reconfirm in this step and stays Draining.
    let t_restart = t_absorb + Duration::from_millis(5);
    let restart_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: rebase_corr2,
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

    // ── N=2 #3: drive the restarted burst to its single finish. The
    // restart is a fresh PreFireBurst (`into_pre_fire_residual` reset
    // a fresh `CertifiedPrior`), so it too is N=2: Batching →
    // Verifying → prime (Unstable ⇒ re-batch) → drain → confirm
    // (Stable; baseline == current, Sub already fired ⇒ B1 dedup, zero
    // effects) → finish_burst_to_idle. This restarted burst carries an
    // EMPTY fire-tail residual, so its finish is the plain
    // direct-finish path: this single test therefore exercises BOTH the
    // residual-restart edge (no finish-then-start flicker) AND the
    // empty-residual "child settles → parent reconfirms at the one
    // finish_burst_to_idle" case. The sweep runs here for the first
    // time since the parent entered Draining; the child is now Idle, so
    // the parent's fresh covered-descendant query is false → reconfirm.
    // The prime sample must NOT finish, sweep, or disturb the Draining
    // parent.
    // Kept inline (not verify_n2): an interleaved mid-N=2 state assert
    // sits between prime and confirm; only the two settle drains
    // collapse to drain_due.
    let t_rdrain = t_restart + SETTLE * 2;
    drain_due(&mut e, t_rdrain);
    let restart_prime = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("restarted burst's Verifying probe in flight (prime sample)");
    let restart_primed = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: restart_prime,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: child_snap.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_rdrain,
    );
    assert!(
        restart_primed.effects().is_empty()
            && !reconfirm_probed(&restart_primed, pid_parent)
            && parent_is_draining(&e),
        "restarted-burst prime must not fire, sweep, or disturb the Draining parent",
    );
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "restarted burst still Active after its prime sample re-batches",
    );
    let t_rconfirm = t_rdrain + SETTLE * 2;
    drain_due(&mut e, t_rconfirm);
    let restart_verify_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid_child))
        .expect("restarted burst's Verifying probe in flight (confirm sample)");
    let finish_out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid_child),
            correlation: restart_verify_corr,
            outcome: ProbeOutcome::SubtreeProven {
                snapshot: child_snap.clone(),
                authority: ProofAuthority::Authoritative,
            },
        }),
        t_rconfirm,
    );
    assert!(
        matches!(
            e.profiles().get(pid_child).unwrap().state(),
            ProfileState::Idle
        ),
        "restarted burst dedup-suppressed (baseline == current) and finished",
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
    // The parent's Seed is Batching-first; drive its N=2
    // quiescence proof to a pinned `Idle` baseline (per-Profile, by-id
    // timer steps so the child Seed below stays untouched).
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
            NO_EVENTS,
            false,
        )),
        now,
    );
    let sid_c = specter_core::testkit::first_attached_sub(&out_c).expect("attach_sub succeeded");
    let pid_child = e.subs().get(sid_c).unwrap().profile;
    // Drive the child's Seed N=2, rebased strictly past the parent
    // Seed's consumed settle windows.
    let child_seed_done = seed_to_idle(&mut e, pid_child, &dir_snap(&[]), parent_seed_done);

    // Child Standard burst FIRST (so it gates the parent), then parent's.
    // The Seed N=2 drives pushed the clock ~4·SETTLE past `now`; rebase
    // the Standard timeline past them.
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

    // ── N=2 #1: parent stabilises while the child gates it → Draining.
    // Layer B opens the parent's fresh Standard burst with
    // a fresh `CertifiedPrior`, so its first Verify sample is
    // `Unstable` by construction (prime ⇒ re-batch); the second
    // settle-spaced hash-equal sample is `Stable`. The child is parked
    // in Verifying with no expirable settle timer, so draining the
    // parent's re-armed settle keeps it gating.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    let n2_parent = verify_n2(&mut e, pid_parent, &dir_snap(&[]), t2);
    assert!(
        n2_parent.primed.effects().is_empty() && !reconfirm_probed(&n2_parent.primed, pid_parent),
        "parent prime sample (prior == None ⇒ Unstable) must not fire",
    );
    let parent_parked_at = n2_parent.confirm_at;
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

    // ── N=2 #2: drive the child through its full fire cycle. Its burst
    // is also fresh (carrier opened `None`), so it needs prime → drain
    // → confirm. The parent is parked in Draining holding only its 6 s
    // BurstDeadline, and the interposed `mid` Profile sits in Seed with
    // no expirable settle timer, so draining the child's re-armed
    // settle disturbs neither. No reconfirm until the child finishes.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child still gates the Draining parent before it stabilises",
    );
    let n2_child = verify_n2(&mut e, pid_child, &dir_snap(&[]), parent_parked_at);
    assert!(
        n2_child.primed.effects().is_empty() && !reconfirm_probed(&n2_child.primed, pid_parent),
        "child prime sample must not fire or reconfirm the parent",
    );
    let child_parked_at = n2_child.confirm_at;
    let stable_out = n2_child.confirmed;
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
        child_parked_at,
    );
    assert!(
        !reconfirm_probed(&stable_out, pid_parent) && !reconfirm_probed(&rebase_out, pid_parent),
        "parent does not reconfirm until the child's burst finishes",
    );

    // Child post-fire N=2 rebase loop. The child stays Active(PostFire)
    // for the whole loop, so the parent does not reconfirm until the
    // child's single finish_burst_to_idle (the Stable verdict).
    let r = rebase_loop_to_idle(&mut e, pid_child, &dir_snap(&[]), child_parked_at);
    assert!(
        !reconfirm_probed(&r.s1, pid_parent),
        "parent does not reconfirm across the child's first rebase sample",
    );
    // RebaseSettle expiry (by id — scoped to the child) → Rebasing #2.
    assert!(
        !reconfirm_probed(&r.rearm, pid_parent),
        "parent does not reconfirm across the child's settle-spaced re-arm",
    );

    // Sample 2 hash-equal ⇒ Stable → child finish_burst_to_idle. The
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
    // Every Seed is Batching-first; drive A's N=2 quiescence
    // proof to a pinned `Idle` baseline (per-Profile, by-id timer steps
    // so the B and P Seeds below stay untouched).
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
    let pid_b = e.subs().get(sid_b).unwrap().profile;
    // Drive B's Seed N=2, rebased strictly past A's consumed settle
    // windows. B then sits pinned at `Idle` — exactly the state the
    // later immediate-reap detach requires.
    let b_seed_done = seed_to_idle(&mut e, pid_b, &dir_snap(&[]), a_seed_done);

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
    // Drive P's Seed N=2, rebased strictly past B's consumed settle
    // windows.
    let p_seed_done = seed_to_idle(&mut e, pid_p, &dir_snap(&[]), b_seed_done);

    // P's Standard burst from its own anchor `foo` (drives only P), then
    // A's own Standard burst from its anchor `src`. The three Seed N=2
    // drives pushed the clock ~6·SETTLE past `now`; rebase the Standard
    // timeline past them.
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

    // ── N=2 #1: A stabilises while P gates it through the chain
    // P → B → A → A enters Draining. Layer B opens A's fresh Standard
    // burst with a fresh `CertifiedPrior`, so its first Verify
    // sample is `Unstable` by construction (prime ⇒ re-batch); the
    // second settle-spaced hash-equal sample is `Stable`. P is parked
    // in Verifying with no expirable settle timer, so draining A's
    // re-armed settle keeps it gating.
    assert!(
        e.profiles()
            .get(pid_p)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "P gates A (via the intermediate B) before A stabilises",
    );
    let n2_a = verify_n2(&mut e, pid_a, &dir_snap(&[]), t2);
    assert!(
        n2_a.primed.effects().is_empty() && !reconfirm_probed(&n2_a.primed, pid_a),
        "A prime sample (prior == None ⇒ Unstable) must not fire",
    );
    let a_parked_at = n2_a.confirm_at;
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

    // ── N=2 #2: drive P through its full fire cycle. P's burst is also
    // fresh (carrier opened `None`), so it needs prime → drain →
    // confirm. A is parked in Draining holding only its 6 s
    // BurstDeadline, so draining P's re-armed settle does not disturb
    // it (and no `finish_burst_to_idle` runs in a drain, so the sweep
    // cannot fire early). At P's single finish the sweep re-checks
    // *every* Draining Profile; A is found and reconfirmed even though
    // P's chain no longer reaches it. (A chain-coupled trigger would
    // strand A here forever.)
    assert!(
        e.profiles()
            .get(pid_p)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "P still gates the Draining A before it stabilises",
    );
    let n2_p = verify_n2(&mut e, pid_p, &dir_snap(&[]), a_parked_at);
    assert!(
        n2_p.primed.effects().is_empty()
            && !reconfirm_probed(&n2_p.primed, pid_a)
            && e.profiles().get(pid_a).unwrap().state().is_draining(),
        "P prime sample must not fire or reconfirm/disturb the Draining A",
    );
    let p_parked_at = n2_p.confirm_at;
    let p_stable = n2_p.confirmed;
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
        p_parked_at,
    );
    assert!(
        !reconfirm_probed(&p_stable, pid_a) && !reconfirm_probed(&p_rebase_out, pid_a),
        "A does not reconfirm until P's burst actually finishes",
    );
    // P's post-fire N=2 rebase loop. P stays Active(PostFire) for the
    // whole loop, so A does not reconfirm until P's single
    // finish_burst_to_idle (the Stable verdict).
    let p_r = rebase_loop_to_idle(&mut e, pid_p, &dir_snap(&[]), p_parked_at);
    assert!(
        !reconfirm_probed(&p_r.s1, pid_a),
        "A does not reconfirm across P's first rebase sample",
    );
    // RebaseSettle expiry (by id — scoped to P) → Rebasing #2.
    assert!(
        !reconfirm_probed(&p_r.rearm, pid_a),
        "A does not reconfirm across P's settle-spaced re-arm",
    );

    // Sample 2 hash-equal ⇒ Stable → P finish_burst_to_idle → the
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
fn co_located_profiles_independently_record_shared_resource_obligation() {
    // Two config_hash-distinct Profiles share one Resource /src. A
    // single FsEvent on the shared resource fans to *every* covering
    // Profile (on_fs_event iterates the covering set), and each Profile
    // records the resource in its *own* dirty_resources (the obligation
    // basis). There is no per-resource global filter that one Profile's
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

    assert_ne!(pid_a, pid_b, "distinct config_hash ⇒ distinct Profiles");
    assert_eq!(
        e.tree().get(r).unwrap().watch_demand(),
        2,
        "both Profiles contribute the shared resource's watch",
    );

    // Drive both Seeds → Idle with a baseline. Each Seed is
    // Batching-first and pins only on its own N=2 proof. Driving A's
    // full N=2 by *A's own* by-id settle steps does not touch B's
    // independent Seed (per-Profile correlation + per-Profile timer
    // id); B is then driven by *B's own* fresh Batching settle id,
    // rebased strictly past A's consumed windows — `seed_to_idle`
    // asserts each `pid` independently reaches `Idle`.
    let a_seed_done = seed_to_idle(&mut e, pid_a, &dir_snap(&[]), now);
    let b_seed_done = seed_to_idle(&mut e, pid_b, &dir_snap(&[]), a_seed_done);

    // One FsEvent on the shared anchor fans to both Profiles; each
    // opens its own Standard burst seeded with the resource. The Seed
    // N=2 drives pushed the clock ~4·SETTLE past `now`; rebase past them.
    let t0 = b_seed_done + Duration::from_millis(10);
    e.step(
        Input::FsEvent {
            resource: r,
            event: FsEvent::Modified,
        },
        t0,
    );
    for pid in [pid_a, pid_b] {
        let pre = match e.profiles().get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre,
            other => panic!("expected Active(PreFire) for {pid:?}, got {other:?}"),
        };
        assert!(
            pre.dirty_resources.contains(&r),
            "{pid:?} independently recorded the shared resource in its own \
             dirty_resources (the obligation basis)",
        );
    }

    // A second external event on the shared resource while both bursts
    // are in flight: again fanned to each Profile independently. No
    // Profile's in-flight burst removes the resource from another's
    // dirty_resources (obligation basis).
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
            pre.dirty_resources.contains(&r),
            "{pid:?} still records the shared resource in its own \
             dirty_resources after a mid-burst event (its own \
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
    /// (after the N=2 quiescence confirm); subsequent steps reuse it so
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
    let pid_child = e.subs().get(sid_child).unwrap().profile;
    // The child's Seed is Batching-first; drive its N=2
    // quiescence proof to a pinned `Idle` baseline (per-Profile, by-id
    // timer steps so the parent Seed below stays untouched).
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
    let pid_parent = e.subs().get(sid_parent).unwrap().profile;
    // Drive the parent's Seed N=2, rebased strictly past the child
    // Seed's consumed settle windows.
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
    // stay isolated. The Seed N=2 drives pushed the clock ~4·SETTLE
    // past `now`; rebase the Standard timeline past them.
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
    // N=2 quiescence: the parent's prime sample (prior == None ⇒
    // Unstable) re-arms its settle timer. The child is parked in
    // Verifying with no expirable timer, so draining the parent's
    // re-armed settle does not advance it — it keeps gating. The
    // hash-equal confirm sample is the Stable verdict that, with the
    // child still in an active Standard burst, parks the parent in
    // Draining.
    assert!(
        e.profiles()
            .get(pid_child)
            .unwrap()
            .state()
            .in_active_standard_burst(),
        "child gates the parent before the parent stabilises",
    );
    let n2 = verify_n2(&mut e, pid_parent, &dir_snap(&[]), t2);
    assert!(
        n2.primed.effects().is_empty(),
        "parent prime sample (prior == None ⇒ Unstable) must not fire",
    );
    let parked_at = n2.confirm_at;
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
