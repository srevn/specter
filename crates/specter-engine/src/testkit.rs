//! Engine-driving test harness.
//!
//! Gated behind the `testkit` feature (and `cfg(test)` for inline unit
//! tests). Written against the **public** `Engine` surface only, so one
//! body serves both the integration suite (`tests/*.rs`) and the inline
//! unit tests (`src/*_tests.rs`) — the access wall that previously
//! forced two divergent copies of every driver is dissolved here.
//!
//! Discipline: every fn drives the engine and returns *all* the
//! `StepOutput`s it produced. Assertions, topology, and the scenario
//! timeline stay at the call site — the harness drives, the test
//! proves.

use crate::Engine;
use specter_core::testkit::{enumerated, proven};
use specter_core::{
    ActiveBurst, ClassSet, DedupKey, DirSnapshot, EffectCompletion, EffectOutcome, EffectScope,
    FS_ROOT_SEGMENT, Input, PatternSpec, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase,
    ProbeCorrelation, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileIdentity,
    ProfileState, PromoterAttachRequest, PromoterId, ResourceId, ResourceKind, ResourceRole,
    ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId, TimerId, TimerKind,
    WatchFailure,
};
use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Debounce window every fixture uses.
pub const SETTLE: Duration = Duration::from_millis(100);
/// Force-fire ceiling every fixture uses.
pub const MAX_SETTLE: Duration = Duration::from_secs(6);
/// The empty event mask (a Profile that bursts only from its own anchor).
pub const NO_EVENTS: ClassSet = ClassSet::EMPTY;

/// Blanket-drain every timer due at `at`, stepping each.
///
/// This is the *parked-siblings* drain discipline: correct when any
/// co-Profile in flight holds no timer expirable at `at` (a Verifying
/// Profile has no settle timer; a Draining one holds only its
/// `MAX_SETTLE` deadline, far past a `start + SETTLE*2` confirm
/// window). A Seed is driven by its own id ([`seed_to_idle`]) instead,
/// precisely so a blanket drain here cannot disturb it.
pub fn drain_due(e: &mut Engine, at: Instant) {
    while let Some(en) = e.pop_expired(at) {
        e.step(
            Input::TimerExpired {
                profile: en.profile,
                kind: en.kind,
                id: en.id,
            },
            at,
        );
    }
}

/// `SubId` → its `ProfileId` via the public registry.
#[must_use]
pub fn pid_of(e: &Engine, sid: SubId) -> ProfileId {
    e.subs().get(sid).expect("sub present").profile()
}

/// First in-flight `Probe` correlation in `out`, if any.
#[must_use]
pub fn first_probe_correlation(out: &StepOutput) -> Option<ProbeCorrelation> {
    out.probe_ops().iter().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.correlation()),
        ProbeOp::Cancel { .. } => None,
    })
}

/// Read `pid`'s `Active(PreFire(Batching))` settle-timer id, or panic
/// with the actual state.
///
/// Stepping a Batching burst by *its own* id (rather than a blanket
/// drain) keeps a multi-Profile setup's sibling Profiles untouched.
#[must_use]
pub fn batching_settle_id(e: &Engine, pid: ProfileId) -> TimerId {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                phase: PreFirePhase::Batching { settle_timer },
                ..
            }),
            _,
        ) => *settle_timer,
        other => panic!("expected {pid:?} in Active(PreFire(Batching)), got {other:?}"),
    }
}

/// Read `pid`'s `Active(PostFire(RebaseSettling))` spacing-timer id, or
/// panic with the actual state — the post-fire sibling of
/// [`batching_settle_id`].
#[must_use]
pub fn rebase_settling_spacing_id(e: &Engine, pid: ProfileId) -> TimerId {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::RebaseSettling { spacing_timer },
                ..
            }),
            _,
        ) => *spacing_timer,
        other => panic!("expected {pid:?} in Active(PostFire(RebaseSettling)), got {other:?}"),
    }
}

/// `out` contains a `Probe` owned by `pid` — the public-surface
/// signal that a Draining Profile reconfirmed (`Draining → Verifying`).
#[must_use]
pub fn reconfirm_probed(out: &StepOutput, pid: ProfileId) -> bool {
    out.probe_ops().iter().any(|op| {
        matches!(op, ProbeOp::Probe { request }
            if request.owner() == ProbeOwner::Profile(pid))
    })
}

/// `pid`'s burst is in the `Draining` phase.
///
/// Phase-only: this does **not** assert the `BurstFinish` directive — a
/// test that pins `ReturnToIdle` vs `Reap` keeps that match inline (it
/// is the test's claim, not drive mechanism).
#[must_use]
pub fn is_draining(e: &Engine, pid: ProfileId) -> bool {
    e.profiles().get(pid).unwrap().state().is_draining()
}

/// Attach a subtree-root Sub at `anchor`, returning the attach
/// `StepOutput` too — for tests asserting on the attach step (probe
/// present/absent, diagnostics).
///
/// Fixture defaults: `SETTLE`, `/bin/true`, `EffectScope::SubtreeRoot`,
/// `log_output = false`. `max_settle` is explicit, not defaulted: it
/// folds into `config_hash`, so a distinct `max_settle` forks a
/// distinct Profile — the multi-Profile fork tests rely on that.
#[must_use]
pub fn attach_returning(
    e: &mut Engine,
    name: &str,
    anchor: SubAttachAnchor,
    cfg: ScanConfig,
    mask: ClassSet,
    max_settle: Duration,
    now: Instant,
) -> (SubId, ProfileId, StepOutput) {
    let out = e.step(
        Input::AttachSub(SubAttachRequest::for_anchor(
            name.into(),
            anchor,
            cfg,
            max_settle,
            SETTLE,
            specter_core::testkit::empty_program(),
            EffectScope::SubtreeRoot,
            mask,
            false,
        )),
        now,
    );
    let sid = specter_core::testkit::first_attached_sub(&out).expect("attach_sub succeeded");
    let pid = pid_of(e, sid);
    (sid, pid, out)
}

/// [`attach_returning`] discarding the attach `StepOutput` — the common
/// case. Returns `(SubId, ProfileId)`.
#[must_use]
pub fn attach(
    e: &mut Engine,
    name: &str,
    anchor: SubAttachAnchor,
    cfg: ScanConfig,
    mask: ClassSet,
    max_settle: Duration,
    now: Instant,
) -> (SubId, ProfileId) {
    let (sid, pid, _) = attach_returning(e, name, anchor, cfg, mask, max_settle, now);
    (sid, pid)
}

/// [`attach`] then drive the fresh Seed burst to pinned `Idle`.
/// Returns `(SubId, ProfileId, seed_done_instant)` — rebase later
/// timelines strictly past `seed_done`.
#[must_use]
pub fn attach_seeded(
    e: &mut Engine,
    name: &str,
    anchor: SubAttachAnchor,
    cfg: ScanConfig,
    mask: ClassSet,
    max_settle: Duration,
    snap: &Arc<DirSnapshot>,
    start: Instant,
) -> (SubId, ProfileId, Instant) {
    let (sid, pid, _) = attach_returning(e, name, anchor, cfg, mask, max_settle, start);
    let done = seed_to_idle(e, pid, snap, start);
    (sid, pid, done)
}

/// Expire `pid`'s own Batching `Settle` window, advancing
/// `Active(PreFire(Batching)) → Verifying` and emitting the probe.
///
/// Steps by the settle's *own* id (not a blanket drain — sibling
/// Profiles in a multi-Profile setup stay untouched). Returns the
/// in-flight probe correlation and the post-expiry instant. The
/// non-atomic primitive [`seed_to_idle_with`] composes twice; callers
/// that terminate the burst on its *first* response (Vanished /
/// Failed) or inspect the intermediate probe shape use it directly.
#[must_use]
pub fn seed_settle_to_verifying(
    e: &mut Engine,
    pid: ProfileId,
    at: Instant,
) -> (ProbeCorrelation, Instant) {
    let at = at + SETTLE;
    let settle_id = batching_settle_id(e, pid);
    e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::Settle,
            id: settle_id,
        },
        at,
    );
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Seed Verifying probe in flight after settle expiry");
    (correlation, at)
}

/// Drive a fresh Batching-first Seed burst for `pid` through its N=2
/// quiescence proof to pinned `Idle`, answering each probe with
/// `make_outcome()`.
///
/// `make_outcome` is invoked once per sample, so the prime and confirm
/// outcomes are equal **by construction** — `proven(snap)` for a Dir
/// anchor, `anchor_ok(file_leaf(..))` for a File anchor, both folding
/// through the N=2 proof identically; the `Stable` requirement cannot
/// be accidentally violated. The first sample is `Unstable` (no prior
/// certified) → graft + re-batch; the second hash-equal sample is
/// `Stable` → pin → `Idle`. A fresh Seed emits no Effects. `start` is
/// the instant the Seed burst's debounce began; returns the final
/// instant (two settle windows later) so the caller rebases later
/// bursts past it.
#[must_use]
pub fn seed_to_idle_with(
    e: &mut Engine,
    pid: ProfileId,
    make_outcome: impl Fn() -> ProbeOutcome,
    start: Instant,
) -> Instant {
    let mut at = start;
    for _ in 0..2 {
        let (correlation, sample_at) = seed_settle_to_verifying(e, pid, at);
        at = sample_at;
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation,
                outcome: make_outcome(),
            }),
            at,
        );
        assert!(
            out.effects().is_empty(),
            "a fresh Seed never fires an Effect (N=2 establishment)",
        );
    }
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "two settle-spaced hash-equal Seed samples pin the baseline → Idle for {pid:?}",
    );
    at
}

/// [`seed_to_idle_with`] answering every probe with `proven(snap)` —
/// the Dir-anchor common case.
#[must_use]
pub fn seed_to_idle(
    e: &mut Engine,
    pid: ProfileId,
    snap: &Arc<DirSnapshot>,
    start: Instant,
) -> Instant {
    seed_to_idle_with(e, pid, || proven(Arc::clone(snap)), start)
}

/// The pre-fire Standard N=2 dance for `pid`.
///
/// `pid` is already in `Verifying` with a probe in flight at `start`:
/// prime (prior `None` ⇒ `Unstable` ⇒ re-batch) → drain the re-armed
/// settle at `start + SETTLE*2` → confirm (hash-equal ⇒ `Stable`).
/// Returns both step outputs and the confirm instant; the caller
/// asserts context-specifically on each.
#[derive(Debug)]
pub struct N2 {
    pub primed: StepOutput,
    pub confirmed: StepOutput,
    pub confirm_at: Instant,
}

/// The pre-fire Standard N=2 dance for `pid`, answering both samples
/// with `make_outcome()`.
///
/// `make_outcome` is invoked once per sample, so prime and confirm are
/// hash-equal **by construction** (`proven(snap)` for a Dir anchor,
/// `anchor_ok(file_leaf(..))` for a File anchor) — the `Stable`
/// requirement cannot be accidentally violated.
#[must_use]
pub fn verify_n2_with(
    e: &mut Engine,
    pid: ProfileId,
    make_outcome: impl Fn() -> ProbeOutcome,
    start: Instant,
) -> N2 {
    let prime_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight (prime sample)");
    let primed = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: prime_corr,
            outcome: make_outcome(),
        }),
        start,
    );
    let confirm_at = start + SETTLE * 2;
    drain_due(e, confirm_at);
    let confirm_corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight (confirm sample)");
    let confirmed = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: confirm_corr,
            outcome: make_outcome(),
        }),
        confirm_at,
    );
    N2 {
        primed,
        confirmed,
        confirm_at,
    }
}

/// [`verify_n2_with`] answering both samples with `proven(snap)` — the
/// Subtree common case (every pre-fire Standard N=2 site today).
#[must_use]
pub fn verify_n2(e: &mut Engine, pid: ProfileId, snap: &Arc<DirSnapshot>, start: Instant) -> N2 {
    verify_n2_with(e, pid, || proven(Arc::clone(snap)), start)
}

/// The clean post-fire rebase N=2 loop for `pid`.
///
/// `pid` is already in `Active(PostFire(Rebasing))` with probe #1 in
/// flight (the caller has stepped `EffectComplete`): sample 1 (prior
/// `None` ⇒ `Unstable` ⇒ `RebaseSettling`) → `RebaseSettle` spacing
/// expiry by id (re-arm `Rebasing`) → sample 2 (hash-equal ⇒ `Stable`
/// ⇒ finish/restart).
///
/// Returns every step output (`s1`, `rearm`, `finish`) and the finish
/// instant so the caller can assert co-Profile state between each.
/// **Carve-out:** a test that injects a custom absorb in the final
/// window or restarts from a fire-tail residual is *not* the clean
/// loop — it composes the finer primitives inline.
#[derive(Debug)]
pub struct RebaseN2 {
    pub s1: StepOutput,
    pub rearm: StepOutput,
    pub finish: StepOutput,
    pub finish_at: Instant,
}

#[must_use]
pub fn rebase_loop_to_idle(
    e: &mut Engine,
    pid: ProfileId,
    snap: &Arc<DirSnapshot>,
    start: Instant,
) -> RebaseN2 {
    let corr1 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("rebase probe #1 in flight");
    let s1 = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr1,
            outcome: proven(Arc::clone(snap)),
        }),
        start,
    );
    let spacing = rebase_settling_spacing_id(e, pid);
    let finish_at = start + SETTLE;
    let rearm = e.step(
        Input::TimerExpired {
            profile: pid,
            kind: TimerKind::RebaseSettle,
            id: spacing,
        },
        finish_at,
    );
    let corr2 = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("RebaseSettle re-arms the Rebasing probe #2");
    let finish = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr2,
            outcome: proven(Arc::clone(snap)),
        }),
        finish_at,
    );
    RebaseN2 {
        s1,
        rearm,
        finish,
        finish_at,
    }
}

/// Respond to `owner`'s single in-flight descent probe with
/// `DirEnumerated(snap)`; the engine advances one path component (or
/// materialises the anchor, opening a Seed).
///
/// Returns the full `StepOutput`: the caller reads the next descent
/// correlation, asserts the terminal Seed/proxy shape, or asserts
/// no-progress. Descent runs outside the Burst lifecycle, so this is a
/// primitive beside [`seed_settle_to_verifying`], composing neither.
#[must_use]
pub fn descent_advance(
    e: &mut Engine,
    owner: ProbeOwner,
    snap: &Arc<DirSnapshot>,
    at: Instant,
) -> StepOutput {
    let correlation = e.pending_probe_for(owner).expect("descent probe in flight");
    e.step(
        Input::ProbeResponse(ProbeResponse {
            owner,
            correlation,
            outcome: enumerated(Arc::clone(snap)),
        }),
        at,
    )
}

/// Drive the burst's single completed Effect (`key`) into the post-fire
/// rebase loop: step `EffectComplete::Ok`, returning the step output
/// and the fresh rebase probe correlation.
///
/// The single-`Ok`, single-effect prologue before
/// [`rebase_loop_to_idle`]. Multi-effect / `Failed`-mix awaiting tests
/// keep their explicit loop inline (they assert the
/// outstanding-count decrement).
#[must_use]
pub fn complete_effect_to_rebasing(
    e: &mut Engine,
    sid: SubId,
    key: DedupKey,
    at: Instant,
) -> (StepOutput, ProbeCorrelation) {
    let pid = pid_of(e, sid);
    let out = e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        at,
    );
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("rebase probe in flight after single-Ok effect completion");
    (out, correlation)
}

/// The fixture `PromoterAttachRequest`: `recursive`, `ClassSet::EMPTY`,
/// `MAX_SETTLE`/`SETTLE`, `/bin/true`, `EffectScope::SubtreeRoot`.
#[must_use]
pub fn promoter_req(name: &str, pattern: &str) -> PromoterAttachRequest {
    PromoterAttachRequest {
        name: name.into(),
        pattern_spec: PatternSpec::parse(pattern).expect("valid test pattern"),
        identity: ProfileIdentity {
            config: ScanConfig::builder().recursive(true).build(),
            max_settle: MAX_SETTLE,
            events: ClassSet::EMPTY,
        },
        settle: SETTLE,
        program: specter_core::testkit::empty_program(),
        scope: EffectScope::SubtreeRoot,
        log_output: false,
    }
}

/// Ensure a `User`-roled `Dir` root named `name`, returning its id.
///
/// The single-root sibling of [`pre_place_dir`] (an FS-root chain): the
/// anchor a subtree-root Sub attaches at. A File-anchor test overrides
/// the kind at the call site (`set_kind(.., File)`) — exactly the shape
/// the hand-rolled copies carried.
pub fn anchor_dir(e: &mut Engine, name: &str) -> ResourceId {
    let r = e.tree_mut().ensure_root(name, ResourceRole::User);
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

/// Pre-place a `User`-roled `Dir` chain (FS-root through `segments`),
/// returning the deepest resource.
pub fn pre_place_dir(e: &mut Engine, segments: &[&str]) -> ResourceId {
    let mut comps = Vec::with_capacity(segments.len() + 1);
    comps.push(FS_ROOT_SEGMENT);
    comps.extend_from_slice(segments);
    let r = e
        .tree_mut()
        .ensure_path(&comps, ResourceRole::User)
        .expect("non-empty fixture");
    e.tree_mut().set_kind(r, ResourceKind::Dir);
    r
}

/// `Input::WatchOpRejected` with the fixture failure
/// (`Pressure { errno: 24 }`).
#[must_use]
pub const fn watch_op_rejected_input(resource: ResourceId) -> Input {
    Input::WatchOpRejected {
        resource,
        failure: WatchFailure::Pressure { errno: 24 },
    }
}

/// Attach the fixture Promoter for `pattern`, returning the attach
/// `StepOutput` too — for tests asserting on the attach step.
#[must_use]
pub fn attach_promoter_returning(
    e: &mut Engine,
    name: &str,
    pattern: &str,
    now: Instant,
) -> (PromoterId, StepOutput) {
    let out = e.step(Input::AttachPromoter(promoter_req(name, pattern)), now);
    let qid =
        specter_core::testkit::first_attached_promoter(&out).expect("attach_promoter succeeded");
    (qid, out)
}

/// [`attach_promoter_returning`] discarding the attach `StepOutput`.
#[must_use]
pub fn attach_promoter(e: &mut Engine, name: &str, pattern: &str, now: Instant) -> PromoterId {
    let (qid, _) = attach_promoter_returning(e, name, pattern, now);
    qid
}

/// The live `(anchor → SubId)` set for Promoter `pid`, derived from
/// `SubRegistry` truth (the single source post-`dynamic_subs` removal).
#[must_use]
pub fn dynamic_subs_of(e: &Engine, pid: PromoterId) -> BTreeMap<ResourceId, SubId> {
    e.subs()
        .iter()
        .filter(|(_, s)| s.source_promoter == Some(pid))
        .map(|(sid, s)| {
            let anchor = e
                .profiles()
                .get(s.profile())
                .expect("a live dynamic Sub's Profile is live")
                .resource();
            (anchor, sid)
        })
        .collect()
}

/// The latest outstanding probe target-path in emission order.
#[must_use]
pub fn last_probe_path(out: &StepOutput) -> Option<PathBuf> {
    out.probe_ops().iter().rev().find_map(|op| match op {
        ProbeOp::Probe { request } => Some(request.target_path().to_path_buf()),
        ProbeOp::Cancel { .. } => None,
    })
}
