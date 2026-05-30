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
    FS_ROOT_SEGMENT, FsEvent, Input, PatternSpec, PostFireBurst, PostFirePhase, PreFireBurst,
    PreFirePhase, ProbeCorrelation, ProbeOp, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId,
    ProfileIdentity, ProfileState, PromoterAttachRequest, PromoterId, ResourceId, ResourceKind,
    ResourceRole, ScanConfig, StepOutput, SubAttachAnchor, SubAttachRequest, SubId, TimerId,
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
///
/// Events-incomplete — `events_witness_quiescence == false`, so fire-bearing
/// bursts owe the verdict floor's N=2 hash-equality channel. Tests that
/// drive a single Authoritative sample to Stable must opt into a CONTENT-
/// containing mask (e.g. [`DEFAULT_EVENTS`]) instead.
pub const NO_EVENTS: ClassSet = ClassSet::EMPTY;
/// Production-realistic `EffectScope::SubtreeRoot` events mask.
///
/// CONTENT in the mask sets `events_witness_quiescence == true`, so a
/// single Authoritative sample closes the verdict floor's hash-equality
/// obligation (witness = `EventsReliable`).
pub const DEFAULT_EVENTS: ClassSet = ClassSet::DEFAULT_SUBTREE_ROOT;

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

/// Read `pid`'s `Active(PostFire(Settling))` settle-timer id, or panic
/// with the actual state — the post-fire sibling of
/// [`batching_settle_id`].
#[must_use]
pub fn post_fire_settle_id(e: &Engine, pid: ProfileId) -> TimerId {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(
            ActiveBurst::PostFire(PostFireBurst {
                phase: PostFirePhase::Settling { settle_timer },
                ..
            }),
            _,
        ) => *settle_timer,
        other => panic!("expected {pid:?} in Active(PostFire(Settling)), got {other:?}"),
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

/// Attach a subtree-root Sub with `events: STRUCTURE`, returning
/// `(SubId, ProfileId)`.
///
/// The resulting Profile fails [`specter_core::Profile::events_witness_quiescence`]:
/// settle-window silence is **not** a sufficient quiescence witness on a
/// `STRUCTURE`-only mask, since in-place writes fire `CONTENT` events
/// that this mask drops at the per-Profile class filter. Fire-bearing
/// bursts on this Profile owe the verdict floor's hash-equality channel.
///
/// Fixture defaults: `MAX_SETTLE`, recursive [`ScanConfig`], name
/// `"build"`. A test needing a non-recursive scan or a different
/// `max_settle` reaches for [`attach`] directly. The helper exists
/// because the `STRUCTURE`-only mask is the canonical regression
/// scenario (the `scp` user-reported bug) — every Layer-C inventory
/// test starts here.
#[must_use]
pub fn attach_structure_only(
    e: &mut Engine,
    anchor: ResourceId,
    now: Instant,
) -> (SubId, ProfileId) {
    let (sid, pid) = attach(
        e,
        "build",
        SubAttachAnchor::Resource(anchor),
        ScanConfig::builder().recursive(true).build(),
        ClassSet::STRUCTURE,
        MAX_SETTLE,
        now,
    );
    debug_assert!(
        !e.profiles()
            .get(pid)
            .expect("attach_structure_only just attached")
            .events_witness_quiescence(),
        "attach_structure_only must produce an events-incomplete Profile",
    );
    (sid, pid)
}

/// Drive a Standard burst from `start_event` through the verdict floor's
/// retry loop to a fire.
///
/// Opens the burst with an [`FsEvent::StructureChanged`] on `start_event`
/// at `now` — the canonical opening event for the events-incomplete
/// `STRUCTURE`-only target the helper is named after; it also opens a
/// burst on any wider mask that contains `STRUCTURE` (the anchor's
/// class-filter bypass is sufficient for narrower masks the helper
/// might be reused against). Then runs one `Batching → settle expiry →
/// Verifying → response` cycle per element of `responses`, returning
/// the first cycle's [`StepOutput`] that carries effects.
///
/// Each cycle advances time by `SETTLE * 2` — well past the freshly-armed
/// settle timer's `last_event_time + SETTLE` expiry without bumping into
/// the burst-deadline ceiling at `now + MAX_SETTLE`.
///
/// Under the Pass-1 verdict floor (`Authoritative ⇒ fire`),
/// `responses.len() == 1` is the canonical shape: the first sample's
/// Authoritative fold fires unconditionally. Under the post-Layer-C
/// verdict floor for an events-incomplete fire-bearing burst, two
/// consecutive samples must agree on the response hash for `Stable`,
/// so the canonical shape is the three-sample slow-writer pattern
/// `[s1, s2, s2]` — `read1`, then `read2 ≠ read1` re-loops, then
/// `read3 = read2` fires. The helper is invariant on which side of the
/// refactor it runs against: the loop reads only the StepOutput's
/// effects, and both verdict shapes converge through it.
///
/// Panics if `responses` is empty or the loop exhausts the slice
/// without firing — the test author chose the sample sequence, so a
/// non-fire is a test setup bug.
#[must_use]
pub fn drive_standard_n2_until_stable(
    e: &mut Engine,
    pid: ProfileId,
    start_event: ResourceId,
    responses: &[Arc<DirSnapshot>],
    now: Instant,
) -> StepOutput {
    assert!(
        !responses.is_empty(),
        "drive_standard_n2_until_stable needs at least one verify-sample snapshot",
    );

    let _ = e.step(
        Input::FsEvent {
            resource: start_event,
            event: FsEvent::StructureChanged,
        },
        now,
    );

    let mut at = now;
    for snap in responses {
        at += SETTLE * 2;
        drain_due(e, at);
        let corr = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight after settle expiry");
        let out = e.step(
            Input::ProbeResponse(ProbeResponse {
                owner: ProbeOwner::Profile(pid),
                correlation: corr,
                outcome: proven(Arc::clone(snap)),
            }),
            at,
        );
        if !out.effects().is_empty() {
            return out;
        }
    }
    panic!(
        "drive_standard_n2_until_stable: {} sample(s) exhausted without firing",
        responses.len()
    );
}

/// Assert `pid` is in cold-arm `Active(PreFire(Verifying))` and read
/// its in-flight probe correlation.
///
/// Post-`start_seed_burst(None)` shape under the cold-arm Verifying-
/// first contract: the probe is in flight at burst construction (no
/// Batching settle to drain on the way in), so the helper is a state
/// projection — it asserts the phase and returns the correlation
/// plus `at` unchanged so call sites stay symmetric with helpers that
/// do advance time.
#[must_use]
pub fn assert_seed_verifying(
    e: &mut Engine,
    pid: ProfileId,
    at: Instant,
) -> (ProbeCorrelation, Instant) {
    match e.profiles().get(pid).unwrap().state() {
        ProfileState::Active(ActiveBurst::PreFire(pre), _) => assert!(
            matches!(pre.phase, PreFirePhase::Verifying { .. }),
            "expected {pid:?} in cold-arm Active(PreFire(Verifying)), got phase {:?}",
            pre.phase,
        ),
        other => panic!("expected {pid:?} Active(PreFire(Verifying)), got {other:?}"),
    }
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-arm Seed Verifying probe in flight at burst construction");
    (correlation, at)
}

/// Drive a fresh cold-arm Seed burst for `pid` through its quiescence
/// verdict to pinned `Idle`, answering the cold walk probe with
/// `make_outcome()`.
///
/// The cold-arm Seed burst pins on the first `Authoritative` response:
/// a `SilentPin` consequence does not owe quiescence proof, so the
/// witness is [`QuiescenceWitness::EventsReliable`] and the fold reaches
/// `Stable(StableReason::Natural)`; dispatch then commits the `SilentPin`
/// (a fresh Seed with no activity) and the burst finishes to Idle. The
/// cold-arm Verifying-first contract puts the probe in flight at burst
/// construction — no Batching settle to drain on the way in.
///
/// `start` is the instant the Seed burst was constructed; the probe
/// response steps at `start + SETTLE` (one settle window past the
/// cold-arm so later bursts get a clean rebase window). Returns the
/// step instant so the caller rebases later bursts past it. A fresh
/// Seed emits no Effects.
#[must_use]
pub fn seed_to_idle_with(
    e: &mut Engine,
    pid: ProfileId,
    make_outcome: impl Fn() -> ProbeOutcome,
    start: Instant,
) -> Instant {
    let correlation = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("cold-arm Seed Verifying probe in flight at burst construction");
    let at = start + SETTLE;
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
        "a fresh Seed never fires an Effect (single Authoritative pin)",
    );
    assert!(
        matches!(e.profiles().get(pid).unwrap().state(), ProfileState::Idle),
        "one Authoritative Seed sample pins the baseline → Idle for {pid:?}",
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

/// The pre-fire verify dispatch outcome for `pid`: the response
/// [`StepOutput`] and the [`Instant`] the response stepped at.
///
/// Single-sample shape: under the verdict floor's
/// `EventsReliable` witness one Authoritative response folds to
/// `Stable`, so this struct carries exactly one [`StepOutput`]. Tests
/// driving an events-incomplete N=2 retry loop chain two
/// [`verify`] calls (first sample → re-Batch; second sample → Stable)
/// or reach for [`drive_standard_n2_until_stable`] when the
/// intermediate state is uninteresting.
#[derive(Debug)]
pub struct Verify {
    pub out: StepOutput,
    pub at: Instant,
}

/// The pre-fire verify dispatch for `pid`, answering the single
/// in-flight probe with `make_outcome()`.
///
/// `pid` is already in `Active(PreFire(Verifying))` (the caller has
/// drained the settle timer driving `Batching → Verifying`). The probe
/// response steps at `start + SETTLE` — one settle window past `start`
/// to give later operations a clean instant — well within the burst's
/// `MAX_SETTLE` ceiling.
///
/// The response folds through
/// [`specter_core::quiescence_verdict`]. For an events-reliable Profile
/// or a non-fire-bearing burst (cold Seed → `SilentPin`) the
/// single Authoritative sample reaches `Stable` and the dispatch fires
/// or pins inline. For an events-incomplete fire-bearing burst the
/// first sample (carrier `prior = None`) folds to
/// [`specter_core::QuiescenceVerdict::Retry`] and the helper returns
/// the re-Batch step; the caller drains the freshly-armed settle timer
/// and calls [`verify`] again for the second sample.
#[must_use]
pub fn verify_with(
    e: &mut Engine,
    pid: ProfileId,
    make_outcome: impl Fn() -> ProbeOutcome,
    start: Instant,
) -> Verify {
    let corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("Verifying probe in flight at verify_with entry");
    let at = start + SETTLE;
    let out = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: make_outcome(),
        }),
        at,
    );
    Verify { out, at }
}

/// [`verify_with`] answering the sample with `proven(snap)` — the
/// `Subtree` common case for the pre-fire verify dispatch.
#[must_use]
pub fn verify(e: &mut Engine, pid: ProfileId, snap: &Arc<DirSnapshot>, start: Instant) -> Verify {
    verify_with(e, pid, || proven(Arc::clone(snap)), start)
}

/// The clean post-fire rebase to Idle for `pid`.
///
/// `pid` is already in `Active(PostFire(Rebasing))` with the rebase
/// probe in flight — `EffectComplete` drove `Awaiting → Rebasing`
/// directly (probe-first), so there is no settle window before the
/// first sample. Answers the in-flight rebase probe with `proven(snap)`;
/// the response folds through `quiescence_verdict` to
/// `Stable(StableReason::Natural)` → commit + finish (or restart on a
/// non-empty residual).
///
/// Returns the finish step output and its instant.
/// **Carve-out:** a test that injects a custom absorb in the final
/// window or exercises the [`specter_core::QuiescenceVerdict::Retry`]
/// loop-back (which re-enters `Settling`) composes the finer primitives
/// inline.
#[derive(Debug)]
pub struct RebasePostFire {
    pub finish: StepOutput,
    pub finish_at: Instant,
}

#[must_use]
pub fn rebase_post_fire_to_idle(
    e: &mut Engine,
    pid: ProfileId,
    snap: &Arc<DirSnapshot>,
    at: Instant,
) -> RebasePostFire {
    let corr = e
        .pending_probe_for(ProbeOwner::Profile(pid))
        .expect("EffectComplete drove Awaiting → Rebasing with the rebase probe in flight");
    let finish = e.step(
        Input::ProbeResponse(ProbeResponse {
            owner: ProbeOwner::Profile(pid),
            correlation: corr,
            outcome: proven(Arc::clone(snap)),
        }),
        at,
    );
    RebasePostFire {
        finish,
        finish_at: at,
    }
}

/// Respond to `owner`'s single in-flight descent probe with
/// `DirEnumerated(snap)`; the engine advances one path component (or
/// materialises the anchor, opening a Seed).
///
/// Returns the full `StepOutput`: the caller reads the next descent
/// correlation, asserts the terminal Seed/proxy shape, or asserts
/// no-progress. Descent runs outside the Burst lifecycle, so this is a
/// primitive beside [`assert_seed_verifying`], composing neither.
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

/// Drive the burst's single completed Effect (`key`) into Rebasing.
///
/// Steps `EffectComplete::Ok`; the burst advances `Awaiting → Rebasing`
/// (probe-first) via `on_effect_complete::LastReached + ReturnToIdle`,
/// minting the `WholeSubtree` rebase probe immediately, and the step
/// output is returned. This is the single-`Ok`, single-effect prologue
/// before [`rebase_post_fire_to_idle`], which answers the now-in-flight
/// probe. Multi-effect / `Failed`-mix awaiting tests keep their explicit
/// loop inline (they assert the outstanding-count decrement).
///
/// **Carve-out callers** that need the rebase probe correlation
/// (e.g. answering with a custom `Vanished` / `Failed` outcome) read it
/// via `pending_probe_for(ProbeOwner::Profile(pid))` right after this
/// step — the probe is already in flight, no settle expiry between.
#[must_use]
pub fn complete_effect_to_rebasing(
    e: &mut Engine,
    sid: SubId,
    key: DedupKey,
    at: Instant,
) -> StepOutput {
    e.step(
        Input::EffectComplete(EffectCompletion {
            sub: sid,
            key,
            outcome: EffectOutcome::Ok,
        }),
        at,
    )
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
