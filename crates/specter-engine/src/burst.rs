//! Burst lifecycle helpers.
//!
//! These helpers are the single source of **category (a):
//! variant/phase transitions** — a phase body, a Burst construction,
//! or a return-to-Idle. Centralizing the timer scheduling, refcount
//! edges, and phase-variant rewrites here prevents drift between the
//! transition-row handlers and the post-`EffectComplete` re-probe path.
//!
//! They are **not** the only writers of `PreFireBurst` /
//! `PostFireBurst` — by construction, not drift. Two other categories
//! own their own single source:
//!
//! - **(b) Single-field load-bearing-invariant edges**: a typed
//!   edge-method on the field's owner in `specter-core` (the method
//!   *is* the floor — total fn, no public setter). `note_effect_completion`
//!   is the surviving member; a phase helper here would only enforce
//!   it at a distance. (The old `apply_dirty_delta` counter edge was
//!   deleted with the `dirty_descendants` refcount — the
//!   `Draining → Verifying` reconfirm is now a fresh query, not a
//!   maintained count.)
//! - **(c) The sanctioned cross-crate emission drain**:
//!   [`Engine::emit_owner_probe`] (in `probe`), sole consumer of
//!   `force_walk_resources`. Its `pub` burst accessors are
//!   load-bearing and deliberately *not* sealed (Rust visibility is
//!   intra-crate; the choke reaches them from another crate).
//!
//! `ActiveBurst` splits into `PreFireBurst` / `PostFireBurst` (see
//! [`specter_core::profile`]); helpers below own a typed view of one or
//! the other. Two typed state-machine moves cross the split: the fire
//! transition (`Verifying → Awaiting`) at
//! [`PreFireBurst::into_post_fire`], and its inverse — the post-rebase
//! residual restart (`Rebasing → Batching`) at
//! [`PostFireBurst::into_pre_fire_residual`].
//!
//! - `start_seed_burst` / `start_standard_burst` — Idle →
//!   `Active(PreFire(_))`.
//! - `event_drives_batching` (FsEvent during pre-fire) /
//!   `unstable_response_drives_batching` (probe-unstable response) /
//!   `transition_to_verifying` (settle-timer expiry, burst-deadline,
//!   Draining → Verifying reconfirm) /
//!   `transition_to_draining` — pre-fire phase swaps (mutate
//!   `PreFireBurst`).
//! - `reschedule_batching` (settle-timer re-point, phase class
//!   unchanged) / `force_pending` (`forced` flag on burst-deadline) —
//!   timer-expiry single-field `PreFireBurst` mutators; the caller
//!   keeps the timer math and the phase-routing decision.
//! - `transition_to_awaiting` — `Active(PreFire(_))` → `Active(PostFire(_))`,
//!   the sole site that crosses the fire boundary (via
//!   `PreFireBurst::into_post_fire`).
//! - `transition_to_rebasing` — `Awaiting → Rebasing` (mutates
//!   `PostFireBurst`).
//! - `absorb_event_into_fire_tail` — FsEvent during post-fire (mutates
//!   `PostFireBurst.force_walk_resources`).
//! - `restart_burst_from_fire_tail_residual` — `Active(PostFire)` →
//!   `Active(PreFire(Batching))` typed move at rebase-ok when a
//!   `ReturnToIdle` burst carries a non-empty residual (origin-agnostic
//!   — Seed-origin restarts too). No refcount edges: the typed move
//!   preserves the watched anchor, neither installing nor releasing a
//!   contribution.
//! - `finish_burst_to_idle` — Active → Idle; then sweeps the Draining
//!   Profiles and reconfirms each whose fresh covered-descendant query
//!   has gone false. Discriminates `PreFire` / `PostFire` at the take.
//!
//! The two batching helpers exist as a deliberate split rather than one
//! helper with a runtime flag: each caller has **static knowledge** of
//! whether a probe is in flight (only `event_drives_batching` may need to
//! emit `ProbeOp::Cancel`). Encoding that knowledge as helper identity
//! makes a stray Cancel on the just-responded path structurally
//! impossible.
//!
//! Probe emission flows through two structural primitives:
//!
//! - [`pre_fire_target`] — pure function returning the `ResourceId` the
//!   next pre-fire probe should target. Centralizes the
//!   `(anchor_kind, intent)` rule (File anchor → anchor; Seed → anchor;
//!   Standard → LCA of `dirty_resources`). Post-fire rebases target the
//!   anchor unconditionally and bypass this helper.
//!   `transition_to_verifying` resolves the target through it and writes
//!   it onto `pre.probe_target` for the choke to read back.
//! - [`Engine::emit_owner_probe`] (in `probe`) — the single
//!   owner-polymorphic emission choke. Each burst-launch helper is
//!   `mint → arm (loud) → emit_owner_probe(owner)`; the choke resolves
//!   the owner's state once, reads the correlation back off the armed
//!   slot, kind-dispatches, and drains the force-walk accumulator.
//!   Unclassified anchors take the Subtree arm — the walker returns
//!   `Vanished` on kind mismatch and the engine recovers via descent.

use crate::Engine;
use smallvec::SmallVec;
use specter_core::{
    ActiveBurst, BurstFinish, BurstHelper, BurstIntent, Diagnostic, FsEvent, LcaIntegritySource,
    PostFirePhase, PreFireBurst, PreFirePhase, ProbeOwner, ProbeSlot, Profile, ProfileId,
    ProfileState, ReapTrigger, ResourceId, ResourceKind, StepOutput, TimerId, TimerKind, Tree,
    TreeSnapshot,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

impl Engine {
    /// Precondition gate for the Active(PreFire(_)) burst helpers. Returns
    /// `true` iff `profile_id` is live AND in `Active(PreFire(_))`; on a
    /// state mismatch emits [`Diagnostic::InvalidBurstTransition`] and
    /// returns `false`. A stale `ProfileId` (no live slot) is a benign
    /// post-detach race and returns `false` silently — the diagnostic is
    /// reserved for genuine state-machine routing breaches.
    ///
    /// **Why a single gate rather than ad-hoc match-and-return.** Every
    /// pre-fire helper opens by reading the Profile's state; the prior
    /// shape used a `let ... else { return; }` projection that silently
    /// dropped misrouted calls. Routing the precondition through this
    /// gate keeps the silent-return semantics on stale ids (the engine
    /// already handles slot reaping at the dispatch level) while
    /// surfacing routing breaches as a typed diagnostic — operators see
    /// the helper name + observed state and can map straight back to the
    /// caller.
    ///
    /// **Pairs with [`Self::require_active_post_fire`] /
    /// [`Self::require_idle`]** — these three cover every helper with a
    /// typed entry precondition. `finish_burst_to_idle` is intentionally
    /// idempotent (handles Idle / Pending as silent no-op) and bypasses
    /// the gate, as do the timer-expiry single-field mutators
    /// [`Self::reschedule_batching`] / [`Self::force_pending`]: their
    /// callers are reached only through `is_timer_referenced`, so the
    /// non-pre-fire arm is structurally unreachable and a routing
    /// diagnostic there would be spurious — they silently no-op instead.
    fn require_active_pre_fire(
        &self,
        profile_id: ProfileId,
        helper: BurstHelper,
        out: &mut StepOutput,
    ) -> bool {
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        if matches!(p.state(), ProfileState::Active(ActiveBurst::PreFire(_), _)) {
            return true;
        }
        out.diagnostics.push(Diagnostic::InvalidBurstTransition {
            profile: profile_id,
            helper,
            observed: p.state().discriminant(),
        });
        false
    }

    /// Precondition gate for the Active(PostFire(_)) burst helpers.
    /// Mirrors [`Self::require_active_pre_fire`] on the post-fire side
    /// of the type split.
    ///
    /// `transition_to_rebasing`'s callers (the `Rebase` arm of
    /// `on_effect_complete` and `handle_gate_deadline`) further narrow
    /// to `PostFirePhase::Awaiting` before invoking, but the gate stops
    /// at the variant level — narrower phase-level preconditions would
    /// duplicate the caller-side check without surfacing additional
    /// routing breaches (a phase-level mismatch within PostFire is
    /// caught by the helper's inner phase match instead).
    fn require_active_post_fire(
        &self,
        profile_id: ProfileId,
        helper: BurstHelper,
        out: &mut StepOutput,
    ) -> bool {
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        if matches!(p.state(), ProfileState::Active(ActiveBurst::PostFire(_), _)) {
            return true;
        }
        out.diagnostics.push(Diagnostic::InvalidBurstTransition {
            profile: profile_id,
            helper,
            observed: p.state().discriminant(),
        });
        false
    }

    /// Precondition gate for the burst-construction helpers
    /// (`start_seed_burst`, `start_standard_burst`). Both transition Idle
    /// → `Active(PreFire(_))`; calling either on a non-Idle Profile is a
    /// routing breach. Stale ids return false silently — same policy as
    /// the Active gates above.
    ///
    /// Replaces the prior `debug_assert!(matches!(p.state,
    /// ProfileState::Idle))` discipline: that variant panicked in
    /// dev/CI and silently misrouted in release. The diagnostic
    /// emission is visible in both build modes and survives via the
    /// usual `StepOutput.diagnostics` plumbing.
    fn require_idle(
        &self,
        profile_id: ProfileId,
        helper: BurstHelper,
        out: &mut StepOutput,
    ) -> bool {
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        if matches!(p.state(), ProfileState::Idle) {
            return true;
        }
        out.diagnostics.push(Diagnostic::InvalidBurstTransition {
            profile: profile_id,
            helper,
            observed: p.state().discriminant(),
        });
        false
    }

    /// Start a Seed burst: no settle wait, immediate Probe.
    ///
    /// **Callers** (post-`Awaiting`/`Rebasing` lifecycle):
    /// - `attach_sub_inner` immediate-Seed path (fresh attach, anchor
    ///   materialised on disk).
    /// - `dispatch_descent_ok` anchor materialization (descent terminus).
    /// - `on_sensor_overflow` Idle path (reseed every Profile in scope).
    /// - `on_sensor_overflow` Active path (after `finish_burst_to_idle`).
    /// - `drive_burst`'s Idle + `current.is_none()` branch (post-Vanished
    ///   re-arming when an event arrives at a still-watched anchor).
    ///
    /// `EffectComplete::Ok` does NOT call this helper; post-Effect rebase
    /// routes through `transition_to_rebasing`.
    ///
    /// Caller has verified `Profile.state == Idle`. Schedules
    /// `burst_deadline` and constructs the `Verifying` phase armed with
    /// a fresh correlation (`mint → arm → emit_owner_probe`). The choke
    /// shapes the request — anchor target
    /// plus `current.subtree_at(anchor)` as `baseline_subtree` when a
    /// post-recovery Seed has one (walker mtime-skip on idempotent
    /// events).
    pub(crate) fn start_seed_burst(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_idle(profile_id, BurstHelper::StartSeedBurst, out) {
            return;
        }
        // `require_idle` confirmed the Profile is live + Idle; the
        // re-borrow below is for captures (`resource`, `max_settle`),
        // not a re-check. In v1's single-threaded `step`, no mutation
        // intervenes between the precondition and this read.
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let resource = p.resource;
        let max_settle = p.max_settle();

        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        // Mint, then construct the `Verifying` phase already armed with
        // the correlation: minting *is* constructing the slot, so a
        // verify phase without its correlation has no representable
        // window. I5 holds by representability (the fresh `Active`
        // carrier owns exactly one slot); the assert is the loud
        // dev/CI backstop that the prior state had nothing in flight.
        let owner = ProbeOwner::Profile(profile_id);
        debug_assert!(
            self.pending_probe_for(owner).is_none(),
            "I5: start_seed_burst on a Profile with a probe already in flight \
             (the construct-armed slot would orphan the prior correlation, \
             profile = {profile_id:?})",
        );
        let correlation = self.mint_probe_correlation();

        // Loud arm. `require_idle` proved the Profile live + Idle, so
        // `get_mut` resolving `None` here means the state machine broke
        // between the precondition and the arm — a silent skip would
        // leave a fresh `Verifying` slot un-constructed while the emit
        // below still fires (no probe, no diagnostic: a wedge). The
        // construct-armed slot makes this site structurally safe, but
        // the loud arm is non-optional at every launch site by one
        // discipline, not a per-site judgement.
        let Some(p) = self.profiles.get_mut(profile_id) else {
            unreachable!(
                "start_seed_burst: Profile {profile_id:?} vanished between \
                 require_idle and the construct-armed transition"
            );
        };
        p.transition_state(ProfileState::Active(
            ActiveBurst::PreFire(PreFireBurst {
                burst_deadline,
                phase: PreFirePhase::Verifying(ProbeSlot::armed(correlation, ())),
                intent: BurstIntent::Seed,
                forced: false,
                dirty_resources: BTreeSet::new(),
                force_walk_resources: BTreeSet::new(),
                // Seed targets the anchor; the field is invariant for the
                // Seed burst's pre-fire lifetime (`transition_to_verifying`
                // re-runs for Seed only on Draining-reconfirm, which Seed
                // bursts never reach because they skip Batching).
                probe_target: resource,
                // Seed bursts skip Batching; the field has no consumer
                // until a fresh FsEvent during the verify routes through
                // `event_drives_batching` and repopulates it.
                last_event_time: None,
            }),
            // Fresh burst — directive starts at `ReturnToIdle`. Flips
            // to `Reap` only on mid-burst `mark_active_for_reap`.
            BurstFinish::ReturnToIdle,
        ));

        // The choke reads the correlation back off the Verifying slot
        // just constructed, resolves the anchor target off
        // `pre.probe_target`, and drains the (empty, fresh-Seed)
        // force-walk accumulator — the caller threads nothing.
        self.emit_owner_probe(owner, out);
    }

    /// Start a Standard burst: schedule settle + `burst_deadline`. No
    /// Probe — that fires on `settle_timer` expiry via
    /// `transition_to_verifying`. No ancestor bookkeeping: the
    /// `Draining → Verifying` reconfirm is a fresh query
    /// ([`crate::coverage::has_active_standard_descendant`]) over the
    /// live tree, so a burst start contributes nothing to maintain.
    ///
    /// `event_resource` is the `FsEvent`'s source. It seeds both
    /// `dirty_resources` (basis for the next probe's LCA) and
    /// `force_walk_resources` (defeats mtime-skip on event-dirty paths).
    pub(crate) fn start_standard_burst(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_idle(profile_id, BurstHelper::StartStandardBurst, out) {
            return;
        }
        // Re-borrow for captures; the precondition has already confirmed
        // the Profile is live + Idle.
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let resource = p.resource;
        let settle = p.settle;
        let max_settle = p.max_settle();

        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);
        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        let mut dirty = BTreeSet::new();
        dirty.insert(event_resource);
        let mut force_walk = BTreeSet::new();
        force_walk.insert(event_resource);

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.transition_state(ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    burst_deadline,
                    phase: PreFirePhase::Batching { settle_timer },
                    intent: BurstIntent::Standard,
                    forced: false,
                    dirty_resources: dirty,
                    force_walk_resources: force_walk,
                    // Initial target = anchor. `transition_to_verifying`
                    // overwrites with the LCA of `dirty_resources` on settle
                    // expiry / force-fire; the initial value carries no
                    // observable consequence (no probe has emitted yet).
                    probe_target: resource,
                    // The burst-start FsEvent IS the first event; seed the
                    // settle-deadline source of truth with `now`. Subsequent
                    // events update this in `event_drives_batching` without
                    // re-inserting a fresh heap entry.
                    last_event_time: Some(now),
                }),
                // Fresh burst — directive starts at `ReturnToIdle`. Flips
                // to `Reap` only on mid-burst `mark_active_for_reap`.
                BurstFinish::ReturnToIdle,
            ));
        }
    }

    /// Caller: `drive_burst` Active branch — an `FsEvent` arrived during a
    /// burst. Cancels any in-flight verify (iff the prior phase was
    /// `Verifying`), accumulates the event into `dirty_resources` and
    /// `force_walk_resources`, updates `last_event_time`, arms a fresh
    /// settle timer **only when re-entering Batching from Verifying or
    /// Draining**, and writes `phase = Batching { settle_timer }`.
    /// `intent`, `forced`, and the `burst_deadline` are preserved.
    ///
    /// Why this is one of two batching mutators rather than a single
    /// helper with a flag: the caller has static knowledge that the
    /// engine has not just received a probe response. If the prior phase
    /// was `Verifying`, a verify is in flight and we must Cancel it. If
    /// the prior phase was `Batching` or `Draining`, no probe is in
    /// flight. Encoding that as a runtime flag is a category error — the
    /// caller always knows the right answer.
    ///
    /// **Settle-timer reuse.** In steady-state Batching the live
    /// settle timer's heap entry is preserved; the per-event update is
    /// just `last_event_time = Some(now)`. The on-expiry handler
    /// (`Engine::on_settle_expired`) reschedules a fresh entry at
    /// `last_event_time + settle` if events arrived since, otherwise
    /// transitions to Verifying. This collapses the per-event
    /// `BinaryHeap::push` that previously orphaned the prior entry to
    /// at most one push per `last_event_time + settle` boundary —
    /// roughly `ceil(burst_duration / settle)` settle-timer entries
    /// per burst, instead of one per `FsEvent`.
    ///
    /// Re-entries from `Verifying` or `Draining` have no live settle
    /// timer to reuse and therefore schedule a fresh entry. `Batching`
    /// re-entries skip the schedule.
    pub(crate) fn event_drives_batching(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::EventDrivesBatching, out) {
            return;
        }
        // Re-borrow for captures + the phase projection used to decide
        // whether a fresh settle timer is needed below. The precondition
        // gate already emitted a diagnostic on any state mismatch (e.g.,
        // a PostFire that should have routed through
        // `absorb_event_into_fire_tail`); the inner projection is the
        // borrow-checker discipline for reading `pre.phase`.
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(ActiveBurst::PreFire(pre), _) = p.state() else {
            return;
        };
        let settle = p.settle;

        // Read phase before mutating self via `cancel_owner_probe`. The
        // Cancel emission doesn't touch `burst.phase`, but it does take
        // `&mut self` and so invalidates the borrow on `burst`. Decide
        // here whether the existing Batching settle timer (if any) carries
        // over, or whether we mint a fresh one for a Verifying/Draining
        // re-entry. The decision is structural: a live Batching has its
        // own timer slot; Verifying/Draining have none.
        let needs_fresh_timer = matches!(
            pre.phase,
            PreFirePhase::Verifying(_) | PreFirePhase::Draining
        );

        // Idempotent: disarms the `Verifying` slot and emits Cancel iff
        // a probe was in flight. For Batching / Draining entries no slot
        // is armed and the helper is a no-op. On the Verifying path this
        // leaves a transient `Verifying(ProbeSlot::empty())` until the
        // phase is rewritten to `Batching` below — a single-step,
        // unobserved, fully representable window; the consume is the
        // disarm here, not the later phase rewrite.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);

        let new_settle_timer = if needs_fresh_timer {
            Some(
                self.timers
                    .schedule(now + settle, profile_id, TimerKind::Settle),
            )
        } else {
            None
        };

        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            pre.last_event_time = Some(now);
            pre.dirty_resources.insert(event_resource);
            pre.force_walk_resources.insert(event_resource);
            if let Some(timer_id) = new_settle_timer {
                pre.phase = PreFirePhase::Batching {
                    settle_timer: timer_id,
                };
            }
            // else: phase already Batching, settle_timer unchanged. The
            // existing timer fires at its scheduled deadline; the expiry
            // handler reads `last_event_time` and reschedules if events
            // have arrived since.
        }
    }

    /// Caller: `dispatch_standard_ok` not-stable + not-forced — a verify
    /// just responded with an unstable verdict. The verify slot was
    /// already disarmed at the top of `on_probe_response`; no Cancel needed.
    /// Arms a fresh settle timer and writes
    /// `phase = Batching { settle_timer }`.
    ///
    /// **`dirty_resources` preserved; `force_walk` empty.** The next
    /// verify re-targets the same LCA via the preserved
    /// `dirty_resources`. Empty `force_walk_resources` is correct: the
    /// prior verify already had the dirty paths' fresh observations,
    /// so the walker can mtime-skip on the second pass. If the disk
    /// has settled, the second verify reuses the prior `current`
    /// (subtree mtime unchanged) and the resulting hash matches the
    /// just-stored response hash — stable verdict, fire.
    ///
    /// **Reachability.** This helper runs *only* when no `FsEvent`
    /// intercepted the verify. An `FsEvent` during `Verifying` routes
    /// through `event_drives_batching`, which Cancels and disarms the
    /// in-flight verify slot; the eventual late response then fails the
    /// `pending_probe_for == Some(received)` staleness gate (the slot
    /// is empty) and drops as `StaleProbeResponse`. The forced +
    /// not-stable case in `dispatch_standard_ok` also bypasses this
    /// helper — forced +
    /// unstable still fires.
    ///
    /// **`last_event_time` pinned to `Some(now)`.** The verify just
    /// responded, so `now` is the timestamp of the latest observation
    /// that drove a transition on this burst (whether a fresh FsEvent or
    /// the verify response itself). Pinning here removes the `Instant`
    /// monotonicity dependency from the on-expiry reschedule check: with
    /// the prior preserve-semantics, the correctness argument was "the
    /// freshly-scheduled settle timer fires at `now + settle`, and the
    /// expiry handler sees `now − last_event_time ≥ settle` because
    /// `now ≥ unstable_response_at + settle ≥ prior_last_event + settle`,
    /// which depends on `Instant` monotonicity". The pinned variant has
    /// the same arithmetic with `last_event_time = now`: the on-expiry
    /// handler sees `expiry_now − now ≥ settle` (true by construction of
    /// the scheduled deadline) and transitions cleanly — independent of
    /// any clock skew between this call and the prior preserve.
    pub(crate) fn unstable_response_drives_batching(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_active_pre_fire(
            profile_id,
            BurstHelper::UnstableResponseDrivesBatching,
            out,
        ) {
            return;
        }
        let Some(settle) = self.profiles.get(profile_id).map(|p| p.settle) else {
            return;
        };
        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);

        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            pre.phase = PreFirePhase::Batching { settle_timer };
            pre.last_event_time = Some(now);
        }
    }

    /// `Batching → Batching` settle-timer re-point. The single-source
    /// `PreFireBurst.phase` mutator for the settle-expiry reschedule:
    /// when an `FsEvent` arrives after the live settle timer was
    /// scheduled but before it fires, `on_settle_expired` schedules a
    /// fresh `Settle` timer at `last_event_time + settle` and routes
    /// the resulting `TimerId` here. The phase *class* is unchanged —
    /// only the timer correlation moves.
    ///
    /// **Not `unstable_response_drives_batching` minus the pin.** That
    /// helper re-enters Batching from `Verifying` and pins
    /// `last_event_time = now` (the verify just responded). This path
    /// is *already* `Batching` and must **not** touch
    /// `last_event_time`: pinning it would push the very deadline the
    /// caller's just-made quiet-window decision is rescheduling toward,
    /// defeating the check that chose to reschedule.
    ///
    /// **Timer math stays with the caller.** `on_settle_expired` owns
    /// the `now − last_event_time < settle` quiet-window decision and
    /// the `TimerHeap::schedule` call; this helper owns only the phase
    /// write, keeping it a pure single-field mutator symmetric with
    /// [`Self::force_pending`].
    ///
    /// **Gate-free by design.** Like [`Self::finish_burst_to_idle`]
    /// this helper carries no `require_active_pre_fire` precondition:
    /// its sole caller has already validated
    /// `Active(PreFire(Batching))` via `is_timer_referenced` plus its
    /// own defensive phase check, so the non-pre-fire arm is
    /// structurally unreachable and a routing diagnostic there would be
    /// spurious. The `debug_assert!` is the loud dev/CI backstop that
    /// the contract held.
    pub(crate) fn reschedule_batching(&mut self, profile_id: ProfileId, settle_timer: TimerId) {
        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            debug_assert!(
                matches!(pre.phase, PreFirePhase::Batching { .. }),
                "reschedule_batching off a non-Batching phase (profile = {profile_id:?})",
            );
            pre.phase = PreFirePhase::Batching { settle_timer };
        }
    }

    /// Set `PreFireBurst.forced` on `BurstDeadline` expiry. The
    /// single-source mutator for the force-fire flag: once the
    /// max-settle deadline elapses, the next probe emission must bypass
    /// the walker's coarse-mtime skip and deliver freshest data —
    /// `forced` lives on the `PreFireBurst` and `emit_owner_probe`
    /// reads it back off the armed `Verifying` slot at emit time.
    ///
    /// **Field write only — the phase decision stays with the caller.**
    /// `handle_burst_deadline` re-reads the phase after this call to
    /// decide whether to drive a verify *now* (`Batching | Draining` —
    /// no probe in flight, so emit) or wait (`Verifying` — a probe is
    /// already in flight and will dispatch with `forced` observed).
    /// That classification is a routing query, not a `PreFireBurst`
    /// mutation, so it is not this helper's concern; keeping the helper
    /// a pure single-field writer mirrors [`Self::reschedule_batching`].
    ///
    /// **Gate-free by design.** Same rationale as
    /// [`Self::reschedule_batching`]: the caller is reached only
    /// through `is_timer_referenced`, which returns false for
    /// `BurstDeadline` in `Awaiting` / `Rebasing`, so only pre-fire
    /// phases arrive. The non-pre-fire arm silently no-ops (preserving
    /// the prior inline `else { return; }`) rather than emitting a
    /// spurious routing diagnostic.
    pub(crate) fn force_pending(&mut self, profile_id: ProfileId) {
        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            pre.forced = true;
        }
    }

    /// Phase: `Batching` (or `Draining`) → `Verifying`. Mints a fresh
    /// correlation; emits `ProbeOp::Probe`.
    ///
    /// **Settle-timer lifecycle on entry.** The `Batching → Verifying`
    /// arm runs only after `on_settle_expired` has decided to transition
    /// (rather than reschedule), and the expired timer entry has already
    /// been removed from the heap by `pop_expired` upstream — the
    /// phase-variant overwrite below drops the engine's reference, not
    /// a heap entry. The `BurstDeadline` arm (force-fire path) leaves
    /// the live settle_timer in the heap when overwriting `burst.phase`;
    /// that stale entry lazy-drops on its original deadline. The
    /// `Draining → Verifying` reconfirm arm has no settle_timer to
    /// orphan (Draining never armed one).
    ///
    /// **Target.** Resolved via [`pre_fire_target`] — File anchors
    /// target the anchor unconditionally; Seed bursts target the
    /// anchor (regardless of phase); Standard bursts target the LCA of
    /// `dirty_resources`. The same rule covers the Draining → Verifying
    /// reconfirm: `dirty_resources` is preserved across the burst's
    /// pre-fire lifetime (only `insert` mutations in production), so
    /// `LCA(dirty)` on the reconfirm matches the LCA computed at the
    /// original Verifying entry up to slot reaping.
    ///
    /// **Emission.** This helper writes the `Verifying` phase + armed
    /// slot + `probe_target`, then calls [`Engine::emit_owner_probe`] —
    /// the single choke that reads the correlation back off the slot,
    /// drains `force_walk_resources` (the walker's force-walk hint, so
    /// events the engine knows about defeat mtime-skip), and reads
    /// `forced` (so the walker bypasses mtime-skip on a force-fire).
    /// New events arriving during `Verifying` accumulate into the
    /// drained `force_walk_resources` set and ship on the next emission.
    pub(crate) fn transition_to_verifying(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::TransitionToVerifying, out) {
            return;
        }
        // Compute target under one immutable borrow window. `&self.tree`
        // and `&self.profiles.get(_)` are disjoint Engine-field borrows;
        // the call returns a `ResourceId` (`Copy`), so neither borrow
        // outlives this block.
        //
        // `pre_fire_target` may emit `LcaIntegrityViolation` via
        // `lca_pair` if the Standard burst's `dirty_resources` walk
        // breaks ancestry; the helper still returns a usable
        // `ResourceId` (folded back to anchor), so the burst proceeds.
        let target = match self.profiles.get(profile_id) {
            Some(p) => match p.state() {
                ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
                    pre_fire_target(p, pre, &self.tree, profile_id, out)
                }
                _ => return,
            },
            None => return,
        };

        // No per-burst consumables are drained here. The post-fire
        // phases never reach this helper — production callers (Settle
        // expiry, BurstDeadline expiry, ancestor reconfirm from
        // `finish_burst_to_idle`) are gated on pre-fire phases via
        // `is_timer_referenced` and the Draining sweep's
        // `is_draining()` filter respectively. A stray call that
        // construct-arms a fresh `Verifying` slot while an effect wait
        // is still in flight would orphan the prior correlation; the
        // loud arm below (`unreachable!()` on a non-pre-fire state) is
        // the guard, and the I5 `debug_assert` is its dev/CI backstop.
        //
        // `force_walk_resources` and `forced` are consumed by
        // `emit_owner_probe` (the single probe-emission choke) off the
        // armed `Verifying` slot it resolves — the transition no longer
        // drains a copy and threads it through a wide constructor.
        // `dirty_resources` is preserved — it carries the LCA basis
        // across the whole burst.

        // Mint, then write the `Verifying` phase already armed with the
        // correlation. The prior phase is `Batching` / `Draining`
        // (gated above), neither of which carries a probe slot, so I5
        // holds by representability; the assert is the loud dev/CI
        // backstop. There is no ordering hazard to manage — the slot
        // *is* the phase, so phase-without-correlation has no window.
        let owner = ProbeOwner::Profile(profile_id);
        debug_assert!(
            self.pending_probe_for(owner).is_none(),
            "I5: transition_to_verifying with a probe already in flight \
             (the construct-armed slot would orphan the prior correlation, \
             profile = {profile_id:?})",
        );
        let correlation = self.mint_probe_correlation();

        // Loud arm. `require_active_pre_fire` proved `Active(PreFire)`,
        // so `pre_fire_burst_mut` resolving `None` means the state
        // machine broke between the gate and the arm. Silent skip ⇒ the
        // emit below reads an un-armed slot and produces no probe and no
        // diagnostic (a wedge); loud ⇒ a crash. Co-required with the
        // choke's read-back: read-back stops an orphaned wire, the loud
        // arm stops the silent wedge — neither subsumes the other.
        let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        else {
            unreachable!(
                "transition_to_verifying: Profile {profile_id:?} not \
                 Active(PreFire) after require_active_pre_fire proved it"
            );
        };
        pre.phase = PreFirePhase::Verifying(ProbeSlot::armed(correlation, ()));
        pre.probe_target = target;

        self.emit_owner_probe(owner, out);
    }

    /// Phase: `Verifying` → `Draining`. Phase swap only — the exit body
    /// (`Draining` → `Verifying` reconfirm) is driven by the
    /// `finish_burst_to_idle` Draining sweep, when this Profile's fresh
    /// `has_active_standard_descendant` query has gone false (no
    /// covered descendant remains in an Active Standard burst).
    ///
    /// `Draining` is a unit variant: the stable snapshot lives on
    /// `Profile.current` (set by `dispatch_standard_ok` immediately
    /// before this call), so no `Arc<TreeSnapshot>` is duplicated on the
    /// phase variant.
    ///
    /// The sole caller (`dispatch_standard_ok`, stable + dirty>0) is
    /// reached only from the Verifying probe response (slot disarmed
    /// before dispatch), so the prior phase is always `Verifying`. The
    /// unit `Draining` has no `ProbeSlot::Drop` tripwire like its
    /// slot-bearing peers; the `debug_assert!` is the symmetric
    /// backstop ([`Self::reschedule_batching`]'s analogue).
    pub(crate) fn transition_to_draining(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::TransitionToDraining, out) {
            return;
        }
        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            debug_assert!(
                matches!(pre.phase, PreFirePhase::Verifying(_)),
                "transition_to_draining off a non-Verifying phase (profile = {profile_id:?})",
            );
            pre.phase = PreFirePhase::Draining;
        }
    }

    /// Phase: `Verifying` → `Awaiting`. The single source of the post-fire
    /// transition: `dispatch_standard_ok`'s stable-fire and forced-fire
    /// branches and `dispatch_seed_ok`'s drift branch call this
    /// immediately after `emit_effects` returns a non-zero
    /// `EmitOutcome.count`. The match is structural (count > 0) —
    /// callers know they pushed Effects.
    ///
    /// `outstanding` is the count of in-flight Effects this Profile owns
    /// (the `EmitOutcome.count` from the just-completed
    /// [`crate::Engine::emit_effects`] call). `EffectComplete` arrivals
    /// decrement it; reaching zero advances to `Rebasing` (or, when
    /// the burst carries [`BurstFinish::Reap`], finishes the burst
    /// directly).
    ///
    /// **Gate timer.** Schedules an `AwaitGateDeadline` at `now +
    /// max_settle * 4` as a recovery hatch for actuator hangs. The
    /// multiplier (v1 default) gives a generous budget — the timer is
    /// not meant to cap normal command runs, only to keep the engine
    /// from wedging if the actuator never reports back. Operator-tunable
    /// knobs are out of scope for v1.
    ///
    /// **`burst_deadline` hand-off.** The pre-fire `BurstDeadline` timer
    /// (scheduled at burst start) stays in the heap but is no longer
    /// structurally relevant — `is_timer_referenced` filters it out of
    /// the post-fire phases, so a late expiry is dropped silently by
    /// `pop_expired`. We do not cancel it eagerly: lazy invalidation is
    /// the cheaper path and `BurstDeadline` carries no payload that
    /// would leak.
    ///
    /// **Defensive precheck before scheduling.** The gate timer is
    /// scheduled only after we verify the Profile is in
    /// `Active(PreFire(_))`. Without the precheck, a defensive miss
    /// (e.g., a future caller bypassing the post-fire phase check that
    /// production gates already enforce) would leave the gate timer
    /// orphaned in the heap; lazy-invalidated by `is_timer_referenced`
    /// since no PostFire exists yet, but still allocated. The precheck
    /// is one `matches!` lookup against a freshly-borrowed Profile —
    /// trivially cheap.
    pub(crate) fn transition_to_awaiting(
        &mut self,
        profile_id: ProfileId,
        outstanding: u32,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::TransitionToAwaiting, out) {
            return;
        }
        // Re-borrow for `max_settle` capture under the same shape-checked
        // window. Anything else here would be a routing breach the
        // precondition would have caught — the inner `matches!` is the
        // borrow-checker discipline for the typed move below, not a
        // duplicated guard.
        let Some(max_settle) = self.profiles.get(profile_id).and_then(|p| {
            matches!(p.state(), ProfileState::Active(ActiveBurst::PreFire(_), _))
                .then_some(p.max_settle())
        }) else {
            return;
        };

        // v1 default: 4× max_settle. Saturating multiplication keeps the
        // arithmetic total — `Duration::saturating_mul` clamps at
        // `Duration::MAX`, leaving the deadline well beyond any
        // reasonable wall-clock horizon.
        let gate_deadline = self.timers.schedule(
            now + max_settle.saturating_mul(4),
            profile_id,
            TimerKind::AwaitGateDeadline,
        );

        // Typed move PreFire → PostFire via `transition_state` (the
        // whole-value swap, returning the prior state). Structurally
        // necessary: `into_post_fire` consumes the pre-fire by value,
        // so we cannot project through `pre_fire_burst_mut`. Bracketing
        // with the matches! shape-check above eliminates the transient
        // Idle window's observability for production callers (a stray
        // observer in dev/CI that races inside the helper would never
        // reach this point on a non-PreFire Profile).
        if let Some(p) = self.profiles.get_mut(profile_id)
            && matches!(p.state(), ProfileState::Active(ActiveBurst::PreFire(_), _))
        {
            let prior = p.transition_state(ProfileState::Idle);
            // Destructure with restore-on-mismatch. The matches! above
            // guarantees the PreFire arm; the fallback exists so a
            // future refactor widening the matches! pattern doesn't
            // silently strand the Profile in `Idle` while dropping the
            // owned burst.
            match prior {
                ProfileState::Active(ActiveBurst::PreFire(pre), finish) => {
                    // Carry `finish` across the fire boundary. PreFire and
                    // PostFire share the post-burst directive — a Reap set
                    // mid-batching survives the fire and is honoured by
                    // `finish_burst_to_idle` at PostFire end.
                    p.transition_state(ProfileState::Active(
                        ActiveBurst::PostFire(pre.into_post_fire(outstanding, gate_deadline)),
                        finish,
                    ));
                }
                other => {
                    p.transition_state(other);
                }
            }
        }
    }

    /// Phase: `Awaiting` → `Rebasing`. The single source of the
    /// post-effect rebase: `on_effect_complete` calls this when
    /// `outstanding` reaches zero (and the burst carries
    /// [`BurstFinish::ReturnToIdle`]), and `handle_gate_deadline`
    /// calls it on the actuator-hang recovery path (also gated on
    /// `ReturnToIdle` — zombie bursts route straight to
    /// `finish_burst_to_idle`).
    ///
    /// **Probe slot.** The fresh correlation is minted and the
    /// `Rebasing` phase is written already armed with it, in one move —
    /// the slot *is* the phase. I5 holds by representability: the prior
    /// phase is `Awaiting` (no probe slot) and the verify slot was
    /// disarmed at its response before `emit_effects` ran. The post-fire
    /// probe targets the anchor (we want the freshest disk state of the
    /// whole watched subtree, not the LCA of the now-stale
    /// `dirty_resources`).
    ///
    /// **`baseline_subtree` for mtime-skip.** The Rebasing probe ships
    /// `Profile.current` as `baseline_subtree`. For an idempotent
    /// command (no writes), the directory's mtime is unchanged and the
    /// walker mtime-skips, returning the prior snapshot — graft is a
    /// no-op and `baseline := current` rebases the (unchanged) view.
    /// For a non-idempotent command, mtime differs and the walker
    /// re-walks, capturing the post-command tree as the new baseline.
    ///
    /// **`force_walk` from absorbed-fire-tail events.** FsEvents that
    /// arrived at descendants during `Awaiting` accumulated into
    /// `Burst.force_walk_resources` via `drive_burst`'s absorb arm.
    /// This helper renders them to walker-facing paths and ships them
    /// as `force_walk`, so the rebase walker re-enumerates the parents
    /// of paths the command touched even when the parent dir's mtime
    /// didn't bump (POSIX content-edit semantics). For idempotent
    /// commands the absorbed set is empty and the walker mtime-skips
    /// at every level — the cheap path is preserved. `mem::take`
    /// consumes the field in one shot, leaving it empty for the
    /// next absorb cycle.
    ///
    /// **Non-Active early return.** Both production callers
    /// (`on_effect_complete`'s last-completion `ReturnToIdle` route and
    /// `handle_gate_deadline`) have already verified `Active(_)` with
    /// phase `Awaiting` before reaching this helper. Defensively
    /// early-returning on non-Active matches `transition_to_verifying`'s
    /// strict policy and avoids the latent state-machine bug where a
    /// stray call would mint a fresh probe correlation and emit a Probe
    /// op while failing to write the phase (the phase-write arm requires
    /// `Active`) — orphaning the correlation, whose late response would
    /// then stale-detect against an unarmed state.
    pub(crate) fn transition_to_rebasing(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_post_fire(profile_id, BurstHelper::TransitionToRebasing, out) {
            return;
        }
        // Mint, then write the `Rebasing` phase already armed with the
        // correlation. The prior phase is `Awaiting` (no probe slot),
        // so I5 holds by representability; the assert is the loud
        // dev/CI backstop. The absorbed-fire-tail `force_walk_resources`
        // accumulator and the anchor target are no longer captured here
        // — `emit_owner_probe` drains `PostFireBurst.force_walk_resources`
        // and resolves the rebase target (the anchor) off state itself.
        let owner = ProbeOwner::Profile(profile_id);
        debug_assert!(
            self.pending_probe_for(owner).is_none(),
            "I5: transition_to_rebasing with a probe already in flight \
             (the construct-armed slot would orphan the prior correlation, \
             profile = {profile_id:?})",
        );
        let correlation = self.mint_probe_correlation();

        // Loud arm. `require_active_post_fire` proved `Active(PostFire)`,
        // so `post_fire_burst_mut` resolving `None` means the state
        // machine broke between the gate and the arm — loud crash, not a
        // silent no-probe wedge (co-required with the choke's read-back).
        let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        else {
            unreachable!(
                "transition_to_rebasing: Profile {profile_id:?} not \
                 Active(PostFire) after require_active_post_fire proved it"
            );
        };
        post.phase = PostFirePhase::Rebasing(ProbeSlot::armed(correlation, ()));

        // The choke reads the correlation back off the `Rebasing` slot,
        // targets the anchor (`forced` is pre-fire-only ⇒ `false`), and
        // drains `PostFireBurst.force_walk_resources` (the absorbed
        // fire-tail events) itself.
        self.emit_owner_probe(owner, out);
    }

    /// Active → Idle. The active burst's timers are not explicitly
    /// cancelled — lazy invalidation in `pop_expired` drops them when
    /// they fire. Idempotent: silent no-op on already-Idle Profiles.
    ///
    /// **Cancel-first entry precondition.** No caller may reach here
    /// with `Profile(profile_id)`'s `ProbeSlot` still armed — the
    /// swap-to-Idle destructures the prior burst, and an armed
    /// Verifying/Rebasing slot reaching that drop trips `ProbeSlot`'s
    /// linearity tripwire. Debug-asserted at entry; the proof that all
    /// callers satisfy it lives at the assert.
    ///
    /// **Draining-exit driver.** After the focal Profile is Idle, sweep
    /// *every* currently-`Draining` Profile and re-evaluate the pure
    /// query [`crate::coverage::has_active_standard_descendant`]; drive
    /// each whose query is now false through `transition_to_verifying`
    /// in the same step. The reconfirm condition can only flip false at
    /// *some* descendant's burst finish, so re-checking all Draining
    /// Profiles at every finish is sufficient — and, unlike walking the
    /// finishing Profile's covering chain, it cannot strand a Draining
    /// ancestor that a mid-burst topology move took off that chain. The
    /// exit is then bounded-latency (it waits for the gating
    /// descendant's own guaranteed, deadline-bounded finish), never a
    /// permanent strand. The reconfirm probe compares against the
    /// Profile's `current` (set when `dispatch_standard_ok` entered
    /// Draining). Same-step ordering means the `StepOutput` reflects
    /// the cascade: child's burst end → parent reconfirm Probe in one
    /// `step` call.
    ///
    /// **Burst-finish directive.** If the prior state's
    /// [`BurstFinish`] is [`BurstFinish::Reap`] (the last Sub was
    /// detached mid-burst, or the anchor's all-dynamic teardown
    /// converged on a still-Active Profile), `Engine::reap_profile`
    /// runs in the same step after the Draining sweep — `via =
    /// DeferredFromBurst` distinguishes this path from the immediate
    /// reap in `detach_sub_inner`. Otherwise the Profile rests at
    /// [`ProfileState::Idle`].
    pub(crate) fn finish_burst_to_idle(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        // Cancel-first entry precondition (debug). The swap-to-Idle
        // below destructures the prior burst; an armed
        // Verifying/Rebasing slot reaching that drop trips ProbeSlot's
        // linearity tripwire. All 15 production callers consume the
        // slot first:
        //  - 10 response-path dispatchers (dispatch_{seed,standard,
        //    rebase}_{ok,vanished,failed}) run only after
        //    on_profile_probe_response disarmed via take_owner_probe;
        //    nothing between that disarm and the call re-arms.
        //  - 2 Awaiting-phase callers (on_effect_complete reap,
        //    handle_gate_deadline zombie) are guarded to
        //    Active(PostFire(Awaiting)) — Awaiting holds no slot.
        //  - 2 pure-teardown callers (finalize_anchor_lost,
        //    on_anchor_terminal_all_dynamic) cancel_owner_probe first.
        //  - 1 overflow caller (on_sensor_overflow Active arm) disarms
        //    first: take_owner_probe on reseed, cancel_owner_probe on
        //    reap.
        // Named at this boundary, not left solely to the far-end
        // ProbeSlot::drop tripwire — that fires frames downstream and
        // is structurally bypassed by every test that pre-consumes the
        // slot. The tripwire stays the release fail-stop; this makes
        // the omission fail in unit tests on the dev (kqueue) platform
        // too. Deliberately the strong form: pending_probe_for also
        // covers a Pending Profile's descent slot, so this additionally
        // forbids reaching here mid-descent armed — vacuous in v1 (no
        // caller does) but free extra defense at the same boundary;
        // narrowing to Active-only would add a phase branch for an
        // unreachable case. Borrow-clean: the &self projection ends
        // before the &mut get_mut, and a stale id yields None so the
        // assert holds trivially ahead of the get_mut early-return.
        debug_assert!(
            self.pending_probe_for(ProbeOwner::Profile(profile_id))
                .is_none(),
            "finish_burst_to_idle: probe slot still armed — the caller \
             must consume it first (cancel_owner_probe on a teardown or \
             overflow-reap path; take_owner_probe on the response or \
             overflow-reseed path); profile = {profile_id:?}",
        );
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };

        // Take the burst-by-value via `transition_state(Idle)` and
        // discriminate on the typed variant. `intent` is not read here:
        // the Draining sweep below is intent-agnostic.
        //
        // After this point `p.state == Idle` for the whole helper
        // window. The subsequent Draining-sweep `transition_to_verifying`
        // / reap calls all run against a focal Profile in Idle — future
        // observers (e.g., a hook firing on state transitions) would see
        // the transition bracket cleanly. Idle-first is also
        // load-bearing for the sweep: the finishing Profile is excluded
        // from its own `has_active_standard_descendant` query precisely
        // because it is no longer in an Active Standard burst.
        let prior = p.transition_state(ProfileState::Idle);
        // Capture `finish` from the consumed prior state. It is captured
        // here — not re-read from `profiles.get(profile_id)` after the
        // swap — so the directive is locked in at burst-end entry; a
        // hypothetical future mid-helper write to a re-borrowed Profile
        // can't flip the reap decision under us. The PostFire burst
        // (whose `ProbeSlot` the cancel-first precondition guarantees
        // disarmed) is dropped at the arm's end, exactly as before. Both
        // Active arms carry the directive identically; the discriminant
        // matters only to drop the right burst payload.
        let finish = match prior {
            ProfileState::Active(ActiveBurst::PreFire(_) | ActiveBurst::PostFire(_), finish) => {
                finish
            }
            other => {
                // Idle / Pending — no burst-end machinery to run. Restore.
                p.transition_state(other);
                return;
            }
        };

        // Intent-agnostic Draining sweep. The reconfirm condition
        // `has_active_standard_descendant(A)` can only flip false at
        // *some* descendant's burst finish; rather than walk the
        // finishing Profile's (possibly topology-moved) covering chain,
        // re-evaluate the pure query for *every* currently-Draining
        // Profile. The focal Profile is already Idle (above), so it is
        // correctly excluded from its own predicate. Pass 1 is pure
        // reads (`&Tree` + `&ProfileMap`, all shared — borrow-clean
        // against the inner `&self.profiles` re-borrow); Draining is a
        // rare, tiny phase (typically 0–1 Profiles). Pass 2 takes
        // `&mut self` for the unchanged downstream reconfirm:
        // `transition_to_verifying` mints a fresh correlation and emits
        // Probe; the response routes through `dispatch_standard_ok` as
        // a normal Standard burst, comparing against the Profile's
        // `current` (set when it entered Draining).
        let reconfirm: SmallVec<[ProfileId; 4]> = self
            .profiles
            .iter()
            .filter_map(|(id, a)| {
                (a.state().is_draining()
                    && !crate::coverage::has_active_standard_descendant(
                        &self.tree,
                        &self.profiles,
                        id,
                    ))
                .then_some(id)
            })
            .collect();
        for ancestor in reconfirm {
            self.transition_to_verifying(ancestor, out);
        }

        // Honour the burst-finish directive captured from the prior
        // state. `Reap` is set by `detach_sub_inner` (last Sub detached
        // mid-burst) or `on_anchor_terminal_all_dynamic` (all-dynamic
        // Promoter teardown); we defer the reap to here so the Profile's
        // burst doesn't fire Effects against a Sub registry that no
        // longer holds the reference. `ReturnToIdle` leaves the Profile
        // resting at Idle (the `mem::replace` above already wrote Idle).
        if matches!(finish, BurstFinish::Reap) {
            self.reap_profile(profile_id, ReapTrigger::DeferredFromBurst, out);
        }
    }

    /// Absorb a post-fire FsEvent into the rebase probe's force-walk
    /// hint. The post-fire phases (`Awaiting | Rebasing`) cannot start a
    /// fresh burst — the rebase probe is already in flight or imminent —
    /// so the engine defers the event to the next probe's
    /// `force_walk_paths`. Closes the POSIX content-edit hole: a
    /// content-only edit at a descendant doesn't bump the anchor's
    /// mtime, so without this hint the rebase walker mtime-skips at
    /// every level and the post-fire baseline retains the stale leaf.
    ///
    /// `event` is threaded purely for the diagnostic so an operator can
    /// correlate logs to the deferred FsEvent.
    pub(crate) fn absorb_event_into_fire_tail(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        event: FsEvent,
        out: &mut StepOutput,
    ) {
        if !self.require_active_post_fire(profile_id, BurstHelper::AbsorbEventIntoFireTail, out) {
            return;
        }
        if let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        {
            post.force_walk_resources.insert(event_resource);
            out.diagnostics.push(Diagnostic::EventAbsorbedByFireTail {
                profile: profile_id,
                resource: event_resource,
                event,
            });
        }
    }

    /// Restart a fresh Standard `Batching` burst from the fire-tail
    /// residual — the consumer for events `absorb_event_into_fire_tail`
    /// accumulated after the rebase probe was already in flight. Single
    /// source of the `Active(PostFire)` → `Active(PreFire(Batching))`
    /// typed move (via [`PostFireBurst::into_pre_fire_residual`]); the
    /// inverse of `transition_to_awaiting`'s fire move.
    ///
    /// **Caller.** `dispatch_rebase_ok` only, after `rebase_baseline`,
    /// and only once it has established the residual is non-empty and
    /// the burst is [`BurstFinish::ReturnToIdle`]. Origin-agnostic — a
    /// Seed-origin residual restarts too; `into_pre_fire_residual` sets
    /// `intent: Standard` because a restarted debounce burst *is*
    /// Standard by definition.
    ///
    /// **No refcount edges.** The typed `PostFire → PreFire` move
    /// preserves the watched anchor: it neither installs nor releases a
    /// contribution, so the restarted burst keeps the original burst's
    /// kernel-watch state without a finish/start round-trip. (There is
    /// no ancestor counter to balance either — the reconfirm is a fresh
    /// query.)
    ///
    /// **Slot-consumed precondition.** The whole-value swap below
    /// destructures the post-fire burst; an armed `Rebasing` slot
    /// reaching that drop trips `ProbeSlot`'s linearity tripwire. The
    /// sole caller runs only after `on_profile_probe_response` disarmed
    /// via `take_owner_probe` — the same precondition
    /// `finish_burst_to_idle` carries on the path this replaces.
    /// Debug-asserted at entry.
    ///
    /// The restart re-enters `Batching` (not an immediate re-probe), so
    /// it is settle-debounced and burst-deadline-bounded exactly like a
    /// fresh `start_standard_burst` — it self-heals at the external
    /// change rate and cannot livelock.
    pub(crate) fn restart_burst_from_fire_tail_residual(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_active_post_fire(
            profile_id,
            BurstHelper::RestartBurstFromFireTailResidual,
            out,
        ) {
            return;
        }
        // Slot-consumed precondition (debug). Borrow-clean: the &self
        // projection ends before the &mut get below, and a stale id
        // yields None so the assert holds trivially ahead of the
        // early-return.
        debug_assert!(
            self.pending_probe_for(ProbeOwner::Profile(profile_id))
                .is_none(),
            "restart_burst_from_fire_tail_residual: probe slot still armed \
             — the caller must consume it first (take_owner_probe on the \
             rebase response path); profile = {profile_id:?}",
        );

        // Re-borrow for captures under the same shape-checked window
        // `transition_to_awaiting` uses for its inverse move: the
        // precondition already proved `Active(PostFire)`; the inner
        // `matches!` is the borrow discipline for the typed move below,
        // not a duplicated guard.
        let Some((resource, settle, max_settle)) = self.profiles.get(profile_id).and_then(|p| {
            matches!(p.state(), ProfileState::Active(ActiveBurst::PostFire(_), _)).then_some((
                p.resource,
                p.settle,
                p.max_settle(),
            ))
        }) else {
            return;
        };

        // The two engine timers a fresh Standard burst arms
        // (`start_standard_burst`): the settle debounce and the
        // burst-deadline force-fire ceiling.
        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);
        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        // Typed move PostFire → PreFire via `transition_state` (the
        // whole-value swap returning the prior state):
        // `into_pre_fire_residual` consumes the post-fire by value, so it
        // cannot project through `post_fire_burst_mut`. Bracketing with
        // the `matches!` above eliminates the transient-Idle window's
        // observability for production callers; the restore-on-mismatch
        // arm keeps a future pattern-widening refactor from stranding
        // the Profile in `Idle` while dropping the owned burst.
        if let Some(p) = self.profiles.get_mut(profile_id)
            && matches!(p.state(), ProfileState::Active(ActiveBurst::PostFire(_), _))
        {
            let prior = p.transition_state(ProfileState::Idle);
            match prior {
                ProfileState::Active(ActiveBurst::PostFire(post), finish) => {
                    // Carry `finish` across the restart. It is
                    // `ReturnToIdle` by the caller's gate; preserving it
                    // (rather than hard-writing) keeps a mid-tail
                    // `mark_active_for_reap` honoured at the restarted
                    // burst's end.
                    p.transition_state(ProfileState::Active(
                        ActiveBurst::PreFire(post.into_pre_fire_residual(
                            burst_deadline,
                            settle_timer,
                            resource,
                            now,
                        )),
                        finish,
                    ));
                }
                other => {
                    p.transition_state(other);
                }
            }
        }
    }
}

// `TreeSnapshot` reachable for downstream consumers via the burst module
// surface — the lifecycle helpers thread `current.subtree_at` references
// through that type.
const _: fn() = || {
    let _ = std::mem::size_of::<TreeSnapshot>();
};

/// "Lowest covering ancestor of all event-dirty Resources" for a
/// Dir-anchored Profile. The single probe target per Standard burst.
///
/// **Caller contract.** This helper is intended for Dir-anchored Profiles
/// only. File-anchored Profiles probe the anchor itself unconditionally
/// (kqueue per-file FDs surface events at the file directly):
/// [`pre_fire_target`] returns the anchor for a File kind and
/// [`Engine::emit_owner_probe`]'s kind dispatch routes it to a
/// `ProbeRequest::AnchorFile` without consulting this helper, so a
/// File-anchored Profile never reaches `lca_target` in production.
/// The `live.contains(&anchor)` short-circuit below remains
/// valid for the Dir-anchor case where the anchor itself is the event
/// source (e.g., an in-place mtime bump on the anchor directory).
///
/// **Invariants.**
/// - Returns a live `ResourceId` (always — defaults to `anchor`).
/// - Result is `ResourceKind::Dir`: descendant LCAs that resolve to a
///   Leaf (or unprobed slot) are promoted to their parent Dir — probes
///   target Dirs because Files are observed as child entries of their
///   parent in the descendant-observation model.
/// - Result is at-or-above every live entry in `dirty`. Reaped entries
///   are filtered first — a stale `ResourceId` whose slot was vacated
///   mid-burst yields `None` on `tree.parent` and would skew the
///   reduction otherwise.
/// - When `dirty` is empty, returns `anchor`: falls back to a full-walk
///   gracefully.
///
/// **Complexity.** O(depth × n_dirty) — pairwise reduction with
/// depth-equalisation + lockstep ancestor walk per pair. No per-pair
/// `BTreeSet` allocation.
pub(crate) fn lca_target(
    anchor: ResourceId,
    dirty: &BTreeSet<ResourceId>,
    tree: &Tree,
    profile: ProfileId,
    out: &mut StepOutput,
) -> ResourceId {
    // 1. Filter stale ResourceIds. A `dirty_resources` entry whose slot
    // was reaped between FsEvent ingestion and probe emission
    // (delete-recreate-different-inode race) is dropped here. This is
    // benign — the slot's prior events are no longer routable. No
    // diagnostic: per-event noise would flood logs during normal
    // delete-recreate churn.
    let live: SmallVec<[ResourceId; 4]> = dirty
        .iter()
        .copied()
        .filter(|&r| tree.get(r).is_some())
        .collect();

    if live.is_empty() {
        return anchor;
    }
    // Anchor in the dirty set ⇒ can't go higher than anchor; trivially LCA.
    if live.contains(&anchor) {
        return promote_to_dir(anchor, anchor, tree);
    }

    // 2. Pairwise LCA reduction. For each new entry, walk both candidates
    // up to a common depth, then up in lockstep until they match. A
    // `None` from `lca_pair` indicates the integrity violation has
    // already been reported via `Diagnostic::LcaIntegrityViolation`;
    // we just fold to anchor and move on.
    let mut acc = live[0];
    for &r in &live[1..] {
        match lca_pair(acc, r, tree, profile, out) {
            Some(joint) => acc = joint,
            None => return anchor,
        }
    }
    promote_to_dir(acc, anchor, tree)
}

/// LCA of two resources via depth-equalisation + lockstep ancestor walk.
/// O(max(depth_a, depth_b)). Returns `None` only when an input slot is
/// stale (`LcaIntegritySource::StaleId`) or a parent walk runs out of
/// ancestors before the candidates align (`LcaIntegritySource::BrokenAncestry`).
/// In either case the helper emits
/// [`Diagnostic::LcaIntegrityViolation`] tagged with the source before
/// returning; the caller folds to anchor so the burst still has a
/// probe target.
///
/// Source-tagging rationale: stale-id ingress at this helper is a
/// fresh class of bug — `lca_target`'s upstream `live` filter is the
/// canonical drop point for reaped slots, and a stale id reaching
/// `lca_pair` means the filter was bypassed (e.g., a future caller
/// constructing the pair from a non-filtered source). Broken ancestry
/// is the parent walk running out before alignment, which indicates
/// the Tree's parent chain is structurally inconsistent.
pub(crate) fn lca_pair(
    a: ResourceId,
    b: ResourceId,
    tree: &Tree,
    profile: ProfileId,
    out: &mut StepOutput,
) -> Option<ResourceId> {
    if a == b {
        return Some(a);
    }
    // Defense-in-depth: upstream `lca_target` filters stale ids, but a
    // future caller bypassing that filter would otherwise manifest as
    // `BrokenAncestry` on the first parent walk. Surfacing it as
    // `StaleId` keeps the operational signal accurate.
    if tree.get(a).is_none() || tree.get(b).is_none() {
        out.diagnostics.push(Diagnostic::LcaIntegrityViolation {
            profile,
            source: LcaIntegritySource::StaleId,
        });
        return None;
    }
    let depth_a = tree.ancestors(a).count();
    let depth_b = tree.ancestors(b).count();
    let mut a = a;
    let mut b = b;
    // Walk the deeper one up to the same depth as the shallower. A
    // `None` here means the parent chain dangled; emit BrokenAncestry
    // and bail.
    for _ in 0..depth_a.saturating_sub(depth_b) {
        a = match tree.parent(a) {
            Some(p) => p,
            None => {
                out.diagnostics.push(Diagnostic::LcaIntegrityViolation {
                    profile,
                    source: LcaIntegritySource::BrokenAncestry,
                });
                return None;
            }
        };
    }
    for _ in 0..depth_b.saturating_sub(depth_a) {
        b = match tree.parent(b) {
            Some(p) => p,
            None => {
                out.diagnostics.push(Diagnostic::LcaIntegrityViolation {
                    profile,
                    source: LcaIntegritySource::BrokenAncestry,
                });
                return None;
            }
        };
    }
    // Walk both up in lockstep until they match.
    while a != b {
        a = match tree.parent(a) {
            Some(p) => p,
            None => {
                out.diagnostics.push(Diagnostic::LcaIntegrityViolation {
                    profile,
                    source: LcaIntegritySource::BrokenAncestry,
                });
                return None;
            }
        };
        b = match tree.parent(b) {
            Some(p) => p,
            None => {
                out.diagnostics.push(Diagnostic::LcaIntegrityViolation {
                    profile,
                    source: LcaIntegritySource::BrokenAncestry,
                });
                return None;
            }
        };
    }
    Some(a)
}

/// Promote a non-Dir candidate to its parent Dir; descendant-observation
/// probes target Dirs. Falls back to `anchor` if the chain crosses a
/// reaped slot or runs out of ancestors. Unprobed slots
/// (`kind() == None`) walk up like File-shape — we don't know what they
/// are, the parent is the safer probe target.
///
/// **Pre-condition.** The caller has filtered out File-anchored Profiles;
/// this helper assumes a Dir anchor and may walk past a non-Dir start to
/// reach the Profile's anchor when `start == anchor` is itself a File
/// (which wouldn't happen for a Dir-anchored Profile).
fn promote_to_dir(start: ResourceId, anchor: ResourceId, tree: &Tree) -> ResourceId {
    let mut current = start;
    loop {
        let Some(r) = tree.get(current) else {
            return anchor;
        };
        if matches!(r.kind(), Some(ResourceKind::Dir)) {
            return current;
        }
        let Some(p) = tree.parent(current) else {
            return anchor;
        };
        current = p;
    }
}

/// Pre-fire probe target for the next emission.
///
/// Centralizes the `(anchor_kind, intent)` rule that drives every
/// pre-fire probe target choice. The three production scenarios
/// (settle-expired Standard burst, force-fire under
/// [`PreFirePhase::Batching`] or [`PreFirePhase::Draining`], the
/// Draining-sweep reconfirm) all resolve here:
///
/// - File anchor (`Profile.kind == Some(File)`) → the anchor itself.
///   kqueue per-file FDs surface events at the file directly, and the
///   walker's [`crate::ProbeRequest::AnchorFile`] arm lstat's the leaf —
///   promoting past the anchor would route the probe outside the
///   Profile's coverage.
/// - Seed intent (Dir / unclassified anchor) → the anchor. Seed bursts
///   compare against fire history rather than against a stable subtree
///   verdict, so they probe at the anchor unconditionally.
/// - Standard intent (Dir / unclassified anchor) → `lca_target` of the
///   burst's `dirty_resources`. The LCA is the deepest shared ancestor
///   of every Resource where an `FsEvent` actually arrived; probing
///   there contains the walk to where events occurred.
///
/// **Draining-reconfirm coverage.** The Draining → Verifying reconfirm
/// path previously special-cased `prior_target` reuse. That arm now
/// folds into the Standard case because `dirty_resources` is preserved
/// across the burst's whole pre-fire lifetime (only `insert` mutations
/// in production), so `LCA(dirty)` on the reconfirm equals the LCA
/// computed for the initial Verifying entry. Slot reaping during
/// Draining makes the new probe **strictly narrower** — reaped paths
/// no longer need reconfirmation — and the stale-`ResourceId` failure
/// mode of the old prior-target reuse is gone (`lca_target` filters
/// reaped slots and falls back to anchor when the live set is empty).
pub(crate) fn pre_fire_target(
    p: &Profile,
    pre: &PreFireBurst,
    tree: &Tree,
    profile: ProfileId,
    out: &mut StepOutput,
) -> ResourceId {
    match (p.kind(), pre.intent) {
        (Some(ResourceKind::File), _) | (_, BurstIntent::Seed) => p.resource,
        _ => lca_target(p.resource, &pre.dirty_resources, tree, profile, out),
    }
}

/// Build the `force_walk` set the walker consumes. Engine-side closure of
/// `force_walk_resources ∩ subtree(target)` rendered to the walker's
/// path-keyed contract.
///
/// The walker checks `force_walk.iter().any(|p| p.starts_with(current))`
/// at every recursion level; pre-filtering by ancestry of `target` keeps
/// the set minimal — out-of-subtree entries cannot affect the walk and
/// would only inflate the walker's per-dir scan.
pub(crate) fn build_force_walk(
    set: &BTreeSet<ResourceId>,
    target: ResourceId,
    tree: &Tree,
) -> BTreeSet<Arc<Path>> {
    set.iter()
        .copied()
        .filter(|&r| r_is_at_or_under(r, target, tree))
        .filter_map(|r| tree.path_of(r))
        .collect()
}

/// Returns true iff `r` is `target` or a descendant of `target` (i.e., `r`
/// lies in `target`'s subtree). The walk goes `r → parent(r) → ...` until
/// it hits `target` (true) or runs out of ancestors (false).
fn r_is_at_or_under(r: ResourceId, target: ResourceId, tree: &Tree) -> bool {
    let mut cur = Some(r);
    while let Some(c) = cur {
        if c == target {
            return true;
        }
        cur = tree.parent(c);
    }
    false
}

#[cfg(test)]
mod tests {
    // Tests prioritize readability over the workspace's pedantic style budget.
    #![allow(
        clippy::manual_let_else,
        clippy::match_wildcard_for_single_variants,
        clippy::missing_const_for_fn,
        clippy::needless_pass_by_value,
        clippy::too_many_lines
    )]

    use crate::Engine;
    use specter_core::{
        ActiveBurst, BurstIntent, ClassSet, Input, PreFirePhase, ProbeOp, ProbeOwner, ProbeRequest,
        ProbeSlot, Profile, ProfileIdentity, ProfileState, ResourceKind, ResourceRole, ScanConfig,
        StepOutput, TimerKind,
    };
    use std::time::{Duration, Instant};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Build an Engine with a single Profile anchored at `/anchor`. Returns the
    /// Engine + the `ProfileId`.
    fn engine_with_profile() -> (Engine, specter_core::ProfileId) {
        let mut e = Engine::new();
        let r = e.tree.ensure_root("anchor", ResourceRole::User);
        e.tree.set_kind(r, ResourceKind::Dir);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r,
                ProfileIdentity {
                    config: ScanConfig::builder().recursive(true).build(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        (e, pid)
    }

    #[test]
    fn start_seed_burst_emits_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);

        // Profile transitioned to Active(Seed Verifying).
        let p = e.profiles.get(pid).unwrap();
        let burst = match p.state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Seed);
        assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
        assert!(!burst.forced);

        // Output: one Probe.
        let probes = out
            .probe_ops()
            .iter()
            .filter(|op| matches!(op, ProbeOp::Probe { .. }))
            .count();
        assert_eq!(probes, 1);

        // Heap: only burst_deadline (Seed has no settle_timer).
        assert_eq!(e.timers.len(), 1);
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn start_standard_burst_schedules_two_timers_no_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );

        let p = e.profiles.get(pid).unwrap();
        let burst = match p.state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Standard);
        assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));

        // Heap holds settle_timer + burst_deadline.
        assert_eq!(e.timers.len(), 2);

        // No probe yet (settle_timer fires first).
        assert!(out.probe_ops().is_empty());
    }

    #[test]
    fn transition_to_verifying_mints_correlation_and_emits_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );
        let mut out = StepOutput::default();

        e.transition_to_verifying(pid, &mut out);

        match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => {
                assert!(matches!(b.phase, PreFirePhase::Verifying(_)));
            }
            _ => panic!("expected Active(PreFire)"),
        }
        let correlation = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight on the state slot");

        // Output: one Probe whose correlation matches.
        let probe_correlation = out.probe_ops().iter().find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        });
        assert_eq!(probe_correlation, Some(correlation));
        let _ = e.cancel_all_in_flight_probes();
    }

    /// Central invariant: a Verifying burst's probe correlation lives on
    /// the state-resident `ProbeSlot` (there is no probe side table).
    /// Assert `pending_probe_for`'s projection equals the correlation
    /// read directly off the `Verifying` slot.
    #[test]
    fn verify_probe_correlation_is_state_resident() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );
        e.transition_to_verifying(pid, &mut out);

        let owner = ProbeOwner::Profile(pid);
        let projected = e
            .pending_probe_for(owner)
            .expect("a verify probe is in flight");

        // Identity is on the Verifying slot itself.
        let slot_correlation = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => match &b.phase {
                PreFirePhase::Verifying(slot) => slot.correlation(),
                other => panic!("expected Verifying, got {other:?}"),
            },
            other => panic!("expected Active(PreFire), got {other:?}"),
        };
        assert_eq!(
            slot_correlation,
            Some(projected),
            "the Verifying slot carries the in-flight correlation",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn event_during_verifying_emits_cancel_and_resets_batching() {
        // FsEvent during Verifying: Cancel emitted; phase becomes Batching
        // with a fresh settle_timer; intent preserved.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out); // Seed → Verifying
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource;

        e.event_drives_batching(pid, r, Instant::now(), &mut out);

        // One Cancel emitted for the in-flight probe.
        let cancel_count = out
            .probe_ops()
            .iter()
            .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
            .count();
        assert_eq!(cancel_count, 1);

        let p = e.profiles.get(pid).unwrap();
        let burst = match p.state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));
        assert_eq!(
            burst.intent,
            BurstIntent::Seed,
            "intent preserved across Verifying → Batching",
        );
    }

    #[test]
    fn event_during_batching_does_not_emit_cancel() {
        // Already in Batching: a fresh FsEvent reschedules without Cancel.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource;
        e.start_standard_burst(pid, r, Instant::now(), &mut out);
        let mut out = StepOutput::default();

        e.event_drives_batching(pid, r, Instant::now(), &mut out);

        let cancels = out
            .probe_ops()
            .iter()
            .filter(|op| matches!(op, ProbeOp::Cancel { .. }))
            .count();
        assert_eq!(cancels, 0);

        // Still Batching; intent preserved.
        let p = e.profiles.get(pid).unwrap();
        let burst = match p.state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert!(matches!(burst.phase, PreFirePhase::Batching { .. }));
        assert_eq!(burst.intent, BurstIntent::Standard);
    }

    #[test]
    fn unstable_response_does_not_emit_cancel() {
        // Standard burst → Batching → Verifying → simulated unstable
        // response → Batching. The transition emits NO Cancel — the helper
        // structurally refuses to emit one because the verify just
        // responded.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let resource = e.profiles.get(pid).unwrap().resource;
        let now = Instant::now();
        e.start_standard_burst(pid, resource, now, &mut out);
        e.transition_to_verifying(pid, &mut out);
        let mut out = StepOutput::default();
        // Production reaches `unstable_response_drives_batching` only
        // from `on_probe_response`, which has already disarmed the
        // Verifying slot via `take_owner_probe`. Mirror that consume.
        let _ = e.take_owner_probe(ProbeOwner::Profile(pid));

        e.unstable_response_drives_batching(pid, now, &mut out);

        assert!(out.probe_ops().is_empty());
        let phase = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => &pre.phase,
            _ => panic!("expected Active(PreFire)"),
        };
        assert!(matches!(phase, PreFirePhase::Batching { .. }));
    }

    /// C2 backstop: `finish_burst_to_idle` debug-asserts the owner's
    /// `ProbeSlot` is already disarmed at function entry. Reaching it
    /// with a *genuinely armed* `Active(PreFire(Verifying))` (the slot
    /// in flight, NOT pre-consumed — the F-CRIT-1 reproduction shape)
    /// must trip the assert loudly rather than silently dropping the
    /// armed slot. `#[should_panic]` on a `debug_assert!` only triggers
    /// under debug assertions; nextest runs the debug profile by
    /// default, so this is exercised.
    #[test]
    #[should_panic(expected = "finish_burst_to_idle: probe slot still armed")]
    fn finish_burst_to_idle_armed_slot_trips_debug_assert() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );
        e.transition_to_verifying(pid, &mut out);
        // Genuinely armed: NO `take_owner_probe` pre-consume. This is
        // the whole point — the slot reaches finish_burst_to_idle armed.
        assert!(
            e.pending_probe_for(ProbeOwner::Profile(pid)).is_some(),
            "fixture: Verifying slot genuinely armed (NOT pre-consumed)",
        );

        // Trips the C2 debug_assert at function entry. The slot is
        // never disarmed, so no `cancel_all_in_flight_probes` teardown
        // is reachable (and unwinding through it would itself trip the
        // ProbeSlot Drop tripwire) — `#[should_panic]` is the contract.
        e.finish_burst_to_idle(pid, &mut out);
    }

    /// C2 positive: the legitimate caller path — a probe response that
    /// already disarmed the slot via `take_owner_probe` — does NOT trip
    /// the assert. `finish_burst_to_idle` returns normally and the
    /// Profile lands in `Idle`. Pins the assert as a precondition
    /// witness, not a blanket ban on the function.
    #[test]
    fn finish_burst_to_idle_after_disarm_returns_to_idle() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_standard_burst(
            pid,
            e.profiles.get(pid).unwrap().resource,
            Instant::now(),
            &mut out,
        );
        e.transition_to_verifying(pid, &mut out);
        // Mirror the real response path: the Verifying slot is disarmed
        // via `take_owner_probe` before burst-end.
        let _ = e.take_owner_probe(ProbeOwner::Profile(pid));
        assert!(
            e.pending_probe_for(ProbeOwner::Profile(pid)).is_none(),
            "slot consumed, mirroring on_probe_response",
        );

        e.finish_burst_to_idle(pid, &mut out);

        assert!(
            matches!(e.profiles.get(pid).unwrap().state(), ProfileState::Idle,),
            "legitimate disarmed caller path returns the Profile to Idle",
        );
    }

    #[test]
    fn event_storm_during_batching_does_not_amplify_settle() {
        // Settle-reuse contract: a storm of FsEvents during Batching does
        // NOT re-insert a fresh settle timer per event; only
        // `last_event_time` updates. The on-expiry handler reschedules at
        // `last_event_time + settle` if events arrived since, otherwise
        // transitions. The semantic invariant — settle deadline pinned at
        // `last_event + settle` — holds without per-event heap pushes.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource;
        let t0 = Instant::now();
        e.start_standard_burst(pid, r, t0, &mut out);

        // Fire ten FsEvents at 50 ms intervals during the Standard burst.
        let mut last_event = t0;
        for k in 1..=10 {
            last_event = t0 + Duration::from_millis(50 * k);
            let mut out = StepOutput::default();
            e.event_drives_batching(pid, r, last_event, &mut out);
        }

        // ── Invariant 1: last_event_time is the source of truth. The
        // most recent event imprints on the field; intermediate events
        // overwrite without trace.
        let burst = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(
            burst.last_event_time,
            Some(last_event),
            "last_event_time pinned to most recent FsEvent",
        );

        // ── Invariant 2: only one settle timer for this profile in the
        // heap. The initial timer from `start_standard_burst` carries
        // through the storm; per-event reschedules are gone.
        let settle_timers: usize = e
            .timers
            .iter()
            .filter(|entry| entry.profile == pid && entry.kind == TimerKind::Settle)
            .count();
        assert_eq!(
            settle_timers, 1,
            "exactly one settle timer per burst (no per-event reinsert)",
        );

        let initial_settle_timer = match burst.phase {
            PreFirePhase::Batching { settle_timer } => settle_timer,
            _ => panic!("expected Batching"),
        };

        // ── Invariant 3: on the initial timer's expiry while events are
        // recent (last_event > initial deadline in this contrived
        // unit-test timeline; saturating_duration_since clamps to 0
        // and the recency check fires), `on_settle_expired` reschedules
        // a fresh timer at `last_event + settle`. The phase stays
        // Batching with the new id.
        let expiry_now = t0 + SETTLE; // initial timer's deadline
        let _ = e.step(
            Input::TimerExpired {
                profile: pid,
                kind: TimerKind::Settle,
                id: initial_settle_timer,
            },
            expiry_now,
        );

        let phase = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => &pre.phase,
            _ => panic!("expected Active(PreFire) after reschedule"),
        };
        let rescheduled_timer = match phase {
            PreFirePhase::Batching { settle_timer } => *settle_timer,
            PreFirePhase::Verifying(_) | PreFirePhase::Draining => {
                panic!("expected Batching after reschedule, got {phase:?}")
            }
        };
        assert_ne!(
            rescheduled_timer, initial_settle_timer,
            "reschedule mints a fresh TimerId; the initial id is no longer referenced",
        );

        // ── Invariant 4: the rescheduled deadline equals
        // `last_event + settle` — the settle deadline tracks the most
        // recent event regardless of which timer carries it.
        let rescheduled_deadline = e
            .timers
            .iter()
            .find(|entry| entry.id == rescheduled_timer)
            .map(|entry| entry.deadline)
            .expect("rescheduled settle timer present in heap");
        assert_eq!(
            rescheduled_deadline,
            last_event + SETTLE,
            "rescheduled deadline pinned at last_event + settle",
        );

        // ── Invariant 5: when the rescheduled timer expires and no
        // further events have come in, on_settle_expired transitions
        // to Verifying — the cycle completes.
        let final_expiry = last_event + SETTLE;
        let _ = e.step(
            Input::TimerExpired {
                profile: pid,
                kind: TimerKind::Settle,
                id: rescheduled_timer,
            },
            final_expiry,
        );
        let final_phase = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => &pre.phase,
            other => panic!("expected Active(PreFire), got {other:?}"),
        };
        assert!(
            matches!(final_phase, PreFirePhase::Verifying(_)),
            "after quiet ≥ settle, on_settle_expired transitions to Verifying; \
             got {final_phase:?}",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn finish_burst_to_idle_returns_profile_to_idle() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        out.watch_ops.clear();
        // Production reaches `finish_burst_to_idle` from a probe-response
        // / effect-complete path that has already disarmed the in-flight
        // slot via `take_owner_probe`. Mirror that consume.
        let _ = e.take_owner_probe(ProbeOwner::Profile(pid));

        e.finish_burst_to_idle(pid, &mut out);

        assert!(matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Idle,
        ));
    }

    #[test]
    fn finish_burst_to_idle_on_idle_is_noop() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.finish_burst_to_idle(pid, &mut out);
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops().is_empty());
    }

    #[test]
    fn burst_deadline_unchanged_across_phase_transitions() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        let burst_deadline_initial = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
            _ => panic!("expected Active(PreFire)"),
        };
        let r = e.profiles.get(pid).unwrap().resource;

        e.event_drives_batching(pid, r, Instant::now(), &mut out);
        let burst_deadline_after = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
            _ => panic!("expected Active(PreFire)"),
        };
        assert_eq!(
            burst_deadline_initial, burst_deadline_after,
            "burst_deadline does not reschedule across Verifying → Batching",
        );
    }

    #[test]
    fn transition_to_draining_swaps_phase_only() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        // Production reaches `transition_to_draining` only from
        // `on_probe_response`, which has already disarmed the Verifying
        // slot via `take_owner_probe`. Mirror that consume here.
        let _ = e.take_owner_probe(ProbeOwner::Profile(pid));

        e.transition_to_draining(pid, &mut out);

        let p = e.profiles.get(pid).unwrap();
        let burst = match p.state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!(),
        };
        assert!(matches!(burst.phase, PreFirePhase::Draining));
        // Intent and forced preserved.
        assert_eq!(burst.intent, BurstIntent::Seed);
    }

    // ---------------------------------------------------------------------------
    // Precondition diagnostics — F-MED-7
    //
    // The Phase 3 precondition gates upgrade silent-return on state mismatch
    // to a typed diagnostic (`InvalidBurstTransition`). The tests below pin
    // each gate variant by invoking a helper on a deliberately wrong state.
    // ---------------------------------------------------------------------------

    #[test]
    fn precondition_diagnoses_active_helper_called_on_idle() {
        // `transition_to_verifying` requires `Active(PreFire(_))`. Calling
        // it on a fresh Idle Profile triggers the precondition: the helper
        // bails without minting a correlation, without emitting a Probe,
        // and surfaces `InvalidBurstTransition` tagged with `observed: Idle`.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();

        e.transition_to_verifying(pid, &mut out);

        assert!(
            out.probe_ops().is_empty(),
            "helper bails before any probe-side side effects",
        );
        let saw = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::InvalidBurstTransition {
                    profile,
                    helper: specter_core::BurstHelper::TransitionToVerifying,
                    observed: specter_core::ProfileStateDiscriminant::Idle,
                } if *profile == pid,
            )
        });
        assert!(
            saw,
            "InvalidBurstTransition emitted with helper + observed tags; got {:?}",
            out.diagnostics,
        );
    }

    #[test]
    fn precondition_diagnoses_idle_helper_called_on_active() {
        // `start_seed_burst` requires `Idle`. Calling it on an
        // already-Active Profile triggers the precondition: the helper
        // bails before re-scheduling timers or re-minting a probe, and
        // surfaces `InvalidBurstTransition` with `observed: ActivePreFire`.
        // Replaces the prior `debug_assert!(matches!(p.state,
        // Idle))` discipline that panicked in dev/CI and silently
        // misrouted in release.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        // Drop the first burst's emissions; only the second call is under test.
        let mut out = StepOutput::default();

        e.start_seed_burst(pid, Instant::now(), &mut out);

        assert!(
            out.probe_ops().is_empty(),
            "second start_seed_burst emits no Probe (precondition bails)",
        );
        let saw = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::InvalidBurstTransition {
                    profile,
                    helper: specter_core::BurstHelper::StartSeedBurst,
                    observed: specter_core::ProfileStateDiscriminant::ActivePreFire,
                } if *profile == pid,
            )
        });
        assert!(
            saw,
            "InvalidBurstTransition emitted with helper=StartSeedBurst, \
             observed=ActivePreFire; got {:?}",
            out.diagnostics,
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn precondition_on_stale_profile_is_silent() {
        // Stale `ProfileId` is a benign post-detach race — no diagnostic.
        // The precondition discriminates "live but wrong state" (loud)
        // from "no longer exists" (silent).
        let (mut e, pid) = engine_with_profile();
        e.profiles.detach(&mut e.tree, pid);
        let mut out = StepOutput::default();

        e.transition_to_verifying(pid, &mut out);

        assert!(
            out.diagnostics.is_empty(),
            "stale ProfileId triggers no diagnostic; got {:?}",
            out.diagnostics,
        );
    }

    // ---------------------------------------------------------------------------
    // LCA + force_walk + transition_to_verifying
    // ---------------------------------------------------------------------------

    use crate::burst::{build_force_walk, lca_pair, lca_target, pre_fire_target};
    use specter_core::{PreFireBurst, TimerId};
    use std::collections::BTreeSet;

    /// Build a tree-shaped Engine: anchor `/root`, two children `a` and `b`.
    fn engine_with_two_children() -> (
        Engine,
        specter_core::ProfileId,
        specter_core::ResourceId,
        specter_core::ResourceId,
        specter_core::ResourceId,
    ) {
        let mut e = Engine::new();
        let root = e.tree.ensure_root("root", ResourceRole::User);
        e.tree.set_kind(root, ResourceKind::Dir);
        let a = e
            .tree
            .ensure_child(root, "a", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(a, ResourceKind::Dir);
        let b = e
            .tree
            .ensure_child(root, "b", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(b, ResourceKind::Dir);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                root,
                ProfileIdentity {
                    config: ScanConfig::builder().recursive(true).build(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );
        (e, pid, root, a, b)
    }

    #[test]
    fn lca_empty_dirty_returns_anchor() {
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let dirty = BTreeSet::new();
        let mut out = StepOutput::default();
        let target = lca_target(
            e.profiles.get(pid).unwrap().resource,
            &dirty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target, root);
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn lca_two_siblings_returns_parent() {
        let (e, pid, root, a, b) = engine_with_two_children();
        let dirty: BTreeSet<_> = [a, b].iter().copied().collect();
        let mut out = StepOutput::default();
        let target = lca_target(
            e.profiles.get(pid).unwrap().resource,
            &dirty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target, root);
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn lca_single_dirty_at_anchor_returns_anchor() {
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let dirty: BTreeSet<_> = std::iter::once(root).collect();
        let mut out = StepOutput::default();
        let target = lca_target(
            e.profiles.get(pid).unwrap().resource,
            &dirty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target, root);
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn lca_single_dirty_deep_returns_self() {
        let (e, pid, _root, a, _b) = engine_with_two_children();
        let dirty: BTreeSet<_> = std::iter::once(a).collect();
        let mut out = StepOutput::default();
        let target = lca_target(
            e.profiles.get(pid).unwrap().resource,
            &dirty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target, a);
        assert!(out.diagnostics.is_empty());
    }

    // ---------------------------------------------------------------------------
    // LCA integrity diagnostics — F-MED-4
    //
    // `lca_pair` emits `LcaIntegrityViolation` source-tagged on either
    // failure mode. The lca_target-level `live` filter stays silent
    // (benign delete-recreate race).
    // ---------------------------------------------------------------------------

    #[test]
    fn lca_pair_on_disjoint_roots_emits_broken_ancestry() {
        // Construct a forest: `a` and `b` both have `parent = None`. The
        // depth-equalisation loops don't run (both depth 0). The
        // lockstep loop attempts `tree.parent(a)?` which returns None,
        // so the helper emits `BrokenAncestry` and bails. This is
        // structurally unreachable from `lca_target` in production (the
        // engine maintains a single FS-root scaffold every attach
        // descends from), but the diagnostic surfaces the invariant
        // break if a future refactor ever produces multi-root Trees.
        let mut e = Engine::new();
        let a = e.tree.ensure_root("alpha", ResourceRole::User);
        let b = e.tree.ensure_root("beta", ResourceRole::User);
        let pid = specter_core::ProfileId::default();
        let mut out = StepOutput::default();

        let res = lca_pair(a, b, &e.tree, pid, &mut out);

        assert_eq!(res, None);
        let saw = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::LcaIntegrityViolation {
                    source: specter_core::LcaIntegritySource::BrokenAncestry,
                    ..
                },
            )
        });
        assert!(
            saw,
            "expected LcaIntegrityViolation(BrokenAncestry); got {:?}",
            out.diagnostics,
        );
    }

    #[test]
    fn lca_pair_on_stale_id_emits_stale_id() {
        // A stale ResourceId reaching `lca_pair` directly bypasses
        // `lca_target`'s upstream `live` filter — a fresh class of bug
        // the diagnostic surfaces. We construct a live Tree, reap one
        // entry, then call `lca_pair` directly.
        let (mut e, pid, _root, a, _b) = engine_with_two_children();
        let mut reap_out = StepOutput::default();
        e.tree.try_reap(a, &mut reap_out);
        let mut out = StepOutput::default();
        let live_id = e.profiles.get(pid).unwrap().resource;

        let res = lca_pair(a, live_id, &e.tree, pid, &mut out);

        assert_eq!(res, None);
        let saw = out.diagnostics.iter().any(|d| {
            matches!(
                d,
                specter_core::Diagnostic::LcaIntegrityViolation {
                    profile,
                    source: specter_core::LcaIntegritySource::StaleId,
                } if *profile == pid,
            )
        });
        assert!(
            saw,
            "expected LcaIntegrityViolation(StaleId); got {:?}",
            out.diagnostics,
        );
    }

    #[test]
    fn lca_filters_stale_resource_ids() {
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        // Reap `a` to make its id stale.
        e.tree.try_reap(a, &mut StepOutput::default());
        // Stale id in the set; LCA must filter and return anchor (since the
        // remaining live entry is empty after the filter). The stale-id
        // drop happens at the `live` filter — no diagnostic; per-event
        // noise during delete-recreate churn would flood logs.
        let dirty: BTreeSet<_> = std::iter::once(a).collect();
        let mut out = StepOutput::default();
        let target = lca_target(
            e.profiles.get(pid).unwrap().resource,
            &dirty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target, root);
        assert!(
            out.diagnostics.is_empty(),
            "live-filter drop is silent (benign delete-recreate race)",
        );
    }

    // ---------------------------------------------------------------------------
    // pre_fire_target — centralizes the (anchor_kind, intent) target rule.
    // Locks the contract independent of `transition_to_verifying`'s body so a
    // refactor of the call site can't silently change the rule.
    // ---------------------------------------------------------------------------

    /// Build a `PreFireBurst` shell for direct `pre_fire_target` calls.
    /// `dirty_resources` is the only field the helper reads (besides
    /// `intent`); the rest are stub values that the helper never inspects.
    fn pre_fire_burst_for_test(
        intent: BurstIntent,
        dirty_resources: BTreeSet<specter_core::ResourceId>,
    ) -> PreFireBurst {
        PreFireBurst {
            burst_deadline: TimerId::default(),
            phase: PreFirePhase::Verifying(ProbeSlot::empty()),
            intent,
            forced: false,
            dirty_resources,
            force_walk_resources: BTreeSet::new(),
            probe_target: specter_core::ResourceId::default(),
            last_event_time: None,
        }
    }

    #[test]
    fn pre_fire_target_file_anchor_returns_anchor() {
        // File-anchored Profile + any intent + any dirty set: target is the
        // anchor itself. kqueue per-file FDs surface events at the file
        // directly; promoting past the anchor would route the probe outside
        // the Profile's coverage.
        let (mut e, pid, _parent, file_anchor) = engine_with_file_anchor();
        let mut out = StepOutput::default();
        let pre = pre_fire_burst_for_test(
            BurstIntent::Standard,
            std::iter::once(file_anchor).collect(),
        );
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree, pid, &mut out);
        assert_eq!(target, file_anchor);

        // Same conclusion even if dirty is empty.
        let pre_empty = pre_fire_burst_for_test(BurstIntent::Standard, BTreeSet::new());
        let target_empty = pre_fire_target(
            e.profiles.get(pid).unwrap(),
            &pre_empty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target_empty, file_anchor);

        // And under Seed intent.
        let pre_seed = pre_fire_burst_for_test(BurstIntent::Seed, BTreeSet::new());
        let target_seed = pre_fire_target(
            e.profiles.get(pid).unwrap(),
            &pre_seed,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target_seed, file_anchor);

        // Silence unused-mut on `e` when no further mutation runs.
        let _ = &mut e;
    }

    #[test]
    fn pre_fire_target_seed_intent_returns_anchor() {
        // Seed intent on a Dir-anchored Profile: target is the anchor,
        // regardless of dirty contents. Seed bursts compare against fire
        // history rather than a stable subtree verdict, so they probe at
        // the anchor unconditionally.
        let (e, pid, root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let pre = pre_fire_burst_for_test(BurstIntent::Seed, std::iter::once(a).collect());
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree, pid, &mut out);
        assert_eq!(target, root);

        // Same with empty dirty.
        let pre_empty = pre_fire_burst_for_test(BurstIntent::Seed, BTreeSet::new());
        let target_empty = pre_fire_target(
            e.profiles.get(pid).unwrap(),
            &pre_empty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target_empty, root);
    }

    #[test]
    fn pre_fire_target_standard_uses_lca_of_dirty() {
        // Standard intent on a Dir-anchored Profile: target is
        // `lca_target(anchor, dirty)`. Two sibling dirty entries reduce to
        // their parent (the anchor here).
        let (e, pid, root, a, b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let dirty: BTreeSet<_> = [a, b].iter().copied().collect();
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, dirty);
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree, pid, &mut out);
        assert_eq!(target, root);

        // Single dirty entry reduces to that entry itself (already a Dir).
        let pre_single =
            pre_fire_burst_for_test(BurstIntent::Standard, std::iter::once(a).collect());
        let target_single = pre_fire_target(
            e.profiles.get(pid).unwrap(),
            &pre_single,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(target_single, a);
    }

    #[test]
    fn pre_fire_target_standard_empty_dirty_falls_back_to_anchor() {
        // Standard intent on a Dir-anchored Profile with empty dirty:
        // `lca_target` falls back to anchor (full-walk reconfirm). This
        // covers the Draining-reconfirm hypothetical where every
        // dirty-Resource was reaped between the original verify and the
        // reconfirm.
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, BTreeSet::new());
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree, pid, &mut out);
        assert_eq!(target, root);
    }

    /// Build an Engine with a File-anchored Profile under a parent dir.
    /// `parent/main.rs` (file) is the anchor; the parent dir exists but is
    /// outside the Profile's coverage. Mirrors the production
    /// `attach_sub` flow's anchor-classification step by stamping
    /// `Profile.kind = Some(File)` post-attach — without it the typed
    /// dispatch in `transition_to_verifying` defaults to Subtree.
    fn engine_with_file_anchor() -> (
        Engine,
        specter_core::ProfileId,
        specter_core::ResourceId,
        specter_core::ResourceId,
    ) {
        let mut e = Engine::new();
        let parent = e.tree.ensure_root("parentdir", ResourceRole::User);
        e.tree.set_kind(parent, ResourceKind::Dir);
        let file_anchor = e
            .tree
            .ensure_child(parent, "main.rs", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(file_anchor, ResourceKind::File);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                file_anchor,
                ProfileIdentity {
                    config: ScanConfig::builder().recursive(false).build(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                Some(ResourceKind::File),
            ),
        );
        (e, pid, parent, file_anchor)
    }

    #[test]
    fn lca_pairwise_reduction_resolves_to_shared_intermediate_ancestor() {
        // Witness for the pairwise LCA reduction. Two leaves under
        // disjoint mid-3 branches share a depth-2 ancestor (`l2`); the
        // reduction must resolve to that ancestor, not collapse to the
        // anchor and not return either leaf.
        let mut e = Engine::new();
        let l0 = e.tree.ensure_root("l0", ResourceRole::User);
        e.tree.set_kind(l0, ResourceKind::Dir);
        let l1 = e
            .tree
            .ensure_child(l0, "l1", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(l1, ResourceKind::Dir);
        let l2 = e
            .tree
            .ensure_child(l1, "l2", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(l2, ResourceKind::Dir);
        let l3a = e
            .tree
            .ensure_child(l2, "a", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(l3a, ResourceKind::Dir);
        let l3b = e
            .tree
            .ensure_child(l2, "b", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(l3b, ResourceKind::Dir);
        let leaf_a = e
            .tree
            .ensure_child(l3a, "x", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(leaf_a, ResourceKind::File);
        let leaf_b = e
            .tree
            .ensure_child(l3b, "y", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(leaf_b, ResourceKind::File);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                l0,
                ProfileIdentity {
                    config: ScanConfig::builder().recursive(true).build(),
                    max_settle: MAX_SETTLE,
                    events: NO_EVENTS,
                },
                SETTLE,
                None,
            ),
        );

        let dirty: BTreeSet<_> = [leaf_a, leaf_b].iter().copied().collect();
        let mut out = StepOutput::default();
        let target = lca_target(
            e.profiles.get(pid).unwrap().resource,
            &dirty,
            &e.tree,
            pid,
            &mut out,
        );
        assert_eq!(
            target, l2,
            "LCA of leaves under l3a and l3b is l2 (their shared depth-2 ancestor)",
        );
        assert!(out.diagnostics.is_empty());
    }

    #[test]
    fn transition_to_verifying_on_file_anchor_targets_anchor() {
        // File-anchored Profile: a Standard burst's probe target must be
        // the anchor itself, not the parent dir. The kind dispatch lives
        // at `transition_to_verifying`'s call site (rather than inside
        // `lca_target`) so the LCA helper has a single, narrow contract:
        // "lowest covering ancestor for a Dir-anchored Profile." This
        // test pins the call-site dispatch — promoting past the anchor
        // would route the probe outside the Profile's coverage and
        // (downstream) wholesale-replace `Profile.current` with a Dir
        // snapshot at the parent.
        let (mut e, pid, _parent, file_anchor) = engine_with_file_anchor();
        let mut start_out = StepOutput::default();
        e.start_standard_burst(pid, file_anchor, Instant::now(), &mut start_out);

        let mut probe_out = StepOutput::default();
        e.transition_to_verifying(pid, &mut probe_out);

        let req = probe_out
            .probe_ops()
            .iter()
            .find_map(|op| match op {
                ProbeOp::Probe { request } => Some(request),
                ProbeOp::Cancel { .. } => None,
            })
            .expect("Standard probe emitted");
        let anchor_path = e.tree.path_of(file_anchor).expect("anchor path resolves");
        assert!(
            matches!(
                req,
                ProbeRequest::AnchorFile { target_path, .. } if *target_path == anchor_path,
            ),
            "Standard burst on a File-anchored Profile must emit ProbeRequest::AnchorFile \
             at the anchor's path",
        );
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn build_force_walk_filters_to_subtree_of_target() {
        let (e, _pid, root, a, b) = engine_with_two_children();
        // target = a; only `a` itself qualifies (b is a sibling).
        let set: BTreeSet<_> = [root, a, b].iter().copied().collect();
        let paths = build_force_walk(&set, a, &e.tree);
        let path_a = e.tree.path_of(a).unwrap();
        assert!(paths.contains(&path_a));
        assert!(!paths.contains(&e.tree.path_of(b).unwrap()));
        // root is an ancestor of a (not a descendant), so it's filtered out.
        assert!(!paths.contains(&e.tree.path_of(root).unwrap()));
    }

    #[test]
    fn transition_to_verifying_standard_uses_lca() {
        let (mut e, pid, _root, a, b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        // Standard burst with two dirty siblings → LCA = root (the anchor).
        e.start_standard_burst(pid, a, now, &mut out);
        // Inject a second dirty resource so LCA computes the sibling parent.
        if let Some(pre) = e.profiles.get_mut(pid).unwrap().pre_fire_burst_mut() {
            pre.dirty_resources.insert(b);
            pre.force_walk_resources.insert(b);
        }
        let mut probe_out = StepOutput::default();
        e.transition_to_verifying(pid, &mut probe_out);

        let req = probe_out
            .probe_ops()
            .iter()
            .find_map(|op| match op {
                ProbeOp::Probe { request } => Some(request),
                ProbeOp::Cancel { .. } => None,
            })
            .expect("Standard probe emitted");
        // a + b's LCA is root (the anchor) because they're siblings under root.
        // Subtree variant carries `target_path` and `force_walk` directly;
        // a Standard burst on a Dir-anchored Profile must produce this variant.
        let anchor_path = e
            .tree
            .path_of(e.profiles.get(pid).unwrap().resource)
            .expect("anchor path resolves");
        match req {
            ProbeRequest::Subtree {
                target_path,
                force_walk,
                ..
            } => {
                assert_eq!(*target_path, anchor_path);
                assert_eq!(force_walk.len(), 2);
            }
            other => panic!(
                "Standard burst on Dir-anchored Profile must emit ProbeRequest::Subtree; \
                 got {other:?}",
            ),
        }
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn transition_to_verifying_clears_force_walk_resources() {
        let (mut e, pid, _root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, a, now, &mut out);
        e.transition_to_verifying(pid, &mut out);

        // After transition_to_verifying, force_walk_resources should be
        // cleared (consumed by this emission); subsequent events accumulate
        // fresh.
        let burst = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!(),
        };
        assert!(burst.force_walk_resources.is_empty());
        // dirty_resources is preserved (LCA basis spans the whole burst).
        assert!(!burst.dirty_resources.is_empty());
        // probe_target was overwritten by the LCA result — non-Optional under
        // the type split, so we assert it equals the expected anchor LCA.
        // For a single dirty event at `a` under root, LCA promotes to `a`
        // (`a` is itself a Dir).
        assert_eq!(burst.probe_target, a);
        let _ = e.cancel_all_in_flight_probes();
    }
}
