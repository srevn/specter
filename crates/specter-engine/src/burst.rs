//! Burst lifecycle helpers.
//!
//! Each helper is the **single source** of one transition kind — a phase
//! transition body, a Burst construction, or a return-to-Idle. Centralizing
//! the timer scheduling, refcount edges, and Burst-struct mutations here
//! prevents drift between the transition-row handlers and the
//! post-`EffectComplete` re-probe path.
//!
//! `ActiveBurst` splits into `PreFireBurst` / `PostFireBurst` (see
//! [`specter_core::profile`]); helpers below own a typed view of one or
//! the other. The fire transition (`Verifying → Awaiting`) is a typed
//! state-machine move at [`PreFireBurst::into_post_fire`].
//!
//! - `start_seed_burst` / `start_standard_burst` — Idle →
//!   `Active(PreFire(_))`.
//! - `event_drives_batching` (FsEvent during pre-fire) /
//!   `unstable_response_drives_batching` (probe-unstable response) /
//!   `transition_to_verifying` (settle-timer expiry, burst-deadline,
//!   Draining → Verifying reconfirm) /
//!   `transition_to_draining` — pre-fire phase swaps (mutate
//!   `PreFireBurst`).
//! - `transition_to_awaiting` — `Active(PreFire(_))` → `Active(PostFire(_))`,
//!   the sole site that crosses the fire boundary (via
//!   `PreFireBurst::into_post_fire`).
//! - `transition_to_rebasing` — `Awaiting → Rebasing` (mutates
//!   `PostFireBurst`).
//! - `absorb_event_into_fire_tail` — FsEvent during post-fire (mutates
//!   `PostFireBurst.force_walk_resources`).
//! - `finish_burst_to_idle` — Active → Idle, single point of `-suppress` and
//!   `propagate(-1)`. Discriminates `PreFire` / `PostFire` at the take.
//!
//! The two batching helpers exist as a deliberate split rather than one
//! helper with a runtime flag: each caller has **static knowledge** of
//! whether a probe is in flight (only `event_drives_batching` may need to
//! emit `ProbeOp::Cancel`). Encoding that knowledge as helper identity
//! makes a stray Cancel on the just-responded path structurally
//! impossible.
//!
//! Probe emission flows through two structural primitives that every
//! burst-launch helper routes through:
//!
//! - [`pre_fire_target`] — pure function returning the `ResourceId` the
//!   next pre-fire probe should target. Centralizes the
//!   `(anchor_kind, intent)` rule (File anchor → anchor; Seed → anchor;
//!   Standard → LCA of `dirty_resources`). Post-fire rebases target the
//!   anchor unconditionally and bypass this helper.
//! - [`Engine::emit_probe_at`] — Engine method that dispatches on
//!   `Profile.kind` and constructs the probe via the typed helpers in
//!   `probe_channel` (`emit_anchor_probe` / `emit_subtree_probe`). The
//!   single site that maps a `(profile_id, target)` pair to a
//!   `ProbeOp::Probe` op. Unclassified anchors take the Subtree arm —
//!   the walker returns `Vanished` on kind mismatch and the engine
//!   recovers via descent.

use crate::Engine;
use crate::path::empty_path;
use crate::probe_channel::OpenKind;
use crate::refcounts::{add_suppress, sub_suppress};
use smallvec::SmallVec;
use specter_core::{
    ActiveBurst, BurstFinish, BurstHelper, BurstIntent, Diagnostic, FsEvent, LcaIntegritySource,
    PostFirePhase, PreFireBurst, PreFirePhase, ProbeCorrelation, ProbeOwner, Profile, ProfileId,
    ProfileState, ReapTrigger, ResourceId, ResourceKind, StepOutput, TimerKind, Tree, TreeSnapshot,
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
    /// the gate.
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
    /// Caller has verified `Profile.state == Idle`. Constructs the Burst,
    /// schedules `burst_deadline`, mints the probe correlation, emits
    /// Probe (with `current.subtree_at(anchor)` as `baseline_subtree`
    /// when post-recovery Seed has one — enables walker mtime-skip on
    /// idempotent events), and `+suppress` on the anchor.
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
        let max_settle = p.max_settle;

        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        // Phase-write BEFORE channel open. The invariant is "phase ==
        // Verifying iff the channel is open for this owner with
        // `OpenKind::ProfileVerifying`"; both directions can be
        // violated only intra-step. Open→write would leave the channel
        // entry live while the phase is still Idle, so a stray observer
        // reading both would see a live correlation in a non-Verifying
        // state; worse, a re-entrant open would trip
        // `ProbeChannel::open`'s unconditional double-open assert.
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.transition_state(ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    burst_deadline,
                    phase: PreFirePhase::Verifying,
                    intent: BurstIntent::Seed,
                    forced: false,
                    dirty_resources: BTreeSet::new(),
                    force_walk_resources: BTreeSet::new(),
                    // Seed targets the anchor; the field is invariant for the
                    // Seed burst's pre-fire lifetime (`transition_to_verifying`
                    // re-runs for Seed only on Draining-reconfirm, which Seed
                    // bursts never reach because they skip Batching).
                    probe_target: resource,
                    suppressed_resources: BTreeSet::new(),
                    // Seed bursts skip Batching; the field has no consumer
                    // until a fresh FsEvent during the verify routes through
                    // `event_drives_batching` and repopulates it.
                    last_event_time: None,
                }),
                // Fresh burst — directive starts at `ReturnToIdle`. Flips
                // to `Reap` only on mid-burst `mark_active_for_reap`.
                BurstFinish::ReturnToIdle,
            ));
        }

        let correlation = self
            .probe_channel
            .open(ProbeOwner::Profile(profile_id), OpenKind::ProfileVerifying);

        add_suppress(&mut self.tree, resource, out);
        // Seed bursts always target the anchor; `force_walk` is empty
        // because no events have been observed yet. An empty `BTreeSet`
        // allocates no heap storage — passing a reference is cheap.
        self.emit_probe_at(
            profile_id,
            correlation,
            resource,
            &BTreeSet::new(),
            false,
            out,
        );
    }

    /// Start a Standard burst: schedule settle + `burst_deadline`,
    /// `+suppress`, propagate(+1). No Probe — that fires on `settle_timer`
    /// expiry via `transition_to_verifying`.
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
        let max_settle = p.max_settle;

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
                    suppressed_resources: BTreeSet::new(),
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

        add_suppress(&mut self.tree, resource, out);
        let _ = crate::stability::propagate(&mut self.profiles, profile_id, 1);
    }

    /// Caller: `drive_burst` Active branch — an `FsEvent` arrived during a
    /// burst. Cancels any in-flight verify (iff the prior phase was
    /// `Verifying`), accumulates the event into `dirty_resources` and
    /// `force_walk_resources`, updates `last_event_time`, arms a fresh
    /// settle timer **only when re-entering Batching from Verifying or
    /// Draining**, emits a per-resource `Suppress` on the first per-burst
    /// occurrence of a non-anchor resource (recorded in
    /// `suppressed_resources` for the symmetric drain at
    /// `transition_to_verifying`), and writes `phase = Batching {
    /// settle_timer }`. `intent`, `forced`, and the `burst_deadline` are
    /// preserved.
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
    ///
    /// **Per-resource Suppress.** The anchor's suppress is bracketed by
    /// `start_*_burst` ↔ `finish_burst_to_idle`. Non-anchor resources
    /// receiving events during the Batching window get their own
    /// suppress bracket: `add_suppress` here (0→1 edge ⇒ one Suppress
    /// op) and the symmetric `sub_suppress` at the next
    /// `transition_to_verifying`. Subsequent events on the same
    /// resource within one Batching window are dedup'd via the burst's
    /// `suppressed_resources` set, so the watcher sees at most one
    /// Suppress / Unsuppress pair per (Burst, non-anchor resource).
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
        let anchor = p.resource;

        // Read phase before mutating self via `cancel_owner_probe`. The
        // Cancel emission doesn't touch `burst.phase`, but it does take
        // `&mut self` and so invalidates the borrow on `burst`. Decide
        // here whether the existing Batching settle timer (if any) carries
        // over, or whether we mint a fresh one for a Verifying/Draining
        // re-entry. The decision is structural: a live Batching has its
        // own timer slot; Verifying/Draining have none.
        let needs_fresh_timer =
            matches!(pre.phase, PreFirePhase::Verifying | PreFirePhase::Draining);

        // Idempotent: emits Cancel iff the probe channel is open
        // (Verifying ⇒ channel open for this owner). For Batching /
        // Draining entries, no probe is in flight and the helper is
        // a no-op — matching the prior `was_verifying` snapshot's role.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);

        let new_settle_timer = if needs_fresh_timer {
            Some(
                self.timers
                    .schedule(now + settle, profile_id, TimerKind::Settle),
            )
        } else {
            None
        };

        // Decide the per-resource Suppress before the mutation block.
        // The anchor is excluded — its suppress is the existing
        // `start_*_burst → finish_burst_to_idle` bracket. Duplicates
        // within one Batching window are excluded via the burst's
        // tracking set so the symmetric drain at `transition_to_verifying`
        // pairs each `add_suppress` with exactly one `sub_suppress`.
        //
        // The borrow on `self.profiles` ends with the `is_some_and` so
        // `add_suppress` (which needs `&mut self.tree`) can run before
        // re-borrowing `self.profiles` mutably below.
        let needs_suppress = event_resource != anchor
            && self
                .profiles
                .get(profile_id)
                .is_some_and(|p| match p.state() {
                    ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
                        !pre.suppressed_resources.contains(&event_resource)
                    }
                    _ => false,
                });
        if needs_suppress {
            add_suppress(&mut self.tree, event_resource, out);
        }

        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            pre.last_event_time = Some(now);
            if needs_suppress {
                pre.suppressed_resources.insert(event_resource);
            }
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
    /// just responded with an unstable verdict. The probe channel was
    /// already closed at the top of `on_probe_response`; no Cancel needed.
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
    /// through `event_drives_batching`, which Cancels the in-flight
    /// probe and closes the channel; the eventual late response then
    /// fails `close_if`'s correlation check and drops as
    /// `StaleProbeResponse`. The forced + not-stable case in
    /// `dispatch_standard_ok` also bypasses this helper — forced +
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
    /// **Emission.** Probe construction lives in [`Engine::emit_probe_at`].
    /// `force_walk_resources` (consumed via `mem::take` below) ships as
    /// the walker's force-walk hint so events the engine knows about
    /// defeat mtime-skip; `Burst.forced` propagates so the walker
    /// bypasses mtime-skip on a force-fire (max-settle deadline
    /// elapsed). New events arriving during `Verifying` accumulate into
    /// the cleared `force_walk_resources` set and ship on the next
    /// emission.
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

        // Take the per-burst consumables in one mutable-borrow window.
        // The post-fire phases never reach this helper — production
        // callers (Settle expiry, BurstDeadline expiry, ancestor
        // reconfirm from `finish_burst_to_idle`) are gated on pre-fire
        // phases via `is_timer_referenced` and the Draining hit-zero
        // check respectively. The early return guards a stray call
        // from opening a fresh probe channel while an effect wait is
        // still in flight — `ProbeChannel::open` would panic
        // unconditionally on the double-open.
        //
        // `force_walk_resources` and `suppressed_resources` are
        // single-use accumulators consumed by this transition;
        // `mem::take` moves them out and leaves the fields empty (no
        // follow-up `clear()` required). `dirty_resources` is preserved
        // — it carries the LCA basis across the whole burst.
        let (force_set, suppressed_drain, forced) = match self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            Some(pre) => (
                std::mem::take(&mut pre.force_walk_resources),
                std::mem::take(&mut pre.suppressed_resources),
                pre.forced,
            ),
            None => return,
        };

        // Drain the per-burst `suppressed_resources` taken above. Each
        // entry was bumped 0→1 (or N→N+1 in the multi-Profile fan-in
        // case) by `event_drives_batching`; `sub_suppress` returns it
        // to 0 (emitting Unsuppress) or to N (no emit). Refcount math
        // holds across overlapping bursts; `StepOutput` sees one
        // Suppress / Unsuppress pair per (Burst, non-anchor resource).
        // BTreeSet iteration is sorted, so emitted Unsuppress ops land
        // in `ResourceId`-ascending order — coherent with
        // `StepOutput.watch_ops`'s sort discipline.
        for r in &suppressed_drain {
            sub_suppress(&mut self.tree, *r, out);
        }

        // Phase-write BEFORE channel open. See the rationale on
        // `start_seed_burst`'s open call site — the invariant
        // ("phase == Verifying iff channel open with ProfileVerifying")
        // is enforced at probe boundaries, and write→open is the
        // strictly safer of the two intra-step ordering options.
        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
            pre.phase = PreFirePhase::Verifying;
            pre.probe_target = target;
        }

        let correlation = self
            .probe_channel
            .open(ProbeOwner::Profile(profile_id), OpenKind::ProfileVerifying);
        self.emit_probe_at(profile_id, correlation, target, &force_set, forced, out);
    }

    /// Phase: `Verifying` → `Draining`. Phase swap only — the exit body
    /// (`Draining` → `Verifying` reconfirm) is driven by
    /// `finish_burst_to_idle` when a child Profile's `propagate(-1)`
    /// returns this Profile in its hit-zero list.
    ///
    /// `Draining` is a unit variant: the stable snapshot lives on
    /// `Profile.current` (set by `dispatch_standard_ok` immediately
    /// before this call), so no `Arc<TreeSnapshot>` is duplicated on the
    /// phase variant.
    pub(crate) fn transition_to_draining(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::TransitionToDraining, out) {
            return;
        }
        if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::pre_fire_burst_mut)
        {
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
                .then_some(p.max_settle)
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
    /// **Probe channel.** `ProbeChannel::open` opens a fresh channel
    /// entry with `OpenKind::ProfileRebasing`. I5 holds because
    /// Verifying closed the channel before `emit_effects` ran, and
    /// Awaiting does not open. The post-fire probe targets the anchor
    /// (we want the freshest disk state of the whole watched subtree,
    /// not the LCA of the now-stale `dirty_resources`).
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
    /// (`on_effect_complete`'s `AwaitAction::Rebase` arm and
    /// `handle_gate_deadline`) have already verified `Active(_)` with
    /// phase `Awaiting` before reaching this helper. Defensively
    /// early-returning on non-Active matches `transition_to_verifying`'s
    /// strict policy and avoids the latent state-machine bug where a
    /// stray call would mint a fresh probe correlation, emit a Probe
    /// op, and then fail to write the phase (because the phase-write
    /// arm requires `Active`) — leaving the probe channel open with no
    /// matching Burst.
    pub(crate) fn transition_to_rebasing(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_post_fire(profile_id, BurstHelper::TransitionToRebasing, out) {
            return;
        }
        // Take absorbed events for the rebase walker's force_walk and
        // capture the anchor (`target = p.resource`) in one mut-borrow
        // window. `force_walk_resources` on `PostFireBurst` is the
        // post-fire accumulator fed by `absorb_event_into_fire_tail`.
        // `post_fire_burst_mut` projects `post`; the precondition
        // guaranteed PostFire above.
        let (force_set, target) = match self.profiles.get_mut(profile_id) {
            Some(p) => {
                let target = p.resource;
                match p.post_fire_burst_mut() {
                    Some(post) => (std::mem::take(&mut post.force_walk_resources), target),
                    None => return,
                }
            }
            None => return,
        };

        // Phase-write BEFORE channel open. See `start_seed_burst`'s
        // rationale.
        if let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        {
            post.phase = PostFirePhase::Rebasing;
        }

        let correlation = self
            .probe_channel
            .open(ProbeOwner::Profile(profile_id), OpenKind::ProfileRebasing);
        // `forced = false`: the rebase probe is never forced — that
        // field is pre-fire-only (a `BurstDeadline` decision) and
        // doesn't survive the typed move into `PostFireBurst`.
        self.emit_probe_at(profile_id, correlation, target, &force_set, false, out);
    }

    /// Active → Idle. Single source of `-suppress` and `propagate(-1)`.
    /// The active burst's timers are not explicitly cancelled — lazy
    /// invalidation in `pop_expired` drops them when they fire.
    /// Idempotent: silent no-op on already-Idle Profiles.
    ///
    /// **Draining-exit driver.** `propagate(-1)` returns ancestors whose
    /// `dirty_descendants` just hit zero AND are in `PreFirePhase::Draining`.
    /// The Engine drives each through `transition_to_verifying` in the
    /// same step — the reconfirm probe compares against the Profile's
    /// `current` (set when `dispatch_standard_ok` entered Draining).
    /// Same-step ordering means the `StepOutput` reflects the cascade:
    /// child's burst end → parent reconfirm Probe in one `step` call.
    ///
    /// **Burst-finish directive.** If the prior state's
    /// [`BurstFinish`] is [`BurstFinish::Reap`] (the last Sub was
    /// detached mid-burst, or the anchor's all-dynamic teardown
    /// converged on a still-Active Profile), `Engine::reap_profile`
    /// runs in the same step after `propagate(-1)` — `via =
    /// DeferredFromBurst` distinguishes this path from the immediate
    /// reap in `detach_sub_inner`. Otherwise the Profile rests at
    /// [`ProfileState::Idle`].
    pub(crate) fn finish_burst_to_idle(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };
        let resource = p.resource;

        // Take the burst-by-value via `transition_state(Idle)` and
        // discriminate on the typed variant. Both pre-fire and post-fire
        // arms preserve `intent` (the propagate(-1) gate); only the
        // pre-fire arm carries `suppressed_resources` to drain. PostFire's
        // `suppressed_resources` is empty by construction —
        // `transition_to_verifying` drained the set immediately before
        // the fire, and `into_post_fire`'s debug_assert catches any
        // future drift.
        //
        // After this point `p.state == Idle` for the whole helper window.
        // The subsequent `sub_suppress` / `propagate(-1)` /
        // `transition_to_verifying` (ancestor reconfirm) / reap calls all
        // run against a focal Profile in Idle — future observers (e.g., a
        // hook firing on state transitions) would see the transition
        // bracket cleanly.
        //
        // The defensive `suppressed_resources` drain catches abnormal-
        // end paths that bypass `transition_to_verifying`
        // (`finalize_anchor_lost` mid-Batching, `reap_profile` mid-burst,
        // config-diff reap). `sub_suppress` on a stale `ResourceId`
        // (slot reaped between event ingestion and burst end via
        // `discard_anchor_state`'s descendant release) is a no-op —
        // `refcounts::sub_suppress` short-circuits on a missing slot.
        // Only Standard bursts call `propagate(+1)` at start (the
        // burst-propagation row), so only Standard bursts call
        // `propagate(-1)` at end. Seed bursts never contribute to
        // ancestor `dirty_descendants`.
        let prior = p.transition_state(ProfileState::Idle);
        // Capture `(was_standard, suppressed_drain, finish)` from the
        // consumed prior state. `finish` is captured here — not re-read
        // from `profiles.get(profile_id)` after the swap — so the
        // directive is locked in at burst-end entry; a hypothetical
        // future mid-helper write to a re-borrowed Profile can't flip
        // the reap decision under us.
        let (was_standard, suppressed_drain, finish) = match prior {
            ProfileState::Active(ActiveBurst::PreFire(pre), finish) => (
                matches!(pre.intent, BurstIntent::Standard),
                pre.suppressed_resources,
                finish,
            ),
            ProfileState::Active(ActiveBurst::PostFire(post), finish) => (
                matches!(post.intent, BurstIntent::Standard),
                BTreeSet::new(),
                finish,
            ),
            other => {
                // Idle / Pending — no burst-end machinery to run. Restore.
                p.transition_state(other);
                return;
            }
        };

        for r in &suppressed_drain {
            sub_suppress(&mut self.tree, *r, out);
        }

        sub_suppress(&mut self.tree, resource, out);

        if was_standard {
            let hit_zero = crate::stability::propagate(&mut self.profiles, profile_id, -1);

            // Draining → Verifying reconfirm for ancestors whose count
            // just hit zero. `transition_to_verifying` mints a fresh
            // correlation and emits Probe; the response routes through
            // `dispatch_standard_ok` as a normal Standard burst.
            for ancestor in hit_zero {
                self.transition_to_verifying(ancestor, out);
            }
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

    /// Emit a probe at `target` on behalf of `Profile(profile_id)`. The
    /// single probe-emit primitive shared by the three burst-launch
    /// helpers — `start_seed_burst`, `transition_to_verifying`,
    /// `transition_to_rebasing` — so the kind dispatch lives in exactly
    /// one place.
    ///
    /// Routes on `Profile.kind`:
    /// - `Some(File)` → [`Engine::emit_anchor_probe`] at `target_path`.
    /// - `Some(Dir | Unknown) | None` → [`Engine::emit_subtree_probe`]
    ///   at `(target, target_path)` carrying the Profile's
    ///   `(config, config_hash)`, `baseline_subtree =
    ///   p.current.subtree_at(target)`, and `force_walk_paths`
    ///   pre-filtered to `target`'s subtree.
    ///
    /// Callers pass only the structural inputs they own: the
    /// just-minted `correlation`, the resolved `target`, the
    /// pre-/post-fire `force_walk_resources` accumulator, and `forced`.
    /// The helper derives `target_path`, `baseline_subtree`,
    /// `force_walk_paths`, and the Profile's `scan_config` /
    /// `config_hash` directly from `(profile_id, target)` — avoiding a
    /// per-call-site preamble that repeated the same Profile borrow.
    ///
    /// Defensive early-return on a stale `profile_id` matches the
    /// surrounding lifecycle helpers' silent-no-op policy: a Profile
    /// reaped between mint and emit yields nothing rather than emitting
    /// a probe against a nonexistent owner. In production every caller
    /// has already validated the Profile under one of the lifecycle
    /// helpers' guards.
    fn emit_probe_at(
        &self,
        profile_id: ProfileId,
        correlation: ProbeCorrelation,
        target: ResourceId,
        force_walk_resources: &BTreeSet<ResourceId>,
        forced: bool,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let owner = ProbeOwner::Profile(profile_id);
        let target_path = self.tree.path_of(target).unwrap_or_else(empty_path);
        match p.kind() {
            Some(ResourceKind::File) => {
                Self::emit_anchor_probe(owner, correlation, target_path, out);
            }
            // Dir or unclassified ⇒ unified Subtree fallback. An Unknown
            // anchor (resource-based attach whose first probe hasn't yet
            // returned, or a slot whose kind never propagated past the
            // default) probes as a Dir; the walker returns `Vanished` on
            // kind mismatch and the engine routes through the
            // dispatch_*_vanished paths to recover via descent.
            Some(ResourceKind::Dir | ResourceKind::Unknown) | None => {
                let baseline_subtree = p
                    .current()
                    .and_then(|s| s.subtree_at(p.resource, target, &self.tree));
                let force_walk_paths = build_force_walk(force_walk_resources, target, &self.tree);
                Self::emit_subtree_probe(
                    owner,
                    correlation,
                    target_path,
                    p.config.clone(),
                    p.config_hash,
                    baseline_subtree,
                    force_walk_paths,
                    forced,
                    out,
                );
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
/// (kqueue per-file FDs surface events at the file directly); the
/// [`pre_fire_target`] / [`Engine::emit_probe_at`] pair routes File
/// anchors to [`Engine::emit_anchor_probe`] without consulting this
/// helper, so a File-anchored Profile never reaches `lca_target` in
/// production. The `live.contains(&anchor)` short-circuit below remains
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
/// [`PreFirePhase::Batching`] or [`PreFirePhase::Draining`], hit-zero
/// Draining reconfirm) all resolve here:
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
        Profile, ProfileState, ResourceKind, ResourceRole, ScanConfig, StepOutput, TimerKind,
        WatchOp,
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
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
                None,
            ),
        );
        (e, pid)
    }

    #[test]
    fn start_seed_burst_emits_probe_and_suppress() {
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
        assert!(matches!(burst.phase, PreFirePhase::Verifying));
        assert!(!burst.forced);

        // Output: one Probe + one Suppress.
        let probes = out
            .probe_ops
            .iter()
            .filter(|op| matches!(op, ProbeOp::Probe { .. }))
            .count();
        assert_eq!(probes, 1);
        let suppresses = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Suppress { .. }))
            .count();
        assert_eq!(suppresses, 1);

        // Heap: only burst_deadline (Seed has no settle_timer).
        assert_eq!(e.timers.len(), 1);
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
        assert!(out.probe_ops.is_empty());
        let suppresses = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Suppress { .. }))
            .count();
        assert_eq!(suppresses, 1);
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
        out.probe_ops.clear();

        e.transition_to_verifying(pid, &mut out);

        match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => {
                assert!(matches!(b.phase, PreFirePhase::Verifying));
            }
            _ => panic!("expected Active(PreFire)"),
        }
        let correlation = e
            .pending_probe_for(ProbeOwner::Profile(pid))
            .expect("Verifying probe in flight on probe channel");

        // Output: one Probe whose correlation matches.
        let probe_correlation = out.probe_ops.iter().find_map(|op| match op {
            ProbeOp::Probe { request } => Some(request.correlation()),
            _ => None,
        });
        assert_eq!(probe_correlation, Some(correlation));
    }

    #[test]
    fn event_during_verifying_emits_cancel_and_resets_batching() {
        // FsEvent during Verifying: Cancel emitted; phase becomes Batching
        // with a fresh settle_timer; intent preserved.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out); // Seed → Verifying
        out.probe_ops.clear();
        let r = e.profiles.get(pid).unwrap().resource;

        e.event_drives_batching(pid, r, Instant::now(), &mut out);

        // One Cancel emitted for the in-flight probe.
        let cancel_count = out
            .probe_ops
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
        out.probe_ops.clear();

        e.event_drives_batching(pid, r, Instant::now(), &mut out);

        let cancels = out
            .probe_ops
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
        out.probe_ops.clear();

        e.unstable_response_drives_batching(pid, now, &mut out);

        assert!(out.probe_ops.is_empty());
        let phase = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => &pre.phase,
            _ => panic!("expected Active(PreFire)"),
        };
        assert!(matches!(phase, PreFirePhase::Batching { .. }));
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
            out.probe_ops.clear();
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
            PreFirePhase::Verifying | PreFirePhase::Draining => {
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
            matches!(final_phase, PreFirePhase::Verifying),
            "after quiet ≥ settle, on_settle_expired transitions to Verifying; \
             got {final_phase:?}",
        );
    }

    #[test]
    fn finish_burst_to_idle_emits_unsuppress() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, Instant::now(), &mut out);
        out.watch_ops.clear();

        e.finish_burst_to_idle(pid, &mut out);

        assert!(matches!(
            e.profiles.get(pid).unwrap().state(),
            ProfileState::Idle,
        ));
        let unsuppresses = out
            .watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unsuppress { .. }))
            .count();
        assert_eq!(unsuppresses, 1);
    }

    #[test]
    fn finish_burst_to_idle_on_idle_is_noop() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.finish_burst_to_idle(pid, &mut out);
        assert!(out.watch_ops.is_empty());
        assert!(out.probe_ops.is_empty());
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
            out.probe_ops.is_empty(),
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
        out.probe_ops.clear();
        out.watch_ops.clear();
        out.diagnostics.clear();

        e.start_seed_burst(pid, Instant::now(), &mut out);

        assert!(
            out.probe_ops.is_empty(),
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
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
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
            phase: PreFirePhase::Verifying,
            intent,
            forced: false,
            dirty_resources,
            force_walk_resources: BTreeSet::new(),
            probe_target: specter_core::ResourceId::default(),
            suppressed_resources: BTreeSet::new(),
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
                ScanConfig::builder().recursive(false).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
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
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
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
            .probe_ops
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
            .probe_ops
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
    }

    // ---------------------------------------------------------------------------
    // Per-resource Suppress reuse during Batching
    //
    // `event_drives_batching` emits a Suppress on the first per-burst event
    // at a non-anchor resource; `transition_to_verifying` drains the set
    // with one Unsuppress per entry. The anchor's suppress is bracketed by
    // `start_*_burst` ↔ `finish_burst_to_idle` and never participates in
    // this set.
    // ---------------------------------------------------------------------------

    /// Number of `WatchOp::Suppress { resource: r }` ops in `out`.
    fn suppress_count_for(out: &StepOutput, r: specter_core::ResourceId) -> usize {
        out.watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Suppress { resource } if *resource == r))
            .count()
    }

    /// Number of `WatchOp::Unsuppress { resource: r }` ops in `out`.
    fn unsuppress_count_for(out: &StepOutput, r: specter_core::ResourceId) -> usize {
        out.watch_ops
            .iter()
            .filter(|op| matches!(op, WatchOp::Unsuppress { resource } if *resource == r))
            .count()
    }

    #[test]
    fn event_drives_batching_emits_suppress_on_first_non_anchor_event() {
        // First per-burst event at a non-anchor resource bumps its
        // suppress_count 0→1 and emits one `Suppress` op; the resource is
        // recorded in `Burst.suppressed_resources` so the symmetric drain
        // at `transition_to_verifying` pairs it with one `Unsuppress`.
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        out.watch_ops.clear();

        e.event_drives_batching(pid, a, now, &mut out);

        assert_eq!(
            suppress_count_for(&out, a),
            1,
            "first non-anchor event emits Suppress(a)",
        );
        let burst = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            burst.suppressed_resources.contains(&a),
            "burst tracks the suppressed non-anchor resource",
        );
        assert_eq!(
            e.tree.get(a).unwrap().suppress_count(),
            1,
            "underlying suppress_count bumped to 1",
        );
    }

    #[test]
    fn event_drives_batching_dedups_suppress_on_repeated_non_anchor_event() {
        // Subsequent events at the same non-anchor resource within one
        // Batching window do NOT re-emit Suppress; the burst's per-burst
        // tracking set is the dedup horizon.
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);

        e.event_drives_batching(pid, a, now, &mut out);
        out.watch_ops.clear();

        e.event_drives_batching(pid, a, now + Duration::from_millis(1), &mut out);
        e.event_drives_batching(pid, a, now + Duration::from_millis(2), &mut out);

        assert_eq!(
            suppress_count_for(&out, a),
            0,
            "repeat events on the same non-anchor resource emit no extra Suppress",
        );
        assert_eq!(
            e.tree.get(a).unwrap().suppress_count(),
            1,
            "underlying suppress_count stays at 1",
        );
    }

    #[test]
    fn event_drives_batching_does_not_suppress_anchor() {
        // FsEvents at the anchor are excluded from suppress bumping: the
        // anchor's suppress is the existing `start_*_burst →
        // finish_burst_to_idle` bracket. `start_standard_burst` already
        // bumped the anchor to 1; additional anchor-targeted events do not
        // bump it further.
        let (mut e, pid, root, _a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        let suppress_after_start = e.tree.get(root).unwrap().suppress_count();
        out.watch_ops.clear();

        e.event_drives_batching(pid, root, now + Duration::from_millis(1), &mut out);
        e.event_drives_batching(pid, root, now + Duration::from_millis(2), &mut out);

        assert_eq!(
            suppress_count_for(&out, root),
            0,
            "anchor events emit no per-resource Suppress",
        );
        let burst = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            !burst.suppressed_resources.contains(&root),
            "anchor never enters suppressed_resources",
        );
        assert_eq!(
            e.tree.get(root).unwrap().suppress_count(),
            suppress_after_start,
            "anchor's suppress_count unchanged across event_drives_batching calls",
        );
    }

    #[test]
    fn transition_to_verifying_drains_suppressed_resources_with_unsuppress() {
        // Multiple non-anchor events accumulate into `suppressed_resources`;
        // `transition_to_verifying` drains the set with one Unsuppress per
        // entry and leaves the set empty.
        let (mut e, pid, root, a, b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        e.event_drives_batching(pid, a, now + Duration::from_millis(1), &mut out);
        e.event_drives_batching(pid, b, now + Duration::from_millis(2), &mut out);
        out.watch_ops.clear();

        e.transition_to_verifying(pid, &mut out);

        assert_eq!(
            unsuppress_count_for(&out, a),
            1,
            "drain emits one Unsuppress(a)",
        );
        assert_eq!(
            unsuppress_count_for(&out, b),
            1,
            "drain emits one Unsuppress(b)",
        );
        let burst = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            burst.suppressed_resources.is_empty(),
            "drain clears the per-burst tracking set",
        );
        assert_eq!(e.tree.get(a).unwrap().suppress_count(), 0);
        assert_eq!(e.tree.get(b).unwrap().suppress_count(), 0);
    }

    #[test]
    fn unstable_verify_then_event_re_emits_suppress() {
        // After `transition_to_verifying` clears `suppressed_resources`,
        // an unstable response routes back to Batching via
        // `unstable_response_drives_batching`. The next FsEvent at the
        // same non-anchor resource sees an empty set and re-emits
        // Suppress (the symmetric drain at the next
        // `transition_to_verifying` pairs it).
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        e.event_drives_batching(pid, a, now + Duration::from_millis(1), &mut out);
        e.transition_to_verifying(pid, &mut out);
        // Unstable response shrinks the burst back to Batching without
        // emitting Cancel — the verify just responded, so no in-flight
        // probe to revoke. (`unstable_response_drives_batching` does not
        // call `cancel_owner_probe`.)
        e.unstable_response_drives_batching(pid, now + Duration::from_millis(2), &mut out);
        out.watch_ops.clear();

        e.event_drives_batching(pid, a, now + Duration::from_millis(3), &mut out);

        assert_eq!(
            suppress_count_for(&out, a),
            1,
            "post-unstable-verify event re-emits Suppress(a)",
        );
        let burst = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            burst.suppressed_resources.contains(&a),
            "burst tracks the resource for the next drain",
        );
    }

    #[test]
    fn finish_burst_to_idle_drains_suppressed_resources_for_abnormal_end() {
        // A burst that reaches `finish_burst_to_idle` without passing
        // through `transition_to_verifying` (abnormal-end path:
        // `finalize_anchor_lost` / config reap mid-Batching) must
        // defensively drain `suppressed_resources` so the anchor's suppress
        // isn't the only release the watcher sees. Symmetric pairing
        // discipline holds even on the abnormal path.
        let (mut e, pid, root, a, b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        e.event_drives_batching(pid, a, now + Duration::from_millis(1), &mut out);
        e.event_drives_batching(pid, b, now + Duration::from_millis(2), &mut out);
        out.watch_ops.clear();

        e.finish_burst_to_idle(pid, &mut out);

        // Each tracked non-anchor resource gets one Unsuppress; the anchor
        // also gets one (from the existing start ↔ finish bracket).
        assert_eq!(unsuppress_count_for(&out, a), 1);
        assert_eq!(unsuppress_count_for(&out, b), 1);
        assert_eq!(unsuppress_count_for(&out, root), 1);
        assert_eq!(e.tree.get(a).unwrap().suppress_count(), 0);
        assert_eq!(e.tree.get(b).unwrap().suppress_count(), 0);
        assert_eq!(e.tree.get(root).unwrap().suppress_count(), 0);
    }

    #[test]
    fn transition_to_verifying_emits_unsuppress_in_resource_id_order() {
        // The drain iterates `BTreeSet<ResourceId>` in sorted order, so
        // emitted Unsuppress ops are in `ResourceId`-ascending order —
        // coherent with `StepOutput.watch_ops`'s sort discipline.
        let (mut e, pid, root, a, b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        // Insert b first, then a — drain order should still be ascending.
        e.event_drives_batching(pid, b, now + Duration::from_millis(1), &mut out);
        e.event_drives_batching(pid, a, now + Duration::from_millis(2), &mut out);
        out.watch_ops.clear();

        e.transition_to_verifying(pid, &mut out);

        let unsuppress_resources: Vec<_> = out
            .watch_ops
            .iter()
            .filter_map(|op| match op {
                WatchOp::Unsuppress { resource } => Some(*resource),
                _ => None,
            })
            .collect();
        let mut sorted = unsuppress_resources.clone();
        sorted.sort();
        assert_eq!(
            unsuppress_resources, sorted,
            "drain emits Unsuppress ops in ResourceId-ascending order",
        );
    }

    #[test]
    fn finish_burst_to_idle_with_empty_suppressed_resources_no_extra_unsuppress() {
        // After a normal `transition_to_verifying` drain,
        // `suppressed_resources` is empty. Reaching
        // `finish_burst_to_idle` from there (e.g., abnormal anchor loss
        // post-verify but pre-fire) must not double-Unsuppress the
        // already-drained resources.
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        let mut out = StepOutput::default();
        let now = Instant::now();
        e.start_standard_burst(pid, root, now, &mut out);
        e.event_drives_batching(pid, a, now + Duration::from_millis(1), &mut out);
        e.transition_to_verifying(pid, &mut out);
        // suppressed_resources is now empty; any abnormal end shouldn't
        // re-emit Unsuppress for `a`.
        out.watch_ops.clear();

        e.finish_burst_to_idle(pid, &mut out);

        assert_eq!(
            unsuppress_count_for(&out, a),
            0,
            "no double-Unsuppress for resource already drained at verify",
        );
        // The anchor's Unsuppress still emits (the existing start↔finish bracket).
        assert_eq!(unsuppress_count_for(&out, root), 1);
    }
}
