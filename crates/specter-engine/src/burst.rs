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
//!   it at a distance. The rebase-loop ceiling lifecycle
//!   (`rebase_ceiling`, `forced`) once lived in cat-b but folded back
//!   into cat-a once `transition_to_settling` opened a true post-fire
//!   settle-debounce window — the writes are co-located with their
//!   phase transitions on this side. (The old `apply_dirty_delta`
//!   counter edge was deleted with the `dirty_descendants` refcount —
//!   the `Draining → Verifying` reconfirm is now a fresh query, not a
//!   maintained count.)
//! - **(c) The sanctioned cross-crate emission reader**:
//!   [`Engine::emit_owner_probe`] (in `probe`) reads the pre-fire
//!   Standard burst's `dirty` to project its captured paths to the
//!   `Chains` obligation. A pure `&self` state→wire projection, not a
//!   writer and not a drain — `dirty` persists across re-batching. Its
//!   `pub` burst accessors are load-bearing and deliberately *not*
//!   sealed (Rust visibility is intra-crate; the choke reaches them
//!   from another crate).
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
//!   `retry_drives_batching` (`QuiescenceVerdict::Retry`) /
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
//! - `arm_rebase_loop_ceiling` (writes `post.rebase_ceiling = Some(t)`
//!   at the `Awaiting → Settling` natural entry) /
//!   `force_pending_post_fire` (writes `post.forced = true; post.rebase_ceiling
//!   = None;` on ceiling expiry or gate-deadline recovery) — post-fire
//!   single-field mutators, the symmetric mirror of pre-fire's
//!   `force_pending`. Each has exactly one production write site, both
//!   in this module; the grep `\.rebase_ceiling =` lands exactly two
//!   hits.
//! - `transition_to_settling` (Awaiting | Rebasing → Settling) /
//!   `transition_to_rebasing` (Settling → Rebasing or
//!   gate-deadline-recovery Awaiting → Rebasing) — post-fire phase
//!   swaps (mutate `PostFireBurst`), the symmetric mirror of pre-fire's
//!   `event_drives_batching` / `transition_to_verifying`.
//! - `reschedule_settling` (settle-timer re-point on absorbed events,
//!   phase class unchanged) — single-field `PostFireBurst.phase`
//!   mutator, the symmetric mirror of pre-fire's `reschedule_batching`.
//! - `absorb_event_into_fire_tail` — FsEvent during post-fire (notes
//!   into `PostFireBurst.final_window_residual` and stamps
//!   `last_event_time`, the symmetric mirror of pre-fire's
//!   `event_drives_batching` write).
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
//! - [`pre_fire_target`] — returns the live `ResourceId` the next
//!   pre-fire probe walks (and the response grafts at). Centralizes the
//!   `(anchor_kind, intent)` rule (File anchor → anchor; Seed → anchor;
//!   Standard → the live slot at the component-LCA of `dirty`'s
//!   captured paths, a File leaf promoted to its parent Dir, anchor on
//!   any resolution miss). Post-fire rebases target the anchor
//!   unconditionally and bypass this helper.
//!   `transition_to_verifying` resolves the target through it and writes
//!   it onto `pre.probe_target` for the choke to read back.
//! - [`Engine::emit_owner_probe`] (in `probe`) — the single
//!   owner-polymorphic emission choke. Each burst-launch helper is
//!   `mint → arm (loud) → emit_owner_probe(owner)`; the choke resolves
//!   the owner's state once, reads the correlation back off the armed
//!   slot, kind-dispatches, and materializes the per-carrier proof
//!   obligation as a pure `&self` read (the pre-fire Standard burst's
//!   `dirty` captured paths as `Chains`; `WholeSubtree` for Seed
//!   and the post-fire Rebase — no accumulator drain, the fire-tail
//!   residual reset is owned by `transition_to_rebasing`). Unclassified
//!   anchors take the Subtree arm — the walker returns `Vanished` on
//!   kind mismatch and the engine recovers via descent.

use crate::Engine;
use smallvec::SmallVec;
use specter_core::{
    ActiveBurst, BurstFinish, BurstHelper, BurstIntent, Diagnostic, DirtyProvenance, FsEvent,
    PostFirePhase, PreFireBurst, PreFirePhase, ProbeOwner, ProbeSlot, Profile, ProfileId,
    ProfileState, ReapTrigger, ResourceId, ResourceKind, StepOutput, TimerId, TimerKind, Tree,
    TreeSnapshot,
};
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
    /// Gates `arm_rebase_loop_ceiling` (from `Awaiting`),
    /// `force_pending_post_fire` (from `Awaiting` /
    /// `Settling` / `Rebasing` — via `handle_rebase_ceiling` and
    /// `handle_gate_deadline`'s non-zombie arm),
    /// `transition_to_settling` (entered from `Awaiting` via
    /// `on_effect_complete::LastReached`, or from `Rebasing` via
    /// `dispatch_rebase_ok::Retry`),
    /// `transition_to_rebasing` (entered from `Settling` via
    /// `handle_post_fire_settle_expired` / `handle_rebase_ceiling`'s
    /// drive-now arm, or from `Awaiting` via `handle_gate_deadline`'s
    /// non-zombie skip), and
    /// `absorb_event_into_fire_tail`. Callers further narrow to a
    /// specific `PostFirePhase` before invoking, but the gate stops at
    /// the variant level — narrower phase-level preconditions would
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

    /// Start a Seed burst. The `trigger` discriminator is **load-
    /// bearing for phase choice**: the cold-attach path
    /// (`trigger = None`) emits a probe immediately
    /// (`Verifying`-at-construction); the triggered path
    /// (`trigger = Some(_)`) opens in `Batching` exactly like
    /// [`Self::start_standard_burst`]. The emission choke stamps
    /// `ProofObligation::WholeSubtree` for both arms (no trusted prior;
    /// the whole subtree is unproven until freshly read).
    ///
    /// **`trigger` is the cold/triggered discriminator.** A Seed burst
    /// is the engine's baseline-establishment surface — every Profile
    /// that reaches a settled `baseline` does so through one. `None`
    /// and `Some` pick out two disjoint origin classes the engine
    /// distinguishes structurally:
    ///
    /// - **`None` ⇔ cold attach** (no driving `FsEvent`). The Seed
    ///   was decided by the engine — a fresh-attach, a descent
    ///   terminus, or an overflow re-seed — not in response to a
    ///   kernel signal. There is no triggering event to record into
    ///   `dirty`, so `dirty.is_empty()` holds at first verify, and
    ///   `seed_owes_first_fire` projects to `false` ⇒ the first
    ///   `Authoritative` response pins the baseline silently. There is
    ///   no event activity to debounce against either, so opening a
    ///   `Batching` phase would amortise nothing — the cold path arms
    ///   `Verifying` at burst construction and emits the cold walk
    ///   immediately. `last_event_time = None` is the first-class
    ///   construction state: no event drove this burst, no settle
    ///   deadline to source.
    /// - **`Some((resource, path))` ⇔ triggered re-Seed.** A driving
    ///   `FsEvent` reached an `Idle + !baseline_is_some()` Profile —
    ///   the post-recovery isolated change reached via the
    ///   `undischarged_consequence` ceiling terminal. The triggering
    ///   event threads into `dirty` so `seed_owes_first_fire` sees the
    ///   activity witness it owes a fire on. The burst opens in
    ///   `Batching { settle_timer }` exactly like `start_standard_burst`,
    ///   so further `FsEvent`s debounce identically; `last_event_time
    ///   = Some(now)` seeds the settle deadline.
    ///
    /// **`burst_deadline` armed in both arms.** The pre-fire bound
    /// holds across the two paths: a triggered Seed's settle / verify
    /// loop, or a cold Seed whose walk runs > `max_settle` (slow disk,
    /// NFS). On the cold path, a `BurstDeadline` expiry routes through
    /// `force_pending` (set `forced = true`) so the late `Authoritative`
    /// response dispatches as a forced fire — the same bounded-fallback
    /// path the triggered Seed reaches. No new ceiling primitive.
    ///
    /// **Intercept correctness (cold path).** An `FsEvent` arriving
    /// during the cold walk's round-trip routes through `drive_burst →
    /// event_drives_batching`, which Cancels the in-flight verify slot
    /// (the late walker response then fails `probe_gate` and drops as
    /// `StaleProbeResponse`), schedules a fresh `settle_timer`, writes
    /// `last_event_time = Some(now)`, notes the event into `dirty`, and
    /// re-enters `Batching`. No new code path: `Verifying → Batching`
    /// is already supported; the only novelty is that the *initial*
    /// phase was `Verifying` without a prior `Batching`. One walk is
    /// wasted on intercept — sub-ms on quiet trees (the common cold-
    /// attach case), bounded by the response-vs-event race window.
    /// The savings on the no-intercept path (cold attach over a quiet
    /// subtree drops from 2·settle to 0·settle, 2 walks to 1) make
    /// this the right trade.
    ///
    /// **Two named constructors, deliberately not one `intent` param.**
    /// The call site always knows whether it wants a Seed (anchor,
    /// `WholeSubtree`, no trusted prior) or a Standard (event-resource,
    /// LCA, `Chains`) burst; a merged constructor behind a runtime
    /// `intent` flag re-introduces exactly the dispatch flag the
    /// burst-helper doctrine rejects. The bodies differ in `intent`,
    /// the trigger discriminator above, and Standard's mandatory
    /// `event_resource`.
    ///
    /// **Callers.** Four cold sites (`None`); one triggered site
    /// (`Some`):
    /// - [`Self::bootstrap_immediate`] — fresh attach with an
    ///   already-materialised anchor on disk (cold).
    /// - `materialize_profile_anchor` (descent terminus reached via
    ///   `dispatch_descent_ok`) — the anchor just became live on disk
    ///   (cold).
    /// - `on_sensor_overflow` Idle path — reseed every Profile in
    ///   the overflow scope (cold).
    /// - `on_sensor_overflow` Active path — after
    ///   `finish_burst_to_idle` flushes the in-flight burst (cold).
    /// - `drive_burst`'s Idle + `!baseline_is_some()` branch — the
    ///   sole triggered call site.
    ///
    /// `EffectComplete::Ok` does NOT call this helper; post-Effect
    /// rebase routes through `transition_to_rebasing`.
    pub(crate) fn start_seed_burst(
        &mut self,
        profile_id: ProfileId,
        trigger: Option<(ResourceId, Arc<Path>)>,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_idle(profile_id, BurstHelper::StartSeedBurst, out) {
            return;
        }
        // Re-borrow for captures; the precondition has already confirmed
        // the Profile is live + Idle.
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let resource = p.resource();
        let settle = p.settle;
        let max_settle = p.max_settle();

        // `burst_deadline` arms unconditionally — the worst-case bound
        // on both the triggered settle/verify loop and the cold walk
        // (slow disk, NFS). On expiry, `force_pending` sets the burst's
        // `forced = true`, and any in-flight cold-walk response then
        // dispatches as a forced fire through the same path the
        // triggered Seed reaches.
        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        // The cold path's emission lives *after* state install (the
        // emission choke reads the correlation back off the armed
        // slot); this flag captures the path choice before the match
        // moves `trigger` into the Batching arm.
        let emit_cold_walk = trigger.is_none();

        let (phase, last_event_time, dirty) = match trigger {
            None => {
                // Cold attach. Mint a correlation, construct-arm the
                // `Verifying` slot inside the phase variant. I5 holds
                // by representability: no prior phase means no slot
                // could hold a competing correlation, and the loud
                // `ProbeSlot::arm` re-acquire assert is unreachable on
                // a fresh variant. `dirty` is empty by construction —
                // `seed_owes_first_fire` reads `!dirty.is_empty()` and
                // folds to `false`, routing the Authoritative response
                // to `SilentPin`. `last_event_time = None` — no event
                // drove this burst, no settle deadline to source.
                debug_assert!(
                    self.pending_probe_for(ProbeOwner::Profile(profile_id))
                        .is_none(),
                    "I5: cold-Seed start with a probe already in flight \
                     for profile {profile_id:?}"
                );
                let correlation = self.mint_probe_correlation();
                (
                    PreFirePhase::Verifying(ProbeSlot::armed(correlation, ())),
                    None,
                    DirtyProvenance::new(),
                )
            }
            Some((trigger_resource, trigger_path)) => {
                // Triggered re-Seed. Batching-first: the triggering
                // `FsEvent` instant seeds the settle deadline; `dirty`
                // notes the event as the activity witness for
                // `seed_owes_first_fire`. Shape mirrors
                // `start_standard_burst` exactly.
                let settle_timer =
                    self.timers
                        .schedule(now + settle, profile_id, TimerKind::Settle);
                let mut dirty = DirtyProvenance::new();
                dirty.note(trigger_resource, trigger_path);
                (PreFirePhase::Batching { settle_timer }, Some(now), dirty)
            }
        };

        self.profiles.transition_state(
            profile_id,
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    burst_deadline,
                    phase,
                    intent: BurstIntent::Seed,
                    forced: false,
                    dirty,
                    // Initial target = anchor, the value `pre_fire_target`
                    // returns for every Seed verify (cold path's emission
                    // choke resolves the same anchor at the wire-render).
                    // `transition_to_verifying` overwrites it with the
                    // same anchor on the triggered path's settle expiry.
                    probe_target: resource,
                    last_event_time,
                    last_certified_hash: None,
                }),
                // Fresh burst — directive starts at `ReturnToIdle`. Flips
                // to `Reap` only on mid-burst `mark_active_for_reap`.
                BurstFinish::ReturnToIdle,
            ),
        );

        // Cold-path emission AFTER state install. The choke
        // (`emit_owner_probe → probe_emission_request`) resolves the
        // owner's state, reads the correlation back off the armed
        // `Verifying` slot, materialises `ProofObligation::WholeSubtree`,
        // and ships the wire. Triggered Seeds do NOT emit here — the
        // settle expiry's `transition_to_verifying` is their emission
        // edge.
        if emit_cold_walk {
            self.emit_owner_probe(ProbeOwner::Profile(profile_id), out);
        }
    }

    /// Start a Standard burst: schedule settle + `burst_deadline`. No
    /// Probe — that fires on `settle_timer` expiry via
    /// `transition_to_verifying`. No ancestor bookkeeping: the
    /// `Draining → Verifying` reconfirm is a fresh query
    /// ([`crate::coverage::has_active_standard_descendant`]) over the
    /// live tree, so a burst start contributes nothing to maintain.
    ///
    /// `event_resource` + `event_path` are the `FsEvent`'s source slot
    /// and its path captured pre-dispatch. They seed `dirty`, whose
    /// captured paths are both the basis for the next probe's scope
    /// (their component-LCA, resolved to a live id) and the source of
    /// its `ProofObligation::Chains` (the chains the walker must freshly
    /// observe — refusing mtime-skip — so the response can certify
    /// quiescence). Sourcing both from the path makes a reaped trigger
    /// unable to collapse the obligation.
    pub(crate) fn start_standard_burst(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        event_path: &Arc<Path>,
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
        let resource = p.resource();
        let settle = p.settle;
        let max_settle = p.max_settle();

        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);
        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);

        let mut dirty = DirtyProvenance::new();
        dirty.note(event_resource, Arc::clone(event_path));

        self.profiles.transition_state(
            profile_id,
            ProfileState::Active(
                ActiveBurst::PreFire(PreFireBurst {
                    burst_deadline,
                    phase: PreFirePhase::Batching { settle_timer },
                    intent: BurstIntent::Standard,
                    forced: false,
                    dirty,
                    // Initial target = anchor. `transition_to_verifying`
                    // overwrites it with the live id at the captured
                    // paths' component-LCA on settle expiry / force-fire;
                    // the initial value carries no observable consequence
                    // (no probe has emitted yet).
                    probe_target: resource,
                    // The burst-start FsEvent IS the first event; seed the
                    // settle-deadline source of truth with `now`. Subsequent
                    // events update this in `event_drives_batching` without
                    // re-inserting a fresh heap entry.
                    last_event_time: Some(now),
                    last_certified_hash: None,
                }),
                // Fresh burst — directive starts at `ReturnToIdle`. Flips
                // to `Reap` only on mid-burst `mark_active_for_reap`.
                BurstFinish::ReturnToIdle,
            ),
        );
    }

    /// Caller: `drive_burst` Active branch — an `FsEvent` arrived during a
    /// burst. Cancels any in-flight verify (iff the prior phase was
    /// `Verifying`), notes `(event_resource, event_path)` into `dirty`
    /// (the captured-path basis for the next verify's probe scope and
    /// its `ProofObligation::Chains`), updates `last_event_time`, arms a
    /// fresh settle timer
    /// **only when re-entering Batching from Verifying or Draining**,
    /// and writes `phase = Batching { settle_timer }`. `intent`,
    /// `forced`, `burst_deadline`, and `last_certified_hash` (the carrier
    /// preserves the prior sample — a net-zero change across an event is
    /// still a valid hash-channel pair for the next Verifying response,
    /// active iff `!events_witness_quiescence`) are preserved.
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
        event_path: &Arc<Path>,
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
            pre.dirty.note(event_resource, Arc::clone(event_path));
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

    /// Sole caller: [`Engine::dispatch_quiescence_ok`]'s
    /// [`specter_core::QuiescenceVerdict::Retry`] arm — a verify just
    /// responded non-terminally (either the hash channel observed
    /// `prior != Some(response)`, or the walker refused on some chain
    /// with a transient non-observation — `EACCES`, a chmod-000 chain)
    /// and the burst-deadline ceiling has not yet fired, so the engine
    /// retries through a fresh settle window. The verify slot was
    /// already disarmed at the top of `on_profile_probe_response`; no
    /// Cancel needed. Arms a fresh settle timer and writes
    /// `phase = Batching { settle_timer }`.
    ///
    /// **`dirty` preserved; no re-commit.** The next verify re-targets
    /// and re-obligates per the carrier's own rule — a Standard burst
    /// the component-LCA of the preserved `dirty` captured paths +
    /// `ProofObligation::Chains` over those paths (the walker must
    /// freshly re-observe the dirty chains, refusing mtime-skip); a
    /// Seed burst the anchor + `ProofObligation::WholeSubtree` (every
    /// frame re-read, no skip). This path does **not** `apply_snapshot`
    /// — an unread region must never become `Profile.current` (the
    /// dedup / Seed baseline). The next verify either certifies
    /// authoritatively (fire-or-pin) or remains undischarged (re-enter
    /// this helper) until the `BurstDeadline` surfaces the terminal.
    ///
    /// **Reachability.** This helper runs *only* on the
    /// [`specter_core::QuiescenceVerdict::Retry`] dispatch arm; an
    /// `FsEvent` arriving during the verify routes through
    /// `event_drives_batching`, which Cancels and disarms the verify
    /// slot first. The `forced` (terminal) cases in the dispatcher
    /// bypass this helper — `Stable(Forced)` fires through, and
    /// `Abandon` surfaces the operator-visible
    /// `QuiescenceCeilingUnreadable` and finishes.
    ///
    /// **`last_event_time` pinned to `Some(now)`.** The verify just
    /// responded, so `now` is the timestamp of the latest observation
    /// that drove a transition on this burst. Pinning here removes the
    /// `Instant` monotonicity dependency from the on-expiry reschedule
    /// check: the freshly-scheduled settle timer fires at `now +
    /// settle`, and the expiry handler sees `expiry_now − now ≥
    /// settle` (true by construction of the scheduled deadline) and
    /// transitions cleanly — independent of any clock skew between
    /// this call and the prior `last_event_time`.
    pub(crate) fn retry_drives_batching(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::RetryDrivesBatching, out) {
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
    /// **Not `retry_drives_batching` minus the pin.** That
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
    /// anchor (regardless of phase); Standard bursts target the live
    /// slot at the component-LCA of `dirty`'s captured paths. The same
    /// rule covers the Draining → Verifying reconfirm: `dirty` is
    /// preserved across the burst's pre-fire lifetime (only `note`d
    /// into), so the component-LCA on the reconfirm matches the one at
    /// the original Verifying entry; a slot reaped in between only
    /// changes the live-id resolution (anchor fallback).
    ///
    /// **Emission.** This helper writes the `Verifying` phase + armed
    /// slot + `probe_target`, then calls [`Engine::emit_owner_probe`] —
    /// the single choke that reads the correlation back off the slot,
    /// materializes the proof obligation off the persisting burst
    /// (`ProofObligation::Chains` from `dirty`'s captured paths for
    /// Standard, `WholeSubtree` for Seed — read immutably, **not**
    /// drained: the burst outlives this probe across re-batching), and
    /// reads `forced`
    /// (so the walker bypasses mtime-skip on a force-fire). New events
    /// arriving during `Verifying` are noted into `dirty`
    /// (via `event_drives_batching`) and reshape the obligation on the
    /// next emission.
    pub(crate) fn transition_to_verifying(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_pre_fire(profile_id, BurstHelper::TransitionToVerifying, out) {
            return;
        }
        // Compute target under one immutable borrow window. `&self.tree`
        // and `&self.profiles.get(_)` are disjoint Engine-field borrows;
        // the call returns a `ResourceId` (`Copy`), so neither borrow
        // outlives this block. `pre_fire_target` is pure path math plus
        // a bounded anchor-rooted Tree descent — it cannot fail, only
        // fall back to the (always-live) anchor, so it threads no
        // diagnostic.
        let target = match self.profiles.get(profile_id) {
            Some(p) => match p.state() {
                ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
                    pre_fire_target(p, pre, &self.tree)
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
        // The proof obligation (`ProofObligation::Chains` from
        // `dirty`'s captured paths, or `WholeSubtree` for Seed) and
        // `forced` are materialized by `emit_owner_probe` (the single
        // probe-emission choke) off the armed `Verifying` slot it
        // resolves — the transition threads nothing. `dirty` is
        // preserved and read immutably by the choke (never drained):
        // its captured paths carry both the probe-scope basis and the
        // proof-obligation chains across the whole burst.

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
    /// `Profile.current` (committed by `fire_or_seal`'s `apply_snapshot`
    /// immediately before classification), so no `Arc<TreeSnapshot>` is
    /// duplicated on the phase variant.
    ///
    /// The sole caller (`gated_fire`, on the deferred fire branch) is
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
    /// transition: `fire_and_settle` calls this immediately after
    /// `emit_effects` returns a non-zero `EmitOutcome.count` — every
    /// fireable Seed/Standard consequence funnels through that one
    /// helper. The match is structural (count > 0) — callers know they
    /// pushed Effects.
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
        if self
            .profiles
            .get(profile_id)
            .is_some_and(|p| matches!(p.state(), ProfileState::Active(ActiveBurst::PreFire(_), _)))
            && let Some(prior) = self
                .profiles
                .transition_state(profile_id, ProfileState::Idle)
        {
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
                    self.profiles.transition_state(
                        profile_id,
                        ProfileState::Active(
                            ActiveBurst::PostFire(pre.into_post_fire(outstanding, gate_deadline)),
                            finish,
                        ),
                    );
                }
                other => {
                    self.profiles.transition_state(profile_id, other);
                }
            }
        }
    }

    /// Arm the post-fire rebase-loop ceiling at the natural `Awaiting
    /// → Settling` entry — the rebase loop's bound, scheduled once
    /// per loop.
    ///
    /// Schedules a [`TimerKind::RebaseCeiling`] timer at `now +
    /// max_settle` and writes
    /// `PostFireBurst.rebase_ceiling = Some(timer)` (`NotStarted →
    /// Armed`) inline. The sole writer of `(None, false) → (Some(t),
    /// false)`; the matching `(Some(t), false) → (None, true)` /
    /// `(None, false) → (None, true)` writes live on
    /// [`Self::force_pending_post_fire`]. The grep
    /// `\.rebase_ceiling =` lands exactly two production hits, here
    /// and there — both in this module.
    ///
    /// **Sole caller.** [`Engine::on_effect_complete`]'s `LastReached
    /// + ReturnToIdle` arm (the natural `Awaiting → Settling` entry,
    /// `outstanding` just hit zero). Invoked **before** calling
    /// [`Self::transition_to_settling`], so the ceiling is armed at
    /// the loop's *start* — its scope covers the whole
    /// `Settling ⇄ Rebasing` loop, not each sample.
    ///
    /// `handle_gate_deadline`'s non-zombie arm does NOT call this
    /// helper — gate-deadline-recovery has already waited 4× max_settle
    /// and the `forced` bit guarantees the next response commits
    /// unconditionally, so no loop bound is needed. It calls
    /// [`Self::force_pending_post_fire`] + [`Self::transition_to_rebasing`]
    /// directly (skip ceiling + Settling), the symmetric mirror of
    /// pre-fire's `handle_burst_deadline → force_pending → drive
    /// Verifying now`.
    ///
    /// **Gate-free by design.** The call site has verified
    /// `Active(PostFire(Awaiting))` before reaching the helper; the
    /// `if let` is the stale-id tolerance, mirroring
    /// [`Self::force_pending`]'s shape on the pre-fire side. The
    /// `debug_assert!` pins the arm-once contract: an
    /// `Armed`/`Reached` find here is a future caller misroute (the
    /// sole caller runs once per loop, at the natural entry), not a
    /// runtime race.
    pub(crate) fn arm_rebase_loop_ceiling(&mut self, profile_id: ProfileId, now: Instant) {
        let Some(max_settle) = self.profiles.get(profile_id).map(Profile::max_settle) else {
            return;
        };
        let timer = self
            .timers
            .schedule(now + max_settle, profile_id, TimerKind::RebaseCeiling);
        if let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        {
            debug_assert!(
                post.rebase_ceiling.is_none() && !post.forced,
                "arm_rebase_loop_ceiling: ceiling already armed/reached \
                 (profile = {profile_id:?})",
            );
            post.rebase_ceiling = Some(timer);
        }
    }

    /// Latch the rebase-loop terminal — sets `post.forced = true` and
    /// drops `post.rebase_ceiling = None` in lockstep. The post-fire
    /// mirror of [`Self::force_pending`] on the pre-fire side; the
    /// single-source mutator of `PostFireBurst.forced` and the
    /// `Some(t) → None` / `None → None` writes of `rebase_ceiling`.
    ///
    /// Once raised, the next probe emission bypasses the walker's
    /// coarse-mtime skip (`forced` projects to the walker's
    /// obligation), the in-flight response folds through
    /// [`specter_core::quiescence_verdict`] with `forced = true`, and
    /// `dispatch_rebase_ok` reads
    /// `Stable(StableReason::Forced { hash_channel_disagreed })`
    /// (commit + diagnose if the channel disagreed; commit silent
    /// otherwise) or [`specter_core::QuiescenceVerdict::Abandon`]
    /// (abandon + diagnose) off the verdict.
    ///
    /// **Lockstep with `rebase_ceiling`.** Sets `forced = true` AND
    /// drops the timer reference `rebase_ceiling = None` in one move
    /// — the invariant that `(rebase_ceiling = Some, forced = true)`
    /// is unreachable (a stale RebaseCeiling-armed-but-forced entry
    /// would re-fire `handle_rebase_ceiling` and double-latch). The
    /// drop is safe whether the timer was consumed by `pop_expired`
    /// (the natural ceiling-expiry path) or was never armed at all
    /// (the gate-deadline-recovery path, which raises `forced` without
    /// an in-heap ceiling entry).
    ///
    /// **Field write only — the phase decision stays with the
    /// caller.** [`Engine::handle_rebase_ceiling`] re-reads the phase
    /// after this call to decide whether to drive a Rebasing verify
    /// *now* (Settling — no probe in flight) or wait (Rebasing — a
    /// probe is already in flight and will dispatch with `forced`
    /// observed). [`Engine::handle_gate_deadline`]'s non-zombie arm
    /// always drives `transition_to_rebasing` after this call (the
    /// Settling path is skipped — we already waited 4× max_settle).
    ///
    /// **Gate-free by design.** Same rationale as
    /// [`Self::force_pending`]: the callers are reached only through
    /// `is_timer_referenced`, which returns false for `RebaseCeiling`
    /// in non-`PostFire` phases (and the gate-deadline-recovery caller
    /// has its own `Active(PostFire(Awaiting))` precondition), so only
    /// post-fire phases arrive. The non-post-fire arm silently no-ops.
    pub(crate) fn force_pending_post_fire(&mut self, profile_id: ProfileId) {
        if let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        {
            post.forced = true;
            post.rebase_ceiling = None;
        }
    }

    /// `Settling → Rebasing` (the natural settle-expiry advance and
    /// the ceiling-driven force) or `Awaiting → Rebasing` (the
    /// gate-deadline-recovery skip). The post-fire baseline-capture
    /// probe's emission edge — single-purpose: mint correlation, clear
    /// residual, write phase, emit probe.
    ///
    /// **Ceiling arming lives elsewhere.** The rebase-loop ceiling is
    /// armed at the natural `Awaiting → Settling` entry by
    /// [`Engine::arm_rebase_loop_ceiling`], so by the time we reach
    /// this helper the ceiling is already in place. The
    /// gate-deadline-recovery caller
    /// ([`Engine::handle_gate_deadline`]'s non-zombie arm) explicitly
    /// skips the ceiling — it calls [`Self::force_pending_post_fire`]
    /// then this helper directly; the `forced` bit drives the next
    /// response to a commit terminal in one walk without needing a
    /// loop bound (the actuator has already hung for 4× max_settle).
    ///
    /// **Probe slot.** A fresh correlation is minted and the
    /// `Rebasing` phase is written already armed with it, in one move
    /// — the slot *is* the phase. I5 holds by representability: the
    /// prior phase is `Settling` or `Awaiting`, neither of which holds
    /// a probe slot (`Settling`'s correlation token is its
    /// `settle_timer`; `Awaiting`'s is its `gate_deadline`). The probe
    /// the loop's previous Rebasing entry emitted was disarmed at
    /// its response's `on_profile_probe_response`, before the loop-
    /// back routed through `transition_to_settling`. `emit_owner_probe`
    /// resolves the target (the anchor — `PostFireBurst` carries no
    /// `probe_target`) and reads the correlation back off the slot.
    ///
    /// **`baseline_subtree` is shipped but not skip-trusted.** The
    /// probe ships `Profile.current` as `baseline_subtree`, but its
    /// obligation is `WholeSubtree` (the command just mutated the tree
    /// — no trustworthy prior), so the walker re-reads the whole
    /// anchor subtree regardless of mtime and the response certifies
    /// the *post-command* tree. An idempotent command still pays the
    /// walk; that cost is the price of soundness (an in-place
    /// descendant edit need not bump an ancestor mtime, so a
    /// chains/mtime skip would re-clone a stale subtree and certify a
    /// false quiet).
    ///
    /// **Fire-tail residual reset, every entry.**
    /// `final_window_residual` (the events
    /// `absorb_event_into_fire_tail` captured) is cleared at *every*
    /// entry. Under `WholeSubtree` it is no longer an obligation
    /// source — the walk observes the tree regardless — only the
    /// final-window restart seed: clearing per entry means an
    /// `Authoritative` terminal sees only events from the *final*
    /// probe round-trip and restarts solely for that genuine race,
    /// not for every tree-touching command. Earlier-round absorbs
    /// are not lost; the next `WholeSubtree` read folded them into
    /// the verdict.
    ///
    /// **Non-Active early return.** Every caller has verified
    /// `Active(PostFire)` (phase `Settling` for the natural and
    /// ceiling-driven entries, `Awaiting` for the
    /// gate-deadline-recovery skip) before reaching here. Defensively
    /// early-returning on non-Active matches `transition_to_verifying`'s
    /// strict policy and avoids the latent bug where a stray call
    /// mints a correlation and emits a Probe op while failing to
    /// write the phase — orphaning the correlation, whose late
    /// response would stale-detect against an unarmed state.
    pub(crate) fn transition_to_rebasing(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if !self.require_active_post_fire(profile_id, BurstHelper::TransitionToRebasing, out) {
            return;
        }
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
        // Reset the fire-tail residual at every Rebasing entry: under
        // `WholeSubtree` it is only the final-window restart seed, not
        // an obligation source, so clearing here keeps a `Stable`
        // terminal from spuriously restarting on every tree-touching
        // command. Earlier-round absorbs are not lost — the
        // `WholeSubtree` walk observes them regardless.
        post.final_window_residual.clear();
        post.phase = PostFirePhase::Rebasing(ProbeSlot::armed(correlation, ()));

        // The choke reads the correlation back off the `Rebasing` slot,
        // targets the anchor (`forced` is pre-fire-only ⇒ `false`), and
        // ships the `WholeSubtree` obligation — no accumulator drain.
        self.emit_owner_probe(owner, out);
    }

    /// `Awaiting | Rebasing → Settling`. The post-fire settle-debounce
    /// entry — the symmetric mirror of pre-fire's `event_drives_batching`
    /// / `retry_drives_batching` pair on the Settling side
    /// (post-fire collapses the two pre-fire callers' work to one
    /// helper, since there is no Cancel-vs-no-Cancel split: the prior
    /// phase carries no probe slot in either arm).
    ///
    /// **Two prior phases.** The helper is reached from:
    ///
    /// 1. `Awaiting → Settling` — [`Engine::on_effect_complete`]'s
    ///    `LastReached + ReturnToIdle` arm (the natural rebase entry,
    ///    `outstanding` just hit zero). The caller invokes
    ///    [`Self::arm_rebase_loop_ceiling`] **before** this helper to
    ///    arm the loop's ceiling at its start; `last_event_time =
    ///    Some(now)` is the EffectComplete instant — the Settling
    ///    window reckons from there.
    /// 2. `Rebasing → Settling` — `dispatch_rebase_ok::Retry` (the
    ///    only surviving post-fire loop-back arm). The ceiling was
    ///    armed at the loop's start (1); no re-arm here.
    ///    `last_event_time = Some(now)` is the response instant — the
    ///    next Settling window reckons from the unfavorable response,
    ///    the same conservative anchor pre-fire's
    ///    `retry_drives_batching` applies on `last_event_time =
    ///    Some(now)` after a verify response.
    ///
    /// Both arms write `phase = Settling { settle_timer }`, pin
    /// `last_event_time = Some(now)`, and arm the fresh
    /// [`TimerKind::PostFireSettle`] timer. Ceiling arming is the
    /// caller's responsibility — this helper is single-purpose.
    ///
    /// **Prior phase carries no probe slot in either arm.** `Awaiting`
    /// has no slot at all (its correlation token is `gate_deadline`).
    /// `Rebasing`'s slot was disarmed by `take_owner_probe` at the
    /// `on_profile_probe_response` entry, before `dispatch_rebase_ok`
    /// ran. So the phase overwrite below drops either a `(outstanding,
    /// gate_deadline)` payload (Awaiting) or an *empty* `ProbeSlot`
    /// (Rebasing) — no linearity tripwire either way. The
    /// `debug_assert!` pins the accepted prior-phase set.
    ///
    /// **`last_event_time` pinned to `Some(now)`.** Same rationale as
    /// pre-fire's `retry_drives_batching` pin: removes the
    /// `Instant` monotonicity dependency from
    /// [`Engine::handle_post_fire_settle_expired`]'s reschedule check.
    /// The freshly-scheduled `settle_timer` fires at `now + settle`,
    /// and the expiry handler sees `expiry_now − now ≥ settle` (true
    /// by construction) and transitions cleanly — independent of any
    /// clock skew between this call and any prior `last_event_time`.
    pub(crate) fn transition_to_settling(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !self.require_active_post_fire(profile_id, BurstHelper::TransitionToSettling, out) {
            return;
        }
        let Some(settle) = self.profiles.get(profile_id).map(|p| p.settle) else {
            return;
        };
        let settle_timer =
            self.timers
                .schedule(now + settle, profile_id, TimerKind::PostFireSettle);

        if let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        {
            debug_assert!(
                matches!(
                    post.phase,
                    PostFirePhase::Awaiting { .. } | PostFirePhase::Rebasing(_),
                ),
                "transition_to_settling off a non-Awaiting/non-Rebasing phase \
                 (profile = {profile_id:?})",
            );
            post.phase = PostFirePhase::Settling { settle_timer };
            post.last_event_time = Some(now);
        }
    }

    /// `Settling → Settling` settle-timer re-point. The single-source
    /// `PostFireBurst.phase` mutator for the settle-expiry reschedule,
    /// the post-fire mirror of [`Self::reschedule_batching`].
    ///
    /// When an `FsEvent` was absorbed by
    /// [`Self::absorb_event_into_fire_tail`] after the live
    /// `settle_timer` was scheduled but before it fires,
    /// `handle_post_fire_settle_expired` schedules a fresh
    /// `PostFireSettle` timer at `last_event_time + settle` and routes
    /// the resulting `TimerId` here. The phase *class* is unchanged —
    /// only the timer correlation moves.
    ///
    /// **Not `transition_to_settling` minus the schedule.** That
    /// helper enters Settling from `Awaiting` (natural) or `Rebasing`
    /// (undischarged loop-back) and pins `last_event_time = now`.
    /// This path is *already* `Settling` and must **not** touch
    /// `last_event_time`: pinning it would push the very deadline the
    /// caller's just-made quiet-window decision is rescheduling
    /// toward, defeating the check that chose to reschedule.
    ///
    /// **Timer math stays with the caller.**
    /// `handle_post_fire_settle_expired` owns the `now −
    /// last_event_time < settle` quiet-window decision and the
    /// `TimerHeap::schedule` call; this helper owns only the phase
    /// write, keeping it a pure single-field mutator symmetric with
    /// [`Self::reschedule_batching`].
    ///
    /// **Gate-free by design.** Like [`Self::reschedule_batching`]
    /// this helper carries no `require_active_post_fire` precondition:
    /// its sole caller has already validated
    /// `Active(PostFire(Settling))` via `is_timer_referenced` plus its
    /// own defensive phase check, so the non-post-fire arm is
    /// structurally unreachable and a routing diagnostic there would
    /// be spurious. The `debug_assert!` is the loud dev/CI backstop
    /// that the contract held.
    pub(crate) fn reschedule_settling(&mut self, profile_id: ProfileId, settle_timer: TimerId) {
        if let Some(post) = self
            .profiles
            .get_mut(profile_id)
            .and_then(Profile::post_fire_burst_mut)
        {
            debug_assert!(
                matches!(post.phase, PostFirePhase::Settling { .. }),
                "reschedule_settling off a non-Settling phase (profile = {profile_id:?})",
            );
            post.phase = PostFirePhase::Settling { settle_timer };
        }
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
    /// permanent strand. The reconfirm verify's verdict folds through
    /// [`specter_core::quiescence_verdict`] against the fresh response,
    /// independent of the splice-mutated `Profile.current` — the verdict
    /// floor reads `(ProofAuthority, forced)`, not the per-Profile
    /// current snapshot. Same-step ordering means the `StepOutput`
    /// reflects the cascade: child's burst end → parent reconfirm Probe
    /// in one `step` call.
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
        let Some(prior) = self
            .profiles
            .transition_state(profile_id, ProfileState::Idle)
        else {
            return;
        };
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
                self.profiles.transition_state(profile_id, other);
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
        // Probe; the response routes via the burst's preserved `intent`
        // (Seed or Standard) through `dispatch_quiescence_ok`, comparing
        // against the Profile's `current` (set when it entered
        // Draining).
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

    /// Absorb a post-fire FsEvent — the self-trigger guard. The
    /// post-fire phases (`Awaiting | Rebasing | Settling`) must not
    /// start a fresh burst: the command the burst just fired writes
    /// to the watched tree, and every such write would otherwise drive
    /// its own burst (the self-trigger loop). The event is deferred
    /// into `PostFireBurst.final_window_residual` (the restart seed)
    /// and the absorb timestamp is recorded on
    /// `PostFireBurst.last_event_time` — the symmetric mirror of
    /// pre-fire's `event_drives_batching` write to the same-named
    /// field.
    ///
    /// The rebase loop's soundness does **not** depend on the residual
    /// set: the rebase probe walks `WholeSubtree`, so every absorbed
    /// event is re-observed by the next sample and folded into the
    /// quiescence verdict whether or not it is recorded here — the
    /// loop exits `Authoritative` only once a full read certifies on
    /// an event-quiet (`forced = false`) path. The residual survives
    /// only as the final-window restart seed (reset at every
    /// `Rebasing` entry by `transition_to_rebasing`); it is the POSIX
    /// content-edit hole's closure for the *restart* decision, not
    /// the walk.
    ///
    /// **`last_event_time` is the settle-debounce deadline source of
    /// truth.** The post-fire `Settling` window's quiet-check reckons
    /// from this field: `handle_post_fire_settle_expired` reads it on
    /// expiry to decide reschedule (events arrived since the timer
    /// was scheduled) vs transition to `Rebasing` (quiet for ≥
    /// settle). The same write contract pre-fire's
    /// `event_drives_batching` follows on the Batching side — the two
    /// halves of the mirror are now structurally symmetric.
    ///
    /// `event` is threaded purely for the diagnostic so an operator can
    /// correlate logs to the deferred FsEvent.
    pub(crate) fn absorb_event_into_fire_tail(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        event_path: &Arc<Path>,
        event: FsEvent,
        now: Instant,
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
            post.final_window_residual
                .note(event_resource, Arc::clone(event_path));
            post.last_event_time = Some(now);
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
                p.resource(),
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
        if self
            .profiles
            .get(profile_id)
            .is_some_and(|p| matches!(p.state(), ProfileState::Active(ActiveBurst::PostFire(_), _)))
            && let Some(prior) = self
                .profiles
                .transition_state(profile_id, ProfileState::Idle)
        {
            match prior {
                ProfileState::Active(ActiveBurst::PostFire(post), finish) => {
                    // Carry `finish` across the restart. It is
                    // `ReturnToIdle` by the caller's gate; preserving it
                    // (rather than hard-writing) keeps a mid-tail
                    // `mark_active_for_reap` honoured at the restarted
                    // burst's end.
                    self.profiles.transition_state(
                        profile_id,
                        ProfileState::Active(
                            ActiveBurst::PreFire(post.into_pre_fire_residual(
                                burst_deadline,
                                settle_timer,
                                resource,
                                now,
                            )),
                            finish,
                        ),
                    );
                }
                other => {
                    self.profiles.transition_state(profile_id, other);
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

/// Resolve a path to the live engine slot that should root the Standard
/// pre-fire probe — and, read back, the response graft — at it.
/// Descends the live `Tree` from the always-live `anchor` by `path`'s
/// anchor-relative components (`Tree::lookup` per segment).
///
/// Any miss — `path` not under the anchor, a non-UTF-8 component, or a
/// reaped-not-recreated intermediate — falls back to the anchor. The
/// fallback is a strictly *wider* root that can never clip a chain:
/// `coverage::covers` routes events only at-or-under the anchor, so the
/// anchor is an ancestor-or-equal of every captured path. The result is
/// always a live `ResourceId` (the anchor at minimum); the caller
/// promotes a non-Dir result to its parent Dir.
fn resolve_under_anchor(anchor: ResourceId, path: &Path, tree: &Tree) -> ResourceId {
    let Some(anchor_path) = tree.get(anchor).map(|r| Arc::clone(r.path())) else {
        return anchor;
    };
    let Ok(rel) = path.strip_prefix(&anchor_path) else {
        return anchor;
    };
    let mut cur = anchor;
    for comp in rel.components() {
        let Some(seg) = comp.as_os_str().to_str() else {
            return anchor;
        };
        match tree.lookup(Some(cur), seg) {
            Some(next) => cur = next,
            None => return anchor,
        }
    }
    cur
}

/// Promote a non-Dir resolved slot to its parent Dir; the
/// descendant-observation model probes Dirs (a File is a child entry of
/// its parent). Walks `start → parent` until a `Dir`, falling back to
/// `anchor` if the chain crosses a reaped slot or runs out of ancestors.
/// Unprobed slots (`kind() == None`) walk up like File-shape — we don't
/// know what they are, the parent is the safer probe target.
///
/// Pairs with [`resolve_under_anchor`]: the component-LCA of ≥2 distinct
/// captured paths is structurally a Dir (it has descendants among the
/// captured set) and is returned as-is; only a lone-location File scope
/// promotes one level to its containing Dir, mirroring the pre-path-LCA
/// behaviour exactly.
///
/// **Pre-condition.** The caller has filtered out File-anchored Profiles
/// (`pre_fire_target` returns the anchor for a File kind), so a Dir
/// anchor is assumed and the walk always terminates at-or-above it.
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

/// Pre-fire probe target for the next emission — the live engine slot
/// the probe walks and the response grafts at.
///
/// Centralizes the `(anchor_kind, intent)` rule. The three production
/// scenarios (settle-expired Standard burst, force-fire under
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
/// - Standard intent (Dir / unclassified anchor) → the live slot at the
///   component-LCA of `dirty`'s captured paths
///   ([`resolve_under_anchor`]), a File leaf promoted to its parent Dir
///   ([`promote_to_dir`]). The LCA is computed over *captured paths*
///   (history), not surviving slot ids, so a slot reaped mid-burst
///   cannot collapse the scope below where an event landed; only the
///   live-id *resolution* may fall back to the anchor (strictly wider,
///   never chain-clipping). An empty `dirty` yields the anchor — a
///   should-never (a Standard burst always notes its trigger); the
///   emission choke pairs that anchor with a `WholeSubtree` obligation
///   under its own `debug_assert`, so the degrade proves the whole
///   subtree rather than silently skipping it.
///
/// **Draining-reconfirm coverage.** The Draining → Verifying reconfirm
/// folds into the Standard case because `dirty` is preserved across the
/// burst's whole pre-fire lifetime (only `note`d into), so the
/// component-LCA on the reconfirm equals the one at the initial
/// Verifying entry. A slot reaped during Draining changes only the
/// live-id resolution (anchor fallback), never the captured-path basis.
pub(crate) fn pre_fire_target(p: &Profile, pre: &PreFireBurst, tree: &Tree) -> ResourceId {
    match (p.kind(), pre.intent) {
        (Some(ResourceKind::File), _) | (_, BurstIntent::Seed) => p.resource(),
        _ => match pre.dirty.lca_path() {
            Some(lca) => promote_to_dir(
                resolve_under_anchor(p.resource(), &lca, tree),
                p.resource(),
                tree,
            ),
            None => p.resource(),
        },
    }
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
        ProbeSlot, Profile, ProfileIdentity, ProfileState, ProofObligation, ResourceKind,
        ResourceRole, ScanConfig, StepOutput, TimerKind,
    };
    use std::time::{Duration, Instant};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Build an Engine with a single Profile anchored at `/anchor`. Returns the
    /// Engine + the `ProfileId`.
    fn engine_with_profile() -> (Engine, specter_core::ProfileId) {
        let mut e = Engine::new();
        let r = e.tree.ensure_root("/anchor", ResourceRole::User);
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

    /// The live resource's materialised path — what the sensor reports as
    /// an `FsEvent`'s `event_path` and what anchor-rooted resolution
    /// re-descends. Panics on a stale id (a fixture bug).
    fn rpath(e: &Engine, id: ResourceId) -> Arc<Path> {
        Arc::clone(e.tree.get(id).expect("live resource").path())
    }

    #[test]
    fn start_seed_burst_cold_arm_verifying_and_emits_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, None, Instant::now(), &mut out);

        // Cold-arm contract: Verifying at construction, ProbeSlot armed.
        // Empty dirty (no driving event), None last_event_time (no
        // settle deadline to source on the cold path).
        let p = e.profiles.get(pid).unwrap();
        let burst = match p.state() {
            ProfileState::Active(ActiveBurst::PreFire(b), _) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Seed);
        assert!(matches!(burst.phase, PreFirePhase::Verifying(_)));
        assert!(!burst.forced);
        assert!(burst.dirty.is_empty(), "cold-Seed dirty starts empty");
        assert!(
            burst.last_event_time.is_none(),
            "cold-Seed has no driving event",
        );

        // Heap holds ONLY burst_deadline (no settle_timer on the cold
        // path — Verifying-at-construction skips Batching).
        assert_eq!(e.timers.len(), 1);

        // Probe emitted at construction.
        assert_eq!(out.probe_ops().len(), 1);
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn start_standard_burst_schedules_two_timers_no_probe() {
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        e.start_standard_burst(pid, r, &rp, Instant::now(), &mut out);

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
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        e.start_standard_burst(pid, r, &rp, Instant::now(), &mut out);
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
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        e.start_standard_burst(pid, r, &rp, Instant::now(), &mut out);
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
        // with a fresh settle_timer; intent preserved. Cold-arm Seed lands
        // directly in Verifying (probe armed at construction) — no need
        // to drive Batching → Verifying first.
        let (mut e, pid) = engine_with_profile();
        let mut out = StepOutput::default();
        e.start_seed_burst(pid, None, Instant::now(), &mut out); // cold Seed → Verifying
        let mut out = StepOutput::default();
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);

        e.event_drives_batching(pid, r, &rp, Instant::now(), &mut out);

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
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        e.start_standard_burst(pid, r, &rp, Instant::now(), &mut out);
        let mut out = StepOutput::default();

        e.event_drives_batching(pid, r, &rp, Instant::now(), &mut out);

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
        let resource = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, resource);
        let now = Instant::now();
        e.start_standard_burst(pid, resource, &rp, now, &mut out);
        e.transition_to_verifying(pid, &mut out);
        let mut out = StepOutput::default();
        // Production reaches `retry_drives_batching`
        // only from `on_profile_probe_response`, which has already
        // disarmed the Verifying slot via `take_owner_probe`. Mirror
        // that consume.
        let _ = e.take_owner_probe(ProbeOwner::Profile(pid));

        e.retry_drives_batching(pid, now, &mut out);

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
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        e.start_standard_burst(pid, r, &rp, Instant::now(), &mut out);
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
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        e.start_standard_burst(pid, r, &rp, Instant::now(), &mut out);
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
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);
        let t0 = Instant::now();
        e.start_standard_burst(pid, r, &rp, t0, &mut out);

        // Fire ten FsEvents at 50 ms intervals during the Standard burst.
        let mut last_event = t0;
        for k in 1..=10 {
            last_event = t0 + Duration::from_millis(50 * k);
            let mut out = StepOutput::default();
            e.event_drives_batching(pid, r, &rp, last_event, &mut out);
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
        e.start_seed_burst(pid, None, Instant::now(), &mut out);
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
        e.start_seed_burst(pid, None, Instant::now(), &mut out);
        let burst_deadline_initial = match e.profiles.get(pid).unwrap().state() {
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.burst_deadline,
            _ => panic!("expected Active(PreFire)"),
        };
        let r = e.profiles.get(pid).unwrap().resource();
        let rp = rpath(&e, r);

        e.event_drives_batching(pid, r, &rp, Instant::now(), &mut out);
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
        e.start_seed_burst(pid, None, Instant::now(), &mut out);
        // Cold-arm: Verifying-at-construction. Production reaches
        // `transition_to_draining` only from a Verifying probe response,
        // which has already disarmed the slot via `take_owner_probe`.
        // Mirror that consume here.
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
        e.start_seed_burst(pid, None, Instant::now(), &mut out);
        // Drop the first burst's emissions; only the second call is under test.
        let mut out = StepOutput::default();

        e.start_seed_burst(pid, None, Instant::now(), &mut out);

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
    // pre_fire_target — the (anchor_kind, intent) probe-target rule.
    //
    // Standard target = the live slot at the component-LCA of `dirty`'s
    // captured *paths* (history, not surviving slot ids), descended from
    // the always-live anchor, a File leaf promoted to its parent Dir,
    // anchor fallback on any resolution miss. Locks the contract
    // independent of `transition_to_verifying`'s body.
    // ---------------------------------------------------------------------------

    use crate::burst::pre_fire_target;
    use specter_core::testkit::dirty_provenance;
    use specter_core::{DirtyProvenance, PreFireBurst, ResourceId, TimerId};
    use std::path::Path;
    use std::sync::Arc;

    /// Build a tree-shaped Engine: anchor `/root`, two children `a` and `b`.
    fn engine_with_two_children() -> (
        Engine,
        specter_core::ProfileId,
        specter_core::ResourceId,
        specter_core::ResourceId,
        specter_core::ResourceId,
    ) {
        let mut e = Engine::new();
        let root = e.tree.ensure_root("/root", ResourceRole::User);
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

    /// Build a `PreFireBurst` shell for direct `pre_fire_target` calls.
    /// `dirty` is the only field the helper reads (besides `intent`); the
    /// rest are stub values the helper never inspects.
    fn pre_fire_burst_for_test(intent: BurstIntent, dirty: DirtyProvenance) -> PreFireBurst {
        PreFireBurst {
            burst_deadline: TimerId::default(),
            phase: PreFirePhase::Verifying(ProbeSlot::empty()),
            intent,
            forced: false,
            dirty,
            probe_target: ResourceId::default(),
            last_event_time: None,
            last_certified_hash: None,
        }
    }

    /// `dirty` carrying each resource id paired with its *real* tree path,
    /// so anchor-rooted resolution genuinely re-descends. The path strings
    /// match `engine_with_two_children`'s absolute `/root` scaffold.
    fn dirty_at(entries: &[(ResourceId, &str)]) -> DirtyProvenance {
        dirty_provenance(entries)
    }

    #[test]
    fn pre_fire_target_standard_empty_dirty_falls_back_to_anchor() {
        // Standard intent, no captured paths: the should-never degrade
        // resolves to the anchor (the emission choke pairs it with a
        // WholeSubtree obligation under its own debug_assert). Also
        // covers the Draining-reconfirm hypothetical where every dirty
        // Resource was reaped between verify and reconfirm.
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, DirtyProvenance::new());
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn pre_fire_target_standard_single_dirty_at_anchor_returns_anchor() {
        // One captured path equal to the anchor: the lone-path LCA is the
        // anchor itself, resolves back to the anchor slot.
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, dirty_at(&[(root, "/root")]));
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn pre_fire_target_standard_single_dirty_deep_returns_self() {
        // One captured path at a deeper Dir: the lone-path LCA is that
        // path; it resolves to that live slot (a Dir, no promotion).
        let (e, pid, _root, a, _b) = engine_with_two_children();
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, dirty_at(&[(a, "/root/a")]));
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(target, a);
    }

    #[test]
    fn pre_fire_target_standard_lone_file_promotes_to_parent_dir() {
        // The dominant single-file-edit case: the lone captured path is
        // a File leaf. Its component-LCA is the file itself; resolution
        // finds the live File slot, which promotes one level to its
        // parent Dir. The descendant-observation model probes Dirs — a
        // File-target probe is misread downstream as anchor removal, so
        // the promotion is load-bearing, not cosmetic.
        let mut e = Engine::new();
        let root = e.tree.ensure_root("/root", ResourceRole::User);
        e.tree.set_kind(root, ResourceKind::Dir);
        let f = e
            .tree
            .ensure_child(root, "f", ResourceRole::User)
            .expect("test live parent");
        e.tree.set_kind(f, ResourceKind::File);
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
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, dirty_at(&[(f, "/root/f")]));
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(
            target, root,
            "a lone File leaf promotes to its parent Dir, not the File slot",
        );
    }

    #[test]
    fn pre_fire_target_standard_two_siblings_resolve_to_parent() {
        // Two captured sibling paths: their component-LCA is the parent
        // (`/root` here), which resolves back to the anchor slot.
        let (e, pid, root, a, b) = engine_with_two_children();
        let pre = pre_fire_burst_for_test(
            BurstIntent::Standard,
            dirty_at(&[(a, "/root/a"), (b, "/root/b")]),
        );
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn pre_fire_target_standard_resolves_to_shared_intermediate_ancestor() {
        // Two leaves under disjoint mid-3 branches share a depth-2
        // ancestor (`l2`). The component-LCA of their captured paths is
        // `/l0/l1/l2`; it resolves to that live slot — not collapsing to
        // the anchor and not returning either leaf.
        let mut e = Engine::new();
        let l0 = e.tree.ensure_root("/l0", ResourceRole::User);
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

        let pre = pre_fire_burst_for_test(
            BurstIntent::Standard,
            dirty_at(&[(leaf_a, "/l0/l1/l2/a/x"), (leaf_b, "/l0/l1/l2/b/y")]),
        );
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(
            target, l2,
            "component-LCA of leaves under l3a and l3b is l2",
        );
    }

    #[test]
    fn pre_fire_target_standard_reaped_slot_falls_back_to_anchor() {
        // The captured path of a since-reaped slot can't resolve a live
        // id under the anchor, so resolution falls back to the anchor —
        // strictly wider, never chain-clipping — and silently (no
        // diagnostic; delete-recreate churn would flood logs).
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        let dirty = dirty_at(&[(a, "/root/a")]);
        e.tree.try_reap(a, &mut StepOutput::default());
        let pre = pre_fire_burst_for_test(BurstIntent::Standard, dirty);
        let target = pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree);
        assert_eq!(
            target, root,
            "a reaped dirty slot widens to the anchor, not below the event",
        );
    }

    #[test]
    fn pre_fire_target_file_anchor_returns_anchor() {
        // File-anchored Profile + any intent + any dirty set: target is the
        // anchor itself. kqueue per-file FDs surface events at the file
        // directly; promoting past the anchor would route the probe outside
        // the Profile's coverage.
        let (e, pid, _parent, file_anchor) = engine_with_file_anchor();

        let pre = pre_fire_burst_for_test(
            BurstIntent::Standard,
            dirty_at(&[(file_anchor, "/parentdir/main.rs")]),
        );
        assert_eq!(
            pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree),
            file_anchor,
        );

        // Same conclusion even if dirty is empty.
        let pre_empty = pre_fire_burst_for_test(BurstIntent::Standard, DirtyProvenance::new());
        assert_eq!(
            pre_fire_target(e.profiles.get(pid).unwrap(), &pre_empty, &e.tree),
            file_anchor,
        );

        // And under Seed intent.
        let pre_seed = pre_fire_burst_for_test(BurstIntent::Seed, DirtyProvenance::new());
        assert_eq!(
            pre_fire_target(e.profiles.get(pid).unwrap(), &pre_seed, &e.tree),
            file_anchor,
        );
    }

    #[test]
    fn pre_fire_target_seed_intent_returns_anchor() {
        // Seed intent on a Dir-anchored Profile: target is the anchor,
        // regardless of dirty contents. Seed bursts compare against fire
        // history rather than a stable subtree verdict, so they probe at
        // the anchor unconditionally.
        let (e, pid, root, a, _b) = engine_with_two_children();

        let pre = pre_fire_burst_for_test(BurstIntent::Seed, dirty_at(&[(a, "/root/a")]));
        assert_eq!(
            pre_fire_target(e.profiles.get(pid).unwrap(), &pre, &e.tree),
            root,
        );

        // Same with empty dirty.
        let pre_empty = pre_fire_burst_for_test(BurstIntent::Seed, DirtyProvenance::new());
        assert_eq!(
            pre_fire_target(e.profiles.get(pid).unwrap(), &pre_empty, &e.tree),
            root,
        );
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
        let parent = e.tree.ensure_root("/parentdir", ResourceRole::User);
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
    fn transition_to_verifying_on_file_anchor_targets_anchor() {
        // File-anchored Profile: a Standard burst's probe target must be
        // the anchor itself, not the parent dir. `pre_fire_target`
        // short-circuits a File-kind anchor before any path-LCA
        // resolution. This test pins that through the emission path —
        // promoting past the anchor would route the probe outside the
        // Profile's coverage and (downstream) wholesale-replace
        // `Profile.current` with a Dir snapshot at the parent.
        let (mut e, pid, _parent, file_anchor) = engine_with_file_anchor();
        let mut start_out = StepOutput::default();
        let fp = rpath(&e, file_anchor);
        e.start_standard_burst(pid, file_anchor, &fp, Instant::now(), &mut start_out);

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
    fn standard_obligation_chains_carry_every_captured_dirty_path() {
        // The emission choke materializes ProofObligation::Chains from
        // *every* captured dirty path, with no subtree-of-target filter
        // (the target is an ancestor-or-equal of every value by
        // construction, so the old filter was a tautology). A captured
        // ancestor path (`/root`) and a sibling (`/root/b`) both survive
        // alongside `/root/a` — none is dropped.
        let (mut e, pid, root, a, b) = engine_with_two_children();
        let (rp, ap, bp) = (rpath(&e, root), rpath(&e, a), rpath(&e, b));
        let now = Instant::now();
        let mut out = StepOutput::default();
        e.start_standard_burst(pid, a, &ap, now, &mut out);
        e.event_drives_batching(pid, root, &rp, now, &mut out);
        e.event_drives_batching(pid, b, &bp, now, &mut out);

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
        match req {
            ProbeRequest::Subtree { obligation, .. } => match obligation {
                ProofObligation::Chains(chains) => {
                    assert!(chains.contains(&ap), "the trigger path is a chain");
                    assert!(chains.contains(&bp), "a sibling chain is not filtered");
                    assert!(
                        chains.contains(&rp),
                        "a captured ancestor path is not filtered out",
                    );
                    assert_eq!(chains.len(), 3, "exactly the captured paths, no more");
                }
                other => panic!("Standard burst obligation must be Chains; got {other:?}"),
            },
            other => panic!("Standard burst must emit ProbeRequest::Subtree; got {other:?}"),
        }
        let _ = e.cancel_all_in_flight_probes();
    }

    #[test]
    fn transition_to_verifying_standard_uses_lca() {
        let (mut e, pid, root, a, b) = engine_with_two_children();
        let (ap, bp) = (rpath(&e, a), rpath(&e, b));
        let now = Instant::now();
        // Standard burst with two dirty siblings → component-LCA of
        // `/root/a` + `/root/b` is `/root`, resolving to the anchor.
        let mut out = StepOutput::default();
        e.start_standard_burst(pid, a, &ap, now, &mut out);
        e.event_drives_batching(pid, b, &bp, now, &mut out);

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
        // Subtree variant carries `target_path` and the proof `obligation`;
        // a Standard burst on a Dir-anchored Profile produces `Chains`.
        let anchor_path = e.tree.path_of(root).expect("anchor path resolves");
        match req {
            ProbeRequest::Subtree {
                target_path,
                obligation,
                ..
            } => {
                assert_eq!(
                    *target_path, anchor_path,
                    "sibling component-LCA resolves to the anchor",
                );
                match obligation {
                    ProofObligation::Chains(chains) => assert_eq!(chains.len(), 2),
                    other => panic!("Standard burst obligation must be Chains; got {other:?}"),
                }
            }
            other => panic!(
                "Standard burst on Dir-anchored Profile must emit ProbeRequest::Subtree; \
                 got {other:?}",
            ),
        }
        let _ = e.cancel_all_in_flight_probes();
    }
}
