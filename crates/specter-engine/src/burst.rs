//! Burst lifecycle helpers.
//!
//! Each helper is the **single source** of one transition kind — a phase
//! transition body, a Burst construction, or a return-to-Idle. Centralizing
//! the timer scheduling, refcount edges, and Burst-struct mutations here
//! prevents drift between the transition-row handlers and the
//! post-`EffectComplete` re-probe path.
//!
//! - `start_seed_burst` / `start_standard_burst` — Idle → Active.
//! - `event_drives_batching` (FsEvent during Active) /
//!   `unstable_response_drives_batching` (probe-unstable response) /
//!   `transition_to_verifying` (settle-timer expiry, burst-deadline,
//!   Draining → Verifying reconfirm) /
//!   `transition_to_draining` — Active → Active phase swaps.
//! - `finish_burst_to_idle` — Active → Idle, single point of `-suppress` and
//!   `propagate(-1)`.
//!
//! The two batching helpers exist as a deliberate split rather than one
//! helper with a runtime flag: each caller has **static knowledge** of
//! whether a probe is in flight (only `event_drives_batching` may need to
//! emit `ProbeOp::Cancel`). Encoding that knowledge as helper identity
//! makes a stray Cancel on the just-responded path structurally
//! impossible.
//!
//! Probe emission flows through the typed helpers in `probe_channel`
//! (`emit_anchor_probe` / `emit_subtree_probe`). Each burst-launch helper
//! reads `Profile.kind` and routes: `Some(File)` ⇒ `emit_anchor_probe`
//! at the anchor; `Some(Dir | Unknown)` or `None` ⇒ `emit_subtree_probe`
//! at the computed target. Unclassified anchors take the same Subtree
//! arm — the walker returns `Vanished` on kind mismatch and the engine
//! recovers via descent.

use crate::Engine;
use crate::refcounts::{add_suppress, sub_suppress};
use smallvec::SmallVec;
use specter_core::{
    Burst, BurstIntent, BurstPhase, ProbeOwner, ProfileId, ProfileState, ResourceId, ResourceKind,
    StepOutput, TimerKind, Tree, TreeSnapshot,
};
use std::collections::BTreeSet;
use std::path::PathBuf;
use std::time::Instant;

impl Engine {
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
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        debug_assert!(
            matches!(p.state, ProfileState::Idle),
            "start_seed_burst: Profile must be Idle on entry",
        );
        let resource = p.resource;
        let max_settle = p.max_settle;
        let anchor_kind = p.kind;
        let scan_config = p.config.clone();
        let captured_with = p.config_hash;
        // Seed targets the anchor; baseline_subtree is current.subtree_at(anchor)
        // for post-Effect Seeds (gives the walker mtime-skip for noop Effects)
        // and None for fresh-Profile / recovery Seeds (no prior observation).
        let baseline_subtree = p
            .current
            .as_ref()
            .and_then(|s| s.subtree_at(resource, &self.tree));

        let burst_deadline =
            self.timers
                .schedule(now + max_settle, profile_id, TimerKind::BurstDeadline);
        let Some(correlation) = self.mint_owner_correlation(ProbeOwner::Profile(profile_id)) else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Active(Burst {
                burst_deadline,
                phase: BurstPhase::Verifying,
                intent: BurstIntent::Seed,
                forced: false,
                dirty_resources: BTreeSet::new(),
                force_walk_resources: BTreeSet::new(),
                probe_target: Some(resource),
                suppressed_resources: BTreeSet::new(),
                // Seed bursts skip Batching; the field has no consumer
                // until a fresh FsEvent during the verify routes through
                // `event_drives_batching` and repopulates it.
                last_event_time: None,
            });
        }

        add_suppress(&mut self.tree, resource, out);
        let target_path = self.tree.path_of(resource).unwrap_or_default();
        match anchor_kind {
            Some(ResourceKind::File) => {
                Self::emit_anchor_probe(
                    ProbeOwner::Profile(profile_id),
                    correlation,
                    target_path,
                    out,
                );
            }
            // Dir or unclassified ⇒ unified Subtree fallback. An Unknown
            // anchor (resource-based attach whose first probe hasn't yet
            // returned, or a slot whose kind never propagated past the
            // default) probes as a Dir; the walker returns `Vanished` on
            // kind mismatch and the engine routes through
            // `dispatch_seed_vanished` to recover via descent.
            Some(ResourceKind::Dir | ResourceKind::Unknown) | None => {
                Self::emit_subtree_probe(
                    ProbeOwner::Profile(profile_id),
                    correlation,
                    resource,
                    target_path,
                    scan_config,
                    captured_with,
                    baseline_subtree,
                    BTreeSet::new(),
                    false,
                    out,
                );
            }
        }
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
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        debug_assert!(
            matches!(p.state, ProfileState::Idle),
            "start_standard_burst: Profile must be Idle on entry",
        );
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
            p.state = ProfileState::Active(Burst {
                burst_deadline,
                phase: BurstPhase::Batching { settle_timer },
                intent: BurstIntent::Standard,
                forced: false,
                dirty_resources: dirty,
                force_walk_resources: force_walk,
                probe_target: None,
                suppressed_resources: BTreeSet::new(),
                // The burst-start FsEvent IS the first event; seed the
                // settle-deadline source of truth with `now`. Subsequent
                // events update this in `event_drives_batching` without
                // re-inserting a fresh heap entry.
                last_event_time: Some(now),
            });
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
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(burst) = &p.state else {
            return;
        };
        let settle = p.settle;
        let anchor = p.resource;

        // Read phase before mutating self via `cancel_pending_probe`. The
        // Cancel emission doesn't touch `burst.phase`, but it does take
        // `&mut self` and so invalidates the borrow on `burst`. Decide
        // here whether the existing Batching settle timer (if any) carries
        // over, or whether we mint a fresh one for a Verifying/Draining
        // re-entry. The decision is structural: a live Batching has its
        // own timer slot; Verifying/Draining have none.
        let needs_fresh_timer =
            matches!(burst.phase, BurstPhase::Verifying | BurstPhase::Draining,);

        // Idempotent: emits Cancel iff the probe channel is open
        // (Verifying ⇒ pending_probe = Some(_)). For Batching / Draining
        // entries, no probe is in flight and the helper is a no-op —
        // matching the prior `was_verifying` snapshot's role.
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
                .is_some_and(|p| match &p.state {
                    ProfileState::Active(b) => !b.suppressed_resources.contains(&event_resource),
                    _ => false,
                });
        if needs_suppress {
            add_suppress(&mut self.tree, event_resource, out);
        }

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.last_event_time = Some(now);
            if needs_suppress {
                burst.suppressed_resources.insert(event_resource);
            }
            burst.dirty_resources.insert(event_resource);
            burst.force_walk_resources.insert(event_resource);
            if let Some(timer_id) = new_settle_timer {
                burst.phase = BurstPhase::Batching {
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
    /// probe and clears `pending_probe`; the eventual late response
    /// then fails the live-slot check and drops as `StaleProbeResponse`.
    /// The forced + not-stable case in `dispatch_standard_ok` also
    /// bypasses this helper — forced + unstable still fires.
    ///
    /// **`last_event_time` preserved.** The verify just responded;
    /// no fresh `FsEvent` drove this transition, so the field carries
    /// its prior value into the new Batching cycle. If no event arrives
    /// before the freshly-scheduled settle timer fires, the on-expiry
    /// handler observes `now − last_event_time ≥ settle`
    /// (`now ≥ unstable_response_at + settle ≥ prior_last_event +
    /// settle`) and transitions cleanly to Verifying — the cycle
    /// completes without spurious reschedules.
    pub(crate) fn unstable_response_drives_batching(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
    ) {
        let Some(settle) = self.profiles.get(profile_id).map(|p| p.settle) else {
            return;
        };
        let settle_timer = self
            .timers
            .schedule(now + settle, profile_id, TimerKind::Settle);

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.phase = BurstPhase::Batching { settle_timer };
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
    /// Standard probes target the LCA of the burst's `dirty_resources`,
    /// ship `current.subtree_at(target)` as the walker's mtime-skip
    /// baseline, ship `force_walk_resources` (rendered to paths) so the
    /// walker re-walks paths whose kqueue actually fired since the last
    /// probe, and propagate `Burst.forced` so the walker bypasses
    /// mtime-skip on a force-fire (max-settle deadline elapsed). Seed
    /// probes target the anchor; the Draining → Verifying reconfirm
    /// reuses `Burst.probe_target` (`dirty_resources` is empty by then so
    /// LCA would degenerate to anchor and lose the correct subtree).
    /// `force_walk_resources` is consumed by this emission; new events
    /// accumulate into the cleared set.
    pub(crate) fn transition_to_verifying(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };

        // Take the per-burst consumables in one mutable-borrow window. The
        // post-fire phases (`Awaiting` / `Rebasing`) never reach this
        // helper — production callers (Settle expiry, BurstDeadline
        // expiry, ancestor reconfirm from `finish_burst_to_idle`) are
        // gated on pre-fire phases via `is_timer_referenced` and the
        // `Draining` hit-zero check respectively. The early return guards
        // a stray call from minting a fresh probe correlation while an
        // effect wait is still in flight (which would either trip the I5
        // debug_assert in `mint_probe_correlation` or, in release,
        // overwrite the post-fire correlation slot).
        //
        // `force_walk_resources` and `suppressed_resources` are
        // single-use accumulators consumed by this transition; `mem::take`
        // moves them out and leaves the fields empty (no follow-up
        // `clear()` required). `dirty_resources` is preserved — it carries
        // the LCA basis across the whole burst, so we clone it here.
        let (intent, phase, prior_target, dirty_for_lca, force_set, suppressed_drain, forced) =
            match &mut p.state {
                ProfileState::Active(b) => {
                    if matches!(b.phase, BurstPhase::Awaiting { .. } | BurstPhase::Rebasing) {
                        return;
                    }
                    (
                        b.intent,
                        phase_kind(&b.phase),
                        b.probe_target,
                        b.dirty_resources.clone(),
                        std::mem::take(&mut b.force_walk_resources),
                        std::mem::take(&mut b.suppressed_resources),
                        b.forced,
                    )
                }
                _ => return,
            };

        // Cached anchor classification. `None` is a resource-based
        // attach whose Seed probe hasn't yet returned (or any case where
        // the kind hasn't propagated past the default); the typed
        // contract routes both `None` and `Some(Dir)` through Subtree
        // emission, so the value is preserved unchanged here.
        let resource = p.resource;
        let anchor_kind = p.kind;
        let scan_config = p.config.clone();
        let captured_with = p.config_hash;

        // Decide target. File anchors always target the anchor itself
        // (kqueue per-file FDs surface events at the file directly,
        // and the walker's AnchorFile arm lstat's the leaf — promoting
        // past the anchor to the parent dir would route the probe
        // outside the Profile's coverage). Dir / unclassified anchors
        // pick LCA / prior / anchor by phase.
        let target = match (intent, phase, anchor_kind) {
            (_, _, Some(ResourceKind::File)) => resource,
            (BurstIntent::Seed, _, _) => resource,
            (BurstIntent::Standard, PhaseKind::Draining, _) => {
                // Reconfirm probe — re-use the previous target. dirty_resources
                // is empty in Draining, so LCA would degenerate to anchor and
                // lose the correct subtree.
                prior_target.unwrap_or(resource)
            }
            // PostFire is unreachable courtesy of the early-return guard
            // above; routed alongside `Batching | Verifying` to keep the
            // match exhaustive without a wildcard.
            (BurstIntent::Standard, _, _) => lca_target(resource, &dirty_for_lca, &self.tree),
        };

        // baseline_subtree at the target. Unused by AnchorFile emission
        // (a leaf has no descendants to mtime-skip) but cheap to compute
        // unconditionally — `subtree_at` is a tree-zipper walk over
        // existing snapshots.
        let baseline_subtree = p
            .current
            .as_ref()
            .and_then(|s| s.subtree_at(target, &self.tree));
        // force_walk paths (filtered to subtree(target); engine-side close).
        let force_walk_paths = build_force_walk(&force_set, target, &self.tree);
        let target_path = self.tree.path_of(target).unwrap_or_default();

        let Some(correlation) = self.mint_owner_correlation(ProbeOwner::Profile(profile_id)) else {
            return;
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

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(b) = &mut p.state
        {
            b.phase = BurstPhase::Verifying;
            b.probe_target = Some(target);
        }

        match anchor_kind {
            Some(ResourceKind::File) => {
                Self::emit_anchor_probe(
                    ProbeOwner::Profile(profile_id),
                    correlation,
                    target_path,
                    out,
                );
            }
            // Dir or unclassified ⇒ Subtree probe at the chosen target.
            // See `start_seed_burst` for the unified-fallback rationale
            // (an Unknown anchor that's actually a File on disk surfaces
            // as a `Vanished` Subtree response and recovers via descent).
            Some(ResourceKind::Dir | ResourceKind::Unknown) | None => {
                Self::emit_subtree_probe(
                    ProbeOwner::Profile(profile_id),
                    correlation,
                    target,
                    target_path,
                    scan_config,
                    captured_with,
                    baseline_subtree,
                    force_walk_paths,
                    forced,
                    out,
                );
            }
        }
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
    pub(crate) fn transition_to_draining(&mut self, profile_id: ProfileId) {
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };
        if let ProfileState::Active(burst) = &mut p.state {
            burst.phase = BurstPhase::Draining;
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
    /// `Profile.reap_pending`, finishes the burst directly).
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
    pub(crate) fn transition_to_awaiting(
        &mut self,
        profile_id: ProfileId,
        outstanding: u32,
        now: Instant,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let max_settle = p.max_settle;

        // v1 default: 4× max_settle. Saturating multiplication keeps the
        // arithmetic total — `Duration::saturating_mul` clamps at
        // `Duration::MAX`, leaving the deadline well beyond any
        // reasonable wall-clock horizon.
        let gate_deadline = self.timers.schedule(
            now + max_settle.saturating_mul(4),
            profile_id,
            TimerKind::AwaitGateDeadline,
        );

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.phase = BurstPhase::Awaiting {
                outstanding,
                gate_deadline,
            };
        }
    }

    /// Phase: `Awaiting` → `Rebasing`. The single source of the
    /// post-effect rebase: `on_effect_complete` calls this when
    /// `outstanding` reaches zero (and `reap_pending` is false), and
    /// `handle_gate_deadline` calls it on the actuator-hang recovery
    /// path.
    ///
    /// **Probe channel.** `mint_probe_correlation` opens a fresh probe
    /// channel — I5 holds because Verifying closed the slot before
    /// `emit_effects` ran, and Awaiting does not mint. The post-fire
    /// probe targets the anchor (we want the freshest disk state of
    /// the whole watched subtree, not the LCA of the now-stale
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
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };

        // Take absorbed events for the rebase walker's force_walk in one
        // mut-borrow window. `force_walk_resources` is the shared
        // accumulator for "events since last probe" — Verifying and
        // Rebasing are sister consumers. Non-Active state is a state-
        // machine bug here (see rustdoc above); early-return rather than
        // silently mint a Probe with no matching phase write.
        let force_set = match &mut p.state {
            ProfileState::Active(burst) => std::mem::take(&mut burst.force_walk_resources),
            _ => return,
        };

        let resource = p.resource;
        let anchor_kind = p.kind;
        let scan_config = p.config.clone();
        let captured_with = p.config_hash;
        let baseline_subtree = p
            .current
            .as_ref()
            .and_then(|s| s.subtree_at(resource, &self.tree));
        let force_walk_paths = build_force_walk(&force_set, resource, &self.tree);
        let target_path = self.tree.path_of(resource).unwrap_or_default();

        let Some(correlation) = self.mint_owner_correlation(ProbeOwner::Profile(profile_id)) else {
            return;
        };

        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.phase = BurstPhase::Rebasing;
            burst.probe_target = Some(resource);
        }

        match anchor_kind {
            Some(ResourceKind::File) => {
                Self::emit_anchor_probe(
                    ProbeOwner::Profile(profile_id),
                    correlation,
                    target_path,
                    out,
                );
            }
            Some(ResourceKind::Dir | ResourceKind::Unknown) | None => {
                Self::emit_subtree_probe(
                    ProbeOwner::Profile(profile_id),
                    correlation,
                    resource,
                    target_path,
                    scan_config,
                    captured_with,
                    baseline_subtree,
                    force_walk_paths,
                    false,
                    out,
                );
            }
        }
    }

    /// Active → Idle. Single source of `-suppress` and `propagate(-1)`.
    /// The active burst's timers are not explicitly cancelled — lazy
    /// invalidation in `pop_expired` drops them when they fire.
    /// Idempotent: silent no-op on already-Idle Profiles.
    ///
    /// **Draining-exit driver.** `propagate(-1)` returns ancestors whose
    /// `dirty_descendants` just hit zero AND are in `BurstPhase::Draining`.
    /// The Engine drives each through `transition_to_verifying` in the
    /// same step — the reconfirm probe compares against the Profile's
    /// `current` (set when `dispatch_standard_ok` entered Draining).
    /// Same-step ordering means the `StepOutput` reflects the cascade:
    /// child's burst end → parent reconfirm Probe in one `step` call.
    ///
    /// **Reap-pending.** If the Profile's `reap_pending` flag is set (its
    /// last Sub was detached mid-burst), `Engine::reap_profile` runs in the
    /// same step after `propagate(-1)` to release watch contributions,
    /// parent edges, and Tree slot.
    pub(crate) fn finish_burst_to_idle(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get_mut(profile_id) else {
            return;
        };
        let resource = p.resource;

        // One mut-borrow window covers two reads (intent for the
        // propagate(-1) gate; suppressed_resources for the defensive
        // drain) and the state→Idle write. Tightened from
        // `!matches!(state, Idle)` to `Active(_)`: the burst-end
        // machinery (`sub_suppress`, `propagate(-1)`) is Active-specific.
        // Pending Profiles never bumped the anchor's suppress_count or
        // the ancestor `dirty_descendants`; running this on Pending
        // would underflow `sub_suppress`. The only documented caller-
        // side guard that this defends against is
        // `finalize_anchor_lost` if a future change relaxes its Pending
        // early-return.
        //
        // The defensive `suppressed_resources` drain catches abnormal-
        // end paths that bypass `transition_to_verifying`
        // (`finalize_anchor_lost` mid-Batching, `reap_profile` mid-burst,
        // config-diff reap). `sub_suppress` on a stale `ResourceId`
        // (slot reaped between event ingestion and burst end via
        // `discard_anchor_state`'s descendant release) is a no-op —
        // `refcounts::sub_suppress` short-circuits on a missing slot.
        // After a normal `transition_to_verifying` drained the set, the
        // `mem::take` here yields an empty BTreeSet — no double-drain.
        // Only Standard bursts call `propagate(+1)` at start (the
        // burst-propagation row), so only Standard bursts call
        // `propagate(-1)` at end. Seed bursts never contribute to
        // ancestor `dirty_descendants`.
        let (was_standard, suppressed_drain) = match &mut p.state {
            ProfileState::Active(b) => (
                matches!(b.intent, BurstIntent::Standard),
                std::mem::take(&mut b.suppressed_resources),
            ),
            _ => return,
        };
        p.state = ProfileState::Idle;

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

        // Reap-pending check. The flag is set by `detach_sub` when the
        // Profile was Active and lost its last Sub; we defer the reap to
        // here so the Profile's burst doesn't fire Effects against a Sub
        // registry that no longer holds the reference.
        let reap_now = self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.reap_pending);
        if reap_now {
            self.reap_profile(profile_id, out);
        }
    }
}

/// Copy projection of `BurstPhase` for `transition_to_verifying`'s
/// `(intent, phase)` match — `Batching`'s `TimerId` is irrelevant at the
/// dispatch site, and `Verifying` is now unit (the probe correlation
/// lives on `Profile.pending_probe`). The post-fire phases
/// (`Awaiting` / `Rebasing`) are filtered by an early-return guard at
/// `transition_to_verifying`'s entry; the projection's `PostFire` arm
/// exists for exhaustiveness only and is structurally unreachable.
#[derive(Copy, Clone, Eq, PartialEq)]
enum PhaseKind {
    Batching,
    Verifying,
    Draining,
    /// Defense-in-depth bucket for `Awaiting` and `Rebasing`. Reaching
    /// `phase_kind` from a post-fire phase is a state-machine bug;
    /// `transition_to_verifying`'s entry guard returns before this is
    /// observed, but the match below must be exhaustive over `BurstPhase`.
    PostFire,
}

const fn phase_kind(p: &BurstPhase) -> PhaseKind {
    match p {
        BurstPhase::Batching { .. } => PhaseKind::Batching,
        BurstPhase::Verifying => PhaseKind::Verifying,
        BurstPhase::Draining => PhaseKind::Draining,
        BurstPhase::Awaiting { .. } | BurstPhase::Rebasing => PhaseKind::PostFire,
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
/// (kqueue per-file FDs surface events at the file directly); the kind
/// dispatch in [`Engine::transition_to_verifying`] routes File anchors to
/// [`Engine::emit_anchor_probe`] without consulting this helper, so a
/// File-anchored Profile never reaches `lca_target` in production. The
/// `live.contains(&anchor)` short-circuit below remains valid for the
/// Dir-anchor case where the anchor itself is the event source (e.g., an
/// in-place mtime bump on the anchor directory).
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
) -> ResourceId {
    // 1. Filter stale ResourceIds. A `dirty_resources` entry whose slot
    // was reaped between FsEvent ingestion and probe emission
    // (delete-recreate-different-inode race) is dropped here.
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
    // up to a common depth, then up in lockstep until they match.
    let mut acc = live[0];
    for &r in &live[1..] {
        match lca_pair(acc, r, tree) {
            Some(joint) => acc = joint,
            None => return anchor,
        }
    }
    promote_to_dir(acc, anchor, tree)
}

/// LCA of two resources via depth-equalisation + lockstep ancestor walk.
/// O(max(depth_a, depth_b)). Returns `None` only when an input slot is
/// stale or a parent walk runs out of ancestors before the candidates
/// align — the caller falls back to anchor.
fn lca_pair(a: ResourceId, b: ResourceId, tree: &Tree) -> Option<ResourceId> {
    if a == b {
        return Some(a);
    }
    let depth_a = tree.ancestors(a).count();
    let depth_b = tree.ancestors(b).count();
    let mut a = a;
    let mut b = b;
    // Walk the deeper one up to the same depth as the shallower.
    for _ in 0..depth_a.saturating_sub(depth_b) {
        a = tree.parent(a)?;
    }
    for _ in 0..depth_b.saturating_sub(depth_a) {
        b = tree.parent(b)?;
    }
    // Walk both up in lockstep until they match.
    while a != b {
        a = tree.parent(a)?;
        b = tree.parent(b)?;
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
) -> BTreeSet<PathBuf> {
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
        BurstIntent, BurstPhase, ClassSet, Input, ProbeOp, ProbeOwner, ProbeRequest, Profile,
        ProfileState, ResourceKind, ResourceRole, ScanConfig, StepOutput, TimerKind, WatchOp,
    };
    use std::time::{Duration, Instant};

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// Build an Engine with a single Profile anchored at `/anchor`. Returns the
    /// Engine + the `ProfileId`.
    fn engine_with_profile() -> (Engine, specter_core::ProfileId) {
        let mut e = Engine::new();
        let r = e.tree.ensure(None, "anchor", ResourceRole::User);
        e.tree.set_kind(r, ResourceKind::Dir);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r,
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
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
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Seed);
        assert!(matches!(burst.phase, BurstPhase::Verifying));
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
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert_eq!(burst.intent, BurstIntent::Standard);
        assert!(matches!(burst.phase, BurstPhase::Batching { .. }));

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

        match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => assert!(matches!(b.phase, BurstPhase::Verifying)),
            _ => panic!("expected Active"),
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
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(matches!(burst.phase, BurstPhase::Batching { .. }));
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
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(matches!(burst.phase, BurstPhase::Batching { .. }));
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

        e.unstable_response_drives_batching(pid, now);

        assert!(out.probe_ops.is_empty());
        let phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(burst) => &burst.phase,
            _ => panic!("expected Active"),
        };
        assert!(matches!(phase, BurstPhase::Batching { .. }));
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
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
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
            BurstPhase::Batching { settle_timer } => settle_timer,
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

        let phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => &b.phase,
            _ => panic!("expected Active after reschedule"),
        };
        let rescheduled_timer = match phase {
            BurstPhase::Batching { settle_timer } => *settle_timer,
            other => panic!("expected Batching after reschedule, got {other:?}"),
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
        let final_phase = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => &b.phase,
            other => panic!("expected Active, got {other:?}"),
        };
        assert!(
            matches!(final_phase, BurstPhase::Verifying),
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
            e.profiles.get(pid).unwrap().state,
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
        let burst_deadline_initial = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b.burst_deadline,
            _ => panic!(),
        };
        let r = e.profiles.get(pid).unwrap().resource;

        e.event_drives_batching(pid, r, Instant::now(), &mut out);
        let burst_deadline_after = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b.burst_deadline,
            _ => panic!(),
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

        e.transition_to_draining(pid);

        let p = e.profiles.get(pid).unwrap();
        let burst = match &p.state {
            ProfileState::Active(b) => b,
            _ => panic!(),
        };
        assert!(matches!(burst.phase, BurstPhase::Draining));
        // Intent and forced preserved.
        assert_eq!(burst.intent, BurstIntent::Seed);
    }

    // ---------------------------------------------------------------------------
    // LCA + force_walk + transition_to_verifying
    // ---------------------------------------------------------------------------

    use crate::burst::{build_force_walk, lca_target};
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
        let root = e.tree.ensure(None, "root", ResourceRole::User);
        e.tree.set_kind(root, ResourceKind::Dir);
        let a = e.tree.ensure(Some(root), "a", ResourceRole::User);
        e.tree.set_kind(a, ResourceKind::Dir);
        let b = e.tree.ensure(Some(root), "b", ResourceRole::User);
        e.tree.set_kind(b, ResourceKind::Dir);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                root,
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );
        (e, pid, root, a, b)
    }

    #[test]
    fn lca_empty_dirty_returns_anchor() {
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let dirty = BTreeSet::new();
        let target = lca_target(e.profiles.get(pid).unwrap().resource, &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn lca_two_siblings_returns_parent() {
        let (e, pid, root, a, b) = engine_with_two_children();
        let dirty: BTreeSet<_> = [a, b].iter().copied().collect();
        let target = lca_target(e.profiles.get(pid).unwrap().resource, &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn lca_single_dirty_at_anchor_returns_anchor() {
        let (e, pid, root, _a, _b) = engine_with_two_children();
        let dirty: BTreeSet<_> = std::iter::once(root).collect();
        let target = lca_target(e.profiles.get(pid).unwrap().resource, &dirty, &e.tree);
        assert_eq!(target, root);
    }

    #[test]
    fn lca_single_dirty_deep_returns_self() {
        let (e, pid, _root, a, _b) = engine_with_two_children();
        let dirty: BTreeSet<_> = std::iter::once(a).collect();
        let target = lca_target(e.profiles.get(pid).unwrap().resource, &dirty, &e.tree);
        assert_eq!(target, a);
    }

    #[test]
    fn lca_filters_stale_resource_ids() {
        let (mut e, pid, root, a, _b) = engine_with_two_children();
        // Reap `a` to make its id stale.
        e.tree.vacate(a, &mut StepOutput::default());
        e.tree.try_reap(a);
        // Stale id in the set; LCA must filter and return anchor (since the
        // remaining live entry is empty after the filter).
        let dirty: BTreeSet<_> = std::iter::once(a).collect();
        let target = lca_target(e.profiles.get(pid).unwrap().resource, &dirty, &e.tree);
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
        let parent = e.tree.ensure(None, "parentdir", ResourceRole::User);
        e.tree.set_kind(parent, ResourceKind::Dir);
        let file_anchor = e.tree.ensure(Some(parent), "main.rs", ResourceRole::User);
        e.tree.set_kind(file_anchor, ResourceKind::File);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                file_anchor,
                ScanConfig::builder().recursive(false).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );
        if let Some(p) = e.profiles.get_mut(pid) {
            p.kind = Some(ResourceKind::File);
        }
        (e, pid, parent, file_anchor)
    }

    #[test]
    fn lca_pairwise_reduction_resolves_to_shared_intermediate_ancestor() {
        // Witness for the pairwise LCA reduction. Two leaves under
        // disjoint mid-3 branches share a depth-2 ancestor (`l2`); the
        // reduction must resolve to that ancestor, not collapse to the
        // anchor and not return either leaf.
        let mut e = Engine::new();
        let l0 = e.tree.ensure(None, "l0", ResourceRole::User);
        e.tree.set_kind(l0, ResourceKind::Dir);
        let l1 = e.tree.ensure(Some(l0), "l1", ResourceRole::User);
        e.tree.set_kind(l1, ResourceKind::Dir);
        let l2 = e.tree.ensure(Some(l1), "l2", ResourceRole::User);
        e.tree.set_kind(l2, ResourceKind::Dir);
        let l3a = e.tree.ensure(Some(l2), "a", ResourceRole::User);
        e.tree.set_kind(l3a, ResourceKind::Dir);
        let l3b = e.tree.ensure(Some(l2), "b", ResourceRole::User);
        e.tree.set_kind(l3b, ResourceKind::Dir);
        let leaf_a = e.tree.ensure(Some(l3a), "x", ResourceRole::User);
        e.tree.set_kind(leaf_a, ResourceKind::File);
        let leaf_b = e.tree.ensure(Some(l3b), "y", ResourceRole::User);
        e.tree.set_kind(leaf_b, ResourceKind::File);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                l0,
                ScanConfig::builder().recursive(true).build(),
                MAX_SETTLE,
                SETTLE,
                NO_EVENTS,
            ),
        );

        let dirty: BTreeSet<_> = [leaf_a, leaf_b].iter().copied().collect();
        let target = lca_target(e.profiles.get(pid).unwrap().resource, &dirty, &e.tree);
        assert_eq!(
            target, l2,
            "LCA of leaves under l3a and l3b is l2 (their shared depth-2 ancestor)",
        );
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
        if let ProfileState::Active(b_burst) = &mut e.profiles.get_mut(pid).unwrap().state {
            b_burst.dirty_resources.insert(b);
            b_burst.force_walk_resources.insert(b);
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
        // Subtree variant carries `target_resource` and `force_walk` directly;
        // a Standard burst on a Dir-anchored Profile must produce this variant.
        match req {
            ProbeRequest::Subtree {
                target_resource,
                force_walk,
                ..
            } => {
                assert_eq!(*target_resource, e.profiles.get(pid).unwrap().resource);
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
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
            _ => panic!(),
        };
        assert!(burst.force_walk_resources.is_empty());
        // dirty_resources is preserved (LCA basis spans the whole burst).
        assert!(!burst.dirty_resources.is_empty());
        // probe_target was set to the LCA result.
        assert!(burst.probe_target.is_some());
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
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            burst.suppressed_resources.contains(&a),
            "burst tracks the suppressed non-anchor resource",
        );
        assert_eq!(
            e.tree.get(a).unwrap().suppress_count,
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
            e.tree.get(a).unwrap().suppress_count,
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
        let suppress_after_start = e.tree.get(root).unwrap().suppress_count;
        out.watch_ops.clear();

        e.event_drives_batching(pid, root, now + Duration::from_millis(1), &mut out);
        e.event_drives_batching(pid, root, now + Duration::from_millis(2), &mut out);

        assert_eq!(
            suppress_count_for(&out, root),
            0,
            "anchor events emit no per-resource Suppress",
        );
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            !burst.suppressed_resources.contains(&root),
            "anchor never enters suppressed_resources",
        );
        assert_eq!(
            e.tree.get(root).unwrap().suppress_count,
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
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
            _ => panic!("expected Active"),
        };
        assert!(
            burst.suppressed_resources.is_empty(),
            "drain clears the per-burst tracking set",
        );
        assert_eq!(e.tree.get(a).unwrap().suppress_count, 0);
        assert_eq!(e.tree.get(b).unwrap().suppress_count, 0);
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
        // call `cancel_pending_probe`.)
        e.unstable_response_drives_batching(pid, now + Duration::from_millis(2));
        out.watch_ops.clear();

        e.event_drives_batching(pid, a, now + Duration::from_millis(3), &mut out);

        assert_eq!(
            suppress_count_for(&out, a),
            1,
            "post-unstable-verify event re-emits Suppress(a)",
        );
        let burst = match &e.profiles.get(pid).unwrap().state {
            ProfileState::Active(b) => b,
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
        assert_eq!(e.tree.get(a).unwrap().suppress_count, 0);
        assert_eq!(e.tree.get(b).unwrap().suppress_count, 0);
        assert_eq!(e.tree.get(root).unwrap().suppress_count, 0);
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
