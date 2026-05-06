//! Per-input dispatch handlers.
//!
//! Each `on_*` method handles one [`Input`] variant for one Profile. They
//! call the burst-lifecycle helpers (`burst.rs`), the refcount edges
//! (`refcounts.rs`), and the reconciliation (`reconcile.rs`). Logic that
//! fits in one row of the transition table stays inline; logic shared across
//! rows (e.g., emit Effects on Standard stable verdict) is factored into
//! private helpers within this module.
//!
//! The match on `(intent, ProbeResult)` is the single dispatch site for the
//! post-probe state-transition chain — six rows, all reachable.

use crate::Engine;
use crate::reconcile::{ensure_descendant, graft, lookup_descendant};
use crate::refcounts::clamp_watch_demand_to_zero;
use smallvec::SmallVec;
use specter_core::{
    BurstIntent, BurstPhase, ClaimKind, ClassSet, CorrelationId, DedupKey, Diagnostic, Effect,
    EffectOutcome, EffectScope, FsEvent, OverflowScope, ProbeResponse, ProbeResult, ProfileId,
    ProfileState, ResourceId, ResourceKind, StepOutput, SubId, SubRegistryDiff, TimerId, TimerKind,
    TreeSnapshot, WatchFailure, WatchOp,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

impl Engine {
    /// Dispatch a normalized [`FsEvent`] for `resource`.
    ///
    /// Routing:
    /// 1. Idempotence guard — `watch_demand == 0` ⇒ `EventOnUnwatchedResource`
    ///    + drop (race between `Unwatch` and the Sensor's drain).
    /// 2. Pending descents whose `current_prefix == resource` get a fresh
    ///    descent probe (`on_descent_event`). Descent prefix watches register
    ///    STRUCTURE-only (D9), so any event reaching here is structurally
    ///    relevant by L4 — descent dispatch is unfiltered.
    /// 3. Idle Profiles whose `watch_root_parent == resource` and whose
    ///    anchor is currently absent (`current.is_none()`) re-enter pending
    ///    descent — auto-recapture on anchor reappearance. Same D9 STRUCTURE
    ///    floor applies.
    /// 4. Per-covering-Profile dispatch with class-aware filter (L5):
    ///    - Anchor events bypass the filter unconditionally per design D8 —
    ///      lifecycle signal continuity trumps user opt-out.
    ///    - Descendant events whose class (per [`fs_event_to_class`]) is
    ///      not in the Profile's `events_union` drop with
    ///      `EventClassDropped` BEFORE driving the burst (per design §6.1
    ///      — class filter sits before dirty-set bumps).
    ///    - Terminal-on-anchor → `on_anchor_terminal_event`. Anything else
    ///      that passes the filter → `drive_burst`.
    pub(crate) fn on_fs_event(
        &mut self,
        resource: ResourceId,
        event: FsEvent,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Idempotence: an FsEvent for a Resource with `watch_demand == 0`
        // is a race between Unwatch and the Sensor's drain.
        let watch_demand = self.tree.get(resource).map_or(0, |r| r.watch_demand);
        if watch_demand == 0 {
            out.diagnostics
                .push(Diagnostic::EventOnUnwatchedResource { resource });
            return;
        }

        // Route events at descent prefixes to `on_descent_event`. Multiple
        // Profiles may share one prefix (two Subs awaiting siblings under
        // the same scaffold); fan out to each. Descent prefix watches
        // register STRUCTURE-only (D9), so any event reaching here is
        // structurally relevant by L4 — no class filter applies.
        let descent_owners = self.descents_at_prefix(resource);
        for pid in &descent_owners {
            self.on_descent_event(*pid, now, out);
        }

        // If an Idle Profile's anchor is absent (current.is_none()) and the
        // event resource is its `watch_root_parent`, re-enter pending descent
        // for this Profile so an anchor reappearance is detected
        // automatically. Pending and Idle are mutually exclusive
        // ProfileState variants — the `matches!(p.state, ProfileState::Idle)`
        // filter already excludes Pending Profiles. The watch-root-parent
        // watch registers STRUCTURE-only (D9) — recovery dispatch is
        // unfiltered, same rationale as descent above.
        let recovery_targets: Vec<ProfileId> = self
            .profiles
            .iter()
            .filter(|(_, p)| {
                p.watch_root_parent == Some(resource)
                    && matches!(p.state, ProfileState::Idle)
                    && p.current.is_none()
            })
            .map(|(pid, _)| pid)
            .collect();
        let recovery_count = recovery_targets.len();
        for pid in recovery_targets {
            self.start_pending_recovery(pid, resource, out);
        }

        // Find covering Profiles (anchor or any covering ancestor). For
        // P4 single-Profile this resolves to 0 or 1; P5 multi-Profile
        // dispatches to each in encounter order.
        let covering = self.covering_profiles(resource);
        if covering.is_empty() && descent_owners.is_empty() && recovery_count == 0 {
            // No consumer: covered by no Profile, no in-flight descent,
            // and no recovery kicked off. Emit `EventNoConsumer` (a
            // benign "watched but no listener" signal — typically a
            // `WatchRootParent` event for something we don't track) and
            // drop. Distinct from `EventOnUnwatchedResource` (the
            // `watch_demand == 0` race earlier) so log levels can diverge.
            out.diagnostics
                .push(Diagnostic::EventNoConsumer { resource });
            return;
        }

        // L5 class-aware routing. Compute the event's class once from the
        // resource's kind; per-Profile dispatch consults the Profile's
        // `events_union` (D3 — every Sub on a Profile shares the same
        // mask, so the union is each Sub's mask).
        let resource_kind = self
            .tree
            .get(resource)
            .map_or(ResourceKind::Unknown, |r| r.kind);
        let event_class = fs_event_to_class(event, resource_kind);
        let is_terminal = matches!(
            event,
            FsEvent::Removed | FsEvent::Renamed | FsEvent::Revoked
        );

        for profile_id in covering {
            let Some((is_anchor, profile_events)) = self
                .profiles
                .get(profile_id)
                .map(|p| (p.resource == resource, p.events_union))
            else {
                continue;
            };

            // D8 — anchor events bypass the class filter unconditionally
            // (lifecycle: anchor disappearance recovery, anchor reappearance
            // detection, etc.). §6.1 — descendant events whose class is
            // not in the Profile's `events_union` drop here, before
            // `drive_burst` extends `dirty_resources` / `force_walk_resources`.
            if !is_anchor && !profile_events.intersects(event_class) {
                out.diagnostics.push(Diagnostic::EventClassDropped {
                    resource,
                    event,
                    profile: profile_id,
                });
                continue;
            }

            if is_terminal && is_anchor {
                self.on_anchor_terminal_event(profile_id, out);
            } else {
                // Modified/StructureChanged/MetadataChanged anywhere that
                // passes the filter, or terminal at a covered descendant
                // whose class matches: drive the burst forward. Descendant
                // terminal events drive the burst; the next probe response
                // reconciles the slot via the diff-against-prior pass.
                self.drive_burst(profile_id, resource, event, now, out);
            }
        }
    }

    /// Re-enter pending descent for an Idle Profile whose anchor is
    /// currently absent. Triggered by an event at the Profile's
    /// `watch_root_parent` ("Watch root deletion" recovery).
    /// The Profile's anchor segment becomes the sole remaining component;
    /// `enter_pending_descent` emits the descent probe at the parent.
    ///
    /// **Recovery overlap.** The parent already holds `+1 STRUCTURE` from
    /// `Profile.watch_root_parent` (set at the original anchor materialization,
    /// never cleared on `on_anchor_terminal_event`). The helper bumps another
    /// `+1` for the descent contribution; the refcount sums to `+2`. The
    /// descent contribution drops at re-materialization while the
    /// `watch_root_parent` contribution persists — see the rustdoc on
    /// `enter_pending_descent` for the full lifecycle.
    fn start_pending_recovery(
        &mut self,
        profile_id: ProfileId,
        parent: ResourceId,
        out: &mut StepOutput,
    ) {
        let Some(anchor) = self.profiles.get(profile_id).map(|p| p.resource) else {
            return;
        };
        let Some(anchor_name) = self.tree.name(anchor).map(str::to_string) else {
            return;
        };
        self.enter_pending_descent(profile_id, parent, vec![anchor_name], out);
    }

    /// Dispatch a [`ProbeResponse`].
    ///
    /// I5 staleness is decided once, against the per-Profile probe channel
    /// (`Profile.pending_probe`): the response is live iff the slot holds
    /// the received correlation. After the live check the channel is closed
    /// before any dispatch arm runs — descent advance, post-Effect Seed,
    /// and Draining → Verifying reconfirm all re-mint via
    /// [`Engine::mint_probe_correlation`], which I5-asserts an empty slot.
    /// State-machine identity (`Pending` vs `Active`) then routes the live
    /// response to the descent or burst dispatch family.
    pub(crate) fn on_probe_response(
        &mut self,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let profile_id = response.profile;
        let received = response.correlation;

        // Single I5 stale-detection check, anchored to the Profile-level
        // probe channel. Catches every stale path: stale ProfileId, response
        // after Cancel, response after a fresh mint (release-build I5
        // overwrite — see `mint_probe_correlation`), out-of-order response.
        let is_live = self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.pending_probe == Some(received));
        if !is_live {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                profile: profile_id,
                correlation: received,
            });
            return;
        }

        // Close the channel BEFORE dispatching. Dispatch arms may re-open a
        // fresh channel (descent advance / rewind, anchor-materialization
        // → Seed, Draining → Verifying reconfirm); they MUST see a closed
        // channel on entry, otherwise the I5 debug_assert in
        // `mint_probe_correlation` fires.
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.pending_probe = None;
        }

        // Route on state-machine identity. The live `pending_probe` belongs
        // to either a descent (`Pending`) or a burst (`Active`); within
        // `Active` the `Rebasing` phase carves out its own dispatch
        // family (post-fire rebase, no stability verdict — graft +
        // baseline := current + finish). The wildcard absorbs `Idle`
        // (defensive — should not occur with `pending_probe = Some(_)`)
        // and any future `non_exhaustive` variant.
        let dispatch = match self.profiles.get(profile_id).map(|p| &p.state) {
            Some(ProfileState::Pending(_)) => ProbeDispatch::Descent,
            Some(ProfileState::Active(burst)) => match &burst.phase {
                BurstPhase::Rebasing => ProbeDispatch::Rebase,
                // Verifying — pre-fire stability check. Awaiting /
                // Batching / Draining never carry an in-flight probe,
                // so a response targeting them slipped past the live
                // check above — but the I5 field discipline guarantees
                // that the slot held a Verifying-minted correlation,
                // so route as Burst with the burst's recorded intent.
                BurstPhase::Verifying
                | BurstPhase::Batching { .. }
                | BurstPhase::Draining
                | BurstPhase::Awaiting { .. } => ProbeDispatch::Burst {
                    intent: burst.intent,
                    forced: burst.forced,
                },
            },
            _ => {
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    profile: profile_id,
                    correlation: received,
                });
                return;
            }
        };

        match dispatch {
            ProbeDispatch::Descent => {
                let arm = match response.result {
                    ProbeResult::Ok(tree_snap) => crate::descent::ProbeResultArm::Ok(tree_snap),
                    ProbeResult::Vanished => crate::descent::ProbeResultArm::Vanished,
                    ProbeResult::Failed { errno } => {
                        crate::descent::ProbeResultArm::Failed { errno }
                    }
                };
                self.dispatch_descent_probe(profile_id, arm, now, out);
            }
            ProbeDispatch::Rebase => match response.result {
                ProbeResult::Ok(tree_snap) => {
                    self.dispatch_rebase_ok(profile_id, tree_snap, out);
                }
                ProbeResult::Vanished => {
                    self.dispatch_rebase_vanished(profile_id, out);
                }
                ProbeResult::Failed { errno } => {
                    self.dispatch_rebase_failed(profile_id, errno, out);
                }
            },
            ProbeDispatch::Burst { intent, forced } => match (intent, response.result) {
                (BurstIntent::Seed, ProbeResult::Ok(tree_snap)) => {
                    self.dispatch_seed_ok(profile_id, tree_snap, now, out);
                }
                (BurstIntent::Seed, ProbeResult::Vanished) => {
                    self.dispatch_seed_vanished(profile_id, out);
                }
                (BurstIntent::Seed, ProbeResult::Failed { errno }) => {
                    self.dispatch_seed_failed(profile_id, errno, out);
                }
                (BurstIntent::Standard, ProbeResult::Ok(tree_snap)) => {
                    self.dispatch_standard_ok(profile_id, tree_snap, forced, now, out);
                }
                (BurstIntent::Standard, ProbeResult::Vanished) => {
                    self.dispatch_standard_vanished(profile_id, out);
                }
                (BurstIntent::Standard, ProbeResult::Failed { errno }) => {
                    self.dispatch_standard_failed(profile_id, errno, out);
                }
            },
        }
    }

    /// Dispatch a [`Input::TimerExpired`].
    ///
    /// `kind` tells us which transition this timer drives — settle expiry
    /// (Batching → Verifying) or burst-deadline expiry (force-fire). The
    /// `id` epoch survives the validation re-check that
    /// [`Engine::is_timer_referenced`] performs against the live burst
    /// slot for that `kind`; `pop_expired` already ran the same check
    /// before `step` was called, so the production path runs it twice
    /// (cheap), and any direct `step(Input::TimerExpired)` from a test
    /// or fuzzer falls through the same gate.
    pub(crate) fn on_timer_expired(
        &mut self,
        profile: ProfileId,
        kind: TimerKind,
        id: TimerId,
        out: &mut StepOutput,
    ) {
        if !Self::is_timer_referenced(&self.profiles, profile, kind, id) {
            out.diagnostics.push(Diagnostic::StaleTimer { id });
            return;
        }
        match kind {
            TimerKind::Settle => self.transition_to_verifying(profile, out),
            TimerKind::BurstDeadline => self.handle_burst_deadline(profile, out),
            TimerKind::AwaitGateDeadline => self.handle_gate_deadline(profile, out),
        }
    }

    /// Dispatch a [`Input::EffectComplete`].
    ///
    /// The Profile is resolved from `key` ([`DedupKey::profile`] is O(1)
    /// post-Phase-09 commit 2); the Sub registry is consulted only for
    /// the unknown-Sub diagnostic.
    ///
    /// Failed arrivals always clear `last_emitted_dir_hash[key]` — a
    /// failed Effect leaves no observable state to deduplicate against,
    /// so the next stable verdict at the same `DedupKey` must fire.
    /// This happens regardless of phase (Awaiting decrement, late
    /// arrival, or unknown — the cleared entry is correct in every
    /// case).
    ///
    /// The phase routing matches the fire-cycle's `Awaiting` counter:
    /// - `Active(Awaiting { outstanding > 1, .. })` ⇒ decrement.
    /// - `Active(Awaiting { outstanding ≤ 1, .. })` + `reap_pending`
    ///   ⇒ finish the burst (the deferred reap inside
    ///   `finish_burst_to_idle` runs in the same step).
    /// - `Active(Awaiting { outstanding ≤ 1, .. })` + `!reap_pending`
    ///   ⇒ `transition_to_rebasing` (post-fire probe at anchor; the
    ///   eventual response rebases `baseline := current` and finishes).
    /// - Anything else (Idle, Pending, Active in a non-Awaiting phase,
    ///   stale Profile) ⇒ `EffectCompleteOutsideAwaiting` Diagnostic.
    pub(crate) fn on_effect_complete(
        &mut self,
        sub: SubId,
        key: &DedupKey,
        result: &EffectOutcome,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        // The Sub registry is consulted only for the unknown-Sub
        // diagnostic in the `Diagnose` arm: a Sub detached mid-Awaiting
        // (the reap-pending case) is gone from the registry by the time
        // its Effects' completions arrive, but the Profile is still
        // alive and waiting for the counter to drain — we must NOT
        // short-circuit here, or the counter would never advance.
        // `key.profile()` is O(1) post-Phase-09 commit 2 and never
        // depends on the Sub registry.
        let profile_id = key.profile();

        // Failed clears the dedup entry regardless of state. The Failed
        // Effect produced no observation worth deduplicating against, so
        // the next stable verdict at the same key must fire fresh.
        if matches!(result, EffectOutcome::Failed { .. })
            && let Some(p) = self.profiles.get_mut(profile_id)
        {
            p.last_emitted_dir_hash.remove(key);
        }

        // Resolve the action under a short read borrow, then mutate.
        // Reading `reap_pending` here means the AwaitAction::Reap branch
        // sees the most recent flag value — covers the race where a Sub
        // detaches between the prior `outstanding == N` step and this
        // completion (the flag flips before this read).
        let phase_action = match self
            .profiles
            .get(profile_id)
            .map(|p| (&p.state, p.reap_pending))
        {
            Some((ProfileState::Active(burst), reap_pending)) => match &burst.phase {
                BurstPhase::Awaiting { outstanding, .. } => {
                    if *outstanding <= 1 {
                        if reap_pending {
                            AwaitAction::Reap
                        } else {
                            AwaitAction::Rebase
                        }
                    } else {
                        AwaitAction::Decrement
                    }
                }
                BurstPhase::Batching { .. }
                | BurstPhase::Verifying
                | BurstPhase::Draining
                | BurstPhase::Rebasing => AwaitAction::Diagnose,
            },
            // Idle, Pending, stale Profile (None): not waiting for this
            // completion — a late arrival the engine no longer tracks.
            _ => AwaitAction::Diagnose,
        };

        match phase_action {
            AwaitAction::Decrement => {
                if let Some(p) = self.profiles.get_mut(profile_id)
                    && let ProfileState::Active(burst) = &mut p.state
                    && let BurstPhase::Awaiting {
                        ref mut outstanding,
                        ..
                    } = burst.phase
                {
                    *outstanding = outstanding.saturating_sub(1);
                }
            }
            AwaitAction::Rebase => {
                self.transition_to_rebasing(profile_id, out);
            }
            AwaitAction::Reap => {
                // Last completion AND reap_pending: skip Rebasing — there
                // are no Subs left to fire for, so re-establishing a
                // baseline against disk reality has no consumer. Routing
                // through `finish_burst_to_idle` runs the burst-end
                // machinery (sub_suppress, propagate(-1) for Standard
                // bursts) and then dispatches `reap_profile` via the
                // reap_pending check — calling `reap_profile` directly
                // would skip those steps and leak the anchor's suppress
                // contribution.
                self.finish_burst_to_idle(profile_id, out);
            }
            AwaitAction::Diagnose => {
                // An unknown Sub at the Diagnose arm is the actionable
                // case: a completion for a Sub the engine never registered
                // (or one that was already reaped without being in a
                // burst). Reach for the Sub-keyed diagnostic since it
                // tells operators the Sub identity. With Sub still in
                // the registry, fall back to the phase-keyed
                // `EffectCompleteOutsideAwaiting` — it pairs the
                // unexpected late delivery with the owning Profile.
                if self.subs.get(sub).is_none() {
                    out.diagnostics
                        .push(Diagnostic::EffectCompleteForUnknownSub { sub });
                } else {
                    out.diagnostics
                        .push(Diagnostic::EffectCompleteOutsideAwaiting {
                            sub,
                            profile: profile_id,
                        });
                }
            }
        }
    }

    /// Dispatch a [`Input::ConfigDiff`].
    ///
    /// Atomic apply in the order **`removed → modified → added`**. Each
    /// `removed` decrements its Sub's Profile refcount (reaping the
    /// Profile if it hits zero, deferring if active); each `modified` is
    /// a remove-then-add (`config_hash` may change ⇒ different Profile);
    /// each `added` materializes the anchor and attaches the Sub.
    ///
    /// Parent-edge recompute is **lazy**: each `detach_sub_inner` /
    /// `attach_sub_inner` calls the appropriate
    /// `StabilityIndex::recompute_parent_edges_for_*` variant. All ops
    /// merge into a single sorted `StepOutput`.
    pub(crate) fn on_config_diff(
        &mut self,
        diff: SubRegistryDiff,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // 1. Removals.
        for sub_id in diff.removed {
            self.detach_sub_inner(sub_id, now, out);
        }
        // 2. Modifications: remove + add. The Sub being modified may
        // share a Profile or move to a different one (different
        // config_hash).
        for (sub_id, req) in diff.modified {
            self.detach_sub_inner(sub_id, now, out);
            let _ = self.attach_sub_inner(req, now, out);
        }
        // 3. Additions.
        for req in diff.added {
            let _ = self.attach_sub_inner(req, now, out);
        }
        // The single-StepOutput sort happens at `step`'s caller.
    }

    /// Dispatch a [`Input::WatchOpRejected`].
    ///
    /// The Sensor failed to install a kernel watch (typically `EMFILE` /
    /// `ENFILE` on FD exhaustion). Three things must happen:
    ///
    /// 1. Clamp `watch_demand := 0` and `events_union := EMPTY` on
    ///    `resource` so the engine's view of "is this slot watched?"
    ///    matches reality.
    /// 2. Walk every Profile that holds a per-Profile claim on
    ///    `resource` (anchor / watch-root parent / descent prefix) and
    ///    clean up its bookkeeping — otherwise the Profile flag
    ///    contradicts the post-clamp counter, and any subsequent
    ///    Profile-driven release path would either see the wrong union
    ///    on recompute or silently drift further out of sync.
    /// 3. Emit one `ProfileClaimPurged` Diagnostic per affected
    ///    (Profile, claim_kind) pair, plus the umbrella
    ///    `WatchOpRejected` diagnostic.
    ///
    /// A single resource may be claimed by multiple Profiles via
    /// different roles — anchor of P, watch-root parent of Q, descent
    /// prefix of R — so the fan-out walks all three claim slots
    /// independently.
    ///
    /// Stale resources (already Unwatched, queue-race) are a no-op +
    /// `WatchOpRejected` diagnostic; the per-claim walk yields nothing
    /// because Profile back-references would have been cleared at reap.
    pub(crate) fn on_watch_op_rejected(
        &mut self,
        resource: ResourceId,
        _op: WatchOp,
        failure: WatchFailure,
        out: &mut StepOutput,
    ) {
        out.diagnostics
            .push(Diagnostic::WatchOpRejected { resource, failure });

        // Snapshot every claimer BEFORE any mutation. Borrow checker
        // (we'll mutate self.profiles in the loop) and we want a stable
        // view of the pre-clamp world: a Profile that's `Pending(d)`
        // with `d.current_prefix == resource` must be detected here,
        // because the helpers we run below transition the Profile to
        // Idle.
        let mut anchor_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut parent_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut descent_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        for (pid, p) in self.profiles.iter() {
            if p.anchor_contribution && p.resource == resource {
                anchor_claimers.push(pid);
            }
            if p.watch_root_parent == Some(resource) {
                parent_claimers.push(pid);
            }
            if let ProfileState::Pending(d) = &p.state
                && d.current_prefix == resource
            {
                descent_claimers.push(pid);
            }
        }

        // Atomic counter zero. Helpers below see counter == 0 ⇒
        // flag-clear only, no `sub_watch_demand` ⇒ no underflow.
        clamp_watch_demand_to_zero(&mut self.tree, resource, out);

        // Anchor claimers: synthesise an anchor-loss. `finalize_anchor_lost`
        // cancels any in-flight Active probe, releases the anchor flag
        // (counter-aware no-op on the post-clamp counter), and finishes
        // the burst to Idle. `finish_burst_to_idle` decrements
        // `suppress_count`; the clamp deliberately does NOT zero
        // `suppress_count` so this decrement balances the burst-start's
        // `add_suppress`.
        for pid in anchor_claimers {
            self.finalize_anchor_lost(pid, out);
            out.diagnostics.push(Diagnostic::ProfileClaimPurged {
                profile: pid,
                claim: ClaimKind::Anchor,
                resource,
                failure,
            });
        }

        // Watch-root parent claimers: clear the flag. The Profile's
        // anchor stays watched (different `resource`), but auto-recovery
        // on rename / recreation is no longer possible — operator
        // restart is required to re-establish the parent watch.
        for pid in parent_claimers {
            self.release_watch_root_parent_claim(pid, out);
            out.diagnostics.push(Diagnostic::ProfileClaimPurged {
                profile: pid,
                claim: ClaimKind::WatchRootParent,
                resource,
                failure,
            });
        }

        // Descent claimers: close the probe channel (idempotent —
        // emits Cancel iff a descent probe was in flight), then release
        // the prefix claim (transitions Profile → Idle). Without the
        // cancel-before-release, a late `ProbeResponse` would arrive
        // after the Profile transitions out of Pending and drop with
        // `StaleProbeResponse` — wasted I/O.
        for pid in descent_claimers {
            self.cancel_pending_probe(pid, out);
            self.release_descent_prefix_claim(pid, out);
            out.diagnostics.push(Diagnostic::ProfileClaimPurged {
                profile: pid,
                claim: ClaimKind::DescentPrefix,
                resource,
                failure,
            });
        }
    }

    /// Sensor reports it dropped events at the kernel level (inotify's
    /// `IN_Q_OVERFLOW`). Reseed every Profile in scope so the engine's
    /// post-probe `dispatch_seed_ok` re-establishes baseline against
    /// disk reality and runs B3 drift detection (a recorded
    /// `last_emitted_dir_hash[Subtree]` disagreement fires Effects once,
    /// then rebases).
    ///
    /// # Per-Profile dispatch
    ///
    /// Each in-scope Profile is reseeded according to its current state:
    ///
    /// - **`Idle`** — direct [`Engine::start_seed_burst`]. The Profile's
    ///   `current` is preserved as the seed probe's `baseline_subtree`
    ///   for mtime-skip; the response `dispatch_seed_ok` rebases or
    ///   fires-on-drift.
    /// - **`Active(_)`** — abandon the in-flight burst via
    ///   [`Engine::finish_burst_to_idle`] (which cancels any pending
    ///   probe, decrements the anchor's `suppress_count`, and runs
    ///   `propagate(-1)` for Standard bursts including its
    ///   Draining→Verifying ancestor cascade), then start a fresh seed
    ///   burst. The Standard burst's accumulated `dirty_resources` are
    ///   discarded — the seed re-baselines against the post-overflow
    ///   tree, which strictly dominates whatever the Standard burst was
    ///   tracking. `reap_pending` Profiles reaped inside
    ///   `finish_burst_to_idle` skip the seed (no Profile to seed).
    /// - **`Pending(_)`** — descent in flight; the anchor doesn't yet
    ///   exist and the Profile holds no baseline to drift-test. Skip.
    ///   The descent's prefix watch continues to deliver future
    ///   `IN_CREATE` events; if the missed event was an `IN_CREATE` for
    ///   the next path component, the descent stalls until a future
    ///   probe / rename / fresh kernel event re-syncs. v1 limitation
    ///   accepted in exchange for handler simplicity.
    ///
    /// # Scope
    ///
    /// [`OverflowScope::Global`] (the v1 inotify backend's only emit)
    /// reseeds every Profile in the registry. [`OverflowScope::Resource`]
    /// reseeds Profiles whose anchor is `r` or a descendant of `r` —
    /// the FSEvents per-stream signal; `profiles_in_subtree(r)` walks
    /// the tree's ancestor chain to compute membership.
    ///
    /// One [`Diagnostic::SensorOverflow`] per call surfaces the event in
    /// operator logs — the bursts the reseed schedules carry no
    /// per-Profile annotation that they were triggered by overflow.
    pub(crate) fn on_sensor_overflow(
        &mut self,
        scope: OverflowScope,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Snapshot the in-scope ProfileId set BEFORE any mutation. The
        // loop below transitions Profiles through Idle and re-into
        // Active(Seed); a fresh `iter()` mid-loop would observe the
        // partial transitions and could double-handle a Profile.
        let profiles_to_reseed: smallvec::SmallVec<[ProfileId; 8]> = match scope {
            OverflowScope::Global => self.profiles.iter().map(|(pid, _)| pid).collect(),
            OverflowScope::Resource(r) => self.profiles_in_subtree(r),
        };

        for pid in profiles_to_reseed {
            // The Profile may have been reaped between snapshot and
            // this iteration via a prior iteration's
            // `finish_burst_to_idle` (a `reap_pending` Profile reaps
            // when its burst transitions to Idle). Stale id ⇒ skip.
            let Some(p) = self.profiles.get(pid) else {
                continue;
            };
            match &p.state {
                ProfileState::Idle => {
                    self.start_seed_burst(pid, now, out);
                }
                ProfileState::Active(_) => {
                    // Abandon the in-flight burst, then reseed. The two
                    // helpers compose: `finish_burst_to_idle` returns
                    // the Profile to Idle (decrementing suppress_count
                    // by one), and `start_seed_burst` adds it back —
                    // the anchor remains suppressed across the
                    // transition. The intervening Idle state is invisible
                    // to external observers (no `StepOutput` ordering
                    // dependency on it). If `finish_burst_to_idle`
                    // reaped the Profile (`reap_pending`), the
                    // `Engine::profiles.get(pid)` inside
                    // `start_seed_burst` returns None and the call
                    // no-ops — correct degenerate behaviour.
                    self.finish_burst_to_idle(pid, out);
                    self.start_seed_burst(pid, now, out);
                }
                ProfileState::Pending(_) => {
                    // Descent in flight; no baseline to drift-test.
                    // The descent's prefix watch keeps delivering
                    // future structural events; if the missed event
                    // was the IN_CREATE we were waiting for, descent
                    // stalls until a re-probe occurs through other
                    // means. Documented v1 limitation.
                }
                // ProfileState is non_exhaustive in core; absorb any
                // future variant defensively rather than panic.
                _ => {}
            }
        }

        out.diagnostics.push(Diagnostic::SensorOverflow { scope });
    }

    /// Enumerate Profiles whose anchor lies in the subtree rooted at
    /// `r` (the anchor itself is `r`, or `r` is on the anchor's
    /// ancestor chain). Used by [`Self::on_sensor_overflow`] to scope a
    /// per-resource overflow signal — the FSEvents-style "this stream's
    /// queue overflowed" case. v1 inotify always emits
    /// [`OverflowScope::Global`] so this is dead-stream-equipment in
    /// the inotify path; kept for the engine API's symmetric handling
    /// across backends.
    ///
    /// Worst-case `O(profiles × tree-depth)`. Acceptable for typical
    /// per-resource overflow rates (rare under healthy invariants).
    fn profiles_in_subtree(&self, r: ResourceId) -> smallvec::SmallVec<[ProfileId; 8]> {
        self.profiles
            .iter()
            .filter(|(_, p)| {
                p.resource == r || self.tree.ancestors(p.resource).any(|a| a == r)
            })
            .map(|(pid, _)| pid)
            .collect()
    }

    /// Start a new burst (Seed if no baseline yet, Standard if baseline
    /// established); pre-fire `Active` → fold the event through
    /// `event_drives_batching` (which accumulates `dirty_resources` +
    /// `force_walk_resources`, emits a Cancel iff a probe was in flight,
    /// and arms a fresh settle timer); post-fire `Active`
    /// (`Awaiting` / `Rebasing`) → absorb the event with a diagnostic.
    ///
    /// `event_resource` is the `FsEvent`'s source. It seeds (Idle path)
    /// or accumulates (pre-fire Active path) the event-tracking sets
    /// the next probe uses to compute LCA + `force_walk`. The post-fire
    /// absorb path does not extend either set: the Rebasing probe at
    /// the anchor captures whatever's on disk regardless, and a fresh
    /// burst against an in-flight one would corrupt the fire-tail.
    ///
    /// `event` is threaded through purely for the absorb diagnostic so
    /// the operator can correlate logs to the dropped FsEvent.
    ///
    /// The "no baseline → Seed" branch handles the degenerate
    /// post-`Vanished` Idle state where `current.is_none()` — a Standard
    /// burst without a baseline cannot dispatch its stability verdict
    /// meaningfully.
    fn drive_burst(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        event: FsEvent,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        match &p.state {
            ProfileState::Idle => {
                if p.current.is_some() {
                    self.start_standard_burst(profile_id, event_resource, now, out);
                } else {
                    self.start_seed_burst(profile_id, now, out);
                }
            }
            ProfileState::Active(burst) => match &burst.phase {
                BurstPhase::Awaiting { .. } | BurstPhase::Rebasing => {
                    out.diagnostics.push(Diagnostic::EventAbsorbedByFireTail {
                        profile: profile_id,
                        resource: event_resource,
                        event,
                    });
                }
                BurstPhase::Batching { .. } | BurstPhase::Verifying | BurstPhase::Draining => {
                    self.event_drives_batching(profile_id, event_resource, now, out);
                }
            },
            // `ProfileState` non_exhaustive: Pending Profiles never reach
            // here — `covering_profiles` filters them at the source — but
            // the wildcard arm absorbs both Pending (defensively) and any
            // future variant.
            _ => {}
        }
    }

    /// Anchor terminal event (Removed/Renamed/Revoked at `Profile.resource`).
    /// Thin wrapper over `finalize_anchor_lost` — the FsEvent dispatcher
    /// and the WatchOpRejected purge share the same "anchor's FD is gone,
    /// finalize the burst" logic.
    fn on_anchor_terminal_event(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        self.finalize_anchor_lost(profile_id, out);
    }

    /// Finalize the loss of a Profile's anchor: cancel any in-flight
    /// probe, release the anchor's `watch_demand` contribution, drop the
    /// stale `baseline` / `current` snapshots, and finish the burst to
    /// Idle if Active.
    ///
    /// **Ordering.** The anchor release runs BEFORE `finish_burst_to_idle`,
    /// so any deferred `reap_profile` (`reap_pending`) sees a cleared
    /// `anchor_contribution` flag and skips its redundant release inside
    /// `reap_profile::release_anchor_claim`. This mirrors the
    /// `dispatch_*_vanished/failed` discipline.
    /// Reverse-ordering would have `finish_burst_to_idle` invoke
    /// `reap_profile`, which would release the anchor; the post-`finish`
    /// release would then see a counter that's already zero and (pre
    /// counter-existence-check) underflow `sub_watch_demand`. The helper
    /// + ordering combination removes both failure modes.
    ///
    /// **Pending exclusion.** `ProfileState::Pending` is defensive here
    /// — `covering_profiles` already filters Pending Profiles at the
    /// source, so the FsEvent path can't deliver a Pending Profile.
    /// `on_watch_op_rejected` calls this directly after iterating the
    /// full registry, where the guard does load-bearing work: a
    /// Pending Profile carries no anchor contribution and participates
    /// in no burst-suppress accounting, so `finish_burst_to_idle`'s
    /// `sub_suppress` would underflow.
    pub(crate) fn finalize_anchor_lost(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        if matches!(p.state, ProfileState::Pending(_)) {
            return;
        }
        let was_active = matches!(p.state, ProfileState::Active(_));

        // Idempotent: emits Cancel iff the probe channel is open
        // (Active+Verifying ⇒ pending_probe = Some(_)). For
        // Active+Batching/Draining no probe is in flight and the helper
        // is a no-op — replaces the prior `was_verifying` snapshot's
        // role with field-discipline equivalence.
        self.cancel_pending_probe(profile_id, out);

        // Release per-descendant `watch_demand` contributions — the
        // helper take-and-walks `Profile.current`, decrementing each
        // covered Tree slot's counter. Must run BEFORE the anchor and
        // burst-end paths so the recompute sees this Profile's
        // descendant claims as gone (closes F-CRIT-1). The take leaves
        // `current = None`, redundant with the explicit clear below
        // but kept for clarity in the snapshot-drop semantic.
        self.release_descendant_claim(profile_id, out);

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            // current is already None — release_descendant_claim took it.
        }

        // Release BEFORE finish_burst_to_idle. See the ordering note
        // above.
        self.release_anchor_claim(profile_id, out);

        if was_active {
            self.finish_burst_to_idle(profile_id, out);
        }
    }

    /// (Seed, Ok).
    ///
    /// Graft the response into `Profile.current` at the burst's
    /// `probe_target` (= anchor for Seeds). Bundle B3 (hash-only): if the
    /// post-graft `current` diverges from a recorded
    /// `last_emitted_dir_hash[Subtree]` for this Profile, fire
    /// `emit_effects` once and route through the same fire-tail as a
    /// Standard burst (`emit_effects` count > 0 ⇒ `transition_to_awaiting`;
    /// the eventual rebase probe captures the post-command tree).
    /// Otherwise rebase directly: `baseline := current` and finish.
    ///
    /// Fresh-attach Seed cannot enter the drift branch — `last_emitted_dir_hash`
    /// is empty by construction at fresh attach, so `b3_seed_drift_observed`
    /// returns false. The drift branch fires only on recovery / post-Effect
    /// rebase paths where the Profile has already emitted at least one
    /// Subtree key.
    fn dispatch_seed_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Seed always targets anchor — `probe_target` was set to the
        // anchor at `start_seed_burst` / `transition_to_verifying`.
        let target = match self.profiles.get(profile_id) {
            Some(p) => match &p.state {
                ProfileState::Active(b) => b.probe_target.unwrap_or(p.resource),
                _ => p.resource,
            },
            None => return,
        };

        match snapshot {
            TreeSnapshot::Dir(arc) => {
                graft(
                    profile_id,
                    target,
                    arc,
                    &mut self.tree,
                    &mut self.profiles,
                    out,
                );
            }
            TreeSnapshot::File(leaf) => {
                // File-anchored Profile: the leaf *is* the snapshot. No
                // graft, no walk_pair (a Leaf has no descendants to
                // materialise).
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.current = Some(TreeSnapshot::File(leaf));
                }
            }
        }

        // Bundle B3 — fire Effects only for the Subtree keys that drifted
        // since the last successful emission. emit_effects' B1 path runs
        // against the freshly grafted current and updates
        // `last_emitted_dir_hash` to the new post-fire hash. The rebase
        // (`baseline := current`) happens in both branches below; on the
        // drift-fires branch it must run before `transition_to_awaiting`
        // so the Profile's view is consistent for any FsEvent absorbed
        // during the post-fire tail (Awaiting/Rebasing).
        let drifted_keys = self.b3_seed_drift_observed(profile_id);
        if !drifted_keys.is_empty() {
            let outcome = self.emit_effects(profile_id, false, Some(&drifted_keys), out);
            if outcome.count > 0 {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.baseline = p.current.clone();
                }
                self.transition_to_awaiting(profile_id, outcome.count, now);
                return;
            }
            // Drift observed but emit produced no effects: the drifted
            // Subs were detached between record and now (their entries
            // would normally be purged on detach, but a same-step race
            // reaches here defensively). Fall through to the finish path.
        }

        // Non-drift Seed (fresh attach, no-drift recovery, or B1-suppressed
        // drift): rebase and finish. No Effect fires, no Awaiting tail.
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = p.current.clone();
        }
        self.finish_burst_to_idle(profile_id, out);
    }

    /// Bundle B3 — per-key hash-only drift check at Seed-Ok. Returns the
    /// `DedupKey::Subtree` keys whose recorded `last_emitted_dir_hash`
    /// differs from the post-graft `current`'s anchor-rooted hash. Empty
    /// vec means "no drift" — fresh-Profile Seed (no prior emission) ⇒
    /// `last_emitted_dir_hash` empty ⇒ empty result ⇒ no fire, preserving
    /// "fresh Seed never fires Effect".
    ///
    /// Per-key scoping (vs prior bool-OR-across-keys design): a multi-Sub
    /// Profile in recovery only re-fires the Subs whose own emission
    /// records have drifted. A Sub whose `last_emitted_dir_hash[Subtree]`
    /// matches the post-recovery hash is skipped — its previous fire is
    /// still consistent with disk reality, so re-running its command
    /// would be a noop with side effects.
    ///
    /// Limitation: `DedupKey::PerFile` entries are not drift sources. The
    /// post-Seed `current` lacks the per-leaf history for a faithful
    /// per-file diff. The dispatcher passes the returned filter to
    /// `emit_effects`, which then skips PerStableFile Subs entirely on
    /// the drift path — PerFile keys fire only via Standard bursts, never
    /// from B3.
    fn b3_seed_drift_observed(&self, profile_id: ProfileId) -> SmallVec<[DedupKey; 2]> {
        let Some(p) = self.profiles.get(profile_id) else {
            return SmallVec::new();
        };
        if p.last_emitted_dir_hash.is_empty() {
            return SmallVec::new();
        }
        let curr_hash: u128 = match p.current.as_ref() {
            Some(TreeSnapshot::Dir(arc)) => arc.dir_hash(),
            Some(TreeSnapshot::File(leaf)) => leaf.leaf_hash(),
            None => return SmallVec::new(),
        };
        p.last_emitted_dir_hash
            .iter()
            .filter_map(|(key, &h)| {
                if matches!(key, DedupKey::Subtree { .. }) && h != curr_hash {
                    Some(key.clone())
                } else {
                    None
                }
            })
            .collect()
    }

    /// (Seed, Vanished).
    ///
    /// Symmetric with `dispatch_standard_vanished` (treats Vanished as an
    /// anchor-disappearance signal): releases the anchor's `watch_demand`
    /// contribution so the trichotomy invariant in `reap_profile` —
    /// `!(Pending && anchor_contribution)` — survives the eventual
    /// `start_pending_recovery` transition.
    ///
    /// Recovery does not depend on the anchor's FD: the kqueue
    /// registration auto-detached on the inode disappearing, and
    /// re-acquisition flows through `watch_root_parent`'s
    /// `StructureChanged` → `start_pending_recovery` → descent →
    /// `dispatch_descent_ok` (anchor materialization, which re-bumps
    /// `anchor.watch_demand` with the Profile's mask).
    fn dispatch_seed_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Seed,
        });
        // Release the per-descendant `watch_demand` contributions
        // encoded in `Profile.current` (F-CRIT-1). The helper takes
        // `current`, walks it, and runs the per-file dedup-hygiene
        // purge. Must run BEFORE `release_anchor_claim` so the
        // recompute (multi-Profile case) sees this Profile's
        // descendant claims as released.
        self.release_descendant_claim(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            // current already None — release_descendant_claim took it.
        }
        // Release BEFORE finish_burst_to_idle so any deferred
        // `reap_profile` (reap_pending) sees a cleared flag — preserves
        // the trichotomy invariant `!(Pending && anchor_contribution)`
        // across the eventual `start_pending_recovery` transition.
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Seed, Failed).
    ///
    /// Symmetric with `dispatch_standard_failed`: the probe failed at the
    /// anchor; release the anchor's `watch_demand` contribution. See
    /// `dispatch_seed_vanished` for the trichotomy-invariant rationale.
    fn dispatch_seed_failed(&mut self, profile_id: ProfileId, errno: i32, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent: BurstIntent::Seed,
            errno,
        });
        self.release_descendant_claim(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
        }
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Standard, Ok).
    ///
    /// Stability verdict is **one `dir_hash` (or `leaf_hash`) comparison**
    /// between the response and `current.subtree_at(target)`. The verdict
    /// is computed BEFORE graft (post-graft comparison would always be
    /// true; graft just put response there). A target with no prior subtree
    /// is conservatively treated as not-stable — there's no "prior probe at
    /// this target" to compare against; the next probe converges.
    fn dispatch_standard_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        forced: bool,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Determine target + pre-graft prior hash at target.
        let (target, prior_target_hash, dirty_zero) = match self.profiles.get(profile_id) {
            Some(p) => {
                let target = match &p.state {
                    ProfileState::Active(b) => b.probe_target.unwrap_or(p.resource),
                    _ => p.resource,
                };
                let prior_hash = crate::reconcile::current_target_hash(p, target, &self.tree);
                (target, prior_hash, p.dirty_descendants == 0)
            }
            None => return,
        };

        // Stability hash from response.
        let response_hash = match &snapshot {
            TreeSnapshot::Dir(arc) => Some(arc.dir_hash()),
            TreeSnapshot::File(leaf) => Some(leaf.leaf_hash()),
        };
        let is_stable = match (prior_target_hash, response_hash) {
            (Some(a), Some(b)) => a == b,
            _ => false,
        };

        // Graft AFTER computing stability — the verdict needs the
        // pre-update prior. graft calls walk_pair (Watch ops) + splice
        // (current update). For File anchors, replace wholesale.
        match snapshot {
            TreeSnapshot::Dir(arc) => {
                graft(
                    profile_id,
                    target,
                    arc,
                    &mut self.tree,
                    &mut self.profiles,
                    out,
                );
            }
            TreeSnapshot::File(leaf) => {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.current = Some(TreeSnapshot::File(leaf));
                }
            }
        }

        if is_stable && dirty_zero {
            // Row 3: stable + dirty=0 → fire Effect. Awaiting on count > 0;
            // finish-to-Idle on count == 0 (B1 suppressed everything, no
            // Subs matched, or `reap_pending` skipped the emit). baseline
            // is NOT pinned here on the firing branch — it will rebase
            // when the Rebasing probe response lands (`dispatch_rebase_ok`).
            // No drift filter — Standard bursts emit for every matching Sub.
            let outcome = self.emit_effects(profile_id, forced, None, out);
            if outcome.count > 0 {
                self.transition_to_awaiting(profile_id, outcome.count, now);
            } else {
                self.finish_burst_to_idle(profile_id, out);
            }
        } else if is_stable {
            // Row 4: stable + dirty>0 → Draining. The stable snapshot lives
            // on `Profile.current` (just spliced in by graft); the reconfirm
            // probe compares against `current`. No need to pin a duplicate
            // snapshot on the phase variant.
            self.transition_to_draining(profile_id);
        } else if forced {
            // Row 5: not-stable + forced → fire Effect with forced=true.
            // Same Awaiting / finish-to-Idle branching as row 3 — `forced`
            // overrides B1 inside `emit_effects`, but a Profile with no
            // matching Subs still returns count == 0.
            let outcome = self.emit_effects(profile_id, true, None, out);
            if outcome.count > 0 {
                self.transition_to_awaiting(profile_id, outcome.count, now);
            } else {
                self.finish_burst_to_idle(profile_id, out);
            }
        } else {
            // Row 5 else: not-stable + !forced → re-arm debounce in
            // `Batching`. By construction no probe is in flight (we're
            // inside the response handler), so no Cancel is emitted.
            self.unstable_response_drives_batching(profile_id, now);
        }
    }

    /// (Standard, Vanished).
    ///
    /// Treat as Removed at anchor: release the anchor's `watch_demand`
    /// contribution. Standard bursts always run on materialized Profiles
    /// (`drive_burst` routes baseline-less `FsEvent`s to Seed instead), so
    /// the guard is effectively unconditional in v1 — kept for robustness
    /// against future routing changes.
    ///
    /// Release runs BEFORE `finish_burst_to_idle` so any deferred
    /// `reap_profile` (`reap_pending`) sees `anchor_contribution=false`
    /// and skips a redundant release. Without this ordering the post-
    /// `finish` release would underflow the now-zero `watch_demand`
    /// counter (debug-assert panic; release-build silent leak).
    fn dispatch_standard_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Standard,
        });
        self.release_descendant_claim(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
        }
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Standard, Failed).
    ///
    /// See `dispatch_standard_vanished` for the release-before-finish
    /// ordering rationale.
    fn dispatch_standard_failed(
        &mut self,
        profile_id: ProfileId,
        errno: i32,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent: BurstIntent::Standard,
            errno,
        });
        self.release_descendant_claim(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
        }
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Rebase, Ok). Post-fire probe response — graft the post-command
    /// snapshot into `Profile.current`, rebase `baseline := current`,
    /// finish the burst to Idle. The Rebasing probe always targets the
    /// anchor (set by `transition_to_rebasing`); no stability verdict
    /// applies (we just fired, drift is expected).
    ///
    /// **No B3.** Recovery / post-Effect drift detection is gated on
    /// Seed-Ok in v1; Rebasing is a phase of the Standard burst (or
    /// the Seed burst's drift tail), not a fresh Seed, so the B3 hash
    /// check would either fire-loop (every fire writes a new hash;
    /// the next rebase would see drift; loop) or be silently a no-op
    /// (the post-fire hash matches itself by construction). The
    /// helper deliberately avoids `b3_seed_drift_observed` here.
    fn dispatch_rebase_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        out: &mut StepOutput,
    ) {
        let target = match self.profiles.get(profile_id) {
            Some(p) => match &p.state {
                ProfileState::Active(b) => b.probe_target.unwrap_or(p.resource),
                _ => p.resource,
            },
            None => return,
        };
        match snapshot {
            TreeSnapshot::Dir(arc) => {
                graft(
                    profile_id,
                    target,
                    arc,
                    &mut self.tree,
                    &mut self.profiles,
                    out,
                );
            }
            TreeSnapshot::File(leaf) => {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.current = Some(TreeSnapshot::File(leaf));
                }
            }
        }
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = p.current.clone();
        }
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Rebase, Vanished). Anchor disappeared between fire and rebase.
    /// Symmetric path with `dispatch_standard_vanished`: clear baseline /
    /// current, release the anchor watch contribution, finish the burst.
    /// Diagnostic carries the burst's actual intent so logs can
    /// distinguish Seed-driven (B3 drift) vs Standard-driven Rebasing;
    /// the lookup falls back to `Standard` only on a stale-Profile or
    /// non-Active defensive path (the routing in `on_probe_response`
    /// guarantees `Active(Rebasing)` at entry).
    fn dispatch_rebase_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        let intent = self.rebase_burst_intent(profile_id);
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent,
        });
        self.release_descendant_claim(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
        }
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Rebase, Failed). Probe failed at the anchor between fire and
    /// rebase. Same shape as `dispatch_rebase_vanished` — clear,
    /// release, finish. Diagnostic carries the burst's actual intent
    /// (Standard fallback on the same defensive path noted there).
    fn dispatch_rebase_failed(&mut self, profile_id: ProfileId, errno: i32, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        let intent = self.rebase_burst_intent(profile_id);
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent,
            errno,
        });
        self.release_descendant_claim(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
        }
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// Resolve the intent of the burst owning the in-flight Rebase
    /// probe. Returns the live `Burst.intent` when the Profile is
    /// `Active(_)` (the production path). Defensive fallback to
    /// [`BurstIntent::Standard`] for the structurally-unreachable
    /// non-Active branch — the `on_probe_response` routing dispatches
    /// `dispatch_rebase_*` only on `BurstPhase::Rebasing`, and that
    /// phase is reachable only from Active. Standard is the right
    /// default because Rebasing is overwhelmingly a Standard-burst tail
    /// (Seed-driven Rebasing requires a recovery + B3 drift, the rare
    /// path).
    fn rebase_burst_intent(&self, profile_id: ProfileId) -> BurstIntent {
        self.profiles
            .get(profile_id)
            .and_then(|p| match &p.state {
                ProfileState::Active(b) => Some(b.intent),
                _ => None,
            })
            .unwrap_or(BurstIntent::Standard)
    }

    /// `burst_deadline` row — sets `forced := true` and either
    /// transitions the phase (Batching/Draining → Verifying) or, if a
    /// probe is already in flight (Verifying), waits for the response.
    ///
    /// Reads phase inline while flipping `forced`: the caller has
    /// already validated the timer is live (via `is_timer_referenced`),
    /// which restricts to pre-fire phases — `is_timer_referenced` for
    /// `BurstDeadline` returns false in `Awaiting` / `Rebasing`, so a
    /// stale fire never reaches here.
    fn handle_burst_deadline(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let needs_verify = if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            match &burst.phase {
                BurstPhase::Batching { .. } | BurstPhase::Draining => {
                    burst.forced = true;
                    true
                }
                // Verifying: probe in flight; no second emission. The
                // response, when it arrives, dispatches with
                // `forced = true`.
                BurstPhase::Verifying => {
                    burst.forced = true;
                    false
                }
                // Awaiting / Rebasing: defense-in-depth no-op. The
                // is_timer_referenced gate filters BurstDeadline out of
                // post-fire phases, so the timer never fires here in
                // production. If a future caller bypasses the gate (e.g.,
                // a direct `step(Input::TimerExpired)` from a fuzzer),
                // we still don't want to flip forced or transition —
                // both would corrupt the in-flight fire-tail.
                BurstPhase::Awaiting { .. } | BurstPhase::Rebasing => false,
            }
        } else {
            return;
        };
        if needs_verify {
            self.transition_to_verifying(profile_id, out);
        }
    }

    /// `gate_deadline` row — actuator-hang recovery. Force-transitions
    /// the burst from `Awaiting` to `Rebasing`. Late `EffectComplete`
    /// arrivals (after this transition) land in
    /// [`Diagnostic::EffectCompleteOutsideAwaiting`].
    ///
    /// Defensive: if the phase has already advanced (e.g., a race with
    /// `finalize_anchor_lost`), the helper no-ops. The
    /// `is_timer_referenced` gate already filters most non-Awaiting
    /// fires; this guard handles the residual same-step ordering window.
    fn handle_gate_deadline(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let outstanding = match self.profiles.get(profile_id).map(|p| &p.state) {
            Some(ProfileState::Active(b)) => match &b.phase {
                BurstPhase::Awaiting { outstanding, .. } => *outstanding,
                _ => return,
            },
            _ => return,
        };
        out.diagnostics.push(Diagnostic::AwaitGateDeadlineElapsed {
            profile: profile_id,
            outstanding,
        });
        self.transition_to_rebasing(profile_id, out);
    }

    /// Emit Effects at a Standard burst's stable verdict. Routes per scope:
    /// `SubtreeRoot` Subs fire one Effect anchored at the Profile's resource;
    /// `PerStableFile` Subs fire one Effect per matching diff entry. The
    /// `Diff` is built at most once and shared across both helpers via `Arc`.
    ///
    /// `drift_filter` narrows emission on the Seed-drift path: when `Some`,
    /// only `SubtreeRoot` Subs whose `DedupKey::Subtree` is in the filter
    /// fire, and PerStableFile Subs are skipped entirely (B3 is a Subtree-
    /// only drift signal — see `b3_seed_drift_observed`). On the Standard
    /// burst path the filter is `None` and every matching Sub emits.
    ///
    /// `Profile.reap_pending` suppresses all emission — the Profile is on its
    /// way out and any remaining Subs (none, by construction of
    /// `reap_pending = sub_refcount == 0`) would fire against a Sub registry
    /// that no longer holds them.
    ///
    /// Returns an [`EmitOutcome`] whose `count` is the number of Effects
    /// pushed onto `out.effects`. Callers consume this to decide whether
    /// to enter the `Awaiting` phase (`count > 0`) or short-circuit to
    /// `finish_burst_to_idle` (B1 suppressed everything, no Subs matched,
    /// or `reap_pending`).
    fn emit_effects(
        &mut self,
        profile_id: ProfileId,
        forced: bool,
        drift_filter: Option<&[DedupKey]>,
        out: &mut StepOutput,
    ) -> EmitOutcome {
        let Some(p) = self.profiles.get(profile_id) else {
            return EmitOutcome::default();
        };
        if p.reap_pending {
            return EmitOutcome::default();
        }
        let resource = p.resource;
        let baseline_snap = p.baseline.clone();
        let current_snap = p.current.clone();
        let pattern = p.config.pattern.clone();

        let anchor_path = self.tree.path_of(resource).unwrap_or_default();
        let anchor_kind = self
            .tree
            .get(resource)
            .map_or(ResourceKind::Unknown, |r| r.kind);
        let anchor_cwd = compute_cwd(&anchor_path, anchor_kind);

        // Lazy-build the Diff Arc only if any Sub needs it AND both a
        // baseline and a current snapshot are present. With baseline pinned
        // across coalesced bursts, `Effect.diff` describes the *net* change
        // since the last EffectComplete::Ok.
        let mut diff_arc: Option<Arc<specter_core::Diff>> = None;
        let ensure_diff = |diff_slot: &mut Option<Arc<specter_core::Diff>>| {
            if diff_slot.is_none()
                && let (Some(b), Some(c)) = (baseline_snap.as_ref(), current_snap.as_ref())
            {
                *diff_slot = Some(Arc::new(specter_core::diff_tree(b, c)));
            }
            diff_slot.clone()
        };

        // Snapshot the post-graft `current` hash once for B1 SubtreeRoot
        // suppression. PerStableFile uses per-leaf hashes (computed inside
        // `emit_effects_per_stable_file`).
        let current_dir_hash: u128 = current_snap.as_ref().map_or(0, |s| match s {
            TreeSnapshot::Dir(arc) => arc.dir_hash(),
            TreeSnapshot::File(leaf) => leaf.leaf_hash(),
        });

        // Snapshot the Sub IDs to avoid holding `&self.subs` across the
        // loop body's `out.effects.push`.
        let sub_ids: Vec<SubId> = self.subs.at(profile_id).to_vec();
        let mut count: u32 = 0;
        for sub_id in sub_ids {
            let (scope, needs_diff, log_output) = match self.subs.get(sub_id) {
                Some(s) => (s.scope, s.needs_diff, s.log_output),
                None => continue,
            };
            match scope {
                EffectScope::SubtreeRoot => {
                    let dk = DedupKey::Subtree {
                        sub: sub_id,
                        profile: profile_id,
                    };
                    // Drift filter (Seed-drift path): emit only when this
                    // Sub's `Subtree` key is in the requested set. The
                    // Standard burst path passes `None` and emits
                    // unconditionally (modulo B1 below).
                    if let Some(allowed) = drift_filter
                        && !allowed.contains(&dk)
                    {
                        continue;
                    }
                    // Bundle B1: suppress when the post-burst hash equals
                    // the hash we last fired against for this DedupKey AND
                    // the burst is not forced. `forced=true` is the
                    // "max-settle elapsed; give up and run" path —
                    // suppressing it would lie about progress.
                    let suppress = !forced
                        && self
                            .profiles
                            .get(profile_id)
                            .and_then(|p| p.last_emitted_dir_hash.get(&dk))
                            == Some(&current_dir_hash);
                    if suppress {
                        continue;
                    }

                    let diff_for_effect = if needs_diff {
                        ensure_diff(&mut diff_arc)
                    } else {
                        None
                    };
                    let correlation = self.next_effect_correlation();
                    let Some(sub) = self.subs.get(sub_id) else {
                        continue;
                    };
                    let (command, env) = specter_core::resolve_effect(
                        sub,
                        &anchor_path,
                        &anchor_path,
                        "",
                        forced,
                        correlation,
                        diff_for_effect.as_deref(),
                    );
                    out.effects.push(Effect {
                        key: dk.clone(),
                        command,
                        env,
                        cwd: anchor_cwd.clone(),
                        forced,
                        correlation,
                        diff: diff_for_effect,
                        capture_output: log_output,
                    });
                    count = count.saturating_add(1);

                    // Record the post-fire hash so the next stable verdict
                    // can suppress an idempotent re-fire.
                    if let Some(p) = self.profiles.get_mut(profile_id) {
                        p.last_emitted_dir_hash.insert(dk, current_dir_hash);
                    }
                }
                EffectScope::PerStableFile => {
                    // B3 is Subtree-only — PerFile keys are not drift
                    // sources (per the helper's documented limitation).
                    // On the Seed-drift path the filter is `Some` and
                    // PerStableFile Subs do not fire; PerFile keys reach
                    // the actuator only via Standard bursts.
                    if drift_filter.is_some() {
                        continue;
                    }
                    // PerStableFile implies `needs_diff = true` at Sub::new;
                    // diff is always built.
                    let Some(diff) = ensure_diff(&mut diff_arc) else {
                        continue;
                    };
                    let pushed = self.emit_effects_per_stable_file(
                        sub_id,
                        resource,
                        forced,
                        pattern.as_ref(),
                        &diff,
                        &anchor_path,
                        &anchor_cwd,
                        out,
                        current_snap.as_ref(),
                    );
                    count = count.saturating_add(pushed);
                }
            }
        }
        EmitOutcome { count }
    }

    /// Per-Diff-entry Effect emission for a `PerStableFile` Sub. Walks
    /// `created`, `modified`, and `renamed.to`; deleted entries do **not**
    /// fire (running a per-file command on a deleted file makes no sense).
    /// The pattern filter is the Profile's `ScanConfig.pattern` — multiple
    /// Subs sharing one Profile share its pattern by design.
    ///
    /// Resource materialization: the diff entry's slot is resolved via
    /// `reconcile`'s `lookup_descendant`-style walk; if the slot isn't yet
    /// in the Tree (defensive — reconcile runs before this and materializes
    /// covered entries), a fresh Resource is created with no `watch_demand`
    /// contribution.
    ///
    /// Returns the number of Effects pushed to `out.effects`. The caller
    /// (`emit_effects`) sums this into the [`EmitOutcome.count`] it returns.
    #[must_use]
    fn emit_effects_per_stable_file(
        &mut self,
        sub_id: SubId,
        anchor: ResourceId,
        forced: bool,
        pattern: Option<&specter_core::GlobPattern>,
        diff: &Arc<specter_core::Diff>,
        anchor_path: &Path,
        anchor_cwd: &Path,
        out: &mut StepOutput,
        current: Option<&TreeSnapshot>,
    ) -> u32 {
        let profile_id = match self.subs.get(sub_id) {
            Some(s) => s.profile,
            None => return 0,
        };
        let mut count: u32 = 0;

        // Collect matching segments + kinds in a single pass, in the order
        // expected — created, then modified, then renamed.to.
        // `EntryRef` carries `kind`; pattern matching applies to Files only
        // (Dirs bypass the pattern per the `covers` predicate).
        let entries = diff
            .created
            .iter()
            .chain(diff.modified.iter())
            .chain(diff.renamed.iter().map(|r| &r.to));

        for entry in entries {
            // PerStableFile is per-FILE: skip Dir and Other (devices /
            // sockets / fifos) entirely — running a per-file command on a
            // directory or device is never the user's intent. Symlinks
            // pass through (they target files in practice).
            if !matches!(
                entry.kind,
                specter_core::EntryKind::File | specter_core::EntryKind::Symlink
            ) {
                continue;
            }
            if let Some(pat) = pattern {
                let path = std::path::PathBuf::from(entry.segment.as_str());
                if !pat.matches_path(&path) {
                    continue;
                }
            }
            // `walk_pair`/`graft` runs before this and materialises every
            // covered diff entry; lookup is the happy path. Fall back to
            // `ensure_descendant` for defense — covers the rare case where
            // reconcile filtered the entry but the Sub's pattern matches
            // it (e.g., reconcile gates Watch on Dir, not on
            // pattern-matching files).
            let resource = match lookup_descendant(&self.tree, anchor, entry.segment.as_str()) {
                Some(r) => r,
                None => match ensure_descendant(
                    &mut self.tree,
                    anchor,
                    entry.segment.as_str(),
                    kind_from_entry(entry.kind),
                ) {
                    Some(r) => r,
                    None => continue,
                },
            };

            let dk = DedupKey::PerFile {
                sub: sub_id,
                profile: profile_id,
                resource,
            };
            // Bundle B1 per-leaf suppression. `lookup_leaf_hash_in_current`
            // returns `None` when current's per-leaf hash isn't reachable
            // (rare; defense-in-depth) — fire conservatively in that case
            // (correctness over efficiency).
            let leaf_hash = lookup_leaf_hash_in_current(current, entry.segment.as_str());
            let suppress = !forced
                && leaf_hash.is_some()
                && self
                    .profiles
                    .get(profile_id)
                    .and_then(|p| p.last_emitted_dir_hash.get(&dk))
                    == leaf_hash.as_ref();
            if suppress {
                continue;
            }

            let target_path = anchor_path.join(entry.segment.as_str());
            let target_rel = entry.segment.as_str();
            let correlation = self.next_effect_correlation();
            // The Sub may have been removed mid-burst; defensive lookup.
            let Some(sub) = self.subs.get(sub_id) else {
                continue;
            };
            let log_output = sub.log_output;
            let (command, env) = specter_core::resolve_effect(
                sub,
                anchor_path,
                &target_path,
                target_rel,
                forced,
                correlation,
                Some(diff),
            );
            out.effects.push(Effect {
                key: dk.clone(),
                command,
                env,
                cwd: anchor_cwd.to_path_buf(),
                forced,
                correlation,
                diff: Some(diff.clone()),
                capture_output: log_output,
            });
            count = count.saturating_add(1);

            // Bundle B1: record the post-fire leaf hash so the next stable
            // verdict at the same DedupKey can suppress an idempotent
            // re-fire. Only insert when we have a real leaf hash; the
            // None-fallback above is intentionally not memoised (we want
            // the next probe to fire too).
            if let Some(h) = leaf_hash
                && let Some(p) = self.profiles.get_mut(profile_id)
            {
                p.last_emitted_dir_hash.insert(dk, h);
            }
        }
        count
    }

    /// Walk `resource` and its strict ancestors looking for Profiles whose
    /// `covers` predicate accepts `resource`. Returns the matching
    /// Profiles in encounter order. P4 single-Profile resolves to 0 or 1.
    ///
    /// **Pending Profiles are filtered at the source.** A Pending
    /// Profile's anchor (`Profile.resource`) is `DescentScaffold`-roled
    /// and carries no `watch_demand` from this Profile — the descent
    /// prefix carries it instead. Events at the prefix route via
    /// `descents_at_prefix` / `on_descent_event`; events at the anchor
    /// or its descendants are structurally unreachable in production
    /// (the anchor's `watch_demand` is 0 ⇒ head guard short-circuits).
    /// Filtering here makes the routing contract explicit:
    /// covering-Profile dispatch (Standard burst, anchor terminal event)
    /// only sees Profiles with a materialized anchor.
    fn covering_profiles(&self, resource: ResourceId) -> smallvec::SmallVec<[ProfileId; 2]> {
        let mut out: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut cur = Some(resource);
        while let Some(rid) = cur {
            for pid in self.profiles.at(rid) {
                let Some(p) = self.profiles.get(pid) else {
                    continue;
                };
                if matches!(p.state, ProfileState::Pending(_)) {
                    continue;
                }
                if crate::coverage::covers(p, resource, &self.tree) && !out.contains(&pid) {
                    out.push(pid);
                }
            }
            cur = self.tree.parent(rid);
        }
        out
    }

    /// Mint a fresh `CorrelationId` for an Effect. Engine-monotonic, sharing
    /// the same `Engine.next_correlation` counter as
    /// [`Engine::mint_probe_correlation`] — the typed wrappers
    /// ([`CorrelationId`] vs `ProbeCorrelation`) keep the spaces disjoint.
    const fn next_effect_correlation(&mut self) -> CorrelationId {
        self.next_correlation = self.next_correlation.saturating_add(1);
        CorrelationId(self.next_correlation)
    }
}

/// Outcome of an [`Engine::emit_effects`] call. `count` is the number of
/// `out.effects.push(...)` invocations that survived B1 suppression and
/// Sub-scope routing — i.e., Effects that the Actuator will actually run.
///
/// `dispatch_*_ok` consumes this to decide whether the Profile should
/// enter the `Awaiting` phase (count > 0, at least one Effect is in
/// flight) or short-circuit to `finish_burst_to_idle` (count == 0: B1
/// suppressed every emission, no Subs matched, or `reap_pending` was
/// set). The `#[must_use]` attribute prevents a future caller from
/// silently dropping the count and re-introducing the post-emit
/// "Idle-but-Effects-in-flight" leakage.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
#[must_use]
pub(crate) struct EmitOutcome {
    pub count: u32,
}

/// Routing classifier for [`Engine::on_effect_complete`]. Computed under
/// a short read borrow on `self.profiles`, then dispatched under
/// `&mut self`. The four arms cover every legitimate outcome:
///
/// - `Decrement`: Awaiting with `outstanding > 1`. Subtract one and
///   stay in Awaiting; more completions are still in flight.
/// - `Rebase`: Awaiting with `outstanding ≤ 1` and `!reap_pending`.
///   Last completion arrived; transition to Rebasing to capture the
///   post-command tree as the new baseline.
/// - `Reap`: Awaiting with `outstanding ≤ 1` and `reap_pending`. Last
///   completion arrived AND the Profile lost its last Sub mid-burst;
///   skip Rebasing and finish the burst (the deferred reap runs inside
///   `finish_burst_to_idle`).
/// - `Diagnose`: any non-Awaiting state (Idle, Pending, Active in
///   another phase, stale Profile). Late completion the engine no
///   longer tracks; emit `EffectCompleteOutsideAwaiting`.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum AwaitAction {
    Decrement,
    Rebase,
    Reap,
    Diagnose,
}

/// Snapshot of `on_probe_response`'s routing decision. Computed under a
/// short `&self.profiles` borrow, then dispatched under `&mut self`.
/// Three variants:
/// - `Descent`: the live response targets `ProfileState::Pending(_)`. Routes
///   to `dispatch_descent_probe`.
/// - `Rebase`: the live response targets `ProfileState::Active(_)` with
///   phase `BurstPhase::Rebasing`. Routes to `dispatch_rebase_*` (post-fire
///   rebase — no stability verdict; graft + `baseline := current` + finish).
/// - `Burst { intent, forced }`: the live response targets a pre-fire
///   `Active` phase (`Verifying`). The intent + forced flags are captured
///   here so the dispatch arm can act on them.
///
/// Stale responses are filtered before this enum is constructed (top-level
/// `pending_probe == Some(received)` check in `on_probe_response`).
enum ProbeDispatch {
    Descent,
    Rebase,
    Burst { intent: BurstIntent, forced: bool },
}

/// Resolve an Effect's `cwd` from the Profile's anchor path + kind.
///
/// `Command::current_dir` requires a directory; spawn fails with `ENOTDIR`
/// otherwise. For File-anchored Profiles the parent directory is the
/// natural cwd (user scripts use `$SPECTER_PATH` to locate the file).
/// `Dir` and `Unknown` (rare; pending paths) anchor at the path itself —
/// for `Unknown`, this may not exist on disk; the actuator surfaces such
/// failures as `EffectOutcome::Failed`.
fn compute_cwd(anchor_path: &Path, kind: ResourceKind) -> PathBuf {
    match kind {
        ResourceKind::File => anchor_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .map_or_else(|| anchor_path.to_path_buf(), Path::to_path_buf),
        ResourceKind::Dir | ResourceKind::Unknown => anchor_path.to_path_buf(),
    }
}

/// L5 event-class assignment. Maps an [`FsEvent`] + the resource's
/// [`ResourceKind`] to the [`ClassSet`] bit it represents.
///
/// Non-terminal events have a fixed class regardless of kind:
/// - [`FsEvent::Modified`] → [`ClassSet::CONTENT`]
/// - [`FsEvent::MetadataChanged`] → [`ClassSet::METADATA`]
/// - [`FsEvent::StructureChanged`] → [`ClassSet::STRUCTURE`]
///
/// Identity events ([`FsEvent::Removed`] / [`FsEvent::Renamed`] /
/// [`FsEvent::Revoked`]) fold by kind per design §2.1 + D7:
/// - `Dir` → [`ClassSet::STRUCTURE`] (the directory's place in its parent
///   changed).
/// - `File` (and `Unknown` via [`ResourceKind::effective`]) →
///   [`ClassSet::CONTENT`] (the file's identity changed — kqexec
///   mapping; the Unknown collapse matches the L4 translator's
///   File-shape default).
///
/// Pure / `const fn`; consulted at the L5 entry filter in [`Engine::on_fs_event`].
const fn fs_event_to_class(event: FsEvent, kind: ResourceKind) -> ClassSet {
    match event {
        FsEvent::Modified => ClassSet::CONTENT,
        FsEvent::MetadataChanged => ClassSet::METADATA,
        FsEvent::StructureChanged => ClassSet::STRUCTURE,
        FsEvent::Removed | FsEvent::Renamed | FsEvent::Revoked => {
            if matches!(kind.effective(), ResourceKind::Dir) {
                ClassSet::STRUCTURE
            } else {
                ClassSet::CONTENT
            }
        }
    }
}

/// Map a diff `EntryKind` to a Tree `ResourceKind`. `Symlink` and `Other`
/// fold into `File` (the slot occupies one file inode regardless of which
/// flavor of non-directory it is); `Dir` maps cleanly. Mirrors
/// `reconcile::kind_from`; kept here so `transitions` doesn't depend on
/// `reconcile`'s private items beyond the explicitly-shared
/// `ensure_descendant` / `lookup_descendant` pair.
const fn kind_from_entry(k: specter_core::EntryKind) -> ResourceKind {
    match k {
        specter_core::EntryKind::File
        | specter_core::EntryKind::Symlink
        | specter_core::EntryKind::Other => ResourceKind::File,
        specter_core::EntryKind::Dir => ResourceKind::Dir,
    }
}

// Keep the `ResourceKind` import used by the burst-side probe-kind decision
// reachable through the engine module surface for tests; this is a no-op at
// runtime but documents the intentional re-export discipline.
const _: fn() = || {
    let _ = ResourceKind::Unknown;
};

/// Resolve a `LeafEntry`'s `leaf_hash` from a snapshot by walking the
/// snapshot's `Dir` chain by relative segment. `rel` is `"a/b/file.txt"`
/// shape (forward-slash separated, never absolute). Returns `None` for:
///
/// - `current == None` (Profile has no snapshot yet).
/// - `current` is a File-anchored snapshot (no Dir children to walk into;
///   the relative path can only refer to the leaf itself, which is the
///   File anchor — handled by callers via the Subtree `DedupKey` path).
/// - The walk crosses an uncovered branch (`subtree: None`) or a Leaf at a
///   non-final segment.
/// - Any segment fails to resolve.
///
/// Bundle B1 `PerStableFile` suppression treats `None` as "fire
/// conservatively" — correctness over efficiency on the rare path where
/// reconcile materialised the slot but the leaf isn't reachable from
/// `current` at emission time (e.g., diff-entry's parent uncovered).
fn lookup_leaf_hash_in_current(current: Option<&TreeSnapshot>, rel: &str) -> Option<u128> {
    let TreeSnapshot::Dir(root) = current? else {
        return None;
    };
    let mut comps = rel.split('/').filter(|s| !s.is_empty()).peekable();
    let first = comps.next()?;
    let mut cur_dir = root.clone();
    let mut next_name = first.to_string();
    loop {
        let entry = cur_dir.entries.get(next_name.as_str())?;
        if comps.peek().is_none() {
            return match entry {
                specter_core::ChildEntry::Leaf(l) => Some(l.leaf_hash()),
                specter_core::ChildEntry::Dir(_) => None,
            };
        }
        let sub = match entry {
            specter_core::ChildEntry::Dir(dc) => dc.subtree.clone()?,
            specter_core::ChildEntry::Leaf(_) => return None,
        };
        cur_dir = sub;
        next_name = comps.next()?.to_string();
    }
}

#[cfg(test)]
#[path = "transitions_tests.rs"]
mod tests;
