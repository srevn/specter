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
use specter_core::{
    BurstIntent, BurstPhase, ClassSet, CorrelationId, DedupKey, Diagnostic, Effect, EffectOutcome,
    EffectScope, FsEvent, ProbeOp, ProbeResponse, ProbeResult, ProfileId, ProfileState, ResourceId,
    ResourceKind, StepOutput, SubId, SubRegistryDiff, TreeSnapshot, WatchOp,
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
                self.on_anchor_terminal_event(profile_id, now, out);
            } else {
                // Modified/StructureChanged/MetadataChanged anywhere that
                // passes the filter, or terminal at a covered descendant
                // whose class matches: drive the burst forward. Descendant
                // terminal events drive the burst; the next probe response
                // reconciles the slot via the diff-against-prior pass.
                self.drive_burst(profile_id, resource, now, out);
            }
        }
    }

    /// Re-enter pending descent for an Idle Profile whose anchor is
    /// currently absent. Triggered by an event at the Profile's
    /// `watch_root_parent` ("Watch root deletion" recovery).
    /// The Profile's anchor segment is added as the sole remaining
    /// component; descent emits a probe at the parent.
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
        // Bump the parent's watch_demand for the descent's contribution.
        // The parent already has +1 from `Profile.watch_root_parent`; the
        // descent contribution is in addition (the refcount sums). D9 —
        // descent prefix contributions are always STRUCTURE regardless of
        // the Sub's user mask.
        crate::refcounts::add_watch_demand(&mut self.tree, parent, ClassSet::STRUCTURE, out);

        let correlation = self.next_probe_correlation();
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.state = ProfileState::Pending(specter_core::DescentState {
                current_prefix: parent,
                remaining_components: vec![anchor_name],
                probe_correlation: Some(correlation),
            });
        }
        self.emit_descent_probe(profile_id, parent, correlation, out);
    }

    /// Dispatch a [`ProbeResponse`].
    ///
    /// A single `match &p.state` decides between Pending (descent dispatch)
    /// and Active(Probing) (burst dispatch). The variants are mutually
    /// exclusive by type — the compiler rules out any "Profile is in both"
    /// race, and there is exactly one place to reason about probe correlation
    /// matching.
    pub(crate) fn on_probe_response(
        &mut self,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let profile_id = response.profile;
        let received_correlation = response.correlation;

        let dispatch = {
            let Some(p) = self.profiles.get(profile_id) else {
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    profile: profile_id,
                    correlation: received_correlation,
                });
                return;
            };
            match &p.state {
                ProfileState::Pending(descent)
                    if descent.probe_correlation == Some(received_correlation) =>
                {
                    ProbeDispatch::Descent
                }
                ProfileState::Active(burst) => match &burst.phase {
                    BurstPhase::Probing { correlation } if *correlation == received_correlation => {
                        ProbeDispatch::Burst {
                            intent: burst.intent,
                            forced: burst.forced,
                        }
                    }
                    _ => ProbeDispatch::Stale,
                },
                // Pending with non-matching correlation, Idle, or any
                // future `non_exhaustive` variant: response is stale.
                _ => ProbeDispatch::Stale,
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
            ProbeDispatch::Burst { intent, forced } => match (intent, response.result) {
                (BurstIntent::Seed, ProbeResult::Ok(tree_snap)) => {
                    self.dispatch_seed_ok(profile_id, tree_snap, now, out);
                }
                (BurstIntent::Seed, ProbeResult::Vanished) => {
                    self.dispatch_seed_vanished(profile_id, now, out);
                }
                (BurstIntent::Seed, ProbeResult::Failed { errno }) => {
                    self.dispatch_seed_failed(profile_id, errno, now, out);
                }
                (BurstIntent::Standard, ProbeResult::Ok(tree_snap)) => {
                    self.dispatch_standard_ok(profile_id, tree_snap, forced, now, out);
                }
                (BurstIntent::Standard, ProbeResult::Vanished) => {
                    self.dispatch_standard_vanished(profile_id, now, out);
                }
                (BurstIntent::Standard, ProbeResult::Failed { errno }) => {
                    self.dispatch_standard_failed(profile_id, errno, now, out);
                }
            },
            ProbeDispatch::Stale => {
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    profile: profile_id,
                    correlation: received_correlation,
                });
            }
        }
    }

    /// Dispatch a [`Input::TimerExpired`].
    pub(crate) fn on_timer_expired(
        &mut self,
        id: specter_core::TimerId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(profile_id) = self.find_profile_referencing_timer(id) else {
            out.diagnostics.push(Diagnostic::StaleTimer { id });
            return;
        };

        let (is_settle, is_burst_deadline, phase_kind) = {
            let Some(p) = self.profiles.get(profile_id) else {
                out.diagnostics.push(Diagnostic::StaleTimer { id });
                return;
            };
            let ProfileState::Active(burst) = &p.state else {
                out.diagnostics.push(Diagnostic::StaleTimer { id });
                return;
            };
            let phase_kind = match &burst.phase {
                BurstPhase::Settling => PhaseKind::Settling,
                BurstPhase::Probing { .. } => PhaseKind::Probing,
                BurstPhase::Draining => PhaseKind::Draining,
            };
            (
                burst.settle_timer == Some(id),
                burst.burst_deadline == id,
                phase_kind,
            )
        };

        if is_settle {
            // Settle timer fires during Settling → transition to Probing.
            // Stale settle timers (settle_timer field is None or non-matching)
            // are filtered upstream by `is_settle == true`.
            if matches!(phase_kind, PhaseKind::Settling) {
                self.transition_to_probing(profile_id, now, out);
            } else {
                out.diagnostics.push(Diagnostic::StaleTimer { id });
            }
        } else if is_burst_deadline {
            self.handle_burst_deadline(profile_id, phase_kind, now, out);
        } else {
            out.diagnostics.push(Diagnostic::StaleTimer { id });
        }
    }

    /// Dispatch a [`Input::EffectComplete`].
    // `key` is taken by-value because the B1 path consumes it as a map key
    // (`p.last_emitted_dir_hash.remove(&key)`). Clippy reads the happy-path
    // `EffectOutcome::Ok` arm where the value isn't used and suggests
    // `&DedupKey`; the Failed-arm's borrow keeps the signature consistent
    // across both arms.
    #[allow(clippy::needless_pass_by_value)]
    pub(crate) fn on_effect_complete(
        &mut self,
        sub: SubId,
        key: DedupKey,
        result: &EffectOutcome,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(s) = self.subs.get(sub) else {
            out.diagnostics
                .push(Diagnostic::EffectCompleteForUnknownSub { sub });
            return;
        };
        let profile_id = s.profile;

        match result {
            EffectOutcome::Ok => {
                let Some(p) = self.profiles.get(profile_id) else {
                    return;
                };
                // reap-pending Profiles drop the result. The Profile's
                // last Sub was detached mid-burst; firing a Seed reseed
                // would race with the deferred reap. Even when the Sub
                // is still in the registry (multi-Sub Profile mid-life),
                // reap_pending implies sub_refcount == 0, so there's
                // nothing to fire next.
                if p.reap_pending {
                    return;
                }
                match &p.state {
                    ProfileState::Idle => {
                        // Post-Effect rebase via Seed burst. baseline and
                        // current are advanced on the Seed's Ok response,
                        // so the next Standard burst diffs against the
                        // post-Effect state.
                        self.start_seed_burst(profile_id, now, out);
                    }
                    ProfileState::Active(_) => {
                        // Drop with Diagnostic. The active burst's own
                        // EffectComplete::Ok will fire the next Seed.
                        out.diagnostics.push(Diagnostic::EffectCompleteWhileActive {
                            sub,
                            profile: profile_id,
                        });
                    }
                    // `ProfileState` is non_exhaustive at the crate
                    // boundary; future variants drop with a Diagnostic.
                    _ => {
                        out.diagnostics.push(Diagnostic::EffectCompleteWhileActive {
                            sub,
                            profile: profile_id,
                        });
                    }
                }
            }
            EffectOutcome::Failed { .. } => {
                // Failed row: no baseline change, no Diagnostic.
                // Bundle B1: clear the `last_emitted_dir_hash` entry for
                // this DedupKey — the failed Effect leaves no observation
                // to deduplicate against, so the next stable verdict at
                // the same key must fire.
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.last_emitted_dir_hash.remove(&key);
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
    /// `ENFILE` on FD exhaustion). The Engine clamps `watch_demand := 0`
    /// on `resource` — losing every contributing Profile's watch demand
    /// atomically — and waits for reconciliation on the parent's next
    /// `StructureChanged` to rebuild the contributions. Heavy reset, but
    /// rare; v1 doesn't engineer per-Profile contribution tracking.
    ///
    /// Stale resources (already Unwatched, queue-race) are a no-op +
    /// Diagnostic.
    pub(crate) fn on_watch_op_rejected(
        &mut self,
        resource: ResourceId,
        _op: WatchOp,
        errno: i32,
        _now: Instant,
        out: &mut StepOutput,
    ) {
        // Always emit the diagnostic for log readability. The clamp helper
        // is a no-op when `watch_demand == 0`, so it's safe to call
        // unconditionally.
        out.diagnostics
            .push(Diagnostic::WatchOpRejected { resource, errno });
        clamp_watch_demand_to_zero(&mut self.tree, resource, out);

        // Vacate any descents whose `current_prefix == resource`. The
        // clamp atomically zeroed the prefix's `watch_demand`, dropping
        // every descent's contribution at once. A subsequent
        // `sub_watch_demand` (probe-Ok-advance, probe-Vanished-rewind, or
        // reap path) would underflow the now-zero counter — purge the
        // descent state to close that race. Cancel any in-flight probe
        // so the prober can skip the syscall under load (best-effort);
        // the late ProbeResponse, if it still arrives, is dropped since
        // the Profile no longer matches a Probing slot.
        let purge_targets = self.descents_at_prefix(resource);
        for pid in purge_targets {
            let had_inflight = self
                .descent_state(pid)
                .is_some_and(|d| d.probe_correlation.is_some());
            // Transition Pending → Idle. The `Profile.anchor_contribution`
            // flag stays false (descent never bumps the anchor; the prefix
            // carried the contribution and the WatchOpRejected clamp already
            // zeroed it). Profile is now stuck Idle without an anchor —
            // operator recovery is required.
            if let Some(p) = self.profiles.get_mut(pid) {
                p.state = ProfileState::Idle;
            }
            if had_inflight {
                out.probe_ops.push(ProbeOp::Cancel { profile: pid });
            }
            out.diagnostics.push(Diagnostic::PendingDescentVacated {
                profile: pid,
                prefix: resource,
                errno,
            });
        }
    }

    /// Start a new burst (Seed if no baseline yet, Standard if baseline
    /// established); Active → accumulate the event into `dirty_resources`
    /// + `force_walk_resources` and reset settle.
    ///
    /// `event_resource` is the `FsEvent`'s source. It seeds (Idle path) or
    /// accumulates (Active path) the event-tracking sets the next probe
    /// uses to compute LCA + `force_walk`.
    ///
    /// The "no baseline → Seed" branch handles the degenerate
    /// post-`Vanished` Idle state where `current.is_none()` — a Standard
    /// burst without a baseline cannot dispatch its stability verdict
    /// meaningfully.
    fn drive_burst(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
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
            ProfileState::Active(_) => {
                // Accumulate the event into `dirty_resources` (LCA basis)
                // and `force_walk_resources` (since-last-probe walker
                // hint) before transitioning. The next probe at this
                // burst's `transition_to_probing` consumes both.
                if let Some(p) = self.profiles.get_mut(profile_id)
                    && let ProfileState::Active(burst) = &mut p.state
                {
                    burst.dirty_resources.insert(event_resource);
                    burst.force_walk_resources.insert(event_resource);
                }
                self.transition_to_settling(profile_id, now, out);
            }
            // `ProfileState` non_exhaustive: Pending Profiles never reach
            // here — `covering_profiles` filters them at the source — but
            // the wildcard arm absorbs both Pending (defensively) and any
            // future variant.
            _ => {}
        }
    }

    /// Anchor terminal event (Removed/Renamed/Revoked at `Profile.resource`).
    /// Thin wrapper over `finalize_anchor_lost` — the FsEvent dispatcher
    /// and the WatchOpRejected purge (Commit 3) share the same
    /// "anchor's FD is gone, finalize the burst" logic.
    fn on_anchor_terminal_event(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        self.finalize_anchor_lost(profile_id, now, out);
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
    /// `dispatch_*_vanished/failed` discipline (transitions.rs ~750).
    /// Reverse-ordering would have `finish_burst_to_idle` invoke
    /// `reap_profile`, which would release the anchor; the post-`finish`
    /// release would then see a counter that's already zero and (pre
    /// counter-existence-check) underflow `sub_watch_demand`. The helper
    /// + ordering combination removes both failure modes.
    ///
    /// **Pending exclusion.** `ProfileState::Pending` is defensive here
    /// — `covering_profiles` already filters Pending Profiles at the
    /// source, so the FsEvent path can't deliver a Pending Profile.
    /// `on_watch_op_rejected` (Commit 3) calls this directly after
    /// iterating the full registry, where the guard does load-bearing
    /// work: a Pending Profile carries no anchor contribution and
    /// participates in no burst-suppress accounting, so
    /// `finish_burst_to_idle`'s `sub_suppress` would underflow.
    pub(crate) fn finalize_anchor_lost(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        if matches!(p.state, ProfileState::Pending(_)) {
            return;
        }
        let was_probing = matches!(
            &p.state,
            ProfileState::Active(b) if matches!(b.phase, BurstPhase::Probing { .. }),
        );
        let was_active = matches!(p.state, ProfileState::Active(_));

        if was_probing {
            out.probe_ops.push(ProbeOp::Cancel {
                profile: profile_id,
            });
        }

        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            p.current = None;
        }

        // Release BEFORE finish_burst_to_idle. See the ordering note
        // above.
        self.release_anchor_claim(profile_id, out);

        if was_active {
            self.finish_burst_to_idle(profile_id, now, out);
        }
    }

    /// (Seed, Ok).
    ///
    /// Graft the response into `Profile.current` at the burst's
    /// `probe_target` (= anchor for Seeds), then rebase
    /// `Profile.baseline := Profile.current`. Bundle B3 (hash-only): if
    /// the post-graft `current` diverges from a recorded
    /// `last_emitted_dir_hash[Subtree]` for this Profile, fire
    /// `emit_effects` once. This narrows "Seed bursts never fire Effects"
    /// to "Seed bursts fire Effects only when `last_emitted_dir_hash[Subtree]`
    /// diverges from the post-Seed view" — handles recovery / post-Effect
    /// drift for `SubtreeRoot` scope. `PerStableFile` is a documented v1
    /// limitation.
    fn dispatch_seed_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Seed always targets anchor — `probe_target` was set to the
        // anchor at `start_seed_burst` / `transition_to_probing`.
        let target = match self.profiles.get(profile_id) {
            Some(p) => match &p.state {
                ProfileState::Active(b) => b.probe_target.unwrap_or(p.resource),
                _ => p.resource,
            },
            None => return,
        };

        match snapshot {
            TreeSnapshot::Dir(arc) => {
                graft(profile_id, target, arc, &mut self.tree, &mut self.profiles, out);
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

        // Bundle B3 — fire Effect once on observed drift, before rebasing
        // baseline. emit_effects' B1 path then runs against the freshly
        // grafted current and updates `last_emitted_dir_hash` to the new
        // post-fire hash, so the rebase below is consistent.
        if self.b3_seed_drift_observed(profile_id) {
            self.emit_effects(profile_id, false, out);
        }

        // Rebase: baseline := current (the grafted snapshot).
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = p.current.clone();
        }

        self.finish_burst_to_idle(profile_id, now, out);
        // No Effect fires from a fresh-Profile Seed burst — B3 gates on
        // `last_emitted_dir_hash` non-empty, which fresh Seeds never have.
    }

    /// Bundle B3 — hash-only drift check at Seed-Ok. Returns true iff the
    /// Profile has fired at least one `DedupKey::Subtree` Effect AND any
    /// such entry's `last_emitted_dir_hash` differs from the post-graft
    /// `current`'s anchor-rooted hash. Fresh-Profile Seed (no prior
    /// emission) ⇒ `last_emitted_dir_hash` empty ⇒ false ⇒ no fire,
    /// preserving "fresh Seed never fires Effect".
    ///
    /// Limitation: `DedupKey::PerFile` entries are not drift sources. The
    /// post-Seed `current` lacks the per-leaf history for a faithful
    /// per-file diff (baseline == current after the rebase, so
    /// `emit_effects`' diff is empty). Documented v1 carve-out.
    fn b3_seed_drift_observed(&self, profile_id: ProfileId) -> bool {
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        if p.last_emitted_dir_hash.is_empty() {
            return false;
        }
        let curr_hash: u128 = match p.current.as_ref() {
            Some(TreeSnapshot::Dir(arc)) => arc.dir_hash(),
            Some(TreeSnapshot::File(leaf)) => leaf.leaf_hash(),
            None => return false,
        };
        p.last_emitted_dir_hash
            .iter()
            .any(|(key, &h)| matches!(key, DedupKey::Subtree { .. }) && h != curr_hash)
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
    fn dispatch_seed_vanished(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            p.current = None;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Seed,
        });
        // Release BEFORE finish_burst_to_idle so any deferred
        // `reap_profile` (reap_pending) sees a cleared flag — preserves
        // the trichotomy invariant `!(Pending && anchor_contribution)`
        // across the eventual `start_pending_recovery` transition.
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, now, out);
    }

    /// (Seed, Failed).
    ///
    /// Symmetric with `dispatch_standard_failed`: the probe failed at the
    /// anchor; release the anchor's `watch_demand` contribution. See
    /// `dispatch_seed_vanished` for the trichotomy-invariant rationale.
    fn dispatch_seed_failed(
        &mut self,
        profile_id: ProfileId,
        errno: i32,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            p.current = None;
        }
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent: BurstIntent::Seed,
            errno,
        });
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, now, out);
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
                graft(profile_id, target, arc, &mut self.tree, &mut self.profiles, out);
            }
            TreeSnapshot::File(leaf) => {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.current = Some(TreeSnapshot::File(leaf));
                }
            }
        }

        if is_stable && dirty_zero {
            // Row 3: stable + dirty=0 → fire Effect, → Idle. baseline pinned
            // (advances on next EffectComplete::Ok Seed).
            self.emit_effects(profile_id, forced, out);
            self.finish_burst_to_idle(profile_id, now, out);
        } else if is_stable {
            // Row 4: stable + dirty>0 → Draining. The stable snapshot lives
            // on `Profile.current` (just spliced in by graft); the reconfirm
            // probe compares against `current`. No need to pin a duplicate
            // snapshot on the phase variant.
            self.transition_to_draining(profile_id);
        } else if forced {
            // Row 5: not-stable + forced → fire Effect with forced=true,
            // → Idle.
            self.emit_effects(profile_id, true, out);
            self.finish_burst_to_idle(profile_id, now, out);
        } else {
            // Row 5 else: not-stable + !forced → reschedule settle, → Settling.
            self.transition_to_settling(profile_id, now, out);
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
    fn dispatch_standard_vanished(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            p.current = None;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Standard,
        });
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, now, out);
    }

    /// (Standard, Failed).
    ///
    /// See `dispatch_standard_vanished` for the release-before-finish
    /// ordering rationale.
    fn dispatch_standard_failed(
        &mut self,
        profile_id: ProfileId,
        errno: i32,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.baseline = None;
            p.current = None;
        }
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent: BurstIntent::Standard,
            errno,
        });
        self.release_anchor_claim(profile_id, out);
        self.finish_burst_to_idle(profile_id, now, out);
    }

    /// `burst_deadline` row — sets `forced := true` and either
    /// transitions the phase (Settling/Draining → Probing) or, if a probe
    /// is already in flight (Probing), waits for the response.
    fn handle_burst_deadline(
        &mut self,
        profile_id: ProfileId,
        phase: PhaseKind,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if let Some(p) = self.profiles.get_mut(profile_id)
            && let ProfileState::Active(burst) = &mut p.state
        {
            burst.forced = true;
        }
        match phase {
            PhaseKind::Settling | PhaseKind::Draining => {
                self.transition_to_probing(profile_id, now, out);
            }
            PhaseKind::Probing => {
                // Probe in flight; no second emission. The response, when
                // it arrives, dispatches with `forced = true`.
            }
        }
    }

    /// Emit Effects at a Standard burst's stable verdict. Routes per scope:
    /// `SubtreeRoot` Subs fire one Effect anchored at the Profile's resource;
    /// `PerStableFile` Subs fire one Effect per matching diff entry. The
    /// `Diff` is built at most once and shared across both helpers via `Arc`.
    ///
    /// `Profile.reap_pending` suppresses all emission — the Profile is on its
    /// way out and any remaining Subs (none, by construction of
    /// `reap_pending = sub_refcount == 0`) would fire against a Sub registry
    /// that no longer holds them.
    #[allow(clippy::too_many_lines)] // diff/B1/scope-routing fan-out is irreducible without churn
    fn emit_effects(&mut self, profile_id: ProfileId, forced: bool, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        if p.reap_pending {
            return;
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
        for sub_id in sub_ids {
            let (scope, needs_diff) = match self.subs.get(sub_id) {
                Some(s) => (s.scope, s.needs_diff),
                None => continue,
            };
            match scope {
                EffectScope::SubtreeRoot => {
                    let dk = DedupKey::Subtree {
                        sub: sub_id,
                        profile: profile_id,
                    };
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
                    });

                    // Record the post-fire hash so the next stable verdict
                    // can suppress an idempotent re-fire.
                    if let Some(p) = self.profiles.get_mut(profile_id) {
                        p.last_emitted_dir_hash.insert(dk, current_dir_hash);
                    }
                }
                EffectScope::PerStableFile => {
                    // PerStableFile implies `needs_diff = true` at Sub::new;
                    // diff is always built.
                    let Some(diff) = ensure_diff(&mut diff_arc) else {
                        continue;
                    };
                    self.emit_effects_per_stable_file(
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
                }
            }
        }
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
    ) {
        let profile_id = match self.subs.get(sub_id) {
            Some(s) => s.profile,
            None => return,
        };

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
            });

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

    /// Find the Profile whose Burst references `id`. O(active profiles).
    /// Acceptable for v1 — typical workloads have few simultaneous Active
    /// profiles.
    fn find_profile_referencing_timer(&self, id: specter_core::TimerId) -> Option<ProfileId> {
        for (pid, p) in self.profiles.iter() {
            if let ProfileState::Active(burst) = &p.state
                && (burst.settle_timer == Some(id) || burst.burst_deadline == id)
            {
                return Some(pid);
            }
        }
        None
    }

    /// Mint a fresh `CorrelationId` for an Effect. Engine-monotonic, sharing
    /// the same counter as `next_probe_correlation` — the spaces don't
    /// collide because they're typed differently.
    const fn next_effect_correlation(&mut self) -> CorrelationId {
        self.next_correlation = self.next_correlation.saturating_add(1);
        CorrelationId(self.next_correlation)
    }
}

#[derive(Copy, Clone)]
enum PhaseKind {
    Settling,
    Probing,
    Draining,
}

/// Snapshot of `on_probe_response`'s routing decision. Computed under a
/// short `&self.profiles` borrow, then dispatched under `&mut self`.
/// Three variants:
/// - `Descent`: the response matches `ProfileState::Pending(d)` where
///   `d.probe_correlation == Some(received)`. Routes to
///   `dispatch_descent_probe`.
/// - `Burst { intent, forced }`: the response matches
///   `ProfileState::Active(b)` with `b.phase == Probing { correlation }`
///   and the correlation matches. The intent + forced flags are
///   captured here so the dispatch arm can act on them.
/// - `Stale`: no live channel matches — emits `StaleProbeResponse`.
enum ProbeDispatch {
    Descent,
    Burst { intent: BurstIntent, forced: bool },
    Stale,
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
/// - `File` → [`ClassSet::CONTENT`] (the file's identity changed —
///   kqexec mapping).
/// - `Unknown` → [`ClassSet::CONTENT`] ("treat as file" default; matches
///   the L4 translator's `Unknown` branch).
///
/// Pure / `const fn`; consulted at the L5 entry filter in [`Engine::on_fs_event`].
const fn fs_event_to_class(event: FsEvent, kind: ResourceKind) -> ClassSet {
    match event {
        FsEvent::Modified => ClassSet::CONTENT,
        FsEvent::MetadataChanged => ClassSet::METADATA,
        FsEvent::StructureChanged => ClassSet::STRUCTURE,
        FsEvent::Removed | FsEvent::Renamed | FsEvent::Revoked => match kind {
            ResourceKind::Dir => ClassSet::STRUCTURE,
            ResourceKind::File | ResourceKind::Unknown => ClassSet::CONTENT,
        },
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
