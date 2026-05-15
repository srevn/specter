//! Per-input dispatch handlers.
//!
//! Each `on_*` method handles one [`Input`] variant for one Profile. They
//! call the burst-lifecycle helpers (`burst.rs`), the refcount edges
//! (`refcounts.rs`), and the reconciliation (`reconcile.rs`). Logic that
//! fits in one row of the transition table stays inline; logic shared across
//! rows (e.g., emit Effects on Standard stable verdict) is factored into
//! private helpers within this module.
//!
//! `on_probe_response` routes each response by
//! [`crate::probe_channel::OpenKind`]: the response handler closes the
//! channel atomically via [`crate::probe_channel::ProbeChannel::close_if`]
//! and matches `(open.kind(), outcome)`. Per-intent fan-out for the
//! Verifying arm lives in `dispatch_burst_outcome`.

use crate::Engine;
use crate::engine::is_timer_referenced;
use crate::probe_channel::OpenKind;
use crate::reconcile::{ensure_descendant, graft, lookup_descendant};
use compact_str::CompactString;
use smallvec::SmallVec;
use specter_core::{
    ActiveBurst, AnchorClaim, BurstFinish, BurstIntent, ClaimKind, ClassSet, DedupKey,
    DescentRemaining, Diagnostic, Effect, EffectOutcome, EffectScope, FsEvent, OverflowScope,
    PostFirePhase, PreFirePhase, ProbeOutcome, ProbeOwner, ProbeResponse, ProfileId, ProfileState,
    PromoterClaimKind, PromoterId, PromoterState, ReapTrigger, Resource, ResourceId, ResourceKind,
    StepOutput, SubId, TimerId, TimerKind, TreeSnapshot, WatchFailure, WatchOp, WatchRegistryDiff,
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
    ///    STRUCTURE-only, so any event reaching here is structurally
    ///    relevant — descent dispatch is unfiltered.
    /// 3. Idle Profiles whose `watch_root_parent == resource` and whose
    ///    anchor is currently absent (`current.is_none()`) re-enter pending
    ///    descent — auto-recapture on anchor reappearance. Same STRUCTURE
    ///    floor applies.
    /// 4. Per-covering-Profile dispatch with class-aware filter:
    ///    - Anchor events bypass the filter unconditionally — lifecycle
    ///      signal continuity trumps user opt-out.
    ///    - Descendant events whose class (per [`fs_event_to_class`]) is
    ///      not in the Profile's `events` drop with
    ///      `EventClassDropped` BEFORE driving the burst — the class filter
    ///      sits before dirty-set bumps.
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
        let watch_demand = self
            .tree
            .get(resource)
            .map_or(0, specter_core::Resource::watch_demand);
        if watch_demand == 0 {
            out.diagnostics
                .push(Diagnostic::EventOnUnwatchedResource { resource });
            return;
        }

        // Snapshot the proxy back-ref BEFORE any dispatch — each
        // `on_promoter_proxy_event` mutates Promoter state, and the
        // enumeration-vanished cascade
        // (`dispatch_promoter_enumeration_vanished` →
        // `unregister_proxy_subtree`, parent enumeration's reverse
        // pass) clears the back-ref of co-resident Promoters
        // mid-loop. The snapshot keeps the dispatch list stable across
        // the iteration. SmallVec inline cap of 1 covers the typical
        // case (one proxy back-ref) without allocation.
        let proxies: SmallVec<[specter_core::PromoterId; 1]> = self
            .tree
            .get(resource)
            .map(|r| r.proxy_promoters.iter().copied().collect())
            .unwrap_or_default();

        // Single-pass classification of the event's carriers: Profiles
        // that "carry" a dispatch responsibility for this resource.
        // Descent prefix and watch-root-parent watches both register
        // STRUCTURE-only, so any event reaching here is structurally
        // relevant for both arms — no class filter applies before
        // dispatch. Mutual exclusion is structural (`Pending` excludes
        // `Idle` at the `ProfileState` sum-type level).
        let carriers = self.classify_event_carriers(resource);
        let descent_count = carriers.descents.len();
        let recovery_count = carriers.recoveries.len();
        for owner in carriers.descents.iter().copied() {
            self.on_descent_event(owner, now, out);
        }
        for pid in carriers.recoveries.iter().copied() {
            self.start_pending_recovery(pid, resource, out);
        }

        // Find covering Profiles (anchor or any covering ancestor). For
        // P4 single-Profile this resolves to 0 or 1; P5 multi-Profile
        // dispatches to each in encounter order.
        let covering = self.covering_profiles(resource);
        if covering.is_empty() && descent_count == 0 && recovery_count == 0 && proxies.is_empty() {
            // No consumer: covered by no Profile, no in-flight descent,
            // no recovery kicked off, and no proxy back-ref. Emit
            // `EventNoConsumer` (a benign "watched but no listener"
            // signal — typically a `WatchRootParent` event for
            // something we don't track) and drop. Distinct from
            // `EventOnUnwatchedResource` (the `watch_demand == 0`
            // race earlier) so log levels can diverge.
            out.diagnostics
                .push(Diagnostic::EventNoConsumer { resource });
            return;
        }

        // Promoter dispatch. Order within the step doesn't affect
        // correctness — proxy events drive enumeration, independent
        // of Profile burst lifecycle. Dispatch BEFORE Profile
        // covering-Profile dispatch for testability: assertions on
        // proxy effects are unaffected by burst ops emitted later in
        // the same step.
        for promoter_id in proxies.iter().copied() {
            self.on_promoter_proxy_event(promoter_id, resource, now, out);
        }

        // Class-aware routing. Compute the event's class once from the
        // resource's kind; per-Profile dispatch consults the Profile's
        // `events` (every Sub on a Profile shares the same mask, so
        // the union is each Sub's mask).
        //
        // Unprobed slots collapse to File-shape per the backend-mask
        // convention — `fs_event_to_class` and the kqueue / inotify
        // translators agree on this default.
        let resource_kind = self
            .tree
            .get(resource)
            .map_or(ResourceKind::File, Resource::kind_or_file);
        let event_class = fs_event_to_class(event, resource_kind);
        let is_terminal = matches!(
            event,
            FsEvent::Removed | FsEvent::Renamed | FsEvent::Revoked
        );

        for profile_id in covering {
            let Some((is_anchor, profile_events)) = self
                .profiles
                .get(profile_id)
                .map(|p| (p.resource == resource, p.events()))
            else {
                continue;
            };

            // Anchor events bypass the class filter unconditionally
            // (lifecycle: anchor disappearance recovery, anchor reappearance
            // detection, etc.). Descendant events whose class is not in
            // the Profile's `events` drop here, before `drive_burst`
            // extends `dirty_resources` / `force_walk_resources`.
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
        let Some(anchor_name) = self.tree.name(anchor).map(CompactString::from) else {
            return;
        };
        // `vec![anchor_name]` is non-empty by construction, so the
        // `from_vec` discriminant is structurally `Some`. `expect`
        // documents the contract.
        let remaining = DescentRemaining::from_vec(vec![anchor_name])
            .expect("start_pending_recovery: single-segment remaining is non-empty");
        self.enter_pending_descent(profile_id, parent, remaining, out);
    }

    /// Dispatch a [`ProbeResponse`] by routing to the per-owner
    /// handler.
    ///
    /// **Staleness.** Each per-owner handler closes the channel
    /// atomically via
    /// [`crate::probe_channel::ProbeChannel::close_if`]; the call
    /// returns `Some(Open)` iff the channel is open AND its
    /// correlation matches the received one. A `None` covers every
    /// stale path — stale id, response after Cancel, out-of-order
    /// response — and yields [`Diagnostic::StaleProbeResponse`].
    ///
    /// **Routing.** The returned [`crate::probe_channel::Open`]
    /// carries the [`crate::probe_channel::OpenKind`] the channel
    /// recorded at open-time. The handlers match `(open.kind(),
    /// outcome)` directly — no per-(state, phase) projection, so
    /// state-phase divergence that used to surface as
    /// `debug_assert!(false, "I5 violated")` is structurally
    /// unrepresentable (a channel opened with `ProfileVerifying`
    /// cannot drift into a state where the dispatch arm rejects it,
    /// because the kind is what's matched).
    ///
    /// **Walker-contract violations.** A descent probe receiving
    /// `AnchorOk` (the walker contracted to return `SubtreeOk` or
    /// `Vanished` for `ProbeRequest::Descent`) is a walker bug. The
    /// dispatch arms `debug_assert!` and emit
    /// [`Diagnostic::StaleProbeResponse`] so the engine never grafts
    /// a kind-mismatched payload. The mirror case (File-anchored
    /// Profile receiving `SubtreeOk`) routes through the existing
    /// dispatch arm — `dispatch_*_ok` synthesises a
    /// `TreeSnapshot::Dir`, and graft handles the kind change at the
    /// snapshot level.
    pub(crate) fn on_probe_response(
        &mut self,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        match response.owner {
            ProbeOwner::Profile(pid) => self.on_profile_probe_response(pid, response, now, out),
            ProbeOwner::Promoter(pid) => self.on_promoter_probe_response(pid, response, now, out),
        }
    }

    /// Profile-side probe response handler. Atomic check-and-take on
    /// the probe channel, then [`OpenKind`]-routed dispatch.
    ///
    /// **Staleness.** [`crate::probe_channel::ProbeChannel::close_if`]
    /// is atomic: it returns `Some(Open)` iff the channel is open AND
    /// its correlation matches `received`, and leaves the entry
    /// intact on mismatch. A `None` covers every stale path — stale
    /// `ProfileId`, response after Cancel, response after a fresh
    /// mint, out-of-order response — and yields
    /// [`Diagnostic::StaleProbeResponse`].
    ///
    /// **Routing.** Match on the [`OpenKind`] discriminant the channel
    /// records at open-time. Profile owners mint with
    /// `ProfileVerifying` / `ProfileRebasing` / `ProfileDescent`; the
    /// Promoter-kind arm catches a cross-affinity regression
    /// (`debug_assert!` + diagnostic). Walker-contract violations
    /// (descent receiving `AnchorOk`) are handled in-arm: production
    /// walkers never emit those shapes.
    fn on_profile_probe_response(
        &mut self,
        profile_id: ProfileId,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let owner = response.owner;
        let received = response.correlation;

        let Some(open) = self.probe_channel.close_if(owner, received) else {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                owner,
                correlation: received,
            });
            return;
        };

        match (open.kind(), response.outcome) {
            // ----- ProfileVerifying -----
            (OpenKind::ProfileVerifying, outcome) => {
                // Read `(intent, forced)` off the PreFireBurst. State
                // agreement is structural: only `start_seed_burst` and
                // `transition_to_verifying` mint with this kind, and
                // both write `phase = Verifying` before opening the
                // channel. The bail covers a test forging a channel
                // open without a matching phase write — never reached
                // in production.
                let Some((intent, forced)) = self.profiles.get(profile_id).and_then(|p| {
                    if let ProfileState::Active(ActiveBurst::PreFire(pre), _) = p.state() {
                        Some((pre.intent, pre.forced))
                    } else {
                        None
                    }
                }) else {
                    debug_assert!(
                        false,
                        "channel ProfileVerifying but profile state diverges \
                         (profile = {profile_id:?})",
                    );
                    out.diagnostics.push(Diagnostic::StaleProbeResponse {
                        owner,
                        correlation: received,
                    });
                    return;
                };
                self.dispatch_burst_outcome(profile_id, intent, forced, outcome, now, out);
            }

            // ----- ProfileRebasing -----
            (OpenKind::ProfileRebasing, ProbeOutcome::AnchorOk(leaf)) => {
                self.dispatch_rebase_ok(profile_id, TreeSnapshot::File(leaf), out);
            }
            (OpenKind::ProfileRebasing, ProbeOutcome::SubtreeOk(arc)) => {
                self.dispatch_rebase_ok(profile_id, TreeSnapshot::Dir(arc), out);
            }
            (OpenKind::ProfileRebasing, ProbeOutcome::Vanished) => {
                self.dispatch_rebase_vanished(profile_id, out);
            }
            (OpenKind::ProfileRebasing, ProbeOutcome::Failed { errno }) => {
                self.dispatch_rebase_failed(profile_id, errno, out);
            }

            // ----- ProfileDescent -----
            (OpenKind::ProfileDescent, ProbeOutcome::SubtreeOk(arc)) => {
                self.dispatch_descent_ok(owner, &arc, now, out);
            }
            (OpenKind::ProfileDescent, ProbeOutcome::Vanished) => {
                self.dispatch_descent_vanished(owner, now, out);
            }
            (OpenKind::ProfileDescent, ProbeOutcome::Failed { errno }) => {
                self.dispatch_descent_failed(owner, errno, out);
            }
            (OpenKind::ProfileDescent, ProbeOutcome::AnchorOk(_)) => {
                // Walker contract: descent probes a Dir prefix, walker
                // returns `SubtreeOk` or `Vanished`. `AnchorOk` here is
                // a walker-side regression.
                debug_assert!(
                    false,
                    "walker contract violated: ProfileDescent received AnchorOk \
                     (profile = {profile_id:?})",
                );
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    owner,
                    correlation: received,
                });
            }

            // ----- Cross-affinity: Profile owner with Promoter kind -----
            // Mint-site discipline forbids; this arm is regression
            // detection. A Promoter-kinded entry under a Profile key
            // requires either a future buggy mint site or test forgery
            // — neither reachable in production.
            (OpenKind::PromoterDescent | OpenKind::PromoterEnumerating { .. }, _) => {
                debug_assert!(
                    false,
                    "owner-affinity violated: Profile owner with Promoter kind \
                     (profile = {profile_id:?}, kind = {:?})",
                    open.kind(),
                );
                out.diagnostics.push(Diagnostic::StaleProbeResponse {
                    owner,
                    correlation: received,
                });
            }
        }
    }

    /// Dispatch a Verifying-phase [`ProbeOutcome`] for `profile_id`.
    ///
    /// Routes the outcome variant to its per-intent dispatch helper:
    /// `AnchorOk → TreeSnapshot::File`, `SubtreeOk → TreeSnapshot::Dir`,
    /// `Vanished` / `Failed` to the per-intent failure helpers.
    ///
    /// **First-classify is delegated.** `Profile.kind` is set
    /// atomically with `Profile.current` by
    /// [`specter_core::Profile::install_dir_current`] /
    /// [`specter_core::Profile::install_file_current`] — the per-intent
    /// dispatchers call these setters (directly or through
    /// [`Engine::apply_snapshot`]) at the snapshot commit point, so the
    /// fallback's `if p.kind.is_none()` write is structurally
    /// redundant. Removing it removes the cross-field invariant
    /// hazard: any read site between the dispatch entry and the
    /// setter call now sees either both fields set or neither, never
    /// a torn write.
    ///
    /// **Boundary kind-mismatch check.** The walker is contracted to
    /// return a `ProbeOutcome` whose variant matches the request's
    /// kind (typed [`crate::ProbeRequest`]). Each per-intent
    /// dispatcher (`dispatch_seed_ok` / `dispatch_standard_ok` /
    /// `dispatch_rebase_ok`) invokes
    /// [`Engine::kind_agrees_or_finalize`] BEFORE the snapshot commit
    /// to catch any future regression and route through
    /// `finalize_anchor_lost` rather than misroute a Dir snapshot
    /// onto a File-kinded Profile (or vice versa).
    fn dispatch_burst_outcome(
        &mut self,
        profile_id: ProfileId,
        intent: BurstIntent,
        forced: bool,
        outcome: ProbeOutcome,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let snapshot = match outcome {
            ProbeOutcome::AnchorOk(leaf) => Some(TreeSnapshot::File(leaf)),
            ProbeOutcome::SubtreeOk(arc) => Some(TreeSnapshot::Dir(arc)),
            ProbeOutcome::Vanished => None,
            ProbeOutcome::Failed { errno } => {
                match intent {
                    BurstIntent::Seed => self.dispatch_seed_failed(profile_id, errno, out),
                    BurstIntent::Standard => self.dispatch_standard_failed(profile_id, errno, out),
                }
                return;
            }
        };
        let Some(snap) = snapshot else {
            match intent {
                BurstIntent::Seed => self.dispatch_seed_vanished(profile_id, out),
                BurstIntent::Standard => self.dispatch_standard_vanished(profile_id, out),
            }
            return;
        };
        match intent {
            BurstIntent::Seed => self.dispatch_seed_ok(profile_id, snap, now, out),
            BurstIntent::Standard => self.dispatch_standard_ok(profile_id, snap, forced, now, out),
        }
    }

    /// Apply a successful probe response's `TreeSnapshot` to the
    /// Profile's `current`. Single home for the "Dir → graft / File →
    /// inline write" dispatch shared by the three `dispatch_*_ok`
    /// helpers.
    ///
    /// `TreeSnapshot::Dir` flows through [`crate::reconcile::graft`]
    /// (splice + reconcile + commit via
    /// `Profile::install_dir_current`); `TreeSnapshot::File` writes
    /// inline through [`specter_core::Profile::install_file_current`]
    /// (a Leaf has no descendants to materialise).
    ///
    /// **Typed prior extraction.** On the Dir arm this helper extracts
    /// the Dir prior from `Profile.current` under one immutable
    /// borrow and threads it to [`graft`] as a typed
    /// `Option<Arc<DirSnapshot>>`. Lifting the extraction here keeps
    /// graft's body Dir-typed end-to-end and centralises the
    /// File-shaped-prior detection at the single boundary that already
    /// owns the Profile borrow shape.
    ///
    /// **Kind agreement is a caller responsibility.** Callers MUST
    /// invoke [`Engine::kind_agrees_or_finalize`] before this helper.
    /// The setters' debug_assert is a defensive backstop for any
    /// future caller bypassing the boundary; production paths through
    /// the dispatchers always pass the agreement check before
    /// reaching here.
    pub(crate) fn apply_snapshot(
        &mut self,
        profile_id: ProfileId,
        target: ResourceId,
        snapshot: TreeSnapshot,
        out: &mut StepOutput,
    ) {
        match snapshot {
            TreeSnapshot::Dir(arc) => {
                // Extract typed prior under one Profile borrow. The
                // `File` arm is structurally unreachable when the
                // dispatcher honoured `kind_agrees_or_finalize` (a Dir
                // response on a `Some(File)`-kinded Profile would have
                // routed through `finalize_anchor_lost` first); the
                // `debug_assert!` is defense-in-depth against a future
                // caller bypassing that boundary.
                let prior = match self.profiles.get(profile_id).and_then(|p| p.current()) {
                    Some(TreeSnapshot::Dir(prior_arc)) => Some(Arc::clone(prior_arc)),
                    None => None,
                    Some(TreeSnapshot::File(_)) => {
                        debug_assert!(
                            false,
                            "apply_snapshot: File-shaped Profile.current paired with \
                             Dir response — kind_agrees_or_finalize boundary breached \
                             (profile = {profile_id:?})",
                        );
                        None
                    }
                };
                graft(
                    profile_id,
                    target,
                    prior,
                    arc,
                    &mut self.tree,
                    &mut self.profiles,
                    out,
                );
            }
            TreeSnapshot::File(leaf) => {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.install_file_current(leaf);
                }
            }
        }
    }

    /// Validate that the response's `TreeSnapshot` shape agrees with
    /// `Profile.kind`. Returns `true` on agreement (or on
    /// `kind == None`, the first-classify case).
    ///
    /// On disagreement — a walker-contract violation, structurally
    /// unreachable in v1 under the typed [`crate::ProbeRequest`]
    /// dispatch chain — emit a
    /// [`Diagnostic::AnchorKindMismatch`] diagnostic and route the
    /// Profile through [`Engine::finalize_anchor_lost`]. The prior
    /// baseline / current become invalid under the new on-disk shape,
    /// the anchor watch is released, and the parent watch is preserved
    /// for descent re-recovery via `Profile.watch_root_parent`.
    ///
    /// Choosing `finalize_anchor_lost` over a `debug_assert! + drop`
    /// is deliberate: the symmetric path with `dispatch_*_vanished`
    /// re-uses a well-tested cleanup chain rather than introducing a
    /// fresh "discard then graft" composition (which leaks watch
    /// contributions and breaks the cross-field invariant — the
    /// original-plan hazard the boundary check exists to prevent).
    pub(crate) fn kind_agrees_or_finalize(
        &mut self,
        profile_id: ProfileId,
        snapshot: &TreeSnapshot,
        out: &mut StepOutput,
    ) -> bool {
        let prior = self
            .profiles
            .get(profile_id)
            .and_then(specter_core::Profile::kind);
        let response_kind = match snapshot {
            TreeSnapshot::Dir(_) => ResourceKind::Dir,
            TreeSnapshot::File(_) => ResourceKind::File,
        };
        match prior {
            None => true,
            Some(prior_kind) if prior_kind == response_kind => true,
            Some(prior_kind) => {
                debug_assert!(
                    false,
                    "walker contract violated: response {response_kind:?} \
                     for kind {prior_kind:?} (profile = {profile_id:?})",
                );
                out.diagnostics.push(Diagnostic::AnchorKindMismatch {
                    profile: profile_id,
                    prior_kind,
                    response_kind,
                });
                self.finalize_anchor_lost(profile_id, out);
                false
            }
        }
    }

    /// Dispatch a [`Input::TimerExpired`].
    ///
    /// `kind` tells us which transition this timer drives — settle expiry
    /// (Batching → Verifying, with possible reschedule), burst-deadline
    /// expiry (force-fire), or gate-deadline expiry (actuator-hang
    /// recovery). The `id` epoch survives the validation re-check that
    /// [`is_timer_referenced`] performs against the live burst slot for
    /// that `kind`; `pop_expired` already ran the same check before
    /// `step` was called, so the production path runs it twice (cheap),
    /// and any direct `step(Input::TimerExpired)` from a test or
    /// fuzzer falls through the same gate.
    ///
    /// `now` flows through to [`Engine::on_settle_expired`]: the settle
    /// expiry handler reads it to decide whether to reschedule for
    /// `last_event_time + settle` (events arrived since) or transition
    /// to Verifying (quiet for ≥ settle). Other dispatch arms ignore
    /// it — `BurstDeadline` and `AwaitGateDeadline` are unconditional
    /// transitions whose decisions depend only on burst state.
    pub(crate) fn on_timer_expired(
        &mut self,
        profile: ProfileId,
        kind: TimerKind,
        id: TimerId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if !is_timer_referenced(&self.profiles, profile, kind, id) {
            out.diagnostics.push(Diagnostic::StaleTimer { id });
            return;
        }
        match kind {
            TimerKind::Settle => self.on_settle_expired(profile, now, out),
            TimerKind::BurstDeadline => self.handle_burst_deadline(profile, out),
            TimerKind::AwaitGateDeadline => self.handle_gate_deadline(profile, out),
        }
    }

    /// Settle-timer expiry. Either reschedule (events arrived since the
    /// timer was scheduled) or transition to Verifying (quiet for ≥
    /// settle).
    ///
    /// Reschedule path: `now − last_event_time < settle`. Schedules a
    /// fresh `TimerKind::Settle` at `last_event_time + settle` and
    /// updates `burst.phase` to point at the new id; the old (just-
    /// expired) id is no longer referenced and would lazily drop on a
    /// subsequent `pop_expired`. The phase stays Batching.
    ///
    /// Transition path: `now − last_event_time ≥ settle` (or
    /// `last_event_time` is `None`, which only occurs as a defensive
    /// fall-through — Standard bursts seed it at burst start, and
    /// Seed-burst Batching re-entries via `event_drives_batching`
    /// populate it before any settle timer is scheduled). Forwards to
    /// [`Engine::transition_to_verifying`].
    ///
    /// **Preconditions** (guaranteed by [`is_timer_referenced`]
    /// upstream): `Profile.state == Active(PreFire(_))` and
    /// `pre.phase == PreFirePhase::Batching { settle_timer == popped_id }`.
    /// The defensive early returns below cover direct
    /// `step(Input::TimerExpired)` calls that bypass `pop_expired`.
    pub(crate) fn on_settle_expired(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(ActiveBurst::PreFire(pre), _) = p.state() else {
            return;
        };
        // is_timer_referenced upstream guarantees Batching, but the
        // direct-step path may bypass it; gate the read defensively.
        if !matches!(pre.phase, PreFirePhase::Batching { .. }) {
            return;
        }
        let settle = p.settle;
        let last = pre.last_event_time;

        // saturating_duration_since handles `now < last` (test mockclock
        // rewind / non-monotonic clocks): returns Duration::ZERO, which
        // satisfies `< settle` and triggers a reschedule. Safe under any
        // clock skew the harness can produce.
        if let Some(last) = last
            && now.saturating_duration_since(last) < settle
        {
            let new_deadline = last + settle;
            let new_timer = self
                .timers
                .schedule(new_deadline, profile_id, TimerKind::Settle);
            if let Some(pre) = self
                .profiles
                .get_mut(profile_id)
                .and_then(specter_core::Profile::pre_fire_burst_mut)
            {
                pre.phase = PreFirePhase::Batching {
                    settle_timer: new_timer,
                };
            }
            return;
        }

        // Quiet for ≥ settle (or last_event_time is None — defensive):
        // proceed with the original Batching → Verifying transition.
        self.transition_to_verifying(profile_id, out);
    }

    /// Dispatch a [`Input::EffectComplete`].
    ///
    /// The Profile is resolved from `key` ([`DedupKey::profile`] is O(1));
    /// the Sub registry is consulted only for the unknown-Sub diagnostic.
    ///
    /// Failed arrivals always remove `key` from `Profile.fired_subs` —
    /// a failed Effect produced no observable state to deduplicate
    /// against, so its fire history is wiped and the next stable
    /// verdict at the same `DedupKey` must fire fresh. This happens
    /// regardless of phase (Awaiting decrement, late arrival, or
    /// unknown — the cleared entry is correct in every case).
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
        // `key.profile()` is O(1) and never depends on the Sub registry.
        let profile_id = key.profile();

        // Failed clears the dedup entry regardless of state. The Failed
        // Effect produced no observation worth deduplicating against, so
        // the next stable verdict at the same key must fire fresh.
        if matches!(result, EffectOutcome::Failed { .. })
            && let Some(p) = self.profiles.get_mut(profile_id)
        {
            p.fired_subs.remove(key);
        }

        // Resolve the action under a short read borrow, then mutate.
        // The `BurstFinish` payload on the Active variant is the source
        // of truth for the Reap-vs-Rebase decision — a Sub that detaches
        // between the prior `outstanding == N` step and this completion
        // mutates the directive in place via `mark_active_for_reap`,
        // and the match arm below sees the post-flip value.
        let phase_action = match self
            .profiles
            .get(profile_id)
            .map(specter_core::Profile::state)
        {
            Some(ProfileState::Active(ActiveBurst::PostFire(post), finish)) => match &post.phase {
                PostFirePhase::Awaiting { outstanding, .. } => {
                    if *outstanding <= 1 {
                        // Exhaustive match on BurstFinish — the
                        // boolean ternary is gone.
                        match finish {
                            BurstFinish::Reap => AwaitAction::Reap,
                            BurstFinish::ReturnToIdle => AwaitAction::Rebase,
                        }
                    } else {
                        AwaitAction::Decrement
                    }
                }
                PostFirePhase::Rebasing => AwaitAction::Diagnose,
            },
            // PreFire phases (Batching / Verifying / Draining), Idle,
            // Pending, stale Profile (None): not waiting for this
            // completion — a late arrival the engine no longer tracks.
            _ => AwaitAction::Diagnose,
        };

        match phase_action {
            AwaitAction::Decrement => {
                if let Some(post) = self
                    .profiles
                    .get_mut(profile_id)
                    .and_then(specter_core::Profile::post_fire_burst_mut)
                    && let PostFirePhase::Awaiting {
                        ref mut outstanding,
                        ..
                    } = post.phase
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
    /// Atomic apply of *both* halves of the [`WatchRegistryDiff`] in
    /// the canonical order:
    ///
    /// 1. **Sub `removed`** — `detach_sub_inner` decrements each
    ///    Sub's Profile refcount (reaping the Profile if it hits zero,
    ///    deferring if active).
    /// 2. **Sub `modified`** — remove-then-add (`config_hash` may
    ///    change ⇒ different Profile).
    /// 3. **Sub `added`** — `attach_sub_inner` materialises the anchor
    ///    and registers the Sub.
    /// 4. **Promoter `removed`** — `reap_promoter_inner` cancels the
    ///    in-flight probe, detaches every dynamic Sub, releases the
    ///    per-Resource watch_demand contributions, and removes the
    ///    Promoter from the registry.
    /// 5. **Promoter `modified`** — wholesale: `reap_promoter_inner`
    ///    then `attach_promoter_inner`. The `name` survives across
    ///    the cycle (the diff keys on it), but the underlying
    ///    `PromoterId` is freshly minted; the bin reconciles via the
    ///    `PromoterAttached` / `PromoterReaped` diagnostic stream.
    /// 6. **Promoter `added`** — `attach_promoter_inner` runs
    ///    descent or immediate-Active per the literal-prefix
    ///    materialisation outcome.
    ///
    /// Sub-side runs first so that any Promoter modification observes
    /// a registry that already reflects the freshly-applied static
    /// Subs — relevant for cross-Promoter / static-Sub `Profile` dedup.
    /// Within each kind, removals run before additions so a
    /// name-recycling rename doesn't transiently alias against the old
    /// entry.
    ///
    /// Parent-edge recompute is **lazy**: each `detach_sub_inner` /
    /// `attach_sub_inner` calls the appropriate
    /// `StabilityIndex::recompute_parent_edges_for_*` variant. All ops
    /// merge into a single sorted `StepOutput`.
    pub(crate) fn on_config_diff(
        &mut self,
        diff: WatchRegistryDiff,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let WatchRegistryDiff { subs, promoters } = diff;

        // ---- Sub side ----
        for sub_id in subs.removed {
            self.detach_sub_inner(sub_id, out);
        }
        for (sub_id, req) in subs.modified {
            self.detach_sub_inner(sub_id, out);
            let _ = self.attach_sub_inner(req, now, out);
        }
        for req in subs.added {
            let _ = self.attach_sub_inner(req, now, out);
        }

        // ---- Promoter side ----
        for pid in promoters.removed {
            self.reap_promoter_inner(pid, out);
        }
        for (pid, req) in promoters.modified {
            self.reap_promoter_inner(pid, out);
            let _ = self.attach_promoter_inner(req, now, out);
        }
        for req in promoters.added {
            let _ = self.attach_promoter_inner(req, now, out);
        }
        // The single-StepOutput sort happens at `step`'s caller.
    }

    /// Dispatch a [`Input::WatchOpRejected`].
    ///
    /// The Sensor failed to install a kernel watch (typically `EMFILE` /
    /// `ENFILE` on FD exhaustion). Three things must happen:
    ///
    /// 1. [`specter_core::Tree::vacate`] the rejected slot — clear
    ///    every contribution and zero `suppress_count` atomically, so
    ///    the engine's view of "is this slot watched?" matches reality.
    /// 2. Walk every Profile that holds a per-Profile claim on
    ///    `resource` (anchor / watch-root parent / descent prefix) and
    ///    clean up its bookkeeping — otherwise the Profile flag
    ///    contradicts the post-vacate counter, and any subsequent
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
        // (we'll mutate self.profiles / self.promoters in the loops)
        // and we want a stable view of the pre-clamp world: a Profile
        // that's `Pending(d)` with `d.current_prefix() == resource` must
        // be detected here, because the helpers we run below transition
        // the Profile to Idle. Same for Promoter state-flips below.
        let mut anchor_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut parent_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut descent_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        for (pid, p) in self.profiles.iter() {
            if matches!(p.anchor_claim(), AnchorClaim::Held) && p.resource == resource {
                anchor_claimers.push(pid);
            }
            if p.watch_root_parent == Some(resource) {
                parent_claimers.push(pid);
            }
            if let ProfileState::Pending(d) = p.state()
                && d.current_prefix() == resource
            {
                descent_claimers.push(pid);
            }
        }

        // Promoter-side claimers — disjoint pair: a single Promoter can
        // claim `resource` either via its literal-prefix descent (5a)
        // or via an `Active` proxy (5b), never both at once (state is
        // a sum-type). Two SmallVecs keep the per-claim purge loops
        // structurally distinct.
        let mut promoter_descent_claimers: smallvec::SmallVec<[PromoterId; 2]> =
            smallvec::SmallVec::new();
        let mut promoter_proxy_claimers: smallvec::SmallVec<[PromoterId; 2]> =
            smallvec::SmallVec::new();
        for (qid, q) in self.promoters.iter() {
            match &q.state {
                PromoterState::PrefixPending(d) if d.current_prefix() == resource => {
                    promoter_descent_claimers.push(qid);
                }
                PromoterState::Active { proxies } if proxies.contains_key(&resource) => {
                    promoter_proxy_claimers.push(qid);
                }
                PromoterState::PrefixPending(_) | PromoterState::Active { .. } => {}
            }
        }

        // Atomic terminus for the rejected slot: clear the
        // contributions map AND zero `suppress_count`, emitting the
        // closing `Unwatch` / `Unsuppress` pair. The per-claimer loops
        // below run their owner-bookkeeping and call `sub_watch` /
        // `sub_suppress`, which short-circuit on the post-vacate state
        // (absent key / zero counter). One slot, one terminus — and
        // the suppress balance is preserved by short-circuit, not by
        // deferring the closing emission.
        self.tree.vacate(resource, out);

        // Anchor claimers: synthesise an anchor-loss. `finalize_anchor_lost`
        // cancels any in-flight Active probe, releases the anchor flag
        // (silent no-op on the post-vacate contributions map), and
        // finishes the burst to Idle. `finish_burst_to_idle` runs
        // `sub_suppress` against the now-zero counter — silent no-op,
        // because `vacate` already emitted the closing `Unsuppress`
        // above. Net Sensor ops match the pre-vacate accounting.
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
            self.cancel_owner_probe(ProbeOwner::Profile(pid), out);
            self.release_descent_prefix_claim(pid, out);
            out.diagnostics.push(Diagnostic::ProfileClaimPurged {
                profile: pid,
                claim: ClaimKind::DescentPrefix,
                resource,
                failure,
            });
        }

        // Promoter descent prefix purge — mirrors the Profile descent
        // loop. Cancel-before-release is unconditional: an in-flight
        // descent probe targets `current_prefix == resource` by
        // construction. Releasing transitions the Promoter to
        // `Active{empty}`. There is no recovery channel for the
        // literal prefix in v1; the Promoter is stranded.
        for qid in promoter_descent_claimers {
            self.cancel_owner_probe(ProbeOwner::Promoter(qid), out);
            self.release_promoter_descent_prefix_claim(qid, out);
            out.diagnostics.push(Diagnostic::PromoterClaimPurged {
                promoter: qid,
                claim: PromoterClaimKind::DescentPrefix,
                resource,
                failure,
            });
        }

        // Promoter `Active` proxy purge — mirror with one twist:
        // cancel only when the in-flight enumeration targets THIS
        // resource. A probe targeting a SIBLING proxy of the same
        // Promoter is unaffected by this rejection and stays in
        // flight. The cancel-first contract on
        // `release_promoter_proxy_claim` gates on this exact
        // condition.
        for qid in promoter_proxy_claimers {
            let target_matches = matches!(
                self.probe_channel
                    .kind_for(ProbeOwner::Promoter(qid)),
                Some(crate::probe_channel::OpenKind::PromoterEnumerating { target })
                    if *target == resource,
            );
            if target_matches {
                self.cancel_owner_probe(ProbeOwner::Promoter(qid), out);
            }
            self.release_promoter_proxy_claim(qid, resource, out);
            out.diagnostics.push(Diagnostic::PromoterClaimPurged {
                promoter: qid,
                claim: PromoterClaimKind::ActiveProxy,
                resource,
                failure,
            });
        }
    }

    /// Sensor reports it dropped events at the kernel level (inotify's
    /// `IN_Q_OVERFLOW`). Reseed every Profile in scope so the engine's
    /// post-probe `dispatch_seed_ok` re-establishes baseline against
    /// disk reality and runs drift detection. Active-mode drift
    /// (`baseline.hash() != current.hash()`) fires once for every
    /// Subtree-scoped key in `fired_subs`, then rebases.
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
            match p.state() {
                ProfileState::Idle => {
                    self.start_seed_burst(pid, now, out);
                }
                ProfileState::Active(_, _) => {
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
            }
        }

        // Snapshot the Promoter set BEFORE any reseed dispatch — the
        // dispatch loop may mutate `pending_enumerations` and emit
        // probes, but the membership of `self.promoters` is stable
        // across the loop (no Promoter reaps, no fresh attaches in
        // this code path).
        let promoters_to_reseed: smallvec::SmallVec<[PromoterId; 4]> = match scope {
            OverflowScope::Global => self.promoters.iter().map(|(qid, _)| qid).collect(),
            OverflowScope::Resource(r) => self.promoters_in_subtree(r),
        };

        for qid in promoters_to_reseed {
            // Project the relevant state into a local enum so the
            // borrow on `self.promoters.get(qid)` ends before the
            // `&mut self` calls below (probe_channel.open,
            // dispatch_next_enumeration). Stale id ⇒ skip without
            // emitting the reseed diagnostic — the Promoter is gone.
            let qowner = ProbeOwner::Promoter(qid);
            let channel_open = self.probe_channel.correlation_for(qowner).is_some();
            let action = match self.promoters.get(qid) {
                None => continue,
                Some(q) => match &q.state {
                    PromoterState::PrefixPending(d) if !channel_open => {
                        PromoterReseedAction::DescentProbe(d.current_prefix())
                    }
                    // PrefixPending with in-flight descent probe: the
                    // probe's response will reflect the post-overflow
                    // state. No double-probe.
                    PromoterState::PrefixPending(_) => PromoterReseedAction::Skip,
                    PromoterState::Active { proxies } => {
                        PromoterReseedAction::Enumerate(proxies.keys().copied().collect())
                    }
                },
            };

            match action {
                PromoterReseedAction::DescentProbe(prefix) => {
                    let correlation = self
                        .probe_channel
                        .open(qowner, crate::probe_channel::OpenKind::PromoterDescent);
                    let target_path = self.tree.path_of(prefix).unwrap_or_default();
                    Self::emit_descent_probe(qowner, correlation, target_path, out);
                }
                PromoterReseedAction::Enumerate(proxy_keys) => {
                    // Enqueue every proxy. Single-slot drain processes
                    // one at a time via the `dispatch_next` chain on
                    // each response. Empty proxies vec is a no-op.
                    if let Some(qmut) = self.promoters.get_mut(qid) {
                        for r in proxy_keys {
                            qmut.pending_enumerations.insert(r);
                        }
                    }
                    self.dispatch_next_enumeration(qid, out);
                }
                PromoterReseedAction::Skip => {}
            }

            out.diagnostics
                .push(Diagnostic::PromoterReseededForOverflow { promoter: qid });
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
            .filter(|(_, p)| p.resource == r || self.tree.ancestors(p.resource).any(|a| a == r))
            .map(|(pid, _)| pid)
            .collect()
    }

    /// Promoter analogue of [`Self::profiles_in_subtree`]. A Promoter is
    /// "in the subtree rooted at `r`" when its watched slot (descent
    /// prefix in `PrefixPending`, OR any proxy in `Active`) is `r` or
    /// has `r` on its ancestor chain.
    ///
    /// Symmetric handling across backends: only FSEvents-style
    /// per-stream overflows ([`OverflowScope::Resource`]) reach here in
    /// practice; v1 inotify always emits [`OverflowScope::Global`].
    /// Worst-case `O(promoters × proxies × tree-depth)`. Acceptable
    /// under healthy invariants.
    fn promoters_in_subtree(&self, r: ResourceId) -> smallvec::SmallVec<[PromoterId; 4]> {
        self.promoters
            .iter()
            .filter(|(_, q)| match &q.state {
                PromoterState::PrefixPending(d) => {
                    d.current_prefix() == r
                        || self.tree.ancestors(d.current_prefix()).any(|a| a == r)
                }
                PromoterState::Active { proxies } => proxies
                    .keys()
                    .any(|&p| p == r || self.tree.ancestors(p).any(|a| a == r)),
            })
            .map(|(qid, _)| qid)
            .collect()
    }

    /// Start a new burst (Seed if no baseline yet, Standard if baseline
    /// established); pre-fire `Active` → fold the event through
    /// `event_drives_batching` (which accumulates `dirty_resources` +
    /// `force_walk_resources`, emits a Cancel iff a probe was in flight,
    /// and arms a fresh settle timer); post-fire `Active`
    /// (`Awaiting` / `Rebasing`) → defer the event to the next post-fire
    /// probe by appending `event_resource` to `force_walk_resources` and
    /// pushing an `EventAbsorbedByFireTail` diagnostic.
    ///
    /// `event_resource` is the `FsEvent`'s source. The pre-fire path
    /// extends both `dirty_resources` (LCA basis) and
    /// `force_walk_resources` (mtime-skip defeat); the post-fire absorb
    /// path extends only `force_walk_resources` because the rebase
    /// probe targets the anchor unconditionally and has no use for an
    /// LCA. The absorb's force_walk hint closes the carve-out where a
    /// content-only descendant edit during the fire-tail would have
    /// left the post-rebase baseline with stale leaves: POSIX content
    /// edits don't bump parent dir mtime, so the rebase walker would
    /// mtime-skip without the hint.
    ///
    /// `event` is threaded through purely for the absorb diagnostic so
    /// the operator can correlate logs to the deferred FsEvent.
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
        match p.state() {
            ProfileState::Idle => {
                if p.current().is_some() {
                    self.start_standard_burst(profile_id, event_resource, now, out);
                } else {
                    self.start_seed_burst(profile_id, now, out);
                }
            }
            // The post-fire absorb arm is *the* typed-disjoint path from
            // the pre-fire `event_drives_batching` arm: mutating
            // `force_walk_resources` and emitting `EventAbsorbedByFireTail`
            // belongs to `PostFireBurst` alone, and the helper owns the
            // mutation in `burst.rs` so `transitions.rs` never reaches
            // for burst internals.
            ProfileState::Active(ActiveBurst::PostFire(_), _) => {
                self.absorb_event_into_fire_tail(profile_id, event_resource, event, out);
            }
            ProfileState::Active(ActiveBurst::PreFire(_), _) => {
                self.event_drives_batching(profile_id, event_resource, now, out);
            }
            // Pending Profiles never reach here — `covering_profiles`
            // filters them at the source. Defensive no-op.
            ProfileState::Pending(_) => {}
        }
    }

    /// Anchor terminal event (Removed/Renamed/Revoked at `Profile.resource`).
    /// Anchor-terminal dispatcher. Splits on whether every Sub on the
    /// Profile is dynamic (originates from a Promoter) versus the
    /// mixed/static case.
    ///
    /// **All-dynamic** ⇒ [`Self::on_anchor_terminal_all_dynamic`]: the
    /// Profile has no static recovery channel; the Promoter re-promotes
    /// on path reappearance, so the Profile is reaped entirely (anchor,
    /// descendants, descent prefix, watch-root parent — the full
    /// quartet) and each source Promoter is notified that its dynamic
    /// Sub has reaped. I-Recovery-Split: the predicate is total over
    /// non-empty Subs.
    ///
    /// **Mixed or pure-static** ⇒ [`Self::finalize_anchor_lost`]: the
    /// existing recovery flow runs. The dynamic Subs (if any) stay
    /// attached — the static Sub keeps the Profile alive via
    /// `Profile.watch_root_parent`'s recovery channel. On
    /// re-materialisation, the Promoter's enumeration's
    /// `dynamic_subs.contains_key(anchor_resource)` check returns
    /// `true` (the engine never minted a fresh Sub for an already-known
    /// anchor), so no engine work is needed for correctness — only the
    /// static Sub's recovery flow drives the burst.
    ///
    /// The empty-Subs case is structurally unreachable: a Profile with
    /// no Subs reaped on the last detach. Routed defensively to
    /// `finalize_anchor_lost` for idempotence.
    fn on_anchor_terminal_event(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let subs = self.subs.at(profile_id);
        if subs.is_empty() {
            self.finalize_anchor_lost(profile_id, out);
            return;
        }
        let all_dynamic = subs.iter().all(|sid| {
            self.subs
                .get(*sid)
                .is_some_and(|s| s.source_promoter.is_some())
        });
        if all_dynamic {
            self.on_anchor_terminal_all_dynamic(profile_id, out);
        } else {
            self.finalize_anchor_lost(profile_id, out);
        }
    }

    /// All-dynamic anchor-terminal teardown. Notifies each source
    /// Promoter (drops the Sub from the Promoter's `dynamic_subs`
    /// map), removes every dynamic Sub from `SubRegistry`, then reaps
    /// the Profile entirely.
    ///
    /// The reap delegates to [`Engine::reap_profile`] /
    /// [`Engine::finish_burst_to_idle`] depending on the Profile's
    /// state — mirrors `detach_sub_inner`'s lifecycle dispatch but
    /// force-runs the deferred-end path synchronously (the anchor is
    /// dead now; we cannot wait for the burst to complete naturally
    /// against a stale anchor).
    ///
    /// Idempotent: re-entering on an already-reaped Profile finds
    /// `subs.at(profile_id)` empty (caller filtered) and never enters
    /// here. The Sub-removal loop is also idempotent: a stale Sub id
    /// on the Profile's `by_profile` list is a structural impossibility
    /// (the registry maintains by_profile in lockstep with subs).
    fn on_anchor_terminal_all_dynamic(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        // 1. Close the probe channel — Active+Verifying may have one
        // in flight. Idempotent on a closed channel.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);

        // 2. Resolve the anchor resource + path ONCE for the per-Sub
        // loop. Every dynamic Sub on this Profile shares the same
        // anchor by the `(resource, config_hash)` find-or-create dedup
        // in `attach_sub_inner`; the resource is precisely the key
        // `try_promote` stamped into each source Promoter's
        // `dynamic_subs` map, and the path is the diagnostic payload.
        // The anchor slot is alive at this point — the Profile is not
        // yet reaped (the slot's anchor_claim contribution is released
        // only by `reap_profile` below) — so `path_of` returns
        // `Some(_)`. Fallbacks are defense-in-depth.
        let anchor_resource: ResourceId = self
            .profiles
            .get(profile_id)
            .map(|p| p.resource)
            .unwrap_or_default();
        let anchor_path: PathBuf = self.tree.path_of(anchor_resource).unwrap_or_else(|| {
            debug_assert!(
                false,
                "on_anchor_terminal_all_dynamic: tree.path_of returned None for live Profile \
                 anchor (profile = {profile_id:?}, resource = {anchor_resource:?})",
            );
            PathBuf::new()
        });

        // 3. Notify each source Promoter; remove each dynamic Sub from
        // SubRegistry. SubRegistry's `by_profile` index drops the
        // entry on the last remove, so the post-loop registry has no
        // back-references for this Profile.
        let dynamic_subs: SmallVec<[SubId; 2]> = self.subs.at(profile_id).iter().copied().collect();
        for sid in dynamic_subs.iter().copied() {
            if let Some(pid) = self.subs.get(sid).and_then(|s| s.source_promoter) {
                self.on_dynamic_sub_reaped(pid, sid, anchor_resource, &anchor_path, out);
            }
        }
        for sid in dynamic_subs {
            let _ = self.subs.remove(sid);
        }

        // 4. Reap the Profile. Active Profiles need their burst force-
        // ended via `finish_burst_to_idle`; Idle / Pending Profiles
        // reap synchronously. The Active branch flips
        // [`BurstFinish::Reap`] via `mark_active_for_reap` so
        // `finish_burst_to_idle` runs `reap_profile` internally with
        // `via = DeferredFromBurst` (single source of truth for the
        // four-claim release + ProfileMap detach).
        let marked = self
            .profiles
            .get_mut(profile_id)
            .is_some_and(specter_core::Profile::mark_active_for_reap);
        if marked {
            self.finish_burst_to_idle(profile_id, out);
        } else if self.profiles.get(profile_id).is_some() {
            // Non-Active arm: the all-dynamic teardown reached a
            // Profile in Idle or Pending. Reap inline.
            self.reap_profile(profile_id, ReapTrigger::Immediate, out);
        }
    }

    /// Finalize the loss of a Profile's anchor: cancel any in-flight
    /// probe, release the anchor's `watch_demand` contribution, drop the
    /// stale `baseline` / `current` snapshots, and finish the burst to
    /// Idle if Active.
    ///
    /// **`watch_root_parent` is intentionally preserved.** After anchor
    /// loss the Profile remains "interested" in anchor reappearance via
    /// the parent's `StructureChanged`. `start_pending_recovery` triggers
    /// descent on such an event; releasing the parent watch here would
    /// close the recovery channel. The contribution is released only
    /// when the Profile itself reaps (`reap_profile` →
    /// `release_watch_root_parent_claim`). Sibling helpers — anchor,
    /// descendants, descent prefix — *are* released here; the asymmetry
    /// is by design.
    ///
    /// **Ordering.** The anchor release runs BEFORE
    /// `finish_burst_to_idle`, so any deferred `reap_profile`
    /// (`reap_pending`) sees an `AnchorClaim::None` and skips its
    /// redundant release inside `reap_profile::release_anchor_claim`.
    /// This mirrors the `dispatch_*_vanished/failed` discipline.
    /// Reverse-ordering would have `finish_burst_to_idle` invoke
    /// `reap_profile`, which would release the anchor; the
    /// post-`finish` release would then see an absent contribution
    /// and silently no-op — correct but redundant. The
    /// "release-then-finish" ordering keeps the cleanup ordered.
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
        if matches!(p.state(), ProfileState::Pending(_)) {
            return;
        }
        // Capture `was_active` BEFORE discard_anchor_state. The helper
        // does not mutate Profile.state (only `finish_burst_to_idle`
        // does), so the read is order-insensitive in v1; pinning it
        // before the helper guards against any future helper change
        // that touches state.
        let was_active = matches!(p.state(), ProfileState::Active(_, _));

        // Idempotent: emits Cancel iff the probe channel is open
        // (Active+Verifying ⇒ channel open). For Active+Batching /
        // Draining no probe is in flight and the helper is a no-op —
        // structural equivalent of the prior `was_verifying` snapshot.
        // Required by discard_anchor_state's cancel-first contract.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);

        // Discard runs BEFORE finish_burst_to_idle. The release-helpers
        // inside emit `AnchorClaim::None` and clear `Profile.kind`
        // before any deferred `reap_profile` (`reap_pending`) fires
        // from `finish_burst_to_idle` — preserves the trichotomy
        // invariant `!(Pending && Held)` across the eventual
        // `start_pending_recovery` transition, and lets the next Seed
        // burst route through the kind-agnostic Subtree probe rather
        // than misroute against a recreated anchor of a different
        // shape.
        self.discard_anchor_state(profile_id, out);

        if was_active {
            self.finish_burst_to_idle(profile_id, out);
        }
    }

    /// (Seed, Ok).
    ///
    /// Graft the response into `Profile.current` at the burst's
    /// `probe_target` (= anchor for Seeds). Hash-only drift check via
    /// [`Engine::seed_drift_observed`] — `true` ⇒ fire `emit_effects`
    /// once over the Subtree subset of `fired_subs` and route through
    /// the same fire-tail as a Standard burst (`emit_effects` count > 0
    /// ⇒ `transition_to_awaiting`; the eventual rebase probe captures
    /// the post-command tree). Otherwise rebase directly: `baseline :=
    /// current` and finish.
    ///
    /// Fresh-attach Seed cannot enter the drift branch — `fired_subs`
    /// is empty by construction at fresh attach, so
    /// `seed_drift_observed` returns false. The drift branch fires only
    /// on recovery / post-Effect rebase paths where the Profile has
    /// already emitted at least one Effect.
    fn dispatch_seed_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Seed always targets anchor — `probe_target` was set to the
        // anchor at `start_seed_burst` / `transition_to_verifying`.
        // First-classify of `Profile.kind` happens atomically with
        // `Profile.current` inside `apply_snapshot`'s `install_*_current`
        // call (see those setters' rustdoc).
        //
        // Seed only reaches here from `Active(PreFire(Verifying))` (the
        // probe-channel dispatcher). The fallback to `p.resource` on
        // non-Active arms is defensive — never observed in v1's
        // single-threaded step, but matches the prior `unwrap_or(anchor)`
        // semantics one-for-one.
        let target = match self.profiles.get(profile_id) {
            Some(p) => match p.state() {
                ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.probe_target,
                _ => p.resource,
            },
            None => return,
        };

        // Boundary check: a walker-contract violation (Dir response for
        // a File-kinded Profile, or symmetric) routes through
        // `finalize_anchor_lost`. Structurally unreachable in v1; the
        // boundary exists as defense-in-depth.
        if !self.kind_agrees_or_finalize(profile_id, &snapshot, out) {
            return;
        }

        self.apply_snapshot(profile_id, target, snapshot, out);

        // Fire Effects only for the Subtree subset of `fired_subs` when
        // drift is observed (post-graft current.hash() differs from the
        // last settled state — `baseline.hash()` in active mode or the
        // `last_settled_hash_at_loss` witness in survival mode). Every
        // Sub that has a fire history on this Profile re-fires once,
        // unconditionally — drift is a per-Profile signal under the
        // cross-field invariant; per-key narrowing is gone.
        //
        // **Why two rebases on the drift branch.** Seed's semantic is
        // `baseline := observed`; the drift detection fires the
        // recovery Effects FIRST, then completes that semantic by
        // calling `rebase_baseline` before `transition_to_awaiting`.
        // The post-Rebasing rebase (in `dispatch_rebase_ok`) sits on
        // top — it's the Standard-style post-command refresh, capturing
        // the disk state AFTER the recovery commands ran. The two
        // rebases serve different roles, not duplicate ones: this one
        // seals the Seed observation; the post-Rebasing one seals the
        // post-Effect view. Standard bursts skip this pre-Awaiting
        // rebase because their baseline was already authoritative at
        // burst start; only Seed completes a `baseline := observed`
        // semantic mid-cycle.
        //
        // The pre-Awaiting rebase here is also forward-defensive: no
        // current code path reads `Profile.baseline` during Awaiting /
        // Rebasing (the absorb arm touches only
        // `PostFireBurst.force_walk_resources`; `transition_to_rebasing`
        // ships `Profile.current`, not `baseline`), but pinning the
        // Profile's view here keeps the cross-field invariant intact
        // against any future absorb / rebase path that does read it.
        if self.seed_drift_observed(profile_id) {
            // Project `DedupKey::Subtree { sub, profile }` → `sub`. The
            // `profile` field is constant across the iteration (it's
            // `profile_id`), and the PerFile variant is filtered out
            // upfront — both of which the `&[SubId]` payload makes
            // unrepresentable. The resulting slice is sorted by `SubId`
            // because `fired_subs` is a `BTreeSet<DedupKey>` and
            // `DedupKey::Subtree`'s `Ord` decomposes to `(sub, profile)`
            // with profile constant.
            let drifted: SmallVec<[SubId; 2]> = self
                .profiles
                .get(profile_id)
                .map(|p| {
                    p.fired_subs
                        .iter()
                        .filter_map(|k| match k {
                            DedupKey::Subtree { sub, .. } => Some(*sub),
                            DedupKey::PerFile { .. } => None,
                        })
                        .collect()
                })
                .unwrap_or_default();
            // `drifted` may be empty if the Profile has only PerFile
            // fires (Subtree filter excludes them) or if every Subtree Sub
            // detached between record and now (a same-step race —
            // `detach_sub_inner` normally purges fired_subs). Fall
            // through to the no-drift finish path in either case.
            if !drifted.is_empty() {
                let outcome =
                    self.emit_effects(profile_id, EmitMode::SeedDrift { drifted: &drifted }, out);
                if outcome.count > 0 {
                    if let Some(p) = self.profiles.get_mut(profile_id) {
                        p.rebase_baseline();
                    }
                    self.transition_to_awaiting(profile_id, outcome.count, now, out);
                    return;
                }
            }
        }

        // Non-drift Seed (fresh attach, no-drift recovery, or
        // dedup-hash-suppressed drift): rebase and finish. No Effect
        // fires, no Awaiting tail.
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.rebase_baseline();
        }
        self.finish_burst_to_idle(profile_id, out);
    }

    /// Decide whether a Seed-Ok should fire conservative-recovery Effects.
    /// Returns `true` iff the Profile has fired before AND the post-graft
    /// `current` snapshot's anchor-rooted hash differs from the most
    /// recent settled state.
    ///
    /// Source of "most recent settled state" depends on mode (mutually
    /// exclusive under the cross-field invariant
    /// `baseline.is_some() ⇒ last_settled_hash_at_loss.is_none()`):
    /// - **Active mode** (`baseline.is_some()`, witness `None`):
    ///   `baseline.hash()` is the witness directly. Covers
    ///   `on_sensor_overflow` reseed, where the prior baseline persists
    ///   across the overflow but `current` is freshly captured.
    /// - **Survival mode** (`baseline.is_none()`, witness `Some`):
    ///   [`crate::claims::Engine::discard_anchor_state`] stashed the
    ///   pre-loss `baseline.hash()` into `last_settled_hash_at_loss`.
    ///   Covers anchor-loss recovery via descent → Seed-Ok.
    ///
    /// The witness is checked first because, when populated, the cross-
    /// field invariant guarantees baseline is `None` — survival mode is
    /// authoritative. The fresh-attach case (both `None`, `fired_subs`
    /// empty by construction) returns `false` and preserves "fresh Seed
    /// never fires Effect".
    ///
    /// The boolean answer is per-Profile; the caller
    /// ([`Engine::dispatch_seed_ok`]) builds the SeedDrift fire filter
    /// from the Subtree subset of [`Profile::fired_subs`].
    fn seed_drift_observed(&self, profile_id: ProfileId) -> bool {
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        if p.fired_subs.is_empty() {
            return false;
        }
        let Some(current) = p.current() else {
            return false;
        };
        let curr = current.hash();
        if let Some(witness) = p.last_settled_hash_at_loss() {
            return witness != curr;
        }
        match p.baseline() {
            Some(b) => b.hash() != curr,
            None => false,
        }
    }

    /// (Seed, Vanished).
    ///
    /// Symmetric with `dispatch_standard_vanished` (treats Vanished as an
    /// anchor-disappearance signal): releases the anchor's `watch_demand`
    /// contribution so the trichotomy invariant in `reap_profile` —
    /// `!(Pending && AnchorClaim::Held)` — survives the eventual
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
        // Discard runs BEFORE finish_burst_to_idle so any deferred
        // `reap_profile` (reap_pending) sees `AnchorClaim::None` —
        // preserves the trichotomy invariant `!(Pending && Held)`
        // across the eventual `start_pending_recovery` transition.
        // Clearing `Profile.kind` lets the next Seed burst route
        // through the kind-agnostic Subtree probe rather than
        // misrouting against a recreated anchor of a different shape.
        self.discard_anchor_state(profile_id, out);
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
        self.discard_anchor_state(profile_id, out);
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
        //
        // Standard Verifying lives in `Active(PreFire(_))`; the fallback
        // arms match the prior `unwrap_or(anchor)` semantics for
        // defensive non-Active routes (production never reaches them).
        let (target, prior_target_hash, dirty_zero) = match self.profiles.get(profile_id) {
            Some(p) => {
                let target = match p.state() {
                    ProfileState::Active(ActiveBurst::PreFire(pre), _) => pre.probe_target,
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

        // Boundary check before any snapshot commit. A walker-contract
        // violation (Dir response on a File-kinded Profile, or
        // symmetric) routes through `finalize_anchor_lost` — the
        // verdict computed above is irrelevant on that branch.
        if !self.kind_agrees_or_finalize(profile_id, &snapshot, out) {
            return;
        }

        // Apply AFTER computing stability — the verdict needs the
        // pre-update prior. `apply_snapshot` routes Dir through `graft`
        // (splice + reconcile + atomic kind/current commit) and File
        // through the inline `install_file_current` setter.
        self.apply_snapshot(profile_id, target, snapshot, out);

        if is_stable && dirty_zero {
            // Stable + dirty=0 → fire Effect. Awaiting on count > 0;
            // finish-to-Idle on count == 0 (dedup-hash suppressed
            // everything, no Subs matched, or `reap_pending` skipped the
            // emit). baseline is NOT pinned here on the firing branch —
            // it will rebase when the Rebasing probe response lands
            // (`dispatch_rebase_ok`). Standard mode: every matching Sub
            // emits, suppress controlled by `forced`.
            //
            // Standard intentionally skips the pre-Awaiting rebase that
            // `dispatch_seed_ok`'s drift branch performs: Standard's
            // baseline was authoritative at burst start; only Seed
            // completes a `baseline := observed` semantic mid-cycle
            // (see `dispatch_seed_ok` for the rationale).
            let outcome = self.emit_effects(profile_id, EmitMode::Standard { forced }, out);
            if outcome.count > 0 {
                self.transition_to_awaiting(profile_id, outcome.count, now, out);
            } else {
                self.finish_burst_to_idle(profile_id, out);
            }
        } else if is_stable {
            // Stable + dirty>0 → Draining. The stable snapshot lives on
            // `Profile.current` (just spliced in by graft); the reconfirm
            // probe compares against `current`. No need to pin a duplicate
            // snapshot on the phase variant.
            self.transition_to_draining(profile_id, out);
        } else if forced {
            // Not-stable + forced → fire Effect with forced=true. Same
            // Awaiting / finish-to-Idle branching as the stable + dirty=0
            // case — `forced` overrides dedup-hash suppression inside
            // `emit_effects`, but a Profile with no matching Subs still
            // returns count == 0.
            let outcome = self.emit_effects(profile_id, EmitMode::Standard { forced: true }, out);
            if outcome.count > 0 {
                self.transition_to_awaiting(profile_id, outcome.count, now, out);
            } else {
                self.finish_burst_to_idle(profile_id, out);
            }
        } else {
            // Not-stable + !forced → re-arm debounce in `Batching`. By
            // construction no probe is in flight (we're inside the
            // response handler), so no Cancel is emitted.
            self.unstable_response_drives_batching(profile_id, now, out);
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
    /// `reap_profile` (`reap_pending`) sees `AnchorClaim::None` and skips
    /// a redundant release. Without this ordering the post-`finish`
    /// release would underflow the now-zero `watch_demand` counter
    /// (debug-assert panic; release-build silent leak).
    fn dispatch_standard_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Standard,
        });
        self.discard_anchor_state(profile_id, out);
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
        self.discard_anchor_state(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Rebase, Ok). Post-fire probe response — graft the post-command
    /// snapshot into `Profile.current`, rebase `baseline := current`,
    /// finish the burst to Idle. The Rebasing probe always targets the
    /// anchor (set by `transition_to_rebasing`); no stability verdict
    /// applies (we just fired, drift is expected).
    ///
    /// **No drift check.** Recovery / post-Effect drift detection is
    /// gated on Seed-Ok in v1; Rebasing is a phase of the Standard burst
    /// (or the Seed burst's drift tail), not a fresh Seed, so the hash
    /// check would either fire-loop (every fire writes a new hash;
    /// the next rebase would see drift; loop) or be silently a no-op
    /// (the post-fire hash matches itself by construction). The
    /// helper deliberately avoids `seed_drift_observed` here.
    fn dispatch_rebase_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        out: &mut StepOutput,
    ) {
        // Rebasing targets the anchor by construction
        // (`transition_to_rebasing` always probes `Profile.resource` and
        // PostFireBurst carries no `probe_target` field).
        let Some(target) = self.profiles.get(profile_id).map(|p| p.resource) else {
            return;
        };
        if !self.kind_agrees_or_finalize(profile_id, &snapshot, out) {
            return;
        }
        self.apply_snapshot(profile_id, target, snapshot, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.rebase_baseline();
        }
        self.finish_burst_to_idle(profile_id, out);
    }

    /// (Rebase, Vanished). Anchor disappeared between fire and rebase.
    /// Symmetric path with `dispatch_standard_vanished`: clear baseline /
    /// current, release the anchor watch contribution, finish the burst.
    /// Diagnostic carries the burst's actual intent so logs can
    /// distinguish Seed-driven (drift) vs Standard-driven Rebasing;
    /// the lookup falls back to `Standard` only on a stale-Profile or
    /// non-Active defensive path (the routing in `on_probe_response`
    /// guarantees `Active(Rebasing)` at entry).
    fn dispatch_rebase_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        // Read intent BEFORE discard_anchor_state. The helper does not
        // mutate Burst.intent (it leaves `state` alone — only
        // `finish_burst_to_idle` flips Active → Idle), so the read is
        // order-insensitive in v1; pinning it before the helper guards
        // against future helpers that might touch state.
        let intent = self.rebase_burst_intent(profile_id);
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent,
        });
        self.discard_anchor_state(profile_id, out);
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
        self.discard_anchor_state(profile_id, out);
        self.finish_burst_to_idle(profile_id, out);
    }

    /// Resolve the intent of the burst owning the in-flight Rebase
    /// probe. Returns the live `Burst.intent` when the Profile is
    /// `Active(_)` (the production path). Defensive fallback to
    /// [`BurstIntent::Standard`] for the structurally-unreachable
    /// non-Active branch — the `on_probe_response` routing dispatches
    /// `dispatch_rebase_*` only on `PostFirePhase::Rebasing`, and that
    /// phase is reachable only from Active. Standard is the right
    /// default because Rebasing is overwhelmingly a Standard-burst tail
    /// (Seed-driven Rebasing requires a recovery + drift, the rare
    /// path).
    fn rebase_burst_intent(&self, profile_id: ProfileId) -> BurstIntent {
        // Rebasing lives in `Active(PostFire(_))` by construction;
        // PostFireBurst carries `intent` precisely for this diagnostic
        // payload. Non-PostFire is the structurally-unreachable
        // defensive arm.
        self.profiles
            .get(profile_id)
            .and_then(|p| match p.state() {
                ProfileState::Active(ActiveBurst::PostFire(post), _) => Some(post.intent),
                ProfileState::Active(ActiveBurst::PreFire(pre), _) => Some(pre.intent),
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
        // Post-fire phases are type-impossible here: PostFireBurst carries
        // no `forced` field, and `is_timer_referenced` filters
        // BurstDeadline to PreFire only. A PostFire match arm would be
        // dead code; instead, falling through the pre-fire match keeps
        // the helper's body PreFire-typed end-to-end.
        //
        // Both pre-fire arms write `pre.forced = true` identically; only
        // the phase-decision differs (Batching/Draining → transition to
        // Verifying on the next probe; Verifying → wait for the in-flight
        // response, which will dispatch with `forced = true`). Lifting
        // the write out makes "burst-deadline elapsed ⇒ forced fire on
        // next emission" the helper's first statement.
        let needs_verify = if let Some(pre) = self
            .profiles
            .get_mut(profile_id)
            .and_then(specter_core::Profile::pre_fire_burst_mut)
        {
            pre.forced = true;
            matches!(
                &pre.phase,
                PreFirePhase::Batching { .. } | PreFirePhase::Draining,
            )
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
    /// **Zombie burst short-circuit.** A burst carrying
    /// [`BurstFinish::Reap`] has no consumer for the rebased baseline
    /// — its Profile is dying. Skip the rebase probe entirely and route
    /// straight through `finish_burst_to_idle`, which runs the
    /// `propagate(-1) / sub_suppress` drain and then dispatches
    /// `reap_profile`. The diagnostic still fires so operators see the
    /// actuator-hang signal; only the wasted rebase round-trip is
    /// elided.
    ///
    /// Defensive: if the phase has already advanced (e.g., a race with
    /// `finalize_anchor_lost`), the helper no-ops. The
    /// `is_timer_referenced` gate already filters most non-Awaiting
    /// fires; this guard handles the residual same-step ordering window.
    fn handle_gate_deadline(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(ActiveBurst::PostFire(post), finish) = p.state() else {
            return;
        };
        let PostFirePhase::Awaiting { outstanding, .. } = &post.phase else {
            return;
        };
        let outstanding = *outstanding;
        let zombie = matches!(finish, BurstFinish::Reap);
        out.diagnostics.push(Diagnostic::AwaitGateDeadlineElapsed {
            profile: profile_id,
            outstanding,
        });
        if zombie {
            self.finish_burst_to_idle(profile_id, out);
        } else {
            self.transition_to_rebasing(profile_id, out);
        }
    }

    /// Emit Effects at a stable verdict. Routes per scope:
    /// `SubtreeRoot` Subs fire one Effect anchored at the Profile's resource;
    /// `PerStableFile` Subs fire one Effect per matching diff entry. The
    /// `Diff` is built at most once and shared across both helpers via `Arc`.
    ///
    /// `mode` ([`EmitMode`]) selects the fire mode — Standard burst
    /// stable verdict vs Seed-drift fire — and carries the per-mode
    /// configuration (Standard's `forced`; SeedDrift's pre-narrowed
    /// `drifted` key set). The variant determines:
    ///
    /// - which `SubtreeRoot` Subs fire (Standard: all; SeedDrift: only
    ///   those whose `DedupKey::Subtree` is in `drifted`),
    /// - whether dedup-hash suppression applies (Standard: yes unless
    ///   `forced`; SeedDrift: structurally unreachable),
    /// - whether `PerStableFile` Subs fire (Standard: yes; SeedDrift:
    ///   skipped — Seed-time drift is Subtree-only), and
    /// - the [`Effect::forced`] value carried into the spawned process.
    ///
    /// A burst flagged [`BurstFinish::Reap`] suppresses all emission —
    /// the Profile is on its way out (its last Sub detached mid-burst)
    /// and any Effect would fire against a Sub registry that no longer
    /// holds the Subs.
    ///
    /// Returns an [`EmitOutcome`] whose `count` is the number of Effects
    /// pushed onto `out.effects`. Callers consume this to decide whether
    /// to enter the `Awaiting` phase (`count > 0`) or short-circuit to
    /// `finish_burst_to_idle` (dedup-hash suppressed everything, no Subs
    /// matched, or the burst is flagged [`BurstFinish::Reap`]).
    fn emit_effects(
        &mut self,
        profile_id: ProfileId,
        mode: EmitMode<'_>,
        out: &mut StepOutput,
    ) -> EmitOutcome {
        let Some(p) = self.profiles.get(profile_id) else {
            return EmitOutcome::default();
        };
        // Burst carrying `BurstFinish::Reap` is on its way out. Any
        // remaining Subs (none, by construction of the directive's
        // writers) would fire against a Sub registry that no longer
        // holds them — suppress emission entirely.
        if matches!(p.state().burst_finish(), Some(BurstFinish::Reap)) {
            return EmitOutcome::default();
        }
        let resource = p.resource;
        let baseline_snap = p.baseline().cloned();
        let current_snap = p.current().cloned();
        let pattern = p.config.pattern.clone();
        // Read the cached anchor classification. `None` falls back to
        // `Dir` — the actuator's `compute_cwd` then anchors at the path
        // itself; if the actuator's later `chdir` discovers the path
        // doesn't behave as a directory, the Effect surfaces
        // `EffectOutcome::Failed`. Reaching `None` here implies a fresh
        // resource-based attach whose Seed probe hasn't returned —
        // `dispatch_seed_ok`'s fallback writes the field on the next
        // Seed-Ok.
        let anchor_kind = p.kind().unwrap_or(ResourceKind::Dir);
        // Substitution-side projection of `ScanConfig.exclude`. The
        // resolver iterates source strings; the sensor consults compiled
        // matchers. The projection is sorted at `Profile::new`.
        let exclude_strings = Arc::clone(&p.exclude_strings);

        // One Arc allocation per `emit_effects` call — every Effect we
        // emit (one per Sub × per matching diff entry for PerFile) Arc-
        // clones the same path. `Tree::path_of` already builds a fresh
        // PathBuf on every call; wrapping it once amortises that cost
        // across the whole burst's emissions.
        let anchor_path: Arc<Path> = Arc::from(self.tree.path_of(resource).unwrap_or_default());

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

        // Per-Profile structural component of B1 dedup. The full Subtree
        // suppress decision combines `nothing_changed` with a per-Sub
        // fire-history check (`fired_subs.contains(&dk)`) inside the loop:
        // a Sub that has never fired suppresses nothing — it is its own
        // "first emission" — even when the tree happens to match. Both
        // reads hit the cached `OnceLock<u128>` on each snapshot; same cost
        // class as the per-Sub map value compare it replaces.
        let nothing_changed = baseline_snap
            .as_ref()
            .zip(current_snap.as_ref())
            .is_some_and(|(b, c)| b.hash() == c.hash());

        let effect_forced = mode.effect_forced();

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
                    // SeedDrift narrows to its pre-filtered Sub set; Standard
                    // emits every Sub (modulo the suppress check below).
                    if let EmitMode::SeedDrift { drifted } = mode
                        && !drifted.contains(&sub_id)
                    {
                        continue;
                    }
                    // B1 suppress = "Sub has fired before AND tree state is
                    // unchanged since the last rebase." `fired_subs.contains`
                    // is the per-Sub fire-history gate; `nothing_changed` is
                    // the per-Profile "no change" structural signal. Both
                    // gates required: a fresh Sub on an unchanged tree must
                    // still fire its first Effect. SeedDrift's `drifted` is
                    // built from drifted keys by construction (see
                    // `seed_drift_observed` + `dispatch_seed_ok`), so the
                    // SeedDrift arm returns `false` directly — the
                    // unreachability is structural, not analytical.
                    let suppress = match mode {
                        EmitMode::Standard { forced } => {
                            !forced
                                && nothing_changed
                                && self
                                    .profiles
                                    .get(profile_id)
                                    .is_some_and(|p| p.fired_subs.contains(&dk))
                        }
                        EmitMode::SeedDrift { .. } => false,
                    };
                    if suppress {
                        continue;
                    }

                    let diff_for_effect = if needs_diff {
                        ensure_diff(&mut diff_arc)
                    } else {
                        None
                    };
                    let correlation = self.effect_correlations.next();
                    let Some(sub) = self.subs.get(sub_id) else {
                        continue;
                    };
                    out.effects.push(Effect {
                        key: dk.clone(),
                        // Subtree's target is the anchor — `resource` was
                        // captured at the function head from
                        // `Profile.resource`. Frozen-at-emit so sort
                        // survives any post-emit state churn without a
                        // ProfileMap lookup.
                        target: resource,
                        forced: effect_forced,
                        correlation,
                        diff: diff_for_effect,
                        capture_output: log_output,
                        sub_name: sub.name.clone(),
                        program: Arc::clone(&sub.program),
                        anchor_path: Arc::clone(&anchor_path),
                        anchor_kind,
                        // Subtree: no per-entry segment. The resolver
                        // derives `${specter.path}` (and `SPECTER_PATH`)
                        // from `anchor_path` directly when
                        // `target_relative` is empty.
                        target_relative: CompactString::new(""),
                        exclude: Arc::clone(&exclude_strings),
                    });
                    count = count.saturating_add(1);

                    if let Some(p) = self.profiles.get_mut(profile_id) {
                        p.fired_subs.insert(dk);
                    }
                }
                EffectScope::PerStableFile => {
                    // SeedDrift skips PerFile entirely — the drift signal
                    // is Subtree-only (PerFile keys lack the per-leaf
                    // history needed for Seed-time drift detection; see
                    // `seed_drift_observed`'s documented limitation).
                    if matches!(mode, EmitMode::SeedDrift { .. }) {
                        continue;
                    }
                    // PerStableFile implies `needs_diff = true` at
                    // Sub::from_request; diff is always built.
                    let Some(diff) = ensure_diff(&mut diff_arc) else {
                        continue;
                    };
                    let pushed = self.emit_effects_per_stable_file(
                        sub_id,
                        resource,
                        effect_forced,
                        pattern.as_ref(),
                        &diff,
                        &anchor_path,
                        anchor_kind,
                        &exclude_strings,
                        out,
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
    #[allow(clippy::too_many_arguments)]
    fn emit_effects_per_stable_file(
        &mut self,
        sub_id: SubId,
        anchor: ResourceId,
        forced: bool,
        pattern: Option<&specter_core::GlobPattern>,
        diff: &Arc<specter_core::Diff>,
        anchor_path: &Arc<Path>,
        anchor_kind: ResourceKind,
        exclude_strings: &Arc<[CompactString]>,
        out: &mut StepOutput,
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

            let correlation = self.effect_correlations.next();
            // The Sub may have been removed mid-burst; defensive lookup.
            let Some(sub) = self.subs.get(sub_id) else {
                continue;
            };
            let log_output = sub.log_output;
            out.effects.push(Effect {
                key: dk.clone(),
                // PerFile's target is the file resource — same value as
                // `dk.resource` by construction. Carried separately so
                // sort doesn't have to peek inside the variant; the pair
                // `(sub_of_key, target)` is uniform across both arms.
                target: resource,
                forced,
                correlation,
                diff: Some(diff.clone()),
                capture_output: log_output,
                sub_name: sub.name.clone(),
                program: Arc::clone(&sub.program),
                anchor_path: Arc::clone(anchor_path),
                anchor_kind,
                // PerFile: the file segment. The resolver derives
                // `${specter.path}` (and `SPECTER_PATH`) by joining
                // `anchor_path` with this at spawn time — deferring the
                // `PathBuf` allocation past the actuator's Latest-coalesce
                // so dropped Effects don't pay for it.
                target_relative: entry.segment.clone(),
                exclude: Arc::clone(exclude_strings),
            });
            count = count.saturating_add(1);

            if let Some(p) = self.profiles.get_mut(profile_id) {
                p.fired_subs.insert(dk);
            }
        }
        count
    }

    /// Walk `resource` and its strict ancestors looking for Profiles whose
    /// `covers` predicate accepts `resource`. Returns the matching
    /// Profiles in encounter order. P4 single-Profile resolves to 0 or 1.
    ///
    /// **Pending Profiles are filtered at the source.** A Pending
    /// Profile carries no anchor-side `watch_demand` from this Profile
    /// — the descent prefix carries it instead (via
    /// [`specter_core::ContribKey::ProfileDescent`]); the anchor slot
    /// itself only receives the
    /// [`specter_core::ContribKey::ProfileAnchor`] contribution at
    /// descent-completion time. Events at the prefix route via
    /// `classify_event_carriers` / `on_descent_event`; events at the
    /// anchor or its descendants are structurally unreachable in
    /// production (the anchor's `watch_demand` is 0 ⇒ head guard
    /// short-circuits). Filtering here makes the routing contract
    /// explicit: covering-Profile dispatch (Standard burst, anchor
    /// terminal event) only sees Profiles with a materialized anchor.
    fn covering_profiles(&self, resource: ResourceId) -> smallvec::SmallVec<[ProfileId; 2]> {
        let mut out: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut cur = Some(resource);
        while let Some(rid) = cur {
            for pid in self.profiles.at(rid) {
                let Some(p) = self.profiles.get(pid) else {
                    continue;
                };
                if matches!(p.state(), ProfileState::Pending(_)) {
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

    /// Single-pass classification of owners that carry a dispatch
    /// responsibility for an [`crate::Input::FsEvent`] at `resource`.
    /// Sole consumer is [`Engine::on_fs_event`].
    ///
    /// Two carrier classes:
    ///
    /// - **Descent** ([`ProbeOwner`]): owners currently descending whose
    ///   `DescentState.current_prefix() == resource`. Both Profile
    ///   (`ProfileState::Pending(d)`) and Promoter
    ///   (`PromoterState::PrefixPending(d)`) descents qualify; the
    ///   Promoter arm closes the bug where a Promoter waiting on a
    ///   missing literal-prefix segment dropped events at the prefix on
    ///   the floor (no consumer matched, so `EventNoConsumer` fired and
    ///   the Promoter could be permanently stuck without a way to
    ///   re-trigger descent). Each descent owner gets a fresh probe via
    ///   [`Engine::on_descent_event`].
    /// - **Recovery** ([`ProfileId`]): `Idle` Profiles whose
    ///   `watch_root_parent == Some(resource)` and whose anchor is
    ///   currently absent (`current.is_none()`). Profile-only —
    ///   Promoters have no analogous recovery channel.
    ///   [`Engine::start_pending_recovery`] re-enters pending descent.
    ///
    /// O(profiles + promoters). A per-resource index keyed by
    /// `current_prefix` and `watch_root_parent` would convert this to
    /// O(matched); not in scope for v1.
    fn classify_event_carriers(&self, resource: ResourceId) -> EventCarriers {
        let mut out = EventCarriers {
            descents: SmallVec::new(),
            recoveries: SmallVec::new(),
        };
        for (pid, p) in self.profiles.iter() {
            match p.state() {
                ProfileState::Pending(d) if d.current_prefix() == resource => {
                    out.descents.push(ProbeOwner::Profile(pid));
                }
                ProfileState::Idle
                    if p.watch_root_parent == Some(resource) && p.current().is_none() =>
                {
                    out.recoveries.push(pid);
                }
                ProfileState::Pending(_) | ProfileState::Idle | ProfileState::Active(_, _) => {}
            }
        }
        for (qid, q) in self.promoters.iter() {
            if let specter_core::PromoterState::PrefixPending(d) = &q.state
                && d.current_prefix() == resource
            {
                out.descents.push(ProbeOwner::Promoter(qid));
            }
        }
        out
    }
}

/// Per-resource dispatch fan-out collected by
/// [`Engine::classify_event_carriers`]. The two SmallVec inline caps of
/// 2 cover the typical "shared scaffold" case (two Subs anchored at
/// sibling children of one parent, or one Profile sharing a prefix with
/// one Promoter) without a heap allocation.
///
/// `descents` is keyed by [`ProbeOwner`] (Profile or Promoter) — the
/// dispatcher [`Engine::on_descent_event`] is owner-polymorphic.
/// `recoveries` is Profile-only — Promoters have no parent-edge
/// reattach channel.
struct EventCarriers {
    descents: SmallVec<[ProbeOwner; 2]>,
    recoveries: SmallVec<[ProfileId; 2]>,
}

/// Outcome of an [`Engine::emit_effects`] call. `count` is the number of
/// `out.effects.push(...)` invocations that survived dedup-hash
/// suppression and Sub-scope routing — i.e., Effects that the Actuator
/// will actually run.
///
/// `dispatch_*_ok` consumes this to decide whether the Profile should
/// enter the `Awaiting` phase (count > 0, at least one Effect is in
/// flight) or short-circuit to `finish_burst_to_idle` (count == 0:
/// dedup-hash suppressed every emission, no Subs matched, or
/// `reap_pending` was set). The `#[must_use]` attribute prevents a future
/// caller from
/// silently dropping the count and re-introducing the post-emit
/// "Idle-but-Effects-in-flight" leakage.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
#[must_use]
pub(crate) struct EmitOutcome {
    pub count: u32,
}

/// Fire-mode for [`Engine::emit_effects`]. Captures the structural
/// distinction between Standard burst stable-verdict emission and
/// Seed-drift emission, replacing the prior `(forced: bool,
/// drift_filter: Option<&[DedupKey]>)` parameter pair where the
/// interaction between the two flags was load-bearing but unmodelled.
///
/// The two modes differ along three axes that all fall out of the
/// variant — no separate field discipline:
///
/// - **Subtree key gating.** Standard fires every `SubtreeRoot` Sub on
///   the Profile (modulo the suppress check). SeedDrift fires only the
///   Subs in `drifted` (one [`SubId`] per drifted Subtree-keyed Sub).
/// - **Suppress.** Standard honours dedup-hash suppression unless
///   `forced` is set. SeedDrift's `drifted` is built from keys where
///   `last_emitted ≠ current` by construction, so suppression is
///   structurally unreachable on this mode and the `match` returns
///   `false` directly (no analytical claim, no debug-assert, just a
///   variant arm).
/// - **PerStableFile.** Standard emits `PerStableFile` Effects per
///   matching diff entry. SeedDrift skips PerFile entirely — the
///   Seed-time drift signal is Subtree-only (per
///   [`Engine::seed_drift_observed`]'s documented limitation: a
///   post-Seed `current` lacks the per-leaf history needed for a
///   faithful per-file diff).
///
/// **Payload type.** `drifted: &[SubId]` rather than `&[DedupKey]`. By
/// construction the slice carries only `DedupKey::Subtree { sub, profile }`
/// entries whose `profile == profile_id` (the focal Profile); projecting
/// to `SubId` upstream drops the redundant profile field AND removes
/// the variant-ambiguity (a `DedupKey::PerFile` cannot be represented
/// in `&[SubId]`). The SeedDrift Subtree-arm filter becomes
/// `drifted.contains(&sub_id)` — same cost class as `contains(&dk)`,
/// stronger type contract.
///
/// [`Effect::forced`] is derived from the variant via
/// [`Self::effect_forced`]: `true` only on `Standard { forced: true }`.
/// SeedDrift always emits with `forced = false` — the engine reached a
/// stable verdict; drift is the trigger, not a time-pressured
/// force-fire. Conflating the two would silently change the meaning of
/// the user-visible `SPECTER_FORCED` env signal.
#[derive(Copy, Clone)]
enum EmitMode<'a> {
    Standard { forced: bool },
    SeedDrift { drifted: &'a [SubId] },
}

impl EmitMode<'_> {
    /// Value to mirror into [`Effect::forced`] for emissions on this
    /// mode. `true` only on `Standard { forced: true }`.
    const fn effect_forced(self) -> bool {
        matches!(self, Self::Standard { forced: true })
    }
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

/// Per-Promoter dispatch projection used by [`Engine::on_sensor_overflow`].
/// Computed under a short `&self.promoters` borrow, then dispatched
/// under `&mut self` — splitting the borrow lifetimes is the only way
/// to thread the post-state-read calls (`probe_channel.open`,
/// `dispatch_next_enumeration`) through Rust's borrow rules without
/// re-querying the registry per access.
///
/// Variants:
/// - `DescentProbe(prefix)`: `PrefixPending` Promoter with no
///   in-flight descent probe; emit one at `prefix`.
/// - `Enumerate(proxies)`: `Active` Promoter; enqueue every proxy and
///   drain the first into a probe via `dispatch_next_enumeration`.
/// - `Skip`: `PrefixPending` Promoter with an in-flight descent probe;
///   the probe's response will reflect the post-overflow state, so a
///   second probe would be redundant.
enum PromoterReseedAction {
    DescentProbe(ResourceId),
    Enumerate(Vec<ResourceId>),
    Skip,
}

/// Event-class assignment. Maps an [`FsEvent`] + the resource's
/// [`ResourceKind`] to the [`ClassSet`] bit it represents.
///
/// Non-terminal events have a fixed class regardless of kind:
/// - [`FsEvent::Modified`] → [`ClassSet::CONTENT`]
/// - [`FsEvent::MetadataChanged`] → [`ClassSet::METADATA`]
/// - [`FsEvent::StructureChanged`] → [`ClassSet::STRUCTURE`]
///
/// Identity events ([`FsEvent::Removed`] / [`FsEvent::Renamed`] /
/// [`FsEvent::Revoked`]) fold by kind:
/// - `Dir` → [`ClassSet::STRUCTURE`] (the directory's place in its parent
///   changed).
/// - `File` (and `Unknown` via [`ResourceKind::effective`]) →
///   [`ClassSet::CONTENT`] (the file's identity changed — kqexec
///   mapping; the Unknown collapse matches the translator's
///   File-shape default).
///
/// Pure / `const fn`; consulted at the entry filter in [`Engine::on_fs_event`].
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

#[cfg(test)]
#[path = "transitions_tests.rs"]
mod tests;
