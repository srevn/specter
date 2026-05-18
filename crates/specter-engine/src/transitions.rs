//! Per-input dispatch handlers.
//!
//! Each `on_*` method handles one [`Input`] variant for one Profile. They
//! call the burst-lifecycle helpers (`burst.rs`), the refcount edges
//! (`refcounts.rs`), and the reconciliation (`reconcile.rs`). Logic that
//! fits in one row of the transition table stays inline; logic shared across
//! rows (e.g., emit Effects on Standard stable verdict) is factored into
//! private helpers within this module.
//!
//! `on_probe_response` routes every response by *state*: the gated
//! correlation lives on a state-resident [`specter_core::ProbeSlot`],
//! and [`Engine::probe_gate`] reads that correlation *and* the routing
//! class in one resolution (pre-disarm); the slot is then disarmed
//! once on dispatch. This holds uniformly for Profile *and* Promoter
//! owners — Promoter enumeration homes on the `Active` variant's slot
//! like every other carrier.
//! Per-intent fan-out for the Verifying route lives in
//! `dispatch_burst_outcome`.

use crate::Engine;
use crate::engine::is_timer_referenced;
use crate::path::empty_path;
use crate::probe::ProbeRoute;
use crate::reconcile::{ensure_descendant, graft, lookup_descendant};
use crate::refcounts::add_watch;
use compact_str::CompactString;
use smallvec::SmallVec;
use specter_core::{
    ActiveBurst, AnchorClaim, AwaitVerdict, BurstFinish, BurstIntent, ClaimKind, ClassSet,
    ContribKey, DedupKey, DescentRemaining, DescentState, Diagnostic, Effect, EffectCommon,
    EffectOutcome, EffectScope, FsEvent, OverflowScope, PatternComponent, PostFirePhase,
    PreFirePhase, ProbeOutcome, ProbeOwner, ProbeResponse, ProbeSlot, ProfileId, ProfileState,
    PromoterClaimKind, PromoterId, PromoterState, ReapTrigger, Resource, ResourceId, ResourceKind,
    StepOutput, SubId, TimerId, TimerKind, TreeSnapshot, WatchFailure, WatchRegistryDiff,
};
use std::path::Path;
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
            .map(|r| r.proxy_promoters().iter().copied().collect())
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
        let promoter_recovery_count = carriers.promoter_recoveries.len();
        for owner in carriers.descents.iter().copied() {
            self.on_descent_event(owner, now, out);
        }
        for pid in carriers.recoveries.iter().copied() {
            self.start_pending_recovery(pid, resource, out);
        }
        for qid in carriers.promoter_recoveries.iter().copied() {
            self.start_promoter_prefix_recovery(qid, resource, out);
        }

        // Find covering Profiles (anchor or any covering ancestor). For
        // P4 single-Profile this resolves to 0 or 1; P5 multi-Profile
        // dispatches to each in encounter order.
        let covering = self.covering_profiles(resource);
        if covering.is_empty()
            && descent_count == 0
            && recovery_count == 0
            && promoter_recovery_count == 0
            && proxies.is_empty()
        {
            // No consumer: covered by no Profile, no in-flight descent,
            // no Profile/Promoter recovery kicked off, and no proxy
            // back-ref. Emit `EventNoConsumer` (a benign "watched but no
            // listener" signal — typically a `WatchRootParent` /
            // `PromoterPrefixParent` event for something we don't track)
            // and drop. Distinct from `EventOnUnwatchedResource` (the
            // `watch_demand == 0` race earlier) so log levels can
            // diverge. The `promoter_recovery_count == 0` term keeps a
            // parent event that *only* triggered a Promoter recovery
            // from being mis-reported as having no consumer (the
            // recovery loop above already acted on it).
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
            self.on_promoter_proxy_event(promoter_id, resource, out);
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

    /// Re-enter `PrefixPending` descent for an `Active { proxies: ∅ }`
    /// Promoter whose terminus was lost. The Promoter twin of
    /// [`Self::start_pending_recovery`]; triggered by an event at the
    /// Promoter's preserved `prefix_parent` edge.
    ///
    /// **Static recovery segment — not `tree.name(terminus)`.** Unlike
    /// the Profile twin, which reads `tree.name(anchor)` (safe there
    /// because `Profile.resource`'s back-ref pins the anchor slot so
    /// `release_anchor_claim` never try-reaps it), the Promoter terminus
    /// has *no* such pin: `unregister_proxy_subtree` →
    /// `release_promoter_proxy_claim` `try_reap`s the terminus slot, so
    /// it may already be gone. The terminus segment is instead the
    /// *static* `pattern.components()[lpl - 1]` — every component in
    /// `0..lpl` is a `Literal` by the parse invariant, so this is
    /// slot-independent and correct even after the terminus slot reaps.
    /// `lpl >= 1` always (synthetic root), and recovery only classifies
    /// when `prefix_parent` is set, which requires `lpl >= 2`
    /// (`lpl == 1` ⇒ `terminus == "/"` ⇒ no parent ⇒ no
    /// `PromoterPrefixParent` ⇒ this carrier never fires), so
    /// `components[lpl - 1]` is a real literal segment in bounds
    /// (`lpl - 1 < lpl < components.len()`).
    ///
    /// **Recovery overlap (`+2`).** `parent` already holds `+1
    /// STRUCTURE` from the preserved
    /// [`ContribKey::PromoterPrefixParent`] (set at the original
    /// materialisation, never cleared on terminus loss). This helper
    /// bumps another `+1` for the [`ContribKey::PromoterPrefix`] descent
    /// contribution; the refcount sums to `+2`. At re-materialisation
    /// `enter_active`'s plain `sub_watch` drops the descent contribution
    /// while the parent edge persists (`set_promoter_prefix_parent`'s
    /// `already_set` skip) — the exact lifecycle of the Profile
    /// `enter_pending_descent` recovery overlap.
    ///
    /// **Ordering: derive segment → mint → state-flip (construct-armed)
    /// → add_watch → emit** — mirrors [`Self::enter_pending_descent`].
    /// Not delegated to that helper: it is Profile-specific (asserts an
    /// `Idle` Profile, writes `ProfileState`); the Promoter
    /// precondition (`Active { proxies: ∅ }`) and state type differ, so
    /// this is an honest parallel rather than a forced abstraction over
    /// two call sites with no shared body.
    fn start_promoter_prefix_recovery(
        &mut self,
        qid: PromoterId,
        parent: ResourceId,
        out: &mut StepOutput,
    ) {
        // Static terminus segment from the pattern, never the
        // possibly-reaped terminus slot. `None` ⇒ Promoter vanished
        // (benign post-classify race) or the parse invariant was
        // breached (a `Glob` in the literal prefix — caught loudly in
        // dev/CI, degrades to "skip recovery" in release exactly as
        // `render_literal_prefix` handles the same invariant). Either
        // way, returning leaves the Promoter `Active { proxies: ∅ }`
        // (the pre-recovery state), never wedged mid-transition.
        let Some(seg) = self.promoters.get(qid).and_then(|q| {
            let components = q.pattern.components();
            let lpl = q.pattern.literal_prefix_len();
            match &components[lpl - 1] {
                PatternComponent::Literal(s) => Some(s.clone()),
                PatternComponent::Glob(_) => {
                    debug_assert!(
                        false,
                        "start_promoter_prefix_recovery: components[lpl - 1] must be \
                         Literal by the literal-prefix parse invariant \
                         (promoter = {qid:?}, lpl = {lpl})",
                    );
                    None
                }
            }
        }) else {
            return;
        };

        // `vec![seg]` is non-empty by construction, so the `from_vec`
        // discriminant is structurally `Some`. `expect` documents the
        // contract (mirror of `start_pending_recovery`).
        let remaining = DescentRemaining::from_vec(vec![seg])
            .expect("start_promoter_prefix_recovery: single-segment remaining is non-empty");

        // Mint first so the re-entered `PrefixPending` is constructed
        // with its descent slot already armed — no window where the
        // phase exists without a correlation (mirror of
        // `enter_pending_descent` step 1).
        let correlation = self.mint_probe_correlation();

        // Loud arm — `classify_event_carriers` proved this Promoter
        // `Active { proxies: ∅ }` this step and nothing between there
        // and here mutates the registry, so `get_mut` resolving `None`
        // is a state-machine breach, not a benign race. A silent skip
        // would leave the slot un-constructed while the emit below
        // still fires (no probe, no diagnostic — a wedge); mirrors
        // `enter_pending_descent`'s loud arm.
        let Some(q) = self.promoters.get_mut(qid) else {
            unreachable!(
                "start_promoter_prefix_recovery: Promoter {qid:?} vanished between \
                 classify_event_carriers and the construct-armed re-entry"
            );
        };
        q.reenter_prefix_pending(DescentState::new(
            parent,
            remaining,
            ProbeSlot::armed(correlation, ()),
        ));

        // Install the descent contribution on the parent (the `+2`
        // overlap with the preserved `PromoterPrefixParent`).
        add_watch(
            &mut self.tree,
            parent,
            ContribKey::PromoterPrefix(qid),
            ClassSet::STRUCTURE,
            out,
        );

        // The choke reads the correlation back off the re-entered
        // `PrefixPending` descent slot and resolves the parent target
        // off state.
        self.emit_owner_probe(ProbeOwner::Promoter(qid), out);
    }

    /// Dispatch a [`ProbeResponse`] by routing to the per-owner
    /// handler.
    ///
    /// **Gate.** Both handlers resolve the owner once through
    /// [`Engine::probe_gate`], which yields the state-resident slot's
    /// own correlation *and* its routing class together — uniform
    /// across Profile and Promoter owners (descent, verify, rebase,
    /// enumeration). The response is gated by the correlation match;
    /// any mismatch or absent gate covers every stale path (stale id,
    /// response after Cancel, response after a fresh mint, out-of-order
    /// response, no probe in flight), leaves live state intact, and
    /// yields [`Diagnostic::StaleProbeResponse`].
    ///
    /// **Routing.** The route is captured *with* the gate (one
    /// [`Copy`] projection before the slot is disarmed); the
    /// correlation — and the enumeration `target` — lives on the
    /// carrier itself, so a routing-vs-identity divergence is
    /// structurally unrepresentable.
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

    /// Profile-side probe response handler. Every Profile probe —
    /// `Pending` descent, `Active(PreFire(Verifying))`,
    /// `Active(PostFire(Rebasing))` — carries its correlation on a
    /// state-resident [`specter_core::ProbeSlot`]. One uniform
    /// sequence, no per-carrier branch:
    ///
    /// **Gate.** `probe_gate(owner)` yields the gated slot's own
    /// correlation and routing class in one resolution. The response
    /// is gated by `correlation == received`; a mismatch, or an absent
    /// gate (stale `ProfileId`, response after Cancel, response after a
    /// fresh mint, out-of-order response, no probe in flight), leaves
    /// live state intact and yields [`Diagnostic::StaleProbeResponse`].
    ///
    /// **Consume-once.** `take_owner_probe` disarms the slot exactly
    /// once, *after* the gate captured the route and *before* any
    /// dispatch. The received correlation is absent from state before
    /// dispatch, so it cannot route twice — disarm *is* the consume.
    ///
    /// **Routing.** [`Engine::probe_gate`] captures the routing class
    /// *with* the staleness correlation, one resolution
    /// ([`crate::probe::ProbeRoute`] is [`Copy`]; the later disarm
    /// leaves the carrier variant intact). The old `Some(c)`/no-route
    /// regression case folds structurally into an absent gate (⇒
    /// stale). A `ProbeRoute::Enumerating` (a Promoter-only class)
    /// reaching the Profile handler stays a loud regression arm — its
    /// gated correlation lives on exactly one Profile carrier. A
    /// descent receiving `AnchorOk` is a walker-contract violation
    /// handled in-arm; production walkers never emit that shape.
    fn on_profile_probe_response(
        &mut self,
        profile_id: ProfileId,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let owner = response.owner;
        let received = response.correlation;

        // One resolution yields the gated correlation *and* the routing
        // class. The route is captured with the gate — before the
        // disarm — so it stays valid through dispatch (disarm empties
        // the slot but leaves the carrier variant intact). An absent
        // gate or a `received` mismatch is every stale path.
        let Some((_, route)) = self.probe_gate(owner).filter(|&(c, _)| c == received) else {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                owner,
                correlation: received,
            });
            return;
        };
        let consumed = self.take_owner_probe(owner);
        debug_assert_eq!(
            consumed,
            Some(received),
            "consume-once: state-slot disarm must yield the gated correlation \
             (profile = {profile_id:?})",
        );
        #[cfg(debug_assertions)]
        self.dispatch_ledger.record(owner, received);

        match route {
            ProbeRoute::Verifying { intent, forced } => {
                self.dispatch_burst_outcome(profile_id, intent, forced, response.outcome, now, out);
            }

            ProbeRoute::Rebasing => match response.outcome {
                ProbeOutcome::AnchorOk(leaf) => {
                    self.dispatch_rebase_ok(profile_id, TreeSnapshot::File(leaf), now, out);
                }
                ProbeOutcome::SubtreeOk(arc) => {
                    self.dispatch_rebase_ok(profile_id, TreeSnapshot::Dir(arc), now, out);
                }
                ProbeOutcome::Vanished => self.dispatch_rebase_vanished(profile_id, out),
                ProbeOutcome::Failed { errno } => {
                    self.dispatch_rebase_failed(profile_id, errno, out);
                }
            },

            ProbeRoute::Descent => match response.outcome {
                ProbeOutcome::SubtreeOk(arc) => self.dispatch_descent_ok(owner, &arc, now, out),
                ProbeOutcome::Vanished => self.dispatch_descent_vanished(owner, now, out),
                ProbeOutcome::Failed { errno } => self.dispatch_descent_failed(owner, errno, out),
                ProbeOutcome::AnchorOk(_) => {
                    // Walker contract: descent probes a Dir prefix and
                    // returns `SubtreeOk` / `Vanished`. `AnchorOk` is a
                    // walker-side regression.
                    debug_assert!(
                        false,
                        "walker contract violated: Profile descent received AnchorOk \
                         (profile = {profile_id:?})",
                    );
                    out.diagnostics.push(Diagnostic::StaleProbeResponse {
                        owner,
                        correlation: received,
                    });
                }
            },

            ProbeRoute::Enumerating { .. } => {
                // `Enumerating` is a Promoter-only routing class —
                // unconstructable for a Profile owner, whose gated
                // correlation lives on exactly one Profile carrier
                // (descent / verify / rebase). The no-route case
                // folded into the stale gate above; this stays the
                // loud regression arm for the cross-owner class.
                debug_assert!(
                    false,
                    "Profile probe response routed to Enumerating (a Promoter-only \
                     class) — the gated correlation must live on a Profile carrier \
                     (profile = {profile_id:?})",
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
    /// **First-classify is delegated.**
    /// [`specter_core::Profile::install_dir_current`] /
    /// [`specter_core::Profile::install_file_current`] classify the
    /// anchor and graft `current` in one move — the sum's discriminant
    /// *is* the kind, so there are no separate kind/current fields to
    /// write out of step. The per-intent dispatchers call these
    /// (directly or through [`Engine::apply_snapshot`]) at the snapshot
    /// commit point; a kind/current disagreement is unrepresentable, not
    /// merely avoided, so no fallback classify write is needed.
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
                // `current_dir()` borrows the prior Dir snapshot
                // directly — `None` for an unclassified, File-kinded, or
                // not-yet-grafted Profile. The anchor sum makes a
                // kind-mismatched prior unrepresentable, so the old
                // kind-agreement defensive arm is now structural.
                let prior = self
                    .profiles
                    .get(profile_id)
                    .and_then(|p| p.current_dir().cloned());
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
    /// fresh `TimerKind::Settle` at `last_event_time + settle`; the
    /// `PreFireBurst.phase` re-point routes through
    /// [`Engine::reschedule_batching`] (the single-source mutator)
    /// while the quiet-window decision and timer math stay here. The
    /// old (just-expired) id is no longer referenced and lazily drops
    /// on a subsequent `pop_expired`. The phase stays Batching.
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
            self.reschedule_batching(profile_id, new_timer);
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
    /// A Failed arrival clears the Sub's per-Sub fire history
    /// ([`specter_core::SubRegistry::clear_fired`]) — only for a
    /// `Subtree` `key`; `PerFile` carries no fire history. A failed
    /// Effect produced no observable state to deduplicate against, so
    /// the next stable verdict for that Sub must fire fresh even on an
    /// unchanged tree. Phase-independent (Awaiting decrement, late
    /// arrival, or unknown), and a no-op if the Sub already detached
    /// (its flag died with the slotmap entry).
    ///
    /// Two passes for borrow shapes (single-threaded `step` ⇒ no change
    /// between them): pass 1 resolves the route (read borrow), pass 2
    /// applies the completion (`&mut`). The counter owns its decrement
    /// and zero-edge ([`specter_core::Profile::note_effect_completion`]);
    /// this only routes the verdict:
    /// - `LastReached` ⇒ route on [`BurstFinish`]: `ReturnToIdle` →
    ///   `transition_to_rebasing`, `Reap` → `finish_burst_to_idle`.
    /// - `Decremented` ⇒ stay Awaiting.
    /// - else (non-Awaiting, stale, `NotAwaiting`) ⇒ late completion:
    ///   `EffectCompleteForUnknownSub` / `EffectCompleteOutsideAwaiting`.
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

        // Failed clears the Sub's fire history regardless of state, so
        // the next stable verdict for it fires fresh even on an
        // unchanged tree. Match `key` (not the `sub` param) for the
        // scope discriminant: only `Subtree` carries fire history.
        if matches!(result, EffectOutcome::Failed(_)) {
            match key {
                DedupKey::Subtree { sub, .. } => self.subs.clear_fired(*sub),
                // PerFile has no fire history (diff membership is the
                // dedup) — nothing to clear.
                DedupKey::PerFile { .. } => {}
            }
        }

        // Pass 1 (read borrow): route only. Capture the `Copy`
        // `BurstFinish` here — a Sub detaching mid-Awaiting flips it via
        // `mark_active_for_reap`, so the captured value is post-flip;
        // capturing keeps pass 2 a single `&mut` borrow.
        let route = match self
            .profiles
            .get(profile_id)
            .map(specter_core::Profile::state)
        {
            Some(ProfileState::Active(ActiveBurst::PostFire(post), finish)) => match &post.phase {
                PostFirePhase::Awaiting { .. } => CompletionRoute::CountDown(*finish),
                PostFirePhase::Rebasing(_) => CompletionRoute::Diagnose,
            },
            // PreFire phases (Batching / Verifying / Draining), Idle,
            // Pending, stale Profile (None): not waiting for this
            // completion — a late arrival the engine no longer tracks.
            _ => CompletionRoute::Diagnose,
        };

        // Pass 2 (`&mut` borrow): the counter owns the decrement and the
        // zero-edge; this dispatcher only routes the verdict.
        match route {
            CompletionRoute::CountDown(finish) => match self
                .profiles
                .get_mut(profile_id)
                .map(specter_core::Profile::note_effect_completion)
            {
                Some(AwaitVerdict::Decremented) => {}
                Some(AwaitVerdict::LastReached) => match finish {
                    BurstFinish::ReturnToIdle => self.transition_to_rebasing(profile_id, out),
                    // No Subs left to rebase for; finish_burst_to_idle
                    // runs the burst-end Draining-sweep reconfirm then
                    // the deferred reap (a direct reap_profile would
                    // skip the sweep).
                    BurstFinish::Reap => self.finish_burst_to_idle(profile_id, out),
                },
                // Off Awaiting between passes (unreachable under
                // single-threaded `step`) or vanished — late completion.
                Some(AwaitVerdict::NotAwaiting) | None => {
                    self.diagnose_late_completion(sub, profile_id, out);
                }
            },
            CompletionRoute::Diagnose => self.diagnose_late_completion(sub, profile_id, out),
        }
    }

    /// Diagnostic for a completion the engine no longer Awaits. Unknown
    /// Sub (detached + reaped) → Sub-keyed
    /// [`Diagnostic::EffectCompleteForUnknownSub`]; still-registered →
    /// Profile-keyed [`Diagnostic::EffectCompleteOutsideAwaiting`].
    fn diagnose_late_completion(&self, sub: SubId, profile_id: ProfileId, out: &mut StepOutput) {
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

    /// Dispatch a [`Input::ConfigDiff`].
    ///
    /// Atomic, name-keyed apply of *both* halves of the
    /// [`WatchRegistryDiff`] in the canonical order. The diff carries
    /// operator names, never engine ids: name → id resolution is a
    /// registry-owner operation and homes here against the engine's
    /// authoritative `by_name` indices, never bin-side off the
    /// order-unguaranteed diagnostic stream.
    ///
    /// 1. **Sub `removed`** — resolve the name. `Some` ⇒
    ///    `detach_sub_inner` (reap the Profile if its last Sub left,
    ///    defer if active). `None` ⇒ [`Diagnostic::ConfigDiffUnknownSub`]
    ///    (a name whose prior attach failed and never entered the
    ///    registry — nothing to detach).
    /// 2. **Sub `modified`** — resolve the name; detach the old id if
    ///    present, then `attach_sub_inner`. A name the engine never
    ///    attached resolves `None` ⇒ attach-only: the modify becomes a
    ///    retrying fresh attach against the now-maybe-valid path
    ///    (strictly more correct than the prior silent-skip-forever).
    /// 3. **Sub `added`** — `attach_sub_inner` materialises the anchor
    ///    and registers the Sub.
    /// 4. **Promoter `removed`** — resolve the name. `Some` ⇒
    ///    `reap_promoter_inner` (cancel the in-flight probe, detach
    ///    every dynamic Sub, release per-Resource contributions, drop
    ///    the registry entry). `None` ⇒
    ///    [`Diagnostic::ConfigDiffUnknownPromoter`] (closes the old
    ///    asymmetry where this case was a silent no-op).
    /// 5. **Promoter `modified`** — wholesale: resolve, reap the old
    ///    id if present, then `attach_promoter_inner`. The `name`
    ///    survives across the cycle (the diff keys on it); the
    ///    `PromoterId` is freshly minted.
    /// 6. **Promoter `added`** — `attach_promoter_inner` runs descent
    ///    or immediate-Active per the literal-prefix materialisation
    ///    outcome.
    ///
    /// Sub-side runs fully before the Promoter side so a static↔dynamic
    /// migration observes a registry that already reflects the
    /// freshly-applied static Subs. Within each kind, removals run
    /// before additions so a name-recycling rename doesn't transiently
    /// alias against the old entry. The three lists per side are
    /// name-disjoint by diff construction, and each `find_by_name`
    /// reads the live registry *after* prior mutations in the same
    /// step — exactly the id the bin mirror used to supply.
    ///
    /// All resulting ops (across every attach / detach in the diff)
    /// merge into a single sorted `StepOutput`.
    pub(crate) fn on_config_diff(
        &mut self,
        diff: WatchRegistryDiff,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let WatchRegistryDiff { subs, promoters } = diff;

        // ---- Sub side (removed → modified → added) ----
        for name in subs.removed {
            match self.subs.find_by_name(&name) {
                Some(sid) => self.detach_sub_inner(sid, out),
                None => out
                    .diagnostics
                    .push(Diagnostic::ConfigDiffUnknownSub { name }),
            }
        }
        for req in subs.modified {
            if let Some(old) = self.subs.find_by_name(&req.params.name) {
                self.detach_sub_inner(old, out);
            }
            let _ = self.attach_sub_inner(req, now, out);
        }
        for req in subs.added {
            let _ = self.attach_sub_inner(req, now, out);
        }

        // ---- Promoter side (structurally identical) ----
        for name in promoters.removed {
            match self.promoters.find_by_name(&name) {
                Some(pid) => self.reap_promoter_inner(pid, out),
                None => out
                    .diagnostics
                    .push(Diagnostic::ConfigDiffUnknownPromoter { name }),
            }
        }
        for req in promoters.modified {
            if let Some(old) = self.promoters.find_by_name(&req.name) {
                self.reap_promoter_inner(old, out);
            }
            let _ = self.attach_promoter_inner(req, out);
        }
        for req in promoters.added {
            let _ = self.attach_promoter_inner(req, out);
        }
        // The single-StepOutput sort happens at `step`'s caller.
    }

    /// Dispatch a [`Input::WatchOpRejected`].
    ///
    /// The Sensor failed to install a kernel watch (typically `EMFILE` /
    /// `ENFILE` on FD exhaustion). Three things must happen:
    ///
    /// 1. [`specter_core::Tree::vacate`] the rejected slot — clear
    ///    every contribution atomically, so the engine's view of "is
    ///    this slot watched?" matches reality.
    /// 2. Walk every Profile *and* Promoter that holds a claim on
    ///    `resource` (Profile: anchor / watch-root parent / descent
    ///    prefix; Promoter: descent prefix / `Active` proxy /
    ///    prefix-parent recovery edge) and clean up its bookkeeping —
    ///    otherwise the owner flag contradicts the post-vacate counter,
    ///    and any subsequent owner-driven release path would either see
    ///    the wrong union on recompute or silently drift further out of
    ///    sync.
    /// 3. Emit one `ProfileClaimPurged` / `PromoterClaimPurged`
    ///    Diagnostic per affected (owner, claim_kind) pair, plus the
    ///    umbrella `WatchOpRejected` diagnostic.
    ///
    /// A single resource may be claimed by several owners via different
    /// roles — anchor of P, watch-root parent of Q, descent prefix of
    /// R, prefix-parent of Promoter S — so the fan-out walks every
    /// claim slot independently. The Promoter prefix-parent purge is
    /// the structural twin of the Profile watch-root-parent purge:
    /// without it an FD-exhaustion clamp of the parent slot would leave
    /// `Promoter.prefix_parent` caching a now-unwatched id, leaking the
    /// stale recovery edge (the release path keys its `sub_watch`
    /// removal off that cache).
    ///
    /// Stale resources (already Unwatched, queue-race) are a no-op +
    /// `WatchOpRejected` diagnostic; the per-claim walk yields nothing
    /// because owner back-references would have been cleared at reap.
    pub(crate) fn on_watch_op_rejected(
        &mut self,
        resource: ResourceId,
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
            if p.watch_root_parent() == Some(resource) {
                parent_claimers.push(pid);
            }
            if let ProfileState::Pending(d) = p.state()
                && d.current_prefix() == resource
            {
                descent_claimers.push(pid);
            }
        }

        // Promoter-side claimers. The descent (5a) / `Active` proxy
        // (5b) pair is state-disjoint — a Promoter holds one XOR the
        // other for a given `resource` (state is a sum-type). The
        // prefix-parent edge (5c) is *orthogonal*: it lives on
        // `Promoter.prefix_parent`, not on `state`, and coexists with
        // proxies — so it is collected by an independent `if`, exactly
        // as a Profile's `watch_root_parent` is collected independently
        // of its descent/anchor state above. Three SmallVecs keep the
        // per-claim purge loops structurally distinct.
        let mut promoter_descent_claimers: smallvec::SmallVec<[PromoterId; 2]> =
            smallvec::SmallVec::new();
        let mut promoter_proxy_claimers: smallvec::SmallVec<[PromoterId; 2]> =
            smallvec::SmallVec::new();
        let mut promoter_prefix_parent_claimers: smallvec::SmallVec<[PromoterId; 2]> =
            smallvec::SmallVec::new();
        for (qid, q) in self.promoters.iter() {
            match q.state() {
                PromoterState::PrefixPending(d) if d.current_prefix() == resource => {
                    promoter_descent_claimers.push(qid);
                }
                PromoterState::Active { proxies, .. } if proxies.contains_key(&resource) => {
                    promoter_proxy_claimers.push(qid);
                }
                PromoterState::PrefixPending(_) | PromoterState::Active { .. } => {}
            }
            if q.prefix_parent() == Some(resource) {
                promoter_prefix_parent_claimers.push(qid);
            }
        }

        // Atomic terminus for the rejected slot: clear the
        // contributions map, emitting the closing `Unwatch`. The
        // per-claimer loops below run their owner-bookkeeping and call
        // `sub_watch`, which short-circuits on the post-vacate state
        // (absent key). One slot, one terminus.
        self.tree.vacate(resource, out);

        // Anchor claimers: synthesise an anchor-loss. `finalize_anchor_lost`
        // cancels any in-flight Active probe, releases the anchor flag
        // (silent no-op on the post-vacate contributions map), and
        // finishes the burst to Idle. Net Sensor ops match the
        // pre-vacate accounting.
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

        // Descent claimers: `cancel_owner_probe` (disarm + Cancel iff a
        // descent probe was in flight, idempotent), then release
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
        // `Active{empty}`. This is FD-exhaustion of the *descent prefix
        // slot* specifically, distinct from terminus loss: a `rm -rf`
        // of the materialised terminus recovers via the preserved
        // `prefix_parent` edge (the prefix-parent purge below, and
        // `start_promoter_prefix_recovery`). FD-clamping the descent
        // prefix itself has no recovery channel and strands the
        // Promoter — accepted v1 debt, exactly symmetric with the
        // Profile descent purge above (equally stranded). A *recovery*
        // descent FD-clamped above its `prefix_parent` keeps that edge
        // (different `resource`) and can still re-trigger.
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
            let target_matches = self
                .promoters
                .get(qid)
                .and_then(|q| q.state().enumeration_target())
                == Some(resource);
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

        // Promoter prefix-parent purge — the structural twin of the
        // Profile watch-root-parent purge above. Clears the preserved
        // recovery edge so `Promoter.prefix_parent` does not cache a
        // now-unwatched id (the release path keys its `sub_watch`
        // removal off that cache; a stale cache would leak the old
        // parent's `+1` and silently disable terminus recovery while
        // pretending it is live). No cancel-first:
        // `release_promoter_prefix_parent_claim` neither flips state nor
        // drops a `ProbeSlot`, so no probe can be orphaned (exactly as
        // the Profile `parent_claimers` loop carries no
        // `cancel_owner_probe`). The Promoter's proxies stay watched
        // (different `resource`); auto-recovery on terminus recreation
        // is no longer possible — operator restart required.
        for qid in promoter_prefix_parent_claimers {
            self.release_promoter_prefix_parent_claim(qid, out);
            out.diagnostics.push(Diagnostic::PromoterClaimPurged {
                promoter: qid,
                claim: PromoterClaimKind::PrefixParent,
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
    /// SubtreeRoot Sub on the Profile that has fired, then rebases.
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
    ///   probe and runs the Draining-sweep reconfirm cascade), then
    ///   start a fresh seed burst. The Standard burst's accumulated
    ///   `dirty_resources` are
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

        // Exclude the snapshot-time `Draining` Profiles. A `Draining`
        // Profile holds a verified-stable `current` plus a
        // descendant-driven, deadline-bounded reconfirm; a Seed re-walk
        // is no fresher (it mtime-skips against that same `current`) and
        // tearing it down to a Seed discards both the verified snapshot
        // and the "ancestor fires once after the gating descendant
        // settles" relationship. The exclusion has to be at snapshot
        // time, not an iteration-time phase guard on the Active arm: a
        // prior iteration's `finish_burst_to_idle` Draining sweep can
        // flip an in-scope Draining ancestor `Draining → Verifying`
        // before the loop reaches it, so by iteration time it is no
        // longer Draining and the guard would never fire. Removing it
        // from the snapshot also means that, once the sweep has armed
        // the lone reconfirm probe for such an ancestor, the loop never
        // reaches a second same-owner emission for it.
        let profiles_to_reseed: smallvec::SmallVec<[ProfileId; 8]> = profiles_to_reseed
            .into_iter()
            .filter(|&pid| {
                self.profiles
                    .get(pid)
                    .is_some_and(|p| !p.state().is_draining())
            })
            .collect();

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
                ProfileState::Active(_, finish) => {
                    // Overflow on an Active burst is reseed-XOR-reap,
                    // not a pure teardown. The in-flight probe's wire
                    // `Cancel` is a syscall-skip optimization only —
                    // `on_profile_probe_response`'s staleness gate is
                    // the sole correctness authority for a late
                    // response; the `Cancel` merely spares a
                    // not-yet-dequeued worker a wasted recursive walk.
                    // Whether it is needed turns on whether a
                    // superseding `submit` follows in THIS step:
                    //
                    //  reseed (will_reap == false): finish_burst_to_idle
                    //    returns the Profile to Idle, then
                    //    start_seed_burst emits a fresh
                    //    Probe{P,C2}. The sensor's per-owner expectation
                    //    map is a last-writer-wins upsert keyed by
                    //    owner, so submit(P,C2) alone supersedes C1: a
                    //    not-yet-dequeued C1 worker self-skips on
                    //    expected[P] != C1. A wire Cancel{P} here would
                    //    be strictly redundant AND the only same-owner
                    //    Cancel+Probe pair the engine can emit — so
                    //    disarm the engine slot only (take_owner_probe,
                    //    no wire op), exactly as the response path does.
                    //
                    //  reap (will_reap == true): finish_burst_to_idle
                    //    reaps the Profile and start_seed_burst then
                    //    no-ops (require_idle finds it detached). No
                    //    superseding submit follows, so the worker would
                    //    run a full doomed walk — emit the wire Cancel
                    //    via cancel_owner_probe, the same syscall-skip
                    //    the pure-teardown sites rely on.
                    //
                    // The disarm MUST precede finish_burst_to_idle: that
                    // helper swaps the Profile to Idle and destructures
                    // the prior burst, so an armed Verifying/Rebasing
                    // slot would reach drop *there* and trip ProbeSlot's
                    // tripwire — before finish_burst_to_idle's own
                    // deferred reap_profile, whose cancel_owner_probe
                    // would by then see an already-Idle Profile (too
                    // late). This pre-finish disarm is the only consume
                    // that reaches the slot in time; it is not redundant
                    // with reap_profile's own.
                    //
                    // `will_reap` is read off the matched `finish`
                    // (BurstFinish is Copy) before any &mut self call,
                    // so NLL ends the &Profile borrow here — the shape
                    // handle_gate_deadline already compiles.
                    let will_reap = matches!(finish, BurstFinish::Reap);
                    let owner = ProbeOwner::Profile(pid);
                    if will_reap {
                        self.cancel_owner_probe(owner, out);
                    } else {
                        let _ = self.take_owner_probe(owner);
                    }
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
            // `&mut self` calls below (mint, descent_state_mut,
            // dispatch_next_enumeration). Stale id ⇒ skip without
            // emitting the reseed diagnostic — the Promoter is gone.
            let qowner = ProbeOwner::Promoter(qid);
            let probe_in_flight = self.pending_probe_for(qowner).is_some();
            let action = match self.promoters.get(qid) {
                None => continue,
                Some(q) => match q.state() {
                    // Target is no longer carried — `emit_owner_probe`
                    // reads `current_prefix` back off the descent slot.
                    PromoterState::PrefixPending(_) if !probe_in_flight => {
                        PromoterReseedAction::DescentProbe
                    }
                    // PrefixPending with in-flight descent probe: the
                    // probe's response will reflect the post-overflow
                    // state. No double-probe.
                    PromoterState::PrefixPending(_) => PromoterReseedAction::Skip,
                    PromoterState::Active { proxies, .. } => {
                        PromoterReseedAction::Enumerate(proxies.keys().copied().collect())
                    }
                },
            };

            match action {
                PromoterReseedAction::DescentProbe => {
                    let correlation = self.mint_probe_correlation();
                    // Loud arm — the classification just above proved
                    // `PrefixPending` under a `promoters.get(qid)` that
                    // resolved `Some`, and nothing mutated `qid` since,
                    // so `descent_state_mut` is structurally `Some`. The
                    // `!probe_in_flight` guard means the slot is empty,
                    // so `arm_probe`'s empty-slot precondition holds.
                    let Some(d) = self.descent_state_mut(qowner) else {
                        unreachable!(
                            "overflow reseed: Promoter {qid:?} left \
                             PrefixPending between classification and re-arm"
                        );
                    };
                    d.arm_probe(correlation);
                    // The choke reads the correlation and the prefix
                    // target back off the descent slot.
                    self.emit_owner_probe(qowner, out);
                }
                PromoterReseedAction::Enumerate(proxy_keys) => {
                    // Enqueue every proxy. Single-slot drain processes
                    // one at a time via the `dispatch_next` chain on
                    // each response. Empty proxies vec is a no-op.
                    if let Some(qmut) = self.promoters.get_mut(qid) {
                        for r in proxy_keys {
                            qmut.enqueue_enumeration(r);
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
            .filter(|(_, q)| match q.state() {
                PromoterState::PrefixPending(d) => {
                    d.current_prefix() == r
                        || self.tree.ancestors(d.current_prefix()).any(|a| a == r)
                }
                PromoterState::Active { proxies, .. } => proxies
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
                if p.current_is_some() {
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
    /// re-materialisation, the Promoter's enumeration's derived dedup
    /// gate (`promoter_already_promoted`) finds the still-attached
    /// dynamic Sub in `SubRegistry` and returns `true` (no fresh Sub
    /// for an already-known anchor), so no engine work is needed for
    /// correctness — only the static Sub's recovery flow drives the
    /// burst.
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
    /// Promoter (operator narration only — the Promoter holds no
    /// mirror to drop since `dynamic_subs` was deleted), removes every
    /// dynamic Sub from `SubRegistry`, then reaps the Profile
    /// entirely.
    ///
    /// The reap delegates to [`Engine::reap_profile`] /
    /// [`Engine::finish_burst_to_idle`] depending on the Profile's
    /// state — mirrors `detach_sub_inner`'s lifecycle dispatch but
    /// force-runs the deferred-end path synchronously (the anchor is
    /// dead now; we cannot wait for the burst to complete naturally
    /// against a stale anchor).
    ///
    /// Idempotent: a guard at entry returns early when `profile_id`
    /// is no longer in the map (the caller filters empty-Subs, not a
    /// vanished Profile). The Sub-removal loop is also idempotent: a
    /// stale Sub id on the Profile's `by_profile` list is a structural
    /// impossibility (the registry maintains by_profile in lockstep
    /// with subs).
    fn on_anchor_terminal_all_dynamic(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        // The caller filtered the empty-Subs case but not a Profile
        // already gone from the map. A vanished Profile would fall
        // through to `path_of(default-id)` → `None` → a
        // `debug_assert!` whose message claims a live anchor, the
        // opposite of the real state. Return early — nothing to tear
        // down.
        if self.profiles.get(profile_id).is_none() {
            return;
        }

        // 1. Disarm + Cancel iff a probe is in flight — Active+Verifying
        // may have one. Idempotent when the slot is already unarmed.
        self.cancel_owner_probe(ProbeOwner::Profile(profile_id), out);

        // 2. Resolve the anchor resource + path ONCE for the per-Sub
        // loop. Every dynamic Sub on this Profile shares the same
        // anchor by the `(resource, config_hash)` find-or-create dedup
        // in `attach_sub_inner`; the path is the operator-facing
        // diagnostic payload. The Profile is present (guarded at
        // entry) and not yet reaped, so its anchor_claim still holds
        // the slot alive and `path_of` resolves. The fallbacks now
        // guard only a present-Profile / dead-anchor regression —
        // loud in dev, degrade in release.
        let anchor_resource: ResourceId = self
            .profiles
            .get(profile_id)
            .map(|p| p.resource)
            .unwrap_or_default();
        let anchor_path: Arc<Path> = self.tree.path_of(anchor_resource).unwrap_or_else(|| {
            debug_assert!(
                false,
                "on_anchor_terminal_all_dynamic: present Profile's anchor slot must be live \
                 until reap_profile (profile = {profile_id:?}, resource = {anchor_resource:?})",
            );
            empty_path()
        });

        // 3. Notify each source Promoter; remove each dynamic Sub from
        // SubRegistry. SubRegistry's `by_profile` index drops the
        // entry on the last remove, so the post-loop registry has no
        // back-references for this Profile.
        let dynamic_subs: SmallVec<[SubId; 2]> = self.subs.at(profile_id).iter().copied().collect();
        for sid in dynamic_subs.iter().copied() {
            if let Some(pid) = self.subs.get(sid).and_then(|s| s.source_promoter) {
                self.on_dynamic_sub_reaped(pid, sid, &anchor_path, out);
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
    /// Pending Profile holds no anchor (it is still descending toward
    /// one) — anchor-loss finalization does not apply to it, and its
    /// descent-prefix watch rejection is handled separately as a
    /// descent-prefix claim purge.
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

        // Idempotent: emits Cancel iff a probe is in flight
        // (Active+Verifying ⇒ slot armed). For Active+Batching /
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
    /// once over the Profile's fired Subs ([`SubRegistry::fired_in`])
    /// and route through the same fire-tail as a Standard burst
    /// (`emit_effects` count > 0 ⇒ `transition_to_awaiting`; the
    /// eventual rebase probe captures the post-command tree). Otherwise
    /// rebase directly: `baseline := current` and finish.
    ///
    /// Fresh-attach Seed cannot enter the drift branch — no Sub on a
    /// fresh Profile has fired, so `seed_drift_observed` returns false.
    /// The drift branch fires only on recovery / post-Effect rebase
    /// paths where the Profile has already emitted at least one Effect.
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
        // probe-response dispatcher). The fallback to `p.resource` on
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

        // Loss→recovery honesty signal, evaluated *before* the rebase
        // below consumes the survival witness. A Seed-Ok that closes an
        // anchor-loss window rebases `baseline := observed`, absorbing
        // the whole loss-window delta in one move: the Subtree side
        // re-fires its drifted Subs from the witness, but a
        // `PerStableFile` Sub has no per-leaf witness, so its
        // loss-window reactions vanish without a trace. This is
        // standalone, not folded into the drift branch below, on
        // purpose — a PerFile-only Profile never enters that branch
        // (its Subs never record a fire ⇒ `seed_drift_observed` is
        // false) yet is exactly the case to flag, so the condition
        // cannot piggy-back on it. A byte-identical recovery
        // (`current == witness`) dropped nothing and emits nothing.
        if let Some(p) = self.profiles.get(profile_id)
            && let Some(witness) = p.survival_witness()
            && let Some(current) = p.current()
            && current.hash() != witness
            && self.subs.has_per_stable_file_sub(profile_id)
        {
            out.diagnostics
                .push(Diagnostic::PerFileDriftDroppedOnRecovery {
                    profile: profile_id,
                });
        }

        // Fire Effects only for the Profile's fired Subs when drift is
        // observed (post-graft current.hash() differs from
        // `settled_hash()` — the active-mode baseline digest or, across
        // the loss→recovery window, the survival witness). Drift is a
        // per-Profile signal: every Sub that has fired re-fires once,
        // unconditionally. (Post-collapse only SubtreeRoot Subs ever
        // record a fire, so the filter is Subtree-only by
        // construction — no explicit per-key narrowing.)
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
            // The Profile's fired Subs, straight off the registry's
            // per-Sub flags. Membership only — the `emit_effects` loop
            // filters with `drifted.contains`, and the observable
            // Effect order is fixed globally by
            // `StepOutput::sort_for_emission`, so the insertion order
            // `fired_in` yields needs no re-sort.
            let drifted: SmallVec<[SubId; 2]> = self.subs.fired_in(profile_id);
            // `drifted` is empty when no attached Sub has fired: PerFile
            // Subs never record a fire, and a detached Sub's flag died
            // with its slotmap entry. Fall through to the no-drift
            // finish path in that case.
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

    /// Decide whether a Seed-Ok should fire conservative-recovery
    /// Effects: `true` iff the Profile has fired before AND the
    /// post-graft `current` snapshot's anchor-rooted hash differs from
    /// the settled reference.
    ///
    /// [`Profile::settled_hash`] is the single settled-reference oracle:
    /// in active mode it digests the baseline snapshot; across the
    /// loss→recovery window it returns the survival witness the anchor
    /// carried through the loss (covering anchor-loss recovery via
    /// descent → Seed-Ok, and `on_sensor_overflow` reseed); a
    /// not-yet-settled anchor yields `None`. The settled snapshot and
    /// the survival witness are mutually exclusive *in the anchor sum*,
    /// so the survival-mode-authoritative priority is structural — there
    /// is no ordering to maintain here and the witness cannot be
    /// silently lost on recovery. `None` (a fresh, never-fired
    /// Profile) preserves "a fresh Seed never fires an Effect".
    ///
    /// The boolean answer is per-Profile; the caller
    /// ([`Engine::dispatch_seed_ok`]) builds the SeedDrift fire filter
    /// from the Profile's fired Subs ([`SubRegistry::fired_in`]).
    fn seed_drift_observed(&self, profile_id: ProfileId) -> bool {
        // Never fired ⇒ no prior emission to re-fire on recovery. The
        // per-Sub flags live on the registry (disjoint field from
        // `profiles`); `any_fired` short-circuits on the first hit.
        if !self.subs.any_fired(profile_id) {
            return false;
        }
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        let Some(current) = p.current() else {
            return false;
        };
        let curr = current.hash();
        match p.settled_hash() {
            Some(settled) => settled != curr,
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
                // Fire-gate component: no covered strict-descendant
                // Profile is still in an Active Standard burst.
                // Evaluated fresh from the live tree — the derived
                // replacement for the deleted `dirty_descendants`
                // refcount. `profile_id` is the ancestor under test;
                // the strict-subtree walk excludes it as a candidate.
                // Borrow-safe: `p`, `&self.tree`, `&self.profiles` are
                // all shared.
                let dirty_zero = !crate::coverage::has_active_standard_descendant(
                    &self.tree,
                    &self.profiles,
                    profile_id,
                );
                (target, prior_hash, dirty_zero)
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
    /// then either restart from the fire-tail residual or finish to
    /// Idle. The Rebasing probe always targets the anchor (set by
    /// `transition_to_rebasing`); no stability verdict applies (we just
    /// fired, drift is expected).
    ///
    /// **Post-rebase residual.** Events absorbed during `Rebasing` while
    /// the probe was already in flight have, until this point, no
    /// consumer. A [`BurstFinish::ReturnToIdle`] burst with a non-empty
    /// residual restarts a fresh debounced burst over the rebased
    /// baseline (`restart_burst_from_fire_tail_residual`) so the change
    /// is not lost — **regardless of origin**. A Seed-origin burst
    /// (Seed drift → fire → rebase) restarts here too: the reconfirm is
    /// a fresh query, not a per-origin refcount, so there is no balance
    /// to keep and `into_pre_fire_residual` simply rejoins it to the
    /// Standard debounce lifecycle (this closes the former
    /// Seed-residual event-loss). The remaining shapes still finish to
    /// Idle: an empty residual (idempotent command — the hot path) or a
    /// zombie `Reap` burst (no consumer for a rebased baseline).
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
        now: Instant,
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

        // Restart-vs-finish on the post-rebase residual. Resolved under
        // one read borrow (the bool carries no borrow out); pass 2 takes
        // `&mut self`. Restart iff the fire-tail residual is non-empty
        // AND the burst returns to Idle. Origin-agnostic: the reconfirm
        // is a fresh query, not a per-origin refcount, so a Seed origin
        // restarts exactly as a Standard one does (see
        // `PostFireBurst::into_pre_fire_residual`).
        let should_restart = match self
            .profiles
            .get(profile_id)
            .map(specter_core::Profile::state)
        {
            Some(ProfileState::Active(ActiveBurst::PostFire(post), finish)) => {
                !post.force_walk_resources.is_empty() && matches!(finish, BurstFinish::ReturnToIdle)
            }
            _ => false,
        };
        if should_restart {
            self.restart_burst_from_fire_tail_residual(profile_id, now, out);
        } else {
            self.finish_burst_to_idle(profile_id, out);
        }
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
    /// The `forced` write is delegated to [`Engine::force_pending`]
    /// (the single-source `PreFireBurst.forced` mutator); the
    /// phase-classification — whether to drive a verify now — stays
    /// here as a routing query, not a mutation. The caller is reached
    /// only through `is_timer_referenced`, which returns false for
    /// `BurstDeadline` in `Awaiting` / `Rebasing`, so only pre-fire
    /// phases arrive and the structurally-unreachable non-pre-fire
    /// re-read folds to "no verify" — a silent no-op preserving the
    /// prior inline `else { return; }`.
    fn handle_burst_deadline(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        // "burst-deadline elapsed ⇒ forced fire on next emission" is the
        // first action; the phase then decides whether that emission is
        // driven now (Batching/Draining — no probe in flight) or by the
        // in-flight verify's response (Verifying), which dispatches with
        // `forced` observed.
        self.force_pending(profile_id);
        let needs_verify = self
            .profiles
            .get(profile_id)
            .and_then(|p| match p.state() {
                ProfileState::Active(ActiveBurst::PreFire(pre), _) => Some(matches!(
                    &pre.phase,
                    PreFirePhase::Batching { .. } | PreFirePhase::Draining,
                )),
                _ => None,
            })
            .unwrap_or(false);
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
    /// Draining-sweep reconfirm and then dispatches `reap_profile`. The
    /// diagnostic still fires so operators see the
    /// actuator-hang signal; only the wasted rebase round-trip is
    /// elided.
    ///
    /// Defensive: if the phase has already advanced (e.g., a race with
    /// `finalize_anchor_lost`), the helper no-ops. The
    /// `is_timer_referenced` gate already filters most non-Awaiting
    /// fires; this guard handles the residual same-step ordering window.
    ///
    /// The `Awaiting.outstanding` access below is a diagnostic-only
    /// *read*; the field's sole writer is `Profile::note_effect_completion`.
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
    /// appended to `out`. Callers consume this to decide whether
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
        let baseline_snap = p.baseline();
        let current_snap = p.current();
        let pattern = p.config().pattern.clone();
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
        let exclude_strings = Arc::clone(p.exclude_strings());

        let anchor_path: Arc<Path> = self.tree.path_of(resource).unwrap_or_else(empty_path);

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
        // suppress decision combines `nothing_changed` with the per-Sub
        // `Sub.has_fired` flag (read once below, alongside scope /
        // needs_diff / log_output, in the loop's single `subs.get`):
        // a Sub that has never fired suppresses nothing — it is its own
        // "first emission" — even when the tree happens to match.
        let nothing_changed = baseline_snap
            .as_ref()
            .zip(current_snap.as_ref())
            .is_some_and(|(b, c)| b.hash() == c.hash());

        let effect_forced = mode.effect_forced();

        // Snapshot the Sub IDs to avoid holding `&self.subs` across the
        // loop body's `out.push_effect`.
        let sub_ids: Vec<SubId> = self.subs.at(profile_id).to_vec();
        let mut count: u32 = 0;
        for sub_id in sub_ids {
            let (scope, needs_diff, log_output, already_fired) = match self.subs.get(sub_id) {
                Some(s) => (s.scope, s.needs_diff, s.log_output, s.has_fired),
                None => continue,
            };
            match fire_decision(mode, scope, sub_id, already_fired, nothing_changed) {
                FireVerdict::SkipScope | FireVerdict::SuppressDedup => continue,
                FireVerdict::Emit => {}
            }
            match scope {
                EffectScope::SubtreeRoot => {
                    let diff_for_effect = if needs_diff {
                        ensure_diff(&mut diff_arc)
                    } else {
                        None
                    };
                    let correlation = self.effect_correlations.next();
                    let Some(sub) = self.subs.get(sub_id) else {
                        continue;
                    };
                    out.push_effect(Effect::subtree(
                        EffectCommon {
                            sub: sub_id,
                            profile: profile_id,
                            // `resource` was captured at the function
                            // head from `Profile.resource`; frozen at
                            // emit so the sort survives post-emit churn
                            // without a ProfileMap lookup.
                            anchor: resource,
                            correlation,
                            forced: effect_forced,
                            capture_output: log_output,
                            sub_name: sub.name.clone(),
                            program: Arc::clone(&sub.program),
                            anchor_path: Arc::clone(&anchor_path),
                            anchor_kind,
                            exclude: Arc::clone(&exclude_strings),
                        },
                        diff_for_effect,
                    ));
                    count = count.saturating_add(1);

                    // Record the per-Sub fire (the `sub` borrow above
                    // ended with `push_effect`; `&mut self.subs` is free).
                    self.subs.mark_fired(sub_id);
                }
                EffectScope::PerStableFile => {
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
    /// Returns the number of Effects appended to `out`. The caller
    /// (`emit_effects`) sums this into the [`EmitOutcome.count`] it returns.
    #[must_use]
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

            let correlation = self.effect_correlations.next();
            // The Sub may have been removed mid-burst; defensive lookup.
            let Some(sub) = self.subs.get(sub_id) else {
                continue;
            };
            let log_output = sub.log_output;
            out.push_effect(Effect::per_file(
                EffectCommon {
                    sub: sub_id,
                    profile: profile_id,
                    anchor,
                    correlation,
                    forced,
                    capture_output: log_output,
                    sub_name: sub.name.clone(),
                    program: Arc::clone(&sub.program),
                    anchor_path: Arc::clone(anchor_path),
                    anchor_kind,
                    exclude: Arc::clone(exclude_strings),
                },
                resource,
                entry.segment.clone(),
                diff.clone(),
            ));
            count = count.saturating_add(1);
            // PerFile records no fire history — the per-file dedup is
            // diff membership itself, not a recorded key.
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
    /// - **Profile recovery** ([`ProfileId`]): `Idle` Profiles whose
    ///   `watch_root_parent == Some(resource)` and whose anchor is
    ///   currently absent (`current.is_none()`).
    ///   [`Engine::start_pending_recovery`] re-enters pending descent.
    /// - **Promoter recovery** ([`PromoterId`]): `Active` Promoters
    ///   whose terminus is lost (`proxies.is_empty()`, the exact
    ///   "terminus gone" discriminant since the terminus is the unique
    ///   proxy-tree root) and whose `prefix_parent == Some(resource)`
    ///   (the preserved parent edge). The structural twin of Profile
    ///   recovery; [`Engine::start_promoter_prefix_recovery`] re-enters
    ///   `PrefixPending` descent rooted at the parent.
    ///
    /// O(profiles + promoters). A per-resource index keyed by
    /// `current_prefix`, `watch_root_parent`, and `prefix_parent` would
    /// convert this to O(matched); not in scope for v1.
    fn classify_event_carriers(&self, resource: ResourceId) -> EventCarriers {
        let mut out = EventCarriers {
            descents: SmallVec::new(),
            recoveries: SmallVec::new(),
            promoter_recoveries: SmallVec::new(),
        };
        for (pid, p) in self.profiles.iter() {
            match p.state() {
                ProfileState::Pending(d) if d.current_prefix() == resource => {
                    out.descents.push(ProbeOwner::Profile(pid));
                }
                ProfileState::Idle
                    if p.watch_root_parent() == Some(resource) && !p.current_is_some() =>
                {
                    out.recoveries.push(pid);
                }
                ProfileState::Pending(_) | ProfileState::Idle | ProfileState::Active(_, _) => {}
            }
        }
        for (qid, q) in self.promoters.iter() {
            match q.state() {
                PromoterState::PrefixPending(d) if d.current_prefix() == resource => {
                    out.descents.push(ProbeOwner::Promoter(qid));
                }
                // Terminus-loss recovery discriminant: `Active` with an
                // empty proxy set ⟺ the terminus (the unique proxy-tree
                // root) is gone, and the preserved parent edge points
                // here. The structural twin of the `Idle Profile +
                // watch_root_parent` recovery arm above.
                PromoterState::Active { proxies, .. }
                    if proxies.is_empty() && q.prefix_parent() == Some(resource) =>
                {
                    out.promoter_recoveries.push(qid);
                }
                PromoterState::PrefixPending(_) | PromoterState::Active { .. } => {}
            }
        }
        out
    }
}

/// Per-resource dispatch fan-out collected by
/// [`Engine::classify_event_carriers`]. The three SmallVec inline caps
/// of 2 cover the typical "shared scaffold" case (two Subs anchored at
/// sibling children of one parent, or one Profile sharing a prefix with
/// one Promoter) without a heap allocation.
///
/// `descents` is keyed by [`ProbeOwner`] (Profile or Promoter) — the
/// dispatcher [`Engine::on_descent_event`] is owner-polymorphic.
/// `recoveries` (Profile, via `watch_root_parent`) and
/// `promoter_recoveries` (Promoter, via `prefix_parent`) are honest
/// parallel fields, *not* one `ProbeOwner`-keyed list: the entry
/// helpers genuinely differ (`start_pending_recovery` asserts an `Idle`
/// Profile, `start_promoter_prefix_recovery` an `Active { proxies: ∅ }`
/// Promoter, with no shared body), so a unified owner key would only
/// force a match-dispatch back into the two distinct helpers — the same
/// shape as the existing `descents` / `recoveries` split.
struct EventCarriers {
    descents: SmallVec<[ProbeOwner; 2]>,
    recoveries: SmallVec<[ProfileId; 2]>,
    promoter_recoveries: SmallVec<[PromoterId; 2]>,
}

/// Outcome of an [`Engine::emit_effects`] call. `count` is the number of
/// `out.push_effect(...)` invocations that survived dedup-hash
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
///   structurally unreachable on this mode — `fire_decision`'s
///   SeedDrift arm yields `Emit` directly (no analytical claim, just
///   a variant arm).
/// - **PerStableFile.** Standard emits `PerStableFile` Effects per
///   matching diff entry. SeedDrift skips PerFile entirely — the
///   Seed-time drift signal is Subtree-only (per
///   [`Engine::seed_drift_observed`]'s documented limitation: a
///   post-Seed `current` lacks the per-leaf history needed for a
///   faithful per-file diff). On a witness-bearing loss→recovery
///   Seed this skip drops the `PerStableFile` Sub's loss-window
///   reactions; that (witness-gated) drop is surfaced via
///   [`Diagnostic::PerFileDriftDroppedOnRecovery`]. A plain
///   `Input::SensorOverflow` reseed of a `Snapshot`-baseline
///   Profile drops them the same way but carries no witness, so it
///   is a further v1 limitation the diagnostic does not cover.
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

/// One Sub's fire verdict in an [`Engine::emit_effects`] pass — the
/// total fold of the three gates that used to sit inline in the loop.
/// Distinguishing `SuppressDedup` from `SkipScope` keeps the *reason*
/// inspectable (unit table, future per-cause metrics) even though the
/// loop currently treats both as "don't emit".
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum FireVerdict {
    /// Emit for this Sub (Subtree: one Effect; PerStableFile: one per
    /// matching diff entry).
    Emit,
    /// B1 dedup suppression — a `SubtreeRoot` Sub that has fired
    /// before on a tree unchanged since the last rebase, not forced.
    SuppressDedup,
    /// This `(scope, mode)` does not fire: a `SubtreeRoot` Sub outside
    /// SeedDrift's `drifted` set, or any `PerStableFile` Sub under
    /// SeedDrift (Seed-time drift is Subtree-only).
    SkipScope,
}

/// Total, pure fire decision over `(scope, mode)` for one Sub. No
/// engine state, no `Effect` sink — exhaustively unit-testable.
/// Folds the three formerly-scattered `continue` gates:
///
/// - **SeedDrift Subtree narrowing.** A `SubtreeRoot` Sub fires under
///   SeedDrift only if it is in the pre-filtered `drifted` set.
/// - **B1 dedup suppress.** A `SubtreeRoot` Sub under `Standard`
///   suppresses iff it is not force-fired, the tree is unchanged
///   since the last rebase (`nothing_changed`), AND it has fired
///   before (`already_fired`) — a never-fired Sub is its own first
///   emission even on an unchanged tree. SeedDrift's `drifted` holds
///   only drifted Subs, so suppression is structurally unreachable on
///   that mode (its arm yields `Emit`).
/// - **PerStableFile under SeedDrift.** Skipped entirely — Seed-time
///   drift is Subtree-only (PerFile keeps no per-leaf fire history).
fn fire_decision(
    mode: EmitMode<'_>,
    scope: EffectScope,
    sub_id: SubId,
    already_fired: bool,
    nothing_changed: bool,
) -> FireVerdict {
    match (scope, mode) {
        (EffectScope::SubtreeRoot, EmitMode::SeedDrift { drifted }) => {
            if drifted.contains(&sub_id) {
                FireVerdict::Emit
            } else {
                FireVerdict::SkipScope
            }
        }
        (EffectScope::SubtreeRoot, EmitMode::Standard { forced }) => {
            if !forced && nothing_changed && already_fired {
                FireVerdict::SuppressDedup
            } else {
                FireVerdict::Emit
            }
        }
        (EffectScope::PerStableFile, EmitMode::SeedDrift { .. }) => FireVerdict::SkipScope,
        (EffectScope::PerStableFile, EmitMode::Standard { .. }) => FireVerdict::Emit,
    }
}

/// Pass-1 routing class for [`Engine::on_effect_complete`]: which way
/// to route once [`specter_core::Profile::note_effect_completion`]'s
/// verdict is known.
///
/// - `CountDown(finish)`: `Active(PostFire(Awaiting))`. Pass 2 applies
///   the completion; the last one routes by the captured
///   [`BurstFinish`] (`ReturnToIdle` → Rebasing, `Reap` → finish).
/// - `Diagnose`: any non-Awaiting state — a late completion.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum CompletionRoute {
    CountDown(BurstFinish),
    Diagnose,
}

/// Per-Promoter dispatch projection used by [`Engine::on_sensor_overflow`].
/// Computed under a short `&self.promoters` borrow, then dispatched
/// under `&mut self` — splitting the borrow lifetimes is the only way
/// to thread the post-state-read calls (`mint_probe_correlation` then
/// the slot arm, `dispatch_next_enumeration`) through Rust's borrow
/// rules without re-querying the registry per access.
///
/// Variants:
/// - `DescentProbe`: `PrefixPending` Promoter with no in-flight descent
///   probe; re-arm and emit. The prefix target is not carried —
///   `emit_owner_probe` reads `current_prefix` back off the descent
///   slot, so a stale snapshot cannot diverge from state.
/// - `Enumerate(proxies)`: `Active` Promoter; enqueue every proxy and
///   drain the first into a probe via `dispatch_next_enumeration`.
/// - `Skip`: `PrefixPending` Promoter with an in-flight descent probe;
///   the probe's response will reflect the post-overflow state, so a
///   second probe would be redundant.
enum PromoterReseedAction {
    DescentProbe,
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

#[cfg(test)]
#[path = "transitions_tests.rs"]
mod tests;
