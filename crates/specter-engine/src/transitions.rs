//! Per-input dispatch handlers.
//!
//! Each `on_*` method handles one [`specter_core::Input`] variant for one Profile. They call the
//! burst-lifecycle helpers (`burst.rs`), the refcount edges (`refcounts.rs`), and the
//! reconciliation (`reconcile.rs`). Logic that fits in one row of the transition table stays
//! inline; logic shared across rows (e.g., emit Effects on Standard stable verdict) is factored
//! into private helpers within this module.
//!
//! `on_probe_response` routes every response by *state*: the gated correlation lives on a
//! state-resident [`specter_core::ProbeSlot`], and the gate ([`Engine::profile_probe_gate`]) reads
//! that correlation *and* the routing class in one resolution (pre-disarm); the slot is then
//! disarmed once on dispatch. The Verifying choke and the post-fire Rebase arm share one certifier,
//! `certify_probe_response`: lower the outcome, verify kind agreement, and fold the single
//! quiescence verdict via [`specter_core::quiescence_verdict`]. `dispatch_burst_outcome` then fans
//! the certified result out per [`specter_core::BurstIntent`]; the Rebase arm maps it to the
//! rebase-loop consequence.

use crate::Engine;
use crate::engine::is_timer_referenced;
use crate::path::empty_path;
use crate::probe::{DescentOutcome, ProfileProbeRoute, ProofOutcome, WalkerContractViolation};
use crate::reconcile::{ensure_descendant, graft, lookup_descendant};
use compact_str::CompactString;
use smallvec::SmallVec;
use specter_core::{
    AbsorbMode, ActiveBurst, AnchorClaim, AwaitVerdict, BurstFinish, BurstIntent, ClaimKind,
    ClassSet, DedupKey, DescentRemaining, DetachReason, Diagnostic, Effect, EffectCommon,
    EffectOutcome, EffectScope, FsEvent, OverflowScope, PostFirePhase, PreFirePhase, ProbeFailure,
    ProbeResponse, Profile, ProfileId, ProfileState, ProofAuthority, QuiescenceVerdict,
    QuiescenceWitness, Reaction, ResourceId, ResourceKind, StableReason, StepOutput,
    SubAttachRequest, SubId, SubRegistryDiff, TimerId, TimerKind, TreeSnapshot, WatchFailure,
    quiescence_verdict,
};
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

impl Engine {
    /// Dispatch a normalized [`FsEvent`] for `resource`.
    ///
    /// Routing:
    /// 1. Idempotence guard — `watch_demand == 0` ⇒ `EventOnUnwatchedResource` + drop (race between
    ///    `Unwatch` and the Sensor's drain).
    /// 2. Pending descents whose `current_prefix == resource` get a fresh descent probe
    ///    (`on_descent_event`). Descent prefix watches register STRUCTURE-only, so any event
    ///    reaching here is structurally relevant — descent dispatch is unfiltered.
    /// 3. Idle Profiles whose `watch_root_parent == resource` and whose anchor is currently absent
    ///    (`current.is_none()`) re-enter pending descent — the probe-`Failed` / watch-rejection
    ///    fallback (observed losses are already `Pending` via their own loss step). Same STRUCTURE
    ///    floor applies.
    /// 4. Per-covering-Profile dispatch with class-aware filter:
    ///    - Anchor events bypass the filter unconditionally — lifecycle signal continuity trumps
    ///      user opt-out.
    ///    - Descendant events whose class (per [`fs_event_to_class`]) is not in the Profile's
    ///      `events` drop with `EventClassDropped` BEFORE driving the burst — the class filter sits
    ///      before dirty-set bumps.
    ///    - Mask-admitted descendant events that cannot move the Profile's proof object — boundary
    ///      Dirs and Dir-METADATA — drop with `EventOutsideProofObject`, identity events exempted
    ///      (the proof-relevance guard below).
    ///    - Terminal-on-anchor → [`Self::finalize_anchor_lost_and_descend`] — anchor loss is
    ///      uniform across static, mixed, and minted Profiles. Anything else that passes the filter
    ///      → `drive_burst`.
    pub(crate) fn on_fs_event(
        &mut self,
        resource: ResourceId,
        event: FsEvent,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Idempotence + the single pre-dispatch resource resolution. One `tree.get`: a stale id or
        // `watch_demand == 0` is a race between Unwatch and the Sensor's drain (drop with
        // `EventOnUnwatchedResource`). A live, watched slot yields the event path, captured here —
        // pre-dispatch, off the proven-live `&Resource`. The path is the staleness-immune
        // historical fact the burst accumulators need: a later covering dispatch can reap the slot,
        // so a post-dispatch `path_of` would be fallible exactly where the obligation must not lose
        // an entry.
        let Some(r) = self.tree.get(resource) else {
            out.diagnostics
                .push(Diagnostic::EventOnUnwatchedResource { resource });
            return;
        };
        if r.watch_demand() == 0 {
            out.diagnostics
                .push(Diagnostic::EventOnUnwatchedResource { resource });
            return;
        }
        // `Arc::clone` of the slot's materialised path — an O(1) refcount bump, total by
        // construction (the slot is live). The kind is captured in the same breath and for the same
        // staleness reason: read off the proven-live `&Resource` now, so a later covering dispatch
        // that reaps the slot can't turn the routing-time classification into a `None`-defaulted
        // guess. Unprobed slots collapse to File-shape (`kind_or_file`) per the backend-mask
        // convention — `fs_event_to_class` and the kqueue / inotify translators agree on this
        // default.
        let event_path = Arc::clone(r.path());
        let resource_kind = r.kind_or_file();

        // Single-pass classification of the event's carriers: Profiles that "carry" a dispatch
        // responsibility for this resource. Descent prefix and watch-root-parent watches both
        // register STRUCTURE-only, so any event reaching here is structurally relevant for both
        // arms — no class filter applies before dispatch. Mutual exclusion is structural (`Pending`
        // excludes `Idle` at the `ProfileState` sum-type level).
        let carriers = self.classify_event_carriers(resource);
        let descent_count = carriers.descents.len();
        let recovery_count = carriers.recoveries.len();
        for owner in carriers.descents.iter().copied() {
            self.on_descent_event(owner, now, out);
        }
        for pid in carriers.recoveries.iter().copied() {
            self.start_pending_recovery(pid, resource, out);
        }

        // Find covering Profiles (anchor or any covering ancestor). For P4 single-Profile this
        // resolves to 0 or 1; P5 multi-Profile dispatches to each in encounter order.
        let covering = crate::coverage::covering_profiles(
            &self.tree,
            &self.profiles,
            resource,
            &mut self.coverage_scratch,
        );
        if covering.is_empty() && descent_count == 0 && recovery_count == 0 {
            // No consumer: covered by no Profile, no in-flight descent, no recovery kicked off.
            // Emit `EventNoConsumer` (a benign "watched but no listener" signal — typically a
            // `WatchRootParent` event for an entry we don't track) and drop. Distinct from
            // `EventOnUnwatchedResource` (the `watch_demand == 0` race earlier) so log levels can
            // diverge.
            out.diagnostics
                .push(Diagnostic::EventNoConsumer { resource });
            return;
        }

        // Class-aware routing. The event's class folds once from the head-captured `resource_kind`
        // (read off the proven-live slot above) and the event; per-Profile dispatch then consults
        // the Profile's `events` (every Sub on a Profile shares the same mask, so the union is each
        // Sub's mask).
        let event_class = fs_event_to_class(event, resource_kind);
        let is_identity = event.is_identity();

        for (profile_id, class) in covering {
            let Some((is_anchor, profile_events)) = self
                .profiles
                .get(profile_id)
                .map(|p| (p.resource() == resource, p.events()))
            else {
                continue;
            };

            // Anchor events bypass the class filter unconditionally (lifecycle: anchor
            // disappearance recovery, anchor reappearance detection, etc.). Descendant events whose
            // class is not in the Profile's `events` drop here, before `drive_burst` notes into the
            // pre-fire or post-fire burst's `dirty`.
            if !is_anchor && !profile_events.intersects(event_class) {
                out.diagnostics.push(Diagnostic::EventClassDropped {
                    resource,
                    event,
                    profile: profile_id,
                });
                continue;
            }

            // Proof-relevance guard: a mask-admitted descendant event still drops when nothing it
            // signals can move this Profile's proof object — one fact on two axes:
            //
            // - **Boundary Dir** (path axis): the shape doesn't descend into the slot, so only its
            //   identity in the parent's enumeration folds; member churn inside it is invisible to
            //   every verdict. The Profile's own FDs no longer exist at boundary Dirs (watch
            //   installation is Interior-gated), so this arm is reached via co-located demand — a
            //   minted Profile's anchor FD sits exactly on the discovery terminus slot, and the
            //   event must drop *for the discovery Profile* while the minted Profile sees it as an
            //   anchor event and correctly drives its own burst.
            // - **Dir-METADATA** (attribute axis): chmod / chown / touch on a directory folds into
            //   no proof object (`dir_hash` deliberately excludes `root_meta.mtime`). kqueue
            //   classes mkdir/rmdir backrefs (`NOTE_LINK`) as STRUCTURE, so dropping Dir-METADATA
            //   loses no structural signal.
            //
            // Identity events are exempt: a deleted terminus folds to STRUCTURE at the boundary
            // slot and must drive the discovery reconcile's reap. Driving a burst on a
            // proof-irrelevant event could only end in a `nothing_changed` verdict — which a
            // never-fired Sub converts into a spurious first fire — and the guard sits before
            // `drive_burst`, so post-fire absorbs are equally stopped (a proof-irrelevant event
            // must not extend a settle loop).
            if !is_anchor && !is_identity {
                let dir_kinded = matches!(resource_kind, ResourceKind::Dir);
                if matches!(class, crate::coverage::CoverageClass::Boundary)
                    || (dir_kinded && event_class == ClassSet::METADATA)
                {
                    out.diagnostics.push(Diagnostic::EventOutsideProofObject {
                        resource,
                        event,
                        profile: profile_id,
                    });
                    continue;
                }
            }

            if is_identity && is_anchor {
                self.finalize_anchor_lost_and_descend(profile_id, out);
            } else {
                // ContentChanged/StructureChanged/MetadataChanged anywhere that passes the filter,
                // or terminal at a covered descendant whose class matches: drive the burst forward.
                // Descendant terminal events drive the burst; the next probe response reconciles
                // the slot via the diff-against-prior pass.
                self.drive_burst(profile_id, resource, &event_path, event, now, out);
            }
        }
    }

    /// Re-enter pending descent for an Idle Profile whose anchor is currently absent — the
    /// event-scan arm of [`Self::classify_event_carriers`], triggered by a `StructureChanged` at
    /// the Profile's `watch_root_parent`.
    ///
    /// **Fallback, not the primary recovery route.** Every *observed* loss (anchor-terminal event,
    /// probe `Vanished`, kind mismatch) re-enters descent inside the loss step itself via
    /// [`Self::finalize_anchor_lost_and_descend`], so this arm never sees those Profiles — they are
    /// `Pending` before any later parent event can arrive. What still parks Idle-anchorless and
    /// needs this arm's event luck: the probe-`Failed` discards (transient `EACCES` — the anchor
    /// typically never left disk; an immediate descent would loop probe → fail tightly) and the
    /// watch-rejection anchor purge (the kernel refused the watch; descending would loop
    /// materialize → watch → reject).
    ///
    /// **The entry event does not latch the descent's activity witness.** The event that selected
    /// this Profile into the recoveries arm is a `StructureChanged` at the *parent* — it can be
    /// sibling churn entirely out of the Sub's scope, and latching it would false-first-fire a
    /// never-fired Sub whose anchor the entry probe then finds unchanged on disk (the transient
    /// probe-`Failed` discard is the canonical shape: the anchor typically never left). Fired Subs
    /// are protected by witness-drift either way. Accepted narrow miss: when the entry event was
    /// itself the replacement *and* the entry probe finds it, a never-fired Sub misses that one
    /// fire. Later events at the prefix latch via `on_descent_event`.
    ///
    /// **Recovery overlap.** The parent already holds `+1 STRUCTURE` from
    /// `Profile.watch_root_parent` (set at the original anchor materialization, never cleared on
    /// anchor loss). The helper bumps another `+1` for the descent contribution; the refcount sums
    /// to `+2`. The descent contribution drops at re-materialization while the `watch_root_parent`
    /// contribution persists — see the rustdoc on `enter_pending_descent` for the full lifecycle.
    fn start_pending_recovery(
        &mut self,
        profile_id: ProfileId,
        parent: ResourceId,
        out: &mut StepOutput,
    ) {
        self.enter_anchor_recovery_descent(profile_id, parent, /* witnessed: */ false, out);
    }

    /// Shared single-segment descent entry for anchor recovery: the anchor's segment becomes the
    /// sole remaining component and `parent` the descent prefix. Both recovery producers funnel
    /// here — the event-scan arm ([`Self::start_pending_recovery`], unwitnessed) and the
    /// observed-loss wrapper ([`Self::finalize_anchor_lost_and_descend`], witnessed) — differing
    /// only in the latch's birth value.
    ///
    /// The early returns are structural no-recovery cases, not races: a reaped Profile has nothing
    /// to recover, and an anchor with no segment name (the FS root) has no parent to descend from.
    fn enter_anchor_recovery_descent(
        &mut self,
        profile_id: ProfileId,
        parent: ResourceId,
        witnessed: bool,
        out: &mut StepOutput,
    ) {
        let Some(anchor) = self.profiles.get(profile_id).map(Profile::resource) else {
            return;
        };
        let Some(anchor_name) = self.tree.name(anchor).map(CompactString::from) else {
            return;
        };
        // `vec![anchor_name]` is non-empty by construction, so the `from_vec` discriminant is
        // structurally `Some`. `expect` documents the contract.
        let remaining = DescentRemaining::from_vec(vec![anchor_name])
            .expect("anchor recovery descent: single-segment remaining is non-empty");
        self.enter_pending_descent(profile_id, parent, remaining, witnessed, out);
    }

    /// Dispatch a [`ProbeResponse`]. Every probe — `Pending` descent, `Active(PreFire(Verifying))`,
    /// `Active(PostFire(Rebasing))` — carries its correlation on a state-resident
    /// [`specter_core::ProbeSlot`]. One uniform sequence, no per-carrier branch:
    ///
    /// **Gate.** `profile_probe_gate(response.owner)` yields the gated slot's own correlation and
    /// routing class in one resolution. The response is gated by `correlation == received`; a
    /// mismatch, or an absent gate (stale `ProfileId`, response after Cancel, response after a
    /// fresh mint, out-of-order response, no probe in flight), leaves live state intact and yields
    /// [`Diagnostic::StaleProbeResponse`].
    ///
    /// **Consume-once.** `take_owner_probe` disarms the slot exactly once, *after* the gate
    /// captured the route and *before* any dispatch. The received correlation is absent from state
    /// before dispatch, so it cannot route twice — disarm *is* the consume.
    ///
    /// **Routing.** [`Engine::profile_probe_gate`] captures the routing class *with* the staleness
    /// correlation, one resolution ([`crate::probe::ProfileProbeRoute`] is [`Copy`]; the later
    /// disarm leaves the carrier variant intact). The old `Some(c)`/no-route regression case folds
    /// structurally into an absent gate (⇒ stale). Each route then *parses* the wire
    /// [`ProbeOutcome`](specter_core::ProbeOutcome) into the typed engine-side outcome that route's
    /// consumers accept — `ProofOutcome` for `Verifying` / `Rebasing`, `DescentOutcome` for
    /// `Descent` — so an illegal `(route, outcome)` pairing is unrepresentable past the parse and
    /// never reaches the certifier or the descent dispatcher. A payload shape the route cannot
    /// accept (a proof route receiving the structural `DirEnumerated`, or a descent receiving an
    /// `AnchorOk` / `SubtreeProven` proof) is a walker-contract violation: the parse fails and the
    /// arm routes to the route-appropriate recovery — the burst recovery finishes the burst to Idle
    /// (anchor/baseline survive; a walker defect is not an anchor-identity change), the descent
    /// recovery abandons the descent prefix — each emitting one honest `WalkerContractViolated` and
    /// self-healing on the next `FsEvent`. The match is total over `ProfileProbeRoute`'s three
    /// variants.
    pub(crate) fn on_probe_response(
        &mut self,
        response: ProbeResponse,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let profile_id = response.owner;
        let received = response.correlation;

        // One resolution yields the gated correlation *and* the routing class. The route is
        // captured with the gate — before the disarm — so it stays valid through dispatch (disarm
        // empties the slot but leaves the carrier variant intact). An absent gate or a `received`
        // mismatch is every stale path.
        let Some((_, route)) = self
            .profile_probe_gate(profile_id)
            .filter(|&(c, _)| c == received)
        else {
            out.diagnostics.push(Diagnostic::StaleProbeResponse {
                owner: profile_id,
                correlation: received,
            });
            return;
        };
        let consumed = self.take_owner_probe(profile_id);
        debug_assert_eq!(
            consumed,
            Some(received),
            "consume-once: state-slot disarm must yield the gated correlation \
             (profile = {profile_id:?})",
        );
        #[cfg(debug_assertions)]
        self.dispatch_ledger.record(profile_id, received);

        // Parse the wire `ProbeOutcome` into the typed engine-side outcome the route's consumers
        // accept; an illegal pairing (the `Err` arm) is a walker-contract violation routed to the
        // route-appropriate recovery, unrepresentable past the parse.
        match route {
            ProfileProbeRoute::Verifying { intent, forced } => {
                match ProofOutcome::try_from(response.outcome) {
                    Ok(proof) => {
                        self.dispatch_burst_outcome(profile_id, intent, forced, proof, now, out);
                    }
                    Err(WalkerContractViolation) => {
                        self.walker_contract_violated_burst(profile_id, out);
                    }
                }
            }

            ProfileProbeRoute::Rebasing { forced } => {
                match ProofOutcome::try_from(response.outcome) {
                    // Same certifier as the Verifying choke — the post-fire rebase response folds
                    // through `quiescence_verdict` over the post-command tree. The verdict drives
                    // the rebase-loop consequence table; `Vanished` / `Failed` route to the
                    // rebase-specific cleanup; `Regressed` (kind mismatch) was already handled
                    // inside the certifier.
                    Ok(proof) => {
                        match self.certify_probe_response(profile_id, proof, forced, out) {
                            CertifiedResponse::Proceed { snapshot, verdict } => {
                                self.dispatch_rebase_ok(profile_id, snapshot, verdict, now, out);
                            }
                            CertifiedResponse::Vanished => {
                                self.dispatch_rebase_vanished(profile_id, out);
                            }
                            CertifiedResponse::Failed(failure) => {
                                self.dispatch_rebase_failed(profile_id, failure, out);
                            }
                            CertifiedResponse::Regressed => {}
                        }
                    }
                    Err(WalkerContractViolation) => {
                        self.walker_contract_violated_burst(profile_id, out);
                    }
                }
            }

            ProfileProbeRoute::Descent => match DescentOutcome::try_from(response.outcome) {
                Ok(descent) => self.dispatch_descent(profile_id, descent, now, out),
                Err(WalkerContractViolation) => {
                    self.walker_contract_violated_descent(profile_id, out);
                }
            },
        }
    }

    /// Recover a pre-fire (`Verifying`) or post-fire (`Rebasing`) burst from a walker-contract
    /// violation — a proof-route probe whose payload resolved to the structural `DirEnumerated`
    /// shape the route cannot accept. The typed [`ProofOutcome`] parse rejected the payload at the
    /// demux seam; this recovers the burst.
    ///
    /// `debug_assert!` in dev/CI (a production walker never emits this shape), then in release emits
    /// [`Diagnostic::WalkerContractViolated`] and routes through [`Self::finish_burst_to_idle`] —
    /// **not** [`Self::finalize_anchor_lost`]: a walker defect is not an anchor-identity change, so
    /// the anchor watch and the prior baseline / current are preserved. The probe slot was disarmed
    /// by `take_owner_probe` before dispatch, so `finish_burst_to_idle`'s cancel-first precondition
    /// holds (this is the tested post-disarm path), and the helper accepts both `PreFire(Verifying)`
    /// and `PostFire(Rebasing)` carriers. Self-healing: the next `FsEvent` opens a fresh burst.
    fn walker_contract_violated_burst(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        debug_assert!(
            false,
            "walker contract violated: a Verifying/Rebasing (proof) probe received \
             a non-proof outcome (DirEnumerated) — a structural enumeration is not a \
             quiescence observation (owner = {profile_id:?})",
        );
        out.diagnostics
            .push(Diagnostic::WalkerContractViolated { owner: profile_id });
        self.finish_burst_to_idle(profile_id, out);
    }

    /// Certify a Verifying / Rebase probe response: lower the typed `ProofOutcome`, guard kind
    /// agreement, and fold the carrier's quiescence verdict — the single verdict-construction site
    /// shared by the Verifying choke ([`Self::dispatch_burst_outcome`]) and the post-fire Rebase arm.
    /// The two routes the engine deliberately keeps separate stay separate: each owns its success
    /// consequence (per-[`BurstIntent`] fire/pin vs. the rebase-loop table) and its own `Vanished` /
    /// `Failed` cleanup; only the lower→kind-check→fold spine is unified here, at the floor.
    ///
    /// **Typed input.** The caller passes a `ProofOutcome`, not the wide wire
    /// [`ProbeOutcome`](specter_core::ProbeOutcome): the proof/descent split is parsed once at the
    /// demux seam, so the structural `DirEnumerated` shape — a walker-contract violation on a
    /// quiescence/rebase probe — is unrepresentable here, with no defensive arm. The certifier sees
    /// only the four shapes a proof probe can legally resolve to.
    ///
    /// **Lowering.** `AnchorOk` → `(File, Authoritative)` — a single `lstat` has no mtime-skip
    /// concept, so an anchor read is definitionally authoritative and the engine injects the
    /// certificate the wire omits. `SubtreeProven { snapshot, authority }` → `(Dir, authority)`.
    /// `Vanished` / `Failed` are returned as-is for the caller's own per-route cleanup (the
    /// certifier is route-agnostic — folding a non-snapshot is meaningless).
    ///
    /// **Fold context.** One immutable Profile resolution (`fold_context`) captures every bit the
    /// fold consumes — `events_witness_quiescence` (invariant across the burst, folds into
    /// `config_hash`), the prior [`specter_core::Profile::kind`], and whether the burst owes a
    /// quiescence proof (`owes_proof_from`, a predicate spanning `profiles + subs`) — before any
    /// `&mut` re-fetch. An absent Profile is a gate breach (the floor is reached only on
    /// `Active(Verifying | Rebasing)` through the `profile_probe_gate` ⇒ `take_owner_probe`
    /// dispatch): `debug_assert!` in dev/CI, `Regressed` in release.
    ///
    /// **Kind agreement, before the fold.** The captured prior kind is compared against the lowered
    /// snapshot's variant, *after* the lowering and *before* the verdict fold. A kind-mismatched
    /// response is not a valid observation of the anchor: folding a verdict over it and advancing the
    /// certified-sample sequence with its hash would be meaningless, so the burst is torn down
    /// through [`Engine::finalize_anchor_lost`] (reusing the tested `dispatch_*_vanished` cleanup
    /// chain rather than a fresh "discard then graft" that leaks watch contributions and breaks the
    /// cross-field invariant), after emitting [`Diagnostic::AnchorKindMismatch`] — so the result is
    /// `Regressed` (already finalized). First-classify (`kind == None`, fresh Seed) passes; the
    /// snapshot's variant *is* the kind at the [`specter_core::Profile::install_dir_current`] /
    /// [`specter_core::Profile::install_file_current`] commit. The guard is unreachable in v1 — the
    /// walker collapses every Dir↔File swap to `Vanished` — but operates on a *successful* lowering,
    /// so it stays a semantic floor distinct from the payload-shape parse at the demux seam.
    ///
    /// **Verdict fold.** [`specter_core::quiescence_verdict`] is the floor — a pure, total
    /// `(ProofAuthority, forced, QuiescenceWitness) → QuiescenceVerdict` projection over three axes:
    ///
    /// - **Authority (C1).** `Authoritative` ⇒ walker certified every obligation chain;
    ///   `Undischarged` ⇒ refused. Set by the walker, threaded as-is.
    /// - **Forced.** Set by the caller (read off the burst by [`Engine::profile_probe_gate`],
    ///   packed onto [`crate::probe::ProfileProbeRoute`]'s `Verifying { forced }` / `Rebasing {
    ///   forced }` payload, threaded here); both carriers — pre-fire `PreFireBurst.forced` (a
    ///   single bit) and post-fire [`specter_core::CeilingState::Reached`] (projected to a bool at
    ///   the gate read) — pass through this one site symmetrically. `forced` distinguishes natural
    ///   fire from the bounded `BurstDeadline` / `RebaseCeiling` fallback.
    /// - **Witness (C2 vs. C3).** [`QuiescenceWitness::EventsReliable`] when settle-window silence
    ///   proves quiescence — the Profile's `events_union` covers in-place writes
    ///   ([`specter_core::Profile::events_witness_quiescence`]) OR the burst's consequence does not
    ///   require proof (cold-Seed `SilentPin`, see `owes_proof_from`).
    ///   [`QuiescenceWitness::HashChannel`] otherwise: this site advances the per-burst
    ///   `last_certified_hash` carrier through the cat-(b) cascade
    ///   ([`specter_core::Profile::advance_certified_sample`]) and reads its prior as the channel's
    ///   input. The advance is gated on `Authoritative ∧ needs_hash_channel` — an `Undischarged`
    ///   observation must not advance (the prior would then reflect an unread region), and the
    ///   `EventsReliable` path skips the carrier entirely (dead write avoided on the cold-attach
    ///   win).
    ///
    /// The callers diverge only on the *consequence* (per-intent fire/pin vs. the rebase-loop table).
    #[must_use]
    fn certify_probe_response(
        &mut self,
        profile_id: ProfileId,
        proof: ProofOutcome,
        forced: bool,
        out: &mut StepOutput,
    ) -> CertifiedResponse {
        // Lower the typed proof outcome to (snapshot, authority). `Vanished` / `Failed` return
        // as-is for the caller's per-route cleanup; `DirEnumerated` is unrepresentable here —
        // parsed out at the demux seam, so no defensive arm is needed.
        let (snap, authority) = match proof {
            ProofOutcome::AnchorOk(leaf) => {
                (TreeSnapshot::File(leaf), ProofAuthority::Authoritative)
            }
            ProofOutcome::SubtreeProven {
                snapshot,
                authority,
            } => (TreeSnapshot::Dir(snapshot), authority),
            ProofOutcome::Vanished => return CertifiedResponse::Vanished,
            ProofOutcome::Failed(failure) => return CertifiedResponse::Failed(failure),
        };

        // One immutable resolution of every Profile bit the fold consumes (events witness, prior
        // kind, proof obligation), held by value so the later `&mut self` re-fetch is borrow-clean.
        // An absent Profile is a gate breach — `profile_probe_gate` ⇒ `take_owner_probe` reaches
        // this floor only on Active(Verifying | Rebasing); degrade to `Regressed`.
        let Some(ctx) = self.fold_context(profile_id) else {
            debug_assert!(
                false,
                "certify_probe_response: absent Profile {profile_id:?} — \
                 profile_probe_gate dispatches only on Active(Verifying | Rebasing)",
            );
            return CertifiedResponse::Regressed;
        };

        // Kind guard before the fold. Unreachable in v1 (the walker collapses Dir↔File swaps to
        // `Vanished`), but a kind-mismatched response is not a valid observation of the anchor:
        // tear the burst down and re-enter descent through `finalize_anchor_lost_and_descend` — a
        // kind swap is itself witnessed evidence the anchor was replaced — rather than fold a
        // verdict over a soon-discarded snapshot. First-classify (`prior == None`, fresh Seed)
        // passes; the snapshot's variant *is* the kind at the `install_*_current` commit.
        let response_kind = match &snap {
            TreeSnapshot::Dir(_) => ResourceKind::Dir,
            TreeSnapshot::File(_) => ResourceKind::File,
        };
        if let Some(prior_kind) = ctx.prior_kind
            && prior_kind != response_kind
        {
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
            self.finalize_anchor_lost_and_descend(profile_id, out);
            return CertifiedResponse::Regressed;
        }

        // Witness selection. The hash channel engages iff the burst owes a proof AND the events
        // stream is insufficient — both captured in the fold context above.
        let response_hash = snap.hash();
        let needs_hash_channel = ctx.owes_proof && !ctx.events_witness;

        // Cat-(b) carrier advance: the cascade (`Profile::advance_certified_sample` →
        // `ActiveBurst::advance_certified_sample` → `PreFireBurst::advance_certified_sample` /
        // `PostFireBurst::advance_certified_sample`) routes to whichever burst is live. Gated on
        // `Authoritative ∧ needs_hash_channel`: an Undischarged observation never advances (its
        // hash reflects an unread region), and the EventsReliable path skips the write entirely.
        let prior = if needs_hash_channel && matches!(authority, ProofAuthority::Authoritative) {
            self.profiles
                .get_mut(profile_id)
                .expect(
                    "certify_probe_response: fold_context proved Profile presence; \
                     the kind-mismatch path returned before reaching here",
                )
                .advance_certified_sample(response_hash)
        } else {
            None
        };
        let witness = if needs_hash_channel {
            QuiescenceWitness::HashChannel {
                prior,
                response: response_hash,
            }
        } else {
            QuiescenceWitness::EventsReliable
        };

        CertifiedResponse::Proceed {
            snapshot: snap,
            verdict: quiescence_verdict(authority, forced, witness),
        }
    }

    /// One immutable Profile resolution into the [`FoldContext`] the verdict fold consumes. `None`
    /// iff the Profile is absent (a gate breach — the floor is reached only on `Active(Verifying |
    /// Rebasing)`); the caller degrades to `Regressed`. Holds no borrow on return (every field is
    /// `Copy`), so the caller is free to take the `&mut self` re-fetch for the cat-(b) advance or
    /// the anchor-loss finalize afterward.
    fn fold_context(&self, profile_id: ProfileId) -> Option<FoldContext> {
        let profile = self.profiles.get(profile_id)?;
        Some(FoldContext {
            events_witness: profile.events_witness_quiescence(),
            prior_kind: profile.kind(),
            owes_proof: self.owes_proof_from(profile, profile_id),
        })
    }

    /// Whether the burst's consequence at the verdict floor requires a tree-quiescence proof
    /// (Contract B — "fire when the tree settles") rather than mere baseline establishment
    /// (Contract A — the cold-Seed `SilentPin` path, which records a reference freely).
    ///
    /// Returns `false` only for the cold-Seed quiet case: a `Seed`- intent `PreFire` burst with
    /// `dirty.is_empty()` AND no prior fires on the Profile. Every other reachable shape — Standard
    /// (any), Seed with witnessed activity, recovery Seed (`any_fired`), any `PostFire` (Rebase) —
    /// owes a quiescence proof.
    ///
    /// **Composed.** Reads `profile.state()` (intent + `dirty.is_empty()`) and
    /// `SubRegistry::any_fired`. The predicate spans two stores, so it lives on `Engine` rather
    /// than as a `Profile` method, taking the already-resolved `&Profile` from
    /// [`Self::fold_context`] (no redundant `get`) plus its `profile_id` for the `subs` lookup.
    ///
    /// **Conservative for recovery Seed.** `any_fired = true` Seed bursts with `dirty.is_empty()`
    /// are marked proof-owing even though [`Self::classify_consequence`] may post-`apply_snapshot`
    /// resolve them to `SilentPin` (no drift). The drift discriminant is computed only after the
    /// verdict commits a snapshot, so the floor must commit conservatively. Cost: one extra settle
    /// window before pinning if drift was absent on a structure-only recovery Seed; no fire missed.
    ///
    /// **Non-`Active` defaults to `true`.** A non-`Active` state at the verdict floor cannot occur
    /// — `fold_context` already proved Profile presence, and the floor is reached only through the
    /// `profile_probe_gate` ⇒ `take_owner_probe` dispatch on `Active(Verifying | Rebasing)`. The
    /// fall-through arm `debug_assert!(false)` to surface a contract violation in dev/CI and
    /// degrades to the proof-owing default in release, preserving the fire-safety invariant rather
    /// than silently bypassing it.
    #[must_use]
    fn owes_proof_from(&self, profile: &Profile, profile_id: ProfileId) -> bool {
        match profile.state() {
            ProfileState::Active(ActiveBurst::PostFire(_), _) => true,
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => match pre.intent {
                BurstIntent::Standard => true,
                BurstIntent::Seed => {
                    // Cold Seed (no activity, never fired) ⇒ `SilentPin` (Contract A); any other
                    // Seed ⇒ fire-bearing (`FreshSeedFire` / `RecoveryFire`) ⇒ Contract B.
                    !pre.dirty.is_empty() || self.subs.any_fired(profile_id)
                }
            },
            ProfileState::Idle | ProfileState::Pending(_) => {
                debug_assert!(
                    false,
                    "owes_proof_from: non-Active Profile {profile_id:?} reached the \
                     verdict floor (profile_probe_gate dispatches only on Active(Verifying | \
                     Rebasing))",
                );
                true
            }
        }
    }

    /// The single Verifying-phase choke: certify the response through
    /// [`Self::certify_probe_response`], then fan the result out per [`BurstIntent`]. The certifier
    /// owns the lower→kind-check→fold spine (shared with the Rebase arm); this routine owns only
    /// the pre-fire consequence.
    ///
    /// `Proceed` ⇒ the verdict feeds the single intent-agnostic [`Self::dispatch_quiescence_ok`]
    /// router (`intent` only selects the consequence split, not a forked path — the
    /// certified-sample machinery is intent-agnostic). `Vanished` / `Failed` ⇒ the per-intent
    /// failure helper (the split lives here, not in the certifier: a vanished anchor's cleanup is
    /// route-specific, and the Rebase arm maps the same two variants to its own helpers).
    /// `Regressed` ⇒ nothing — the certifier already emitted the diagnostic / tore the burst down.
    fn dispatch_burst_outcome(
        &mut self,
        profile_id: ProfileId,
        intent: BurstIntent,
        forced: bool,
        proof: ProofOutcome,
        now: Instant,
        out: &mut StepOutput,
    ) {
        match self.certify_probe_response(profile_id, proof, forced, out) {
            CertifiedResponse::Proceed { snapshot, verdict } => {
                self.dispatch_quiescence_ok(profile_id, snapshot, verdict, intent, now, out);
            }
            CertifiedResponse::Vanished => match intent {
                BurstIntent::Seed => self.dispatch_seed_vanished(profile_id, out),
                BurstIntent::Standard => self.dispatch_standard_vanished(profile_id, out),
            },
            CertifiedResponse::Failed(failure) => match intent {
                BurstIntent::Seed => self.dispatch_seed_failed(profile_id, failure, out),
                BurstIntent::Standard => self.dispatch_standard_failed(profile_id, failure, out),
            },
            CertifiedResponse::Regressed => {}
        }
    }

    /// Apply a successful probe response's `TreeSnapshot` to the Profile's `current`. Single home for
    /// the "Dir → graft / File → inline write" dispatch shared by the three `dispatch_*_ok` helpers.
    ///
    /// `TreeSnapshot::Dir` flows through [`crate::reconcile::graft`] (splice + reconcile + commit
    /// via `Profile::install_dir_current`); `TreeSnapshot::File` writes inline through
    /// [`specter_core::Profile::install_file_current`] (a Leaf has no descendants to materialise).
    ///
    /// **Typed prior extraction.** On the Dir arm this helper extracts the Dir prior from
    /// `Profile.current` under one immutable borrow and threads it to [`graft`] as a typed
    /// `Option<Arc<DirSnapshot>>`. Lifting the extraction here keeps graft's body Dir-typed
    /// end-to-end and centralises the File-shaped-prior detection at the single boundary that
    /// already owns the Profile borrow shape.
    ///
    /// **Kind agreement is a caller responsibility.** Production callers reach this helper only
    /// after [`Engine::certify_probe_response`]'s inline kind guard passed (Verifying via
    /// `dispatch_burst_outcome`, Rebasing via the post-fire arm). The setters' debug_assert is a
    /// defensive backstop for any future caller bypassing the boundary.
    pub(crate) fn apply_snapshot(
        &mut self,
        profile_id: ProfileId,
        target: ResourceId,
        snapshot: TreeSnapshot,
        out: &mut StepOutput,
    ) {
        match snapshot {
            TreeSnapshot::Dir(arc) => {
                // `current_dir()` borrows the prior Dir snapshot directly — `None` for an
                // unclassified, File-kinded, or not-yet-grafted Profile. The anchor sum makes a
                // kind-mismatched prior unrepresentable, so the old kind-agreement defensive arm is
                // now structural.
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
                    &mut self.coverage_scratch,
                );
            }
            TreeSnapshot::File(leaf) => {
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.install_file_current(leaf);
                }
            }
        }
    }

    /// Dispatch a [`specter_core::Input::TimerExpired`].
    ///
    /// `kind` tells us which transition this timer drives — settle expiry (Batching → Verifying,
    /// with possible reschedule), burst-deadline expiry (force-fire), gate-deadline expiry
    /// (actuator-hang recovery), or the post-fire rebase loop's spacing / ceiling. The `id` epoch
    /// survives the validation re-check that [`is_timer_referenced`] performs against the live
    /// burst slot for that `kind`; `pop_expired` already ran the same check before `step` was
    /// called, so the production path runs it twice (cheap), and any direct
    /// `step(Input::TimerExpired)` from a test or fuzzer falls through the same gate.
    ///
    /// `now` flows through to every handler that schedules a follow-up:
    /// [`Engine::on_settle_expired`]'s reschedule, [`Engine::handle_post_fire_settle_expired`]'s
    /// reschedule fork (the post-fire symmetric mirror of pre-fire's),
    /// [`Engine::handle_gate_deadline`] (drives `Awaiting → Rebasing` skip), and
    /// [`Engine::handle_rebase_ceiling`] (sets `forced` and drives `Settling → Rebasing` now if no
    /// probe is in flight). `BurstDeadline` is the only arm that ignores `now` —
    /// `handle_burst_deadline` sets `forced` and re-points the phase, scheduling nothing.
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
            TimerKind::PostFireSettle => self.handle_post_fire_settle_expired(profile, now, out),
            TimerKind::RebaseCeiling => self.handle_rebase_ceiling(profile, out),
        }
    }

    /// Settle-timer expiry. Either reschedule (events arrived since the timer was scheduled) or
    /// transition to Verifying (quiet for ≥ settle).
    ///
    /// Reschedule path: `now − last_event_time < settle`. Schedules a fresh `TimerKind::Settle` at
    /// `last_event_time + settle`; the `PreFireBurst.phase` re-point routes through
    /// [`Engine::reschedule_batching`] (the single-source mutator) while the quiet-window decision
    /// and timer math stay here. The old (just-expired) id is no longer referenced and lazily drops
    /// on a subsequent `pop_expired`. The phase stays Batching.
    ///
    /// Transition path: `now − last_event_time ≥ settle`. Forwards to
    /// [`Engine::transition_to_verifying`].
    ///
    /// **Structurally unreachable: `last_event_time = None` on a Batching expiry.** Every constructor
    /// that lands a burst in `Batching` pins `Some(now)`: `start_standard_burst`'s burst-start
    /// `FsEvent`; `start_seed_burst`'s triggered arm (`Some(trigger)` ⇒ Batching-first with
    /// `Some(now)`); the `event_drives_batching` re-entry from a Verifying/Draining cancel. The
    /// cold-Seed arm constructs `Verifying` directly with `None` and never schedules a `Settle`
    /// timer. The match's `None` arm is therefore unreachable in production; it carries
    /// `debug_assert!(false)` + the safe transition default to surface a future writer that opens the
    /// unreachable shape, the same convention `owes_proof_from` and `verifying_probe_target` use.
    ///
    /// **Preconditions** (guaranteed by [`is_timer_referenced`] upstream): `Profile.state ==
    /// Active(PreFire(_))` and `pre.phase == PreFirePhase::Batching { settle_timer == popped_id }`.
    /// The defensive early returns below cover direct `step(Input::TimerExpired)` calls that bypass
    /// `pop_expired`.
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
        // is_timer_referenced upstream guarantees Batching, but the direct-step path may bypass it;
        // gate the read defensively.
        if !matches!(pre.phase, PreFirePhase::Batching { .. }) {
            return;
        }
        let settle = p.settle;

        // saturating_duration_since handles `now < last` (test mockclock rewind / non-monotonic
        // clocks): returns Duration::ZERO, which satisfies `< settle` and triggers a reschedule.
        // Safe under any clock skew the harness can produce.
        match pre.last_event_time {
            Some(last) if now.saturating_duration_since(last) < settle => {
                let new_deadline = last + settle;
                let new_timer = self
                    .timers
                    .schedule(new_deadline, profile_id, TimerKind::Settle);
                self.reschedule_batching(profile_id, new_timer);
            }
            Some(_) => self.transition_to_verifying(profile_id, out),
            None => {
                debug_assert!(
                    false,
                    "on_settle_expired: last_event_time = None on Batching expiry \
                     for Profile {profile_id:?} — every Batching constructor pins \
                     Some(now); reaching here means a future writer opened the \
                     unreachable arm",
                );
                self.transition_to_verifying(profile_id, out);
            }
        }
    }

    /// Dispatch a [`specter_core::Input::EffectComplete`].
    ///
    /// The Profile is resolved from `key` ([`DedupKey::profile`] is O(1)); the Sub registry is
    /// consulted only for the unknown-Sub diagnostic.
    ///
    /// A Failed arrival clears the Sub's per-Sub fire history
    /// ([`specter_core::SubRegistry::clear_fired`]) — only for a `Subtree` `key`; `PerFile` carries
    /// no fire history. A failed Effect produced no observable state to deduplicate against, so the
    /// next stable verdict for that Sub must fire fresh even on an unchanged tree.
    /// Phase-independent (Awaiting decrement, late arrival, or unknown), and a no-op if the Sub
    /// already detached (its flag died with the slotmap entry).
    ///
    /// Two passes for borrow shapes (single-threaded `step` ⇒ no change between them): pass 1
    /// resolves the route (read borrow), pass 2 applies the completion (`&mut`). The counter owns
    /// its decrement and zero-edge ([`specter_core::Profile::note_effect_completion`]); this only
    /// routes the verdict:
    /// - `LastReached` ⇒ route on [`BurstFinish`]: `ReturnToIdle` → arm the rebase-loop ceiling at
    ///   the `Awaiting → Rebasing` edge ([`Engine::arm_rebase_loop_ceiling`], scheduled at `now +
    ///   max_settle`), then [`Engine::transition_to_rebasing`] to probe the post-command tree
    ///   immediately — probe-first, no driving FS event to debounce (the Cold-Seed invariant);
    ///   `Reap` → `finish_burst_to_idle`.
    /// - `Decremented` ⇒ stay Awaiting.
    /// - else (non-Awaiting, stale, `NotAwaiting`) ⇒ late completion: `EffectCompleteForUnknownSub`
    ///   / `EffectCompleteOutsideAwaiting`.
    ///
    /// `now` is the wall-clock instant of this completion — the actual `Awaiting → Rebasing` edge.
    /// The ceiling timer's deadline (`now + max_settle`) anchors on it; the immediate rebase probe
    /// itself needs no `now` (the `WholeSubtree` walk reckons from its response, not from a window
    /// deadline).
    pub(crate) fn on_effect_complete(
        &mut self,
        sub: SubId,
        key: &DedupKey,
        outcome: &EffectOutcome,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // The Sub registry is consulted only for the unknown-Sub diagnostic in the `Diagnose` arm:
        // a Sub detached mid-Awaiting (the reap-pending case) is gone from the registry by the time
        // its Effects' completions arrive, but the Profile is still alive and waiting for the
        // counter to drain — we must NOT short-circuit here, or the counter would never advance.
        // `key.profile()` is O(1) and never depends on the Sub registry.
        let profile_id = key.profile();

        // Failed clears the Sub's fire history regardless of state, so the next stable verdict for
        // it fires fresh even on an unchanged tree. Match `key` (not the `sub` param) for the scope
        // discriminant: only `Subtree` carries fire history.
        if matches!(outcome, EffectOutcome::Failed(_)) {
            match key {
                DedupKey::Subtree { sub, .. } => self.subs.clear_fired(*sub),
                // PerFile has no fire history (diff membership is the dedup) — nothing to clear.
                DedupKey::PerFile { .. } => {}
            }
        }

        // Pass 1 (read borrow): route only. Capture the `Copy` `BurstFinish` here — a Sub detaching
        // mid-Awaiting flips it via `mark_active_for_reap`, so the captured value is post-flip;
        // capturing keeps pass 2 a single `&mut` borrow.
        let route = match self
            .profiles
            .get(profile_id)
            .map(specter_core::Profile::state)
        {
            Some(ProfileState::Active(ActiveBurst::PostFire(post), finish)) => match &post.phase {
                PostFirePhase::Awaiting { .. } => CompletionRoute::CountDown(*finish),
                // The counter drained at the `Awaiting → Rebasing` edge; a completion arriving
                // anywhere in the rebase loop (in-flight Rebasing probe or the HashChannel spacing
                // Settling) is a late, untracked arrival.
                PostFirePhase::Rebasing(_) | PostFirePhase::Settling { .. } => {
                    CompletionRoute::Diagnose
                }
            },
            // PreFire phases (Batching / Verifying / Draining), Idle, Pending, stale Profile
            // (None): not waiting for this completion — a late arrival the engine no longer tracks.
            _ => CompletionRoute::Diagnose,
        };

        // Pass 2 (`&mut` borrow): the counter owns the decrement and the zero-edge; this dispatcher
        // only routes the verdict.
        match route {
            CompletionRoute::CountDown(finish) => match self
                .profiles
                .get_mut(profile_id)
                .map(specter_core::Profile::note_effect_completion)
            {
                Some(AwaitVerdict::Decremented) => {}
                Some(AwaitVerdict::LastReached) => match finish {
                    BurstFinish::ReturnToIdle => {
                        // No driving FS event — the command's own writes were absorbed during
                        // Awaiting and the WholeSubtree rebase re-observes them regardless, so go
                        // probe-first (the Cold-Seed invariant). Arm the loop ceiling at its start
                        // (the sole natural arming site), then drive the rebase now. NOT
                        // force_pending_post_fire — that is the gate-deadline forced variant; the
                        // natural path folds the verdict normally, so a HashChannel Profile still
                        // proves quiescence over N>=2 samples.
                        self.arm_rebase_loop_ceiling(profile_id, now);
                        self.transition_to_rebasing(profile_id, out);
                    }
                    // No Subs left to rebase for; finish_burst_to_idle runs the burst-end
                    // Draining-sweep reconfirm then the deferred reap (a direct reap_profile would
                    // skip the sweep).
                    BurstFinish::Reap => self.finish_burst_to_idle(profile_id, out),
                },
                // Off Awaiting between passes (unreachable under single-threaded `step`) or
                // vanished — late completion.
                Some(AwaitVerdict::NotAwaiting) | None => {
                    self.diagnose_late_completion(sub, profile_id, out);
                }
            },
            CompletionRoute::Diagnose => self.diagnose_late_completion(sub, profile_id, out),
        }
    }

    /// Diagnostic for a completion the engine no longer Awaits. Unknown Sub (detached + reaped) →
    /// Sub-keyed [`Diagnostic::EffectCompleteForUnknownSub`]; still-registered → Profile-keyed
    /// [`Diagnostic::EffectCompleteOutsideAwaiting`].
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

    /// Dispatch a [`specter_core::Input::ConfigDiff`].
    ///
    /// Atomic, name-keyed apply of the [`SubRegistryDiff`] buckets in the canonical order. The diff
    /// carries operator names, never engine ids: name → id resolution is a registry-owner operation
    /// and homes here against the engine's authoritative `by_name` index, never bin-side off the
    /// order-unguaranteed diagnostic stream.
    ///
    /// # Four buckets, validate-then-act
    ///
    /// The `modified` bucket is split into two semantically distinct transformations; the engine's
    /// response collapses to each arm's natural shape:
    ///
    /// 1. **Sub `removed`** — resolve the name. `Some` ⇒ `detach_sub_inner` (reap the Profile if
    ///    its last Sub left, defer if active). `None` ⇒ [`Diagnostic::ConfigDiffUnknownSub`] (a
    ///    name whose prior attach failed and never entered the registry — nothing to detach).
    /// 2. **Sub `modified_params`** — anchor + identity unchanged; only per-Sub fields differ.
    ///    Resolve the name; on `Some`, [`Self::rebind_sub_inner`] rebinds the Sub in place via the
    ///    [`specter_core::SubRegistry::rebind`] edge — no Profile churn, no kernel-watch flap, no
    ///    baseline loss. On `None`, the prior attach failed and the Sub never entered the registry;
    ///    the engine degrades the entry to a fresh attach
    ///    ([`Diagnostic::ConfigDiffRebindFallbackAttach`] narrates the reason).
    /// 3. **Sub `modified_identity`** — path / scan / max_settle / events changed; the Sub must
    ///    move to a different Profile partition. [`Self::validate_sub_attach`] pre-checks the only
    ///    fallible boundary (the new anchor's parse); on success, the engine detaches the old Sub
    ///    (if present) and attaches the new. On validation failure the old Sub stays in place —
    ///    **structural rollback** at the composition layer: the validate site captures nothing,
    ///    attach re-derives, so the state-mid-operation problem doesn't arise.
    /// 4. **Sub `added`** — `attach_sub_inner` materialises the anchor and registers the Sub.
    ///
    /// **Ordering: removed → params → identity → added.** `removed` first frees name slots a
    /// downstream identity-arm might want (defense in depth — the four buckets are name-disjoint by
    /// diff construction). `modified_params` next is the cheapest path (in-place rebind, no Profile
    /// churn) and locks in the new params before any reap could drop the Sub. `modified_identity`
    /// next: validation precedes detach so a malformed new path doesn't tear down a live attachment
    /// for nothing. `added` last, after every detach has freed its name slot. The buckets are
    /// name-disjoint by diff construction, and each `find_by_name` reads the live registry *after*
    /// prior mutations in the same step, resolving the current id.
    ///
    /// All resulting ops (across every attach / detach / rebind in the diff) merge into a single
    /// sorted `StepOutput`.
    pub(crate) fn on_config_diff(
        &mut self,
        subs: SubRegistryDiff,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // removed → modified_params → modified_identity → added
        for name in subs.removed {
            match self.subs.find_by_name(&name) {
                Some(sid) => self.detach_sub_inner(sid, DetachReason::ConfigDiffRemoved, out),
                None => out
                    .diagnostics
                    .push(Diagnostic::ConfigDiffUnknownSub { name }),
            }
        }
        for req in subs.modified_params {
            match self.subs.find_by_name(&req.params.name) {
                Some(sid) => {
                    let SubAttachRequest { params, .. } = req;
                    self.rebind_sub_inner(sid, params, out);
                }
                None => {
                    // Prior attach failed; the Sub never entered the registry. Params alone cannot
                    // apply to a non-existent Sub — degrade to a fresh attach, narrating the
                    // *reason* (the fallback attach emits its own lifecycle diagnostics).
                    out.diagnostics
                        .push(Diagnostic::ConfigDiffRebindFallbackAttach {
                            name: req.params.name.clone(),
                        });
                    let _ = self.attach_sub_inner(req, now, out);
                }
            }
        }
        for req in subs.modified_identity {
            // Validate-then-act: a malformed new anchor leaves the old Sub in place. The validate
            // is a pure read; on success the attach re-derives, so no engine state is captured
            // across the detach-attach boundary.
            if !self.validate_sub_attach(&req, out) {
                continue;
            }
            if let Some(old) = self.subs.find_by_name(&req.params.name) {
                self.detach_sub_inner(old, DetachReason::ConfigDiffIdentityChanged, out);
            }
            let _ = self.attach_sub_inner(req, now, out);
        }
        for req in subs.added {
            let _ = self.attach_sub_inner(req, now, out);
        }
        // The single-StepOutput sort happens at `step`'s caller.
    }

    /// Dispatch a [`specter_core::Input::WatchOpRejected`].
    ///
    /// The Sensor failed to install a kernel watch (typically `EMFILE` / `ENFILE` on FD
    /// exhaustion). Three things must happen:
    ///
    /// 1. [`specter_core::Tree::vacate`] the rejected slot — clear every contribution atomically,
    ///    so the engine's view of "is this slot watched?" matches reality.
    /// 2. Walk every Profile that holds a claim on `resource` (anchor / watch-root parent / descent
    ///    prefix) and clean up its bookkeeping — otherwise the owner flag contradicts the
    ///    post-vacate counter, and any subsequent owner-driven release path would either see the
    ///    wrong union on recompute or silently drift further out of sync.
    /// 3. Emit one `ProfileClaimPurged` Diagnostic per affected (owner, claim_kind) pair, plus the
    ///    umbrella `WatchOpRejected` diagnostic.
    ///
    /// A single resource may be claimed by several owners via different roles — anchor of P,
    /// watch-root parent of Q, descent prefix of R — so the fan-out walks every claim slot
    /// independently.
    ///
    /// Stale resources (already Unwatched, queue-race) are a no-op + `WatchOpRejected` diagnostic;
    /// the per-claim walk yields nothing because owner back-references would have been cleared at
    /// reap.
    pub(crate) fn on_watch_op_rejected(
        &mut self,
        resource: ResourceId,
        failure: WatchFailure,
        out: &mut StepOutput,
    ) {
        out.diagnostics
            .push(Diagnostic::WatchOpRejected { resource, failure });

        // Snapshot every claimer BEFORE any mutation. Borrow checker (we'll mutate self.profiles in
        // the loops) and we want a stable view of the pre-clamp world: a Profile that's
        // `Pending(d)` with `d.current_prefix() == resource` must be detected here, because the
        // helpers we run below transition the Profile to Idle.
        let mut anchor_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut parent_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        let mut descent_claimers: smallvec::SmallVec<[ProfileId; 2]> = smallvec::SmallVec::new();
        for (pid, p) in self.profiles.iter() {
            if matches!(p.anchor_claim(), AnchorClaim::Held) && p.resource() == resource {
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

        // Atomic terminus for the rejected slot: clear the contributions map, emitting the closing
        // `Unwatch`. The per-claimer loops below run their owner-bookkeeping and call `sub_watch`,
        // which short-circuits on the post-vacate state (absent key). One slot, one terminus.
        //
        // The three claimer classes below — anchor, watch-root parent, descent prefix — are the
        // only owners notified. A vacated `ProfileDescendant` contribution notifies no one, and the
        // post-graft reconciler re-watches only diff-created entries, so a rejected chain /
        // descendant watch never self-heals while the entry stays in the snapshot; observation
        // converges opportunistically through probes from still-watched ancestors. Accepted debt:
        // watch rejection is EMFILE-rare.
        self.tree.vacate(resource, out);

        // Anchor claimers: synthesise an anchor-loss. Bare `finalize_anchor_lost` — deliberately
        // not the descend wrapper: the anchor still exists on disk, the *kernel* refused the watch,
        // so a descent would re-materialize and loop materialize → Watch → reject. The coordinator
        // cancels any in-flight Active probe, releases the anchor flag (silent no-op on the
        // post-vacate contributions map), and finishes the burst to Idle. Net Sensor ops match the
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

        // Watch-root parent claimers: clear the flag. The Profile's anchor stays watched (different
        // `resource`), but re-establishing the parent watch after a rename / recreation requires an
        // operator restart; there is no auto-recovery.
        for pid in parent_claimers {
            self.release_watch_root_parent_claim(pid, out);
            out.diagnostics.push(Diagnostic::ProfileClaimPurged {
                profile: pid,
                claim: ClaimKind::WatchRootParent,
                resource,
                failure,
            });
        }

        // Descent claimers: `cancel_owner_probe` (disarm + Cancel iff a descent probe was in
        // flight, idempotent), then release the prefix claim (transitions Profile → Idle). Without
        // the cancel-before-release, a late `ProbeResponse` would arrive after the Profile
        // transitions out of Pending and drop with `StaleProbeResponse` — wasted I/O.
        for pid in descent_claimers {
            self.cancel_owner_probe(pid, out);
            self.release_descent_prefix_claim(pid, out);
            out.diagnostics.push(Diagnostic::ProfileClaimPurged {
                profile: pid,
                claim: ClaimKind::DescentPrefix,
                resource,
                failure,
            });
        }
    }

    /// Sensor reports it dropped events at the kernel level (inotify's `IN_Q_OVERFLOW`). Reseed
    /// every Profile in scope so the engine's post-probe Seed-Ok (`dispatch_quiescence_ok`)
    /// re-establishes baseline against disk reality and runs drift detection. Active-mode drift
    /// (`baseline.hash() != current.hash()`) fires once for every SubtreeRoot Sub on the Profile
    /// that has fired, then rebases.
    ///
    /// # Per-Profile dispatch
    ///
    /// Each in-scope Profile is reseeded according to its current state:
    ///
    /// - **`Idle`** — direct [`Engine::start_seed_burst`]. The Profile's `current` is preserved as
    ///   the seed probe's `baseline_subtree` for mtime-skip; the response
    ///   (`dispatch_quiescence_ok`) rebases or fires-on-drift.
    /// - **`Active(_)`** — abandon the in-flight burst via [`Engine::finish_burst_to_idle`] (which
    ///   cancels any pending probe and runs the Draining-sweep reconfirm cascade), then start a
    ///   fresh seed burst. The Standard burst's accumulated `dirty` provenance is discarded — the
    ///   seed re-baselines against the post-overflow tree, which strictly dominates whatever the
    ///   Standard burst was tracking. `reap_pending` Profiles reaped inside `finish_burst_to_idle`
    ///   skip the seed (no Profile to seed).
    /// - **`Pending(_)`** — the anchor doesn't yet exist and the Profile holds no baseline to
    ///   drift-test, so there is nothing to re-Seed; instead the descent re-probes via
    ///   [`Engine::on_descent_event`]. A disarmed descent (awaiting an `IN_CREATE` for the next
    ///   path component) that lost that event to the overflow window would otherwise wedge until
    ///   some unrelated event at the prefix; the fresh probe reads the post-overflow tree directly.
    ///   Skips internally when a probe is already in flight — its response reflects the
    ///   post-overflow state.
    ///
    /// # Scope
    ///
    /// [`OverflowScope::Global`] (the v1 inotify backend's only emit) reseeds every Profile in the
    /// registry. [`OverflowScope::Resource`] reseeds Profiles whose anchor is `r` or a descendant
    /// of `r` — the FSEvents per-stream signal; `profiles_in_subtree(r)` walks the tree's ancestor
    /// chain to compute membership.
    ///
    /// One [`Diagnostic::SensorOverflow`] per call surfaces the event in operator logs — the bursts
    /// the reseed schedules carry no per-Profile annotation that they were triggered by overflow.
    pub(crate) fn on_sensor_overflow(
        &mut self,
        scope: OverflowScope,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Snapshot the in-scope ProfileId set BEFORE any mutation. The loop below transitions
        // Profiles through Idle and re-into Active(Seed); a fresh `iter()` mid-loop would observe
        // the partial transitions and could double-handle a Profile.
        let profiles_to_reseed: smallvec::SmallVec<[ProfileId; 8]> = match scope {
            OverflowScope::Global => self.profiles.iter().map(|(pid, _)| pid).collect(),
            OverflowScope::Resource(r) => self.profiles_in_subtree(r),
        };

        // Exclude the snapshot-time `Draining` Profiles. A `Draining` Profile holds a verified-stable
        // `current` plus a descendant-driven, deadline-bounded reconfirm; a Seed re-walk is no
        // fresher (it mtime-skips against that same `current`) and tearing it down to a Seed discards
        // both the verified snapshot and the "ancestor fires once after the gating descendant
        // settles" relationship. The exclusion has to be at snapshot time, not an iteration-time
        // phase guard on the Active arm: a prior iteration's `finish_burst_to_idle` Draining sweep
        // can flip an in-scope Draining ancestor `Draining → Verifying` before the loop reaches it,
        // so by iteration time it is no longer Draining and the guard would never fire. Removing it
        // from the snapshot also means that, once the sweep has armed the lone reconfirm probe for
        // such an ancestor, the loop never reaches a second same-owner emission for it.
        let profiles_to_reseed: smallvec::SmallVec<[ProfileId; 8]> = profiles_to_reseed
            .into_iter()
            .filter(|&pid| {
                self.profiles
                    .get(pid)
                    .is_some_and(|p| !p.state().is_draining())
            })
            .collect();

        for pid in profiles_to_reseed {
            // The Profile may have been reaped between snapshot and this iteration via a prior
            // iteration's `finish_burst_to_idle` (a `reap_pending` Profile reaps when its burst
            // transitions to Idle). Stale id ⇒ skip.
            let Some(p) = self.profiles.get(pid) else {
                continue;
            };
            match p.state() {
                ProfileState::Idle => {
                    self.start_seed_burst(pid, None, now, out);
                }
                ProfileState::Active(_, finish) => {
                    // Overflow on an Active burst is reseed-XOR-reap, not a pure teardown. The
                    // in-flight probe's wire `Cancel` is a syscall-skip optimization only —
                    // `on_profile_probe_response`'s staleness gate is the sole correctness
                    // authority for a late response; the `Cancel` merely spares a not-yet-dequeued
                    // worker a wasted recursive walk. Whether it is needed turns on whether a
                    // superseding `submit` follows in THIS step:
                    //
                    //  reseed (will_reap == false): finish_burst_to_idle returns the Profile to
                    //  Idle, then start_seed_burst emits a fresh Probe{P,C2}. The sensor's
                    //  per-owner expectation map is a last-writer-wins upsert keyed by owner, so
                    //  submit(P,C2) alone supersedes C1: a not-yet-dequeued C1 worker self-skips on
                    //  expected[P] != C1. A wire Cancel{P} here would be strictly redundant AND the
                    //  only same-owner Cancel+Probe pair the engine can emit — so disarm the engine
                    //  slot only (take_owner_probe, no wire op), exactly as the response path does.
                    //
                    //  reap (will_reap == true): finish_burst_to_idle reaps the Profile and
                    //  start_seed_burst then no-ops (require_idle finds it detached). No
                    //  superseding submit follows, so the worker would run a full doomed walk —
                    //  emit the wire Cancel via cancel_owner_probe, the same syscall-skip the
                    //  pure-teardown sites rely on.
                    //
                    // The disarm MUST precede finish_burst_to_idle: that helper swaps the Profile
                    // to Idle and destructures the prior burst, so an armed Verifying/Rebasing slot
                    // would reach drop *there* and trip ProbeSlot's tripwire — before
                    // finish_burst_to_idle's own deferred reap_profile, whose cancel_owner_probe
                    // would by then see an already-Idle Profile (too late). This pre-finish disarm
                    // is the only consume that reaches the slot in time; it is not redundant with
                    // reap_profile's own.
                    //
                    // A Seed in `Batching` (or any burst in Batching/Draining/Awaiting) holds no
                    // probe slot — the slot lives on the Verifying/Rebasing phase variant. Both
                    // take_owner_probe and cancel_owner_probe are idempotent no-ops there, so the
                    // disarm above is harmless on a slot-less burst and the only states it does
                    // real work on are exactly the slot-bearing ones the tripwire argument covers.
                    //
                    // `will_reap` is read off the matched `finish` (BurstFinish is Copy) before any
                    // &mut self call, so NLL ends the &Profile borrow here — the shape
                    // handle_gate_deadline already compiles.
                    let will_reap = matches!(finish, BurstFinish::Reap);
                    if will_reap {
                        self.cancel_owner_probe(pid, out);
                    } else {
                        let _ = self.take_owner_probe(pid);
                    }
                    self.finish_burst_to_idle(pid, out);
                    self.start_seed_burst(pid, None, now, out);
                }
                ProfileState::Pending(_) => {
                    // No baseline to drift-test — re-probe the descent prefix instead, so an
                    // IN_CREATE lost to the unreliable window can't wedge the descent (a disarmed
                    // slot would otherwise wait forever for an event the kernel already dropped).
                    // Skips internally when a probe is already in flight (its response reflects the
                    // post-overflow tree). No per-Profile diagnostic — consistent with the
                    // Idle/Active arms; the step's SensorOverflow diagnostic covers it.
                    self.on_descent_event(pid, now, out);
                }
            }
        }

        out.diagnostics.push(Diagnostic::SensorOverflow { scope });
    }

    /// Enumerate Profiles whose anchor lies in the subtree rooted at `r` (the anchor itself is `r`,
    /// or `r` is on the anchor's ancestor chain). Used by [`Self::on_sensor_overflow`] to scope a
    /// per-resource overflow signal — the FSEvents-style "this stream's queue overflowed" case. v1
    /// inotify always emits [`OverflowScope::Global`] so this is dead-stream-equipment in the
    /// inotify path; kept for the engine API's symmetric handling across backends.
    ///
    /// Worst-case `O(profiles × tree-depth)`. Acceptable for typical per-resource overflow rates
    /// (rare under healthy invariants).
    fn profiles_in_subtree(&self, r: ResourceId) -> smallvec::SmallVec<[ProfileId; 8]> {
        self.profiles
            .iter()
            .filter(|(_, p)| p.resource() == r || self.tree.ancestors(p.resource()).any(|a| a == r))
            .map(|(pid, _)| pid)
            .collect()
    }

    /// Start a new burst (Seed if no baseline yet, Standard if baseline established); pre-fire
    /// `Active` → fold the event through `event_drives_batching` (notes into `dirty`, emits a
    /// Cancel iff a probe was in flight, arms a fresh settle timer); post-fire `Active` (`Awaiting`
    /// / `Rebasing`) → defer it via `absorb_event_into_fire_tail`.
    ///
    /// Pre-fire, `dirty` is the single accumulator — its captured paths both derive the probe scope
    /// (their component-LCA, resolved to a live id) and are projected to the `Chains` proof
    /// obligation at the emission choke. The post-fire absorb notes `(event_resource, event_path)`
    /// into the post-fire `dirty` (the fire-tail residual): the next `WholeSubtree` rebase read
    /// re-observes it by construction (that obligation never mtime-skips), and a non-empty residual
    /// restarts a fresh Standard burst instead of finishing to Idle — the post-command self-trigger
    /// guard.
    ///
    /// `event` is threaded purely for the `EventAbsorbedByFireTail` diagnostic so the operator can
    /// correlate logs to the deferred FsEvent. The "no baseline → Seed" branch handles the
    /// degenerate post-`Vanished` Idle state (`current.is_none()`): a Standard burst with no
    /// baseline cannot dispatch a stability verdict.
    fn drive_burst(
        &mut self,
        profile_id: ProfileId,
        event_resource: ResourceId,
        event_path: &Arc<Path>,
        event: FsEvent,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        match p.state() {
            ProfileState::Idle => {
                // The fork's real question is "do I have a trustworthy *settled baseline* to
                // debounce against, or must I re-Seed?". `current_is_some()` was only ever a
                // faithful proxy because `current` and `settled` were once installed atomically; a
                // Seed that grafts `current` while deferring the pin (the `Retry` re-batch loop, or
                // a loop terminated by the `Abandon` ceiling) leaves Idle with `current = Some` but
                // no settled baseline — routing that to a Standard burst would make a never-fired
                // Profile *fire* on first quiescence where Seed deliberately stays silent. Branch
                // on the baseline's presence directly. (`classify_event_carriers`'s
                // `!current_is_some()` is a different question — "is the anchor absent ⇒
                // loss-recovery candidate?" — and correctly stays `current_is_some()`.)
                if p.baseline_is_some() {
                    self.start_standard_burst(profile_id, event_resource, event_path, now, out);
                } else {
                    // Thread the triggering FsEvent into the Seed's provenance so an isolated
                    // post-recovery change (Idle+!baseline reached via the `undischarged_consequence`
                    // ceiling terminal) is witnessed — symmetric with `start_standard_burst`.
                    self.start_seed_burst(
                        profile_id,
                        Some((event_resource, Arc::clone(event_path))),
                        now,
                        out,
                    );
                }
            }
            // The post-fire absorb arm is *the* typed-disjoint path from the pre-fire
            // `event_drives_batching` arm: noting into `dirty` and emitting
            // `EventAbsorbedByFireTail` belongs to `PostFireBurst` alone, and the helper owns the
            // mutation in `burst.rs` so `transitions.rs` never reaches for burst internals.
            ProfileState::Active(ActiveBurst::PostFire(_), _) => {
                self.absorb_event_into_fire_tail(
                    profile_id,
                    event_resource,
                    event_path,
                    event,
                    now,
                    out,
                );
            }
            ProfileState::Active(ActiveBurst::PreFire(_), _) => {
                self.event_drives_batching(profile_id, event_resource, event_path, now, out);
            }
            // Pending Profiles never reach here — `covering_profiles` filters them at the source.
            // Defensive no-op.
            ProfileState::Pending(_) => {}
        }
    }

    /// Arm (or re-arm) the operator `absorb` window on a Profile — the
    /// [`specter_core::Input::ArmAbsorb`] handler. Deriving the window's `(expiry, mode)` from the
    /// operator's `duration` is the one place this policy lives: `None` ⇒ a one-shot window one
    /// `settle` interval wide ([`AbsorbMode::ConsumeOnFirst`], to cover a single expected
    /// replication); `Some(d)` ⇒ a `d`-wide window ([`AbsorbMode::PersistUntil`], to cover a run of
    /// them). A `--for 0s` yields `expiry == now`, which `absorb_window_live` reads inert — a
    /// harmless operator-owned no-op, no validation.
    ///
    /// [`Profile::arm_absorb`] sets the window **and** retro-latches any in-flight pre-fire burst
    /// in one operation (the reverse race — the replication's events opened a burst before the
    /// signal arrived — so a burst already batching folds too). A stale `profile` (reaped between
    /// the driver's name resolution and this step) no-ops silently: there is no Profile for the
    /// window to live on. Narrates via [`Diagnostic::AbsorbArmed`] so a `tail` sees the arm, not
    /// only the eventual [`Diagnostic::QuiescenceAbsorbed`] fold.
    pub(crate) fn on_arm_absorb(
        &mut self,
        profile_id: ProfileId,
        duration: Option<Duration>,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if let Some(p) = self.profiles.get_mut(profile_id) {
            let (expiry, mode) = match duration {
                Some(d) => (now + d, AbsorbMode::PersistUntil),
                None => (now + p.settle, AbsorbMode::ConsumeOnFirst),
            };
            p.arm_absorb(expiry, mode);
            out.diagnostics.push(Diagnostic::AbsorbArmed {
                profile: profile_id,
                mode,
            });
        }
    }

    /// Finalize the loss of a Profile's anchor: cancel any in-flight probe, release the anchor's
    /// `watch_demand` contribution, drop the stale `baseline` / `current` snapshots, and finish the
    /// burst to Idle if Active.
    ///
    /// **The single anchor-loss coordinator.** Every loss route funnels here: the five observed-loss
    /// sites through [`Self::finalize_anchor_lost_and_descend`] (which re-enters descent after this
    /// returns), the three probe-`Failed` dispatches and `on_watch_op_rejected`'s anchor purge
    /// directly (no descent — their failure modes would loop it). The probe-dispatch callers reach
    /// this with their slot already disarmed (`take_owner_probe` ran before dispatch), so the
    /// `cancel_owner_probe` below is a no-op there — no wire Cancel; for the event/rejection routes
    /// it cancels a genuinely in-flight Verifying/Rebasing probe.
    ///
    /// **`watch_root_parent` is intentionally preserved.** It is both the descent prefix the
    /// observed-loss wrapper re-enters against immediately, and — for the bare-finalize callers —
    /// the channel through which a later parent `StructureChanged` re-enters descent
    /// (`start_pending_recovery`); releasing it here would close recovery in both shapes. The
    /// contribution is released only when the Profile itself reaps (`reap_profile` →
    /// `release_watch_root_parent_claim`). Sibling helpers — anchor, descendants, descent prefix —
    /// *are* released here; the asymmetry is by design.
    ///
    /// **Ordering.** The anchor release runs BEFORE `finish_burst_to_idle`, so any deferred
    /// `reap_profile` (`reap_pending`) sees an `AnchorClaim::None` and skips its redundant release
    /// inside `reap_profile::release_anchor_claim`. Reverse-ordering would have
    /// `finish_burst_to_idle` invoke `reap_profile`, which would release the anchor; the
    /// post-`finish` release would then see an absent contribution and silently no-op — correct but
    /// redundant. The "release-then-finish" ordering keeps the cleanup ordered.
    ///
    /// **Pending exclusion.** `ProfileState::Pending` is defensive on the FsEvent route
    /// (`covering_profiles` filters Pending Profiles at the source) and vacuous on the
    /// probe-dispatch routes (`profile_probe_gate` proves `Active(Verifying | Rebasing)`).
    /// `on_watch_op_rejected` calls this directly after iterating the full registry, where the
    /// guard does load-bearing work: a Pending Profile holds no anchor (it is still descending
    /// toward one) — anchor-loss finalization does not apply to it, and its descent-prefix watch
    /// rejection is handled separately as a descent-prefix claim purge.
    pub(crate) fn finalize_anchor_lost(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        if matches!(p.state(), ProfileState::Pending(_)) {
            return;
        }
        // Capture `was_active` BEFORE discard_anchor_state. The helper does not mutate
        // Profile.state (only `finish_burst_to_idle` does), so the read is order-insensitive in v1;
        // pinning it before the helper guards against any future helper change that touches state.
        let was_active = matches!(p.state(), ProfileState::Active(_, _));

        // Idempotent: emits Cancel iff a probe is in flight (Active+Verifying ⇒ slot armed). For
        // Active+Batching / Draining no probe is in flight and the helper is a no-op — structural
        // equivalent of the prior `was_verifying` snapshot. Required by discard_anchor_state's
        // cancel-first contract.
        self.cancel_owner_probe(profile_id, out);

        // Discard runs BEFORE finish_burst_to_idle. The release-helpers inside emit
        // `AnchorClaim::None` and clear `Profile.kind` before any deferred `reap_profile`
        // (`reap_pending`) fires from `finish_burst_to_idle` — preserves the trichotomy invariant
        // `!(Pending && Held)` across the eventual `start_pending_recovery` transition, and lets
        // the next Seed burst route through the kind-agnostic Subtree probe rather than misroute
        // against a recreated anchor of a different shape.
        self.discard_anchor_state(profile_id, out);

        if was_active {
            self.finish_burst_to_idle(profile_id, out);
        }
    }

    /// Observed-loss terminal: [`Self::finalize_anchor_lost`], then re-enter pending descent at the
    /// anchor's recovery parent *inside the loss step itself* — witnessed, so the terminus Seed
    /// owes a fire (`seed_owes_first_fire` / witness drift).
    ///
    /// "Anchor lost" and "anchor doesn't yet exist" are the same state: Resource identity is
    /// `(parent, segment)` and survives delete-and-recreate, so an observed loss re-enters the same
    /// descent a not-yet-existing attach runs, and operators see an honest `Pending`. Descending
    /// immediately — rather than parking Idle-anchorless and betting on a *future*
    /// `watch_root_parent` event — is what kills the two atomic-save races: the loss signal
    /// postdates the rename, so the descent probe reads post-rename truth (a parent event that
    /// *preceded* the terminal can no longer strand the Profile), and there is no anchorless window
    /// in which the next save's early tmp-create event could re-capture the dying inode.
    ///
    /// Callers are the five observed-loss sites: the anchor-terminal event path, the kind-mismatch
    /// certifier arm, and the three probe-`Vanished` dispatches. The three probe-`Failed`
    /// dispatches and `on_watch_op_rejected`'s anchor purge stay on bare
    /// [`Self::finalize_anchor_lost`] — descending there would loop tightly (probe → fail;
    /// materialize → watch → reject) against a condition the descent cannot fix; those Profiles
    /// keep the event-scan re-entry ([`Self::start_pending_recovery`]).
    fn finalize_anchor_lost_and_descend(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        self.finalize_anchor_lost(profile_id, out);
        // `finish_burst_to_idle` may have honoured a deferred reap (`BurstFinish::Reap`); a reaped
        // Profile has nothing to recover.
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        // Post-finalize the Profile is Idle-anchorless on every reachable path (finalize returns
        // early on `Pending` and never constructs it). The guard is the recoveries-arm predicate
        // verbatim, kept as a structural backstop rather than an assert: a non-Idle or still-anchored
        // Profile here means finalize declined, and descending would tear live state.
        if !matches!(p.state(), ProfileState::Idle) || p.current_is_some() {
            return;
        }
        // The recovery channel, NOT `tree.parent(anchor)`: `None` only after a parent-watch
        // rejection purge, where a descent would loop materialize → watch → reject — stay latent
        // (operator restart is the documented recovery there).
        let Some(parent) = p.watch_root_parent() else {
            return;
        };
        self.enter_anchor_recovery_descent(profile_id, parent, /* witnessed: */ true, out);
    }

    /// Emit [`Diagnostic::PerFileDriftDroppedOnRecovery`] iff a live survival witness exists, the
    /// post-graft `current` drifted from it, and the Profile carries a `PerStableFile` Sub. A
    /// Seed-Ok that closes an anchor-loss window rebases `baseline := observed`, absorbing the
    /// whole loss-window delta in one move: the Subtree side re-fires its drifted Subs from the
    /// witness, but a `PerStableFile` Sub has no per-leaf witness, so its loss-window reactions
    /// vanish without a trace — exactly the case to flag.
    ///
    /// Standalone witness-drift predicate, **not** folded into the drift fork: a PerFile-only
    /// recovery never records a fire (so it classifies [`Consequence::SilentPin`], never
    /// [`Consequence::RecoveryFire`]) yet is precisely this signal's target, so the condition
    /// cannot piggy-back on the fork. Invoked while the witness is still live — at the recovery
    /// fire (post-gate, once) and on the seal-only terminal, each *before*
    /// [`specter_core::Profile::rebase_baseline`] consumes it. A byte-identical recovery (`current
    /// == witness`) dropped nothing and emits nothing.
    fn warn_perfile_dropped_on_recovery(&self, profile_id: ProfileId, out: &mut StepOutput) {
        if let Some(p) = self.profiles.get(profile_id)
            && let Some(witness) = p.survival_witness()
            && let Some(current_h) = p.current_hash()
            && current_h != witness
            && self.subs.has_per_stable_file_sub(profile_id)
        {
            out.diagnostics
                .push(Diagnostic::PerFileDriftDroppedOnRecovery {
                    profile: profile_id,
                });
        }
    }

    /// Map a fireable verdict to its [`Consequence`] for a burst of known
    /// [`specter_core::BurstIntent`] — the single home of the [`Engine::seed_owes_first_fire`] /
    /// [`Engine::seed_drift_observed`] fork. Pure `&self`; reads the post-graft state the caller
    /// ([`Engine::fire_or_seal`]) committed immediately before.
    ///
    /// **The shape layer precedes both the intent fork and the fold override.** A
    /// `MatchChain`-shaped Profile classifies [`Consequence::Reconcile`] for *any* intent: cold
    /// Seed (first enumeration), triggered Seed, Standard, and post-recovery Seed all converge on
    /// the same idempotent reconcile (the registry dedup query makes a re-reconcile a no-op), so
    /// the Seed-flag machinery below is never consulted for discovery. The early return also
    /// bypasses the fold override *structurally* rather than by assertion: an operator `absorb`
    /// window suppresses Effects, discovery emits none, so minting proceeds under absorb and the
    /// latch stays unconsumed — disabling the discovery Sub is the lever to stop minting.
    ///
    /// A Standard burst always fires the Standard consequence. A Seed burst splits three disjoint
    /// ways: a fresh Profile that witnessed activity owes a *first* fire (the Standard consequence
    /// — no baseline yet, the post-command rebase establishes it); a recovery whose tree drifted
    /// re-fires its previously-fired Subs and seals the witness; everything else (fresh-static
    /// restart, no-drift or empty-fired recovery) is a silent witness-sealing pin.
    /// `seed_owes_first_fire` and `seed_drift_observed` stay the disjoint building blocks, **not**
    /// flattened: a Seed-Ok that owes a first fire and one that re-fires a recovery are different
    /// consequences reached through different settled-reference reasoning, and the per-Sub vs
    /// per-Profile split is load-bearing for B1 dedup.
    ///
    /// **The fold override is the single, final layer** atop the intent fork: a burst that froze
    /// the fold latch at birth (a live operator `absorb` window —
    /// [`specter_core::ProfileState::burst_fold_latched`]) folds rather than fires. A firing `base`
    /// ([`Consequence::is_firing`]) becomes [`Consequence::AbsorbFold`]; a non-firing `base`
    /// ([`Consequence::SilentPin`]) passes through, so a redundant Cold-Seed leaves the window
    /// armed for the first fireable burst. This is the **sole** verdict-time consult of the fold
    /// decision, and it reads the burst's frozen latch, never the window — the
    /// orthogonal-to-[`specter_core::BurstIntent`] terminal-consequence switch the latch was
    /// designed to be.
    fn classify_consequence(&self, profile_id: ProfileId, intent: BurstIntent) -> Consequence {
        // Shape layer: a discovery Profile's stable verdict reconciles the match set, whatever the
        // intent. Returning before the intent match keeps the fold override unreachable —
        // absorb-inertness is structural, not asserted.
        if self
            .profiles
            .get(profile_id)
            .is_some_and(|p| p.config().match_chain().is_some())
        {
            return Consequence::Reconcile;
        }

        let base = match intent {
            BurstIntent::Standard => Consequence::StandardFire,
            BurstIntent::Seed => {
                if self.seed_owes_first_fire(profile_id) {
                    Consequence::FreshSeedFire
                } else if self.seed_drift_observed(profile_id) {
                    // `seed_drift_observed` implies `any_fired`, which within this single `&self`
                    // resolution implies a non-empty fired set — so the empty arm is a
                    // registry-timing defensive (a detached Sub's flag died with its slotmap
                    // entry). It falls to the silent seal, exactly as the prior pin's
                    // `!drifted.is_empty()` guard did.
                    let drifted = self.subs.fired_in(profile_id);
                    if drifted.is_empty() {
                        Consequence::SilentPin
                    } else {
                        Consequence::RecoveryFire(drifted)
                    }
                } else {
                    Consequence::SilentPin
                }
            }
        };

        // The single fold override. A burst born (or retro-latched) under a live `absorb` window
        // folds instead of firing: a firing `base` is replaced with the silent baseline advance.
        // Read the *latch* the burst froze at birth — never the window — so a long transfer that
        // outlived its settle window still folds (the window may already read inert; the latch does
        // not). A non-firing `base` passes through untouched, so a Cold-Seed `SilentPin` (which
        // proves nothing) leaves the window unconsumed for the first genuinely fireable burst.
        if base.is_firing()
            && self
                .profiles
                .get(profile_id)
                .is_some_and(|p| p.state().burst_fold_latched())
        {
            Consequence::AbsorbFold
        } else {
            base
        }
    }

    /// Commit the observed tree, then route the verdict's [`Consequence`]. The shared
    /// [`QuiescenceVerdict::Stable`] consequence for **both** intents — the certified-sample
    /// machinery is intent-agnostic, so there is no Seed special case to fork at this layer.
    ///
    /// `apply_snapshot` runs *before* `classify_consequence`: the drift read needs the post-graft
    /// `current`, and `seed_owes_first_fire`'s inputs (`any_fired`, the pre-fire `dirty`
    /// accumulator) are invariant under the graft, so the classification is order-stable.
    ///
    /// The three non-firing arms — [`Consequence::SilentPin`], [`Consequence::AbsorbFold`], and
    /// [`Consequence::Reconcile`] — share the silent-seal terminus
    /// ([`Engine::seal_baseline_silently`]): no Effect to defer, so none consults the Draining
    /// gate. `AbsorbFold` and `Reconcile` each run a per-cause prologue first (the
    /// [`Diagnostic::QuiescenceAbsorbed`] narration + count bump; the reconcile's mint pass), since
    /// the seal may reap the Profile. The three firing consequences cross the single gate in
    /// [`Engine::gated_fire`].
    fn fire_or_seal(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        target: ResourceId,
        forced: bool,
        intent: BurstIntent,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Kind agreement is verified upstream at the `dispatch_burst_outcome` choke (hoisted, once
        // for Seed + Standard, before this dispatch).
        self.apply_snapshot(profile_id, target, snapshot, out);
        match self.classify_consequence(profile_id, intent) {
            Consequence::StandardFire => {
                self.gated_fire(
                    profile_id,
                    EmitMode::Standard { forced },
                    forced,
                    false,
                    now,
                    out,
                );
            }
            Consequence::FreshSeedFire => {
                self.gated_fire(
                    profile_id,
                    EmitMode::Standard { forced },
                    forced,
                    true,
                    now,
                    out,
                );
            }
            Consequence::RecoveryFire(drifted) => {
                self.gated_fire(
                    profile_id,
                    EmitMode::SeedDrift { drifted: &drifted },
                    forced,
                    false,
                    now,
                    out,
                );
            }
            Consequence::Reconcile => {
                // Prologue-before-seal, `AbsorbFold`'s ordering rationale: the seal can reap the
                // Profile (a template detached mid-burst marks the burst for reap), so the mint
                // pass lands while it is live. `reconcile_matches` derives its template set from
                // the live registry, so that same zombie burst mints nothing — and it touches no
                // burst state; the burst exits through the seal's `finish_burst_to_idle` alone.
                self.reconcile_matches(profile_id, now, out);
                self.seal_baseline_silently(profile_id, out);
            }
            Consequence::SilentPin => self.seal_baseline_silently(profile_id, out),
            Consequence::AbsorbFold => {
                // Per-cause prologue, then the SAME silent-seal terminus as `SilentPin`. The
                // bookkeeping runs *before* the seal: `seal_baseline_silently` finishes the burst,
                // which can reap the Profile (and its Subs), so the diagnostic and the count
                // bump/consume must land while the Profile is still live.
                out.diagnostics.push(Diagnostic::QuiescenceAbsorbed {
                    profile: profile_id,
                });
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.note_absorb_fold();
                }
                self.seal_baseline_silently(profile_id, out);
            }
        }
    }

    /// Seal `baseline := current` silently and finish the burst — the shared terminus for a
    /// non-firing commit.
    ///
    /// Flags the per-file drop *while the witness is still live* (the predicate self-gates to a
    /// live witness + a `PerStableFile` Sub), then rebases and finishes. The warn must precede
    /// [`specter_core::Profile::rebase_baseline`], which consumes the witness it reads. No Effect
    /// to defer ⇒ never Draining-gated. The caller ([`Engine::fire_or_seal`]) has already committed
    /// the observed tree (`apply_snapshot`, once for every consequence), so this seals over the
    /// post-graft `current`.
    fn seal_baseline_silently(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        self.warn_perfile_dropped_on_recovery(profile_id, out);
        if let Some(p) = self.profiles.get_mut(profile_id) {
            p.rebase_baseline();
        }
        self.finish_burst_to_idle(profile_id, out);
    }

    /// The single Draining-gate site. Fire iff `forced` (the `max_settle` ceiling expired — out of
    /// time, fire regardless of a churning covered descendant) **or** no covered strict-descendant
    /// Profile is in an Active Standard burst; otherwise defer via
    /// [`Engine::transition_to_draining`] (don't fire an ancestor's "tree settled" command while a
    /// nested watched subtree churns — the `finish_burst_to_idle` sweep reconfirms once the
    /// descendant settles). The descendant query is evaluated fresh here, never cached; `forced`
    /// short-circuits it, so a deadline-fire neither pays for nor consults the gate. `forced` is
    /// the [`QuiescenceVerdict::Stable`]([`StableReason::Forced`]) ceiling case (the bounded
    /// `BurstDeadline` / `RebaseCeiling` expired) — a hard ceiling fires through, never defers.
    ///
    /// On the fire branch, post-gate and exactly once: the fresh-Seed per-file-skip narration
    /// (`fresh_seed`), then — for the recovery consequence (`EmitMode::SeedDrift`) — the
    /// per-file-drop honesty, while the witness is still live ([`Engine::fire_and_settle`] seals
    /// after the emit). The defer emits neither; both re-derive at the reconfirm terminal, so each
    /// surfaces exactly once.
    fn gated_fire(
        &mut self,
        profile_id: ProfileId,
        mode: EmitMode<'_>,
        forced: bool,
        fresh_seed: bool,
        now: Instant,
        out: &mut StepOutput,
    ) {
        if forced
            || !crate::coverage::has_active_standard_descendant(
                &self.tree,
                &self.profiles,
                profile_id,
            )
        {
            if fresh_seed {
                self.warn_perfile_skipped_on_fresh_seed(profile_id, out);
            }
            if matches!(mode, EmitMode::SeedDrift { .. }) {
                self.warn_perfile_dropped_on_recovery(profile_id, out);
            }
            self.fire_and_settle(profile_id, mode, now, out);
        } else {
            self.transition_to_draining(profile_id, out);
        }
    }

    /// (Seed | Standard, Ok) — map the quiescence `verdict` to its consequence. One router for both
    /// intents: `intent` only selects the [`Consequence`] split in [`Engine::classify_consequence`]
    /// and threads onto the [`Diagnostic::QuiescenceCeilingUnreadable`] /
    /// [`Diagnostic::QuiescenceCeilingForcedDespiteChange`] payloads — there is no forked path.
    ///
    /// - [`QuiescenceVerdict::Stable`]([`StableReason::Natural`]) — the natural fire path.
    ///   [`Engine::fire_or_seal`] commits, classifies, then either gated-fires or silently seals
    ///   the witness. No diagnostic owed (the witness held).
    /// - [`QuiescenceVerdict::Stable`]([`StableReason::Forced`]) — the bounded `BurstDeadline`
    ///   fallback fired. `fire_or_seal` runs the same fire path with `forced = true`, which
    ///   propagates onto [`specter_core::Effect::forced`] via [`EmitMode::Standard`] and crosses
    ///   the Draining gate via the `forced` disjunct in [`Engine::gated_fire`]. On
    ///   `hash_channel_disagreed = true` (strong signal — the hash channel observed `prior !=
    ///   response` before the ceiling expired) the dispatch emits
    ///   [`Diagnostic::QuiescenceCeilingForcedDespiteChange`]; the quiet `false` case stays silent
    ///   — a forced *fire* carries the bit on its [`specter_core::Effect`], and a forced silent
    ///   seal ([`Consequence::SilentPin`]) observed no change worth flagging.
    /// - [`QuiescenceVerdict::Retry`] — non-firing, non-terminal: the walker certified but the hash
    ///   channel observed `prior != Some(response)` (events-incomplete fire-bearing burst), or the
    ///   walker refused on some chain (transient non-observation — `EACCES`, a chmod-000 chain).
    ///   Both origins route through [`Engine::retry_drives_batching`]; never commit (the prior
    ///   carrier value is the last walker-certified sample, not a quiescent one; an unread region
    ///   must not poison `current`). The bounded `BurstDeadline` ceiling eventually surfaces a
    ///   `Stable(Forced)` (channel-disagreement path) or `Abandon` (walker-refused path) terminal.
    /// - [`QuiescenceVerdict::Abandon`] — the bounded ceiling already fired and the probe could not
    ///   discharge its obligation. Surface `first_unread` via
    ///   [`Diagnostic::QuiescenceCeilingUnreadable`] and finish the burst **without** committing —
    ///   an unread region must never become the dedup / Seed baseline.
    fn dispatch_quiescence_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        verdict: QuiescenceVerdict,
        intent: BurstIntent,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Profile-presence guard + the snapshot-commit target read back off the in-flight Verifying
        // probe (the latest emitted probe target). The covered-descendant fire-gate lives at the
        // single gate site ([`Engine::gated_fire`]), short-circuited by `forced`.
        let Some(target) = self.verifying_probe_target(profile_id) else {
            return;
        };

        match verdict {
            QuiescenceVerdict::Stable(StableReason::Natural) => {
                // Natural fire path. `forced = false` propagates onto `Effect.forced`; the Draining
                // gate is consulted via `gated_fire` (no `forced` short-circuit).
                self.fire_or_seal(profile_id, snapshot, target, false, intent, now, out);
            }
            QuiescenceVerdict::Stable(StableReason::Forced {
                hash_channel_disagreed,
            }) => {
                // Bounded-ceiling path. The strong-signal diagnostic emits only when the hash
                // channel observed a concrete prior/response disagreement before the deadline
                // expired; the quiet `false` case stays silent — a forced fire carries the bit on
                // its `Effect`, a forced silent seal observed no change worth flagging. The
                // fire-or-seal routing is identical either way; on the firing arms `forced = true`
                // triggers the Draining-gate bypass in `gated_fire`.
                if hash_channel_disagreed {
                    out.diagnostics
                        .push(Diagnostic::QuiescenceCeilingForcedDespiteChange {
                            profile: profile_id,
                            intent,
                        });
                }
                self.fire_or_seal(profile_id, snapshot, target, true, intent, now, out);
            }
            QuiescenceVerdict::Retry => {
                // Two operationally-identical origins collapse here: the hash channel observed
                // `prior != Some(response)` (the tree is moving under the verify window), or the
                // walker refused on some chain (transient non-observation — `EACCES`, a chmod-000
                // chain). Re-arm the settle window for another sample; never commit (the prior
                // carrier value is the last walker-certified sample, not a quiescent one; an unread
                // region must not poison `current`). The bounded `BurstDeadline` ceiling eventually
                // surfaces the operator-visible terminal — `Stable(Forced)` on persistent
                // disagreement, `Abandon` on a persistent unread chain.
                self.retry_drives_batching(profile_id, now, out);
            }
            QuiescenceVerdict::Abandon { first_unread } => {
                // Bounded terminal: the ceiling already fired and the walker still refused. No
                // commit; surface the unread path with the burst's `intent` so operators can
                // distinguish a Seed-baseline failure from a Standard reconfirm failure.
                out.diagnostics
                    .push(Diagnostic::QuiescenceCeilingUnreadable {
                        profile: profile_id,
                        first_unread,
                        intent,
                    });
                self.finish_burst_to_idle(profile_id, out);
            }
        }
    }

    /// Decide whether a Seed-Ok should fire conservative-recovery Effects: `true` iff the Profile
    /// has fired before AND the post-graft `current` snapshot's anchor-rooted hash differs from the
    /// settled reference.
    ///
    /// [`Profile::settled_hash`] is the single settled-reference oracle: in active mode it digests
    /// the baseline snapshot; across the loss→recovery window it returns the survival witness the
    /// anchor carried through the loss (covering anchor-loss recovery via descent → Seed-Ok, and
    /// `on_sensor_overflow` reseed); a not-yet-settled anchor yields `None`. The settled snapshot
    /// and the survival witness are mutually exclusive *in the anchor sum*, so the
    /// survival-mode-authoritative priority is structural — there is no ordering to maintain here
    /// and the witness cannot be silently lost on recovery. `None` (a fresh, never-fired Profile)
    /// preserves "a fresh Seed with **no witnessed activity** never fires an Effect" — a fresh Seed
    /// that *did* witness activity is diverted upstream by [`Engine::seed_owes_first_fire`] and
    /// never reaches this predicate.
    ///
    /// The boolean answer is per-Profile; the caller ([`Engine::classify_consequence`]) builds the
    /// SeedDrift fire filter from the Profile's fired Subs ([`specter_core::SubRegistry::fired_in`]).
    fn seed_drift_observed(&self, profile_id: ProfileId) -> bool {
        // Never fired ⇒ no prior emission to re-fire on recovery. The per-Sub flags live on the
        // registry (disjoint field from `profiles`); `any_fired` short-circuits on the first hit.
        if !self.subs.any_fired(profile_id) {
            return false;
        }
        let Some(p) = self.profiles.get(profile_id) else {
            return false;
        };
        let Some(curr) = p.current_hash() else {
            return false;
        };
        match p.settled_hash() {
            Some(settled) => settled != curr,
            None => false,
        }
    }

    /// Whether this Seed-Ok must fire a *first* time: a fresh, never-fired Profile that **witnessed
    /// activity** during the Seed window (`!dirty.is_empty()`). The discriminant for the three
    /// disjoint Seed-Ok consequences, split on the `any_fired` axis (and, for `!any_fired`, the
    /// witnessed-activity axis):
    ///
    /// 1. `!any_fired && !dirty.is_empty()` ⇒ **true** — a fresh Seed that saw events. Specter's
    ///    contract ("fire when the tree settles") owes a fire; the consequence is
    ///    [`Consequence::FreshSeedFire`] (no baseline yet ⇒ the fire's diff is `None`, and the
    ///    post-command rebase establishes the baseline).
    /// 2. `!any_fired && dirty.is_empty()` ⇒ false — a fresh Seed over a static tree (a daemon
    ///    restart; Specter persists no baseline). Restart-safe silent [`Consequence::SilentPin`].
    /// 3. `any_fired` ⇒ false — a recovery Seed: [`Consequence::RecoveryFire`]
    ///    ([`Self::seed_drift_observed`] re-fires the drifted Subs from the survival witness) or,
    ///    with no drift, the silent [`Consequence::SilentPin`].
    ///
    /// Mutually exclusive with [`Self::seed_drift_observed`] by construction (that predicate is
    /// `false` whenever `!any_fired`), so the fresh-first-fire and recovery-drift paths never
    /// overlap.
    fn seed_owes_first_fire(&self, profile_id: ProfileId) -> bool {
        if self.subs.any_fired(profile_id) {
            return false;
        }
        self.profiles
            .get(profile_id)
            .and_then(specter_core::Profile::pre_fire_burst)
            .is_some_and(|pre| !pre.dirty.is_empty())
    }

    /// Read back the in-flight Verifying probe's `target` — the pre-fire snapshot-commit resource —
    /// plus the Profile-presence guard. The shared up-front read for
    /// [`Engine::dispatch_quiescence_ok`].
    ///
    /// This is the **read-back twin** of the standalone target rule
    /// [`crate::burst::pre_fire_target`], not a second computation of it: that function *computes*
    /// the dirty-LCA target at [`Engine::transition_to_verifying`] and writes it onto
    /// [`specter_core::PreFirePhase::Verifying`]'s payload (immutable for the variant's lifetime);
    /// this method reads it back when the probe responds, so the snapshot grafts at the resource the
    /// probe was scoped to. The same value is also read back at the emission choke
    /// ([`Engine::probe_emission_request`]) to render the wire: computed once at the transition, read
    /// back wherever the probe's scope is needed. The `p.resource` fallback on the
    /// structurally-unreachable non-Verifying arm matches the historical `unwrap_or(anchor)`. `None`
    /// only on the structurally-unreachable absent-Profile path (the caller arms then return).
    ///
    /// Both fall-through arms `debug_assert!(false)` to surface a dispatch-contract violation in
    /// dev/CI and degrade silently in release: `dispatch_quiescence_ok` is reached only after
    /// [`Self::certify_probe_response`]'s entry guard proved the Profile present, and after
    /// `profile_probe_gate` proved its state is `Active(PreFire(Verifying))`.
    ///
    /// The covered-descendant fire-gate is **not** read here. It is a fire-only concern, so it
    /// lives at the single gate site ([`Engine::gated_fire`]) — evaluated fresh from the live tree
    /// only when a fire reaches it, and short-circuited entirely on a `forced` deadline-fire.
    /// Computing it here would re-introduce the "derived then discarded on the non-fire arms" shape
    /// this unification dissolves.
    fn verifying_probe_target(&self, profile_id: ProfileId) -> Option<ResourceId> {
        let Some(p) = self.profiles.get(profile_id) else {
            debug_assert!(
                false,
                "verifying_probe_target: absent Profile {profile_id:?} — \
                 certify_probe_response's entry guard proves presence at this depth",
            );
            return None;
        };
        let Some(pre) = p.pre_fire_burst() else {
            debug_assert!(
                false,
                "verifying_probe_target: non-PreFire Profile {profile_id:?} \
                 reached dispatch_quiescence_ok (profile_probe_gate dispatches \
                 Verifying only on Active(PreFire))",
            );
            return Some(p.resource());
        };
        Some(match &pre.phase {
            PreFirePhase::Verifying { target, .. } => *target,
            PreFirePhase::Batching { .. } | PreFirePhase::Draining => {
                debug_assert!(
                    false,
                    "verifying_probe_target: non-Verifying pre-fire phase on \
                     Profile {profile_id:?} reached dispatch_quiescence_ok \
                     (profile_probe_gate dispatches Verifying only on \
                     PreFirePhase::Verifying)",
                );
                p.resource()
            }
        })
    }

    /// Emit Effects for `mode`, seal the survival witness for the recovery consequence, then settle
    /// the burst: `Awaiting` when ≥1 Effect was pushed, else finish to Idle. The single home for
    /// the shared `emit_effects → count>0 ? awaiting : finish` triple — the one funnel every
    /// fireable Seed/Standard consequence reaches.
    ///
    /// **The seal is keyed on `EmitMode::SeedDrift`, not a separate discriminant.** Only a recovery
    /// drift-fire emits `SeedDrift`, so the mode *is* the witness-consumption signal:
    /// [`specter_core::Profile::rebase_baseline`] (`Witness → Snapshot`) runs before the
    /// `Awaiting`/finish split, regardless of `count`. This binds witness consumption to the
    /// certified-recovery decision — the two settle-spaced equal Seed-Ok reads — rather than
    /// deferring it into the post-fire phase machinery. It is load-bearing, not defensive: the
    /// post-fire rebase loop's "keep the prior baseline" terminals (a ceiling hit on an unreadable
    /// region; the Vanished / Failed cleanup) then need no per-intent witness reasoning, because by
    /// then the prior baseline is always a legitimate recovered `Snapshot`. Without the seal a
    /// recovery that hit such a terminal would finish in the loss window and re-fire on the next
    /// event. Sealing even when `count == 0` (every recovery Sub dedup-suppressed, or detached) is
    /// intentional and matches the prior pin exactly.
    fn fire_and_settle(
        &mut self,
        profile_id: ProfileId,
        mode: EmitMode<'_>,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let outcome = self.emit_effects(profile_id, mode, now, out);
        if matches!(mode, EmitMode::SeedDrift { .. })
            && let Some(p) = self.profiles.get_mut(profile_id)
        {
            p.rebase_baseline();
        }
        // The "fire emitted ≥1 Effect" test IS the `NonZeroU32` constructor: `Some` carries the
        // invariant into `transition_to_awaiting` as a type; the zero case finishes the burst
        // directly.
        match NonZeroU32::new(outcome.count) {
            Some(count) => self.transition_to_awaiting(profile_id, count, now, out),
            None => self.finish_burst_to_idle(profile_id, out),
        }
    }

    /// Emit [`Diagnostic::PerFileFireSkippedOnFreshSeed`] iff the Profile carries a `PerStableFile`
    /// Sub — the single home for the fresh-Seed per-file-skip narration. A fresh Profile has no
    /// baseline, so `emit_effects` builds no per-leaf diff and the per-file reactions have nothing
    /// to enumerate on the first fire (they begin from the post-command baseline). Called only on
    /// the genuine fresh-Seed fire — never the Draining-gated defer, which re-enters here on the
    /// reconfirm pass, so the note emits exactly once.
    fn warn_perfile_skipped_on_fresh_seed(&self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.subs.has_per_stable_file_sub(profile_id) {
            out.diagnostics
                .push(Diagnostic::PerFileFireSkippedOnFreshSeed {
                    profile: profile_id,
                });
        }
    }

    /// (Seed, Vanished).
    ///
    /// Symmetric with `dispatch_standard_vanished` (treats Vanished as an anchor-disappearance
    /// signal): routes through [`Self::finalize_anchor_lost_and_descend`], whose discard releases
    /// the anchor's `watch_demand` contribution so the trichotomy invariant in `reap_profile` —
    /// `!(Pending && AnchorClaim::Held)` — holds when the wrapper re-enters descent within this
    /// same step.
    ///
    /// Recovery does not depend on the anchor's FD: the kqueue registration auto-detached on the
    /// inode disappearing, and the wrapper's witnessed descent re-acquires it at materialization
    /// (`dispatch_descent_ok`'s anchor arm re-bumps `anchor.watch_demand` with the Profile's mask).
    fn dispatch_seed_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Seed,
        });
        self.finalize_anchor_lost_and_descend(profile_id, out);
    }

    /// (Seed, Failed).
    ///
    /// Symmetric with `dispatch_standard_failed`: the probe failed at the anchor; the coordinator
    /// releases the anchor's `watch_demand` contribution.
    ///
    /// Deliberately bare [`Self::finalize_anchor_lost`], **not** the descend wrapper: `Failed` is a
    /// read error against an anchor that typically still exists on disk (transient `EACCES`, a
    /// chmod-000 chain), so an immediate descent would loop probe → fail tightly against the same
    /// condition. The Profile parks Idle-anchorless and the event-scan arm
    /// ([`Self::start_pending_recovery`]) retries on the parent's next `StructureChanged`.
    fn dispatch_seed_failed(
        &mut self,
        profile_id: ProfileId,
        failure: ProbeFailure,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent: BurstIntent::Seed,
            failure,
        });
        self.finalize_anchor_lost(profile_id, out);
    }

    /// (Standard, Vanished).
    ///
    /// A Standard probe targets the dirty-LCA — possibly a *descendant*, not the anchor — so
    /// `Vanished` here means "the probed subtree root is gone", which an `rm -rf` racing the walk
    /// produces at a descendant while the anchor itself survives. Routing through
    /// [`Self::finalize_anchor_lost_and_descend`] resolves the ambiguity in one hop instead of
    /// trusting either reading: the witnessed descent re-probes the anchor's parent; a surviving
    /// anchor re-materializes into a triggered Seed (the `rm` *was* a change — the fire is owed), a
    /// genuinely-gone anchor parks as an honest `Pending` until it reappears.
    ///
    /// Standard bursts always run on materialized Profiles (`drive_burst` routes baseline-less
    /// `FsEvent`s to Seed instead), so the guard is effectively unconditional in v1 — kept for
    /// robustness against future routing changes.
    fn dispatch_standard_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent: BurstIntent::Standard,
        });
        self.finalize_anchor_lost_and_descend(profile_id, out);
    }

    /// (Standard, Failed).
    ///
    /// Bare [`Self::finalize_anchor_lost`] — unlike the `Vanished` sibling, no immediate descent;
    /// see `dispatch_seed_failed` for the tight-loop rationale.
    fn dispatch_standard_failed(
        &mut self,
        profile_id: ProfileId,
        failure: ProbeFailure,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent: BurstIntent::Standard,
            failure,
        });
        self.finalize_anchor_lost(profile_id, out);
    }

    /// (Rebase, Ok). The shared certifier folded the quiescence verdict over the *post-command* tree
    /// (events-reliable witness silence for CONTENT-subscribed Profiles, the `last_certified_hash`
    /// N=2 channel otherwise); this routine maps the post-fire `verdict` to a consequence. The
    /// Rebasing probe always targets the anchor (set by [`Engine::transition_to_rebasing`]).
    ///
    /// - [`QuiescenceVerdict::Stable`] — walker certified AND quiescence proven. The outer arm runs
    ///   the single commit-and-rebase prelude (`apply_snapshot` + `rebase_baseline`); the inner
    ///   [`StableReason`] selects what happens after the commit:
    ///   - [`StableReason::Natural`] — genuinely quiescent post-command tree (settle silence held,
    ///     or the hash channel agreed). Restart from the fire-tail residual or finish to Idle.
    ///   - [`StableReason::Forced`] — the bounded `RebaseCeiling` fired but the walker still
    ///     certified. Pin the freshest observation (a deliberate, loud terminal — not a wedge) and
    ///     emit one [`Diagnostic::RebaseCeilingForced`] carrying `observed_change =
    ///     hash_channel_disagreed`: `true` is the strong signal (the hash channel observed `prior !=
    ///     response` before the ceiling expired); `false` is the quiet ceiling (the channel agreed at
    ///     the last sample, was on its first sample, or was inactive on an events-reliable Profile).
    /// - [`QuiescenceVerdict::Retry`] — non-firing, non-terminal: the walker certified but the hash
    ///   channel observed `prior != Some(response)` (events-incomplete fire-bearing burst), or the
    ///   walker refused on some chain (transient non-observation — `EACCES`, a chmod-000 chain).
    ///   Never commit (the prior is the last walker-certified sample, not a quiescent one; an
    ///   unread region must not poison `current`); settle-space the next sample via
    ///   [`Engine::transition_to_settling`]. The bounded `RebaseCeiling` eventually surfaces a
    ///   `Stable(Forced)` / `Abandon` terminal if the failing condition persists.
    /// - [`QuiescenceVerdict::Abandon`] — ceiling reached on an unread response. Refuse to rebase
    ///   blind: surface [`Diagnostic::RebaseCeilingUnreadable`] and finish without committing — the
    ///   prior baseline stays in place. Safe to keep it on a recovery: the certified-recovery
    ///   decision already sealed `Witness → Snapshot` (the `EmitMode::SeedDrift` seal in
    ///   `fire_and_settle`), so by here the prior baseline is a legitimate recovered `Snapshot`,
    ///   not a loss-window witness — this terminal needs no per-intent witness reasoning, and the
    ///   next event routes Standard rather than re-firing the recovery.
    ///
    /// **Single commit-and-rebase prelude.** The shared `apply_snapshot` + `rebase_baseline` work
    /// for both `Stable(Natural)` and `Stable(Forced)` factors structurally into the outer
    /// `Stable(_)` arm — the inner match on `StableReason` selects the post-commit divergence
    /// (restart-or- finish vs. diagnose-and-finish). The non-`Stable` arms never commit, so the
    /// prelude is correctly scoped to `Stable(_)`.
    ///
    /// **Post-rebase residual.** On the natural Stable terminal a [`BurstFinish::ReturnToIdle`]
    /// burst with a non-empty fire-tail residual restarts a fresh debounced burst over the rebased
    /// baseline (`restart_burst_from_fire_tail_residual`) so a final-window change is not lost —
    /// origin-agnostic (a Seed-origin drift → fire → rebase restarts too: the reconfirm is a fresh
    /// query, not a per-origin refcount, so `into_pre_fire_residual` rejoins it to the Standard
    /// debounce lifecycle). An empty residual or a zombie `Reap` burst finishes to Idle. The
    /// ceiling, `Retry`, and `Abandon` terminals never restart (the loop is bounded by the rebase
    /// ceiling, not raced).
    ///
    /// **Why the verdict applies.** Kind agreement and the verdict fold are owned upstream by the
    /// shared certifier ([`Engine::certify_probe_response`]). The verdict is a pure projection of
    /// `(ProofAuthority, forced, QuiescenceWitness)` — independent of `current` / `baseline`, so
    /// acting on it is sound even though the fire just mutated those snapshots.
    fn dispatch_rebase_ok(
        &mut self,
        profile_id: ProfileId,
        snapshot: TreeSnapshot,
        verdict: QuiescenceVerdict,
        now: Instant,
        out: &mut StepOutput,
    ) {
        // Rebasing targets the anchor by construction (`transition_to_rebasing` always probes
        // `Profile.resource`; the post-fire side carries no probe target on its variant —
        // Rebasing's target is structurally fixed). Kind agreement and the verdict fold are owned
        // upstream by the shared certifier.
        let Some(target) = self.profiles.get(profile_id).map(Profile::resource) else {
            return;
        };

        match verdict {
            QuiescenceVerdict::Stable(reason) => {
                // Single commit-and-rebase prelude — shared by the Natural and Forced fire paths.
                // Both observe the freshest tree (Natural: the witness held; Forced: ceiling bypass
                // against the last sample), so the graft + baseline rebase land identically; only
                // the post-commit branch (restart vs. diagnose-and-finish) diverges on `reason`.
                self.apply_snapshot(profile_id, target, snapshot, out);
                if let Some(p) = self.profiles.get_mut(profile_id) {
                    p.rebase_baseline();
                }
                match reason {
                    StableReason::Natural => {
                        // Restart iff the final-window residual is non-empty AND the burst returns
                        // to Idle. Resolved under one read borrow; the bool carries no borrow out.
                        // Origin-agnostic (see `PostFireBurst::into_pre_fire_residual`).
                        let should_restart = match self
                            .profiles
                            .get(profile_id)
                            .map(specter_core::Profile::state)
                        {
                            Some(ProfileState::Active(ActiveBurst::PostFire(post), finish)) => {
                                !post.final_window_residual.is_empty()
                                    && matches!(finish, BurstFinish::ReturnToIdle)
                            }
                            _ => false,
                        };
                        if should_restart {
                            self.restart_burst_from_fire_tail_residual(profile_id, now, out);
                        } else {
                            self.finish_burst_to_idle(profile_id, out);
                        }
                    }
                    StableReason::Forced {
                        hash_channel_disagreed,
                    } => {
                        // Bounded terminal: the `RebaseCeiling` already fired but the walker
                        // certified anyway. Emit one diagnostic unconditionally — no `Effect`
                        // records the forced fallback downstream (the principled asymmetry with the
                        // pre-fire mirror) — carrying the disagreement bit as `observed_change`.
                        let intent = self.rebase_burst_intent(profile_id);
                        out.diagnostics.push(Diagnostic::RebaseCeilingForced {
                            profile: profile_id,
                            intent,
                            observed_change: hash_channel_disagreed,
                        });
                        self.finish_burst_to_idle(profile_id, out);
                    }
                }
            }

            QuiescenceVerdict::Retry => {
                // Two operationally-identical origins collapse here: the hash channel observed
                // `prior != Some(response)` (the post-command tree is moving under the rebase
                // loop), or the walker refused on some chain (transient non-observation). Never
                // commit (the prior carrier value is the last walker-certified sample, not a
                // quiescent one; an unread region must not poison `current`). Settle-space the next
                // sample via Rebasing → Settling; the `RebaseCeiling` (armed at the loop's start)
                // eventually surfaces the operator-visible terminal.
                self.transition_to_settling(profile_id, now, out);
            }

            QuiescenceVerdict::Abandon { first_unread } => {
                // Ceiling reached on an unread response: refuse to rebase blind. No commit, no
                // rebase — the prior baseline stays in place.
                let intent = self.rebase_burst_intent(profile_id);
                out.diagnostics.push(Diagnostic::RebaseCeilingUnreadable {
                    profile: profile_id,
                    first_unread,
                    intent,
                });
                self.finish_burst_to_idle(profile_id, out);
            }
        }
    }

    /// (Rebase, Vanished). Anchor disappeared between fire and rebase. Symmetric path with
    /// `dispatch_standard_vanished`: route through [`Self::finalize_anchor_lost_and_descend`]. A
    /// command whose effect atomically replaces its own anchor lands here and re-fires once per
    /// settle window until its output stabilizes (witness equality / B1 terminate idempotent
    /// commands after one extra cycle) — the same semantics the anchor-terminal route already
    /// carries for self-replacing commands. Diagnostic carries the burst's actual intent so logs
    /// can distinguish Seed-driven (drift) vs Standard-driven Rebasing; the lookup falls back to
    /// `Standard` only on a stale-Profile or non-Active defensive path (the routing in
    /// `on_probe_response` guarantees `Active(Rebasing)` at entry).
    fn dispatch_rebase_vanished(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        // Read intent BEFORE the loss wrapper. The coordinator does not mutate Burst.intent (its
        // finish_burst_to_idle flips Active → Idle but never rewrites the burst payload), so the
        // read is order-insensitive in v1; pinning it before the call guards against future
        // coordinator changes that might touch state.
        let intent = self.rebase_burst_intent(profile_id);
        out.diagnostics.push(Diagnostic::ProbeVanished {
            profile: profile_id,
            intent,
        });
        self.finalize_anchor_lost_and_descend(profile_id, out);
    }

    /// (Rebase, Failed). Probe failed at the anchor between fire and rebase. Narrate, then bare
    /// [`Self::finalize_anchor_lost`] — unlike the `Vanished` sibling, no immediate descent; see
    /// `dispatch_seed_failed` for the tight-loop rationale. Diagnostic carries the burst's actual
    /// intent (Standard fallback on the same defensive path noted on the `Vanished` sibling; read
    /// before the call for the same reason).
    fn dispatch_rebase_failed(
        &mut self,
        profile_id: ProfileId,
        failure: ProbeFailure,
        out: &mut StepOutput,
    ) {
        if self.profiles.get(profile_id).is_none() {
            return;
        }
        let intent = self.rebase_burst_intent(profile_id);
        out.diagnostics.push(Diagnostic::ProbeFailed {
            profile: profile_id,
            intent,
            failure,
        });
        self.finalize_anchor_lost(profile_id, out);
    }

    /// Resolve the intent of the burst owning the in-flight Rebase probe. Returns
    /// `PostFireBurst.intent` on the production path — the only path `profile_probe_gate` ⇒
    /// `take_owner_probe` reaches the `dispatch_rebase_*` callers from
    /// (`Active(PostFire(Rebasing))`).
    ///
    /// Every other arm `debug_assert!(false)`s a dispatch-contract violation and degrades to a safe
    /// default in release: PreFire keeps `pre.intent` (most accurate residual), and
    /// absent/Idle/Pending fall back to [`BurstIntent::Standard`] (Rebasing is overwhelmingly a
    /// Standard-burst tail; Seed-driven Rebasing requires a recovery + drift, the rare path).
    fn rebase_burst_intent(&self, profile_id: ProfileId) -> BurstIntent {
        let Some(profile) = self.profiles.get(profile_id) else {
            debug_assert!(
                false,
                "rebase_burst_intent: absent Profile {profile_id:?} — \
                 certify_probe_response's entry guard proves presence at this depth",
            );
            return BurstIntent::Standard;
        };
        match profile.state() {
            ProfileState::Active(ActiveBurst::PostFire(post), _) => post.intent,
            ProfileState::Active(ActiveBurst::PreFire(pre), _) => {
                debug_assert!(
                    false,
                    "rebase_burst_intent: PreFire Profile {profile_id:?} reached \
                     dispatch_rebase_* (profile_probe_gate dispatches Rebasing only on \
                     Active(PostFire(Rebasing)))",
                );
                pre.intent
            }
            ProfileState::Idle | ProfileState::Pending(_) => {
                debug_assert!(
                    false,
                    "rebase_burst_intent: non-Active Profile {profile_id:?} \
                     reached dispatch_rebase_* (profile_probe_gate dispatches Rebasing only \
                     on Active(PostFire(Rebasing)))",
                );
                BurstIntent::Standard
            }
        }
    }

    /// `burst_deadline` row — sets `forced := true` and either transitions the phase
    /// (Batching/Draining → Verifying) or, if a probe is already in flight (Verifying), waits for
    /// the response.
    ///
    /// The `forced` write is delegated to [`Engine::force_pending`] (the single-source
    /// `PreFireBurst.forced` mutator); the phase-classification — whether to drive a verify now —
    /// stays here as a routing query, not a mutation. The caller is reached only through
    /// `is_timer_referenced`, which returns false for `BurstDeadline` in `Awaiting` / `Rebasing`,
    /// so only pre-fire phases arrive and the structurally-unreachable non-pre-fire re-read folds
    /// to "no verify" — a silent no-op preserving the prior inline `else { return; }`.
    fn handle_burst_deadline(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        // "burst-deadline elapsed ⇒ forced fire on next emission" is the first action; the phase then
        // decides whether that emission is driven now (Batching/Draining — no probe in flight) or by
        // the in-flight verify's response (Verifying), which dispatches with `forced` observed.
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

    /// `gate_deadline` row — actuator-hang recovery. Force-transitions the burst from `Awaiting`
    /// directly to `Rebasing`, skipping the `Settling` debounce window: the actuator has already hung
    /// for 4× max_settle, so the bounded wait is spent. Raises `forced` via
    /// [`Engine::force_pending_post_fire`] (the symmetric mirror of pre-fire's `handle_burst_deadline
    /// → force_pending`), then drives [`Engine::transition_to_rebasing`] for the final walk.
    ///
    /// The next probe response folds through `quiescence_verdict (authority, forced=true)`: an
    /// `Authoritative` certifies and commits on the first walk (the
    /// [`QuiescenceVerdict::Stable`]([`StableReason::Forced`]) arm), an `Undischarged` folds to
    /// [`QuiescenceVerdict::Abandon`] and surfaces `RebaseCeilingUnreadable` before finishing. No
    /// ceiling timer is armed — the loop has no second sample to bound against.
    ///
    /// Late `EffectComplete` arrivals (after this transition) land in
    /// [`Diagnostic::EffectCompleteOutsideAwaiting`].
    ///
    /// **Zombie burst short-circuit.** A burst carrying [`BurstFinish::Reap`] has no consumer for
    /// the rebased baseline — its Profile is dying. Skip the rebase probe entirely and route
    /// straight through `finish_burst_to_idle`, which runs the Draining-sweep reconfirm and then
    /// dispatches `reap_profile`. The diagnostic still fires so operators see the actuator-hang
    /// signal; only the wasted rebase round-trip is elided.
    ///
    /// Defensive: if the phase has already advanced (e.g., a race with `finalize_anchor_lost`), the
    /// helper no-ops. The `is_timer_referenced` gate already filters most non-Awaiting fires; this
    /// guard handles the residual same-step ordering window.
    ///
    /// The `Awaiting.outstanding` access below is a diagnostic-only *read*; the field's sole writer
    /// is `Profile::note_effect_completion`.
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

        // Engine→actuator effect-cancel emission — the single abandonment site, structural dual of
        // `cancel_owner_probe` for probes. Emitted *before* the phase change so the actuator sees
        // the cancel ahead of any rebase probe response that could (in a future cross-step
        // sequence) trigger a `restart_burst_from_fire_tail_residual` and re-submit effects for the
        // same profile. The actuator's `handle_cancel` SIGTERMs in-flight children for this profile
        // and drops queued work; the wait threads still drive natural reap, and the engine routes
        // the late `EffectComplete` to `EffectCompleteOutsideAwaiting` (zombie case routes to
        // `EffectCompleteForUnknownSub`). Same emission shape for both zombie and force-rebasing —
        // the OS resources held by hung children must be released regardless of whether the Profile
        // has a consumer for the rebased baseline.
        out.push_cancel_effect(profile_id);

        out.diagnostics.push(if zombie {
            Diagnostic::AwaitGateDeadlineReap {
                profile: profile_id,
                outstanding,
            }
        } else {
            Diagnostic::AwaitGateDeadlineForceRebasing {
                profile: profile_id,
                outstanding,
            }
        });
        if zombie {
            self.finish_burst_to_idle(profile_id, out);
        } else {
            // Symmetric mirror of pre-fire's `handle_burst_deadline → force_pending → drive
            // Verifying now`. Shares the `Awaiting → Rebasing` edge with the natural completion
            // path (`on_effect_complete`) but takes it *forced*: no follow-up timer is scheduled —
            // the `forced` bit drives the next probe response to a commit terminal in one walk.
            self.force_pending_post_fire(profile_id);
            self.transition_to_rebasing(profile_id, out);
        }
    }

    /// `PostFireSettle` row — the HashChannel re-sample spacing expiry (the only surviving
    /// post-fire `Settling` window; the natural rebase entry is probe-first, `Awaiting →
    /// Rebasing`). The symmetric mirror of [`Engine::on_settle_expired`] on the pre-fire side,
    /// including the reschedule fork.
    ///
    /// **Reschedule path**: `now − last_event_time < settle`. `absorb_event_into_fire_tail` updated
    /// `last_event_time` after the settle timer was scheduled; the quiet window is not yet closed.
    /// Schedules a fresh `TimerKind::PostFireSettle` at `last_event_time + settle` and routes the
    /// new id through [`Engine::reschedule_settling`] (the single-source phase mutator) — the old
    /// (just-expired) id is no longer referenced and lazily drops on a subsequent `pop_expired`.
    /// The phase stays `Settling`.
    ///
    /// **Transition path**: `now − last_event_time ≥ settle`. Forwards to
    /// [`Engine::transition_to_rebasing`] for the next sample.
    ///
    /// **Structurally unreachable: `last_event_time = None` on a `Settling` expiry.** The sole
    /// `Settling` entry (`Rebasing → Settling` via [`Engine::transition_to_settling`]) pins
    /// `Some(now)`. The match's `None` arm is therefore unreachable in production; it carries
    /// `debug_assert!(false)` + the safe transition default (the pre-fire mirror at
    /// `on_settle_expired`).
    ///
    /// **Preconditions** (guaranteed by [`is_timer_referenced`]
    /// upstream): `Profile.state == Active(PostFire(Settling {
    /// settle_timer == popped_id }))`. The defensive early returns below cover direct
    /// `step(Input::TimerExpired)` calls that bypass `pop_expired`.
    fn handle_post_fire_settle_expired(
        &mut self,
        profile_id: ProfileId,
        now: Instant,
        out: &mut StepOutput,
    ) {
        let Some(p) = self.profiles.get(profile_id) else {
            return;
        };
        let ProfileState::Active(ActiveBurst::PostFire(post), _) = p.state() else {
            return;
        };
        if !matches!(post.phase, PostFirePhase::Settling { .. }) {
            return;
        }
        let settle = p.settle;

        // saturating_duration_since handles `now < last` (test mockclock rewind / non-monotonic
        // clocks): returns Duration::ZERO, which satisfies `< settle` and triggers a reschedule.
        // Safe under any clock skew the harness can produce.
        match post.last_event_time {
            Some(last) if now.saturating_duration_since(last) < settle => {
                let new_deadline = last + settle;
                let new_timer =
                    self.timers
                        .schedule(new_deadline, profile_id, TimerKind::PostFireSettle);
                self.reschedule_settling(profile_id, new_timer);
            }
            Some(_) => self.transition_to_rebasing(profile_id, out),
            None => {
                debug_assert!(
                    false,
                    "handle_post_fire_settle_expired: last_event_time = None on \
                     Settling expiry for Profile {profile_id:?} — \
                     transition_to_settling pins Some(now) at the sole Settling \
                     entry; reaching here means a future writer opened the \
                     unreachable arm",
                );
                self.transition_to_rebasing(profile_id, out);
            }
        }
    }

    /// `RebaseCeiling` row — the rebase loop's bound, the forced-mirror of
    /// [`Engine::handle_burst_deadline`]. Latches [`specter_core::CeilingState::Reached`] via
    /// [`Engine::force_pending_post_fire`] (the single-source [`specter_core::CeilingState::Armed`]
    /// → `Reached` writer), then mirrors `handle_burst_deadline`'s phase routing exactly: in
    /// `Settling` no probe is in flight (the `Batching` analogue — the HashChannel re-sample
    /// spacing window, now the only post-fire `Settling`), so drive the final sample *now* via
    /// [`Engine::transition_to_rebasing`]; in `Rebasing` a probe is already in flight (the
    /// `Verifying` analogue), so set-only — its response carries the terminal. `Awaiting` is
    /// unreachable (the ceiling is armed only at the natural `Awaiting → Rebasing` entry, and the
    /// burst leaves `Awaiting` in that same step) and folds to the no-op default, as does a
    /// vanished Profile.
    fn handle_rebase_ceiling(&mut self, profile_id: ProfileId, out: &mut StepOutput) {
        // Single-source latch: `post.ceiling = CeilingState::Reached`, collapsing the prior two-field
        // `forced = true; rebase_ceiling = None` lockstep into one write. `is_timer_referenced` only
        // routes `RebaseCeiling` here while `Armed`, so the live timer entry we just popped is
        // dropped from the phase reference in the same move that latches the terminal.
        self.force_pending_post_fire(profile_id);

        // Mirror `handle_burst_deadline`: drive the final sample now iff no probe is in flight
        // (`Settling` — the `Batching` analogue). In `Rebasing` the in-flight response applies the
        // terminal via `dispatch_rebase_ok`'s `Stable(StableReason::Forced)` / `Abandon` arms.
        let needs_rebase = self
            .profiles
            .get(profile_id)
            .and_then(|p| match p.state() {
                ProfileState::Active(ActiveBurst::PostFire(post), _) => {
                    Some(matches!(&post.phase, PostFirePhase::Settling { .. }))
                }
                _ => None,
            })
            .unwrap_or(false);
        if needs_rebase {
            self.transition_to_rebasing(profile_id, out);
        }
    }

    /// Emit Effects at a stable verdict. Routes per scope: `SubtreeRoot` Subs fire one Effect
    /// anchored at the Profile's resource; `PerStableFile` Subs fire one Effect per matching diff
    /// entry. The `Diff` is built at most once and shared across both helpers via `Arc`.
    ///
    /// `mode` ([`EmitMode`]) selects the fire mode — Standard burst stable verdict vs Seed-drift
    /// fire — and carries the per-mode configuration (Standard's `forced`; SeedDrift's pre-narrowed
    /// `drifted` key set). The variant determines:
    ///
    /// - which `SubtreeRoot` Subs fire (Standard: all; SeedDrift: only those whose
    ///   `DedupKey::Subtree` is in `drifted`),
    /// - whether dedup-hash suppression applies (Standard: yes unless `forced`; SeedDrift:
    ///   structurally unreachable),
    /// - whether `PerStableFile` Subs fire (Standard: yes; SeedDrift: skipped — Seed-time drift is
    ///   Subtree-only), and
    /// - the [`Effect::forced`] value carried into the spawned process.
    ///
    /// A burst flagged [`BurstFinish::Reap`] suppresses all emission — the Profile is on its way
    /// out (its last Sub detached mid-burst) and any Effect would fire against a Sub registry that
    /// no longer holds the Subs.
    ///
    /// Returns an [`EmitOutcome`] whose `count` is the number of Effects appended to `out`. Callers
    /// consume this to decide whether to enter the `Awaiting` phase (`count > 0`) or short-circuit
    /// to `finish_burst_to_idle` (dedup-hash suppressed everything, no Subs matched, or the burst
    /// is flagged [`BurstFinish::Reap`]).
    ///
    /// **Per-Sub observational bookkeeping.** Each emission triggers
    /// [`specter_core::SubRegistry::record_fired`] (bumps `fire_count`, stamps `last_fired_at =
    /// now`) and pushes one [`Diagnostic::SubFired`] carrying the aggregated per-pass count (1 for
    /// SubtreeRoot, the per-leaf count for PerStableFile). A `FireVerdict::SuppressDedup` verdict
    /// instead bumps `dedup_suppressed_count` and emits nothing. The B1-dedup-load-bearing
    /// [`specter_core::SubRegistry::mark_fired`] stays the SubtreeRoot edge — separate signal,
    /// separate writer.
    fn emit_effects(
        &mut self,
        profile_id: ProfileId,
        mode: EmitMode<'_>,
        now: Instant,
        out: &mut StepOutput,
    ) -> EmitOutcome {
        let Some(p) = self.profiles.get(profile_id) else {
            return EmitOutcome::default();
        };
        // Burst carrying `BurstFinish::Reap` is on its way out. Any remaining Subs (none, by
        // construction of the directive's writers) would fire against a Sub registry that no longer
        // holds them — suppress emission entirely.
        if matches!(p.state().burst_finish(), Some(BurstFinish::Reap)) {
            return EmitOutcome::default();
        }
        let resource = p.resource();
        let baseline_snap = p.baseline();
        let current_snap = p.current();
        // Read the cached anchor classification. `None` falls back to `Dir` — the actuator's
        // `compute_cwd` then anchors at the path itself; if the actuator's later `chdir` discovers
        // the path doesn't behave as a directory, the Effect surfaces `EffectOutcome::Failed`.
        // Reaching `None` here implies a fresh resource-based attach whose Seed probe hasn't
        // returned — `dispatch_quiescence_ok`'s fallback writes the field on the next Seed-Ok.
        let anchor_kind = p.kind().unwrap_or(ResourceKind::Dir);
        // Substitution-side projection of `ScanConfig.exclude`. The resolver iterates source strings;
        // the sensor consults compiled matchers. The order is the exclude list's build-time
        // canonical form — `ScanConfigBuilder::build` sorts and dedups — which `Profile::new` copies
        // out verbatim, so the projection is already canonical without a re-sort here.
        let exclude_strings = Arc::clone(p.exclude_strings());

        let anchor_path: Arc<Path> = self.tree.path_of(resource).unwrap_or_else(empty_path);

        // Lazy-build the Diff Arc only if any Sub needs it AND both a baseline and a current
        // snapshot are present. With baseline pinned across coalesced bursts, `Effect.diff`
        // describes the *net* change since the last EffectComplete::Ok.
        let mut diff_arc: Option<Arc<specter_core::Diff>> = None;
        let ensure_diff = |diff_slot: &mut Option<Arc<specter_core::Diff>>| {
            if diff_slot.is_none()
                && let (Some(b), Some(c)) = (baseline_snap.as_ref(), current_snap.as_ref())
            {
                *diff_slot = Some(Arc::new(specter_core::diff_tree(b, c)));
            }
            diff_slot.clone()
        };

        // Per-Profile structural component of B1 dedup. The full Subtree suppress decision combines
        // `nothing_changed` with the per-Sub `has_fired` flag (read once below, alongside scope /
        // needs_diff / log_output, in the loop's single `subs.get`): a Sub that has never fired
        // suppresses nothing — it is its own "first emission" — even when the tree happens to match.
        let nothing_changed = p
            .baseline_hash()
            .zip(p.current_hash())
            .is_some_and(|(b, c)| b == c);

        let effect_forced = mode.effect_forced();

        // Snapshot the Sub IDs to avoid holding `&self.subs` across the loop body's
        // `out.push_effect`.
        let sub_ids: Vec<SubId> = self.subs.at(profile_id).to_vec();
        let mut count: u32 = 0;
        for sub_id in sub_ids {
            let (scope, needs_diff, log_output, already_fired) = match self.subs.get(sub_id) {
                Some(s) => match s.reaction() {
                    Reaction::Spawn { spec, history, .. } => (
                        spec.scope(),
                        spec.needs_diff(),
                        spec.log_output(),
                        history.has_fired,
                    ),
                    // A Mint Sub never fires — discovery's "fire" is the reconcile's batch of
                    // attachments, and the verdict floor classifies its Profile to Reconcile before
                    // the fire fork. Reaching here is a routing breach; the skip is the
                    // semantically correct degrade (a Mint must not fire), loud in dev so the
                    // misrouting dispatcher gets fixed, never daemon-fatal in release.
                    Reaction::Mint(_) => {
                        debug_assert!(
                            false,
                            "emit_effects reached a Mint Sub \
                             (sub = {sub_id:?}, profile = {profile_id:?})",
                        );
                        continue;
                    }
                },
                None => continue,
            };
            match fire_decision(mode, scope, sub_id, already_fired, nothing_changed) {
                FireVerdict::SkipScope => continue,
                FireVerdict::SuppressDedup => {
                    // Observational only: count the B1-dedup-suppressed verdict so the
                    // operator-facing `list --wide` surfaces how often a Sub's reaction *would*
                    // have fired against an unchanged tree.
                    self.subs.record_dedup_suppressed(sub_id);
                    continue;
                }
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
                    // The funnel head classified this Sub as Spawn; a None here would mean the
                    // reaction changed under a live borrow-free window, which rebind cannot do
                    // (variant-typed Spawn↔Spawn) — the silent skip mirrors the stale-id arm.
                    let Some(spec) = sub.spawn_spec() else {
                        continue;
                    };
                    out.push_effect(Effect::subtree(
                        EffectCommon {
                            sub: sub_id,
                            profile: profile_id,
                            // `resource` was captured at the function head from `Profile.resource`;
                            // frozen at emit so the sort survives post-emit churn without a
                            // ProfileMap lookup.
                            anchor: resource,
                            correlation,
                            forced: effect_forced,
                            capture_output: log_output,
                            sub_name: sub.name.clone(),
                            program: Arc::clone(spec.program()),
                            anchor_path: Arc::clone(&anchor_path),
                            anchor_kind,
                            exclude: Arc::clone(&exclude_strings),
                        },
                        diff_for_effect,
                    ));
                    count = count.saturating_add(1);

                    // Record the per-Sub fire (the `sub` borrow above ended with `push_effect`;
                    // `&mut self.subs` is free). `mark_fired` is the load-bearing B1-dedup edge;
                    // `record_fired` is the observational counter pair that drives the
                    // operator-facing `list` projection.
                    self.subs.mark_fired(sub_id);
                    self.subs.record_fired(sub_id, 1, now);
                    out.diagnostics.push(Diagnostic::SubFired {
                        sub: sub_id,
                        profile: profile_id,
                        count: 1,
                    });
                }
                EffectScope::PerStableFile => {
                    // PerStableFile implies `needs_diff = true` at Sub::from_request; diff is
                    // always built.
                    let Some(diff) = ensure_diff(&mut diff_arc) else {
                        continue;
                    };
                    let pushed = self.emit_effects_per_stable_file(
                        sub_id,
                        resource,
                        effect_forced,
                        &diff,
                        &anchor_path,
                        anchor_kind,
                        &exclude_strings,
                        out,
                    );
                    if pushed > 0 {
                        // Aggregated: one `SubFired` + one `record_fired(pushed)` per pass,
                        // regardless of how many leaf files matched. Keeps the wire stream linear
                        // in Sub count, not in diff size — the per-leaf Effects themselves carry
                        // the per-file granularity downstream.
                        self.subs.record_fired(sub_id, pushed, now);
                        out.diagnostics.push(Diagnostic::SubFired {
                            sub: sub_id,
                            profile: profile_id,
                            count: pushed,
                        });
                    }
                    count = count.saturating_add(pushed);
                }
            }
        }
        EmitOutcome { count }
    }

    /// Per-Diff-entry Effect emission for a `PerStableFile` Sub. Walks `created`, `modified`, and
    /// `renamed.to`; deleted entries do **not** fire (running a per-file command on a deleted file
    /// makes no sense).
    ///
    /// Resource materialization: the diff entry's slot is resolved via `reconcile`'s
    /// `lookup_descendant`-style walk; if the slot isn't yet in the Tree (defensive — reconcile
    /// runs before this and materializes covered entries), a fresh Resource is created with no
    /// `watch_demand` contribution.
    ///
    /// Returns the number of Effects appended to `out`. The caller (`emit_effects`) sums this into
    /// the [`EmitOutcome`]'s `count` it returns.
    #[must_use]
    fn emit_effects_per_stable_file(
        &mut self,
        sub_id: SubId,
        anchor: ResourceId,
        forced: bool,
        diff: &Arc<specter_core::Diff>,
        anchor_path: &Arc<Path>,
        anchor_kind: ResourceKind,
        exclude_strings: &Arc<[CompactString]>,
        out: &mut StepOutput,
    ) -> u32 {
        let profile_id = match self.subs.get(sub_id) {
            Some(s) => s.profile(),
            None => return 0,
        };
        let mut count: u32 = 0;

        // Collect matching segments + kinds in a single pass, in the order expected — created, then
        // modified, then renamed.to.
        let entries = diff
            .created
            .iter()
            .chain(diff.modified.iter())
            .chain(diff.renamed.iter().map(|r| &r.to));

        for entry in entries {
            // PerStableFile is per-FILE: skip Dir and Other (devices / sockets / fifos) entirely —
            // running a per-file command on a directory or device is never the user's intent.
            // Symlinks pass through (they target files in practice).
            if !matches!(
                entry.kind,
                specter_core::EntryKind::File | specter_core::EntryKind::Symlink
            ) {
                continue;
            }
            // `graft`'s `apply_diff_to_tree` runs before this and materialises every covered diff
            // entry; lookup is the happy path. Fall back to `ensure_descendant` for defense — covers
            // the rare case where reconcile filtered the entry (e.g., reconcile gates Watch on Dir,
            // not on every leaf the Sub can fire against).
            let resource = match lookup_descendant(&self.tree, anchor, entry.segment.as_str()) {
                Some(r) => r,
                None => match ensure_descendant(
                    &mut self.tree,
                    anchor,
                    entry.segment.as_str(),
                    entry.kind.into(),
                ) {
                    Some(r) => r,
                    None => continue,
                },
            };

            let correlation = self.effect_correlations.next();
            // The Sub may have been removed mid-burst; defensive lookup. The spawn-spec read shares
            // the same silent-skip discipline — the funnel's PerStableFile dispatch already proved
            // the variant.
            let Some(sub) = self.subs.get(sub_id) else {
                continue;
            };
            let Some(spec) = sub.spawn_spec() else {
                continue;
            };
            out.push_effect(Effect::per_file(
                EffectCommon {
                    sub: sub_id,
                    profile: profile_id,
                    anchor,
                    correlation,
                    forced,
                    capture_output: spec.log_output(),
                    sub_name: sub.name.clone(),
                    program: Arc::clone(spec.program()),
                    anchor_path: Arc::clone(anchor_path),
                    anchor_kind,
                    exclude: Arc::clone(exclude_strings),
                },
                resource,
                entry.segment.clone(),
                diff.clone(),
            ));
            count = count.saturating_add(1);
            // PerFile records no fire history — the per-file dedup is diff membership itself, not a
            // recorded key.
        }
        count
    }

    /// Single-pass classification of owners that carry a dispatch responsibility for an
    /// [`specter_core::Input::FsEvent`] at `resource`. Sole consumer is [`Engine::on_fs_event`].
    ///
    /// Two carrier classes:
    ///
    /// - **Descent** ([`ProfileId`]): Profiles currently descending whose
    ///   `DescentState.current_prefix() == resource` (`ProfileState::Pending(d)`). Each descent
    ///   owner gets a fresh probe via [`Engine::on_descent_event`].
    /// - **Recovery** ([`ProfileId`]): `Idle` Profiles whose `watch_root_parent == Some(resource)`
    ///   and whose anchor is currently absent (`current.is_none()`).
    ///   [`Engine::start_pending_recovery`] re-enters pending descent. This is the probe-`Failed` /
    ///   watch-rejection fallback only: an *observed* loss re-enters descent inside the loss step
    ///   itself (`finalize_anchor_lost_and_descend`) and is `Pending` — the descents arm — before
    ///   any later parent event can select it here.
    ///
    /// **O(1) carrier gate.** The scan body is O(profiles), but under a sustained storm every
    /// Profile is in a steady `Active` burst, so it iterates the full registry only to return
    /// empty. The registry maintains a `nonsteady` count of the carrier-*eligible* owners
    /// ([`Profile::is_nonsteady`]); when it is zero no carrier of either class can exist, so the
    /// scan is provably empty and skipped in O(1) — the keeps-up-storm win an operator feels as the
    /// daemon no longer pegging a core during a build.
    ///
    /// The count is over a pure state(+anchor) bucket, deliberately *not* the per-resource index a
    /// naïve reading invites. The recovery predicate couples multiple fields (`state` +
    /// `watch_root_parent` + anchor presence), and [`Profile::materialize_anchor`] writes `state`
    /// outside the [`specter_core::ProfileMap::transition_state`] chokepoint — a state-keyed index
    /// silently desyncs at that bypass. The bucket instead over-approximates to a single-field-ish
    /// predicate that is invariant under the bypass by construction (`Pending` and anchorless
    /// `Idle` are the same counted bucket) and sound (every true carrier is counted), so a zero
    /// gate is never a false skip; it is also *tight* — a healthy anchored `Idle` Profile is
    /// excluded, so a quiet watcher coexisting with a storm does not defeat the gate. A
    /// `#[cfg(debug_assertions)]` full recount tripwire below pins each maintained count every
    /// call; release pays only the O(1) compare.
    fn classify_event_carriers(&self, resource: ResourceId) -> EventCarriers {
        #[cfg(debug_assertions)]
        {
            debug_assert_eq!(
                self.profiles.nonsteady(),
                self.profiles
                    .iter()
                    .filter(|(_, p)| p.is_nonsteady())
                    .count(),
                "ProfileMap.nonsteady desynced from a full carrier recount",
            );
        }
        if self.profiles.nonsteady() == 0 {
            return EventCarriers::empty();
        }
        let mut out = EventCarriers::empty();
        for (pid, p) in self.profiles.iter() {
            match p.state() {
                ProfileState::Pending(d) if d.current_prefix() == resource => {
                    out.descents.push(pid);
                }
                ProfileState::Idle
                    if p.watch_root_parent() == Some(resource) && !p.current_is_some() =>
                {
                    out.recoveries.push(pid);
                }
                ProfileState::Pending(_) | ProfileState::Idle | ProfileState::Active(_, _) => {}
            }
        }
        out
    }
}

/// Per-resource dispatch fan-out collected by [`Engine::classify_event_carriers`]. The SmallVec
/// inline caps of 2 cover the typical "shared scaffold" case (two Subs anchored at sibling children
/// of one parent) without a heap allocation.
///
/// `descents` (via `current_prefix`) and `recoveries` (via `watch_root_parent`) are honest parallel
/// fields: the entry helpers genuinely differ ([`Engine::on_descent_event`] re-probes a live
/// descent; `start_pending_recovery` asserts an `Idle` Profile and re-enters descent), so a merged
/// list would only force a match-dispatch back into the two distinct helpers. After an observed
/// loss the same Profile moves from the recoveries class to the descents class within the loss step
/// itself — only probe-`Failed` / watch-rejection Profiles linger in the recoveries class.
struct EventCarriers {
    descents: SmallVec<[ProfileId; 2]>,
    recoveries: SmallVec<[ProfileId; 2]>,
}

impl EventCarriers {
    /// The no-carrier value: the O(1) carrier-gate return and the seed the scan pushes into. Both
    /// `SmallVec`s start inline-empty, no allocation.
    fn empty() -> Self {
        Self {
            descents: SmallVec::new(),
            recoveries: SmallVec::new(),
        }
    }
}

/// Outcome of an [`Engine::emit_effects`] call. `count` is the number of `out.push_effect(...)`
/// invocations that survived dedup-hash suppression and Sub-scope routing — i.e., Effects that the
/// Actuator will actually run.
///
/// `dispatch_*_ok` consumes this to decide whether the Profile should enter the `Awaiting` phase
/// (count > 0, at least one Effect is in flight) or short-circuit to `finish_burst_to_idle` (count
/// == 0: dedup-hash suppressed every emission, no Subs matched, or `reap_pending` was set). The
/// `#[must_use]` attribute prevents a future caller from silently dropping the count and
/// re-introducing the post-emit "Idle-but-Effects-in-flight" leakage.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
#[must_use]
pub(crate) struct EmitOutcome {
    pub count: u32,
}

/// Fire-mode for [`Engine::emit_effects`]. Captures the structural distinction between Standard
/// burst stable-verdict emission and Seed-drift emission, replacing the prior `(forced: bool,
/// drift_filter: Option<&[DedupKey]>)` parameter pair where the interaction between the two flags
/// was load-bearing but unmodelled.
///
/// The two modes differ along three axes that all fall out of the variant — no separate field
/// discipline:
///
/// - **Subtree key gating.** Standard fires every `SubtreeRoot` Sub on the Profile (modulo the
///   suppress check). SeedDrift fires only the Subs in `drifted` (one [`SubId`] per drifted
///   Subtree-keyed Sub).
/// - **Suppress.** Standard honours dedup-hash suppression unless `forced` is set. SeedDrift's
///   `drifted` is built from keys where `last_emitted ≠ current` by construction, so suppression is
///   structurally unreachable on this mode — `fire_decision`'s SeedDrift arm yields `Emit` directly
///   (no analytical claim, just a variant arm).
/// - **PerStableFile.** Standard emits `PerStableFile` Effects per matching diff entry. SeedDrift
///   skips PerFile entirely — the Seed-time drift signal is Subtree-only (per
///   [`Engine::seed_drift_observed`]'s documented limitation: a post-Seed `current` lacks the
///   per-leaf history needed for a faithful per-file diff). On a witness-bearing loss→recovery Seed
///   this skip drops the `PerStableFile` Sub's loss-window reactions; that (witness-gated) drop is
///   surfaced via [`Diagnostic::PerFileDriftDroppedOnRecovery`]. A plain `Input::SensorOverflow`
///   reseed of a `Snapshot`-baseline Profile drops them the same way but carries no witness, so it
///   is a further v1 limitation the diagnostic does not cover.
///
/// **Payload type.** `drifted: &[SubId]` rather than `&[DedupKey]`. By construction the slice
/// carries only `DedupKey::Subtree { sub, profile }` entries whose `profile == profile_id` (the
/// focal Profile); projecting to `SubId` upstream drops the redundant profile field AND removes the
/// variant-ambiguity (a `DedupKey::PerFile` cannot be represented in `&[SubId]`). The SeedDrift
/// Subtree-arm filter becomes `drifted.contains(&sub_id)` — same cost class as `contains(&dk)`,
/// stronger type contract.
///
/// [`Effect::forced`] is derived from the variant via [`Self::effect_forced`]: `true` only on
/// `Standard { forced: true }`. SeedDrift always emits with `forced = false` — the engine reached a
/// stable verdict; drift is the trigger, not a time-pressured force-fire. Conflating the two would
/// silently change the meaning of the user-visible `SPECTER_FORCED` env signal.
#[derive(Copy, Clone)]
enum EmitMode<'a> {
    Standard { forced: bool },
    SeedDrift { drifted: &'a [SubId] },
}

impl EmitMode<'_> {
    /// Value to mirror into [`Effect::forced`] for emissions on this mode. `true` only on `Standard
    /// { forced: true }`.
    const fn effect_forced(self) -> bool {
        matches!(self, Self::Standard { forced: true })
    }
}

/// The consequence of a fireable quiescence verdict ([`QuiescenceVerdict::Stable`], inner
/// [`StableReason::Natural`] or [`StableReason::Forced`]) for a burst whose
/// [`specter_core::BurstIntent`] is known — computed once by [`Engine::classify_consequence`]
/// *after* the observed tree is committed, so the drift read sees the post-graft `current`.
///
/// **Owned payload by design.** [`Self::RecoveryFire`] carries the drifted Sub set so the
/// classifier stays a total `&self` function; the `&[SubId]` borrow [`EmitMode::SeedDrift`] needs
/// is taken at the `emit_effects` boundary, never stored on the classification (the classify and
/// the emit are different stack frames — a stored borrow would outlive the local it points at).
///
/// Each variant's emission mode, Draining-gate participation, witness seal, and honesty narration
/// are a total function of the variant. No derived `bool` or `Seal` discriminant is threaded
/// alongside: the seal and the per-file-drop narration are read off `EmitMode::SeedDrift` at the
/// single fire site, and the fresh-Seed skip narration — the one fact `EmitMode` cannot carry,
/// since [`Self::StandardFire`] and [`Self::FreshSeedFire`] share `EmitMode::Standard` — is the
/// only bit passed on.
#[derive(Debug)]
#[must_use]
enum Consequence {
    /// Standard burst. `EmitMode::Standard`; Draining-gated; no seal; no honesty narration.
    StandardFire,
    /// Fresh Profile that witnessed activity ([`Engine::seed_owes_first_fire`]).
    /// `EmitMode::Standard`; Draining-gated; no seal; [`Diagnostic::PerFileFireSkippedOnFreshSeed`]
    /// post-gate.
    FreshSeedFire,
    /// Recovery whose post-graft tree drifted from the settled reference, with a non-empty fired set.
    /// `EmitMode::SeedDrift`; Draining-gated; **seals the survival witness at this certified-recovery
    /// decision** (pre-`Awaiting`); [`Diagnostic::PerFileDriftDroppedOnRecovery`] post-gate.
    RecoveryFire(SmallVec<[SubId; 2]>),
    /// The non-firing arm reached by the *intent classification itself*. Four disjoint origins reach
    /// it: a fresh-static daemon restart (no witnessed activity, no baseline to drift against), a
    /// no-drift recovery, a recovery whose fired set is empty (PerFile-only), or a recovery whose
    /// fired set is empty (all Subs detached). No emission; rebases the baseline (consuming a live
    /// witness if any) and finishes; **never Draining-gated** — no Effect to defer.
    /// [`Diagnostic::PerFileDriftDroppedOnRecovery`] (self-gating predicate) before the rebase.
    SilentPin,
    /// The non-firing arm reached by the operator `absorb` *override* of a would-have-fired
    /// verdict: the live pre-fire burst carries the fold latch
    /// ([`specter_core::ProfileState::burst_fold_latched`]), so a firing `base`
    /// ([`Self::is_firing`]) is replaced with a silent baseline advance — the echo of an expected
    /// replication is folded into the settled reference instead of re-fired.
    ///
    /// Shares [`Self::SilentPin`]'s seal *terminus* ([`Engine::seal_baseline_silently`]) but is a
    /// **distinct variant**, not a `SilentPin` flag: it differs in *bookkeeping* (one
    /// [`Diagnostic::QuiescenceAbsorbed`] + a [`specter_core::Profile::note_absorb_fold`]
    /// bump/consume) and in *cause* (an operator window, not the intent fork), so burying it in
    /// `SilentPin`'s four-origin arm would hide an operator-visible event. No emission; **never
    /// Draining-gated** — like `SilentPin`, no Effect to defer.
    AbsorbFold,
    /// The discovery consequence — the Profile's scan shape is `MatchChain`, so a stable verdict
    /// reconciles the match set ([`Engine::reconcile_matches`]: mint a dynamic Sub per chain
    /// terminus × template) instead of firing Effects. Reached by the *shape pre-check* for any
    /// intent, before the intent fork and the fold override run: reconcile is idempotent via the
    /// registry dedup query, so first enumeration and re-reconcile are the same operation, and a
    /// forced ceiling reconciles from the forced graft identically (`forced` is ignored — fresh
    /// data, same consequence). Non-firing: never Draining-gated, never absorb-folded (structural —
    /// the early return precedes the override), exits through the silent seal after the mints.
    Reconcile,
}

impl Consequence {
    /// True for the three arms that run the Subs' reactions ([`Self::StandardFire`] /
    /// [`Self::FreshSeedFire`] / [`Self::RecoveryFire`]); false for the silent-seal arms
    /// ([`Self::SilentPin`] / [`Self::AbsorbFold`]) and for [`Self::Reconcile`] (discovery mints
    /// attachments, not Effects — there is nothing for an `absorb` window to suppress). The
    /// [`Engine::classify_consequence`] override consults this to decide whether a fold latch has a
    /// fire to override: a non-firing `base` passes through untouched (so a Cold-Seed `SilentPin`
    /// leaves the `absorb` window unconsumed for the first genuinely fireable burst). Wildcard-free
    /// — a future firing variant is a compile error here, not a silently-unfoldable fire.
    #[must_use]
    const fn is_firing(&self) -> bool {
        match self {
            Self::StandardFire | Self::FreshSeedFire | Self::RecoveryFire(_) => true,
            Self::SilentPin | Self::AbsorbFold | Self::Reconcile => false,
        }
    }
}

/// One Sub's fire verdict in an [`Engine::emit_effects`] pass — the total fold of the three fire
/// gates. Distinguishing `SuppressDedup` from `SkipScope` keeps the *reason* inspectable (unit
/// table, future per-cause metrics) even though the loop currently treats both as "don't emit".
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum FireVerdict {
    /// Emit for this Sub (Subtree: one Effect; PerStableFile: one per matching diff entry).
    Emit,
    /// B1 dedup suppression — a `SubtreeRoot` Sub that has fired before on a tree unchanged since
    /// the last rebase, not forced.
    SuppressDedup,
    /// This `(scope, mode)` does not fire: a `SubtreeRoot` Sub outside SeedDrift's `drifted` set,
    /// or any `PerStableFile` Sub under SeedDrift (Seed-time drift is Subtree-only).
    SkipScope,
}

/// Total, pure fire decision over `(scope, mode)` for one Sub. No engine state, no `Effect` sink —
/// exhaustively unit-testable. Folds three fire gates:
///
/// - **SeedDrift Subtree narrowing.** A `SubtreeRoot` Sub fires under SeedDrift only if it is in
///   the pre-filtered `drifted` set.
/// - **B1 dedup suppress.** A `SubtreeRoot` Sub under `Standard` suppresses iff it is not
///   force-fired, the tree is unchanged since the last rebase (`nothing_changed`), AND it has fired
///   before (`already_fired`) — a never-fired Sub is its own first emission even on an unchanged
///   tree. SeedDrift's `drifted` holds only drifted Subs, so suppression is structurally
///   unreachable on that mode (its arm yields `Emit`).
/// - **PerStableFile under SeedDrift.** Skipped entirely — Seed-time drift is Subtree-only (PerFile
///   keeps no per-leaf fire history).
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

/// Certified outcome of a Verifying / Rebase probe response — the shared result of
/// [`Engine::certify_probe_response`], routed differently by the two callers.
///
/// The certifier accepts a typed `ProofOutcome` (a proof-route probe resolves to exactly `AnchorOk`
/// / `SubtreeProven` / `Vanished` / `Failed`; the structural `DirEnumerated` shape is parsed out at
/// the demux seam, so it is unrepresentable here), then performs the operation common to the
/// Verifying choke and the post-fire Rebase arm — lower the outcome, guard kind agreement, fold the
/// quiescence verdict (events-reliable witness for CONTENT-subscribed bursts, or the
/// `last_certified_hash` channel otherwise) — and yields this 4-variant result. The callers own the
/// consequence: Verifying fans `Proceed` out per [`BurstIntent`]; Rebase maps the verdict to the
/// rebase-loop table. One verdict-construction site at the floor, two routes preserved.
///
/// - `Proceed`: lowered, kind-agreed, verdict folded — the caller acts on `(snapshot, verdict)`.
/// - `Vanished` / `Failed`: anchor disappeared / I/O error at the probe root; the caller routes to
///   its own per-route cleanup (the certifier is route-agnostic — folding a non-snapshot is
///   meaningless).
/// - `Regressed`: the certifier resolved a terminal state and the caller does nothing. Two
///   producers, both contract-violation degrades: a kind mismatch (the certifier emitted
///   [`Diagnostic::AnchorKindMismatch`] and tore the burst down through
///   [`Engine::finalize_anchor_lost`]), or an absent Profile at the floor (a gate breach — nothing
///   to tear down).
///
/// **Reachability.** Every `Regressed` producer is a contract-violation degrade, and all are
/// unreachable on a correct sensor: the payload-shape violation (a proof route receiving
/// `DirEnumerated`) is rejected before the certifier by the typed demux decode; the absent Profile
/// cannot occur because the gate dispatches only on `Active(Verifying | Rebasing)`; the kind
/// mismatch cannot occur because the walker collapses every on-disk Dir↔File swap to `Vanished`
/// rather than returning a kind-divergent snapshot. The channel exists to degrade these violations
/// gracefully, not to handle a reachable fault.
#[derive(Debug)]
enum CertifiedResponse {
    Proceed {
        snapshot: TreeSnapshot,
        verdict: QuiescenceVerdict,
    },
    Vanished,
    Failed(ProbeFailure),
    Regressed,
}

/// The Profile bits [`Engine::certify_probe_response`]'s verdict fold consumes, captured in one
/// immutable resolution ([`Engine::fold_context`]) before any `&mut` re-fetch. Every field is
/// `Copy`, so the context holds no borrow on the Profile — the caller is free to take the cat-(b)
/// `&mut self` advance or the anchor-loss finalize afterward.
///
/// - `events_witness`: whether the Profile's `events_union` covers in-place writes
///   ([`specter_core::Profile::events_witness_quiescence`]) — invariant across the burst (folds
///   into `config_hash`).
/// - `prior_kind`: the prior [`specter_core::Profile::kind`], or `None` for a fresh
///   (first-classify) Seed.
/// - `owes_proof`: whether the burst's consequence requires a tree-quiescence proof
///   ([`Engine::owes_proof_from`]).
struct FoldContext {
    events_witness: bool,
    prior_kind: Option<ResourceKind>,
    owes_proof: bool,
}

/// Pass-1 routing class for [`Engine::on_effect_complete`]: which way to route once
/// [`specter_core::Profile::note_effect_completion`]'s verdict is known.
///
/// - `CountDown(finish)`: `Active(PostFire(Awaiting))`. Pass 2 applies the completion; the last one
///   routes by the captured [`BurstFinish`] (`ReturnToIdle` → Rebasing, `Reap` → finish).
/// - `Diagnose`: any non-Awaiting state — a late completion.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
enum CompletionRoute {
    CountDown(BurstFinish),
    Diagnose,
}

/// Event-class assignment. Maps an [`FsEvent`] + the resource's [`ResourceKind`] to the
/// [`ClassSet`] bit it represents.
///
/// Non-terminal events have a fixed class regardless of kind:
/// - [`FsEvent::ContentChanged`] → [`ClassSet::CONTENT`]
/// - [`FsEvent::MetadataChanged`] → [`ClassSet::METADATA`]
/// - [`FsEvent::StructureChanged`] → [`ClassSet::STRUCTURE`]
///
/// Identity events ([`FsEvent::Removed`] / [`FsEvent::Renamed`] / [`FsEvent::Revoked`]) fold by kind:
/// - `Dir` → [`ClassSet::STRUCTURE`] (the directory's place in its parent changed).
/// - `File` (and `Unknown` via [`ResourceKind::effective`]) → [`ClassSet::CONTENT`] (the file's
///   identity changed — kqexec mapping; the Unknown collapse matches the translator's File-shape
///   default).
///
/// Pure / `const fn`; consulted at the entry filter in [`Engine::on_fs_event`].
const fn fs_event_to_class(event: FsEvent, kind: ResourceKind) -> ClassSet {
    match event {
        FsEvent::ContentChanged => ClassSet::CONTENT,
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

#[cfg(test)]
#[path = "transitions_tests.rs"]
mod tests;
