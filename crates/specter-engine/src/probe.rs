//! Engine-side probe wiring.
//!
//! Every probe-bearing fact now homes on the owner's own state: a
//! `Profile`'s descent / verify / rebase slots, a `Promoter`'s descent
//! / enumeration slots. What remains engine-resident is the
//! *irreducible floor* — the global monotone [`ProbeCorrelation`]
//! counter — plus the thin state-derived surface the response path
//! reads through:
//!
//! 1. **Correlation monotonicity for the probe id space.** The
//!    engine-wide mint floor is the bare `Engine.correlations`
//!    [`MonotonicCounter`] field, driven solely by
//!    [`Engine::mint_probe_correlation`]. The phantom-typed counter
//!    makes cross-space misuse (minting a
//!    [`specter_core::CorrelationId`] from it, or vice versa) a compile
//!    error, and saturation an unconditional panic.
//! 2. **State-derived projections.** [`Engine::probe_gate`] is the
//!    response path's sole projection: one `profiles`/`promoters`
//!    resolution yielding the gated correlation (the staleness
//!    identity) *and* the routing class together, so a `ProbeResponse`
//!    resolves the owner's state twice (gate, then the
//!    [`Engine::take_owner_probe`] disarm) instead of three times.
//!    [`Engine::pending_probe_for`] stays the standalone *liveness*
//!    projection every launch guard, double-arm backstop, and
//!    integration test reads — `probe_gate` is additive, not its
//!    replacement. "At most one probe per owner" (I5) is structural:
//!    one owner is in one state variant, which holds exactly one
//!    [`specter_core::ProbeSlot`]. The consume triad is layered:
//!    `ProbeSlot::disarm` is the slot-level consume,
//!    [`Engine::take_owner_probe`] the state-level owner-polymorphic
//!    disarm, [`Engine::cancel_owner_probe`] the engine+wire
//!    consume-plus-`Cancel` choke every abandon site routes through.
//! 3. **Request emission.** [`Engine::emit_owner_probe`] is the sole
//!    [`ProbeOp::Probe`] construction site — one owner-polymorphic
//!    choke that resolves the owner's state *once*, reads the
//!    correlation **back off the armed slot** (so armed-iff-emitted is
//!    structural — an empty/absent slot emits nothing and no second
//!    copy of the correlation can diverge), materializes the
//!    per-carrier proof obligation (the pre-fire Standard burst's
//!    `dirty` captured paths as `Chains`; `WholeSubtree` for Seed
//!    and the post-fire Rebase, neither of which has a trustworthy
//!    prior to skip against), and renders the kind-dispatched wire.
//!    Every read is immutable: the choke is a pure `&self` state→wire
//!    projection (like Descent), with no accumulator drain — the
//!    fire-tail residual reset is owned by `transition_to_rebasing`,
//!    not the emission path. It reads engine state, so it homes here
//!    without the SRP compromise the prior stateless `emit_*`
//!    constructors carried. Read-back and the launch sites' loud arm
//!    are **co-required**, not redundant: read-back guarantees no
//!    orphaned correlation reaches the wire; the loud arm guarantees an
//!    arm-guard miss is a crash, not a silent no-probe wedge. Neither
//!    subsumes the other. [`Engine::probe_gate`] (item 2) is the
//!    response-side twin — same one-resolution shape, disjoint concern
//!    (route demux vs. wire emission), so emission never depends on the
//!    response-shaped `ProbeRoute`.
//! 4. **Consume-once tripwire.** [`DispatchLedger`] (debug builds only)
//!    records the high-water correlation dispatched per owner. The
//!    structural laws (arm-once on the core slot, disarm-once via
//!    [`Engine::take_owner_probe`]) make a double-dispatch
//!    unconstructable; the ledger is the cross-step runtime witness
//!    that pins it under fuzzing and property tests.

use crate::Engine;
use crate::path::empty_path;
use specter_core::{
    ActiveBurst, BurstIntent, PostFirePhase, PreFirePhase, ProbeCorrelation, ProbeOp, ProbeOwner,
    ProbeRequest, Profile, ProfileState, Promoter, PromoterState, ProofObligation, ResourceId,
    ResourceKind, StepOutput, subtree_at_dir,
};

/// State-derived routing class for a probe response — what the
/// dispatcher needs that the response wire does not supply.
///
/// Computed by [`Engine::probe_gate`] from the owner's *current* state
/// alongside the gated correlation, so it is the minimal non-derivable
/// read. It is [`Copy`] and is captured *before* the slot is disarmed:
/// disarm empties the slot but leaves the carrier's variant intact, so
/// a route captured first stays valid through dispatch.
///
/// `Verifying` carries `(intent, forced)` because those drive the
/// per-intent fan-out and are not recoverable from the state variant
/// alone. `Enumerating` carries the proxy `target` because the
/// enumeration wire is path-only — the slot's tag is the sole
/// authority for the dispatch key. `Rebasing` and `Descent` need no
/// payload — the variant (and, for descent, the owner the handler
/// already holds) is the whole routing decision.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum ProbeRoute {
    /// Pending-path descent (Profile `Pending` or Promoter
    /// `PrefixPending`). The handler routes on the owner it already
    /// holds; the outcome variant selects advance / rewind / fail.
    Descent,
    /// Profile pre-fire stability probe. `intent` / `forced` are read
    /// off the `PreFireBurst` for the per-intent dispatch fan-out.
    Verifying { intent: BurstIntent, forced: bool },
    /// Profile post-fire rebase probe. The outcome routes through the
    /// shared `certify_probe_response` certifier (kind-check + N=2
    /// quiescence fold over the post-command tree) and then
    /// `dispatch_rebase_*`; no payload — the variant is the whole
    /// routing decision.
    Rebasing,
    /// Promoter proxy enumeration (`Active`). `target` is the proxy the
    /// probe enumerates, read from the enumeration slot's tag — the
    /// wire is path-only, so it is the canonical dispatch key across
    /// every outcome (`DirEnumerated` / `Vanished` / `Failed`).
    Enumerating { target: ResourceId },
}

/// Debug-only consume-once tripwire.
///
/// The structural laws make a double-dispatch unconstructable: the
/// core slot's `arm` asserts arm-once, and [`Engine::take_owner_probe`]
/// disarms exactly once before any dispatch. This ledger is the
/// *cross-step runtime witness* of that property — a
/// [`specter_core::ProbeSlot`] is a single-step value and cannot carry
/// per-owner history, so the engine owns the dispatch-once half of the
/// proof. Correlations are minted off one monotone floor, so a
/// correctly-consumed sequence dispatches strictly increasing
/// correlations per owner; re-dispatching an already-consumed
/// correlation is necessarily not above the per-owner high-water mark
/// and trips the assert. Debug-only: zero cost, zero footprint in
/// release.
#[cfg(debug_assertions)]
#[derive(Debug, Default)]
pub(crate) struct DispatchLedger {
    high_water: std::collections::BTreeMap<ProbeOwner, ProbeCorrelation>,
}

#[cfg(debug_assertions)]
impl DispatchLedger {
    /// Record that `correlation` was routed into a `dispatch_*` arm for
    /// `owner`, asserting it is strictly greater than every correlation
    /// previously dispatched for that owner. Sole callers: the two
    /// response handlers, immediately after the slot is disarmed and
    /// before the outcome is dispatched.
    pub(crate) fn record(&mut self, owner: ProbeOwner, correlation: ProbeCorrelation) {
        let prior = self.high_water.insert(owner, correlation);
        debug_assert!(
            prior.is_none_or(|p| correlation > p),
            "consume-once tripwire: correlation {correlation:?} dispatched for \
             {owner:?} is not strictly greater than the prior dispatched \
             {prior:?} — a probe correlation reached a dispatch arm more than once",
        );
    }
}

impl Engine {
    /// The owner's in-flight probe correlation, or `None` if it has
    /// none. A pure projection over the owner's state: every
    /// probe-bearing carrier (Profile descent / verify / rebase,
    /// Promoter descent / enumeration) homes its correlation on a
    /// [`specter_core::ProbeSlot`] in exactly one state variant, so
    /// reading that variant's slot is the single source of truth. This
    /// is the staleness identity the response path gates on.
    ///
    /// `pub` (not `pub(crate)`) — the engine crate's `tests/`
    /// directory is an external crate from a Rust visibility
    /// standpoint, and ~35 integration-test call sites depend on this
    /// for probe-state assertions.
    #[must_use]
    pub fn pending_probe_for(&self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        match owner {
            ProbeOwner::Profile(pid) => self
                .profiles
                .get(pid)
                .and_then(|p| p.state().probe_correlation()),
            ProbeOwner::Promoter(qid) => self
                .promoters
                .get(qid)
                .and_then(|q| q.state().probe_correlation()),
        }
    }

    /// The owner's response-path gate: its in-flight probe correlation
    /// **and** routing class, from one state resolution.
    ///
    /// Folds the staleness identity and the routing class into a single
    /// `profiles`/`promoters` lookup. The correlation is read through
    /// the public state surface
    /// ([`specter_core::ProfileState::probe_correlation`] /
    /// [`specter_core::PromoterState::probe_correlation`] — the same
    /// projection [`Self::pending_probe_for`] exposes, so the engine
    /// never reaches past that surface into a descent's slot), so a
    /// `ProbeResponse` resolves the owner's state twice (here, then the
    /// [`Self::take_owner_probe`] disarm) rather than three times. The
    /// route is [`Copy`]; the caller gates on the correlation, disarms
    /// once, then dispatches on the route — the disarm leaves the
    /// carrier variant intact, so the route stays valid through
    /// dispatch.
    ///
    /// `Some((correlation, route))` iff the owner is in a probe-bearing
    /// carrier with an armed slot; `None` otherwise (⇒ the caller emits
    /// `StaleProbeResponse`). "Armed slot but no route" is
    /// unrepresentable: every state whose `probe_correlation` is `Some`
    /// is, by the same case split, a routable carrier, so the dead
    /// armed-but-unroutable arm the old open-coded staleness-gate +
    /// route-snapshot pair carried as a loud regression bail folds
    /// structurally into this single `None`. The `Active` enumeration
    /// arm reads the proxy `target` off the slot's tag — the wire is
    /// path-only, so that tag is the route's sole authority for the
    /// dispatch key.
    ///
    /// Distinct from [`Self::pending_probe_for`], which stays the
    /// standalone *liveness* projection the launch guards, double-arm
    /// backstops, and integration tests read; `probe_gate` is the
    /// response path's additive fold, not a replacement.
    pub(crate) fn probe_gate(&self, owner: ProbeOwner) -> Option<(ProbeCorrelation, ProbeRoute)> {
        match owner {
            ProbeOwner::Profile(pid) => {
                let state = self.profiles.get(pid)?.state();
                let correlation = state.probe_correlation()?;
                let route = match state {
                    ProfileState::Pending(_) => Some(ProbeRoute::Descent),
                    ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
                        PreFirePhase::Verifying(_) => Some(ProbeRoute::Verifying {
                            intent: pre.intent,
                            forced: pre.forced,
                        }),
                        PreFirePhase::Batching { .. } | PreFirePhase::Draining => None,
                    },
                    ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
                        PostFirePhase::Rebasing(_) => Some(ProbeRoute::Rebasing),
                        PostFirePhase::Awaiting { .. } | PostFirePhase::RebaseSettling { .. } => {
                            None
                        }
                    },
                    ProfileState::Idle => None,
                }?;
                Some((correlation, route))
            }
            ProbeOwner::Promoter(qid) => {
                let state = self.promoters.get(qid)?.state();
                let correlation = state.probe_correlation()?;
                let route = match state {
                    PromoterState::PrefixPending(_) => Some(ProbeRoute::Descent),
                    PromoterState::Active { enumerating, .. } => enumerating
                        .tag()
                        .map(|target| ProbeRoute::Enumerating { target }),
                }?;
                Some((correlation, route))
            }
        }
    }

    /// Mint a fresh [`ProbeCorrelation`] off the engine-wide monotone
    /// floor (`self.correlations`) — the sole mint driver for every
    /// state-resident probe slot. One id space, so slot-held
    /// correlations never collide; saturation panics unconditionally
    /// via [`crate::counter::MonotonicCounter::next`].
    ///
    /// **Deferred type-honest end-state.** The `mint → arm` gap is the
    /// one edge [`specter_core::ProbeSlot`]'s linear discipline does not
    /// cover — its `arm` re-acquire assert and `Drop` tripwire begin
    /// only once the correlation is on the slot. The honest end-state is
    /// a linear non-`Copy` `#[must_use]` mint token consumed by
    /// `arm`/`armed`, with a `Drop` tripwire if dropped un-armed,
    /// turning `mint`-without-`arm` into a compile/`Drop` error (the
    /// same shape `ids.rs` defers for `Minted<T>`). Not built: the
    /// emission choke ([`Self::emit_owner_probe`]'s read-back) already
    /// makes armed-iff-emitted structural and the launch sites' loud arm
    /// makes an arm miss fatal, so the token would buy only
    /// compile-error-vs-panic on an already-unreachable branch — not
    /// worth threading a move-only token through every launch path under
    /// single-user alpha.
    #[must_use]
    pub(crate) fn mint_probe_correlation(&mut self) -> ProbeCorrelation {
        self.correlations.next()
    }

    /// Consume the owner's in-flight probe and return its correlation
    /// (`None` if none was in flight). Disarms the owner's
    /// state-resident slot — Profile descent / verify / rebase, or
    /// Promoter descent / enumeration. The single consume primitive
    /// both the response path and the cancel path route through; one
    /// owner is in one state variant holding one slot, so the disarm is
    /// unambiguous.
    #[must_use]
    pub(crate) fn take_owner_probe(&mut self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        match owner {
            ProbeOwner::Profile(pid) => self.profiles.get_mut(pid).and_then(Profile::take_probe),
            ProbeOwner::Promoter(qid) => self.promoters.get_mut(qid).and_then(Promoter::take_probe),
        }
    }

    /// Consume the owner's in-flight probe and emit [`ProbeOp::Cancel`]
    /// iff one was in flight. The disarm *is* the consume, atomic
    /// with the Cancel emission within this one `&mut self` window.
    ///
    /// Sole "consume + emit Cancel" choke point used at every cancel
    /// site — `event_drives_batching`, `finalize_anchor_lost`,
    /// `on_watch_op_rejected` descent / proxy purges, `reap_profile`,
    /// `reap_promoter_inner`. Idempotent when no probe is in flight.
    /// Inlining at each site loses the named contract that "you must
    /// Cancel if-and-only-if a probe was outstanding".
    pub(crate) fn cancel_owner_probe(&mut self, owner: ProbeOwner, out: &mut StepOutput) {
        if self.take_owner_probe(owner).is_some() {
            out.push_probe_op(ProbeOp::Cancel { owner });
        }
    }

    /// Abandon **every** in-flight probe across all Profiles and
    /// Promoters, emitting one [`ProbeOp::Cancel`] per owner that had
    /// one outstanding. Returns the sealed [`StepOutput`] the caller
    /// forwards to the prober.
    ///
    /// This is the **graceful-shutdown probe drain**. The linear
    /// [`specter_core::ProbeSlot`] `Drop` tripwire fires if the
    /// `Engine` is dropped with a probe still armed, and a normal
    /// shutdown routinely coincides with one in flight (every
    /// settle / verify / rebase / descent window). The driver calls
    /// this once when a tick resolves to shutdown, before the engine
    /// is torn down; every consume routes through the crate-private
    /// `cancel_owner_probe` — the same disarm-then-`Cancel` choke
    /// every internal abandon site uses — so the slot is consumed
    /// (not forgotten, not leaked) and a graceful exit is silent.
    /// The guard stays fully effective: a genuine mid-`step` orphan
    /// still panics during that step, long before any shutdown drain
    /// runs.
    ///
    /// Tests that freeze a Profile / Promoter mid-flight reuse this
    /// for the *same* teardown before dropping a local `Engine`, so a
    /// test models the real shutdown path, not a test-only fiction.
    /// `pub` because the driver crate and the engine's external
    /// `tests/` crate are both out-of-crate callers.
    ///
    /// Snapshot-then-consume: the `probe_correlation` projection
    /// borrows `&self`, the disarm needs `&mut self`, so they can't
    /// overlap; one owner is one state variant holding one slot, so
    /// every armed slot is enumerated exactly once. Output is
    /// [`StepOutput::sort_for_emission`]-sealed like every other
    /// `StepOutput`-returning entry point — `forward` assumes a
    /// resealed value.
    #[must_use]
    pub fn cancel_all_in_flight_probes(&mut self) -> StepOutput {
        let mut out = StepOutput::default();
        let owners: Vec<ProbeOwner> = self
            .profiles
            .iter()
            .filter_map(|(pid, p)| {
                p.state()
                    .probe_correlation()
                    .map(|_| ProbeOwner::Profile(pid))
            })
            .chain(self.promoters.iter().filter_map(|(qid, q)| {
                q.state()
                    .probe_correlation()
                    .map(|_| ProbeOwner::Promoter(qid))
            }))
            .collect();
        for owner in owners {
            self.cancel_owner_probe(owner, &mut out);
        }
        out.sort_for_emission();
        out
    }

    /// The sole [`ProbeOp::Probe`] emission choke. Every launch path —
    /// Seed / Verify / Rebase, Profile & Promoter descent, Promoter
    /// enumeration — is `mint → arm (loud) → emit_owner_probe(owner)`;
    /// the caller passes nothing but the owner. Pushes **exactly one**
    /// `ProbeOp::Probe` for an owner whose slot is armed, and **nothing**
    /// for an owner whose slot is empty/absent: armed-iff-emitted is
    /// structural here, because the correlation on the wire is the one
    /// read back off the slot ([`Self::probe_emission_request`]), not a
    /// caller-threaded copy that could outlive a skipped arm.
    ///
    /// This is one half of a co-required pair, not a standalone
    /// guarantee. Read-back makes *armed ⇒ the wire carries exactly the
    /// slot's correlation* and *not-armed ⇒ no wire*. The launch sites'
    /// loud arm makes *arm-guard miss ⇒ crash*. Without the loud arm a
    /// missed arm would be a silent no-probe wedge (worse than the old
    /// orphan-stall — it emits no diagnostic at all); without read-back a
    /// missed arm would orphan a threaded correlation. Each kills a
    /// distinct failure; neither is redundant.
    pub(crate) fn emit_owner_probe(&self, owner: ProbeOwner, out: &mut StepOutput) {
        if let Some(request) = self.probe_emission_request(owner) {
            out.push_probe_op(ProbeOp::Probe { request });
        }
    }

    /// Resolve `owner`'s state **once** into the wire it should emit, or
    /// `None` if its slot is empty/absent (⇒ no probe). The emission-side
    /// twin of [`Self::probe_gate`]: the same one-resolution shape, the
    /// disjoint concern. `probe_gate` yields the `Copy` route the
    /// *response* demuxes on; this yields the owned `ProbeRequest` the
    /// *request* carries — heavier (`ScanConfig`, baseline `Arc`, the
    /// proof obligation), so it is deliberately a separate function
    /// rather than a `probe_gate` caller; emission never depends on the
    /// response-shaped [`ProbeRoute`].
    ///
    /// A pure `&self` state→wire projection — like Descent, no
    /// accumulator drain. The post-fire fire-tail residual reset is
    /// owned by `transition_to_rebasing` (the category-(a) phase
    /// helper), not the emission path, so this choke reaches no
    /// burst-mut at all.
    ///
    /// Two passes under **one** `profiles`/`promoters` resolution:
    ///
    /// 1. **Classify + read back.** Match the owner's state; read the
    ///    correlation *off the armed slot* (`?` ⇒ an empty slot returns
    ///    `None` and nothing is emitted — the structural armed-iff-
    ///    emitted property), and resolve `(target, forced)` from the
    ///    same match.
    /// 2. **Render the wire** via `&self.tree`. Descent / enumeration
    ///    are path-only; Verify / Rebase kind-dispatch — `Some(File)`
    ///    ⇒ `AnchorFile`, else ⇒ `Subtree` with the Profile's
    ///    `(config, config_hash)`, `baseline_subtree`, and the
    ///    per-carrier [`specter_core::ProofObligation`] (Standard ⇒
    ///    `Chains` from the persisting `dirty`'s captured paths, or
    ///    `WholeSubtree` under a `debug_assert` if empty; Seed and
    ///    Rebase ⇒ `WholeSubtree` — no trustworthy prior — built
    ///    lazily, never for a File anchor). The kind rule lives here
    ///    exactly once, so the prior positional constructors' fan-out
    ///    dissolves into struct literals.
    fn probe_emission_request(&self, owner: ProbeOwner) -> Option<ProbeRequest> {
        // `Copy` carrier classification: which carrier, and (for the
        // pre-fire carrier) the target + `forced` + `intent` read off
        // state. No obligation source is carried here — the borrowed
        // `dirty` provenance is not `Copy`; it is read immutably off the
        // still-borrowed Profile in the render pass, keyed by this.
        #[derive(Clone, Copy)]
        enum Carrier {
            /// Profile `Pending` / Promoter `PrefixPending` /
            /// Promoter `Active` enumeration — all path-only `Descent`
            /// wires; the target is fully resolved here. No
            /// proof obligation (a structural query is not a
            /// quiescence observation).
            Descent(ResourceId),
            /// Profile `Verifying`. `target` = `pre.probe_target`
            /// (the live id `pre_fire_target` resolved from the
            /// captured paths' LCA), `forced` = `pre.forced`. `intent`
            /// selects the obligation kind: Seed ⇒ `WholeSubtree` (no
            /// trustworthy prior); Standard ⇒ `Chains` from the
            /// *persisting* `dirty`'s captured paths (read immutably in
            /// the render pass — the burst outlives this probe across
            /// re-batching).
            PreFire {
                target: ResourceId,
                forced: bool,
                intent: BurstIntent,
            },
            /// Profile `Rebasing` — target is the anchor, `forced` is
            /// pre-fire-only (⇒ `false`). Obligation = `WholeSubtree`:
            /// the command just mutated the tree, so there is no
            /// trustworthy prior to skip against (exactly as Seed).
            /// No accumulator drain — the fire-tail residual reset is
            /// owned by `transition_to_rebasing`.
            Rebase,
        }

        match owner {
            ProbeOwner::Profile(pid) => {
                let p = self.profiles.get(pid)?;
                let anchor = p.resource();

                // Read the correlation BACK off the armed slot via the
                // *same* pub projection `pending_probe_for` reads
                // (`probe_correlation` is `pub(crate)` to `core`; the
                // engine never reaches past the state surface into a
                // carrier's private slot). `?` on an empty slot ⇒
                // `None` ⇒ no probe: armed-iff-emitted, structurally.
                // Then classify the carrier — target / `forced` /
                // `intent` / kind-dispatch — independently of the
                // correlation just read. A `Some` correlation *implies*
                // a probe-bearing carrier (Batching / Draining /
                // Awaiting / RebaseSettling / Idle hold no slot), so
                // those arms are structurally dead when the read-back
                // succeeded; they fold to `None` exactly as
                // `probe_gate`'s twin arms do.
                let correlation = p.state().probe_correlation()?;
                let carrier = match p.state() {
                    ProfileState::Pending(d) => Carrier::Descent(d.current_prefix()),
                    ProfileState::Active(ActiveBurst::PreFire(pre), _) => match &pre.phase {
                        PreFirePhase::Verifying(_) => Carrier::PreFire {
                            target: pre.probe_target,
                            forced: pre.forced,
                            intent: pre.intent,
                        },
                        PreFirePhase::Batching { .. } | PreFirePhase::Draining => return None,
                    },
                    ProfileState::Active(ActiveBurst::PostFire(post), _) => match &post.phase {
                        PostFirePhase::Rebasing(_) => Carrier::Rebase,
                        PostFirePhase::Awaiting { .. } | PostFirePhase::RebaseSettling { .. } => {
                            return None;
                        }
                    },
                    ProfileState::Idle => return None,
                };

                // The Rebase target is the anchor (`PostFireBurst`
                // carries no `probe_target`); `forced` is pre-fire-only
                // so `false`. No mutation here — the Rebase obligation
                // is `WholeSubtree` (built in the render pass), so this
                // resolution no longer needs `&mut` to drain anything.
                let (target, forced) = match carrier {
                    Carrier::Descent(prefix) => (prefix, false),
                    Carrier::PreFire { target, forced, .. } => (target, forced),
                    Carrier::Rebase => (anchor, false),
                };

                // Render via `&self.tree`. Descent is path-only; the
                // pre-fire / rebase carriers kind-dispatch.
                let target_path = self.tree.path_of(target).unwrap_or_else(empty_path);
                Some(match carrier {
                    Carrier::Descent(_) => ProbeRequest::Descent {
                        owner,
                        correlation,
                        target_path,
                    },
                    Carrier::PreFire { .. } | Carrier::Rebase => match p.kind() {
                        Some(ResourceKind::File) => ProbeRequest::AnchorFile {
                            owner,
                            correlation,
                            target_path,
                        },
                        // Dir or still-unclassified ⇒ the kind-agnostic
                        // Subtree walk; the walker returns `Vanished` on
                        // a kind mismatch and the engine recovers via
                        // descent. The proof obligation is materialized
                        // here — lazily (never for a File anchor) and
                        // per carrier.
                        _ => {
                            let scan_config = p.config().clone();
                            let captured_with = p.config_hash();
                            let baseline_subtree = p
                                .current_dir()
                                .and_then(|root| subtree_at_dir(root, anchor, target, &self.tree));
                            let obligation = match carrier {
                                // Rebase / Seed: no trustworthy prior
                                // exists, so nothing under the anchor
                                // may be skipped — the whole subtree is
                                // unproven until freshly read. Rebase
                                // because the command just mutated the
                                // tree (an in-place descendant edit need
                                // not bump an ancestor mtime, so a
                                // chains-only skip would certify a false
                                // quiet); Seed because it has never
                                // observed the tree.
                                Carrier::Rebase
                                | Carrier::PreFire {
                                    intent: BurstIntent::Seed,
                                    ..
                                } => ProofObligation::WholeSubtree,
                                // Standard: the event-dirty root→leaf
                                // chains, the captured paths off the
                                // *persisting* `dirty` (re-read
                                // immutably — the carrier classified
                                // PreFire and the stable `&Profile`
                                // borrow makes an intervening state
                                // change unrepresentable). Every captured
                                // path is at-or-under `target` by
                                // construction (`pre_fire_target`
                                // resolved the captured paths' LCA), so
                                // no subtree filter is needed. An empty
                                // `dirty` is a should-never (a Standard
                                // burst notes its trigger); degrade to
                                // `WholeSubtree` so the response proves
                                // the whole subtree rather than
                                // silently skipping it, with the
                                // `debug_assert` as the dev/CI tripwire
                                // for an ingest path that forgot to note.
                                Carrier::PreFire {
                                    intent: BurstIntent::Standard,
                                    ..
                                } => {
                                    let ProfileState::Active(ActiveBurst::PreFire(pre), _) =
                                        p.state()
                                    else {
                                        unreachable!(
                                            "probe_emission_request: Profile {pid:?} left \
                                             PreFire between carrier classification and the \
                                             obligation build under one stable &Profile \
                                             borrow"
                                        )
                                    };
                                    if pre.dirty.is_empty() {
                                        debug_assert!(
                                            false,
                                            "Standard obligation empty: every ingest site \
                                             must note(id, path) (profile {pid:?})"
                                        );
                                        ProofObligation::WholeSubtree
                                    } else {
                                        ProofObligation::Chains(pre.dirty.chains())
                                    }
                                }
                                // Descent emits ProbeRequest::Descent in
                                // the outer arm and never reaches the
                                // Subtree obligation builder.
                                Carrier::Descent(_) => unreachable!(
                                    "probe_emission_request: Descent carrier in the \
                                     Subtree obligation builder"
                                ),
                            };
                            ProbeRequest::Subtree {
                                owner,
                                correlation,
                                target_path,
                                scan_config,
                                captured_with,
                                baseline_subtree,
                                obligation,
                                forced,
                            }
                        }
                    },
                })
            }
            ProbeOwner::Promoter(qid) => {
                // Descent / enumeration are path-only; no proof
                // obligation, no kind dispatch, so an immutable `get`
                // suffices. The enumeration slot's tag is the proxy
                // target the path-only wire cannot echo back.
                let q = self.promoters.get(qid)?;
                let correlation = q.state().probe_correlation()?;
                let target = match q.state() {
                    PromoterState::PrefixPending(d) => d.current_prefix(),
                    // `Active` enumeration: the slot tag is the proxy
                    // target the path-only wire cannot echo back.
                    PromoterState::Active { .. } => q.state().enumeration_target()?,
                };
                let target_path = self.tree.path_of(target).unwrap_or_else(empty_path);
                Some(ProbeRequest::Descent {
                    owner,
                    correlation,
                    target_path,
                })
            }
        }
    }
}
