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
//! 2. **State-derived projections.** [`Engine::pending_probe_for`] (the
//!    staleness identity), [`Engine::probe_route`] (the routing class),
//!    and [`Engine::take_owner_probe`] (the single consume) read or
//!    disarm the owner's state slot directly. "At most one probe per
//!    owner" (I5) is structural: one owner is in one state variant,
//!    which holds exactly one [`specter_core::ProbeSlot`].
//!    [`Engine::cancel_owner_probe`] is the consume-plus-`Cancel`
//!    choke every abandon site routes through.
//! 3. **Request emission.** [`Engine::emit_anchor_probe`] /
//!    [`Engine::emit_subtree_probe`] / [`Engine::emit_descent_probe`]
//!    are the sole [`ProbeOp::Probe`] construction sites — stateless
//!    typed constructors that live here because they belong to "probe
//!    wiring" even though they touch no engine state.
//! 4. **Consume-once tripwire.** [`DispatchLedger`] (debug builds only)
//!    records the high-water correlation dispatched per owner. The
//!    structural laws (arm-once on the core slot, disarm-once via
//!    [`Engine::take_owner_probe`]) make a double-dispatch
//!    unconstructable; the ledger is the cross-step runtime witness
//!    that pins it under fuzzing and property tests.

use crate::Engine;
use specter_core::{
    ActiveBurst, BurstIntent, DirSnapshot, PostFirePhase, PreFirePhase, ProbeCorrelation, ProbeOp,
    ProbeOwner, ProbeRequest, Profile, ProfileState, Promoter, PromoterState, ResourceId,
    ScanConfig, StepOutput,
};
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// State-derived routing class for a probe response — what the
/// dispatcher needs that the response wire does not supply.
///
/// Computed by [`Engine::probe_route`] from the owner's *current*
/// state, so it is the minimal non-derivable read. It is [`Copy`] and
/// is snapshotted *before* the slot is disarmed: disarm empties the
/// slot but leaves the carrier's variant intact, so a route captured
/// first stays valid through dispatch.
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
    /// Profile post-fire baseline-capture probe. The outcome routes
    /// straight through `dispatch_rebase_*`.
    Rebasing,
    /// Promoter proxy enumeration (`Active`). `target` is the proxy the
    /// probe enumerates, read from the enumeration slot's tag — the
    /// wire is path-only, so it is the canonical dispatch key across
    /// every outcome (`SubtreeOk` / `Vanished` / `Failed`).
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

    /// The owner's probe routing class derived purely from its current
    /// state, or `None` if the owner is in no probe-bearing carrier.
    ///
    /// Owner-symmetric with [`Self::pending_probe_for`] /
    /// [`Self::take_owner_probe`]; it is the routing twin of the
    /// staleness gate. The caller snapshots this *before*
    /// [`Self::take_owner_probe`] (the route is [`Copy`], the disarm
    /// leaves the carrier variant intact), then dispatches on it.
    ///
    /// Total over the state space: a probe-bearing carrier with an
    /// armed slot yields its route; every other state (including a
    /// disarmed slot) yields `None`. The `Active` enumeration arm reads
    /// the proxy `target` off the slot's tag — the wire is path-only,
    /// so that tag is the route's sole authority for the dispatch key.
    pub(crate) fn probe_route(&self, owner: ProbeOwner) -> Option<ProbeRoute> {
        match owner {
            ProbeOwner::Profile(pid) => match self.profiles.get(pid)?.state() {
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
                    PostFirePhase::Awaiting { .. } => None,
                },
                ProfileState::Idle => None,
            },
            ProbeOwner::Promoter(qid) => match self.promoters.get(qid)?.state() {
                PromoterState::PrefixPending(_) => Some(ProbeRoute::Descent),
                PromoterState::Active { enumerating, .. } => enumerating
                    .tag()
                    .map(|target| ProbeRoute::Enumerating { target }),
            },
        }
    }

    /// Mint a fresh [`ProbeCorrelation`] off the engine-wide monotone
    /// floor (`self.correlations`) — the sole mint driver for every
    /// state-resident probe slot. One id space, so slot-held
    /// correlations never collide; saturation panics unconditionally
    /// via [`crate::counter::MonotonicCounter::next`].
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
            out.probe_ops.push(ProbeOp::Cancel { owner });
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
    /// is torn down; every consume routes through
    /// [`Self::cancel_owner_probe`] — the same disarm-then-`Cancel`
    /// choke every internal abandon site uses — so the slot is
    /// consumed (not forgotten, not leaked) and a graceful exit is
    /// silent. The guard stays fully effective: a genuine mid-`step`
    /// orphan still panics during that step, long before any shutdown
    /// drain runs.
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

    /// Emit [`ProbeRequest::AnchorFile`]. Walker runs a single `lstat`
    /// and returns `ProbeOutcome::AnchorOk` / `Vanished` / `Failed`.
    ///
    /// `correlation` must be the value just minted via
    /// [`Engine::mint_probe_correlation`] and armed onto the owner's
    /// slot (mint precedes this call within the same `&mut self`
    /// window). Associated function — no Engine-state dependency.
    pub(crate) fn emit_anchor_probe(
        owner: ProbeOwner,
        correlation: ProbeCorrelation,
        target_path: Arc<Path>,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::AnchorFile {
                owner,
                correlation,
                target_path,
            },
        });
    }

    /// Emit [`ProbeRequest::Subtree`]. Recursive Dir walk honouring
    /// `scan_config`; walker returns
    /// `ProbeOutcome::SubtreeOk(Arc<DirSnapshot>)` rooted at
    /// `target_path`.
    ///
    /// `scan_config` / `captured_with` come from the Profile — the
    /// caller already holds a `&Profile` borrow at every call site and
    /// threads `(p.config.clone(), p.config_hash)` through here. The
    /// helper does not re-borrow `self` to look them up.
    ///
    /// The wire carries `target_path` only. Engine-side identity (the
    /// `ResourceId` the engine probed) stays on the owner's state slot
    /// — the walker never needs the engine's `Tree`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn emit_subtree_probe(
        owner: ProbeOwner,
        correlation: ProbeCorrelation,
        target_path: Arc<Path>,
        scan_config: ScanConfig,
        captured_with: u64,
        baseline_subtree: Option<Arc<DirSnapshot>>,
        force_walk: BTreeSet<Arc<Path>>,
        forced: bool,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::Subtree {
                owner,
                correlation,
                target_path,
                scan_config,
                captured_with,
                baseline_subtree,
                force_walk,
                forced,
            },
        });
    }

    /// Emit [`ProbeRequest::Descent`]. Single-level enumeration of the
    /// prefix; walker hardcodes the override config
    /// (`recursive=false`, `hidden=true`, no exclude/pattern, no
    /// `max_depth`) — the Profile's user-facing filters would mask the
    /// very segment descent is searching for.
    ///
    /// The wire carries `target_path` only. The engine reads the
    /// dispatch identity off its own state at response time:
    /// `descent.current_prefix()` for Profile / Promoter descent, the
    /// `Active` enumeration slot's tag for Promoter enumeration.
    pub(crate) fn emit_descent_probe(
        owner: ProbeOwner,
        correlation: ProbeCorrelation,
        target_path: Arc<Path>,
        out: &mut StepOutput,
    ) {
        out.probe_ops.push(ProbeOp::Probe {
            request: ProbeRequest::Descent {
                owner,
                correlation,
                target_path,
            },
        });
    }
}
