//! Probe channel — engine↔Prober communication primitive.
//!
//! [`ProbeChannel`] owns the per-owner outstanding-probe map, the
//! correlation counter, and (for [`OpenKind::PromoterEnumerating`]) the
//! sibling state the dispatcher reads to identify the proxy across all
//! three response outcomes (`SubtreeOk`, `Vanished`, `Failed`). The
//! structure encodes three invariants that were previously split between
//! core fields and engine helpers:
//!
//! 1. **I5 — at most one outstanding probe per [`ProbeOwner`].** Enforced
//!    structurally: a single [`std::collections::BTreeMap`] entry per
//!    owner; [`ProbeChannel::open`] panics unconditionally on a
//!    double-open (matches [`crate::counter::MonotonicCounter::next`]'s
//!    saturation discipline — silent overwrite would corrupt
//!    stale-response detection).
//! 2. **Correlation monotonicity for the probe id space.** The counter
//!    lives inside the channel; mint is the sole drive site. Cross-counter
//!    misuse (minting a [`specter_core::CorrelationId`] from this counter
//!    or vice versa) is a compile error via the phantom-typed wrapper.
//! 3. **Per-owner sibling state.** [`OpenKind::PromoterEnumerating`]
//!    carries the proxy [`ResourceId`] the enumeration response refers
//!    to. The wire payload is path-only — the dispatcher reads the
//!    target off the channel's variant payload uniformly, regardless of
//!    which `ProbeOutcome` variant came back. Pairing the target with
//!    the channel state in one structural slot keeps the dispatch key
//!    in lockstep with the correlation.
//!
//! Response dispatchers route on [`Open::kind`] rather than inspecting
//! Profile / Promoter state, so phase-mismatch cases that used to require
//! `debug_assert!(false, "I5 violated")` arms are now unrepresentable.
//!
//! The associated [`Engine::emit_anchor_probe`] /
//! [`Engine::emit_subtree_probe`] / [`Engine::emit_descent_probe`] helpers
//! are the sole construction sites for [`ProbeOp::Probe`] requests; they
//! live alongside the channel type because both belong to "probe wiring"
//! even though the helpers are stateless typed constructors.

use crate::Engine;
use crate::counter::MonotonicCounter;
use specter_core::{
    ActiveBurst, BurstIntent, DirSnapshot, PostFirePhase, PreFirePhase, ProbeCorrelation, ProbeOp,
    ProbeOwner, ProbeRequest, Profile, ProfileState, PromoterState, ResourceId, ScanConfig,
    StepOutput,
};
use std::collections::btree_map::Entry;
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::Arc;

/// Engine-resident probe-channel state. Owns the open-map and the
/// correlation counter.
///
/// **Construction.** [`Engine::probe_channel`] is initialised via
/// `Default` at [`Engine::new`]; both the map and the counter start empty.
///
/// **Invariants.**
/// - Every entry in `open` is for a live [`ProbeOwner`] — closure happens
///   in [`Engine::cancel_owner_probe`] (which precedes every reap path)
///   and the per-owner response dispatchers' [`Self::close_if`] step.
///   Reap-time `debug_assert!` in [`Engine::reap_profile`] /
///   [`Engine::reap_promoter_inner`] catches missed closures.
/// - `counter` advances monotonically; saturation panics
///   unconditionally via [`MonotonicCounter::next`] (release-runnable).
#[derive(Debug, Default)]
pub(crate) struct ProbeChannel {
    open: BTreeMap<ProbeOwner, Open>,
    counter: MonotonicCounter<ProbeCorrelation>,
}

/// Per-owner channel-state record. Carries the correlation the channel
/// was opened with and a typed [`OpenKind`] discriminant the dispatcher
/// reads to route the response.
///
/// Construction is closed under [`ProbeChannel::open`] — fields are
/// private; engine code accesses them via [`Self::correlation`] /
/// [`Self::kind`].
#[derive(Debug)]
pub(crate) struct Open {
    correlation: ProbeCorrelation,
    kind: OpenKind,
}

impl Open {
    #[must_use]
    pub(crate) const fn correlation(&self) -> ProbeCorrelation {
        self.correlation
    }

    #[must_use]
    pub(crate) const fn kind(&self) -> &OpenKind {
        &self.kind
    }
}

/// Typed routing tag for the one remaining channel-resident probe
/// class: Promoter proxy enumeration.
///
/// Every other probe (Profile descent, Profile verify, Profile rebase,
/// Promoter literal-prefix descent) is state-resident — its correlation
/// lives on a [`specter_core::ProbeSlot`] in the owner's state and is
/// routed by inspecting that state, not by an open-map entry. Promoter
/// enumeration is the sole holdout: its dispatch key (the proxy
/// `target`) is not derivable from the wire (path-only) and does not
/// yet live on Promoter state, so the channel still carries it.
///
/// [`Self::PromoterEnumerating`] carries the proxy [`ResourceId`] the
/// enumeration probe targets. The dispatcher reads it on every outcome
/// (`SubtreeOk`, `Vanished`, `Failed`) — the wire is path-only, and the
/// engine's dispatch identity lives on the channel state rather than
/// being echoed by the walker.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub(crate) enum OpenKind {
    /// Promoter proxy enumeration. `target` identifies the proxy the
    /// probe is enumerating. The dispatcher reads it on every outcome
    /// (`SubtreeOk` / `Vanished` / `Failed`) — the wire is path-only,
    /// so this variant is the canonical source of the dispatch key.
    PromoterEnumerating { target: ResourceId },
}

/// State-derived routing class for a probe response — the replacement
/// for matching a channel-resident [`OpenKind`] on the response path.
///
/// Computed by [`Engine::probe_route`] from the owner's *current*
/// state, so it is the minimal non-derivable read the dispatcher needs
/// that the response wire does not supply. It is [`Copy`] and is
/// snapshotted *before* the slot is disarmed: disarm empties the slot
/// but leaves the carrier's variant intact, so a route captured first
/// stays valid through dispatch.
///
/// `Verifying` carries `(intent, forced)` because those drive the
/// per-intent fan-out and are *not* recoverable from the state variant
/// alone. `Rebasing` and `Descent` need no payload — the variant (and,
/// for descent, the owner the handler already holds) is the whole
/// routing decision.
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
}

impl ProbeChannel {
    /// Open the channel for `owner`. Mints a fresh
    /// [`ProbeCorrelation`] from the channel's monotonic counter,
    /// stamps it onto a new [`Open`] keyed by `owner`, and returns the
    /// correlation for the caller to embed in the outgoing
    /// [`ProbeRequest`].
    ///
    /// **I5 enforcement is unconditional.** A double-open panics in
    /// both debug and release: the channel is the sole mint site, so a
    /// second open without a matching close is a programming error.
    /// Silent overwrite would orphan the prior probe's response (its
    /// correlation no longer matches anything in the map, so
    /// [`Self::close_if`] would reject it as stale even though the
    /// engine asked for it). Crashing loudly is the only correct
    /// outcome.
    ///
    /// **Counter saturation.** Inherited from
    /// [`MonotonicCounter::next`]: unconditional panic at
    /// [`u64::MAX`].
    #[must_use]
    pub(crate) fn open(&mut self, owner: ProbeOwner, kind: OpenKind) -> ProbeCorrelation {
        let correlation = self.counter.next();
        let prior = self.open.insert(owner, Open { correlation, kind });
        assert!(
            prior.is_none(),
            "I5 violated: opening probe channel for {owner:?} while already open \
             (prior = {prior:?}, attempted kind = {kind:?})",
        );
        correlation
    }

    /// Mint a fresh [`ProbeCorrelation`] from the monotonic counter
    /// **without** inserting an open-map entry. The counter is the
    /// engine-wide probe-id floor: state-resident probe slots mint
    /// through here so their correlations stay globally unique against
    /// every channel-minted one (same counter, one id space).
    ///
    /// **Counter saturation.** Inherited from
    /// [`MonotonicCounter::next`]: unconditional panic at [`u64::MAX`].
    #[must_use]
    pub(crate) fn mint(&mut self) -> ProbeCorrelation {
        self.counter.next()
    }

    /// Atomic check-and-take. Returns `Some(Open)` iff a channel is
    /// open for `owner` AND its correlation matches `received`;
    /// otherwise returns `None` and leaves any existing entry intact.
    ///
    /// The "leave intact on mismatch" semantics matter: a late
    /// response carrying a *stale* correlation must NOT displace the
    /// legitimately-outstanding entry. Production callers
    /// (`on_*_probe_response`) emit
    /// [`specter_core::Diagnostic::StaleProbeResponse`] on `None` and
    /// proceed.
    ///
    /// Implemented via [`std::collections::btree_map::Entry`] so the
    /// "find then maybe remove" decision happens under a single
    /// navigation rather than the prior peek → check → write
    /// three-step at every response site.
    pub(crate) fn close_if(
        &mut self,
        owner: ProbeOwner,
        received: ProbeCorrelation,
    ) -> Option<Open> {
        match self.open.entry(owner) {
            Entry::Occupied(e) if e.get().correlation == received => Some(e.remove()),
            Entry::Occupied(_) | Entry::Vacant(_) => None,
        }
    }

    /// Unconditional close. Returns the prior [`Open`] entry (if any)
    /// so the caller can use its fields for diagnostics. Cancel paths
    /// use this — by then the caller has decided the channel must die
    /// regardless of correlation (reap, force-cancel on
    /// event-during-Verifying, etc.).
    pub(crate) fn close(&mut self, owner: ProbeOwner) -> Option<Open> {
        self.open.remove(&owner)
    }

    /// Read the correlation an open channel holds, or `None` if
    /// closed. Used by reap-time invariant checks and by
    /// `on_descent_event`'s I5 short-circuit ("drop the event if a
    /// probe is already in flight").
    #[must_use]
    pub(crate) fn correlation_for(&self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        self.open.get(&owner).map(Open::correlation)
    }

    /// Read the [`OpenKind`] discriminant of an open channel, or
    /// `None` if closed. Used by:
    ///
    /// - `on_watch_op_rejected`'s Promoter proxy-purge: cancel only
    ///   when the in-flight probe targets the rejected proxy.
    /// - `release_promoter_proxy_claim`'s cancel-first
    ///   `debug_assert!`: assert the in-flight enumeration (if any)
    ///   targets some OTHER proxy of the same Promoter.
    ///
    /// Pattern-matching at the call site is the natural idiom; a
    /// dedicated `enumeration_target_for` helper would just hide the
    /// match.
    #[must_use]
    pub(crate) fn kind_for(&self, owner: ProbeOwner) -> Option<&OpenKind> {
        self.open.get(&owner).map(Open::kind)
    }

    /// Test-only counter prime. Saturation tests jump to `u64::MAX`
    /// without consuming the counter via repeated `open` calls.
    #[cfg(test)]
    pub(crate) fn prime_counter(&mut self, value: u64) {
        self.counter.prime(value);
    }

    /// Test-only counter peek for "fresh channel starts at zero"
    /// fixtures.
    #[cfg(test)]
    pub(crate) fn counter_peek(&self) -> u64 {
        self.counter.peek()
    }
}

impl Engine {
    /// The owner's in-flight probe correlation, or `None` if it has
    /// none. Single source of truth across both homes: the
    /// state-resident slot (Profile descent / verify / rebase, Promoter
    /// descent) when armed, else the channel (Promoter enumeration
    /// only). The two are disjoint per owner — a carrier is in exactly
    /// one state — so at most one home answers.
    ///
    /// `pub` (not `pub(crate)`) — the engine crate's `tests/`
    /// directory is an external crate from a Rust visibility
    /// standpoint, and ~35 integration-test call sites depend on this
    /// for probe-state assertions.
    #[must_use]
    pub fn pending_probe_for(&self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        let from_state = match owner {
            ProbeOwner::Profile(pid) => self
                .profiles
                .get(pid)
                .and_then(|p| p.state().probe_correlation()),
            ProbeOwner::Promoter(qid) => self
                .promoters
                .get(qid)
                .and_then(|q| q.state.probe_correlation()),
        };
        from_state.or_else(|| self.probe_channel.correlation_for(owner))
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
    /// Promoter enumeration has no state-resident route yet (its
    /// dispatch key lives on the channel), so a Promoter `Active`
    /// returns `None` here and that owner's handler still routes on the
    /// channel's [`OpenKind`].
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
            ProbeOwner::Promoter(qid) => match &self.promoters.get(qid)?.state {
                PromoterState::PrefixPending(_) => Some(ProbeRoute::Descent),
                PromoterState::Active { .. } => None,
            },
        }
    }

    /// Mint a fresh [`ProbeCorrelation`] off the engine-wide monotonic
    /// floor — the sole mint driver for every state-resident probe
    /// slot. Shares the channel's counter so slot- and channel-minted
    /// correlations never collide (one id space).
    #[must_use]
    pub(crate) fn mint_probe_correlation(&mut self) -> ProbeCorrelation {
        self.probe_channel.mint()
    }

    /// Consume the owner's in-flight probe and return its correlation
    /// (`None` if none was in flight). Disarms the state-resident slot
    /// (Profile descent / verify / rebase, Promoter descent) if armed;
    /// otherwise closes the channel entry (Promoter enumeration). The
    /// single consume primitive both the response path and the cancel
    /// path route through — disjoint per owner, so exactly one home
    /// fires.
    pub(crate) fn take_owner_probe(&mut self, owner: ProbeOwner) -> Option<ProbeCorrelation> {
        let from_slot = match owner {
            ProbeOwner::Profile(pid) => self.profiles.get_mut(pid).and_then(Profile::take_probe),
            ProbeOwner::Promoter(qid) => self
                .promoters
                .get_mut(qid)
                .and_then(|q| q.state.take_probe()),
        };
        from_slot.or_else(|| self.probe_channel.close(owner).map(|o| o.correlation()))
    }

    /// Consume the owner's in-flight probe and emit [`ProbeOp::Cancel`]
    /// iff one was in flight. The disarm/close *is* the consume, atomic
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

    /// Emit [`ProbeRequest::AnchorFile`]. Walker runs a single `lstat`
    /// and returns `ProbeOutcome::AnchorOk` / `Vanished` / `Failed`.
    ///
    /// `correlation` must be the value returned by the matching
    /// [`ProbeChannel::open`] (the caller's mint precedes this call
    /// within the same `&mut self` window). Associated function — no
    /// Engine-state dependency.
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
    /// `ResourceId` the engine probed) stays on `ProbeChannel`'s open
    /// kind or on the relevant `Profile` / burst state — the walker
    /// never needs the engine's `Tree`.
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
    /// `descent.current_prefix()` for Profile / Promoter descent,
    /// `OpenKind::PromoterEnumerating { target }` for Promoter
    /// enumeration.
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

#[cfg(test)]
mod tests {
    use super::{OpenKind, ProbeChannel};
    use crate::Engine;
    use specter_core::{
        ClassSet, ProbeCorrelation, ProbeOp, ProbeOwner, Profile, ProfileIdentity, ResourceId,
        ResourceRole, ScanConfig, StepOutput,
    };
    use std::time::Duration;

    const SETTLE: Duration = Duration::from_millis(100);
    const MAX_SETTLE: Duration = Duration::from_secs(6);

    /// Attach a fresh `Idle` Profile at a synthetic anchor, returning
    /// the engine, the new [`ProbeOwner`], and the anchor
    /// [`ResourceId`] (handy as a channel-mechanism `OpenKind` target).
    /// The Profile carries no Subs and no claims — purely a vehicle for
    /// exercising the channel state in isolation.
    fn fresh_engine_with_idle_profile() -> (Engine, ProbeOwner, ResourceId) {
        let mut e = Engine::new();
        let r = e.tree.ensure_root("anchor", ResourceRole::User);
        let pid = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r,
                ProfileIdentity {
                    config: ScanConfig::builder().build(),
                    max_settle: MAX_SETTLE,
                    events: ClassSet::EMPTY,
                },
                SETTLE,
                None,
            ),
        );
        (e, ProbeOwner::Profile(pid), r)
    }

    /// Open returns a fresh correlation; channel reports it on
    /// `correlation_for`.
    #[test]
    fn open_returns_correlation_and_records_kind() {
        let (mut e, owner, r) = fresh_engine_with_idle_profile();
        let corr = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
        assert_eq!(e.pending_probe_for(owner), Some(corr));
        assert_eq!(
            e.probe_channel.kind_for(owner),
            Some(&OpenKind::PromoterEnumerating { target: r }),
        );
    }

    /// I5: double-open panics. Unconditional `assert!` — survives
    /// release builds (deliberately distinct from the pre-Phase-3
    /// `debug_assert!`-gated regression check).
    #[test]
    #[should_panic(expected = "I5 violated")]
    fn open_panics_on_double_open() {
        let (mut e, owner, r) = fresh_engine_with_idle_profile();
        let _ = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
        let _ = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r }); // panics
    }

    /// Counter saturation — release-runnable. Pairs with the
    /// [`crate::counter::MonotonicCounter`] unit tests; this site test
    /// proves the channel routes through the counter at the `open`
    /// boundary rather than reimplementing the bump.
    #[test]
    #[should_panic(expected = "MonotonicCounter")]
    fn open_panics_on_counter_saturation() {
        let (mut e, owner, r) = fresh_engine_with_idle_profile();
        e.probe_channel.prime_counter(u64::MAX);
        let _ = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
    }

    /// `close_if` succeeds on matched correlation and returns the
    /// `Open` with the expected kind.
    #[test]
    fn close_if_matched_returns_open() {
        let (mut e, owner, r) = fresh_engine_with_idle_profile();
        let corr = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
        let open = e
            .probe_channel
            .close_if(owner, corr)
            .expect("matched correlation closes");
        assert_eq!(open.correlation(), corr);
        assert_eq!(open.kind(), &OpenKind::PromoterEnumerating { target: r });
        assert!(
            e.pending_probe_for(owner).is_none(),
            "channel closed post-close_if",
        );
    }

    /// `close_if` rejects mismatched correlation; the in-flight entry
    /// stays intact (a stale response must NOT displace a
    /// legitimately-outstanding probe).
    #[test]
    fn close_if_mismatch_preserves_entry() {
        let (mut e, owner, r) = fresh_engine_with_idle_profile();
        let corr = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
        let bogus = ProbeCorrelation::from(corr.as_u64() + 9_999);
        assert!(
            e.probe_channel.close_if(owner, bogus).is_none(),
            "mismatched correlation returns None",
        );
        assert_eq!(
            e.pending_probe_for(owner),
            Some(corr),
            "channel entry preserved",
        );
    }

    /// `close_if` on a closed channel is a clean `None`. No surprise.
    #[test]
    fn close_if_closed_returns_none() {
        let (mut e, owner, _r) = fresh_engine_with_idle_profile();
        let bogus = ProbeCorrelation::from(42);
        assert!(e.probe_channel.close_if(owner, bogus).is_none());
        assert!(e.pending_probe_for(owner).is_none());
    }

    /// `cancel_owner_probe` on closed channel = no-op. Load-bearing
    /// for `event_drives_batching` which invokes it on every event
    /// regardless of phase.
    #[test]
    fn cancel_owner_probe_idempotent_on_closed_channel() {
        let (mut e, owner, _r) = fresh_engine_with_idle_profile();
        assert!(e.pending_probe_for(owner).is_none());
        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner, &mut out);
        assert!(out.probe_ops.is_empty());
        assert!(e.pending_probe_for(owner).is_none());
    }

    /// `cancel_owner_probe` on open channel: single Cancel + close.
    #[test]
    fn cancel_owner_probe_emits_and_clears_on_open_channel() {
        let (mut e, owner, r) = fresh_engine_with_idle_profile();
        let corr = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
        assert_eq!(e.pending_probe_for(owner), Some(corr));
        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner, &mut out);
        assert_eq!(out.probe_ops.len(), 1);
        assert!(matches!(out.probe_ops[0], ProbeOp::Cancel { owner: o } if o == owner));
        assert!(e.pending_probe_for(owner).is_none());
    }

    /// Cancel is per-owner: closing one owner's channel doesn't touch
    /// another's. Cross-owner concurrency drives descent fan-out
    /// (multiple Pending Profiles awaiting siblings under one prefix).
    #[test]
    fn cancel_owner_probe_is_per_owner() {
        let mut e = Engine::new();
        let r1 = e.tree.ensure_root("a", ResourceRole::User);
        let r2 = e.tree.ensure_root("b", ResourceRole::User);
        let cfg = ScanConfig::builder().build();
        let pid1 = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r1,
                ProfileIdentity {
                    config: cfg.clone(),
                    max_settle: MAX_SETTLE,
                    events: ClassSet::EMPTY,
                },
                SETTLE,
                None,
            ),
        );
        let pid2 = e.profiles.attach(
            &mut e.tree,
            Profile::new(
                r2,
                ProfileIdentity {
                    config: cfg,
                    max_settle: MAX_SETTLE,
                    events: ClassSet::EMPTY,
                },
                SETTLE,
                None,
            ),
        );
        let owner1 = ProbeOwner::Profile(pid1);
        let owner2 = ProbeOwner::Profile(pid2);
        let c1 = e
            .probe_channel
            .open(owner1, OpenKind::PromoterEnumerating { target: r1 });
        let c2 = e
            .probe_channel
            .open(owner2, OpenKind::PromoterEnumerating { target: r2 });

        let mut out = StepOutput::default();
        e.cancel_owner_probe(owner1, &mut out);

        assert!(e.pending_probe_for(owner1).is_none());
        assert_eq!(e.pending_probe_for(owner2), Some(c2));
        assert_ne!(c1, c2);
    }

    /// `kind_for` round-trips the variant data — relied on by the
    /// proxy-purge call site to detect "in-flight enumeration of
    /// THIS proxy". The channel itself doesn't validate owner-kind
    /// affinity (mint discipline lives at call sites); a Promoter
    /// owner paired with `PromoterEnumerating` is the natural shape.
    #[test]
    fn kind_for_round_trips_promoter_enumerating_target() {
        let mut e = Engine::new();
        let r = e.tree.ensure_root("proxy", ResourceRole::User);
        let owner = ProbeOwner::Promoter(specter_core::PromoterId::default());
        let _ = e
            .probe_channel
            .open(owner, OpenKind::PromoterEnumerating { target: r });
        let kind = e.probe_channel.kind_for(owner).expect("channel open");
        assert!(
            matches!(kind, OpenKind::PromoterEnumerating { target } if *target == r),
            "kind carries the target ResourceId verbatim",
        );
    }

    /// Fresh channel reports a zero counter.
    #[test]
    fn default_channel_starts_at_zero_counter() {
        let c = ProbeChannel::default();
        assert_eq!(c.counter_peek(), 0);
        assert_eq!(
            c.kind_for(ProbeOwner::Profile(specter_core::ProfileId::default())),
            None,
        );
    }
}
