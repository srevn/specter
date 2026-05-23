//! Diagnostic fan-out to operator IPC subscribers.
//!
//! Every [`super::EngineDriver::forward`] call ships one
//! [`Diagnostic`] to `log_diagnostic` (operator log line) AND to
//! [`Broker::dispatch`] (fan-out to live subscribers). Single
//! emission, one wall-clock stamp per fanout pass — every subscriber
//! sees byte-identical `at` for the same engine emission, regardless
//! of their per-client delivery cadence.
//!
//! Lives on [`super::EngineDriver`] as a plain struct. There is no
//! `BrokerHandle`, no `Arc<Mutex<Broker>>`. Subscriber registration
//! lands as a `RequestPayload::Subscribe` arm in [`super::ipc`]; that
//! handler holds the unique `&mut self.broker` via the same
//! single-thread invariant `forward()` does. The two writer paths
//! cannot race because both are bodies of [`super::EngineDriver::tick`].
//!
//! # Add-before-ack ordering
//!
//! The Subscribe arm calls [`Broker::add_subscriber`] *before* it
//! `try_send`s the `SubscribeAck` through the reply channel. The
//! per-connection thread is blocked on `reply_rx.recv_timeout` until
//! the ack lands, so the broker holds the subscriber by the time the
//! client receives the ack line — no diagnostic emitted between add
//! and the client's first read can leak past the registration.

use crossbeam::channel::{Sender, TrySendError};
use specter_core::{Diagnostic, SubId};
use std::time::SystemTime;

use crate::ipc::wire::BrokerEvent;

/// Fan-out registry of operator-IPC subscribers.
///
/// Plain `Vec` storage: subscriber count is bounded by
/// [`crate::ipc::server::MAX_IPC_CONNS`] and lookups are linear
/// scans by design — broadcast is the dominant operation, and a
/// bounded subscriber count makes the linear walk indistinguishable
/// from any other constant-factor cost.
pub(super) struct Broker {
    subs: Vec<Subscriber>,
}

/// Per-subscriber state — the back-pressure marker accumulator, the
/// per-Sub filter, and the channel the per-conn thread reads from.
struct Subscriber {
    tx: Sender<BrokerEvent>,
    /// Accumulator for the per-subscriber back-pressure marker.
    /// Increments on every dropped `try_send`; flushed lazily on the
    /// next successful send via a [`BrokerEvent::Missed`] emitted
    /// *before* the next `Diag`. `saturating_add` guards the
    /// `u32`-overflow case (practically unreachable; a wedged
    /// subscriber dropping `2^32` events would hit the disconnect
    /// path long prior).
    missed: u32,
    /// `Some(sid)` ⇒ deliver only events that name `sid` (per-Sub
    /// `wait` subscription); `None` ⇒ unfiltered (the `tail` shape).
    /// Resolved server-side at Subscribe-arm dispatch (the
    /// `name → SubId` lookup happens once, on the driver thread, so
    /// a typo / race never reaches the broker).
    filter_sub: Option<SubId>,
}

impl Broker {
    /// Empty broker — no subscribers, no allocations.
    pub(super) const fn new() -> Self {
        Self { subs: Vec::new() }
    }

    /// Register a fresh subscriber. Sole call site:
    /// `EngineDriver::handle_ipc`'s `RequestPayload::Subscribe` arm,
    /// *before* the corresponding `SubscribeAck` is sent through
    /// the reply channel — see the module rustdoc for the
    /// add-before-ack ordering contract.
    pub(super) fn add_subscriber(&mut self, tx: Sender<BrokerEvent>, filter_sub: Option<SubId>) {
        self.subs.push(Subscriber {
            tx,
            missed: 0,
            filter_sub,
        });
    }

    /// One emission → fan out to every live subscriber → GC the
    /// dead. See [`Subscriber::step`] for the per-subscriber state
    /// machine.
    ///
    /// Short-circuits on the no-subscriber path so the `Diagnostic`
    /// clone never runs in production deploys that have no operator
    /// tail attached.
    pub(super) fn dispatch(&mut self, diag: &Diagnostic, at: SystemTime) {
        if self.subs.is_empty() {
            return;
        }
        let diag_sub = diag_sub_id(diag);
        self.subs.retain_mut(|s| s.step(diag, diag_sub, at));
    }

    /// Live subscriber count — surfaces in unit tests that pin the
    /// GC-on-disconnect contract. Not exposed beyond the driver.
    #[cfg(test)]
    pub(super) const fn len(&self) -> usize {
        self.subs.len()
    }
}

impl Default for Broker {
    fn default() -> Self {
        Self::new()
    }
}

impl std::fmt::Debug for Broker {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Broker")
            .field("subscriber_count", &self.subs.len())
            .finish()
    }
}

impl Subscriber {
    /// Per-subscriber state machine — one transition per dispatch call.
    ///
    /// Returns `true` to keep the subscriber in the registry, `false`
    /// to GC it (the channel is disconnected; the per-conn thread is
    /// gone).
    ///
    /// Three branches:
    ///
    /// 1. **Filter**. A per-Sub subscription drops events that don't
    ///    name `want`. A Profile-keyed event (no `SubId` on the diag)
    ///    reaches every unfiltered subscriber, never a filtered one —
    ///    by design, `wait <name>` is per-Sub.
    /// 2. **Flush pending `Missed`**. The back-pressure marker is the
    ///    operator's "I missed N events before this point" signal, so
    ///    it must precede the next `Diag` in causal order. A still-full
    ///    channel keeps the subscriber alive but defers the flush
    ///    (`missed` carries forward); a disconnect GCs.
    /// 3. **The `Diag`**. Same disconnect/full semantics, but a full
    ///    send increments `missed` for the next pass to flush.
    fn step(&mut self, diag: &Diagnostic, diag_sub: Option<SubId>, at: SystemTime) -> bool {
        if let Some(want) = self.filter_sub
            && diag_sub != Some(want)
        {
            return true;
        }

        if self.missed > 0 {
            match self.tx.try_send(BrokerEvent::Missed {
                count: self.missed,
                at,
            }) {
                Ok(()) => self.missed = 0,
                Err(TrySendError::Full(_)) => {
                    // Still backed up: the pending Missed marker has not
                    // delivered, and this dispatch's Diag will also not
                    // reach the subscriber. Count THIS dispatch toward
                    // `missed` so the eventual flush reflects every
                    // dropped event, not just the first one. Without
                    // this bump, a sustained back-pressure window
                    // under-counts by one per dispatch — the marker would
                    // lie about how many events were lost.
                    self.missed = self.missed.saturating_add(1);
                    return true;
                }
                Err(TrySendError::Disconnected(_)) => return false,
            }
        }

        match self.tx.try_send(BrokerEvent::Diag {
            diag: diag.clone(),
            at,
        }) {
            Ok(()) => true,
            Err(TrySendError::Full(_)) => {
                self.missed = self.missed.saturating_add(1);
                true
            }
            Err(TrySendError::Disconnected(_)) => false,
        }
    }
}

/// Project a [`Diagnostic`] to the [`SubId`] it names, if any.
///
/// Total over the [`Diagnostic`] enum — a new core variant is a
/// compile error here (the exhaustive `match` is the structural
/// wall, same discipline as
/// [`crate::ipc::wire::WireDiagnostic::from`]).
///
/// Per-Sub variants project to their `sub`. Profile-keyed variants
/// (`ProfileReaped`, `ReapPendingCancelled`, etc.) return `None`
/// and reach unfiltered subscribers only.
///
/// The verbose `None`-arm enumeration is deliberate: a future
/// `Diagnostic` variant carrying a `SubId` that this function
/// silently projects to `None` would be a per-Sub `wait` bug; the
/// exhaustive `match` forces the author to pick a side at the point
/// of variant introduction.
const fn diag_sub_id(d: &Diagnostic) -> Option<SubId> {
    use Diagnostic as D;
    match d {
        D::SubAttached { sub, .. }
        | D::SubFired { sub, .. }
        | D::SubDetached { sub, .. }
        | D::SubRebound { sub }
        | D::DetachUnknownSub { sub }
        | D::RebindUnknownSub { sub }
        | D::EffectCompleteForUnknownSub { sub }
        | D::EffectCompleteOutsideAwaiting { sub, .. } => Some(*sub),

        D::StaleProbeResponse { .. }
        | D::StaleTimer { .. }
        | D::ConfigDiffUnknownSub { .. }
        | D::ConfigDiffUnknownPromoter { .. }
        | D::ConfigDiffRebindFallbackAttach { .. }
        | D::ProbeVanished { .. }
        | D::ProbeFailed { .. }
        | D::EventClassDropped { .. }
        | D::EventOnUnwatchedResource { .. }
        | D::EventNoConsumer { .. }
        | D::WatchOpRejected { .. }
        | D::PendingPathProbeVanished { .. }
        | D::PendingPathProbeFailed { .. }
        | D::ReapPendingCancelled { .. }
        | D::ProfileReaped { .. }
        | D::ProfileClaimPurged { .. }
        | D::PromoterClaimPurged { .. }
        | D::AttachPathInvalid { .. }
        | D::AttachResourceStale { .. }
        | D::AnchorKindMismatch { .. }
        | D::SpliceCrossedUncovered { .. }
        | D::EventAbsorbedByFireTail { .. }
        | D::AwaitGateDeadlineForceRebasing { .. }
        | D::AwaitGateDeadlineReap { .. }
        | D::QuiescenceCeilingUnreadable { .. }
        | D::RebaseCeilingStillChanging { .. }
        | D::RebaseCeilingUnreadable { .. }
        | D::SensorOverflow { .. }
        | D::PromoterReseededForOverflow { .. }
        | D::PerFileDriftDroppedOnRecovery { .. }
        | D::PerFileFireSkippedOnFreshSeed { .. }
        | D::PromoterAttached { .. }
        | D::PromoterReaped { .. }
        | D::PromoterDescentVanished { .. }
        | D::PromoterDescentFailed { .. }
        | D::PromotionKindObserved { .. }
        | D::PromoterFanoutThreshold { .. }
        | D::PromoterProxyStaleEvent { .. }
        | D::PromoterEnumerationVanished { .. }
        | D::PromoterEnumerationFailed { .. }
        | D::DynamicSubReaped { .. }
        | D::InvalidBurstTransition { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::{Broker, diag_sub_id};
    use crate::ipc::wire::BrokerEvent;
    use compact_str::CompactString;
    use crossbeam::channel::bounded;
    use slotmap::KeyData;
    use specter_core::{
        BurstIntent, DetachReason, Diagnostic, ProbeCorrelation, ProbeOwner, ProfileId,
        ReapTrigger, SubId,
    };
    use std::time::SystemTime;

    /// Mint a non-default `SubId` from a raw FFI value — the broker's
    /// dispatch logic keys on `Some(sid)` vs `None`, so a slotmap
    /// default would be indistinguishable from an absent id.
    fn sid(raw: u64) -> SubId {
        SubId::from(KeyData::from_ffi(raw))
    }

    fn pid(raw: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(raw))
    }

    /// A per-Sub diagnostic — `SubFired` is the dominant case operators
    /// `wait` and `tail` against. Carries `sub` so the filter machinery
    /// has something to match on.
    fn sub_fired_for(s: SubId) -> Diagnostic {
        Diagnostic::SubFired {
            sub: s,
            profile: pid(0xAA),
            count: 1,
        }
    }

    /// A Profile-keyed diagnostic — `ProfileReaped` is the cleanest
    /// witness because no per-Sub `wait` should ever match it.
    fn profile_reaped() -> Diagnostic {
        Diagnostic::ProfileReaped {
            profile: pid(0xAA),
            via: ReapTrigger::Immediate,
        }
    }

    #[test]
    fn dispatch_no_subscribers_is_noop() {
        let mut broker = Broker::new();
        // No panic, no allocation — short-circuits before the diag clone.
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now());
        assert_eq!(broker.len(), 0);
    }

    #[test]
    fn dispatch_fans_out_to_every_subscriber() {
        let mut broker = Broker::new();
        let (tx_a, rx_a) = bounded::<BrokerEvent>(8);
        let (tx_b, rx_b) = bounded::<BrokerEvent>(8);
        broker.add_subscriber(tx_a, None);
        broker.add_subscriber(tx_b, None);

        let at = SystemTime::now();
        broker.dispatch(&sub_fired_for(sid(1)), at);

        // Same `(diag, at)` reaches both.
        let ev_a = rx_a.try_recv().expect("subscriber A receives");
        let ev_b = rx_b.try_recv().expect("subscriber B receives");
        match (&ev_a, &ev_b) {
            (BrokerEvent::Diag { at: at_a, .. }, BrokerEvent::Diag { at: at_b, .. }) => {
                assert_eq!(at_a, at_b, "same wall-clock at fanout");
                assert_eq!(*at_a, at);
            }
            other => panic!("expected two Diag events, got {other:?}"),
        }
    }

    #[test]
    fn dispatch_filter_sub_drops_unmatched() {
        let mut broker = Broker::new();
        let (tx, rx) = bounded::<BrokerEvent>(8);
        let want = sid(1);
        let other = sid(2);
        broker.add_subscriber(tx, Some(want));

        // Two emissions: one matches `want`, one doesn't.
        broker.dispatch(&sub_fired_for(other), SystemTime::now());
        broker.dispatch(&sub_fired_for(want), SystemTime::now());

        let ev = rx.try_recv().expect("matching event arrives");
        match ev {
            BrokerEvent::Diag {
                diag: Diagnostic::SubFired { sub, .. },
                ..
            } => assert_eq!(sub, want, "only the matching sub passes the filter"),
            other => panic!("expected Diag(SubFired), got {other:?}"),
        }
        assert!(
            rx.try_recv().is_err(),
            "non-matching event must NOT reach the subscriber",
        );
    }

    #[test]
    fn dispatch_filter_sub_drops_profile_keyed() {
        let mut broker = Broker::new();
        let (tx, rx) = bounded::<BrokerEvent>(8);
        broker.add_subscriber(tx, Some(sid(1)));

        // Profile-keyed events have no `SubId` — filtered subscribers
        // never see them. `wait <name>` is per-Sub by design.
        broker.dispatch(&profile_reaped(), SystemTime::now());
        assert!(
            rx.try_recv().is_err(),
            "ProfileReaped must not reach a per-Sub-filtered subscriber",
        );
    }

    #[test]
    fn dispatch_full_channel_accumulates_missed() {
        let mut broker = Broker::new();
        // `bounded(1)`: the first `Diag` fills the slot; subsequent
        // sends fail with `Full` and bump `missed`.
        let (tx, rx) = bounded::<BrokerEvent>(1);
        broker.add_subscriber(tx, None);

        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now());
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now());
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now());

        // One Diag in the slot; the broker is still tracking the
        // subscriber (not GC'd) and `missed` is non-zero internally.
        assert!(matches!(rx.try_recv(), Ok(BrokerEvent::Diag { .. })));
        assert!(
            rx.try_recv().is_err(),
            "no further events while the receiver hasn't drained",
        );
        assert_eq!(broker.len(), 1, "subscriber kept under back-pressure");
    }

    #[test]
    fn dispatch_flushes_missed_before_next_diag() {
        let mut broker = Broker::new();
        let (tx, rx) = bounded::<BrokerEvent>(1);
        broker.add_subscriber(tx, None);

        // Saturate, accumulate `missed`, drain, dispatch again.
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now());
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now()); // dropped → missed=1
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now()); // dropped → missed=2

        // Drain the first Diag so the channel has room.
        assert!(matches!(rx.try_recv(), Ok(BrokerEvent::Diag { .. })));

        // Next dispatch flushes `Missed` first, then the new `Diag`.
        let at = SystemTime::now();
        broker.dispatch(&sub_fired_for(sid(1)), at);
        match rx.try_recv().expect("Missed flushed first") {
            BrokerEvent::Missed { count, at: at_m } => {
                assert_eq!(count, 2, "accumulated missed count");
                assert_eq!(at_m, at, "Missed at == emission at");
            }
            other @ BrokerEvent::Diag { .. } => panic!("expected Missed first, got {other:?}"),
        }
        // The next slot's Diag landed only if the channel had room AFTER
        // Missed. Drain — both can't coexist in a bounded(1) slot, so
        // the Diag may be dropped, which the next pass would resurrect
        // via missed=1. The structural contract pinned here is "Missed
        // comes first in causal order"; the bounded(1) starvation
        // pattern is exercised in `dispatch_full_channel_accumulates_missed`.
        let _ = rx.try_recv();
    }

    #[test]
    fn dispatch_gc_drops_disconnected() {
        let mut broker = Broker::new();
        let (tx, rx) = bounded::<BrokerEvent>(8);
        broker.add_subscriber(tx, None);
        assert_eq!(broker.len(), 1);

        // Client side closes — the broker observes `Disconnected` on
        // the next dispatch and GCs the entry.
        drop(rx);
        broker.dispatch(&sub_fired_for(sid(1)), SystemTime::now());
        assert_eq!(broker.len(), 0, "GC'd the disconnected subscriber");
    }

    /// A per-Sub `Diagnostic` projects to its `sub`. Pins the
    /// load-bearing arms of `diag_sub_id`; a regression here is a
    /// `wait <name>` bug.
    #[test]
    fn diag_sub_id_per_sub_variants() {
        let s = sid(1);
        let p = pid(0xAA);
        assert_eq!(
            diag_sub_id(&Diagnostic::SubAttached {
                sub: s,
                name: CompactString::const_new("x"),
                source_promoter: None,
            }),
            Some(s)
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::SubFired {
                sub: s,
                profile: p,
                count: 1
            }),
            Some(s)
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::SubDetached {
                sub: s,
                profile: p,
                reason: DetachReason::IpcDisabled,
            }),
            Some(s)
        );
        assert_eq!(diag_sub_id(&Diagnostic::SubRebound { sub: s }), Some(s));
        assert_eq!(
            diag_sub_id(&Diagnostic::DetachUnknownSub { sub: s }),
            Some(s)
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::RebindUnknownSub { sub: s }),
            Some(s)
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::EffectCompleteForUnknownSub { sub: s }),
            Some(s)
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::EffectCompleteOutsideAwaiting { sub: s, profile: p }),
            Some(s)
        );
    }

    /// Profile-keyed and metadata-only variants project to `None` —
    /// they reach unfiltered subscribers, never per-Sub filtered ones.
    #[test]
    fn diag_sub_id_profile_keyed_returns_none() {
        let p = pid(0xAA);
        assert_eq!(diag_sub_id(&profile_reaped()), None);
        assert_eq!(
            diag_sub_id(&Diagnostic::ReapPendingCancelled { profile: p }),
            None
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::ProbeVanished {
                profile: p,
                intent: BurstIntent::Standard,
            }),
            None
        );
        assert_eq!(
            diag_sub_id(&Diagnostic::StaleProbeResponse {
                owner: ProbeOwner::Profile(p),
                correlation: ProbeCorrelation::from(7),
            }),
            None
        );
    }
}
