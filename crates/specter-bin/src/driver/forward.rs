//! Downstream dispatch — the terminal-consumer drain of a
//! [`StepOutput`], and the [`Diagnostic`] → tracing map.
//!
//! [`EngineDriver::forward`] ships every [`StepOutput`] onward:
//! `watch_ops` → `watch_ops_tx` with a `wake_handle.wake()` per
//! successful send (the bounded-channel deadlock the per-send wake
//! defends against is documented on the method); `probe_ops` →
//! `prober.submit` / `cancel`; `effects` → `effects_tx`;
//! `diagnostics` → [`log_diagnostic`]. `log_diagnostic` is the
//! per-variant hand-mapping of a [`Diagnostic`] to a tracing event;
//! its severities are the operator-facing catalogue.

use super::EngineDriver;
use specter_core::{Diagnostic, ProbeOp, StepOutput};

impl EngineDriver {
    /// Push a [`StepOutput`] to its downstream consumers.
    ///
    /// `watch_ops` queue to `watch_ops_tx` and `wake_handle.wake()` fires
    /// after **every** successful send. The wake-per-send protocol is
    /// load-bearing: `watch_ops_tx` is bounded(1024), and a single Seed
    /// burst against a large tree can produce 10k+ Watch ops in one
    /// `StepOutput`. With a "wake once at end of loop" rule, the engine
    /// would fill the channel, block on `Sender::send` at op 1025, and
    /// never reach the end-of-loop wake — leaving the watcher asleep in
    /// `kevent` forever. Wakes coalesce kernel-side via `EVFILT_USER`'s
    /// `EV_CLEAR`, so the per-send cost is one `kevent` syscall (~1µs)
    /// regardless of whether the watcher is awake. `probe_ops` dispatch
    /// directly to the prober per owner — [`StepOutput::into_parts`]
    /// yields them as an owner-keyed map (at most one op per owner *by
    /// construction*), the producer-side image of the prober's own
    /// `expected` map. So the prober's non-commuting `submit` /
    /// `cancel` never see a superseded same-owner pair: the shape
    /// collapses it before the wire. Probe *correctness* is the
    /// engine's — a stale or superseded response folds to
    /// `StaleProbeResponse` at its response gate, never this drain's.
    /// `effects` queue to `effects_tx`. `diagnostics` log per variant
    /// via [`log_diagnostic`].
    ///
    /// `Send` errors on disconnected channels are warn-logged and
    /// dropped — the only path here is a downstream-thread crash mid-
    /// shutdown. Takes `&self` because every downstream send is
    /// channel-based or trait-object dispatch (`Sender::send`,
    /// `Prober::submit`, `WakeHandle::wake`, `tracing::*`) — none
    /// requires `&mut self`.
    pub(in crate::driver) fn forward(&self, out: StepOutput) {
        // Terminal-consumer drain: `out` is already resealed (every
        // `StepOutput`-returning entry point sorts before returning), so
        // a single by-value destructure preserves the sort and the
        // dispatch order below without one clone per Effect.
        let (watch_ops, probe_ops, effects, diagnostics) = out.into_parts();

        for op in watch_ops {
            match self.sides.watch_ops_tx.send(op) {
                Ok(()) => self.wake_handle.wake(),
                Err(_) => tracing::warn!("watch_ops channel disconnected; dropping op"),
            }
        }

        for op in probe_ops.into_values() {
            match op {
                ProbeOp::Probe { request } => self.prober.submit(request),
                ProbeOp::Cancel { owner } => self.prober.cancel(owner),
            }
        }

        for eff in effects {
            if self.sides.effects_tx.send(eff).is_err() {
                tracing::warn!("effects channel disconnected; dropping effect");
            }
        }

        for diag in diagnostics {
            log_diagnostic(&diag);
        }
    }
}

/// Map a [`Diagnostic`] to a tracing event.
///
/// Most variants are `warn` (drops + race conditions). With auto-reload
/// landed, `EffectCompleteForUnknownSub` is `warn` too — the auto-reload
/// path makes the detach-during-effect race routine; engine bugs surface
/// via test assertions on the `Diagnostic::` variant rather than via log
/// severity. `ProfileReaped` and `ReapPendingCancelled` are `info`
/// (informational; the Profile was reaped — see `via` for the
/// trigger — or the deferred reap was pre-empted by a revival).
pub(in crate::driver) fn log_diagnostic(d: &Diagnostic) {
    match d {
        Diagnostic::StaleProbeResponse { owner, correlation } => tracing::warn!(
            ?owner,
            ?correlation,
            "stale probe response (state mismatch)"
        ),
        Diagnostic::StaleTimer { id } => tracing::warn!(?id, "stale timer expiration"),
        Diagnostic::EffectCompleteOutsideAwaiting { sub, profile } => tracing::warn!(
            ?sub,
            ?profile,
            "effect_complete arrived outside Awaiting (gate-deadline force-transition or anchor-loss); dropped",
        ),
        Diagnostic::EffectCompleteForUnknownSub { sub } => tracing::warn!(
            ?sub,
            "effect_complete for unknown Sub (hot-reload race or engine bug; dropped)",
        ),
        Diagnostic::DetachUnknownSub { sub } => tracing::warn!(
            ?sub,
            "detach for unknown Sub (hot-reload race or stale id; dropped)",
        ),
        Diagnostic::ConfigDiffUnknownSub { name } => tracing::info!(
            %name,
            "config reload removed a watch the engine never attached \
             (likely a prior path error); nothing to detach",
        ),
        Diagnostic::ConfigDiffUnknownPromoter { name } => tracing::info!(
            %name,
            "config reload removed a dynamic watch the engine never \
             attached (likely a prior path error); nothing to reap",
        ),
        Diagnostic::ProbeVanished { profile, intent } => {
            tracing::warn!(?profile, ?intent, "probe returned Vanished");
        }
        Diagnostic::ProbeFailed {
            profile,
            intent,
            errno,
        } => tracing::warn!(?profile, ?intent, errno, "probe failed"),
        Diagnostic::EventClassDropped {
            resource,
            event,
            profile,
        } => tracing::trace!(
            ?resource,
            ?event,
            ?profile,
            "fs event dropped (class not in profile.events)",
        ),
        Diagnostic::EventOnUnwatchedResource { resource } => {
            tracing::warn!(?resource, "FsEvent on unwatched resource (race; dropped)");
        }
        Diagnostic::EventNoConsumer { resource } => {
            // Benign: a watched resource (typically a `WatchRootParent`)
            // fired an event no Profile cared about this step. Logging at
            // TRACE so it doesn't pollute operator logs.
            tracing::trace!(
                ?resource,
                "FsEvent had no consumer (watched, but no covering Profile / descent / recovery)"
            );
        }
        Diagnostic::WatchOpRejected { resource, failure } => {
            tracing::warn!(
                ?resource,
                ?failure,
                errno = failure.errno(),
                "watch op rejected by sensor",
            );
        }
        Diagnostic::PendingPathProbeVanished { profile, prefix } => {
            tracing::warn!(?profile, ?prefix, "pending-path descent probe Vanished");
        }
        Diagnostic::PendingPathProbeFailed {
            profile,
            prefix,
            errno,
        } => tracing::warn!(
            ?profile,
            ?prefix,
            errno,
            "pending-path descent probe Failed",
        ),
        Diagnostic::ReapPendingCancelled { profile } => tracing::debug!(
            ?profile,
            "reap-pending Profile revived (fresh attach pre-empted deferred reap)",
        ),
        Diagnostic::ProfileReaped { profile, via } => {
            tracing::info!(?profile, ?via, "Profile reaped");
        }
        Diagnostic::ProfileClaimPurged {
            profile,
            claim,
            resource,
            failure,
        } => tracing::warn!(
            ?profile,
            ?claim,
            ?resource,
            ?failure,
            errno = failure.errno(),
            "profile claim purged (WatchOpRejected at claimed resource)",
        ),
        Diagnostic::PromoterClaimPurged {
            promoter,
            claim,
            resource,
            failure,
        } => tracing::warn!(
            ?promoter,
            ?claim,
            ?resource,
            ?failure,
            errno = failure.errno(),
            "promoter claim purged (WatchOpRejected at claimed resource)",
        ),
        Diagnostic::AttachPathInvalid { path, hint } => {
            tracing::error!(
                path = %path.display(),
                hint,
                "attach path invalid; request dropped",
            );
        }
        Diagnostic::AttachResourceStale { resource } => {
            tracing::warn!(
                ?resource,
                "attach resource stale (no live Tree slot); request dropped",
            );
        }
        Diagnostic::SpliceCrossedUncovered {
            profile,
            target,
            cause,
        } => tracing::warn!(
            ?profile,
            ?target,
            ?cause,
            "splice crossed uncovered subtree (graft contract violation; \
             prior view kept, response dropped)",
        ),
        Diagnostic::AnchorKindMismatch {
            profile,
            prior_kind,
            response_kind,
        } => tracing::error!(
            ?profile,
            ?prior_kind,
            ?response_kind,
            "probe response shape disagrees with cached Profile.kind \
             (walker contract violation; routing through anchor-loss recovery)",
        ),
        Diagnostic::EventAbsorbedByFireTail {
            profile,
            resource,
            event,
        } => tracing::trace!(
            ?profile,
            ?resource,
            ?event,
            "fs event absorbed by fire-tail (Awaiting/Rebasing); folded into post-fire rebase",
        ),
        Diagnostic::AwaitGateDeadlineElapsed {
            profile,
            outstanding,
        } => tracing::warn!(
            ?profile,
            outstanding,
            "await-gate deadline elapsed; force-transitioning to Rebasing (actuator likely hung)",
        ),
        Diagnostic::QuiescenceCeilingUnreadable {
            profile,
            first_unread,
            intent,
        } => tracing::warn!(
            ?profile,
            ?intent,
            ?first_unread,
            "quiescence ceiling unreadable (obligation chain frame mtime-skipped/degraded); \
             refused to fire/pin, burst finished to Idle (self-recovers if transient)",
        ),
        Diagnostic::RebaseCeilingStillChanging { profile, intent } => tracing::warn!(
            ?profile,
            ?intent,
            "post-fire rebase ceiling reached while the post-command tree was still changing; \
             pinned the freshest observation as baseline and finished the burst (a streaming \
             command, or settle shorter than its write cadence)",
        ),
        Diagnostic::RebaseCeilingUnreadable {
            profile,
            first_unread,
            intent,
        } => tracing::warn!(
            ?profile,
            ?intent,
            ?first_unread,
            "post-fire rebase ceiling reached on an unreadable response (obligation chain frame \
             mtime-skipped/degraded); refused to rebase blind, prior baseline kept, burst \
             finished to Idle (self-recovers if transient)",
        ),
        Diagnostic::SensorOverflow { scope } => tracing::warn!(
            ?scope,
            "sensor reported overflow (kernel queue dropped events); reseeding in-scope Profiles",
        ),
        Diagnostic::PromoterReseededForOverflow { promoter } => tracing::debug!(
            ?promoter,
            "promoter reseeded after sensor overflow (descent re-probed or proxies re-enumerated)",
        ),
        Diagnostic::PerFileDriftDroppedOnRecovery { profile } => tracing::warn!(
            ?profile,
            "per-file Sub's loss-window reactions dropped on recovery (no per-leaf survival witness)",
        ),
        Diagnostic::PerFileFireSkippedOnFreshSeed { profile } => tracing::info!(
            ?profile,
            "per-file Sub skipped on first-ever fire (fresh Profile has no baseline diff); \
             per-file reactions begin from the post-command baseline",
        ),
        Diagnostic::SubAttached {
            sub,
            name,
            source_promoter,
        } => match source_promoter {
            // Static (operator-declared) attach: high signal, low rate
            // (one per `[[watch]]` block per reload). INFO is the
            // operator-facing default per the catalog severity table.
            None => tracing::info!(?sub, %name, "sub attached"),
            // Dynamic (Promoter-spawned) attach: same lifecycle event
            // but emitted once per pattern match, which can be many
            // per enumeration. DEBUG keeps operator logs uncluttered;
            // `PromotionKindObserved` already carries the path-level
            // signal at the same severity.
            Some(promoter) => tracing::debug!(
                ?sub,
                %name,
                ?promoter,
                "dynamic sub attached (promoter-spawned)",
            ),
        },
        Diagnostic::PromoterAttached { promoter, name } => tracing::info!(
            ?promoter,
            %name,
            "promoter attached",
        ),
        Diagnostic::PromoterReaped { promoter } => tracing::info!(?promoter, "promoter reaped",),
        Diagnostic::PromoterDescentVanished { promoter, prefix } => tracing::debug!(
            ?promoter,
            ?prefix,
            "promoter descent / enumeration probe Vanished",
        ),
        Diagnostic::PromoterDescentFailed {
            promoter,
            prefix,
            errno,
        } => tracing::warn!(
            ?promoter,
            ?prefix,
            errno,
            "promoter descent / enumeration probe Failed",
        ),
        Diagnostic::PromotionKindObserved {
            promoter,
            path,
            kind,
        } => tracing::debug!(
            ?promoter,
            path = %path.display(),
            ?kind,
            "promoter promotion observed (dynamic Sub minted)",
        ),
        Diagnostic::PromoterFanoutThreshold { promoter, count } => tracing::warn!(
            ?promoter,
            count,
            "promoter fanout exceeded warning threshold (consider tightening pattern)",
        ),
        Diagnostic::PromoterProxyStaleEvent { promoter, resource } => tracing::debug!(
            ?promoter,
            ?resource,
            "fs event for promoter proxy that was unregistered earlier in step (stale; dropped)",
        ),
        Diagnostic::PromoterEnumerationVanished { promoter, proxy } => tracing::debug!(
            ?promoter,
            ?proxy,
            "promoter enumeration probe Vanished (proxy gone; subtree unwound)",
        ),
        Diagnostic::PromoterEnumerationFailed {
            promoter,
            proxy,
            errno,
        } => tracing::warn!(
            ?promoter,
            ?proxy,
            errno,
            "promoter enumeration probe Failed (retaining proxy state)",
        ),
        Diagnostic::DynamicSubReaped {
            promoter,
            sub,
            path,
        } => tracing::debug!(
            ?promoter,
            ?sub,
            path = %path.display(),
            "dynamic Sub reaped (anchor terminal — Profile all-dynamic teardown)",
        ),
        Diagnostic::InvalidBurstTransition {
            profile,
            helper,
            observed,
        } => tracing::warn!(
            ?profile,
            ?helper,
            ?observed,
            "burst lifecycle helper precondition failed (state-machine routing breach)",
        ),
    }
}
