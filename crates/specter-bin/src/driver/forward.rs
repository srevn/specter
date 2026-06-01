//! Downstream dispatch — the terminal-consumer drain of a
//! [`StepOutput`], the [`Diagnostic`] → tracing map, and the
//! diagnostic fan-out to operator-IPC subscribers.
//!
//! [`EngineDriver::forward`] ships every [`StepOutput`] onward:
//! `watch_ops` → [`crate::driver::Reactor::apply_watch_ops`] (inline
//! against the owned watcher; rejected ops queue as
//! [`Input::WatchOpRejected`] in the deferred-input replay queue);
//! `probe_ops` → `prober.submit` / `cancel`; `cancel_effects` +
//! `effects` → `actuator_io.effects_tx` (lifted to
//! [`EffectOp::Cancel`] / [`EffectOp::Submit`], cancels first so a
//! defensive same-step cancel + submit for one profile would kill
//! stale before spawn new); `diagnostics` → [`log_diagnostic`] AND
//! [`crate::driver::Hub::dispatch_to_subscribers`] (fan-out
//! to live IPC subscribers, with a single `SystemTime::now()` capture
//! per `StepOutput` so every subscriber sees byte-identical `at` for
//! the same emission).
//!
//! **No `shutdown_engine_rx` arm.** Signals dispatch inline on the
//! reactor thread via [`super::EngineDriver::dispatch_signal`], and a
//! wedged effects channel surfaces through `try_send` rather than
//! blocking the reactor mid-tick. Per-channel disconnect policy is
//! explicit at the call site; the rationale lives on the method.
//!
//! `log_diagnostic` is the per-variant hand-mapping of a [`Diagnostic`]
//! to a tracing event; its severities are the operator-facing catalogue.

use super::EngineDriver;
use crate::ipc::wire::WireTime;
use crossbeam::channel::TrySendError;
use specter_core::{Diagnostic, EffectOp, Input, ProbeOp, StepOutput, SubId};
use specter_sensor::FsWatcher;
use std::ops::ControlFlow;
use std::time::SystemTime;

impl<W: FsWatcher> EngineDriver<W> {
    /// Push a [`StepOutput`] to its downstream consumers.
    ///
    /// **`watch_ops` dispatch inline against the owned watcher.** The
    /// driver thread owns [`crate::driver::Reactor`]'s watcher
    /// directly — no channel, no consumer-side disconnect to race.
    /// Rejected ops surface as `(resource, failure)` pairs which we
    /// queue into [`super::EngineDriver::deferred_inputs`] as
    /// [`Input::WatchOpRejected`]; the next tick's
    /// `replay_deferred_inputs` runs each rejection through
    /// `engine.step` BEFORE the mio Poll re-blocks, so the engine's
    /// claim-purge fires this tick's cycle.
    ///
    /// **`effects_tx` is `try_send` with advisory drop on `Full`.**
    /// The engine's `gate_deadline` recovery contract covers a missed
    /// Submit identically to "actuator wedged" — both produce a
    /// `EffectComplete` that never arrives, so the burst force-
    /// transitions `Awaiting → Rebasing` on the same timer. Advisory
    /// drop on `Full` is therefore safe; the cost is a single missed
    /// effect, recovered through the same path operators already
    /// observe under actuator pressure. `Disconnected` remains terminal
    /// (actuator thread is gone — no recovery possible).
    ///
    /// **Probe ops dispatch directly.** [`StepOutput::into_parts`]
    /// yields an owner-keyed map (at most one op per owner by
    /// construction), the producer-side image of the prober's own
    /// `expected` map. So the prober's non-commuting `submit` /
    /// `cancel` never see a superseded same-owner pair: the shape
    /// collapses it before the wire. Probe *correctness* is the
    /// engine's — a stale or superseded response folds to
    /// `StaleProbeResponse` at its response gate, never this drain's.
    ///
    /// **Diagnostic fan-out lives on [`Self::forward_diagnostics`].**
    /// The wall-clock + per-subscriber byte-identical `at` contract
    /// is documented there; the call site below is one line so the
    /// terminal-drain ordering above is unobstructed by it.
    pub(super) fn forward(&mut self, out: StepOutput) -> ControlFlow<()> {
        // Terminal-consumer drain: `out` is already resealed (every
        // `StepOutput`-returning entry point sorts before returning), so
        // a single by-value destructure preserves the sort and the
        // dispatch order below without one clone per Effect.
        let (watch_ops, probe_ops, effects, cancel_effects, diagnostics) = out.into_parts();

        // Apply watch ops inline against the Reactor-owned watcher.
        // The rejected list is the producer side of the
        // `Input::WatchOpRejected` replay queue — every rejection runs
        // through `engine.step` on the next tick's
        // `replay_deferred_inputs` pass, so the engine's claim-purge
        // dispatch fires within one tick of the failure.
        for (resource, failure) in self.reactor.apply_watch_ops(&watch_ops) {
            self.deferred_inputs
                .push_back(Input::WatchOpRejected { resource, failure });
        }

        for op in probe_ops.into_values() {
            match op {
                ProbeOp::Probe { request } => self.prober.submit(request),
                ProbeOp::Cancel { owner } => self.prober.cancel(owner),
            }
        }

        // Cancel-effects dispatch BEFORE submits over the same
        // `effects_tx`. Defense in depth: by construction a same-step
        // cancel + submit for one profile cannot occur
        // (`handle_gate_deadline` emits no Effects; the post-fire
        // stable verdict path emits no cancel), but if a future
        // emission site ever introduced the cross-stream race, "kill
        // stale before spawn new" is the right ordering and is
        // structural here, not a documented convention. The single
        // `effects_tx` channel keeps the engine→actuator FIFO single,
        // so causal order between same-profile cancel and a later
        // submit (in a later step) is preserved automatically.
        for profile in cancel_effects {
            match self
                .actuator_io
                .effects_tx
                .try_send(EffectOp::Cancel { profile })
            {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    tracing::warn!(
                        ?profile,
                        "effects channel saturated; dropping cancel \
                         (engine gate_deadline recovers on the same path)",
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    tracing::error!("actuator disconnected; shutting down");
                    return ControlFlow::Break(());
                }
            }
        }

        for eff in effects {
            match self.actuator_io.effects_tx.try_send(EffectOp::Submit(eff)) {
                Ok(()) => {}
                Err(TrySendError::Full(_)) => {
                    tracing::warn!(
                        "effects channel saturated; dropping submit \
                         (engine gate_deadline recovers on the same path)",
                    );
                }
                Err(TrySendError::Disconnected(_)) => {
                    tracing::error!("actuator disconnected; shutting down");
                    return ControlFlow::Break(());
                }
            }
        }

        self.forward_diagnostics(&diagnostics);

        ControlFlow::Continue(())
    }

    /// Log each [`Diagnostic`] via tracing AND fan it out to live
    /// IPC subscribers through
    /// [`crate::driver::Hub::dispatch_to_subscribers`].
    ///
    /// **Wall-clock capture is once per call.** A single
    /// [`SystemTime::now`] threads into every `dispatch_to_subscribers`
    /// call — every subscriber sees byte-identical `at` for one engine
    /// emission, regardless of per-client delivery cadence. A slow
    /// client cannot rewrite history; a fast client cannot observe a
    /// slower client's `at`.
    ///
    /// **`WireTime` projection is also once per call.** The
    /// `humantime::format_rfc3339_seconds` allocation runs once here;
    /// every per-diag [`crate::ipc::wire::WireDiagnostic::from`]
    /// projection bumps the `Arc<str>` refcount instead of
    /// re-formatting. For a high-fanout `StepOutput` (e.g., a
    /// 256-fanout `PromoterFanoutThreshold` burst) this collapses N
    /// format calls into 1 + N atomic refcount bumps.
    ///
    /// **Empty-slice short-circuit.** Most ticks emit zero
    /// diagnostics; the early return keeps the [`SystemTime::now`]
    /// syscall and the `WireTime` projection off the common path.
    ///
    /// **Subscriber-empty short-circuit.** When no operator is
    /// subscribed ([`crate::driver::Hub::has_any_subscriber`] →
    /// `false`), only [`log_diagnostic`] runs per diag — the per-
    /// emission [`SystemTime::now`] / [`WireTime::from`] /
    /// [`diag_sub_id`] / `dispatch_to_subscribers` work is skipped
    /// entirely. For a `StepOutput` carrying N diags on a quiet
    /// daemon this collapses to N tracing emits plus one conn-map
    /// walk, rather than N times the same walk inside the dispatch
    /// path's defensive inner gate.
    fn forward_diagnostics(&mut self, diagnostics: &[Diagnostic]) {
        if diagnostics.is_empty() {
            return;
        }
        if !self.ipc.has_any_subscriber() {
            for diag in diagnostics {
                log_diagnostic(diag);
            }
            return;
        }
        let wall_now = SystemTime::now();
        let wire_at = WireTime::from(wall_now);
        for diag in diagnostics {
            log_diagnostic(diag);
            let diag_sub = diag_sub_id(diag);
            self.ipc
                .dispatch_to_subscribers(diag, wall_now, &wire_at, diag_sub);
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
///
/// These per-variant levels are also the client-side severity
/// catalogue: `specter tail` colours each line by the same judgment in
/// `crate::ipc::render::diag::severity` (`error!`→red, `warn!`→yellow,
/// `info!`/`debug!`/`trace!`→unstyled, with `SubFired` elevated to
/// green). Re-judging a variant means moving it in both places.
pub(super) fn log_diagnostic(d: &Diagnostic) {
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
        Diagnostic::ConfigDiffRebindFallbackAttach { name } => tracing::info!(
            %name,
            "config reload rebind found no live Sub (prior attach likely \
             failed); degrading to fresh attach",
        ),
        Diagnostic::ProbeVanished { profile, intent } => {
            tracing::warn!(?profile, ?intent, "probe returned Vanished");
        }
        Diagnostic::ProbeFailed {
            profile,
            intent,
            failure,
        } => tracing::warn!(
            ?profile,
            ?intent,
            ?failure,
            errno = failure.errno(),
            "probe failed",
        ),
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
            failure,
        } => tracing::warn!(
            ?profile,
            ?prefix,
            ?failure,
            errno = failure.errno(),
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
        Diagnostic::AwaitGateDeadlineForceRebasing {
            profile,
            outstanding,
        } => tracing::warn!(
            ?profile,
            outstanding,
            "await-gate deadline elapsed; force-transitioning to Rebasing (actuator likely hung)",
        ),
        Diagnostic::AwaitGateDeadlineReap {
            profile,
            outstanding,
        } => tracing::warn!(
            ?profile,
            outstanding,
            "await-gate deadline elapsed on zombie burst; \
             skipping rebase and reaping profile (actuator likely hung, sole sub detached)",
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
        Diagnostic::QuiescenceCeilingForcedDespiteChange { profile, intent } => tracing::warn!(
            ?profile,
            ?intent,
            "pre-fire burst deadline ceiling reached AND the hash channel observed concrete \
             change (prior ≠ response) at the last sample; fired against the freshest \
             observation anyway (tree visibly moving when the deadline expired)",
        ),
        Diagnostic::RebaseCeilingForced {
            profile,
            intent,
            observed_change,
        } => {
            if *observed_change {
                tracing::warn!(
                    ?profile,
                    ?intent,
                    "post-fire rebase ceiling reached AND the hash channel observed concrete \
                     change (prior ≠ response) at the last sample; pinned the freshest \
                     observation as baseline anyway (post-command tree visibly moving when \
                     the ceiling expired)",
                );
            } else {
                tracing::warn!(
                    ?profile,
                    ?intent,
                    "post-fire rebase ceiling reached without the hash channel confirming \
                     quiescence (samples agreed at the last read, the ceiling forced the \
                     first sample, or the channel was inactive); pinned the freshest \
                     observation as baseline and finished the burst",
                );
            }
        }
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
        Diagnostic::SubFired {
            sub,
            profile,
            count,
        } => tracing::info!(
            ?sub,
            ?profile,
            count,
            "sub fired (aggregated per emit_effects pass; SubtreeRoot count=1, \
             PerStableFile count=per-leaf matches)",
        ),
        Diagnostic::QuiescenceAbsorbed { profile } => tracing::info!(
            ?profile,
            "burst folded by an armed absorb window — baseline advanced, no fire \
             (expected replication absorbed)",
        ),
        Diagnostic::AbsorbArmed { profile, mode } => tracing::info!(
            ?profile,
            ?mode,
            "absorb window armed (next fireable burst folds instead of firing)",
        ),
        Diagnostic::SubDetached {
            sub,
            profile,
            reason,
        } => tracing::info!(?sub, ?profile, ?reason, "sub detached",),
        Diagnostic::SubRebound { sub } => tracing::info!(
            ?sub,
            "sub rebound (per-Sub fields updated in place; baseline preserved)",
        ),
        Diagnostic::RebindUnknownSub { sub } => tracing::warn!(
            ?sub,
            "rebind targeted an unknown Sub (dispatcher routing breach)",
        ),
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
            failure,
        } => tracing::warn!(
            ?promoter,
            ?prefix,
            ?failure,
            errno = failure.errno(),
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
            failure,
        } => tracing::warn!(
            ?promoter,
            ?proxy,
            ?failure,
            errno = failure.errno(),
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
        Diagnostic::WalkerContractViolated { owner } => tracing::error!(
            ?owner,
            "probe response payload shape contradicts the requested route \
             (walker contract violation; recovered route-appropriately)",
        ),
    }
}

/// Project a [`Diagnostic`] to the [`SubId`] it names, if any.
///
/// Called by [`EngineDriver::forward`] once per emitted diagnostic to
/// resolve the `diag_sub` filter argument
/// [`super::Hub::dispatch_to_subscribers`] consumes — the
/// projection lives on the engine-output side of the boundary, the
/// reactor module never names [`Diagnostic`] internals.
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
/// [`Diagnostic`] variant carrying a [`SubId`] that this function
/// silently projects to `None` would be a per-Sub `wait` bug; the
/// exhaustive `match` forces the author to pick a side at the point
/// of variant introduction.
pub(super) const fn diag_sub_id(d: &Diagnostic) -> Option<SubId> {
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
        | D::QuiescenceCeilingForcedDespiteChange { .. }
        | D::RebaseCeilingForced { .. }
        | D::RebaseCeilingUnreadable { .. }
        | D::SensorOverflow { .. }
        | D::PromoterReseededForOverflow { .. }
        | D::PerFileDriftDroppedOnRecovery { .. }
        | D::PerFileFireSkippedOnFreshSeed { .. }
        | D::QuiescenceAbsorbed { .. }
        | D::AbsorbArmed { .. }
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
        | D::InvalidBurstTransition { .. }
        | D::WalkerContractViolated { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::diag_sub_id;
    use compact_str::CompactString;
    use slotmap::KeyData;
    use specter_core::{
        BurstIntent, DetachReason, Diagnostic, ProbeCorrelation, ProbeOwner, ProfileId, SubId,
    };

    /// Mint a non-default [`SubId`] from a raw FFI value — the
    /// fan-out filter keys on `Some(sid)` vs `None`, so a slotmap
    /// default would be indistinguishable from an absent id.
    fn sid(raw: u64) -> SubId {
        SubId::from(KeyData::from_ffi(raw))
    }

    fn pid(raw: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(raw))
    }

    /// Every per-Sub [`Diagnostic`] variant projects to its `sub`.
    /// Pins the load-bearing arms of [`diag_sub_id`]; a regression
    /// here is a `wait <name>` bug.
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
    /// they reach unfiltered subscribers, never per-Sub filtered
    /// ones. Catches a future variant that carries a [`SubId`] but
    /// lands in the wrong arm.
    #[test]
    fn diag_sub_id_profile_keyed_returns_none() {
        let p = pid(0xAA);
        assert_eq!(
            diag_sub_id(&Diagnostic::ProfileReaped {
                profile: p,
                via: specter_core::ReapTrigger::Immediate,
            }),
            None
        );
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
