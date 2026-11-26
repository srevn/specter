//! Downstream dispatch — the terminal-consumer drain of a [`StepOutput`], the [`Diagnostic`] →
//! tracing map, and the diagnostic fan-out to operator-IPC subscribers.
//!
//! [`EngineDriver::forward`] ships every [`StepOutput`] onward: `watch_ops` →
//! [`crate::driver::Reactor::apply_watch_ops`] (inline against the owned watcher; rejected ops queue
//! as [`Input::WatchOpRejected`] in the deferred-input replay queue); `probe_ops` → `prober.submit`
//! / `cancel`; `cancel_effects` + `effects` → `actuator_io.effects_tx` (lifted to
//! [`EffectOp::Cancel`] / [`EffectOp::Submit`], cancels first so a defensive same-step cancel +
//! submit for one profile would kill stale before spawn new); `diagnostics` → [`log_diagnostic`]
//! AND [`crate::driver::Hub::dispatch_to_subscribers`] (fan-out to live IPC subscribers, with a
//! single `SystemTime::now()` capture per `StepOutput` so every subscriber sees byte-identical `at`
//! for the same emission).
//!
//! **No `shutdown_engine_rx` arm.** Signals dispatch inline on the reactor thread via
//! [`super::EngineDriver::dispatch_signal`], and a wedged effects channel surfaces through `try_send`
//! rather than blocking the reactor mid-tick. Per-channel disconnect policy is explicit at the call
//! site; the rationale lives on the method.
//!
//! `log_diagnostic` is the per-variant hand-mapping of a [`Diagnostic`] to a tracing event; its
//! severities are the operator-facing catalogue.

use super::EngineDriver;
use super::ipc::conns::DiagSub;
use crate::ipc::wire::WireTime;
use crossbeam::channel::TrySendError;
use specter_core::{Diagnostic, EffectOp, Input, ProbeOp, StepOutput};
use specter_sensor::FsWatcher;
use std::ops::ControlFlow;
use std::time::SystemTime;

impl<W: FsWatcher> EngineDriver<W> {
    /// Push a [`StepOutput`] to its downstream consumers.
    ///
    /// **`watch_ops` dispatch inline against the owned watcher.** The driver thread owns
    /// [`crate::driver::Reactor`]'s watcher directly — no channel, no consumer-side disconnect to
    /// race. Rejected ops surface as `(resource, failure)` pairs which we queue into
    /// [`super::EngineDriver::deferred_inputs`] as [`Input::WatchOpRejected`]; the next tick's
    /// `replay_deferred_inputs` runs each rejection through `engine.step` BEFORE the mio Poll
    /// re-blocks, so the engine's claim-purge fires this tick's cycle.
    ///
    /// **`effects_tx` is `try_send` with advisory drop on `Full`.** The engine's `gate_deadline`
    /// recovery contract covers a missed Submit identically to "actuator wedged" — both produce a
    /// `EffectComplete` that never arrives, so the burst force-transitions `Awaiting → Rebasing` on
    /// the same timer. Advisory drop on `Full` is therefore safe; the cost is a single missed effect,
    /// recovered through the same path operators already observe under actuator pressure.
    /// `Disconnected` remains terminal (actuator thread is gone — no recovery possible).
    ///
    /// **Probe ops dispatch directly.** [`StepOutput::into_parts`] yields an owner-keyed map (at
    /// most one op per owner by construction), the producer-side image of the prober's own `expected`
    /// map. So the prober's non-commuting `submit` / `cancel` never see a superseded same-owner pair:
    /// the shape collapses it before the wire. Probe *correctness* is the engine's — a stale or
    /// superseded response folds to `StaleProbeResponse` at its response gate, never this drain's.
    ///
    /// **Diagnostic fan-out lives on [`Self::forward_diagnostics`].** The wall-clock + per-subscriber
    /// byte-identical `at` contract is documented there; the call site below is one line so the
    /// terminal-drain ordering above is unobstructed by it.
    pub(super) fn forward(&mut self, out: StepOutput) -> ControlFlow<()> {
        // Terminal-consumer drain: `out` is already resealed (every `StepOutput`-returning entry
        // point sorts before returning), so a single by-value destructure preserves the sort and
        // the dispatch order below without one clone per Effect.
        let (watch_ops, probe_ops, effects, cancel_effects, diagnostics) = out.into_parts();

        // Apply watch ops inline against the Reactor-owned watcher. The rejected list is the producer
        // side of the `Input::WatchOpRejected` replay queue — every rejection runs through
        // `engine.step` on the next tick's `replay_deferred_inputs` pass, so the engine's claim-purge
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

        // Cancel-effects dispatch BEFORE submits over the same `effects_tx`. Defense in depth: by
        // construction a same-step cancel + submit for one profile cannot occur
        // (`handle_gate_deadline` emits no Effects; the post-fire stable verdict path emits no
        // cancel), but if a future emission site ever introduced the cross-stream race, "kill stale
        // before spawn new" is the right ordering and is structural here, not a documented
        // convention. The single `effects_tx` channel keeps the engine→actuator FIFO single, so
        // causal order between same-profile cancel and a later submit (in a later step) is preserved
        // automatically.
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

    /// Log each [`Diagnostic`] via tracing AND fan it out to live IPC subscribers through
    /// [`crate::driver::Hub::dispatch_to_subscribers`].
    ///
    /// **Wall-clock capture is once per call.** A single [`SystemTime::now`] threads into every
    /// `dispatch_to_subscribers` call — every subscriber sees byte-identical `at` for one engine
    /// emission, regardless of per-client delivery cadence. A slow client cannot rewrite history; a
    /// fast client cannot observe a slower client's `at`.
    ///
    /// **`WireTime` projection is also once per call.** The RFC 3339 render allocation runs once
    /// here; every per-diag [`crate::ipc::wire::WireDiagnostic::from`] projection bumps the
    /// `Arc<str>` refcount instead of re-formatting. For a high-fanout `StepOutput` (e.g., a
    /// 256-fanout `DiscoveryMinted` burst) this collapses N format calls into 1 + N atomic refcount
    /// bumps.
    ///
    /// **Empty-slice short-circuit.** Most ticks emit zero diagnostics; the early return keeps the
    /// [`SystemTime::now`] syscall and the `WireTime` projection off the common path.
    ///
    /// **Subscriber-empty short-circuit.** When no operator is subscribed
    /// ([`crate::driver::Hub::has_any_subscriber`] → `false`), only [`log_diagnostic`] runs per diag
    /// — the per-emission [`SystemTime::now`] / [`WireTime::from`] / [`diag_sub`] /
    /// `dispatch_to_subscribers` work is skipped entirely. For a `StepOutput` carrying N diags on a
    /// quiet daemon this collapses to N tracing emits plus one conn-map walk, rather than N times
    /// the same walk inside the dispatch path's defensive inner gate.
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
            self.ipc
                .dispatch_to_subscribers(diag, wall_now, &wire_at, diag_sub(diag));
        }
    }
}

/// Map a [`Diagnostic`] to a tracing event.
///
/// Most variants are `warn` (drops + race conditions). With auto-reload landed,
/// `EffectCompleteForUnknownSub` is `warn` too — the auto-reload path makes the detach-during-effect
/// race routine; engine bugs surface via test assertions on the `Diagnostic::` variant rather than
/// via log severity. `ProfileReaped` and `ReapPendingCancelled` are `info` (informational; the
/// Profile was reaped — see `via` for the trigger — or the deferred reap was pre-empted by a
/// revival). `StaleProbeResponse` is `debug`: the response gate fires it for designed supersedes
/// (an overflow reseed, a post-cancel arrival, a fresh mint), so `warn` would misread a routine,
/// self-fencing race as an operator-actionable fault.
///
/// These per-variant levels are also the client-side severity catalogue: `specter tail` colours each
/// line by the same judgment in `crate::ipc::render::diag::severity` (`error!`→red, `warn!`→yellow,
/// `info!`/`debug!`/`trace!`→unstyled, with `SubFired` elevated to green). Re-judging a variant means
/// moving it in both places.
pub(super) fn log_diagnostic(d: &Diagnostic) {
    match d {
        Diagnostic::StaleProbeResponse { owner, correlation } => tracing::debug!(
            ?owner,
            ?correlation,
            "stale probe response (state mismatch)"
        ),
        Diagnostic::StaleTimer { id } => tracing::warn!(?id, "stale timer expiration"),
        Diagnostic::EffectCompleteOutsideAwaiting {
            sub,
            profile,
            minted_by,
        } => tracing::warn!(
            ?sub,
            ?profile,
            ?minted_by,
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
        Diagnostic::EventOutsideProofObject {
            resource,
            event,
            profile,
        } => tracing::trace!(
            ?resource,
            ?event,
            ?profile,
            "fs event dropped (cannot move the profile's proof object)",
        ),
        Diagnostic::EventOnUnwatchedResource { resource } => {
            tracing::warn!(?resource, "FsEvent on unwatched resource (race; dropped)");
        }
        Diagnostic::EventNoConsumer { resource } => {
            // Benign: a watched resource (typically a `WatchRootParent`) fired an event no Profile
            // cared about this step. Logging at TRACE so it doesn't pollute operator logs.
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
        Diagnostic::PendingPathAwaitingSegment {
            profile,
            prefix,
            segment,
        } => tracing::debug!(
            ?profile,
            ?prefix,
            %segment,
            "recovery descent parked: awaiting segment under prefix",
        ),
        Diagnostic::PendingPathRetriesExhausted {
            profile,
            prefix,
            retries,
            errno,
        } => tracing::warn!(
            ?profile,
            ?prefix,
            retries,
            errno,
            "pending-path descent gave up retrying a signal-bearing probe; \
             awaiting a future prefix event or overflow",
        ),
        Diagnostic::PendingPathMaterialized { profile, anchor } => tracing::info!(
            ?profile,
            ?anchor,
            "pending-path descent materialized its anchor (recovery complete; baseline pending)",
        ),
        Diagnostic::ReapPendingCancelled { profile } => tracing::debug!(
            ?profile,
            "reap-pending Profile revived (fresh attach pre-empted deferred reap)",
        ),
        Diagnostic::ProfileReaped { profile, via } => {
            tracing::info!(?profile, ?via, "Profile reaped");
        }
        Diagnostic::ProfileParked { profile, recovery } => tracing::warn!(
            ?profile,
            ?recovery,
            "Profile parked (anchorless; recovery via parent/anchor event, overflow, or re-attach)",
        ),
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
            "quiescence ceiling unreadable (the walk could not read first_unread, a directory it \
             had cause to read); refused to fire/pin, burst finished to Idle (self-recovers once \
             readable, or excluded)",
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
            "post-fire rebase ceiling reached on an unreadable response (the walk could not read \
             first_unread, a directory it had cause to read); refused to rebase blind, prior \
             baseline kept, burst finished to Idle (self-recovers once readable, or excluded)",
        ),
        Diagnostic::ChangeOutsideEventMask {
            profile,
            intent,
            retries,
        } => tracing::warn!(
            ?profile,
            ?intent,
            retries,
            "forced ceiling reached after consecutive event-silent retry windows that each \
             hashed differently: the proof object is moving via change classes outside the \
             profile's subscribed `events` mask (e.g. in-place writes under events=STRUCTURE); \
             committed the freshest observation anyway — widen `events` to the classes that \
             actually move the tree",
        ),
        Diagnostic::SensorOverflow { scope } => tracing::warn!(
            ?scope,
            "sensor reported overflow (kernel queue dropped events); reseeding in-scope Profiles",
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
            minted_by,
        } => match minted_by {
            // Operator-declared attach: high signal, low rate (one per `[[watch]]` block per reload).
            // INFO is the operator-facing default per the catalog severity table.
            None => tracing::info!(?sub, %name, "sub attached"),
            // A minted attach: one per match, which can be many per pass. DEBUG keeps operator logs
            // uncluttered; `DiscoveryMinted` carries the path at the same severity.
            Some(template) => tracing::debug!(?sub, %name, ?template, "minted sub attached"),
        },
        Diagnostic::SubFired {
            sub,
            profile,
            count,
            minted_by,
        } => tracing::info!(
            ?sub,
            ?profile,
            count,
            ?minted_by,
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
            minted_by,
        } => tracing::info!(?sub, ?profile, ?reason, ?minted_by, "sub detached"),
        Diagnostic::SubRebound { sub, minted_by } => tracing::info!(
            ?sub,
            ?minted_by,
            "sub rebound (per-Sub fields updated in place; baseline preserved)",
        ),
        Diagnostic::RebindUnknownSub { sub } => tracing::warn!(
            ?sub,
            "rebind targeted an unknown Sub (dispatcher routing breach)",
        ),
        Diagnostic::DiscoveryMinted {
            template,
            path,
            kind,
            entry,
        } => tracing::debug!(
            ?template,
            path = %path.display(),
            ?kind,
            ?entry,
            "discovery minted a sub for a match",
        ),
        Diagnostic::DiscoveryUnsupportedAnchorKind {
            template,
            path,
            kind,
        } => tracing::warn!(
            ?template,
            path = %path.display(),
            ?kind,
            "discovery match skipped: terminus kind cannot anchor a watch \
             (regular files and directories only); later unsupported matches \
             under this template skip silently",
        ),
        Diagnostic::DiscoveryFanoutThreshold { template, count } => tracing::warn!(
            ?template,
            count,
            "discovery fanout exceeded warning threshold (consider tightening pattern)",
        ),
        Diagnostic::DiscoverySubReaped {
            template,
            sub,
            path,
        } => tracing::debug!(
            ?template,
            ?sub,
            path = %path.display(),
            "minted sub reaped (its match left the certified match set)",
        ),
        Diagnostic::DiscoveryDenied { template, path } => tracing::warn!(
            ?template,
            path = %path.display(),
            "discovery went on without a directory it may not read; its matches there are unknown \
             until it can be read",
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

/// What a [`Diagnostic`] is about, as a filtered stream reads it ([`DiagSub`]): the `about`
/// [`super::Hub::dispatch_to_subscribers`] takes, projected here so the hub never names
/// [`Diagnostic`] internals. A Profile-keyed line (`ProfileReaped`, `ReapPendingCancelled`, …) is
/// about no Sub and reaches unfiltered subscribers only.
///
/// Total over the [`Diagnostic`] enum — a new core variant is a compile error here (the exhaustive
/// `match` is the structural wall, same discipline as [`crate::ipc::wire::WireDiagnostic::from`]).
/// The verbose `None`-arm enumeration is deliberate: a future variant carrying a Sub that this
/// function silently projects to `None` would be a `wait` bug; the exhaustive `match` forces the
/// author to pick a side at the point of variant introduction.
pub(super) const fn diag_sub(d: &Diagnostic) -> Option<DiagSub> {
    use Diagnostic as D;
    match d {
        D::SubAttached { sub, minted_by, .. }
        | D::SubFired { sub, minted_by, .. }
        | D::SubDetached { sub, minted_by, .. }
        | D::SubRebound { sub, minted_by }
        | D::EffectCompleteOutsideAwaiting { sub, minted_by, .. } => Some(DiagSub {
            sub: *sub,
            minted_by: *minted_by,
        }),

        // A Sub the engine no longer holds names no template.
        D::DetachUnknownSub { sub }
        | D::RebindUnknownSub { sub }
        | D::EffectCompleteForUnknownSub { sub } => Some(DiagSub {
            sub: *sub,
            minted_by: None,
        }),

        // A reap is about the match, and so the template that minted it.
        D::DiscoverySubReaped { template, sub, .. } => Some(DiagSub {
            sub: *sub,
            minted_by: Some(*template),
        }),

        // A pass's lines are about the template that runs it.
        D::DiscoveryMinted { template, .. }
        | D::DiscoveryUnsupportedAnchorKind { template, .. }
        | D::DiscoveryFanoutThreshold { template, .. }
        | D::DiscoveryDenied { template, .. } => Some(DiagSub {
            sub: *template,
            minted_by: None,
        }),

        D::StaleProbeResponse { .. }
        | D::StaleTimer { .. }
        | D::ConfigDiffUnknownSub { .. }
        | D::ConfigDiffRebindFallbackAttach { .. }
        | D::ProbeVanished { .. }
        | D::ProbeFailed { .. }
        | D::EventClassDropped { .. }
        | D::EventOutsideProofObject { .. }
        | D::EventOnUnwatchedResource { .. }
        | D::EventNoConsumer { .. }
        | D::WatchOpRejected { .. }
        | D::PendingPathProbeVanished { .. }
        | D::PendingPathProbeFailed { .. }
        | D::PendingPathAwaitingSegment { .. }
        | D::PendingPathRetriesExhausted { .. }
        | D::PendingPathMaterialized { .. }
        | D::ReapPendingCancelled { .. }
        | D::ProfileReaped { .. }
        | D::ProfileParked { .. }
        | D::ProfileClaimPurged { .. }
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
        | D::ChangeOutsideEventMask { .. }
        | D::SensorOverflow { .. }
        | D::PerFileDriftDroppedOnRecovery { .. }
        | D::PerFileFireSkippedOnFreshSeed { .. }
        | D::QuiescenceAbsorbed { .. }
        | D::AbsorbArmed { .. }
        | D::InvalidBurstTransition { .. }
        | D::WalkerContractViolated { .. } => None,
    }
}

#[cfg(test)]
mod tests {
    use super::diag_sub;
    use crate::driver::ipc::conns::DiagSub;
    use compact_str::CompactString;
    use slotmap::KeyData;
    use specter_core::{BurstIntent, DetachReason, Diagnostic, ProbeCorrelation, ProfileId, SubId};
    use std::path::Path;
    use std::sync::Arc;

    /// Mint a non-default [`SubId`] from a raw FFI value — the fan-out filter keys on `Some(sid)`
    /// vs `None`, so a slotmap default would be indistinguishable from an absent id.
    fn sid(raw: u64) -> SubId {
        SubId::from(KeyData::from_ffi(raw))
    }

    fn pid(raw: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(raw))
    }

    /// Every per-Sub [`Diagnostic`] variant projects to its Sub, a line about a Sub the engine holds
    /// with the template that minted it, a reap with both the match and its template, and a pass's
    /// line to its template. A regression here is a `wait <name>` bug.
    #[test]
    fn diag_sub_pairs_each_line_with_its_template() {
        let (s, t, p) = (sid(1), sid(2), pid(0xAA));
        let about = |sub, minted_by| Some(DiagSub { sub, minted_by });
        assert_eq!(
            diag_sub(&Diagnostic::SubAttached {
                sub: s,
                name: CompactString::const_new("t@/srv/a"),
                minted_by: Some(t),
            }),
            about(s, Some(t)),
        );
        assert_eq!(
            diag_sub(&Diagnostic::SubFired {
                sub: s,
                profile: p,
                count: 1,
                minted_by: Some(t),
            }),
            about(s, Some(t)),
        );
        assert_eq!(
            diag_sub(&Diagnostic::SubDetached {
                sub: s,
                profile: p,
                reason: DetachReason::TemplateDetached,
                minted_by: Some(t),
            }),
            about(s, Some(t)),
        );
        assert_eq!(
            diag_sub(&Diagnostic::SubRebound {
                sub: s,
                minted_by: Some(t),
            }),
            about(s, Some(t)),
        );
        assert_eq!(
            diag_sub(&Diagnostic::EffectCompleteOutsideAwaiting {
                sub: s,
                profile: p,
                minted_by: Some(t),
            }),
            about(s, Some(t)),
        );
        // A Sub the engine no longer holds names no template.
        for gone in [
            Diagnostic::DetachUnknownSub { sub: s },
            Diagnostic::RebindUnknownSub { sub: s },
            Diagnostic::EffectCompleteForUnknownSub { sub: s },
        ] {
            assert_eq!(diag_sub(&gone), about(s, None), "{gone:?}");
        }
        let path: Arc<Path> = Arc::from(Path::new("/srv/a"));
        assert_eq!(
            diag_sub(&Diagnostic::DiscoverySubReaped {
                template: t,
                sub: s,
                path: Arc::clone(&path),
            }),
            about(s, Some(t)),
        );
        assert_eq!(
            diag_sub(&Diagnostic::DiscoveryDenied { template: t, path }),
            about(t, None),
        );
    }

    /// Profile-keyed and metadata-only variants project to `None` — they reach unfiltered
    /// subscribers, never filtered ones. Catches a future variant that carries a [`SubId`] but lands
    /// in the wrong arm.
    #[test]
    fn diag_sub_profile_keyed_returns_none() {
        let p = pid(0xAA);
        assert_eq!(
            diag_sub(&Diagnostic::ProfileReaped {
                profile: p,
                via: specter_core::ReapTrigger::Immediate,
            }),
            None
        );
        assert_eq!(
            diag_sub(&Diagnostic::ReapPendingCancelled { profile: p }),
            None
        );
        assert_eq!(
            diag_sub(&Diagnostic::ProbeVanished {
                profile: p,
                intent: BurstIntent::Standard,
            }),
            None
        );
        assert_eq!(
            diag_sub(&Diagnostic::StaleProbeResponse {
                owner: p,
                correlation: ProbeCorrelation::from(7),
            }),
            None
        );
    }
}
