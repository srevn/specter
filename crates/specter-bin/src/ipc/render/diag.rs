//! `specter tail` / `specter wait` per-event human renderer.
//!
//! One event per line; format:
//!
//! ```text
//! <wall-clock>  <variant>  key=value  key=value  ...
//! ```
//!
//! - `<wall-clock>` is the RFC 3339 timestamp the daemon captured at
//!   `forward()` fanout ([`WireTime`], monospaced).
//! - `<variant>` is the same tag `tail --filter` accepts (matches
//!   [`WireDiagnostic::variant_name`]). Operators copy a tag from a
//!   tail line directly into the next invocation's `--filter`.
//! - Field pairs are space-separated `key=value`. Values escape
//!   nothing (paths and names are not expected to contain
//!   whitespace; [`WireTime`] is a pre-formatted RFC 3339 token);
//!   operators wanting structured data use `-o json`.
//!
//! Layout discipline: timestamp first, variant second, fields after.
//! Same row shape on every line so a column-aligning eye can scan
//! quickly without `column -t`.
//!
//! Pure writer: `(&mut String, &WireDiagnostic)`. No I/O, no styling.
//! The caller owns the buffer's lifetime so the per-event allocation
//! amortizes across stream-loop iterations
//! ([`crate::ipc::client::tail::run`] reuses one buffer for the
//! lifetime of the subscription).

use std::fmt::Write as _;

use crate::ipc::wire::{
    WireAbsorbMode, WireBurstHelper, WireBurstIntent, WireClaimKind, WireDetachReason,
    WireDiagnostic, WireFsEvent, WireProfileStateDiscriminant, WirePromoterClaimKind,
    WireReapTrigger, WireResourceKind, WireSpliceFailureCause, WireTime,
};

/// Render one event as a single newline-terminated line into the
/// caller's buffer.
///
/// Writer-shape so the call site amortizes the line buffer across
/// iterations — `specter tail` reuses one [`String`] for the lifetime
/// of the stream loop, symmetric with
/// [`crate::ipc::client::subscribe::Subscription`]'s reused inbound
/// `line_buf`. A 1000-evt/s tail carries no per-event allocation
/// through the human path; the three compound-enum Display impls
/// ([`crate::ipc::wire::WireProbeOwner`] /
/// [`crate::ipc::wire::WireOverflowScope`] /
/// [`crate::ipc::wire::WireWatchFailure`]) write through the same
/// formatter so the compound fields likewise carry no allocation.
pub(crate) fn render(out: &mut String, d: &WireDiagnostic) {
    let _ = write!(out, "{}  {}", at_field(d), d.variant_name());
    write_fields(out, d);
    out.push('\n');
}

/// Project the variant's `at` field through the structural commitment
/// that every [`WireDiagnostic`] variant declares `at: WireTime` as
/// its first field. Single or-pattern arm — a new variant without an
/// `at` field is a compile error here.
const fn at_field(d: &WireDiagnostic) -> &WireTime {
    match d {
        WireDiagnostic::StaleProbeResponse { at, .. }
        | WireDiagnostic::StaleTimer { at, .. }
        | WireDiagnostic::EffectCompleteOutsideAwaiting { at, .. }
        | WireDiagnostic::EffectCompleteForUnknownSub { at, .. }
        | WireDiagnostic::DetachUnknownSub { at, .. }
        | WireDiagnostic::ConfigDiffUnknownSub { at, .. }
        | WireDiagnostic::ConfigDiffUnknownPromoter { at, .. }
        | WireDiagnostic::ConfigDiffRebindFallbackAttach { at, .. }
        | WireDiagnostic::ProbeVanished { at, .. }
        | WireDiagnostic::ProbeFailed { at, .. }
        | WireDiagnostic::EventClassDropped { at, .. }
        | WireDiagnostic::EventOnUnwatchedResource { at, .. }
        | WireDiagnostic::EventNoConsumer { at, .. }
        | WireDiagnostic::WatchOpRejected { at, .. }
        | WireDiagnostic::PendingPathProbeVanished { at, .. }
        | WireDiagnostic::PendingPathProbeFailed { at, .. }
        | WireDiagnostic::ReapPendingCancelled { at, .. }
        | WireDiagnostic::ProfileReaped { at, .. }
        | WireDiagnostic::ProfileClaimPurged { at, .. }
        | WireDiagnostic::PromoterClaimPurged { at, .. }
        | WireDiagnostic::AttachPathInvalid { at, .. }
        | WireDiagnostic::AttachResourceStale { at, .. }
        | WireDiagnostic::AnchorKindMismatch { at, .. }
        | WireDiagnostic::SpliceCrossedUncovered { at, .. }
        | WireDiagnostic::EventAbsorbedByFireTail { at, .. }
        | WireDiagnostic::AwaitGateDeadlineForceRebasing { at, .. }
        | WireDiagnostic::AwaitGateDeadlineReap { at, .. }
        | WireDiagnostic::QuiescenceCeilingUnreadable { at, .. }
        | WireDiagnostic::QuiescenceCeilingForcedDespiteChange { at, .. }
        | WireDiagnostic::RebaseCeilingStillChanging { at, .. }
        | WireDiagnostic::RebaseCeilingForcedDespiteChange { at, .. }
        | WireDiagnostic::RebaseCeilingUnreadable { at, .. }
        | WireDiagnostic::SensorOverflow { at, .. }
        | WireDiagnostic::PromoterReseededForOverflow { at, .. }
        | WireDiagnostic::PerFileDriftDroppedOnRecovery { at, .. }
        | WireDiagnostic::PerFileFireSkippedOnFreshSeed { at, .. }
        | WireDiagnostic::SubAttached { at, .. }
        | WireDiagnostic::SubFired { at, .. }
        | WireDiagnostic::QuiescenceAbsorbed { at, .. }
        | WireDiagnostic::AbsorbArmed { at, .. }
        | WireDiagnostic::SubDetached { at, .. }
        | WireDiagnostic::SubRebound { at, .. }
        | WireDiagnostic::RebindUnknownSub { at, .. }
        | WireDiagnostic::PromoterAttached { at, .. }
        | WireDiagnostic::PromoterReaped { at, .. }
        | WireDiagnostic::PromoterDescentVanished { at, .. }
        | WireDiagnostic::PromoterDescentFailed { at, .. }
        | WireDiagnostic::PromotionKindObserved { at, .. }
        | WireDiagnostic::PromoterFanoutThreshold { at, .. }
        | WireDiagnostic::PromoterProxyStaleEvent { at, .. }
        | WireDiagnostic::PromoterEnumerationVanished { at, .. }
        | WireDiagnostic::PromoterEnumerationFailed { at, .. }
        | WireDiagnostic::DynamicSubReaped { at, .. }
        | WireDiagnostic::InvalidBurstTransition { at, .. }
        | WireDiagnostic::WalkerContractViolated { at, .. }
        | WireDiagnostic::Missed { at, .. } => at,
    }
}

/// Append every non-`at` field as ` key=value` pairs. Exhaustive
/// match — a new variant lands a compile error here, paired with the
/// matching arm in [`WireDiagnostic::variant_name`] and the
/// `KNOWN_WIRE_VARIANTS` tag list.
///
/// Field order mirrors the variant's declaration order so the human
/// form and the JSON form present fields in the same sequence.
///
/// One arm per variant — fewer would mean a less specific format.
fn write_fields(out: &mut String, d: &WireDiagnostic) {
    match d {
        WireDiagnostic::StaleProbeResponse {
            owner, correlation, ..
        } => {
            let _ = write!(out, "  owner={owner}  correlation={correlation}");
        }
        WireDiagnostic::StaleTimer { id, .. } => {
            let _ = write!(out, "  id={id}");
        }
        WireDiagnostic::EffectCompleteOutsideAwaiting { sub, profile, .. } => {
            let _ = write!(out, "  sub={}  profile={}", sub.0, profile.0);
        }
        WireDiagnostic::EffectCompleteForUnknownSub { sub, .. } => {
            let _ = write!(out, "  sub={}", sub.0);
        }
        WireDiagnostic::DetachUnknownSub { sub, .. } => {
            let _ = write!(out, "  sub={}", sub.0);
        }
        WireDiagnostic::ConfigDiffUnknownSub { name, .. }
        | WireDiagnostic::ConfigDiffUnknownPromoter { name, .. }
        | WireDiagnostic::ConfigDiffRebindFallbackAttach { name, .. } => {
            let _ = write!(out, "  name={name}");
        }
        WireDiagnostic::ProbeVanished {
            profile, intent, ..
        } => {
            let _ = write!(
                out,
                "  profile={}  intent={}",
                profile.0,
                burst_intent_str(*intent),
            );
        }
        WireDiagnostic::ProbeFailed {
            profile,
            intent,
            errno,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  intent={}  errno={errno}",
                profile.0,
                burst_intent_str(*intent),
            );
        }
        WireDiagnostic::EventClassDropped {
            resource,
            event,
            profile,
            ..
        } => {
            let _ = write!(
                out,
                "  resource={}  event={}  profile={}",
                resource.0,
                fs_event_str(*event),
                profile.0,
            );
        }
        WireDiagnostic::EventOnUnwatchedResource { resource, .. }
        | WireDiagnostic::EventNoConsumer { resource, .. }
        | WireDiagnostic::AttachResourceStale { resource, .. } => {
            let _ = write!(out, "  resource={}", resource.0);
        }
        WireDiagnostic::WatchOpRejected {
            resource, failure, ..
        } => {
            let _ = write!(out, "  resource={}  failure={failure}", resource.0);
        }
        WireDiagnostic::PendingPathProbeVanished {
            profile, prefix, ..
        } => {
            let _ = write!(out, "  profile={}  prefix={}", profile.0, prefix.0);
        }
        WireDiagnostic::PendingPathProbeFailed {
            profile,
            prefix,
            errno,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  prefix={}  errno={errno}",
                profile.0, prefix.0,
            );
        }
        WireDiagnostic::ReapPendingCancelled { profile, .. }
        | WireDiagnostic::PerFileDriftDroppedOnRecovery { profile, .. }
        | WireDiagnostic::PerFileFireSkippedOnFreshSeed { profile, .. }
        | WireDiagnostic::QuiescenceAbsorbed { profile, .. } => {
            let _ = write!(out, "  profile={}", profile.0);
        }
        WireDiagnostic::AbsorbArmed { profile, mode, .. } => {
            let _ = write!(
                out,
                "  profile={}  mode={}",
                profile.0,
                absorb_mode_str(*mode),
            );
        }
        WireDiagnostic::ProfileReaped { profile, via, .. } => {
            let _ = write!(
                out,
                "  profile={}  via={}",
                profile.0,
                reap_trigger_str(*via),
            );
        }
        WireDiagnostic::ProfileClaimPurged {
            profile,
            claim,
            resource,
            failure,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  claim={}  resource={}  failure={failure}",
                profile.0,
                claim_kind_str(*claim),
                resource.0,
            );
        }
        WireDiagnostic::PromoterClaimPurged {
            promoter,
            claim,
            resource,
            failure,
            ..
        } => {
            let _ = write!(
                out,
                "  promoter={}  claim={}  resource={}  failure={failure}",
                promoter.0,
                promoter_claim_kind_str(*claim),
                resource.0,
            );
        }
        WireDiagnostic::AttachPathInvalid { path, hint, .. } => {
            let _ = write!(out, "  path={path}  hint={hint}");
        }
        WireDiagnostic::AnchorKindMismatch {
            profile,
            prior_kind,
            response_kind,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  prior_kind={}  response_kind={}",
                profile.0,
                resource_kind_str(*prior_kind),
                resource_kind_str(*response_kind),
            );
        }
        WireDiagnostic::SpliceCrossedUncovered {
            profile,
            target,
            cause,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  target={}  cause={}",
                profile.0,
                target.0,
                splice_failure_cause_str(*cause),
            );
        }
        WireDiagnostic::EventAbsorbedByFireTail {
            profile,
            resource,
            event,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  resource={}  event={}",
                profile.0,
                resource.0,
                fs_event_str(*event),
            );
        }
        WireDiagnostic::AwaitGateDeadlineForceRebasing {
            profile,
            outstanding,
            ..
        }
        | WireDiagnostic::AwaitGateDeadlineReap {
            profile,
            outstanding,
            ..
        } => {
            let _ = write!(out, "  profile={}  outstanding={outstanding}", profile.0);
        }
        WireDiagnostic::QuiescenceCeilingUnreadable {
            profile,
            first_unread,
            intent,
            ..
        }
        | WireDiagnostic::RebaseCeilingUnreadable {
            profile,
            first_unread,
            intent,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  first_unread={first_unread}  intent={}",
                profile.0,
                burst_intent_str(*intent),
            );
        }
        WireDiagnostic::RebaseCeilingStillChanging {
            profile, intent, ..
        }
        | WireDiagnostic::QuiescenceCeilingForcedDespiteChange {
            profile, intent, ..
        }
        | WireDiagnostic::RebaseCeilingForcedDespiteChange {
            profile, intent, ..
        } => {
            let _ = write!(
                out,
                "  profile={}  intent={}",
                profile.0,
                burst_intent_str(*intent),
            );
        }
        WireDiagnostic::SensorOverflow { scope, .. } => {
            let _ = write!(out, "  scope={scope}");
        }
        WireDiagnostic::PromoterReseededForOverflow { promoter, .. }
        | WireDiagnostic::PromoterReaped { promoter, .. } => {
            let _ = write!(out, "  promoter={}", promoter.0);
        }
        WireDiagnostic::SubAttached {
            sub,
            name,
            source_promoter,
            ..
        } => {
            let _ = write!(out, "  sub={}  name={name}", sub.0);
            if let Some(p) = source_promoter {
                let _ = write!(out, "  source_promoter={}", p.0);
            }
        }
        WireDiagnostic::SubFired {
            sub,
            profile,
            count,
            ..
        } => {
            let _ = write!(out, "  sub={}  profile={}  count={count}", sub.0, profile.0);
        }
        WireDiagnostic::SubDetached {
            sub,
            profile,
            reason,
            ..
        } => {
            let _ = write!(
                out,
                "  sub={}  profile={}  reason={}",
                sub.0,
                profile.0,
                detach_reason_str(*reason),
            );
        }
        WireDiagnostic::SubRebound { sub, .. } | WireDiagnostic::RebindUnknownSub { sub, .. } => {
            let _ = write!(out, "  sub={}", sub.0);
        }
        WireDiagnostic::PromoterAttached { promoter, name, .. } => {
            let _ = write!(out, "  promoter={}  name={name}", promoter.0);
        }
        WireDiagnostic::PromoterDescentVanished {
            promoter, prefix, ..
        } => {
            let _ = write!(out, "  promoter={}  prefix={}", promoter.0, prefix.0);
        }
        WireDiagnostic::PromoterDescentFailed {
            promoter,
            prefix,
            errno,
            ..
        } => {
            let _ = write!(
                out,
                "  promoter={}  prefix={}  errno={errno}",
                promoter.0, prefix.0,
            );
        }
        WireDiagnostic::PromotionKindObserved {
            promoter,
            path,
            kind,
            ..
        } => {
            let _ = write!(
                out,
                "  promoter={}  path={path}  kind={}",
                promoter.0,
                resource_kind_str(*kind),
            );
        }
        WireDiagnostic::PromoterFanoutThreshold {
            promoter, count, ..
        } => {
            let _ = write!(out, "  promoter={}  count={count}", promoter.0);
        }
        WireDiagnostic::PromoterProxyStaleEvent {
            promoter, resource, ..
        } => {
            let _ = write!(out, "  promoter={}  resource={}", promoter.0, resource.0);
        }
        WireDiagnostic::PromoterEnumerationVanished {
            promoter, proxy, ..
        } => {
            let _ = write!(out, "  promoter={}  proxy={}", promoter.0, proxy.0);
        }
        WireDiagnostic::PromoterEnumerationFailed {
            promoter,
            proxy,
            errno,
            ..
        } => {
            let _ = write!(
                out,
                "  promoter={}  proxy={}  errno={errno}",
                promoter.0, proxy.0,
            );
        }
        WireDiagnostic::DynamicSubReaped {
            promoter,
            sub,
            path,
            ..
        } => {
            let _ = write!(out, "  promoter={}  sub={}  path={path}", promoter.0, sub.0);
        }
        WireDiagnostic::InvalidBurstTransition {
            profile,
            helper,
            observed,
            ..
        } => {
            let _ = write!(
                out,
                "  profile={}  helper={}  observed={}",
                profile.0,
                burst_helper_str(*helper),
                profile_state_discriminant_str(*observed),
            );
        }
        WireDiagnostic::WalkerContractViolated { owner, .. } => {
            let _ = write!(out, "  owner={owner}");
        }
        WireDiagnostic::Missed { count, .. } => {
            let _ = write!(out, "  count={count}");
        }
    }
}

/// Operator-visible label for [`WireBurstIntent`]. Mirrors the
/// `snake_case` serde rename so the human view matches the JSON.
const fn burst_intent_str(i: WireBurstIntent) -> &'static str {
    match i {
        WireBurstIntent::Standard => "standard",
        WireBurstIntent::Seed => "seed",
    }
}

/// Operator-visible label for [`WireFsEvent`]. Mirrors the
/// `snake_case` serde rename.
const fn fs_event_str(e: WireFsEvent) -> &'static str {
    match e {
        WireFsEvent::Modified => "modified",
        WireFsEvent::MetadataChanged => "metadata_changed",
        WireFsEvent::StructureChanged => "structure_changed",
        WireFsEvent::Renamed => "renamed",
        WireFsEvent::Removed => "removed",
        WireFsEvent::Revoked => "revoked",
    }
}

/// Operator-visible label for [`WireReapTrigger`].
const fn reap_trigger_str(t: WireReapTrigger) -> &'static str {
    match t {
        WireReapTrigger::Immediate => "immediate",
        WireReapTrigger::DeferredFromBurst => "deferred_from_burst",
    }
}

/// Operator-visible label for [`WireResourceKind`].
const fn resource_kind_str(k: WireResourceKind) -> &'static str {
    match k {
        WireResourceKind::File => "file",
        WireResourceKind::Dir => "dir",
        WireResourceKind::Unknown => "unknown",
    }
}

/// Operator-visible label for [`WireClaimKind`].
const fn claim_kind_str(c: WireClaimKind) -> &'static str {
    match c {
        WireClaimKind::Anchor => "anchor",
        WireClaimKind::WatchRootParent => "watch_root_parent",
        WireClaimKind::DescentPrefix => "descent_prefix",
    }
}

/// Operator-visible label for [`WirePromoterClaimKind`].
const fn promoter_claim_kind_str(c: WirePromoterClaimKind) -> &'static str {
    match c {
        WirePromoterClaimKind::DescentPrefix => "descent_prefix",
        WirePromoterClaimKind::ActiveProxy => "active_proxy",
        WirePromoterClaimKind::PrefixParent => "prefix_parent",
    }
}

/// Operator-visible label for [`WireSpliceFailureCause`].
const fn splice_failure_cause_str(c: WireSpliceFailureCause) -> &'static str {
    match c {
        WireSpliceFailureCause::TargetOutsideAnchorSubtree => "target_outside_anchor_subtree",
        WireSpliceFailureCause::SlotReapedMidGraft => "slot_reaped_mid_graft",
        WireSpliceFailureCause::IntermediateUncovered => "intermediate_uncovered",
    }
}

/// Operator-visible label for [`WireDetachReason`].
const fn detach_reason_str(r: WireDetachReason) -> &'static str {
    match r {
        WireDetachReason::ConfigDiffRemoved => "config_diff_removed",
        WireDetachReason::ConfigDiffIdentityChanged => "config_diff_identity_changed",
        WireDetachReason::IpcDisabled => "ipc_disabled",
        WireDetachReason::PromoterReaped => "promoter_reaped",
    }
}

/// Operator-visible label for [`WireBurstHelper`].
const fn burst_helper_str(h: WireBurstHelper) -> &'static str {
    match h {
        WireBurstHelper::StartSeedBurst => "start_seed_burst",
        WireBurstHelper::StartStandardBurst => "start_standard_burst",
        WireBurstHelper::EventDrivesBatching => "event_drives_batching",
        WireBurstHelper::RetryDrivesBatching => "retry_drives_batching",
        WireBurstHelper::TransitionToVerifying => "transition_to_verifying",
        WireBurstHelper::TransitionToDraining => "transition_to_draining",
        WireBurstHelper::TransitionToAwaiting => "transition_to_awaiting",
        WireBurstHelper::TransitionToRebasing => "transition_to_rebasing",
        WireBurstHelper::TransitionToSettling => "transition_to_settling",
        WireBurstHelper::AbsorbEventIntoFireTail => "absorb_event_into_fire_tail",
        WireBurstHelper::RestartBurstFromFireTailResidual => {
            "restart_burst_from_fire_tail_residual"
        }
    }
}

/// Operator-visible label for [`WireProfileStateDiscriminant`].
const fn profile_state_discriminant_str(d: WireProfileStateDiscriminant) -> &'static str {
    match d {
        WireProfileStateDiscriminant::Idle => "idle",
        WireProfileStateDiscriminant::Pending => "pending",
        WireProfileStateDiscriminant::ActivePreFire => "active_pre_fire",
        WireProfileStateDiscriminant::ActivePostFire => "active_post_fire",
    }
}

/// Operator-visible label for [`WireAbsorbMode`] on the `tail` stream.
/// Mirrors the snake_case serde rename so the human view matches the
/// JSON (the `show` renderer uses its own hyphenated label table).
const fn absorb_mode_str(m: WireAbsorbMode) -> &'static str {
    match m {
        WireAbsorbMode::ConsumeOnFirst => "consume_on_first",
        WireAbsorbMode::PersistUntil => "persist_until",
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::ipc::protocol::WireId;
    use crate::ipc::wire::{
        WireBurstIntent, WireDetachReason, WireDiagnostic, WireFsEvent, WireOverflowScope,
        WireProbeOwner, WireTime,
    };
    use std::time::UNIX_EPOCH;

    /// The renderer's row shape: timestamp first, variant second,
    /// fields after, newline last. Pin per-Sub event details so the
    /// operator-visible fields stay structural.
    #[test]
    fn render_sub_fired_carries_timestamp_tag_sub_profile_count() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::SubFired {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(11),
                profile: WireId(22),
                count: 3,
            },
        );
        assert!(
            s.starts_with("1970-01-01T00:00:00Z  sub_fired"),
            "leading timestamp + tag: {s:?}",
        );
        assert!(s.contains("  sub=11"), "sub field present: {s:?}");
        assert!(s.contains("  profile=22"), "profile field present: {s:?}");
        assert!(s.contains("  count=3"), "count field present: {s:?}");
        assert!(s.ends_with('\n'), "newline-terminated: {s:?}");
    }

    /// `Missed` is the back-pressure marker. Its tag is the
    /// underscore-prefixed `_missed` (the only variant with an
    /// override) — operators reading a `_missed N` line know the
    /// daemon dropped events upstream.
    #[test]
    fn render_missed_marker_is_underscore_prefixed() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::Missed {
                at: WireTime::from(UNIX_EPOCH),
                count: 5,
            },
        );
        assert!(
            s.starts_with("1970-01-01T00:00:00Z  _missed"),
            "leading `_missed` tag: {s:?}",
        );
        assert!(s.contains("  count=5"), "count field present: {s:?}");
        assert!(s.ends_with('\n'), "newline-terminated: {s:?}");
    }

    /// `SubAttached.source_promoter = Some(_)` renders an extra
    /// `source_promoter=N` field; `None` omits it entirely. Operators
    /// distinguishing static vs promoter-minted Subs read the
    /// presence/absence of the field.
    #[test]
    fn render_sub_attached_promoter_field_optional() {
        let mut static_attach = String::new();
        render(
            &mut static_attach,
            &WireDiagnostic::SubAttached {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(1),
                name: "static_watch".into(),
                source_promoter: None,
            },
        );
        assert!(static_attach.contains("name=static_watch"));
        assert!(
            !static_attach.contains("source_promoter"),
            "None must omit source_promoter: {static_attach:?}",
        );

        let mut dynamic_attach = String::new();
        render(
            &mut dynamic_attach,
            &WireDiagnostic::SubAttached {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(2),
                name: "p@/tmp/x".into(),
                source_promoter: Some(WireId(99)),
            },
        );
        assert!(dynamic_attach.contains("source_promoter=99"));
    }

    /// `SubDetached.reason` renders through the typed
    /// [`WireDetachReason`] label table (`config_diff_removed`,
    /// `ipc_disabled`, etc.). Mirrors the snake_case serde rename so
    /// the human form matches the JSON.
    #[test]
    fn render_sub_detached_reason_label() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::SubDetached {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(1),
                profile: WireId(2),
                reason: WireDetachReason::IpcDisabled,
            },
        );
        assert!(s.contains("  reason=ipc_disabled"), "got: {s:?}");
    }

    /// A Profile-keyed variant (`ProfileReaped`) renders without any
    /// `sub=` field — operators reading a tail know the event is not
    /// per-Sub by the absence of the column. Distinct from
    /// per-Sub variants that always carry `sub=N`.
    #[test]
    fn render_profile_keyed_has_no_sub_field() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::ProfileReaped {
                at: WireTime::from(UNIX_EPOCH),
                profile: WireId(7),
                via: crate::ipc::wire::WireReapTrigger::DeferredFromBurst,
            },
        );
        assert!(s.contains("  profile=7"));
        assert!(s.contains("  via=deferred_from_burst"));
        assert!(
            !s.contains("sub="),
            "Profile-keyed variant must not carry a sub field: {s:?}",
        );
    }

    /// Promoter-keyed variants (`PromoterAttached`) render the
    /// `promoter=N` key. Sanity-checks the cross-cutting helper coverage.
    #[test]
    fn render_promoter_attached_carries_promoter_and_name() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::PromoterAttached {
                at: WireTime::from(UNIX_EPOCH),
                promoter: WireId(42),
                name: "watch_glob".into(),
            },
        );
        assert!(s.contains("  promoter=42"));
        assert!(s.contains("  name=watch_glob"));
    }

    /// Compound enums render through their helper — `WireProbeOwner`
    /// projects through `probe_owner_str` to `<kind>/<id>` form, which
    /// is more operator-readable than two separate fields.
    #[test]
    fn render_probe_owner_compound_label() {
        let mut profile_owner = String::new();
        render(
            &mut profile_owner,
            &WireDiagnostic::StaleProbeResponse {
                at: WireTime::from(UNIX_EPOCH),
                owner: WireProbeOwner::Profile { profile: WireId(1) },
                correlation: 9,
            },
        );
        assert!(profile_owner.contains("  owner=profile/1"));
        assert!(profile_owner.contains("  correlation=9"));

        let mut promoter_owner = String::new();
        render(
            &mut promoter_owner,
            &WireDiagnostic::StaleProbeResponse {
                at: WireTime::from(UNIX_EPOCH),
                owner: WireProbeOwner::Promoter {
                    promoter: WireId(2),
                },
                correlation: 11,
            },
        );
        assert!(promoter_owner.contains("  owner=promoter/2"));
    }

    /// `EventClassDropped` is the canonical multi-field per-Resource
    /// variant; its rendered line carries resource, event, profile in
    /// declaration order with the correct enum label for the event.
    #[test]
    fn render_event_class_dropped_three_fields_in_order() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::EventClassDropped {
                at: WireTime::from(UNIX_EPOCH),
                resource: WireId(100),
                event: WireFsEvent::MetadataChanged,
                profile: WireId(200),
            },
        );
        // Field order matches declaration order.
        let idx_resource = s.find("resource=").expect("resource present");
        let idx_event = s.find("event=").expect("event present");
        let idx_profile = s.find("profile=").expect("profile present");
        assert!(
            idx_resource < idx_event && idx_event < idx_profile,
            "fields not in declaration order: {s:?}",
        );
        assert!(s.contains("event=metadata_changed"), "got: {s:?}");
    }

    /// `WireOverflowScope::Global` is the bare-tag variant —
    /// rendering must emit `scope=global` without any `/id` tail.
    #[test]
    fn render_sensor_overflow_global_scope_bare() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::SensorOverflow {
                at: WireTime::from(UNIX_EPOCH),
                scope: WireOverflowScope::Global,
            },
        );
        assert!(s.contains("  scope=global"), "got: {s:?}");
        assert!(
            !s.contains("scope=global/"),
            "global must NOT carry a trailing id: {s:?}",
        );
    }

    /// Cross-variant coverage: every variant on the [`super::super::wire::KNOWN_WIRE_VARIANTS`]
    /// list renders to a non-empty line that contains its tag. Catches
    /// a missing `write_fields` arm before the operator hits it. Uses
    /// the same witness fixture the wire round-trip test consumes —
    /// duplicating would invite drift.
    #[test]
    fn every_variant_renders_a_nonempty_line_with_its_tag() {
        // The wire test fixture lives in wire's test module; we
        // can't reach it cross-module, so we exercise the
        // KNOWN_WIRE_VARIANTS surface here via the JSON wire round
        // trip — every wire-variant witness round-trips, and the
        // renderer reads back the structural shape. Reach the
        // wire-side test seam through serde directly.
        //
        // For each variant tag, synthesize the minimum-field JSON
        // shape and feed it through deserialize → render. A wire
        // variant whose JSON shape doesn't round-trip would already
        // fail the wire's own round-trip test; this guard reports
        // any render-side regression separately.
        let timestamp = serde_json::to_value(WireTime::from(UNIX_EPOCH)).unwrap();
        for tag in crate::ipc::wire::KNOWN_WIRE_VARIANTS {
            let value = synthesize_min_value(tag, &timestamp);
            let wire: WireDiagnostic = serde_json::from_value(value).unwrap_or_else(|e| {
                panic!("failed to deserialize witness for tag {tag}: {e}");
            });
            let mut line = String::new();
            render(&mut line, &wire);
            assert!(
                line.contains(tag),
                "render line missing tag {tag}: {line:?}"
            );
            assert!(line.ends_with('\n'), "no trailing LF for tag {tag}");
            assert!(
                line.starts_with("1970-01-01T00:00:00Z  "),
                "leading timestamp absent for tag {tag}: {line:?}",
            );
        }
    }

    /// Minimum-field JSON value for one variant tag. Constructs the
    /// smallest acceptable payload so [`WireDiagnostic::deserialize`]
    /// succeeds; any new variant added without a paired arm here
    /// fails [`every_variant_renders_a_nonempty_line_with_its_tag`]
    /// loudly.
    fn synthesize_min_value(tag: &str, at: &serde_json::Value) -> serde_json::Value {
        use serde_json::json;
        let id = json!(1);
        let profile_owner = json!({ "kind": "profile", "profile": id });
        let global_scope = json!({ "scope": "global" });
        let pressure = json!({ "kind": "pressure", "errno": 24 });
        match tag {
            "stale_probe_response" => {
                json!({"diag": tag, "at": at, "owner": profile_owner, "correlation": 1})
            }
            "stale_timer" => json!({"diag": tag, "at": at, "id": 1}),
            "effect_complete_outside_awaiting" => {
                json!({"diag": tag, "at": at, "sub": id, "profile": id})
            }
            "effect_complete_for_unknown_sub" | "detach_unknown_sub" => {
                json!({"diag": tag, "at": at, "sub": id})
            }
            "config_diff_unknown_sub"
            | "config_diff_unknown_promoter"
            | "config_diff_rebind_fallback_attach" => {
                json!({"diag": tag, "at": at, "name": "x"})
            }
            "probe_vanished" => {
                json!({"diag": tag, "at": at, "profile": id, "intent": "standard"})
            }
            "probe_failed" => {
                json!({"diag": tag, "at": at, "profile": id, "intent": "standard", "errno": 0})
            }
            "event_class_dropped" => {
                json!({"diag": tag, "at": at, "resource": id, "event": "modified", "profile": id})
            }
            "event_on_unwatched_resource" | "event_no_consumer" | "attach_resource_stale" => {
                json!({"diag": tag, "at": at, "resource": id})
            }
            "watch_op_rejected" => {
                json!({"diag": tag, "at": at, "resource": id, "failure": pressure})
            }
            "pending_path_probe_vanished" => {
                json!({"diag": tag, "at": at, "profile": id, "prefix": id})
            }
            "pending_path_probe_failed" => {
                json!({"diag": tag, "at": at, "profile": id, "prefix": id, "errno": 0})
            }
            "reap_pending_cancelled"
            | "per_file_drift_dropped_on_recovery"
            | "per_file_fire_skipped_on_fresh_seed" => {
                json!({"diag": tag, "at": at, "profile": id})
            }
            "profile_reaped" => {
                json!({"diag": tag, "at": at, "profile": id, "via": "immediate"})
            }
            "profile_claim_purged" => json!({
                "diag": tag, "at": at, "profile": id, "claim": "anchor",
                "resource": id, "failure": pressure,
            }),
            "promoter_claim_purged" => json!({
                "diag": tag, "at": at, "promoter": id, "claim": "active_proxy",
                "resource": id, "failure": pressure,
            }),
            "attach_path_invalid" => {
                json!({"diag": tag, "at": at, "path": "/x", "hint": "h"})
            }
            "anchor_kind_mismatch" => json!({
                "diag": tag, "at": at, "profile": id,
                "prior_kind": "dir", "response_kind": "file",
            }),
            "splice_crossed_uncovered" => json!({
                "diag": tag, "at": at, "profile": id, "target": id,
                "cause": "target_outside_anchor_subtree",
            }),
            "event_absorbed_by_fire_tail" => json!({
                "diag": tag, "at": at, "profile": id, "resource": id,
                "event": "modified",
            }),
            "await_gate_deadline_force_rebasing" | "await_gate_deadline_reap" => {
                json!({"diag": tag, "at": at, "profile": id, "outstanding": 1})
            }
            "quiescence_ceiling_unreadable" | "rebase_ceiling_unreadable" => json!({
                "diag": tag, "at": at, "profile": id,
                "first_unread": "/x", "intent": "standard",
            }),
            "rebase_ceiling_still_changing"
            | "quiescence_ceiling_forced_despite_change"
            | "rebase_ceiling_forced_despite_change" => {
                json!({"diag": tag, "at": at, "profile": id, "intent": "standard"})
            }
            "sensor_overflow" => json!({"diag": tag, "at": at, "scope": global_scope}),
            "promoter_reseeded_for_overflow" | "promoter_reaped" => {
                json!({"diag": tag, "at": at, "promoter": id})
            }
            "sub_attached" => json!({
                "diag": tag, "at": at, "sub": id, "name": "x", "source_promoter": null,
            }),
            "sub_fired" => {
                json!({"diag": tag, "at": at, "sub": id, "profile": id, "count": 1})
            }
            "quiescence_absorbed" => json!({"diag": tag, "at": at, "profile": id}),
            "absorb_armed" => {
                json!({"diag": tag, "at": at, "profile": id, "mode": "consume_on_first"})
            }
            "sub_detached" => json!({
                "diag": tag, "at": at, "sub": id, "profile": id, "reason": "ipc_disabled",
            }),
            "sub_rebound" | "rebind_unknown_sub" => json!({"diag": tag, "at": at, "sub": id}),
            "promoter_attached" => {
                json!({"diag": tag, "at": at, "promoter": id, "name": "p"})
            }
            "promoter_descent_vanished" => {
                json!({"diag": tag, "at": at, "promoter": id, "prefix": id})
            }
            "promoter_descent_failed" => json!({
                "diag": tag, "at": at, "promoter": id, "prefix": id, "errno": 0,
            }),
            "promotion_kind_observed" => json!({
                "diag": tag, "at": at, "promoter": id, "path": "/x", "kind": "dir",
            }),
            "promoter_fanout_threshold" => {
                json!({"diag": tag, "at": at, "promoter": id, "count": 0})
            }
            "promoter_proxy_stale_event" => {
                json!({"diag": tag, "at": at, "promoter": id, "resource": id})
            }
            "promoter_enumeration_vanished" => {
                json!({"diag": tag, "at": at, "promoter": id, "proxy": id})
            }
            "promoter_enumeration_failed" => json!({
                "diag": tag, "at": at, "promoter": id, "proxy": id, "errno": 0,
            }),
            "dynamic_sub_reaped" => json!({
                "diag": tag, "at": at, "promoter": id, "sub": id, "path": "/x",
            }),
            "invalid_burst_transition" => json!({
                "diag": tag, "at": at, "profile": id,
                "helper": "transition_to_verifying", "observed": "idle",
            }),
            "walker_contract_violated" => {
                json!({"diag": tag, "at": at, "owner": profile_owner})
            }
            "_missed" => json!({"diag": tag, "at": at, "count": 1}),
            other => panic!("synthesize_min_value: unknown tag {other}"),
        }
    }

    /// `WireBurstIntent` labels mirror the snake_case wire form.
    #[test]
    fn render_probe_failed_burst_intent_label() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::ProbeFailed {
                at: WireTime::from(UNIX_EPOCH),
                profile: WireId(1),
                intent: WireBurstIntent::Seed,
                errno: 13,
            },
        );
        assert!(s.contains("  intent=seed"), "got: {s:?}");
        assert!(s.contains("  errno=13"), "got: {s:?}");
    }
}
