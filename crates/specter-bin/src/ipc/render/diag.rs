//! `specter tail` / `specter wait` per-event human renderer.
//!
//! One event per line; format:
//!
//! ```text
//! <wall-clock>  <variant>  key=value  key=value  ...
//! ```
//!
//! - `<wall-clock>` is the RFC 3339 timestamp the daemon captured at `forward()` fanout
//!   ([`WireTime`], monospaced).
//! - `<variant>` is the same tag `tail --filter` accepts (matches [`WireDiagnostic::variant_name`]).
//!   Operators copy a tag from a tail line directly into the next invocation's `--filter`.
//! - Field pairs are space-separated `key=value`. Values escape nothing (paths and names are not
//!   expected to contain whitespace; [`WireTime`] is a pre-formatted RFC 3339 token); operators
//!   wanting structured data use `-o json`.
//!
//! Layout discipline: timestamp first, variant second, fields after. Same row shape on every line
//! so a column-aligning eye can scan quickly without `column -t`.
//!
//! Pure writer: `(&mut String, &WireDiagnostic, Styler)`. No I/O. The leading timestamp paints
//! [`style::SECONDARY`], the variant tag its [`severity`] hue, and each field's key/`=` paint
//! [`style::LABEL`] / [`style::DELIM`] (values stay unstyled). Under `Styler::Plain` the line is
//! byte-identical to the pre-color form, and the painted path stays allocation-free — the caller
//! owns the buffer's lifetime so the per-event work amortizes across stream-loop iterations
//! ([`crate::ipc::client::tail::run`] reuses one buffer for the lifetime of the subscription).

use std::fmt::{Display, Write as _};

use crate::ipc::render::style::{self, Severity, Styler};
use crate::ipc::wire::{WireDiagnostic, WireTime};

/// Render one event as a single newline-terminated line into the caller's buffer.
///
/// Writer-shape so the call site amortizes the line buffer across iterations — `specter tail`
/// reuses one [`String`] for the lifetime of the stream loop, symmetric with
/// [`crate::ipc::client::subscribe::Subscription`]'s reused inbound `line_buf`. A 1000-evt/s tail
/// carries no per-event allocation through the human path; the compound-enum Display impls
/// ([`crate::ipc::wire::WireOverflowScope`] / [`crate::ipc::wire::WireWatchFailure`]) write through
/// the same formatter so the compound fields likewise carry no allocation.
pub(crate) fn render(out: &mut String, d: &WireDiagnostic, sty: Styler) {
    let _ = write!(
        out,
        "{}  {}",
        sty.paint(style::SECONDARY, at_field(d)),
        sty.paint(style::severity_style(severity(d)), d.variant_name()),
    );
    write_fields(out, d, sty);
    out.push('\n');
}

/// Project the variant's `at` field through the structural commitment that every [`WireDiagnostic`]
/// variant declares `at: WireTime` as its first field. Single or-pattern arm — a new variant
/// without an `at` field is a compile error here.
const fn at_field(d: &WireDiagnostic) -> &WireTime {
    match d {
        WireDiagnostic::StaleProbeResponse { at, .. }
        | WireDiagnostic::StaleTimer { at, .. }
        | WireDiagnostic::EffectCompleteOutsideAwaiting { at, .. }
        | WireDiagnostic::EffectCompleteForUnknownSub { at, .. }
        | WireDiagnostic::DetachUnknownSub { at, .. }
        | WireDiagnostic::ConfigDiffUnknownSub { at, .. }
        | WireDiagnostic::ConfigDiffRebindFallbackAttach { at, .. }
        | WireDiagnostic::ProbeVanished { at, .. }
        | WireDiagnostic::ProbeFailed { at, .. }
        | WireDiagnostic::EventClassDropped { at, .. }
        | WireDiagnostic::EventOutsideProofObject { at, .. }
        | WireDiagnostic::EventOnUnwatchedResource { at, .. }
        | WireDiagnostic::EventNoConsumer { at, .. }
        | WireDiagnostic::WatchOpRejected { at, .. }
        | WireDiagnostic::PendingPathProbeVanished { at, .. }
        | WireDiagnostic::PendingPathProbeFailed { at, .. }
        | WireDiagnostic::PendingPathAwaitingSegment { at, .. }
        | WireDiagnostic::ReapPendingCancelled { at, .. }
        | WireDiagnostic::ProfileReaped { at, .. }
        | WireDiagnostic::ProfileClaimPurged { at, .. }
        | WireDiagnostic::AttachPathInvalid { at, .. }
        | WireDiagnostic::AttachResourceStale { at, .. }
        | WireDiagnostic::AnchorKindMismatch { at, .. }
        | WireDiagnostic::SpliceCrossedUncovered { at, .. }
        | WireDiagnostic::EventAbsorbedByFireTail { at, .. }
        | WireDiagnostic::AwaitGateDeadlineForceRebasing { at, .. }
        | WireDiagnostic::AwaitGateDeadlineReap { at, .. }
        | WireDiagnostic::QuiescenceCeilingUnreadable { at, .. }
        | WireDiagnostic::QuiescenceCeilingForcedDespiteChange { at, .. }
        | WireDiagnostic::RebaseCeilingForced { at, .. }
        | WireDiagnostic::RebaseCeilingUnreadable { at, .. }
        | WireDiagnostic::ChangeOutsideEventMask { at, .. }
        | WireDiagnostic::SensorOverflow { at, .. }
        | WireDiagnostic::PerFileDriftDroppedOnRecovery { at, .. }
        | WireDiagnostic::PerFileFireSkippedOnFreshSeed { at, .. }
        | WireDiagnostic::SubAttached { at, .. }
        | WireDiagnostic::SubFired { at, .. }
        | WireDiagnostic::QuiescenceAbsorbed { at, .. }
        | WireDiagnostic::AbsorbArmed { at, .. }
        | WireDiagnostic::SubDetached { at, .. }
        | WireDiagnostic::SubRebound { at, .. }
        | WireDiagnostic::RebindUnknownSub { at, .. }
        | WireDiagnostic::DiscoveryMinted { at, .. }
        | WireDiagnostic::DiscoveryUnsupportedAnchorKind { at, .. }
        | WireDiagnostic::DiscoveryFanoutThreshold { at, .. }
        | WireDiagnostic::DiscoverySubReaped { at, .. }
        | WireDiagnostic::InvalidBurstTransition { at, .. }
        | WireDiagnostic::WalkerContractViolated { at, .. }
        | WireDiagnostic::Missed { at, .. } => at,
    }
}

/// Severity tier of a diagnostic — the hue [`render`] paints its variant tag with. Exhaustive
/// or-pattern match (a new variant without a tier is a compile error here).
///
/// The tiers mirror the daemon's own per-variant tracing levels in
/// `crate::driver::forward::log_diagnostic` — the established operator-facing severity catalogue —
/// so a `tail` line's tail colour matches the level the same event carries in the daemon log:
///
/// - `error!` → [`Severity::Error`] — a violated engine invariant the daemon flags loudly (a
///   malformed attach request, an anchor-kind mismatch, a walker-contract breach). Exactly three
///   variants.
/// - `warn!` → [`Severity::Warn`] — a degraded-but-recovered edge: a failed / vanished probe (the
///   errno self-recovers), a purged claim, a forced ceiling, an overflow reseed, a gate deadline, a
///   routing breach the helper bailed on.
/// - `info!` / `debug!` / `trace!` → [`Severity::Info`] — routine lifecycle, benign races, and
///   class / consumer drops.
///
/// Two deliberate departures from a literal level mirror:
///
/// - `SubFired` is `info!` in the daemon log but elevated to [`Severity::Ok`] here — a fire is the
///   headline positive event and reads green on a `tail`.
/// - `Missed` is wire-only (the slow-subscriber back-pressure marker has no `log_diagnostic` arm);
///   it is [`Severity::Warn`], the data-loss signal it shares with `SensorOverflow`.
///
/// Re-judging a variant means moving it here AND in `log_diagnostic` (its rustdoc carries the
/// reciprocal note). Tiers are an operator-triage projection, not a wire contract: re-tiering
/// changes only a line's colour, never its bytes.
const fn severity(d: &WireDiagnostic) -> Severity {
    use WireDiagnostic as W;
    match d {
        // `info!` in the log, elevated: the headline positive event.
        W::SubFired { .. } => Severity::Ok,

        // `error!` in the log — a violated engine invariant.
        W::AttachPathInvalid { .. }
        | W::AnchorKindMismatch { .. }
        | W::WalkerContractViolated { .. } => Severity::Error,

        // `warn!` in the log — a degraded-but-recovered edge — plus the wire-only `Missed`
        // data-loss marker.
        W::StaleProbeResponse { .. }
        | W::StaleTimer { .. }
        | W::EffectCompleteOutsideAwaiting { .. }
        | W::EffectCompleteForUnknownSub { .. }
        | W::DetachUnknownSub { .. }
        | W::ProbeVanished { .. }
        | W::ProbeFailed { .. }
        | W::EventOnUnwatchedResource { .. }
        | W::WatchOpRejected { .. }
        | W::PendingPathProbeVanished { .. }
        | W::PendingPathProbeFailed { .. }
        | W::ProfileClaimPurged { .. }
        | W::AttachResourceStale { .. }
        | W::SpliceCrossedUncovered { .. }
        | W::AwaitGateDeadlineForceRebasing { .. }
        | W::AwaitGateDeadlineReap { .. }
        | W::QuiescenceCeilingUnreadable { .. }
        | W::QuiescenceCeilingForcedDespiteChange { .. }
        | W::RebaseCeilingForced { .. }
        | W::RebaseCeilingUnreadable { .. }
        | W::ChangeOutsideEventMask { .. }
        | W::SensorOverflow { .. }
        | W::PerFileDriftDroppedOnRecovery { .. }
        | W::RebindUnknownSub { .. }
        | W::DiscoveryUnsupportedAnchorKind { .. }
        | W::DiscoveryFanoutThreshold { .. }
        | W::InvalidBurstTransition { .. }
        | W::Missed { .. } => Severity::Warn,

        // `info!` / `debug!` / `trace!` — routine lifecycle, benign races, class / consumer drops.
        W::ConfigDiffUnknownSub { .. }
        | W::PendingPathAwaitingSegment { .. }
        | W::ConfigDiffRebindFallbackAttach { .. }
        | W::EventClassDropped { .. }
        | W::EventOutsideProofObject { .. }
        | W::EventNoConsumer { .. }
        | W::ReapPendingCancelled { .. }
        | W::ProfileReaped { .. }
        | W::EventAbsorbedByFireTail { .. }
        | W::PerFileFireSkippedOnFreshSeed { .. }
        | W::SubAttached { .. }
        | W::QuiescenceAbsorbed { .. }
        | W::AbsorbArmed { .. }
        | W::SubDetached { .. }
        | W::SubRebound { .. }
        | W::DiscoveryMinted { .. }
        | W::DiscoverySubReaped { .. } => Severity::Info,
    }
}

/// Append one ` key=value` field — `key` painted [`style::LABEL`], the `=` painted
/// [`style::DELIM`], the value unstyled. The two leading spaces are the inter-field separator every
/// line uses, so a run of
/// `field` calls reproduces the pre-color `  k=v  k2=v2` shape
/// byte-for-byte under `Styler::Plain`.
fn field(out: &mut String, sty: Styler, key: &str, value: impl Display) {
    let _ = write!(
        out,
        "  {}{}{value}",
        sty.paint(style::LABEL, key),
        sty.paint(style::DELIM, "="),
    );
}

/// Append every non-`at` field as ` key=value` pairs via [`field`]. Exhaustive match — a new
/// variant lands a compile error here, paired with the matching arm in
/// [`WireDiagnostic::variant_name`] and the `KNOWN_WIRE_VARIANTS` tag list.
///
/// Field order mirrors the variant's declaration order so the human form and the JSON form present
/// fields in the same sequence.
///
/// Snake-rename'd wire enum fields render through `Display`; see the per-enum `as_str` impls in
/// [`super::super::wire`] and [`super::super::protocol`].
fn write_fields(out: &mut String, d: &WireDiagnostic, sty: Styler) {
    match d {
        WireDiagnostic::StaleProbeResponse {
            owner, correlation, ..
        } => {
            field(out, sty, "owner", owner.0);
            field(out, sty, "correlation", correlation);
        }
        WireDiagnostic::StaleTimer { id, .. } => {
            field(out, sty, "id", id);
        }
        WireDiagnostic::EffectCompleteOutsideAwaiting { sub, profile, .. } => {
            field(out, sty, "sub", sub.0);
            field(out, sty, "profile", profile.0);
        }
        WireDiagnostic::EffectCompleteForUnknownSub { sub, .. } => {
            field(out, sty, "sub", sub.0);
        }
        WireDiagnostic::DetachUnknownSub { sub, .. } => {
            field(out, sty, "sub", sub.0);
        }
        WireDiagnostic::ConfigDiffUnknownSub { name, .. }
        | WireDiagnostic::ConfigDiffRebindFallbackAttach { name, .. } => {
            field(out, sty, "name", name);
        }
        WireDiagnostic::ProbeVanished {
            profile, intent, ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "intent", intent);
        }
        WireDiagnostic::ProbeFailed {
            profile,
            intent,
            errno,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "intent", intent);
            field(out, sty, "errno", errno);
        }
        WireDiagnostic::EventClassDropped {
            resource,
            event,
            profile,
            ..
        }
        | WireDiagnostic::EventOutsideProofObject {
            resource,
            event,
            profile,
            ..
        } => {
            field(out, sty, "resource", resource.0);
            field(out, sty, "event", event);
            field(out, sty, "profile", profile.0);
        }
        WireDiagnostic::EventOnUnwatchedResource { resource, .. }
        | WireDiagnostic::EventNoConsumer { resource, .. }
        | WireDiagnostic::AttachResourceStale { resource, .. } => {
            field(out, sty, "resource", resource.0);
        }
        WireDiagnostic::WatchOpRejected {
            resource, failure, ..
        } => {
            field(out, sty, "resource", resource.0);
            field(out, sty, "failure", failure);
        }
        WireDiagnostic::PendingPathProbeVanished {
            profile, prefix, ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "prefix", prefix.0);
        }
        WireDiagnostic::PendingPathProbeFailed {
            profile,
            prefix,
            errno,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "prefix", prefix.0);
            field(out, sty, "errno", errno);
        }
        WireDiagnostic::PendingPathAwaitingSegment {
            profile,
            prefix,
            segment,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "prefix", prefix.0);
            field(out, sty, "segment", segment);
        }
        WireDiagnostic::ReapPendingCancelled { profile, .. }
        | WireDiagnostic::PerFileDriftDroppedOnRecovery { profile, .. }
        | WireDiagnostic::PerFileFireSkippedOnFreshSeed { profile, .. }
        | WireDiagnostic::QuiescenceAbsorbed { profile, .. } => {
            field(out, sty, "profile", profile.0);
        }
        WireDiagnostic::AbsorbArmed { profile, mode, .. } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "mode", mode);
        }
        WireDiagnostic::ProfileReaped { profile, via, .. } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "via", via);
        }
        WireDiagnostic::ProfileClaimPurged {
            profile,
            claim,
            resource,
            failure,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "claim", claim);
            field(out, sty, "resource", resource.0);
            field(out, sty, "failure", failure);
        }
        WireDiagnostic::AttachPathInvalid { path, hint, .. } => {
            field(out, sty, "path", path);
            field(out, sty, "hint", hint);
        }
        WireDiagnostic::AnchorKindMismatch {
            profile,
            prior_kind,
            response_kind,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "prior_kind", prior_kind);
            field(out, sty, "response_kind", response_kind);
        }
        WireDiagnostic::SpliceCrossedUncovered {
            profile,
            target,
            cause,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "target", target.0);
            field(out, sty, "cause", cause);
        }
        WireDiagnostic::EventAbsorbedByFireTail {
            profile,
            resource,
            event,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "resource", resource.0);
            field(out, sty, "event", event);
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
            field(out, sty, "profile", profile.0);
            field(out, sty, "outstanding", outstanding);
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
            field(out, sty, "profile", profile.0);
            field(out, sty, "first_unread", first_unread);
            field(out, sty, "intent", intent);
        }
        WireDiagnostic::QuiescenceCeilingForcedDespiteChange {
            profile, intent, ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "intent", intent);
        }
        WireDiagnostic::RebaseCeilingForced {
            profile,
            intent,
            observed_change,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "intent", intent);
            field(out, sty, "observed_change", observed_change);
        }
        WireDiagnostic::ChangeOutsideEventMask {
            profile,
            intent,
            retries,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "intent", intent);
            field(out, sty, "retries", retries);
        }
        WireDiagnostic::SensorOverflow { scope, .. } => {
            field(out, sty, "scope", scope);
        }
        WireDiagnostic::SubAttached {
            sub,
            name,
            minted_by,
            ..
        } => {
            field(out, sty, "sub", sub.0);
            field(out, sty, "name", name);
            if let Some(src) = minted_by {
                field(out, sty, "minted_by", src.0);
            }
        }
        WireDiagnostic::SubFired {
            sub,
            profile,
            count,
            ..
        } => {
            field(out, sty, "sub", sub.0);
            field(out, sty, "profile", profile.0);
            field(out, sty, "count", count);
        }
        WireDiagnostic::SubDetached {
            sub,
            profile,
            reason,
            ..
        } => {
            field(out, sty, "sub", sub.0);
            field(out, sty, "profile", profile.0);
            field(out, sty, "reason", reason);
        }
        WireDiagnostic::SubRebound { sub, .. } | WireDiagnostic::RebindUnknownSub { sub, .. } => {
            field(out, sty, "sub", sub.0);
        }
        WireDiagnostic::DiscoveryMinted {
            source, path, kind, ..
        } => {
            field(out, sty, "source", source.0);
            field(out, sty, "path", path);
            field(out, sty, "kind", kind);
        }
        WireDiagnostic::DiscoveryUnsupportedAnchorKind {
            source, path, kind, ..
        } => {
            field(out, sty, "source", source.0);
            field(out, sty, "path", path);
            field(out, sty, "kind", kind);
        }
        WireDiagnostic::DiscoveryFanoutThreshold { source, count, .. } => {
            field(out, sty, "source", source.0);
            field(out, sty, "count", count);
        }
        WireDiagnostic::DiscoverySubReaped {
            source, sub, path, ..
        } => {
            field(out, sty, "source", source.0);
            field(out, sty, "sub", sub.0);
            field(out, sty, "path", path);
        }
        WireDiagnostic::InvalidBurstTransition {
            profile,
            helper,
            observed,
            ..
        } => {
            field(out, sty, "profile", profile.0);
            field(out, sty, "helper", helper);
            field(out, sty, "observed", observed);
        }
        WireDiagnostic::WalkerContractViolated { owner, .. } => {
            field(out, sty, "owner", owner.0);
        }
        WireDiagnostic::Missed { count, .. } => {
            field(out, sty, "count", count);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use crate::ipc::protocol::WireId;
    use crate::ipc::render::style::Styler;
    use crate::ipc::wire::{
        WireBurstIntent, WireDetachReason, WireDiagnostic, WireFsEvent, WireOverflowScope, WireTime,
    };
    use std::time::UNIX_EPOCH;

    /// The renderer's row shape: timestamp first, variant second, fields after, newline last. Pin
    /// per-Sub event details so the operator-visible fields stay structural.
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
            Styler::Plain,
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

    /// `Missed` is the back-pressure marker. Its tag is the underscore-prefixed `_missed` (the only
    /// variant with an override) — operators reading a `_missed N` line know the daemon dropped
    /// events upstream.
    #[test]
    fn render_missed_marker_is_underscore_prefixed() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::Missed {
                at: WireTime::from(UNIX_EPOCH),
                count: 5,
            },
            Styler::Plain,
        );
        assert!(
            s.starts_with("1970-01-01T00:00:00Z  _missed"),
            "leading `_missed` tag: {s:?}",
        );
        assert!(s.contains("  count=5"), "count field present: {s:?}");
        assert!(s.ends_with('\n'), "newline-terminated: {s:?}");
    }

    /// `SubAttached.minted_by = Some(_)` renders an extra `minted_by=N` field; `None` omits it
    /// entirely. Operators distinguishing operator-declared vs discovery-minted Subs read the
    /// presence/absence of the field.
    #[test]
    fn render_sub_attached_discovery_field_optional() {
        let mut static_attach = String::new();
        render(
            &mut static_attach,
            &WireDiagnostic::SubAttached {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(1),
                name: "static_watch".into(),
                minted_by: None,
            },
            Styler::Plain,
        );
        assert!(static_attach.contains("name=static_watch"));
        assert!(
            !static_attach.contains("minted_by"),
            "None must omit minted_by: {static_attach:?}",
        );

        let mut dynamic_attach = String::new();
        render(
            &mut dynamic_attach,
            &WireDiagnostic::SubAttached {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(2),
                name: "t@/tmp/x".into(),
                minted_by: Some(WireId(99)),
            },
            Styler::Plain,
        );
        assert!(dynamic_attach.contains("minted_by=99"));
    }

    /// `SubDetached.reason` renders through the typed [`WireDetachReason`] label table
    /// (`config_diff_removed`, `ipc_disabled`, etc.). Mirrors the snake_case serde rename so the
    /// human form matches the JSON.
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
            Styler::Plain,
        );
        assert!(s.contains("  reason=ipc_disabled"), "got: {s:?}");
    }

    /// A Profile-keyed variant (`ProfileReaped`) renders without any `sub=` field — operators
    /// reading a tail know the event is not per-Sub by the absence of the column. Distinct from
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
            Styler::Plain,
        );
        assert!(s.contains("  profile=7"));
        assert!(s.contains("  via=deferred_from_burst"));
        assert!(
            !s.contains("sub="),
            "Profile-keyed variant must not carry a sub field: {s:?}",
        );
    }

    /// `EventClassDropped` is the canonical multi-field per-Resource variant; its rendered line
    /// carries resource, event, profile in declaration order with the correct enum label for the
    /// event.
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
            Styler::Plain,
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

    /// `WireOverflowScope::Global` is the bare-tag variant — rendering must emit `scope=global`
    /// without any `/id` tail.
    #[test]
    fn render_sensor_overflow_global_scope_bare() {
        let mut s = String::new();
        render(
            &mut s,
            &WireDiagnostic::SensorOverflow {
                at: WireTime::from(UNIX_EPOCH),
                scope: WireOverflowScope::Global,
            },
            Styler::Plain,
        );
        assert!(s.contains("  scope=global"), "got: {s:?}");
        assert!(
            !s.contains("scope=global/"),
            "global must NOT carry a trailing id: {s:?}",
        );
    }

    /// Cross-variant coverage: every variant on the [`super::super::wire::KNOWN_WIRE_VARIANTS`]
    /// list renders to a non-empty line that contains its tag. Catches a missing `write_fields` arm
    /// before the operator hits it. Uses the same witness fixture the wire round-trip test consumes
    /// — duplicating would invite drift.
    #[test]
    fn every_variant_renders_a_nonempty_line_with_its_tag() {
        // The wire test fixture lives in wire's test module; we can't reach it cross-module, so we
        // exercise the KNOWN_WIRE_VARIANTS surface here via the JSON wire round trip — every
        // wire-variant witness round-trips, and the renderer reads back the structural shape. Reach
        // the wire-side test seam through serde directly.
        //
        // For each variant tag, synthesize the minimum-field JSON shape and feed it through
        // deserialize → render. A wire variant whose JSON shape doesn't round-trip would already fail
        // the wire's own round-trip test; this guard reports any render-side regression separately.
        let timestamp = serde_json::to_value(WireTime::from(UNIX_EPOCH)).unwrap();
        for tag in crate::ipc::wire::KNOWN_WIRE_VARIANTS {
            let value = synthesize_min_value(tag, &timestamp);
            let wire: WireDiagnostic = serde_json::from_value(value).unwrap_or_else(|e| {
                panic!("failed to deserialize witness for tag {tag}: {e}");
            });
            let mut line = String::new();
            render(&mut line, &wire, Styler::Plain);
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

    /// Minimum-field JSON value for one variant tag. Constructs the smallest acceptable payload so
    /// [`WireDiagnostic::deserialize`] succeeds; any new variant added without a paired arm here
    /// fails [`every_variant_renders_a_nonempty_line_with_its_tag`] loudly.
    fn synthesize_min_value(tag: &str, at: &serde_json::Value) -> serde_json::Value {
        use serde_json::json;
        let id = json!(1);
        let global_scope = json!({ "scope": "global" });
        let pressure = json!({ "kind": "pressure", "errno": 24 });
        match tag {
            "stale_probe_response" => {
                json!({"diag": tag, "at": at, "owner": id, "correlation": 1})
            }
            "stale_timer" => json!({"diag": tag, "at": at, "id": 1}),
            "effect_complete_outside_awaiting" => {
                json!({"diag": tag, "at": at, "sub": id, "profile": id})
            }
            "effect_complete_for_unknown_sub" | "detach_unknown_sub" => {
                json!({"diag": tag, "at": at, "sub": id})
            }
            "config_diff_unknown_sub" | "config_diff_rebind_fallback_attach" => {
                json!({"diag": tag, "at": at, "name": "x"})
            }
            "probe_vanished" => {
                json!({"diag": tag, "at": at, "profile": id, "intent": "standard"})
            }
            "probe_failed" => {
                json!({"diag": tag, "at": at, "profile": id, "intent": "standard", "errno": 0})
            }
            "event_class_dropped" | "event_outside_proof_object" => {
                json!({"diag": tag, "at": at, "resource": id, "event": "content_changed", "profile": id})
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
            "pending_path_awaiting_segment" => {
                json!({"diag": tag, "at": at, "profile": id, "prefix": id, "segment": "x"})
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
                "event": "content_changed",
            }),
            "await_gate_deadline_force_rebasing" | "await_gate_deadline_reap" => {
                json!({"diag": tag, "at": at, "profile": id, "outstanding": 1})
            }
            "quiescence_ceiling_unreadable" | "rebase_ceiling_unreadable" => json!({
                "diag": tag, "at": at, "profile": id,
                "first_unread": "/x", "intent": "standard",
            }),
            "quiescence_ceiling_forced_despite_change" => {
                json!({"diag": tag, "at": at, "profile": id, "intent": "standard"})
            }
            "rebase_ceiling_forced" => json!({
                "diag": tag, "at": at, "profile": id,
                "intent": "standard", "observed_change": false,
            }),
            "change_outside_event_mask" => json!({
                "diag": tag, "at": at, "profile": id,
                "intent": "standard", "retries": 3,
            }),
            "sensor_overflow" => json!({"diag": tag, "at": at, "scope": global_scope}),
            "sub_attached" => json!({
                "diag": tag, "at": at, "sub": id, "name": "x", "minted_by": null,
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
            "discovery_minted" => json!({
                "diag": tag, "at": at, "source": id, "path": "/x", "kind": "dir",
            }),
            "discovery_unsupported_anchor_kind" => json!({
                "diag": tag, "at": at, "source": id, "path": "/x", "kind": "symlink",
            }),
            "discovery_fanout_threshold" => json!({
                "diag": tag, "at": at, "source": id, "count": 1024,
            }),
            "discovery_sub_reaped" => json!({
                "diag": tag, "at": at, "source": id, "sub": id, "path": "/x",
            }),
            "invalid_burst_transition" => json!({
                "diag": tag, "at": at, "profile": id,
                "helper": "transition_to_verifying", "observed": "idle",
            }),
            "walker_contract_violated" => {
                json!({"diag": tag, "at": at, "owner": id})
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
            Styler::Plain,
        );
        assert!(s.contains("  intent=seed"), "got: {s:?}");
        assert!(s.contains("  errno=13"), "got: {s:?}");
    }

    /// `severity` classifies one representative per tier. Exhaustiveness is the compiler's (the
    /// or-pattern match owns every variant); this pins the four anchors so a mis-tiered
    /// representative is caught.
    #[test]
    fn severity_classifies_one_representative_per_tier() {
        use super::severity;
        use crate::ipc::render::style::Severity;

        // Ok — the headline positive event.
        assert_eq!(
            severity(&WireDiagnostic::SubFired {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(1),
                profile: WireId(2),
                count: 1,
            }),
            Severity::Ok,
        );
        // Error — a violated engine invariant (the daemon's `error!`).
        assert_eq!(
            severity(&WireDiagnostic::WalkerContractViolated {
                at: WireTime::from(UNIX_EPOCH),
                owner: WireId(1),
            }),
            Severity::Error,
        );
        // Warn — a probe failure carries an errno but self-recovers, so it mirrors the daemon's
        // `warn!`, NOT `error!`. Pinned here so the variant stays at Warn and does not drift to
        // Error.
        assert_eq!(
            severity(&WireDiagnostic::ProbeFailed {
                at: WireTime::from(UNIX_EPOCH),
                profile: WireId(1),
                intent: WireBurstIntent::Seed,
                errno: 13,
            }),
            Severity::Warn,
        );
        // Info — routine lifecycle narration.
        assert_eq!(
            severity(&WireDiagnostic::SubAttached {
                at: WireTime::from(UNIX_EPOCH),
                sub: WireId(1),
                name: "w".into(),
                minted_by: None,
            }),
            Severity::Info,
        );
    }

    /// Under `Styler::Active` the variant tag wears its severity hue (`Ok`→green for `sub_fired`,
    /// `Error`→red for `probe_failed`) and the field keys gain SGR; stripping every escape
    /// reproduces the `Plain` line byte-for-byte (color is purely additive).
    #[test]
    fn active_colors_tag_by_severity_and_strips_to_plain() {
        use crate::ipc::render::style::{Severity, severity_style, strip_ansi};

        let fired = WireDiagnostic::SubFired {
            at: WireTime::from(UNIX_EPOCH),
            sub: WireId(11),
            profile: WireId(22),
            count: 3,
        };
        let mut active = String::new();
        render(&mut active, &fired, Styler::Active);
        let green = severity_style(Severity::Ok).render().to_string();
        assert!(
            active.contains(&format!("{green}sub_fired")),
            "Ok tag wears the green severity hue: {active:?}",
        );
        let mut plain = String::new();
        render(&mut plain, &fired, Styler::Plain);
        assert_eq!(strip_ansi(&active), plain, "stripping Active yields Plain");

        // A probe failure is the daemon's `warn!` — yellow, not red — even though an errno rides
        // the line (the cause self-recovers).
        let failed = WireDiagnostic::ProbeFailed {
            at: WireTime::from(UNIX_EPOCH),
            profile: WireId(1),
            intent: WireBurstIntent::Seed,
            errno: 13,
        };
        let mut active = String::new();
        render(&mut active, &failed, Styler::Active);
        let yellow = severity_style(Severity::Warn).render().to_string();
        assert!(
            active.contains(&format!("{yellow}probe_failed")),
            "Warn tag wears the yellow severity hue: {active:?}",
        );

        // Red is reserved for a violated invariant.
        let breach = WireDiagnostic::WalkerContractViolated {
            at: WireTime::from(UNIX_EPOCH),
            owner: WireId(1),
        };
        let mut active = String::new();
        render(&mut active, &breach, Styler::Active);
        let red = severity_style(Severity::Error).render().to_string();
        assert!(
            active.contains(&format!("{red}walker_contract_violated")),
            "Error tag wears the red severity hue: {active:?}",
        );
    }
}
