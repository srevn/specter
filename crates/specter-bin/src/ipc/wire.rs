//! Wire-side projection of `specter_core::Diagnostic` and every
//! enum it transitively reaches.
//!
//! # The structural wall
//!
//! [`WireDiagnostic`]'s [`From<(&Diagnostic, SystemTime)>`] is an
//! exhaustive `match` — no wildcard, no fallback. A new
//! [`specter_core::Diagnostic`] variant is a compile error here, and
//! the same discipline mirrors out across every per-core-type
//! `Wire*` enum: a new core variant fails the matching `From` arm.
//! Adding a wire variant is a paired edit (declare it, write its
//! `From` arm) so no schema change can land silently.
//!
//! # Deserialize policy
//!
//! [`WireDiagnostic`] is **two-way**: the daemon serializes for the
//! broker fan-out (the [`From<(&Diagnostic, SystemTime)>`] projection
//! at write time), and operator clients (`specter tail`, `specter
//! wait`) deserialize the streamed JSON lines back into the typed
//! enum. Every wire enum it transitively reaches carries both
//! `Serialize` and `Deserialize`; round-trip is structural over the
//! `#[serde]` tags.
//!
//! Adding a [`WireDiagnostic`] variant is a paired edit: declare it,
//! write its [`From<(&Diagnostic, SystemTime)>`] arm, add the matching
//! arm in [`WireDiagnostic::variant_name`], and add a tag entry in
//! [`KNOWN_WIRE_VARIANTS`]. The first three edits are exhaustive
//! `match` arms so the compiler refuses the change without them; the
//! fourth is pinned by a drift test that fails on either side
//! diverging from the witness set.
//!
//! `WireTime` owns its own formatting via
//! `humantime::format_rfc3339_seconds` rather than going through
//! serde for the *outgoing* path; deserialization treats it as the
//! transparent `String` it serializes to. Pre-epoch `SystemTime` is
//! clamped to `UNIX_EPOCH` on conversion to defuse `humantime`'s
//! pre-epoch panic.
//!
//! # Field ordering
//!
//! Every [`WireDiagnostic`] variant declares `at: WireTime` as its
//! first field so it serializes immediately after the `diag` tag.
//! `jq` filters and operator inspection both benefit from a
//! predictable timestamp position.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. Operator clients ship inside the
//! same binary, so the wire surface stays a bin-internal contract.

use serde::{Deserialize, Serialize};
use specter_core::{
    BurstHelper, BurstIntent, ClaimKind, DetachReason, Diagnostic, EffectScope, FsEvent,
    OverflowScope, ProbeOwner, ProfileStateDiscriminant, PromoterClaimKind, ReapTrigger,
    ResourceKind, SpliceFailureCause, StateLabel, WatchFailure,
};
use std::path::Path;
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::protocol::WireId;

/// RFC 3339 wall-clock projection at second resolution.
///
/// Second precision matches the `SPECTER_AT` subprocess env
/// ([`specter_actuator`]'s `format_now`) so operators see one
/// timestamp shape across both surfaces. Sub-second digits would be
/// unread by every current consumer and precise-but-NTP-inaccurate
/// on the synthesized `last_fired_at` projection
/// (`project::project_wall`), so the wire disclaims them.
///
/// `humantime::format_rfc3339_seconds` panics on pre-epoch
/// `SystemTime`; the clamp to [`UNIX_EPOCH`] defuses it. NTP
/// stepping, an operator `date` reset, or container clock skew at
/// boot can all produce a pre-epoch value in the wild, so the clamp
/// is defense-in-depth, not a theoretical concern.
///
/// `#[serde(transparent)]` makes the JSON form a bare quoted string
/// (`"2026-05-23T15:30:00Z"`), not a wrapped object.
///
/// Client-side `Deserialize` recovers the same `String` shape the
/// server wrote — the round-trip skips re-parsing humantime, treating
/// the field as an opaque RFC 3339 token the renderer reproduces
/// verbatim.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub(crate) struct WireTime(String);

impl From<SystemTime> for WireTime {
    fn from(t: SystemTime) -> Self {
        let clamped = t.max(UNIX_EPOCH);
        if clamped != t {
            tracing::warn!(
                ?t,
                "specter ipc: pre-epoch SystemTime clamped to UNIX_EPOCH",
            );
        }
        Self(humantime::format_rfc3339_seconds(clamped).to_string())
    }
}

impl std::fmt::Display for WireTime {
    /// Renderers reproduce the RFC 3339 token verbatim through
    /// `Display`, so the token is a zero-alloc `&str` write — the
    /// only consumer outside the JSON wire path.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Broker-to-subscriber envelope.
///
/// Not [`Serialize`]: the per-connection thread projects
/// `&BrokerEvent → WireDiagnostic` at write time, so wire-string
/// allocation never blocks the broker's dispatch lock.
///
/// - [`Self::Diag`] carries a cloned [`Diagnostic`] paired with the
///   wall-clock instant captured once per `forward()` fanout.
///   The same `at` reaches every subscriber for one engine
///   emission — operators correlating events across `tail` clients
///   see identical timestamps for the same underlying event.
/// - [`Self::Missed`] is the broker's back-pressure marker, flushed
///   lazily on the next successful send after one or more dropped
///   `try_send`s.
#[derive(Clone, Debug)]
pub(crate) enum BrokerEvent {
    Diag { diag: Diagnostic, at: SystemTime },
    Missed { count: u32, at: SystemTime },
}

impl From<&BrokerEvent> for WireDiagnostic {
    fn from(ev: &BrokerEvent) -> Self {
        match ev {
            BrokerEvent::Diag { diag, at } => Self::from((diag, *at)),
            BrokerEvent::Missed { count, at } => Self::Missed {
                at: WireTime::from(*at),
                count: *count,
            },
        }
    }
}

/// JSON-line projection of `specter_core::Diagnostic` plus the
/// broker's `_missed` back-pressure marker.
///
/// Internally tagged on `diag`; every variant's `at` field
/// serializes immediately after the tag.
///
/// Two-way derive (server serializes for fan-out, client
/// deserializes for tail/wait) — see the module rustdoc's
/// `Deserialize policy` section for the structural invariants the
/// paired edit must preserve.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "diag")]
pub(crate) enum WireDiagnostic {
    StaleProbeResponse {
        at: WireTime,
        owner: WireProbeOwner,
        correlation: u64,
    },
    StaleTimer {
        at: WireTime,
        id: u64,
    },
    EffectCompleteOutsideAwaiting {
        at: WireTime,
        sub: WireId,
        profile: WireId,
    },
    EffectCompleteForUnknownSub {
        at: WireTime,
        sub: WireId,
    },
    DetachUnknownSub {
        at: WireTime,
        sub: WireId,
    },
    ConfigDiffUnknownSub {
        at: WireTime,
        name: String,
    },
    ConfigDiffUnknownPromoter {
        at: WireTime,
        name: String,
    },
    ConfigDiffRebindFallbackAttach {
        at: WireTime,
        name: String,
    },
    ProbeVanished {
        at: WireTime,
        profile: WireId,
        intent: WireBurstIntent,
    },
    ProbeFailed {
        at: WireTime,
        profile: WireId,
        intent: WireBurstIntent,
        errno: i32,
    },
    EventClassDropped {
        at: WireTime,
        resource: WireId,
        event: WireFsEvent,
        profile: WireId,
    },
    EventOnUnwatchedResource {
        at: WireTime,
        resource: WireId,
    },
    EventNoConsumer {
        at: WireTime,
        resource: WireId,
    },
    WatchOpRejected {
        at: WireTime,
        resource: WireId,
        failure: WireWatchFailure,
    },
    PendingPathProbeVanished {
        at: WireTime,
        profile: WireId,
        prefix: WireId,
    },
    PendingPathProbeFailed {
        at: WireTime,
        profile: WireId,
        prefix: WireId,
        errno: i32,
    },
    ReapPendingCancelled {
        at: WireTime,
        profile: WireId,
    },
    ProfileReaped {
        at: WireTime,
        profile: WireId,
        via: WireReapTrigger,
    },
    ProfileClaimPurged {
        at: WireTime,
        profile: WireId,
        claim: WireClaimKind,
        resource: WireId,
        failure: WireWatchFailure,
    },
    PromoterClaimPurged {
        at: WireTime,
        promoter: WireId,
        claim: WirePromoterClaimKind,
        resource: WireId,
        failure: WireWatchFailure,
    },
    AttachPathInvalid {
        at: WireTime,
        path: String,
        /// Operator-visible explanation of *why* the path was
        /// rejected. The core-side carrier is a `&'static str`
        /// literal; on the wire it becomes an owned [`String`] so
        /// the symmetric client deserialize lifts cleanly into the
        /// same shape every other text field on this enum carries.
        hint: String,
    },
    AttachResourceStale {
        at: WireTime,
        resource: WireId,
    },
    AnchorKindMismatch {
        at: WireTime,
        profile: WireId,
        prior_kind: WireResourceKind,
        response_kind: WireResourceKind,
    },
    SpliceCrossedUncovered {
        at: WireTime,
        profile: WireId,
        target: WireId,
        cause: WireSpliceFailureCause,
    },
    EventAbsorbedByFireTail {
        at: WireTime,
        profile: WireId,
        resource: WireId,
        event: WireFsEvent,
    },
    AwaitGateDeadlineForceRebasing {
        at: WireTime,
        profile: WireId,
        outstanding: u32,
    },
    AwaitGateDeadlineReap {
        at: WireTime,
        profile: WireId,
        outstanding: u32,
    },
    QuiescenceCeilingUnreadable {
        at: WireTime,
        profile: WireId,
        first_unread: String,
        intent: WireBurstIntent,
    },
    RebaseCeilingStillChanging {
        at: WireTime,
        profile: WireId,
        intent: WireBurstIntent,
    },
    RebaseCeilingUnreadable {
        at: WireTime,
        profile: WireId,
        first_unread: String,
        intent: WireBurstIntent,
    },
    SensorOverflow {
        at: WireTime,
        scope: WireOverflowScope,
    },
    PromoterReseededForOverflow {
        at: WireTime,
        promoter: WireId,
    },
    PerFileDriftDroppedOnRecovery {
        at: WireTime,
        profile: WireId,
    },
    PerFileFireSkippedOnFreshSeed {
        at: WireTime,
        profile: WireId,
    },
    SubAttached {
        at: WireTime,
        sub: WireId,
        name: String,
        source_promoter: Option<WireId>,
    },
    SubFired {
        at: WireTime,
        sub: WireId,
        profile: WireId,
        count: u32,
    },
    SubDetached {
        at: WireTime,
        sub: WireId,
        profile: WireId,
        reason: WireDetachReason,
    },
    SubRebound {
        at: WireTime,
        sub: WireId,
    },
    RebindUnknownSub {
        at: WireTime,
        sub: WireId,
    },
    PromoterAttached {
        at: WireTime,
        promoter: WireId,
        name: String,
    },
    PromoterReaped {
        at: WireTime,
        promoter: WireId,
    },
    PromoterDescentVanished {
        at: WireTime,
        promoter: WireId,
        prefix: WireId,
    },
    PromoterDescentFailed {
        at: WireTime,
        promoter: WireId,
        prefix: WireId,
        errno: i32,
    },
    PromotionKindObserved {
        at: WireTime,
        promoter: WireId,
        path: String,
        kind: WireResourceKind,
    },
    PromoterFanoutThreshold {
        at: WireTime,
        promoter: WireId,
        count: usize,
    },
    PromoterProxyStaleEvent {
        at: WireTime,
        promoter: WireId,
        resource: WireId,
    },
    PromoterEnumerationVanished {
        at: WireTime,
        promoter: WireId,
        proxy: WireId,
    },
    PromoterEnumerationFailed {
        at: WireTime,
        promoter: WireId,
        proxy: WireId,
        errno: i32,
    },
    DynamicSubReaped {
        at: WireTime,
        promoter: WireId,
        sub: WireId,
        path: String,
    },
    InvalidBurstTransition {
        at: WireTime,
        profile: WireId,
        helper: WireBurstHelper,
        observed: WireProfileStateDiscriminant,
    },
    /// Broker back-pressure marker — not derived from any
    /// `specter_core::Diagnostic`. The underscore-prefix protects
    /// against collision with any future core variant named
    /// `Missed`; `#[serde(rename = "_missed")]` overrides the
    /// PascalCase default.
    #[serde(rename = "_missed")]
    Missed {
        at: WireTime,
        count: u32,
    },
}

impl From<(&Diagnostic, SystemTime)> for WireDiagnostic {
    fn from((d, at): (&Diagnostic, SystemTime)) -> Self {
        let at = WireTime::from(at);
        match d {
            Diagnostic::StaleProbeResponse { owner, correlation } => Self::StaleProbeResponse {
                at,
                owner: WireProbeOwner::from(*owner),
                correlation: correlation.as_u64(),
            },
            Diagnostic::StaleTimer { id } => Self::StaleTimer {
                at,
                id: id.as_u64(),
            },
            Diagnostic::EffectCompleteOutsideAwaiting { sub, profile } => {
                Self::EffectCompleteOutsideAwaiting {
                    at,
                    sub: WireId::from(*sub),
                    profile: WireId::from(*profile),
                }
            }
            Diagnostic::EffectCompleteForUnknownSub { sub } => Self::EffectCompleteForUnknownSub {
                at,
                sub: WireId::from(*sub),
            },
            Diagnostic::DetachUnknownSub { sub } => Self::DetachUnknownSub {
                at,
                sub: WireId::from(*sub),
            },
            Diagnostic::ConfigDiffUnknownSub { name } => Self::ConfigDiffUnknownSub {
                at,
                name: name.to_string(),
            },
            Diagnostic::ConfigDiffUnknownPromoter { name } => Self::ConfigDiffUnknownPromoter {
                at,
                name: name.to_string(),
            },
            Diagnostic::ConfigDiffRebindFallbackAttach { name } => {
                Self::ConfigDiffRebindFallbackAttach {
                    at,
                    name: name.to_string(),
                }
            }
            Diagnostic::ProbeVanished { profile, intent } => Self::ProbeVanished {
                at,
                profile: WireId::from(*profile),
                intent: WireBurstIntent::from(*intent),
            },
            Diagnostic::ProbeFailed {
                profile,
                intent,
                errno,
            } => Self::ProbeFailed {
                at,
                profile: WireId::from(*profile),
                intent: WireBurstIntent::from(*intent),
                errno: *errno,
            },
            Diagnostic::EventClassDropped {
                resource,
                event,
                profile,
            } => Self::EventClassDropped {
                at,
                resource: WireId::from(*resource),
                event: WireFsEvent::from(*event),
                profile: WireId::from(*profile),
            },
            Diagnostic::EventOnUnwatchedResource { resource } => Self::EventOnUnwatchedResource {
                at,
                resource: WireId::from(*resource),
            },
            Diagnostic::EventNoConsumer { resource } => Self::EventNoConsumer {
                at,
                resource: WireId::from(*resource),
            },
            Diagnostic::WatchOpRejected { resource, failure } => Self::WatchOpRejected {
                at,
                resource: WireId::from(*resource),
                failure: WireWatchFailure::from(*failure),
            },
            Diagnostic::PendingPathProbeVanished { profile, prefix } => {
                Self::PendingPathProbeVanished {
                    at,
                    profile: WireId::from(*profile),
                    prefix: WireId::from(*prefix),
                }
            }
            Diagnostic::PendingPathProbeFailed {
                profile,
                prefix,
                errno,
            } => Self::PendingPathProbeFailed {
                at,
                profile: WireId::from(*profile),
                prefix: WireId::from(*prefix),
                errno: *errno,
            },
            Diagnostic::ReapPendingCancelled { profile } => Self::ReapPendingCancelled {
                at,
                profile: WireId::from(*profile),
            },
            Diagnostic::ProfileReaped { profile, via } => Self::ProfileReaped {
                at,
                profile: WireId::from(*profile),
                via: WireReapTrigger::from(*via),
            },
            Diagnostic::ProfileClaimPurged {
                profile,
                claim,
                resource,
                failure,
            } => Self::ProfileClaimPurged {
                at,
                profile: WireId::from(*profile),
                claim: WireClaimKind::from(*claim),
                resource: WireId::from(*resource),
                failure: WireWatchFailure::from(*failure),
            },
            Diagnostic::PromoterClaimPurged {
                promoter,
                claim,
                resource,
                failure,
            } => Self::PromoterClaimPurged {
                at,
                promoter: WireId::from(*promoter),
                claim: WirePromoterClaimKind::from(*claim),
                resource: WireId::from(*resource),
                failure: WireWatchFailure::from(*failure),
            },
            Diagnostic::AttachPathInvalid { path, hint } => Self::AttachPathInvalid {
                at,
                path: arc_path_to_wire(path),
                hint: (*hint).to_owned(),
            },
            Diagnostic::AttachResourceStale { resource } => Self::AttachResourceStale {
                at,
                resource: WireId::from(*resource),
            },
            Diagnostic::AnchorKindMismatch {
                profile,
                prior_kind,
                response_kind,
            } => Self::AnchorKindMismatch {
                at,
                profile: WireId::from(*profile),
                prior_kind: WireResourceKind::from(*prior_kind),
                response_kind: WireResourceKind::from(*response_kind),
            },
            Diagnostic::SpliceCrossedUncovered {
                profile,
                target,
                cause,
            } => Self::SpliceCrossedUncovered {
                at,
                profile: WireId::from(*profile),
                target: WireId::from(*target),
                cause: WireSpliceFailureCause::from(*cause),
            },
            Diagnostic::EventAbsorbedByFireTail {
                profile,
                resource,
                event,
            } => Self::EventAbsorbedByFireTail {
                at,
                profile: WireId::from(*profile),
                resource: WireId::from(*resource),
                event: WireFsEvent::from(*event),
            },
            Diagnostic::AwaitGateDeadlineForceRebasing {
                profile,
                outstanding,
            } => Self::AwaitGateDeadlineForceRebasing {
                at,
                profile: WireId::from(*profile),
                outstanding: *outstanding,
            },
            Diagnostic::AwaitGateDeadlineReap {
                profile,
                outstanding,
            } => Self::AwaitGateDeadlineReap {
                at,
                profile: WireId::from(*profile),
                outstanding: *outstanding,
            },
            Diagnostic::QuiescenceCeilingUnreadable {
                profile,
                first_unread,
                intent,
            } => Self::QuiescenceCeilingUnreadable {
                at,
                profile: WireId::from(*profile),
                first_unread: arc_path_to_wire(first_unread),
                intent: WireBurstIntent::from(*intent),
            },
            Diagnostic::RebaseCeilingStillChanging { profile, intent } => {
                Self::RebaseCeilingStillChanging {
                    at,
                    profile: WireId::from(*profile),
                    intent: WireBurstIntent::from(*intent),
                }
            }
            Diagnostic::RebaseCeilingUnreadable {
                profile,
                first_unread,
                intent,
            } => Self::RebaseCeilingUnreadable {
                at,
                profile: WireId::from(*profile),
                first_unread: arc_path_to_wire(first_unread),
                intent: WireBurstIntent::from(*intent),
            },
            Diagnostic::SensorOverflow { scope } => Self::SensorOverflow {
                at,
                scope: WireOverflowScope::from(*scope),
            },
            Diagnostic::PromoterReseededForOverflow { promoter } => {
                Self::PromoterReseededForOverflow {
                    at,
                    promoter: WireId::from(*promoter),
                }
            }
            Diagnostic::PerFileDriftDroppedOnRecovery { profile } => {
                Self::PerFileDriftDroppedOnRecovery {
                    at,
                    profile: WireId::from(*profile),
                }
            }
            Diagnostic::PerFileFireSkippedOnFreshSeed { profile } => {
                Self::PerFileFireSkippedOnFreshSeed {
                    at,
                    profile: WireId::from(*profile),
                }
            }
            Diagnostic::SubAttached {
                sub,
                name,
                source_promoter,
            } => Self::SubAttached {
                at,
                sub: WireId::from(*sub),
                name: name.to_string(),
                source_promoter: source_promoter.map(WireId::from),
            },
            Diagnostic::SubFired {
                sub,
                profile,
                count,
            } => Self::SubFired {
                at,
                sub: WireId::from(*sub),
                profile: WireId::from(*profile),
                count: *count,
            },
            Diagnostic::SubDetached {
                sub,
                profile,
                reason,
            } => Self::SubDetached {
                at,
                sub: WireId::from(*sub),
                profile: WireId::from(*profile),
                reason: WireDetachReason::from(*reason),
            },
            Diagnostic::SubRebound { sub } => Self::SubRebound {
                at,
                sub: WireId::from(*sub),
            },
            Diagnostic::RebindUnknownSub { sub } => Self::RebindUnknownSub {
                at,
                sub: WireId::from(*sub),
            },
            Diagnostic::PromoterAttached { promoter, name } => Self::PromoterAttached {
                at,
                promoter: WireId::from(*promoter),
                name: name.to_string(),
            },
            Diagnostic::PromoterReaped { promoter } => Self::PromoterReaped {
                at,
                promoter: WireId::from(*promoter),
            },
            Diagnostic::PromoterDescentVanished { promoter, prefix } => {
                Self::PromoterDescentVanished {
                    at,
                    promoter: WireId::from(*promoter),
                    prefix: WireId::from(*prefix),
                }
            }
            Diagnostic::PromoterDescentFailed {
                promoter,
                prefix,
                errno,
            } => Self::PromoterDescentFailed {
                at,
                promoter: WireId::from(*promoter),
                prefix: WireId::from(*prefix),
                errno: *errno,
            },
            Diagnostic::PromotionKindObserved {
                promoter,
                path,
                kind,
            } => Self::PromotionKindObserved {
                at,
                promoter: WireId::from(*promoter),
                path: arc_path_to_wire(path),
                kind: WireResourceKind::from(*kind),
            },
            Diagnostic::PromoterFanoutThreshold { promoter, count } => {
                Self::PromoterFanoutThreshold {
                    at,
                    promoter: WireId::from(*promoter),
                    count: *count,
                }
            }
            Diagnostic::PromoterProxyStaleEvent { promoter, resource } => {
                Self::PromoterProxyStaleEvent {
                    at,
                    promoter: WireId::from(*promoter),
                    resource: WireId::from(*resource),
                }
            }
            Diagnostic::PromoterEnumerationVanished { promoter, proxy } => {
                Self::PromoterEnumerationVanished {
                    at,
                    promoter: WireId::from(*promoter),
                    proxy: WireId::from(*proxy),
                }
            }
            Diagnostic::PromoterEnumerationFailed {
                promoter,
                proxy,
                errno,
            } => Self::PromoterEnumerationFailed {
                at,
                promoter: WireId::from(*promoter),
                proxy: WireId::from(*proxy),
                errno: *errno,
            },
            Diagnostic::DynamicSubReaped {
                promoter,
                sub,
                path,
            } => Self::DynamicSubReaped {
                at,
                promoter: WireId::from(*promoter),
                sub: WireId::from(*sub),
                path: arc_path_to_wire(path),
            },
            Diagnostic::InvalidBurstTransition {
                profile,
                helper,
                observed,
            } => Self::InvalidBurstTransition {
                at,
                profile: WireId::from(*profile),
                helper: WireBurstHelper::from(*helper),
                observed: WireProfileStateDiscriminant::from(*observed),
            },
        }
    }
}

/// Project an [`Arc<Path>`] to its operator-visible string form.
/// Non-UTF-8 bytes ride as `U+FFFD REPLACEMENT CHARACTER`; the
/// schema stays JSON-safe and matches `tracing`'s own lossy path
/// projection.
fn arc_path_to_wire(p: &Arc<Path>) -> String {
    p.as_ref().to_string_lossy().into_owned()
}

impl WireDiagnostic {
    /// Wire tag for this variant — the same `"diag"` field value the
    /// JSON form carries. Mirrors the serde tag exactly: PascalCase
    /// by default (so [`Self::StaleProbeResponse`] →
    /// `"StaleProbeResponse"`) or the explicit `#[serde(rename =
    /// "...")]` override for [`Self::Missed`] → `"_missed"`.
    ///
    /// Exhaustive `match` — a new variant without a paired arm fails
    /// to compile, keeping the tag vocabulary single-source against
    /// [`KNOWN_WIRE_VARIANTS`].
    ///
    /// Used by `specter tail --filter <variant>` to dispatch lines
    /// client-side without re-serializing through
    /// `serde_json::Value`, and by per-event renderers that want the
    /// variant tag as a column without re-walking the JSON.
    pub(crate) const fn variant_name(&self) -> &'static str {
        match self {
            Self::StaleProbeResponse { .. } => "StaleProbeResponse",
            Self::StaleTimer { .. } => "StaleTimer",
            Self::EffectCompleteOutsideAwaiting { .. } => "EffectCompleteOutsideAwaiting",
            Self::EffectCompleteForUnknownSub { .. } => "EffectCompleteForUnknownSub",
            Self::DetachUnknownSub { .. } => "DetachUnknownSub",
            Self::ConfigDiffUnknownSub { .. } => "ConfigDiffUnknownSub",
            Self::ConfigDiffUnknownPromoter { .. } => "ConfigDiffUnknownPromoter",
            Self::ConfigDiffRebindFallbackAttach { .. } => "ConfigDiffRebindFallbackAttach",
            Self::ProbeVanished { .. } => "ProbeVanished",
            Self::ProbeFailed { .. } => "ProbeFailed",
            Self::EventClassDropped { .. } => "EventClassDropped",
            Self::EventOnUnwatchedResource { .. } => "EventOnUnwatchedResource",
            Self::EventNoConsumer { .. } => "EventNoConsumer",
            Self::WatchOpRejected { .. } => "WatchOpRejected",
            Self::PendingPathProbeVanished { .. } => "PendingPathProbeVanished",
            Self::PendingPathProbeFailed { .. } => "PendingPathProbeFailed",
            Self::ReapPendingCancelled { .. } => "ReapPendingCancelled",
            Self::ProfileReaped { .. } => "ProfileReaped",
            Self::ProfileClaimPurged { .. } => "ProfileClaimPurged",
            Self::PromoterClaimPurged { .. } => "PromoterClaimPurged",
            Self::AttachPathInvalid { .. } => "AttachPathInvalid",
            Self::AttachResourceStale { .. } => "AttachResourceStale",
            Self::AnchorKindMismatch { .. } => "AnchorKindMismatch",
            Self::SpliceCrossedUncovered { .. } => "SpliceCrossedUncovered",
            Self::EventAbsorbedByFireTail { .. } => "EventAbsorbedByFireTail",
            Self::AwaitGateDeadlineForceRebasing { .. } => "AwaitGateDeadlineForceRebasing",
            Self::AwaitGateDeadlineReap { .. } => "AwaitGateDeadlineReap",
            Self::QuiescenceCeilingUnreadable { .. } => "QuiescenceCeilingUnreadable",
            Self::RebaseCeilingStillChanging { .. } => "RebaseCeilingStillChanging",
            Self::RebaseCeilingUnreadable { .. } => "RebaseCeilingUnreadable",
            Self::SensorOverflow { .. } => "SensorOverflow",
            Self::PromoterReseededForOverflow { .. } => "PromoterReseededForOverflow",
            Self::PerFileDriftDroppedOnRecovery { .. } => "PerFileDriftDroppedOnRecovery",
            Self::PerFileFireSkippedOnFreshSeed { .. } => "PerFileFireSkippedOnFreshSeed",
            Self::SubAttached { .. } => "SubAttached",
            Self::SubFired { .. } => "SubFired",
            Self::SubDetached { .. } => "SubDetached",
            Self::SubRebound { .. } => "SubRebound",
            Self::RebindUnknownSub { .. } => "RebindUnknownSub",
            Self::PromoterAttached { .. } => "PromoterAttached",
            Self::PromoterReaped { .. } => "PromoterReaped",
            Self::PromoterDescentVanished { .. } => "PromoterDescentVanished",
            Self::PromoterDescentFailed { .. } => "PromoterDescentFailed",
            Self::PromotionKindObserved { .. } => "PromotionKindObserved",
            Self::PromoterFanoutThreshold { .. } => "PromoterFanoutThreshold",
            Self::PromoterProxyStaleEvent { .. } => "PromoterProxyStaleEvent",
            Self::PromoterEnumerationVanished { .. } => "PromoterEnumerationVanished",
            Self::PromoterEnumerationFailed { .. } => "PromoterEnumerationFailed",
            Self::DynamicSubReaped { .. } => "DynamicSubReaped",
            Self::InvalidBurstTransition { .. } => "InvalidBurstTransition",
            Self::Missed { .. } => "_missed",
        }
    }
}

/// Operator-visible tag for every [`WireDiagnostic`] variant — the
/// vocabulary `specter tail --filter` validates against and the
/// suggestion list the handler prints on a rejected token.
///
/// Hand-maintained alongside [`WireDiagnostic::variant_name`]; the
/// `known_wire_variants_matches_variant_name` drift test fails if
/// either side adds or drops an entry. Iteration order is the
/// variant declaration order on [`WireDiagnostic`] so operators
/// reading the "Known filters: ..." list see the same order the
/// source declares them in.
pub(crate) const KNOWN_WIRE_VARIANTS: &[&str] = &[
    "StaleProbeResponse",
    "StaleTimer",
    "EffectCompleteOutsideAwaiting",
    "EffectCompleteForUnknownSub",
    "DetachUnknownSub",
    "ConfigDiffUnknownSub",
    "ConfigDiffUnknownPromoter",
    "ConfigDiffRebindFallbackAttach",
    "ProbeVanished",
    "ProbeFailed",
    "EventClassDropped",
    "EventOnUnwatchedResource",
    "EventNoConsumer",
    "WatchOpRejected",
    "PendingPathProbeVanished",
    "PendingPathProbeFailed",
    "ReapPendingCancelled",
    "ProfileReaped",
    "ProfileClaimPurged",
    "PromoterClaimPurged",
    "AttachPathInvalid",
    "AttachResourceStale",
    "AnchorKindMismatch",
    "SpliceCrossedUncovered",
    "EventAbsorbedByFireTail",
    "AwaitGateDeadlineForceRebasing",
    "AwaitGateDeadlineReap",
    "QuiescenceCeilingUnreadable",
    "RebaseCeilingStillChanging",
    "RebaseCeilingUnreadable",
    "SensorOverflow",
    "PromoterReseededForOverflow",
    "PerFileDriftDroppedOnRecovery",
    "PerFileFireSkippedOnFreshSeed",
    "SubAttached",
    "SubFired",
    "SubDetached",
    "SubRebound",
    "RebindUnknownSub",
    "PromoterAttached",
    "PromoterReaped",
    "PromoterDescentVanished",
    "PromoterDescentFailed",
    "PromotionKindObserved",
    "PromoterFanoutThreshold",
    "PromoterProxyStaleEvent",
    "PromoterEnumerationVanished",
    "PromoterEnumerationFailed",
    "DynamicSubReaped",
    "InvalidBurstTransition",
    "_missed",
];

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireBurstIntent {
    Standard,
    Seed,
}

impl From<BurstIntent> for WireBurstIntent {
    fn from(i: BurstIntent) -> Self {
        match i {
            BurstIntent::Standard => Self::Standard,
            BurstIntent::Seed => Self::Seed,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireFsEvent {
    Modified,
    MetadataChanged,
    StructureChanged,
    Renamed,
    Removed,
    Revoked,
}

impl From<FsEvent> for WireFsEvent {
    fn from(e: FsEvent) -> Self {
        match e {
            FsEvent::Modified => Self::Modified,
            FsEvent::MetadataChanged => Self::MetadataChanged,
            FsEvent::StructureChanged => Self::StructureChanged,
            FsEvent::Renamed => Self::Renamed,
            FsEvent::Removed => Self::Removed,
            FsEvent::Revoked => Self::Revoked,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "scope", rename_all = "snake_case")]
pub(crate) enum WireOverflowScope {
    Resource { resource: WireId },
    Global,
}

impl From<OverflowScope> for WireOverflowScope {
    fn from(s: OverflowScope) -> Self {
        match s {
            OverflowScope::Resource(r) => Self::Resource {
                resource: WireId::from(r),
            },
            OverflowScope::Global => Self::Global,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum WireProbeOwner {
    Profile { profile: WireId },
    Promoter { promoter: WireId },
}

impl From<ProbeOwner> for WireProbeOwner {
    fn from(o: ProbeOwner) -> Self {
        match o {
            ProbeOwner::Profile(p) => Self::Profile {
                profile: WireId::from(p),
            },
            ProbeOwner::Promoter(p) => Self::Promoter {
                promoter: WireId::from(p),
            },
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum WireWatchFailure {
    Pressure { errno: i32 },
    Resource { errno: i32 },
    Invariant { errno: i32 },
}

impl From<WatchFailure> for WireWatchFailure {
    fn from(f: WatchFailure) -> Self {
        match f {
            WatchFailure::Pressure { errno } => Self::Pressure { errno },
            WatchFailure::Resource { errno } => Self::Resource { errno },
            WatchFailure::Invariant { errno } => Self::Invariant { errno },
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireReapTrigger {
    Immediate,
    DeferredFromBurst,
}

impl From<ReapTrigger> for WireReapTrigger {
    fn from(t: ReapTrigger) -> Self {
        match t {
            ReapTrigger::Immediate => Self::Immediate,
            ReapTrigger::DeferredFromBurst => Self::DeferredFromBurst,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireResourceKind {
    File,
    Dir,
    Unknown,
}

impl From<ResourceKind> for WireResourceKind {
    fn from(k: ResourceKind) -> Self {
        match k {
            ResourceKind::File => Self::File,
            ResourceKind::Dir => Self::Dir,
            ResourceKind::Unknown => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireClaimKind {
    Anchor,
    WatchRootParent,
    DescentPrefix,
}

impl From<ClaimKind> for WireClaimKind {
    fn from(c: ClaimKind) -> Self {
        match c {
            ClaimKind::Anchor => Self::Anchor,
            ClaimKind::WatchRootParent => Self::WatchRootParent,
            ClaimKind::DescentPrefix => Self::DescentPrefix,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WirePromoterClaimKind {
    DescentPrefix,
    ActiveProxy,
    PrefixParent,
}

impl From<PromoterClaimKind> for WirePromoterClaimKind {
    fn from(c: PromoterClaimKind) -> Self {
        match c {
            PromoterClaimKind::DescentPrefix => Self::DescentPrefix,
            PromoterClaimKind::ActiveProxy => Self::ActiveProxy,
            PromoterClaimKind::PrefixParent => Self::PrefixParent,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireSpliceFailureCause {
    TargetOutsideAnchorSubtree,
    SlotReapedMidGraft,
    IntermediateUncovered,
}

impl From<SpliceFailureCause> for WireSpliceFailureCause {
    fn from(c: SpliceFailureCause) -> Self {
        match c {
            SpliceFailureCause::TargetOutsideAnchorSubtree => Self::TargetOutsideAnchorSubtree,
            SpliceFailureCause::SlotReapedMidGraft => Self::SlotReapedMidGraft,
            SpliceFailureCause::IntermediateUncovered => Self::IntermediateUncovered,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireDetachReason {
    ConfigDiffRemoved,
    ConfigDiffIdentityChanged,
    IpcDisabled,
    PromoterReaped,
}

impl From<DetachReason> for WireDetachReason {
    fn from(r: DetachReason) -> Self {
        match r {
            DetachReason::ConfigDiffRemoved => Self::ConfigDiffRemoved,
            DetachReason::ConfigDiffIdentityChanged => Self::ConfigDiffIdentityChanged,
            DetachReason::IpcDisabled => Self::IpcDisabled,
            DetachReason::PromoterReaped => Self::PromoterReaped,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireBurstHelper {
    StartSeedBurst,
    StartStandardBurst,
    EventDrivesBatching,
    UnstableResponseDrivesBatching,
    TransitionToVerifying,
    TransitionToDraining,
    TransitionToAwaiting,
    TransitionToRebasing,
    RebaseUnstableLoopsSettling,
    AbsorbEventIntoFireTail,
    RestartBurstFromFireTailResidual,
}

impl From<BurstHelper> for WireBurstHelper {
    fn from(h: BurstHelper) -> Self {
        match h {
            BurstHelper::StartSeedBurst => Self::StartSeedBurst,
            BurstHelper::StartStandardBurst => Self::StartStandardBurst,
            BurstHelper::EventDrivesBatching => Self::EventDrivesBatching,
            BurstHelper::UnstableResponseDrivesBatching => Self::UnstableResponseDrivesBatching,
            BurstHelper::TransitionToVerifying => Self::TransitionToVerifying,
            BurstHelper::TransitionToDraining => Self::TransitionToDraining,
            BurstHelper::TransitionToAwaiting => Self::TransitionToAwaiting,
            BurstHelper::TransitionToRebasing => Self::TransitionToRebasing,
            BurstHelper::RebaseUnstableLoopsSettling => Self::RebaseUnstableLoopsSettling,
            BurstHelper::AbsorbEventIntoFireTail => Self::AbsorbEventIntoFireTail,
            BurstHelper::RestartBurstFromFireTailResidual => Self::RestartBurstFromFireTailResidual,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireProfileStateDiscriminant {
    Idle,
    Pending,
    ActivePreFire,
    ActivePostFire,
}

impl From<ProfileStateDiscriminant> for WireProfileStateDiscriminant {
    fn from(d: ProfileStateDiscriminant) -> Self {
        match d {
            ProfileStateDiscriminant::Idle => Self::Idle,
            ProfileStateDiscriminant::Pending => Self::Pending,
            ProfileStateDiscriminant::ActivePreFire => Self::ActivePreFire,
            ProfileStateDiscriminant::ActivePostFire => Self::ActivePostFire,
        }
    }
}

/// Operator-display phase. Mirrors `specter_core::StateLabel`'s
/// eight phases verbatim; landing here keeps the wire projection
/// layer cohesive even though `StateLabel` is not currently
/// referenced by [`Diagnostic`].
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireStateLabel {
    Idle,
    Pending,
    Batching,
    Verifying,
    Draining,
    Awaiting,
    Rebasing,
    RebaseSettling,
}

impl From<StateLabel> for WireStateLabel {
    fn from(s: StateLabel) -> Self {
        match s {
            StateLabel::Idle => Self::Idle,
            StateLabel::Pending => Self::Pending,
            StateLabel::Batching => Self::Batching,
            StateLabel::Verifying => Self::Verifying,
            StateLabel::Draining => Self::Draining,
            StateLabel::Awaiting => Self::Awaiting,
            StateLabel::Rebasing => Self::Rebasing,
            StateLabel::RebaseSettling => Self::RebaseSettling,
        }
    }
}

/// Sub effect-scope projection. Mirrors `specter_core::EffectScope`
/// verbatim; surfaces in `SubDetails.scope`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireEffectScope {
    SubtreeRoot,
    PerStableFile,
}

impl From<EffectScope> for WireEffectScope {
    fn from(s: EffectScope) -> Self {
        match s {
            EffectScope::SubtreeRoot => Self::SubtreeRoot,
            EffectScope::PerStableFile => Self::PerStableFile,
        }
    }
}

/// Reload-trigger projection of `crate::driver::ReloadTrigger`.
/// Lives in this module to keep every wire enum (core- or bin-sourced)
/// in one place; surfaces in `StatusResponse.last_reload_via`.
///
/// `AutoReload` projects to operator-facing `auto` — the engine-internal
/// `AutoReload` name carries the "settle-expiry observed drift"
/// mechanism that doesn't belong in the operator vocabulary.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireReloadTrigger {
    Sighup,
    Auto,
    Ipc,
}

impl From<crate::driver::ReloadTrigger> for WireReloadTrigger {
    fn from(t: crate::driver::ReloadTrigger) -> Self {
        use crate::driver::ReloadTrigger as R;
        match t {
            R::Sighup => Self::Sighup,
            R::AutoReload => Self::Auto,
            R::Ipc => Self::Ipc,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{
        KNOWN_WIRE_VARIANTS, WireBurstHelper, WireBurstIntent, WireClaimKind, WireDetachReason,
        WireDiagnostic, WireFsEvent, WireOverflowScope, WireProbeOwner,
        WireProfileStateDiscriminant, WirePromoterClaimKind, WireReapTrigger, WireResourceKind,
        WireSpliceFailureCause, WireTime, WireWatchFailure,
    };
    use crate::ipc::protocol::WireId;
    use std::collections::BTreeSet;
    use std::time::{Duration, UNIX_EPOCH};

    /// Pre-epoch `SystemTime` clamps to `UNIX_EPOCH` and serializes
    /// as the epoch literal — does not panic. Defends against NTP
    /// step-backwards, operator `date` reset, container clock skew.
    #[test]
    fn wire_time_clamps_pre_epoch_to_unix_epoch() {
        let pre = UNIX_EPOCH - Duration::from_secs(1);
        let wire = WireTime::from(pre);
        let json = serde_json::to_string(&wire).unwrap();
        assert_eq!(json, r#""1970-01-01T00:00:00Z""#);

        let epoch = WireTime::from(UNIX_EPOCH);
        assert_eq!(
            serde_json::to_string(&epoch).unwrap(),
            r#""1970-01-01T00:00:00Z""#
        );
    }

    /// Witness fixture covering every [`WireDiagnostic`] variant.
    ///
    /// Two cross-cutting pins read this list:
    ///
    /// 1. [`wire_diagnostic_round_trips_via_serde`] — serialize each
    ///    witness, deserialize, re-serialize, assert byte equality.
    ///    Catches a serde derive macro regression on either half of
    ///    the round-trip.
    /// 2. [`known_wire_variants_matches_variant_name`] — every tag
    ///    here matches its variant's
    ///    [`WireDiagnostic::variant_name`] and appears in
    ///    [`KNOWN_WIRE_VARIANTS`] (and vice versa).
    ///
    /// A new [`WireDiagnostic`] variant needs (a) its
    /// `From<(&Diagnostic, SystemTime)>` arm (compile-time, exhaustive),
    /// (b) its `variant_name` arm (compile-time, exhaustive), (c) a tag
    /// in [`KNOWN_WIRE_VARIANTS`], and (d) a witness here. The drift
    /// test fails when (c) or (d) lag.
    fn variant_witnesses() -> Vec<WireDiagnostic> {
        let at = || WireTime::from(UNIX_EPOCH);
        vec![
            WireDiagnostic::StaleProbeResponse {
                at: at(),
                owner: WireProbeOwner::Profile { profile: WireId(1) },
                correlation: 7,
            },
            WireDiagnostic::StaleTimer { at: at(), id: 9 },
            WireDiagnostic::EffectCompleteOutsideAwaiting {
                at: at(),
                sub: WireId(11),
                profile: WireId(22),
            },
            WireDiagnostic::EffectCompleteForUnknownSub {
                at: at(),
                sub: WireId(13),
            },
            WireDiagnostic::DetachUnknownSub {
                at: at(),
                sub: WireId(17),
            },
            WireDiagnostic::ConfigDiffUnknownSub {
                at: at(),
                name: "foo".into(),
            },
            WireDiagnostic::ConfigDiffUnknownPromoter {
                at: at(),
                name: "bar".into(),
            },
            WireDiagnostic::ConfigDiffRebindFallbackAttach {
                at: at(),
                name: "baz".into(),
            },
            WireDiagnostic::ProbeVanished {
                at: at(),
                profile: WireId(31),
                intent: WireBurstIntent::Standard,
            },
            WireDiagnostic::ProbeFailed {
                at: at(),
                profile: WireId(32),
                intent: WireBurstIntent::Seed,
                errno: 5,
            },
            WireDiagnostic::EventClassDropped {
                at: at(),
                resource: WireId(40),
                event: WireFsEvent::Modified,
                profile: WireId(41),
            },
            WireDiagnostic::EventOnUnwatchedResource {
                at: at(),
                resource: WireId(42),
            },
            WireDiagnostic::EventNoConsumer {
                at: at(),
                resource: WireId(43),
            },
            WireDiagnostic::WatchOpRejected {
                at: at(),
                resource: WireId(44),
                failure: WireWatchFailure::Pressure { errno: 24 },
            },
            WireDiagnostic::PendingPathProbeVanished {
                at: at(),
                profile: WireId(50),
                prefix: WireId(51),
            },
            WireDiagnostic::PendingPathProbeFailed {
                at: at(),
                profile: WireId(52),
                prefix: WireId(53),
                errno: 13,
            },
            WireDiagnostic::ReapPendingCancelled {
                at: at(),
                profile: WireId(60),
            },
            WireDiagnostic::ProfileReaped {
                at: at(),
                profile: WireId(61),
                via: WireReapTrigger::Immediate,
            },
            WireDiagnostic::ProfileClaimPurged {
                at: at(),
                profile: WireId(70),
                claim: WireClaimKind::Anchor,
                resource: WireId(71),
                failure: WireWatchFailure::Resource { errno: 28 },
            },
            WireDiagnostic::PromoterClaimPurged {
                at: at(),
                promoter: WireId(80),
                claim: WirePromoterClaimKind::ActiveProxy,
                resource: WireId(81),
                failure: WireWatchFailure::Invariant { errno: 22 },
            },
            WireDiagnostic::AttachPathInvalid {
                at: at(),
                path: "/tmp/x".into(),
                hint: "relative".into(),
            },
            WireDiagnostic::AttachResourceStale {
                at: at(),
                resource: WireId(90),
            },
            WireDiagnostic::AnchorKindMismatch {
                at: at(),
                profile: WireId(91),
                prior_kind: WireResourceKind::Dir,
                response_kind: WireResourceKind::File,
            },
            WireDiagnostic::SpliceCrossedUncovered {
                at: at(),
                profile: WireId(92),
                target: WireId(93),
                cause: WireSpliceFailureCause::TargetOutsideAnchorSubtree,
            },
            WireDiagnostic::EventAbsorbedByFireTail {
                at: at(),
                profile: WireId(100),
                resource: WireId(101),
                event: WireFsEvent::StructureChanged,
            },
            WireDiagnostic::AwaitGateDeadlineForceRebasing {
                at: at(),
                profile: WireId(110),
                outstanding: 1,
            },
            WireDiagnostic::AwaitGateDeadlineReap {
                at: at(),
                profile: WireId(111),
                outstanding: 2,
            },
            WireDiagnostic::QuiescenceCeilingUnreadable {
                at: at(),
                profile: WireId(120),
                first_unread: "/tmp/x/y".into(),
                intent: WireBurstIntent::Standard,
            },
            WireDiagnostic::RebaseCeilingStillChanging {
                at: at(),
                profile: WireId(121),
                intent: WireBurstIntent::Standard,
            },
            WireDiagnostic::RebaseCeilingUnreadable {
                at: at(),
                profile: WireId(122),
                first_unread: "/tmp/z".into(),
                intent: WireBurstIntent::Seed,
            },
            WireDiagnostic::SensorOverflow {
                at: at(),
                scope: WireOverflowScope::Global,
            },
            WireDiagnostic::PromoterReseededForOverflow {
                at: at(),
                promoter: WireId(130),
            },
            WireDiagnostic::PerFileDriftDroppedOnRecovery {
                at: at(),
                profile: WireId(140),
            },
            WireDiagnostic::PerFileFireSkippedOnFreshSeed {
                at: at(),
                profile: WireId(141),
            },
            WireDiagnostic::SubAttached {
                at: at(),
                sub: WireId(150),
                name: "watch".into(),
                source_promoter: None,
            },
            WireDiagnostic::SubFired {
                at: at(),
                sub: WireId(151),
                profile: WireId(152),
                count: 3,
            },
            WireDiagnostic::SubDetached {
                at: at(),
                sub: WireId(153),
                profile: WireId(154),
                reason: WireDetachReason::IpcDisabled,
            },
            WireDiagnostic::SubRebound {
                at: at(),
                sub: WireId(155),
            },
            WireDiagnostic::RebindUnknownSub {
                at: at(),
                sub: WireId(156),
            },
            WireDiagnostic::PromoterAttached {
                at: at(),
                promoter: WireId(160),
                name: "p".into(),
            },
            WireDiagnostic::PromoterReaped {
                at: at(),
                promoter: WireId(161),
            },
            WireDiagnostic::PromoterDescentVanished {
                at: at(),
                promoter: WireId(162),
                prefix: WireId(163),
            },
            WireDiagnostic::PromoterDescentFailed {
                at: at(),
                promoter: WireId(164),
                prefix: WireId(165),
                errno: 2,
            },
            WireDiagnostic::PromotionKindObserved {
                at: at(),
                promoter: WireId(166),
                path: "/tmp/p/x".into(),
                kind: WireResourceKind::Dir,
            },
            WireDiagnostic::PromoterFanoutThreshold {
                at: at(),
                promoter: WireId(167),
                count: 256,
            },
            WireDiagnostic::PromoterProxyStaleEvent {
                at: at(),
                promoter: WireId(168),
                resource: WireId(169),
            },
            WireDiagnostic::PromoterEnumerationVanished {
                at: at(),
                promoter: WireId(170),
                proxy: WireId(171),
            },
            WireDiagnostic::PromoterEnumerationFailed {
                at: at(),
                promoter: WireId(172),
                proxy: WireId(173),
                errno: 13,
            },
            WireDiagnostic::DynamicSubReaped {
                at: at(),
                promoter: WireId(174),
                sub: WireId(175),
                path: "/tmp/p/dyn".into(),
            },
            WireDiagnostic::InvalidBurstTransition {
                at: at(),
                profile: WireId(180),
                helper: WireBurstHelper::TransitionToVerifying,
                observed: WireProfileStateDiscriminant::Idle,
            },
            WireDiagnostic::Missed { at: at(), count: 5 },
        ]
    }

    /// Every [`WireDiagnostic`] variant round-trips through serde
    /// identity: serialize → JSON bytes → deserialize → re-serialize
    /// → same bytes.
    ///
    /// Identity check is via re-serialization because
    /// [`WireDiagnostic`] is intentionally not `PartialEq` — adding
    /// the derive would propagate the bound to every transitively-
    /// reached enum and the canonical wire bytes already are the
    /// identity. A serde derive macro regression on either half of
    /// the round-trip fails this test.
    #[test]
    fn wire_diagnostic_round_trips_via_serde() {
        for w in variant_witnesses() {
            let bytes = serde_json::to_string(&w).expect("serialize");
            let back: WireDiagnostic =
                serde_json::from_str(&bytes).expect("deserialize the same bytes");
            let again = serde_json::to_string(&back).expect("re-serialize");
            assert_eq!(
                again,
                bytes,
                "round-trip identity broke for variant tagged {}",
                w.variant_name(),
            );
        }
    }

    /// [`KNOWN_WIRE_VARIANTS`] aligns with [`WireDiagnostic::variant_name`]
    /// and the [`variant_witnesses`] fixture: same set, same size,
    /// no duplicates. A new variant that lands without a tag entry
    /// (or a tag without a witness) fails here loudly.
    #[test]
    fn known_wire_variants_matches_variant_name() {
        let witnesses = variant_witnesses();

        // (a) Every witness's variant_name appears in KNOWN_WIRE_VARIANTS.
        for w in &witnesses {
            let tag = w.variant_name();
            assert!(
                KNOWN_WIRE_VARIANTS.contains(&tag),
                "{tag} reported by variant_name but absent from KNOWN_WIRE_VARIANTS",
            );
        }

        // (b) Counts agree — a stale entry in either side fails here.
        assert_eq!(
            witnesses.len(),
            KNOWN_WIRE_VARIANTS.len(),
            "witness count ({}) ≠ KNOWN_WIRE_VARIANTS count ({})",
            witnesses.len(),
            KNOWN_WIRE_VARIANTS.len(),
        );

        // (c) Set equality across both surfaces; catches duplicates
        //     and reorderings that (b) would silently accept.
        let from_witness: BTreeSet<&str> =
            witnesses.iter().map(WireDiagnostic::variant_name).collect();
        let from_const: BTreeSet<&str> = KNOWN_WIRE_VARIANTS.iter().copied().collect();
        assert_eq!(
            from_witness, from_const,
            "set mismatch — variants in one source missing from the other",
        );
        assert_eq!(
            from_const.len(),
            KNOWN_WIRE_VARIANTS.len(),
            "duplicate entry in KNOWN_WIRE_VARIANTS",
        );
    }
}
