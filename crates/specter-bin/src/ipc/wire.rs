//! Wire-side projection of `specter_core::Diagnostic` and every
//! enum it transitively reaches.
//!
//! # The structural wall
//!
//! [`WireDiagnostic`]'s [`From<(&Diagnostic, &WireTime)>`] is an
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
//! per-conn fan-out (the [`From<(&Diagnostic, &WireTime)>`] projection
//! at write time, called once per dispatch from
//! [`crate::driver::Hub::dispatch_to_subscribers`]), and
//! operator clients (`specter tail`, `specter wait`) deserialize the
//! streamed JSON lines back into the typed enum. Every wire enum it
//! transitively reaches carries both `Serialize` and `Deserialize`;
//! round-trip is structural over the `#[serde]` tags.
//!
//! Adding a [`WireDiagnostic`] variant is a paired edit: declare it,
//! write its [`From<(&Diagnostic, &WireTime)>`] arm, add the matching
//! arm in [`WireDiagnostic::variant_name`], and add a tag entry in
//! [`KNOWN_WIRE_VARIANTS`]. The first three edits are exhaustive
//! `match` arms so the compiler refuses the change without them; the
//! fourth is pinned by a drift test that fails on either side
//! diverging from the witness set.
//!
//! `WireTime` owns its own formatting via
//! `humantime::format_rfc3339_seconds` on the outgoing path and
//! validates via `humantime::parse_rfc3339` on the incoming path:
//! every wire value is RFC 3339 by construction in *both*
//! directions. Pre-epoch `SystemTime` is clamped to `UNIX_EPOCH` on
//! the server-side projection to defuse `humantime`'s pre-epoch
//! panic.
//!
//! The inner storage is `Arc<str>`. The fan-out path builds one
//! `WireTime` per `StepOutput` and the `From<(&Diagnostic, &WireTime)>`
//! projection bumps the refcount per diag — `humantime` formats once
//! per emission regardless of diag count.
//!
//! # Field ordering
//!
//! Every [`WireDiagnostic`] variant declares `at: WireTime` as its
//! first field so it serializes immediately after the `diag` tag.
//! `jq` filters and operator inspection both benefit from a
//! predictable timestamp position.
//!
//! # Small-string idiom
//!
//! `CompactString` is the wire-uniform small-string idiom across the
//! IPC layer. Request-side fields ([`crate::ipc::protocol`]'s
//! `WireRequest` Subscribe / Show name) and diag-side fields (the
//! five name-bearing variants on [`WireDiagnostic`]) both use it, so
//! a hypothetical shared carrier wouldn't have to pick — and the
//! source-side core enum stores `CompactString` already, so the
//! projection is a refcount-free SSO-fit clone rather than a fresh
//! `String` allocation per emission.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. Operator clients ship inside the
//! same binary, so the wire surface stays a bin-internal contract.

use compact_str::CompactString;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use specter_core::{
    AbsorbMode, BurstHelper, BurstIntent, ClaimKind, DetachReason, Diagnostic, EffectScope,
    FsEvent, OverflowScope, ProbeOwner, ProfileStateDiscriminant, PromoterClaimKind, ReapTrigger,
    ResourceKind, SpliceFailureCause, StateLabel, WatchFailure,
};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use super::framing::InfallibleSerialize;
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
/// # Symmetric validation
///
/// Both [`Serialize`] and [`Deserialize`] are manual and both gate
/// the same RFC 3339 vocabulary. Serialize writes the inner `&str`
/// verbatim (it is invariant-by-construction UTF-8 RFC 3339 thanks to
/// [`Self::from(SystemTime)`]). Deserialize takes any JSON string,
/// validates it with [`humantime::parse_rfc3339`], and stores the
/// validated bytes. A non-RFC-3339 token fails the boundary — the
/// wire layer cannot accept opaque text masquerading as a timestamp.
///
/// JSON form is a bare quoted string (`"2026-05-23T15:30:00Z"`), not
/// a wrapped object.
///
/// # Shared allocation
///
/// The inner storage is `Arc<str>`. The fan-out path
/// ([`crate::driver::EngineDriver::forward_diagnostics`]) builds a
/// single `WireTime` per `StepOutput`; every per-diag
/// [`From<(&Diagnostic, &WireTime)>`] projection bumps the refcount
/// instead of re-formatting. The `Display` consumer (status / list /
/// show / tail human renderers) writes the inner `&str` verbatim, so
/// the only consumer outside the JSON wire path is also zero-alloc.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct WireTime(Arc<str>);

impl Serialize for WireTime {
    /// Wire form is a bare JSON string — invariant-by-construction
    /// RFC 3339 second-resolution UTF-8 thanks to [`Self::from`] and
    /// the matching [`Deserialize`] gate.
    fn serialize<S: Serializer>(&self, ser: S) -> Result<S::Ok, S::Error> {
        ser.serialize_str(&self.0)
    }
}

impl<'de> Deserialize<'de> for WireTime {
    /// Parse-on-deserialize: every incoming byte sequence is checked
    /// against [`humantime::parse_rfc3339`] before it becomes a
    /// `WireTime`. The shape mirrors the server-side `From` projection
    /// so the wire vocabulary is invariant in both directions; a
    /// future malformed daemon emit or a hostile client cannot smuggle
    /// arbitrary text past this gate.
    fn deserialize<D: Deserializer<'de>>(de: D) -> Result<Self, D::Error> {
        let raw = String::deserialize(de)?;
        humantime::parse_rfc3339(&raw)
            .map_err(|e| serde::de::Error::custom(format!("invalid RFC 3339 timestamp: {e}")))?;
        Ok(Self(Arc::from(raw)))
    }
}

impl From<SystemTime> for WireTime {
    fn from(t: SystemTime) -> Self {
        let clamped = t.max(UNIX_EPOCH);
        if clamped != t {
            tracing::warn!(
                ?t,
                "specter ipc: pre-epoch SystemTime clamped to UNIX_EPOCH",
            );
        }
        Self(Arc::from(
            humantime::format_rfc3339_seconds(clamped).to_string(),
        ))
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

/// Wire-shape projection of a filesystem path.
///
/// The schema is JSON-safe: non-UTF-8 bytes ride as
/// `U+FFFD REPLACEMENT CHARACTER` via [`Path::to_string_lossy`],
/// matching `tracing`'s own lossy path projection. The lossy
/// projection runs once at construct time (the [`From`] impls
/// below); the wire serialization is then a structural copy of an
/// already-validated UTF-8 string, so the daemon's `serde_json`
/// path cannot panic on a non-UTF-8 [`PathBuf`] / [`Arc<Path>`] —
/// the structural floor closes the daemon-panic surface the audit's
/// F-CRIT-1 finding identified.
///
/// `#[serde(transparent)]` makes the JSON form a bare quoted string
/// (`"/etc/specter.toml"`), not a wrapped object. The shape is
/// symmetric across Serialize and Deserialize: server emits a
/// lossy-projected UTF-8 token, client parses the same bytes back
/// into the inner [`String`].
///
/// `Deserialize` accepts any UTF-8 string (the server-side projection
/// is the gating shape; the client treats the inner bytes as opaque
/// path-display text). The renderer reproduces the value verbatim
/// through [`Display`].
///
/// Construction is one-way: every [`From`] impl projects *into*
/// `WirePath`. There is no `From<String>` — a `WirePath` is built
/// from a path-typed source, not from an unconstrained string. The
/// discipline mirrors [`WireId`]'s "no `From<u64>` from clients" rule.
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(transparent)]
pub(crate) struct WirePath(String);

impl From<&Path> for WirePath {
    fn from(p: &Path) -> Self {
        Self(p.to_string_lossy().into_owned())
    }
}

impl From<&PathBuf> for WirePath {
    /// Convenience for the common `WirePath::from(&driver_state.socket_path)`
    /// shape. Delegates to the `&Path` projection.
    fn from(p: &PathBuf) -> Self {
        Self::from(p.as_path())
    }
}

impl From<&Arc<Path>> for WirePath {
    /// Diagnostic-side projection — the engine emits `path: Arc<Path>`
    /// fields and the [`WireDiagnostic::from`] arms reach this impl
    /// to project them. Delegates to the `&Path` projection.
    fn from(p: &Arc<Path>) -> Self {
        Self::from(p.as_ref())
    }
}

impl std::fmt::Display for WirePath {
    /// Renderers reproduce the projected path verbatim through
    /// `Display`, so the token is a zero-alloc `&str` write — used
    /// by `status -o human` / `list -o human` / `show -o human` /
    /// `tail -o human` and the embedded `path={path}` fields on
    /// diagnostic lines.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// JSON-line projection of `specter_core::Diagnostic` plus the
/// fan-out path's `_missed` back-pressure marker.
///
/// Internally tagged on `diag`; every variant's `at` field
/// serializes immediately after the tag.
///
/// Tag vocabulary is snake_case (`#[serde(rename_all = "snake_case")]`)
/// so a single operator-visible vocabulary covers `tail --filter`,
/// the streamed JSON's `diag` field, and every other `Wire*` enum
/// in this module (they all carry the same serde rename). The
/// only exception is [`Self::Missed`], which keeps an explicit
/// `#[serde(rename = "_missed")]` override — the underscore prefix
/// is the collision-protection token reserved for the fan-out
/// back-pressure marker, never a Rust identifier and so unreachable
/// from a future `specter_core::Diagnostic` variant.
///
/// Two-way derive (server serializes for fan-out, client
/// deserializes for tail/wait) — see the module rustdoc's
/// `Deserialize policy` section for the structural invariants the
/// paired edit must preserve.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "diag", rename_all = "snake_case")]
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
        name: CompactString,
    },
    ConfigDiffUnknownPromoter {
        at: WireTime,
        name: CompactString,
    },
    ConfigDiffRebindFallbackAttach {
        at: WireTime,
        name: CompactString,
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
        path: WirePath,
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
        first_unread: WirePath,
        intent: WireBurstIntent,
    },
    RebaseCeilingStillChanging {
        at: WireTime,
        profile: WireId,
        intent: WireBurstIntent,
    },
    QuiescenceCeilingForcedDespiteChange {
        at: WireTime,
        profile: WireId,
        intent: WireBurstIntent,
    },
    RebaseCeilingForcedDespiteChange {
        at: WireTime,
        profile: WireId,
        intent: WireBurstIntent,
    },
    RebaseCeilingUnreadable {
        at: WireTime,
        profile: WireId,
        first_unread: WirePath,
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
        name: CompactString,
        source_promoter: Option<WireId>,
    },
    SubFired {
        at: WireTime,
        sub: WireId,
        profile: WireId,
        count: u32,
    },
    QuiescenceAbsorbed {
        at: WireTime,
        profile: WireId,
    },
    AbsorbArmed {
        at: WireTime,
        profile: WireId,
        mode: WireAbsorbMode,
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
        name: CompactString,
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
        path: WirePath,
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
        path: WirePath,
    },
    InvalidBurstTransition {
        at: WireTime,
        profile: WireId,
        helper: WireBurstHelper,
        observed: WireProfileStateDiscriminant,
    },
    /// Fan-out back-pressure marker — not derived from any
    /// `specter_core::Diagnostic`. Emitted by
    /// [`crate::driver::Hub::dispatch_to_subscribers`]
    /// when a wedged subscriber's queue overflowed and the dispatch
    /// loop had to drop diag lines; the marker tells the operator
    /// how many were skipped before the next reachable line. The
    /// underscore-prefix protects against collision with any future
    /// core variant named `Missed`; `#[serde(rename = "_missed")]`
    /// overrides the enum's `rename_all = "snake_case"` default (which
    /// would otherwise emit the bare `missed` token a core variant
    /// could legitimately occupy).
    #[serde(rename = "_missed")]
    Missed {
        at: WireTime,
        count: u32,
    },
}

impl From<(&Diagnostic, &WireTime)> for WireDiagnostic {
    /// The wall-clock projection is the caller's concern — the fan-out
    /// path builds one [`WireTime`] per `StepOutput`
    /// ([`crate::driver::EngineDriver::forward_diagnostics`]) and threads
    /// `&WireTime` through every per-diag construction. Each arm bumps
    /// the `Arc<str>` refcount via [`Clone`] instead of re-formatting,
    /// so `humantime` runs once per emission regardless of diag count.
    fn from((d, at): (&Diagnostic, &WireTime)) -> Self {
        match d {
            Diagnostic::StaleProbeResponse { owner, correlation } => Self::StaleProbeResponse {
                at: at.clone(),
                owner: WireProbeOwner::from(*owner),
                correlation: correlation.as_u64(),
            },
            Diagnostic::StaleTimer { id } => Self::StaleTimer {
                at: at.clone(),
                id: id.as_u64(),
            },
            Diagnostic::EffectCompleteOutsideAwaiting { sub, profile } => {
                Self::EffectCompleteOutsideAwaiting {
                    at: at.clone(),
                    sub: WireId::from(*sub),
                    profile: WireId::from(*profile),
                }
            }
            Diagnostic::EffectCompleteForUnknownSub { sub } => Self::EffectCompleteForUnknownSub {
                at: at.clone(),
                sub: WireId::from(*sub),
            },
            Diagnostic::DetachUnknownSub { sub } => Self::DetachUnknownSub {
                at: at.clone(),
                sub: WireId::from(*sub),
            },
            Diagnostic::ConfigDiffUnknownSub { name } => Self::ConfigDiffUnknownSub {
                at: at.clone(),
                name: name.clone(),
            },
            Diagnostic::ConfigDiffUnknownPromoter { name } => Self::ConfigDiffUnknownPromoter {
                at: at.clone(),
                name: name.clone(),
            },
            Diagnostic::ConfigDiffRebindFallbackAttach { name } => {
                Self::ConfigDiffRebindFallbackAttach {
                    at: at.clone(),
                    name: name.clone(),
                }
            }
            Diagnostic::ProbeVanished { profile, intent } => Self::ProbeVanished {
                at: at.clone(),
                profile: WireId::from(*profile),
                intent: WireBurstIntent::from(*intent),
            },
            Diagnostic::ProbeFailed {
                profile,
                intent,
                failure,
            } => Self::ProbeFailed {
                at: at.clone(),
                profile: WireId::from(*profile),
                intent: WireBurstIntent::from(*intent),
                // Wire carries the operator-visible integer; the
                // typed routing-target variant is engine-internal.
                errno: failure.errno(),
            },
            Diagnostic::EventClassDropped {
                resource,
                event,
                profile,
            } => Self::EventClassDropped {
                at: at.clone(),
                resource: WireId::from(*resource),
                event: WireFsEvent::from(*event),
                profile: WireId::from(*profile),
            },
            Diagnostic::EventOnUnwatchedResource { resource } => Self::EventOnUnwatchedResource {
                at: at.clone(),
                resource: WireId::from(*resource),
            },
            Diagnostic::EventNoConsumer { resource } => Self::EventNoConsumer {
                at: at.clone(),
                resource: WireId::from(*resource),
            },
            Diagnostic::WatchOpRejected { resource, failure } => Self::WatchOpRejected {
                at: at.clone(),
                resource: WireId::from(*resource),
                failure: WireWatchFailure::from(*failure),
            },
            Diagnostic::PendingPathProbeVanished { profile, prefix } => {
                Self::PendingPathProbeVanished {
                    at: at.clone(),
                    profile: WireId::from(*profile),
                    prefix: WireId::from(*prefix),
                }
            }
            Diagnostic::PendingPathProbeFailed {
                profile,
                prefix,
                failure,
            } => Self::PendingPathProbeFailed {
                at: at.clone(),
                profile: WireId::from(*profile),
                prefix: WireId::from(*prefix),
                errno: failure.errno(),
            },
            Diagnostic::ReapPendingCancelled { profile } => Self::ReapPendingCancelled {
                at: at.clone(),
                profile: WireId::from(*profile),
            },
            Diagnostic::ProfileReaped { profile, via } => Self::ProfileReaped {
                at: at.clone(),
                profile: WireId::from(*profile),
                via: WireReapTrigger::from(*via),
            },
            Diagnostic::ProfileClaimPurged {
                profile,
                claim,
                resource,
                failure,
            } => Self::ProfileClaimPurged {
                at: at.clone(),
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
                at: at.clone(),
                promoter: WireId::from(*promoter),
                claim: WirePromoterClaimKind::from(*claim),
                resource: WireId::from(*resource),
                failure: WireWatchFailure::from(*failure),
            },
            Diagnostic::AttachPathInvalid { path, hint } => Self::AttachPathInvalid {
                at: at.clone(),
                path: WirePath::from(path),
                hint: (*hint).to_owned(),
            },
            Diagnostic::AttachResourceStale { resource } => Self::AttachResourceStale {
                at: at.clone(),
                resource: WireId::from(*resource),
            },
            Diagnostic::AnchorKindMismatch {
                profile,
                prior_kind,
                response_kind,
            } => Self::AnchorKindMismatch {
                at: at.clone(),
                profile: WireId::from(*profile),
                prior_kind: WireResourceKind::from(*prior_kind),
                response_kind: WireResourceKind::from(*response_kind),
            },
            Diagnostic::SpliceCrossedUncovered {
                profile,
                target,
                cause,
            } => Self::SpliceCrossedUncovered {
                at: at.clone(),
                profile: WireId::from(*profile),
                target: WireId::from(*target),
                cause: WireSpliceFailureCause::from(*cause),
            },
            Diagnostic::EventAbsorbedByFireTail {
                profile,
                resource,
                event,
            } => Self::EventAbsorbedByFireTail {
                at: at.clone(),
                profile: WireId::from(*profile),
                resource: WireId::from(*resource),
                event: WireFsEvent::from(*event),
            },
            Diagnostic::AwaitGateDeadlineForceRebasing {
                profile,
                outstanding,
            } => Self::AwaitGateDeadlineForceRebasing {
                at: at.clone(),
                profile: WireId::from(*profile),
                outstanding: *outstanding,
            },
            Diagnostic::AwaitGateDeadlineReap {
                profile,
                outstanding,
            } => Self::AwaitGateDeadlineReap {
                at: at.clone(),
                profile: WireId::from(*profile),
                outstanding: *outstanding,
            },
            Diagnostic::QuiescenceCeilingUnreadable {
                profile,
                first_unread,
                intent,
            } => Self::QuiescenceCeilingUnreadable {
                at: at.clone(),
                profile: WireId::from(*profile),
                first_unread: WirePath::from(first_unread),
                intent: WireBurstIntent::from(*intent),
            },
            Diagnostic::RebaseCeilingStillChanging { profile, intent } => {
                Self::RebaseCeilingStillChanging {
                    at: at.clone(),
                    profile: WireId::from(*profile),
                    intent: WireBurstIntent::from(*intent),
                }
            }
            Diagnostic::QuiescenceCeilingForcedDespiteChange { profile, intent } => {
                Self::QuiescenceCeilingForcedDespiteChange {
                    at: at.clone(),
                    profile: WireId::from(*profile),
                    intent: WireBurstIntent::from(*intent),
                }
            }
            Diagnostic::RebaseCeilingForcedDespiteChange { profile, intent } => {
                Self::RebaseCeilingForcedDespiteChange {
                    at: at.clone(),
                    profile: WireId::from(*profile),
                    intent: WireBurstIntent::from(*intent),
                }
            }
            Diagnostic::RebaseCeilingUnreadable {
                profile,
                first_unread,
                intent,
            } => Self::RebaseCeilingUnreadable {
                at: at.clone(),
                profile: WireId::from(*profile),
                first_unread: WirePath::from(first_unread),
                intent: WireBurstIntent::from(*intent),
            },
            Diagnostic::SensorOverflow { scope } => Self::SensorOverflow {
                at: at.clone(),
                scope: WireOverflowScope::from(*scope),
            },
            Diagnostic::PromoterReseededForOverflow { promoter } => {
                Self::PromoterReseededForOverflow {
                    at: at.clone(),
                    promoter: WireId::from(*promoter),
                }
            }
            Diagnostic::PerFileDriftDroppedOnRecovery { profile } => {
                Self::PerFileDriftDroppedOnRecovery {
                    at: at.clone(),
                    profile: WireId::from(*profile),
                }
            }
            Diagnostic::PerFileFireSkippedOnFreshSeed { profile } => {
                Self::PerFileFireSkippedOnFreshSeed {
                    at: at.clone(),
                    profile: WireId::from(*profile),
                }
            }
            Diagnostic::SubAttached {
                sub,
                name,
                source_promoter,
            } => Self::SubAttached {
                at: at.clone(),
                sub: WireId::from(*sub),
                name: name.clone(),
                source_promoter: source_promoter.map(WireId::from),
            },
            Diagnostic::SubFired {
                sub,
                profile,
                count,
            } => Self::SubFired {
                at: at.clone(),
                sub: WireId::from(*sub),
                profile: WireId::from(*profile),
                count: *count,
            },
            Diagnostic::QuiescenceAbsorbed { profile } => Self::QuiescenceAbsorbed {
                at: at.clone(),
                profile: WireId::from(*profile),
            },
            Diagnostic::AbsorbArmed { profile, mode } => Self::AbsorbArmed {
                at: at.clone(),
                profile: WireId::from(*profile),
                mode: WireAbsorbMode::from(*mode),
            },
            Diagnostic::SubDetached {
                sub,
                profile,
                reason,
            } => Self::SubDetached {
                at: at.clone(),
                sub: WireId::from(*sub),
                profile: WireId::from(*profile),
                reason: WireDetachReason::from(*reason),
            },
            Diagnostic::SubRebound { sub } => Self::SubRebound {
                at: at.clone(),
                sub: WireId::from(*sub),
            },
            Diagnostic::RebindUnknownSub { sub } => Self::RebindUnknownSub {
                at: at.clone(),
                sub: WireId::from(*sub),
            },
            Diagnostic::PromoterAttached { promoter, name } => Self::PromoterAttached {
                at: at.clone(),
                promoter: WireId::from(*promoter),
                name: name.clone(),
            },
            Diagnostic::PromoterReaped { promoter } => Self::PromoterReaped {
                at: at.clone(),
                promoter: WireId::from(*promoter),
            },
            Diagnostic::PromoterDescentVanished { promoter, prefix } => {
                Self::PromoterDescentVanished {
                    at: at.clone(),
                    promoter: WireId::from(*promoter),
                    prefix: WireId::from(*prefix),
                }
            }
            Diagnostic::PromoterDescentFailed {
                promoter,
                prefix,
                failure,
            } => Self::PromoterDescentFailed {
                at: at.clone(),
                promoter: WireId::from(*promoter),
                prefix: WireId::from(*prefix),
                errno: failure.errno(),
            },
            Diagnostic::PromotionKindObserved {
                promoter,
                path,
                kind,
            } => Self::PromotionKindObserved {
                at: at.clone(),
                promoter: WireId::from(*promoter),
                path: WirePath::from(path),
                kind: WireResourceKind::from(*kind),
            },
            Diagnostic::PromoterFanoutThreshold { promoter, count } => {
                Self::PromoterFanoutThreshold {
                    at: at.clone(),
                    promoter: WireId::from(*promoter),
                    count: *count,
                }
            }
            Diagnostic::PromoterProxyStaleEvent { promoter, resource } => {
                Self::PromoterProxyStaleEvent {
                    at: at.clone(),
                    promoter: WireId::from(*promoter),
                    resource: WireId::from(*resource),
                }
            }
            Diagnostic::PromoterEnumerationVanished { promoter, proxy } => {
                Self::PromoterEnumerationVanished {
                    at: at.clone(),
                    promoter: WireId::from(*promoter),
                    proxy: WireId::from(*proxy),
                }
            }
            Diagnostic::PromoterEnumerationFailed {
                promoter,
                proxy,
                failure,
            } => Self::PromoterEnumerationFailed {
                at: at.clone(),
                promoter: WireId::from(*promoter),
                proxy: WireId::from(*proxy),
                errno: failure.errno(),
            },
            Diagnostic::DynamicSubReaped {
                promoter,
                sub,
                path,
            } => Self::DynamicSubReaped {
                at: at.clone(),
                promoter: WireId::from(*promoter),
                sub: WireId::from(*sub),
                path: WirePath::from(path),
            },
            Diagnostic::InvalidBurstTransition {
                profile,
                helper,
                observed,
            } => Self::InvalidBurstTransition {
                at: at.clone(),
                profile: WireId::from(*profile),
                helper: WireBurstHelper::from(*helper),
                observed: WireProfileStateDiscriminant::from(*observed),
            },
        }
    }
}

impl WireDiagnostic {
    /// Wire tag for this variant — the same `"diag"` field value the
    /// JSON form carries. Mirrors the serde tag exactly: snake_case
    /// by default (so [`Self::StaleProbeResponse`] →
    /// `"stale_probe_response"`) or the explicit `#[serde(rename =
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
            Self::StaleProbeResponse { .. } => "stale_probe_response",
            Self::StaleTimer { .. } => "stale_timer",
            Self::EffectCompleteOutsideAwaiting { .. } => "effect_complete_outside_awaiting",
            Self::EffectCompleteForUnknownSub { .. } => "effect_complete_for_unknown_sub",
            Self::DetachUnknownSub { .. } => "detach_unknown_sub",
            Self::ConfigDiffUnknownSub { .. } => "config_diff_unknown_sub",
            Self::ConfigDiffUnknownPromoter { .. } => "config_diff_unknown_promoter",
            Self::ConfigDiffRebindFallbackAttach { .. } => "config_diff_rebind_fallback_attach",
            Self::ProbeVanished { .. } => "probe_vanished",
            Self::ProbeFailed { .. } => "probe_failed",
            Self::EventClassDropped { .. } => "event_class_dropped",
            Self::EventOnUnwatchedResource { .. } => "event_on_unwatched_resource",
            Self::EventNoConsumer { .. } => "event_no_consumer",
            Self::WatchOpRejected { .. } => "watch_op_rejected",
            Self::PendingPathProbeVanished { .. } => "pending_path_probe_vanished",
            Self::PendingPathProbeFailed { .. } => "pending_path_probe_failed",
            Self::ReapPendingCancelled { .. } => "reap_pending_cancelled",
            Self::ProfileReaped { .. } => "profile_reaped",
            Self::ProfileClaimPurged { .. } => "profile_claim_purged",
            Self::PromoterClaimPurged { .. } => "promoter_claim_purged",
            Self::AttachPathInvalid { .. } => "attach_path_invalid",
            Self::AttachResourceStale { .. } => "attach_resource_stale",
            Self::AnchorKindMismatch { .. } => "anchor_kind_mismatch",
            Self::SpliceCrossedUncovered { .. } => "splice_crossed_uncovered",
            Self::EventAbsorbedByFireTail { .. } => "event_absorbed_by_fire_tail",
            Self::AwaitGateDeadlineForceRebasing { .. } => "await_gate_deadline_force_rebasing",
            Self::AwaitGateDeadlineReap { .. } => "await_gate_deadline_reap",
            Self::QuiescenceCeilingUnreadable { .. } => "quiescence_ceiling_unreadable",
            Self::RebaseCeilingStillChanging { .. } => "rebase_ceiling_still_changing",
            Self::QuiescenceCeilingForcedDespiteChange { .. } => {
                "quiescence_ceiling_forced_despite_change"
            }
            Self::RebaseCeilingForcedDespiteChange { .. } => "rebase_ceiling_forced_despite_change",
            Self::RebaseCeilingUnreadable { .. } => "rebase_ceiling_unreadable",
            Self::SensorOverflow { .. } => "sensor_overflow",
            Self::PromoterReseededForOverflow { .. } => "promoter_reseeded_for_overflow",
            Self::PerFileDriftDroppedOnRecovery { .. } => "per_file_drift_dropped_on_recovery",
            Self::PerFileFireSkippedOnFreshSeed { .. } => "per_file_fire_skipped_on_fresh_seed",
            Self::SubAttached { .. } => "sub_attached",
            Self::SubFired { .. } => "sub_fired",
            Self::QuiescenceAbsorbed { .. } => "quiescence_absorbed",
            Self::AbsorbArmed { .. } => "absorb_armed",
            Self::SubDetached { .. } => "sub_detached",
            Self::SubRebound { .. } => "sub_rebound",
            Self::RebindUnknownSub { .. } => "rebind_unknown_sub",
            Self::PromoterAttached { .. } => "promoter_attached",
            Self::PromoterReaped { .. } => "promoter_reaped",
            Self::PromoterDescentVanished { .. } => "promoter_descent_vanished",
            Self::PromoterDescentFailed { .. } => "promoter_descent_failed",
            Self::PromotionKindObserved { .. } => "promotion_kind_observed",
            Self::PromoterFanoutThreshold { .. } => "promoter_fanout_threshold",
            Self::PromoterProxyStaleEvent { .. } => "promoter_proxy_stale_event",
            Self::PromoterEnumerationVanished { .. } => "promoter_enumeration_vanished",
            Self::PromoterEnumerationFailed { .. } => "promoter_enumeration_failed",
            Self::DynamicSubReaped { .. } => "dynamic_sub_reaped",
            Self::InvalidBurstTransition { .. } => "invalid_burst_transition",
            Self::Missed { .. } => "_missed",
        }
    }
}

/// [`WireDiagnostic`] is structurally infallible to serialize: every
/// variant payload is plain data ([`WireTime`]'s manual
/// `serialize_str` over an invariant-by-construction RFC-3339 token,
/// [`WireId`] / `Wire*` enum derives, [`CompactString`] /
/// [`WirePath`] as quoted strings, primitives). No field uses a
/// `serialize_with` adapter that could return `Err`. Marks the
/// diag-fan-out path
/// ([`crate::driver::Hub::dispatch_to_subscribers`]), the
/// back-pressure `_missed` marker emit
/// ([`crate::driver::ipc::conns::ConnState::try_dispatch_diag`]), and the
/// client `tail -o json` re-emit ([`crate::ipc::client::tail`]) safe
/// for [`crate::ipc::framing::encode_line`] without an
/// `.expect`-at-a-distance.
impl InfallibleSerialize for WireDiagnostic {}

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
    "stale_probe_response",
    "stale_timer",
    "effect_complete_outside_awaiting",
    "effect_complete_for_unknown_sub",
    "detach_unknown_sub",
    "config_diff_unknown_sub",
    "config_diff_unknown_promoter",
    "config_diff_rebind_fallback_attach",
    "probe_vanished",
    "probe_failed",
    "event_class_dropped",
    "event_on_unwatched_resource",
    "event_no_consumer",
    "watch_op_rejected",
    "pending_path_probe_vanished",
    "pending_path_probe_failed",
    "reap_pending_cancelled",
    "profile_reaped",
    "profile_claim_purged",
    "promoter_claim_purged",
    "attach_path_invalid",
    "attach_resource_stale",
    "anchor_kind_mismatch",
    "splice_crossed_uncovered",
    "event_absorbed_by_fire_tail",
    "await_gate_deadline_force_rebasing",
    "await_gate_deadline_reap",
    "quiescence_ceiling_unreadable",
    "rebase_ceiling_still_changing",
    "quiescence_ceiling_forced_despite_change",
    "rebase_ceiling_forced_despite_change",
    "rebase_ceiling_unreadable",
    "sensor_overflow",
    "promoter_reseeded_for_overflow",
    "per_file_drift_dropped_on_recovery",
    "per_file_fire_skipped_on_fresh_seed",
    "sub_attached",
    "sub_fired",
    "quiescence_absorbed",
    "absorb_armed",
    "sub_detached",
    "sub_rebound",
    "rebind_unknown_sub",
    "promoter_attached",
    "promoter_reaped",
    "promoter_descent_vanished",
    "promoter_descent_failed",
    "promotion_kind_observed",
    "promoter_fanout_threshold",
    "promoter_proxy_stale_event",
    "promoter_enumeration_vanished",
    "promoter_enumeration_failed",
    "dynamic_sub_reaped",
    "invalid_burst_transition",
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

impl std::fmt::Display for WireOverflowScope {
    /// Operator-visible label — `resource/<id>` for the per-resource
    /// arm, bare `global` for the daemon-wide arm. Mirrors
    /// [`WireTime`] / [`WirePath`]: the renderer writes the projection
    /// verbatim through the formatter, no per-event allocation.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Resource { resource } => write!(f, "resource/{}", resource.0),
            Self::Global => f.write_str("global"),
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

impl std::fmt::Display for WireProbeOwner {
    /// Operator-visible label — `<kind>/<id>` so the owner tag and the
    /// inner id read as one token in the renderer's `owner=` field.
    /// Mirrors [`WireTime`] / [`WirePath`]'s `Display`-as-projection
    /// precedent so per-event diag rendering carries no allocation.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Profile { profile } => write!(f, "profile/{}", profile.0),
            Self::Promoter { promoter } => write!(f, "promoter/{}", promoter.0),
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

impl std::fmt::Display for WireWatchFailure {
    /// Operator-visible label — `<class>(errno=<n>)` so the failure
    /// class and the raw kernel errno read together. Operators chasing
    /// kernel-pressure incidents see the errno without consulting
    /// `errno.h` separately. Mirrors [`WireTime`] / [`WirePath`]'s
    /// `Display`-as-projection precedent.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Pressure { errno } => write!(f, "pressure(errno={errno})"),
            Self::Resource { errno } => write!(f, "resource(errno={errno})"),
            Self::Invariant { errno } => write!(f, "invariant(errno={errno})"),
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
    RetryDrivesBatching,
    TransitionToVerifying,
    TransitionToDraining,
    TransitionToAwaiting,
    TransitionToRebasing,
    TransitionToSettling,
    AbsorbEventIntoFireTail,
    RestartBurstFromFireTailResidual,
}

impl From<BurstHelper> for WireBurstHelper {
    fn from(h: BurstHelper) -> Self {
        match h {
            BurstHelper::StartSeedBurst => Self::StartSeedBurst,
            BurstHelper::StartStandardBurst => Self::StartStandardBurst,
            BurstHelper::EventDrivesBatching => Self::EventDrivesBatching,
            BurstHelper::RetryDrivesBatching => Self::RetryDrivesBatching,
            BurstHelper::TransitionToVerifying => Self::TransitionToVerifying,
            BurstHelper::TransitionToDraining => Self::TransitionToDraining,
            BurstHelper::TransitionToAwaiting => Self::TransitionToAwaiting,
            BurstHelper::TransitionToRebasing => Self::TransitionToRebasing,
            BurstHelper::TransitionToSettling => Self::TransitionToSettling,
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
    Settling,
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
            StateLabel::Settling => Self::Settling,
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

/// Retirement-discipline projection of [`specter_core::AbsorbMode`].
/// Surfaces both in [`WireDiagnostic::AbsorbArmed`] (so a `tail` sees
/// the arm's mode) and in [`WireAbsorbWindow::mode`] on the `show`
/// detail block. The `show` human renderer maps it to the operator
/// labels `consume-on-first` / `persist`.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireAbsorbMode {
    ConsumeOnFirst,
    PersistUntil,
}

impl From<AbsorbMode> for WireAbsorbMode {
    fn from(m: AbsorbMode) -> Self {
        match m {
            AbsorbMode::ConsumeOnFirst => Self::ConsumeOnFirst,
            AbsorbMode::PersistUntil => Self::PersistUntil,
        }
    }
}

/// `show`-detail projection of an armed [`specter_core::AbsorbWindow`].
///
/// Constructed field-by-field in the `show` projection
/// ([`crate::driver::ipc::project`]) rather than through a `From`: the
/// window's expiry is an engine-monotonic [`std::time::Instant`] with
/// no wall-clock of its own, so the projection threads it through the
/// driver's startup-anchor pair (`project_wall`) to reach a
/// [`WireTime`]. The projection is live-gated at the call site — an
/// inert window (`expiry <= now`) projects to `None`, never a stale
/// `Some`. Surfaces in [`crate::ipc::protocol::SubDetails::absorb`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct WireAbsorbWindow {
    /// Wall-clock projection of the window's expiry instant.
    pub(crate) expiry: WireTime,
    /// Retirement discipline — see [`WireAbsorbMode`].
    pub(crate) mode: WireAbsorbMode,
}

/// Reload-trigger projection of [`crate::driver::ReloadTrigger`].
/// The enum lives here to keep every wire shape (core- or bin-sourced)
/// declared in one module; the `From` projection lives at the source
/// ([`crate::driver::state`]) so a new `ReloadTrigger` variant fails
/// to compile at its declaration site, keeping the wire layer a leaf
/// (no `crate::driver` import here). Surfaces in
/// `StatusResponse.last_reload_via`.
///
/// `AutoReload` projects to operator-facing `auto` — the engine-internal
/// `AutoReload` name carries the "settle-expiry observed drift"
/// mechanism that doesn't belong in the operator vocabulary. `Startup`
/// projects verbatim and surfaces when boot-time TOCTOU drift drove the
/// reload (the daemon's first config-handling action was an apply on
/// freshly-detected drift, not a steady-state operator pulse).
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireReloadTrigger {
    Sighup,
    Auto,
    Ipc,
    Startup,
}

#[cfg(test)]
mod tests {
    use super::{
        KNOWN_WIRE_VARIANTS, WireAbsorbMode, WireBurstHelper, WireBurstIntent, WireClaimKind,
        WireDetachReason, WireDiagnostic, WireFsEvent, WireOverflowScope, WirePath, WireProbeOwner,
        WireProfileStateDiscriminant, WirePromoterClaimKind, WireReapTrigger, WireResourceKind,
        WireSpliceFailureCause, WireTime, WireWatchFailure,
    };
    use crate::ipc::protocol::WireId;
    use std::collections::BTreeSet;
    use std::path::Path;
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

    /// `Deserialize` for [`WireTime`] is the symmetric gate — a
    /// non-RFC-3339 token is rejected at the boundary, not stored as
    /// opaque text. Pins the contract change behind the audit's
    /// F-HIGH-3 finding: the wire layer validates *both* directions
    /// of the round-trip. One witness covers the gate; humantime's
    /// internal failure modes are humantime's contract, not ours.
    #[test]
    fn wire_time_rejects_malformed_string() {
        let err = serde_json::from_str::<WireTime>(r#""not a date""#)
            .expect_err("malformed RFC 3339 must be rejected by WireTime::deserialize");
        let msg = err.to_string();
        assert!(
            msg.contains("RFC 3339"),
            "rejection message should name the format; got {msg:?}",
        );
    }

    /// A well-formed RFC 3339 token round-trips through serde with
    /// byte-identical output — the deserialize gate accepts valid
    /// input and the Serialize impl preserves the bytes verbatim. The
    /// server-emit shape is pinned separately by
    /// [`wire_time_clamps_pre_epoch_to_unix_epoch`]; this test pins
    /// the complementary half (Deserialize ↔ Serialize byte stability).
    #[test]
    fn wire_time_round_trips_rfc3339() {
        let bytes = r#""2026-05-23T15:30:00Z""#;
        let parsed: WireTime = serde_json::from_str(bytes).expect("valid RFC 3339 deserialize");
        let again = serde_json::to_string(&parsed).expect("serialize");
        assert_eq!(again, bytes, "round-trip preserves wire bytes");
    }

    /// A non-UTF-8 path projects to U+FFFD-bearing UTF-8 at construct
    /// time and round-trips cleanly through serde — `serde_json` never
    /// sees the offending bytes, so the daemon-panic surface the
    /// audit's F-CRIT-1 finding identified is structurally closed.
    ///
    /// The construct-time projection (`Path::to_string_lossy`) is the
    /// load-bearing barrier: a `WirePath` whose inner [`String`]
    /// already holds valid UTF-8 cannot panic at JSON-serialize time,
    /// regardless of how exotic the source `PathBuf` / `Arc<Path>`
    /// bytes are.
    ///
    /// Runs unix-only because [`OsStr::from_bytes`] is the standard
    /// way to manufacture non-UTF-8 path bytes; the projection is the
    /// same on every platform but the witness needs a non-UTF-8
    /// `OsStr` to be meaningful.
    #[cfg(unix)]
    #[test]
    fn wire_path_projects_non_utf8_to_replacement_and_round_trips() {
        use std::ffi::OsStr;
        use std::os::unix::ffi::OsStrExt;

        // 0xff 0xfe is an invalid UTF-8 prefix on every Unix.
        let raw: &OsStr = OsStr::from_bytes(&[0xff, 0xfe]);
        let path = Path::new(raw);
        let wire = WirePath::from(path);

        // U+FFFD is the lossy-projection sentinel. Path::to_string_lossy
        // emits one per invalid byte sequence; both bytes here form one
        // invalid sequence (0xff is not a valid lead byte), so each
        // shows as its own replacement. Reach through Display to read
        // the projected form back as a String witness.
        let projected = wire.to_string();
        assert!(
            projected.chars().any(|c| c == '\u{FFFD}'),
            "non-UTF-8 path projects through U+FFFD; got {projected:?}",
        );

        // Construct-time validity ⇒ JSON-serialize cannot panic; the
        // bytes survive a full round-trip (structural equality on
        // WirePath is via the inner String).
        let json = serde_json::to_string(&wire).expect("WirePath serialization is infallible");
        let back: WirePath = serde_json::from_str(&json).expect("deserialize");
        assert_eq!(back, wire);
    }

    /// [`WireProbeOwner`]'s `Display` projects to `<kind>/<id>` — the
    /// operator-visible token the `owner=` column on
    /// `StaleProbeResponse` lines carries. Extends [`WireTime`] /
    /// [`WirePath`]'s `Display`-as-projection precedent across the
    /// compound enums; the renderer writes verbatim through the
    /// formatter (no per-event allocation).
    #[test]
    fn wire_probe_owner_display_projects_kind_slash_id() {
        assert_eq!(
            WireProbeOwner::Profile { profile: WireId(7) }.to_string(),
            "profile/7",
        );
        assert_eq!(
            WireProbeOwner::Promoter {
                promoter: WireId(42),
            }
            .to_string(),
            "promoter/42",
        );
    }

    /// [`WireOverflowScope`]'s `Display` is asymmetric by design — the
    /// `Resource` arm carries an id (`resource/<n>`), the `Global` arm
    /// is the bare tag. Operators reading `scope=` on `SensorOverflow`
    /// lines distinguish daemon-wide overflow from a single-resource
    /// queue overrun by the absence of the trailing `/<id>`.
    #[test]
    fn wire_overflow_scope_display_resource_carries_id_global_is_bare() {
        assert_eq!(
            WireOverflowScope::Resource {
                resource: WireId(13),
            }
            .to_string(),
            "resource/13",
        );
        assert_eq!(WireOverflowScope::Global.to_string(), "global");
    }

    /// [`WireWatchFailure`]'s `Display` carries `(errno=<n>)` for every
    /// arm — the raw kernel errno is the operator's index into
    /// `errno.h` and stays paired with the failure class in the
    /// rendered `failure=` field.
    #[test]
    fn wire_watch_failure_display_includes_errno() {
        assert_eq!(
            WireWatchFailure::Pressure { errno: 24 }.to_string(),
            "pressure(errno=24)",
        );
        assert_eq!(
            WireWatchFailure::Resource { errno: 28 }.to_string(),
            "resource(errno=28)",
        );
        assert_eq!(
            WireWatchFailure::Invariant { errno: 22 }.to_string(),
            "invariant(errno=22)",
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
    /// `From<(&Diagnostic, &WireTime)>` arm (compile-time, exhaustive),
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
                path: WirePath::from(Path::new("/tmp/x")),
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
                first_unread: WirePath::from(Path::new("/tmp/x/y")),
                intent: WireBurstIntent::Standard,
            },
            WireDiagnostic::RebaseCeilingStillChanging {
                at: at(),
                profile: WireId(121),
                intent: WireBurstIntent::Standard,
            },
            WireDiagnostic::QuiescenceCeilingForcedDespiteChange {
                at: at(),
                profile: WireId(123),
                intent: WireBurstIntent::Standard,
            },
            WireDiagnostic::RebaseCeilingForcedDespiteChange {
                at: at(),
                profile: WireId(124),
                intent: WireBurstIntent::Seed,
            },
            WireDiagnostic::RebaseCeilingUnreadable {
                at: at(),
                profile: WireId(122),
                first_unread: WirePath::from(Path::new("/tmp/z")),
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
            WireDiagnostic::QuiescenceAbsorbed {
                at: at(),
                profile: WireId(157),
            },
            WireDiagnostic::AbsorbArmed {
                at: at(),
                profile: WireId(158),
                mode: WireAbsorbMode::ConsumeOnFirst,
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
                path: WirePath::from(Path::new("/tmp/p/x")),
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
                path: WirePath::from(Path::new("/tmp/p/dyn")),
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
