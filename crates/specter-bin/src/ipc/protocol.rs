//! Operator IPC protocol — request layering, response carriers,
//! wire-id newtype, and error-code constants.
//!
//! # Three-type request layering
//!
//! ```text
//!                        ┌─ JSON line ─┐    ┌─ channel ─┐
//! client ──write──> [WireRequest]  ───────> [RequestPayload]
//!                                            + [IpcRequest { reply_tx }]
//!                                            ────send────> driver
//! ```
//!
//! - [`WireRequest`] is the only type the daemon parses from the
//!   socket. Deserialize-only: operators address by name, not by id,
//!   so the daemon refuses to admit `WireId` values from clients.
//! - [`RequestPayload`] is the channel-bound shape the driver
//!   receives. Its [`RequestPayload::Subscribe`] arm carries the
//!   `Sender<BrokerEvent>` the broker fans into — a routing identity
//!   with no wire representation, so the type derives neither
//!   `Serialize` nor `Deserialize` by construction.
//! - [`IpcRequest`] is the envelope: payload + a `bounded(1)` reply
//!   channel. One verb, one response.
//!
//! # Response shape
//!
//! [`ResponsePayload`] is internally tagged on `kind`; every variant
//! flattens into a single JSON object keyed by `kind`, symmetric with
//! [`super::wire::WireDiagnostic`]'s `diag` tag. Operators
//! filter the entire wire surface with one `jq` pattern.
//!
//! # Visibility
//!
//! Every export is `pub(crate)`. The driver IPC drain, server thread,
//! projection helpers, and client verbs consume each type at its
//! own point of use.

use compact_str::CompactString;
use crossbeam::channel::Sender;
use serde::{Deserialize, Serialize};
use slotmap::Key;
use specter_core::{ProfileId, PromoterId, ResourceId, SubId};
use std::borrow::Cow;
use std::path::PathBuf;

use super::wire::{BrokerEvent, WireEffectScope, WireReloadTrigger, WireStateLabel, WireTime};

/// Operator-facing wire request — the shape the daemon parses from
/// the socket and the client constructs at write time.
///
/// The type carries name-only addressing by construction: no variant
/// holds a [`WireId`] field, so clients cannot round-trip a slotmap
/// key whose generation has since expired engine-side (a fresh
/// disable/enable would reuse the slot index and the cached id would
/// silently resolve to a different Sub). The structural floor lives
/// in the field shapes, not in a missing `Serialize` impl — both
/// directions of the wire round-trip the same JSON object.
///
/// Tagged internally on `op`; both the tag value and the field names
/// use `snake_case` so the wire form matches the operator vocabulary
/// (`{"op":"status"}`, `{"op":"show","name":"foo"}`).
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub(crate) enum WireRequest {
    Status,
    List,
    Show {
        name: CompactString,
    },
    Disable {
        name: CompactString,
    },
    Enable {
        name: CompactString,
    },
    Reload,
    /// Subscribe to the diagnostic stream. `name` is optional — `None`
    /// is an unfiltered tail; `Some(n)` scopes the subscription to
    /// events naming that Sub, collapsing the historical
    /// resolve-then-subscribe race window for the `wait` verb.
    Subscribe {
        #[serde(default)]
        name: Option<CompactString>,
    },
}

/// Channel-bound request payload — the shape the driver receives on
/// its IPC arm.
///
/// Distinct from [`WireRequest`] only by the [`RequestPayload::Subscribe`]
/// variant: subscribing carries a `Sender<BrokerEvent>` clone for the
/// broker to fan diagnostics into. The sender is a routing identity,
/// not a wire value, which is why the two-type split exists.
pub(crate) enum RequestPayload {
    Status,
    List,
    Show {
        name: CompactString,
    },
    Disable {
        name: CompactString,
    },
    Enable {
        name: CompactString,
    },
    Reload,
    Subscribe {
        tx: Sender<BrokerEvent>,
        name: Option<CompactString>,
    },
}

impl std::fmt::Debug for RequestPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Status => f.write_str("Status"),
            Self::List => f.write_str("List"),
            Self::Show { name } => f.debug_struct("Show").field("name", name).finish(),
            Self::Disable { name } => f.debug_struct("Disable").field("name", name).finish(),
            Self::Enable { name } => f.debug_struct("Enable").field("name", name).finish(),
            Self::Reload => f.write_str("Reload"),
            Self::Subscribe { name, tx: _ } => f
                .debug_struct("Subscribe")
                .field("name", name)
                .finish_non_exhaustive(),
        }
    }
}

/// Channel envelope — pairs a payload with the per-request reply
/// channel. The reply channel is constructed `bounded(1)` by the
/// server thread: one verb, one response, no queueing.
pub(crate) struct IpcRequest {
    pub(crate) payload: RequestPayload,
    pub(crate) reply_tx: Sender<ResponsePayload>,
}

impl std::fmt::Debug for IpcRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IpcRequest")
            .field("payload", &self.payload)
            .finish_non_exhaustive()
    }
}

/// Operator-facing response.
///
/// Internally tagged on `kind`, `snake_case`. Every variant
/// serializes as a flat JSON object — newtype-around-struct variants
/// (`Status`, `List`, `Show`) inline their carrier's fields next to
/// `"kind"`; struct variants (`SubscribeAck`, `Err`) carry their own
/// fields; unit variants (`Ok`) carry only the tag.
///
/// `Deserialize` is symmetric with `Serialize` so client code can
/// parse responses back from the wire — every variant payload carries
/// `Deserialize`. The wire schema is one canonical shape; the same
/// JSON object the daemon writes is what the client reads.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub(crate) enum ResponsePayload {
    Ok,
    Status(StatusResponse),
    List(ListResponse),
    Show(ShowResponse),
    /// Ack for a Subscribe; `sub` carries the resolved [`WireId`]
    /// when the request named a Sub, `None` for the unfiltered tail.
    SubscribeAck {
        sub: Option<WireId>,
    },
    /// Structured error: `code` is one of the [`ERR_*`](self) static
    /// constants on the server side — clients branch on it. `error`
    /// is the human-readable amplification.
    ///
    /// `code` is `Cow<'static, str>` so the type carries the
    /// trichotomy of construction sites in one shape: the server
    /// constructs with `Cow::Borrowed(ERR_*)` (zero-alloc, type-
    /// checked against the static constants), the client deserializes
    /// into `Cow::Owned(String)` (no `'static` borrow exists in the
    /// wire bytes). Both halves derive cleanly via the
    /// [`err_code_serde`] helper — `Cow<'static, str>` is not directly
    /// `Deserialize`-able because the `'de: 'static` bound demands
    /// statically-rooted input, which `serde_json` cannot supply.
    Err {
        #[serde(with = "err_code_serde")]
        code: Cow<'static, str>,
        error: String,
    },
}

/// Custom serde adapter for the [`ResponsePayload::Err::code`] field.
///
/// `Cow<'static, str>` cannot directly derive `Deserialize`: serde's
/// blanket `Cow<'a, str>: Deserialize<'de>` impl requires `'de: 'a`,
/// and no non-`'static` deserializer (i.e. every real-world one)
/// satisfies `'de: 'static`. The two free functions below deserialize
/// into an owned `String` first and lift into `Cow::Owned`, and
/// serialize through `str` directly so the wire form is identical
/// regardless of whether the server constructed `Borrowed(ERR_*)` or
/// the client round-tripped through `Owned`.
mod err_code_serde {
    use serde::{Deserialize, Deserializer, Serializer};
    use std::borrow::Cow;

    // `#[serde(with = "...")]` fixes the serializer's first parameter
    // shape (`&T` where `T` is the field type), so the `clippy::ptr_arg`
    // suggestion of `&str` cannot apply here — the signature is the
    // serde contract, not a free choice.
    #[allow(clippy::ptr_arg)]
    pub(super) fn serialize<S: Serializer>(
        code: &Cow<'static, str>,
        ser: S,
    ) -> Result<S::Ok, S::Error> {
        ser.serialize_str(code)
    }

    pub(super) fn deserialize<'de, D: Deserializer<'de>>(
        de: D,
    ) -> Result<Cow<'static, str>, D::Error> {
        String::deserialize(de).map(Cow::Owned)
    }
}

/// Stable wire-side projection of a `specter-core` slotmap key.
///
/// The wire form is the bare `u64` returned by
/// [`slotmap::KeyData::as_ffi`] — documented as
/// `(generation << 32) | index`, stable across slotmap minor
/// releases. `Debug` would order differently across versions and is
/// not used here.
///
/// One-way only: every [`From`] impl is *into* `WireId`. There is no
/// `From<u64>` for the four core key types because the daemon must
/// not admit a [`WireId`] from a client: see [`WireRequest`].
///
/// `#[serde(transparent)]` collapses the JSON form to a bare integer
/// (`42`), not a wrapped object (`{"WireId":42}`).
///
/// `Deserialize` rounds out the wire shape so a client-side response
/// carrier (`StatusResponse`, etc.) parses the embedded ids back into
/// `WireId` newtypes. The asymmetry around `From<SubId>` etc. (one-way
/// projections) is preserved: `Deserialize` is the wire shape, the
/// `From` impls are the engine-side projection.
#[derive(Clone, Copy, Debug, Deserialize, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
#[serde(transparent)]
pub(crate) struct WireId(pub(crate) u64);

impl From<SubId> for WireId {
    fn from(k: SubId) -> Self {
        Self(k.data().as_ffi())
    }
}

impl From<ProfileId> for WireId {
    fn from(k: ProfileId) -> Self {
        Self(k.data().as_ffi())
    }
}

impl From<ResourceId> for WireId {
    fn from(k: ResourceId) -> Self {
        Self(k.data().as_ffi())
    }
}

impl From<PromoterId> for WireId {
    fn from(k: PromoterId) -> Self {
        Self(k.data().as_ffi())
    }
}

/// Why an operator-declared Sub is currently not in the engine
/// registry. Surfaced on [`ListRow::disabled`] and
/// [`ShowResponse::Disabled::source`] so operators distinguish "I
/// disabled this via IPC and haven't re-enabled" from "the TOML has
/// `enabled = false`".
///
/// Lives in this module rather than alongside the other wire
/// projections because the discrimination is bin-local: there is no
/// `specter-core` analogue to project from.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum DisabledSource {
    /// Operator ran `specter disable <name>` via IPC and has not yet
    /// run `enable`. Cleared by `enable`, or implicitly by the TOML
    /// `[[watch]]` entry leaving the file entirely.
    Runtime,
    /// `[[watch]]` entry carries `enabled = false`. Cleared by the
    /// operator editing the config and triggering a reload.
    Toml,
}

/// Daemon-wide status snapshot. Surfaced by `specter status` —
/// uptime, reload bookkeeping, Sub / Profile / Promoter counts, and
/// the canonical paths the daemon is currently bound to.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct StatusResponse {
    /// Seconds since daemon boot — `DriverState.start_instant.elapsed()`.
    pub(crate) uptime_secs: u64,
    /// Wall-clock of daemon boot — sampled in the same constructor
    /// as `start_instant` so the two anchors agree.
    pub(crate) start_wall: WireTime,
    /// Cumulative successful reloads (SIGHUP + auto-reload + IPC).
    pub(crate) reload_count: u64,
    /// Wall-clock of the most recent successful reload, `None`
    /// before the first one fires.
    pub(crate) last_reload_at: Option<WireTime>,
    /// Trigger of the most recent successful reload. Typed enum
    /// (not `&'static str`) so a future `ReloadTrigger` variant is
    /// a compile error here, in the same shape as the rest of the
    /// wire projection layer.
    pub(crate) last_reload_via: Option<WireReloadTrigger>,
    /// `engine.subs().len()` — every currently-attached Sub.
    pub(crate) sub_total: usize,
    /// `config.disabled_names().0.len()` — TOML-disabled rows
    /// (`enabled = false`).
    pub(crate) sub_disabled_toml: usize,
    /// Operator-runtime-disabled Subs (`disabled_runtime.len()`).
    pub(crate) sub_disabled_runtime: usize,
    /// `engine.profiles().active_count()` — Profiles not in
    /// `ProfileState::Idle`.
    pub(crate) profile_active: usize,
    /// `engine.promoters().len()` — Promoters live in the registry.
    pub(crate) promoter_active: usize,
    /// Currently-loaded config's source path. Every code path through
    /// `App::run` resolves one; a future stdin-TOML / ephemeral-config
    /// mode would widen this to `Option<PathBuf>` honestly.
    pub(crate) config_path: PathBuf,
    /// UNIX-socket path the IPC server is bound to.
    pub(crate) socket_path: PathBuf,
}

/// `specter list` response — every operator-declared Sub keyed by
/// name. Attached, runtime-disabled, and TOML-disabled populations
/// union into one alphabetically-ordered row set.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ListResponse {
    pub(crate) rows: Vec<ListRow>,
}

/// One row in [`ListResponse`]. Fields scoped per row type:
///
/// - Attached rows fill every field; `disabled` is `None`.
/// - Runtime-disabled rows fill `name` + `disabled =
///   Some(Runtime)`; engine-derived fields are `None` (the Sub is
///   not in `engine.subs()`, so the Profile / state / anchor /
///   counters do not exist).
/// - TOML-disabled rows fill `name` + `disabled = Some(Toml)`; same
///   reason.
///
/// `Option<u64>` on the counter columns over plain `u64 = 0` for
/// missing rows makes "field doesn't apply" structural — JSON-schema
/// generators distinguish "never fired" from "not attached".
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct ListRow {
    /// Operator-facing name. Static Subs: `[[watch]].name`. Dynamic
    /// Subs: `<promoter_name>@<resolved_path>`.
    pub(crate) name: String,
    /// Eight-phase operator-display state. `None` for non-attached
    /// rows.
    pub(crate) state: Option<WireStateLabel>,
    /// Anchor path. `None` for non-attached rows (no Profile, no
    /// resource).
    pub(crate) anchor: Option<PathBuf>,
    /// Wall-clock projection of `Sub.last_fired_at`. `None` for
    /// never-fired Subs and non-attached rows.
    pub(crate) last_fired_at: Option<WireTime>,
    /// `Sub.fire_count`. `None` for non-attached rows.
    pub(crate) fire_count: Option<u64>,
    /// `Sub.dedup_suppressed_count`. `None` for non-attached rows.
    pub(crate) dedup_suppressed_count: Option<u64>,
    /// `Sub.settle.as_millis()` (or the Profile's settle,
    /// equivalent). `None` for non-attached rows.
    pub(crate) settle_ms: Option<u64>,
    /// Disable-source discriminator. `None` for attached rows.
    pub(crate) disabled: Option<DisabledSource>,
    /// `SubId` projection. `None` for non-attached rows.
    pub(crate) sub: Option<WireId>,
    /// `ProfileId` projection of the Sub's hosting Profile. `None`
    /// for non-attached rows.
    pub(crate) profile: Option<WireId>,
    /// `Sub.source_promoter` projection — `Some(_)` iff the Sub is
    /// promoter-minted, `None` for static (operator-declared) Subs
    /// and for non-attached rows.
    pub(crate) source_promoter: Option<WireId>,
}

/// `specter show <name>` response — internally tagged on `status` so
/// the three operator outcomes (Active / Disabled / Unknown) carry
/// their own field set without an outer envelope.
///
/// The outer envelope is [`ResponsePayload::Show`], which tags on
/// `kind`; the inner `status` tag flattens alongside. Both tags
/// appear in the same JSON object (`{"kind":"show","status":"active",
/// "name":"foo",…}`) — they do not collide, and the operator can
/// dispatch on either independently.
#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub(crate) enum ShowResponse {
    /// Sub is in `engine.subs()` — the full [`SubDetails`] block.
    Active(SubDetails),
    /// Sub is operator-declared but not attached. `source` says why.
    Disabled {
        name: String,
        source: DisabledSource,
    },
    /// Sub is not operator-declared at all (no `[[watch]]`, no
    /// runtime disable record).
    Unknown { name: String },
}

/// `specter show <name>` detail block for an attached Sub.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct SubDetails {
    /// Operator-facing name.
    pub(crate) name: String,
    /// `SubId` projection.
    pub(crate) sub: WireId,
    /// Hosting `ProfileId` projection.
    pub(crate) profile: WireId,
    /// Operator-display state (`StateLabel`).
    pub(crate) state: WireStateLabel,
    /// Anchor path (`engine.tree().path_of(profile.resource)`). `None`
    /// signals "anchor vanished / not yet resolved" — symmetric with
    /// [`ListRow::anchor`], so operators reading `list -o json` and
    /// `show -o json` decode vanish identically (`null`).
    pub(crate) anchor: Option<PathBuf>,
    /// Wall-clock projection of `Sub.last_fired_at`. `None` until
    /// the first successful fire.
    pub(crate) last_fired_at: Option<WireTime>,
    /// Cumulative fires — per-leaf for `PerStableFile`, per-burst
    /// for `SubtreeRoot`.
    pub(crate) fire_count: u64,
    /// Cumulative B1-dedup suppressions.
    pub(crate) dedup_suppressed_count: u64,
    /// `Sub.settle.as_millis()`.
    pub(crate) settle_ms: u64,
    /// `Sub.source_promoter` projection — `Some(_)` iff the Sub
    /// was minted by a Promoter. Distinct from a TOML-declared Sub
    /// with the same anchor: the promoter id locates which dynamic
    /// pattern produced the entry.
    pub(crate) source_promoter: Option<WireId>,
    /// `Sub.scope` projection.
    pub(crate) scope: WireEffectScope,
    /// One line per `ActionProgram` instruction. Rendering rules
    /// live with the projection helper (`specter-bin`'s
    /// `ipc::project::program`); this field pins only the shape.
    pub(crate) program: Vec<String>,
}

/// Sub name not in the engine registry, the operator-runtime disable
/// set, or the TOML watches.
pub(crate) const ERR_UNKNOWN_SUB: &str = "unknown_sub";

/// Operator targeted a dynamic (promoter-spawned) Sub with an op
/// the bin refuses to apply: the synthesised name would silently
/// evaporate on the next reload's promoter prune pass.
///
/// Consumed by the `disable` / `enable` IPC handlers in
/// [`crate::driver`].
pub(crate) const ERR_DYNAMIC_SUB_NO_OP: &str = "dynamic_sub_no_op";

/// `enable` / `disable` precondition: the targeted Sub is not in
/// the state the verb expects (`enable` against a Sub that is not
/// runtime-disabled, or `disable` against one already disabled).
pub(crate) const ERR_NOT_DISABLED: &str = "not_disabled";

/// `enable` cleared the runtime override but the Sub is also
/// TOML-disabled (or absent from the file entirely) — the daemon
/// cannot re-attach until the operator edits the config and
/// reloads.
pub(crate) const ERR_TOML_DISABLED: &str = "toml_disabled";

/// Connection cap reached — too many concurrent operator clients.
pub(crate) const ERR_BUSY: &str = "busy";

/// Daemon is in the shutdown path; no further requests served.
pub(crate) const ERR_SHUTDOWN: &str = "shutdown";

/// Request line failed JSON parse, or carries an unknown `op`.
pub(crate) const ERR_MALFORMED: &str = "malformed";

#[cfg(test)]
mod tests {
    use super::{DisabledSource, ERR_UNKNOWN_SUB, ResponsePayload, WireId, WireRequest};
    use slotmap::KeyData;
    use specter_core::{ProfileId, PromoterId, ResourceId, SubId};
    use std::borrow::Cow;

    /// Every slotmap key family projects through `KeyData::as_ffi()`
    /// to the same canonical `WireId(u64)`. A regression in any
    /// `From` impl, or a future-added key family that grew an impl
    /// returning a different shape, fails here.
    ///
    /// The canonical value comes from `KeyData::as_ffi()` itself,
    /// not from the raw bits handed to `from_ffi`: `KeyData::from_ffi`
    /// normalizes the generation's high bit so a freshly minted key
    /// is non-default, so an arbitrary `u64 → from_ffi → as_ffi`
    /// round-trip is not guaranteed to preserve every bit.
    #[test]
    fn wire_id_round_trips_slotmap_as_ffi() {
        let raw: u64 = 0x1234_5678_9abc_def0;
        let canonical = KeyData::from_ffi(raw).as_ffi();
        assert_eq!(
            WireId::from(SubId::from(KeyData::from_ffi(raw))).0,
            canonical
        );
        assert_eq!(
            WireId::from(ProfileId::from(KeyData::from_ffi(raw))).0,
            canonical
        );
        assert_eq!(
            WireId::from(ResourceId::from(KeyData::from_ffi(raw))).0,
            canonical
        );
        assert_eq!(
            WireId::from(PromoterId::from(KeyData::from_ffi(raw))).0,
            canonical
        );
    }

    /// `WireRequest` deserializes the unit-shaped verbs, the
    /// name-bearing struct variants, and the absent-name form of
    /// Subscribe. `#[serde(default)]` on `Subscribe.name` is the
    /// load-bearing detail — missing field yields `None`, not parse
    /// failure.
    #[test]
    fn wire_request_parses_unit_and_struct_variants() {
        let parsed: WireRequest = serde_json::from_str(r#"{"op":"status"}"#).unwrap();
        assert!(matches!(parsed, WireRequest::Status));

        let parsed: WireRequest = serde_json::from_str(r#"{"op":"show","name":"foo"}"#).unwrap();
        match parsed {
            WireRequest::Show { name } => assert_eq!(name.as_str(), "foo"),
            other => panic!("expected Show, got {other:?}"),
        }

        let parsed: WireRequest = serde_json::from_str(r#"{"op":"subscribe"}"#).unwrap();
        match parsed {
            WireRequest::Subscribe { name } => assert_eq!(name, None),
            other => panic!("expected Subscribe(None), got {other:?}"),
        }

        let parsed: WireRequest =
            serde_json::from_str(r#"{"op":"subscribe","name":"foo"}"#).unwrap();
        match parsed {
            WireRequest::Subscribe { name } => {
                assert_eq!(name.as_deref(), Some("foo"));
            }
            other => panic!("expected Subscribe(Some), got {other:?}"),
        }
    }

    /// A typoed `op` is a parse failure — the wire surface refuses
    /// to silently match nothing. Catches operator typos
    /// (e.g. `sub_fire` for `status`) at the daemon boundary.
    #[test]
    fn wire_request_rejects_unknown_op() {
        assert!(serde_json::from_str::<WireRequest>(r#"{"op":"sub_fire"}"#).is_err());
        assert!(serde_json::from_str::<WireRequest>(r#"{"op":"STATUS"}"#).is_err());
        assert!(serde_json::from_str::<WireRequest>(r"{}").is_err());
    }

    /// `ResponsePayload`'s internal tag is the load-bearing wire
    /// commitment: every variant flattens into one JSON object
    /// keyed by `kind`. A retrofit to external tagging would change
    /// the operator-visible shape and fail this test.
    #[test]
    fn response_payload_round_trips_internal_tag() {
        let ok = serde_json::to_string(&ResponsePayload::Ok).unwrap();
        assert_eq!(ok, r#"{"kind":"ok"}"#);

        let err = serde_json::to_string(&ResponsePayload::Err {
            code: Cow::Borrowed(ERR_UNKNOWN_SUB),
            error: "no watch named foo".into(),
        })
        .unwrap();
        assert_eq!(
            err,
            r#"{"kind":"err","code":"unknown_sub","error":"no watch named foo"}"#
        );

        // Wire round-trip — server emits `Cow::Borrowed`; client
        // deserializes through `Cow::Owned`. Both halves observe the
        // same canonical bytes (`{"kind":"err","code":...,"error":...}`).
        let round_trip: ResponsePayload = serde_json::from_str(&err).unwrap();
        match round_trip {
            ResponsePayload::Err { code, error } => {
                assert_eq!(code.as_ref(), ERR_UNKNOWN_SUB);
                assert_eq!(error, "no watch named foo");
            }
            other => panic!("expected Err, got {other:?}"),
        }

        let ack = serde_json::to_string(&ResponsePayload::SubscribeAck { sub: None }).unwrap();
        assert_eq!(ack, r#"{"kind":"subscribe_ack","sub":null}"#);

        let ack_some = serde_json::to_string(&ResponsePayload::SubscribeAck {
            sub: Some(WireId(7)),
        })
        .unwrap();
        assert_eq!(ack_some, r#"{"kind":"subscribe_ack","sub":7}"#);

        let disabled = serde_json::to_string(&DisabledSource::Runtime).unwrap();
        assert_eq!(disabled, r#""runtime""#);
    }
}
