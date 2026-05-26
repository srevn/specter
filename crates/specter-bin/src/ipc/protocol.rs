//! Operator IPC protocol — wire-side request shape, response carriers,
//! wire-id newtype, and the [`WireErrorCode`] closed-vocabulary enum.
//!
//! # Single-type request shape
//!
//! ```text
//!                        ┌─ JSON line ─┐
//! client ──write──> [WireRequest] ────────> driver (parses + dispatches inline)
//! ```
//!
//! [`WireRequest`] is the only request type the daemon ever sees: the
//! mio-reactor driver reads bytes off each per-conn stream, parses one
//! `WireRequest` per line, and dispatches inline on the same thread —
//! no channel envelope, no per-request reply channel, no per-conn
//! worker thread.
//!
//! Deserialize-only: operators address by name, not by id, so the
//! daemon refuses to admit `WireId` values from clients.
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
//! Every export is `pub(crate)`. The driver IPC drain, projection
//! helpers, and client verbs consume each type at its own point of
//! use.

use compact_str::CompactString;
use serde::{Deserialize, Serialize};
use slotmap::Key;
use specter_core::{ProfileId, PromoterId, ResourceId, SubId};

use super::framing::InfallibleSerialize;
use super::wire::{WireEffectScope, WirePath, WireReloadTrigger, WireStateLabel, WireTime};

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
///
/// Unknown-field strictness lives at the wire boundary, not on the
/// type: [`crate::ipc::framing::parse_strict`] round-trip-validates
/// every incoming request line so a typoed JSON
/// (`{"op":"subscribe","names":"build"}`) reaches the handler as
/// [`WireErrorCode::Malformed`] rather than silently dropping the
/// typo'd field. The boundary discipline applies uniformly to
/// `WireRequest` (daemon read) and [`ResponsePayload`] (client read);
/// `WireDiagnostic` is deliberately exempt — streamed diags stay
/// forward-compatible with older `tail`/`wait` clients.
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
///
/// Unknown-field strictness on the client side lives at the wire
/// boundary via [`crate::ipc::framing::parse_strict`] — a future
/// daemon field this client doesn't know about reaches the verb
/// handler as a parse error rather than silently dropping the value.
/// Symmetric with the daemon-side request gate (see
/// [`WireRequest`]'s rustdoc).
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
    /// Structured error: `code` is a closed-set [`WireErrorCode`]
    /// vocabulary the daemon emits — clients branch on it. `error`
    /// is the human-readable amplification.
    Err {
        code: WireErrorCode,
        error: String,
    },
}

/// Closed-set error vocabulary for [`ResponsePayload::Err::code`].
///
/// Every variant is a unit value; `#[serde(rename_all = "snake_case")]`
/// makes the wire form a bare quoted token (`"unknown_sub"`,
/// `"already_subscribed"`, …) symmetric with the rest of the
/// wire-projection layer's discipline ([`WireFsEvent`](super::wire),
/// [`WireStateLabel`](super::wire), etc.). Serialize and Deserialize
/// validate the same finite set, so a client receiving a daemon-emitted
/// error parses into the same typed variant the daemon constructed —
/// no host type behind the field, no separate adapter, no `Cow`.
///
/// [`Display`](std::fmt::Display) writes the exact wire form via
/// [`Self::as_str`], so the existing client renderer
/// (`eprintln!("specter <verb>: {code}: {error}")`) emits the same
/// bytes after the refactor as before.
///
/// The vocabulary is intentionally closed (no `#[serde(other)]`
/// fallback): a client that hits a daemon emitting a code it doesn't
/// understand surfaces the failure loudly through the verb's catch-all
/// arm (`unexpected response: ...`), rather than silently parsing a
/// future code into an opaque sink. Per the audit's policy split,
/// request/response carriers favor strictness; diag fan-out favors
/// forgiveness.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub(crate) enum WireErrorCode {
    /// Sub name not in the engine registry, the operator-runtime
    /// disable set, or the TOML watches.
    UnknownSub,
    /// Operator targeted a dynamic (promoter-spawned) Sub with an op
    /// the bin refuses to apply: the synthesised name would silently
    /// evaporate on the next reload's promoter prune pass.
    ///
    /// Consumed by the `disable` / `enable` IPC handlers in
    /// [`crate::driver`].
    DynamicSubNoOp,
    /// `enable` / `disable` precondition: the targeted Sub is not in
    /// the state the verb expects (`enable` against a Sub that is
    /// not runtime-disabled, or `disable` against one already
    /// disabled).
    NotDisabled,
    /// `enable` cleared the runtime override but the Sub is also
    /// TOML-disabled (or absent from the file entirely) — the daemon
    /// cannot re-attach until the operator edits the config and
    /// reloads.
    TomlDisabled,
    /// Connection cap reached — too many concurrent operator clients.
    Busy,
    /// Daemon refused a response whose serialized bytes would exceed
    /// the per-conn write-queue accept cap. Symmetric with [`Self::Busy`]:
    /// both are cap-class failures, both close the conn after the
    /// structured Err line flushes, both let operator scripts branch
    /// on a single closed-vocabulary token rather than parsing an
    /// `UnexpectedEof`.
    ///
    /// Emitted by [`crate::driver::Hub::enqueue_response`]'s Refused
    /// arm; the structured line is queued into the per-conn reserve
    /// (sized so the Err always fits regardless of queue state) and
    /// `close_after_flush` is armed by the upstream `push_response`.
    /// The `error` field carries the byte counts as
    /// `response of N bytes exceeds per-conn cap of M bytes`,
    /// renderable verbatim by the operator client.
    ResponseTooBig,
    /// Request line failed JSON parse, or carries an unknown `op`.
    Malformed,
    /// `Subscribe` invoked on a conn that already flipped to
    /// subscriber role. Subscribe is one-shot per conn — a repeat
    /// call would silently overwrite the prior `name` filter and
    /// drop any pending back-pressure accounting. The handler refuses
    /// with this structured error so the operator sees a deterministic
    /// failure instead of an invisible state mutation.
    ///
    /// The wire vocabulary is pinned by
    /// [`tests::wire_error_code_round_trips_every_variant`]; the
    /// handler-side gate that reaches this variant lives on
    /// [`crate::driver::EngineDriver`]'s Subscribe arm.
    AlreadySubscribed,
    /// Daemon is winding down — mutating verbs (`Reload`, `Disable`,
    /// `Enable`) refused. The gate fires iff
    /// `EngineDriver::first_term.is_some()` — the operator pulsed
    /// SIGINT / SIGTERM and the actuator is in the middle of its
    /// SIGTERM → grace → SIGKILL → reap-drain pipeline. Read-only
    /// verbs (`Status`, `List`, `Show`, `Subscribe`) continue
    /// working so operators can observe in-flight shutdown via
    /// `specter tail`; the engine still emits diagnostics during
    /// shutdown (late effect completions, post-fire rebase ceilings,
    /// etc.).
    ///
    /// Operator-script branch: a `code == "shutting_down"` Err
    /// surfaces "daemon is going away; don't retry against this PID";
    /// the operator can re-issue the verb against the next boot.
    /// Subscribe stays accessible because its mutation is
    /// bin-local — flipping `conn.role` doesn't touch engine state
    /// and is safe to enact during shutdown.
    ShuttingDown,
}

impl WireErrorCode {
    /// Wire-form token for this variant — the same `code` field value
    /// the JSON shape carries. Mirrors the
    /// `#[serde(rename_all = "snake_case")]` projection exactly.
    ///
    /// Exhaustive `match` — a new variant without a paired arm fails
    /// to compile, keeping the wire vocabulary single-source against
    /// [`tests::wire_error_code_round_trips_every_variant`]'s
    /// JSON-form pin.
    ///
    /// Takes `self` by value: the enum is `Copy` and a single byte
    /// wide, so the by-value form is a register-passing convention
    /// rather than the indirection a `&self` borrow would imply.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::UnknownSub => "unknown_sub",
            Self::DynamicSubNoOp => "dynamic_sub_no_op",
            Self::NotDisabled => "not_disabled",
            Self::TomlDisabled => "toml_disabled",
            Self::Busy => "busy",
            Self::ResponseTooBig => "response_too_big",
            Self::Malformed => "malformed",
            Self::AlreadySubscribed => "already_subscribed",
            Self::ShuttingDown => "shutting_down",
        }
    }
}

impl std::fmt::Display for WireErrorCode {
    /// Operator-visible rendering — writes the snake_case wire token
    /// verbatim via [`Self::as_str`]. The client's
    /// `eprintln!("specter <verb>: {code}: {error}")` reaches here,
    /// so the human view and the JSON form share one source of truth.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
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
    /// Most-recent reload — wall-clock + attribution as one
    /// observable. `None` before the first one fires. Flattens on
    /// the wire so the JSON form carries `last_reload_at` and
    /// `last_reload_via` directly alongside the other fields on the
    /// `Some` side; both keys are omitted entirely when `None`.
    /// Mirrors the source-side `Option<crate::driver::state::LastReload>`.
    #[serde(flatten, default, skip_serializing_if = "Option::is_none")]
    pub(crate) last_reload: Option<WireLastReload>,
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
    /// mode would widen this to `Option<WirePath>` honestly.
    pub(crate) config_path: WirePath,
    /// UNIX-socket path the IPC server is bound to.
    pub(crate) socket_path: WirePath,
}

/// Most-recent successful reload as one wire observable — wall-clock
/// at the reload-record call site plus the attribution enum that
/// drove it. Mirrors [`crate::driver::state::LastReload`] on the wire; the
/// JSON form stays flat (the two inner fields inline directly
/// alongside [`StatusResponse`]'s siblings via `#[serde(flatten)]`)
/// so operator scripts that parsed `status.last_reload_at` /
/// `status.last_reload_via` continue to read the same byte shape on
/// the `Some` side. The `None` side omits both keys entirely — the
/// type-driven encoding of "never reloaded yet."
///
/// Field renames keep the Rust ergonomics (`lr.at`, `lr.via`)
/// without touching the JSON contract — see [`Self::at`] and
/// [`Self::via`].
#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
pub(crate) struct WireLastReload {
    /// Wall-clock of the most recent successful reload. Renamed to
    /// `last_reload_at` on the wire so the JSON form lines up with
    /// the pre-lift shape on the `Some` side.
    #[serde(rename = "last_reload_at")]
    pub(crate) at: WireTime,
    /// Trigger of the most recent successful reload. Renamed to
    /// `last_reload_via` on the wire so the JSON form lines up with
    /// the pre-lift shape on the `Some` side. Typed enum (not
    /// `&'static str`) so a future variant on
    /// [`super::wire::WireReloadTrigger`] is a compile error at the
    /// declaration site, in the same shape as the rest of the wire
    /// projection layer.
    #[serde(rename = "last_reload_via")]
    pub(crate) via: WireReloadTrigger,
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
    pub(crate) anchor: Option<WirePath>,
    /// Wall-clock projection of `Sub.last_fired_at`. `None` for
    /// never-fired Subs and non-attached rows.
    pub(crate) last_fired_at: Option<WireTime>,
    /// `Sub.fire_count`. `None` for non-attached rows.
    pub(crate) fire_count: Option<u64>,
    /// `Sub.dedup_suppressed_count`. `None` for non-attached rows.
    pub(crate) dedup_suppressed_count: Option<u64>,
    /// `Sub.settle.as_millis()` — the per-Sub debounce floor. Distinct
    /// from `Profile.settle` (engine-recomputed as
    /// `min(remaining_subs.settles)` across the Profile's attached
    /// Subs); the wire field is the per-Sub value the operator
    /// declared, not the Profile-level fold. `None` for non-attached
    /// rows.
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

/// [`WireRequest`] is structurally infallible to serialize: every
/// variant carries only plain-data fields ([`CompactString`] / unit),
/// every derived `Serialize` walks through `serialize_str` /
/// `serialize_unit` against in-memory values, and no field uses a
/// `serialize_with` adapter that could fail. Marks the client-side
/// request path
/// ([`crate::ipc::client::connect::write_request`]) safe for the
/// `Vec<u8>`-returning wrapper without an `.expect`-at-a-distance.
impl InfallibleSerialize for WireRequest {}

/// [`ResponsePayload`] is structurally infallible to serialize: the
/// variant payloads are plain-data structs ([`StatusResponse`],
/// [`ListResponse`], [`ShowResponse`], [`SubDetails`]) whose fields
/// are primitives, [`String`], [`Option`], [`Vec`], [`WireId`],
/// [`WireTime`] (manual `serialize_str` over an invariant-by-
/// construction RFC-3339 token), [`WirePath`] (transparent `String`),
/// or other `Wire*` enums (closed-set derives). Marks the daemon-side
/// response paths
/// ([`crate::driver::Hub::enqueue_response`] +
/// [`crate::driver::Hub::drain_accept`]'s cap-arm
/// best-effort Busy write) safe for the wrapper.
impl InfallibleSerialize for ResponsePayload {}

/// `specter show <name>` detail block for an attached Sub.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub(crate) struct SubDetails {
    /// Operator-facing name.
    pub(crate) name: String,
    /// `SubId` projection.
    pub(crate) sub: WireId,
    /// Hosting `ProfileId` projection.
    pub(crate) profile: WireId,
    /// Operator-display state (`StateLabel`). `None` signals the
    /// same engine-invariant breach [`ListRow::state`] does: the
    /// attached Sub's Profile is gone. Symmetric across
    /// `show -o json` and `list -o json` — operators decode the
    /// breach identically on both verbs, and the IPC layer projects
    /// the missing Profile as `None` rather than panicking the
    /// daemon out from under every concurrent operator session.
    /// The breach loudness lives engine-side (the `ProbeSlot`
    /// tripwire, the registries' `debug_assert!`), not in the
    /// projection layer.
    pub(crate) state: Option<WireStateLabel>,
    /// Anchor path (`engine.tree().path_of(profile.resource)`). `None`
    /// signals "anchor vanished / not yet resolved" — symmetric with
    /// [`ListRow::anchor`], so operators reading `list -o json` and
    /// `show -o json` decode vanish identically (`null`).
    pub(crate) anchor: Option<WirePath>,
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

#[cfg(test)]
mod tests {
    use super::{
        DisabledSource, ResponsePayload, StatusResponse, WireErrorCode, WireId, WireLastReload,
        WireRequest,
    };
    use crate::ipc::framing::parse_strict;
    use crate::ipc::wire::{WirePath, WireReloadTrigger, WireTime};
    use slotmap::KeyData;
    use specter_core::{ProfileId, PromoterId, ResourceId, SubId};
    use std::path::Path;
    use std::time::{Duration, UNIX_EPOCH};

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

    /// `parse_strict` rejects a typoed payload field on a struct
    /// variant — the bug class the boundary gate exists to prevent.
    /// `{"op":"subscribe","names":"build"}` (note the trailing `s`)
    /// would silently parse with `name = None` under bare
    /// `serde_json::from_slice` and the daemon would subscribe
    /// unfiltered, ignoring the operator's intent. With the
    /// boundary gate the typo surfaces as an unknown-field rejection
    /// the [`crate::driver::ipc`] handler turns into
    /// [`WireErrorCode::Malformed`].
    ///
    /// Bare `from_slice` is exercised first to pin the bug shape
    /// (silent accept) the boundary gate fixes; the strict parse
    /// is the structural floor.
    #[test]
    fn parse_strict_rejects_typoed_payload_field_on_wire_request() {
        let bytes = br#"{"op":"subscribe","names":"build"}"#;
        // Baseline: bare serde silently accepts the typo, dropping `names`.
        let lenient: WireRequest = serde_json::from_slice(bytes).expect("bare parse succeeds");
        match lenient {
            WireRequest::Subscribe { name } => assert_eq!(
                name, None,
                "bare parse silently drops the typoed `names` key",
            ),
            other => panic!("expected Subscribe, got {other:?}"),
        }
        // Boundary: strict parse rejects.
        let err = parse_strict::<WireRequest>(bytes).expect_err("strict parse rejects typo");
        assert!(
            err.to_string().contains("unknown field `names`"),
            "rejection names the typoed field; got {err}",
        );
    }

    /// `parse_strict` rejects an unknown top-level field on a
    /// [`ResponsePayload`] — the daemon-bug class the client-side
    /// boundary gate ([`crate::ipc::client::connect::read_response`])
    /// surfaces as [`std::io::ErrorKind::InvalidData`] rather than
    /// silently dropping the value. Symmetric protection with the
    /// daemon-side request gate.
    #[test]
    fn parse_strict_rejects_unknown_field_on_response_payload() {
        let bytes = br#"{"kind":"ok","stray":1}"#;
        let err = parse_strict::<ResponsePayload>(bytes).expect_err("strict parse rejects unknown");
        assert!(
            err.to_string().contains("unknown field `stray`"),
            "rejection names the unknown field; got {err}",
        );
    }

    /// `ResponsePayload`'s internal tag is the load-bearing wire
    /// commitment: every variant flattens into one JSON object
    /// keyed by `kind`. A retrofit to external tagging would change
    /// the operator-visible shape and fail this test.
    ///
    /// The Err arm exercises one [`WireErrorCode`] variant inline so
    /// the test's reach across [`ResponsePayload`] stays intact;
    /// every-variant coverage of the error-code vocabulary lives in
    /// [`wire_error_code_round_trips_every_variant`].
    #[test]
    fn response_payload_round_trips_internal_tag() {
        let ok = serde_json::to_string(&ResponsePayload::Ok).unwrap();
        assert_eq!(ok, r#"{"kind":"ok"}"#);

        let err = serde_json::to_string(&ResponsePayload::Err {
            code: WireErrorCode::UnknownSub,
            error: "no watch named foo".into(),
        })
        .unwrap();
        assert_eq!(
            err,
            r#"{"kind":"err","code":"unknown_sub","error":"no watch named foo"}"#
        );

        let round_trip: ResponsePayload = serde_json::from_str(&err).unwrap();
        match round_trip {
            ResponsePayload::Err { code, error } => {
                assert_eq!(code, WireErrorCode::UnknownSub);
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

    /// Every [`WireErrorCode`] variant projects to its
    /// snake_case wire token and round-trips identically through
    /// serde + [`WireErrorCode::as_str`] + [`std::fmt::Display`].
    ///
    /// One iteration pins three surfaces in lockstep:
    /// the JSON byte form clients parse, the `as_str()` table the
    /// daemon's Display reaches, and the renderer-visible
    /// `"specter <verb>: {code}: ..."` line. A hand-edit to any
    /// variant — adding one, renaming a tag, drifting the as_str
    /// arm — fails here loudly, replacing the per-variant copy-paste
    /// pin the prior `ERR_*` constants used to require.
    #[test]
    fn wire_error_code_round_trips_every_variant() {
        // Compile-time exhaustive — a new variant without an entry
        // here is a missing-arm match below.
        const ALL: &[WireErrorCode] = &[
            WireErrorCode::UnknownSub,
            WireErrorCode::DynamicSubNoOp,
            WireErrorCode::NotDisabled,
            WireErrorCode::TomlDisabled,
            WireErrorCode::Busy,
            WireErrorCode::ResponseTooBig,
            WireErrorCode::Malformed,
            WireErrorCode::AlreadySubscribed,
            WireErrorCode::ShuttingDown,
        ];
        for &code in ALL {
            let json = serde_json::to_string(&code).expect("serialize");
            let stripped = json
                .strip_prefix('"')
                .and_then(|s| s.strip_suffix('"'))
                .expect("JSON form is a bare quoted token");
            assert_eq!(
                stripped,
                code.as_str(),
                "JSON form ({json}) must equal as_str() ({}) for {code:?}",
                code.as_str(),
            );
            assert_eq!(
                code.to_string(),
                code.as_str(),
                "Display must write as_str() for {code:?}",
            );
            let round_trip: WireErrorCode =
                serde_json::from_str(&json).expect("deserialize wire form");
            assert_eq!(round_trip, code);
        }

        // Spot-check the embedded shape: a `code` field deserialized
        // from a daemon-emitted Err line yields the matching variant.
        let err: ResponsePayload = serde_json::from_str(
            r#"{"kind":"err","code":"already_subscribed","error":"conn already in subscribe mode"}"#,
        )
        .unwrap();
        match err {
            ResponsePayload::Err { code, error } => {
                assert_eq!(code, WireErrorCode::AlreadySubscribed);
                assert_eq!(error, "conn already in subscribe mode");
            }
            other => panic!("expected Err(AlreadySubscribed), got {other:?}"),
        }
    }

    /// `StatusResponse.last_reload` flattens onto the wire — `Some`
    /// inlines `last_reload_at` and `last_reload_via` next to the
    /// peer fields; `None` omits both keys entirely. No wrapper key
    /// (`last_reload`) ever appears in the JSON form, on either
    /// side. The wire shape is the lift's load-bearing contract:
    /// operator scripts that read `status.last_reload_at` /
    /// `status.last_reload_via` keep working unchanged on the
    /// `Some` side.
    #[test]
    fn status_response_flattens_last_reload_pair() {
        let with = sample_status(Some(WireLastReload {
            at: WireTime::from(UNIX_EPOCH + Duration::from_mins(2)),
            via: WireReloadTrigger::Sighup,
        }));
        let json = serde_json::to_value(&with).expect("serialize");
        assert!(
            json.get("last_reload_at").is_some(),
            "Some inlines last_reload_at: {json}",
        );
        assert!(
            json.get("last_reload_via").is_some(),
            "Some inlines last_reload_via: {json}",
        );
        assert!(
            json.get("last_reload").is_none(),
            "no wrapper key appears: {json}",
        );

        let without = sample_status(None);
        let json = serde_json::to_value(&without).expect("serialize");
        assert!(
            json.get("last_reload_at").is_none(),
            "None omits last_reload_at entirely: {json}",
        );
        assert!(
            json.get("last_reload_via").is_none(),
            "None omits last_reload_via entirely: {json}",
        );
    }

    /// `StatusResponse` round-trips through serde with the flattened
    /// `last_reload` pair preserved on the `Some` side. The
    /// serialize-deserialize chain reaches the wire field renames
    /// (`at` ↔ `last_reload_at`, `via` ↔ `last_reload_via`) in
    /// lockstep — a rename drift on either side fails this test
    /// rather than silently dropping the value.
    #[test]
    fn status_response_round_trips_last_reload_pair() {
        let original = sample_status(Some(WireLastReload {
            at: WireTime::from(UNIX_EPOCH + Duration::from_mins(1)),
            via: WireReloadTrigger::Ipc,
        }));
        let json = serde_json::to_string(&original).expect("serialize");
        let parsed: StatusResponse = serde_json::from_str(&json).expect("deserialize");
        let lr = parsed
            .last_reload
            .as_ref()
            .expect("Some on the round-trip side too");
        assert_eq!(lr.via, WireReloadTrigger::Ipc);
        assert_eq!(lr, original.last_reload.as_ref().expect("Some originally"));
    }

    /// A partial wire shape — one of the two `last_reload_*` keys
    /// present, the other absent — fails the strict boundary parse.
    /// The structural witness operates in two layers:
    ///
    /// 1. Bare [`serde_json::from_slice`] over the flattened
    ///    `Option<WireLastReload>` silently collapses partial input
    ///    to `None`; serde's `flatten + default` machinery does not
    ///    surface the missing-sibling as a parse error.
    /// 2. [`parse_strict`] — the boundary every client response read
    ///    routes through ([`crate::ipc::client::connect::read_response`])
    ///    — catches the partial via its round-trip walk: the parsed
    ///    `StatusResponse` round-trips with the orphan key omitted,
    ///    so the original's `last_reload_at` (or `last_reload_via`)
    ///    is reported as `unknown field` at the boundary.
    ///
    /// The wire-side invariant therefore holds at the operator-facing
    /// surface even though the inner deserialize tolerates partial
    /// input — pinning both behaviors keeps the test honest about
    /// where the structural gate actually lives.
    #[test]
    fn status_response_partial_last_reload_shape_caught_at_boundary() {
        let at_only = partial_status_bytes("last_reload_via");
        let via_only = partial_status_bytes("last_reload_at");

        // Layer 1 — bare serde tolerates the partial shape by
        // collapsing to `None`. The behavior is a serde-flatten
        // property, not a load-bearing wire invariant.
        let parsed: StatusResponse =
            serde_json::from_slice(&at_only).expect("bare serde tolerates partial flatten");
        assert!(
            parsed.last_reload.is_none(),
            "bare serde collapses partial flatten to None",
        );

        // Layer 2 — `parse_strict` catches it. The boundary gate
        // every client response read routes through reports the
        // orphan key as `unknown field` because the parsed
        // `StatusResponse` round-trips without it.
        let err = parse_strict::<StatusResponse>(&at_only)
            .expect_err("parse_strict catches partial last_reload_at");
        assert!(
            err.to_string().contains("last_reload_at"),
            "rejection names the orphan key; got {err}",
        );
        let err = parse_strict::<StatusResponse>(&via_only)
            .expect_err("parse_strict catches partial last_reload_via");
        assert!(
            err.to_string().contains("last_reload_via"),
            "rejection names the orphan key; got {err}",
        );
    }

    /// Construct a [`StatusResponse`] with constant scaffolding so the
    /// flatten / round-trip / partial-shape tests focus on
    /// `last_reload`. Defaults are minimal — the surrounding fields
    /// carry no semantic load for these tests.
    fn sample_status(last_reload: Option<WireLastReload>) -> StatusResponse {
        StatusResponse {
            uptime_secs: 0,
            start_wall: WireTime::from(UNIX_EPOCH),
            reload_count: 0,
            last_reload,
            sub_total: 0,
            sub_disabled_toml: 0,
            sub_disabled_runtime: 0,
            profile_active: 0,
            promoter_active: 0,
            config_path: WirePath::from(Path::new("/etc/specter.toml")),
            socket_path: WirePath::from(Path::new("/tmp/specter.sock")),
        }
    }

    /// Serialize a fully-populated [`StatusResponse`] and drop
    /// `drop_key` from the JSON object, yielding the partial wire
    /// shape the strict boundary test asserts against. Building the
    /// bytes through serde rather than hand-typing the JSON keeps
    /// the test resilient to a future field rename / reorder on
    /// `StatusResponse`.
    fn partial_status_bytes(drop_key: &str) -> Vec<u8> {
        let complete = sample_status(Some(WireLastReload {
            at: WireTime::from(UNIX_EPOCH + Duration::from_mins(2)),
            via: WireReloadTrigger::Sighup,
        }));
        let mut value = serde_json::to_value(&complete).expect("serialize");
        value
            .as_object_mut()
            .expect("StatusResponse serializes as a JSON object")
            .remove(drop_key)
            .unwrap_or_else(|| panic!("complete sample_status must contain {drop_key}"));
        serde_json::to_vec(&value).expect("re-serialize")
    }
}
