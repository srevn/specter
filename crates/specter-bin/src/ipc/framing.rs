//! Wire-line framing — envelope constants, serialization, and
//! strict-parse helpers for the operator IPC protocol's LF-delimited
//! JSON object lines.
//!
//! # Framing contract
//!
//! One JSON object per line, terminated by `\n`. Per-line length cap
//! is [`MAX_LINE_BYTES`] — a line past that is structurally hostile
//! and the reader path terminates the conn. Every send path on both
//! client and server converges on this shape: the daemon's response
//! writer, the diag fan-out, the back-pressure marker flush, the
//! structured-busy reply on the accept cap, the client's request
//! shipping, and the operator-side `tail -o json` echo. This module
//! owns the envelope contract — the cap, the "build the wire-ready
//! bytes" step, and the strict-parse gate on incoming object lines —
//! so a future change to the framing discipline (length-prefix,
//! batching, larger cap, etc.) lands here once.
//!
//! # Boundary strictness
//!
//! The request/response surface admits no unknown fields:
//! [`parse_strict`] round-trip-validates every incoming JSON object
//! against its own re-serialized form so a typoed operator JSON
//! (`{"op":"subscribe","names":"build"}`) or a daemon-bug response
//! carrying a stale field name reaches the caller as a parse error
//! rather than silently dropping the value. The discipline lives
//! here because internally-tagged enums ([`super::protocol::WireRequest`],
//! [`super::protocol::ResponsePayload`], [`super::protocol::ShowResponse`])
//! cannot use serde derive's `deny_unknown_fields` — a long-standing
//! serde-derive gap — so the gate is implemented at the wire boundary
//! instead. Streamed [`super::wire::WireDiagnostic`] lines are
//! deliberately exempt: an older `tail`/`wait` client reading from a
//! newer daemon stays forward-compatible.
//!
//! # Infallible serialization discipline
//!
//! Every production send path threads a wire type whose `Serialize`
//! impl cannot fail in practice — the carriers are derive-Serialize
//! over plain-data fields, with no custom adapters that could panic
//! or error mid-emit. [`InfallibleSerialize`] is the marker trait
//! that captures the property at the type level;
//! [`encode_line`] is the `Vec<u8>`-returning wrapper
//! every production caller routes through. The fallible primitive
//! [`serialize_line`] is module-private, so a future caller cannot
//! reach the wire bytes without either marking their type
//! `InfallibleSerialize` or lifting the primitive's visibility — both
//! explicit, reviewed moves rather than a silent `.expect`-at-a-
//! distance scattered across the binary.
//!
//! # Visibility
//!
//! `pub(crate)` on the wrappers and constants; the fallible primitive
//! stays private. Both the server-side mio reactor
//! ([`crate::driver::Hub`]) and the client-side verb
//! handlers ([`super::client`]) consume
//! [`encode_line`] / [`parse_strict`] and read
//! [`MAX_LINE_BYTES`] for envelope enforcement.

use std::io;

/// Per-line byte cap. A line past this is structurally hostile —
/// operator IPC verbs are well under 1 KiB (the largest verb today,
/// `Subscribe`, is ~60 bytes), so 256 KiB is 256× headroom against
/// any legitimate use and the conn gets terminated rather than
/// allowed to monopolise the driver thread on a malformed line.
///
/// The cap lives here so envelope enforcement is single-source:
/// [`crate::driver::Hub::read_conn_into_lines`] checks it
/// on incoming bytes, and the per-conn write-queue high-water mark
/// in `crate::driver::ipc::conns` is set to this value so an oversize
/// response has the same backpressure footprint as a hostile read.
pub(crate) const MAX_LINE_BYTES: usize = 256 * 1024;

/// Serialize `value` as JSON and append a trailing `\n` so the
/// bytes are wire-ready for the operator IPC protocol's
/// LF-delimited framing.
///
/// Module-private — production callers route through
/// [`encode_line`], which carries the
/// [`InfallibleSerialize`] bound that structurally rules out the
/// error path for every wire type. Keeping this primitive private
/// closes the surface so a future caller cannot reach the fallible
/// `Vec<u8>` path without an explicit `pub(crate)` lift; the tests
/// in this module reach it directly to exercise both the happy and
/// error arms in one place.
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidData`] wrapping the underlying
/// [`serde_json::Error`] when serialization fails. Every production
/// caller marks the value type [`InfallibleSerialize`], so the error
/// arm is unreachable on the production wire path; the kind is
/// pinned for the in-module tests and any future caller that opts
/// back into the fallible primitive.
fn serialize_line<T: serde::Serialize>(value: &T) -> io::Result<Vec<u8>> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    bytes.push(b'\n');
    Ok(bytes)
}

/// Marker trait declaring that `T`'s [`serde::Serialize`] impl cannot
/// fail in practice — every field is plain data, every custom
/// `Serialize` is a `serialize_str` over an invariant-by-construction
/// `&str`, and no nested type carries an adapter that can return
/// `Err` from inside `serialize`.
///
/// Implementing this trait is a *promise* the implementer makes
/// after auditing the type's full serialize tree. The promise is
/// structurally checked at the call site (the bound on
/// [`encode_line`]) but the body of the promise is a
/// human judgement, not a compiler verification. The cost of a
/// wrong impl is a panic at runtime, not memory unsafety, so the
/// trait is safe rather than `unsafe`.
///
/// Implemented for the three wire surfaces that production code
/// sends through the boundary: [`super::wire::WireDiagnostic`] (diag
/// fan-out + client `tail -o json` re-emit),
/// [`super::protocol::ResponsePayload`] (daemon response enqueue +
/// over-cap busy reply), and [`super::protocol::WireRequest`]
/// (client request shipping). Each impl is co-located with its type
/// so the audit lives next to the structural floor it certifies;
/// adding a new wire type that needs to reach
/// [`encode_line`] is a paired edit (declare it, impl
/// the marker) so a forgotten impl becomes a compile error at the
/// call site, not a silent fallthrough to the fallible primitive.
pub(crate) trait InfallibleSerialize: serde::Serialize {}

/// Serialize `value` as a wire-ready LF-terminated JSON line.
///
/// The [`InfallibleSerialize`] bound asserts the type's
/// [`serde::Serialize`] cannot fail; the body invokes
/// [`serialize_line`] and unwraps with [`Result::expect`]. The
/// `.expect` message names the contract the impl carries so a
/// future regression — a wire field gaining a custom `Serialize`
/// adapter that *can* fail — surfaces as a load-bearing panic
/// pointing at the broken promise rather than a silent miss
/// scattered across four call sites.
///
/// Production wire path: diag fan-out
/// ([`crate::driver::Hub::dispatch_to_subscribers`]),
/// back-pressure `_missed` marker
/// (`crate::driver::ipc::conns::ConnState::try_dispatch_diag`), response
/// enqueue ([`crate::driver::Hub::enqueue_response`] and
/// [`crate::driver::Hub::drain_accept`]'s cap-arm
/// best-effort Busy write), client request shipping
/// ([`super::client::connect::write_request`]), and client `tail -o json`
/// re-emit ([`super::client::tail`]).
pub(crate) fn encode_line<T: InfallibleSerialize>(value: &T) -> Vec<u8> {
    serialize_line(value).expect(
        "InfallibleSerialize contract: wire-type serialization cannot fail; \
         a fallible `serialize_with` adapter would violate the marker impl",
    )
}

/// Parse `bytes` as JSON into `T`, rejecting object keys that `T`'s
/// derived [`serde::Deserialize`] would silently ignore.
///
/// # Mechanism
///
/// 1. Parse `bytes` to a [`serde_json::Value`] (admits any well-
///    formed JSON).
/// 2. Parse the same value to `T` via [`serde_json::from_value`] —
///    surfaces missing required fields, type mismatches, and unknown
///    variant tags through `T`'s normal deserialize gate.
/// 3. Re-serialize the parsed `T` back to a `serde_json::Value`. The
///    derived [`serde::Serialize`] emits exactly the keys `T`
///    recognises.
/// 4. Recursively walk the original value against the round-trip:
///    every key in the original must also exist in the round-trip.
///    A key in the round-trip but missing from the original is
///    benign (an `Option<_>::None` field that serialized to `null`
///    versus an omitted field both deserialize identically), so the
///    asymmetric walk admits the abbreviated form without false-
///    flagging it.
///
/// # Why a round-trip helper rather than `#[serde(deny_unknown_fields)]`
///
/// Serde derive's `deny_unknown_fields` is incompatible with
/// internally tagged enums (`#[serde(tag = "...")]`): the attribute
/// either fails to generate the [`Deserialize`](serde::Deserialize)
/// impl entirely (when placed on the enum) or treats the tag key
/// itself as unknown on the inner variant payload's struct (when
/// placed on the struct). The wire surface's three request/response
/// carriers are all internally tagged, and the embedded structs
/// ([`super::protocol::StatusResponse`], etc.) are reached as newtype
/// variant payloads — so neither the enum nor its inner struct can
/// carry the attribute. Round-trip validation sidesteps the derive
/// gap by treating the derived `Serialize` as the schema, with no
/// per-type attrs and no hand-maintained field whitelist.
///
/// # Cost
///
/// Three serde passes per line (Value, T, Value-again). Operator
/// IPC verbs are sub-KB and not throughput-sensitive; the extra
/// cost is invisible to operators.
///
/// # Errors
///
/// - First `from_slice` failure: malformed JSON.
/// - `from_value` failure: missing fields, type mismatch, unknown
///   variant tag.
/// - Round-trip walk failure: unknown key in the original object
///   not produced by `T`'s `Serialize` — surfaced as a
///   [`serde::de::Error::custom`] message naming the unknown field
///   and its path within the value (e.g.
///   `unknown field \`names\` at \`\``,
///   `unknown field \`extra\` at \`.rows\[0\]\``).
pub(crate) fn parse_strict<T>(bytes: &[u8]) -> Result<T, serde_json::Error>
where
    T: serde::de::DeserializeOwned + serde::Serialize,
{
    use serde::de::Error as _;

    let original: serde_json::Value = serde_json::from_slice(bytes)?;
    let parsed: T = serde_json::from_value(original.clone())?;
    let round_trip: serde_json::Value =
        serde_json::to_value(&parsed).map_err(serde_json::Error::custom)?;
    let mut path = String::new();
    reject_unknown_keys(&original, &round_trip, &mut path).map_err(serde_json::Error::custom)?;
    Ok(parsed)
}

/// Recursive walk for [`parse_strict`]: every object key in `orig`
/// must exist in `rt`. Returns the first violation; the error
/// message includes the path within the value (`""` at the root,
/// `.rows[0].name` at a nested array element).
///
/// The walk descends into matched object keys and matched array
/// element pairs. Mismatched shapes (`Object` vs `Array`, or scalar
/// vs collection) are not reachable in production — `parsed`'s
/// re-serialized form mirrors the original's shape modulo
/// `Option::None` differences — but the no-op fallback arm keeps
/// the helper total.
fn reject_unknown_keys(
    orig: &serde_json::Value,
    rt: &serde_json::Value,
    path: &mut String,
) -> Result<(), String> {
    use serde_json::Value;
    use std::fmt::Write as _;

    match (orig, rt) {
        (Value::Object(o), Value::Object(r)) => {
            for (key, orig_v) in o {
                let Some(rt_v) = r.get(key) else {
                    return Err(format!("unknown field `{key}` at `{path}`"));
                };
                let saved = path.len();
                let _ = write!(path, ".{key}");
                reject_unknown_keys(orig_v, rt_v, path)?;
                path.truncate(saved);
            }
        }
        (Value::Array(o), Value::Array(r)) => {
            for (i, (orig_v, rt_v)) in o.iter().zip(r.iter()).enumerate() {
                let saved = path.len();
                let _ = write!(path, "[{i}]");
                reject_unknown_keys(orig_v, rt_v, path)?;
                path.truncate(saved);
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{parse_strict, serialize_line};
    use serde::{Deserialize, Serialize, Serializer, ser::Error};
    use std::io::ErrorKind;

    /// Happy path: a `T: Serialize` carrier renders as `<json>\n` —
    /// the trailing LF is the load-bearing framing delimiter, not
    /// part of the JSON object itself. Any future change that
    /// dropped the LF would break every line-oriented reader on the
    /// wire (the daemon's `read_conn_into_lines` LF-splitter, the
    /// client's `BufReader::read_line` ack drain).
    #[test]
    fn appends_trailing_newline_to_serialized_bytes() {
        #[derive(Serialize)]
        struct Carrier {
            x: u32,
            name: &'static str,
        }
        let line = serialize_line(&Carrier { x: 7, name: "go" }).unwrap();
        assert_eq!(&line[..], b"{\"x\":7,\"name\":\"go\"}\n");
    }

    /// A `Serialize` impl that errors maps to
    /// [`ErrorKind::InvalidData`]. The kind is the wire-uniform
    /// signal call sites branch on; mapping every serialize failure
    /// to the same kind keeps the operator-visible error surface
    /// consistent across send paths. Pinning the kind here defends
    /// against a future refactor that propagated the raw
    /// [`serde_json::Error`] (its `io::Error::other`-style fallback
    /// would surface as `ErrorKind::Other`, breaking the contract
    /// the callers depend on).
    #[test]
    fn maps_serde_error_to_invalid_data_kind() {
        struct AlwaysFails;
        impl Serialize for AlwaysFails {
            fn serialize<S: Serializer>(&self, _ser: S) -> Result<S::Ok, S::Error> {
                Err(S::Error::custom("intentional test-driven failure"))
            }
        }
        let err = serialize_line(&AlwaysFails).expect_err("Serialize impl errors");
        assert_eq!(err.kind(), ErrorKind::InvalidData);
    }

    /// `parse_strict` accepts a well-formed object whose every key is
    /// recognised by the target's derive — the happy path, identical
    /// in outcome to a bare `serde_json::from_slice`.
    #[test]
    fn parse_strict_accepts_well_formed_object() {
        #[derive(Deserialize, Serialize)]
        struct Carrier {
            x: u32,
            name: String,
        }
        let parsed: Carrier = parse_strict(br#"{"x":7,"name":"go"}"#).expect("well-formed");
        assert_eq!(parsed.x, 7);
        assert_eq!(parsed.name, "go");
    }

    /// `parse_strict` rejects an extra top-level key — the structural
    /// claim the helper enforces against typoed operator JSON and
    /// daemon-bug responses. The error message names the unknown
    /// key so the rejection is operator-actionable.
    #[test]
    fn parse_strict_rejects_extra_top_level_key() {
        #[derive(Debug, Deserialize, Serialize)]
        struct Carrier {
            x: u32,
        }
        let err = parse_strict::<Carrier>(br#"{"x":7,"y":8}"#).expect_err("extra key");
        assert!(
            err.to_string().contains("unknown field `y`"),
            "error names the unknown key; got {err}",
        );
    }

    /// `parse_strict` rejects an extra key nested inside an array
    /// element — the recursive walk catches drift at any depth, so
    /// the [`super::super::protocol::ListResponse::rows`] / per-row
    /// nesting is covered without per-type attrs on each
    /// [`super::super::protocol::ListRow`].
    #[test]
    fn parse_strict_rejects_extra_key_inside_array_element() {
        #[derive(Debug, Deserialize, Serialize)]
        struct Row {
            name: String,
        }
        #[derive(Debug, Deserialize, Serialize)]
        struct Wrap {
            rows: Vec<Row>,
        }
        let err = parse_strict::<Wrap>(br#"{"rows":[{"name":"a","extra":1}]}"#)
            .expect_err("extra key in row");
        assert!(
            err.to_string().contains("unknown field `extra`"),
            "error names the unknown key; got {err}",
        );
        assert!(
            err.to_string().contains(".rows[0]"),
            "error path locates the nested element; got {err}",
        );
    }

    /// An `Option<T>::None` field that the wire omits (rather than
    /// emitting as `null`) deserializes cleanly through `parse_strict`
    /// — the round-trip walk only flags keys IN the original NOT in
    /// the re-serialized form. A round-trip key absent from the
    /// original is benign, so the abbreviated form is admitted.
    #[test]
    fn parse_strict_admits_omitted_option_field() {
        #[derive(Deserialize, Serialize)]
        struct Carrier {
            x: u32,
            #[serde(default)]
            opt: Option<u32>,
        }
        let parsed: Carrier = parse_strict(br#"{"x":7}"#).expect("omitted optional");
        assert_eq!(parsed.x, 7);
        assert_eq!(parsed.opt, None);
    }
}
