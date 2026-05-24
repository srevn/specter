//! Wire-line framing — serialization helpers for the operator IPC
//! protocol's LF-delimited JSON object lines.
//!
//! # Framing contract
//!
//! One JSON object per line, terminated by `\n`. Every send path on
//! both client and server converges on this shape: the daemon's
//! response writer, the diag fan-out, the back-pressure marker
//! flush, the structured-busy reply on the accept cap, the client's
//! request shipping, and the operator-side `tail -o json` echo. This
//! module owns the "build the wire-ready bytes" step so the framing
//! discipline is single-source — a future change to the protocol
//! envelope (length-prefix, batching, etc.) lands here once.
//!
//! # Visibility
//!
//! `pub(crate)` — both the server-side mio reactor
//! ([`crate::driver::hub::DriverHub`]) and the client-side verb
//! handlers ([`super::client`]) consume [`serialize_line`].

use std::io;

/// Serialize `value` as JSON and append a trailing `\n` so the
/// bytes are wire-ready for the operator IPC protocol's
/// LF-delimited framing.
///
/// # Errors
///
/// Returns [`io::ErrorKind::InvalidData`] wrapping the underlying
/// [`serde_json::Error`] when serialization fails. The daemon's
/// wire types ([`super::protocol::WireRequest`],
/// [`super::protocol::ResponsePayload`],
/// [`super::wire::WireDiagnostic`]) are `Serialize`-derive over
/// plain-data fields — their serialization is structurally
/// infallible, and reaching the error path indicates a
/// programmer-error class regression worth surfacing through the
/// `io` chain rather than silently logging. Call sites with those
/// infallible carriers convert the result via `.expect("infallible
/// by construction")`; call sites that accept a generic
/// `T: Serialize` (the client's
/// [`super::client::connect::write_request`], the daemon's
/// [`crate::driver::hub::DriverHub::enqueue_response`]) propagate
/// via `?`.
pub(crate) fn serialize_line<T: serde::Serialize>(value: &T) -> io::Result<Vec<u8>> {
    let mut bytes =
        serde_json::to_vec(value).map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
    bytes.push(b'\n');
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::serialize_line;
    use serde::{Serialize, Serializer, ser::Error};
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
}
