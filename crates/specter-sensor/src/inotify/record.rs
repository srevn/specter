//! `inotify_event` record parser.
//!
//! Per `inotify(7)`, the kernel emits a stream of variable-length
//! records on `read(inotify_fd)`:
//!
//! ```text
//!     struct inotify_event {
//!         __s32 wd;       /* watch descriptor (or 0 / -1 sentinel) */
//!         __u32 mask;     /* event mask */
//!         __u32 cookie;   /* IN_MOVED_FROM ↔ IN_MOVED_TO pairing */
//!         __u32 len;      /* bytes after the header, including NUL padding */
//!         char  name[];   /* optional NUL-terminated, NUL-padded basename */
//!     };
//! ```
//!
//! Each field is in native byte order (the kernel's ABI is host-native;
//! see `linux/inotify.h`). The parser reads through `from_ne_bytes`,
//! which works on any Linux architecture without endianness ceremony.
//!
//! ## What v1 keeps and what it throws away
//!
//! v1 collapses every name-bearing structure event into
//! [`specter_core::FsEvent::StructureChanged`] and re-discovers the
//! delta via a parent probe — `cookie` and `name` are never consumed.
//! The parser preserves them anyway so a future optimization that mints
//! a typed `Input::ChildEvent { parent, name, kind }` directly from
//! inotify does not have to reshape the record API. Cost is one
//! byte-slice borrow per record on the hot path.
//!
//! ## Truncation discipline
//!
//! The kernel returns `EINVAL` if the user buffer is below the per-event
//! minimum (`sizeof(struct inotify_event) + NAME_MAX + 1` ≈ 273 bytes,
//! per `inotify(7)`). The watcher sizes its drain buffer well above
//! that — see [`super::watcher`] — so `read` never returns a truncated
//! record. The defensive guards below silently end
//! iteration on a malformed buffer rather than panic, but those branches
//! are observationally dead under healthy invariants.

use libc::c_int;

/// Size of the fixed-width header preceding the variable-length name.
/// Bound to [`libc::inotify_event`] so a future libc binding change
/// (the kernel ABI itself is stable) becomes a compile-time concern
/// rather than a silent drift between our parser and the kernel
/// layout.
const HEADER_SIZE: usize = std::mem::size_of::<libc::inotify_event>();

/// One inotify record borrowed from the watcher's drain buffer.
///
/// `name` is the basename portion with trailing NUL padding stripped:
/// non-name records (`IN_DELETE_SELF`, `IN_UNMOUNT`, etc.) report
/// `len = 0`, surfaced here as an empty slice.
#[derive(Debug, Clone, Copy)]
pub(super) struct Record<'a> {
    /// Watch descriptor identifying which install fired this event.
    /// Maps to `ResourceId` via the watcher's `wd → r` index.
    pub wd: c_int,
    /// Bit set of `IN_*` flags; consumed by [`super::normalize`].
    pub mask: u32,
    /// Pairing token for `IN_MOVED_FROM` ↔ `IN_MOVED_TO`. v1 does not
    /// match move pairs (the engine probes the parent on
    /// `StructureChanged`); preserved for v3+ named-event handling.
    #[allow(dead_code)]
    pub cookie: u32,
    /// Basename portion (NUL padding stripped). Empty for non-name
    /// events. Borrows from the watcher's drain buffer; lifetime
    /// matches the parser's input slice.
    pub name: &'a [u8],
}

/// Iterate every record packed into `buf`. Iteration ends at the first
/// truncated record; under healthy invariants `buf` always ends on a
/// record boundary, so the iterator drains exactly the populated prefix.
#[must_use]
pub(super) const fn parse(buf: &[u8]) -> RecordIter<'_> {
    RecordIter { remaining: buf }
}

#[derive(Debug)]
pub(super) struct RecordIter<'a> {
    remaining: &'a [u8],
}

impl<'a> Iterator for RecordIter<'a> {
    type Item = Record<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        // Defensive: a truncated header is impossible from a real
        // `read(inotify_fd)` (the kernel never half-fills a record), but
        // ending iteration silently is safer than panicking on a fuzzed
        // input.
        if self.remaining.len() < HEADER_SIZE {
            return None;
        }

        let header: &[u8; HEADER_SIZE] = self.remaining[..HEADER_SIZE]
            .try_into()
            .expect("slicing 16 bytes from a 16+ byte slice always yields a [u8; 16]");

        let wd = i32::from_ne_bytes([header[0], header[1], header[2], header[3]]);
        let mask = u32::from_ne_bytes([header[4], header[5], header[6], header[7]]);
        let cookie = u32::from_ne_bytes([header[8], header[9], header[10], header[11]]);
        // `len` is `__u32` per `linux/inotify.h`; widening to `usize`
        // is exact on 64-bit (specter-sensor's only Linux target — see
        // the 32-bit compile_error in `super::super::inotify`).
        let len = u32::from_ne_bytes([header[12], header[13], header[14], header[15]]) as usize;

        let payload_end = match HEADER_SIZE.checked_add(len) {
            Some(end) if end <= self.remaining.len() => end,
            // Either the kernel claimed a length the buffer can't honour
            // (truncation — see the discipline note in the module
            // header), or `len` arithmetic overflowed `usize` (impossible
            // on 64-bit but defended for completeness).
            _ => return None,
        };

        let name_with_padding = &self.remaining[HEADER_SIZE..payload_end];
        // Strip trailing NUL padding to get the basename. The kernel
        // pads to align the next record; `position` returns `None` only
        // when `len = 0` (no name), in which case `name` is an empty
        // slice.
        let name_end = name_with_padding
            .iter()
            .position(|&b| b == 0)
            .unwrap_or(name_with_padding.len());
        let name = &name_with_padding[..name_end];

        self.remaining = &self.remaining[payload_end..];

        Some(Record {
            wd,
            mask,
            cookie,
            name,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{HEADER_SIZE, parse};

    /// Convenience: encode one record into a `Vec<u8>` ready for
    /// [`parse`]. Mirrors what the kernel writes onto the inotify_fd's
    /// read buffer (native byte order, padded name).
    fn encode(wd: i32, mask: u32, cookie: u32, name: &[u8], padding: usize) -> Vec<u8> {
        let len = name.len() + padding;
        let mut out = Vec::with_capacity(HEADER_SIZE + len);
        out.extend_from_slice(&wd.to_ne_bytes());
        out.extend_from_slice(&mask.to_ne_bytes());
        out.extend_from_slice(&cookie.to_ne_bytes());
        out.extend_from_slice(&u32::try_from(len).unwrap().to_ne_bytes());
        out.extend_from_slice(name);
        out.extend(std::iter::repeat_n(0u8, padding));
        out
    }

    #[test]
    fn parses_single_record_with_no_name() {
        // IN_MOVE_SELF — slot-final, no name, len = 0.
        let buf = encode(5, 0x40, 0x1234, b"", 0);
        let recs: Vec<_> = parse(&buf).collect();
        assert_eq!(recs.len(), 1);
        let r = &recs[0];
        assert_eq!(r.wd, 5);
        assert_eq!(r.mask, 0x40);
        assert_eq!(r.cookie, 0x1234);
        assert!(r.name.is_empty());
    }

    #[test]
    fn parses_multiple_records_with_names() {
        let mut buf = encode(3, libc::IN_CREATE, 0, b"hello", 3);
        buf.extend(encode(4, libc::IN_DELETE, 0, b"x", 3));

        let recs: Vec<_> = parse(&buf).collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].wd, 3);
        assert_eq!(recs[0].name, b"hello");
        assert_eq!(recs[1].wd, 4);
        assert_eq!(recs[1].name, b"x");
    }

    #[test]
    fn cookie_paired_records_round_trip() {
        // IN_MOVED_FROM and IN_MOVED_TO share a cookie under the
        // kernel's pairing protocol; v1 doesn't pair them but the
        // parser preserves the cookie for v3+ named-event handling.
        let cookie = 0xDEAD_BEEF;
        let mut buf = encode(7, libc::IN_MOVED_FROM, cookie, b"old", 1);
        buf.extend(encode(7, libc::IN_MOVED_TO, cookie, b"new", 1));
        let recs: Vec<_> = parse(&buf).collect();
        assert_eq!(recs.len(), 2);
        assert_eq!(recs[0].cookie, cookie);
        assert_eq!(recs[1].cookie, cookie);
        assert_eq!(recs[0].name, b"old");
        assert_eq!(recs[1].name, b"new");
    }

    #[test]
    fn name_with_no_padding_round_trips() {
        // len = name.len() exactly; no NUL padding; name has no NULs.
        let buf = encode(1, libc::IN_CREATE, 0, b"abcd", 0);
        let recs: Vec<_> = parse(&buf).collect();
        assert_eq!(recs.len(), 1);
        assert_eq!(recs[0].name, b"abcd");
    }

    #[test]
    fn truncated_header_yields_zero_records() {
        // First 8 bytes only — header is 16 bytes; iter must stop.
        let buf = vec![0u8; 8];
        assert_eq!(parse(&buf).count(), 0);
    }

    #[test]
    fn truncated_after_header_yields_zero_records() {
        // Header claims len = 20 but only 4 payload bytes follow.
        let mut buf: Vec<u8> = Vec::new();
        buf.extend_from_slice(&1_i32.to_ne_bytes());
        buf.extend_from_slice(&0x1_u32.to_ne_bytes());
        buf.extend_from_slice(&0_u32.to_ne_bytes());
        buf.extend_from_slice(&20_u32.to_ne_bytes());
        buf.extend_from_slice(&[0u8; 4]);
        assert_eq!(parse(&buf).count(), 0);
    }

    #[test]
    fn empty_buffer_yields_zero_records() {
        assert_eq!(parse(&[]).count(), 0);
    }

    #[test]
    fn iter_advances_through_padded_records_in_order() {
        // Three back-to-back records with varying name lengths and
        // padding; verify the iterator hands them back in order.
        let mut buf = encode(10, libc::IN_CREATE, 0, b"a", 3);
        buf.extend(encode(11, libc::IN_DELETE, 0, b"bb", 2));
        buf.extend(encode(12, libc::IN_ATTRIB, 0, b"ccc", 1));

        let wds: Vec<_> = parse(&buf).map(|r| r.wd).collect();
        assert_eq!(wds, vec![10, 11, 12]);
    }
}
