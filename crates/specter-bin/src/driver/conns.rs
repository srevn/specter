//! Per-connection state owned by [`super::hub::DriverHub`] via
//! `BTreeMap<Token, ConnState>`.
//!
//! Each conn is one mio-registered [`mio::net::UnixStream`] plus the
//! read accumulator, the partial-write residue, and the per-conn role
//! (`Reqs` pre-subscribe, `Sub` after).
//!
//! # State-transition ownership
//!
//! Every mutation that touches the per-conn role, the `missed` /
//! `first_dropped_at` back-pressure accounting, the `close_after_flush`
//! flag, or the write-queue bytes flows through a typed `&mut self`
//! method on [`ConnState`]. The Hub's reactor surface (kernel-fd loops,
//! mio Poll registry, accept) keeps its responsibilities; per-conn
//! state-machine mechanics live here as the sole writer set. A future
//! dispatch path that re-implements a capacity gate or a role flip
//! cannot diverge — there is no public field setter to reach for.
//!
//! # Visibility
//!
//! Every export is `pub(super)`. The only consumers are
//! [`super::hub::DriverHub`] (per-conn read/write helpers, fan-out
//! dispatch) and the driver-side IPC handler (the Subscribe arm
//! flips the role through [`ConnState::transition_to_sub`]).

use crate::ipc::framing::serialize_line;
use crate::ipc::wire::WireDiagnostic;
use specter_core::SubId;
use std::collections::VecDeque;
use std::time::SystemTime;

/// Per-conn write-queue high-water mark. A subscriber that can't keep
/// up sees its queue grow; past this watermark the dispatch loop
/// counts the dropped diag against the `Missed` marker rather than
/// pushing more bytes into a stalled queue. Matches the framing-level
/// [`crate::ipc::framing::MAX_LINE_BYTES`] cap so a single oversize
/// response (a saturated `list` projection on a busy daemon) wedging
/// the queue has the same backpressure footprint as a hostile read.
/// The two constants own different invariants (framing envelope vs
/// per-conn backpressure) and stay split so a future divergence
/// (chunked diag fan-out with a larger queue, etc.) doesn't have to
/// untangle them.
pub(super) const WRITE_QUEUE_HIGH_WATER: usize = 256 * 1024;

/// Connection state held on [`super::hub::DriverHub`]'s
/// `conns: BTreeMap<Token, ConnState>`.
///
/// One per accepted operator IPC client. Drops when
/// `DriverHub::terminate_conn` removes the entry from the map,
/// closing the underlying socket fd and (after explicit
/// `Registry::deregister` at the call site) freeing the mio-side
/// registration.
pub(super) struct ConnState {
    /// The mio-wrapped accepted stream. Owned for the lifetime of the
    /// entry; `Drop` closes the fd. Holds mio's `IoSource` wrapping
    /// of the underlying `std::os::unix::net::UnixStream` — the
    /// non-blocking flag is set by mio at accept time, so no extra
    /// `set_nonblocking` is required at this constructor.
    pub(super) stream: mio::net::UnixStream,
    /// The mio Token registered against the [`super::hub::DriverHub`]
    /// Poll registry. Kept on the struct so `DriverHub::drain_writable`
    /// / `terminate_conn` can reach the registry without re-deriving
    /// the token from the map key.
    pub(super) token: mio::Token,
    /// Line accumulator. `DriverHub::read_conn_into_lines` appends
    /// every `read` chunk here and slices out LF-delimited lines.
    /// Pre-allocated to `1024` bytes — comfortably above the typical
    /// `Status` / `Show` verb size while keeping the cold start
    /// allocation small.
    pub(super) read_buf: Vec<u8>,
    /// Pending write residue: response bytes the previous
    /// `drain_writable` could not flush due to kernel-side backpressure,
    /// plus any diag bytes queued by the fan-out path in this tick.
    /// `VecDeque` lets `drain_writable` slice off the consumed prefix
    /// without copying the tail forward.
    pub(super) write_queue: VecDeque<u8>,
    /// Conn role — pre-subscribe `Reqs` or post-subscribe `Sub`. The
    /// role gate controls whether the fan-out path delivers diags into
    /// `write_queue`: only `Sub` conns receive them.
    pub(super) role: ConnRole,
    /// Set when the next successful flush of `write_queue` should
    /// terminate the conn. Lifecycle:
    ///
    /// - **Oversize input** (the read accumulator exceeds
    ///   [`crate::ipc::framing::MAX_LINE_BYTES`] or a complete line
    ///   crosses the cap): set true via
    ///   [`ConnState::arm_close_after_flush`].
    /// - **Over-watermark response** (a verb projection serializes
    ///   into a payload larger than [`WRITE_QUEUE_HIGH_WATER`]):
    ///   set true via [`ConnState::push_response`]'s overflow arm.
    ///
    /// `drain_writable` observes the flag when the queue empties via a
    /// successful flush; the Hub-level `try_terminate_if_idle` handles
    /// the queue-empty-at-arm case (where no WRITABLE edge would ever
    /// run drain_writable). Together the two paths cover both arming
    /// shapes: bytes-queued → flush-then-terminate, empty-queue →
    /// terminate-inline.
    pub(super) close_after_flush: bool,
}

/// Per-conn role axis. A fresh conn enters [`ConnRole::Reqs`]; the
/// Subscribe verb's handler flips it to [`ConnRole::Sub`] via
/// [`ConnState::transition_to_sub`] *after* the SubscribeAck bytes
/// are already in the write_queue — the ack-before-fanout ordering
/// the wire-side regression test pins.
pub(super) enum ConnRole {
    /// Request/response shape: every line is parsed as a `WireRequest`
    /// and answered with a `ResponsePayload`. The fan-out path skips
    /// `Reqs` conns.
    Reqs,
    /// Diagnostic subscriber shape: the fan-out path appends
    /// `WireDiagnostic` lines to the write_queue. `filter` scopes the
    /// stream to a single Sub when set (the `wait <name>` use case);
    /// `missed` accumulates dropped-due-to-backpressure diags for the
    /// next `Missed` marker emission.
    Sub {
        /// `Some(sid)` ⇒ deliver only diags whose `diag_sub_id` resolves
        /// to `sid` (per-Sub `wait`). `None` ⇒ unfiltered (`tail`).
        /// Resolved server-side at Subscribe time so a typo never
        /// reaches the fan-out path.
        filter: Option<SubId>,
        /// Back-pressure marker: number of diags dropped since the
        /// last successful emission. Flushed lazily as a
        /// `WireDiagnostic::Missed` line before the next `Diag` so
        /// causal order is preserved on the wire. `saturating_add`
        /// guards the `u32`-overflow corner (practically unreachable
        /// — a wedged subscriber dropping `2^32` events would have
        /// hit the disconnect path first).
        missed: u32,
        /// Wall-clock of the first drop in the currently-open `missed`
        /// window. Captured on the `0 → 1` transition; cleared back to
        /// `None` when the marker flushes. Threading the first-drop
        /// time as the marker's `at` gives operators the start-of-
        /// window timestamp — when the drops actually began — rather
        /// than the flush-time stamp the marker would otherwise carry
        /// (the queue has already drained by the time the marker
        /// reaches the wire, so flush-time is misleading for incident
        /// forensics).
        ///
        /// Two-field invariant: `missed > 0 ⇒ first_dropped_at.is_some()`.
        /// Upheld by [`ConnState::try_dispatch_diag`] as the sole
        /// mutator of both fields.
        first_dropped_at: Option<SystemTime>,
    },
}

/// Outcome of one [`ConnState::try_dispatch_diag`] attempt. The Hub's
/// fan-out loop discards the value (per-conn missed accounting is
/// internal to the method); the discriminant is exposed for unit-test
/// observation of the axis the call took.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(super) enum DispatchOutcome {
    /// Diag bytes (possibly preceded by a flushed Missed marker) are
    /// in the write_queue.
    Accepted,
    /// Diag dropped — capacity gate refused. `missed` was bumped;
    /// `first_dropped_at` was captured on the `0 → 1` transition.
    Dropped,
    /// Conn not eligible (`close_after_flush`, `role != Sub`, or
    /// per-Sub filter mismatch). No mutation.
    Skipped,
}

/// Outcome of one [`ConnState::push_response`] attempt.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(super) enum PushOutcome {
    /// Bytes pushed; the caller may want to arm WRITABLE interest
    /// (done in bulk by [`super::hub::DriverHub::arm_writable_interests`]
    /// at end of tick).
    Accepted,
    /// Bytes did NOT fit; `close_after_flush` was armed internally.
    /// The Hub-level caller pairs this with
    /// [`super::hub::DriverHub::try_terminate_if_idle`] so a refusal
    /// against a previously-empty queue terminates the conn inline
    /// (rather than lingering with an armed close that no WRITABLE
    /// edge will ever observe).
    Refused,
}

impl ConnState {
    /// Construct a fresh conn from an accepted mio stream + the token
    /// it will register under.
    ///
    /// The stream is already non-blocking via mio's accept path
    /// (`mio::net::UnixListener::accept` returns a non-blocking
    /// stream irrespective of the listener's flag, per mio's
    /// `IoSource` convention) — no extra `set_nonblocking` call is
    /// required here.
    ///
    /// Buffers pre-allocate `1024` bytes apiece: above the typical
    /// IPC verb / response size, below the page boundary that would
    /// otherwise dirty a fresh page for every cold-start client.
    pub(super) fn new(stream: mio::net::UnixStream, token: mio::Token) -> Self {
        Self {
            stream,
            token,
            read_buf: Vec::with_capacity(1024),
            write_queue: VecDeque::with_capacity(1024),
            role: ConnRole::Reqs,
            close_after_flush: false,
        }
    }

    /// Flip the conn's role to [`ConnRole::Sub`].
    ///
    /// The ack-ordering contract (`SubscribeAck` bytes precede every
    /// future diag on the wire) is enforced *at the call site* by
    /// enqueueing the ack into `write_queue` BEFORE invoking this
    /// method. The fan-out path skips `Reqs` conns, so any diag
    /// emission between the ack enqueue and this role flip is
    /// structurally absent from this conn's queue.
    ///
    /// One-shot-per-conn is the contract: a fresh conn enters `Reqs`,
    /// flips to `Sub` once, and never flips back. The *structural*
    /// gate lives at the [`super::ipc`] Subscribe handler — a repeat
    /// Subscribe on a `Sub` conn returns
    /// [`crate::ipc::protocol::ERR_ALREADY_SUBSCRIBED`] before
    /// reaching this method. The `debug_assert!` here is the contract
    /// witness: any future caller that bypasses the handler gate
    /// fails loudly in debug builds.
    pub(super) fn transition_to_sub(&mut self, filter: Option<SubId>) {
        debug_assert!(
            matches!(self.role, ConnRole::Reqs),
            "transition_to_sub on non-Reqs role — the Subscribe handler gate is bypassed",
        );
        self.role = ConnRole::Sub {
            filter,
            missed: 0,
            first_dropped_at: None,
        };
    }

    /// Set `close_after_flush`. Idempotent — repeated calls are a
    /// no-op (a peer triggering close from multiple paths in one tick,
    /// e.g. an oversize line and an over-water response, legitimately
    /// arms the same flag through different entry points).
    ///
    /// Does NOT terminate. Termination is a Hub concern (needs the mio
    /// Poll registry to deregister the stream); the Hub-level call
    /// sites pair this with
    /// [`super::hub::DriverHub::try_terminate_if_idle`] at the right
    /// moment — typically *after* the in-flight processing pass that
    /// may have pushed response bytes into the queue.
    pub(super) const fn arm_close_after_flush(&mut self) {
        self.close_after_flush = true;
    }

    /// Try to enqueue `diag_line` (one LF-terminated JSON line) into
    /// this conn's write_queue.
    ///
    /// Sole mutator of `self.role.Sub.missed` and
    /// `self.role.Sub.first_dropped_at`; the two-field invariant
    /// (`missed > 0 ⇒ first_dropped_at.is_some()`) is upheld here as
    /// long as no other code path writes those fields.
    ///
    /// Axes evaluated in order:
    /// 1. `close_after_flush` → `Skipped` (closing conn doesn't
    ///    accumulate fresh diags).
    /// 2. `role != Sub` → `Skipped` (pre-subscribe conn).
    /// 3. Per-Sub `filter` mismatch → `Skipped`.
    /// 4. Capacity gate: combined `(queue_len + marker_bytes + diag_line)`
    ///    must fit under [`WRITE_QUEUE_HIGH_WATER`]. When it does not,
    ///    the diag drops, `missed` is bumped, and `first_dropped_at`
    ///    is captured on the `0 → 1` transition.
    /// 5. Otherwise: flush any pending Missed marker (carrying the
    ///    captured `first_dropped_at` as the marker's `at` so
    ///    operators see start-of-window time, not flush time), reset
    ///    the bookkeeping, then push the diag bytes.
    ///
    /// `at` threads from the caller so every conn observes a
    /// byte-identical timestamp for one engine emission; the marker
    /// path uses the previously-captured `first_dropped_at` instead,
    /// falling back to `at` if the invariant is ever violated by a
    /// future cross-cutting change.
    pub(super) fn try_dispatch_diag(
        &mut self,
        diag_line: &[u8],
        diag_sub: Option<SubId>,
        at: SystemTime,
    ) -> DispatchOutcome {
        if self.close_after_flush {
            return DispatchOutcome::Skipped;
        }
        let ConnRole::Sub {
            filter,
            missed,
            first_dropped_at,
        } = &mut self.role
        else {
            return DispatchOutcome::Skipped;
        };
        if let Some(want) = filter
            && diag_sub != Some(*want)
        {
            return DispatchOutcome::Skipped;
        }

        // Pre-build the marker line if a missed window is open. The
        // marker carries `first_dropped_at` (set at the 0→1 transition)
        // as its `at` — the `unwrap_or(at)` fallback is defensive
        // against a future caller mutating `missed` without capturing
        // a first-drop time.
        let marker_bytes: Option<Vec<u8>> = (*missed > 0).then(|| {
            let marker = WireDiagnostic::Missed {
                at: first_dropped_at.unwrap_or(at).into(),
                count: *missed,
            };
            serialize_line(&marker)
                .expect("WireDiagnostic::Missed serialization is infallible by construction")
        });

        let marker_len = marker_bytes.as_ref().map_or(0, Vec::len);
        let queue_len = self.write_queue.len();
        let combined = queue_len
            .saturating_add(marker_len)
            .saturating_add(diag_line.len());
        if combined > WRITE_QUEUE_HIGH_WATER {
            if *missed == 0 {
                *first_dropped_at = Some(at);
            }
            *missed = missed.saturating_add(1);
            return DispatchOutcome::Dropped;
        }

        if let Some(mb) = marker_bytes {
            self.write_queue.extend(mb);
            *missed = 0;
            *first_dropped_at = None;
        }
        self.write_queue.extend(diag_line);
        DispatchOutcome::Accepted
    }

    /// Push response bytes into the write_queue.
    ///
    /// Capacity-gated: if the projected queue length would exceed
    /// [`WRITE_QUEUE_HIGH_WATER`], the queue is left untouched and
    /// `close_after_flush` is armed via
    /// [`Self::arm_close_after_flush`]. The Hub-level
    /// [`super::hub::DriverHub::enqueue_response`] pairs the `Refused`
    /// outcome with a `try_terminate_if_idle` call to handle the
    /// queue-empty-at-arm case.
    pub(super) fn push_response(&mut self, bytes: &[u8]) -> PushOutcome {
        if self.write_queue.len().saturating_add(bytes.len()) > WRITE_QUEUE_HIGH_WATER {
            self.arm_close_after_flush();
            return PushOutcome::Refused;
        }
        self.write_queue.extend(bytes);
        PushOutcome::Accepted
    }
}

impl std::fmt::Debug for ConnState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConnState")
            .field("token", &self.token)
            .field("read_buf_len", &self.read_buf.len())
            .field("write_queue_len", &self.write_queue.len())
            .field("role", &self.role)
            .field("close_after_flush", &self.close_after_flush)
            .finish_non_exhaustive()
    }
}

impl std::fmt::Debug for ConnRole {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Reqs => f.write_str("Reqs"),
            Self::Sub {
                filter,
                missed,
                first_dropped_at,
            } => f
                .debug_struct("Sub")
                .field("filter", filter)
                .field("missed", missed)
                .field("first_dropped_at", first_dropped_at)
                .finish(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use slotmap::KeyData;
    use std::time::Duration;

    /// Pair a `ConnState` with one end of a `UnixStream::pair()`. The
    /// other end is returned as `_peer` so tests can `drop(_peer)` if
    /// they want to model peer-gone, or simply let it live through the
    /// test's lifetime. The wrapping reflects that every ConnState
    /// needs a real fd to construct (mio's accepted-stream requirement),
    /// even when the tests don't exercise the IO path.
    fn fresh_conn() -> (ConnState, mio::net::UnixStream) {
        let (a, b) = mio::net::UnixStream::pair().expect("socketpair");
        (ConnState::new(a, mio::Token(0x100)), b)
    }

    /// Synthesize a deterministic [`SubId`] for filter-axis tests. The
    /// raw u64 is arbitrary — the only requirement is that it round-
    /// trips through [`slotmap::KeyData::from_ffi`] to the same id
    /// across calls in one test.
    fn make_sid(raw: u64) -> SubId {
        SubId::from(KeyData::from_ffi(raw))
    }

    /// A fresh conn lands in `Reqs` with empty buffers and no pending
    /// close — the structural floor every other test builds on.
    #[test]
    fn new_conn_starts_in_reqs_with_empty_buffers() {
        let (conn, _peer) = fresh_conn();
        assert!(matches!(conn.role, ConnRole::Reqs));
        assert!(conn.read_buf.is_empty());
        assert!(conn.write_queue.is_empty());
        assert!(!conn.close_after_flush);
        assert_eq!(conn.token, mio::Token(0x100));
    }

    /// `transition_to_sub` flips the role and initializes the Sub-side
    /// accounting (filter, missed, first_dropped_at) to the
    /// fresh-window defaults. Pins the post-flip shape.
    #[test]
    fn transition_to_sub_initializes_sub_state() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        match conn.role {
            ConnRole::Sub {
                filter,
                missed,
                first_dropped_at,
            } => {
                assert_eq!(filter, None);
                assert_eq!(missed, 0);
                assert_eq!(first_dropped_at, None);
            }
            ConnRole::Reqs => panic!("expected Sub role after transition"),
        }
    }

    /// `arm_close_after_flush` is idempotent — repeated calls are
    /// silent no-ops. The fan-out path and the read drain both reach
    /// the method in the same tick when a peer streams an oversize
    /// payload that also triggers an over-water response; the second
    /// arm must not panic in release nor debug.
    #[test]
    fn arm_close_after_flush_is_idempotent() {
        let (mut conn, _peer) = fresh_conn();
        conn.arm_close_after_flush();
        conn.arm_close_after_flush();
        assert!(conn.close_after_flush);
    }

    /// `try_dispatch_diag` returns `Skipped` when the conn is armed
    /// for close — closing conns drain their existing residue then
    /// terminate; no fresh diags accumulate on the way out.
    #[test]
    fn try_dispatch_diag_skips_on_close_after_flush() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        conn.arm_close_after_flush();
        let line = b"{\"diag\":\"placeholder\"}\n";
        let outcome = conn.try_dispatch_diag(line, None, SystemTime::UNIX_EPOCH);
        assert_eq!(outcome, DispatchOutcome::Skipped);
        assert!(conn.write_queue.is_empty(), "no bytes leak past the gate");
    }

    /// `try_dispatch_diag` returns `Skipped` when the conn is still
    /// in `Reqs` role — pre-subscribe conns don't receive diags.
    #[test]
    fn try_dispatch_diag_skips_on_reqs_role() {
        let (mut conn, _peer) = fresh_conn();
        let line = b"{\"diag\":\"placeholder\"}\n";
        let outcome = conn.try_dispatch_diag(line, None, SystemTime::UNIX_EPOCH);
        assert_eq!(outcome, DispatchOutcome::Skipped);
        assert!(conn.write_queue.is_empty());
    }

    /// `try_dispatch_diag` returns `Skipped` when the per-Sub filter
    /// rejects the diag's `diag_sub`. The unfiltered-tail (filter=None)
    /// path is covered by `try_dispatch_diag_pushes_on_unfiltered_sub`.
    #[test]
    fn try_dispatch_diag_skips_on_filter_mismatch() {
        let (mut conn, _peer) = fresh_conn();
        let want = make_sid(0x42);
        conn.transition_to_sub(Some(want));
        let line = b"{\"diag\":\"placeholder\"}\n";
        let other = make_sid(0x99);
        let outcome = conn.try_dispatch_diag(line, Some(other), SystemTime::UNIX_EPOCH);
        assert_eq!(outcome, DispatchOutcome::Skipped);
        assert!(conn.write_queue.is_empty());
    }

    /// Happy path: an unfiltered Sub-mode conn under the capacity cap
    /// accepts the diag bytes verbatim into the write_queue.
    #[test]
    fn try_dispatch_diag_pushes_on_unfiltered_sub() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        let line: &[u8] = b"{\"diag\":\"placeholder\"}\n";
        let outcome = conn.try_dispatch_diag(line, None, SystemTime::UNIX_EPOCH);
        assert_eq!(outcome, DispatchOutcome::Accepted);
        let queued: Vec<u8> = conn.write_queue.iter().copied().collect();
        assert_eq!(queued, line);
    }

    /// Capacity gate: a diag whose projected queue length crosses
    /// [`WRITE_QUEUE_HIGH_WATER`] drops, bumping `missed` to 1. No
    /// bytes leak into the queue.
    #[test]
    fn try_dispatch_diag_drops_and_bumps_missed_on_capacity_overflow() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        // Pre-fill near the high-water mark so a small diag overflows.
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', WRITE_QUEUE_HIGH_WATER - 10));
        let queue_len_before = conn.write_queue.len();
        let line: &[u8] = b"{\"diag\":\"too_big_to_fit_in_remaining_capacity\"}\n";
        assert!(line.len() > 10, "test precondition");

        let outcome = conn.try_dispatch_diag(line, None, SystemTime::UNIX_EPOCH);
        assert_eq!(outcome, DispatchOutcome::Dropped);
        assert_eq!(conn.write_queue.len(), queue_len_before, "queue untouched");
        match &conn.role {
            ConnRole::Sub {
                missed,
                first_dropped_at,
                ..
            } => {
                assert_eq!(*missed, 1);
                assert_eq!(*first_dropped_at, Some(SystemTime::UNIX_EPOCH));
            }
            ConnRole::Reqs => panic!("expected Sub role"),
        }
    }

    /// First-drop time is captured at the `0 → 1` transition of
    /// `missed`. Subsequent drops in the same window do NOT
    /// re-capture, so the marker carries the start-of-window
    /// timestamp (operators see when drops began, not when the
    /// system got around to mentioning them).
    #[test]
    fn try_dispatch_diag_preserves_first_dropped_at_across_subsequent_drops() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', WRITE_QUEUE_HIGH_WATER - 5));
        let line: &[u8] = b"{\"diag\":\"too_big_to_fit\"}\n";

        let at1 = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        let at2 = SystemTime::UNIX_EPOCH + Duration::from_secs(200);
        let at3 = SystemTime::UNIX_EPOCH + Duration::from_secs(301);
        assert_eq!(
            conn.try_dispatch_diag(line, None, at1),
            DispatchOutcome::Dropped
        );
        assert_eq!(
            conn.try_dispatch_diag(line, None, at2),
            DispatchOutcome::Dropped
        );
        assert_eq!(
            conn.try_dispatch_diag(line, None, at3),
            DispatchOutcome::Dropped
        );
        match &conn.role {
            ConnRole::Sub {
                missed,
                first_dropped_at,
                ..
            } => {
                assert_eq!(*missed, 3);
                assert_eq!(
                    *first_dropped_at,
                    Some(at1),
                    "first_dropped_at sticks to the 0→1 transition's at",
                );
            }
            ConnRole::Reqs => panic!("expected Sub role"),
        }
    }

    /// Causal ordering: when a missed window is open and a fresh diag
    /// fits, the marker line is queued BEFORE the diag bytes so
    /// operators see "lost N events" preceding the next reachable
    /// diag on the wire.
    #[test]
    fn try_dispatch_diag_flushes_missed_marker_before_diag_in_causal_order() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        // Drive one drop to open a missed window.
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', WRITE_QUEUE_HIGH_WATER - 5));
        let big: &[u8] = b"{\"diag\":\"too_big_to_fit\"}\n";
        let at_drop = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        assert_eq!(
            conn.try_dispatch_diag(big, None, at_drop),
            DispatchOutcome::Dropped
        );

        // Drain to simulate the wire making room.
        conn.write_queue.clear();

        // Dispatch a small diag — it fits, AND the marker must
        // precede it.
        let small: &[u8] = b"{\"diag\":\"placeholder\"}\n";
        let at_flush = SystemTime::UNIX_EPOCH + Duration::from_secs(200);
        assert_eq!(
            conn.try_dispatch_diag(small, None, at_flush),
            DispatchOutcome::Accepted
        );
        let queued: Vec<u8> = conn.write_queue.iter().copied().collect();
        let lines: Vec<&[u8]> = queued
            .split(|&b| b == b'\n')
            .filter(|l| !l.is_empty())
            .collect();
        assert_eq!(lines.len(), 2, "marker then diag");
        let first_v: serde_json::Value =
            serde_json::from_slice(lines[0]).expect("marker is valid JSON");
        assert_eq!(first_v["diag"], "_missed");
        let second_v: serde_json::Value =
            serde_json::from_slice(lines[1]).expect("diag is valid JSON");
        assert_eq!(second_v["diag"], "placeholder");
    }

    /// Wire-level pin for the marker's `at`: it is the first-drop
    /// time, not the flush-time. Operators reading a `_missed`
    /// marker see when drops began, not when the system got around
    /// to mentioning them.
    #[test]
    fn try_dispatch_diag_marker_uses_first_dropped_at_not_flush_at() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', WRITE_QUEUE_HIGH_WATER - 5));
        let big: &[u8] = b"{\"diag\":\"too_big_to_fit\"}\n";
        let at_drop = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        assert_eq!(
            conn.try_dispatch_diag(big, None, at_drop),
            DispatchOutcome::Dropped,
        );

        conn.write_queue.clear();

        let small: &[u8] = b"{\"diag\":\"placeholder\"}\n";
        let at_flush = SystemTime::UNIX_EPOCH + Duration::from_secs(500);
        assert_eq!(
            conn.try_dispatch_diag(small, None, at_flush),
            DispatchOutcome::Accepted,
        );

        let queued: Vec<u8> = conn.write_queue.iter().copied().collect();
        let marker_line: &[u8] = queued
            .split(|&b| b == b'\n')
            .find(|l| !l.is_empty())
            .expect("marker first");
        let marker_v: serde_json::Value = serde_json::from_slice(marker_line).expect("marker JSON");
        let expected_at = humantime::format_rfc3339_seconds(at_drop).to_string();
        assert_eq!(
            marker_v["at"].as_str().expect("at is a string"),
            expected_at,
            "marker carries first-drop time, not flush time",
        );
    }

    /// Combined-capacity check: when the marker+diag combined size
    /// crosses [`WRITE_QUEUE_HIGH_WATER`], the dispatch drops without
    /// pushing the marker. `missed` re-accumulates; the marker tries
    /// again on the next dispatch.
    #[test]
    fn try_dispatch_diag_drops_when_marker_plus_diag_does_not_fit() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        // Drive one drop to open the marker window.
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', WRITE_QUEUE_HIGH_WATER - 5));
        let big: &[u8] = b"{\"diag\":\"too_big_to_fit\"}\n";
        let at_drop = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        assert_eq!(
            conn.try_dispatch_diag(big, None, at_drop),
            DispatchOutcome::Dropped,
        );

        // Keep the queue near the cap. The marker bytes + a fresh diag
        // both have to fit — even with the marker alone fitting, the
        // combined size overflows.
        let queue_len_before = conn.write_queue.len();
        let line: &[u8] = b"{\"diag\":\"still_too_big\"}\n";
        let at2 = SystemTime::UNIX_EPOCH + Duration::from_secs(200);
        assert_eq!(
            conn.try_dispatch_diag(line, None, at2),
            DispatchOutcome::Dropped
        );
        assert_eq!(
            conn.write_queue.len(),
            queue_len_before,
            "no bytes pushed when combined overflows",
        );
        match &conn.role {
            ConnRole::Sub {
                missed,
                first_dropped_at,
                ..
            } => {
                assert_eq!(*missed, 2, "missed re-accumulates on combined-overflow");
                assert_eq!(
                    *first_dropped_at,
                    Some(at_drop),
                    "first_dropped_at sticks to the original 0→1 capture",
                );
            }
            ConnRole::Reqs => panic!("expected Sub role"),
        }
    }

    /// After a successful marker flush, `missed` resets to 0 and
    /// `first_dropped_at` resets to `None` — the invariant
    /// `missed > 0 ⇒ first_dropped_at.is_some()` is restored on the
    /// flush edge.
    #[test]
    fn try_dispatch_diag_resets_first_dropped_at_on_flush() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', WRITE_QUEUE_HIGH_WATER - 5));
        let big: &[u8] = b"{\"diag\":\"too_big_to_fit\"}\n";
        let at_drop = SystemTime::UNIX_EPOCH + Duration::from_secs(100);
        assert_eq!(
            conn.try_dispatch_diag(big, None, at_drop),
            DispatchOutcome::Dropped,
        );

        conn.write_queue.clear();

        let small: &[u8] = b"{\"diag\":\"placeholder\"}\n";
        let at_flush = SystemTime::UNIX_EPOCH + Duration::from_secs(200);
        assert_eq!(
            conn.try_dispatch_diag(small, None, at_flush),
            DispatchOutcome::Accepted
        );
        match &conn.role {
            ConnRole::Sub {
                missed,
                first_dropped_at,
                ..
            } => {
                assert_eq!(*missed, 0);
                assert_eq!(*first_dropped_at, None);
            }
            ConnRole::Reqs => panic!("expected Sub role"),
        }
    }

    /// Happy path: `push_response` lands the bytes verbatim when the
    /// queue has room.
    #[test]
    fn push_response_pushes_within_capacity() {
        let (mut conn, _peer) = fresh_conn();
        let bytes = b"{\"kind\":\"ok\"}\n";
        let outcome = conn.push_response(bytes);
        assert_eq!(outcome, PushOutcome::Accepted);
        let queued: Vec<u8> = conn.write_queue.iter().copied().collect();
        assert_eq!(queued, bytes);
        assert!(!conn.close_after_flush);
    }

    /// Over-water response: the queue stays untouched and
    /// `close_after_flush` is armed. The bytes-discarded-on-overflow
    /// arm is the structural fix for the linger bug: a refused
    /// response into a previously-empty queue is paired with
    /// `try_terminate_if_idle` at the Hub call site.
    #[test]
    fn push_response_arms_close_on_capacity_overflow() {
        let (mut conn, _peer) = fresh_conn();
        let huge = vec![b'x'; WRITE_QUEUE_HIGH_WATER + 1];
        let outcome = conn.push_response(&huge);
        assert_eq!(outcome, PushOutcome::Refused);
        assert!(
            conn.write_queue.is_empty(),
            "refused bytes never reach the queue",
        );
        assert!(conn.close_after_flush, "arm fires on overflow");
    }

    /// An already-armed conn (close_after_flush=true) still accepts
    /// further `push_response` calls within capacity — the arm only
    /// gates the conn's future termination, not the in-flight queue
    /// extension. Pins that a single arming path doesn't race-
    /// disable subsequent legitimate responses.
    #[test]
    fn push_response_within_capacity_after_arm_still_succeeds() {
        let (mut conn, _peer) = fresh_conn();
        conn.arm_close_after_flush();
        let bytes = b"{\"kind\":\"ok\"}\n";
        let outcome = conn.push_response(bytes);
        assert_eq!(outcome, PushOutcome::Accepted);
        let queued: Vec<u8> = conn.write_queue.iter().copied().collect();
        assert_eq!(queued, bytes);
    }
}
