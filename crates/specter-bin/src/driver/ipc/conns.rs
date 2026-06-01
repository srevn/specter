//! Per-connection state owned by [`super::hub::Hub`] via
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
//! method on [`ConnState`]. The Hub's kernel-fd surface (listener,
//! per-conn streams, registry clone, accept) keeps its responsibilities;
//! per-conn state-machine mechanics live here as the sole writer set. A future
//! dispatch path that re-implements a capacity gate or a role flip
//! cannot diverge — there is no public field setter to reach for.
//!
//! # Visibility
//!
//! Every export is `pub(in crate::driver)`. The only consumers are
//! [`super::hub::Hub`] (per-conn read/write helpers, fan-out
//! dispatch) and the driver-side IPC handler (the Subscribe arm
//! flips the role through [`ConnState::transition_to_sub`]).

use crate::ipc::framing::encode_line;
use crate::ipc::wire::WireDiagnostic;
use specter_core::SubId;
use std::collections::VecDeque;
use std::time::SystemTime;

/// Per-conn write-queue hard ceiling. No byte path ever pushes past
/// this point — the queue's memory footprint is bounded above by
/// this constant. Matches the framing-level
/// [`crate::ipc::framing::MAX_LINE_BYTES`] cap so a single oversize
/// response (a saturated `list` projection on a busy daemon) wedging
/// the queue has the same backpressure footprint as a hostile read.
/// The two constants own different invariants (framing envelope vs
/// per-conn backpressure) and stay split so a future divergence
/// (chunked diag fan-out with a larger queue, etc.) doesn't have to
/// untangle them.
///
/// # Two-cap discipline
///
/// `WRITE_QUEUE_HIGH_WATER` is the *hard* ceiling. The *soft* cap
/// every legitimate push honours is [`ACCEPT_CAP`], which sits
/// [`RESPONSE_TOO_BIG_RESERVE`] bytes below the hard ceiling. The
/// reserve is sacrosanct: only [`ConnState::push_err_in_reserve`]
/// reaches it, and only with a single structured
/// [`crate::ipc::protocol::WireErrorCode::ResponseTooBig`] Err line.
/// Every other writer ([`ConnState::push_response`],
/// [`ConnState::try_dispatch_diag`]) refuses bytes that would push
/// the queue past `ACCEPT_CAP`. Reaching the hard ceiling is then a
/// structural invariant of the reserve writer — not a runtime check
/// against pathological accumulation.
pub(in crate::driver) const WRITE_QUEUE_HIGH_WATER: usize = 256 * 1024;

/// Bytes held at the top of [`WRITE_QUEUE_HIGH_WATER`] for the
/// structured [`crate::ipc::protocol::WireErrorCode::ResponseTooBig`]
/// Err line. Sized for one render of:
///
/// ```text
/// {"kind":"err","code":"response_too_big",
///  "error":"response of N bytes exceeds per-conn cap of M bytes"}\n
/// ```
///
/// (~120 bytes worst case at u64-sized byte counts) — the 1 KiB
/// figure is 8× headroom against the actual line width, structurally
/// witnessed by the `debug_assert!` in
/// [`ConnState::push_err_in_reserve`]. The reserve is owned by the
/// ResponseTooBig path — no other writer reaches it — so the headroom
/// also absorbs the pathological case where a peer pipelines several
/// over-water requests on one conn and stacks multiple Err lines into
/// the reserve before the first flush.
pub(in crate::driver) const RESPONSE_TOO_BIG_RESERVE: usize = 1024;

/// Soft cap every legitimate write path honours. Bytes above this
/// drop into the structured-Err path
/// ([`ConnState::push_response`]'s Refused arm via the upstream
/// [`super::hub::Hub::enqueue_response`] Refused arm) or the missed-
/// marker accumulator ([`ConnState::try_dispatch_diag`]'s Dropped
/// arm). Computed as
/// [`WRITE_QUEUE_HIGH_WATER`] − [`RESPONSE_TOO_BIG_RESERVE`] so the
/// arithmetic relationship between the two cap surfaces is single-
/// source: bumping either constant moves `ACCEPT_CAP` with it.
pub(in crate::driver) const ACCEPT_CAP: usize = WRITE_QUEUE_HIGH_WATER - RESPONSE_TOO_BIG_RESERVE;

/// Connection state held on [`super::hub::Hub`]'s
/// `conns: BTreeMap<Token, ConnState>`.
///
/// One per accepted operator IPC client. Drops when
/// `Hub::terminate_conn` removes the entry from the map,
/// closing the underlying socket fd and (after explicit
/// `Registry::deregister` at the call site) freeing the mio-side
/// registration.
pub(in crate::driver) struct ConnState {
    /// The mio-wrapped accepted stream. Owned for the lifetime of the
    /// entry; `Drop` closes the fd. Holds mio's `IoSource` wrapping
    /// of the underlying `std::os::unix::net::UnixStream` — the
    /// non-blocking flag is set by mio at accept time, so no extra
    /// `set_nonblocking` is required at this constructor.
    pub(in crate::driver) stream: mio::net::UnixStream,
    /// The mio Token registered against the [`super::hub::Hub`]
    /// Poll registry. Kept on the struct so `Hub::drain_writable`
    /// / `terminate_conn` can reach the registry without re-deriving
    /// the token from the map key.
    pub(in crate::driver) token: mio::Token,
    /// Line accumulator. `Hub::read_conn_into_lines` appends
    /// every `read` chunk here and slices out LF-delimited lines.
    /// Pre-allocated to `1024` bytes — comfortably above the typical
    /// `Status` / `Show` verb size while keeping the cold start
    /// allocation small.
    pub(in crate::driver) read_buf: Vec<u8>,
    /// Pending write residue: response bytes the previous
    /// `drain_writable` could not flush due to kernel-side backpressure,
    /// plus any diag bytes queued by the fan-out path in this tick.
    /// `VecDeque` lets `drain_writable` slice off the consumed prefix
    /// without copying the tail forward.
    pub(in crate::driver) write_queue: VecDeque<u8>,
    /// Conn role — pre-subscribe `Reqs` or post-subscribe `Sub`. The
    /// role gate controls whether the fan-out path delivers diags into
    /// `write_queue`: only `Sub` conns receive them.
    pub(in crate::driver) role: ConnRole,
    /// Set when the next successful flush of `write_queue` should
    /// terminate the conn. Lifecycle:
    ///
    /// - **Oversize input** (the read accumulator exceeds
    ///   [`crate::ipc::framing::MAX_LINE_BYTES`] or a complete line
    ///   crosses the cap): set true via
    ///   [`ConnState::arm_close_after_flush`].
    /// - **Over-cap response** (a verb projection serializes into a
    ///   payload that would push the queue past [`ACCEPT_CAP`]): set
    ///   true via [`ConnState::push_response`]'s overflow arm. The
    ///   structured [`crate::ipc::protocol::WireErrorCode::ResponseTooBig`]
    ///   Err line is queued into the reserve via
    ///   [`ConnState::push_err_in_reserve`] by the upstream
    ///   [`super::hub::Hub::enqueue_response`], so the flush carries
    ///   the operator-actionable signal ahead of the close.
    ///
    /// `drain_writable` observes the flag when the queue empties via a
    /// successful flush; the Hub-level `try_terminate_if_idle` handles
    /// the queue-empty-at-arm case (where no WRITABLE edge would ever
    /// run drain_writable). Together the two paths cover both arming
    /// shapes: bytes-queued → flush-then-terminate, empty-queue →
    /// terminate-inline.
    pub(in crate::driver) close_after_flush: bool,
}

/// Per-conn role axis. A fresh conn enters [`ConnRole::Reqs`]; the
/// Subscribe verb's handler flips it to [`ConnRole::Sub`] via
/// [`ConnState::transition_to_sub`] *after* the SubscribeAck bytes
/// are already in the write_queue — the ack-before-fanout ordering
/// the wire-side regression test pins.
pub(in crate::driver) enum ConnRole {
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
        /// Back-pressure marker window — count + start-of-window
        /// timestamp pair, or the closed state. Flushed lazily as a
        /// `WireDiagnostic::Missed` line before the next dispatched
        /// diag so causal order is preserved on the wire. Mutated
        /// exclusively through [`MissedWindow::record_drop`] (capacity
        /// refusal) and [`MissedWindow::take`] (marker-flushed edge);
        /// the sum type makes the prior two-field invariant
        /// (`count > 0 ⇒ since.is_some()`) structurally unrepresentable.
        missed: MissedWindow,
    },
}

/// Back-pressure marker window. Either no drops are pending since the
/// last flush ([`MissedWindow::Closed`]), or one-or-more diags have
/// been dropped since `since`, the wall-clock the `0 → 1` transition
/// captured ([`MissedWindow::Open`]).
///
/// The sum type makes the impossible state (`count > 0 ∧
/// since.is_none()`) structurally unrepresentable — the
/// `count > 0 ⇒ since.is_some()` invariant is enforced by the type,
/// not a rustdoc rule, so [`ConnState::try_dispatch_diag`] needs no
/// defensive `unwrap_or(at)` at marker construction.
///
/// Operators reading a `_missed` marker see start-of-window time
/// (when the drops began), not flush time (when the daemon got
/// around to mentioning them) — the marker's `at` is the captured
/// `since`, not the dispatch-time stamp.
#[derive(Debug, Default, Eq, PartialEq)]
pub(in crate::driver) enum MissedWindow {
    /// No drops pending since the last successful marker flush.
    #[default]
    Closed,
    /// `count` diags dropped since `since`, the wall-clock captured
    /// at the 0→1 transition. `count` is `saturating_add`-bumped on
    /// every subsequent drop; a wedged subscriber dropping `2^32`
    /// events would saturate at `u32::MAX`, though in practice the
    /// per-conn write-side disconnect fires long before.
    Open {
        /// Number of diags dropped in the open window.
        count: u32,
        /// Wall-clock of the first drop. Threaded as the marker's
        /// `at` when the window flushes — operators see when drops
        /// began, not flush time.
        since: SystemTime,
    },
}

impl MissedWindow {
    /// Record one capacity-refused diag. The `Closed → Open` edge
    /// captures `at` as `since`; subsequent drops in the same window
    /// `saturating_add`-bump `count` and preserve `since`.
    pub(in crate::driver) const fn record_drop(&mut self, at: SystemTime) {
        match self {
            Self::Closed => {
                *self = Self::Open {
                    count: 1,
                    since: at,
                }
            }
            Self::Open { count, .. } => *count = count.saturating_add(1),
        }
    }

    /// Take the open window's `(count, since)` and reset to
    /// [`MissedWindow::Closed`]. Returns `None` when no window was
    /// open. Callers that only need the reset (the marker bytes
    /// were pre-built ahead of the capacity gate) may `let _ = .take()`.
    pub(in crate::driver) const fn take(&mut self) -> Option<(u32, SystemTime)> {
        match std::mem::replace(self, Self::Closed) {
            Self::Closed => None,
            Self::Open { count, since } => Some((count, since)),
        }
    }
}

/// Outcome of one [`ConnState::try_dispatch_diag`] attempt. The Hub's
/// fan-out loop discards the value (per-conn missed accounting is
/// internal to the method); the discriminant is exposed for unit-test
/// observation of the axis the call took.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(in crate::driver) enum DispatchOutcome {
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
pub(in crate::driver) enum PushOutcome {
    /// Bytes pushed; the caller may want to arm WRITABLE interest
    /// (done in bulk by [`super::hub::Hub::arm_writable_interests`]
    /// at end of tick).
    Accepted,
    /// Bytes did NOT fit; `close_after_flush` was armed internally.
    /// The Hub-level caller pairs this with
    /// [`super::hub::Hub::try_terminate_if_idle`] so a refusal
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
    pub(in crate::driver) fn new(stream: mio::net::UnixStream, token: mio::Token) -> Self {
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
    /// gate lives at the [`super::dispatch`] Subscribe handler — a
    /// repeat Subscribe on a `Sub` conn returns
    /// [`crate::ipc::protocol::WireErrorCode::AlreadySubscribed`]
    /// before reaching this method. The `debug_assert!` here is the
    /// contract witness: any future caller that bypasses the handler
    /// gate fails loudly in debug builds.
    pub(in crate::driver) fn transition_to_sub(&mut self, filter: Option<SubId>) {
        debug_assert!(
            matches!(self.role, ConnRole::Reqs),
            "transition_to_sub on non-Reqs role — the Subscribe handler gate is bypassed",
        );
        self.role = ConnRole::Sub {
            filter,
            missed: MissedWindow::Closed,
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
    /// [`super::hub::Hub::try_terminate_if_idle`] at the right
    /// moment — typically *after* the in-flight processing pass that
    /// may have pushed response bytes into the queue.
    pub(in crate::driver) const fn arm_close_after_flush(&mut self) {
        self.close_after_flush = true;
    }

    /// Try to enqueue `diag_line` (one LF-terminated JSON line) into
    /// this conn's write_queue.
    ///
    /// Sole mutator of [`ConnRole::Sub::missed`]; the open-window
    /// invariant lives on [`MissedWindow`] itself, so this method's
    /// concern is the dispatch / capacity verdict, not the field shape.
    ///
    /// Axes evaluated in order:
    /// 1. `close_after_flush` → `Skipped` (closing conn doesn't
    ///    accumulate fresh diags).
    /// 2. `role != Sub` → `Skipped` (pre-subscribe conn).
    /// 3. Per-Sub `filter` mismatch → `Skipped`.
    /// 4. Capacity gate: combined `(queue_len + marker_bytes + diag_line)`
    ///    must fit under [`ACCEPT_CAP`]. The diag fan-out path
    ///    honours the response-cap reserve too — a streaming diag
    ///    that could fill the reserve would deny the next
    ///    [`crate::ipc::protocol::WireErrorCode::ResponseTooBig`] Err
    ///    its guaranteed-fit space. When the combined size does not
    ///    fit, the diag drops and [`MissedWindow::record_drop`] bumps
    ///    the window — opening one on the `Closed → Open` edge or
    ///    `saturating_add`-bumping `count` on a subsequent drop.
    /// 5. Otherwise: flush any pending Missed marker (carrying the
    ///    open window's `since` as the marker's `at` so operators see
    ///    start-of-window time, not flush time), reset the window via
    ///    [`MissedWindow::take`], then push the diag bytes.
    ///
    /// `at` threads from the caller so every conn observes a
    /// byte-identical timestamp for one engine emission; the marker
    /// path uses the open window's captured `since` instead.
    pub(in crate::driver) fn try_dispatch_diag(
        &mut self,
        diag_line: &[u8],
        diag_sub: Option<SubId>,
        at: SystemTime,
    ) -> DispatchOutcome {
        if self.close_after_flush {
            return DispatchOutcome::Skipped;
        }
        let ConnRole::Sub { filter, missed } = &mut self.role else {
            return DispatchOutcome::Skipped;
        };
        if let Some(want) = filter
            && diag_sub != Some(*want)
        {
            return DispatchOutcome::Skipped;
        }

        // Pre-build the marker line if a missed window is open. The
        // sum type guarantees `since` is present whenever `count > 0` —
        // no defensive fallback is needed.
        let marker_bytes: Option<Vec<u8>> = match missed {
            MissedWindow::Open { count, since } => Some(encode_line(&WireDiagnostic::Missed {
                at: (*since).into(),
                count: *count,
            })),
            MissedWindow::Closed => None,
        };

        let marker_len = marker_bytes.as_ref().map_or(0, Vec::len);
        let queue_len = self.write_queue.len();
        let combined = queue_len
            .saturating_add(marker_len)
            .saturating_add(diag_line.len());
        if combined > ACCEPT_CAP {
            missed.record_drop(at);
            return DispatchOutcome::Dropped;
        }

        if let Some(mb) = marker_bytes {
            self.write_queue.extend(mb);
            let _ = missed.take();
        }
        self.write_queue.extend(diag_line);
        DispatchOutcome::Accepted
    }

    /// Push response bytes into the write_queue.
    ///
    /// Capacity-gated on the soft cap: if the projected queue length
    /// would exceed [`ACCEPT_CAP`] (the soft cap, sitting
    /// [`RESPONSE_TOO_BIG_RESERVE`] bytes below
    /// [`WRITE_QUEUE_HIGH_WATER`]), the queue is left untouched and
    /// `close_after_flush` is armed via
    /// [`Self::arm_close_after_flush`]. The Hub-level
    /// [`super::hub::Hub::enqueue_response`] pairs the `Refused`
    /// outcome with [`Self::push_err_in_reserve`] (queueing the
    /// structured `ResponseTooBig` Err into the reserve so operators
    /// see an actionable signal ahead of the close) and then
    /// [`super::hub::Hub::try_terminate_if_idle`] (a no-op once the
    /// Err line populated the queue — the flush-then-terminate path
    /// on the next WRITABLE drain handles teardown).
    pub(in crate::driver) fn push_response(&mut self, bytes: &[u8]) -> PushOutcome {
        if self.write_queue.len().saturating_add(bytes.len()) > ACCEPT_CAP {
            self.arm_close_after_flush();
            return PushOutcome::Refused;
        }
        self.write_queue.extend(bytes);
        PushOutcome::Accepted
    }

    /// Append the pre-encoded `ResponseTooBig` Err line bytes into the
    /// per-conn reserve unconditionally. Total fn — the structural
    /// invariant the [`ACCEPT_CAP`] / [`RESPONSE_TOO_BIG_RESERVE`]
    /// split encodes guarantees the reserve has room regardless of
    /// the prior queue state, so refusal is unrepresentable.
    ///
    /// Sole caller: [`super::hub::Hub::enqueue_response`]'s Refused
    /// arm — paired with the upstream [`Self::push_response`] refusal
    /// that already armed `close_after_flush`. The Hub builds the
    /// `ResponsePayload::Err { code: ResponseTooBig, error: ... }`,
    /// encodes the line, and calls through here; this method owns
    /// the reserve-write half of the contract and stays unaware of
    /// the wire vocabulary so the cap arithmetic lives single-source
    /// alongside the rest of the per-conn state.
    ///
    /// # Invariant (debug-asserted)
    ///
    /// `bytes.len() <= RESPONSE_TOO_BIG_RESERVE`. The current Err
    /// shape (`{"kind":"err","code":"response_too_big","error":"response
    /// of N bytes exceeds per-conn cap of M bytes"}\n`) is ~120 bytes
    /// at u64-sized counts; the 1 KiB reserve carries 8× headroom.
    /// A future Err rendering that grew past the reserve would trip
    /// the assert in debug builds, prompting the operator to widen
    /// `RESPONSE_TOO_BIG_RESERVE` rather than silently overrun
    /// [`WRITE_QUEUE_HIGH_WATER`].
    pub(in crate::driver) fn push_err_in_reserve(&mut self, bytes: &[u8]) {
        debug_assert!(
            bytes.len() <= RESPONSE_TOO_BIG_RESERVE,
            "ResponseTooBig Err line ({} bytes) exceeds RESPONSE_TOO_BIG_RESERVE ({} bytes); \
             widen the reserve",
            bytes.len(),
            RESPONSE_TOO_BIG_RESERVE,
        );
        self.write_queue.extend(bytes);
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
            Self::Sub { filter, missed } => f
                .debug_struct("Sub")
                .field("filter", filter)
                .field("missed", missed)
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
    /// accounting (filter unset, missed window closed) to the
    /// fresh-window defaults. Pins the post-flip shape.
    #[test]
    fn transition_to_sub_initializes_sub_state() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        match conn.role {
            ConnRole::Sub { filter, missed } => {
                assert_eq!(filter, None);
                assert_eq!(missed, MissedWindow::Closed);
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
    /// [`ACCEPT_CAP`] drops, bumping `missed` to 1. No
    /// bytes leak into the queue.
    #[test]
    fn try_dispatch_diag_drops_and_bumps_missed_on_capacity_overflow() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        // Pre-fill near the soft cap so a small diag overflows.
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP - 10));
        let queue_len_before = conn.write_queue.len();
        let line: &[u8] = b"{\"diag\":\"too_big_to_fit_in_remaining_capacity\"}\n";
        assert!(line.len() > 10, "test precondition");

        let outcome = conn.try_dispatch_diag(line, None, SystemTime::UNIX_EPOCH);
        assert_eq!(outcome, DispatchOutcome::Dropped);
        assert_eq!(conn.write_queue.len(), queue_len_before, "queue untouched");
        match &conn.role {
            ConnRole::Sub { missed, .. } => {
                assert_eq!(
                    *missed,
                    MissedWindow::Open {
                        count: 1,
                        since: SystemTime::UNIX_EPOCH,
                    },
                );
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
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP - 5));
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
            ConnRole::Sub { missed, .. } => {
                assert_eq!(
                    *missed,
                    MissedWindow::Open {
                        count: 3,
                        since: at1,
                    },
                    "open window pins count to 3 and since to the 0→1 transition's at",
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
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP - 5));
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
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP - 5));
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
    /// crosses [`ACCEPT_CAP`], the dispatch drops without
    /// pushing the marker. `missed` re-accumulates; the marker tries
    /// again on the next dispatch.
    #[test]
    fn try_dispatch_diag_drops_when_marker_plus_diag_does_not_fit() {
        let (mut conn, _peer) = fresh_conn();
        conn.transition_to_sub(None);
        // Drive one drop to open the marker window.
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP - 5));
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
            ConnRole::Sub { missed, .. } => {
                assert_eq!(
                    *missed,
                    MissedWindow::Open {
                        count: 2,
                        since: at_drop,
                    },
                    "open window count re-accumulates on combined-overflow; \
                     since sticks to the original 0→1 capture",
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
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP - 5));
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
            ConnRole::Sub { missed, .. } => {
                assert_eq!(
                    *missed,
                    MissedWindow::Closed,
                    "window resets to Closed on the flush edge",
                );
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
        let huge = vec![b'x'; ACCEPT_CAP + 1];
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

    /// `push_err_in_reserve` extends the write_queue unconditionally —
    /// the reserve carries no soft-cap gate. The pre-existing queue
    /// state is preserved; the new bytes append at the tail. Pins the
    /// total-fn contract `Hub::enqueue_response`'s Refused arm depends
    /// on (the reserve invariant
    /// `ACCEPT_CAP + RESPONSE_TOO_BIG_RESERVE = WRITE_QUEUE_HIGH_WATER`
    /// makes refusal unrepresentable).
    #[test]
    fn push_err_in_reserve_extends_queue_unconditionally() {
        let (mut conn, _peer) = fresh_conn();
        // Pre-fill at exactly the soft cap — push_err_in_reserve must
        // still succeed because the reserve sits above ACCEPT_CAP.
        conn.write_queue
            .extend(std::iter::repeat_n(b'x', ACCEPT_CAP));
        let err_line: &[u8] = b"{\"kind\":\"err\",\"code\":\"response_too_big\",\"error\":\"x\"}\n";
        assert!(
            err_line.len() <= RESPONSE_TOO_BIG_RESERVE,
            "sample Err line fits the reserve invariant",
        );
        conn.push_err_in_reserve(err_line);
        assert_eq!(
            conn.write_queue.len(),
            ACCEPT_CAP + err_line.len(),
            "Err line appended verbatim past the soft cap into the reserve",
        );
    }

    /// In debug builds, an Err line wider than [`RESPONSE_TOO_BIG_RESERVE`]
    /// trips the [`ConnState::push_err_in_reserve`] `debug_assert!`.
    /// Pins the structural witness that the reserve sizing assumption
    /// is checked in lockstep with the rendered Err shape — a future
    /// rendering that grew past the reserve would fail this test loudly,
    /// prompting the operator to widen `RESPONSE_TOO_BIG_RESERVE` before
    /// silently overrunning `WRITE_QUEUE_HIGH_WATER` in release.
    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "exceeds RESPONSE_TOO_BIG_RESERVE")]
    fn push_err_in_reserve_oversize_panics_in_debug() {
        let (mut conn, _peer) = fresh_conn();
        let oversize = vec![b'x'; RESPONSE_TOO_BIG_RESERVE + 1];
        conn.push_err_in_reserve(&oversize);
    }
}
