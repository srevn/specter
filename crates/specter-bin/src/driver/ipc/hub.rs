//! [`Hub`] — owner of the operator-IPC kernel-fd surface:
//! the bound listener, the per-conn map, the per-conn Token
//! allocator, and the [`Registry`] clone the conn lifecycle uses for
//! register / deregister.
//!
//! Constructed once by `App::run` from the [`Registry`] clone
//! minted via [`crate::driver::Reactor::registry_clone`]. Owned by
//! [`crate::driver::EngineDriver`] for the rest of the daemon's lifetime,
//! dropped BEFORE [`crate::driver::Reactor`] so the explicit `Drop` impl
//! below can deregister listener + every live conn stream against a
//! still-live Poll selector.
//!
//! # Drop order
//!
//! Field order on [`Hub`] is the drop order, and the explicit
//! [`Drop`] impl runs before any field-order drop touches a Source.
//! The discipline:
//!
//! 1. **`Drop::drop` runs first** — `Registry::deregister` for the
//!    listener and every live conn stream. The Reactor's Poll
//!    selector is still live at this moment (the
//!    [`crate::driver::EngineDriver`] field order places `ipc` before
//!    `reactor`), so the deregister calls succeed against the
//!    underlying selector via this server's [`Registry`] clone.
//!    Errors are best-effort: `NotFound` is benign on a stream whose
//!    fd was already closed by a prior `terminate_conn`.
//! 2. **`listener`** drops — closes the bound socket fd; any racing
//!    client connect that observed the listener now sees the socket
//!    gone before any other server state is torn down.
//! 3. **`conns`** drops — each [`super::conns::ConnState`]'s stream
//!    Drop fires; the corresponding mio-side registration was
//!    explicitly deregistered above so a `ConnState` reaching Drop
//!    without going through `terminate_conn` does not strand a
//!    registration. The conn map is the only path that removes
//!    entries — the audit grep `conns.remove\|conns.drain` returns
//!    only [`Hub::terminate_conn`].
//! 4. **`registry`** drops — the [`Registry::try_clone()`] handle
//!    releases one Arc-reference on the underlying selector. The
//!    Reactor's Poll still holds the other reference; the selector
//!    closes when Reactor drops.
//! 5. **`next_conn_token`** drops — a plain `usize`; no resource
//!    implications.
//!
//! # Visibility
//!
//! Every export is `pub(in crate::driver)` or `pub(crate)`. The crate-visible
//! surface is [`Hub::new`] (called by `App::run`); every other
//! method is `pub(in crate::driver)` — only the surrounding `driver` module
//! reaches them. `tick.rs` drives [`Hub::drain_accept`],
//! [`Hub::drain_writable`], [`Hub::terminate_conn`],
//! [`Hub::arm_writable_interests`]; the IPC verb handler
//! reaches per-conn helpers; `forward.rs` drives
//! [`Hub::dispatch_to_subscribers`].

use super::conns::{self, ConnRole, ConnState, PushOutcome};
use crate::ipc::framing::{InfallibleSerialize, MAX_LINE_BYTES, encode_line};
use crate::ipc::protocol::{ResponsePayload, WireErrorCode};
use crate::ipc::wire::{WireDiagnostic, WireTime};
use mio::{Interest, Registry, Token};
use specter_core::{Diagnostic, SubId};
use std::collections::BTreeMap;
use std::io;
use std::time::SystemTime;

/// First per-conn Token value. Chosen well above the static set so a
/// new static Source can be added (Token(5..0xFF) are reserved) without
/// migrating per-conn allocation; the gap also makes a debug-mode
/// `assert!(token.0 >= TOKEN_CONN_BASE)` in per-conn arms a cheap
/// structural check against accidental static-token aliasing.
///
/// Defined here (not in `reactor.rs`) because allocation is an
/// Hub concern; the reactor dispatch arm imports it for the
/// per-conn catch-all match.
pub(in crate::driver) const TOKEN_CONN_BASE: usize = 0x100;

/// Concurrent IPC client cap. Bound by the operator hand-control
/// envelope (one engineer typically holds ≤2 sessions: a `tail` and
/// a verb shell); the cap survives mostly as a DoS floor against a
/// misbehaving client opening connections in a loop.
pub(in crate::driver) const MAX_IPC_CONNS: usize = 8;

/// Owner of the operator-IPC kernel-fd surface. See module rustdoc
/// for the drop-order discipline the field order + explicit [`Drop`]
/// impl encode.
pub(crate) struct Hub {
    /// The bound operator-IPC socket. Drops after the explicit
    /// `Drop::drop` deregisters it; closing the bind fd then means
    /// any racing client connect sees the socket gone before any
    /// per-conn teardown reaches the kernel.
    listener: mio::net::UnixListener,
    /// Per-conn state map. `BTreeMap` over `HashMap`: the conn count
    /// is small (≤[`MAX_IPC_CONNS`]) and `BTreeMap` carries no random
    /// state for the iteration order — the dispatch loop walks in
    /// Token order, which makes test assertions deterministic without
    /// extra sort calls.
    conns: BTreeMap<Token, ConnState>,
    /// Cloned [`Registry`] handle pointing at the same underlying
    /// selector as the Reactor's Poll. Used for per-conn `register` /
    /// `reregister` / `deregister`; the clone lets the per-call
    /// ergonomics stay clean (no `&Registry` parameter threading).
    /// Dropped after `conns` so the deregister calls in `Drop::drop`
    /// have a live registry to reach.
    registry: Registry,
    /// Monotone counter for fresh per-conn Token allocation. Starts
    /// at [`TOKEN_CONN_BASE`]; [`Self::allocate_conn_token`] panics
    /// loudly via `checked_add` on the unreachable `usize` overflow
    /// edge — a silent wrap would alias a new accept against the
    /// static-token set and misroute kernel events.
    next_conn_token: usize,
}

/// Outcome of one [`Hub::enqueue_response`] call.
///
/// The Hub-side wrapper around
/// [`super::conns::ConnState::push_response`] threads the
/// per-conn capacity verdict back to the caller along with a
/// "conn-not-in-map" signal — the IPC handler's Subscribe arm needs
/// to know whether the ack actually landed before flipping the role.
/// Other handlers (`Reload`, `Disable`, `Enable`, `Absorb`, projection
/// paths) `let _ = ...` the outcome benignly: a refused or gone conn is
/// already on the path to termination, and re-acking is pointless.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(in crate::driver) enum EnqueueOutcome {
    /// Bytes pushed into the write_queue; will flush on the next
    /// WRITABLE drain.
    Accepted,
    /// Response did not fit under the per-conn accept cap. The arm
    /// emitted a structured
    /// [`crate::ipc::protocol::WireErrorCode::ResponseTooBig`] Err
    /// line into the per-conn reserve and armed `close_after_flush`;
    /// the next [`Hub::drain_writable`] flushes the Err (and any
    /// prior queued bytes) and terminates the conn at the flush-empty
    /// edge.
    Refused,
    /// `token` is not in the conn map — the caller addressed a conn
    /// that closed between an earlier point in this tick and the
    /// enqueue (a read drain that observed EOF terminated it, or a
    /// write failure removed it). Discriminated from `Refused` so the
    /// Subscribe handler can avoid a no-op role flip on a gone conn.
    ConnGone,
}

/// Outcome of one [`Hub::read_conn_into_lines`] call.
///
/// The two variants distinguish the termination semantics the read
/// drain triggers:
///
/// - `Continue` — the read end is alive (or pending). The caller
///   processes any drained lines and then pairs with
///   [`Hub::try_terminate_if_idle`] in case an oversize-line
///   guard armed `close_after_flush` against an empty queue.
/// - `PeerGone` — peer EOF or a non-recoverable read transport error.
///   The caller terminates the conn unconditionally; any pending
///   write-queue bytes are wasted because the peer's read end has
///   closed.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(in crate::driver) enum ReadOutcome {
    Continue,
    PeerGone,
}

/// Outcome of one [`Hub::drain_writable`] call.
///
/// Three discriminants — the tick's WRITABLE pass needs to know not
/// only "does this conn keep going" but also "did the conn vanish
/// mid-tick" (the read drain may have terminated it earlier this tick,
/// in which case the WRITABLE pass's reach is a benign no-op).
///
/// The three discriminants are:
/// - `Continue` ⇒ keep going,
/// - `Terminate` ⇒ caller terminates,
/// - `ConnGone` ⇒ conn already gone.
///
/// Lifting the three states into one enum makes the tick-side match
/// total without an `io::ErrorKind::NotFound` arm masquerading as a
/// transport error. Any other transport error folds into `Terminate`
/// — the conn is closed by the same path either way.
#[must_use]
#[derive(Debug, Eq, PartialEq)]
pub(in crate::driver) enum DrainWritableOutcome {
    /// Queue still has bytes, OR queue empty without close-flag. The
    /// tick keeps the conn open; mio re-arming happens at end-of-tick
    /// via [`Hub::arm_writable_interests`].
    Continue,
    /// Queue empty + `close_after_flush` set, OR a transport-level
    /// write error. The caller terminates the conn via
    /// [`Hub::terminate_conn`].
    Terminate,
    /// `token` no longer in the conn map — the read drain earlier
    /// this tick already terminated it. The caller silently skips
    /// (the mio-side registration is gone too).
    ConnGone,
}

impl Hub {
    /// Construct the server from a bound listener + the Registry
    /// clone the Reactor handed back.
    ///
    /// `listener` is the bound `std::os::unix::net::UnixListener` from
    /// [`crate::ipc::sockpath::bind_socket_atomic`]; we re-wrap it
    /// into mio's flavor after setting non-blocking. `registry` is
    /// the [`Registry::try_clone()`] handle minted via
    /// [`crate::driver::Reactor::registry_clone`]; it shares the
    /// underlying selector with the Reactor's Poll.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `set_nonblocking` or the
    /// listener registration — both are programmer-error or
    /// kernel-pressure failures. The caller treats any error as
    /// startup-fatal.
    pub(crate) fn new(
        listener: std::os::unix::net::UnixListener,
        registry: Registry,
    ) -> io::Result<Self> {
        // Listener: must be non-blocking before mio observes it
        // (mio's `from_std` does not set the flag), or the next
        // `accept()` would block the reactor thread on an empty
        // listen queue.
        listener.set_nonblocking(true)?;
        let mut listener = mio::net::UnixListener::from_std(listener);
        registry.register(
            &mut listener,
            super::super::reactor::TOKEN_LISTENER,
            Interest::READABLE,
        )?;
        Ok(Self {
            listener,
            conns: BTreeMap::new(),
            registry,
            next_conn_token: TOKEN_CONN_BASE,
        })
    }

    /// Accept every pending connection in this tick, bounded above by
    /// `MAX_IPC_CONNS * 2` iterations (see [`MAX_IPC_CONNS`]).
    ///
    /// Edge-triggered: loops `accept(2)` non-blockingly. The natural
    /// termination is `WouldBlock` (accept queue empty); the per-tick
    /// cap is the DoS guard for the alternative case where a hostile
    /// peer keeps the queue continuously non-empty.
    ///
    /// # Per-tick cap
    ///
    /// The bound is `MAX_IPC_CONNS * 2`: enough headroom for
    /// [`MAX_IPC_CONNS`] legitimate accepts (filling the conn map)
    /// plus the same again of cap-busy refusals stacked behind them in
    /// the listen queue. A peer pushing accepts faster than we drain
    /// would otherwise starve every other Source on the reactor; this
    /// bound forces a yield to the rest of the tick's drain pass after
    /// at most `2 * MAX_IPC_CONNS` accepts.
    ///
    /// # Cap-hit re-arm (mio EPOLLET / EV_CLEAR contract)
    ///
    /// Returning from the loop short of `WouldBlock` leaves the kernel-
    /// side accept queue non-empty. Under edge-triggered mio (EPOLLET
    /// on Linux, EV_CLEAR on kqueue — see `mio` 1.2.0
    /// `sys/unix/selector/{epoll,kqueue}.rs`), no future
    /// `TOKEN_LISTENER` edge fires until the queue transitions
    /// empty→non-empty; subsequent SYNs into a non-empty queue do NOT
    /// re-fire. To preserve forward progress, the cap-hit path
    /// re-registers the listener: Linux `EPOLL_CTL_MOD` calls
    /// `ep_modify` which re-checks current readiness and queues a
    /// wake-up if ready; kqueue's `EV_ADD | EV_CLEAR` re-arms with
    /// identical semantics. The next `poll` re-fires `TOKEN_LISTENER`
    /// and the next tick drains another batch.
    ///
    /// # Per-conn cap (`MAX_IPC_CONNS`)
    ///
    /// Independent of the per-tick bound: on reaching
    /// [`MAX_IPC_CONNS`] live conns, extra accepts receive a one-shot
    /// non-blocking best-effort write of a structured
    /// [`WireErrorCode::Busy`] JSON line; then the stream drops. The
    /// stream from [`mio::net::UnixListener::accept`] is already
    /// non-blocking (mio's `IoSource` convention), so `write(2)` here
    /// either lands the ~80 bytes into the kernel send buffer
    /// (healthy peer, microseconds) or returns `WouldBlock` (hostile
    /// peer with a wedged receive buffer) and we drop without bytes
    /// on the wire. Either outcome preserves the reactor's
    /// no-blocking floor — no `set_nonblocking(false)` flip, no
    /// kernel-level `SO_SNDTIMEO`, no blocking-write fallback that
    /// could stall the accept loop.
    ///
    /// The cap rejects rather than queues — operator IPC is not
    /// throughput-sensitive, and a queue would let a misbehaving
    /// client wedge the daemon's resource budget.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `register` (mio programmer-error;
    /// always startup-fatal on a fresh fd) and from the cap-hit
    /// `reregister` (same class — a failure here means the listener
    /// fd is gone, which is structurally unrecoverable). `accept()`
    /// errors other than `WouldBlock` propagate too — under normal
    /// operation these are `ECONNABORTED` (client closed between SYN
    /// and accept), which is rare enough to be terminal here.
    pub(in crate::driver) fn drain_accept(&mut self) -> io::Result<()> {
        for _ in 0..MAX_IPC_CONNS.saturating_mul(2) {
            match self.listener.accept() {
                Ok((mut stream, _addr)) => {
                    if self.conns.len() >= MAX_IPC_CONNS {
                        // Cap reached. One non-blocking write attempt
                        // against the already-non-blocking mio stream;
                        // discard the result (WouldBlock against a
                        // hostile peer is fine — the stream drops next
                        // either way). The healthy-peer wire surface
                        // stays a structured Busy line; the hostile-peer
                        // path takes no bytes but preserves the
                        // reactor's no-blocking floor.
                        use std::io::Write;
                        let resp = ResponsePayload::Err {
                            code: WireErrorCode::Busy,
                            error: "max concurrent connections".into(),
                        };
                        let _ = stream.write(&encode_line(&resp));
                        drop(stream);
                        continue;
                    }
                    let token = self.allocate_conn_token();
                    let mut conn = ConnState::new(stream, token);
                    self.registry
                        .register(&mut conn.stream, token, Interest::READABLE)?;
                    self.conns.insert(token, conn);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => return Ok(()),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => return Err(e),
            }
        }
        // Hit the per-tick bound without reaching `WouldBlock`. Under
        // edge-triggered mio the listener won't re-fire until the
        // accept queue transitions empty→non-empty; force a fresh
        // readiness evaluation via `reregister` so the next poll
        // re-fires `TOKEN_LISTENER` and the next tick drains another
        // batch. See method rustdoc for the kernel-level rationale.
        self.registry.reregister(
            &mut self.listener,
            super::super::reactor::TOKEN_LISTENER,
            Interest::READABLE,
        )
    }

    /// Mint a fresh per-conn [`Token`]. Monotone from
    /// [`TOKEN_CONN_BASE`].
    ///
    /// `checked_add` is loud on the unreachable overflow edge:
    /// `usize::MAX - TOKEN_CONN_BASE` accepts is &gt;10^18 on 64-bit, so
    /// hitting the panic requires either an inflation bug at the
    /// caller (the accept loop somehow runs without ever
    /// terminating a conn) or hardware running long enough to outlive
    /// every realistic deployment. A loud panic on the unreachable
    /// edge beats a silent wrap-and-alias against the static token
    /// set (which would collide a new accept with `TOKEN_WATCHER` /
    /// `TOKEN_SIGNAL` and route fs events into the conn dispatch
    /// arm).
    const fn allocate_conn_token(&mut self) -> Token {
        let raw = self.next_conn_token;
        self.next_conn_token = raw
            .checked_add(1)
            .expect("conn token allocation overflowed usize — unreachable in practice");
        Token(raw)
    }

    /// Serialize a [`Diagnostic`] once and append its JSON line to
    /// every subscriber conn's write_queue.
    ///
    /// Subscriber storage IS the per-conn map: every conn whose role
    /// is [`super::conns::ConnRole::Sub`] is a subscriber,
    /// the per-Sub filter and back-pressure marker accumulator live
    /// on the role payload, and the line bytes are pushed directly
    /// into [`ConnState::write_queue`] on the reactor thread — no
    /// envelope, no worker-thread re-serialize hop.
    ///
    /// **`diag_sub` is caller-computed** — the projection
    /// `&Diagnostic → Option<SubId>` is an exhaustive walk of the
    /// core enum that lives next to its only call site
    /// ([`crate::driver::forward::diag_sub_id`]); this method takes
    /// the resolved value to keep the Diagnostic-walking concern out
    /// of the Hub module.
    ///
    /// **Time threads through two channels.** `wire_at` is the
    /// pre-formatted RFC 3339 token the per-diag construction reuses
    /// across the StepOutput (built once in
    /// [`crate::driver::EngineDriver::forward_diagnostics`], passed
    /// by reference here so the `humantime::format_rfc3339_seconds`
    /// allocation is amortized over the whole batch). `at` is the
    /// full-precision [`SystemTime`] that
    /// [`ConnState::try_dispatch_diag`] needs for the per-conn
    /// `first_dropped_at` back-pressure accounting — distinct
    /// timestamps (the marker's own `WireTime` is built from
    /// `first_dropped_at`, not `wire_at`).
    ///
    /// **One serialize per dispatch.** The JSON bytes are built once
    /// before the conn loop and appended verbatim per subscriber via
    /// [`super::conns::ConnState::try_dispatch_diag`], which
    /// owns the per-conn five-axis verdict (close, role, filter,
    /// capacity, marker flush).
    ///
    /// **Interest re-arming is deferred** to
    /// [`Self::arm_writable_interests`], which runs once at the end
    /// of the tick's drain pass. Per-tick byte pushes therefore
    /// translate into one interest rearm per ready conn rather than
    /// N (one per `dispatch_to_subscribers` call) — the
    /// re-registration syscall amortizes across the whole tick.
    ///
    /// **Subscriber-empty short-circuit is defensive.** The hot path
    /// already gates fan-out at the StepOutput-scoped caller
    /// ([`crate::driver::EngineDriver::forward_diagnostics`] reads
    /// [`Self::has_any_subscriber`] once per emission and skips the
    /// per-diag `WireTime` / `diag_sub_id` work entirely). The inner
    /// check below catches a future caller that bypasses the outer
    /// gate; the cost is one `O(MAX_IPC_CONNS)` discriminator walk
    /// before bailing, well below the `WireDiagnostic::from` +
    /// `encode_line` build it stands in front of.
    pub(in crate::driver) fn dispatch_to_subscribers(
        &mut self,
        diag: &Diagnostic,
        at: SystemTime,
        wire_at: &WireTime,
        diag_sub: Option<SubId>,
    ) {
        if !self.has_any_subscriber() {
            return;
        }
        let wire = WireDiagnostic::from((diag, wire_at));
        let line = encode_line(&wire);
        for conn in self.conns.values_mut() {
            // The five-axis verdict (close, role, filter, capacity,
            // marker dance) and the per-conn missed-window
            // bookkeeping live on ConnState — the Hub's job
            // here is serialize-once + iterate.
            let _ = conn.try_dispatch_diag(&line, diag_sub, at);
        }
    }

    /// `true` iff at least one conn in the map is in
    /// [`super::conns::ConnRole::Sub`] role — i.e., at least one
    /// operator `tail` / `wait` session is live and would receive
    /// fan-out lines from [`Self::dispatch_to_subscribers`].
    ///
    /// Composed by
    /// [`crate::driver::EngineDriver::forward_diagnostics`] once per
    /// `StepOutput` so the no-subscriber common path skips the
    /// per-emission [`SystemTime::now`] / [`WireTime::from`] /
    /// [`crate::driver::forward::diag_sub_id`] work entirely.
    /// The walk is bounded above by [`MAX_IPC_CONNS`] enum-discriminator
    /// reads — well below the syscall + humantime allocation the
    /// outer gate would otherwise pay per emission.
    ///
    /// [`Self::dispatch_to_subscribers`] retains the same predicate
    /// as a defensive inner short-circuit; the accessor keeps the
    /// rule single-source so the two surfaces cannot drift.
    pub(in crate::driver) fn has_any_subscriber(&self) -> bool {
        self.conns
            .values()
            .any(|c| matches!(c.role, ConnRole::Sub { .. }))
    }

    /// Borrow a per-conn state mutably by [`Token`]. Returns `None`
    /// if the conn closed between the tick's drain and the caller's
    /// reach (e.g., an oversize line set `close_after_flush`, the
    /// flush completed, and `terminate_conn` ran).
    ///
    /// Used by the IPC dispatcher's `Subscribe` arm to flip the conn's
    /// role *after* the ack bytes have been enqueued — the
    /// ack-before-fanout ordering pinned by the wire-side regression
    /// test.
    pub(in crate::driver) fn conn_mut(&mut self, token: Token) -> Option<&mut ConnState> {
        self.conns.get_mut(&token)
    }

    /// Borrow a per-conn state immutably by [`Token`]. Returns `None`
    /// when the conn is no longer in the map — same lifecycle as
    /// [`Self::conn_mut`] (terminate_conn between tick boundaries can
    /// remove the entry).
    ///
    /// Read-only complement to `conn_mut` for handler-side predicates
    /// that need to inspect the conn's role / `close_after_flush` /
    /// queue state without mutating, and for test introspection of
    /// the post-tick per-conn shape. The Subscribe handler gates on
    /// the conn's existing role here before deciding whether to
    /// flip via [`super::conns::ConnState::transition_to_sub`]
    /// or refuse with
    /// [`crate::ipc::protocol::WireErrorCode::AlreadySubscribed`].
    pub(in crate::driver) fn conn_ref(&self, token: Token) -> Option<&ConnState> {
        self.conns.get(&token)
    }

    /// Drain the per-conn read end into LF-delimited lines.
    ///
    /// Edge-triggered: loops `read(2)` until `WouldBlock`. EOF
    /// (`Ok(0)`) returns [`ReadOutcome::PeerGone`] so the caller
    /// terminates unconditionally. `Interrupted` retries; any other
    /// read error returns `PeerGone` too (the peer-side is gone, no
    /// further bytes will arrive).
    ///
    /// Complete lines are sliced off the front of `conn.read_buf`
    /// via `drain(..=nl)` (one allocation per line, no tail copy).
    /// Each line carries its trailing `\n` — the dispatcher trims
    /// it once before serde sees the bytes (mirroring
    /// `BufRead::read_line`'s convention).
    ///
    /// **Oversize line guard.** A line exceeding
    /// [`MAX_LINE_BYTES`] is structurally hostile (operator IPC verbs
    /// are small JSON objects; the largest verb is `Subscribe { name:
    /// <CompactString> }` at ~60 bytes). The guard arms
    /// `close_after_flush` via
    /// [`super::conns::ConnState::arm_close_after_flush`],
    /// breaks the line loop, and returns [`ReadOutcome::Continue`] —
    /// the caller pairs this with `try_terminate_if_idle` (so an
    /// armed-close + empty-queue terminates inline rather than
    /// lingering for a WRITABLE edge that will never come) AND lets
    /// `drain_writable` flush any partially-queued response bytes
    /// before teardown.
    ///
    /// **Read-accumulator size guard.** A peer streaming bytes
    /// without ever emitting an LF would grow `read_buf` unboundedly
    /// across ticks — every tick more bytes arrive, no LF, no line
    /// gets dispatched, no post-process termination fires. The
    /// post-loop check on `read_buf.len()` arms `close_after_flush`
    /// for that case; the same `try_terminate_if_idle` pairing
    /// terminates the conn at end-of-tick.
    ///
    /// # Errors
    ///
    /// Returns `Err(NotFound)` only if `token` is not in the conn
    /// map — that would be a tick-body bug (we drain tokens that
    /// arrived this tick; if the conn isn't there, we never accepted
    /// it). All transport-level read failures map to
    /// `Ok(ReadOutcome::PeerGone)` so the caller's terminate path
    /// is uniform.
    pub(in crate::driver) fn read_conn_into_lines(
        &mut self,
        token: Token,
        out: &mut Vec<Vec<u8>>,
    ) -> io::Result<ReadOutcome> {
        use std::io::Read;
        let conn = self.conns.get_mut(&token).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::NotFound,
                format!("ipc read: no conn for {token:?}"),
            )
        })?;
        let mut buf = [0u8; 4096];
        let mut peer_gone = false;
        loop {
            match conn.stream.read(&mut buf) {
                Ok(0) => {
                    peer_gone = true;
                    break;
                }
                Ok(n) => conn.read_buf.extend_from_slice(&buf[..n]),
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    tracing::debug!(?token, ?e, "ipc read failed; closing conn");
                    peer_gone = true;
                    break;
                }
            }
        }
        while let Some(nl) = conn.read_buf.iter().position(|&b| b == b'\n') {
            // `drain(..=nl)` consumes the inclusive prefix in O(n)
            // — the tail of `read_buf` shifts forward once per line.
            // For the steady-state "one verb per line" cadence the
            // tail is empty, so the shift is a no-op.
            let line: Vec<u8> = conn.read_buf.drain(..=nl).collect();
            if line.len() > MAX_LINE_BYTES {
                tracing::warn!(
                    ?token,
                    len = line.len(),
                    "ipc request line exceeds MAX_LINE_BYTES; closing conn",
                );
                conn.arm_close_after_flush();
                break;
            }
            out.push(line);
        }
        if conn.read_buf.len() > MAX_LINE_BYTES {
            tracing::warn!(
                ?token,
                buffered = conn.read_buf.len(),
                "ipc read accumulator exceeds MAX_LINE_BYTES with no LF; closing conn",
            );
            conn.arm_close_after_flush();
        }
        Ok(if peer_gone {
            ReadOutcome::PeerGone
        } else {
            ReadOutcome::Continue
        })
    }

    /// Serialize a response value and append the JSON line to the
    /// conn's write_queue (the trailing `\n` is added by
    /// [`encode_line`]; framing is line-delimited).
    ///
    /// Capacity-gated via
    /// [`super::conns::ConnState::push_response`]: if the projected
    /// queue length would exceed [`super::conns::ACCEPT_CAP`] (the
    /// soft cap, sitting [`super::conns::RESPONSE_TOO_BIG_RESERVE`]
    /// bytes below [`super::conns::WRITE_QUEUE_HIGH_WATER`]), the
    /// queue is left untouched, `close_after_flush` is armed, and this
    /// returns [`EnqueueOutcome::Refused`]. The refusal arm then
    /// emits a structured
    /// [`crate::ipc::protocol::WireErrorCode::ResponseTooBig`] Err
    /// line into the per-conn reserve via
    /// [`super::conns::ConnState::push_err_in_reserve`] — the
    /// reserve's existence guarantees the Err line fits regardless of
    /// queue state, so the operator branches on `response_too_big`
    /// instead of decoding an `UnexpectedEof`. With the Err line
    /// queued, the conn is no longer idle, so the trailing
    /// [`Self::try_terminate_if_idle`] is a structural no-op; the
    /// next WRITABLE drain flushes the Err and observes
    /// `close_after_flush` at the flush-empty edge. The path
    /// converges with the queue-non-empty case: the in-flight bytes
    /// drain, then the Err drains, then the conn terminates.
    ///
    /// The cap-class signal vocabulary now matches the accept-cap's
    /// `Err { code: Busy }` shape ([`Self::drain_accept`]'s cap arm):
    /// both cap surfaces emit a closed-vocabulary structured token
    /// before close. Operator scripts decode either with one wire-
    /// stable `code` field.
    ///
    /// The `T: InfallibleSerialize` bound is the structural floor
    /// for the serializer-cannot-fail claim — every call site passes
    /// a type whose marker impl carries an audit of its serialize
    /// tree, so the wrapper's `expect` lives at the contract source
    /// in [`crate::ipc::framing`] rather than scattered as one-off
    /// `.expect("…")` strings at every enqueue arm.
    pub(in crate::driver) fn enqueue_response<T: InfallibleSerialize>(
        &mut self,
        token: Token,
        response: &T,
    ) -> EnqueueOutcome {
        let bytes = encode_line(response);
        // Scope the conn borrow so the post-push reaches into &mut self
        // cleanly on the Refused arm. PushOutcome is Copy, so the
        // binding outlives the borrow without effort.
        let outcome = match self.conns.get_mut(&token) {
            Some(conn) => conn.push_response(&bytes),
            None => return EnqueueOutcome::ConnGone,
        };
        match outcome {
            PushOutcome::Accepted => EnqueueOutcome::Accepted,
            PushOutcome::Refused => {
                tracing::warn!(
                    ?token,
                    response_len = bytes.len(),
                    accept_cap = conns::ACCEPT_CAP,
                    "ipc response over accept cap; emitting ResponseTooBig + arming close",
                );
                let err_payload = ResponsePayload::Err {
                    code: WireErrorCode::ResponseTooBig,
                    error: format!(
                        "response of {} bytes exceeds per-conn cap of {} bytes",
                        bytes.len(),
                        conns::ACCEPT_CAP,
                    ),
                };
                let err_bytes = encode_line(&err_payload);
                // push_response just refused (so conn was in the map
                // and close_after_flush is armed). The same &mut self
                // borrow continues here — nothing between the two
                // reaches has had the opportunity to remove the entry.
                // The reserve invariant
                // (ACCEPT_CAP + RESPONSE_TOO_BIG_RESERVE = WRITE_QUEUE_HIGH_WATER)
                // makes push_err_in_reserve total.
                self.conns
                    .get_mut(&token)
                    .expect(
                        "conn was in map at push_response — same &mut self.conns borrow continues",
                    )
                    .push_err_in_reserve(&err_bytes);
                // No-op once the Err line populated the queue (queue
                // is no longer empty) — a defensive call against the
                // unreachable queue-empty post-condition costs one map
                // lookup and a flag read.
                self.try_terminate_if_idle(token);
                EnqueueOutcome::Refused
            }
        }
    }

    /// Terminate `token` iff the conn is armed for close AND the
    /// write_queue is empty (nothing left to flush). Returns `true`
    /// on terminate, `false` otherwise (queue still holding bytes,
    /// close not armed, or conn already gone).
    ///
    /// Called from [`crate::driver::EngineDriver::drain_ipc_lines`]
    /// after a conn's lines are processed — the post-process pass
    /// folds in any response bytes the handler may have pushed, so
    /// the queue state at THIS point is the conn's settled state
    /// for the tick. The read drain's oversize-line guard arms
    /// `close_after_flush` without queueing anything, so an armed
    /// guard against an empty queue terminates inline here rather
    /// than waiting for a WRITABLE edge that will never come.
    ///
    /// Also called defensively at the tail of [`Self::enqueue_response`]'s
    /// Refused arm. There the structured ResponseTooBig Err line
    /// populates the queue ahead of this call, so the
    /// `write_queue.is_empty()` precondition does not hold and the
    /// call is a structural no-op; the flush-then-terminate path on
    /// the next WRITABLE drain handles teardown uniformly with the
    /// queue-non-empty case.
    ///
    /// Termination needs the cloned [`Registry`] to deregister the
    /// stream, which is why this lives on Hub rather than on
    /// `ConnState`.
    pub(in crate::driver) fn try_terminate_if_idle(&mut self, token: Token) -> bool {
        let should_terminate = self
            .conns
            .get(&token)
            .is_some_and(|c| c.close_after_flush && c.write_queue.is_empty());
        if should_terminate {
            self.terminate_conn(token);
        }
        should_terminate
    }

    /// Drain the conn's write_queue to the kernel non-blockingly.
    ///
    /// Edge-triggered: loops `write(2)` against the queue's front
    /// slice until WouldBlock, the queue empties, or the peer is
    /// gone. `Interrupted` retries; an `Ok(0)` is treated as
    /// "no progress this iteration" and breaks the loop (the next
    /// WRITABLE edge re-enters).
    ///
    /// **Queue-empty bookkeeping.** When the queue empties:
    /// 1. If `close_after_flush` is set (oversize line, over-water
    ///    response, etc.), return [`DrainWritableOutcome::Terminate`]
    ///    so the caller runs `terminate_conn` — the reregister-to-
    ///    READABLE is short-circuited (the conn is about to be
    ///    deregistered entirely).
    ///    The arm-for-close → drain → close sequence guarantees the
    ///    last queued bytes reach the wire before teardown.
    /// 2. Otherwise re-register the conn with `READABLE` only — the
    ///    WRITABLE interest was a transient "drain me" flag; leaving
    ///    it armed against an empty queue would have mio fire on
    ///    every socket-send-buffer-room edge.
    ///
    /// Three-state return ([`DrainWritableOutcome`]) — `Continue`
    /// (keep the conn open), `Terminate` (caller closes), `ConnGone`
    /// (already terminated this tick; benign no-op). Any non-
    /// `WouldBlock` write error folds into `Terminate`; a re-register
    /// failure on the queue-empty edge does the same (the conn's fd
    /// is somehow gone, which means the next read-or-write would
    /// fail anyway; close it now).
    pub(in crate::driver) fn drain_writable(&mut self, token: Token) -> DrainWritableOutcome {
        use std::io::Write;
        let Some(conn) = self.conns.get_mut(&token) else {
            return DrainWritableOutcome::ConnGone;
        };
        while !conn.write_queue.is_empty() {
            // `as_slices().0` is the contiguous front slice — no
            // need to defragment for the write. `VecDeque::drain(..n)`
            // peels off the consumed prefix in O(n) (the tail shifts
            // forward); under steady-state the queue is small and
            // the shift is negligible.
            let front = conn.write_queue.as_slices().0;
            match conn.stream.write(front) {
                Ok(0) => break,
                Ok(n) => {
                    conn.write_queue.drain(..n);
                }
                Err(e) if e.kind() == io::ErrorKind::WouldBlock => break,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) => {
                    tracing::debug!(?token, ?e, "ipc write failed; closing conn");
                    return DrainWritableOutcome::Terminate;
                }
            }
        }
        if conn.write_queue.is_empty() {
            // Close-after-flush short-circuits the reregister: the
            // caller terminates the conn on `Terminate`, which
            // deregisters the stream entirely — re-registering with
            // READABLE-only first would be a wasted syscall on the
            // close-after-flush edge.
            if conn.close_after_flush {
                return DrainWritableOutcome::Terminate;
            }
            if let Err(e) = self
                .registry
                .reregister(&mut conn.stream, token, Interest::READABLE)
            {
                tracing::debug!(?token, ?e, "ipc reregister-to-READABLE failed; closing");
                return DrainWritableOutcome::Terminate;
            }
        }
        DrainWritableOutcome::Continue
    }

    /// Remove a conn from the map, deregistering its mio source
    /// first so no straggler event for this Token reaches the
    /// dispatch loop after teardown.
    ///
    /// The deregister-before-drop discipline matches mio's
    /// "deregister explicitly before drop" contract — a
    /// `ConnState::stream` that drops without going through this
    /// function would strand a registration, and the next `poll`
    /// would fire events against a Token whose conn has gone
    /// (silent miss; harmless in practice but technically a
    /// programmer-error).
    ///
    /// Deregister errors are logged at debug — the most common
    /// cause is "fd already closed" (a write failure earlier in the
    /// tick); the path is best-effort, the next `terminate_conn`
    /// for the same Token would be a `None` on the map.
    ///
    /// Audit grep: `conns.remove\|conns.drain` returns only this
    /// function — the single removal path keeps the
    /// deregister-before-drop discipline structurally enforced.
    pub(in crate::driver) fn terminate_conn(&mut self, token: Token) {
        let Some(mut conn) = self.conns.remove(&token) else {
            return;
        };
        if let Err(e) = self.registry.deregister(&mut conn.stream) {
            tracing::debug!(
                ?token,
                ?e,
                "ipc deregister failed (fd likely already closed)"
            );
        }
        // `conn` drops here — `ConnState::stream`'s Drop closes
        // the fd; the read/write buffers drop their allocations.
    }

    /// Walk every conn and re-register the mio interest to include
    /// `WRITABLE` for any conn with pending bytes.
    ///
    /// Called once at the end of the tick's drain pass — see
    /// [`Self::dispatch_to_subscribers`] for the "amortize one
    /// rearm across the whole tick" rationale. A conn whose queue
    /// is empty stays at `READABLE` only (the disarm happens in
    /// `drain_writable` when the queue empties via a successful
    /// write).
    ///
    /// # Per-conn failure isolation
    ///
    /// A failed `reregister` on a single conn (the conn's fd is gone
    /// or an out-of-band deregister beat us to it) terminates that
    /// conn and continues the walk; the daemon stays alive. The
    /// per-tick failure footprint is therefore at most one conn per
    /// failing reregister, never the whole reactor.
    ///
    /// Failures are deferred to a post-loop drain so the inner walk
    /// can mutate per-conn fields (`reregister` borrows `&mut
    /// conn.stream`) without conflicting with the
    /// [`Self::terminate_conn`] removal it would otherwise need to
    /// fire inline. Same shape as [`Self::drain_writable`]'s
    /// `DrainWritableOutcome::Terminate` arm — single-conn failure
    /// ⇒ single-conn termination, not daemon shutdown.
    pub(in crate::driver) fn arm_writable_interests(&mut self) {
        // Collect the tokens that failed; `Vec::new()` doesn't
        // allocate until the first `push`, so the steady-state cost
        // is zero for the happy path.
        let mut to_terminate: Vec<Token> = Vec::new();
        for conn in self.conns.values_mut() {
            if conn.write_queue.is_empty() {
                continue;
            }
            if let Err(e) = self.registry.reregister(
                &mut conn.stream,
                conn.token,
                Interest::READABLE | Interest::WRITABLE,
            ) {
                tracing::debug!(
                    token = ?conn.token,
                    ?e,
                    "ipc writable-interest rearm failed; terminating conn",
                );
                to_terminate.push(conn.token);
            }
        }
        for token in to_terminate {
            self.terminate_conn(token);
        }
    }

    /// Test-only read of the conn-map size. Used to assert
    /// "accept happened" / "terminate happened" without observing the
    /// wire (the wire-side assertion is the load-bearing one; this is
    /// belt-and-braces for tests where the conn lifecycle is the
    /// subject and the wire payload is incidental).
    #[cfg(test)]
    pub(in crate::driver) fn conn_count(&self) -> usize {
        self.conns.len()
    }

    /// Test-only: deregister `token`'s stream out-of-band so the next
    /// [`Self::arm_writable_interests`] call observes a stale
    /// kernel-side registration and routes that conn through the
    /// defer-terminate path. The conn entry stays in the map until
    /// `arm_writable_interests` runs `terminate_conn` against it.
    ///
    /// Used by [`super::super::tests::arm_writable_interests_per_conn_failure_terminates_only_failing_conn`]
    /// to exercise the per-conn failure-isolation contract on Linux,
    /// where `EPOLL_CTL_MOD` returns `ENOENT` for an unregistered fd.
    /// macOS's kqueue `EV_ADD | EV_CLEAR` is add-or-modify and would
    /// succeed on the same surface, so the behavioral test is
    /// Linux-gated; the cross-platform contract is enforced by the
    /// signature change ([`Self::arm_writable_interests`] returns
    /// `()`, not `io::Result`, so per-conn failures cannot propagate
    /// to daemon shutdown structurally).
    #[cfg(all(test, target_os = "linux"))]
    pub(in crate::driver) fn force_deregister_conn_for_test(&mut self, token: Token) {
        let conn = self
            .conns
            .get_mut(&token)
            .expect("force_deregister_conn_for_test: token must be in map");
        self.registry
            .deregister(&mut conn.stream)
            .expect("test: deregister against live Poll");
    }
}

impl Drop for Hub {
    /// Explicit `deregister` for the listener + every live conn
    /// stream ahead of the field-order drop that closes their fds.
    /// mio's contract calls for "deregister before drop"; relying on
    /// the source's own `Drop` to release the registration is a
    /// contract violation, even though in practice the field-order
    /// drop reaches each fd before the underlying selector closes.
    ///
    /// The Reactor's Poll selector is still live at this moment
    /// (the [`crate::driver::EngineDriver`] field order places `ipc`
    /// before `reactor`), so the deregister calls reach the same
    /// kernel-side state the registrations created.
    ///
    /// Errors are best-effort. `NotFound` is benign on a stream whose
    /// fd was already closed by a prior `terminate_conn`. A
    /// non-`NotFound` error here is a programmer-error worth knowing
    /// about but not worth panicking — Drop must not unwind. The log
    /// channel may already be gone at this teardown phase, which is
    /// fine: the next process boot re-creates everything against a
    /// fresh selector regardless.
    fn drop(&mut self) {
        let _ = self.registry.deregister(&mut self.listener);
        for conn in self.conns.values_mut() {
            let _ = self.registry.deregister(&mut conn.stream);
        }
        // Field-order drop (listener → conns → registry → counter)
        // runs after this method returns.
    }
}

impl std::fmt::Debug for Hub {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Hub")
            .field("conn_count", &self.conns.len())
            .field("next_conn_token", &self.next_conn_token)
            .finish_non_exhaustive()
    }
}
