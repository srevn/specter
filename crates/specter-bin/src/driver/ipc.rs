//! Operator-IPC verb dispatch — drains per-conn read events the mio
//! tick collected, parses each LF-delimited [`WireRequest`], and
//! routes through the projection helpers, the reload pipeline, or the
//! per-conn role flip.
//!
//! Lives between [`super::tick`] (which collects per-conn readiness
//! into the [`super::hub::DrainedTick`]) and the downstream sinks
//! ([`super::hub::DriverHub::dispatch_to_subscribers`] for fan-out,
//! [`super::EngineDriver::dispatch_reload`] for the reload pipeline,
//! and the [`crate::ipc::project`] free functions for status / list /
//! show projections). Every handler returns [`ControlFlow<()>`] so a
//! mid-handler shutdown (a [`super::EngineDriver::forward`] that
//! observed a downstream disconnect, or a `dispatch_reload` that
//! observed shutdown mid-apply) propagates back through the tick and
//! into [`super::EngineDriver::begin_shutdown`].
//!
//! # Visibility
//!
//! `pub(super)` — the only caller is [`super::tick`]. The per-verb
//! handlers are private to this module; `drain_ipc_lines` is the
//! single seam.
//!
//! # No envelope, no reply channel, no worker thread
//!
//! The IPC pipeline is single-threaded: the mio reactor drains
//! per-conn bytes inline, parses one [`WireRequest`] per line, and
//! writes the response into the conn's write_queue. There is no
//! [`crossbeam::channel`] envelope, no `bounded(1)` reply channel
//! (the same thread that parsed the line also writes the response),
//! and no per-request `Arc<AtomicBool>` shutdown gate (the driver's
//! signal handler arms `begin_shutdown` directly).
//!
//! # Engine in-unwind silence
//!
//! [`specter_engine::Engine`] MUST NOT be wrapped in `catch_unwind` —
//! `ProbeSlot`'s linear-edge tripwire (`specter_core::probe`) depends
//! on a mid-`step` panic being fatal. An IPC request that drives
//! `engine.step` to panic therefore crashes the daemon. A future
//! "recover and continue" handler would need to thread its disarm
//! site through the engine's probe lattice first.

use super::EngineDriver;
use super::conns::ConnRole;
use super::hub::{EnqueueOutcome, ReadOutcome};
use super::state::ReloadTrigger;
use crate::ipc::project;
use crate::ipc::protocol::{
    ERR_ALREADY_SUBSCRIBED, ERR_DYNAMIC_SUB_NO_OP, ERR_MALFORMED, ERR_NOT_DISABLED,
    ERR_TOML_DISABLED, ERR_UNKNOWN_SUB, ResponsePayload, WireId, WireRequest,
};
use compact_str::CompactString;
use mio::Token;
use specter_core::Input;
use specter_sensor::FsWatcher;
use std::borrow::Cow;
use std::ops::ControlFlow;
use std::time::Instant;

impl<W: FsWatcher> EngineDriver<W> {
    /// Drain the per-conn readiness this tick into IPC verb
    /// dispatches.
    ///
    /// Walks every per-conn Token in `read_tokens`, asks the Hub to
    /// pull bytes off the kernel buffer into LF-delimited line
    /// chunks, and dispatches each line through
    /// [`Self::handle_ipc_line`]. The two termination semantics are
    /// distinguished by the [`ReadOutcome`] return:
    ///
    /// - [`ReadOutcome::PeerGone`] (EOF or unrecoverable read error)
    ///   ⇒ unconditional [`super::hub::DriverHub::terminate_conn`].
    ///   Any pending write-queue bytes are wasted because the peer's
    ///   read end has closed.
    /// - [`ReadOutcome::Continue`] ⇒ pair with
    ///   [`super::hub::DriverHub::try_terminate_if_idle`]. The read
    ///   drain may have armed `close_after_flush` (oversize line, or
    ///   over-cap read accumulator); the handler loop may have
    ///   enqueued response bytes; the queue state at this point is
    ///   the conn's settled state for the tick. If armed AND empty,
    ///   the conn terminates inline. If armed AND non-empty,
    ///   [`super::hub::DriverHub::drain_writable`] handles the
    ///   terminate on the flush edge.
    ///
    /// A read-side `Err` from the Hub is the "no conn for token"
    /// shape — a tick-body bug that nonetheless terminates the
    /// (presumably already-gone) conn defensively.
    ///
    /// Write-side termination (peer-gone observed during a
    /// `drain_writable` call, or a `close_after_flush` that flushed
    /// cleanly) lives on the tick's WRITABLE pass — that pass
    /// terminates directly via [`super::hub::DriverHub::terminate_conn`]
    /// because it doesn't need any IPC handler state.
    ///
    /// Returns [`ControlFlow::Break`] iff a handler observed
    /// shutdown mid-apply (the `Reload`/`Disable`/`Enable` arms can
    /// drive `engine.step` + `forward`, and a downstream-disconnect
    /// `forward` propagates Break upward). All other paths return
    /// [`ControlFlow::Continue`] — including malformed JSON, unknown
    /// names, and read failures, which surface to the operator as a
    /// structured `Err` response or a clean conn close.
    pub(super) fn drain_ipc_lines(
        &mut self,
        read_tokens: &[Token],
        now: Instant,
    ) -> ControlFlow<()> {
        for &token in read_tokens {
            // Per-conn line buffer — re-allocated each loop iteration
            // because the line bytes are consumed by serde during
            // dispatch (no benefit to reuse). `Vec<Vec<u8>>` keeps
            // the line-framing explicit; `Vec<u8>` would force a
            // re-scan for LFs at every dispatch.
            let mut lines: Vec<Vec<u8>> = Vec::new();
            let outcome = match self.hub.read_conn_into_lines(token, &mut lines) {
                Ok(o) => o,
                Err(e) => {
                    tracing::debug!(?token, ?e, "ipc read pipeline failed; closing conn");
                    self.hub.terminate_conn(token);
                    continue;
                }
            };
            for line in lines {
                if self.handle_ipc_line(token, &line, now).is_break() {
                    return ControlFlow::Break(());
                }
            }
            match outcome {
                ReadOutcome::Continue => {
                    // Handler loop above may have pushed response
                    // bytes into the queue; the close-arm may have
                    // been set by the read drain's oversize-line
                    // guard. try_terminate_if_idle resolves the
                    // four combinations into the right action.
                    self.hub.try_terminate_if_idle(token);
                }
                ReadOutcome::PeerGone => {
                    self.hub.terminate_conn(token);
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// Parse and dispatch one LF-delimited line as a [`WireRequest`].
    ///
    /// Malformed JSON enqueues an [`ResponsePayload::Err`] response
    /// with `code = ERR_MALFORMED` and continues; the client gets
    /// one structured error frame and the conn stays open for the
    /// next line. (A repeat-offender peer trips the
    /// read-accumulator-size guard in
    /// [`super::hub::DriverHub::read_conn_into_lines`] and the conn
    /// terminates on the next drain pass.)
    ///
    /// The trailing `\n` is stripped before serde sees the bytes —
    /// mirror of the standard `BufRead::read_line` convention; the
    /// JSON parser would reject a trailing newline as a structural
    /// token.
    fn handle_ipc_line(&mut self, token: Token, line: &[u8], now: Instant) -> ControlFlow<()> {
        let trimmed = line.strip_suffix(b"\n").unwrap_or(line);
        let request: WireRequest = match serde_json::from_slice(trimmed) {
            Ok(r) => r,
            Err(e) => {
                let resp = ResponsePayload::Err {
                    code: Cow::Borrowed(ERR_MALFORMED),
                    error: format!("json parse: {e}"),
                };
                let _ = self.hub.enqueue_response(token, &resp);
                return ControlFlow::Continue(());
            }
        };
        match request {
            WireRequest::Status => {
                let resp = project::status(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    &self.loader.current_config,
                    &self.config_path,
                );
                let _ = self
                    .hub
                    .enqueue_response(token, &ResponsePayload::Status(resp));
                ControlFlow::Continue(())
            }
            WireRequest::List => {
                let resp = project::list(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    &self.loader.current_config,
                );
                let _ = self
                    .hub
                    .enqueue_response(token, &ResponsePayload::List(resp));
                ControlFlow::Continue(())
            }
            WireRequest::Show { name } => {
                let resp = project::show(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    &self.loader.current_config,
                    name.as_str(),
                );
                let _ = self
                    .hub
                    .enqueue_response(token, &ResponsePayload::Show(resp));
                ControlFlow::Continue(())
            }
            WireRequest::Subscribe { name } => self.handle_subscribe(token, name.as_ref()),
            WireRequest::Reload => {
                // Single-source attribution: `ReloadTrigger::Ipc` is
                // constructed AT this call site, not inferred from a
                // peer pulse. The reload's success rotates the loader
                // and bumps `driver_state`'s reload counters with this
                // trigger.
                let outcome = self.dispatch_reload(ReloadTrigger::Ipc, now);
                let _ = self.hub.enqueue_response(token, &ResponsePayload::Ok);
                outcome
            }
            WireRequest::Disable { name } => self.handle_disable(token, name, now),
            WireRequest::Enable { name } => self.handle_enable(token, name.as_str(), now),
        }
    }

    /// Subscribe arm — three precondition gates, then ack-before-
    /// fanout ordering.
    ///
    /// 1. **Already-subscribed gate.** A repeat Subscribe on a conn
    ///    that already flipped to [`ConnRole::Sub`] is a client-side
    ///    bug; left ungated it silently overwrites the prior filter
    ///    and drops the accumulated `missed` window. The handler
    ///    refuses with [`ERR_ALREADY_SUBSCRIBED`] so the operator
    ///    sees a deterministic failure rather than an invisible
    ///    state mutation.
    /// 2. **Unknown-name gate.** A `Some(name)` that does not
    ///    resolve through the engine's `find_by_name` index returns
    ///    [`ERR_UNKNOWN_SUB`]. The conn stays in `Reqs` (no role
    ///    flip), so a retry with a valid name still goes through
    ///    the unfiltered path.
    /// 3. **Ack-then-flip.** With `conn.role` still in `Reqs`, any
    ///    concurrent `dispatch_to_subscribers` call skips this conn
    ///    — no diag can interleave between the ack enqueue and the
    ///    role flip. The ack bytes are already in the write_queue
    ///    when [`ConnRole::Sub`] takes effect, so the wire-order
    ///    contract (`SubscribeAck` precedes every future diag) holds
    ///    structurally. Pinned by the `subscribe_ack_precedes_diag_on_wire`
    ///    regression test.
    fn handle_subscribe(&mut self, token: Token, name: Option<&CompactString>) -> ControlFlow<()> {
        // Gate 1: refuse a second Subscribe with a structured error.
        // `is_some_and` consumes the conn_ref borrow before the
        // following `enqueue_response`'s `&mut self.hub` reach.
        let already_subscribed = self
            .hub
            .conn_ref(token)
            .is_some_and(|c| matches!(c.role, ConnRole::Sub { .. }));
        if already_subscribed {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_ALREADY_SUBSCRIBED),
                error: "conn already in subscribe mode".into(),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        }
        // Gate 2: refuse Some(name) that doesn't resolve. None
        // (unfiltered tail) always resolves.
        let resolved = name.and_then(|n| self.engine.subs().find_by_name(n));
        if let Some(n) = name
            && resolved.is_none()
        {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_UNKNOWN_SUB),
                error: format!("no watch named {n}"),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        }
        // Ack-then-flip: enqueue the ack while `conn.role == Reqs`
        // (fan-out skips this conn); flip iff the ack landed (a
        // Refused or ConnGone outcome means the conn is already on
        // its way out and the role flip would be a no-op against a
        // gone conn or a flush-in-progress one).
        let ack = ResponsePayload::SubscribeAck {
            sub: resolved.map(WireId::from),
        };
        if matches!(
            self.hub.enqueue_response(token, &ack),
            EnqueueOutcome::Accepted
        ) && let Some(conn) = self.hub.conn_mut(token)
        {
            conn.transition_to_sub(resolved);
        }
        ControlFlow::Continue(())
    }

    /// Operator-requested runtime disable of a static Sub by name.
    ///
    /// Three precondition gates ahead of the apply:
    ///
    /// 1. A name absent from the engine's `by_name` index is refused
    ///    with [`ERR_UNKNOWN_SUB`].
    /// 2. The resolved Sub's `source_promoter.is_some()` (a dynamic,
    ///    promoter-spawned Sub) is refused with
    ///    [`ERR_DYNAMIC_SUB_NO_OP`] — a runtime override against a
    ///    synthesised name has no TOML anchor and would evaporate at
    ///    the next reload's prune pass.
    /// 3. A name already in `disabled_runtime` is refused with
    ///    [`ERR_NOT_DISABLED`] — the verb's precondition (sub is
    ///    runtime-enabled) is violated.
    ///
    /// On the apply path, the override-set insert runs BEFORE the
    /// engine's [`Input::DetachSub`] step. The ordering binds the
    /// `disabled_runtime ↔ engine` invariant across the engine step:
    /// any same-tick observer that reads `disabled_runtime`
    /// membership sees the override in place before the detach
    /// emission, so a reload running on the same tick refuses to
    /// re-attach in the same pass.
    ///
    /// The reply is acked unconditionally after the work — matching
    /// the [`WireRequest::Reload`] ack discipline. State mutations
    /// happen before [`Self::forward`], so the `Ok` ack is truthful
    /// regardless of whether `forward` observes shutdown mid-flight;
    /// the [`ControlFlow`] return propagates the shutdown signal so
    /// the tick can resolve cleanly.
    fn handle_disable(
        &mut self,
        token: Token,
        name: CompactString,
        now: Instant,
    ) -> ControlFlow<()> {
        let Some(sid) = self.engine.subs().find_by_name(name.as_str()) else {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_UNKNOWN_SUB),
                error: format!("no watch named {name}"),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        };
        let sub = self
            .engine
            .subs()
            .get(sid)
            .expect("by_name resolves to live SubId — registry lockstep invariant");
        if sub.source_promoter.is_some() {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_DYNAMIC_SUB_NO_OP),
                error: "cannot disable a promoter-spawned dynamic sub \
                        (synthesised names cannot persist as runtime overrides)"
                    .into(),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        }
        if self.disabled_runtime.contains(name.as_str()) {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_NOT_DISABLED),
                error: "already disabled at runtime".into(),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        }
        self.disabled_runtime.insert(name);
        let out = self.engine.step(Input::DetachSub(sid), now);
        let outcome = self.forward(out);
        let _ = self.hub.enqueue_response(token, &ResponsePayload::Ok);
        outcome
    }

    /// Operator-requested clear of a runtime disable, with best-
    /// effort re-attach.
    ///
    /// Two-step semantics:
    ///
    /// 1. **Clear the override.** [`std::collections::BTreeSet::remove`]
    ///    returns the membership flag atomically with the removal; a
    ///    `false` return surfaces as [`ERR_NOT_DISABLED`]. The
    ///    override IS cleared before step 2 runs, so even on the
    ///    [`ERR_TOML_DISABLED`] failure path the runtime state is
    ///    consistent: the operator's "no longer want this suppressed"
    ///    intent is honoured regardless of whether the TOML can
    ///    re-attach.
    /// 2. **Re-attach iff the TOML carries the entry active.**
    ///    [`specter_config::Config::find_active_watch`] returns
    ///    `None` for both "entry has `enabled = false`" AND "entry
    ///    left the TOML entirely"; both surface the same
    ///    [`ERR_TOML_DISABLED`] to the operator. On a hit, a fresh
    ///    [`Input::AttachSub`] step drives re-attach, and per-anchor
    ///    outcomes (Pending descent, AttachPathInvalid) surface via
    ///    diag fan-out
    ///    ([`super::hub::DriverHub::dispatch_to_subscribers`]), not
    ///    the verb ack.
    ///
    /// The reply is acked unconditionally after the work — same
    /// discipline as [`Self::handle_disable`].
    fn handle_enable(&mut self, token: Token, name: &str, now: Instant) -> ControlFlow<()> {
        if !self.disabled_runtime.remove(name) {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_NOT_DISABLED),
                error: "not disabled at runtime".into(),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        }
        let Some(spec) = self.loader.current_config.find_active_watch(name) else {
            let resp = ResponsePayload::Err {
                code: Cow::Borrowed(ERR_TOML_DISABLED),
                error: "runtime override cleared, but the watch is not active in the \
                        current TOML (enabled = false or removed); edit config and \
                        reload to re-attach"
                    .into(),
            };
            let _ = self.hub.enqueue_response(token, &resp);
            return ControlFlow::Continue(());
        };
        let out = self
            .engine
            .step(Input::AttachSub(spec.to_attach_request()), now);
        let outcome = self.forward(out);
        let _ = self.hub.enqueue_response(token, &ResponsePayload::Ok);
        outcome
    }
}
