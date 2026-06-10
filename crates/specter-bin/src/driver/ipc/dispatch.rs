//! Operator-IPC verb dispatch — drains per-conn read events the mio tick collected, parses each
//! LF-delimited [`WireRequest`], and routes through the projection helpers, the reload pipeline, or
//! the per-conn role flip.
//!
//! Lives between [`EngineDriver::tick`](crate::driver::EngineDriver::tick) (which collects per-conn
//! readiness into the [`crate::driver::reactor::DrainedTick`]) and the downstream sinks
//! ([`super::Hub::dispatch_to_subscribers`] for fan-out,
//! [`crate::driver::EngineDriver::dispatch_reload`] for the reload pipeline, and the
//! [`super::project`] free functions for status / list / show projections). Every handler returns
//! [`ControlFlow<()>`] so a mid-handler shutdown (a [`crate::driver::EngineDriver::forward`] that
//! observed a downstream disconnect, or a `dispatch_reload` that observed shutdown mid-apply)
//! propagates back through the tick and into [`crate::driver::EngineDriver::begin_shutdown`].
//!
//! # Visibility
//!
//! `pub(in crate::driver)` — the only caller is
//! [`EngineDriver::tick`](crate::driver::EngineDriver::tick). The per-verb handlers are private to
//! this module; `drain_ipc_lines` is the single seam.
//!
//! # No envelope, no reply channel, no worker thread
//!
//! The IPC pipeline is single-threaded: the mio reactor drains per-conn bytes inline, parses one
//! [`WireRequest`] per line, and writes the response into the conn's write_queue. There is no
//! [`crossbeam::channel`] envelope, no `bounded(1)` reply channel (the same thread that parsed the
//! line also writes the response), and no per-request `Arc<AtomicBool>` shutdown gate (the driver's
//! signal handler arms `begin_shutdown` directly).
//!
//! # Engine in-unwind silence
//!
//! [`specter_engine::Engine`] MUST NOT be wrapped in `catch_unwind` — `ProbeSlot`'s linear-edge
//! tripwire (`specter_core::probe`) depends on a mid-`step` panic being fatal. An IPC request that
//! drives `engine.step` to panic therefore crashes the daemon. A future "recover and continue"
//! handler would need to thread its disarm site through the engine's probe lattice first.

use super::conns::ConnRole;
use super::hub::{EnqueueOutcome, ReadOutcome};
use super::project;
use crate::driver::EngineDriver;
use crate::driver::state::ReloadTrigger;
use crate::ipc::framing::{InfallibleSerialize, parse_strict};
use crate::ipc::protocol::{ResponsePayload, WireErrorCode, WireId, WireRequest};
use compact_str::CompactString;
use mio::Token;
use specter_core::Input;
use specter_sensor::FsWatcher;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

/// Upper bound on an operator `absorb --for` window, clamped at the driver before the engine step.
///
/// The driver is the sole [`Input::ArmAbsorb`] producer, and the engine derives the window expiry
/// as `now + duration` (`specter_engine`'s `on_arm_absorb`). `Instant + Duration` panics on
/// overflow — on a u64-nanosecond `Instant` (macOS) a multi-century `Duration` overflows — so the
/// duration is clamped here, where every client is covered, rather than guarded with defensive
/// arithmetic in the (rightly-trusting, as it trusts config) engine.
///
/// **Not a policy cap.** An `absorb` window covers a replication transfer — seconds to hours — so ten
/// years is unreachable by operator intent, yet leaves >50× headroom below the tightest `Instant`
/// representation (a u64-nanosecond monotonic counter overflows ~584 years past its epoch).
/// `parse_duration` (CLI) and the engine impose no cap by design; this is the lone overflow guard.
const MAX_ABSORB_WINDOW: Duration = Duration::from_hours(10 * 365 * 24);

impl<W: FsWatcher> EngineDriver<W> {
    /// Drain the per-conn readiness this tick into IPC verb dispatches.
    ///
    /// Walks every per-conn Token in `read_tokens`, asks the [`super::Hub`] to pull bytes off the
    /// kernel buffer into LF-delimited line chunks, and dispatches each line through
    /// [`Self::handle_ipc_line`]. The two termination semantics are distinguished by the
    /// [`ReadOutcome`] return:
    ///
    /// - [`ReadOutcome::PeerGone`] (EOF or unrecoverable read error) ⇒ unconditional
    ///   [`super::Hub::terminate_conn`]. Any pending write-queue bytes are wasted because the
    ///   peer's read end has closed.
    /// - [`ReadOutcome::Continue`] ⇒ pair with [`super::Hub::try_terminate_if_idle`]. The read drain
    ///   may have armed `close_after_flush` (oversize line, or over-cap read accumulator); the
    ///   handler loop may have enqueued response bytes; the queue state at this point is the conn's
    ///   settled state for the tick. If armed AND empty, the conn terminates inline. If armed AND
    ///   non-empty, [`super::Hub::drain_writable`] handles the terminate on the flush edge.
    ///
    /// A read-side `Err` from the Hub is the "no conn for token" shape — a tick-body bug that
    /// nonetheless terminates the (presumably already-gone) conn defensively.
    ///
    /// Write-side termination (peer-gone observed during a `drain_writable` call, or a
    /// `close_after_flush` that flushed cleanly) lives on the tick's WRITABLE pass — that pass
    /// terminates directly via [`super::Hub::terminate_conn`] because it doesn't need any IPC
    /// handler state.
    ///
    /// Returns [`ControlFlow::Break`] iff a handler observed shutdown mid-apply (the
    /// `Reload`/`Disable`/`Enable`/`Absorb` arms can drive `engine.step` + `forward`, and a
    /// downstream-disconnect `forward` propagates Break upward). All other paths return
    /// [`ControlFlow::Continue`] — including malformed JSON, unknown names, and read failures,
    /// which surface to the operator as a structured `Err` response or a clean conn close.
    pub(in crate::driver) fn drain_ipc_lines(
        &mut self,
        read_tokens: &[Token],
        now: Instant,
    ) -> ControlFlow<()> {
        for &token in read_tokens {
            // Per-conn line buffer — re-allocated each loop iteration because the line bytes are
            // consumed by serde during dispatch (no benefit to reuse). `Vec<Vec<u8>>` keeps the
            // line-framing explicit; `Vec<u8>` would force a re-scan for LFs at every dispatch.
            let mut lines: Vec<Vec<u8>> = Vec::new();
            let outcome = match self.ipc.read_conn_into_lines(token, &mut lines) {
                Ok(o) => o,
                Err(e) => {
                    tracing::debug!(?token, ?e, "ipc read pipeline failed; closing conn");
                    self.ipc.terminate_conn(token);
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
                    // Handler loop above may have pushed response bytes into the queue; the
                    // close-arm may have been set by the read drain's oversize-line guard.
                    // try_terminate_if_idle resolves the four combinations into the right action.
                    self.ipc.try_terminate_if_idle(token);
                }
                ReadOutcome::PeerGone => {
                    self.ipc.terminate_conn(token);
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// Enqueue an infallible response to `token`'s write_queue, swallowing the [`EnqueueOutcome`].
    ///
    /// Every verb handler outside the Subscribe ack flow discards the outcome benignly:
    ///
    /// - [`EnqueueOutcome::Accepted`] is the happy path.
    /// - [`EnqueueOutcome::Refused`] is already on the close-after- flush teardown path — the
    ///   over-water response armed the close, and the Hub either inline-terminated via
    ///   `try_terminate_if_idle` (queue-empty arm) or queued bytes for the next flush edge to
    ///   drain-and-terminate. Re-acking would be a no-op against a conn the next pass tears down.
    /// - [`EnqueueOutcome::ConnGone`] means the read drain observed EOF between the line drain and
    ///   this dispatch (or a write failure removed the entry mid-tick). Re-acking is impossible —
    ///   the conn is gone.
    ///
    /// The Subscribe ack flow is the lone exception: [`Self::handle_subscribe`] gates the
    /// [`super::conns::ConnState::transition_to_sub`] role flip on the `Accepted` discriminant and
    /// therefore reaches [`super::Hub::enqueue_response`] directly.
    fn respond<T: InfallibleSerialize>(&mut self, token: Token, payload: &T) {
        let _ = self.ipc.enqueue_response(token, payload);
    }

    /// Parse and dispatch one LF-delimited line as a [`WireRequest`].
    ///
    /// Strict parse via [`parse_strict`]: a typoed operator JSON
    /// (`{"op":"subscribe","names":"build"}`) is rejected with the unknown-field surface rather than
    /// silently dropping the typo'd key — the gate uses the derived [`serde::Serialize`] as the
    /// schema and round-trip-validates the input against it. Either kind of parse failure (malformed
    /// JSON or unknown field) enqueues a [`ResponsePayload::Err`] with `code =
    /// WireErrorCode::Malformed` and continues; the client gets one structured error frame and the
    /// conn stays open for the next line. (A repeat-offender peer trips the read- accumulator-size
    /// guard in [`super::Hub::read_conn_into_lines`] and the conn terminates on the next drain pass.)
    ///
    /// The trailing `\n` is stripped before serde sees the bytes — mirror of the standard
    /// `BufRead::read_line` convention; the JSON parser would reject a trailing newline as a
    /// structural token. The [`Option::expect`] is the framing invariant:
    /// [`super::Hub::read_conn_into_lines`] produces every line via `drain(..=nl)` (LF inclusive),
    /// so a missing trailing LF would mean an upstream framing breach. A loud panic at this seam
    /// beats a misleading "json parse" surfacing one frame later.
    ///
    /// # Mutating-verb shutdown gate
    ///
    /// `Reload`, `Disable`, `Enable`, and `Absorb` are gated on
    /// `EngineDriver::first_term.is_none()` — once the driver has observed a SIGINT / SIGTERM,
    /// mutating verbs refuse with [`WireErrorCode::ShuttingDown`]. The actuator is in the middle of
    /// its grace pipeline; admitting a fresh `engine.step` from `Reload` (which can attach new Subs
    /// that arm probes), `Disable` / `Enable` (which mutate engine state), or `Absorb` (which arms
    /// a window and may retro-latch an in-flight burst) would invalidate the shutdown's premise
    /// that the engine is winding down. Read-only verbs (`Status`, `List`, `Show`) and `Subscribe`
    /// (bin-local mutation only — flips `conn.role` without touching engine state) stay accessible
    /// so operators can `tail` the wind-down.
    fn handle_ipc_line(&mut self, token: Token, line: &[u8], now: Instant) -> ControlFlow<()> {
        let trimmed = line
            .strip_suffix(b"\n")
            .expect("framing invariant: line carries trailing LF");
        let request: WireRequest = match parse_strict(trimmed) {
            Ok(r) => r,
            Err(e) => {
                self.respond(
                    token,
                    &ResponsePayload::Err {
                        code: WireErrorCode::Malformed,
                        error: format!("json parse: {e}"),
                    },
                );
                return ControlFlow::Continue(());
            }
        };
        // Mutating-verb shutdown gate — see the rustdoc above. Lives at this seam (not on each
        // handler) so the four mutating-verb arms below stay focused on their happy path; the
        // refusal carries the structured `ShuttingDown` code so operator scripts can branch
        // deterministically.
        if self.first_term.is_some()
            && matches!(
                request,
                WireRequest::Reload
                    | WireRequest::Disable { .. }
                    | WireRequest::Enable { .. }
                    | WireRequest::Absorb { .. }
            )
        {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::ShuttingDown,
                    error: "daemon shutting down; mutating verbs refused".into(),
                },
            );
            return ControlFlow::Continue(());
        }
        match request {
            WireRequest::Status => {
                let resp = project::status(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    self.loader.current_config(),
                    &self.config_path,
                );
                self.respond(token, &ResponsePayload::Status(resp));
                ControlFlow::Continue(())
            }
            WireRequest::List => {
                let resp = project::list(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    self.loader.current_config(),
                );
                self.respond(token, &ResponsePayload::List(resp));
                ControlFlow::Continue(())
            }
            WireRequest::Show { name } => {
                let resp = project::show(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    self.loader.current_config(),
                    name.as_str(),
                    now,
                );
                self.respond(token, &ResponsePayload::Show(resp));
                ControlFlow::Continue(())
            }
            WireRequest::Subscribe { name } => self.handle_subscribe(token, name.as_ref()),
            WireRequest::Reload => {
                // Single-source attribution: `ReloadTrigger::Ipc` is constructed AT this call site,
                // not inferred from a peer pulse. The reload's success rotates the loader and bumps
                // `driver_state`'s reload counters with this trigger.
                let outcome = self.dispatch_reload(ReloadTrigger::Ipc, now);
                self.respond(token, &ResponsePayload::Ok);
                outcome
            }
            WireRequest::Disable { name } => self.handle_disable(token, name, now),
            WireRequest::Enable { name } => self.handle_enable(token, name.as_str(), now),
            WireRequest::Absorb { name, duration_ms } => {
                self.handle_absorb(token, name.as_str(), duration_ms, now)
            }
        }
    }

    /// Subscribe arm — three precondition gates, then ack-before- fanout ordering.
    ///
    /// 1. **Already-subscribed gate.** A repeat Subscribe on a conn that already flipped to
    ///    [`ConnRole::Sub`] is a client-side bug; left ungated it silently overwrites the prior
    ///    filter and drops the accumulated `missed` window. The handler refuses with
    ///    [`WireErrorCode::AlreadySubscribed`] so the operator sees a deterministic failure rather
    ///    than an invisible state mutation.
    /// 2. **Unknown-name gate.** A `Some(name)` that does not resolve through the engine's
    ///    `find_by_name` index returns [`WireErrorCode::UnknownSub`]. The conn stays in `Reqs` (no
    ///    role flip), so a retry with a valid name still goes through the unfiltered path.
    /// 3. **Ack-then-flip.** With `conn.role` still in `Reqs`, any concurrent
    ///    `dispatch_to_subscribers` call skips this conn — no diag can interleave between the ack
    ///    enqueue and the role flip. The ack bytes are already in the write_queue when
    ///    [`ConnRole::Sub`] takes effect, so the wire-order contract (`SubscribeAck` precedes every
    ///    future diag) holds structurally. Pinned by the `subscribe_ack_precedes_diag_on_wire`
    ///    regression test.
    fn handle_subscribe(&mut self, token: Token, name: Option<&CompactString>) -> ControlFlow<()> {
        // Gate 1: refuse a second Subscribe with a structured error. `is_some_and` consumes the
        // conn_ref borrow before the following `respond`'s `&mut self` reach.
        let already_subscribed = self
            .ipc
            .conn_ref(token)
            .is_some_and(|c| matches!(c.role, ConnRole::Sub { .. }));
        if already_subscribed {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::AlreadySubscribed,
                    error: "conn already in subscribe mode".into(),
                },
            );
            return ControlFlow::Continue(());
        }
        // Gate 2: refuse Some(name) that doesn't resolve. None (unfiltered tail) always resolves.
        let resolved = name.and_then(|n| self.engine.subs().find_by_name(n));
        if let Some(n) = name
            && resolved.is_none()
        {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::UnknownSub,
                    error: format!("no watch named {n}"),
                },
            );
            return ControlFlow::Continue(());
        }
        // Ack-then-flip: enqueue the ack while `conn.role == Reqs` (fan-out skips this conn); flip
        // iff the ack landed (a Refused or ConnGone outcome means the conn is already on its way
        // out and the role flip would be a no-op against a gone conn or a flush-in-progress one).
        let ack = ResponsePayload::SubscribeAck {
            sub: resolved.map(WireId::from),
        };
        if matches!(
            self.ipc.enqueue_response(token, &ack),
            EnqueueOutcome::Accepted
        ) {
            // Structural invariant: an `Accepted` outcome means the conn was in the map at enqueue
            // time, and the same `&mut self.ipc` borrow continues here — nothing between the two
            // reaches has had the opportunity to remove the entry. A `None` from `conn_mut` here
            // would be a Hub-internal lifecycle bug; the `expect` documents the load-bearing reason
            // a defensive Option-branch is not appropriate at this seam.
            self.ipc
                .conn_mut(token)
                .expect("just-Accepted conn must remain in map — same &mut self.ipc borrow")
                .transition_to_sub(resolved);
        }
        ControlFlow::Continue(())
    }

    /// Operator-requested runtime disable of a static Sub by name.
    ///
    /// Three precondition gates ahead of the apply:
    ///
    /// 1. A name absent from the engine's `by_name` index is refused with
    ///    [`WireErrorCode::UnknownSub`].
    /// 2. A discovery-minted Sub (`is_dynamic()`) is refused with
    ///    [`WireErrorCode::DynamicSubNoOp`] — the discovery Profile's next reconcile would simply
    ///    re-mint it; disabling the *template* is the lever (its cascade reaps the minted set).
    /// 3. A name already in `disabled_runtime` is refused with [`WireErrorCode::NotDisabled`] — the
    ///    verb's precondition (sub is runtime-enabled) is violated.
    ///
    /// On the apply path, the override-set insert runs BEFORE the engine's [`Input::DetachSub`]
    /// step. The ordering binds the `disabled_runtime ↔ engine` invariant across the engine step:
    /// any same-tick observer that reads `disabled_runtime` membership sees the override in place
    /// before the detach emission, so a reload running on the same tick refuses to re-attach in the
    /// same pass.
    ///
    /// The reply is acked unconditionally after the work — matching the [`WireRequest::Reload`] ack
    /// discipline. State mutations happen before [`Self::forward`], so the `Ok` ack is truthful
    /// regardless of whether `forward` observes shutdown mid-flight; the [`ControlFlow`] return
    /// propagates the shutdown signal so the tick can resolve cleanly.
    fn handle_disable(
        &mut self,
        token: Token,
        name: CompactString,
        now: Instant,
    ) -> ControlFlow<()> {
        let Some(sid) = self.engine.subs().find_by_name(name.as_str()) else {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::UnknownSub,
                    error: format!("no watch named {name}"),
                },
            );
            return ControlFlow::Continue(());
        };
        let sub = self
            .engine
            .subs()
            .get(sid)
            .expect("by_name resolves to live SubId — registry lockstep invariant");
        if sub.is_dynamic() {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::DynamicSubNoOp,
                    error: "cannot disable a discovery-minted sub (the next \
                            reconcile would re-mint it; disable the template instead)"
                        .into(),
                },
            );
            return ControlFlow::Continue(());
        }
        if self.disabled_runtime.contains(name.as_str()) {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::NotDisabled,
                    error: "already disabled at runtime".into(),
                },
            );
            return ControlFlow::Continue(());
        }
        self.disabled_runtime.insert(name);
        let out = self.engine.step(Input::DetachSub(sid), now);
        let outcome = self.forward(out);
        self.respond(token, &ResponsePayload::Ok);
        outcome
    }

    /// Operator-requested clear of a runtime disable, with best- effort re-attach.
    ///
    /// Two-step semantics:
    ///
    /// 1. **Clear the override.** [`std::collections::BTreeSet::remove`] returns the membership flag
    ///    atomically with the removal; a `false` return surfaces as [`WireErrorCode::NotDisabled`].
    ///    The override IS cleared before step 2 runs, so even on the [`WireErrorCode::TomlDisabled`]
    ///    failure path the runtime state is consistent: the operator's "no longer want this
    ///    suppressed" intent is honoured regardless of whether the TOML can re-attach.
    /// 2. **Re-attach iff the TOML carries the entry active.**
    ///    [`specter_config::Config::find_active_watch`] returns `None` for both "entry has `enabled
    ///    = false`" AND "entry left the TOML entirely"; both surface the same
    ///    [`WireErrorCode::TomlDisabled`] to the operator. On a hit, a fresh [`Input::AttachSub`]
    ///    step drives re-attach, and per-anchor outcomes (Pending descent, AttachPathInvalid)
    ///    surface via diag fan-out ([`super::Hub::dispatch_to_subscribers`]), not the verb ack.
    ///
    /// The reply is acked unconditionally after the work — same discipline as
    /// [`Self::handle_disable`].
    fn handle_enable(&mut self, token: Token, name: &str, now: Instant) -> ControlFlow<()> {
        if !self.disabled_runtime.remove(name) {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::NotDisabled,
                    error: "not disabled at runtime".into(),
                },
            );
            return ControlFlow::Continue(());
        }
        let Some(spec) = self.loader.current_config().find_active_watch(name) else {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::TomlDisabled,
                    error: "runtime override cleared, but the watch is not active in the \
                            current TOML (enabled = false or removed); edit config and \
                            reload to re-attach"
                        .into(),
                },
            );
            return ControlFlow::Continue(());
        };
        let out = self
            .engine
            .step(Input::AttachSub(spec.to_attach_request()), now);
        let outcome = self.forward(out);
        self.respond(token, &ResponsePayload::Ok);
        outcome
    }

    /// Operator-requested `absorb` window on a static Sub's Profile — the runtime fold-without-fire
    /// signal. Arms a window so the next fireable burst (or an in-flight one, retro-latched)
    /// advances the baseline silently instead of firing, folding an expected replication into the
    /// settled reference rather than echoing it.
    ///
    /// Mirrors [`Self::handle_disable`]'s resolution + gate shape, minus the `disabled_runtime`
    /// interaction (absorb does not detach):
    ///
    /// 1. A name absent from the engine's `by_name` index is refused with
    ///    [`WireErrorCode::UnknownSub`].
    /// 2. A discovery-minted Sub (`is_dynamic()`) is refused with
    ///    [`WireErrorCode::DynamicSubNoOp`] — minted Subs vanish and re-mint with the match set,
    ///    the same reason `disable` refuses them.
    ///
    /// On the apply path the operator's `duration_ms` is rebuilt into a [`Duration`] **and
    /// clamped** to [`MAX_ABSORB_WINDOW`] before the engine step — the lone overflow guard for the
    /// feature (see that constant). A `None` duration threads through as the engine's
    /// consume-on-first default (one `settle` interval).
    ///
    /// The reply is acked unconditionally after the work — same discipline as
    /// [`Self::handle_disable`] / [`Self::handle_enable`].
    fn handle_absorb(
        &mut self,
        token: Token,
        name: &str,
        duration_ms: Option<u64>,
        now: Instant,
    ) -> ControlFlow<()> {
        let Some(sid) = self.engine.subs().find_by_name(name) else {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::UnknownSub,
                    error: format!("no watch named {name}"),
                },
            );
            return ControlFlow::Continue(());
        };
        let sub = self
            .engine
            .subs()
            .get(sid)
            .expect("by_name resolves to live SubId — registry lockstep invariant");
        if sub.is_dynamic() {
            self.respond(
                token,
                &ResponsePayload::Err {
                    code: WireErrorCode::DynamicSubNoOp,
                    error: "cannot absorb on a discovery-minted sub (minted Subs \
                            vanish and re-mint with the match set; target an \
                            operator-declared sub)"
                        .into(),
                },
            );
            return ControlFlow::Continue(());
        }
        // `ProfileId` is `Copy`; capturing it ends the `&self.engine` borrow before the `&mut
        // self.engine` step below.
        let profile = sub.profile();
        let duration = duration_ms.map(|ms| Duration::from_millis(ms).min(MAX_ABSORB_WINDOW));
        let out = self
            .engine
            .step(Input::ArmAbsorb { profile, duration }, now);
        let outcome = self.forward(out);
        self.respond(token, &ResponsePayload::Ok);
        outcome
    }
}
