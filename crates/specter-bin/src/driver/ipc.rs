//! Operator-IPC request drain — pulls [`crate::ipc::protocol::IpcRequest`]s
//! off [`crate::channels::EngineSide::ipc_request_rx`] and dispatches them
//! on the driver thread.
//!
//! Lives between [`super::tick`] (the drain caller) and the broker /
//! reload pipeline / projection helpers (the dispatch sinks). Every
//! handler returns [`ControlFlow<()>`] so a mid-handler shutdown
//! (`forward → broker.dispatch` race against shutdown, or a
//! `handle_reload` that observes shutdown mid-apply) propagates back
//! through the tick and into [`super::EngineDriver::begin_shutdown`]
//! — the same exit shape every other inbound drain uses.
//!
//! # Visibility
//!
//! `pub(super)` — the only caller is [`super::tick`]. `handle_ipc`
//! is private to this module; `drain_ipc` is the public seam.
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
use super::state::ReloadTrigger;
use crate::ipc::project;
use crate::ipc::protocol::{
    ERR_DYNAMIC_SUB_NO_OP, ERR_NOT_DISABLED, ERR_TOML_DISABLED, ERR_UNKNOWN_SUB, IpcRequest,
    RequestPayload, ResponsePayload, WireId,
};
use crossbeam::channel::TryRecvError;
use specter_core::Input;
use std::borrow::Cow;
use std::ops::ControlFlow;
use std::time::Instant;

impl EngineDriver {
    /// Drain every queued IPC request on this tick. Returns
    /// [`ControlFlow::Break`] when the producer-side senders all
    /// disconnect (the IPC server thread is the sole sender; its
    /// disconnection means shutdown is in flight) or when a
    /// handler observes shutdown mid-apply.
    pub(super) fn drain_ipc(&mut self, now: Instant) -> ControlFlow<()> {
        loop {
            match self.sides.ipc_request_rx.try_recv() {
                Ok(req) => {
                    if self.handle_ipc(req, now).is_break() {
                        return ControlFlow::Break(());
                    }
                }
                Err(TryRecvError::Empty) => return ControlFlow::Continue(()),
                Err(TryRecvError::Disconnected) => return ControlFlow::Break(()),
            }
        }
    }

    /// Dispatch one IPC request — route to the projection helper
    /// (`Status` / `List` / `Show`), the broker (`Subscribe`), the
    /// reload pipeline (`Reload`), or the operator-mutation verbs
    /// (`Disable` / `Enable`).
    ///
    /// `try_send` on the reply channel is intentional: the channel
    /// is `bounded(1)` per request and the per-conn thread is
    /// waiting on `reply_rx.recv_timeout`. Its absence at this
    /// moment is its own bug (the connection died mid-flight); we
    /// never block the driver thread on a client.
    fn handle_ipc(&mut self, req: IpcRequest, now: Instant) -> ControlFlow<()> {
        use RequestPayload as P;
        match req.payload {
            P::Status => {
                let resp = project::status(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    &self.loader.current_config,
                    &self.config_path,
                );
                let _ = req.reply_tx.try_send(ResponsePayload::Status(resp));
                ControlFlow::Continue(())
            }
            P::List => {
                let resp = project::list(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    &self.loader.current_config,
                );
                let _ = req.reply_tx.try_send(ResponsePayload::List(resp));
                ControlFlow::Continue(())
            }
            P::Show { name } => {
                let resp = project::show(
                    &self.engine,
                    &self.driver_state,
                    &self.disabled_runtime,
                    &self.loader.current_config,
                    name.as_str(),
                );
                let _ = req.reply_tx.try_send(ResponsePayload::Show(resp));
                ControlFlow::Continue(())
            }
            P::Subscribe { tx, name } => {
                // Resolve `name → SubId` server-side, atomic with
                // `add_subscriber`. A typo or a race with `disable`
                // fails the Subscribe immediately; no event window
                // can leak past the resolve.
                let resolved = name
                    .as_deref()
                    .and_then(|n| self.engine.subs().find_by_name(n));
                if let Some(n) = &name
                    && resolved.is_none()
                {
                    let _ = req.reply_tx.try_send(ResponsePayload::Err {
                        code: Cow::Borrowed(ERR_UNKNOWN_SUB),
                        error: format!("no watch named {n}"),
                    });
                    return ControlFlow::Continue(());
                }
                // Add THEN ack. The broker holds the subscriber by
                // the time the per-conn thread writes the ack line
                // to the client — no diagnostic can leak past
                // registration. See driver::broker module rustdoc.
                self.broker.add_subscriber(tx, resolved);
                let _ = req.reply_tx.try_send(ResponsePayload::SubscribeAck {
                    sub: resolved.map(WireId::from),
                });
                ControlFlow::Continue(())
            }
            P::Reload => {
                // Single-source attribution: `ReloadTrigger::Ipc` is
                // constructed AT this call site, not inferred from a
                // peer pulse. `handle_reload`'s success rotates the
                // loader and bumps driver_state's reload counters
                // with this trigger.
                let outcome = self.handle_reload(ReloadTrigger::Ipc, now);
                // Ack regardless of forward-side shutdown — the
                // reload either applied or is mid-shutdown; both
                // honour the operator's request.
                let _ = req.reply_tx.try_send(ResponsePayload::Ok);
                outcome
            }
            P::Disable { name } => self.handle_disable(&req.reply_tx, name, now),
            P::Enable { name } => self.handle_enable(&req.reply_tx, name.as_str(), now),
        }
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
    ///    the next reload's prune pass. Discrimination is a property
    ///    of the resolved Sub, not a syntactic name shape.
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
    /// the [`RequestPayload::Reload`] ack discipline. State mutations
    /// happen before [`Self::forward`], so the `Ok` ack is truthful
    /// regardless of whether `forward` observes shutdown mid-flight;
    /// the [`ControlFlow`] return propagates the shutdown signal so
    /// the tick can resolve cleanly.
    fn handle_disable(
        &mut self,
        reply_tx: &crossbeam::channel::Sender<ResponsePayload>,
        name: compact_str::CompactString,
        now: Instant,
    ) -> ControlFlow<()> {
        let Some(sid) = self.engine.subs().find_by_name(name.as_str()) else {
            let _ = reply_tx.try_send(ResponsePayload::Err {
                code: Cow::Borrowed(ERR_UNKNOWN_SUB),
                error: format!("no watch named {name}"),
            });
            return ControlFlow::Continue(());
        };
        let sub = self
            .engine
            .subs()
            .get(sid)
            .expect("by_name resolves to live SubId — registry lockstep invariant");
        if sub.source_promoter.is_some() {
            let _ = reply_tx.try_send(ResponsePayload::Err {
                code: Cow::Borrowed(ERR_DYNAMIC_SUB_NO_OP),
                error: "cannot disable a promoter-spawned dynamic sub \
                        (synthesised names cannot persist as runtime overrides)"
                    .into(),
            });
            return ControlFlow::Continue(());
        }
        if self.disabled_runtime.contains(name.as_str()) {
            let _ = reply_tx.try_send(ResponsePayload::Err {
                code: Cow::Borrowed(ERR_NOT_DISABLED),
                error: "already disabled at runtime".into(),
            });
            return ControlFlow::Continue(());
        }
        self.disabled_runtime.insert(name);
        let out = self.engine.step(Input::DetachSub(sid), now);
        let outcome = self.forward(out);
        let _ = reply_tx.try_send(ResponsePayload::Ok);
        outcome
    }

    /// Operator-requested clear of a runtime disable, with best-
    /// effort re-attach.
    ///
    /// Two-step semantics:
    ///
    /// 1. **Clear the override.** [`std::collections::BTreeSet::remove`]
    ///    returns the membership flag atomically with the removal; a
    ///    `false` return surfaces as [`ERR_NOT_DISABLED`] (the verb's
    ///    precondition — sub is currently runtime-disabled — is
    ///    violated). The override IS cleared before step 2 runs, so
    ///    even on the [`ERR_TOML_DISABLED`] failure path the runtime
    ///    state is consistent: the operator's "no longer want this
    ///    suppressed" intent is honoured regardless of whether the
    ///    TOML can re-attach.
    /// 2. **Re-attach iff the TOML carries the entry active.**
    ///    [`specter_config::Config::find_active_watch`] returns
    ///    `None` for both "entry has `enabled = false`" AND "entry
    ///    left the TOML entirely"; both surface the same
    ///    [`ERR_TOML_DISABLED`] to the operator — the resolution
    ///    (edit config + reload) is the same. On a hit, a fresh
    ///    [`Input::AttachSub`] step drives re-attach, and per-anchor
    ///    outcomes (Pending descent, AttachPathInvalid) surface via
    ///    broker fan-out, not the verb ack.
    ///
    /// The reply is acked unconditionally after the work — same
    /// discipline as [`Self::handle_disable`].
    fn handle_enable(
        &mut self,
        reply_tx: &crossbeam::channel::Sender<ResponsePayload>,
        name: &str,
        now: Instant,
    ) -> ControlFlow<()> {
        if !self.disabled_runtime.remove(name) {
            let _ = reply_tx.try_send(ResponsePayload::Err {
                code: Cow::Borrowed(ERR_NOT_DISABLED),
                error: "not disabled at runtime".into(),
            });
            return ControlFlow::Continue(());
        }
        let Some(spec) = self.loader.current_config.find_active_watch(name) else {
            let _ = reply_tx.try_send(ResponsePayload::Err {
                code: Cow::Borrowed(ERR_TOML_DISABLED),
                error: "runtime override cleared, but the watch is not active in the \
                        current TOML (enabled = false or removed); edit config and \
                        reload to re-attach"
                    .into(),
            });
            return ControlFlow::Continue(());
        };
        let out = self
            .engine
            .step(Input::AttachSub(spec.to_attach_request()), now);
        let outcome = self.forward(out);
        let _ = reply_tx.try_send(ResponsePayload::Ok);
        outcome
    }
}
