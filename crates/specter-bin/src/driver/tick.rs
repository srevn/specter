//! One pass through the driver's drain order — the load-bearing
//! single-iteration body, and the module new inbound-path work lands
//! in.
//!
//! [`EngineDriver::tick`] consults the [`super::Reactor`] for the
//! multi-source mio poll, then drains the partitioned
//! [`super::reactor::DrainedTick`] in canonical order: deferred-replay
//! → listener accept → sensor inputs → expired timers → signals →
//! config-event + settle expiry → effect completions → actuator-gone
//! gate → operator-IPC verbs → WRITABLE drain / interest rearm. The
//! settle-expiry filter and `dispatch_reload` itself live in
//! [`super::reload`]; downstream dispatch in [`super::forward`]; IPC
//! dispatch in [`super::ipc`]; per-conn state machinery on
//! [`super::Hub`].
//!
//! # IPC drain placement
//!
//! IPC verb dispatch sits LAST in the engine-input drain order
//! (immediately before the writable drain). Every read verb projects
//! engine state, so draining IPC after sensor / timers / signals /
//! reload / effects guarantees each projection observes the freshest
//! engine state for this tick — `status.profile_active` reflects every
//! in-flight burst, including those that transitioned to/from Idle in
//! this same tick's drains.
//!
//! # Drain order rationale
//!
//! - **Deferred-input replay BEFORE the mio poll.** `forward()` in
//!   the prior tick may have queued [`Input::WatchOpRejected`] here;
//!   processing them first guarantees the engine sees their
//!   consequences before any fresh kernel events arrive. The block
//!   timeout collapses to [`Duration::ZERO`] when the queue is
//!   non-empty so replay isn't deferred behind a long wait.
//! - **Sensor inputs BEFORE effect completions.** The fire-cycle's
//!   post-fire tail (`PostFirePhase::Awaiting` / `Rebasing`) absorbs
//!   `FsEvent`s and folds their disk state into the rebase, while
//!   `EffectComplete` arrivals transition `Awaiting → Rebasing`. If
//!   the order were inverted, an `EffectComplete` could move the
//!   burst into Rebasing before the engine had seen `FsEvent`s queued
//!   in the same tick — those events would then route to the wrong
//!   burst (or kick off a fresh burst against an in-flight rebase).
//!   Sensor-first preserves the "fire-tail absorbs concurrent edits"
//!   contract documented on `PostFirePhase::Awaiting`.
//! - **Signal dispatch AFTER timers.** A SIGHUP arriving during a
//!   timer-driven burst should reload after the burst's timer takes
//!   effect — the reload's apply runs on the same tick but *after*
//!   the timer's emission. Timer drains are bounded (the heap has
//!   finitely many expired entries); a SIGHUP behind them costs at
//!   most "next tick" in operator-visible latency.
//! - **Config-event pulse AFTER signals.** A SIGHUP that pre-empts
//!   the auto-reload settle wakes the reload pipeline first; the
//!   subsequent settle-expiry filter then compares against the
//!   freshly-rotated `loader.config_meta` and silently no-ops.
//! - **WRITABLE drain LAST.** Every engine-input drain above can
//!   push bytes into per-conn write queues (via `forward()`'s
//!   diagnostic fan-out or the IPC handler's response enqueue). The
//!   per-tick WRITABLE pass then flushes both pre-existing residue
//!   (conns whose WRITABLE fired this tick) and any newly-armed
//!   interest (conns whose queue just gained bytes).
//!
//! # Auto-reload settle pipeline
//!
//! The config-event drain arms `config_settle_until` to
//! `now + CONFIG_SETTLE` per pulse — sustained editor bursts
//! (atomic-save sequences, in-place writes) defer the reload until
//! quiet. Apply-side: on settle expiry, a single `lstat` of
//! `config_path` filters phantom pulses (kqueue parent-dir spillover
//! from sibling writes); on confirmed [`FileMeta`](specter_config::FileMeta)
//! drift the driver runs the same [`super::EngineDriver::dispatch_reload`]
//! SIGHUP uses, so meta-rotation discipline converges across the two
//! pulse sources.
//!
//! # Disconnect policy
//!
//! The kernel-side inputs (watcher, config-watcher, signal pipe,
//! waker, channel receivers) are owned directly by the Reactor; the
//! listener and per-conn streams are owned by the Hub. Both live on
//! this thread — there is no upstream sender that can disconnect.
//! The two wake'd channels (`prober_response_rx`, `effect_complete_rx`)
//! and the outbound `effects_tx` are the only producer-driven seams.
//!
//! - **`effect_complete_rx` Disconnected** is the load-bearing
//!   actuator-gone signal. The actuator thread's
//!   `Box<dyn EffectCompleteSender>` adapter holds a [`super::WakingSink`]
//!   whose [`Drop`] closes the `Sender<Input>` clone BEFORE pulsing
//!   the wake edge (the close-then-wake symmetry of the send-then-wake
//!   protocol). The Reactor's drain surfaces the Disconnected as
//!   [`super::reactor::DrainedTick::actuator_gone`]; the tick body
//!   routes through [`super::EngineDriver::begin_shutdown`]. This is
//!   how a clean actuator exit (its `run` returned) AND a crash
//!   (panic unwound the closure) converge on one shutdown path.
//! - **`prober_response_rx` Disconnected** is benign — the prober
//!   pool's workers disconnect only at pool shutdown, which happens
//!   during App teardown *after* the driver is gone. The drain
//!   absorbs the variant silently.
//! - **`effects_tx` `try_send(Disconnected)`** in
//!   [`super::EngineDriver::forward`] is the producer-side mirror:
//!   the actuator's `effects_rx` is gone, so the driver can't ship
//!   work. Propagates `ControlFlow::Break` upward and the tick
//!   routes through [`super::EngineDriver::begin_shutdown`] — the
//!   redundant-but-idempotent path that converges with the
//!   actuator_gone arm above.

use super::ipc::hub;
use super::{EngineDriver, TickOutcome};
use specter_core::{FsEvent, Input, ResourceId};
use specter_sensor::FsWatcher;
use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

/// Auto-reload settle window. Each config-event pulse arms
/// [`super::EngineDriver::config_settle_until`] to `now + CONFIG_SETTLE`;
/// quiet for a full window expires the deadline and the driver runs
/// the lstat-vs-`loader.config_meta` filter (and on drift,
/// `dispatch_reload`).
///
/// `100ms` covers common editor patterns — atomic save (vim, Helix:
/// write-tmp → rename → fsync; ~10–30ms wall) and in-place modify
/// (`echo > file`; ~1–5ms per syscall, sustained bursts well under
/// 100ms). Fixed, not operator-tunable, under the "minimal config
/// surface" rule.
const CONFIG_SETTLE: Duration = Duration::from_millis(100);

/// Soft cap on the [`super::EngineDriver::deferred_inputs`] queue. The
/// queue is unbounded by type; the debug assertion fires if it ever
/// approaches this size, signaling that the engine's emission shape
/// changed in a way that produces unbounded same-tick rejections. The
/// floor is comfortably above the worst-case fan-out from a single
/// rejected watch op (a Profile claim purge can cascade O(claim count)
/// at most a few dozen entries in practice).
const DEFERRED_INPUTS_SOFT_CAP: usize = 256;

impl<W: FsWatcher> EngineDriver<W> {
    /// One pass through the drain order. Public for unit tests
    /// (sibling tests drive a single tick with synthetic Hub state).
    ///
    /// When the pass resolves to shutdown (operator signal or a
    /// downstream channel disconnect) it runs
    /// [`Self::begin_shutdown`] before returning
    /// [`TickOutcome::Shutdown`], so the engine is probe-drained
    /// whether the daemon ([`Self::run`]) or a test drove the tick.
    ///
    /// MUST NOT be wrapped in `catch_unwind`: `ProbeSlot`'s in-unwind
    /// silence (`specter_core::probe`) depends on a mid-`step` panic
    /// being fatal — the graceful drain above is the *only* sanctioned
    /// path to a probe-free engine; catching a `step` panic would
    /// bypass it and resume on torn-down probe state.
    #[must_use]
    pub fn tick(&mut self) -> TickOutcome {
        let now = Instant::now();

        // Replay deferred inputs BEFORE the mio poll. Same-tick
        // engine consequences from the prior tick's rejections land
        // before any fresh kernel arrival is admitted.
        if self.replay_deferred_inputs(now).is_break() {
            return self.begin_shutdown();
        }

        // Compute the block-until timeout. `None` blocks forever
        // (mio honors `None` directly); `Some(ZERO)` polls once
        // non-blockingly so the next iteration's drains run without
        // a spurious wait.
        let timeout = self.compute_block_timeout(now);

        // Block on mio + drain every ready static Source non-blockingly.
        // `poll_and_drain` partitions the ready set into per-source
        // buckets on [`super::reactor::DrainedTick`]; the listener's
        // readiness surfaces as a bool flag (the accept itself runs
        // through [`super::Hub::drain_accept`] below).
        let mut drained = match self.reactor.poll_and_drain(timeout) {
            Ok(d) => d,
            Err(e) => {
                tracing::error!(?e, "mio poll failed; shutting down");
                return self.begin_shutdown();
            }
        };
        // Refresh `now` after the block — `poll_and_drain` may have
        // waited up to `timeout`, and downstream `engine.step` /
        // settle-expiry / timer pops compare against this value.
        let now = Instant::now();

        // 0. Listener accept — runs BEFORE every engine-input drain so
        //    a freshly accepted conn observes the same engine state
        //    this tick's projections will return. The dispatch loop in
        //    `poll_and_drain` only sets `drained.listener_ready`; the
        //    actual `accept(2)` loop lives on [`super::Hub`].
        //    Operator-perception of "ack happens before any sub diag"
        //    is preserved by the per-conn role-gate, independent of
        //    accept ordering.
        if drained.listener_ready
            && let Err(e) = self.ipc.drain_accept()
        {
            tracing::error!(?e, "ipc accept failed; shutting down");
            return self.begin_shutdown();
        }

        // 1. Sensor inputs — fs events + overflows + probe responses.
        if self.drain_sensor_inputs(&mut drained, now).is_break() {
            return self.begin_shutdown();
        }

        // 2. Expired timers — the engine hands back a `TimerEntry`
        //    pre-validated against the owning Profile's burst slot;
        //    forward (profile, kind, id) verbatim so the engine's
        //    dispatch routes directly without re-deriving owner/role.
        while let Some(entry) = self.engine.pop_expired(now) {
            let out = self.engine.step(
                Input::TimerExpired {
                    profile: entry.profile,
                    kind: entry.kind,
                    id: entry.id,
                },
                now,
            );
            if self.forward(out).is_break() {
                return self.begin_shutdown();
            }
        }

        // 3. Signals — iterate the batch; short-circuit on first
        //    Break (SIGINT/SIGTERM observed). SIGHUP routes through
        //    `dispatch_reload` and continues; the second SIGINT/SIGTERM
        //    within `HARD_EXIT_WINDOW` escalates inside
        //    `dispatch_signal_with_exit_fn`.
        for sig in drained.signals.iter().copied() {
            if self.dispatch_signal(sig, now).is_break() {
                return self.begin_shutdown();
            }
        }

        // 4. Config-event pulse + settle expiry. Order matters: arm
        //    the deadline BEFORE checking expiry so a same-tick pulse
        //    pushes the deadline forward (sustained editor bursts
        //    keep deferring the reload until edits actually settle).
        //    Inverting (expiry-then-pulse) would fire a reload in the
        //    middle of an in-flight burst.
        if drained.config_event_pulse {
            self.config_settle_until = Some(now + CONFIG_SETTLE);
        }
        if self.apply_config_settle_expiry(now).is_break() {
            return self.begin_shutdown();
        }

        // 5. Effect completions — already drained off the
        //    TOKEN_WAKER arm into `drained.effect_completions`. Each
        //    entry is an `Input::EffectComplete { .. }` envelope.
        for input in drained.effect_completions.drain(..) {
            let out = self.engine.step(input, now);
            if self.forward(out).is_break() {
                return self.begin_shutdown();
            }
        }

        // 5b. Actuator-gone signal. The drain above set
        //     `drained.actuator_gone` iff `effect_complete_rx`
        //     observed Disconnected (the actuator thread's
        //     `Box<dyn EffectCompleteSender>` dropped, closing its
        //     `Sender<Input>` clone via `WakingSink::Drop`). The check
        //     sits AFTER the completion drain so any in-flight
        //     completions queued ahead of the disconnect still flow
        //     through `engine.step` — losing those would orphan
        //     post-fire transitions on the way out. `begin_shutdown`
        //     is idempotent over an empty in-flight probe set; the
        //     same path covers both the clean exit (actuator finished
        //     shutdown phases naturally) and the crash exit (actuator
        //     panic). The hard-exit handshake stays installed up to
        //     and through this point because the SignalPipe lives on
        //     the Reactor lives on this driver — the second-SIGINT
        //     escalation reaches `dispatch_signal_inner`'s HardExit
        //     arm naturally.
        if drained.actuator_gone {
            return self.begin_shutdown();
        }

        // 6. Operator-IPC verb dispatch — read every ready conn,
        //    parse + handle each LF-delimited line, enqueue responses.
        //    Sits last in the engine-input drain order so every
        //    projection observes the freshest engine state.
        if self.drain_ipc_lines(&drained.ready_reads, now).is_break() {
            return self.begin_shutdown();
        }

        // 7. WRITABLE drain — flush write_queue residue for conns
        //    whose WRITABLE fired this tick. A conn whose drain
        //    reaches `Terminate` (close-after-flush or peer-gone) is
        //    closed immediately so the next tick's poll has no stale
        //    registration. `ConnGone` is the benign "read drain
        //    earlier this tick already terminated this conn" arm.
        for &token in &drained.ready_writes {
            match self.ipc.drain_writable(token) {
                hub::DrainWritableOutcome::Continue => {}
                hub::DrainWritableOutcome::Terminate => self.ipc.terminate_conn(token),
                hub::DrainWritableOutcome::ConnGone => {}
            }
        }
        // 8. Arm WRITABLE interest on any conn whose queue gained
        //    bytes this tick (via forward()'s fan-out or the IPC
        //    handler's response enqueue). One re-register pass at
        //    end-of-tick amortizes a per-conn syscall across the
        //    whole tick. Per-conn rearm failures terminate the
        //    failing conn in place ([`Hub::arm_writable_interests`]
        //    defer-terminates); the daemon stays alive.
        self.ipc.arm_writable_interests();

        TickOutcome::Continue
    }

    /// Compute the block-until duration handed to
    /// [`super::Reactor::poll_and_drain`].
    ///
    /// Three cases:
    /// - **`deferred_inputs` non-empty**: `Some(ZERO)` — return
    ///   immediately so the next iteration runs the replay before
    ///   re-blocking.
    /// - **No deadline armed**: `None` — mio blocks forever until any
    ///   Source readies. No `FOREVER_TIMEOUT` workaround is needed;
    ///   the previous epoch's crossbeam `Select::ready_timeout`
    ///   required a finite `Duration` (1 day fallback) because the
    ///   API didn't accept "wait indefinitely."
    /// - **Deadline armed (engine timer or settle expiry)**: the
    ///   `min` of the active deadlines, clamped to a non-negative
    ///   duration from `now`. `saturating_duration_since` returns
    ///   `ZERO` if the deadline already elapsed — the next poll is
    ///   non-blocking, and the body's drain pass fires the timer /
    ///   settle-expiry immediately.
    pub(super) fn compute_block_timeout(&self, now: Instant) -> Option<Duration> {
        if !self.deferred_inputs.is_empty() {
            return Some(Duration::ZERO);
        }
        [self.engine.next_deadline(), self.config_settle_until]
            .into_iter()
            .flatten()
            .min()
            .map(|d| d.saturating_duration_since(now))
    }

    /// Drain the deferred-input queue through `engine.step`. The
    /// queue is the producer-side counterpart of the inline
    /// `apply_watch_ops` rejection collector — a rejected op queues
    /// here, the next tick's pre-poll pass runs the rejection through
    /// the engine, and the resulting `forward` cycle dispatches the
    /// claim-purge.
    ///
    /// The debug-only soft cap catches a hypothetical future engine
    /// emission shape that produces unbounded same-tick rejections —
    /// the queue is unbounded by type, but a runaway producer would
    /// degrade tick latency unbounded.
    pub(super) fn replay_deferred_inputs(&mut self, now: Instant) -> ControlFlow<()> {
        while let Some(input) = self.deferred_inputs.pop_front() {
            let out = self.engine.step(input, now);
            if self.forward(out).is_break() {
                return ControlFlow::Break(());
            }
            debug_assert!(
                self.deferred_inputs.len() < DEFERRED_INPUTS_SOFT_CAP,
                "deferred_inputs growing unboundedly ({} entries) — \
                 engine emission shape changed?",
                self.deferred_inputs.len(),
            );
        }
        ControlFlow::Continue(())
    }

    /// Drain sensor inputs (fs events + overflows + probe responses)
    /// through `engine.step`, collapsing same-tick redundant recency
    /// hints into a single emission.
    ///
    /// The drain visits three [`super::reactor::DrainedTick`] fields:
    /// 1. `fs_events` — per-resource recency hints + identity-class
    ///    edges from the watcher fd.
    /// 2. `sensor_overflows` — kernel-level overflow markers
    ///    (inotify's `IN_Q_OVERFLOW`; kqueue never emits).
    /// 3. `probe_responses` — already-lifted `Input::ProbeResponse(_)`
    ///    envelopes from the wake'd channel.
    ///
    /// Recency-class [`FsEvent`]s (`Modified` / `MetadataChanged` /
    /// `StructureChanged`) are lossy hints whose sole truth is the
    /// next probe — same-tick duplicates collapse via a per-tick
    /// `BTreeSet<(ResourceId, FsEvent)>` horizon. Every other input
    /// is a barrier (identity-class lifecycle facts, probe responses,
    /// overflow markers) — barriers clear the horizon and step
    /// verbatim. The split is exhaustive over the [`FsEvent`] enum,
    /// so any future variant defaults to the safe (barrier) side via
    /// the `is_recency()` projection.
    ///
    /// Soundness rests on the engine's lossy-hint contract, not a
    /// fragile bit-for-bit `StepOutput` identity. A dropped same-tick
    /// duplicate, had it been delivered, would have either no-op'd an
    /// idempotent dispatch guard or elided only re-work the next
    /// probe re-establishes regardless — it can never change a
    /// fire/no-fire verdict, probe target, timer deadline, or
    /// baseline.
    fn drain_sensor_inputs(
        &mut self,
        drained: &mut super::reactor::DrainedTick,
        now: Instant,
    ) -> ControlFlow<()> {
        let mut seen: BTreeSet<(ResourceId, FsEvent)> = BTreeSet::new();
        for (resource, event) in drained.fs_events.drain(..) {
            if event.is_recency() {
                // First occurrence drives the engine; a later
                // same-`(resource, event)` recency hint is provably
                // redundant — dropped before step/forward.
                if !seen.insert((resource, event)) {
                    continue;
                }
            } else {
                // Barrier: identity FsEvent. Drop the horizon so the
                // next same-`(resource, event)` recency hint reaches
                // the engine — a `Removed` between two `Modified`s
                // must not be hidden behind the dedup.
                seen.clear();
            }
            let out = self.engine.step(Input::FsEvent { resource, event }, now);
            if self.forward(out).is_break() {
                return ControlFlow::Break(());
            }
        }
        // Overflows AFTER fs events: an `IN_Q_OVERFLOW` reseed
        // covers the kernel-lost edges; running it after the
        // visible-edge drain means the reseed sees the freshest
        // engine state for the in-scope Profiles.
        for scope in drained.sensor_overflows.drain(..) {
            seen.clear();
            let out = self.engine.step(Input::SensorOverflow { scope }, now);
            if self.forward(out).is_break() {
                return ControlFlow::Break(());
            }
        }
        // Probe responses AFTER overflow: an overflow that arrived
        // mid-burst should reseed before the in-flight probe response
        // would have been stale-fenced anyway. Step order preserves
        // the engine's correlation-gate discipline.
        for input in drained.probe_responses.drain(..) {
            let out = self.engine.step(input, now);
            if self.forward(out).is_break() {
                return ControlFlow::Break(());
            }
        }
        ControlFlow::Continue(())
    }
}
