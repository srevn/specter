//! One pass through the driver's drain order — the load-bearing
//! single-iteration body, and the module new inbound-path work lands
//! in.
//!
//! [`EngineDriver::tick`] drains, in order: sensor inputs → expired
//! timers → reload (SIGHUP) pulses → config-event pulses + settle
//! expiry → effect completions → then blocks on `Select::ready_timeout`
//! until any source readies (a timer deadline elapses, or shutdown).
//! The settle-expiry filter and `handle_reload` itself live in
//! [`super::reload`]; downstream dispatch in [`super::forward`].
//!
//! **Drain order rationale.** Sensor inputs (FsEvents) drain *before*
//! effect completions because the fire-cycle's post-fire tail
//! (`PostFirePhase::Awaiting` / `Rebasing`) absorbs FsEvents and folds
//! their disk state into the rebase, while `EffectComplete` arrivals
//! transition `Awaiting → Rebasing`. If the order were inverted, an
//! `EffectComplete` could move the burst into Rebasing before the
//! engine had seen FsEvents queued in the same tick — those events
//! would then route to the wrong burst (or kick off a fresh burst
//! against an in-flight rebase). Sensor-first preserves the
//! "fire-tail absorbs concurrent edits" contract documented on
//! `PostFirePhase::Awaiting`.
//!
//! **Auto-reload settle pipeline.** The config-event drain re-arms
//! `config_settle_until` to `now + CONFIG_SETTLE` per pulse — sustained
//! editor bursts (atomic-save sequences, in-place writes) defer the
//! reload until quiet. Apply-side: on settle expiry, a single `lstat`
//! of `config_path` filters phantom pulses (kqueue parent-dir
//! spillover from sibling writes); on confirmed [`FileMeta`] drift the
//! driver runs the same [`Self::handle_reload`] SIGHUP uses, so
//! meta-rotation discipline converges across the two pulse sources.
//! Config-event drain sits *after* the SIGHUP drain so an in-flight
//! SIGHUP rotates `loader.config_meta` first — the subsequent
//! settle-expiry's lstat then compares against the freshly-rotated
//! identity and silent-drops the redundant edit. Drain sits *before*
//! effect completions for the same reason as SIGHUP: file I/O latency
//! lands on this thread, and effect drain stays tight by following
//! both reload sources.
//!
//! **Disconnect policy.** Every inbound drain uses an explicit
//! `try_recv` match: `Empty` breaks the loop, `Disconnected` returns
//! [`TickOutcome::Shutdown`] via [`EngineDriver::begin_shutdown`].
//! `sensor_in_rx` Disconnect is canonically Terminal (the sole truth
//! source for fs state); `reload_signal_rx`, `config_event_rx` (when
//! present), and `effect_in_rx` are defensively Terminal — their
//! producers' deaths also drop `shutdown_engine_tx` (signal thread)
//! or the channel the engine needs to make further progress
//! (actuator), so the canonical shutdown converges either way. The
//! auto-reload arm is gated on `Option<Receiver>`: under
//! `--no-config-watch` or a watcher-init failure neither the drain
//! nor the `Select` arm registers, so a missing producer is the
//! absence of a wire rather than a disconnected wire. A `forward`
//! that observes shutdown mid-stream (via the per-send race in
//! [`super::forward::EngineDriver::forward`]) propagates `Break`
//! upward and joins the same `begin_shutdown` exit.
//!
//! `Select::ready_timeout` is a *peek* primitive — the message stays in
//! its channel and the next iteration's `try_recv` drain re-imposes
//! the drain ordering. The deadline math feeds `next_deadline` from the
//! engine's timer heap; `None` (no timers armed) maps to a 1-day fallback.

use super::state::ReloadTrigger;
use super::{EngineDriver, TickOutcome};
use crossbeam::channel::{Select, TryRecvError};
use specter_core::{FsEvent, Input, ResourceId};
use std::collections::BTreeSet;
use std::ops::ControlFlow;
use std::time::{Duration, Instant};

/// `1 day` — the fallback timeout when the engine has no armed timers.
/// `Select::ready_timeout` requires a `Duration`; "never" needs to be
/// an absurdly-long-but-finite span. A spurious wake every 24h is not
/// a concern; the next tick re-blocks identically.
const FOREVER_TIMEOUT: Duration = Duration::from_hours(24);

/// Auto-reload settle window. Each config-event pulse re-arms
/// `EngineDriver::config_settle_until` to `now + CONFIG_SETTLE`;
/// quiet for a full window expires the deadline and the driver runs
/// the lstat-vs-`loader.config_meta` filter (and on drift,
/// `handle_reload`).
///
/// `100ms` covers the editor patterns the design targets — atomic save
/// (vim, Helix: write-tmp → rename → fsync; ~10–30ms wall) and
/// in-place modify (`echo > file` ; ~1–5ms per syscall, sustained
/// bursts well under 100ms). Fixed in v1; not operator-tunable per the
/// project's "minimal config surface" alpha rule.
const CONFIG_SETTLE: Duration = Duration::from_millis(100);

impl EngineDriver {
    /// One pass through the drain order. Public for unit tests
    /// (sibling tests drive a single tick with mock channels).
    ///
    /// When the pass resolves to shutdown (operator signal or sensor
    /// disconnect) it runs [`Self::begin_shutdown`] before returning
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

        // Drain sensor inputs, collapsing same-tick redundant recency
        // hints. `Break` ⇒ either `sensor_in` is disconnected or a
        // downstream `forward` raced shutdown; both route through the
        // shared `begin_shutdown` exit below.
        if self.drain_sensor(now).is_break() {
            return self.begin_shutdown();
        }

        // Drain expired timers. The engine hands back a `TimerEntry`
        // pre-validated against the owning Profile's burst slot; we
        // forward (profile, kind, id) verbatim so the engine's dispatch
        // routes directly without re-deriving owner/role.
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

        // Drain reload pulses (file I/O on this thread). SIGHUP is
        // the only pulse source for this channel — auto-reload uses
        // `config_event_rx` + settle-expiry — so every drained pulse
        // here is attributed to `Sighup`.
        loop {
            match self.sides.reload_signal_rx.try_recv() {
                Ok(()) => {
                    if self.handle_reload(ReloadTrigger::Sighup, now).is_break() {
                        return self.begin_shutdown();
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return self.begin_shutdown(),
            }
        }

        // Drain auto-reload pulses (re-arm settle per pulse), then
        // check whether the settle window has elapsed and (on
        // confirmed meta drift) run handle_reload. Gated on the
        // option: under `--no-config-watch` or a watcher-init failure
        // the engine bundle carries no receiver, so neither the
        // drain nor the Select arm exists — a disconnected receiver
        // can't busy-loop the tick because the receiver itself isn't
        // there. Order matters: drain-then-expiry implements
        // "settle resets per pulse" — a pulse arriving in the same
        // tick as a stale deadline pushes the deadline forward, so a
        // sustained editor burst keeps deferring the reload until the
        // edits actually settle. Inverting (expiry-then-drain) would
        // fire a reload in the middle of an in-flight burst.
        if let Some(rx) = &self.sides.config_event_rx {
            loop {
                match rx.try_recv() {
                    Ok(()) => {
                        self.config_settle_until = Some(now + CONFIG_SETTLE);
                    }
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => return self.begin_shutdown(),
                }
            }
        }
        if self.apply_config_settle_expiry(now).is_break() {
            return self.begin_shutdown();
        }

        // Drain effect completions. Disconnected here is terminal:
        // the actuator thread is the sole producer, so its death
        // means outstanding effects will never reap and further
        // engine progress is wedged on `gate_deadline` recovery
        // alone — shut down instead.
        loop {
            match self.sides.effect_in_rx.try_recv() {
                Ok(input) => {
                    let out = self.engine.step(input, now);
                    if self.forward(out).is_break() {
                        return self.begin_shutdown();
                    }
                }
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return self.begin_shutdown(),
            }
        }

        // Block until any source readies or timer fires. Deadlines come
        // from two independent sources: the engine's internal timer
        // heap, and the auto-reload settle window. Both are
        // `Option<Instant>`; `flatten` discards un-armed sources and
        // `min` picks the soonest.
        let timeout = [self.engine.next_deadline(), self.config_settle_until]
            .into_iter()
            .flatten()
            .min()
            .map_or(FOREVER_TIMEOUT, |d| {
                d.saturating_duration_since(Instant::now())
            });

        // Scope the `Select`: it borrows `&self.sides.*`, while the
        // shutdown drain below needs `&mut self`. Resolving to a
        // `bool` inside the block drops `sel` (and its borrows) before
        // `begin_shutdown` takes the mutable borrow.
        let shutting_down = {
            let mut sel = Select::new();
            let _i_sensor = sel.recv(&self.sides.sensor_in_rx);
            let _i_effect = sel.recv(&self.sides.effect_in_rx);
            let _i_reload = sel.recv(&self.sides.reload_signal_rx);
            // Auto-reload wakes the driver from a long block when a
            // config-event pulse arrives. The arm registers only when
            // the engine bundle carries a receiver — i.e., the config
            // watcher thread spawned successfully. Under
            // `--no-config-watch` (or a watcher init failure) the arm
            // is omitted entirely, so crossbeam's `Select::ready_timeout`
            // cannot report a non-existent (or disconnected) receiver
            // as immediately-ready. The shutdown arm's `i_shutdown` is
            // computed at registration, so the `idx == i_shutdown`
            // comparison below is index-agnostic — it works whether
            // the auto-reload arm sits at slot 3 (off) or 4 (on).
            if let Some(rx) = self.sides.config_event_rx.as_ref() {
                sel.recv(rx);
            }
            let i_shutdown = sel.recv(&self.sides.shutdown_engine_rx);
            matches!(sel.ready_timeout(timeout), Ok(idx) if idx == i_shutdown)
        };

        if shutting_down {
            self.begin_shutdown()
        } else {
            TickOutcome::Continue
        }
    }

    /// Drain queued sensor inputs, collapsing same-tick redundant
    /// recency hints into a single `engine.step`.
    ///
    /// `sensor_in_rx` carries exactly four shapes: the watcher
    /// thread's [`Input::FsEvent`] and [`Input::SensorOverflow`],
    /// `apply_watch_op`'s [`Input::WatchOpRejected`], and the prober
    /// pool's [`Input::ProbeResponse`]. A *recency-class*
    /// [`Input::FsEvent`] ([`FsEvent::is_recency`] —
    /// `Modified` / `MetadataChanged` / `StructureChanged`) is a lossy
    /// "this resource changed in this class" hint whose sole truth is
    /// the next probe. [`Self::tick`] samples `now` once and threads
    /// it through the whole drain, so the engine's settle deadline
    /// (`last_event_time`) is byte-identical whether a second
    /// same-`(resource, event)` hint *within this tick* is delivered
    /// or dropped — same-tick is therefore the maximal lossless
    /// collapse boundary (collapsing across ticks would move the
    /// deadline). The first occurrence drives the engine; later
    /// duplicates are dropped before `step` (hence before `forward`,
    /// so no redundant watch-op send or `wake`).
    ///
    /// Every other input is a **barrier**: an identity
    /// [`Input::FsEvent`] (`Removed` / `Renamed` / `Revoked` —
    /// terminal lifecycle facts), a [`Input::ProbeResponse`] (can
    /// move a Profile `Verifying → PostFire`), a
    /// [`Input::SensorOverflow`] (reseeds in-scope Profiles), a
    /// [`Input::WatchOpRejected`] (can purge a claim). Each can
    /// change how a later same-`(resource, event)` hint dispatches,
    /// so the dedup horizon is cleared and the input stepped
    /// verbatim, preserving drain order. The recency/barrier split is
    /// *total*, so any future `Input` reaching this channel defaults
    /// to the safe (barrier) side.
    ///
    /// Soundness rests on the engine's lossy-hint contract, not a
    /// fragile bit-for-bit `StepOutput` identity. A dropped same-tick
    /// duplicate, had it been delivered, would have either no-op'd an
    /// idempotent dispatch guard (a `Batching` burst re-notes the
    /// same `(id, path)` at the same `now`; a descent or
    /// anchor-recovery hits its in-flight-probe guard) or elided only
    /// re-work the next probe re-establishes regardless (a
    /// trace-level fire-tail diagnostic; one redundant — and
    /// idempotent — promoter enumeration). It can never change a
    /// fire/no-fire verdict, probe target, timer deadline, or
    /// baseline. The horizon is a per-tick local bounded by the
    /// distinct dirty resources between barriers and freed at every
    /// barrier — never a second unbounded buffer behind the unbounded
    /// `sensor_in` channel.
    ///
    /// [`ControlFlow::Break`] ⇒ either `sensor_in` is disconnected or
    /// a downstream `forward` raced shutdown. The caller
    /// ([`Self::tick`]) routes the carrier through
    /// [`Self::begin_shutdown`] at the one tick-level boundary, the
    /// same shape the other lifecycle helpers here use.
    fn drain_sensor(&mut self, now: Instant) -> ControlFlow<()> {
        let mut seen: BTreeSet<(ResourceId, FsEvent)> = BTreeSet::new();
        loop {
            match self.sides.sensor_in_rx.try_recv() {
                Ok(Input::FsEvent { resource, event }) if event.is_recency() => {
                    // First occurrence this tick drives the engine; a
                    // later same-`(resource, event)` recency hint is
                    // provably redundant — dropped before step/forward.
                    if seen.insert((resource, event)) {
                        let out = self.engine.step(Input::FsEvent { resource, event }, now);
                        if self.forward(out).is_break() {
                            return ControlFlow::Break(());
                        }
                    }
                }
                Ok(other) => {
                    // Barrier: identity FsEvent or any non-FsEvent
                    // input. Drop the horizon, step verbatim, in order.
                    seen.clear();
                    let out = self.engine.step(other, now);
                    if self.forward(out).is_break() {
                        return ControlFlow::Break(());
                    }
                }
                Err(TryRecvError::Empty) => return ControlFlow::Continue(()),
                Err(TryRecvError::Disconnected) => return ControlFlow::Break(()),
            }
        }
    }
}
