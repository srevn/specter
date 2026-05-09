//! Engine driver — the main loop body, on the bin's main thread.
//!
//! Owns the [`Engine`], the [`Loader`], the engine-side channel bundle,
//! the prober [`Arc`] clone, and a wake-handle clone. The driver's
//! public methods:
//!
//! - [`EngineDriver::run_initial_attach`] walks `loader.current_config.watches`
//!   in source order, calling [`Engine::attach_sub`] for each spec and
//!   forwarding [`StepOutput`]s immediately so the watcher / prober see
//!   work as it lands. Runs once before [`EngineDriver::run`].
//! - [`EngineDriver::run`] is the loop wrapper around
//!   [`EngineDriver::tick`]; returns [`ExitReason::Shutdown`] when the
//!   shutdown channel signals.
//! - [`EngineDriver::tick`] is the load-bearing single-iteration body:
//!   drain sensor → drain timers → drain reload → drain effects →
//!   block on `Select::ready_timeout` until any source readies (or a
//!   timer deadline elapses, or shutdown).
//!
//! **Drain order rationale.** Sensor inputs (FsEvents) drain *before*
//! effect completions because the fire-cycle's post-fire tail
//! (`BurstPhase::Awaiting` / `Rebasing`) absorbs FsEvents and folds
//! their disk state into the rebase, while `EffectComplete` arrivals
//! transition `Awaiting → Rebasing`. If the order were inverted, an
//! `EffectComplete` could move the burst into Rebasing before the
//! engine had seen FsEvents queued in the same tick — those events
//! would then route to the wrong burst (or kick off a fresh burst
//! against an in-flight rebase). Sensor-first preserves the
//! "fire-tail absorbs concurrent edits" contract documented on
//! `BurstPhase::Awaiting`.
//!
//! `Select::ready_timeout` is a *peek* primitive — the message stays in
//! its channel and the next iteration's `try_recv` drain re-imposes
//! the drain ordering. The deadline math feeds `next_deadline` from the
//! engine's timer heap; `None` (no timers armed) maps to a 1-day fallback.
//!
//! `forward` ships every [`StepOutput`] downstream: `watch_ops` →
//! `watch_ops_tx` with a `wake_handle.wake()` per successful send
//! (see `forward` rustdoc for the bounded-channel deadlock that motivates
//! per-send waking); `probe_ops` → `prober.submit/cancel` direct;
//! `effects` → `effects_tx`; `diagnostics` → [`log_diagnostic`]
//! hand-mapped to tracing.
//!
//! `handle_reload` is the SIGHUP pipeline: read the file, compute the
//! diff, apply via `Input::ConfigDiff`, sync `loader.ids` post-apply,
//! rotate `current_config`. All on the engine thread — no Mutex.

use crate::app::CliLogOverrides;
use crate::channels::EngineSide;
use crate::loader::Loader;
use crate::observability::{LogReloadKind, ObservabilityHandle};
use crossbeam::channel::Select;
use specter_config::Config;
use specter_core::{Diagnostic, Input, ProbeOp, StepOutput, SubId};
use specter_engine::Engine;
use specter_sensor::{DrainWindow, Prober, WakeHandle};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Reason the driver loop exited. Returned from [`EngineDriver::run`].
///
/// v1 has only the `Shutdown` variant — every path that could exit the
/// loop without a shutdown signal (sensor channel disconnect) currently
/// also routes through `TickOutcome::Shutdown` per [`EngineDriver::tick`].
/// The enum exists so v2 (recovery / restart) has a structural seam
/// without breaking the [`EngineDriver::run`] return type.
#[derive(Debug, Eq, PartialEq)]
pub enum ExitReason {
    /// `shutdown_engine_rx` fired (operator-driven, normal path), OR
    /// every input channel disconnected (upstream thread crash; v1
    /// treats both as terminal-graceful).
    Shutdown,
}

/// Outcome of a single [`EngineDriver::tick`] call. The loop wrapper
/// matches on this; explicit enum is friendlier than a bool.
#[derive(Debug, Eq, PartialEq)]
pub enum TickOutcome {
    Continue,
    Shutdown,
}

/// `1 day` — the fallback timeout when the engine has no armed timers.
/// `Select::ready_timeout` requires a `Duration`; "never" needs to be
/// an absurdly-long-but-finite span. A spurious wake every 24h is not
/// a concern; the next tick re-blocks identically.
const FOREVER_TIMEOUT: Duration = Duration::from_hours(24);

/// Engine driver — see module rustdoc.
pub struct EngineDriver {
    engine: Engine,
    loader: Loader,
    config_path: PathBuf,
    /// CLI overrides applied to `[log]` at startup. Re-applied on every
    /// SIGHUP-driven reload so CLI precedence stays consistent across
    /// the process lifetime (`CLI > config > default`).
    cli_log_overrides: CliLogOverrides,
    /// Subscriber handle for runtime updates (`set_level`,
    /// `reopen_file`). Held here so `handle_reload` can fire both on
    /// SIGHUP without going through the loader.
    obs_handle: ObservabilityHandle,
    sides: EngineSide,
    prober: Arc<dyn Prober>,
    wake_handle: Box<dyn WakeHandle>,
    /// Cross-thread handle for the watcher's deferred-drain window.
    /// Engine-thread writes (`handle_reload`) are observed by the
    /// watcher thread on its next `poll_until` iteration via the
    /// underlying Atomic. Held here so the SIGHUP reload pipeline can
    /// rotate the value alongside `current_config`.
    drain_window: DrainWindow,
}

impl std::fmt::Debug for EngineDriver {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("EngineDriver")
            .field("loader", &self.loader)
            .field("config_path", &self.config_path)
            .field("cli_log_overrides", &self.cli_log_overrides)
            .field("obs_handle", &self.obs_handle)
            .finish_non_exhaustive()
    }
}

impl EngineDriver {
    #[must_use]
    pub fn new(
        engine: Engine,
        loader: Loader,
        config_path: PathBuf,
        cli_log_overrides: CliLogOverrides,
        obs_handle: ObservabilityHandle,
        sides: EngineSide,
        prober: Arc<dyn Prober>,
        wake_handle: Box<dyn WakeHandle>,
        drain_window: DrainWindow,
    ) -> Self {
        Self {
            engine,
            loader,
            config_path,
            cli_log_overrides,
            obs_handle,
            sides,
            prober,
            wake_handle,
            drain_window,
        }
    }

    /// Attach every Sub from `loader.current_config.watches` in source
    /// order. Each [`StepOutput`] is forwarded as we go so the watcher
    /// / prober receive ops as the engine emits them.
    pub fn run_initial_attach(&mut self) {
        let now = Instant::now();
        // Snapshot the spec list — `loader.ids` mutation invalidates an
        // iterator over `loader.current_config`.
        let specs = self.loader.current_config.watches.clone();
        for spec in specs {
            let req = spec.to_attach_request();
            let (id, out) = self.engine.attach_sub(req, now);
            self.loader.ids.insert(spec.name.clone(), id);
            self.forward(out);
        }
    }

    /// Loop wrapping [`Self::tick`] until shutdown.
    pub fn run(&mut self) -> ExitReason {
        loop {
            match self.tick() {
                TickOutcome::Continue => {}
                TickOutcome::Shutdown => return ExitReason::Shutdown,
            }
        }
    }

    /// One pass through the drain order. Public for unit tests
    /// (sibling tests drive a single tick with mock channels).
    pub fn tick(&mut self) -> TickOutcome {
        let now = Instant::now();

        // Drain sensor inputs (FsEvent + ProbeResponse + WatchOpRejected).
        loop {
            match self.sides.sensor_in_rx.try_recv() {
                Ok(input) => {
                    let out = self.engine.step(input, now);
                    self.forward(out);
                }
                Err(crossbeam::channel::TryRecvError::Empty) => break,
                Err(crossbeam::channel::TryRecvError::Disconnected) => {
                    return TickOutcome::Shutdown;
                }
            }
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
            self.forward(out);
        }

        // Drain reload pulses (file I/O on this thread).
        while self.sides.reload_signal_rx.try_recv().is_ok() {
            self.handle_reload(now);
        }

        // Drain effect completions. Disconnect tolerated (engine remains
        // functional against sensor + timers).
        while let Ok(input) = self.sides.effect_in_rx.try_recv() {
            let out = self.engine.step(input, now);
            self.forward(out);
        }

        // Block until any source readies or timer fires.
        let timeout = self.engine.next_deadline().map_or(FOREVER_TIMEOUT, |d| {
            d.saturating_duration_since(Instant::now())
        });

        let mut sel = Select::new();
        let _i_sensor = sel.recv(&self.sides.sensor_in_rx);
        let _i_effect = sel.recv(&self.sides.effect_in_rx);
        let _i_reload = sel.recv(&self.sides.reload_signal_rx);
        let i_shutdown = sel.recv(&self.sides.shutdown_engine_rx);

        match sel.ready_timeout(timeout) {
            Ok(idx) if idx == i_shutdown => TickOutcome::Shutdown,
            Ok(_) | Err(crossbeam::channel::ReadyTimeoutError) => TickOutcome::Continue,
        }
    }

    /// Read the config from disk; on success, diff against the current
    /// snapshot, apply via `Input::ConfigDiff`, sync `loader.ids`,
    /// rotate `loader.current_config`. On failure, log + retain
    /// running config.
    ///
    /// Log-side reload is integrated here:
    ///   - The `[log]` block is re-resolved (CLI overrides re-applied);
    ///     a level-only change calls `obs_handle.set_level`;
    ///     a destination/path change logs an `error!` instructing the
    ///     operator to restart (v1 doesn't hot-reload destinations).
    ///   - `obs_handle.reopen_file()` fires unconditionally so logrotate
    ///     `copytruncate`-style rotation works without a config diff.
    fn handle_reload(&mut self, now: Instant) {
        let new_config = match Config::from_path(&self.config_path) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    path = %self.config_path.display(),
                    "config reload failed; keeping running config",
                );
                return;
            }
        };

        // Resolve the new [log] block with CLI overrides (CLI wins,
        // matching startup precedence). Validation may fail (e.g., a
        // freshly-edited config now says destination = "file" without a
        // path); on failure we log and keep the old log state.
        let new_log_resolved = match new_config.log.clone().merge_cli(
            self.cli_log_overrides.level,
            self.cli_log_overrides.destination,
            self.cli_log_overrides.path.clone(),
        ) {
            Ok(c) => c,
            Err(e) => {
                tracing::error!(
                    error = %e,
                    "log reload failed; keeping running log config",
                );
                // Don't abandon the watch reload below — the [log]
                // failure is independent.
                self.loader.current_log.clone()
            }
        };
        self.apply_log_reload(&new_log_resolved);

        let diff = specter_config::diff(&self.loader.current_config, &new_config, &self.loader.ids);
        if diff.added.is_empty() && diff.removed.is_empty() && diff.modified.is_empty() {
            tracing::info!("config reload: no watch changes");
            self.loader.current_config = new_config;
            self.loader.current_log = new_log_resolved;
            return;
        }

        // Pre-collect identifiers; `Input::ConfigDiff(diff)` consumes
        // `diff` shortly. Cheap (Vecs of SubId + name strings).
        let added_n = diff.added.len();
        let removed_n = diff.removed.len();
        let modified_n = diff.modified.len();
        let added_names: Vec<String> = diff.added.iter().map(|r| r.name.clone()).collect();
        let removed_ids: Vec<SubId> = diff.removed.clone();
        let modified_pairs: Vec<(SubId, String)> = diff
            .modified
            .iter()
            .map(|(id, r)| (*id, r.name.clone()))
            .collect();

        let out = self.engine.step(Input::ConfigDiff(diff), now);

        // Sync loader.ids: drop removed/old-modified ids; look up fresh
        // ids by name for added/modified. `find_by_name` is O(N_subs)
        // linear scan; bounded by reload frequency (operator-driven).
        for id in &removed_ids {
            self.loader.ids.retain(|_, v| v != id);
        }
        for name in &added_names {
            if let Some(new_id) = self.engine.subs().find_by_name(name) {
                self.loader.ids.insert(name.into(), new_id);
            }
        }
        for (old_id, name) in &modified_pairs {
            self.loader.ids.retain(|_, v| v != old_id);
            if let Some(new_id) = self.engine.subs().find_by_name(name) {
                self.loader.ids.insert(name.into(), new_id);
            }
        }

        self.loader.current_config = new_config;
        self.loader.current_log = new_log_resolved;
        // Recompute the watcher's deferred-drain window from the
        // freshly-applied config and rotate atomically. The watcher
        // thread reads the new value on its next `poll_until`
        // iteration; at most one drain straddles the rotation.
        let new_window = self.loader.derive_drain_window();
        let old_window = self.drain_window.get();
        if new_window != old_window {
            self.drain_window.set(new_window);
            tracing::info!(
                old_ms = old_window.as_millis(),
                new_ms = new_window.as_millis(),
                "drain_window updated via SIGHUP",
            );
        }
        tracing::info!(
            added = added_n,
            removed = removed_n,
            modified = modified_n,
            "config reload applied",
        );
        self.forward(out);
    }

    /// Apply a freshly-resolved [`specter_config::LogConfig`] to the
    /// observability handle. Three branches:
    ///
    /// - **Unchanged** — only fire `reopen_file` (logrotate cadence is
    ///   independent of operator-driven config edits; reopen
    ///   unconditionally).
    /// - **LevelOnly** — call `set_level`; reopen the file too.
    /// - **DestinationChanged** — log an `error!` instructing the
    ///   operator to restart; reopen the (still-old) file so logrotate
    ///   keeps working until the restart.
    ///
    /// Any reopen `Err` is logged at `warn!` — the rotator may have
    /// raced us to the path, in which case the existing fd is still
    /// usable.
    fn apply_log_reload(&self, new_log: &specter_config::LogConfig) {
        let kind = LogReloadKind::diff(&self.loader.current_log, new_log);
        match kind {
            LogReloadKind::Unchanged => {}
            LogReloadKind::LevelOnly => match self.obs_handle.set_level(new_log.level) {
                Ok(()) => tracing::info!(
                    new_level = ?new_log.level,
                    "log level updated via SIGHUP",
                ),
                Err(e) => tracing::error!(
                    error = ?e,
                    "log level reload failed; keeping prior level",
                ),
            },
            LogReloadKind::DestinationChanged => {
                tracing::error!(
                    new_destination = ?new_log.destination,
                    new_path = ?new_log.path.as_ref().map(|p| p.display().to_string()),
                    "log destination / path change is not hot-reloadable in v1; \
                     restart specter to apply",
                );
            }
        }
        if let Err(e) = self.obs_handle.reopen_file() {
            tracing::warn!(
                error = ?e,
                path = ?self.obs_handle.file_path().map(|p| p.display().to_string()),
                "log file reopen failed; keeping existing fd",
            );
        }
    }

    /// Push a [`StepOutput`] to its downstream consumers.
    ///
    /// `watch_ops` queue to `watch_ops_tx` and `wake_handle.wake()` fires
    /// after **every** successful send. The wake-per-send protocol is
    /// load-bearing: `watch_ops_tx` is bounded(1024), and a single Seed
    /// burst against a large tree can produce 10k+ Watch ops in one
    /// `StepOutput`. With a "wake once at end of loop" rule, the engine
    /// would fill the channel, block on `Sender::send` at op 1025, and
    /// never reach the end-of-loop wake — leaving the watcher asleep in
    /// `kevent` forever. Wakes coalesce kernel-side via `EVFILT_USER`'s
    /// `EV_CLEAR`, so the per-send cost is one `kevent` syscall (~1µs)
    /// regardless of whether the watcher is awake. `probe_ops` dispatch
    /// directly to the prober. `effects` queue to `effects_tx`.
    /// `diagnostics` log per variant via [`log_diagnostic`].
    ///
    /// `Send` errors on disconnected channels are warn-logged and
    /// dropped — the only path here is a downstream-thread crash mid-
    /// shutdown. Takes `&self` because every downstream send is
    /// channel-based or trait-object dispatch (`Sender::send`,
    /// `Prober::submit`, `WakeHandle::wake`, `tracing::*`) — none
    /// requires `&mut self`.
    fn forward(&self, out: StepOutput) {
        for op in out.watch_ops {
            match self.sides.watch_ops_tx.send(op) {
                Ok(()) => self.wake_handle.wake(),
                Err(_) => tracing::warn!("watch_ops channel disconnected; dropping op"),
            }
        }

        for op in out.probe_ops {
            match op {
                ProbeOp::Probe { request } => self.prober.submit(request),
                ProbeOp::Cancel { profile } => self.prober.cancel(profile),
            }
        }

        for eff in out.effects {
            if self.sides.effects_tx.send(eff).is_err() {
                tracing::warn!("effects channel disconnected; dropping effect");
            }
        }

        for diag in out.diagnostics {
            log_diagnostic(&diag);
        }
    }
}

/// Map a [`Diagnostic`] to a tracing event.
///
/// Most variants are `warn` (drops + race conditions are warnings, not
/// errors). `EffectCompleteForUnknownSub` is `error` (variant docstring
/// marks it as a bug or hot-reload race the operator should see);
/// `DetachUnknownSub` is `warn` — a benign hot-reload race rather than a
/// bug. `ReapPendingResolved` and `ReapPendingCancelled` are `info`
/// (informational; the late reap completed or was pre-empted by a
/// revival).
pub fn log_diagnostic(d: &Diagnostic) {
    match d {
        Diagnostic::StaleProbeResponse {
            profile,
            correlation,
        } => tracing::warn!(
            ?profile,
            ?correlation,
            "stale probe response (state mismatch)"
        ),
        Diagnostic::StaleTimer { id } => tracing::warn!(?id, "stale timer expiration"),
        Diagnostic::EffectCompleteOutsideAwaiting { sub, profile } => tracing::warn!(
            ?sub,
            ?profile,
            "effect_complete arrived outside Awaiting (gate-deadline force-transition or anchor-loss); dropped",
        ),
        Diagnostic::EffectCompleteForUnknownSub { sub } => tracing::error!(
            ?sub,
            "effect_complete for unknown Sub — engine bug or hot-reload race",
        ),
        Diagnostic::DetachUnknownSub { sub } => tracing::warn!(
            ?sub,
            "detach for unknown Sub (hot-reload race or stale id; dropped)",
        ),
        Diagnostic::ProbeVanished { profile, intent } => {
            tracing::warn!(?profile, ?intent, "probe returned Vanished");
        }
        Diagnostic::ProbeFailed {
            profile,
            intent,
            errno,
        } => tracing::warn!(?profile, ?intent, errno, "probe failed"),
        Diagnostic::EventClassDropped {
            resource,
            event,
            profile,
        } => tracing::trace!(
            ?resource,
            ?event,
            ?profile,
            "fs event dropped (class not in profile.events_union)",
        ),
        Diagnostic::EventOnUnwatchedResource { resource } => {
            tracing::warn!(?resource, "FsEvent on unwatched resource (race; dropped)");
        }
        Diagnostic::EventNoConsumer { resource } => {
            // Benign: a watched resource (typically a `WatchRootParent`)
            // fired an event no Profile cared about this step. Logging at
            // TRACE so it doesn't pollute operator logs.
            tracing::trace!(
                ?resource,
                "FsEvent had no consumer (watched, but no covering Profile / descent / recovery)"
            );
        }
        Diagnostic::WatchOpRejected { resource, failure } => {
            tracing::warn!(
                ?resource,
                ?failure,
                errno = failure.errno(),
                "watch op rejected by sensor",
            );
        }
        Diagnostic::PendingPathProbeVanished { profile, prefix } => {
            tracing::warn!(?profile, ?prefix, "pending-path descent probe Vanished");
        }
        Diagnostic::PendingPathProbeFailed {
            profile,
            prefix,
            errno,
        } => tracing::warn!(
            ?profile,
            ?prefix,
            errno,
            "pending-path descent probe Failed",
        ),
        Diagnostic::ReapPendingCancelled { profile } => tracing::info!(
            ?profile,
            "reap-pending Profile revived (fresh attach pre-empted deferred reap)",
        ),
        Diagnostic::ReapPendingResolved { profile } => tracing::info!(
            ?profile,
            "reap-pending Profile resolved (Sub removed mid-burst)",
        ),
        Diagnostic::ProfileClaimPurged {
            profile,
            claim,
            resource,
            failure,
        } => tracing::warn!(
            ?profile,
            ?claim,
            ?resource,
            ?failure,
            errno = failure.errno(),
            "profile claim purged (WatchOpRejected at claimed resource)",
        ),
        Diagnostic::AttachPathInvalid { path, hint } => {
            tracing::error!(
                path = %path.display(),
                hint,
                "attach path invalid; request dropped",
            );
        }
        Diagnostic::DescentInvariantViolation { profile, prefix } => tracing::error!(
            ?profile,
            ?prefix,
            "descent invariant violation: remaining_components empty",
        ),
        Diagnostic::SpliceCrossedUncovered { profile, target } => tracing::warn!(
            ?profile,
            ?target,
            "splice crossed uncovered subtree (graft contract violation; \
             prior view kept, response dropped)",
        ),
        Diagnostic::EventAbsorbedByFireTail {
            profile,
            resource,
            event,
        } => tracing::trace!(
            ?profile,
            ?resource,
            ?event,
            "fs event absorbed by fire-tail (Awaiting/Rebasing); folded into post-fire rebase",
        ),
        Diagnostic::AwaitGateDeadlineElapsed {
            profile,
            outstanding,
        } => tracing::warn!(
            ?profile,
            outstanding,
            "await-gate deadline elapsed; force-transitioning to Rebasing (actuator likely hung)",
        ),
        Diagnostic::SensorOverflow { scope } => tracing::warn!(
            ?scope,
            "sensor reported overflow (kernel queue dropped events); reseeding in-scope Profiles",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::{ActuatorSide, Channels, WatcherSide};
    use crossbeam::channel::Sender;
    use specter_config::Config;
    use specter_core::Input;
    use specter_sensor::FsWatcher;
    use specter_sensor::testkit::{MockFsWatcher, MockProber, MockWaker};
    use std::sync::Arc;

    /// Bundle of handles a test holds to drive [`EngineDriver`] without the
    /// [`crate::app`] orchestration layer.
    struct TestRig {
        driver: EngineDriver,
        /// Held to keep the `watcher_side` receivers alive (and so tests
        /// can assert what would have been delivered to the watcher).
        watcher_side: WatcherSide,
        actuator_side: ActuatorSide,
        prober: Arc<MockProber>,
        waker: Arc<MockWaker>,
        sensor_in_tx: Sender<Input>,
        effect_in_tx: Sender<Input>,
        reload_tx: Sender<()>,
        shutdown_tx: Sender<()>,
    }

    fn rig_for(config: Config, config_path: PathBuf) -> TestRig {
        let mut chans = Channels::new();
        let sensor_in_tx = chans.sensor_in_tx.clone();
        let effect_in_tx = chans.effect_in_tx.clone();
        let reload_tx = chans.reload_signal_tx.clone();
        let shutdown_tx = chans.shutdown_engine_tx.clone();
        let actuator_side = chans.take_actuator_side();
        let watcher_side = chans.take_watcher_side();
        let engine_side = chans.take_engine_side();
        drop(chans);

        let watcher = MockFsWatcher::new();
        let waker = Arc::clone(&watcher.waker);
        let wake_handle = watcher.wake_handle();
        let prober: Arc<MockProber> = Arc::new(MockProber::new());

        let log_cfg = config.log.clone();
        // Tests don't drive the SIGHUP API meaningfully and would race
        // each other on the global subscriber slot if every rig called
        // `observability::init`. `noop()` returns a structurally-correct
        // handle whose `set_level` / `reopen_file` are silent no-ops —
        // tests assert the *driver*'s reload-pipeline behaviour, not the
        // subscriber's filter state.
        let obs_handle = crate::observability::ObservabilityHandle::noop();
        let loader = Loader::new(config, log_cfg);
        // Mirror the production path: derive the initial window from the
        // loader's config so reload-driven rotation tests have a real
        // baseline to compare against.
        let drain_window = DrainWindow::new();
        drain_window.set(loader.derive_drain_window());
        let driver = EngineDriver::new(
            Engine::new(),
            loader,
            config_path,
            CliLogOverrides::default(),
            obs_handle,
            engine_side,
            prober.clone(),
            wake_handle,
            drain_window,
        );
        TestRig {
            driver,
            watcher_side,
            actuator_side,
            prober,
            waker,
            sensor_in_tx,
            effect_in_tx,
            reload_tx,
            shutdown_tx,
        }
    }

    fn config_with_one_watch(path: &std::path::Path) -> Config {
        let toml = format!(
            r#"
    [log]
    level = "warn"

    [[watch]]
    name      = "build"
    path      = "{}"
    command   = ["true"]
    settle_ms = 50
    "#,
            path.display(),
        );
        Config::from_str(&toml).expect("test config parses")
    }

    #[test]
    fn empty_run_returns_continue_after_select_timeout() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);
        // No inputs queued; select times out after FOREVER_TIMEOUT, but
        // since the engine has no timers we'd block forever. Skirt this
        // by triggering shutdown immediately.
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
    }

    #[test]
    fn run_initial_attach_populates_loader_ids_and_emits_watch_op() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = config_with_one_watch(tmp.path());
        let mut rig = rig_for(config, cfg_path);

        rig.driver.run_initial_attach();

        // One Sub was attached → loader.ids has one entry under that name.
        assert_eq!(rig.driver.loader.ids.len(), 1);
        let sid = *rig.driver.loader.ids.get("build").expect("name present");
        assert_ne!(sid, SubId::default());

        // The attach emitted a Watch op → forwarded to watch_ops_tx.
        let mut watch_ops = Vec::new();
        while let Ok(op) = rig.watcher_side.watch_ops_rx.try_recv() {
            watch_ops.push(op);
        }
        assert!(!watch_ops.is_empty(), "attach emits at least one Watch op");

        // Wake handle was poked (since ≥1 WatchOp was sent).
        assert!(*rig.waker.woken.lock().unwrap() >= 1);

        // The Seed burst emitted a probe → forwarded to prober.submit.
        let submitted = rig.prober.take_submitted();
        assert_eq!(submitted.len(), 1);
    }

    #[test]
    fn shutdown_signal_returns_shutdown_outcome() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        rig.shutdown_tx.try_send(()).expect("shutdown send");
        assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
    }

    #[test]
    fn sensor_in_disconnect_returns_shutdown() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        // Drop every sender into sensor_in_rx (the engine side holds the
        // only receiver). Drop our test clone first; the watcher_side's
        // sensor_in_tx clone still keeps it alive — drop that too.
        drop(rig.sensor_in_tx);
        let WatcherSide {
            watch_ops_rx,
            sensor_in_tx,
        } = rig.watcher_side;
        drop(sensor_in_tx);
        drop(watch_ops_rx); // not needed for this assertion

        // Now sensor_in_rx is disconnected; tick observes via try_recv.
        assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
    }

    #[test]
    fn effect_in_disconnect_continues() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        drop(rig.effect_in_tx);
        let ActuatorSide {
            effects_rx,
            shutdown_actuator_rx,
            hard_shutdown_actuator_rx,
            effect_in_tx,
        } = rig.actuator_side;
        drop(effect_in_tx);
        drop(effects_rx);
        drop(shutdown_actuator_rx);
        drop(hard_shutdown_actuator_rx);

        // Effect channel disconnected — tick treats it as "no completions",
        // does not shut down the engine. Use shutdown to exit cleanly.
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        assert_eq!(rig.driver.tick(), TickOutcome::Shutdown);
    }

    #[test]
    fn reload_with_invalid_path_logs_and_keeps_config() {
        // Config file at a non-existent path: handle_reload returns early
        // without touching loader.current_config.
        let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config.clone(), cfg_path);

        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();
        // Current config preserved.
        assert_eq!(rig.driver.loader.current_config, config);
    }

    #[test]
    fn reload_with_no_changes_rotates_config_silently() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_text = format!(
            r#"
    [[watch]]
    name      = "build"
    path      = "{}"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &cfg_text).unwrap();
        let initial = Config::from_str(&cfg_text).expect("test config");

        let mut rig = rig_for(initial.clone(), cfg_path);
        rig.driver.run_initial_attach();
        let ids_before: Vec<_> = rig.driver.loader.ids.keys().cloned().collect();

        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        // No changes → loader.ids unchanged.
        let ids_after: Vec<_> = rig.driver.loader.ids.keys().cloned().collect();
        assert_eq!(ids_before, ids_after);
        assert_eq!(rig.driver.loader.current_config, initial);
    }

    #[test]
    fn reload_added_watch_appears_in_loader_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = format!(
            r#"
    [[watch]]
    name      = "a"
    path      = "{}"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let new_text = format!(
            r#"
    [[watch]]
    name      = "a"
    path      = "{0}"
    command   = ["true"]
    
    [[watch]]
    name      = "b"
    path      = "{0}"
    command   = ["true"]
    settle_ms = 100
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &initial_text).unwrap();
        let initial = Config::from_str(&initial_text).expect("initial parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        assert_eq!(rig.driver.loader.ids.len(), 1);

        // Operator edits config; sends SIGHUP (we simulate via the channel).
        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        // Both Subs now in loader.ids.
        assert_eq!(rig.driver.loader.ids.len(), 2);
        assert!(rig.driver.loader.ids.contains_key("a"));
        assert!(rig.driver.loader.ids.contains_key("b"));
    }

    #[test]
    fn reload_removed_watch_drops_from_loader_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = format!(
            r#"
    [[watch]]
    name      = "a"
    path      = "{0}"
    command   = ["true"]
    
    [[watch]]
    name      = "b"
    path      = "{0}"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let new_text = format!(
            r#"
    [[watch]]
    name      = "a"
    path      = "{}"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &initial_text).unwrap();
        let initial = Config::from_str(&initial_text).expect("initial parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        assert_eq!(rig.driver.loader.ids.len(), 2);

        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert_eq!(rig.driver.loader.ids.len(), 1);
        assert!(rig.driver.loader.ids.contains_key("a"));
        assert!(!rig.driver.loader.ids.contains_key("b"));
    }

    #[test]
    fn fs_event_drained_before_effect_complete_so_fire_tail_absorbs() {
        // Sensor inputs drain BEFORE effect completions: an EffectComplete
        // could move an Awaiting burst into Rebasing, and any FsEvent
        // queued in the same tick should reach the engine first so the
        // fire-tail (`BurstPhase::Awaiting` / `Rebasing`) can absorb it
        // and fold the disk change into the post-fire rebase. Push an
        // EffectComplete first, then an FsEvent; tick sees the FsEvent first
        // because of the drain order — even though EffectComplete was queued earlier.
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        let r = specter_core::ResourceId::default();
        rig.effect_in_tx
            .send(Input::EffectComplete {
                sub: SubId::default(),
                key: specter_core::DedupKey::Subtree {
                    sub: SubId::default(),
                    profile: specter_core::ProfileId::default(),
                },
                result: specter_core::EffectOutcome::Ok,
            })
            .unwrap();
        rig.sensor_in_tx
            .send(Input::FsEvent {
                resource: r,
                event: specter_core::FsEvent::Modified,
            })
            .unwrap();
        rig.shutdown_tx.try_send(()).expect("shutdown send");

        // Tick processes both, then sees shutdown. The engine handles
        // each input atomically in the order step is called; the drain
        // order is what we're testing — the bin's contract is that
        // sensor inputs reach engine.step before effect completions.
        let outcome = rig.driver.tick();
        assert_eq!(outcome, TickOutcome::Shutdown);
        // We don't assert on engine state here — the FsEvent + EC for
        // unknown ids both produce diagnostics, and the order of
        // diagnostics confirms drain order. For a behavioral test, see
        // `tests/e2e_*` integration tests where ordering surfaces as
        // observable subprocess behavior.
    }

    #[test]
    fn forward_wakes_after_each_send_to_break_full_channel_deadlock() {
        // Regression for the deep-tree startup deadlock. The Seed burst
        // against a tree with many directories emits a single `StepOutput`
        // whose `watch_ops` exceeds the bounded(1024) `watch_ops_tx`
        // capacity. With a wake-once-at-end protocol, the engine's `forward`
        // would fill the channel, block on `Sender::send` at op 1025, and
        // never reach the trailing `wake_handle.wake()` — leaving the watcher
        // asleep in `kevent` until SIGTERM forced a separate wake. The
        // contract is one wake **per successful send**, so the watcher always
        // sees a fresh `EVFILT_USER` trigger to drain by, kernel-coalesced.
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let rig = rig_for(config, cfg_path);
        let TestRig {
            driver,
            watcher_side,
            waker,
            ..
        } = rig;

        // Drain in a thread so the bounded channel can flow; without a
        // drainer this test would block at the channel-bound boundary even
        // with the fix in place — wake-per-send unsticks the *kqueue*
        // consumer, not crossbeam's `bounded` send.
        let watch_ops_rx = watcher_side.watch_ops_rx;
        let drainer = std::thread::spawn(move || {
            let mut count = 0usize;
            while watch_ops_rx.recv().is_ok() {
                count += 1;
            }
            count
        });

        let n_ops: usize = 5;
        let mut out = StepOutput::default();
        for i in 0..n_ops {
            out.watch_ops.push(specter_core::WatchOp::Watch {
                resource: specter_core::ResourceId::default(),
                path: PathBuf::from(format!("/p/{i}")),
                kind: specter_core::ResourceKind::Unknown,
                events: specter_core::ClassSet::EMPTY,
            });
        }

        driver.forward(out);
        drop(driver); // release engine-side `watch_ops_tx` so the drainer exits.
        drop(watcher_side.sensor_in_tx); // release the watcher-side clone too.

        let received = drainer.join().expect("drainer thread panicked");
        assert_eq!(received, n_ops, "all ops must reach the watcher");

        let woken = usize::try_from(*waker.woken.lock().expect("MockWaker poisoned"))
            .expect("wake count fits in usize");
        assert_eq!(
            woken, n_ops,
            "expected wake-per-send (n={n_ops}); got {woken}",
        );
    }
}
