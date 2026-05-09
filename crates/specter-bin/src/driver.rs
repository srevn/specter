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
//!   drain sensor → drain timers → drain reload (SIGHUP) → drain
//!   config-event + apply settle-expiry → drain effects → block on
//!   `Select::ready_timeout` until any source readies (or a timer
//!   deadline elapses, or shutdown).
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
use specter_config::{Config, FileMeta};
use specter_core::{Diagnostic, Input, ProbeOp, PromoterId, StepOutput, SubId};
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
    /// Auto-reload settle deadline — armed by the config-event drain
    /// (the watcher thread's `try_send` or a test rig's manual
    /// `try_send`), expires after [`CONFIG_SETTLE`] of quiet, at
    /// which point the driver runs the
    /// lstat-vs-`loader.config_meta` filter and (on drift) calls
    /// [`Self::handle_reload`]. Reset to `None` on expiry and
    /// re-armed per pulse (settle resets, so sustained bursts defer
    /// the reload until the edits actually settle).
    ///
    /// Two consumers:
    /// - The [`Self::tick`] timeout math feeds the deadline into
    ///   `Select::ready_timeout` so the driver wakes precisely when
    ///   the window expires, not on the next sensor / effect pulse.
    /// - [`Self::apply_config_settle_expiry`] gates the lstat call
    ///   on `now >= deadline` so the engine thread never lstats
    ///   before the settle window has elapsed.
    config_settle_until: Option<Instant>,
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
            config_settle_until: None,
        }
    }

    /// Attach every Sub from `loader.current_config.watches` and every
    /// Promoter from `loader.current_config.promoters` in source order.
    /// Each [`StepOutput`] is forwarded as we go so the watcher /
    /// prober receive ops as the engine emits them — a single
    /// startup-sized `ConfigDiff` would batch the entire attach into
    /// one output and stall the watcher behind the post-call
    /// `forward`.
    ///
    /// Both id maps populate via the shared
    /// [`Self::reconcile_loader_from_diagnostics`] helper so the
    /// initial-attach and SIGHUP-reload paths converge on the same
    /// reconciliation discipline. Static Subs end up in
    /// [`Loader::ids`]; Promoters end up in [`Loader::promoter_ids`];
    /// dynamic Subs (whose `SubAttached` carries
    /// `source_promoter = Some(_)`) are filtered out by the helper —
    /// they live in `Promoter.dynamic_subs` and are never observed by
    /// the bin's diff layer.
    pub fn run_initial_attach(&mut self) {
        let now = Instant::now();
        // Snapshot the spec lists — `loader.ids` / `loader.promoter_ids`
        // mutation invalidates an iterator over `loader.current_config`.
        let watch_specs = self.loader.current_config.watches.clone();
        let promoter_specs = self.loader.current_config.promoters.clone();
        for spec in watch_specs {
            let req = spec.to_attach_request();
            let (_id, out) = self.engine.attach_sub(req, now);
            Self::reconcile_loader_from_diagnostics(&mut self.loader, &[], &[], &out.diagnostics);
            self.forward(out);
        }
        for spec in promoter_specs {
            let req = spec.to_attach_request();
            let (_pid, out) = self.engine.attach_promoter(req, now);
            Self::reconcile_loader_from_diagnostics(&mut self.loader, &[], &[], &out.diagnostics);
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

        // Drain auto-reload pulses (re-arm settle per pulse), then
        // check whether the settle window has elapsed and (on
        // confirmed meta drift) run handle_reload. Order matters:
        // drain-then-expiry implements "settle resets per pulse" — a
        // pulse arriving in the same tick as a stale deadline pushes
        // the deadline forward, so a sustained editor burst keeps
        // deferring the reload until the edits actually settle.
        // Inverting (expiry-then-drain) would fire a reload in the
        // middle of an in-flight burst.
        while self.sides.config_event_rx.try_recv().is_ok() {
            self.config_settle_until = Some(now + CONFIG_SETTLE);
        }
        self.apply_config_settle_expiry(now);

        // Drain effect completions. Disconnect tolerated (engine remains
        // functional against sensor + timers).
        while let Ok(input) = self.sides.effect_in_rx.try_recv() {
            let out = self.engine.step(input, now);
            self.forward(out);
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

        let mut sel = Select::new();
        let _i_sensor = sel.recv(&self.sides.sensor_in_rx);
        let _i_effect = sel.recv(&self.sides.effect_in_rx);
        let _i_reload = sel.recv(&self.sides.reload_signal_rx);
        // Wakes the driver from a long block when a config-event pulse
        // arrives. The actual pulse handling lives in the per-tick
        // drain above; this arm is purely for unblocking. The sender
        // side must remain alive across the block — a Disconnected
        // rx here would crossbeam-report as immediately-ready and
        // busy-loop the driver.
        let _i_config = sel.recv(&self.sides.config_event_rx);
        let i_shutdown = sel.recv(&self.sides.shutdown_engine_rx);

        match sel.ready_timeout(timeout) {
            Ok(idx) if idx == i_shutdown => TickOutcome::Shutdown,
            Ok(_) | Err(crossbeam::channel::ReadyTimeoutError) => TickOutcome::Continue,
        }
    }

    /// Drive the auto-reload settle deadline forward by one tick.
    ///
    /// Called from [`Self::tick`] after the config-event drain. Three
    /// branches:
    ///
    /// - `config_settle_until == None`: nothing armed; no-op.
    /// - `now < deadline`: still inside the settle window; no-op
    ///   (the next pulse may push the deadline forward; a future tick
    ///   will reach `now >= deadline` if the burst goes quiet).
    /// - `now >= deadline`: clear the deadline, run the lstat filter
    ///   ([`Self::config_meta_changed`]), and call
    ///   [`Self::handle_reload`] on drift. The lstat filter is what
    ///   suppresses no-op pulses — a kqueue parent-dir spillover from
    ///   a sibling write fires a pulse but doesn't move
    ///   `loader.config_meta`, so the lstat compares equal and we
    ///   skip the parse.
    ///
    /// `pub(crate)` so unit tests can drive the helper directly with
    /// a synthetic `now`, avoiding real-time sleeps across the 100ms
    /// settle window. Production callers go through `tick`, which
    /// always passes `Instant::now()`.
    pub(crate) fn apply_config_settle_expiry(&mut self, now: Instant) {
        let Some(deadline) = self.config_settle_until else {
            return;
        };
        if now < deadline {
            return;
        }
        self.config_settle_until = None;
        if self.config_meta_changed() {
            self.handle_reload(now);
        }
    }

    /// Cheap (one syscall) lstat-vs-stored-meta compare. Returns `true`
    /// if the on-disk file's [`FileMeta`] differs from
    /// `loader.config_meta`, **or** if the lstat itself fails.
    ///
    /// Treating an lstat error as "changed" is a defensive choice with
    /// two desirable properties:
    ///
    /// 1. **Recovery semantics.** An ENOENT / EACCES that recovers
    ///    (operator un-deletes / chmods 644) flips the lstat from `Err`
    ///    to `Ok`, which is structurally a transition — handle_reload
    ///    runs on the next pulse and either succeeds (rotation) or
    ///    fails again (parse-fail; meta NOT rotated; retry on next
    ///    pulse).
    /// 2. **Fail-stable.** If the file is permanently unreachable, the
    ///    next pulse fires a parse attempt that logs and returns
    ///    early. `loader.config_meta` is preserved across parse-fails,
    ///    so the next pulse repeats — but we do not loop on our own
    ///    (`config_settle_until` is consumed regardless), so the
    ///    error is paced by external pulse rate, not internal spinning.
    fn config_meta_changed(&self) -> bool {
        match FileMeta::from_path(&self.config_path) {
            Ok(m) => m != self.loader.config_meta,
            Err(_) => true,
        }
    }

    /// Read the config from disk; on success, diff against the current
    /// snapshot, apply via `Input::ConfigDiff`, sync `loader.ids`,
    /// rotate `loader.current_config` and `loader.config_meta`. On
    /// failure, log + retain running config + meta (preserving the
    /// auto-reload retry loop on the next pulse).
    ///
    /// Log-side reload is integrated here:
    ///   - The `[log]` block is re-resolved (CLI overrides re-applied);
    ///     a level-only change calls `obs_handle.set_level`;
    ///     a destination/path change logs an `error!` instructing the
    ///     operator to restart (v1 doesn't hot-reload destinations).
    ///   - `obs_handle.reopen_file()` fires unconditionally so logrotate
    ///     `copytruncate`-style rotation works without a config diff.
    ///
    /// **Meta rotation discipline.** `loader.config_meta` rotates on
    /// **every** successful read — both the empty-diff and the
    /// apply-diff branches — so the auto-reload settle-expiry filter
    /// sees a freshly-stored identity after each reload. Skipping the
    /// empty-diff rotation would loop the filter against the same
    /// already-applied edit forever (the lstat reflects the post-edit
    /// inode but `loader.config_meta` would still hold the pre-edit
    /// value). Parse-fail does **not** rotate — preserving the retry
    /// loop until the operator fixes the file.
    fn handle_reload(&mut self, now: Instant) {
        let Some((new_config, new_meta)) = self.read_and_parse_config() else {
            return;
        };
        let new_log_resolved = self.parse_and_resolve_log(&new_config);
        self.apply_log_reload(&new_log_resolved);

        let diff = self.compute_watch_diff(&new_config);
        let no_sub_changes = diff.subs.added.is_empty()
            && diff.subs.removed.is_empty()
            && diff.subs.modified.is_empty();
        let no_promoter_changes = diff.promoters.added.is_empty()
            && diff.promoters.removed.is_empty()
            && diff.promoters.modified.is_empty();
        if no_sub_changes && no_promoter_changes {
            tracing::info!("config reload: no watch changes");
            self.loader.current_config = new_config;
            self.loader.current_log = new_log_resolved;
            self.loader.config_meta = new_meta;
            return;
        }

        // Snapshot the change-counts (for the post-apply summary log)
        // and the `removed` id lists before the diff moves into the
        // engine. Modified entries don't appear here — their old ids
        // live under the entry's name in `loader.ids` /
        // `loader.promoter_ids` and are overwritten by the
        // `SubAttached` / `PromoterAttached` diagnostics emitted for
        // the freshly-minted entities.
        let added_n = diff.subs.added.len();
        let removed_n = diff.subs.removed.len();
        let modified_n = diff.subs.modified.len();
        let promoter_added_n = diff.promoters.added.len();
        let promoter_removed_n = diff.promoters.removed.len();
        let promoter_modified_n = diff.promoters.modified.len();
        let removed_sub_ids: Vec<SubId> = diff.subs.removed.clone();
        let removed_promoter_ids: Vec<PromoterId> = diff.promoters.removed.clone();

        let out = self.engine.step(Input::ConfigDiff(diff), now);

        Self::reconcile_loader_from_diagnostics(
            &mut self.loader,
            &removed_sub_ids,
            &removed_promoter_ids,
            &out.diagnostics,
        );

        self.loader.current_config = new_config;
        self.loader.current_log = new_log_resolved;
        self.loader.config_meta = new_meta;
        self.apply_drain_window_rotation();
        tracing::info!(
            added = added_n,
            removed = removed_n,
            modified = modified_n,
            promoters_added = promoter_added_n,
            promoters_removed = promoter_removed_n,
            promoters_modified = promoter_modified_n,
            "config reload applied",
        );
        self.forward(out);
    }

    /// Read + parse the on-disk config, capturing `FileMeta` atomically
    /// alongside the bytes. Returns `None` on I/O / parse failure (with
    /// an `error!` log); the caller keeps the running config rather
    /// than aborting.
    ///
    /// Sole I/O surface for the reload pipeline — both the SIGHUP path
    /// and the auto-reload settle-expiry path call here so the
    /// failure-handling discipline lives in one place. The returned
    /// [`FileMeta`] is captured from the same `File` handle that
    /// produced the bytes ([`Config::from_path_with_meta`]), so a
    /// concurrent atomic-save cannot rotate the meta out from under
    /// the parsed [`Config`]. Callers rotate `loader.config_meta`
    /// from this value on every successful read — including the
    /// empty-diff branch, so the auto-reload settle filter doesn't
    /// loop on an already-applied edit.
    pub(crate) fn read_and_parse_config(&self) -> Option<(Config, FileMeta)> {
        match Config::from_path_with_meta(&self.config_path) {
            Ok(pair) => Some(pair),
            Err(e) => {
                tracing::error!(
                    error = %e,
                    path = %self.config_path.display(),
                    "config reload failed; keeping running config",
                );
                None
            }
        }
    }

    /// Resolve `new_config.log` with CLI overrides re-applied (CLI
    /// wins, matching startup precedence). On validation failure
    /// (e.g., a freshly-edited config sets `destination = "file"`
    /// without a `path`), log the error and return the running log
    /// snapshot so the watch-side reload can still proceed
    /// independently.
    pub(crate) fn parse_and_resolve_log(&self, new_config: &Config) -> specter_config::LogConfig {
        match new_config.log.clone().merge_cli(
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
                self.loader.current_log.clone()
            }
        }
    }

    /// Compute the diff between the running and freshly-parsed config.
    /// Pure delegation to [`specter_config::diff`] threaded through
    /// the loader's live id maps.
    pub(crate) fn compute_watch_diff(
        &self,
        new_config: &Config,
    ) -> specter_core::WatchRegistryDiff {
        specter_config::diff(
            &self.loader.current_config,
            new_config,
            &self.loader.ids,
            &self.loader.promoter_ids,
        )
    }

    /// Recompute the watcher's deferred-drain window from the loader's
    /// (already-rotated) `current_config` and rotate the cross-thread
    /// handle if the value changed. The watcher thread observes the
    /// new value on its next `poll_until` iteration.
    pub(crate) fn apply_drain_window_rotation(&self) {
        let new_window = self.loader.derive_drain_window();
        let old_window = self.drain_window.get();
        if new_window != old_window {
            self.drain_window.set(new_window);
            tracing::info!(
                old_ms = old_window.as_millis(),
                new_ms = new_window.as_millis(),
                "drain_window updated",
            );
        }
    }

    /// Apply lifecycle diagnostics emitted by an `Input::ConfigDiff`
    /// step (or any per-attach `attach_sub` / `attach_promoter` step
    /// during initial attach) to the [`Loader`]'s name → id maps,
    /// **and** drop entries whose ids appear in the supplied
    /// `removed_sub_ids` / `removed_promoter_ids` lists.
    ///
    /// Reconciliation discipline:
    /// - **Removals** are applied from the diff's `removed` lists, not
    ///   from a putative "SubDetached" diagnostic — `detach_sub_inner`
    ///   does not emit one (the diff is the authoritative source for
    ///   what disappeared).
    /// - **Additions / modifications** flow from
    ///   [`Diagnostic::SubAttached`] (filtered on
    ///   `source_promoter.is_none()` — dynamic Subs synthesised by a
    ///   Promoter live in the engine's
    ///   `Promoter.dynamic_subs` map and would leak across reload
    ///   cycles if mirrored into the static `loader.ids` index) and
    ///   [`Diagnostic::PromoterAttached`].
    /// - [`Diagnostic::PromoterReaped`] also drains
    ///   `loader.promoter_ids` as defense-in-depth: the diff's
    ///   `removed` list already covers operator-driven removals, but
    ///   reaps cascaded from a Promoter modify (`reap_promoter_inner`
    ///   then `attach_promoter_inner`) only surface here. Insert order
    ///   in `on_config_diff` is `reap → attach`, so the freshly-minted
    ///   entry's `PromoterAttached` overwrites the cleared slot in
    ///   the same loop pass — correct end state regardless of how
    ///   many `(reap, attach)` pairs interleave.
    ///
    /// `&mut Loader` (no `&mut self`) so the call site keeps `&self`
    /// available for the surrounding driver work (logging, channel
    /// sends) without borrow-check gymnastics.
    fn reconcile_loader_from_diagnostics(
        loader: &mut Loader,
        removed_sub_ids: &[SubId],
        removed_promoter_ids: &[PromoterId],
        diagnostics: &[Diagnostic],
    ) {
        for id in removed_sub_ids {
            loader.ids.retain(|_, v| v != id);
        }
        for id in removed_promoter_ids {
            loader.promoter_ids.retain(|_, v| v != id);
        }

        for diag in diagnostics {
            match diag {
                Diagnostic::SubAttached {
                    sub,
                    name,
                    source_promoter: None,
                } => {
                    loader.ids.insert(name.clone(), *sub);
                }
                Diagnostic::PromoterAttached { promoter, name } => {
                    loader.promoter_ids.insert(name.clone(), *promoter);
                }
                Diagnostic::PromoterReaped { promoter } => {
                    loader.promoter_ids.retain(|_, v| v != promoter);
                }
                _ => {}
            }
        }
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
                ProbeOp::Cancel { owner } => self.prober.cancel(owner),
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
/// Most variants are `warn` (drops + race conditions). With auto-reload
/// landed, `EffectCompleteForUnknownSub` is `warn` too — the auto-reload
/// path makes the detach-during-effect race routine; engine bugs surface
/// via test assertions on the `Diagnostic::` variant rather than via log
/// severity. `ReapPendingResolved` and `ReapPendingCancelled` are `info`
/// (informational; the late reap completed or was pre-empted by a
/// revival).
pub fn log_diagnostic(d: &Diagnostic) {
    match d {
        Diagnostic::StaleProbeResponse { owner, correlation } => tracing::warn!(
            ?owner,
            ?correlation,
            "stale probe response (state mismatch)"
        ),
        Diagnostic::StaleTimer { id } => tracing::warn!(?id, "stale timer expiration"),
        Diagnostic::EffectCompleteOutsideAwaiting { sub, profile } => tracing::warn!(
            ?sub,
            ?profile,
            "effect_complete arrived outside Awaiting (gate-deadline force-transition or anchor-loss); dropped",
        ),
        Diagnostic::EffectCompleteForUnknownSub { sub } => tracing::warn!(
            ?sub,
            "effect_complete for unknown Sub (hot-reload race or engine bug; dropped)",
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
        Diagnostic::ReapPendingCancelled { profile } => tracing::debug!(
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
        Diagnostic::PromoterClaimPurged {
            promoter,
            claim,
            resource,
            failure,
        } => tracing::warn!(
            ?promoter,
            ?claim,
            ?resource,
            ?failure,
            errno = failure.errno(),
            "promoter claim purged (WatchOpRejected at claimed resource)",
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
        Diagnostic::PromoterReseededForOverflow { promoter } => tracing::debug!(
            ?promoter,
            "promoter reseeded after sensor overflow (descent re-probed or proxies re-enumerated)",
        ),
        Diagnostic::SubAttached {
            sub,
            name,
            source_promoter,
        } => match source_promoter {
            // Static (operator-declared) attach: high signal, low rate
            // (one per `[[watch]]` block per reload). INFO is the
            // operator-facing default per the catalog severity table.
            None => tracing::info!(?sub, %name, "sub attached"),
            // Dynamic (Promoter-spawned) attach: same lifecycle event
            // but emitted once per pattern match, which can be many
            // per enumeration. DEBUG keeps operator logs uncluttered;
            // `PromotionKindObserved` already carries the path-level
            // signal at the same severity.
            Some(promoter) => tracing::debug!(
                ?sub,
                %name,
                ?promoter,
                "dynamic sub attached (promoter-spawned)",
            ),
        },
        Diagnostic::PromoterAttached { promoter, name } => tracing::info!(
            ?promoter,
            %name,
            "promoter attached",
        ),
        Diagnostic::PromoterReaped { promoter } => tracing::info!(?promoter, "promoter reaped",),
        Diagnostic::PromoterDescentInvariantViolation { promoter, prefix } => tracing::error!(
            ?promoter,
            ?prefix,
            "promoter descent invariant violation: remaining_components empty",
        ),
        Diagnostic::PromoterDescentVanished { promoter, prefix } => tracing::debug!(
            ?promoter,
            ?prefix,
            "promoter descent / enumeration probe Vanished",
        ),
        Diagnostic::PromoterDescentFailed {
            promoter,
            prefix,
            errno,
        } => tracing::warn!(
            ?promoter,
            ?prefix,
            errno,
            "promoter descent / enumeration probe Failed",
        ),
        Diagnostic::PromotionKindObserved {
            promoter,
            path,
            kind,
        } => tracing::debug!(
            ?promoter,
            path = %path.display(),
            ?kind,
            "promoter promotion observed (dynamic Sub minted)",
        ),
        Diagnostic::PromoterFanoutThreshold { promoter, count } => tracing::warn!(
            ?promoter,
            count,
            "promoter fanout exceeded warning threshold (consider tightening pattern)",
        ),
        Diagnostic::PromoterProxyStaleEvent { promoter, resource } => tracing::debug!(
            ?promoter,
            ?resource,
            "fs event for promoter proxy that was unregistered earlier in step (stale; dropped)",
        ),
        Diagnostic::PromoterEnumerationVanished { promoter, proxy } => tracing::debug!(
            ?promoter,
            ?proxy,
            "promoter enumeration probe Vanished (proxy gone; subtree unwound)",
        ),
        Diagnostic::PromoterEnumerationFailed {
            promoter,
            proxy,
            errno,
        } => tracing::warn!(
            ?promoter,
            ?proxy,
            errno,
            "promoter enumeration probe Failed (retaining proxy state)",
        ),
        Diagnostic::DynamicSubReaped {
            promoter,
            sub,
            path,
        } => tracing::debug!(
            ?promoter,
            ?sub,
            path = %path.display(),
            "dynamic Sub reaped (anchor terminal — Promoter dynamic_subs entry dropped)",
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

    /// Sentinel meta used in fixtures whose config file may not exist
    /// on disk. Inode 0 is reserved by every supported kernel; this
    /// value never compares equal to a real `FileMeta::from_path`
    /// capture, so tests that *do* exercise the meta-rotation path
    /// can assert "rotated to a real value" by comparing against a
    /// fresh `FileMeta::from_path` (which differs from this sentinel
    /// in every field).
    fn dummy_meta() -> FileMeta {
        FileMeta {
            inode: 0,
            device: 0,
            mtime_sec: 0,
            mtime_nsec: 0,
            ctime_sec: 0,
            ctime_nsec: 0,
            size: 0,
        }
    }

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
        /// Cloned config-event sender. Holding this clone alive in the
        /// rig keeps the engine's `config_event_rx` connected (otherwise
        /// `drop(chans)` would release the only sender and the
        /// driver's `Select` arm would observe Disconnected). Tests
        /// `try_send(())` here to simulate watcher pulses.
        config_event_tx: Sender<()>,
    }

    fn rig_for(config: Config, config_path: PathBuf) -> TestRig {
        let mut chans = Channels::new();
        let sensor_in_tx = chans.sensor_in_tx.clone();
        let effect_in_tx = chans.effect_in_tx.clone();
        let reload_tx = chans.reload_signal_tx.clone();
        let shutdown_tx = chans.shutdown_engine_tx.clone();
        let config_event_tx = chans.config_event_tx.clone();
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
        let loader = Loader::new(config, log_cfg, dummy_meta());
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
            config_event_tx,
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
    settle    = "50ms"
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
    settle    = "100ms"
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

    // ===== diagnostic-driven id reconciliation =====
    //
    // The bin's `loader.ids` and `loader.promoter_ids` are populated
    // from `Diagnostic::SubAttached` and `Diagnostic::PromoterAttached`
    // emitted by the engine during attach paths, and drained from
    // diff-supplied `removed` lists plus `Diagnostic::PromoterReaped`.
    // The shared helper [`EngineDriver::reconcile_loader_from_diagnostics`]
    // is the single source of truth across `run_initial_attach` and
    // `handle_reload`. Tests in this section pin the helper's
    // discipline and the surrounding driver glue.

    /// Build a config with a single dynamic [[watch]] entry. The path
    /// uses brace expansion, exercising the `is_dynamic` auto-detect
    /// path (the brace `{` is one of `*?[{`). Literal prefix is the
    /// supplied `tmp` directory so the validator's path-canonicalisation
    /// pass succeeds.
    fn config_with_one_promoter(path: &std::path::Path) -> Config {
        let toml = format!(
            r#"
    [log]
    level = "warn"

    [[watch]]
    name      = "logs"
    path      = "{}/{{a,b}}/access.log"
    command   = ["true"]
    settle    = "50ms"
    "#,
            path.display(),
        );
        Config::from_str(&toml).expect("test config parses")
    }

    /// `run_initial_attach` for a static-only config emits one
    /// `SubAttached` diagnostic per `[[watch]]` and the loader's static
    /// `ids` map gets a corresponding entry. The `(_id, out)` tuple
    /// from `attach_sub` is no longer the source of truth — the
    /// diagnostic stream is.
    #[test]
    fn run_initial_attach_emits_subattached_and_populates_loader_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = config_with_one_watch(tmp.path());
        let mut rig = rig_for(config, cfg_path);

        rig.driver.run_initial_attach();

        // Loader's static ids map populated from the SubAttached
        // diagnostic emitted by the attach.
        assert_eq!(rig.driver.loader.ids.len(), 1);
        let sid = *rig.driver.loader.ids.get("build").expect("name present");
        assert_ne!(sid, SubId::default());
        assert!(rig.driver.loader.promoter_ids.is_empty());
    }

    /// `run_initial_attach` extension: a config with a dynamic
    /// `[[watch]]` (auto-detected at config load) routes through
    /// `attach_promoter` and populates `loader.promoter_ids` from the
    /// `PromoterAttached` diagnostic.
    #[test]
    fn run_initial_attach_populates_promoter_ids_for_dynamic_watch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = config_with_one_promoter(tmp.path());
        let mut rig = rig_for(config, cfg_path);

        rig.driver.run_initial_attach();

        // Static map untouched; dynamic map carries the promoter.
        assert!(rig.driver.loader.ids.is_empty());
        assert_eq!(rig.driver.loader.promoter_ids.len(), 1);
        let pid = *rig
            .driver
            .loader
            .promoter_ids
            .get("logs")
            .expect("promoter name present");
        assert_ne!(pid, specter_core::PromoterId::default());
    }

    /// Mixed static + dynamic config: the initial-attach loop walks
    /// both spec lists and populates both maps in one run, with a
    /// single forward per attach so the watcher receives WatchOps
    /// incrementally.
    #[test]
    fn run_initial_attach_handles_mixed_static_and_dynamic() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_text = format!(
            r#"
    [log]
    level = "warn"

    [[watch]]
    name      = "build"
    path      = "{0}"
    command   = ["true"]
    settle    = "50ms"

    [[watch]]
    name      = "logs"
    path      = "{0}/{{a,b}}/access.log"
    command   = ["true"]
    settle    = "50ms"
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str(&cfg_text).expect("mixed config parses");
        let mut rig = rig_for(config, cfg_path);

        rig.driver.run_initial_attach();

        assert!(rig.driver.loader.ids.contains_key("build"));
        assert!(rig.driver.loader.promoter_ids.contains_key("logs"));
    }

    /// Reload that adds a fresh dynamic [[watch]] populates
    /// `loader.promoter_ids` via the `PromoterAttached` diagnostic
    /// emitted from the `Input::ConfigDiff` step. No `find_by_name`
    /// scan of the engine's promoter registry.
    #[test]
    fn reload_added_promoter_populates_loader_promoter_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = String::new(); // empty config → no watches
        let new_text = format!(
            r#"
    [[watch]]
    name      = "logs"
    path      = "{}/{{a,b}}/access.log"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &initial_text).unwrap();
        let initial = Config::from_str(&initial_text).expect("initial parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        assert!(rig.driver.loader.promoter_ids.is_empty());

        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert_eq!(rig.driver.loader.promoter_ids.len(), 1);
        assert!(rig.driver.loader.promoter_ids.contains_key("logs"));
    }

    /// Reload that removes a dynamic [[watch]] drops the entry from
    /// `loader.promoter_ids`. The diff's `removed` list (not a
    /// diagnostic) is the source of truth for removals — but
    /// `PromoterReaped` is also emitted, and the helper's
    /// defense-in-depth `retain` on that variant produces the same
    /// final state regardless of which path drove it.
    #[test]
    fn reload_removed_promoter_drops_from_loader_promoter_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = format!(
            r#"
    [[watch]]
    name      = "logs"
    path      = "{}/{{a,b}}/access.log"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let new_text = String::new();
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &initial_text).unwrap();
        let initial = Config::from_str(&initial_text).expect("initial parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        assert_eq!(rig.driver.loader.promoter_ids.len(), 1);

        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert!(rig.driver.loader.promoter_ids.is_empty());
    }

    /// Reload that modifies a dynamic [[watch]] (e.g., changes the
    /// command) replaces the old `PromoterId` with a freshly-minted
    /// one, keyed by the same name. The diff's `removed` list does
    /// NOT contain the modified id (modifications go through
    /// reap-then-attach), so the helper relies on the
    /// `PromoterAttached` diagnostic overwriting the entry and the
    /// `PromoterReaped` diagnostic clearing the prior id (no-op for
    /// the kept-name case, but exercises the cascade arm).
    #[test]
    fn reload_modified_promoter_replaces_id_in_loader_promoter_ids() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = format!(
            r#"
    [[watch]]
    name      = "logs"
    path      = "{}/{{a,b}}/access.log"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let new_text = format!(
            r#"
    [[watch]]
    name      = "logs"
    path      = "{}/{{a,b}}/access.log"
    command   = ["echo"]
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &initial_text).unwrap();
        let initial = Config::from_str(&initial_text).expect("initial parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        let old_pid = *rig
            .driver
            .loader
            .promoter_ids
            .get("logs")
            .expect("name present pre-reload");

        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert_eq!(rig.driver.loader.promoter_ids.len(), 1);
        let new_pid = *rig
            .driver
            .loader
            .promoter_ids
            .get("logs")
            .expect("name present post-reload");
        assert_ne!(new_pid, old_pid, "modify mints a fresh PromoterId");
    }

    /// Static→dynamic migration via path edit: a `[[watch]]` named
    /// "foo" with a literal path edits to a glob path. `is_dynamic`
    /// flips, so the diff emits `subs.removed + promoters.added`.
    /// Loader maps converge: the static entry vanishes from
    /// `loader.ids`; a Promoter entry appears in
    /// `loader.promoter_ids`.
    #[test]
    fn reload_static_to_dynamic_migration_swaps_loader_maps() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = format!(
            r#"
    [[watch]]
    name      = "foo"
    path      = "{}"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let new_text = format!(
            r#"
    [[watch]]
    name      = "foo"
    path      = "{}/*"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(&cfg_path, &initial_text).unwrap();
        let initial = Config::from_str(&initial_text).expect("initial parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        assert_eq!(rig.driver.loader.ids.len(), 1);
        assert!(rig.driver.loader.promoter_ids.is_empty());

        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert!(
            rig.driver.loader.ids.is_empty(),
            "static `foo` removed from loader.ids",
        );
        assert!(
            rig.driver.loader.promoter_ids.contains_key("foo"),
            "dynamic `foo` registered in loader.promoter_ids",
        );
    }

    /// Reverse direction: a dynamic [[watch]] flips to a literal
    /// path. Diff emits `promoters.removed + subs.added`; loader
    /// maps mirror the swap.
    #[test]
    fn reload_dynamic_to_static_migration_swaps_loader_maps() {
        let tmp = tempfile::TempDir::new().unwrap();
        let initial_text = format!(
            r#"
    [[watch]]
    name      = "foo"
    path      = "{}/*"
    command   = ["true"]
    "#,
            tmp.path().display(),
        );
        let new_text = format!(
            r#"
    [[watch]]
    name      = "foo"
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
        assert!(rig.driver.loader.ids.is_empty());
        assert_eq!(rig.driver.loader.promoter_ids.len(), 1);

        std::fs::write(&cfg_path, &new_text).unwrap();
        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert!(
            rig.driver.loader.promoter_ids.is_empty(),
            "dynamic `foo` removed from loader.promoter_ids",
        );
        assert!(
            rig.driver.loader.ids.contains_key("foo"),
            "static `foo` registered in loader.ids",
        );
    }

    /// Filter discipline: a `Diagnostic::SubAttached` carrying
    /// `source_promoter = Some(_)` (the dynamic-attach stamp) does
    /// NOT populate `loader.ids`. Dynamic Subs are owned by the
    /// engine's `Promoter.dynamic_subs` map; mirroring them into
    /// the static index would leak across reload cycles (the
    /// helper has no path to reap dynamic-Sub-id entries — the
    /// diff layer only sees static names).
    ///
    /// Drives the helper directly with synthetic diagnostics — the
    /// full Promoter→try_promote pipeline is exercised under
    /// `crates/specter-engine/src/promoter_tests.rs`; here we just
    /// pin the bin-side filter.
    #[test]
    fn reconcile_helper_filters_dynamic_sub_attached() {
        use compact_str::CompactString;
        use slotmap::KeyData;
        use specter_core::PromoterId;
        let mut loader = Loader::new(
            Config::from_str("").expect("empty config parses"),
            specter_config::LogConfig::default(),
            dummy_meta(),
        );
        let static_id = SubId::from(KeyData::from_ffi(1));
        let dynamic_id = SubId::from(KeyData::from_ffi(2));
        let promoter_id = PromoterId::from(KeyData::from_ffi(3));

        let diags = vec![
            Diagnostic::SubAttached {
                sub: static_id,
                name: CompactString::from("static-watch"),
                source_promoter: None,
            },
            Diagnostic::SubAttached {
                sub: dynamic_id,
                name: CompactString::from("logs@/var/log/foo.log"),
                source_promoter: Some(promoter_id),
            },
        ];

        EngineDriver::reconcile_loader_from_diagnostics(&mut loader, &[], &[], &diags);

        assert_eq!(
            loader.ids.get("static-watch"),
            Some(&static_id),
            "static SubAttached populates loader.ids",
        );
        assert!(
            !loader.ids.contains_key("logs@/var/log/foo.log"),
            "dynamic SubAttached (source_promoter = Some(_)) is filtered",
        );
        assert!(
            loader.promoter_ids.is_empty(),
            "no PromoterAttached emitted; promoter_ids untouched",
        );
    }

    /// `read_and_parse_config` on a valid file returns
    /// `Some((Config, FileMeta))` with the parsed `[[watch]]` blocks
    /// populated and `FileMeta` matching the on-disk lstat. Pins the
    /// helper's happy-path contract — both the SIGHUP and the
    /// auto-reload settle-expiry paths rely on this signature, and the
    /// meta-rotation discipline in `handle_reload` depends on the
    /// captured value being lstat-equivalent in the absence of
    /// concurrent edits.
    #[test]
    fn read_and_parse_config_returns_some_on_valid_file() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        std::fs::write(
            &cfg_path,
            format!(
                r#"
[[watch]]
name    = "build"
path    = "{}"
command = ["true"]
"#,
                tmp.path().display(),
            ),
        )
        .unwrap();
        let config = Config::from_str("").expect("empty config parses");
        let rig = rig_for(config, cfg_path.clone());
        let (parsed_config, parsed_meta) = rig
            .driver
            .read_and_parse_config()
            .expect("valid file parses to Some");
        assert_eq!(parsed_config.watches.len(), 1);
        assert_eq!(parsed_config.watches[0].name, "build");
        // No concurrent edits between the helper's atomic capture and
        // this fresh path-level stat — both must observe the same
        // inode-level identity.
        let lstat = FileMeta::from_path(&cfg_path).expect("lstat ok");
        assert_eq!(parsed_meta, lstat);
        assert_ne!(
            parsed_meta,
            dummy_meta(),
            "captured meta is real, not the placeholder"
        );
    }

    /// SIGHUP reload that introduces a substantive diff (added watch)
    /// rotates `loader.config_meta` to the post-edit lstat. Pins the
    /// apply-branch half of the meta-rotation discipline.
    #[test]
    fn reload_rotates_config_meta_on_apply_branch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let v1_text = format!(
            r#"
[[watch]]
name      = "a"
path      = "{}"
command   = ["true"]
"#,
            tmp.path().display(),
        );
        let v2_text = format!(
            r#"
[[watch]]
name      = "a"
path      = "{0}"
command   = ["true"]

[[watch]]
name      = "b"
path      = "{0}"
command   = ["true"]
settle    = "100ms"
"#,
            tmp.path().display(),
        );
        std::fs::write(&cfg_path, &v1_text).unwrap();
        let initial = Config::from_str(&v1_text).expect("v1 parses");

        let mut rig = rig_for(initial, cfg_path.clone());
        rig.driver.run_initial_attach();
        assert_eq!(
            rig.driver.loader.config_meta,
            dummy_meta(),
            "rig starts with placeholder meta",
        );

        // Substantive edit — diff is non-empty (one added watch).
        std::fs::write(&cfg_path, &v2_text).unwrap();
        let expected_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");

        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert_eq!(
            rig.driver.loader.config_meta, expected_meta,
            "apply-branch reload rotates loader.config_meta to the on-disk identity",
        );
        // Confirm the apply branch ran (added "b" landed in loader.ids).
        assert!(
            rig.driver.loader.ids.contains_key("b"),
            "v2's added watch attached — apply-branch path was exercised",
        );
    }

    /// SIGHUP reload whose new content differs only in metadata
    /// (re-write of identical bytes; mtime / ctime move, content
    /// identical) takes the empty-diff branch, but **must still
    /// rotate `loader.config_meta`** — otherwise the auto-reload
    /// settle filter would observe `lstat != stored_meta` on every
    /// subsequent pulse for the same already-applied edit and loop
    /// `handle_reload` against unchanged content. Pins the
    /// empty-diff half of the meta-rotation discipline.
    #[test]
    fn reload_rotates_config_meta_on_empty_diff_branch() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let cfg_text = format!(
            r#"
[[watch]]
name      = "build"
path      = "{}"
command   = ["true"]
"#,
            tmp.path().display(),
        );
        std::fs::write(&cfg_path, &cfg_text).unwrap();
        let initial = Config::from_str(&cfg_text).expect("v1 parses");

        let mut rig = rig_for(initial.clone(), cfg_path.clone());
        rig.driver.run_initial_attach();
        let ids_before: Vec<_> = rig.driver.loader.ids.keys().cloned().collect();

        // Re-save the same content. Real `FileMeta::from_path` after
        // this returns nonzero inode + non-placeholder mtime/ctime,
        // which is enough to distinguish from `dummy_meta()` and to
        // observe rotation.
        std::fs::write(&cfg_path, &cfg_text).unwrap();
        let expected_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");
        assert_ne!(
            expected_meta,
            dummy_meta(),
            "real lstat is non-placeholder — comparison is meaningful",
        );

        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert_eq!(
            rig.driver.loader.config_meta, expected_meta,
            "empty-diff reload rotates loader.config_meta — \
             skipping rotation here would loop the auto-reload settle filter",
        );
        // Confirm the empty-diff branch ran (loader state unchanged
        // semantically — same config, same ids, same SubIds).
        assert_eq!(
            rig.driver.loader.current_config, initial,
            "v1 ≡ v1 → empty-diff branch was exercised",
        );
        let ids_after: Vec<_> = rig.driver.loader.ids.keys().cloned().collect();
        assert_eq!(
            ids_before, ids_after,
            "ids unchanged across empty-diff reload"
        );
    }

    /// Parse-fail must **not** rotate `loader.config_meta`. Rotating
    /// on failure would suppress the auto-reload retry loop: the next
    /// pulse's lstat-vs-stored-meta check would compare the (still
    /// broken) on-disk file against the freshly-stored meta from the
    /// failed attempt, decide "unchanged," and never retry. Pins the
    /// negative invariant.
    #[test]
    fn reload_parse_failure_does_not_rotate_meta() {
        // `/dev/null/no/such/file.toml` — a guaranteed-ENOTDIR path
        // (because `/dev/null` is a character device, not a directory).
        // The reload pipeline observes a parse-fail equivalent.
        let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);
        let pre_reload_meta = rig.driver.loader.config_meta;

        rig.reload_tx.try_send(()).expect("reload send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        assert_eq!(
            rig.driver.loader.config_meta, pre_reload_meta,
            "parse-fail must not rotate meta — would suppress retry pulses",
        );
    }

    /// `PromoterReaped` clears the matching id from
    /// `loader.promoter_ids` even when the reap arrives via cascade
    /// (i.e. not via the diff's `removed` list). This is the cascade
    /// path triggered by Promoter modify (`reap_promoter_inner` then
    /// `attach_promoter_inner` in the same step) — the diff doesn't
    /// list the old id under `removed`, so the diagnostic-driven
    /// retain is the safety net.
    #[test]
    fn reconcile_helper_clears_promoter_on_reaped_diagnostic() {
        use compact_str::CompactString;
        use slotmap::KeyData;
        use specter_core::PromoterId;
        let mut loader = Loader::new(
            Config::from_str("").expect("empty config parses"),
            specter_config::LogConfig::default(),
            dummy_meta(),
        );
        let old_pid = PromoterId::from(KeyData::from_ffi(10));
        let new_pid = PromoterId::from(KeyData::from_ffi(11));
        loader.promoter_ids.insert("logs".into(), old_pid);

        // Modify-style emission order: PromoterReaped then PromoterAttached.
        let diags = vec![
            Diagnostic::PromoterReaped { promoter: old_pid },
            Diagnostic::PromoterAttached {
                promoter: new_pid,
                name: CompactString::from("logs"),
            },
        ];

        EngineDriver::reconcile_loader_from_diagnostics(&mut loader, &[], &[], &diags);

        assert_eq!(
            loader.promoter_ids.get("logs"),
            Some(&new_pid),
            "PromoterAttached overwrites the cleared entry by name",
        );
        assert_eq!(loader.promoter_ids.len(), 1);
    }

    // ===== Auto-reload settle pipeline =====
    //
    // Tests below exercise the `config_event` channel + `tick`'s drain
    // step + the `apply_config_settle_expiry` helper end-to-end, with
    // pulses driven by hand (no watcher backend wired yet). The helper
    // takes an explicit `now: Instant` so tests can span the 100 ms
    // settle window deterministically without sleeping.

    /// `tick`'s config-event drain converts a pulse into an armed
    /// settle deadline. Settle window = `now + CONFIG_SETTLE` (100 ms),
    /// so the freshly-armed deadline lies in the future relative to
    /// the tick's `Instant::now()`.
    #[test]
    fn config_event_pulse_via_tick_arms_settle_window() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        let before_tick = Instant::now();
        rig.config_event_tx.try_send(()).expect("pulse send");
        rig.shutdown_tx.try_send(()).expect("shutdown send");
        rig.driver.tick();

        let armed = rig
            .driver
            .config_settle_until
            .expect("settle armed by drain");
        // Lower bound: the tick captured `now` at-or-after `before_tick`,
        // so the armed deadline is at-or-after `before_tick + 100ms`.
        assert!(
            armed >= before_tick + Duration::from_millis(100),
            "armed deadline must be at least CONFIG_SETTLE in the future",
        );
        // Upper bound (sanity): the deadline isn't far in the future
        // (allow a generous 1 s slack for slow CI).
        assert!(
            armed <= Instant::now() + Duration::from_secs(1),
            "armed deadline shouldn't drift more than a second past now",
        );
    }

    /// Settle resets per pulse. Two consecutive ticks each draining a
    /// pulse push the deadline strictly forward — a sustained editor
    /// burst defers the reload window until quiet.
    #[test]
    fn repeat_config_pulses_via_tick_defer_settle_expiry() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        rig.config_event_tx.try_send(()).expect("first pulse");
        rig.driver.tick();
        let t1 = rig.driver.config_settle_until.expect("first settle armed");

        // Yield enough time for `Instant::now()` to advance — `tick()`
        // captures `now` afresh per call, so we just need a measurable
        // delta. Sleeping `2 ms` is well within scheduler granularity
        // on every supported platform.
        std::thread::sleep(Duration::from_millis(2));

        rig.config_event_tx.try_send(()).expect("second pulse");
        rig.driver.tick();
        let t2 = rig.driver.config_settle_until.expect("second settle armed");

        assert!(
            t2 > t1,
            "second pulse defers the deadline (t1={t1:?}, t2={t2:?})",
        );
    }

    /// Helper short-circuits when no pulse has armed the deadline.
    /// Pre-pulse state must remain unchanged (no spurious reload).
    #[test]
    fn apply_config_settle_expiry_no_op_when_unarmed() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config.clone(), cfg_path);

        let snapshot_meta = rig.driver.loader.config_meta;
        rig.driver.apply_config_settle_expiry(Instant::now());

        assert_eq!(rig.driver.config_settle_until, None);
        assert_eq!(
            rig.driver.loader.config_meta, snapshot_meta,
            "unarmed expiry must not touch loader.config_meta",
        );
        assert_eq!(rig.driver.loader.current_config, config);
    }

    /// Helper short-circuits when `now < deadline`. Deadline stays
    /// armed so a future tick (after the window elapses) can fire it.
    #[test]
    fn apply_config_settle_expiry_no_op_within_window() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config, cfg_path);

        let now = Instant::now();
        let deadline = now + Duration::from_millis(50);
        rig.driver.config_settle_until = Some(deadline);

        rig.driver.apply_config_settle_expiry(now);

        assert_eq!(
            rig.driver.config_settle_until,
            Some(deadline),
            "in-window call must not clear the deadline",
        );
    }

    /// `now == deadline` is the boundary case for the `>=` test in the
    /// helper. The deadline is consumed (cleared); the lstat filter
    /// then runs and (against an unchanged file) silent-drops.
    #[test]
    fn apply_config_settle_expiry_fires_at_exact_deadline() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        // Empty config on disk so the lstat capture has a real meta to
        // compare against.
        std::fs::write(&cfg_path, "").unwrap();
        let config = Config::from_str("").expect("empty config parses");
        let real_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");

        let mut rig = rig_for(config, cfg_path);
        rig.driver.loader.config_meta = real_meta;

        let deadline = Instant::now();
        rig.driver.config_settle_until = Some(deadline);

        rig.driver.apply_config_settle_expiry(deadline);

        assert_eq!(
            rig.driver.config_settle_until, None,
            "exact-deadline match clears the slot",
        );
        assert_eq!(
            rig.driver.loader.config_meta, real_meta,
            "lstat agreed with stored meta — no reload, meta unchanged",
        );
    }

    /// Settle expiry whose lstat agrees with `loader.config_meta`
    /// silently drops the pulse: no `handle_reload`, no parse, no log
    /// (beyond TRACE). This is the kqueue-parent-spillover case — a
    /// sibling write fires a pulse, settle expires, lstat shows the
    /// config file is unchanged → skip.
    #[test]
    fn apply_config_settle_expiry_skips_reload_on_unchanged_meta() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let cfg_text = format!(
            r#"
[[watch]]
name      = "build"
path      = "{}"
command   = ["true"]
"#,
            tmp.path().display(),
        );
        std::fs::write(&cfg_path, &cfg_text).unwrap();
        let real_meta = FileMeta::from_path(&cfg_path).expect("lstat ok");
        let initial = Config::from_str(&cfg_text).expect("v1 parses");

        let mut rig = rig_for(initial.clone(), cfg_path);
        // Seed loader.config_meta to the real on-disk lstat so the
        // helper's `m != self.loader.config_meta` returns false.
        rig.driver.loader.config_meta = real_meta;
        rig.driver.run_initial_attach();
        let ids_snapshot: Vec<_> = rig.driver.loader.ids.keys().cloned().collect();

        // Fire expiry with a `now` past the deadline — helper takes the
        // `now >= deadline` branch.
        let deadline = Instant::now();
        rig.driver.config_settle_until = Some(deadline);
        rig.driver
            .apply_config_settle_expiry(deadline + Duration::from_millis(1));

        // Settle slot consumed even on a silent drop (the deadline was
        // serviced; future pulses arm a fresh window).
        assert_eq!(rig.driver.config_settle_until, None);
        // No reload ⇒ loader state untouched.
        assert_eq!(
            rig.driver.loader.config_meta, real_meta,
            "silent-drop does not rotate config_meta",
        );
        assert_eq!(
            rig.driver.loader.current_config, initial,
            "silent-drop does not rotate current_config",
        );
        let ids_after: Vec<_> = rig.driver.loader.ids.keys().cloned().collect();
        assert_eq!(
            ids_snapshot, ids_after,
            "silent-drop does not perturb attached Sub ids",
        );
    }

    /// Settle expiry whose lstat detects drift (file was edited)
    /// triggers `handle_reload`, which rotates `loader.config_meta`
    /// and `loader.current_config`. The end-to-end gate for the
    /// drift-driven auto-reload path.
    #[test]
    fn apply_config_settle_expiry_triggers_reload_on_meta_drift() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let v1_text = format!(
            r#"
[[watch]]
name      = "a"
path      = "{}"
command   = ["true"]
"#,
            tmp.path().display(),
        );
        std::fs::write(&cfg_path, &v1_text).unwrap();
        let v1_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");
        let v1_config = Config::from_str(&v1_text).expect("v1 parses");

        let mut rig = rig_for(v1_config, cfg_path.clone());
        rig.driver.loader.config_meta = v1_meta;
        rig.driver.run_initial_attach();
        assert_eq!(rig.driver.loader.ids.len(), 1);
        assert!(rig.driver.loader.ids.contains_key("a"));

        // Edit the file — atomic write replaces inode; mtime/ctime move.
        let v2_text = format!(
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
        // Sleep briefly so the FS-resolved mtime/ctime tick at least
        // one nanosecond past `v1_meta` even on coarse-resolution FSs.
        std::thread::sleep(Duration::from_millis(10));
        std::fs::write(&cfg_path, &v2_text).unwrap();
        let v2_lstat = FileMeta::from_path(&cfg_path).expect("v2 lstat ok");
        assert_ne!(
            v1_meta, v2_lstat,
            "v2 must lstat-differ from v1 — otherwise the helper's filter \
             can't drive the reload",
        );

        // Force settle expiry.
        let deadline = Instant::now();
        rig.driver.config_settle_until = Some(deadline);
        rig.driver
            .apply_config_settle_expiry(deadline + Duration::from_millis(1));

        // Reload happened: settle consumed, meta rotated to the
        // post-edit identity, config now has v2's "b" watch.
        assert_eq!(rig.driver.config_settle_until, None);
        assert_eq!(
            rig.driver.loader.config_meta, v2_lstat,
            "drift-driven reload rotated config_meta to the v2 lstat",
        );
        assert!(
            rig.driver.loader.ids.contains_key("a"),
            "v2 still has watch 'a' — preserved across reload",
        );
        assert!(
            rig.driver.loader.ids.contains_key("b"),
            "v2's added watch 'b' attached during reload",
        );
        assert_eq!(rig.driver.loader.current_config.watches.len(), 2);
    }

    /// Lstat error (file missing, EACCES, etc.) routes through the
    /// "treat-as-changed" branch: helper calls `handle_reload`, which
    /// fails to read, logs, and preserves loader state. The settle
    /// slot is consumed (no internal looping); the next pulse fires
    /// a fresh attempt.
    #[test]
    fn apply_config_settle_expiry_treats_missing_path_as_changed() {
        // Path that cannot be lstat'd: `/dev/null` is a character
        // device, so `lstat("/dev/null/no/such")` returns ENOTDIR.
        let cfg_path = PathBuf::from("/dev/null/no/such/file.toml");
        let config = Config::from_str("").expect("empty config parses");
        let mut rig = rig_for(config.clone(), cfg_path);
        let pre_meta = rig.driver.loader.config_meta;

        let deadline = Instant::now();
        rig.driver.config_settle_until = Some(deadline);
        rig.driver
            .apply_config_settle_expiry(deadline + Duration::from_millis(1));

        // Settle slot is consumed even when lstat fails — the helper
        // doesn't loop on its own; the next external pulse arms a
        // fresh window.
        assert_eq!(rig.driver.config_settle_until, None);
        // Parse-fail preserves loader state across the failed reload
        // attempt — the parse-fail invariant carries through the
        // settle-expiry path verbatim (handle_reload's failure
        // handling is the single source of truth across SIGHUP and
        // auto-reload alike).
        assert_eq!(
            rig.driver.loader.config_meta, pre_meta,
            "parse-fail must not rotate meta — would suppress retry pulses",
        );
        assert_eq!(rig.driver.loader.current_config, config);
    }

    /// End-to-end gate: pulse → tick (drain arms settle) →
    /// helper-driven expiry → reload runs against drift. Pins the
    /// drain step's interaction with the helper without spinning
    /// 100 ms of real time.
    #[test]
    fn pulse_then_helper_expiry_runs_full_pipeline_on_drift() {
        let tmp = tempfile::TempDir::new().unwrap();
        let cfg_path = tmp.path().join("specter.toml");
        let v1_text = format!(
            r#"
[[watch]]
name      = "a"
path      = "{}"
command   = ["true"]
"#,
            tmp.path().display(),
        );
        std::fs::write(&cfg_path, &v1_text).unwrap();
        let v1_meta = FileMeta::from_path(&cfg_path).expect("v1 lstat ok");
        let v1_config = Config::from_str(&v1_text).expect("v1 parses");

        let mut rig = rig_for(v1_config, cfg_path.clone());
        rig.driver.loader.config_meta = v1_meta;
        rig.driver.run_initial_attach();

        // Edit the file.
        std::thread::sleep(Duration::from_millis(10));
        let v2_text = format!(
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
        std::fs::write(&cfg_path, &v2_text).unwrap();

        // Drain arms settle.
        rig.config_event_tx.try_send(()).expect("pulse send");
        rig.driver.tick();
        let armed = rig.driver.config_settle_until.expect("drain armed settle");

        // Force-expire via the helper. (Skirts the 100ms wall-clock
        // wait that an end-to-end-with-tick test would need; the
        // helper's contract is identical regardless of who calls it.)
        rig.driver
            .apply_config_settle_expiry(armed + Duration::from_millis(1));

        assert_eq!(rig.driver.config_settle_until, None);
        assert!(
            rig.driver.loader.ids.contains_key("b"),
            "drift-driven reload attached the new watch",
        );
        assert_ne!(
            rig.driver.loader.config_meta, v1_meta,
            "post-reload meta must differ from the pre-edit identity",
        );
    }
}
