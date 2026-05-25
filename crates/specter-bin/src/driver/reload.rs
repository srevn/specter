//! The reload pipeline — SIGHUP and auto-reload, both on the engine
//! thread (no Mutex).
//!
//! [`EngineDriver::dispatch_reload`] is the shared apply path:
//! `obs_handle.reopen_file()` fires first (so logrotate cadence is
//! independent of the config-parse outcome), then the file is read
//! with an atomic [`FileMeta`] capture, the name-keyed diff is
//! applied via `Input::ConfigDiff` (the engine resolves names to ids
//! against its own registries), and `loader`'s config / log / meta
//! commit in a single post-apply step. The `[log]` block is
//! re-resolved (CLI overrides re-applied) and `obs_handle` updated
//! in the same pass.
//!
//! Two entry points converge here. SIGHUP calls `dispatch_reload`
//! directly. Auto-reload is settle-debounced: [`super::tick`] arms
//! `config_settle_until` per config-event pulse;
//! [`EngineDriver::apply_config_settle_expiry`] fires on quiet,
//! filters phantom pulses with a single `lstat`
//! ([`EngineDriver::config_meta_changed`]) and calls the same
//! `dispatch_reload` on confirmed [`FileMeta`] drift — so the
//! meta-rotation discipline converges across both pulse sources.

use super::EngineDriver;
use super::state::ReloadTrigger;
use specter_config::{Config, FileMeta, LogConfig};
use specter_core::Input;
use specter_sensor::FsWatcher;
use std::ops::ControlFlow;
use std::time::Instant;

impl<W: FsWatcher> EngineDriver<W> {
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
    ///   [`Self::dispatch_reload`] on drift. The lstat filter is what
    ///   suppresses no-op pulses — a kqueue parent-dir spillover from
    ///   a sibling write fires a pulse but doesn't move
    ///   `loader.config_meta`, so the lstat compares equal and we
    ///   skip the parse.
    ///
    /// `pub(super)` so the driver's own tests can drive the helper
    /// directly with a synthetic `now`, avoiding real-time sleeps
    /// across the settle window. Production callers go through
    /// `tick`, which always passes `Instant::now()`.
    ///
    /// Returns [`ControlFlow::Break`] iff [`Self::dispatch_reload`]
    /// observed shutdown (`forward`'s outbound `effects_tx`
    /// disconnected mid-apply). `dispatch_reload` drains in-flight
    /// probes internally on `Break`; the tick-layer outer
    /// `begin_shutdown` after this returns is redundant but
    /// idempotent.
    pub(super) fn apply_config_settle_expiry(&mut self, now: Instant) -> ControlFlow<()> {
        let Some(deadline) = self.config_settle_until else {
            return ControlFlow::Continue(());
        };
        if now < deadline {
            return ControlFlow::Continue(());
        }
        self.config_settle_until = None;
        if self.config_meta_changed() {
            self.dispatch_reload(ReloadTrigger::AutoReload, now)
        } else {
            ControlFlow::Continue(())
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
    ///    to `Ok`, which is structurally a transition — dispatch_reload
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

    /// Read the config from disk, diff against the running snapshot,
    /// apply via `Input::ConfigDiff` (the engine resolves names to
    /// ids), and commit `loader`'s `current_config` / `current_log` /
    /// `config_meta` in one post-apply step covering both diff
    /// branches. On parse failure, log + retain running config + log
    /// (preserving the auto-reload retry loop on the next pulse) and
    /// capture a best-effort post-fail [`FileMeta`] so a recovery
    /// edit drives a fresh attempt.
    ///
    /// **Reopen-first.** [`Self::reopen_log_file`] runs at the top of
    /// the body, before the parse early-return. Logrotate cadence is
    /// independent of operator-driven config edits — placing the
    /// reopen upstream of every branch means a `logrotate -f` + a
    /// broken-TOML edit + SIGHUP still rotates the log file. The
    /// structural position is the contract; no branch below can
    /// bypass it.
    ///
    /// **Log-side dispatch.** The `[log]` block is re-resolved (CLI
    /// overrides re-applied) and applied per-field via
    /// [`Self::apply_log_reload`]: a level change calls
    /// `obs_handle.set_level`; a destination / path change logs an
    /// `error!` instructing the operator to restart — destinations
    /// are not hot-reloadable. The actually-applied state rotates
    /// into [`crate::loader::Loader::current_log`] — so a destination
    /// flip-back compares runtime-vs-runtime on the next reload and
    /// avoids a phantom "restart to apply" warning.
    ///
    /// **Meta rotation discipline.** `loader.config_meta` rotates on
    /// every observed file state: [`crate::loader::Loader::rotate_apply`]
    /// on success (both empty-diff and apply-diff branches converge
    /// there) and [`crate::loader::Loader::rotate_meta_only`] on
    /// parse-fail with a successful post-fail lstat — the methods
    /// carry the per-branch rationale. If the post-fail lstat itself
    /// fails (parent dir gone, etc.), the existing meta is preserved
    /// — [`Self::config_meta_changed`]'s "Err = treat as changed"
    /// semantics keep the retry loop alive on the next pulse.
    ///
    /// **Reload-counter discipline.** [`super::state::DriverState::record_reload`]
    /// fires immediately after a successful parse — covering both
    /// the empty-diff and apply-diff branches (the operator's pulse
    /// is honoured either way; only the engine-side work differs).
    /// A parse-fail short-circuits *before* this site, so the
    /// counters never advance on a failed reload. The `trigger`
    /// argument carries per-caller attribution (SIGHUP vs auto-
    /// reload settle expiry).
    ///
    /// Returns the [`ControlFlow`] from the apply-branch `forward` so
    /// a shutdown observed mid-apply (`forward`'s outbound `effects_tx`
    /// disconnect) propagates back to the caller. On `Break` the
    /// cancel-first probe drain via
    /// [`super::EngineDriver::begin_shutdown`] runs internally,
    /// symmetric with [`super::EngineDriver::run_initial_attach`] —
    /// every engine-mutating entry point that may arm probes (the
    /// diff's `added` Subs / Promoters enter Seed-Verifying with an
    /// armed `ProbeSlot`) is responsible for the cleanup of those
    /// probes on its own `Break`. A caller that drops the driver
    /// after observing `Break` cannot trip `ProbeSlot::drop`'s
    /// linear-edge tripwire. The tick-layer outer drain that
    /// also calls `begin_shutdown` on `Break` propagation is
    /// redundant for this path but idempotent (the cancel-first
    /// drain iterates over an already-empty in-flight-probe set).
    ///
    /// The loader rotation runs even on the `Break` path: the engine
    /// has the diff applied, so the loader matches the engine on the
    /// way out, and the driver is about to shut down anyway — keeping
    /// the rotation in front of the return is the clearer invariant.
    pub(crate) fn dispatch_reload(
        &mut self,
        trigger: ReloadTrigger,
        now: Instant,
    ) -> ControlFlow<()> {
        // A manual reload (SIGHUP / IPC) supersedes any pending
        // auto-reload settle deadline: the operator's explicit pulse
        // pre-empts the debounced pulse, and a deadline left armed
        // would either fire on a freshly-applied config (no-op via
        // `config_meta_changed`'s lstat filter) or, worse, race a
        // mid-window config edit and trigger a second apply. Clearing
        // the deadline here is idempotent for the
        // `apply_config_settle_expiry` caller (it cleared the slot
        // before delegating here) and load-bearing for the other two
        // entry points (`handle_ipc::P::Reload`, signal-thread SIGHUP).
        self.config_settle_until = None;

        self.reopen_log_file();

        let Some((new_config, new_meta)) = self.read_and_parse_config() else {
            if let Ok(post_fail_meta) = FileMeta::from_path(&self.config_path) {
                self.loader.rotate_meta_only(post_fail_meta);
            }
            return ControlFlow::Continue(());
        };

        // Parse succeeded — bump the reload counters. The bump records
        // that the operator's pulse was acknowledged, not that the
        // engine-side apply succeeded: a shutdown observed mid-apply
        // (forward returning Break below) still leaves the bump
        // standing, mirroring the loader-rotation discipline ("we
        // accepted the work; we may not finish it cleanly"). Both the
        // empty-diff and apply-diff branches converge here for the
        // same reason — an operator re-save of unchanged bytes is
        // still a reload from the operator's view.
        self.driver_state.record_reload(trigger);

        let applied_log = self.apply_log_reload(&new_config);

        // Filter the diff against `disabled_runtime` BEFORE the
        // engine sees it: any attach / re-attach / re-bind / detach
        // for a runtime-disabled Sub is suppressed so the engine
        // stays consistent with the operator's "off" preference
        // across the apply.
        let diff = self.compute_watch_diff(&new_config);
        let outcome = if diff.is_empty() {
            tracing::info!("config reload: no watch changes");
            ControlFlow::Continue(())
        } else {
            // Snapshot the change-counts for the post-apply summary
            // log before the diff moves into the engine. Name-keyed:
            // the engine resolves each name to its live id against
            // its own registries, so the bin keeps no `name → id`
            // mirror to reconcile afterwards. The Sub side reports
            // `modified_identity` and `modified_params` separately
            // so triage can see at a glance whether a reload tore
            // down Profiles (identity) or only rebound per-Sub
            // fields (params).
            let added_n = diff.subs.added.len();
            let removed_n = diff.subs.removed.len();
            let modified_identity_n = diff.subs.modified_identity.len();
            let modified_params_n = diff.subs.modified_params.len();
            let promoter_added_n = diff.promoters.added.len();
            let promoter_removed_n = diff.promoters.removed.len();
            let promoter_modified_n = diff.promoters.modified.len();

            let out = self.engine.step(Input::ConfigDiff(diff), now);
            tracing::info!(
                added = added_n,
                removed = removed_n,
                modified_identity = modified_identity_n,
                modified_params = modified_params_n,
                promoters_added = promoter_added_n,
                promoters_removed = promoter_removed_n,
                promoters_modified = promoter_modified_n,
                "config reload applied",
            );
            self.forward(out)
        };

        self.loader.rotate_apply(new_config, applied_log, new_meta);

        // Prune `disabled_runtime` AFTER the loader rotation so the
        // helper reads the freshly-applied TOML. The prune runs on
        // every successful parse (both empty-diff and apply-diff
        // branches converge here) — an edit that removes a
        // runtime-disabled entry must clear the override on the same
        // pulse, even when no other diff bits moved.
        self.prune_disabled_runtime_against_current_config();

        // Cancel-first probe drain on `Break`, symmetric with
        // `run_initial_attach`'s internal drain. The apply-branch
        // `engine.step` may have transitioned freshly-added Subs to
        // Seed-Verifying with armed `ProbeSlot`s; on a `Break` from
        // `forward` the engine retains those probes. Without this
        // drain a caller that does not loop back through `tick` (the
        // boot-time call from `App::run`) would drop the driver with
        // armed probes and trip `ProbeSlot::drop`'s linear-edge
        // tripwire. The drain is idempotent — tick-layer callers
        // also call `begin_shutdown` on the propagated `Break`, and
        // the second pass iterates over an empty in-flight set.
        if outcome.is_break() {
            let _ = self.begin_shutdown();
        }

        outcome
    }

    /// Re-open the log file destination so `logrotate`'s
    /// `copytruncate` / move-then-create rotation cycles pick up a
    /// freshly-opened path without a daemon restart. Errors land at
    /// `warn!` — the rotator may have raced us to the path, in which
    /// case the existing fd is still usable.
    ///
    /// Sole call site is the top of [`Self::dispatch_reload`]; every
    /// branch below it is structurally downstream, so a broken-TOML
    /// edit landing during a `logrotate` cycle still rotates the log
    /// file — the reopen does not depend on the parse outcome.
    fn reopen_log_file(&self) {
        if let Err(e) = self.obs_handle.reopen_file() {
            tracing::warn!(
                error = ?e,
                path = ?self.obs_handle.file_path().map(|p| p.display().to_string()),
                "log file reopen failed; keeping existing fd",
            );
        }
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
    pub(super) fn read_and_parse_config(&self) -> Option<(Config, FileMeta)> {
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
    /// wins, matching startup precedence) and apply per-field against
    /// the running runtime state. Returns the *actually-applied*
    /// [`LogConfig`] — the shape that rotates into
    /// [`crate::loader::Loader::current_log`].
    ///
    /// Per-field discipline — level and destination dispatch
    /// independently, no enum intermediate:
    ///
    /// - **level** — hot-reloadable. When the request differs from
    ///   runtime, calls `obs_handle.set_level`. On `Ok` the applied
    ///   level is the new one; on `Err` it falls back to the running
    ///   value (with an `error!` log).
    /// - **destination / path** — NOT hot-reloadable. When the
    ///   request differs from runtime, logs an `error!` instructing
    ///   the operator to restart. The applied destination / path are
    ///   always the running values; the appender does not move.
    ///
    /// On `merge_cli` validation failure (operator typo, missing
    /// `path` under `destination = "file"`), logs the issue and
    /// returns the running log unchanged — the watch-side reload can
    /// still proceed independently. `merge_cli` surfaces a bare
    /// [`specter_config::ValidationIssue`]; the structured `error`
    /// field renders the issue's `Display` directly
    /// (`<field>: <detail> (<kind>)`), without the `<inline>: N
    /// validation error(s)` envelope the
    /// [`specter_config::ConfigError::Validate`] shape would impose on
    /// a CLI-merge failure.
    ///
    /// The applied-state framing closes a phantom-warning class:
    /// because the rotated `current_log` carries the running shape, a
    /// destination flip-back compares runtime-vs-runtime on the next
    /// reload and reports no change — instead of comparing
    /// requested-vs-running and re-firing "restart to apply" on every
    /// flip. The trade-off is that a *sustained* mismatch (operator
    /// requests a new destination and leaves it that way) re-fires
    /// the warning on every reload until restart — accepted because
    /// the warning is correct on every firing and reload frequency is
    /// operator-bounded.
    fn apply_log_reload(&self, new_config: &Config) -> LogConfig {
        let requested = match new_config.log.clone().merge_cli(
            self.cli_log_overrides.level,
            self.cli_log_overrides.destination,
            self.cli_log_overrides.path.clone(),
        ) {
            Ok(r) => r,
            Err(issue) => {
                tracing::error!(
                    issue = %issue,
                    "log reload failed; keeping running log config",
                );
                return self.loader.current_log.clone();
            }
        };

        let running = &self.loader.current_log;

        let applied_level = if requested.level == running.level {
            running.level
        } else {
            match self.obs_handle.set_level(requested.level) {
                Ok(()) => {
                    tracing::info!(
                        new_level = ?requested.level,
                        "log level updated via SIGHUP",
                    );
                    requested.level
                }
                Err(e) => {
                    tracing::error!(
                        error = ?e,
                        "log level reload failed; keeping prior level",
                    );
                    running.level
                }
            }
        };

        if requested.destination != running.destination || requested.path != running.path {
            tracing::error!(
                new_destination = ?requested.destination,
                new_path = ?requested.path.as_ref().map(|p| p.display().to_string()),
                "log destination / path change is not hot-reloadable; \
                 restart specter to apply",
            );
        }

        LogConfig {
            level: applied_level,
            destination: running.destination,
            path: running.path.clone(),
        }
    }

    /// Compute the name-keyed diff between the running and
    /// freshly-parsed config, then strip every bucket of any Sub
    /// whose name lives in [`Self::disabled_runtime`].
    ///
    /// The unfiltered diff is the raw `specter_config::diff` output;
    /// the filter is the negative side of the `disabled_runtime ↔
    /// engine` invariant — an attach / re-attach / re-bind / detach
    /// for a runtime-disabled Sub would re-introduce or churn the
    /// engine on a Sub the operator has suppressed. The four Sub
    /// buckets (`added`, `removed`, `modified_identity`,
    /// `modified_params`) are name-disjoint by construction, so
    /// retaining by name on each is exhaustive. Promoters are not in
    /// scope for the runtime-disable override and pass through
    /// unchanged.
    pub(super) fn compute_watch_diff(
        &self,
        new_config: &Config,
    ) -> specter_core::WatchRegistryDiff {
        let mut diff = specter_config::diff(&self.loader.current_config, new_config);
        let disabled = &self.disabled_runtime;
        diff.subs
            .added
            .retain(|r| !disabled.contains(r.params.name.as_str()));
        diff.subs
            .modified_identity
            .retain(|r| !disabled.contains(r.params.name.as_str()));
        diff.subs
            .modified_params
            .retain(|r| !disabled.contains(r.params.name.as_str()));
        diff.subs.removed.retain(|n| !disabled.contains(n.as_str()));
        diff
    }

    /// Retain only those `disabled_runtime` entries whose `[[watch]]`
    /// entry still exists in the freshly-applied TOML (regardless of
    /// the per-row `enabled` flag).
    ///
    /// A name that left the TOML entirely is structurally invalid as
    /// a runtime override: the operator's "off" preference cannot
    /// anchor against a missing entry, and a future re-declaration is
    /// a fresh operator decision, not a revival. A `enabled = false`
    /// row stays operator-declared, so the runtime override stacked
    /// over it is preserved (the operator's "off twice" choice).
    ///
    /// Reads through the FULL [`Config::watches`] list so the
    /// TOML-disabled retention case works correctly — filtering
    /// through [`Config::active_watches`] would drop those entries.
    ///
    /// Sole production caller is [`Self::dispatch_reload`] *after*
    /// [`crate::loader::Loader::rotate_apply`] — so `current_config`
    /// is the newly-applied state. `pub(super)` so the driver's own
    /// tests can exercise the helper directly without driving the
    /// full reload pipeline.
    pub(super) fn prune_disabled_runtime_against_current_config(&mut self) {
        self.disabled_runtime.retain(|name| {
            self.loader
                .current_config
                .watches
                .iter()
                .any(|s| s.name.as_str() == name.as_str())
        });
    }
}
