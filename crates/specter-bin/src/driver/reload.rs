//! The reload pipeline — SIGHUP and auto-reload, both on the engine
//! thread (no Mutex).
//!
//! [`EngineDriver::handle_reload`] is the shared apply path: read the
//! file with an atomic [`FileMeta`] capture, compute the name-keyed
//! diff, apply via `Input::ConfigDiff` (the engine resolves names to
//! ids against its own registries), then rotate `loader`'s config /
//! log / meta. The `[log]` block is re-resolved (CLI overrides
//! re-applied) and `obs_handle` updated in the same pass.
//!
//! Two entry points converge here. SIGHUP calls `handle_reload`
//! directly. Auto-reload is settle-debounced: [`super::tick`] arms
//! `config_settle_until` per config-event pulse;
//! [`EngineDriver::apply_config_settle_expiry`] fires on quiet,
//! filters phantom pulses with a single `lstat`
//! ([`EngineDriver::config_meta_changed`]) and calls the same
//! `handle_reload` on confirmed [`FileMeta`] drift — so the
//! meta-rotation discipline converges across both pulse sources.

use super::EngineDriver;
use crate::observability::LogReloadKind;
use specter_config::{Config, FileMeta};
use specter_core::Input;
use std::time::Instant;

impl EngineDriver {
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
    /// `pub(in crate::driver)` so the driver's own tests can drive
    /// the helper directly with a synthetic `now`, avoiding real-time
    /// sleeps across the settle window. Production callers go through
    /// `tick`, which always passes `Instant::now()`.
    pub(in crate::driver) fn apply_config_settle_expiry(&mut self, now: Instant) {
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
    /// snapshot, apply via `Input::ConfigDiff` (the engine resolves
    /// names to ids), rotate `loader.current_config` and
    /// `loader.config_meta`. On failure, log + retain running config +
    /// meta (preserving the auto-reload retry loop on the next pulse).
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
    /// **every** observed file state — both successful reads (empty
    /// diff and apply diff) **and** parse / open failures (best-effort
    /// post-failure lstat). The success branches keep the auto-reload
    /// settle-expiry filter from looping against an already-applied
    /// edit. The parse-fail rotation closes the chmod-EACCES recovery
    /// hole: with mode/uid/gid as the access-side fingerprint (see
    /// [`FileMeta`]), stored meta would freeze at the pre-tighten
    /// value if parse-fail preserved it, and the post-loosen lstat
    /// would compare equal — silently breaking auto-recovery. The
    /// post-fail lstat captures the locked-out state instead, so the
    /// recovery chmod's lstat differs and re-fires `handle_reload`.
    /// If the post-fail lstat itself fails (parent dir gone, etc.),
    /// preserve the existing meta — `config_meta_changed`'s
    /// "Err = treat as changed" semantics keep the retry loop alive.
    pub(in crate::driver) fn handle_reload(&mut self, now: Instant) {
        let Some((new_config, new_meta)) = self.read_and_parse_config() else {
            if let Ok(post_fail_meta) = FileMeta::from_path(&self.config_path) {
                self.loader.config_meta = post_fail_meta;
            }
            return;
        };
        let new_log_resolved = self.parse_and_resolve_log(&new_config);
        self.apply_log_reload(&new_log_resolved);

        let diff = self.compute_watch_diff(&new_config);
        if diff.is_empty() {
            tracing::info!("config reload: no watch changes");
            self.loader.current_config = new_config;
            self.loader.current_log = new_log_resolved;
            self.loader.config_meta = new_meta;
            return;
        }

        // Snapshot the change-counts for the post-apply summary log
        // before the diff moves into the engine. Name-keyed: the
        // engine resolves each name to its live id against its own
        // registries, so the bin keeps no `name → id` mirror to
        // reconcile afterwards. The Sub side reports `modified_identity`
        // and `modified_params` separately so triage can see at a
        // glance whether a reload tore down Profiles (identity) or
        // only rebound per-Sub fields (params).
        let added_n = diff.subs.added.len();
        let removed_n = diff.subs.removed.len();
        let modified_identity_n = diff.subs.modified_identity.len();
        let modified_params_n = diff.subs.modified_params.len();
        let promoter_added_n = diff.promoters.added.len();
        let promoter_removed_n = diff.promoters.removed.len();
        let promoter_modified_n = diff.promoters.modified.len();

        let out = self.engine.step(Input::ConfigDiff(diff), now);

        self.loader.current_config = new_config;
        self.loader.current_log = new_log_resolved;
        self.loader.config_meta = new_meta;
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
    pub(in crate::driver) fn read_and_parse_config(&self) -> Option<(Config, FileMeta)> {
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
    ///
    /// `merge_cli` surfaces a bare [`specter_config::ValidationIssue`];
    /// the structured `error` field renders the issue's `Display` directly
    /// (`<field>: <detail> (<kind>)`), without the `<inline>: N validation
    /// error(s)` envelope the [`specter_config::ConfigError::Validate`]
    /// shape would impose on a CLI-merge failure.
    pub(in crate::driver) fn parse_and_resolve_log(
        &self,
        new_config: &Config,
    ) -> specter_config::LogConfig {
        match new_config.log.clone().merge_cli(
            self.cli_log_overrides.level,
            self.cli_log_overrides.destination,
            self.cli_log_overrides.path.clone(),
        ) {
            Ok(c) => c,
            Err(issue) => {
                tracing::error!(
                    issue = %issue,
                    "log reload failed; keeping running log config",
                );
                self.loader.current_log.clone()
            }
        }
    }

    /// Compute the name-keyed diff between the running and
    /// freshly-parsed config. Pure delegation to [`specter_config::diff`]
    /// — no id maps; the engine resolves names to ids at apply time.
    pub(in crate::driver) fn compute_watch_diff(
        &self,
        new_config: &Config,
    ) -> specter_core::WatchRegistryDiff {
        specter_config::diff(&self.loader.current_config, new_config)
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
}
