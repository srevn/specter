//! `Loader` — the bin's persistent reload state.
//!
//! Holds the most recently applied [`Config`] (the snapshot the next reload diffs against) plus its
//! resolved `[log]` block and on-disk [`FileMeta`] identity. It carries **no** `name → id` map:
//! hot-reload diffs are name-keyed and the engine resolves names to ids through its own
//! authoritative registries, so the bin never mirrors engine identity.
//!
//! Lives on the engine driver thread — file I/O on SIGHUP runs there too, eliminating the Mutex an
//! early design sketch anticipated.
//!
//! **Encapsulation.** Fields are private; construction goes through [`Loader::new`] at boot and
//! rotation through [`Loader::rotate_apply`] / [`Loader::rotate_meta_only`]. Read access is via
//! [`Loader::current_config`], [`Loader::current_log`], and [`Loader::config_meta`]. Falsifiable in
//! one grep — production code contains zero `loader.{field} =` assignments, because the field names
//! are not in scope outside this module.

use specter_config::{Config, FileMeta, LogConfig};

/// Bin-side reload state. See module rustdoc.
#[derive(Debug)]
pub(crate) struct Loader {
    current_config: Config,
    /// `[log]` block as resolved at startup or after the last successful reload — *with* CLI
    /// overrides folded in. Compared against the next reload's resolved value to decide whether to
    /// call `obs_handle.set_level` and / or fail-with-error on a destination change.
    current_log: LogConfig,
    /// Inode-level identity of `current_config`'s on-disk source — captured atomically with the
    /// content read via [`Config::from_path_with_meta`] (the `f.metadata()` call binds to the same
    /// `File` handle that produced the bytes, so a concurrent `rename(2)` cannot rotate the meta
    /// out from under the bytes). Rotated alongside `current_config` on every successful reload —
    /// **including the empty-diff branch**, so a re-saved-but-identical file still updates the
    /// stored identity. Without that rotation, the auto-reload settle-expiry filter would observe a
    /// fresh lstat that differs from the stored value forever, looping `dispatch_reload` against
    /// the same content.
    config_meta: FileMeta,
}

impl Loader {
    /// Build the boot-time loader from the freshly-parsed config, the resolved log block (CLI
    /// overrides folded in), and the atomic [`FileMeta`] capture that produced the config bytes.
    /// The single production construction site is `App::run`; tests construct via the same path so
    /// the seeding rule stays single-source.
    #[must_use]
    pub(crate) const fn new(
        current_config: Config,
        current_log: LogConfig,
        config_meta: FileMeta,
    ) -> Self {
        Self {
            current_config,
            current_log,
            config_meta,
        }
    }

    /// Borrow the currently-applied [`Config`].
    pub(crate) const fn current_config(&self) -> &Config {
        &self.current_config
    }

    /// Borrow the currently-applied [`LogConfig`] — the runtime shape, with CLI overrides folded in.
    pub(crate) const fn current_log(&self) -> &LogConfig {
        &self.current_log
    }

    /// Read the stored [`FileMeta`]. `FileMeta` is [`Copy`] (a flat inode-level fingerprint), so
    /// the by-value return matches the comparison-then-discard usage on the auto-reload
    /// settle-expiry path without forcing callers to deref.
    pub(crate) const fn config_meta(&self) -> FileMeta {
        self.config_meta
    }

    /// Atomic rotation across all three fields on a successful reload — both the apply-diff and
    /// empty-diff branches of `EngineDriver::dispatch_reload` converge here. The empty-diff branch
    /// passes `config == self.current_config` and an identical log shape; rotation is still
    /// required because `meta` must advance to the fresh lstat so the auto-reload settle filter
    /// doesn't loop against an already-applied edit. The `meta` value is captured atomically with
    /// the parsed bytes ([`Config::from_path_with_meta`]), so a concurrent `rename(2)` can't rotate
    /// it out from under the content this snapshot represents.
    pub(crate) fn rotate_apply(&mut self, config: Config, log: LogConfig, meta: FileMeta) {
        self.current_config = config;
        self.current_log = log;
        self.config_meta = meta;
    }

    /// Meta-only rotation on parse-failure with a successful post-fail lstat. Closes the
    /// chmod-EACCES recovery loop: without this rotation, stored meta would freeze at the
    /// pre-tighten state, and the post-loosen lstat would compare equal — silently breaking
    /// auto-recovery (see [`FileMeta`]'s rustdoc on mode/uid/gid as the access-side fingerprint).
    /// The post-fail lstat captures the locked-out state instead, so the recovery chmod's lstat
    /// differs and re-fires `dispatch_reload`.
    pub(crate) const fn rotate_meta_only(&mut self, meta: FileMeta) {
        self.config_meta = meta;
    }
}
