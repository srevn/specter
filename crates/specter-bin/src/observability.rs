//! Engine-telemetry subscriber bootstrap.
//!
//! The bin's ergonomic surface is:
//!
//! ```ignore
//! let (obs_handle, _obs_guard) = observability::init(&log_cfg)?;
//! // give `obs_handle` to the engine driver — it calls `set_level` /
//! // `reopen_file` on SIGHUP. Keep `_obs_guard` on the App's stack so it
//! // outlives every trailing `tracing::*` call.
//! ```
//!
//! The split — control-half ([`ObservabilityHandle`]) vs guard-half
//! ([`ObservabilityGuard`]) — is load-bearing for `destination = file`:
//! the appender's worker thread is shut down when the `WorkerGuard` drops,
//! and any subsequent `tracing::*` event is dropped on the floor by
//! `tracing-appender::non_blocking`. If the engine driver owned the
//! guard, every event between `drop(driver)` and end-of-`run()`
//! ("watcher thread joined", "specter exited cleanly", etc.) would be
//! silently lost. Holding the guard on `App::run`'s stack frame defers
//! the appender shutdown until the very end of the process lifetime.
//!
//! Two SIGHUP-driven hooks live on [`ObservabilityHandle`]:
//!
//! - **Level reload** — a `tracing_subscriber::reload::Layer` wraps the
//!   `EnvFilter`. The bin re-applies the level via
//!   [`ObservabilityHandle::set_level`] when SIGHUP brings a new
//!   `[log] level`. Path / destination changes are *not* hot-reloaded
//!   in v1 (an explicit `error!` instructs the operator to restart).
//!
//! - **Reopen-on-SIGHUP** — when destination = `file`, the underlying
//!   `Arc<Mutex<File>>` is swappable. The bin calls
//!   [`ObservabilityHandle::reopen_file`] unconditionally on SIGHUP so
//!   `logrotate copytruncate` / move-then-create rotation cycles see
//!   their freshly-opened path picked up without restart.
//!
//! Subprocess output is *not* this module's concern. See the actuator
//! and `Effect.capture_output` for how user-visible bytes are routed.

use specter_config::{LogConfig, LogDestination, LogLevel};
use std::fs::OpenOptions;
use std::io::{self, IsTerminal, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_subscriber::{
    EnvFilter, Registry, layer::SubscriberExt as _, reload, util::SubscriberInitExt as _,
};

/// Control surface for SIGHUP-driven runtime updates.
///
/// Held by the engine driver; the bin's `App::run` keeps the paired
/// [`ObservabilityGuard`] separately so `Drop` order doesn't truncate
/// trailing log events.
///
/// Cheap to construct via [`ObservabilityHandle::noop`] for tests that
/// don't exercise the SIGHUP API.
pub struct ObservabilityHandle {
    /// Reload handle for the level filter. `None` for noop handles
    /// (tests). Path / destination changes are not hot-reloaded in v1.
    level_reload: Option<reload::Handle<EnvFilter, Registry>>,
    /// `Some` iff destination = `file`. Reopens the configured path
    /// on demand (logrotate `copytruncate`-style rotation needs this).
    file_reopen: Option<ReopenHandle>,
    /// The path we wrote to at init time. Surfaced so the driver can
    /// log a coherent "reopened X" message on SIGHUP without touching
    /// the config snapshot.
    file_path: Option<PathBuf>,
}

impl std::fmt::Debug for ObservabilityHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservabilityHandle")
            .field("file_path", &self.file_path)
            .field("level_reload", &self.level_reload.is_some())
            .field("file_reopen", &self.file_reopen.is_some())
            .finish()
    }
}

impl ObservabilityHandle {
    /// Replace the active level. Errors are reported to the caller (the
    /// engine driver) at `error!` and otherwise ignored — a failed reload
    /// leaves the existing level in place. Returns `Ok(())` for noop
    /// handles (the level is already a no-op).
    pub fn set_level(&self, level: LogLevel) -> Result<(), reload::Error> {
        let Some(handle) = &self.level_reload else {
            return Ok(());
        };
        let new_filter = EnvFilter::new(level_directive(level));
        handle.modify(|f| *f = new_filter)
    }

    /// Reopen the file destination. `Ok(())` when destination is
    /// `Stderr` (no-op). Production callers (the SIGHUP path) call this
    /// unconditionally; logrotate's `copytruncate` mode benefits even
    /// when the path didn't change in config.
    pub fn reopen_file(&self) -> io::Result<()> {
        let Some(handle) = &self.file_reopen else {
            return Ok(());
        };
        let Some(path) = &self.file_path else {
            return Ok(());
        };
        handle.reopen(path)
    }

    /// The file destination's path, if any. Used by the driver for log
    /// messages — never to make routing decisions.
    #[must_use]
    pub fn file_path(&self) -> Option<&Path> {
        self.file_path.as_deref()
    }

    /// Construct a handle whose [`Self::set_level`] and
    /// [`Self::reopen_file`] are both no-ops. Intended for unit tests
    /// that need an `ObservabilityHandle`-shaped value but don't drive
    /// the SIGHUP API; calling [`init`] in tests would either fight the
    /// global subscriber installed by a sibling test or pollute stderr
    /// with engine logs.
    #[cfg(test)]
    #[must_use]
    pub(crate) const fn noop() -> Self {
        Self {
            level_reload: None,
            file_reopen: None,
            file_path: None,
        }
    }
}

/// Lifetime guard for the file-destination appender's worker thread.
///
/// `Drop` flushes pending events and joins the worker. Hold on the
/// process's outermost stack frame so trailing `tracing::*` events
/// (post-driver-shutdown sequence in `App::run`) still reach disk.
///
/// `Stderr`-destination guards are empty (the field is `None`); they
/// drop in O(1) and exist purely to keep the API symmetric.
///
/// The `file_guard` field is held purely for its [`Drop`] side effect
/// (`WorkerGuard::drop` flushes pending events and joins the worker
/// thread); the field is never read directly.
#[must_use = "drop the guard *last* — `tracing::*` events fired on a \
              dropped guard's appender are silently discarded"]
pub struct ObservabilityGuard {
    file_guard: Option<WorkerGuard>,
}

impl std::fmt::Debug for ObservabilityGuard {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ObservabilityGuard")
            .field("file_guard", &self.file_guard.is_some())
            .finish()
    }
}

/// Install the global subscriber. Must be called exactly once per
/// process; a second call returns
/// [`io::ErrorKind::AlreadyExists`] (the global subscriber slot is
/// taken).
///
/// Validation of `cfg` is the caller's responsibility — for
/// [`LogDestination::File`], `cfg.path` must be `Some` and absolute,
/// the parent must exist. The function still attempts to open the file
/// once; an `io::Error` propagates with the offending path embedded so
/// the bin can surface a sane startup failure message before the
/// subscriber is alive.
pub fn init(cfg: &LogConfig) -> io::Result<(ObservabilityHandle, ObservabilityGuard)> {
    let directive = level_directive(cfg.level);
    let env_filter = EnvFilter::new(directive);
    let (filter_layer, level_reload) = reload::Layer::new(env_filter);

    match cfg.destination {
        LogDestination::Stderr => {
            let ansi = std::io::stderr().is_terminal();
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_target(true)
                .with_thread_names(true)
                .with_file(false)
                .with_line_number(false)
                .with_ansi(ansi);
            tracing_subscriber::registry()
                .with(filter_layer)
                .with(fmt_layer)
                .try_init()
                .map_err(|e| already_installed_err(&e))?;
            Ok((
                ObservabilityHandle {
                    level_reload: Some(level_reload),
                    file_reopen: None,
                    file_path: None,
                },
                ObservabilityGuard { file_guard: None },
            ))
        }
        LogDestination::File => {
            let path = cfg.path.as_ref().ok_or_else(|| {
                io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "LogDestination::File requires a path; caller must validate",
                )
            })?;
            // Single open here — the bin's calling layer surfaces this
            // as "config load failed" with the file path included so
            // operators can grep for it without parsing tracing output.
            let writer = ReopenableFile::open(path).map_err(|e| {
                io::Error::new(
                    e.kind(),
                    format!("failed to open log file `{}`: {e}", path.display()),
                )
            })?;
            let reopen = writer.handle();
            let (non_blocking, guard) = tracing_appender::non_blocking(writer);
            let fmt_layer = tracing_subscriber::fmt::layer()
                .with_writer(non_blocking)
                .with_target(true)
                .with_thread_names(true)
                .with_file(false)
                .with_line_number(false)
                .with_ansi(false);
            tracing_subscriber::registry()
                .with(filter_layer)
                .with(fmt_layer)
                .try_init()
                .map_err(|e| already_installed_err(&e))?;
            Ok((
                ObservabilityHandle {
                    level_reload: Some(level_reload),
                    file_reopen: Some(reopen),
                    file_path: Some(path.clone()),
                },
                ObservabilityGuard {
                    file_guard: Some(guard),
                },
            ))
        }
    }
}

#[must_use]
const fn level_directive(level: LogLevel) -> &'static str {
    match level {
        LogLevel::Trace => "trace",
        LogLevel::Debug => "debug",
        LogLevel::Info => "info",
        LogLevel::Warn => "warn",
        LogLevel::Error => "error",
    }
}

/// Map `try_init`'s opaque error to `AlreadyExists` so `App::run` can
/// distinguish "subscriber slot taken" (a programming bug or test-suite
/// state leak) from genuine I/O failures.
fn already_installed_err(e: &tracing_subscriber::util::TryInitError) -> io::Error {
    io::Error::new(
        io::ErrorKind::AlreadyExists,
        format!("global tracing subscriber already installed: {e}"),
    )
}

/// Reopenable file writer for the `file` destination.
///
/// The `Arc<Mutex<File>>` lives behind both [`ReopenableFile`] (which
/// implements [`Write`] and is consumed by `tracing-appender`) and a
/// detached [`ReopenHandle`] held by [`ObservabilityHandle`]. Reopening
/// swaps the inner `File`; the appender's worker thread keeps locking
/// and writing, oblivious. Mutex contention is negligible because the
/// worker thread is the sole writer and reopens happen at SIGHUP cadence.
struct ReopenableFile {
    inner: Arc<Mutex<std::fs::File>>,
}

impl ReopenableFile {
    fn open(path: &Path) -> io::Result<Self> {
        let file = OpenOptions::new().create(true).append(true).open(path)?;
        Ok(Self {
            inner: Arc::new(Mutex::new(file)),
        })
    }

    fn handle(&self) -> ReopenHandle {
        ReopenHandle {
            inner: Arc::clone(&self.inner),
        }
    }
}

impl Write for ReopenableFile {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // A poisoned mutex means a previous holder panicked mid-write;
        // we can't claim anything about the underlying File state, so we
        // surface the error and let `tracing-appender`'s NonBlocking
        // writer count it as a drop.
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("reopenable-file mutex poisoned"))?;
        guard.write(buf)
    }
    fn flush(&mut self) -> io::Result<()> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|_| io::Error::other("reopenable-file mutex poisoned"))?;
        guard.flush()
    }
}

struct ReopenHandle {
    inner: Arc<Mutex<std::fs::File>>,
}

impl ReopenHandle {
    fn reopen(&self, path: &Path) -> io::Result<()> {
        let new_file = OpenOptions::new().create(true).append(true).open(path)?;
        // Reopening replaces the inner File wholesale; we don't read any
        // mid-panic state from it, so a poisoned mutex is recoverable.
        // `PoisonError::into_inner` returns the guard but leaves the
        // Mutex flagged poisoned — `Mutex::clear_poison` (stable since
        // 1.77) actually resets the flag so subsequent `lock()` calls
        // succeed. Order matters: clear *after* the swap, since
        // `clear_poison` is `&Mutex` and the guard holds an exclusive
        // borrow; we drop the guard first, then clear.
        let was_poisoned = self.inner.is_poisoned();
        let mut guard = self.inner.lock().unwrap_or_else(|e| {
            tracing::warn!("reopenable-file mutex was poisoned; recovering on reopen");
            e.into_inner()
        });
        *guard = new_file;
        drop(guard);
        if was_poisoned {
            self.inner.clear_poison();
        }
        Ok(())
    }
}

/// Type alias for `LogConfig` reload outcome — the driver tracks
/// what changed in the new config to decide which hooks to fire.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum LogReloadKind {
    /// Level changed; everything else identical.
    LevelOnly,
    /// Destination or path changed; v1 cannot hot-reload these — the
    /// driver logs an `error!` instructing the operator to restart.
    DestinationChanged,
    /// No change.
    Unchanged,
}

impl LogReloadKind {
    /// Compare two [`LogConfig`]s. The comparison is total: any change
    /// to `destination` or `path` returns [`Self::DestinationChanged`]
    /// even when `level` also moved.
    #[must_use]
    pub fn diff(old: &LogConfig, new: &LogConfig) -> Self {
        if old.destination != new.destination || old.path != new.path {
            Self::DestinationChanged
        } else if old.level != new.level {
            Self::LevelOnly
        } else {
            Self::Unchanged
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn reopenable_file_round_trips_writes() {
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("specter.log");
        let mut w = ReopenableFile::open(&path).unwrap();
        w.write_all(b"first\n").unwrap();
        w.flush().unwrap();
        drop(w);
        let body = std::fs::read_to_string(&path).unwrap();
        assert_eq!(body, "first\n");
    }

    #[test]
    fn reopen_swaps_to_fresh_inode() {
        // Simulate `mv specter.log specter.log.1 && touch specter.log`
        // (logrotate "copy then rename" mode). Pre-reopen writes go to
        // the rotated file; post-reopen writes go to the new file.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("specter.log");
        let mut w = ReopenableFile::open(&path).unwrap();
        let h = w.handle();
        w.write_all(b"before\n").unwrap();
        w.flush().unwrap();

        // Simulate a rotator: rename out from under us.
        let rotated = tmp.path().join("specter.log.1");
        std::fs::rename(&path, &rotated).unwrap();
        // The bin opens a fresh empty file at `path` — that's what
        // logrotate's `create` directive does. We mirror it by opening
        // via `reopen`.
        h.reopen(&path).unwrap();

        w.write_all(b"after\n").unwrap();
        w.flush().unwrap();
        drop(w);

        assert_eq!(std::fs::read_to_string(&rotated).unwrap(), "before\n");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "after\n");
    }

    #[test]
    fn reopen_recovers_from_poisoned_mutex() {
        // A panicking write would normally leave the mutex poisoned and
        // permanently break logrotate. Verify reopen recovers by clearing
        // the poison and swapping in a fresh File.
        let tmp = TempDir::new().unwrap();
        let path = tmp.path().join("specter.log");
        let w = ReopenableFile::open(&path).unwrap();
        let h = w.handle();
        let inner = Arc::clone(&w.inner);
        // Poison the mutex via a panic inside a held guard. Suppress the
        // child thread's panic message so the test output stays focused
        // on the reopen-recovery assertion.
        let prior_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let _ = std::thread::spawn(move || {
            let _guard = inner.lock().unwrap();
            panic!("intentional poison for test");
        })
        .join();
        std::panic::set_hook(prior_hook);

        assert!(w.inner.is_poisoned(), "mutex should be poisoned");
        // Reopen recovers — both the data (new File) and the poison flag.
        h.reopen(&path).expect("reopen recovers from poison");
        assert!(
            !w.inner.is_poisoned(),
            "reopen must clear the poison flag, not just swap the inner",
        );
        // Subsequent locks succeed without unwrap_or_else.
        let _guard = w.inner.lock().expect("subsequent lock succeeds");
    }

    #[test]
    fn open_failure_error_includes_path() {
        // ENOENT on the parent directory should surface a message that
        // names the offending log path, not just "No such file or
        // directory" — operators need to see which file failed.
        let cfg = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::File,
            path: Some(PathBuf::from("/nonexistent-parent-x9z/specter.log")),
        };
        let err = init(&cfg).expect_err("nonexistent parent should fail");
        let msg = err.to_string();
        assert!(
            msg.contains("/nonexistent-parent-x9z/specter.log"),
            "error message must contain the path; got: {msg}",
        );
    }

    #[test]
    fn noop_handle_set_level_succeeds() {
        let h = ObservabilityHandle::noop();
        assert!(h.set_level(LogLevel::Debug).is_ok());
    }

    #[test]
    fn noop_handle_reopen_file_succeeds() {
        let h = ObservabilityHandle::noop();
        assert!(h.reopen_file().is_ok());
    }

    #[test]
    fn noop_handle_has_no_file_path() {
        let h = ObservabilityHandle::noop();
        assert!(h.file_path().is_none());
    }

    #[test]
    fn log_reload_kind_unchanged_when_identical() {
        let a = LogConfig::default();
        let b = LogConfig::default();
        assert_eq!(LogReloadKind::diff(&a, &b), LogReloadKind::Unchanged);
    }

    #[test]
    fn log_reload_kind_level_only_when_only_level_differs() {
        let a = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::Stderr,
            path: None,
        };
        let b = LogConfig {
            level: LogLevel::Debug,
            destination: LogDestination::Stderr,
            path: None,
        };
        assert_eq!(LogReloadKind::diff(&a, &b), LogReloadKind::LevelOnly);
    }

    #[test]
    fn log_reload_kind_destination_changed_when_destination_changes() {
        let a = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::Stderr,
            path: None,
        };
        let b = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::File,
            path: Some(PathBuf::from("/var/log/x.log")),
        };
        assert_eq!(
            LogReloadKind::diff(&a, &b),
            LogReloadKind::DestinationChanged,
        );
    }

    #[test]
    fn log_reload_kind_destination_changed_when_path_changes() {
        let a = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::File,
            path: Some(PathBuf::from("/var/log/a.log")),
        };
        let b = LogConfig {
            level: LogLevel::Info,
            destination: LogDestination::File,
            path: Some(PathBuf::from("/var/log/b.log")),
        };
        assert_eq!(
            LogReloadKind::diff(&a, &b),
            LogReloadKind::DestinationChanged,
        );
    }
}
