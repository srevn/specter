//! Process-spawn / child-wait / child-signal traits.
//!
//! The split — three small traits instead of one — exists because
//! `std::process::Child::wait` consumes `&mut self`, and the wait thread
//! owns the `Child` for the duration of the wait, while the controller
//! thread may need to send signals to the same child during shutdown.
//! [`ChildWaiter`] is owned by the wait thread; [`ChildSignaler`] is
//! owned by the controller; both share an `Arc<AtomicBool>` (production
//! impl detail) so signals to a reaped child short-circuit instead of
//! racing PID-reuse.

use specter_core::EffectOutcome;
use std::borrow::Cow;
use std::io;
use std::path::Path;

/// One `(name, value)` env-var pair the spawner passes to the child.
///
/// `key` is `&'static str` because every env-var name the resolver
/// emits is a literal (`"SPECTER_*"`); allocating those at the trait
/// boundary would be pure waste. `value` is `Cow<'_, str>` so the
/// resolver can borrow from the [`specter_core::Effect`] (anchor path
/// lossy-rendered, `target_relative`, `sub_name`, etc.) when the data
/// is already there, and own only the strings it genuinely synthesises
/// (newline-joined diff lists, formatted timestamp, parent-dir lossy,
/// joined target path). The trait shape thereby matches the producer's
/// natural lifetimes instead of forcing a flatten-to-owned hop.
///
/// The lifetime parameter `'a` ties the borrow to the source data: in
/// production, the resolver returns a `Vec<EnvVar<'_>>` borrowing from
/// the `Effect` and the optional diff-tmp path, both of which outlive
/// the synchronous `Spawner::spawn` call.
#[derive(Clone, Debug)]
pub struct EnvVar<'a> {
    pub key: &'static str,
    pub value: Cow<'a, str>,
}

/// Process spawner — the single I/O seam between the actuator's
/// (otherwise pure) state machine and the OS.
///
/// Production = [`super::OsSpawner`] (`std::process::Command`); tests =
/// `testkit::MockSpawner` (controllable). `Send + Sync` so the bin can
/// hold one `Arc<dyn Spawner>` and share across the controller thread
/// + any test orchestration.
pub trait Spawner: Send + Sync {
    /// Spawn a child for the given argv + env + cwd + stdio policy.
    /// Returns paired handles: the `waiter` (consumed by the wait thread)
    /// and the `signaler` (held by the controller, used for SIGTERM/SIGKILL).
    ///
    /// `capture_output` is the per-Effect stdio policy mirrored from the
    /// owning Sub's `log_output`. `false` (the default) routes
    /// stdout/stderr to `/dev/null`; `true` inherits Specter's own
    /// stdio so the parent's supervisor (systemd journal, launchd
    /// `StandardOutPath`, FreeBSD `daemon -o`) captures the bytes.
    /// Stdin is unconditionally `/dev/null` regardless — a watch-action
    /// command never reads from the parent's tty.
    fn spawn(
        &self,
        argv: &[String],
        env: &[EnvVar<'_>],
        cwd: &Path,
        capture_output: bool,
    ) -> io::Result<SpawnHandles>;
}

/// Paired handles for a freshly-spawned child.
///
/// The `waiter` is moved to the wait thread (consumed via `Box<Self>`
/// at wait-call time); the `signaler` stays on the controller, used at
/// shutdown to deliver SIGTERM and (after the grace window) SIGKILL.
/// Production impls share an `Arc<AtomicBool>` between the two so
/// post-reap signals are no-ops (closes the PID-reuse race).
pub struct SpawnHandles {
    pub pid: u32,
    pub waiter: Box<dyn ChildWaiter>,
    pub signaler: Box<dyn ChildSignaler>,
}

impl std::fmt::Debug for SpawnHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnHandles")
            .field("pid", &self.pid)
            .finish_non_exhaustive()
    }
}

/// Owned by the wait thread; consumed via `Box<Self>` at wait time.
/// Single-use.
pub trait ChildWaiter: Send {
    /// Block until the child exits; return the outcome. `io::Error` on
    /// system-level wait failure (rare; e.g. ECHILD from external
    /// reaping); the wait thread treats this as
    /// `Failed { exit_code: None, signal: None }`.
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome>;
}

/// Held by the controller; consulted on shutdown.
///
/// `Send + Sync` lets the controller move signaler boxes between
/// thread-local maps without ceremony. Implementations short-circuit
/// when their paired waiter has already returned (the production impl
/// uses an `Arc<AtomicBool>`); ESRCH from the actual `kill(2)` is
/// collapsed to `Ok(())` as a defense-in-depth layer.
///
/// Neither layer fully closes the PID-reuse race: between `child.wait()`
/// returning (kernel reaps the zombie; pid eligible for reuse) and the
/// waiter setting the shared flag, a brief window exists where a signal
/// could land on an unrelated process at the same pid. ESRCH-collapse
/// does not help here — a reused pid points at a real process and the
/// syscall returns success. The window is small but live on systems
/// with high pid pressure; v2 may switch to process descriptors
/// (pidfd / pdfork).
pub trait ChildSignaler: Send + Sync {
    /// Send SIGTERM. ESRCH (child already gone) is collapsed to `Ok(())`.
    /// Short-circuits to `Ok(())` if the paired waiter has already
    /// reported completion.
    fn signal_term(&self) -> io::Result<()>;
    /// Send SIGKILL. Same ESRCH-collapse + short-circuit as
    /// [`Self::signal_term`].
    fn signal_kill(&self) -> io::Result<()>;
}
