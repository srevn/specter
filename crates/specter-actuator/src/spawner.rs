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
use std::sync::Arc;

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

    /// Spawn N stages wired stdout→stdin. The producer-side of stage
    /// K writes to a pipe whose read-end is the consumer-side stdin of
    /// stage K+1; the kernel handles the SIGPIPE chain when an
    /// upstream stage exits.
    ///
    /// Returns paired handles:
    /// - `waiter` — single aggregating waiter that drains every stage
    ///   sequentially and applies pipefail-on semantics (any non-zero
    ///   stage exit ⇒ aggregated `Failed`, carrying the last non-zero
    ///   exit in spawn order and the first observed signal).
    /// - `combined_signaler` — fans SIGTERM/SIGKILL out to every stage.
    ///   Used by the controller's shutdown path.
    /// - `stage_signalers` — parallel-indexed with the input stage
    ///   slice; the controller hands each one to its per-stage timer
    ///   thread (if the stage's [`specter_core::ExecAction::timeout`]
    ///   is set).
    ///
    /// On partial-spawn failure (stage K's spawn raises
    /// [`io::Error`]), implementations must roll back: SIGKILL +
    /// `reap_blocking` every prior stage and close every pipe fd in
    /// the parent before returning the error. Returning leaves no
    /// zombies and no orphan pipe fds.
    ///
    /// `capture_output` controls the **last** stage's stdout: `true` ⇒
    /// inherit Specter's stdout; `false` ⇒ route to `/dev/null`.
    /// Earlier stages' stdouts are always plumbed to the next stage's
    /// stdin. Every stage's stderr follows the `capture_output`
    /// policy (inherit vs `/dev/null`). Stage 0's stdin is always
    /// `/dev/null` (a pipe never reads from the parent's tty).
    fn spawn_pipe(
        &self,
        stages: &[StageSpec<'_>],
        cwd: &Path,
        capture_output: bool,
    ) -> io::Result<PipeSpawnHandles>;
}

/// One stage of a pipe spawn. Borrow shape mirrors
/// [`Spawner::spawn`]: argv as `&[String]` (the resolver's owning
/// slice), env as `&[EnvVar<'_>]` (resolver borrows from the Effect /
/// diff-tmp path / time string).
#[derive(Debug)]
pub struct StageSpec<'a> {
    pub argv: &'a [String],
    pub env: &'a [EnvVar<'a>],
}

/// Paired handles for a freshly-spawned pipe of N stages.
///
/// See [`Spawner::spawn_pipe`] for the contract. The `combined_signaler`
/// is `Arc<dyn>` because it's shared between the controller's shutdown
/// path and (in the aggregating waiter) the SIGTERM-cascade-on-first-
/// failure path; `stage_signalers` are `Arc<dyn>` so the per-stage
/// timer threads can co-own with the waiter without ceremony.
pub struct PipeSpawnHandles {
    /// Pid of the *last* stage — what an operator inspecting the
    /// pipe via `ps` would call "the pid of this pipe". The actuator
    /// surfaces this via [`crate::pool::state::RunningJob::pid`] and
    /// uses it only for log lines; the per-stage signalers carry
    /// their own pids internally for syscall routing.
    pub last_pid: u32,
    /// Aggregating waiter: drains every stage sequentially and
    /// applies pipefail-on semantics. Consumed once via
    /// `Box<dyn ChildWaiter>::wait`.
    pub waiter: Box<dyn ChildWaiter>,
    /// Shutdown signaler: fans SIGTERM/SIGKILL out to every stage.
    pub combined_signaler: Arc<dyn ChildSignaler>,
    /// Per-stage signalers, parallel-indexed with the input stage
    /// slice. The controller hands each one to its per-stage timer
    /// thread; not all stages need a timer (only those whose
    /// `ExecAction.timeout` is set).
    pub stage_signalers: Box<[Arc<dyn ChildSignaler>]>,
}

impl std::fmt::Debug for PipeSpawnHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PipeSpawnHandles")
            .field("last_pid", &self.last_pid)
            .field("stages", &self.stage_signalers.len())
            .finish_non_exhaustive()
    }
}

/// Paired handles for a freshly-spawned child.
///
/// The `waiter` is moved to the wait thread (consumed via `Box<Self>`
/// at wait-call time); the `signaler` stays on the controller, used at
/// shutdown to deliver SIGTERM and (after the grace window) SIGKILL.
/// Production impls share an `Arc<AtomicBool>` between the two so
/// post-reap signals are no-ops (closes the PID-reuse race).
///
/// `signaler` is `Arc<dyn>` because the controller installs the signaler
/// into the per-job `RunningJob::signaler` slot and *also* clones it
/// into the per-step timer thread (when [`specter_core::ExecAction`]'s
/// `timeout` is set) — both paths need to outlive each other
/// independently. The pipe path's `PipeSpawnHandles::stage_signalers`
/// are likewise `Arc`, so the spawn surface speaks a single
/// signaler-ownership shape.
pub struct SpawnHandles {
    pub pid: u32,
    pub waiter: Box<dyn ChildWaiter>,
    pub signaler: Arc<dyn ChildSignaler>,
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
    /// `Failed(Termination::Internal)`.
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome>;
}

/// Held by the controller; consulted on shutdown and on the
/// wait-thread-spawn-failure recovery path.
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
    /// Synchronously reap the child via blocking `waitpid(2)`. Used as
    /// the recovery path when the paired [`ChildWaiter`] cannot be
    /// driven (i.e. the wait thread spawn failed after fork+exec, so
    /// the closure that owned the waiter was dropped without ever
    /// calling `wait`). Without this, an orphan child that has been
    /// SIGKILLed would linger as a zombie until process exit.
    ///
    /// **Caller invariant**: no other party is `waitpid`-ing on this
    /// child. The production caller is the wait-thread-spawn-failure
    /// branch where the wait thread does not exist, so this trivially
    /// holds. Returns `Ok(())` once the kernel has reaped the child;
    /// `ECHILD` (child already gone, e.g. external reap or never
    /// existed) is collapsed to `Ok(())` so the recovery path is
    /// idempotent. `EINTR` is retried internally.
    ///
    /// Production impls set the shared `dead` flag on success so any
    /// later `signal_term` / `signal_kill` against the (reusable) pid
    /// short-circuits at the protocol layer — same guarantee the
    /// paired waiter provides on the normal path.
    fn reap_blocking(&self) -> io::Result<()>;
    /// `true` once the paired waiter has reported completion (or
    /// [`Self::reap_blocking`] has run). Lets co-owning threads short-
    /// circuit on natural completion — used by the per-step timer
    /// thread to skip SIGTERM/SIGKILL when a child finishes within its
    /// deadline. Mirrors the same `dead` flag the existing
    /// short-circuits in `signal_term` / `signal_kill` consult; exposed
    /// as its own method so callers don't have to issue a no-op signal
    /// to probe the state.
    fn is_dead(&self) -> bool;
}
