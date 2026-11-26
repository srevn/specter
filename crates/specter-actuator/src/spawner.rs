//! Process-spawn / child-wait / child-signal traits.
//!
//! The split — three small traits instead of one — exists because `std::process::Child::wait`
//! consumes `&mut self`, and the wait thread owns the `Child` for the duration of the wait, while
//! the controller thread may need to send signals to the same child during shutdown. [`ChildWaiter`]
//! is owned by the wait thread; [`ChildSignaler`] is owned by the controller; both share a one-shot
//! lifecycle flag (the `crate::lifecycle::DeadFlag` protocol primitive in production *and* in the
//! test mock) so signals to a reaped child short-circuit at the protocol layer instead of racing
//! PID-reuse.

use specter_core::EffectOutcome;
use std::borrow::Cow;
use std::io;
use std::path::Path;
use std::sync::Arc;

/// One `(name, value)` env-var pair the spawner passes to the child.
///
/// `key` is `&'static str` because every env-var name the resolver emits is a literal
/// (`"SPECTER_*"`); allocating those at the trait boundary would be pure waste. `value` is `Cow<'_,
/// str>` so the resolver can borrow from the [`specter_core::Effect`] (anchor path lossy-rendered,
/// `target_relative`, `sub_name`, etc.) when the data is already there, and own only the strings it
/// genuinely synthesises (newline-joined diff lists, formatted timestamp, parent-dir lossy, joined
/// target path). The trait shape thereby matches the producer's natural lifetimes instead of forcing
/// a flatten-to-owned hop.
///
/// The lifetime `'a` ties the borrow to the source data — the [`specter_core::Effect`] and the
/// optional diff-tmp path, both of which outlive the synchronous [`Spawner`] call.
#[derive(Clone, Debug)]
pub struct EnvVar<'a> {
    pub key: &'static str,
    pub value: Cow<'a, str>,
}

/// Process spawner — the single I/O seam between the actuator's (otherwise pure) state machine and
/// the OS.
///
/// It spawns a step's processes and wires a pipe's links, and nothing more: what the stages are to
/// the pool — each stage's timer, a pipe joined into one job ([`crate::pipe::join`]) — is decided
/// above it, the same for every implementation.
///
/// Production = [`super::OsSpawner`] (`std::process::Command`); tests = `testkit::MockSpawner`. The
/// actuator controller is single-threaded and is the sole caller — `Send` lets the bin's `Box<dyn
/// Spawner>` move into the actuator thread (see `specter-bin`'s `spawn_actuator_thread`); no caller
/// ever shares `&dyn Spawner` across threads, so `Sync` is not part of the contract.
pub trait Spawner: Send {
    /// Spawns one process per stage, each stage's stdout the next stage's stdin, and returns each
    /// stage's handles in stage order; an exec is one stage. Where stage K fails to spawn,
    /// implementations SIGKILL and reap every stage before K and close every link before returning
    /// the error: no zombie, no orphan descriptor.
    fn spawn(&self, stages: &[StageSpec<'_>], cwd: &Path) -> io::Result<Vec<SpawnHandles>>;
}

/// One stage's command as the resolver rendered it: the program, its arguments, and the env it adds
/// to the daemon's own.
#[derive(Debug)]
pub struct StageSpec<'a> {
    pub program: String,
    pub args: Vec<String>,
    pub env: Vec<EnvVar<'a>>,
}

/// A running child's pid, waiter and signaler — one stage's, as the spawner hands it back, or a whole
/// pipe's, which [`crate::pipe::join`] makes one child under its last stage's pid.
///
/// The `waiter` is moved to the wait thread (consumed via `Box<Self>` at wait-call time); the
/// `signaler` stays on the controller, used at shutdown to deliver SIGTERM and (after the grace
/// window) SIGKILL. The pair shares a one-shot lifecycle flag (production and mock both use the
/// `crate::lifecycle::DeadFlag` newtype) so post-reap signals are no-ops at the protocol layer —
/// closes the PID-reuse race independently of the kernel-level ESRCH-collapse fallback.
///
/// `signaler` is `Arc<dyn>` because its co-owners outlive each other independently: the job's
/// `RunningJob::signaler`, the wait thread's backstop, a timed stage's timer thread, and a joined
/// pipe's waiter and combined signaler.
pub struct SpawnHandles {
    pub pid: u32,
    pub waiter: Box<dyn ChildWaiter>,
    pub signaler: Arc<dyn ChildSignaler>,
}

impl std::fmt::Debug for SpawnHandles {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SpawnHandles").field("pid", &self.pid).finish_non_exhaustive()
    }
}

/// Owned by the wait thread; consumed via `Box<Self>` at wait time. Single-use.
pub trait ChildWaiter: Send {
    /// Block until the child exits; return the outcome. `io::Error` on system-level wait failure
    /// (rare; e.g. ECHILD from external reaping); the wait thread treats this as
    /// `Failed(Termination::Internal)`.
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome>;
}

/// Held by the controller; consulted on shutdown and on the wait-thread-spawn-failure recovery path.
///
/// `Send + Sync` lets the controller move signaler boxes between thread-local maps without ceremony.
/// Implementations short-circuit when their paired waiter has already returned (production and the
/// test mock both use the shared `crate::lifecycle::DeadFlag`); ESRCH from the actual `kill(2)` is
/// collapsed to `Ok(())` as a defense-in-depth layer.
///
/// The lifecycle ratchet — [`Self::is_dead`] (read) paired with [`Self::mark_dead`] (write) — is
/// the protocol-level surface. The `ChildWaiter` impl is the canonical early publisher (the
/// crate-private `OsChildWaiter::wait` marks dead before returning); wrapper sites that
/// `catch_unwind` a waiter call [`Self::mark_dead`] as a backstop in case the impl panicked before
/// reaching its own publish.
///
/// Neither layer fully closes the PID-reuse race: between `child.wait()` returning (kernel reaps
/// the zombie; pid eligible for reuse) and the waiter setting the shared flag, a brief window exists
/// where a signal could land on an unrelated process at the same pid. ESRCH-collapse does not help
/// here — a reused pid points at a real process and the syscall returns success. The window is small
/// but live on systems with high pid pressure; v2 may switch to process descriptors (pidfd / pdfork).
///
/// `OsChildWaiter::wait` lives in the crate-private `super::os`.
pub trait ChildSignaler: Send + Sync {
    /// Send SIGTERM. ESRCH (child already gone) is collapsed to `Ok(())`. Short-circuits to `Ok(())`
    /// if the paired waiter has already reported completion.
    fn signal_term(&self) -> io::Result<()>;
    /// Send SIGKILL. Same ESRCH-collapse + short-circuit as [`Self::signal_term`].
    fn signal_kill(&self) -> io::Result<()>;
    /// Synchronously reap the child via blocking `waitpid(2)`. Used as the recovery path when the
    /// paired [`ChildWaiter`] cannot be driven (i.e. the wait thread spawn failed after fork+exec,
    /// so the closure that owned the waiter was dropped without ever calling `wait`). Without this,
    /// an orphan child that has been SIGKILLed would linger as a zombie until process exit.
    ///
    /// **Caller invariant**: no other party is `waitpid`-ing on this child. The production caller
    /// is the wait-thread-spawn-failure branch where the wait thread does not exist, so this
    /// trivially holds. Returns `Ok(())` once the kernel has reaped the child; `ECHILD` (child
    /// already gone, e.g. external reap or never existed) is collapsed to `Ok(())` so the recovery
    /// path is idempotent. `EINTR` is retried internally.
    ///
    /// Production impls mark the shared dead-flag unconditionally after the underlying `waitpid`
    /// returns (success OR failure) so any later `signal_term` / `signal_kill` against the (reusable)
    /// pid short-circuits at the protocol layer — same guarantee the paired waiter provides on the
    /// normal path.
    fn reap_blocking(&self) -> io::Result<()>;
    /// `true` once the paired waiter has reported completion (or [`Self::reap_blocking`] /
    /// [`Self::mark_dead`] has run). Lets co-owning threads short-circuit on natural completion —
    /// used by the per-step timer thread to skip SIGTERM/SIGKILL when a child finishes within its
    /// deadline. Reads the same shared dead-flag the `signal_term` / `signal_kill` short-circuits
    /// consult; exposed as its own method so callers don't have to issue a no-op signal to probe
    /// the state.
    fn is_dead(&self) -> bool;
    /// Publish "paired child is reaped; signal syscalls would race PID reuse." Idempotent — the
    /// underlying flag is a one-shot ratchet.
    ///
    /// The protocol-level write that pairs with [`Self::is_dead`]. Wrapper sites that `catch_unwind`
    /// a [`ChildWaiter::wait`] invocation (the controller's `wait_loop`, and the per-stage closures
    /// inside the crate-private `PipeWaiter`) call this after the catch, so a panicking waiter impl
    /// that didn't reach its own write site still publishes — closing the PID-reuse window at the
    /// protocol layer. On the happy path the [`ChildWaiter::wait`] impl is the early publisher; this
    /// trait-level write is the additive backstop, idempotent against it.
    ///
    /// The crate-private `CombinedSignaler::mark_dead` fans the call out to every per-stage signaler,
    /// so a pipe-level wrapper closes the window for all stages with one call.
    fn mark_dead(&self);
}
