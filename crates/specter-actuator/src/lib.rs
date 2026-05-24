//! `specter-actuator` — subprocess pool, coalescing, env vars.
//!
//! The actuator's surface is intentionally narrow: a single
//! [`SubprocessActuator`] consumes [`specter_core::Effect`]s over a channel,
//! coalesces by [`specter_core::DedupKey`] (`Latest`-only in v1),
//! gates spawns on a global semaphore (default `2 × num_cpus`) and
//! a per-Sub semaphore (hardcoded `1`), spawns children via
//! `std::process::Command`, materializes the diff tmp file when the
//! [`Effect`] carries a [`Diff`], reaps children in per-process wait
//! threads, and ships effect completions back to the engine through an
//! [`EffectCompleteSender`] wired in at constructor time.
//!
//! Failure policy is `Ignore`-only in v1: non-zero exit, signal,
//! or spawn failure all surface as [`EffectOutcome::Failed`]; the engine
//! refuses to re-baseline on Failed. Shutdown sequences SIGTERM → 5s
//! grace → SIGKILL.
//!
//! Process I/O is abstracted behind three small traits — [`Spawner`],
//! [`ChildWaiter`], [`ChildSignaler`] — so coalescing / concurrency /
//! shutdown logic is testable without forking real children. The
//! engine-facing completion sink is similarly behind a trait
//! ([`EffectCompleteSender`]) so the actuator never names the engine's
//! `Input` vocabulary. Production impls are [`OsSpawner`] and the
//! bin-side `DriverEffectSender`; the `testkit` Cargo feature exposes
//! `testkit::MockSpawner`.
//!
//! [`Diff`]: specter_core::Diff
//! [`Effect`]: specter_core::Effect
//! [`EffectOutcome::Failed`]: specter_core::EffectOutcome::Failed

// Actuator needs `unsafe` for process control on Unix; `warn` is the
// looser-than-workspace setting, with per-call-site allows at FFI sites.
#![warn(unsafe_code)]

// The actuator's production surface (fork+exec, pipe(2), waitpid,
// signal delivery) is unix-only; every internal module would otherwise
// carry `#[cfg(unix)]`. A single crate-root gate is the honest
// expression of the target constraint and removes drift: a new module
// added later cannot silently lose its `#[cfg(unix)]` marker, because
// the crate fails to compile on non-unix outright. The `bin` already
// fails to link on non-unix (it calls `default_spawner`), so this is a
// transparent failure shift — not a new constraint.
#[cfg(not(unix))]
compile_error!("specter-actuator requires a unix target (linux / macOS / freebsd)");

mod env;
mod lifecycle;
mod os;
mod permits;
mod pipe;
mod pool;
mod resolve;
mod spawner;
mod timer;
mod tmp;

#[cfg(feature = "testkit")]
pub mod testkit;

pub use os::OsSpawner;
pub use permits::{Permit, Permits};
pub use pool::{DEFAULT_CONCURRENCY, Reaped, SubprocessActuator};
pub use spawner::{ChildSignaler, ChildWaiter, EnvVar, SpawnHandles, Spawner};

use specter_core::{DedupKey, EffectOutcome, SubId};

/// Sink for effect completions produced by the actuator.
///
/// The actuator crate owns *what* it delivers (a `(sub, key, outcome)`
/// triple) — the engine-facing `Input::EffectComplete` envelope is the
/// concern of whoever wires the actuator into a driver. This trait is
/// the seam: implementors translate the actuator's completion
/// vocabulary into whatever transport the bin holds.
///
/// # Threading
///
/// `Send + Sync + 'static` so a future refactor can share a single
/// implementor across the controller and any auxiliary thread (per-step
/// timer, panic-recovery worker) without re-cloning the underlying
/// transport. The controller thread is the sole caller today; the
/// bound only widens the option set.
///
/// # Semantics
///
/// Fire-and-forget. A successful [`send`](Self::send) leaves no further
/// obligation on the caller; an [`Err`](SendError) means the consumer
/// is gone (the engine driver dropped its receiver). The actuator's
/// terminal arms swallow the error — its own [`SubprocessActuator::run`]
/// loop will observe the engine teardown via one of its other exit
/// channels (`effects_rx` disconnect, `shutdown_rx`,
/// `hard_shutdown_rx`) and break out shortly after. The trait does not
/// carry the rejected `(sub, key, outcome)` back on error — completions
/// do not retry once the engine is gone.
pub trait EffectCompleteSender: Send + Sync + 'static {
    /// Deliver one effect completion. Returns `Ok(())` on enqueue;
    /// `Err(SendError::Disconnected)` if the engine-side consumer is
    /// gone.
    fn send(&self, sub: SubId, key: DedupKey, result: EffectOutcome) -> Result<(), SendError>;
}

/// Sender-side error vocabulary for [`EffectCompleteSender::send`].
///
/// One variant today; reserved as an `enum` rather than `()` so future
/// transports (bounded backpressure, batch submit) can extend the
/// vocabulary without churning every actuator emit site.
#[derive(Debug)]
pub enum SendError {
    /// The consumer dropped its receiver. No further `send` will
    /// succeed on this sender; the caller (the controller) will exit
    /// its loop via one of its other shutdown channels shortly after.
    Disconnected,
}

impl std::fmt::Display for SendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Disconnected => f.write_str("effect-complete consumer disconnected"),
        }
    }
}

impl std::error::Error for SendError {}

/// Construct the platform's default spawner as a `Box<dyn Spawner>`.
///
/// `Box<dyn>` matches the existing `&dyn Spawner` calling convention in
/// the actuator's state machine — one vtable hop per spawn, on a rare
/// (op/sec scale) hot path. A future non-Unix backend slots in by
/// providing an alternative spawner constructor here.
#[must_use]
pub fn default_spawner() -> Box<dyn Spawner> {
    Box::new(OsSpawner::new())
}
