//! `specter-actuator` — subprocess pool, coalescing, env vars.
//!
//! The actuator's surface is intentionally narrow: a single
//! [`SubprocessActuator`] consumes [`specter_core::Effect`]s over a channel,
//! coalesces by [`specter_core::DedupKey`] (`Latest`-only in v1),
//! gates spawns on a global semaphore (default `2 × num_cpus`) and
//! a per-Sub semaphore (hardcoded `1`), spawns children via
//! `std::process::Command`, materializes the diff tmp file when the
//! [`Effect`] carries a [`Diff`], reaps children in per-process wait
//! threads, and ships [`Input::EffectComplete`] back to the engine.
//!
//! Failure policy is `Ignore`-only in v1: non-zero exit, signal,
//! or spawn failure all surface as [`EffectOutcome::Failed`]; the engine
//! refuses to re-baseline on Failed. Shutdown sequences SIGTERM → 5s
//! grace → SIGKILL.
//!
//! Process I/O is abstracted behind three small traits — [`Spawner`],
//! [`ChildWaiter`], [`ChildSignaler`] — so coalescing / concurrency /
//! shutdown logic is testable without forking real children. Production
//! impl is [`OsSpawner`]; the `testkit` Cargo feature exposes
//! `testkit::MockSpawner`.
//!
//! [`Diff`]: specter_core::Diff
//! [`Effect`]: specter_core::Effect
//! [`EffectOutcome::Failed`]: specter_core::EffectOutcome::Failed
//! [`Input::EffectComplete`]: specter_core::Input::EffectComplete

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
