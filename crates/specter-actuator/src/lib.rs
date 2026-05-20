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

#[cfg(unix)]
mod env;
#[cfg(unix)]
mod lifecycle;
#[cfg(unix)]
mod os;
#[cfg(unix)]
mod permits;
#[cfg(unix)]
mod pipe;
#[cfg(unix)]
mod pool;
#[cfg(unix)]
mod resolve;
#[cfg(unix)]
mod spawner;
#[cfg(unix)]
mod timer;
#[cfg(unix)]
mod tmp;

#[cfg(all(unix, feature = "testkit"))]
pub mod testkit;

#[cfg(unix)]
pub use os::OsSpawner;
#[cfg(unix)]
pub use permits::{Permit, Permits};
#[cfg(unix)]
pub use pool::{DEFAULT_CONCURRENCY, Reaped, SubprocessActuator};
#[cfg(unix)]
pub use spawner::{ChildSignaler, ChildWaiter, EnvVar, SpawnHandles, Spawner};

/// Construct the platform's default spawner as a `Box<dyn Spawner>`.
///
/// `Box<dyn>` matches the existing `&dyn Spawner` calling convention in
/// the actuator's state machine — one vtable hop per spawn, on a rare
/// (op/sec scale) hot path. A future non-Unix backend slots in by
/// gating `OsSpawner` on the relevant `cfg`.
#[cfg(unix)]
#[must_use]
pub fn default_spawner() -> Box<dyn Spawner> {
    Box::new(OsSpawner::new())
}
