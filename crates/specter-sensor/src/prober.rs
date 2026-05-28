//! Multi-threaded probe worker pool.
//!
//! `WorkerProber` owns N worker threads consuming `ProbeRequest`s from a
//! single crossbeam unbounded channel. Each worker performs a recursive walk
//! against the filesystem and ships the resulting `ProbeResponse` back
//! to the engine's `engine_inbound` channel as `Input::ProbeResponse(...)`.
//!
//! Layered:
//! - [`pool`]: `WorkerProber` lifecycle, the cancellation expectation
//!   map, and the shared `run_worker` loop.
//! - [`walk`]: pure-IO `probe_anchor_file`, `probe_subtree`,
//!   `probe_descent` walkers — one per [`specter_core::ProbeRequest`]
//!   variant.
//!
//! # Cancellation
//!
//! Best-effort. The pool tracks the *latest* per-Profile
//! correlation in an `Arc<Mutex<BTreeMap<ProfileId, ProbeCorrelation>>>`.
//! `submit` writes; `cancel` removes; the worker checks before running
//! a popped request. A stale correlation (a `cancel` raced ahead of the
//! worker's `recv`, or a fresh `submit` overwrote with a new
//! correlation) causes the worker to skip the syscall and discard the
//! request silently. In-flight probes are *not* interrupted — the
//! engine drops late responses via stale-correlation discipline.
//!
//! # Panic recovery
//!
//! Each `run_probe` call is wrapped in `catch_unwind`. A panic converts
//! to [`specter_core::ProbeOutcome::Failed`] carrying
//! [`specter_core::ProbeFailure::Anchor`] with `errno = EIO` and the
//! worker continues on the next request. Aborts (`panic = "abort"`
//! profile) bypass this and kill the worker — v1 uses the workspace's
//! default unwind profile.

mod pool;
mod walk;

#[cfg(test)]
mod tests;

pub use pool::{WorkerProber, default_concurrency};
