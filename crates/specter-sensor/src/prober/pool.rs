//! `WorkerProber` — N-thread probe pool with per-correlation
//! cancellation.
//!
//! # Lifecycle
//!
//! 1. [`WorkerProber::new(out, concurrency)`](WorkerProber::new) spawns
//!    `concurrency.max(1)` threads named `specter-prober-{i}`. Each
//!    holds clones of the queue receiver, the `engine_inbound` sender,
//!    and the expectation map.
//! 2. The bin maps each `ProbeOp::Probe` from the engine to
//!    [`Prober::submit`]; each `ProbeOp::Cancel` to
//!    [`Prober::cancel`].
//! 3. Workers loop `recv → check expectation → catch_unwind(run_probe) →
//!    cleanup → send`. Panics convert to `Failed(EIO)`; the worker
//!    survives.
//! 4. On bin shutdown, [`WorkerProber::shutdown`] drops the queue
//!    sender and joins every worker thread.
//!
//! # Cancellation discipline
//!
//! The `expected: Arc<Mutex<BTreeMap<ProfileId, ProbeCorrelation>>>`
//! map records the *latest* expected correlation per Profile.
//!
//! - `submit(req)`: insert `(req.profile, req.correlation)`, then
//!   channel-send. The lock-then-send order guarantees the worker that
//!   races to `recv()` already sees the expectation.
//! - `cancel(profile)`: remove the entry. Queued requests with the
//!   stale correlation get skipped at worker-side check time.
//! - `submit` again with a fresh correlation: overwrites the entry.
//!   The prior request's worker-side check fails on its own
//!   correlation; the new request runs.
//! - Worker post-run cleanup: remove iff still equal to *our*
//!   correlation. A racing `submit` that wrote a fresh entry between
//!   our `recv` and our cleanup must not be clobbered.

use crate::Prober;
use crate::prober::walk::{probe_dir, probe_file};
use crossbeam::channel::{Receiver, Sender};
use specter_core::{
    Input, ProbeCorrelation, ProbeKind, ProbeRequest, ProbeResponse, ProbeResult, ProfileId,
};
use std::collections::BTreeMap;
use std::io;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Default worker count. The bin's `--probe-concurrency` CLI flag
/// overrides; this constant is the fallback.
pub const DEFAULT_CONCURRENCY: usize = 4;

/// Per-Profile correlation expectation map. `Arc<Mutex<...>>` is
/// shared across the [`WorkerProber`] and every worker thread; the
/// `Mutex` body holds for ~10ns (`BTreeMap` lookup + insert/remove on a
/// short map), so contention is negligible at v1's expected probe
/// rates. Visible to sibling tests so they can drive `run_worker`
/// directly with a hand-seeded map.
pub(super) type ExpectedMap = Arc<Mutex<BTreeMap<ProfileId, ProbeCorrelation>>>;

/// Multi-threaded probe pool. See module rustdoc for the cancellation
/// contract and lifecycle.
pub struct WorkerProber {
    queue_tx: Sender<ProbeRequest>,
    workers: Vec<JoinHandle<()>>,
    expected: ExpectedMap,
}

impl std::fmt::Debug for WorkerProber {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WorkerProber")
            .field("workers", &self.workers.len())
            .finish_non_exhaustive()
    }
}

impl WorkerProber {
    /// Spawn the worker pool. `concurrency.max(1)` workers — a
    /// zero-worker pool would queue requests forever.
    ///
    /// On any worker spawn failure (typically `EAGAIN` from the
    /// process-wide thread limit), drops the queue sender so
    /// already-spawned workers exit on `Disconnected`, joins them, and
    /// returns the underlying `io::Error`.
    pub fn new(out: &Sender<Input>, concurrency: usize) -> io::Result<Self> {
        let concurrency = concurrency.max(1);
        let (queue_tx, queue_rx) = crossbeam::channel::unbounded::<ProbeRequest>();
        let expected: ExpectedMap = Arc::new(Mutex::new(BTreeMap::new()));

        let mut workers = Vec::with_capacity(concurrency);
        for i in 0..concurrency {
            let rx = queue_rx.clone();
            let out_clone = out.clone();
            let expected_clone = Arc::clone(&expected);
            let spawned = thread::Builder::new()
                .name(format!("specter-prober-{i}"))
                .spawn(move || run_worker(&rx, &out_clone, &expected_clone, run_probe));
            match spawned {
                Ok(h) => workers.push(h),
                Err(e) => {
                    drop(queue_tx);
                    for h in workers {
                        let _ = h.join();
                    }
                    return Err(e);
                }
            }
        }
        Ok(Self {
            queue_tx,
            workers,
            expected,
        })
    }

    /// Drop the queue sender (workers exit on `Disconnected` at next
    /// `recv`) and join every worker handle. Returns the
    /// per-worker `thread::Result<()>` for the bin to log — `Err` here
    /// means the worker thread itself panicked outside of `run_probe`'s
    /// catch-unwind, which is a v1 bug to investigate.
    pub fn shutdown(self) -> Vec<thread::Result<()>> {
        drop(self.queue_tx);
        self.workers.into_iter().map(JoinHandle::join).collect()
    }

    /// Test-only inspection of the expectation map size; used by
    /// sibling unit tests to assert post-run cleanup mechanics.
    #[cfg(test)]
    pub(super) fn expected_len(&self) -> usize {
        self.expected.lock().expect("poisoned").len()
    }
}

impl Prober for WorkerProber {
    fn submit(&self, req: ProbeRequest) {
        // Lock-then-send: the map insert commits before the channel
        // send, so the worker that races to `recv` cannot observe the
        // request without also observing the expectation. The lock is
        // released before the send; the BTreeMap insert is durable
        // across the lock release.
        {
            let mut e = self.expected.lock().expect("prober expected map poisoned");
            e.insert(req.profile, req.correlation);
        }
        if let Err(crossbeam::channel::SendError(req)) = self.queue_tx.send(req) {
            tracing::error!(
                profile = ?req.profile,
                correlation = ?req.correlation,
                "prober queue closed; submit dropped",
            );
        }
    }

    fn cancel(&self, profile: ProfileId) {
        self.expected
            .lock()
            .expect("prober expected map poisoned")
            .remove(&profile);
        tracing::trace!(?profile, "prober cancel");
    }
}

/// Production probe-runner: dispatch on `ProbeKind`. Pure-IO; no
/// awareness of the worker loop or the expectation map.
fn run_probe(req: &ProbeRequest) -> ProbeResult {
    match req.kind {
        ProbeKind::File => probe_file(&req.target_path),
        ProbeKind::Directory => probe_dir(
            &req.target_path,
            req.target_resource,
            &req.scan_config,
            req.captured_with,
            req.baseline_subtree.as_ref(),
            &req.force_walk,
            req.forced,
        ),
    }
}

/// The worker loop body, parameterized over the probe runner so unit
/// tests can inject panics or canned results without touching the
/// production [`run_probe`] path.
///
/// Production [`WorkerProber::new`] passes [`run_probe`] directly; tests
/// in `prober/tests.rs` pass closures that `panic!()`, simulate
/// concurrent expectation-map writes, etc. The closure is invoked from
/// inside `catch_unwind(AssertUnwindSafe(...))`, so a test-injected
/// panic is recovered exactly as a production panic would be.
pub(super) fn run_worker<F>(
    rx: &Receiver<ProbeRequest>,
    out: &Sender<Input>,
    expected: &ExpectedMap,
    probe: F,
) where
    F: Fn(&ProbeRequest) -> ProbeResult,
{
    while let Ok(req) = rx.recv() {
        // Pre-run cancel check: the request was queued at submit time;
        // a `cancel` since then (or a fresh `submit` with a new
        // correlation) means our `(profile, correlation)` no longer
        // matches the latest expectation. Skip the syscall *and* the
        // response — the engine has structurally exited
        // `BurstPhase::Verifying` on cancel-emit, so a missing response
        // is harmless.
        let still_wanted = expected
            .lock()
            .expect("prober expected map poisoned")
            .get(&req.profile)
            .copied()
            == Some(req.correlation);
        if !still_wanted {
            tracing::debug!(
                profile = ?req.profile,
                correlation = ?req.correlation,
                "probe cancelled before dispatch",
            );
            continue;
        }

        let result =
            std::panic::catch_unwind(AssertUnwindSafe(|| probe(&req))).unwrap_or_else(|_| {
                tracing::error!(
                    profile = ?req.profile,
                    correlation = ?req.correlation,
                    "prober worker panicked; emitting Failed(EIO)",
                );
                ProbeResult::Failed { errno: libc::EIO }
            });

        // Post-run cleanup: remove iff still ours. A racing fresh
        // `submit` may have written a new correlation between our recv
        // and now; clobbering would spuriously skip the new request.
        {
            let mut e = expected.lock().expect("prober expected map poisoned");
            if e.get(&req.profile).copied() == Some(req.correlation) {
                e.remove(&req.profile);
            }
        }

        let response = ProbeResponse {
            profile: req.profile,
            correlation: req.correlation,
            result,
        };
        if out.send(Input::ProbeResponse(response)).is_err() {
            tracing::warn!("prober out channel closed; worker exiting");
            return;
        }
    }
    // `Err(Disconnected)`: queue sender dropped (clean shutdown).
}
