//! `WorkerProber` — N-thread probe pool with per-correlation
//! cancellation.
//!
//! # Lifecycle
//!
//! 1. [`WorkerProber::new(out, concurrency)`](WorkerProber::new)
//!    spawns `concurrency.max(1)` threads named `sp-prober-{i}`. The
//!    prefix is abbreviated (not the full `specter-prober-`) so the
//!    name fits Linux's `TASK_COMM_LEN` (15 visible bytes + null) at
//!    every legal `--probe-concurrency` value — preserving the index
//!    in `/proc/<pid>/task/<tid>/comm`. Each worker holds clones of the
//!    queue receiver, the response sender, and the expectation map.
//! 2. The bin maps each `ProbeOp::Probe` from the engine to
//!    [`Prober::submit`]; each `ProbeOp::Cancel` to
//!    [`Prober::cancel`].
//! 3. Workers loop `recv → check expectation → catch_unwind(run_probe) →
//!    cleanup → send`. Panics convert to `Failed(EIO)`; the worker
//!    survives.
//! 4. On bin shutdown, [`WorkerProber::shutdown`] drops the queue
//!    sender and joins every worker thread.
//!
//! # Shutdown observability
//!
//! The bin drops the engine driver as part of its shutdown sequence
//! (see `App::run`). The drop releases the engine-side receiver, so
//! the next [`ProberResponseSender::send`] from a worker returns
//! [`SendError::Disconnected`]. The worker logs that exit at
//! `debug!` and unwinds its loop — the bin owns whatever shutdown
//! cause logging the operator needs, at the right severity, on its
//! own thread.
//!
//! # Cancellation discipline
//!
//! The `expected: Arc<Mutex<BTreeMap<ProbeOwner, ProbeCorrelation>>>`
//! map records the *latest* expected correlation per probe-channel
//! owner (Profile in v1; future owner kinds plug in via the
//! [`ProbeOwner`] enum).
//!
//! - `submit(req)`: insert `(req.owner(), req.correlation())`, then
//!   channel-send. The lock-then-send order guarantees the worker that
//!   races to `recv()` already sees the expectation.
//! - `cancel(owner)`: remove the entry. Queued requests with the
//!   stale correlation get skipped at worker-side check time.
//! - `submit` again with a fresh correlation: overwrites the entry.
//!   The prior request's worker-side check fails on its own
//!   correlation; the new request runs.
//! - Worker post-run cleanup: remove iff still equal to *our*
//!   correlation. A racing `submit` that wrote a fresh entry between
//!   our `recv` and our cleanup must not be clobbered.
//!
//! # Panic recovery
//!
//! Two layered primitives keep a worker thread alive across local
//! panics:
//!
//! - The inner `catch_unwind` in [`run_worker`] catches probe-side
//!   panics and emits `Failed { errno: EIO }`. The worker survives and
//!   resumes its `recv → check → run → cleanup → send` loop.
//! - Lock-acquisition panics (e.g., a panicking allocator during
//!   `BTreeMap::insert`) recover silently via [`lock_expected`]. The
//!   `BTreeMap` operations we run inside the lock are exception-safe
//!   under Rust's allocator-panic semantics, so the recovered map is
//!   structurally consistent and the surviving worker continues
//!   dispatching probes against it.
//!
//! There is no outer `catch_unwind` around the worker loop body: with
//! the two primitives above, the only remaining panic surface is
//! `out.send` (which returns `SendError`, never panics) and the
//! channel-disconnect path (the clean-shutdown signal).

use crate::prober::walk::{probe_anchor_file, probe_descent, probe_subtree};
use crate::{Prober, ProberResponseSender};
use crossbeam::channel::{Receiver, Sender};
use specter_core::{ProbeCorrelation, ProbeOutcome, ProbeOwner, ProbeRequest, ProbeResponse};
use std::collections::BTreeMap;
use std::io;
use std::panic::AssertUnwindSafe;
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};

/// Default worker count. The bin's `--probe-concurrency` CLI flag
/// overrides; this constant is the fallback.
pub const DEFAULT_CONCURRENCY: usize = 4;

/// Per-owner correlation expectation map. `Arc<Mutex<...>>` is shared
/// across the [`WorkerProber`] and every worker thread; the `Mutex`
/// body holds for ~10ns (`BTreeMap` lookup + insert/remove on a short
/// map), so contention is negligible at v1's expected probe rates.
/// Visible to sibling tests so they can drive `run_worker` directly
/// with a hand-seeded map.
pub(super) type ExpectedMap = Arc<Mutex<BTreeMap<ProbeOwner, ProbeCorrelation>>>;

/// Lock the expectation map, recovering from `Mutex` poisoning by
/// extracting the inner state via `PoisonError::into_inner`.
///
/// Every lock body in this module is `BTreeMap` get / insert / remove
/// on a small map. Those operations are exception-safe under Rust's
/// allocator-panic semantics — allocation happens *before* tree
/// mutation, so an allocator panic inside `insert` cannot leave the
/// recovered map structurally torn. A worker that panics while holding
/// the lock therefore corrupts no invariant we depend on, and silent
/// recovery is the right policy: surviving workers continue
/// dispatching probes against an unchanged-or-cleanly-updated
/// expectation map rather than panic-cascading the whole pool.
///
/// This is the single panic-recovery primitive in the prober. The
/// `catch_unwind` boundary in [`run_worker`] handles probe panics;
/// this helper handles lock-acquisition panics. Together they keep a
/// worker thread alive across any local panic — probe, allocator, or
/// any future allocation we add inside the lock.
pub(super) fn lock_expected(
    expected: &ExpectedMap,
) -> std::sync::MutexGuard<'_, BTreeMap<ProbeOwner, ProbeCorrelation>> {
    expected
        .lock()
        .unwrap_or_else(std::sync::PoisonError::into_inner)
}

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
    /// `out` is `Box<dyn>`-in / `Arc<dyn>`-internal: the constructor
    /// owns the sink, converts it into an `Arc` once, and then
    /// `Arc::clone`s per worker. The Box-shaped boundary makes the
    /// ownership transfer truthful (no by-value-but-only-cloned
    /// smell) while the trait-object form keeps the pool non-generic
    /// at the type level — symmetric with the `Arc<dyn Prober>`
    /// handle the bin clones onto its driver — and lets the
    /// underlying transport (a crossbeam `Sender<Input>` in
    /// production; a test wrapper in unit tests) stay invisible to
    /// the worker loop.
    ///
    /// On any worker spawn failure (typically `EAGAIN` from the
    /// process-wide thread limit), drops the queue sender so
    /// already-spawned workers exit on `Disconnected`, joins them, and
    /// returns the underlying `io::Error`.
    pub fn new(out: Box<dyn ProberResponseSender>, concurrency: usize) -> io::Result<Self> {
        let concurrency = concurrency.max(1);
        let (queue_tx, queue_rx) = crossbeam::channel::unbounded::<ProbeRequest>();
        let expected: ExpectedMap = Arc::new(Mutex::new(BTreeMap::new()));
        let out: Arc<dyn ProberResponseSender> = Arc::from(out);

        let mut workers = Vec::with_capacity(concurrency);
        for i in 0..concurrency {
            let rx = queue_rx.clone();
            let out_clone = Arc::clone(&out);
            let expected_clone = Arc::clone(&expected);
            let spawned = thread::Builder::new()
                .name(format!("sp-prober-{i}"))
                .spawn(move || {
                    run_worker(&rx, &*out_clone, &expected_clone, run_probe);
                });
            match spawned {
                Ok(h) => workers.push(h),
                Err(e) => {
                    // Partial-spawn cleanup: drop the queue Sender so
                    // any worker already in `recv` exits on
                    // `Disconnected`, then join each spawned handle.
                    // A panic here means a worker died before it ever
                    // entered its loop body — log it so the operator
                    // sees the real failure alongside the spawn error
                    // we're about to return.
                    drop(queue_tx);
                    for (worker, h) in workers.into_iter().enumerate() {
                        if let Err(panic) = h.join() {
                            tracing::error!(
                                worker,
                                ?panic,
                                "prober worker panicked during partial-spawn cleanup; \
                                 the original spawn error will still be returned",
                            );
                        }
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
    /// `recv`) and join every worker handle. Returns each worker's
    /// `(index, thread::Result<()>)` for the bin to log — `Err` here
    /// means the worker thread itself panicked outside of `run_probe`'s
    /// catch-unwind, which is a v1 bug to investigate.
    ///
    /// The index matches the spawn order (the thread is named
    /// `sp-prober-{i}` for the same `i`), so post-mortem logs can
    /// correlate a panicking handle back to its thread name without
    /// reaching for thread-local state.
    #[must_use]
    pub fn shutdown(self) -> Vec<(usize, thread::Result<()>)> {
        drop(self.queue_tx);
        self.workers
            .into_iter()
            .enumerate()
            .map(|(i, h)| (i, h.join()))
            .collect()
    }

    /// Test-only inspection of the expectation map size; used by
    /// sibling unit tests to assert post-run cleanup mechanics.
    #[cfg(test)]
    pub(super) fn expected_len(&self) -> usize {
        lock_expected(&self.expected).len()
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
            let mut e = lock_expected(&self.expected);
            e.insert(req.owner(), req.correlation());
        }
        if let Err(crossbeam::channel::SendError(dropped)) = self.queue_tx.send(req) {
            // Symmetric with the response-side `debug!` at the bottom of
            // `run_worker`: the queue closes only when every worker has
            // exited (the receivers are dropped), which under current
            // ownership only happens via `WorkerProber::shutdown` — i.e.
            // clean teardown. `debug!` matches that severity.
            tracing::debug!(
                owner = ?dropped.owner(),
                correlation = ?dropped.correlation(),
                "prober queue closed; submit dropped",
            );
        }
    }

    fn cancel(&self, owner: ProbeOwner) {
        lock_expected(&self.expected).remove(&owner);
    }
}

/// Production probe-runner: dispatch on the `ProbeRequest` variant.
/// Pure-IO; no awareness of the worker loop or the expectation map.
pub(super) fn run_probe(req: &ProbeRequest) -> ProbeOutcome {
    match req {
        ProbeRequest::AnchorFile { target_path, .. } => probe_anchor_file(target_path),
        ProbeRequest::Subtree {
            target_path,
            scan_config,
            captured_with,
            baseline_subtree,
            obligation,
            forced,
            ..
        } => probe_subtree(
            target_path,
            scan_config,
            *captured_with,
            baseline_subtree.as_ref(),
            obligation,
            *forced,
        ),
        ProbeRequest::Descent { target_path, .. } => probe_descent(target_path),
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
    out: &dyn ProberResponseSender,
    expected: &ExpectedMap,
    probe: F,
) where
    F: Fn(&ProbeRequest) -> ProbeOutcome,
{
    while let Ok(req) = rx.recv() {
        let owner = req.owner();
        let correlation = req.correlation();
        // Pre-run cancel check: the request was queued at submit time;
        // a `cancel` since then (or a fresh `submit` with a new
        // correlation) means our `(owner, correlation)` no longer
        // matches the latest expectation. Skip the syscall *and* the
        // response — the engine has structurally closed the per-owner
        // probe channel on cancel-emit, so a missing response is
        // harmless.
        let still_wanted = lock_expected(expected).get(&owner).copied() == Some(correlation);
        if !still_wanted {
            tracing::debug!(?owner, ?correlation, "probe cancelled before dispatch",);
            continue;
        }

        let outcome =
            std::panic::catch_unwind(AssertUnwindSafe(|| probe(&req))).unwrap_or_else(|_| {
                tracing::error!(
                    ?owner,
                    ?correlation,
                    "prober worker panicked; emitting Failed(EIO)",
                );
                ProbeOutcome::Failed { errno: libc::EIO }
            });

        // Post-run cleanup: remove iff still ours. A racing fresh
        // `submit` may have written a new correlation between our recv
        // and now; clobbering would spuriously skip the new request.
        {
            let mut e = lock_expected(expected);
            if e.get(&owner).copied() == Some(correlation) {
                e.remove(&owner);
            }
        }

        let response = ProbeResponse {
            owner,
            correlation,
            outcome,
        };
        if out.send(response).is_err() {
            tracing::debug!("prober response sink disconnected; worker exiting");
            return;
        }
    }
    // `Err(Disconnected)`: queue sender dropped (clean shutdown).
}
