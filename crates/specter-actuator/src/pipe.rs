//! Shared pipe-spawn helpers used by both the production [`OsSpawner`]
//! and the [`MockSpawner`] testkit. The two types here are the
//! "aggregating" side of the pipe machinery — they coordinate
//! per-stage outcomes (waiter) and per-stage shutdown signals
//! (signaler) into a single `ChildWaiter` / `ChildSignaler` pair the
//! controller can reason about uniformly.
//!
//! Lives outside of `os.rs` so [`crate::testkit::MockSpawner`] can
//! construct pipe handles against its own per-stage mock
//! signalers/waiters without re-implementing the aggregation rules.
//! The shapes ARE the contract — `Spawner::spawn_pipe`'s pipefail-on
//! semantics and shutdown-fan-out behaviour both live in this module.
//!
//! [`OsSpawner`]: crate::OsSpawner
//! [`MockSpawner`]: crate::testkit::MockSpawner

use crate::spawner::{ChildSignaler, ChildWaiter};
use crossbeam::channel::{Receiver, unbounded};
use specter_core::{EffectOutcome, Termination};
use std::io;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::thread;

/// Combined signaler — fans SIGTERM/SIGKILL out to every stage.
///
/// Used as `PipeSpawnHandles.combined_signaler`: the controller's
/// shutdown path calls `signal_term` / `signal_kill` against it and
/// the call propagates to every stage. The aggregating waiter
/// ([`PipeWaiter`]) holds its own per-stage signaler references for
/// the SIGTERM-cascade-on-first-failure path; the combined signaler
/// is for callers that want to address the pipe as a whole.
///
/// Errors from individual stage signalers are aggregated: the first
/// non-`Ok` is propagated as the combined result; subsequent stages
/// still get the signal (best-effort). This matches the controller's
/// "log and continue" stance — a shutdown signal is best-effort by
/// nature, and one stage's `ESRCH` shouldn't suppress the next
/// stage's SIGTERM. Every per-stage error is logged at
/// `tracing::debug!` with the stage index and the call kind so an
/// operator triaging "some stages didn't terminate" sees every cause,
/// not just the first.
pub(crate) struct CombinedSignaler {
    stages: Arc<[Arc<dyn ChildSignaler>]>,
}

impl CombinedSignaler {
    pub(crate) fn new(stages: Arc<[Arc<dyn ChildSignaler>]>) -> Self {
        Self { stages }
    }
}

/// Fan a single signal call out to every stage, log every error, and
/// return the first non-`Ok` (if any) as the aggregated result. Used
/// by [`CombinedSignaler`]'s three methods — the only thing that
/// varies is the per-stage method invoked, so the fan-out + log-then-
/// retain-first shape lives in one place.
fn fan_out<F>(
    stages: &[Arc<dyn ChildSignaler>],
    call_kind: &'static str,
    mut call: F,
) -> io::Result<()>
where
    F: FnMut(&Arc<dyn ChildSignaler>) -> io::Result<()>,
{
    let mut first_err: Option<io::Error> = None;
    for (idx, s) in stages.iter().enumerate() {
        if let Err(e) = call(s) {
            tracing::debug!(
                stage = idx,
                call = call_kind,
                ?e,
                "CombinedSignaler: per-stage call failed",
            );
            if first_err.is_none() {
                first_err = Some(e);
            }
        }
    }
    first_err.map_or(Ok(()), Err)
}

impl ChildSignaler for CombinedSignaler {
    fn signal_term(&self) -> io::Result<()> {
        fan_out(&self.stages, "signal_term", |s| s.signal_term())
    }

    fn signal_kill(&self) -> io::Result<()> {
        fan_out(&self.stages, "signal_kill", |s| s.signal_kill())
    }

    fn reap_blocking(&self) -> io::Result<()> {
        fan_out(&self.stages, "reap_blocking", |s| s.reap_blocking())
    }

    fn is_dead(&self) -> bool {
        // The combined signaler is "dead" iff every stage is dead.
        // Per-step timer threads address individual stages via
        // `Arc<dyn>` handles; this aggregate predicate is for
        // shutdown plumbing that wants a one-shot "is this pipe
        // done" probe.
        self.stages.iter().all(|s| s.is_dead())
    }
}

/// Aggregating waiter for a pipe.
///
/// On `wait`, spawns one OS thread per stage and watches every stage's
/// waiter concurrently. The first stage that reports `Failed` triggers
/// a SIGTERM cascade to every other still-alive stage, so a hung
/// stage doesn't keep the pipe alive after another stage fails — the
/// kernel's SIGPIPE chain handles the happy case, but a stage that's
/// not reading from stdin (or that's blocked on something else) needs
/// an explicit signal.
///
/// **Why parallel.** A sequential drain of N stage waiters head-of-
/// lines on stage 0: if stage 0 blocks indefinitely (infinite sleep,
/// blocked read), the cascade can't fire until stage 0 returns, and a
/// downstream stage's per-step timeout can never propagate back. The
/// parallel shape eliminates that hazard at the cost of one OS thread
/// per stage; the per-stage timer threads (one per stage with a
/// `timeout`) already grow O(N), so the thread budget is the same
/// order.
///
/// Aggregated outcome:
/// - **all Ok** ⇒ [`EffectOutcome::Ok`].
/// - **any Failed** ⇒ [`EffectOutcome::Failed`] carrying the *last*
///   non-zero exit in spawn order (pipefail-on: the last failure
///   observable from a shell's perspective dominates) and/or the
///   *first* signal seen in spawn order (so timer-driven SIGTERMs
///   surface). One-or-both present yields
///   [`Termination::Exit`] / [`Termination::Signal`] /
///   [`Termination::PipeMixed`]; neither yields
///   [`Termination::Internal`].
///
/// The spawn-order aggregation is preserved by collecting reports
/// into a stage-indexed `Vec<Option<EffectOutcome>>` and iterating
/// `0..n` at fold time — completion order doesn't bleed into the
/// reported outcome.
///
/// `signalers` is parallel-indexed with `waiters`. The cascade
/// iterates every signaler whose index differs from the first-Failed
/// stage and whose paired waiter hasn't reported yet (`is_dead` short-
/// circuits already-reaped stages — their waiter set the shared flag
/// before returning).
pub(crate) struct PipeWaiter {
    waiters: Vec<Box<dyn ChildWaiter>>,
    signalers: Arc<[Arc<dyn ChildSignaler>]>,
}

impl PipeWaiter {
    pub(crate) fn new(
        waiters: Vec<Box<dyn ChildWaiter>>,
        signalers: Arc<[Arc<dyn ChildSignaler>]>,
    ) -> Self {
        debug_assert_eq!(
            waiters.len(),
            signalers.len(),
            "PipeWaiter: per-stage waiter/signaler counts must match",
        );
        Self { waiters, signalers }
    }
}

/// One per-stage report carried over the aggregator channel.
struct StageReport {
    idx: usize,
    outcome: EffectOutcome,
}

impl ChildWaiter for PipeWaiter {
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
        let Self { waiters, signalers } = *self;
        let n = signalers.len();
        debug_assert_eq!(waiters.len(), n);

        // Empty pipes are rejected by the builder/lowering; guard
        // anyway so a degenerate value (e.g. tests constructing
        // `PipeWaiter::new` with zero stages) collapses to Ok rather
        // than blocking on an empty channel below.
        if n == 0 {
            return Ok(EffectOutcome::Ok);
        }

        let (tx, rx) = unbounded::<StageReport>();
        let mut handles: Vec<thread::JoinHandle<()>> = Vec::with_capacity(n);
        let mut spawn_failed: Option<(usize, io::Error)> = None;
        let mut leftover: Vec<Box<dyn ChildWaiter>> = Vec::new();

        for (idx, w) in waiters.into_iter().enumerate() {
            if spawn_failed.is_some() {
                // After the first spawn failure we keep collecting
                // remaining waiters so the recovery path can drain
                // them synchronously (SIGKILL above makes wait fast).
                leftover.push(w);
                continue;
            }
            let tx_stage = tx.clone();
            let spawn_result = thread::Builder::new()
                // pthread_setname_np on Linux truncates at 15+null;
                // "act-pipe-NN" is 11 chars even at idx=99, leaving
                // headroom for triple-digit stage counts (unreachable
                // in v1 — pipes top out far below).
                .name(format!("act-pipe-{idx}"))
                .spawn(move || {
                    let outcome = match std::panic::catch_unwind(AssertUnwindSafe(|| w.wait())) {
                        Ok(Ok(o)) => o,
                        Ok(Err(_)) | Err(_) => EffectOutcome::Failed(Termination::Internal),
                    };
                    // Channel is unbounded and owned by the
                    // aggregator (this function); send cannot block,
                    // and a Disconnected error here would only happen
                    // if the aggregator panicked — we drop the report
                    // silently in that case.
                    let _ = tx_stage.send(StageReport { idx, outcome });
                });
            match spawn_result {
                Ok(h) => handles.push(h),
                Err(e) => {
                    // Closure (and waiter `w`) was dropped along with
                    // the failed spawn attempt; the child process is
                    // alive but has no paired waiter. Subsequent loop
                    // iterations route into the leftover branch.
                    spawn_failed = Some((idx, e));
                }
            }
        }
        // Drop our local sender so `rx.recv()` returns Disconnected
        // once every spawned thread completes its send. The cloned
        // senders inside the threads keep the channel alive in the
        // meantime.
        drop(tx);

        if let Some((failed_idx, e)) = spawn_failed {
            return Ok(recover_from_spawn_failure(
                failed_idx, &e, &signalers, leftover, handles, &rx,
            ));
        }

        // Aggregate.
        let mut reports: Vec<Option<EffectOutcome>> = (0..n).map(|_| None).collect();
        let mut cascade_fired = false;
        while let Ok(StageReport { idx, outcome }) = rx.recv() {
            debug_assert!(reports[idx].is_none(), "duplicate report for stage {idx}");
            let is_failed = matches!(&outcome, EffectOutcome::Failed(_));
            reports[idx] = Some(outcome);

            if is_failed && !cascade_fired {
                cascade_fired = true;
                cascade_sigterm(&signalers, idx);
            }
        }

        // All senders dropped ⇒ every spawned thread finished its
        // send; joining is bounded by stack-unwind time (no syscall).
        for h in handles {
            let _ = h.join();
        }

        Ok(fold_reports(reports))
    }
}

/// Fan SIGTERM out to every stage other than `failed_idx` whose
/// paired waiter hasn't yet returned. `is_dead` short-circuits
/// already-reaped stages — without it we'd queue stale signals
/// against pids that have already been recycled by the kernel.
/// Per-stage failures are logged at `tracing::debug!` and otherwise
/// collapsed — the cascade is best-effort and one stage's ESRCH
/// should not stop the next stage from receiving its signal.
fn cascade_sigterm(signalers: &[Arc<dyn ChildSignaler>], failed_idx: usize) {
    for (s_idx, s) in signalers.iter().enumerate() {
        if s_idx == failed_idx {
            continue;
        }
        if s.is_dead() {
            continue;
        }
        if let Err(e) = s.signal_term() {
            tracing::debug!(stage = s_idx, ?e, "PipeWaiter: cascade SIGTERM failed",);
        }
    }
}

/// Aggregate stage-indexed reports into the pipe's [`EffectOutcome`].
///
/// Iterates `0..n` (spawn order) so the outcome is independent of
/// completion order: the last non-zero exit wins (pipefail-on) and the
/// first observed signal wins. A missing report (`None` — the
/// wait-thread panicked before send) folds as
/// [`Termination::Internal`]. Total: every stage `Termination`
/// projects to an `(exit, signal)` pair, and the aggregate recomposes
/// from that pair with no unreachable arm.
fn fold_reports(reports: Vec<Option<EffectOutcome>>) -> EffectOutcome {
    let mut last_failed_exit: Option<i32> = None;
    let mut first_signal: Option<i32> = None;
    let mut any_failed = false;
    for outcome_opt in reports {
        let outcome = outcome_opt.unwrap_or(EffectOutcome::Failed(Termination::Internal));
        let EffectOutcome::Failed(term) = outcome else {
            continue;
        };
        any_failed = true;
        let (exit, signal) = match term {
            Termination::Internal => (None, None),
            Termination::Exit(c) => (Some(c), None),
            Termination::Signal(s) => (None, Some(s)),
            Termination::PipeMixed {
                last_exit,
                first_signal: sig,
            } => (Some(last_exit), Some(sig)),
        };
        if let Some(c) = exit {
            last_failed_exit = Some(c);
        }
        if first_signal.is_none() && signal.is_some() {
            first_signal = signal;
        }
    }
    if !any_failed {
        return EffectOutcome::Ok;
    }
    EffectOutcome::Failed(match (last_failed_exit, first_signal) {
        (None, None) => Termination::Internal,
        (Some(c), None) => Termination::Exit(c),
        (None, Some(s)) => Termination::Signal(s),
        (Some(c), Some(s)) => Termination::PipeMixed {
            last_exit: c,
            first_signal: s,
        },
    })
}

/// Cleanup path when a stage's wait-thread `Builder::spawn` failed.
/// The child process for `failed_idx` is alive but its waiter was
/// dropped along with the failed spawn closure, so the controller
/// must SIGKILL + sync-reap it via the signaler — same shape as
/// [`crate::pool::state::recover_orphan_after_wait_thread_failure`]
/// for the single-process path.
///
/// SIGKILLing every stage before draining serves two purposes:
/// 1. The already-spawned waiter threads' children exit promptly so
///    their threads return.
/// 2. The leftover (unspawned) waiters' children exit promptly so the
///    synchronous `wait()` in this function doesn't reintroduce a
///    head-of-line block on a still-running stage.
///
/// Returns the synthesised aggregate outcome
/// [`Termination::Internal`] — semantically equivalent to a wait-thread
/// spawn failure on a single-process step.
fn recover_from_spawn_failure(
    failed_idx: usize,
    err: &io::Error,
    signalers: &[Arc<dyn ChildSignaler>],
    leftover: Vec<Box<dyn ChildWaiter>>,
    spawned_handles: Vec<thread::JoinHandle<()>>,
    rx: &Receiver<StageReport>,
) -> EffectOutcome {
    tracing::error!(
        failed_idx,
        ?err,
        "pipe stage wait-thread spawn failed; SIGKILL + sync reap orphan",
    );

    // SIGKILL every stage that's still alive. Already-reaped stages
    // (their waiter set the dead flag) short-circuit. Errors are
    // logged inside the signaler impls.
    for (s_idx, s) in signalers.iter().enumerate() {
        if s.is_dead() {
            continue;
        }
        if let Err(e) = s.signal_kill() {
            tracing::debug!(stage = s_idx, ?e, "PipeWaiter recovery: SIGKILL failed",);
        }
    }

    // Reap the orphan whose waiter was lost — without this the
    // kernel keeps the zombie until process exit.
    if let Err(e) = signalers[failed_idx].reap_blocking() {
        tracing::warn!(
            stage = failed_idx,
            ?e,
            "PipeWaiter recovery: orphan reap_blocking failed",
        );
    }

    // Synchronously drain the unspawned waiters. SIGKILL above
    // ensures wait returns quickly; the outcome is discarded because
    // the aggregate is unconditionally Failed below.
    for w in leftover {
        let _ = w.wait();
    }

    // Drain the spawned threads' channel and join their handles.
    while rx.recv().is_ok() {
        // Discard reports — the recovery path returns Failed
        // regardless of which stages reported what.
    }
    for h in spawned_handles {
        let _ = h.join();
    }

    EffectOutcome::Failed(Termination::Internal)
}

#[cfg(test)]
mod tests {
    //! Direct tests for the aggregator types. The wiring into
    //! `OsSpawner::spawn_pipe` and `MockSpawner::spawn_pipe` is
    //! exercised by their respective test suites; here we just pin
    //! the aggregation rules.
    use super::{CombinedSignaler, PipeWaiter};
    use crate::lifecycle::DeadFlag;
    use crate::spawner::{ChildSignaler, ChildWaiter};
    use crossbeam::channel::{Receiver, Sender, bounded};
    use specter_core::{EffectOutcome, Termination};
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    /// Per-stage probe: counts every signal call so the cascade
    /// assertions can pin which stages received SIGTERM. Holds a
    /// [`DeadFlag`] paired with a [`StaticWaiter`] (or
    /// [`BlockingWaiter`]) so the test fixture mirrors the production
    /// pairing shape exactly.
    struct Probe {
        dead: DeadFlag,
        term: AtomicU32,
        kill: AtomicU32,
    }

    impl Probe {
        fn new() -> Self {
            Self {
                dead: DeadFlag::new(),
                term: AtomicU32::new(0),
                kill: AtomicU32::new(0),
            }
        }
        fn mark_dead(&self) {
            self.dead.mark_dead();
        }
    }

    impl ChildSignaler for Probe {
        fn signal_term(&self) -> io::Result<()> {
            self.term.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            self.kill.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            self.dead.mark_dead();
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.is_dead()
        }
    }

    /// Trivial waiter for the aggregator tests — returns a
    /// pre-baked outcome (or `io::Error`). Single-use.
    struct StaticWaiter {
        outcome: io::Result<EffectOutcome>,
        dead: Arc<Probe>,
    }

    impl ChildWaiter for StaticWaiter {
        fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
            self.dead.mark_dead();
            self.outcome
        }
    }

    fn probes_into_signalers(probes: &[Arc<Probe>]) -> Arc<[Arc<dyn ChildSignaler>]> {
        probes
            .iter()
            .map(|p| Arc::clone(p) as Arc<dyn ChildSignaler>)
            .collect()
    }

    /// Probe paired with a [`BlockingWaiter`]: the waiter blocks on a
    /// crossbeam channel until any signal method runs, mirroring a
    /// real child process that exits when SIGTERM/SIGKILL is delivered.
    ///
    /// Used to deterministically exercise the parallel cascade: an
    /// "alive sibling" (still running when the cascade fires) is
    /// modeled as a `BlockingProbe`/`BlockingWaiter` pair, where the
    /// waiter's `wait` returns only after the controller's
    /// `signal_term` writes to the unblock channel. This is the
    /// shape no purely-static fixture can give us — without it the
    /// in-process race between "stage 0 reports Failed" and
    /// "stage 1 reports Ok" would make cascade assertions
    /// nondeterministic.
    struct BlockingProbe {
        dead: DeadFlag,
        term: AtomicU32,
        kill: AtomicU32,
        return_outcome: EffectOutcome,
        unblock: Sender<()>,
    }

    impl BlockingProbe {
        /// Build a (probe, waiter) pair. The waiter consumes the
        /// receiver; the probe owns the sender so signal methods
        /// can unblock the paired waiter. The waiter returns
        /// `return_outcome` after unblock.
        fn new(return_outcome: EffectOutcome) -> (Arc<Self>, BlockingWaiter) {
            // bounded(1) + try_send: the unblock is one-shot. If
            // multiple signal calls land before the waiter consumes,
            // the second send drops silently — the receiver only
            // needs one.
            let (tx, rx) = bounded::<()>(1);
            let probe = Arc::new(Self {
                dead: DeadFlag::new(),
                term: AtomicU32::new(0),
                kill: AtomicU32::new(0),
                return_outcome,
                unblock: tx,
            });
            let waiter = BlockingWaiter {
                rx,
                probe: Arc::clone(&probe),
            };
            (probe, waiter)
        }
    }

    impl ChildSignaler for BlockingProbe {
        fn signal_term(&self) -> io::Result<()> {
            self.term.fetch_add(1, Ordering::SeqCst);
            // Non-blocking: dropping the message on a full slot
            // (already unblocked) is the right behaviour for a
            // one-shot ready signal.
            let _ = self.unblock.try_send(());
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            self.kill.fetch_add(1, Ordering::SeqCst);
            let _ = self.unblock.try_send(());
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            self.dead.mark_dead();
            let _ = self.unblock.try_send(());
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.is_dead()
        }
    }

    /// Waiter half of [`BlockingProbe`] — blocks on `rx.recv()` until
    /// any signal arrives, then returns the configured outcome and
    /// marks the probe dead.
    struct BlockingWaiter {
        rx: Receiver<()>,
        probe: Arc<BlockingProbe>,
    }

    impl ChildWaiter for BlockingWaiter {
        fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
            // recv returns Err iff the sender drops without sending;
            // in tests we always send through a signaler before
            // dropping the probe, so a Disconnected here is a fixture
            // bug. Collapse rather than panic so the surrounding
            // PipeWaiter still aggregates a Failed outcome.
            let _ = self.rx.recv();
            self.probe.dead.mark_dead();
            Ok(self.probe.return_outcome.clone())
        }
    }

    /// All Ok ⇒ aggregated Ok; no cascade SIGTERMs.
    #[test]
    fn all_ok_outcome_is_ok_no_cascade() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Ok),
                dead: Arc::clone(&p0),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Ok),
                dead: Arc::clone(&p1),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Ok),
                dead: Arc::clone(&p2),
            }),
        ];
        let outcome = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        assert_eq!(outcome, EffectOutcome::Ok);
        assert_eq!(p0.term.load(Ordering::SeqCst), 0);
        assert_eq!(p1.term.load(Ordering::SeqCst), 0);
        assert_eq!(p2.term.load(Ordering::SeqCst), 0);
    }

    /// First stage Failed ⇒ aggregated Failed; SIGTERM cascade to
    /// alive siblings.
    ///
    /// Siblings are modeled with [`BlockingWaiter`] so the cascade
    /// fires against stages that are still running — a static
    /// "returns Ok and immediately marks dead" waiter would race the
    /// aggregator's cascade decision and skip the SIGTERM under
    /// `is_dead`. The blocking shape pins the parallel cascade
    /// invariant: a sibling that hasn't yet returned receives
    /// SIGTERM and aborts.
    #[test]
    fn first_failed_cascades_sigterm_to_alive_siblings() {
        let p0 = Arc::new(Probe::new());
        let (p1, p1_waiter) = BlockingProbe::new(EffectOutcome::Failed(Termination::Signal(15)));
        let (p2, p2_waiter) = BlockingProbe::new(EffectOutcome::Failed(Termination::Signal(15)));
        let signalers: Arc<[Arc<dyn ChildSignaler>]> = vec![
            Arc::clone(&p0) as Arc<dyn ChildSignaler>,
            Arc::clone(&p1) as Arc<dyn ChildSignaler>,
            Arc::clone(&p2) as Arc<dyn ChildSignaler>,
        ]
        .into();

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::Exit(7))),
                dead: Arc::clone(&p0),
            }),
            Box::new(p1_waiter),
            Box::new(p2_waiter),
        ];
        let outcome = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        match outcome {
            EffectOutcome::Failed(Termination::PipeMixed {
                last_exit,
                first_signal,
            }) => {
                assert_eq!(last_exit, 7, "last non-zero exit propagates");
                assert_eq!(
                    first_signal, 15,
                    "first observed signal in spawn order (from cascaded siblings)",
                );
            }
            other => panic!("expected Failed(PipeMixed); got {other:?}"),
        }
        // Stage 0 does NOT receive a cascade SIGTERM (it's the
        // failing stage, already reaped). Stages 1 and 2 do.
        assert_eq!(p0.term.load(Ordering::SeqCst), 0, "stage 0 not cascaded");
        assert_eq!(p1.term.load(Ordering::SeqCst), 1, "stage 1 cascaded");
        assert_eq!(p2.term.load(Ordering::SeqCst), 1, "stage 2 cascaded");
    }

    /// Last stage Failed ⇒ cascade fires *backward* to alive earlier
    /// stages. The sequential design's `idx+1..n` skip was sound only
    /// because earlier stages had already been drained by the time
    /// cascade fired; under parallel drain, earlier stages may still
    /// be alive, so cascade must visit them.
    ///
    /// This case is impossible to express under the old sequential
    /// PipeWaiter — the test pins the parallel-only invariant.
    #[test]
    fn last_failed_cascades_backward_to_alive_earlier_stages() {
        let (p0, p0_waiter) = BlockingProbe::new(EffectOutcome::Failed(Termination::Signal(15)));
        let (p1, p1_waiter) = BlockingProbe::new(EffectOutcome::Failed(Termination::Signal(15)));
        let p2 = Arc::new(Probe::new());
        let signalers: Arc<[Arc<dyn ChildSignaler>]> = vec![
            Arc::clone(&p0) as Arc<dyn ChildSignaler>,
            Arc::clone(&p1) as Arc<dyn ChildSignaler>,
            Arc::clone(&p2) as Arc<dyn ChildSignaler>,
        ]
        .into();
        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(p0_waiter),
            Box::new(p1_waiter),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::Exit(9))),
                dead: Arc::clone(&p2),
            }),
        ];
        let outcome = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        match outcome {
            EffectOutcome::Failed(Termination::PipeMixed {
                last_exit,
                first_signal,
            }) => {
                // last_exit = last non-zero in spawn order = 9 (stage 2);
                // first_signal = first observed in spawn order = 15 (stage 0).
                assert_eq!(last_exit, 9);
                assert_eq!(first_signal, 15);
            }
            other => panic!("expected Failed(PipeMixed); got {other:?}"),
        }
        assert_eq!(
            p0.term.load(Ordering::SeqCst),
            1,
            "stage 0 cascaded (backward)"
        );
        assert_eq!(
            p1.term.load(Ordering::SeqCst),
            1,
            "stage 1 cascaded (backward)"
        );
        assert_eq!(
            p2.term.load(Ordering::SeqCst),
            0,
            "stage 2 (failing) not cascaded"
        );
    }

    /// Stage 0's waiter blocks indefinitely (modelled as
    /// [`BlockingWaiter`]); stage 1 reports Failed promptly. The
    /// parallel waiter design must let stage 1's report fire the
    /// cascade SIGTERM, which unblocks stage 0's waiter via its
    /// [`BlockingProbe`]'s `signal_term`. A sequential waiter would
    /// head-of-line block on stage 0 and never see stage 1's report.
    ///
    /// The timing bound (under 5 seconds) is a generous proxy: a
    /// deadlock manifests as the test hanging until nextest's
    /// per-test timeout fires.
    #[test]
    fn blocked_first_stage_unblocked_by_cascade_does_not_deadlock() {
        let (p0, p0_waiter) = BlockingProbe::new(EffectOutcome::Failed(Termination::Signal(15)));
        let p1 = Arc::new(Probe::new());
        let signalers: Arc<[Arc<dyn ChildSignaler>]> = vec![
            Arc::clone(&p0) as Arc<dyn ChildSignaler>,
            Arc::clone(&p1) as Arc<dyn ChildSignaler>,
        ]
        .into();
        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(p0_waiter),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::Exit(3))),
                dead: Arc::clone(&p1),
            }),
        ];
        let start = Instant::now();
        let outcome = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        let elapsed = start.elapsed();
        assert!(
            elapsed < Duration::from_secs(5),
            "parallel waiter must not block on stage 0 (elapsed = {elapsed:?})",
        );
        match outcome {
            EffectOutcome::Failed(Termination::PipeMixed {
                last_exit,
                first_signal,
            }) => {
                assert_eq!(last_exit, 3);
                assert_eq!(first_signal, 15);
            }
            other => panic!("expected Failed(PipeMixed); got {other:?}"),
        }
        assert_eq!(p0.term.load(Ordering::SeqCst), 1, "stage 0 cascaded");
        assert_eq!(
            p1.term.load(Ordering::SeqCst),
            0,
            "stage 1 (failing) not cascaded"
        );
    }

    /// Multiple failures: aggregated exit_code is the LAST non-zero
    /// exit in spawn order; signal is the FIRST observed signal.
    #[test]
    fn multiple_failures_last_exit_first_signal() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::Signal(15))),
                dead: Arc::clone(&p0),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::Exit(2))),
                dead: Arc::clone(&p1),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::PipeMixed {
                    last_exit: 7,
                    first_signal: 11,
                })),
                dead: Arc::clone(&p2),
            }),
        ];
        let outcome = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        match outcome {
            EffectOutcome::Failed(Termination::PipeMixed {
                last_exit,
                first_signal,
            }) => {
                assert_eq!(last_exit, 7, "last non-zero exit (stage 2)");
                assert_eq!(first_signal, 15, "first observed signal (stage 0)");
            }
            other => panic!("expected Failed(PipeMixed); got {other:?}"),
        }
    }

    /// Cascade skips already-dead siblings — `is_dead` short-
    /// circuits the SIGTERM syscall.
    #[test]
    fn cascade_skips_already_dead_siblings() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        // p1 already dead before the cascade decision.
        p1.mark_dead();
        let signalers = probes_into_signalers(&[p0.clone(), p1.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed(Termination::Exit(1))),
                dead: Arc::clone(&p0),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Ok),
                dead: Arc::clone(&p1),
            }),
        ];
        let _ = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        assert_eq!(
            p1.term.load(Ordering::SeqCst),
            0,
            "already-dead sibling skipped by is_dead short-circuit",
        );
    }

    /// CombinedSignaler fans every signal call out to every stage.
    #[test]
    fn combined_signaler_fans_out_to_all_stages() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let combined =
            CombinedSignaler::new(probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]));

        combined.signal_term().unwrap();
        combined.signal_term().unwrap();
        combined.signal_kill().unwrap();

        for p in [&p0, &p1, &p2] {
            assert_eq!(p.term.load(Ordering::SeqCst), 2);
            assert_eq!(p.kill.load(Ordering::SeqCst), 1);
        }
    }

    /// `is_dead` returns true only when every stage is dead.
    #[test]
    fn combined_signaler_is_dead_requires_all_stages_dead() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let combined = CombinedSignaler::new(probes_into_signalers(&[p0.clone(), p1.clone()]));
        assert!(!combined.is_dead());
        p0.mark_dead();
        assert!(!combined.is_dead(), "one alive ⇒ combined not dead");
        p1.mark_dead();
        assert!(combined.is_dead(), "all dead ⇒ combined dead");
    }
}
