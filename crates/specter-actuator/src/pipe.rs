//! A step's spawned stages, made one job by [`join`]: a lone stage is its own job; a pipe's waiter
//! ([`PipeWaiter`]) folds every stage's report and its signaler ([`CombinedSignaler`]) reaches every
//! stage. Both are the actuator's policy over the stages, whichever spawner made them.

use crate::spawner::{ChildSignaler, ChildWaiter, SpawnHandles};
use crossbeam::channel::{Sender, unbounded};
use specter_core::{DedupKey, EffectOutcome, Termination};
use std::io;
use std::ops::Range;
use std::panic::AssertUnwindSafe;
use std::sync::Arc;
use std::thread;

/// A joined pipe's signaler: the controller's cancel, shutdown and orphan recovery address the pipe
/// as a whole through it, and each call reaches every stage. Its [`PipeWaiter`] shares the slice,
/// to stop the stages upstream of a failure.
///
/// Errors from individual stage signalers are aggregated: the first non-`Ok` is propagated as the
/// combined result; subsequent stages still get the signal (best-effort). This matches the
/// controller's "log and continue" stance — a shutdown signal is best-effort by nature, and one
/// stage's `ESRCH` shouldn't suppress the next stage's SIGTERM. Every per-stage error is logged at
/// `tracing::debug!` with the stage index and the call kind so an operator triaging "some stages
/// didn't terminate" sees every cause, not just the first.
pub(crate) struct CombinedSignaler {
    stages: Arc<[Arc<dyn ChildSignaler>]>,
}

/// One job over a step's spawned stages: a lone stage's own handles; for a pipe, the waiter that
/// folds every stage's report and the signaler that reaches every stage, under the last stage's pid
/// — the one a shell's `$!` names for a pipeline. The waiter's lines name the plan by its `key`.
pub(crate) fn join(key: DedupKey, stages: Vec<SpawnHandles>) -> SpawnHandles {
    match <[SpawnHandles; 1]>::try_from(stages) {
        Ok([lone]) => lone,
        Err(stages) => {
            let pid = stages.last().expect("a spawn hands back a stage at least").pid;
            let signalers: Arc<[Arc<dyn ChildSignaler>]> =
                stages.iter().map(|stage| Arc::clone(&stage.signaler)).collect();
            SpawnHandles {
                pid,
                waiter: Box::new(PipeWaiter { key, stages, signalers: Arc::clone(&signalers) }),
                signaler: Arc::new(CombinedSignaler { stages: signalers }),
            }
        }
    }
}

/// Fan a single signal call out to every stage, log every error, and return the first non-`Ok` (if
/// any) as the aggregated result. Used by [`CombinedSignaler`]'s three methods — the only thing that
/// varies is the per-stage method invoked, so the fan-out + log-then-retain-first shape lives in
/// one place.
fn fan_out<F>(
    stages: &[Arc<dyn ChildSignaler>], call_kind: &'static str, mut call: F,
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
        self.stages.iter().all(|s| s.is_dead())
    }

    fn mark_dead(&self) {
        // `wait_loop`'s backstop reaches every stage's ratchet through this one call, whatever ended
        // the pipe's wait: a panic that took it down before a stage thread marked its stage among
        // them. Idempotent against every per-stage writer.
        for s in self.stages.iter() {
            s.mark_dead();
        }
    }
}

/// A joined pipe's waiter: it waits on every stage at once, a thread each, and folds their reports
/// as [`Pipefail`], sending SIGTERM to the stages each failure stops.
///
/// The stages are waited on at once so that a failure is seen when it happens: a stage upstream of
/// it that blocks without writing (`sleep 60`) would otherwise hold the wait before the failure was
/// ever read. The stage threads are scoped to the wait, so one that dies before it reports makes
/// the wait itself panic: a stage whose ending is unknown never folds as one that succeeded.
pub(crate) struct PipeWaiter {
    key: DedupKey,
    stages: Vec<SpawnHandles>,
    signalers: Arc<[Arc<dyn ChildSignaler>]>,
}

impl ChildWaiter for PipeWaiter {
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
        let Self { key, stages, signalers } = *self;
        let (reports, reported) = unbounded();
        Ok(thread::scope(|scope| {
            let mut stages = stages.into_iter().enumerate();
            for (stage, handles) in stages.by_ref() {
                let report = reports.clone();
                let started = thread::Builder::new()
                    .name(format!("act-pipe-{stage}"))
                    .spawn_scoped(scope, move || wait_stage(key, stage, handles, &report));
                if let Err(e) = started {
                    return fail_unstarted(key, stage, &e, &signalers, stages);
                }
            }
            // The stage threads hold the only senders left, so the loop ends with the last report.
            drop(reports);
            let mut pipefail = Pipefail::default();
            for (stage, outcome) in &reported {
                for upstream in pipefail.record(stage, outcome) {
                    if let Err(e) = signalers[upstream].signal_term() {
                        tracing::debug!(?key, stage = upstream, ?e, "upstream SIGTERM failed");
                    }
                }
            }
            pipefail.outcome()
        }))
    }
}

/// A stage thread's body: waits on the stage and reports how it ended. A failed or panicked wait
/// ends the stage `Internal` and says which, as `pool::state::wait_loop` says it for a job.
fn wait_stage(
    key: DedupKey, stage: usize, handles: SpawnHandles, report: &Sender<(usize, EffectOutcome)>,
) {
    let SpawnHandles { pid, waiter, signaler } = handles;
    let outcome = match std::panic::catch_unwind(AssertUnwindSafe(|| waiter.wait())) {
        Ok(Ok(outcome)) => outcome,
        Ok(Err(e)) => {
            tracing::warn!(?key, stage, pid, ?e, "stage wait failed");
            EffectOutcome::Failed(Termination::Internal)
        }
        Err(_) => {
            tracing::error!(?key, stage, pid, "stage wait panicked");
            EffectOutcome::Failed(Termination::Internal)
        }
    };
    // A waiter that panicked never marked its stage.
    signaler.mark_dead();
    // The receiver outlives the scope, so the send never fails.
    let _ = report.send((stage, outcome));
}

/// A pipe's stage reports, folded as `set -o pipefail` folds them: the pipe fails when any stage
/// fails, and ends as its rightmost failed stage ended.
///
/// That stage is also the edge of what the failures stop. A stage upstream of it can change neither
/// how the pipe ends nor what reaches the pipe's end, since all it writes flows into the failure; a
/// stage downstream of it can change both, so it is left to read what the failure wrote and to finish
/// on its own. [`PipeWaiter::wait`] sends each stage upstream of a failure one SIGTERM: a stage it
/// signals has a failure downstream of it, so the rightmost failure is never one it caused.
#[derive(Default)]
struct Pipefail {
    /// The rightmost failed stage reported so far, and how it ended.
    rightmost: Option<(usize, Termination)>,
}

impl Pipefail {
    /// Folds in `stage`'s report and returns the stages it newly stops: those upstream of `stage`
    /// that no earlier failure stopped. A success stops none, nor does a failure upstream of one
    /// already folded in.
    #[must_use = "the stages it returns are the caller's to stop"]
    fn record(&mut self, stage: usize, outcome: EffectOutcome) -> Range<usize> {
        let stopped = self.rightmost.as_ref().map_or(0, |&(failed, _)| failed + 1);
        match outcome {
            EffectOutcome::Failed(ended) if stage >= stopped => {
                self.rightmost = Some((stage, ended));
                stopped..stage
            }
            EffectOutcome::Failed(_) | EffectOutcome::Ok => 0..0,
        }
    }

    fn outcome(self) -> EffectOutcome {
        self.rightmost.map_or(EffectOutcome::Ok, |(_, ended)| EffectOutcome::Failed(ended))
    }
}

/// Ends a pipe whose stage `stage` got no thread to wait on it: that stage's waiter went with the
/// thread's closure, and the stages after it (`unstarted`) got no thread either. Every stage is
/// killed, so each stage thread already started returns; the orphan is reaped through its signaler
/// and the unstarted stages are waited on here. The pipe ends `Internal`, as a step does whose own
/// wait thread failed to start.
fn fail_unstarted(
    key: DedupKey, stage: usize, err: &io::Error, signalers: &[Arc<dyn ChildSignaler>],
    unstarted: impl Iterator<Item = (usize, SpawnHandles)>,
) -> EffectOutcome {
    tracing::error!(?key, stage, ?err, "pipe stage thread spawn failed; killing the pipe");
    for (idx, signaler) in signalers.iter().enumerate() {
        if let Err(e) = signaler.signal_kill() {
            tracing::debug!(?key, stage = idx, ?e, "SIGKILL of an unwaited pipe failed");
        }
    }
    if let Err(e) = signalers[stage].reap_blocking() {
        tracing::warn!(?key, stage, ?e, "reap of a pipe stage with no wait thread failed");
    }
    for (_, SpawnHandles { waiter, .. }) in unstarted {
        let _ = waiter.wait();
    }
    EffectOutcome::Failed(Termination::Internal)
}

#[cfg(test)]
mod tests {
    //! The waiter and the combined signaler, each pipe built through [`super::join`] as the pool
    //! builds one, over stub stages; [`super::Pipefail`] on its own, as a value. `os.rs`'s tests
    //! drive real children.
    use super::{CombinedSignaler, Pipefail};
    use crate::lifecycle::DeadFlag;
    use crate::spawner::{ChildSignaler, ChildWaiter, SpawnHandles};
    use crossbeam::channel::{Receiver, Sender, bounded};
    use specter_core::{DedupKey, EffectOutcome, Termination};
    use std::io;
    use std::panic::AssertUnwindSafe;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, Ordering};

    /// Per-stage probe: counts the signals it is sent, so a test can pin which stages were stopped.
    /// Holds a [`DeadFlag`] paired with a [`StaticWaiter`], and keeps the signaler's contract: a
    /// stage already reaped is sent nothing.
    struct Probe {
        dead: DeadFlag,
        term: AtomicU32,
        kill: AtomicU32,
    }

    impl Probe {
        fn new() -> Self {
            Self { dead: DeadFlag::new(), term: AtomicU32::new(0), kill: AtomicU32::new(0) }
        }
    }

    impl ChildSignaler for Probe {
        fn signal_term(&self) -> io::Result<()> {
            if !self.dead.is_dead() {
                self.term.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            if !self.dead.is_dead() {
                self.kill.fetch_add(1, Ordering::SeqCst);
            }
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            self.dead.mark_dead();
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.is_dead()
        }
        fn mark_dead(&self) {
            self.dead.mark_dead();
        }
    }

    /// Trivial waiter for the aggregator tests — returns a pre-baked outcome (or `io::Error`).
    /// Single-use.
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
        probes.iter().map(|p| Arc::clone(p) as Arc<dyn ChildSignaler>).collect()
    }

    /// The pipe [`super::join`] makes of these stages, each waiter beside its stage's signaler.
    fn pipe(
        waiters: Vec<Box<dyn ChildWaiter>>, signalers: &[Arc<dyn ChildSignaler>],
    ) -> Box<dyn ChildWaiter> {
        let stages = waiters
            .into_iter()
            .zip(signalers)
            .zip(0..)
            .map(|((waiter, signaler), pid)| SpawnHandles {
                pid,
                waiter,
                signaler: Arc::clone(signaler),
            })
            .collect();
        super::join(DedupKey::default(), stages).waiter
    }

    /// Probe paired with a [`BlockingWaiter`]: the waiter blocks on a crossbeam channel until any
    /// signal method runs, mirroring a real child process that exits when SIGTERM/SIGKILL is
    /// delivered — a stage still running when a failure downstream of it is reported, whatever the
    /// order the threads run in.
    struct BlockingProbe {
        dead: DeadFlag,
        term: AtomicU32,
        return_outcome: EffectOutcome,
        unblock: Sender<()>,
    }

    impl BlockingProbe {
        /// Build a (probe, waiter) pair. The waiter consumes the receiver; the probe owns the sender
        /// so signal methods can unblock the paired waiter. The waiter returns `return_outcome` after
        /// unblock.
        fn new(return_outcome: EffectOutcome) -> (Arc<Self>, BlockingWaiter) {
            // bounded(1) + try_send: the unblock is one-shot. If multiple signal calls land before
            // the waiter consumes, the second send drops silently — the receiver only needs one.
            let (tx, rx) = bounded::<()>(1);
            let probe = Arc::new(Self {
                dead: DeadFlag::new(),
                term: AtomicU32::new(0),
                return_outcome,
                unblock: tx,
            });
            let waiter = BlockingWaiter { rx, probe: Arc::clone(&probe) };
            (probe, waiter)
        }
    }

    impl ChildSignaler for BlockingProbe {
        fn signal_term(&self) -> io::Result<()> {
            if !self.dead.is_dead() {
                self.term.fetch_add(1, Ordering::SeqCst);
                // Non-blocking: dropping the message on a full slot (already unblocked) is the right
                // behaviour for a one-shot ready signal.
                let _ = self.unblock.try_send(());
            }
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            if !self.dead.is_dead() {
                let _ = self.unblock.try_send(());
            }
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
        fn mark_dead(&self) {
            self.dead.mark_dead();
        }
    }

    /// Waiter half of [`BlockingProbe`] — blocks on `rx.recv()` until any signal arrives, then
    /// returns the configured outcome and marks the probe dead.
    struct BlockingWaiter {
        rx: Receiver<()>,
        probe: Arc<BlockingProbe>,
    }

    impl ChildWaiter for BlockingWaiter {
        fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
            // recv returns Err iff the sender drops without sending; in tests we always send through
            // a signaler before dropping the probe, so a Disconnected here is a fixture bug. Collapse
            // rather than panic so the surrounding PipeWaiter still aggregates a Failed outcome.
            let _ = self.rx.recv();
            self.probe.dead.mark_dead();
            Ok(self.probe.return_outcome.clone())
        }
    }

    fn failed(ended: Termination) -> EffectOutcome {
        EffectOutcome::Failed(ended)
    }

    /// The outcome [`Pipefail`] folds `reports` into, in the order they arrive.
    fn fold(reports: impl IntoIterator<Item = (usize, EffectOutcome)>) -> EffectOutcome {
        let mut pipefail = Pipefail::default();
        for (stage, outcome) in reports {
            let _ = pipefail.record(stage, outcome);
        }
        pipefail.outcome()
    }

    /// A failure stops the stages between it and the rightmost failure before it: no stage twice,
    /// none downstream of a failure.
    #[test]
    fn pipefail_stops_each_stage_upstream_of_a_failure_once() {
        assert_eq!(
            Pipefail::default().record(0, failed(Termination::Exit(1))),
            0..0,
            "a first stage's failure stops nothing"
        );
        let mut pipefail = Pipefail::default();
        assert_eq!(pipefail.record(2, EffectOutcome::Ok), 0..0, "a success stops nothing");
        assert_eq!(pipefail.record(1, failed(Termination::Exit(1))), 0..1);
        assert_eq!(
            pipefail.record(0, failed(Termination::Signal(15))),
            0..0,
            "a failure upstream of one folded in stops nothing"
        );
        assert_eq!(pipefail.record(3, failed(Termination::Exit(3))), 2..3);
    }

    /// The pipe ends as its rightmost failed stage ended, whichever order the reports arrive in.
    #[test]
    fn pipefail_ends_as_the_rightmost_failure() {
        let upstream = failed(Termination::Signal(13));
        let rightmost = failed(Termination::Exit(4));
        assert_eq!(
            fold([(0, upstream.clone()), (1, rightmost.clone()), (2, EffectOutcome::Ok)]),
            rightmost
        );
        assert_eq!(
            fold([(1, rightmost.clone()), (2, EffectOutcome::Ok), (0, upstream)]),
            rightmost
        );
        assert_eq!(
            fold([(0, EffectOutcome::Ok), (1, EffectOutcome::Ok), (2, EffectOutcome::Ok)]),
            EffectOutcome::Ok
        );
    }

    /// Every stage succeeds: the pipe is `Ok`, and nothing is stopped.
    #[test]
    fn all_ok_outcome_is_ok_and_stops_nothing() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p0) }),
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p1) }),
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p2) }),
        ];
        let outcome = pipe(waiters, &signalers).wait().unwrap();
        assert_eq!(outcome, EffectOutcome::Ok);
        assert_eq!(p0.term.load(Ordering::SeqCst), 0);
        assert_eq!(p1.term.load(Ordering::SeqCst), 0);
        assert_eq!(p2.term.load(Ordering::SeqCst), 0);
    }

    /// The last stage fails while the two before it block until they are signalled, as `sleep 60`
    /// does: its failure stops both, and the pipe ends as the failure did, not as the stages it
    /// stopped. A wait that took the stages in turn would block on stage 0 and never see the failure.
    #[test]
    fn a_failure_stops_the_live_stages_upstream_of_it() {
        let (p0, p0_waiter) = BlockingProbe::new(failed(Termination::Signal(15)));
        let (p1, p1_waiter) = BlockingProbe::new(failed(Termination::Signal(15)));
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
                outcome: Ok(failed(Termination::Exit(9))),
                dead: Arc::clone(&p2),
            }),
        ];
        let outcome = pipe(waiters, &signalers).wait().unwrap();
        assert_eq!(outcome, failed(Termination::Exit(9)));
        assert_eq!(p0.term.load(Ordering::SeqCst), 1, "stage 0 stopped");
        assert_eq!(p1.term.load(Ordering::SeqCst), 1, "stage 1 stopped");
        assert_eq!(p2.term.load(Ordering::SeqCst), 0, "the failed stage is not signalled");
    }

    /// A signaler whose `mark_dead` panics: its stage thread dies after its wait, before it reports.
    struct UnmarkableSignaler;

    impl ChildSignaler for UnmarkableSignaler {
        fn signal_term(&self) -> io::Result<()> {
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            Ok(())
        }
        fn is_dead(&self) -> bool {
            false
        }
        fn mark_dead(&self) {
            panic!("UnmarkableSignaler: a stage thread dies before it reports");
        }
    }

    /// A stage whose ending nobody learns fails the wait itself, which the job's wait thread reports
    /// as a panicked wait; folded without it, the pipe would pass for one that succeeded.
    #[test]
    fn a_stage_thread_that_dies_unreported_fails_the_wait() {
        let p0 = Arc::new(Probe::new());
        let signalers: Arc<[Arc<dyn ChildSignaler>]> =
            vec![Arc::clone(&p0) as Arc<dyn ChildSignaler>, Arc::new(UnmarkableSignaler)].into();
        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p0) }),
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::new(Probe::new()) }),
        ];
        let waited =
            std::panic::catch_unwind(AssertUnwindSafe(|| pipe(waiters, &signalers).wait()));
        assert!(waited.is_err(), "the wait must panic, not end {waited:?}");
    }

    /// CombinedSignaler fans every signal call out to every stage.
    #[test]
    fn combined_signaler_fans_out_to_all_stages() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let combined = CombinedSignaler {
            stages: probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]),
        };

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
        let combined =
            CombinedSignaler { stages: probes_into_signalers(&[p0.clone(), p1.clone()]) };
        assert!(!combined.is_dead());
        p0.mark_dead();
        assert!(!combined.is_dead(), "one alive ⇒ combined not dead");
        p1.mark_dead();
        assert!(combined.is_dead(), "all dead ⇒ combined dead");
    }

    /// `mark_dead` on the combined signaler fans the write out to every per-stage flag. The
    /// wrapper-layer backstop in `pool::state::wait_loop` relies on this fan-out to close the whole
    /// pipe's PID-reuse window with a single call.
    #[test]
    fn combined_signaler_mark_dead_fans_out_to_all_stages() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let combined = CombinedSignaler {
            stages: probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]),
        };

        combined.mark_dead();

        assert!(p0.is_dead());
        assert!(p1.is_dead());
        assert!(p2.is_dead());
    }

    /// Single-use waiter that panics inside `wait()`. Models a custom [`ChildWaiter`] impl that never
    /// reaches its own happy-path `mark_dead` write — the per-stage `catch_unwind` backstop inside
    /// [`super::PipeWaiter::wait`] is the only writer that can flip the paired signaler's flag.
    struct PanickingWaiter;

    impl ChildWaiter for PanickingWaiter {
        fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
            panic!("PanickingWaiter: intentional panic to exercise per-stage backstop");
        }
    }

    /// Stage 1's waiter panics before its own self-mark. Its stage thread's catch ends the stage
    /// `Internal`, the rightmost failure here, and marks the stage dead: without that mark a racing
    /// controller signal could land on a recycled pid.
    #[test]
    fn a_stage_whose_wait_panics_ends_internal_and_is_marked_dead() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p0) }),
            Box::new(PanickingWaiter),
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p2) }),
        ];
        let outcome = pipe(waiters, &signalers).wait().unwrap();

        assert_eq!(outcome, failed(Termination::Internal));
        assert!(p1.is_dead(), "stage 1's thread marks it dead though its waiter panicked first");
    }

    /// Stage 1's wait fails, as an `ECHILD` would: its ending is unknown, so the stage ends
    /// `Internal`, the rightmost failure here, and never passes for one that succeeded.
    #[test]
    fn a_stage_whose_wait_fails_ends_internal() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let signalers = probes_into_signalers(&[p0.clone(), p1.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter { outcome: Ok(EffectOutcome::Ok), dead: Arc::clone(&p0) }),
            Box::new(StaticWaiter {
                outcome: Err(io::Error::from(io::ErrorKind::Other)),
                dead: Arc::clone(&p1),
            }),
        ];
        assert_eq!(pipe(waiters, &signalers).wait().unwrap(), failed(Termination::Internal));
    }
}
