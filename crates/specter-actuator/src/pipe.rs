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
use specter_core::EffectOutcome;
use std::io;
use std::sync::Arc;

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
/// stage's SIGTERM.
pub(crate) struct CombinedSignaler {
    stages: Box<[Arc<dyn ChildSignaler>]>,
}

impl CombinedSignaler {
    pub(crate) fn new(stages: Box<[Arc<dyn ChildSignaler>]>) -> Self {
        Self { stages }
    }
}

impl ChildSignaler for CombinedSignaler {
    fn signal_term(&self) -> io::Result<()> {
        let mut first_err: Option<io::Error> = None;
        for s in &self.stages {
            if let Err(e) = s.signal_term()
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    fn signal_kill(&self) -> io::Result<()> {
        let mut first_err: Option<io::Error> = None;
        for s in &self.stages {
            if let Err(e) = s.signal_kill()
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    fn reap_blocking(&self) -> io::Result<()> {
        let mut first_err: Option<io::Error> = None;
        for s in &self.stages {
            if let Err(e) = s.reap_blocking()
                && first_err.is_none()
            {
                first_err = Some(e);
            }
        }
        first_err.map_or(Ok(()), Err)
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
/// Drains every stage's waiter sequentially in spawn order. On the
/// first non-Ok stage, cascades SIGTERM to every alive sibling
/// (`idx+1..n`) so a hung downstream stage doesn't keep the pipe
/// alive after an upstream failure — the kernel's SIGPIPE chain
/// handles the happy case, but a stage that's not reading from
/// stdin (or that's blocked on something else) needs an explicit
/// signal.
///
/// Aggregated outcome:
/// - **all Ok** ⇒ [`EffectOutcome::Ok`].
/// - **any Failed** ⇒ [`EffectOutcome::Failed`] with
///   `exit_code` = the *last* non-zero exit in spawn order
///   (pipefail-on: the last failure observable from a shell's
///   perspective dominates) and `signal` = the *first* signal seen
///   (so timer-driven SIGTERMs surface in the reported outcome).
///
/// `signalers` is parallel-indexed with `waiters`; the cascade
/// iterates `signalers[idx+1..]` after observing `waiters[idx]` non-
/// `Ok`. `is_dead` short-circuits already-reaped siblings (their
/// waiter has already returned and set the shared flag).
pub(crate) struct PipeWaiter {
    waiters: Vec<Box<dyn ChildWaiter>>,
    signalers: Box<[Arc<dyn ChildSignaler>]>,
}

impl PipeWaiter {
    pub(crate) fn new(
        waiters: Vec<Box<dyn ChildWaiter>>,
        signalers: Box<[Arc<dyn ChildSignaler>]>,
    ) -> Self {
        debug_assert_eq!(
            waiters.len(),
            signalers.len(),
            "PipeWaiter: per-stage waiter/signaler counts must match",
        );
        Self { waiters, signalers }
    }
}

impl ChildWaiter for PipeWaiter {
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
        let signalers = self.signalers;
        let mut last_failed_exit: Option<i32> = None;
        let mut first_signal: Option<i32> = None;
        let mut any_failed = false;

        for (idx, w) in self.waiters.into_iter().enumerate() {
            // A waiter that errors out (e.g. mock's BrokenPipe when
            // the test never delivers a completion) is folded into
            // an unspecified `Failed` — same as the single-process
            // `wait_loop` does for the engine-side outcome.
            let outcome = w.wait().unwrap_or(EffectOutcome::Failed {
                exit_code: None,
                signal: None,
            });
            match outcome {
                EffectOutcome::Ok => {}
                EffectOutcome::Failed { exit_code, signal } => {
                    if !any_failed {
                        any_failed = true;
                        // Cascade SIGTERM to alive siblings `idx+1..`.
                        // `is_dead` short-circuits already-reaped
                        // ones; errors are logged inside the signaler
                        // and collapsed at this call site.
                        for s in signalers.iter().skip(idx + 1) {
                            if !s.is_dead()
                                && let Err(e) = s.signal_term()
                            {
                                tracing::debug!(?e, "PipeWaiter: cascade SIGTERM failed");
                            }
                        }
                    }
                    if let Some(c) = exit_code {
                        // Pipefail-on: the *last* non-zero exit
                        // dominates (matches `set -o pipefail` for
                        // the user-facing result, the last-pipeline
                        // exit semantics in a shell).
                        last_failed_exit = Some(c);
                    }
                    if first_signal.is_none() && signal.is_some() {
                        first_signal = signal;
                    }
                }
            }
        }

        if any_failed {
            Ok(EffectOutcome::Failed {
                exit_code: last_failed_exit,
                signal: first_signal,
            })
        } else {
            Ok(EffectOutcome::Ok)
        }
    }
}

#[cfg(test)]
mod tests {
    //! Direct tests for the aggregator types. The wiring into
    //! `OsSpawner::spawn_pipe` and `MockSpawner::spawn_pipe` is
    //! exercised by their respective test suites; here we just pin
    //! the aggregation rules.
    use super::{CombinedSignaler, PipeWaiter};
    use crate::spawner::{ChildSignaler, ChildWaiter};
    use specter_core::EffectOutcome;
    use std::io;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

    /// Per-stage probe: counts every signal call so the cascade
    /// assertions can pin which stages received SIGTERM.
    struct Probe {
        dead: AtomicBool,
        term: AtomicU32,
        kill: AtomicU32,
    }

    impl Probe {
        fn new() -> Self {
            Self {
                dead: AtomicBool::new(false),
                term: AtomicU32::new(0),
                kill: AtomicU32::new(0),
            }
        }
        fn mark_dead(&self) {
            self.dead.store(true, Ordering::SeqCst);
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
            self.dead.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.load(Ordering::SeqCst)
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

    fn boxed_probes_into_signalers(probes: &[Arc<Probe>]) -> Box<[Arc<dyn ChildSignaler>]> {
        probes
            .iter()
            .map(|p| Arc::clone(p) as Arc<dyn ChildSignaler>)
            .collect()
    }

    /// All Ok ⇒ aggregated Ok; no cascade SIGTERMs.
    #[test]
    fn all_ok_outcome_is_ok_no_cascade() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = boxed_probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

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
    /// alive siblings (idx 1, 2).
    #[test]
    fn first_failed_cascades_sigterm_to_alive_siblings() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = boxed_probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed {
                    exit_code: Some(7),
                    signal: None,
                }),
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
        match outcome {
            EffectOutcome::Failed { exit_code, signal } => {
                assert_eq!(exit_code, Some(7), "last non-zero exit propagates");
                assert_eq!(signal, None, "no signal seen");
            }
            EffectOutcome::Ok => panic!("expected Failed; got Ok"),
        }
        // Stage 0 does NOT receive a cascade SIGTERM (it's the
        // failing stage, already reaped). Stages 1 and 2 do.
        assert_eq!(p0.term.load(Ordering::SeqCst), 0, "stage 0 not cascaded");
        assert_eq!(p1.term.load(Ordering::SeqCst), 1, "stage 1 cascaded");
        assert_eq!(p2.term.load(Ordering::SeqCst), 1, "stage 2 cascaded");
    }

    /// Multiple failures: aggregated exit_code is the LAST non-zero
    /// exit in spawn order; signal is the FIRST observed signal.
    #[test]
    fn multiple_failures_last_exit_first_signal() {
        let p0 = Arc::new(Probe::new());
        let p1 = Arc::new(Probe::new());
        let p2 = Arc::new(Probe::new());
        let signalers = boxed_probes_into_signalers(&[p0.clone(), p1.clone(), p2.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed {
                    exit_code: None,
                    signal: Some(15),
                }),
                dead: Arc::clone(&p0),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed {
                    exit_code: Some(2),
                    signal: None,
                }),
                dead: Arc::clone(&p1),
            }),
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed {
                    exit_code: Some(7),
                    signal: Some(11),
                }),
                dead: Arc::clone(&p2),
            }),
        ];
        let outcome = Box::new(PipeWaiter::new(waiters, signalers))
            .wait()
            .unwrap();
        match outcome {
            EffectOutcome::Failed { exit_code, signal } => {
                assert_eq!(exit_code, Some(7), "last non-zero exit (stage 2)");
                assert_eq!(signal, Some(15), "first observed signal (stage 0)");
            }
            EffectOutcome::Ok => panic!("expected Failed; got Ok"),
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
        let signalers = boxed_probes_into_signalers(&[p0.clone(), p1.clone()]);

        let waiters: Vec<Box<dyn ChildWaiter>> = vec![
            Box::new(StaticWaiter {
                outcome: Ok(EffectOutcome::Failed {
                    exit_code: Some(1),
                    signal: None,
                }),
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
        let combined = CombinedSignaler::new(boxed_probes_into_signalers(&[
            p0.clone(),
            p1.clone(),
            p2.clone(),
        ]));

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
            CombinedSignaler::new(boxed_probes_into_signalers(&[p0.clone(), p1.clone()]));
        assert!(!combined.is_dead());
        p0.mark_dead();
        assert!(!combined.is_dead(), "one alive ⇒ combined not dead");
        p1.mark_dead();
        assert!(combined.is_dead(), "all dead ⇒ combined dead");
    }
}
