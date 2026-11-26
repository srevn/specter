//! A stage's `timeout`: a detached thread sends the stage's process SIGTERM at the deadline and
//! SIGKILL a grace later, and logs each.
//!
//! The thread shares the stage's [`crate::spawner::ChildSignaler`] and nothing else — no join handle,
//! no cancellation channel. Before each signal it reads the dead flag the stage's waiter sets when
//! it reaps, so a stage that ended in time is never signalled: its pid may be another process's by
//! then. A stage that ends between that read and the signal is caught by the signaler, which reads
//! the same flag, and a pid already gone collapses `ESRCH`.
//!
//! # Why one thread per timed stage
//!
//! It mirrors the wait thread: same lifetime, same ownership, the same `Arc<dyn ChildSignaler>`
//! shared. A consolidated timer heap would amortise OS thread cost at high pid volumes but adds a
//! coordination point nothing else needs.
//!
//! # Spawn-failure policy
//!
//! `thread::Builder::spawn` can fail (OOM, EAGAIN, ulimit) — extremely rare. [`arm_timer`] treats
//! this as **best-effort**: logs at `tracing::error!` and proceeds without timer enforcement for
//! that single stage. The alternative (kill the child to fail the plan) would race the step's wait
//! thread, risking a double-`EffectComplete` for the same Effect. The user-visible regression is
//! "this one stage ran without its deadline"; the plan otherwise completes normally. The policy lives
//! at one site (this module) — callers don't choose it per call.
//!
//! # Thread name
//!
//! `act-tmr-{pid}`, named for the stage it times: Linux keeps 15 bytes of a thread's name, and the
//! 8-byte prefix leaves a 7-digit pid whole. The key, cursor and stage are in its lines.

use crate::spawner::ChildSignaler;
use specter_core::DedupKey;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// The stage a timer times, as its thread's name and its lines say it.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TimerContext {
    pub key: DedupKey,
    pub cursor: u32,
    pub stage: usize,
    pub pid: u32,
}

/// Arms the timer for the stage `signaler` signals: SIGTERM at `deadline`, SIGKILL `grace` later.
///
/// `deadline > Duration::ZERO` is config validation's to uphold (`IssueKind::TimeoutZero`); the
/// `debug_assert!` catches a regression there.
pub(crate) fn arm_timer(
    deadline: Duration, grace: Duration, signaler: Arc<dyn ChildSignaler>, ctx: TimerContext,
) {
    debug_assert!(
        deadline > Duration::ZERO,
        "arm_timer requires deadline > 0; config validation must reject zero",
    );
    if let Err(e) = thread::Builder::new()
        .name(format!("act-tmr-{}", ctx.pid))
        .spawn(move || run_timer(deadline, grace, &*signaler, ctx))
    {
        let TimerContext { key, cursor, stage, pid } = ctx;
        tracing::error!(
            ?key,
            cursor,
            stage,
            pid,
            ?deadline,
            ?e,
            "step timer thread spawn failed; deadline not enforced",
        );
    }
}

/// The timer thread's body, apart from its spawn so the tests run it on their own thread.
///
/// Every line is a `warn`: the deadline and the grace are bounds the operator set, and a signal that
/// fails is a kernel error past the signaler's own collapse of a reaped or vanished child (an `EPERM`
/// on a recycled pid), which an operator triaging a stage that outlived its SIGKILL needs.
fn run_timer(deadline: Duration, grace: Duration, signaler: &dyn ChildSignaler, ctx: TimerContext) {
    let TimerContext { key, cursor, stage, pid } = ctx;
    thread::sleep(deadline);
    if signaler.is_dead() {
        return;
    }
    tracing::warn!(?key, cursor, stage, pid, ?deadline, "timeout elapsed; SIGTERM");
    if let Err(e) = signaler.signal_term() {
        tracing::warn!(?key, cursor, stage, pid, ?e, "timeout's SIGTERM failed");
    }
    thread::sleep(grace);
    if signaler.is_dead() {
        return;
    }
    tracing::warn!(?key, cursor, stage, pid, ?grace, "grace elapsed; SIGKILL");
    if let Err(e) = signaler.signal_kill() {
        tracing::warn!(?key, cursor, stage, pid, ?e, "timeout's SIGKILL failed");
    }
}

#[cfg(test)]
mod tests {
    use super::{TimerContext, run_timer};
    use crate::spawner::ChildSignaler;
    use specter_core::DedupKey;
    use std::io;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    /// Test signaler with controllable `dead` and call-count recording. Used by every unit test in
    /// this module — keeps the assertions declarative.
    struct Probe {
        dead: AtomicBool,
        term_count: AtomicU32,
        kill_count: AtomicU32,
        when: Mutex<Vec<(&'static str, Instant)>>,
    }

    impl Probe {
        fn new() -> Self {
            Self {
                dead: AtomicBool::new(false),
                term_count: AtomicU32::new(0),
                kill_count: AtomicU32::new(0),
                when: Mutex::new(Vec::new()),
            }
        }
        fn record(&self, label: &'static str) {
            self.when.lock().unwrap().push((label, Instant::now()));
        }
        fn term_calls(&self) -> u32 {
            self.term_count.load(Ordering::SeqCst)
        }
        fn kill_calls(&self) -> u32 {
            self.kill_count.load(Ordering::SeqCst)
        }
    }

    impl ChildSignaler for Probe {
        fn signal_term(&self) -> io::Result<()> {
            self.record("term");
            self.term_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            self.record("kill");
            self.kill_count.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.load(Ordering::SeqCst)
        }
        fn mark_dead(&self) {
            self.dead.store(true, Ordering::SeqCst);
        }
    }

    fn ctx() -> TimerContext {
        TimerContext { key: DedupKey::default(), cursor: 0, stage: 0, pid: 0 }
    }

    /// Natural completion before the deadline: timer wakes, sees `dead == true`, exits without
    /// signalling.
    #[test]
    fn timer_short_circuits_on_natural_completion() {
        let probe = Probe::new();
        probe.mark_dead();
        run_timer(Duration::from_millis(5), Duration::from_millis(5), &probe, ctx());
        assert_eq!(probe.term_calls(), 0, "no SIGTERM after natural completion");
        assert_eq!(probe.kill_calls(), 0, "no SIGKILL after natural completion");
    }

    /// Child still alive at deadline: timer sends SIGTERM, then SIGKILL after the grace.
    #[test]
    fn timer_sigterms_then_sigkills_when_child_stays_alive() {
        let probe = Probe::new();
        run_timer(Duration::from_millis(10), Duration::from_millis(10), &probe, ctx());
        assert_eq!(probe.term_calls(), 1, "SIGTERM at deadline");
        assert_eq!(probe.kill_calls(), 1, "SIGKILL after grace");
    }

    /// Child dies during the grace window after SIGTERM: timer wakes from the grace sleep, observes
    /// `dead`, and skips SIGKILL.
    #[test]
    fn timer_skips_sigkill_when_child_dies_during_grace() {
        let probe = Probe::new();
        std::thread::scope(|s| {
            s.spawn(|| {
                // Wait for the observable SIGTERM before marking dead — racing two wall-clock sleeps
                // flakes under scheduler skew.
                while probe.term_calls() == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                probe.mark_dead();
            });
            run_timer(Duration::from_millis(5), Duration::from_millis(50), &probe, ctx());
        });
        assert_eq!(probe.term_calls(), 1, "SIGTERM fired");
        assert_eq!(probe.kill_calls(), 0, "SIGKILL skipped — child died in grace");
    }
}
