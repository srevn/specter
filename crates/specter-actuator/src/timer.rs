//! Per-step deadline timer threads.
//!
//! When an [`specter_core::ExecAction`] carries `timeout: Some(d)`, the
//! actuator spawns a one-shot timer thread alongside the wait thread.
//! The timer thread shares the child's [`crate::spawner::ChildSignaler`]
//! via `Arc<dyn ChildSignaler>` and enforces the deadline:
//!
//! 1. Sleep `deadline`.
//! 2. If the child completed naturally (the wait thread set the shared
//!    `dead` flag), exit silently — the SIGTERM/SIGKILL pair would race
//!    a pid-reusing process.
//! 3. Otherwise send SIGTERM.
//! 4. Sleep `grace` (the actuator's `shutdown_grace`).
//! 5. If still alive, send SIGKILL.
//!
//! The thread is **detached**: there is no join handle, no shared
//! cancellation channel. Natural completion is observed via the same
//! `dead` flag the wait thread sets before sending `Reaped`; that flag
//! is the single seam between the two threads. Worst case (child runs
//! exactly to its deadline, then dies of natural causes between the
//! `is_dead` check and the syscall) the kernel SIGTERMs an already-
//! reaped pid — the production signaler short-circuits via the same
//! flag, plus the syscall layer ESRCH-collapses. Either path makes the
//! signal a no-op.
//!
//! # Why one thread per step
//!
//! v1 uses a thread-per-step to mirror the existing wait-thread shape:
//! same lifetime, same ownership model, same `Arc<dyn ChildSignaler>`
//! sharing pattern. A consolidated timer heap would amortise OS thread
//! cost at high pid volumes but reintroduces a single coordination point
//! that v1 doesn't need. Assumption A3 in the action-types expansion
//! plan documents the v2 evolution.
//!
//! # Spawn-failure policy
//!
//! `thread::Builder::spawn` can fail (OOM, EAGAIN, ulimit) — extremely
//! rare. The caller treats this as **best-effort**: log an error and
//! proceed without timer enforcement for that single step. The
//! alternative (kill the child to fail the plan) would race the wait
//! thread that's already alive, risking a double-`EffectComplete` for
//! the same Effect. The user-visible regression is "this one step ran
//! without its deadline"; the plan otherwise completes normally.

use crate::spawner::ChildSignaler;
use std::io;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Spawn a detached one-shot timer thread that enforces `deadline`
/// against the child paired with `signaler`. See module docs for the
/// algorithm and the spawn-failure contract.
///
/// `name` is folded into the OS thread name (`act-timer-{name}`). The
/// length is unbounded at this layer — Linux's `pthread_setname_np`
/// truncates to 15 bytes plus a null, so long names are still safe;
/// macOS allows 64 bytes. The truncation is informational-only.
///
/// Returns `io::Result<()>` rather than a join handle: the thread is
/// detached and there is nothing useful to await. The `Result` surfaces
/// only the `thread::Builder::spawn` outcome; the caller logs failures
/// and proceeds.
pub(crate) fn spawn_timer(
    name: &str,
    deadline: Duration,
    grace: Duration,
    signaler: Arc<dyn ChildSignaler>,
) -> io::Result<()> {
    thread::Builder::new()
        .name(format!("act-timer-{name}"))
        .spawn(move || run_timer(deadline, grace, &*signaler))?;
    Ok(())
}

/// Timer-thread body. Extracted for direct unit testing without
/// standing up a `thread::Builder::spawn` call (the spawned variant
/// would race the test's polling).
///
/// Takes the signaler as `&dyn` rather than `Arc<dyn>` so the unit test
/// can construct it on the stack against a custom impl. The production
/// caller [`spawn_timer`] passes `&*signaler` (the `Arc::deref`) into
/// this function from the spawned closure.
fn run_timer(deadline: Duration, grace: Duration, signaler: &dyn ChildSignaler) {
    thread::sleep(deadline);
    if signaler.is_dead() {
        return;
    }
    if let Err(e) = signaler.signal_term() {
        tracing::debug!(?e, "per-step timer SIGTERM failed");
    }
    thread::sleep(grace);
    if signaler.is_dead() {
        return;
    }
    if let Err(e) = signaler.signal_kill() {
        tracing::debug!(?e, "per-step timer SIGKILL failed");
    }
}

#[cfg(test)]
mod tests {
    use super::run_timer;
    use crate::spawner::ChildSignaler;
    use std::io;
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
    use std::time::{Duration, Instant};

    /// Test signaler with controllable `dead` and call-count recording.
    /// Used by every unit test in this module — keeps the assertions
    /// declarative.
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
        fn mark_dead(&self) {
            self.dead.store(true, Ordering::SeqCst);
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
    }

    /// Natural completion before the deadline: timer wakes, sees
    /// `dead == true`, exits without signalling.
    #[test]
    fn timer_short_circuits_on_natural_completion() {
        let probe = Probe::new();
        probe.mark_dead();
        run_timer(Duration::from_millis(5), Duration::from_millis(5), &probe);
        assert_eq!(probe.term_calls(), 0, "no SIGTERM after natural completion");
        assert_eq!(probe.kill_calls(), 0, "no SIGKILL after natural completion");
    }

    /// Child still alive at deadline: timer sends SIGTERM, then SIGKILL
    /// after the grace.
    #[test]
    fn timer_sigterms_then_sigkills_when_child_stays_alive() {
        let probe = Probe::new();
        run_timer(Duration::from_millis(10), Duration::from_millis(10), &probe);
        assert_eq!(probe.term_calls(), 1, "SIGTERM at deadline");
        assert_eq!(probe.kill_calls(), 1, "SIGKILL after grace");
    }

    /// Child dies during the grace window after SIGTERM: timer wakes
    /// from the grace sleep, observes `dead`, and skips SIGKILL.
    #[test]
    fn timer_skips_sigkill_when_child_dies_during_grace() {
        let probe = Probe::new();
        std::thread::scope(|s| {
            s.spawn(|| {
                // SIGTERM races out at ~5ms; this thread marks dead at
                // ~10ms, well inside the 30ms grace. The timer's
                // post-grace is_dead check then short-circuits SIGKILL.
                std::thread::sleep(Duration::from_millis(10));
                probe.mark_dead();
            });
            run_timer(Duration::from_millis(5), Duration::from_millis(30), &probe);
        });
        assert_eq!(probe.term_calls(), 1, "SIGTERM fired");
        assert_eq!(
            probe.kill_calls(),
            0,
            "SIGKILL skipped — child died in grace"
        );
    }
}
