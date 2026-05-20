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
//! that v1 doesn't need.
//!
//! # Spawn-failure policy
//!
//! `thread::Builder::spawn` can fail (OOM, EAGAIN, ulimit) — extremely
//! rare. [`arm_timer`] treats this as **best-effort**: logs at
//! `tracing::error!` and proceeds without timer enforcement for that
//! single step. The alternative (kill the child to fail the plan) would
//! race the wait thread that's already alive, risking a double-
//! `EffectComplete` for the same Effect. The user-visible regression is
//! "this one step ran without its deadline"; the plan otherwise completes
//! normally. The policy lives at one site (this module) — callers don't
//! choose it per call.
//!
//! # Thread name budget
//!
//! Linux `pthread_setname_np` truncates to 15 bytes plus a null; macOS
//! allows 64. The names built by [`TimerContext::os_thread_name`] are
//! shaped around what `ps -T` / `gdb info threads` can render:
//!
//! - Exec: `act-timer-c{cursor}-pid{pid}` — fits a 9-char pid unscathed
//!   at `cursor=0`.
//! - PipeStage: `act-timer-pipe-c{cursor}-s{stage}-pid{pid}` — exceeds
//!   the Linux ceiling at any non-trivial stage/pid, so live-system
//!   inspection on Linux must cross-reference via the `tracing` log
//!   keyed on the same `pid` (the spawn-failure error line and the
//!   `tracing::debug!("spawned …")` line both carry `?key` + `pid`).
//!
//! The sub identifier is intentionally omitted — adding it would push
//! even the Exec name past the Linux ceiling.

use crate::spawner::ChildSignaler;
use specter_core::DedupKey;
use std::io;
use std::sync::Arc;
use std::thread;
use std::time::Duration;

/// Observability identity for a timer arm — the structured fields the
/// spawn-failure log emits and the typed parts the OS thread name is
/// built from. Kept as a typed enum (rather than loose `&str` + opaque
/// log payload) so the two variants — exec single-process and pipe
/// per-stage — share one shape at the seam and the variant-specific
/// name / log differences are encapsulated in this module.
#[derive(Debug, Clone, Copy)]
pub(crate) enum TimerContext {
    /// Single-process [`specter_core::program::SpawnBody::Exec`] step.
    Exec {
        key: DedupKey,
        cursor: u32,
        pid: u32,
    },
    /// One stage of a multi-stage
    /// [`specter_core::program::SpawnBody::Pipe`] step. `stage` is the
    /// index inside the pipe's stage list; `pid` is the per-stage pid.
    PipeStage {
        key: DedupKey,
        cursor: u32,
        stage: u32,
        pid: u32,
    },
}

impl TimerContext {
    /// Build the OS thread name from typed parts. Single allocation per
    /// arm. The Linux 15-byte ceiling lives in the module docs, not at
    /// the call site.
    fn os_thread_name(&self) -> String {
        match *self {
            Self::Exec { cursor, pid, .. } => format!("act-timer-c{cursor}-pid{pid}"),
            Self::PipeStage {
                cursor, stage, pid, ..
            } => format!("act-timer-pipe-c{cursor}-s{stage}-pid{pid}"),
        }
    }

    /// Emit the spawn-failure log at `tracing::error!`. The deadline is
    /// included so an operator triaging "this step ran past its declared
    /// timeout" can correlate; `?e` carries the kernel error
    /// (typically `EAGAIN` / `ENOMEM` / a ulimit).
    fn log_spawn_failure(&self, deadline: Duration, e: &io::Error) {
        match *self {
            Self::Exec { key, cursor, pid } => tracing::error!(
                ?key,
                cursor,
                pid,
                ?deadline,
                ?e,
                "per-step timer thread spawn failed; deadline not enforced",
            ),
            Self::PipeStage {
                key,
                cursor,
                stage,
                pid,
            } => tracing::error!(
                ?key,
                cursor,
                stage,
                pid,
                ?deadline,
                ?e,
                "per-stage timer thread spawn failed; deadline not enforced",
            ),
        }
    }
}

/// Arm a detached one-shot timer thread that enforces `deadline` against
/// the child paired with `signaler`. See module docs for the algorithm,
/// the spawn-failure contract, and the thread-name budget.
///
/// No return value: the thread is detached (nothing useful to await) and
/// the `thread::Builder::spawn` outcome is policy-handled inside this
/// function — log at `tracing::error!` via [`TimerContext::log_spawn_failure`]
/// and proceed. Pinning the policy here means the two production call
/// sites (exec / pipe stage) can't drift on log severity or structured
/// fields.
///
/// `deadline > Duration::ZERO` is upheld by config validation
/// (`IssueKind::TimeoutZero`); the `debug_assert!` is defense-in-depth
/// for a future regression in that layer.
pub(crate) fn arm_timer(
    deadline: Duration,
    grace: Duration,
    signaler: Arc<dyn ChildSignaler>,
    ctx: TimerContext,
) {
    debug_assert!(
        deadline > Duration::ZERO,
        "arm_timer requires deadline > 0; config validation must reject zero",
    );
    let name = ctx.os_thread_name();
    if let Err(e) = thread::Builder::new()
        .name(name)
        .spawn(move || run_timer(deadline, grace, &*signaler))
    {
        ctx.log_spawn_failure(deadline, &e);
    }
}

/// Timer-thread body. Extracted for direct unit testing without
/// standing up a `thread::Builder::spawn` call (the spawned variant
/// would race the test's polling).
///
/// Takes the signaler as `&dyn` rather than `Arc<dyn>` so the unit test
/// can construct it on the stack against a custom impl. The production
/// caller [`arm_timer`] passes `&*signaler` (the `Arc::deref`) into this
/// function from the spawned closure.
///
/// Signal-failure logs at `tracing::warn!`: the production
/// [`crate::spawner::ChildSignaler`] impl ESRCH-collapses and re-checks
/// `is_dead` internally, so a `signal_term`/`signal_kill` `Err`
/// surfacing here is a non-ESRCH, non-already-dead kernel boundary
/// error (e.g. EPERM after PID-reuse landing on another user's process).
/// An operator triaging "child never terminated despite SIGKILL" needs
/// this surfaced.
fn run_timer(deadline: Duration, grace: Duration, signaler: &dyn ChildSignaler) {
    thread::sleep(deadline);
    if signaler.is_dead() {
        return;
    }
    if let Err(e) = signaler.signal_term() {
        tracing::warn!(?e, "per-step timer SIGTERM failed");
    }
    thread::sleep(grace);
    if signaler.is_dead() {
        return;
    }
    if let Err(e) = signaler.signal_kill() {
        tracing::warn!(?e, "per-step timer SIGKILL failed");
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
                // Wait for the observable SIGTERM before marking dead —
                // racing two wall-clock sleeps flakes under scheduler skew.
                while probe.term_calls() == 0 {
                    std::thread::sleep(Duration::from_millis(1));
                }
                probe.mark_dead();
            });
            run_timer(Duration::from_millis(5), Duration::from_millis(50), &probe);
        });
        assert_eq!(probe.term_calls(), 1, "SIGTERM fired");
        assert_eq!(
            probe.kill_calls(),
            0,
            "SIGKILL skipped — child died in grace"
        );
    }
}
