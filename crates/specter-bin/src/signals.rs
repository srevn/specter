//! Signal-hook iterator + dispatch.
//!
//! [`spawn_signal_thread`] starts the `specter-signal` thread; the
//! thread's body installs handlers for SIGHUP / SIGINT / SIGTERM via
//! `signal_hook::iterator::Signals`, then loops `signals.forever()`
//! routing each signal through [`dispatch_signal`]. The dispatch is
//! pulled into a free function so sibling tests can drive it directly
//! without going through `Signals::forever` (which registers
//! process-wide handlers — fragile under cargo test parallelism).
//!
//! Behavior:
//! - **SIGHUP** → `reload_signal_tx.try_send(())`. Bounded(1) coalesces
//!   redundant signals at the channel layer; the kernel's signal queue
//!   coalesces at its own layer. Net: a flurry of SIGHUPs results in
//!   ≤2 reloads (one in flight + one queued).
//! - **SIGINT / SIGTERM (first)** → broadcast shutdown:
//!   `shutdown_flag.store(true)`; `try_send(())` on both shutdown
//!   channels; `wake_handle.wake()` to interrupt the watcher's
//!   `poll_until`. Records `first_term = Some(now)`.
//! - **SIGINT / SIGTERM (second within `HARD_EXIT_WINDOW`)** →
//!   pre-empt the actuator's 5s SIGTERM grace via
//!   `hard_shutdown_actuator_tx`, briefly yield so phase 3 (SIGKILL
//!   stragglers) lands before the parent dies, then call the
//!   injectable `exit_fn` (default: `std::process::exit(130)`). Without
//!   the pre-empt,
//!   stubborn children that ignored phase 1's SIGTERM survive as orphans
//!   reparented to PID 1. The injectable `exit_fn` parameter lets tests
//!   assert escalation without killing the test runner.
//!
//! After the 2-second window, a fresh termination signal *replaces*
//! `first_term` — the next signal can re-escalate against the new
//! recorded time.

use crate::channels::SignalSide;
use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use specter_sensor::WakeHandle;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// Max gap between two terminations before the second escalates to a
/// hard exit. Operator pressing Ctrl-C twice in <2s → "I'm done waiting."
pub const HARD_EXIT_WINDOW: Duration = Duration::from_secs(2);

/// Exit code conventionally used for "killed by SIGINT" (128 + 2).
pub const HARD_EXIT_CODE: i32 = 130;

/// Outcome of a single [`dispatch_signal`] call. Production code
/// ignores it; tests assert on it.
#[derive(Debug, Eq, PartialEq)]
pub enum SignalOutcome {
    /// SIGHUP routed through `reload_signal_tx`.
    ReloadRequested,
    /// First SIGINT/SIGTERM observed; shutdown broadcast.
    ShutdownInitiated,
    /// Second SIGINT/SIGTERM within the hard-exit window. Production
    /// path calls `exit_fn` (typically `process::exit(130)`); the
    /// outcome captures that the escalation triggered so tests can
    /// assert without exiting.
    HardExitTriggered,
    /// Unrecognized signum (kernel queued something we didn't register
    /// for). No-op — defensive.
    Ignored,
}

/// Mutable state the dispatch keeps across calls. Held inside the
/// signal thread; sibling tests construct one and drive
/// [`dispatch_signal`] manually.
#[derive(Debug, Default)]
pub struct SignalState {
    pub first_term: Option<Instant>,
}

/// Pure-ish dispatch — separated from [`Signals::forever`] so tests
/// can exercise every branch without registering process-wide handlers.
///
/// `exit_fn` is invoked on the hard-exit path (second SIGINT/SIGTERM
/// within [`HARD_EXIT_WINDOW`]). Production passes `|code| std::process::exit(code)`;
/// tests pass a closure that records the request.
pub fn dispatch_signal<F>(
    sig: i32,
    now: Instant,
    state: &mut SignalState,
    side: &SignalSide,
    shutdown_flag: &AtomicBool,
    wake_handle: &dyn WakeHandle,
    exit_fn: F,
) -> SignalOutcome
where
    F: FnOnce(i32),
{
    match sig {
        SIGHUP => {
            tracing::info!("SIGHUP — config reload requested");
            // try_send returns Err(Full) if a previous SIGHUP is
            // already queued — which is what we want (coalesce).
            let _ = side.reload_signal_tx.try_send(());
            SignalOutcome::ReloadRequested
        }
        SIGINT | SIGTERM => {
            if let Some(prev) = state.first_term
                && now.duration_since(prev) < HARD_EXIT_WINDOW
            {
                eprintln!("specter: second termination within 2s — exiting hard");
                // Pre-empt the actuator's SIGTERM grace so it
                // SIGKILLs running children before we abort the
                // process — otherwise stubborn children survive as
                // PID-1 orphans.
                let _ = side.hard_shutdown_actuator_tx.try_send(());
                // Brief yield so the actuator's `select!` observes
                // the hard-shutdown signal and reaches phase 3
                // (SIGKILL) before `exit_fn` aborts the process.
                // 50ms is enough margin for the cross-thread hop +
                // SIGKILL syscall on every running child.
                std::thread::sleep(std::time::Duration::from_millis(50));
                exit_fn(HARD_EXIT_CODE);
                return SignalOutcome::HardExitTriggered;
            }
            state.first_term = Some(now);
            tracing::info!(signal = sig, "termination signal — shutdown initiated");
            shutdown_flag.store(true, Ordering::SeqCst);
            let _ = side.shutdown_engine_tx.try_send(());
            let _ = side.shutdown_actuator_tx.try_send(());
            wake_handle.wake();
            SignalOutcome::ShutdownInitiated
        }
        _ => SignalOutcome::Ignored,
    }
}

/// Spawn the signal thread. Registers SIGHUP / SIGINT / SIGTERM
/// process-wide. Returns a [`JoinHandle`] the bin holds for graceful
/// shutdown (in v1, the signal thread is allowed to outlive the process
/// — `signal_hook::iterator::Signals` doesn't expose a programmatic
/// teardown that doesn't race with in-flight signals).
///
/// On `Signals::new` failure (rare — only `EAGAIN` from the
/// thread-private signal mask, basically a fork-bomb scenario), the
/// inner closure logs and returns; the bin proceeds without signal
/// handling and exits via Ctrl-C's kernel-default action.
pub fn spawn_signal_thread(
    side: SignalSide,
    shutdown_flag: Arc<AtomicBool>,
    wake_handle: Box<dyn WakeHandle>,
) -> JoinHandle<()> {
    thread::Builder::new()
        .name("specter-signal".into())
        .spawn(move || {
            let mut signals = match Signals::new([SIGHUP, SIGINT, SIGTERM]) {
                Ok(s) => s,
                Err(e) => {
                    tracing::error!(?e, "signal-hook init failed; signal handling disabled");
                    return;
                }
            };
            let mut state = SignalState::default();
            for sig in signals.forever() {
                let outcome = dispatch_signal(
                    sig,
                    Instant::now(),
                    &mut state,
                    &side,
                    &shutdown_flag,
                    wake_handle.as_ref(),
                    |code| std::process::exit(code),
                );
                if matches!(outcome, SignalOutcome::HardExitTriggered) {
                    // exit_fn returned (test path); break the loop.
                    return;
                }
            }
        })
        .expect("spawn signal thread")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::Channels;
    use specter_sensor::{FsWatcher, testkit::MockFsWatcher};
    use std::sync::atomic::AtomicBool;
    use std::time::Duration;

    /// Tracks whether the injected `exit_fn` was called and with what code.
    #[derive(Debug, Default)]
    struct ExitRecorder {
        code: std::sync::Mutex<Option<i32>>,
    }

    impl ExitRecorder {
        fn record(&self, code: i32) {
            *self.code.lock().expect("ExitRecorder poisoned") = Some(code);
        }
        fn taken(&self) -> Option<i32> {
            *self.code.lock().expect("ExitRecorder poisoned")
        }
    }

    /// Build the dispatch fixtures: channels, flag, mock waker, exit recorder.
    fn fixture() -> (Channels, Arc<AtomicBool>, MockFsWatcher, Arc<ExitRecorder>) {
        let chans = Channels::new();
        let flag = Arc::new(AtomicBool::new(false));
        let watcher = MockFsWatcher::new();
        let recorder = Arc::new(ExitRecorder::default());
        (chans, flag, watcher, recorder)
    }

    #[test]
    fn sighup_sends_reload_pulse() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            SIGHUP,
            Instant::now(),
            &mut state,
            &side,
            &flag,
            wake.as_ref(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::ReloadRequested);
        assert!(chans.reload_signal_tx.is_full(), "pulse queued");
        assert!(!flag.load(Ordering::SeqCst));
        assert_eq!(recorder.taken(), None);
    }

    #[test]
    fn sigterm_first_initiates_shutdown() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let waker = Arc::clone(&watcher.waker);
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            SIGTERM,
            Instant::now(),
            &mut state,
            &side,
            &flag,
            wake.as_ref(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::ShutdownInitiated);
        assert!(state.first_term.is_some());
        assert!(flag.load(Ordering::SeqCst));
        assert!(chans.shutdown_engine_tx.is_full());
        assert!(chans.shutdown_actuator_tx.is_full());
        assert_eq!(*waker.woken.lock().unwrap(), 1);
        assert_eq!(recorder.taken(), None);
    }

    #[test]
    fn sigint_first_initiates_shutdown() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            SIGINT,
            Instant::now(),
            &mut state,
            &side,
            &flag,
            wake.as_ref(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::ShutdownInitiated);
        assert!(flag.load(Ordering::SeqCst));
    }

    #[test]
    fn second_sigint_within_window_triggers_hard_exit() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();

        let t0 = Instant::now();
        dispatch_signal(SIGINT, t0, &mut state, &side, &flag, wake.as_ref(), |c| {
            recorder.record(c);
        });
        assert_eq!(recorder.taken(), None);

        // 100ms later — well within HARD_EXIT_WINDOW (2s).
        let t1 = t0 + Duration::from_millis(100);
        let outcome = dispatch_signal(SIGINT, t1, &mut state, &side, &flag, wake.as_ref(), |c| {
            recorder.record(c);
        });
        assert_eq!(outcome, SignalOutcome::HardExitTriggered);
        assert_eq!(recorder.taken(), Some(HARD_EXIT_CODE));
    }

    #[test]
    fn hard_exit_pre_empts_actuator_shutdown_grace() {
        // The signal thread must fire `hard_shutdown_actuator_tx` BEFORE
        // calling `exit_fn(130)` so the actuator pre-empts its 5s SIGTERM
        // grace and SIGKILLs running children. Without this signal,
        // stubborn children (those that ignored phase 1's SIGTERM) survive
        // as PID-1 orphans after `process::exit`.
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();

        let t0 = Instant::now();
        dispatch_signal(SIGINT, t0, &mut state, &side, &flag, wake.as_ref(), |c| {
            recorder.record(c);
        });
        // First SIGINT does NOT fire hard-shutdown.
        assert!(
            chans.hard_shutdown_actuator_tx.is_empty(),
            "first SIGINT must not preempt grace"
        );

        let t1 = t0 + Duration::from_millis(100);
        let outcome = dispatch_signal(SIGINT, t1, &mut state, &side, &flag, wake.as_ref(), |c| {
            recorder.record(c);
        });
        assert_eq!(outcome, SignalOutcome::HardExitTriggered);
        // Second SIGINT enqueued the hard-shutdown signal so the actuator
        // can SIGKILL its children before the parent exits.
        assert!(
            !chans.hard_shutdown_actuator_tx.is_empty(),
            "second SIGINT must fire hard_shutdown_actuator_tx"
        );
    }

    #[test]
    fn second_sigint_outside_window_does_not_escalate() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();

        let t0 = Instant::now();
        dispatch_signal(SIGTERM, t0, &mut state, &side, &flag, wake.as_ref(), |c| {
            recorder.record(c);
        });

        // 3s later — outside the 2s window.
        let t1 = t0 + Duration::from_secs(3);
        let outcome = dispatch_signal(SIGTERM, t1, &mut state, &side, &flag, wake.as_ref(), |c| {
            recorder.record(c);
        });
        assert_eq!(outcome, SignalOutcome::ShutdownInitiated);
        assert_eq!(recorder.taken(), None, "exit not triggered");
        assert_eq!(state.first_term, Some(t1), "first_term updated");
    }

    #[test]
    fn unknown_signal_is_ignored() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            signal_hook::consts::SIGUSR1,
            Instant::now(),
            &mut state,
            &side,
            &flag,
            wake.as_ref(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::Ignored);
        assert!(!flag.load(Ordering::SeqCst));
    }

    #[test]
    fn redundant_sighup_coalesces_at_bounded_channel() {
        let (chans, flag, watcher, recorder) = fixture();
        let side = chans.signal_side();
        let wake = watcher.wake_handle();
        let mut state = SignalState::default();
        let now = Instant::now();
        for _ in 0..5 {
            dispatch_signal(SIGHUP, now, &mut state, &side, &flag, wake.as_ref(), |c| {
                recorder.record(c);
            });
        }
        // bounded(1) — exactly one pulse queued; the rest dropped silently.
        assert_eq!(chans.reload_signal_tx.len(), 1);
    }
}
