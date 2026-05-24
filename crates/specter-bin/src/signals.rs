//! Signal-hook iterator + dispatch.
//!
//! [`spawn_signal_thread`] starts the `specter-signal` thread around a
//! pre-constructed [`Signals`] (registered in `App::run`'s prologue —
//! see [`crate::app::run`] for the why). The thread's body loops
//! `signals.forever()` routing each signal through [`dispatch_signal`].
//! The dispatch is pulled into a free function so sibling tests can
//! drive it directly without going through `Signals::forever` (which
//! registers process-wide handlers — fragile under cargo test
//! parallelism).
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
//!   `hard_shutdown_actuator_tx`, wait for the actuator's phase 3
//!   confirmation pulse (or sender-drop, or
//!   [`HARD_SHUTDOWN_CONFIRM_TIMEOUT`] fallback) on
//!   `hard_shutdown_done_rx`, then call the injectable `exit_fn`
//!   (default: `std::process::exit(130)`). Without the pre-empt and
//!   confirmation, stubborn children that ignored phase 1's SIGTERM
//!   survive as orphans reparented to PID 1. The injectable `exit_fn`
//!   parameter lets tests assert escalation without killing the test
//!   runner.
//!
//! After the 2-second window, a fresh termination signal *replaces*
//! `first_term` — the next signal can re-escalate against the new
//! recorded time.

use crate::channels::SignalSide;
use signal_hook::consts::{SIGHUP, SIGINT, SIGTERM};
use signal_hook::iterator::Signals;
use specter_sensor::WakeHandle;
use std::io;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

/// The signal set the bin handles end-to-end: SIGHUP for reload,
/// SIGINT / SIGTERM for shutdown (with double-tap escalation). Pinned
/// in one place so the registration site ([`crate::app::run`]'s
/// prologue) and the signal thread's iterator can't drift apart.
pub(crate) const HANDLED_SIGNALS: [i32; 3] = [SIGHUP, SIGINT, SIGTERM];

/// Register `sa_sigaction` handlers for [`HANDLED_SIGNALS`] and return
/// the [`Signals`] iterator the signal thread will eventually drain.
///
/// Called from `App::run`'s prologue — *before* config load,
/// observability init, and channel allocation. The kernel installs
/// the handlers synchronously: any signal arriving in the
/// initialisation window after this call is captured by signal-hook's
/// internal pipe (owned by the returned `Signals`) and surfaces on
/// the first `signals.forever()` iteration once the signal thread
/// runs. Without this lift, every line of init ran with SIGTERM's
/// kernel-default disposition (immediate process death) — see
/// [`crate::app::run`] for the longer rationale.
pub(crate) fn register_signal_handlers() -> io::Result<Signals> {
    Signals::new(HANDLED_SIGNALS)
}

/// Max gap between two terminations before the second escalates to a
/// hard exit. Operator pressing Ctrl-C twice in <2s → "I'm done waiting."
pub(crate) const HARD_EXIT_WINDOW: Duration = Duration::from_secs(2);

/// Exit code conventionally used for "killed by SIGINT" (128 + 2).
pub(crate) const HARD_EXIT_CODE: i32 = 130;

/// Upper bound on how long the signal thread waits for the actuator's
/// phase 3 confirmation pulse before calling `exit_fn` regardless.
///
/// Healthy phase 3 fanout is microseconds per child (a `kill(2)`
/// syscall); the pulse arrives well inside this window. The timeout
/// is the bound for a *wedged* actuator — a wait thread deadlocked,
/// a panic during the fanout, etc. — past which the parent must die
/// even without confirmation: the kernel reaps surviving children
/// on parent exit, and an orphan window > a few hundred milliseconds
/// is already operator-visible.
///
/// `200ms` is 4× the historical 50ms sleep heuristic, generous enough
/// for cross-thread hop + SIGKILL syscalls on a large child set under
/// scheduler contention, tight enough to keep double-Ctrl-C
/// responsive.
pub(crate) const HARD_SHUTDOWN_CONFIRM_TIMEOUT: Duration = Duration::from_millis(200);

/// Outcome of a single [`dispatch_signal`] call. The caller maps
/// each variant to post-dispatch side effects via [`apply_outcome`]
/// (set shutdown flag + wake the watcher on
/// [`SignalOutcome::ShutdownInitiated`]; return from the loop on
/// [`SignalOutcome::HardExitTriggered`]).
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum SignalOutcome {
    /// SIGHUP routed through the `on_sighup` closure.
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

/// Routing outcome returned by [`dispatch_signal`]'s `on_sighup`
/// closure. Distinguishes a fresh pulse landing in the empty slot
/// from a coalesce (slot already queued or consumer disconnected) so
/// the dispatch can pick a log severity without leaking the
/// underlying channel error vocabulary into the closure signature.
///
/// Both variants correspond to [`SignalOutcome::ReloadRequested`] —
/// the operator's intent to reload is the same regardless of how the
/// pulse routed; the discriminant only drives the log line.
#[derive(Debug, Eq, PartialEq)]
pub(crate) enum ReloadDelivery {
    /// Pulse landed in the empty slot; the driver's next tick observes
    /// it and runs `handle_reload`.
    Queued,
    /// Slot already full (a prior reload is still in flight) or the
    /// consumer disconnected during shutdown. The in-flight reload
    /// absorbs this pulse too — no operator intent is lost.
    Coalesced,
}

/// Mutable state the dispatch keeps across calls. Held inside the
/// signal thread; sibling tests construct one and drive
/// [`dispatch_signal`] manually.
#[derive(Debug, Default)]
pub(crate) struct SignalState {
    pub(crate) first_term: Option<Instant>,
}

/// Pure-ish dispatch — separated from [`Signals::forever`] so tests
/// can exercise every branch without registering process-wide handlers.
///
/// The dispatch is the canonical "translate a signum into a
/// [`SignalOutcome`] + log the event" state machine. Side effects
/// that don't depend on dispatch-internal state (the shutdown flag
/// store, the watcher wake) live in [`apply_outcome`] and run after
/// the dispatch returns — keeping the dispatch's parameter list to
/// the bits it actually reads.
///
/// `on_sighup` is invoked on the SIGHUP branch to route the reload
/// pulse. Production wires it to `side.reload_signal_tx.try_send(())`;
/// the returned [`ReloadDelivery`] drives the dispatch's log severity.
/// `exit_fn` is invoked on the hard-exit path (second SIGINT/SIGTERM
/// within [`HARD_EXIT_WINDOW`]). Production passes
/// `|code| std::process::exit(code)`; tests pass a closure that
/// records the request.
pub(crate) fn dispatch_signal<R, F>(
    sig: i32,
    now: Instant,
    state: &mut SignalState,
    side: &SignalSide,
    on_sighup: R,
    exit_fn: F,
) -> SignalOutcome
where
    R: FnOnce() -> ReloadDelivery,
    F: FnOnce(i32),
{
    match sig {
        SIGHUP => {
            // Branch the log on the routing outcome so an operator
            // sees the structural state per pulse. Both `Coalesced`
            // sub-cases — slot already pulsed, or consumer
            // disconnected during shutdown — collapse to debug; the
            // operator's intent to reload is satisfied either way.
            match on_sighup() {
                ReloadDelivery::Queued => {
                    tracing::info!("SIGHUP — config reload queued");
                }
                ReloadDelivery::Coalesced => {
                    tracing::debug!("SIGHUP — coalesced (prior reload still pending)");
                }
            }
            SignalOutcome::ReloadRequested
        }
        SIGINT | SIGTERM => {
            if let Some(prev) = state.first_term
                && now.duration_since(prev) < HARD_EXIT_WINDOW
            {
                // Synchronous stderr — `process::exit` below skips
                // destructors, so the tracing-appender's worker thread
                // dies with the process. `eprintln!` lands the line
                // before exit; `tracing::*` could be silently dropped.
                eprintln!(
                    "specter: second termination within {}s — exiting hard",
                    HARD_EXIT_WINDOW.as_secs(),
                );
                // Pre-empt the actuator's SIGTERM grace so it
                // SIGKILLs running children before we abort the
                // process — otherwise stubborn children survive as
                // PID-1 orphans.
                let _ = side.hard_shutdown_actuator_tx.try_send(());
                // Wait for the actuator's "phase 3 SIGKILL fanout
                // complete" pulse. Three terminal paths, all OK:
                //   - `Ok(())` — confirmation received; the kernel
                //     has been told to kill every running child.
                //   - `Err(Disconnected)` — actuator thread already
                //     exited (sender dropped); kernel reap pending,
                //     parent safe to die.
                //   - `Err(Timeout)` — fallback bound for a wedged
                //     actuator; parent dies, kernel reaps on exit.
                let _ = side
                    .hard_shutdown_done_rx
                    .recv_timeout(HARD_SHUTDOWN_CONFIRM_TIMEOUT);
                exit_fn(HARD_EXIT_CODE);
                return SignalOutcome::HardExitTriggered;
            }
            state.first_term = Some(now);
            tracing::info!(signal = sig, "termination signal — shutdown initiated");
            let _ = side.shutdown_engine_tx.try_send(());
            let _ = side.shutdown_actuator_tx.try_send(());
            SignalOutcome::ShutdownInitiated
        }
        _ => SignalOutcome::Ignored,
    }
}

/// Apply the side effects [`dispatch_signal`] deferred to the caller:
/// on [`SignalOutcome::ShutdownInitiated`], set the shared shutdown
/// flag and wake the watcher so its blocking `poll_until` exits and
/// observes the flag on the next loop iteration.
///
/// All other outcomes are no-ops here — [`SignalOutcome::ReloadRequested`]
/// is fully handled inside the dispatch (the reload pulse is the only
/// effect); [`SignalOutcome::HardExitTriggered`] needs no flag/wake
/// because the process is about to die and the caller returns from
/// its loop; [`SignalOutcome::Ignored`] is defensive.
///
/// Shared between [`spawn_signal_thread`] and sibling tests so the
/// post-dispatch glue is exercised by the same code path in both.
fn apply_outcome(
    outcome: &SignalOutcome,
    shutdown_flag: &AtomicBool,
    wake_handle: &dyn WakeHandle,
) {
    if matches!(outcome, SignalOutcome::ShutdownInitiated) {
        shutdown_flag.store(true, Ordering::SeqCst);
        wake_handle.wake();
    }
}

/// Spawn the signal thread around a pre-registered [`Signals`]
/// iterator (constructed by [`register_signal_handlers`] in
/// `App::run`'s prologue). The thread loops `signals.forever()`,
/// routing each signal through [`dispatch_signal`].
///
/// Returns a [`JoinHandle`] the bin holds for graceful shutdown (in
/// v1, the signal thread is allowed to outlive the process —
/// `signal_hook::iterator::Signals` doesn't expose a programmatic
/// teardown that doesn't race with in-flight signals), or
/// [`io::Error`] on `thread::Builder::spawn` failure. The caller
/// translates the error to a startup-fail [`std::process::ExitCode`],
/// mirroring the uniform "startup failure → exit 1" contract every
/// other init path in [`crate::app::run`] honours.
pub(crate) fn spawn_signal_thread(
    mut signals: Signals,
    side: SignalSide,
    shutdown_flag: Arc<AtomicBool>,
    wake_handle: Box<dyn WakeHandle>,
) -> io::Result<JoinHandle<()>> {
    thread::Builder::new()
        .name("specter-signal".into())
        .spawn(move || {
            let mut state = SignalState::default();
            for sig in signals.forever() {
                let outcome = dispatch_signal(
                    sig,
                    Instant::now(),
                    &mut state,
                    &side,
                    || match side.reload_signal_tx.try_send(()) {
                        Ok(()) => ReloadDelivery::Queued,
                        Err(_) => ReloadDelivery::Coalesced,
                    },
                    |code| std::process::exit(code),
                );
                apply_outcome(&outcome, &shutdown_flag, wake_handle.as_ref());
                if matches!(outcome, SignalOutcome::HardExitTriggered) {
                    // exit_fn returned (test path); break the loop.
                    return;
                }
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::channels::Channels;
    use specter_sensor::{FsWatcher, testkit::MockFsWatcher};
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

    /// Dispatch fixtures: a fresh [`Channels`] bundle plus an exit
    /// recorder. The whole bundle survives the fixture call so every
    /// paired sender/receiver in the topology stays connected (the
    /// rx for `reload_signal_tx`, `shutdown_engine_tx`,
    /// `shutdown_actuator_tx`, `hard_shutdown_actuator_tx` lives on
    /// `EnginePieces` / `ActuatorSide`, so a partial-move would
    /// resolve those `try_send` calls as `Disconnected`).
    ///
    /// The shutdown flag and watcher wake handle aren't part of the
    /// dispatch fixture — those are [`apply_outcome`]'s inputs and
    /// the apply-outcome tests construct them inline.
    fn fixture() -> (Channels, Arc<ExitRecorder>) {
        (Channels::new(), Arc::new(ExitRecorder::default()))
    }

    /// SIGHUP routing closure mirroring [`spawn_signal_thread`]'s body —
    /// try-send the reload pulse and translate the result into the
    /// dispatch's [`ReloadDelivery`] vocabulary. Tests share this with
    /// production so the routing edge is exercised by one shape.
    fn route_sighup(side: &SignalSide) -> impl FnOnce() -> ReloadDelivery + '_ {
        move || match side.reload_signal_tx.try_send(()) {
            Ok(()) => ReloadDelivery::Queued,
            Err(_) => ReloadDelivery::Coalesced,
        }
    }

    /// `on_sighup` closure for tests whose signum is not SIGHUP —
    /// trips a panic if the dispatch invokes it (asserts the
    /// SIGINT/SIGTERM/Ignored branches don't accidentally route
    /// through the SIGHUP edge).
    fn unused_sighup_route() -> impl FnOnce() -> ReloadDelivery {
        || unreachable!("on_sighup must not be invoked for a non-SIGHUP signal")
    }

    #[test]
    fn sighup_sends_reload_pulse() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            SIGHUP,
            Instant::now(),
            &mut state,
            &side,
            route_sighup(&side),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::ReloadRequested);
        assert!(side.reload_signal_tx.is_full(), "pulse queued");
        assert_eq!(recorder.taken(), None);
    }

    #[test]
    fn sigterm_first_initiates_shutdown() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            SIGTERM,
            Instant::now(),
            &mut state,
            &side,
            unused_sighup_route(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::ShutdownInitiated);
        assert!(state.first_term.is_some());
        assert!(side.shutdown_engine_tx.is_full());
        assert!(side.shutdown_actuator_tx.is_full());
        assert_eq!(recorder.taken(), None);
    }

    #[test]
    fn sigint_first_initiates_shutdown() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            SIGINT,
            Instant::now(),
            &mut state,
            &side,
            unused_sighup_route(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::ShutdownInitiated);
    }

    #[test]
    fn second_sigint_within_window_triggers_hard_exit() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        // Pre-plant the actuator's "phase 3 complete" pulse so the
        // dispatch's `recv_timeout` returns `Ok(())` immediately —
        // models the production fast path where the actuator confirms
        // before the signal thread parks. Without this, the test would
        // wait the full `HARD_SHUTDOWN_CONFIRM_TIMEOUT`.
        chans
            .actuator
            .hard_shutdown_done_tx
            .try_send(())
            .expect("plant pulse");
        let mut state = SignalState::default();

        let t0 = Instant::now();
        dispatch_signal(SIGINT, t0, &mut state, &side, unused_sighup_route(), |c| {
            recorder.record(c);
        });
        assert_eq!(recorder.taken(), None);

        // 100ms later — well within HARD_EXIT_WINDOW (2s).
        let t1 = t0 + Duration::from_millis(100);
        let outcome = dispatch_signal(SIGINT, t1, &mut state, &side, unused_sighup_route(), |c| {
            recorder.record(c);
        });
        assert_eq!(outcome, SignalOutcome::HardExitTriggered);
        assert_eq!(recorder.taken(), Some(HARD_EXIT_CODE));
    }

    #[test]
    fn hard_exit_pre_empts_actuator_shutdown_grace() {
        // The signal thread must fire `hard_shutdown_actuator_tx` BEFORE
        // waiting on `hard_shutdown_done_rx`, then call `exit_fn(130)`.
        // The pre-empt lets the actuator SIGKILL stragglers; the
        // back-channel pulse is the confirmation that fanout completed,
        // replacing the historical 50ms sleep heuristic.
        let (chans, recorder) = fixture();
        let side = chans.signal;
        // Plant the confirmation pulse so dispatch proceeds without
        // waiting `HARD_SHUTDOWN_CONFIRM_TIMEOUT` (the timeout path is
        // covered structurally: any of pulse / disconnect / timeout
        // end at `exit_fn`).
        chans
            .actuator
            .hard_shutdown_done_tx
            .try_send(())
            .expect("plant pulse");
        let mut state = SignalState::default();

        let t0 = Instant::now();
        dispatch_signal(SIGINT, t0, &mut state, &side, unused_sighup_route(), |c| {
            recorder.record(c);
        });
        // First SIGINT does NOT fire hard-shutdown.
        assert!(
            side.hard_shutdown_actuator_tx.is_empty(),
            "first SIGINT must not preempt grace"
        );

        let t1 = t0 + Duration::from_millis(100);
        let outcome = dispatch_signal(SIGINT, t1, &mut state, &side, unused_sighup_route(), |c| {
            recorder.record(c);
        });
        assert_eq!(outcome, SignalOutcome::HardExitTriggered);
        // Second SIGINT enqueued the hard-shutdown signal so the actuator
        // can SIGKILL its children before the parent exits.
        assert!(
            !side.hard_shutdown_actuator_tx.is_empty(),
            "second SIGINT must fire hard_shutdown_actuator_tx"
        );
    }

    #[test]
    fn second_sigint_outside_window_does_not_escalate() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        let mut state = SignalState::default();

        let t0 = Instant::now();
        dispatch_signal(SIGTERM, t0, &mut state, &side, unused_sighup_route(), |c| {
            recorder.record(c);
        });

        // 3s later — outside the 2s window.
        let t1 = t0 + Duration::from_secs(3);
        let outcome = dispatch_signal(SIGTERM, t1, &mut state, &side, unused_sighup_route(), |c| {
            recorder.record(c);
        });
        assert_eq!(outcome, SignalOutcome::ShutdownInitiated);
        assert_eq!(recorder.taken(), None, "exit not triggered");
        assert_eq!(state.first_term, Some(t1), "first_term updated");
    }

    #[test]
    fn unknown_signal_is_ignored() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        let mut state = SignalState::default();
        let outcome = dispatch_signal(
            signal_hook::consts::SIGUSR1,
            Instant::now(),
            &mut state,
            &side,
            unused_sighup_route(),
            |c| recorder.record(c),
        );
        assert_eq!(outcome, SignalOutcome::Ignored);
    }

    #[test]
    fn redundant_sighup_coalesces_at_bounded_channel() {
        let (chans, recorder) = fixture();
        let side = chans.signal;
        let mut state = SignalState::default();
        let now = Instant::now();
        for _ in 0..5 {
            dispatch_signal(SIGHUP, now, &mut state, &side, route_sighup(&side), |c| {
                recorder.record(c);
            });
        }
        // bounded(1) — exactly one pulse queued; the rest dropped silently.
        assert_eq!(side.reload_signal_tx.len(), 1);
    }

    #[test]
    fn apply_outcome_shutdown_initiated_sets_flag_and_wakes() {
        // Pin the post-dispatch glue: the [`SignalOutcome::ShutdownInitiated`]
        // branch must store-true the shared shutdown flag and wake the
        // watcher's `poll_until` so the watcher / config-watcher / IPC
        // threads observe the flag on their next loop iteration.
        let flag = AtomicBool::new(false);
        let watcher = MockFsWatcher::new();
        let wake = watcher.wake_handle();
        let waker = Arc::clone(&watcher.waker);
        apply_outcome(&SignalOutcome::ShutdownInitiated, &flag, wake.as_ref());
        assert!(flag.load(Ordering::SeqCst));
        assert_eq!(*waker.woken.lock().unwrap(), 1);
    }

    #[test]
    fn apply_outcome_non_shutdown_is_no_op() {
        // Three non-shutdown variants share the same early-out path:
        // ReloadRequested — fully handled inside dispatch (reload
        // pulse only); HardExitTriggered — process is dying, caller
        // returns from its loop; Ignored — defensive. None of them
        // touch the shutdown flag or the watcher wake.
        let flag = AtomicBool::new(false);
        let watcher = MockFsWatcher::new();
        let wake = watcher.wake_handle();
        let waker = Arc::clone(&watcher.waker);
        for outcome in [
            SignalOutcome::ReloadRequested,
            SignalOutcome::HardExitTriggered,
            SignalOutcome::Ignored,
        ] {
            apply_outcome(&outcome, &flag, wake.as_ref());
        }
        assert!(!flag.load(Ordering::SeqCst));
        assert_eq!(*waker.woken.lock().unwrap(), 0);
    }
}
