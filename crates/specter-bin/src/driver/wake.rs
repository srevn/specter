//! Wake-bearing cross-thread sinks to the engine.
//!
//! Two `pub(crate)` primitives:
//!
//! - [`WakeHandle`] — newtype around `Arc<mio::Waker>`. The only call to [`mio::Waker::new`] in the
//!   bin lives in [`WakeHandle::new`], making mio's "one Waker per Poll" a structural property: a
//!   second construction site requires a fresh `use mio::Waker` import, which `grep "use
//!   mio::Waker"` makes a code-review red flag.
//! - [`WakingSink`] — a `Sender<Input>` paired with a [`WakeHandle`]. The single body of the
//!   send-THEN-wake protocol — every wake-bearing crossing of the cross-thread boundary into the
//!   engine routes through one method ([`WakingSink::send`]). The sink's [`Drop`] is the structural
//!   dual on the *channel close* edge: it closes the `Sender<Input>` BEFORE pulsing the wake, so
//!   the driver's next `try_recv` observes `Disconnected` and the actuator-gone path in
//!   [`crate::driver::Reactor::poll_and_drain`] routes through to
//!   [`crate::driver::EngineDriver::begin_shutdown`] without a race window where the driver wakes
//!   on an Empty channel and re-sleeps before the disconnect lands.
//!
//! Adapters in [`crate::app`] wrap a `WakingSink` to implement
//! [`specter_sensor::ProberResponseSender`] / [`specter_actuator::EffectCompleteSender`]: each
//! adapter is one content-lift line, the wake-and-send protocol stays single-sourced. A future
//! wake-bearing sink drops in as a third adapter with the same shape — single-Waker stays
//! structural by typing, not by convention.

use crossbeam::channel::Sender;
use mio::{Registry, Token, Waker};
use specter_core::{Input, SendError};
use std::io;
use std::sync::Arc;

/// Cloneable handle to the Reactor's single mio Waker. [`Clone`] is structurally cheap —
/// [`Arc::clone`] of the underlying [`mio::Waker`], one atomic refcount bump per call.
///
/// The `pub(crate)` constructor [`WakeHandle::new`] is the only call site of [`mio::Waker::new`] in
/// the bin — `grep "Waker::new"` returns one production hit (here). A future contributor adding a
/// second wake-bearing sink either threads a clone of an existing [`WakeHandle`] through, or
/// imports [`mio::Waker`] afresh — the latter is a code-review red flag `grep "use mio::Waker"`
/// makes visible at one grep.
#[derive(Clone, Debug)]
pub(crate) struct WakeHandle(Arc<Waker>);

impl WakeHandle {
    /// Construct the single [`mio::Waker`] for this Poll registry.
    ///
    /// The `token` argument is supplied by the caller so this module stays token-agnostic —
    /// [`crate::driver::Reactor`] passes its `TOKEN_WAKER` constant in. Any other Poll registry is
    /// a bug: the project's structural invariant is one Poll per process; constructing a
    /// `WakeHandle` against a foreign registry would silently miss every wake edge.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `Waker::new` — kernel-pressure (`EMFILE` on the Waker fd) or
    /// programmer-error. Callers treat either as startup-fatal.
    pub(crate) fn new(registry: &Registry, token: Token) -> io::Result<Self> {
        Ok(Self(Arc::new(Waker::new(registry, token)?)))
    }

    /// Pulse the wake edge.
    ///
    /// Errors only when the paired Poll has been destroyed — the shutdown path; every production
    /// caller threads the result through `let _ = ...` because the next surviving wake (none, once
    /// Poll is gone) is the right reader. The bin's only production wake source is
    /// [`WakingSink::send`] post-channel send.
    pub(crate) fn wake(&self) -> io::Result<()> {
        self.0.wake()
    }
}

/// Wake-bearing cross-thread sink to the engine's [`Input`] channel.
///
/// Sealed primitive: a [`Sender<Input>`] paired with a [`WakeHandle`]. [`Self::send`] runs the
/// send-THEN-wake protocol — the channel must hold the message before the wake edge fires, or the
/// driver's `try_recv` after `Poll::poll` returns sees `Empty` and the message strands until the
/// next unrelated wake.
///
/// **Drop runs close-THEN-wake**, the symmetric protocol on the channel-close edge. The bin's
/// actuator-thread closure owns a `Box<dyn EffectCompleteSender>` whose inner sink is this struct;
/// when that closure exits (clean or panic), the Box drops, the adapter drops, and this [`Drop`]
/// fires. The [`Sender<Input>`] closes FIRST (via [`Option::take`] + implicit end-of-statement
/// drop), so by the time the wake edge fires the driver's next `try_recv` on the paired `Receiver`
/// is guaranteed to see `TryRecvError::Disconnected` — the actuator-gone signal in
/// [`crate::driver::Reactor::poll_and_drain`] (via
/// [`crate::driver::reactor::DrainedTick::actuator_gone`]) lands on the next tick body. Reversing
/// the order opens a race window where the driver wakes, observes `Empty`, sleeps again, and
/// strands on Disconnected until an unrelated wake fires.
///
/// The `Option<Sender<Input>>` wrapper is layout-free for the channel-sender shape ([`Sender<Input>`]
/// is a thin `Arc<Channel<Input>>` internally, so `Option<Sender<Input>>` gets the null-pointer
/// optimization). The cost on the hot [`Self::send`] path is one register-resident, well-predicted
/// `Option::as_ref` branch per send — well below the noise floor for the cross-thread channel hop.
///
/// Construction requires a [`WakeHandle`], obtainable from [`crate::driver::Reactor::wake_handle`].
/// New wake-bearing sinks drop in as `WakingSink::new(tx, reactor.wake_handle())` — the mio "one
/// Waker per Poll" invariant stays structural.
pub(crate) struct WakingSink {
    /// Wrapped in [`Option`] so [`Drop`] can [`Option::take`] the inner [`Sender<Input>`] and drop
    /// it BEFORE pulsing the wake edge. The drop-then-wake ordering is the structural floor of the
    /// actuator-gone signal — see the struct rustdoc for the race-window rationale. Steady-state
    /// [`Self::send`] reads via [`Option::as_ref`]; the only call site that takes the inner value
    /// is the [`Drop`] impl below.
    tx: Option<Sender<Input>>,
    waker: WakeHandle,
}

impl WakingSink {
    /// Wrap a [`Sender<Input>`] and a [`WakeHandle`] into the wake-bearing sink. The `waker` argument
    /// is typically a [`WakeHandle`] clone minted via [`crate::driver::Reactor::wake_handle`].
    pub(crate) const fn new(tx: Sender<Input>, waker: WakeHandle) -> Self {
        Self {
            tx: Some(tx),
            waker,
        }
    }

    /// Push an [`Input`] onto the channel then pulse the wake edge.
    ///
    /// `tx.send` on the wrapped unbounded crossbeam channel is non-blocking — it only fails with
    /// `Disconnected` once the driver-side receiver drops, at which point the calling worker
    /// surfaces the error and exits its loop.
    ///
    /// The post-send `wake` is best-effort: it errors only on a destroyed Poll (the shutdown path),
    /// and the next surviving wake's drain picks up our message — nothing to propagate.
    ///
    /// The [`Option::as_ref`] branch surfaces `SendError::Disconnected` if the sink's [`Drop`] has
    /// already run (only reachable from a hypothetical post-Drop send — every production caller
    /// holds the sink by reference for the call's lifetime, so the branch is unreachable in steady
    /// state and well-predicted regardless).
    pub(crate) fn send(&self, input: Input) -> Result<(), SendError> {
        let tx = self.tx.as_ref().ok_or(SendError::Disconnected)?;
        tx.send(input).map_err(|_| SendError::Disconnected)?;
        let _ = self.waker.wake();
        Ok(())
    }
}

impl Drop for WakingSink {
    /// Close the channel BEFORE pulsing the wake edge.
    ///
    /// [`Option::take`] returns the inner [`Sender<Input>`] in an [`Option`]; the returned value is
    /// dropped at the end of the statement (no binding) which drops the sender. The drop transitions
    /// the paired [`crossbeam::channel::Receiver`] into the disconnected state — the driver's next
    /// `try_recv` returns [`crossbeam::channel::TryRecvError::Disconnected`] rather than `Empty`.
    ///
    /// The subsequent [`WakeHandle::wake`] pulse is best-effort: a destroyed Poll (the shutdown
    /// path) returns Err which we discard. The next surviving wake's drain picks up the disconnect;
    /// mio's late-wake-on-destroyed-Poll contract is a silent no-op so there's nothing to propagate
    /// on the way out.
    ///
    /// Ordering matters: if the wake fired BEFORE the sender drop, the driver could wake, observe
    /// `Empty` on the still-connected channel, sleep again, and strand on the eventual Disconnected
    /// until an unrelated wake fired. Wake-after-close collapses the race — the wake-and-disconnect
    /// pair lands as one atomic observation from the driver's side.
    fn drop(&mut self) {
        self.tx.take();
        let _ = self.waker.wake();
    }
}

#[cfg(test)]
mod tests {
    use super::{WakeHandle, WakingSink};
    use crossbeam::channel::{TryRecvError, unbounded};
    use mio::{Events, Poll, Token};
    use specter_core::{Input, ProfileId, TimerId, TimerKind};
    use std::time::{Duration, Instant};

    // Both tests below construct a [`WakeHandle`] directly (no Reactor) and clone it test-side so
    // the underlying [`mio::Waker`] outlives the [`WakingSink::Drop`]. The clone is the test's
    // analogue of the production anchor in [`super::super::Reactor`]: there the Reactor's `waker`
    // field holds one Arc reference for the Poll's lifetime; here the test's `wake` binding holds
    // one Arc reference for the test function's lifetime. Both encode the same invariant —
    // [`Arc<mio::Waker>`] refcount ≥ 1 across the [`WakingSink`]'s drop edge — which mio's contract
    // requires for the post-drop wake to be observable.
    //
    // Without the clone the sink's [`Drop`] runs while holding the sole reference; on Linux that
    // closes the Waker's eventfd inside the Drop body and the pending wake-write is lost (Linux
    // auto-deregisters closed fds from epoll's interest and ready lists). On macOS the
    // `EVFILT_USER` trigger queues on the kqueue's own state and survives the filter teardown, so
    // the same setup passes on macOS even without the clone — passing for the wrong reason.

    /// Constructing a [`WakingSink`] and dropping it closes the paired
    /// [`crossbeam::channel::Receiver`] *and* pulses the `mio::Waker` exactly once — the production
    /// actuator-gone signal.
    ///
    /// The test isolates the close-then-wake ordering: any queued message must drain cleanly
    /// (`Ok(_)`) BEFORE the receiver observes `Disconnected`, and the post-drop `Poll::poll` must
    /// return immediately with `TOKEN_WAKER` ready (proving the Drop-fired wake edge landed).
    #[test]
    fn drop_closes_channel_then_pulses_wake() {
        let mut poll = Poll::new().expect("Poll");
        let waker_token = Token(0xABC);
        let wake = WakeHandle::new(poll.registry(), waker_token).expect("WakeHandle");
        let (tx, rx) = unbounded::<Input>();

        let sink = WakingSink::new(tx, wake.clone());
        // Send one Input so the channel has a queued message; the Drop-fired Disconnected must
        // surface AFTER the drain.
        sink.send(Input::TimerExpired {
            profile: ProfileId::default(),
            kind: TimerKind::Settle,
            id: TimerId::default(),
        })
        .expect("send");
        // Drain the prior send-then-wake pulse so the assertion below observes the Drop-fired edge
        // in isolation.
        let mut events = Events::with_capacity(4);
        poll.poll(&mut events, Some(Duration::from_millis(10)))
            .expect("drain prior send-then-wake pulse");

        drop(sink);

        // Drop-fired wake edge: the next poll returns immediately with TOKEN_WAKER ready.
        let start = Instant::now();
        poll.poll(&mut events, Some(Duration::from_secs(2)))
            .expect("Drop-fired wake unblocks poll");
        assert!(
            start.elapsed() < Duration::from_millis(500),
            "Drop-fired wake unblocks immediately, took {:?}",
            start.elapsed(),
        );
        let observed: Vec<Token> = events.iter().map(mio::event::Event::token).collect();
        assert!(
            observed.contains(&waker_token),
            "Drop must pulse TOKEN_WAKER; observed {observed:?}",
        );

        // The queued send arrives normally (Ok), then the receiver observes Disconnected on the
        // next try_recv — proving the sender drop ran BEFORE the wake (and therefore happens-before
        // any consumer reaction to the wake edge).
        assert!(
            matches!(rx.try_recv(), Ok(Input::TimerExpired { .. })),
            "queued send drains cleanly before Disconnected lands",
        );
        assert!(
            matches!(rx.try_recv(), Err(TryRecvError::Disconnected)),
            "post-drain try_recv observes Disconnected, NOT Empty",
        );

        // Explicitly drop wake at the end of the test to extend its lifetime
        drop(wake);
    }

    /// Drop on an empty channel still pulses the wake edge AND surfaces Disconnected immediately on
    /// the next `try_recv` — the pristine actuator-startup-fails scenario.
    #[test]
    fn drop_on_empty_channel_surfaces_disconnected_immediately() {
        let mut poll = Poll::new().expect("Poll");
        let waker_token = Token(0xDEF);
        let wake = WakeHandle::new(poll.registry(), waker_token).expect("WakeHandle");
        let (tx, rx) = unbounded::<Input>();

        let sink = WakingSink::new(tx, wake.clone());
        drop(sink);

        // Disconnected lands immediately — no queued sends to drain.
        assert!(
            matches!(rx.try_recv(), Err(TryRecvError::Disconnected)),
            "empty-channel drop surfaces Disconnected on the next try_recv",
        );
        // Wake edge fired on Drop, so the next poll returns immediately with TOKEN_WAKER ready.
        let mut events = Events::with_capacity(4);
        poll.poll(&mut events, Some(Duration::from_secs(2)))
            .expect("Drop-fired wake unblocks poll");
        assert!(
            events
                .iter()
                .map(mio::event::Event::token)
                .any(|t| t == waker_token),
            "Drop pulses TOKEN_WAKER",
        );

        // Explicitly drop wake at the end of the test to extend its lifetime
        drop(wake);
    }
}
