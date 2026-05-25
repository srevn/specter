//! Wake-bearing cross-thread sinks to the engine.
//!
//! Two `pub(crate)` primitives:
//!
//! - [`WakeHandle`] — newtype around `Arc<mio::Waker>`. The only call
//!   to [`mio::Waker::new`] in the bin lives in [`WakeHandle::new`],
//!   making mio's "one Waker per Poll" a structural property: a second
//!   construction site requires a fresh `use mio::Waker` import, which
//!   `grep "use mio::Waker"` makes a code-review red flag.
//! - [`WakingSink`] — a `Sender<Input>` paired with a [`WakeHandle`].
//!   The single body of the send-THEN-wake protocol — every wake-bearing
//!   crossing of the cross-thread boundary into the engine routes
//!   through one method ([`WakingSink::send`]).
//!
//! Adapters in [`crate::app`] wrap a `WakingSink` to implement
//! [`specter_sensor::ProberResponseSender`] /
//! [`specter_actuator::EffectCompleteSender`]: each adapter is one
//! content-lift line, the wake-and-send protocol stays single-sourced.
//! A future wake-bearing sink drops in as a third adapter with the
//! same shape — single-Waker stays structural by typing, not by
//! convention.

use crossbeam::channel::Sender;
use mio::{Registry, Token, Waker};
use specter_core::{Input, SendError};
use std::io;
use std::sync::Arc;

/// Cloneable handle to the Hub's single mio Waker.
///
/// The `pub(crate)` constructor [`WakeHandle::new`] is the only call
/// site of [`mio::Waker::new`] in the bin — `grep "Waker::new"`
/// returns one production hit (here). A future contributor adding a
/// second wake-bearing sink either threads a clone of an existing
/// [`WakeHandle`] through, or imports [`mio::Waker`] afresh — the
/// latter is a code-review red flag `grep "use mio::Waker"` makes
/// visible at one grep.
#[derive(Clone, Debug)]
pub(crate) struct WakeHandle(Arc<Waker>);

impl WakeHandle {
    /// Construct the single [`mio::Waker`] for this Poll registry.
    ///
    /// The `token` argument is supplied by the caller so this module
    /// stays token-agnostic — [`crate::driver::hub::DriverHub`] passes
    /// its `TOKEN_WAKER` constant in. Any other Poll registry is a
    /// bug: the project's structural invariant is one Poll per
    /// process; constructing a `WakeHandle` against a foreign
    /// registry would silently miss every wake edge.
    ///
    /// # Errors
    ///
    /// Propagates [`io::Error`] from `Waker::new` — kernel-pressure
    /// (`EMFILE` on the Waker fd) or programmer-error. Callers treat
    /// either as startup-fatal.
    pub(crate) fn new(registry: &Registry, token: Token) -> io::Result<Self> {
        Ok(Self(Arc::new(Waker::new(registry, token)?)))
    }

    /// Pulse the wake edge.
    ///
    /// Errors only when the paired Poll has been destroyed — the
    /// shutdown path; every production caller threads the result
    /// through `let _ = ...` because the next surviving wake (none,
    /// once Poll is gone) is the right reader. The bin's only
    /// production wake source is [`WakingSink::send`] post-channel
    /// send.
    pub(crate) fn wake(&self) -> io::Result<()> {
        self.0.wake()
    }
}

/// Wake-bearing cross-thread sink to the engine's [`Input`] channel.
///
/// Sealed primitive: a [`Sender<Input>`] paired with a [`WakeHandle`].
/// [`Self::send`] runs the send-THEN-wake protocol — the channel must
/// hold the message before the wake edge fires, or the driver's
/// `try_recv` after `Poll::poll` returns sees `Empty` and the message
/// strands until the next unrelated wake.
///
/// Construction requires a [`WakeHandle`], which means routing
/// through [`crate::driver::hub::DriverHub::new`]'s return value. New
/// wake-bearing sinks drop in as `WakingSink::new(tx, handle.clone())`
/// — the mio "one Waker per Poll" invariant stays structural.
pub(crate) struct WakingSink {
    tx: Sender<Input>,
    waker: WakeHandle,
}

impl WakingSink {
    /// Wrap a [`Sender<Input>`] and a [`WakeHandle`] into the
    /// wake-bearing sink. The `waker` argument is typically a
    /// [`WakeHandle::clone`] of the handle returned from
    /// [`crate::driver::hub::DriverHub::new`].
    pub(crate) const fn new(tx: Sender<Input>, waker: WakeHandle) -> Self {
        Self { tx, waker }
    }

    /// Push an [`Input`] onto the channel then pulse the wake edge.
    ///
    /// `tx.send` on the wrapped unbounded crossbeam channel is
    /// non-blocking — it only fails with `Disconnected` once the
    /// driver-side receiver drops, at which point the calling worker
    /// surfaces the error and exits its loop.
    ///
    /// The post-send `wake` is best-effort: it errors only on a
    /// destroyed Poll (the shutdown path), and the next surviving
    /// wake's drain picks up our message — nothing to propagate.
    pub(crate) fn send(&self, input: Input) -> Result<(), SendError> {
        self.tx.send(input).map_err(|_| SendError::Disconnected)?;
        let _ = self.waker.wake();
        Ok(())
    }
}
