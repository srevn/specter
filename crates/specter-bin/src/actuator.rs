//! Driver-actuator channel topology.
//!
//! The driver thread owns every kernel-side fd directly via
//! [`crate::driver::Reactor`], so cross-thread coordination
//! is limited to the two seams where blocking syscalls cannot collapse
//! onto the reactor: the engine ↔ actuator seam (this module — the
//! actuator thread spawns subprocesses and waits on `waitpid`
//! synchronously) and the engine ↔ prober seam (see below — the
//! prober pool's workers block on `lstat` / `readdir` during the
//! directory walk). Each seam pulses the Reactor's
//! [`crate::driver::WakeHandle`] (the bin's sole [`mio::Waker`]) to
//! lift the reactor out of `Poll::poll` when a response is ready.
//!
//! Two channel-bundle types pair through one [`ActuatorIO::pair`]
//! call:
//!
//! - [`ActuatorIO`] — driver-side handles. Lives on
//!   [`crate::driver::EngineDriver`] for the daemon's lifetime;
//!   [`crate::driver::EngineDriver::dispatch_signal_with_exit_fn`]
//!   pulses `shutdown_actuator_tx` / `hard_shutdown_actuator_tx` and
//!   waits on `hard_shutdown_done_rx`. `effects_tx` carries every
//!   emitted [`EffectOp`].
//! - [`RunWiring`] — actuator-thread handles. Owned by the actuator's
//!   [`specter_actuator::SubprocessActuator::run`] for the lifetime
//!   of the call. There is no bin-side mirror name: the actuator's
//!   own contract is the one struct that crosses the seam.
//!
//! `effect_complete_tx` is NOT in either struct — it's lifted into
//! a [`specter_actuator::EffectCompleteSender`] trait object (the
//! bin's `WakingEffectCompleteSender`) at `App::run`'s wiring
//! point so the actuator never names [`specter_core::Input`]. The
//! trait object is passed alongside the [`RunWiring`] into the
//! actuator-thread spawn.
//!
//! Prober traffic does NOT live here. The prober's response channel
//! pairs the driver's [`crate::driver::Reactor`]
//! `prober_response_rx` with the bin's
//! `WakingProberResponseSender` wrapper — the pair is
//! allocated inline in `App::run` because both halves are wrapped
//! before they cross any constructor boundary.
//!
//! # Capacities
//!
//! - `effects_tx` is **`bounded(1024)`** — headroom for one large
//!   initial-attach burst (hundreds of effects per Sub on a fresh
//!   daemon). The driver's [`crate::driver::EngineDriver::forward`]
//!   uses `try_send` with advisory drop on `Full` (the engine's
//!   `gate_deadline` recovery covers the missed Submit); `Disconnected`
//!   is terminal.
//! - **Shutdown legs** are `bounded(1)` — coalesces redundant pulses
//!   at the kernel queue layer; the consumer drains via `try_recv`
//!   before the next pulse can land.

use crossbeam::channel::{Receiver, Sender, bounded};
use specter_actuator::RunWiring;
use specter_core::EffectOp;

/// Driver-side actuator-coordination channel bundle.
///
/// Holds the four channel halves the driver thread uses to talk to
/// the actuator: the effects pipe and the three shutdown-handshake
/// legs (soft pulse, hard pulse, confirm-receive). Constructed by
/// [`Self::pair`] paired with [`RunWiring`].
///
/// Threaded into [`crate::driver::EngineDriver::new`] at `App::run`
/// time; the soft / hard / confirm legs are pulsed from
/// [`crate::driver::EngineDriver::dispatch_signal_with_exit_fn`] on
/// observed SIGINT / SIGTERM.
#[derive(Debug)]
#[must_use]
pub struct ActuatorIO {
    /// Effects pipe. The driver dispatches every emitted
    /// [`EffectOp`] here; the actuator's controller drains it via
    /// `select!` against its shutdown legs. `bounded(1024)` — see
    /// the module rustdoc for the cap rationale.
    pub effects_tx: Sender<EffectOp>,
    /// Soft-shutdown pulse. The driver pulses once on the first
    /// SIGINT / SIGTERM observation; the actuator's controller drains
    /// this to enter its graceful-stop arm (SIGTERM-then-wait fanout
    /// with a grace window).
    pub shutdown_actuator_tx: Sender<()>,
    /// Hard-shutdown pulse. The driver pulses on the second
    /// SIGINT / SIGTERM within
    /// [`crate::signals::HARD_EXIT_WINDOW`]. The actuator's
    /// controller pre-empts its grace window and runs SIGKILL fanout
    /// against every running child.
    pub hard_shutdown_actuator_tx: Sender<()>,
    /// Hard-shutdown confirmation receiver. The actuator pulses once
    /// after SIGKILL fanout completes. The driver's hard-exit path
    /// waits on this (bounded by
    /// [`crate::signals::HARD_SHUTDOWN_CONFIRM_TIMEOUT`])
    /// before calling [`std::process::exit`] so the parent doesn't
    /// abort while children are still being signaled.
    pub hard_shutdown_done_rx: Receiver<()>,
}

impl ActuatorIO {
    /// Allocate the four channel pairs and distribute halves into
    /// the driver-side ([`ActuatorIO`]) and the actuator's owned
    /// [`RunWiring`] in one move.
    pub fn pair() -> (Self, RunWiring) {
        let (effects_tx, effects_rx) = bounded::<EffectOp>(1024);
        let (shutdown_actuator_tx, shutdown_rx) = bounded::<()>(1);
        let (hard_shutdown_actuator_tx, hard_shutdown_rx) = bounded::<()>(1);
        let (hard_shutdown_done_tx, hard_shutdown_done_rx) = bounded::<()>(1);
        (
            Self {
                effects_tx,
                shutdown_actuator_tx,
                hard_shutdown_actuator_tx,
                hard_shutdown_done_rx,
            },
            RunWiring {
                effects_rx,
                shutdown_rx,
                hard_shutdown_rx,
                hard_shutdown_done_tx,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel::TrySendError;

    /// [`ActuatorIO::pair`] yields a `bounded(1024)` effects channel.
    /// Pins the cap against accidental relaxation; the driver's
    /// `try_send` policy and the actuator's drain cadence are
    /// calibrated around this width.
    #[test]
    fn pair_creates_bounded_effects_at_1024() {
        let (io, wiring) = ActuatorIO::pair();
        // Saturate via `try_send`s and assert the 1025th rejects.
        for _ in 0..1024 {
            io.effects_tx
                .try_send(EffectOp::Cancel {
                    profile: specter_core::ProfileId::default(),
                })
                .expect("first 1024 fit");
        }
        let next = io.effects_tx.try_send(EffectOp::Cancel {
            profile: specter_core::ProfileId::default(),
        });
        assert!(matches!(next, Err(TrySendError::Full(_))));
        // Sender → Receiver carries the EffectOp verbatim across the
        // bundle seam.
        assert!(matches!(
            wiring.effects_rx.try_recv(),
            Ok(EffectOp::Cancel { .. })
        ));
    }

    /// All three shutdown legs are `bounded(1)` — redundant pulses
    /// coalesce at the channel layer. A second `try_send` on a
    /// pending slot returns `Full` rather than queueing.
    #[test]
    fn pair_creates_bounded_shutdown_legs_at_1() {
        let (io, _wiring) = ActuatorIO::pair();
        io.shutdown_actuator_tx
            .try_send(())
            .expect("first slot fits");
        assert!(matches!(
            io.shutdown_actuator_tx.try_send(()),
            Err(TrySendError::Full(()))
        ));

        io.hard_shutdown_actuator_tx
            .try_send(())
            .expect("first slot fits");
        assert!(matches!(
            io.hard_shutdown_actuator_tx.try_send(()),
            Err(TrySendError::Full(()))
        ));
    }

    /// The actuator-side `hard_shutdown_done_tx` pulse reaches the
    /// driver-side `hard_shutdown_done_rx`. Pins the confirmation
    /// edge the hard-exit path relies on.
    #[test]
    fn pair_routes_hard_shutdown_confirmation() {
        let (io, wiring) = ActuatorIO::pair();
        wiring
            .hard_shutdown_done_tx
            .try_send(())
            .expect("actuator can pulse the confirm leg");
        assert!(io.hard_shutdown_done_rx.try_recv().is_ok());
    }
}
