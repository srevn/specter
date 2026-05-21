//! Channel topology — the bin's cross-thread plumbing in one place.
//!
//! Senders are cloneable; receivers move into a single consumer thread.
//! [`Channels::new`] allocates every *unconditional* channel pair and
//! distributes halves into per-thread bundles ([`EnginePieces`],
//! [`WatcherSide`], [`ActuatorSide`], [`SignalSide`]). Each bundle
//! partial-moves into its consumer; the discipline is compiler-enforced
//! (no `take_*`, no `Option<Receiver>` storage on the dispenser, no
//! panic-on-second-take).
//!
//! Auto-reload (`config_event`) is **not** allocated here. It is
//! conditional on the config watcher thread spawning successfully, so
//! the channel pair is allocated inline by `App::run` and threaded
//! into [`EngineSide`] via [`EnginePieces::finalize`] — the
//! `config_event_rx` parameter is `Some` iff the watcher spawned.
//! The driver's tick conditions both its drain and its `Select` arm
//! on the resulting `Option<Receiver>`. Under `--no-config-watch` (or
//! a watcher-init failure) the arm never registers, so crossbeam's
//! `Select::ready_timeout` cannot report a non-existent (or
//! disconnected) receiver as immediately-ready — the bug the previous
//! stack-bound keepalive workaround addressed. The absence of the
//! channel is the absence-signal.
//!
//! `bounded(1)` for shutdown / reload coalesces redundant signals at
//! the kernel-queue layer. `bounded(1024)` for `watch_ops` / `effects`
//! holds one large initial-attach burst (hundreds of ops per Sub).
//! Inbound `sensor_in` + `effect_in` are `unbounded` — the driver's
//! `drain_sensor` same-tick coalescing owns the recency horizon.
//!
//! There is no `probe_ops` channel — engine driver calls
//! `Prober::submit/cancel` directly via an `Arc<dyn Prober>` clone.

use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
use specter_core::{Effect, Input, WatchOp};

/// All channel handles for the bin, materialized as per-consumer-thread
/// bundles. [`Channels::new`] allocates every unconditional pair and
/// distributes halves into the bundles below in one move; each field
/// partial-moves into its consumer (watcher / actuator / signal
/// thread) or — for the engine side — first folds through
/// [`EnginePieces::finalize`] to attach the conditional auto-reload
/// receiver.
#[derive(Debug)]
#[must_use]
pub struct Channels {
    pub engine: EnginePieces,
    pub watcher: WatcherSide,
    pub actuator: ActuatorSide,
    pub signal: SignalSide,
}

/// Engine-bound channel halves yielded by [`Channels::new`].
/// `App::run` converts this into [`EngineSide`] via [`Self::finalize`]
/// once the auto-reload decision has landed — the `config_event_rx`
/// parameter is `Some` iff the config watcher thread spawned
/// successfully.
///
/// Distinct type from [`EngineSide`] so the conditional
/// `config_event_rx` edge is a compiler-enforced constructor
/// parameter rather than a post-construction field mutation: the
/// compiler refuses to build an [`EngineSide`] without a decision on
/// auto-reload, and a future refactor can't silently leave the engine
/// running without an arm by skipping a setter.
#[derive(Debug)]
#[must_use]
pub struct EnginePieces {
    pub sensor_in_rx: Receiver<Input>,
    pub effect_in_rx: Receiver<Input>,
    pub reload_signal_rx: Receiver<()>,
    pub shutdown_engine_rx: Receiver<()>,
    pub watch_ops_tx: Sender<WatchOp>,
    pub effects_tx: Sender<Effect>,
}

impl EnginePieces {
    /// Finalize into [`EngineSide`] with the auto-reload decision.
    /// Pass `Some(rx)` when the config watcher thread spawned;
    /// `None` under `--no-config-watch` or `default_config_watcher`
    /// init failure.
    pub fn finalize(self, config_event_rx: Option<Receiver<()>>) -> EngineSide {
        let Self {
            sensor_in_rx,
            effect_in_rx,
            reload_signal_rx,
            shutdown_engine_rx,
            watch_ops_tx,
            effects_tx,
        } = self;
        EngineSide {
            sensor_in_rx,
            effect_in_rx,
            reload_signal_rx,
            shutdown_engine_rx,
            watch_ops_tx,
            effects_tx,
            config_event_rx,
        }
    }
}

/// Receivers + sender clones the engine driver thread owns. Built from
/// [`EnginePieces`] via [`EnginePieces::finalize`] once `App::run`
/// has decided whether to wire auto-reload.
#[derive(Debug)]
#[must_use]
pub struct EngineSide {
    pub sensor_in_rx: Receiver<Input>,
    pub effect_in_rx: Receiver<Input>,
    pub reload_signal_rx: Receiver<()>,
    pub shutdown_engine_rx: Receiver<()>,
    pub watch_ops_tx: Sender<WatchOp>,
    pub effects_tx: Sender<Effect>,
    /// Auto-reload pulse drain — `Some` only when the config watcher
    /// thread spawned, `None` under `--no-config-watch` or a watcher
    /// init failure. The driver's tick gates both its drain loop and
    /// the `Select::ready_timeout` arm on this option, so the absence
    /// of the channel is the structural signal that auto-reload is
    /// off.
    pub config_event_rx: Option<Receiver<()>>,
}

/// Receivers + sender clones the watcher thread owns.
///
/// `sensor_in_tx` is also borrowed by `WorkerProber::new` at startup;
/// the prober pool clones it internally per worker, so the borrow
/// ends before this bundle moves into the watcher thread.
#[derive(Debug)]
#[must_use]
pub struct WatcherSide {
    pub watch_ops_rx: Receiver<WatchOp>,
    pub sensor_in_tx: Sender<Input>,
}

/// Receivers + sender clones the actuator thread owns.
///
/// `hard_shutdown_done_tx` is the back-channel to the signal thread:
/// the actuator pulses it once at the close of phase 3 (SIGKILL
/// fanout), signalling that every running child has been told to die.
/// The signal thread waits on the paired receiver in [`SignalSide`]
/// before calling `process::exit(130)` — without this confirmation,
/// the parent could die while the actuator was still mid-fanout,
/// leaving stubborn children as PID-1 orphans.
#[derive(Debug)]
#[must_use]
pub struct ActuatorSide {
    pub effects_rx: Receiver<Effect>,
    pub shutdown_actuator_rx: Receiver<()>,
    pub hard_shutdown_actuator_rx: Receiver<()>,
    pub effect_in_tx: Sender<Input>,
    pub hard_shutdown_done_tx: Sender<()>,
}

/// Channel halves the signal thread owns. Four senders fan signals
/// out to the engine / actuator / reload pipeline; one receiver
/// observes the actuator's phase-3 confirmation pulse so the hard-exit
/// path can wait for SIGKILL fanout to complete before calling
/// `process::exit(130)`.
#[derive(Debug)]
#[must_use]
pub struct SignalSide {
    pub reload_signal_tx: Sender<()>,
    pub shutdown_engine_tx: Sender<()>,
    pub shutdown_actuator_tx: Sender<()>,
    pub hard_shutdown_actuator_tx: Sender<()>,
    pub hard_shutdown_done_rx: Receiver<()>,
}

/// Sender clone the auto-reload config watcher thread owns.
/// Constructed inline by `App::run` when the config watcher spawns;
/// no factory method on [`Channels`].
///
/// `_tx` postfix is the workspace convention; the
/// `struct_field_names` lint is silenced for that reason.
#[derive(Debug)]
#[must_use]
#[allow(clippy::struct_field_names)]
pub struct ConfigWatcherSide {
    pub config_event_tx: Sender<()>,
}

impl Channels {
    /// Allocate every unconditional channel pair and distribute halves
    /// into per-thread bundles in one move. The auto-reload
    /// `config_event` channel is not allocated here; see `App::run`
    /// for the conditional path.
    ///
    /// The struct itself is `#[must_use]`, which subsumes the
    /// per-function `#[must_use]` (the bundles below carry the same
    /// attribute, so a discarded field is caught at the move site).
    pub fn new() -> Self {
        let (sensor_in_tx, sensor_in_rx) = unbounded();
        let (effect_in_tx, effect_in_rx) = unbounded();
        // Headroom for one large `[[watch]]` block's initial-attach
        // burst (hundreds of WatchOps per Sub).
        let (watch_ops_tx, watch_ops_rx) = bounded(1024);
        // Symmetric headroom against the actuator's per-tick effect
        // emission burst.
        let (effects_tx, effects_rx) = bounded(1024);
        // `bounded(1)` for signal channels — coalesces redundant pulses
        // at the kernel-queue layer (the consumer drains via `try_recv`
        // before the next pulse can land).
        let (reload_signal_tx, reload_signal_rx) = bounded(1);
        let (shutdown_engine_tx, shutdown_engine_rx) = bounded(1);
        let (shutdown_actuator_tx, shutdown_actuator_rx) = bounded(1);
        let (hard_shutdown_actuator_tx, hard_shutdown_actuator_rx) = bounded(1);
        // `bounded(1)`: the actuator emits exactly one pulse per
        // shutdown (after phase 3 SIGKILL fanout). Soft-shutdown emits
        // it too — nobody drains it, the slot fills, no semantic
        // impact. The signal thread drains via `recv_timeout` only on
        // the hard-exit path.
        let (hard_shutdown_done_tx, hard_shutdown_done_rx) = bounded(1);

        Self {
            engine: EnginePieces {
                sensor_in_rx,
                effect_in_rx,
                reload_signal_rx,
                shutdown_engine_rx,
                watch_ops_tx,
                effects_tx,
            },
            watcher: WatcherSide {
                watch_ops_rx,
                sensor_in_tx,
            },
            actuator: ActuatorSide {
                effects_rx,
                shutdown_actuator_rx,
                hard_shutdown_actuator_rx,
                effect_in_tx,
                hard_shutdown_done_tx,
            },
            signal: SignalSide {
                reload_signal_tx,
                shutdown_engine_tx,
                shutdown_actuator_tx,
                hard_shutdown_actuator_tx,
                hard_shutdown_done_rx,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use specter_core::ResourceId;

    #[test]
    fn new_creates_unbounded_inbound() {
        let chans = Channels::new();
        // Sending into an unbounded channel never blocks; verify by
        // pushing many messages without reader and observing no error.
        for _ in 0..2048 {
            chans
                .watcher
                .sensor_in_tx
                .send(Input::TimerExpired {
                    profile: specter_core::ProfileId::default(),
                    kind: specter_core::TimerKind::Settle,
                    id: specter_core::TimerId::default(),
                })
                .expect("unbounded sensor_in_tx send");
        }
        for _ in 0..2048 {
            chans
                .actuator
                .effect_in_tx
                .send(Input::TimerExpired {
                    profile: specter_core::ProfileId::default(),
                    kind: specter_core::TimerKind::Settle,
                    id: specter_core::TimerId::default(),
                })
                .expect("unbounded effect_in_tx send");
        }
    }

    #[test]
    fn new_creates_bounded_watch_ops_at_1024() {
        let chans = Channels::new();
        for _ in 0..1024 {
            chans
                .engine
                .watch_ops_tx
                .try_send(WatchOp::Unwatch {
                    resource: ResourceId::default(),
                })
                .expect("first 1024 fit");
        }
        let result = chans.engine.watch_ops_tx.try_send(WatchOp::Unwatch {
            resource: ResourceId::default(),
        });
        assert!(matches!(
            result,
            Err(crossbeam::channel::TrySendError::Full(_))
        ));
    }

    #[test]
    fn new_creates_bounded_effects_at_1024() {
        use compact_str::CompactString;
        use specter_core::testkit::single_exec_program;
        use specter_core::{
            ArgPart, ArgTemplate, CorrelationId, EffectCommon, ProfileId, ResourceKind, SubId,
        };
        use std::path::PathBuf;
        use std::sync::Arc;
        let chans = Channels::new();
        let dummy = || {
            let common = EffectCommon {
                sub: SubId::default(),
                profile: ProfileId::default(),
                anchor: ResourceId::default(),
                correlation: CorrelationId::default(),
                forced: false,
                capture_output: false,
                sub_name: CompactString::new(""),
                program: single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])]),
                anchor_path: Arc::from(PathBuf::new()),
                anchor_kind: ResourceKind::Dir,
                exclude: Arc::from(Vec::<CompactString>::new()),
            };
            Effect::subtree(common, None)
        };
        for _ in 0..1024 {
            chans
                .engine
                .effects_tx
                .try_send(dummy())
                .expect("first 1024 fit");
        }
        let result = chans.engine.effects_tx.try_send(dummy());
        assert!(matches!(
            result,
            Err(crossbeam::channel::TrySendError::Full(_))
        ));
    }

    #[test]
    fn new_creates_bounded_signal_channels_at_1() {
        let chans = Channels::new();
        chans
            .signal
            .reload_signal_tx
            .try_send(())
            .expect("first slot");
        assert!(matches!(
            chans.signal.reload_signal_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));

        chans
            .signal
            .shutdown_engine_tx
            .try_send(())
            .expect("first slot");
        assert!(matches!(
            chans.signal.shutdown_engine_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));

        chans
            .signal
            .shutdown_actuator_tx
            .try_send(())
            .expect("first slot");
        assert!(matches!(
            chans.signal.shutdown_actuator_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));
    }

    #[test]
    fn new_distributes_clones_across_bundles() {
        // The dispenser-era `take_engine_side_moves_receivers_and_clones_senders`
        // asserted that taking the engine side did not invalidate the
        // dispenser's sender clones. Post-refactor the senders distribute
        // directly into the bundles at construction; this test pins the
        // distribution by sending across the engine ↔ watcher seam.
        let chans = Channels::new();
        // Engine's `watch_ops_tx` clone reaches the watcher's
        // `watch_ops_rx`.
        chans
            .engine
            .watch_ops_tx
            .try_send(WatchOp::Unwatch {
                resource: ResourceId::default(),
            })
            .expect("engine ⇒ watcher send");
        assert!(matches!(
            chans.watcher.watch_ops_rx.try_recv(),
            Ok(WatchOp::Unwatch { .. })
        ));
        // Watcher's `sensor_in_tx` clone reaches the engine's
        // `sensor_in_rx`.
        chans
            .watcher
            .sensor_in_tx
            .send(Input::TimerExpired {
                profile: specter_core::ProfileId::default(),
                kind: specter_core::TimerKind::Settle,
                id: specter_core::TimerId::default(),
            })
            .expect("watcher ⇒ engine send");
        assert!(matches!(
            chans.engine.sensor_in_rx.try_recv(),
            Ok(Input::TimerExpired { .. })
        ));
    }

    #[test]
    fn engine_pieces_finalize_carries_config_event_rx() {
        // `finalize` with `Some(rx)` wires the auto-reload arm; the
        // resulting `EngineSide.config_event_rx` carries that
        // receiver verbatim. Compile-time check: the `Option<Receiver>`
        // shape is the structural signal the driver's tick reads off.
        let chans = Channels::new();
        let (_tx, rx) = bounded::<()>(1);
        let side = chans.engine.finalize(Some(rx));
        assert!(side.config_event_rx.is_some());
    }

    #[test]
    fn engine_pieces_finalize_without_config_event_yields_none() {
        // `finalize(None)` is the `--no-config-watch` / init-failure
        // path: the engine carries no `config_event_rx`, so the
        // driver's tick skips both the drain and the Select arm.
        let chans = Channels::new();
        let side = chans.engine.finalize(None);
        assert!(side.config_event_rx.is_none());
    }
}
