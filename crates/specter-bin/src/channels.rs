//! Channel topology — the bin's cross-thread plumbing in one place.
//!
//! Senders are cloneable (`Sender<T>: Clone`); receivers move into a
//! single consumer thread. Per-thread bundles ([`EngineSide`],
//! [`WatcherSide`], [`ActuatorSide`], [`ConfigWatcherSide`],
//! [`SignalSide`]) are produced via `take_*_side` (one-shot, panics on
//! second call) or sibling clone-only accessors ([`Self::signal_side`],
//! [`Self::config_watcher_side`]) that simply clone the senders their
//! consumer needs. The [`Channels`] struct itself is a one-shot
//! dispenser: once every side has been taken, it can be dropped (the
//! originals release; cloned handles in the threads keep each channel
//! alive).
//!
//! Two-channel inbound (`sensor_in` + `effect_in`) is load-bearing for
//! drain ordering. `bounded(1)` for shutdown / reload / config-event
//! coalesces redundant signals at the kernel-queue layer — a sustained
//! editor burst on `config_event` lands as one pulse the driver
//! debounces via `config_settle_until`, not 1000 redundant try_sends.
//!
//! There is no `probe_ops` channel — engine driver calls
//! `Prober::submit/cancel` directly via an `Arc<dyn Prober>` clone.

use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
use specter_core::{Effect, Input, WatchOp};

/// All channel handles for the bin process. Construct once at startup;
/// drain via the `take_*_side` / clone-side accessors; drop (originals
/// release).
#[derive(Debug)]
pub struct Channels {
    pub sensor_in_tx: Sender<Input>,
    pub effect_in_tx: Sender<Input>,
    pub watch_ops_tx: Sender<WatchOp>,
    pub effects_tx: Sender<Effect>,
    pub reload_signal_tx: Sender<()>,
    pub shutdown_engine_tx: Sender<()>,
    pub shutdown_actuator_tx: Sender<()>,
    /// Hard-shutdown signal — the actuator pre-empts its 5s SIGTERM
    /// grace and goes straight to phase 3 (SIGKILL stragglers). Fired
    /// by the signal thread on second SIGINT/SIGTERM within the
    /// `HARD_EXIT_WINDOW`.
    pub hard_shutdown_actuator_tx: Sender<()>,
    /// Auto-reload pulse channel — the config watcher thread produces
    /// `()` per kernel event observed for the config file or its
    /// parent dir; the engine driver consumes via [`EngineSide`] +
    /// the per-tick drain in `EngineDriver::tick`. `bounded(1)` so a
    /// sustained editor burst coalesces at the channel layer (the
    /// driver-side `config_settle_until` deadline does the time-based
    /// debounce). Distinct from [`Self::reload_signal_tx`] (SIGHUP) so
    /// SIGHUP retains its immediate-handle semantics — both terminate
    /// at `EngineDriver::handle_reload`.
    pub config_event_tx: Sender<()>,

    sensor_in_rx: Option<Receiver<Input>>,
    effect_in_rx: Option<Receiver<Input>>,
    watch_ops_rx: Option<Receiver<WatchOp>>,
    effects_rx: Option<Receiver<Effect>>,
    reload_signal_rx: Option<Receiver<()>>,
    shutdown_engine_rx: Option<Receiver<()>>,
    shutdown_actuator_rx: Option<Receiver<()>>,
    hard_shutdown_actuator_rx: Option<Receiver<()>>,
    config_event_rx: Option<Receiver<()>>,
}

/// Receivers + sender clones the engine driver thread owns.
#[derive(Debug)]
pub struct EngineSide {
    pub sensor_in_rx: Receiver<Input>,
    pub effect_in_rx: Receiver<Input>,
    pub reload_signal_rx: Receiver<()>,
    pub shutdown_engine_rx: Receiver<()>,
    /// Auto-reload pulse — drained per tick (re-arms
    /// `config_settle_until`); also wired into the tick's `Select`
    /// arm so a pulse wakes the driver from a long block.
    pub config_event_rx: Receiver<()>,
    pub watch_ops_tx: Sender<WatchOp>,
    pub effects_tx: Sender<Effect>,
}

/// Receivers + sender clones the watcher thread owns.
#[derive(Debug)]
pub struct WatcherSide {
    pub watch_ops_rx: Receiver<WatchOp>,
    pub sensor_in_tx: Sender<Input>,
}

/// Receivers + sender clones the actuator thread owns.
#[derive(Debug)]
pub struct ActuatorSide {
    pub effects_rx: Receiver<Effect>,
    pub shutdown_actuator_rx: Receiver<()>,
    pub hard_shutdown_actuator_rx: Receiver<()>,
    pub effect_in_tx: Sender<Input>,
}

/// Sender clone the auto-reload config watcher thread owns. The watcher
/// is the sole producer (one `try_send` per kernel event observed for
/// the config file or its parent dir); the engine drains via
/// [`EngineSide::config_event_rx`].
///
/// `_tx` postfix is the workspace convention; the
/// `struct_field_names` lint is silenced for that reason.
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct ConfigWatcherSide {
    pub config_event_tx: Sender<()>,
}

/// Sender clones the signal thread owns. The signal thread never reads
/// from any channel — it only fans signals out to the other actors.
///
/// All fields are senders by design (`_tx` postfix is the workspace
/// convention); the `struct_field_names` lint is silenced for that
/// reason.
#[derive(Debug, Clone)]
#[allow(clippy::struct_field_names)]
pub struct SignalSide {
    pub reload_signal_tx: Sender<()>,
    pub shutdown_engine_tx: Sender<()>,
    pub shutdown_actuator_tx: Sender<()>,
    pub hard_shutdown_actuator_tx: Sender<()>,
}

impl Channels {
    /// Allocate every channel pair with the spec'd bounds.
    #[must_use]
    pub fn new() -> Self {
        let (sensor_in_tx, sensor_in_rx) = unbounded();
        let (effect_in_tx, effect_in_rx) = unbounded();
        let (watch_ops_tx, watch_ops_rx) = bounded(1024);
        let (effects_tx, effects_rx) = bounded(1024);
        let (reload_signal_tx, reload_signal_rx) = bounded(1);
        let (shutdown_engine_tx, shutdown_engine_rx) = bounded(1);
        let (shutdown_actuator_tx, shutdown_actuator_rx) = bounded(1);
        let (hard_shutdown_actuator_tx, hard_shutdown_actuator_rx) = bounded(1);
        let (config_event_tx, config_event_rx) = bounded(1);
        Self {
            sensor_in_tx,
            effect_in_tx,
            watch_ops_tx,
            effects_tx,
            reload_signal_tx,
            shutdown_engine_tx,
            shutdown_actuator_tx,
            hard_shutdown_actuator_tx,
            config_event_tx,
            sensor_in_rx: Some(sensor_in_rx),
            effect_in_rx: Some(effect_in_rx),
            watch_ops_rx: Some(watch_ops_rx),
            effects_rx: Some(effects_rx),
            reload_signal_rx: Some(reload_signal_rx),
            shutdown_engine_rx: Some(shutdown_engine_rx),
            shutdown_actuator_rx: Some(shutdown_actuator_rx),
            hard_shutdown_actuator_rx: Some(hard_shutdown_actuator_rx),
            config_event_rx: Some(config_event_rx),
        }
    }

    /// Move all engine-side receivers + clone the engine's outbound
    /// senders into a single bundle. Panics on second call (the
    /// receivers have already moved).
    pub fn take_engine_side(&mut self) -> EngineSide {
        EngineSide {
            sensor_in_rx: self.sensor_in_rx.take().expect("engine side already taken"),
            effect_in_rx: self.effect_in_rx.take().expect("engine side already taken"),
            reload_signal_rx: self
                .reload_signal_rx
                .take()
                .expect("engine side already taken"),
            shutdown_engine_rx: self
                .shutdown_engine_rx
                .take()
                .expect("engine side already taken"),
            config_event_rx: self
                .config_event_rx
                .take()
                .expect("engine side already taken"),
            watch_ops_tx: self.watch_ops_tx.clone(),
            effects_tx: self.effects_tx.clone(),
        }
    }

    /// Move the watcher's `watch_ops_rx` + clone its `sensor_in_tx`
    /// (used for `Input::FsEvent` and `Input::WatchOpRejected`).
    pub fn take_watcher_side(&mut self) -> WatcherSide {
        WatcherSide {
            watch_ops_rx: self
                .watch_ops_rx
                .take()
                .expect("watcher side already taken"),
            sensor_in_tx: self.sensor_in_tx.clone(),
        }
    }

    /// Move the actuator's `effects_rx` + `shutdown_actuator_rx` and
    /// clone its `effect_in_tx` (used for `Input::EffectComplete`).
    pub fn take_actuator_side(&mut self) -> ActuatorSide {
        ActuatorSide {
            effects_rx: self.effects_rx.take().expect("actuator side already taken"),
            shutdown_actuator_rx: self
                .shutdown_actuator_rx
                .take()
                .expect("actuator side already taken"),
            hard_shutdown_actuator_rx: self
                .hard_shutdown_actuator_rx
                .take()
                .expect("actuator side already taken"),
            effect_in_tx: self.effect_in_tx.clone(),
        }
    }

    /// Clone the signal thread's outbound senders. Idempotent (no
    /// receivers to move); the signal thread can be re-spawned in tests
    /// without re-creating the [`Channels`] dispenser.
    #[must_use]
    pub fn signal_side(&self) -> SignalSide {
        SignalSide {
            reload_signal_tx: self.reload_signal_tx.clone(),
            shutdown_engine_tx: self.shutdown_engine_tx.clone(),
            shutdown_actuator_tx: self.shutdown_actuator_tx.clone(),
            hard_shutdown_actuator_tx: self.hard_shutdown_actuator_tx.clone(),
        }
    }

    /// Clone the config-watcher's outbound sender. Idempotent for the
    /// same reason as [`Self::signal_side`]: the side bundle is
    /// clone-only — the watcher is the producer, the engine drains via
    /// [`EngineSide::config_event_rx`].
    ///
    /// Production calls this exactly once (in `App::run`) and either
    /// hands the bundle to the spawned watcher thread or projects out
    /// `config_event_tx` as a stack-bound keepalive — either way the
    /// engine's `config_event_rx` keeps a live producer (a
    /// disconnected rx would crossbeam-report as immediately-ready
    /// and busy-loop the tick).
    #[must_use]
    pub fn config_watcher_side(&self) -> ConfigWatcherSide {
        ConfigWatcherSide {
            config_event_tx: self.config_event_tx.clone(),
        }
    }
}

impl Default for Channels {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crossbeam::channel::TryRecvError;
    use specter_core::ResourceId;

    #[test]
    fn new_creates_unbounded_inbound() {
        let chans = Channels::new();
        // Sending into an unbounded channel never blocks; verify by
        // pushing many messages without reader and observing no error.
        for _ in 0..2048 {
            chans
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
                .watch_ops_tx
                .try_send(WatchOp::Unwatch {
                    resource: ResourceId::default(),
                })
                .expect("first 1024 fit");
        }
        let result = chans.watch_ops_tx.try_send(WatchOp::Unwatch {
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
        use specter_core::{ArgPart, ArgTemplate, CorrelationId, DedupKey, ResourceKind};
        use std::path::PathBuf;
        use std::sync::Arc;
        let chans = Channels::new();
        let dummy = || Effect {
            key: DedupKey::default(),
            target: ResourceId::default(),
            forced: false,
            correlation: CorrelationId::default(),
            diff: None,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: single_exec_program([ArgTemplate::new([ArgPart::literal("/bin/true")])]),
            anchor_path: Arc::from(PathBuf::new()),
            anchor_kind: ResourceKind::Dir,
            target_relative: CompactString::new(""),
            exclude: Arc::from(Vec::<CompactString>::new()),
        };
        for _ in 0..1024 {
            chans.effects_tx.try_send(dummy()).expect("first 1024 fit");
        }
        let result = chans.effects_tx.try_send(dummy());
        assert!(matches!(
            result,
            Err(crossbeam::channel::TrySendError::Full(_))
        ));
    }

    #[test]
    fn new_creates_bounded_signal_channels_at_1() {
        let chans = Channels::new();
        chans.reload_signal_tx.try_send(()).expect("first slot");
        assert!(matches!(
            chans.reload_signal_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));

        chans.shutdown_engine_tx.try_send(()).expect("first slot");
        assert!(matches!(
            chans.shutdown_engine_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));

        chans.shutdown_actuator_tx.try_send(()).expect("first slot");
        assert!(matches!(
            chans.shutdown_actuator_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));

        // Config-event coalesces redundant pulses at the kernel-queue
        // layer; a sustained editor burst that fills the slot relies on
        // the consumer (driver tick) to drain via try_recv before the
        // next pulse can land.
        chans.config_event_tx.try_send(()).expect("first slot");
        assert!(matches!(
            chans.config_event_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));
    }

    #[test]
    fn take_engine_side_moves_receivers_and_clones_senders() {
        let mut chans = Channels::new();
        let engine = chans.take_engine_side();

        // Senders survive on `chans` (they were cloned, not moved).
        chans
            .watch_ops_tx
            .try_send(WatchOp::Unwatch {
                resource: ResourceId::default(),
            })
            .expect("watch_ops_tx clone alive");
        // Engine's clone of `watch_ops_tx` also survives.
        engine
            .watch_ops_tx
            .try_send(WatchOp::Unwatch {
                resource: ResourceId::default(),
            })
            .expect("engine watch_ops_tx clone alive");

        // Receivers moved out — re-taking panics.
        assert!(chans.sensor_in_rx.is_none());
        assert!(chans.effect_in_rx.is_none());
    }

    #[test]
    #[should_panic(expected = "engine side already taken")]
    fn take_engine_side_panics_on_second_call() {
        let mut chans = Channels::new();
        let _first = chans.take_engine_side();
        let _second = chans.take_engine_side(); // panic
    }

    #[test]
    #[should_panic(expected = "watcher side already taken")]
    fn take_watcher_side_panics_on_second_call() {
        let mut chans = Channels::new();
        let _first = chans.take_watcher_side();
        let _second = chans.take_watcher_side(); // panic
    }

    #[test]
    #[should_panic(expected = "actuator side already taken")]
    fn take_actuator_side_panics_on_second_call() {
        let mut chans = Channels::new();
        let _first = chans.take_actuator_side();
        let _second = chans.take_actuator_side(); // panic
    }

    #[test]
    fn signal_side_is_idempotent_and_clones_senders() {
        let chans = Channels::new();
        let sig1 = chans.signal_side();
        let sig2 = chans.signal_side();
        sig1.reload_signal_tx.try_send(()).expect("first slot");
        // Second clone targets the same channel: bounded(1) is now full.
        assert!(matches!(
            sig2.reload_signal_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));
    }

    #[test]
    fn config_watcher_side_clones_tx_and_is_idempotent() {
        let mut chans = Channels::new();
        // Take the engine side first so we hold a live receiver to assert
        // delivery against (the second-best test signal — that the
        // cloned sender talks to the same underlying channel).
        let engine = chans.take_engine_side();

        let cw1 = chans.config_watcher_side();
        let cw2 = chans.config_watcher_side();
        // Both clones target the same `bounded(1)` slot — the second
        // try_send is Full because the first filled the slot.
        cw1.config_event_tx.try_send(()).expect("first slot");
        assert!(matches!(
            cw2.config_event_tx.try_send(()),
            Err(crossbeam::channel::TrySendError::Full(()))
        ));
        // Engine side observes the pulse — full round-trip across the
        // cloned sender.
        engine
            .config_event_rx
            .try_recv()
            .expect("config_event pulse delivered");
    }

    #[test]
    fn take_engine_side_moves_config_event_rx() {
        let mut chans = Channels::new();
        let engine = chans.take_engine_side();
        // Receiver has moved out of `chans` — the original slot is None.
        assert!(chans.config_event_rx.is_none());
        // The engine-side rx is alive: a sender clone can still talk
        // to it (the dispenser's original tx held by `chans` keeps the
        // channel open until `drop(chans)`).
        let cw = chans.config_watcher_side();
        cw.config_event_tx
            .try_send(())
            .expect("send across channel");
        engine
            .config_event_rx
            .try_recv()
            .expect("engine receives pulse");
    }

    #[test]
    fn dropping_channels_keeps_taken_senders_alive() {
        // Cross-side flow survives Channels drop: a sensor_in_tx clone
        // taken with `WatcherSide` can send to the engine side's
        // sensor_in_rx even after the dispenser drops.
        let watcher_side;
        let engine_side;
        {
            let mut chans = Channels::new();
            watcher_side = chans.take_watcher_side();
            engine_side = chans.take_engine_side();
            // chans drops here; original sender / receiver halves release.
        }
        watcher_side
            .sensor_in_tx
            .send(Input::TimerExpired {
                profile: specter_core::ProfileId::default(),
                kind: specter_core::TimerKind::Settle,
                id: specter_core::TimerId::default(),
            })
            .expect("sensor_in_tx clone outlives Channels");
        // Engine side's receiver picks the message up.
        assert!(matches!(
            engine_side.sensor_in_rx.try_recv(),
            Ok(Input::TimerExpired { .. }),
        ));
        // No watch_ops were sent — channel is empty (not disconnected; the
        // engine_side still holds a watch_ops_tx clone too).
        assert!(matches!(
            watcher_side.watch_ops_rx.try_recv(),
            Err(TryRecvError::Empty),
        ));
    }
}
