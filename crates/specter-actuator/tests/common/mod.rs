//! Shared test fixtures for the integration suite. Real subprocesses
//! via [`OsSpawner`].

#![allow(
    dead_code,
    clippy::manual_assert,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::redundant_clone,
    clippy::useless_conversion,
    clippy::wildcard_enum_match_arm
)]

use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
use specter_actuator::{OsSpawner, SubprocessActuator};
use specter_core::{
    CommandResolved, CorrelationId, DedupKey, Effect, Input, ProfileId, ResourceId, SubId,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Process-wide monotonic correlation counter for tests. Production
/// correlations come from the engine; tests need their own unique
/// stream so parallel tests within one binary don't collide on the
/// actuator's `(pid, correlation)`-keyed tmp file path.
static NEXT_CORR: AtomicU64 = AtomicU64::new(0xdead_0000);

pub fn next_corr() -> u64 {
    NEXT_CORR.fetch_add(1, Ordering::SeqCst)
}

pub fn unique_sub_id(seed: u64) -> SubId {
    use slotmap::KeyData;
    SubId::from(KeyData::from_ffi(seed))
}

pub fn unique_resource_id(seed: u64) -> ResourceId {
    use slotmap::KeyData;
    ResourceId::from(KeyData::from_ffi(seed))
}

pub fn unique_profile_id(seed: u64) -> ProfileId {
    use slotmap::KeyData;
    ProfileId::from(KeyData::from_ffi(seed))
}

pub struct Harness {
    pub effects_tx: Sender<Effect>,
    pub shutdown_tx: Sender<()>,
    pub hard_shutdown_tx: Sender<()>,
    pub engine_in: Receiver<Input>,
    pub join: Option<thread::JoinHandle<()>>,
}

impl Harness {
    pub fn new(concurrency: usize) -> Self {
        let (effects_tx, effects_rx) = bounded::<Effect>(1024);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
        let (hard_shutdown_tx, hard_shutdown_rx) = bounded::<()>(1);
        let (engine_tx, engine_rx) = unbounded::<Input>();
        let join = thread::Builder::new()
            .name("test-actuator-controller".into())
            .spawn(move || {
                let spawner = Arc::new(OsSpawner::new());
                let mut a = SubprocessActuator::new(concurrency);
                a.run(
                    effects_rx,
                    shutdown_rx,
                    hard_shutdown_rx,
                    engine_tx,
                    spawner.as_ref(),
                );
            })
            .expect("spawn controller");
        Self {
            effects_tx,
            shutdown_tx,
            hard_shutdown_tx,
            engine_in: engine_rx,
            join: Some(join),
        }
    }

    pub fn submit(&self, e: Effect) {
        self.effects_tx.send(e).expect("submit");
    }

    pub fn shutdown(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(j) = self.join.take() {
            j.join().expect("controller join");
        }
    }

    pub fn wait_for_effect_completes(&self, n: usize, dur: Duration) -> Vec<Input> {
        let deadline = Instant::now() + dur;
        let mut received = Vec::new();
        while received.len() < n {
            let now = Instant::now();
            if now >= deadline {
                panic!("expected {n} EffectCompletes; saw {}", received.len());
            }
            match self.engine_in.recv_timeout(deadline - now) {
                Ok(i) => received.push(i),
                Err(crossbeam::channel::RecvTimeoutError::Timeout) => {
                    panic!(
                        "timeout waiting for EffectCompletes; saw {}",
                        received.len()
                    )
                }
                Err(crossbeam::channel::RecvTimeoutError::Disconnected) => break,
            }
        }
        received
    }
}

impl Drop for Harness {
    fn drop(&mut self) {
        if self.join.is_some() {
            self.shutdown();
        }
    }
}

/// Build a PerFile Effect with a literal `argv` and the given correlation.
pub fn perfile_effect(
    sub_seed: u64,
    res_seed: u64,
    corr: u64,
    argv: Vec<String>,
    cwd: PathBuf,
) -> Effect {
    Effect {
        key: DedupKey::PerFile {
            sub: unique_sub_id(sub_seed),
            resource: unique_resource_id(res_seed),
        },
        command: CommandResolved { argv },
        env: Vec::new(),
        cwd,
        forced: false,
        correlation: CorrelationId(corr),
        diff: None,
    }
}

/// Build a Subtree Effect with a literal `argv`.
pub fn subtree_effect(
    sub_seed: u64,
    profile_seed: u64,
    corr: u64,
    argv: Vec<String>,
    cwd: PathBuf,
) -> Effect {
    Effect {
        key: DedupKey::Subtree {
            sub: unique_sub_id(sub_seed),
            profile: unique_profile_id(profile_seed),
        },
        command: CommandResolved { argv },
        env: Vec::new(),
        cwd,
        forced: false,
        correlation: CorrelationId(corr),
        diff: None,
    }
}
