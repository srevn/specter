//! Test fixtures behind the `testkit` Cargo feature.
//!
//! [`MockSpawner`] is a pure-Rust [`crate::Spawner`] that records every
//! `spawn` call into a `Mutex<Vec<SpawnRecord>>` and returns handles
//! whose `wait` blocks on a per-call channel until the test signals
//! [`MockSpawner::complete`]. The signaler records `signal_term` /
//! `signal_kill` calls. This lets coalescing, concurrency, and
//! shutdown logic be tested deterministically without forking real
//! children.

use crate::lifecycle::DeadFlag;
use crate::spawner::{
    ChildSignaler, ChildWaiter, EnvVar, PipeSpawnHandles, SpawnHandles, Spawner, StageSpec,
};
use crossbeam::channel::{Receiver, Sender, bounded};
use specter_core::EffectOutcome;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::{Arc, Mutex};

/// Recorded spawn — what the controller passed to
/// [`Spawner::spawn`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpawnRecord {
    pub pid: u32,
    pub argv: Vec<String>,
    pub env: Vec<(String, String)>,
    pub cwd: PathBuf,
    pub capture_output: bool,
}

/// Recorded signal call. `Reap` records the synchronous reap path
/// taken on wait-thread-spawn-failure recovery.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalRecord {
    Term(u32),
    Kill(u32),
    Reap(u32),
}

/// Pure-Rust [`Spawner`] test fixture.
#[derive(Debug)]
pub struct MockSpawner {
    next_pid: AtomicU32,
    spawns: Mutex<Vec<SpawnRecord>>,
    signals: Arc<Mutex<Vec<SignalRecord>>>,
    /// Per-pid completion sender; the waiter reads from the
    /// corresponding receiver. Tests inject outcomes via `complete`.
    completions: Mutex<BTreeMap<u32, Sender<EffectOutcome>>>,
    /// If set, every `spawn` returns this error instead of recording the
    /// call. Used to test the spawn-fail synthesis path.
    inject_spawn_error: Mutex<Option<io::ErrorKind>>,
}

impl Default for MockSpawner {
    fn default() -> Self {
        Self {
            next_pid: AtomicU32::new(10000),
            spawns: Mutex::new(Vec::new()),
            signals: Arc::new(Mutex::new(Vec::new())),
            completions: Mutex::new(BTreeMap::new()),
            inject_spawn_error: Mutex::new(None),
        }
    }
}

impl MockSpawner {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drain and return the recorded spawn calls.
    pub fn take_spawns(&self) -> Vec<SpawnRecord> {
        std::mem::take(&mut self.spawns.lock().unwrap())
    }

    /// Snapshot the recorded spawn calls without draining.
    pub fn spawns(&self) -> Vec<SpawnRecord> {
        self.spawns.lock().unwrap().clone()
    }

    /// Drain and return the recorded signal calls.
    pub fn take_signals(&self) -> Vec<SignalRecord> {
        std::mem::take(&mut self.signals.lock().unwrap())
    }

    /// Snapshot the recorded signal calls without draining.
    pub fn signals(&self) -> Vec<SignalRecord> {
        self.signals.lock().unwrap().clone()
    }

    /// Cause the spawned child with `pid` to return `outcome` from its
    /// `wait`. Returns `Err` if `pid` was never spawned or if the
    /// waiter is no longer listening.
    pub fn complete(&self, pid: u32, outcome: EffectOutcome) -> Result<(), &'static str> {
        // Clone the Sender so the lock is released before send (avoids
        // holding the Mutex across the channel operation).
        let tx = self
            .completions
            .lock()
            .unwrap()
            .get(&pid)
            .cloned()
            .ok_or("unknown pid")?;
        tx.send(outcome).map_err(|_| "waiter dropped channel")
    }

    /// Configure subsequent `spawn` calls to return the given error
    /// kind.
    pub fn inject_spawn_error(&self, kind: io::ErrorKind) {
        *self.inject_spawn_error.lock().unwrap() = Some(kind);
    }

    /// Clear any previously-set spawn-error injection.
    pub fn clear_spawn_error(&self) {
        *self.inject_spawn_error.lock().unwrap() = None;
    }
}

impl MockSpawner {
    /// Allocate one spawn slot: mint a pid, register the completion
    /// channel, record the [`SpawnRecord`]. Returns the primitives the
    /// caller assembles into per-stage waiter + signaler pair shapes
    /// — `Box<dyn>` for single-spawn (`Self::spawn`) or
    /// `Arc<dyn ChildSignaler>` for pipe stages (`Self::spawn_pipe`).
    ///
    /// Centralising the id minting / channel bookkeeping here keeps
    /// `MockSpawner::complete(pid, outcome)` a single contract: any
    /// pid returned from `take_spawns()` / `spawns()` is wired into
    /// exactly one in-flight completion channel.
    fn allocate_spawn(
        &self,
        argv: Vec<String>,
        env: Vec<(String, String)>,
        cwd: PathBuf,
        capture_output: bool,
    ) -> (u32, Receiver<EffectOutcome>, DeadFlag) {
        let pid = self.next_pid.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = bounded::<EffectOutcome>(1);
        self.completions.lock().unwrap().insert(pid, tx);
        self.spawns.lock().unwrap().push(SpawnRecord {
            pid,
            argv,
            env,
            cwd,
            capture_output,
        });
        (pid, rx, DeadFlag::new())
    }

    /// Owned `Vec<(String, String)>` mirror of the spawner's
    /// borrowed env slice. Shared by [`Self::spawn`] and
    /// [`Self::spawn_pipe`] so the recorded env shape is stable.
    fn env_to_owned(env: &[EnvVar<'_>]) -> Vec<(String, String)> {
        env.iter()
            .map(|e| (e.key.to_owned(), e.value.as_ref().to_owned()))
            .collect()
    }
}

impl Spawner for MockSpawner {
    fn spawn(
        &self,
        argv: &[String],
        env: &[EnvVar<'_>],
        cwd: &Path,
        capture_output: bool,
    ) -> io::Result<SpawnHandles> {
        // Copy out of the lock before checking — Mutex guard's
        // significant Drop should not span the if-let body.
        let injected = *self.inject_spawn_error.lock().unwrap();
        if let Some(kind) = injected {
            return Err(io::Error::from(kind));
        }
        let (pid, rx, dead) = self.allocate_spawn(
            argv.to_vec(),
            Self::env_to_owned(env),
            cwd.to_owned(),
            capture_output,
        );
        Ok(SpawnHandles {
            pid,
            waiter: Box::new(MockChildWaiter {
                rx,
                dead: dead.clone(),
            }),
            signaler: Arc::new(MockChildSignaler {
                pid,
                dead,
                signals: Arc::clone(&self.signals),
            }),
        })
    }

    fn spawn_pipe(
        &self,
        stages: &[StageSpec<'_>],
        cwd: &Path,
        capture_output: bool,
    ) -> io::Result<PipeSpawnHandles> {
        // Same injection point as `spawn` — tests can flip the flag
        // and observe a pipe spawn failure as a stage-0 failure.
        let injected = *self.inject_spawn_error.lock().unwrap();
        if let Some(kind) = injected {
            return Err(io::Error::from(kind));
        }
        if stages.len() < 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "MockSpawner::spawn_pipe requires at least two stages",
            ));
        }
        let mut stage_waiters: Vec<Box<dyn ChildWaiter>> = Vec::with_capacity(stages.len());
        let mut stage_signalers: Vec<Arc<dyn ChildSignaler>> = Vec::with_capacity(stages.len());
        let mut last_pid: u32 = 0;
        for stage in stages {
            let (pid, rx, dead) = self.allocate_spawn(
                stage.argv.to_vec(),
                Self::env_to_owned(stage.env),
                cwd.to_owned(),
                capture_output,
            );
            last_pid = pid;
            stage_waiters.push(Box::new(MockChildWaiter {
                rx,
                dead: dead.clone(),
            }));
            stage_signalers.push(Arc::new(MockChildSignaler {
                pid,
                dead,
                signals: Arc::clone(&self.signals),
            }));
        }
        // Mirror `OsSpawner::spawn_pipe`: one `Arc<[_]>` backs the
        // aggregating waiter, the combined signaler, and the
        // controller's `stage_signalers` handle. Tests reading
        // `handles.stage_signalers.len()` see the same shape as
        // production.
        let stage_signalers: Arc<[Arc<dyn ChildSignaler>]> = Arc::from(stage_signalers);
        let combined: Arc<dyn ChildSignaler> = Arc::new(crate::pipe::CombinedSignaler::new(
            Arc::clone(&stage_signalers),
        ));
        let waiter: Box<dyn ChildWaiter> = Box::new(crate::pipe::PipeWaiter::new(
            stage_waiters,
            Arc::clone(&stage_signalers),
        ));
        Ok(PipeSpawnHandles {
            last_pid,
            waiter,
            combined_signaler: combined,
            stage_signalers,
        })
    }
}

struct MockChildWaiter {
    rx: Receiver<EffectOutcome>,
    dead: DeadFlag,
}

impl ChildWaiter for MockChildWaiter {
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
        let result = self.rx.recv();
        // Mirror OsChildWaiter — mark dead unconditionally before
        // returning so the protocol contract holds even on error.
        self.dead.mark_dead();
        match result {
            Ok(o) => Ok(o),
            Err(_) => Err(io::Error::new(
                io::ErrorKind::BrokenPipe,
                "completion channel dropped",
            )),
        }
    }
}

struct MockChildSignaler {
    pid: u32,
    dead: DeadFlag,
    signals: Arc<Mutex<Vec<SignalRecord>>>,
}

impl ChildSignaler for MockChildSignaler {
    fn signal_term(&self) -> io::Result<()> {
        if self.dead.is_dead() {
            return Ok(());
        }
        self.signals
            .lock()
            .unwrap()
            .push(SignalRecord::Term(self.pid));
        Ok(())
    }
    fn signal_kill(&self) -> io::Result<()> {
        if self.dead.is_dead() {
            return Ok(());
        }
        self.signals
            .lock()
            .unwrap()
            .push(SignalRecord::Kill(self.pid));
        Ok(())
    }
    fn reap_blocking(&self) -> io::Result<()> {
        if self.dead.is_dead() {
            return Ok(());
        }
        self.signals
            .lock()
            .unwrap()
            .push(SignalRecord::Reap(self.pid));
        self.dead.mark_dead();
        Ok(())
    }
    fn is_dead(&self) -> bool {
        self.dead.is_dead()
    }
    fn mark_dead(&self) {
        // Trait-level publish — delegates to the shared DeadFlag the
        // MockChildWaiter writes after its rx.recv() returns. Idempotent
        // against the waiter's own mark and against reap_blocking.
        self.dead.mark_dead();
    }
}
