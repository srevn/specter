//! Test fixtures behind the `testkit` Cargo feature.
//!
//! [`MockSpawner`] is a pure-Rust [`crate::Spawner`] that records every
//! `spawn` call into a `Mutex<Vec<SpawnRecord>>` and returns handles
//! whose `wait` blocks on a per-call channel until the test signals
//! [`MockSpawner::complete`]. The signaler records `signal_term` /
//! `signal_kill` calls. This lets coalescing, concurrency, and
//! shutdown logic be tested deterministically without forking real
//! children.

use crate::spawner::{ChildSignaler, ChildWaiter, EnvVar, SpawnHandles, Spawner};
use crossbeam::channel::{Receiver, Sender, bounded};
use specter_core::EffectOutcome;
use std::collections::BTreeMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
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

/// Recorded signal call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SignalRecord {
    Term(u32),
    Kill(u32),
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
        let pid = self.next_pid.fetch_add(1, Ordering::SeqCst);
        let (tx, rx) = bounded::<EffectOutcome>(1);
        self.completions.lock().unwrap().insert(pid, tx);
        // SpawnRecord stores owned `(String, String)` so existing test
        // assertions compare against literal keys/values without
        // tracking the borrow lifetime; one owning hop at the test
        // boundary is the price of preserving the trait's borrow shape.
        let env_owned: Vec<(String, String)> = env
            .iter()
            .map(|e| (e.key.to_owned(), e.value.as_ref().to_owned()))
            .collect();
        self.spawns.lock().unwrap().push(SpawnRecord {
            pid,
            argv: argv.to_vec(),
            env: env_owned,
            cwd: cwd.to_owned(),
            capture_output,
        });
        let dead = Arc::new(AtomicBool::new(false));
        Ok(SpawnHandles {
            pid,
            waiter: Box::new(MockChildWaiter {
                rx,
                dead: Arc::clone(&dead),
            }),
            signaler: Box::new(MockChildSignaler {
                pid,
                dead,
                signals: Arc::clone(&self.signals),
            }),
        })
    }
}

struct MockChildWaiter {
    rx: Receiver<EffectOutcome>,
    dead: Arc<AtomicBool>,
}

impl ChildWaiter for MockChildWaiter {
    fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
        let result = self.rx.recv();
        // Mirror OsChildWaiter — set dead unconditionally before
        // returning so the protocol contract holds even on error.
        self.dead.store(true, Ordering::SeqCst);
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
    dead: Arc<AtomicBool>,
    signals: Arc<Mutex<Vec<SignalRecord>>>,
}

impl ChildSignaler for MockChildSignaler {
    fn signal_term(&self) -> io::Result<()> {
        if self.dead.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.signals
            .lock()
            .unwrap()
            .push(SignalRecord::Term(self.pid));
        Ok(())
    }
    fn signal_kill(&self) -> io::Result<()> {
        if self.dead.load(Ordering::SeqCst) {
            return Ok(());
        }
        self.signals
            .lock()
            .unwrap()
            .push(SignalRecord::Kill(self.pid));
        Ok(())
    }
}
