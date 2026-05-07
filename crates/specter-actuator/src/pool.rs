//! Subprocess pool controller — single thread, drains submits + reaps +
//! shutdown, owns slot state.
//!
//! Channel topology:
//!
//! ```text
//! bin --(effects, bounded(1024))--> Controller
//! Controller --(engine_inbound, unbounded)--> Engine
//! Controller <--(reap_rx, bounded(64))-- WaitThread × N
//! bin --(shutdown, bounded(1) broadcast)--> Controller
//! ```
//!
//! Shutdown sequence: SIGTERM all running, drain reaps for 5s,
//! SIGKILL stragglers, drain remaining reaps.

mod state;
use crate::spawner::Spawner;
use crossbeam::channel::{Receiver, Sender};
use specter_core::{CorrelationId, DedupKey, Effect, EffectOutcome, Input, SubId};
use state::ActuatorState;
use std::num::NonZeroUsize;
use std::time::{Duration, Instant};

/// Sentinel for "default concurrency" passed to [`SubprocessActuator::new`].
/// Resolved to `2 * num_cpus` (or `4` if `num_cpus` is unavailable). The
/// bin's CLI flag passes a non-zero value when set.
pub const DEFAULT_CONCURRENCY: usize = 0;

/// Signal from a wait thread back to the controller.
#[derive(Debug)]
pub struct Reaped {
    pub key: DedupKey,
    pub sub: SubId,
    pub correlation: CorrelationId,
    pub outcome: EffectOutcome,
}

/// The actuator's controller. One per process. Owns the slot map, ready
/// queue, per-Sub counter, and global semaphore. Blocks in [`Self::run`]
/// for the lifetime of the bin process.
#[derive(Debug)]
pub struct SubprocessActuator {
    state: ActuatorState,
    reap_tx: Sender<Reaped>,
    reap_rx: Receiver<Reaped>,
    shutdown_grace: Duration,
}

impl SubprocessActuator {
    /// Construct with `concurrency` global permits. Pass
    /// [`DEFAULT_CONCURRENCY`] (`0`) to resolve to `2 * num_cpus`; non-zero
    /// values pass through. The `0`-sentinel is the only place this
    /// crate resolves "default concurrency"; everything below
    /// [`ActuatorState::new`] receives a [`NonZeroUsize`] and trusts it.
    #[must_use]
    pub fn new(concurrency: usize) -> Self {
        let fallback = NonZeroUsize::new(4).expect("4 is non-zero");
        let resolved = NonZeroUsize::new(concurrency).unwrap_or_else(|| {
            std::thread::available_parallelism()
                .ok()
                .and_then(|n| NonZeroUsize::new(n.get().saturating_mul(2)))
                .unwrap_or(fallback)
        });
        let (reap_tx, reap_rx) = crossbeam::channel::bounded::<Reaped>(64);
        Self {
            state: ActuatorState::new(resolved),
            reap_tx,
            reap_rx,
            shutdown_grace: Duration::from_secs(5),
        }
    }

    /// Test-only constructor with a custom shutdown grace.
    ///
    /// Gated to match the test module (`cfg(all(test, feature = "testkit"))`)
    /// — without `testkit`, the test module that consumes this constructor
    /// is excluded too, so the function would otherwise be flagged as
    /// dead code under `cargo test --lib` (no features).
    #[cfg(all(test, feature = "testkit"))]
    pub(crate) fn new_with_grace(concurrency: usize, grace: Duration) -> Self {
        let mut s = Self::new(concurrency);
        s.shutdown_grace = grace;
        s
    }

    /// Block until shutdown. Drains effects, dispatches to spawner,
    /// reaps wait threads, propagates [`Input::EffectComplete`].
    /// Returns when `effects_rx` disconnects or `shutdown_rx` signals;
    /// performs the SIGTERM → 5s grace → SIGKILL sequence on the
    /// way out. If `hard_shutdown_rx` fires (operator pressed Ctrl-C
    /// twice within `HARD_EXIT_WINDOW`), the grace is pre-empted: the
    /// loop breaks immediately, the SIGTERM phase still runs (cheap;
    /// gives well-behaved children a chance to exit cleanly), then
    /// phase 2's grace becomes a near-zero wait before phase 3 SIGKILLs
    /// everything still alive.
    ///
    /// Channels are taken by value: the controller owns them for the
    /// lifetime of [`Self::run`], so the caller hands off and is freed
    /// from any borrow-tracking.
    #[allow(clippy::needless_pass_by_value)]
    pub fn run(
        &mut self,
        effects_rx: Receiver<Effect>,
        shutdown_rx: Receiver<()>,
        hard_shutdown_rx: Receiver<()>,
        engine_in: Sender<Input>,
        spawner: &dyn Spawner,
    ) {
        let mut hard = false;
        loop {
            crossbeam::select! {
                recv(effects_rx) -> msg => match msg {
                    Ok(effect) => self.state.handle_submit(effect, spawner, &self.reap_tx, &engine_in),
                    Err(_)     => break, // bin closed channel
                },
                recv(self.reap_rx) -> msg => match msg {
                    Ok(r)  => self.state.handle_reap(r, &engine_in, spawner, &self.reap_tx),
                    Err(_) => {
                        // Controller holds reap_tx, so the rx cannot disconnect under
                        // current invariants. Logged break preserves orderly shutdown
                        // if a future refactor reshuffles ownership.
                        tracing::error!("reap channel disconnected; controller invariant broken");
                        break;
                    }
                },
                recv(shutdown_rx) -> _ => break,
                recv(hard_shutdown_rx) -> _ => { hard = true; break; }
            }
        }
        self.shutdown(&engine_in, hard, &hard_shutdown_rx);
    }

    fn shutdown(&mut self, engine_in: &Sender<Input>, hard: bool, hard_shutdown_rx: &Receiver<()>) {
        // Phase 1: SIGTERM all running.
        tracing::info!("shutdown phase 1: SIGTERM running children");
        for slot in self.state.slots.values() {
            if let Some(job) = slot.running.as_ref()
                && let Err(e) = job.signaler.signal_term()
            {
                tracing::debug!(pid = job.pid, ?e, "SIGTERM failed");
            }
        }
        // Phase 2: drain reaps for shutdown_grace. No pump — pending
        // effects are dropped, not respawned. If `hard` was already set
        // when we entered shutdown (operator double-Ctrl-C), skip the
        // grace entirely. Otherwise the loop also watches
        // `hard_shutdown_rx` and breaks early if it fires mid-grace.
        let deadline = Instant::now() + self.shutdown_grace;
        let mut grace = !hard;
        while self.has_running() && grace {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            crossbeam::select! {
                recv(self.reap_rx) -> r => match r {
                    Ok(r) => self.state.handle_reap_no_pump(r, engine_in),
                    Err(crossbeam::channel::RecvError) => break,
                },
                recv(hard_shutdown_rx) -> _ => { grace = false; }
                default(deadline - now) => break,
            }
        }
        // Phase 3: SIGKILL stragglers.
        if self.has_running() {
            tracing::info!("shutdown phase 3: SIGKILL stragglers");
            for slot in self.state.slots.values() {
                if let Some(job) = slot.running.as_ref()
                    && let Err(e) = job.signaler.signal_kill()
                {
                    tracing::debug!(pid = job.pid, ?e, "SIGKILL failed");
                }
            }
        }
        // Phase 4: drain remaining reaps. SIGKILL is uninterruptible
        // (kernel guarantee), so the wait threads must return
        // eventually. Cap with a wall-clock guard to avoid hanging on
        // misbehaving mocks; in production this loop terminates within
        // microseconds of phase 3.
        let final_deadline = Instant::now() + self.shutdown_grace;
        while self.has_running() {
            let now = Instant::now();
            if now >= final_deadline {
                tracing::warn!("shutdown phase 4: final-drain deadline elapsed");
                break;
            }
            match self.reap_rx.recv_timeout(final_deadline - now) {
                Ok(r) => self.state.handle_reap_no_pump(r, engine_in),
                Err(
                    crossbeam::channel::RecvTimeoutError::Timeout
                    | crossbeam::channel::RecvTimeoutError::Disconnected,
                ) => break,
            }
        }
        tracing::info!("shutdown complete");
    }

    fn has_running(&self) -> bool {
        self.state.slots.values().any(|s| s.running.is_some())
    }
}

#[cfg(all(test, feature = "testkit"))]
#[allow(
    clippy::items_after_statements,
    clippy::missing_const_for_fn,
    clippy::needless_pass_by_value,
    clippy::too_many_lines
)]
mod tests {
    use crate::SubprocessActuator;
    use crate::testkit::{MockSpawner, SignalRecord};
    use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
    use specter_core::{
        CommandResolved, CorrelationId, DedupKey, Effect, EffectOutcome, Input, ProfileId,
        ResourceId, SubId,
    };
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    // ---------- helpers ----------

    fn unique_sub_id(seed: u64) -> SubId {
        use slotmap::KeyData;
        SubId::from(KeyData::from_ffi(seed))
    }

    fn unique_resource_id(seed: u64) -> ResourceId {
        use slotmap::KeyData;
        ResourceId::from(KeyData::from_ffi(seed))
    }

    fn unique_profile_id(seed: u64) -> ProfileId {
        use slotmap::KeyData;
        ProfileId::from(KeyData::from_ffi(seed))
    }

    fn make_effect_perfile(sub_seed: u64, profile_seed: u64, res_seed: u64, corr: u64) -> Effect {
        let resource = unique_resource_id(res_seed);
        Effect {
            key: DedupKey::PerFile {
                sub: unique_sub_id(sub_seed),
                profile: unique_profile_id(profile_seed),
                resource,
            },
            target: resource,
            command: CommandResolved {
                argv: vec!["/bin/true".into()],
            },
            env: Vec::new(),
            cwd: PathBuf::from("/tmp"),
            forced: false,
            correlation: CorrelationId(corr),
            diff: None,
            capture_output: false,
        }
    }

    fn make_effect_subtree(sub_seed: u64, profile_seed: u64, corr: u64) -> Effect {
        Effect {
            key: DedupKey::Subtree {
                sub: unique_sub_id(sub_seed),
                profile: unique_profile_id(profile_seed),
            },
            target: unique_resource_id(profile_seed),
            command: CommandResolved {
                argv: vec!["/bin/true".into()],
            },
            env: Vec::new(),
            cwd: PathBuf::from("/tmp"),
            forced: false,
            correlation: CorrelationId(corr),
            diff: None,
            capture_output: false,
        }
    }

    /// Spawn the controller in a thread; return the channels + a join
    /// handle. `concurrency` is the global cap.
    struct Harness {
        effects_tx: Sender<Effect>,
        shutdown_tx: Sender<()>,
        hard_shutdown_tx: Sender<()>,
        engine_in: Receiver<Input>,
        spawner: Arc<MockSpawner>,
        join: Option<thread::JoinHandle<()>>,
    }

    impl Harness {
        fn new(concurrency: usize) -> Self {
            Self::new_with_grace(concurrency, Duration::from_secs(5))
        }

        fn new_with_grace(concurrency: usize, grace: Duration) -> Self {
            let (effects_tx, effects_rx) = bounded::<Effect>(1024);
            let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
            let (hard_shutdown_tx, hard_shutdown_rx) = bounded::<()>(1);
            let (engine_tx, engine_rx) = unbounded::<Input>();
            let spawner = Arc::new(MockSpawner::new());
            let spawner_clone = Arc::clone(&spawner);
            let join = thread::Builder::new()
                .name("test-actuator-controller".into())
                .spawn(move || {
                    let mut a = SubprocessActuator::new_with_grace(concurrency, grace);
                    a.run(
                        effects_rx,
                        shutdown_rx,
                        hard_shutdown_rx,
                        engine_tx,
                        spawner_clone.as_ref(),
                    );
                })
                .expect("spawn controller");
            Self {
                effects_tx,
                shutdown_tx,
                hard_shutdown_tx,
                engine_in: engine_rx,
                spawner,
                join: Some(join),
            }
        }

        fn submit(&self, e: Effect) {
            self.effects_tx.send(e).expect("submit");
        }

        fn shutdown(&mut self) {
            let _ = self.shutdown_tx.send(());
            if let Some(j) = self.join.take() {
                j.join().expect("controller join");
            }
        }

        /// Block until `MockSpawner` has recorded at least `n` spawns.
        /// Times out after `dur`; returns the actual recorded list.
        fn wait_for_spawns(&self, n: usize, dur: Duration) -> Vec<crate::testkit::SpawnRecord> {
            let deadline = Instant::now() + dur;
            loop {
                let s = self.spawner.spawns();
                if s.len() >= n {
                    return s;
                }
                assert!(
                    Instant::now() < deadline,
                    "expected {n} spawns; saw {}",
                    s.len()
                );
                thread::sleep(Duration::from_millis(5));
            }
        }

        /// Block until the engine receives at least `n` EffectComplete
        /// messages.
        fn wait_for_effect_completes(&self, n: usize, dur: Duration) -> Vec<Input> {
            let deadline = Instant::now() + dur;
            let mut received = Vec::new();
            while received.len() < n {
                let now = Instant::now();
                assert!(
                    now < deadline,
                    "expected {n} EffectCompletes; saw {}",
                    received.len()
                );
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

    // ---------- coalescing ----------

    #[test]
    fn submit_to_empty_slot_spawns_immediately() {
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        assert_eq!(s.len(), 1);
        h.spawner
            .complete(s[0].pid, EffectOutcome::Ok)
            .expect("complete");
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn capture_output_threads_from_effect_to_spawner() {
        // The actuator must surface `Effect.capture_output` to the
        // `Spawner::spawn` call so the production OsSpawner can switch
        // between Stdio::null() (false) and Stdio::inherit() (true).
        let mut h = Harness::new(4);
        let mut e_off = make_effect_subtree(1, 1, 1);
        e_off.capture_output = false;
        let mut e_on = make_effect_subtree(2, 2, 2);
        e_on.capture_output = true;
        h.submit(e_off);
        h.submit(e_on);
        let s = h.wait_for_spawns(2, Duration::from_secs(1));
        // Spawn order matches submit order under the global gate.
        assert!(!s[0].capture_output);
        assert!(s[1].capture_output);
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        h.spawner.complete(s[1].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn submit_to_running_slot_replaces_pending() {
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        // While the first is "running", submit two more on the same key.
        h.submit(make_effect_perfile(1, 1, 1, 2));
        h.submit(make_effect_perfile(1, 1, 1, 3));
        // No second spawn yet (first still "running").
        thread::sleep(Duration::from_millis(50));
        assert_eq!(h.spawner.spawns().len(), 1, "running blocks new spawn");
        // Complete the first; pump should pick up the *latest* (corr=3,
        // corr=2 was dropped by Latest coalesce).
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        let s2 = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s2.len(), 2, "second spawn fires after reap");
        // Complete the second.
        h.spawner.complete(s2[1].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn reap_with_no_pending_emits_completion_and_clears_slot() {
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete { result, .. } => {
                assert!(matches!(result, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete; got {other:?}"),
        }
        h.shutdown();
    }

    // ---------- concurrency ----------

    #[test]
    fn concurrency_cap_blocks_excess() {
        let mut h = Harness::new(2);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        h.submit(make_effect_perfile(2, 2, 2, 2));
        h.submit(make_effect_perfile(3, 3, 3, 3));
        let s = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s.len(), 2, "only 2 spawns under cap=2");
        thread::sleep(Duration::from_millis(50));
        assert_eq!(h.spawner.spawns().len(), 2, "third blocked");
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        let s3 = h.wait_for_spawns(3, Duration::from_secs(1));
        assert_eq!(
            s3.len(),
            3,
            "third spawned after first reap released permit"
        );
        h.spawner.complete(s3[1].pid, EffectOutcome::Ok).unwrap();
        h.spawner.complete(s3[2].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(3, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn per_sub_serializes_two_per_file_keys() {
        let mut h = Harness::new(4);
        // Same Sub, different Resources → both PerFile keys, one Sub.
        h.submit(make_effect_perfile(7, 7, 1, 1));
        h.submit(make_effect_perfile(7, 7, 2, 2));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            1,
            "per-Sub gate forces serialization"
        );
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        let s2 = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s2.len(), 2, "second spawn after first reap");
        h.spawner.complete(s2[1].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn per_sub_does_not_serialize_distinct_subs() {
        let mut h = Harness::new(4);
        // Different Subs → no per-Sub gating.
        h.submit(make_effect_perfile(1, 1, 1, 1));
        h.submit(make_effect_perfile(2, 2, 2, 2));
        let s = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s.len(), 2, "distinct Subs run concurrently");
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        h.spawner.complete(s[1].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn subtree_and_per_file_for_same_sub_serialize() {
        let mut h = Harness::new(4);
        h.submit(make_effect_subtree(5, 1, 1));
        h.submit(make_effect_perfile(5, 5, 2, 2));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            1,
            "Subtree and PerFile for the same Sub serialize"
        );
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        let s2 = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s2.len(), 2);
        h.spawner.complete(s2[1].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    // ---------- shutdown ----------

    #[test]
    fn shutdown_with_no_running_returns_immediately() {
        let mut h = Harness::new(4);
        h.shutdown();
        assert!(
            h.spawner.signals().is_empty(),
            "no signals when nothing is running"
        );
    }

    #[test]
    fn shutdown_sigterms_running_then_drains_reap() {
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        let pid = s[0].pid;
        // Trigger shutdown; it'll send SIGTERM then wait up to grace.
        let shutdown_tx = h.shutdown_tx.clone();
        let spawner = Arc::clone(&h.spawner);
        let waiter_thread = thread::spawn(move || {
            // After shutdown trigger, briefly wait, then complete the
            // child (mock waiters block until told). This simulates the
            // child responding to SIGTERM gracefully.
            thread::sleep(Duration::from_millis(50));
            spawner
                .complete(
                    pid,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: Some(15),
                    },
                )
                .unwrap();
        });
        shutdown_tx.send(()).unwrap();
        h.join
            .take()
            .unwrap()
            .join()
            .expect("controller join after graceful shutdown");
        waiter_thread.join().unwrap();
        let signals = h.spawner.take_signals();
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, SignalRecord::Term(p) if *p == pid)),
            "SIGTERM delivered: {signals:?}"
        );
    }

    #[test]
    fn shutdown_grace_expires_then_sigkills_stragglers() {
        // Use a short grace so the test runs quickly.
        let mut h = Harness::new_with_grace(4, Duration::from_millis(150));
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        let pid = s[0].pid;

        let shutdown_tx = h.shutdown_tx.clone();
        let spawner = Arc::clone(&h.spawner);
        // After the grace window, complete the child (simulating SIGKILL
        // forcing it down). The controller should have sent SIGKILL by then.
        let waiter_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            // Complete with signal=9 (the result of SIGKILL).
            spawner
                .complete(
                    pid,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: Some(9),
                    },
                )
                .unwrap();
        });
        shutdown_tx.send(()).unwrap();
        h.join
            .take()
            .unwrap()
            .join()
            .expect("controller join after forced shutdown");
        waiter_thread.join().unwrap();

        let signals = h.spawner.take_signals();
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, SignalRecord::Term(p) if *p == pid)),
            "SIGTERM first: {signals:?}"
        );
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, SignalRecord::Kill(p) if *p == pid)),
            "SIGKILL after grace: {signals:?}"
        );
    }

    #[test]
    fn hard_shutdown_skips_grace_and_sigkills_immediately() {
        // Operator double-Ctrl-C: the signal thread fires
        // `hard_shutdown_actuator_tx` *before* `exit_fn(130)`. The actuator
        // must SIGTERM all running children (phase 1), bypass the 5s grace
        // wait (phase 2), and SIGKILL stragglers (phase 3). With a long
        // grace (5s) configured, this test asserts that the SIGKILL lands
        // *well* before the grace would have elapsed.
        let mut h = Harness::new_with_grace(4, Duration::from_secs(5));
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        let pid = s[0].pid;

        let hard_tx = h.hard_shutdown_tx.clone();
        let spawner = Arc::clone(&h.spawner);
        // Resolve the child only after we expect SIGKILL to have landed.
        // Cap latency low so the assertion below catches a regression
        // (5-second grace not bypassed).
        let waiter_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(300));
            spawner
                .complete(
                    pid,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: Some(9),
                    },
                )
                .unwrap();
        });

        let t0 = Instant::now();
        hard_tx.send(()).unwrap();
        h.join
            .take()
            .unwrap()
            .join()
            .expect("controller join after hard shutdown");
        let elapsed = t0.elapsed();
        waiter_thread.join().unwrap();

        let signals = h.spawner.take_signals();
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, SignalRecord::Term(p) if *p == pid)),
            "phase 1 SIGTERM still runs: {signals:?}"
        );
        assert!(
            signals
                .iter()
                .any(|s| matches!(s, SignalRecord::Kill(p) if *p == pid)),
            "phase 3 SIGKILL after hard-shutdown: {signals:?}"
        );
        assert!(
            elapsed < Duration::from_secs(2),
            "hard-shutdown bypassed the grace period (elapsed: {elapsed:?})"
        );
    }

    #[test]
    fn shutdown_drops_pending_effects() {
        let mut h = Harness::new(1);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        // Submit a second effect on the same key — it becomes pending.
        h.submit(make_effect_perfile(1, 1, 1, 2));
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            1,
            "pending stays pending while running"
        );

        // Trigger shutdown; complete the running child.
        let shutdown_tx = h.shutdown_tx.clone();
        let spawner = Arc::clone(&h.spawner);
        let pid = s[0].pid;
        let waiter_thread = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            spawner.complete(pid, EffectOutcome::Ok).unwrap();
        });
        shutdown_tx.send(()).unwrap();
        h.join.take().unwrap().join().expect("controller join");
        waiter_thread.join().unwrap();

        // Only the running effect's EffectComplete should arrive — pending
        // is silently dropped on shutdown.
        let mut received = Vec::new();
        while let Ok(i) = h.engine_in.try_recv() {
            received.push(i);
        }
        assert_eq!(received.len(), 1, "only running's reap was emitted");
        // Total spawns: 1 (pending was never spawned).
        assert_eq!(h.spawner.spawns().len(), 1);
    }

    // ---------- failure synthesis ----------

    #[test]
    fn spawn_failure_synthesizes_failed_outcome() {
        let mut h = Harness::new(4);
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete { result, .. } => {
                assert!(matches!(
                    result,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: None
                    }
                ));
            }
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
        // No actual spawn was recorded (inject_spawn_error short-circuits).
        assert!(h.spawner.spawns().is_empty());
        h.shutdown();
    }

    #[test]
    fn spawn_failure_releases_permit() {
        // After a spawn failure the permit must be released — otherwise
        // subsequent submits would never spawn.
        let mut h = Harness::new(1);
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        // Clear the injection; submit again — should spawn.
        h.spawner.clear_spawn_error();
        h.submit(make_effect_perfile(2, 2, 2, 2));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        // Engine receives one more EffectComplete (the prior call drained
        // the first); waiting for one more here is the right count.
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    #[test]
    fn spawn_failure_does_not_block_same_sub_on_different_key() {
        // Regression fence for a (now-closed) race in the synth-Reap
        // teardown: when spawn failures synthesised a `Reaped` via a
        // channel hop, an interleaved same-key submit could replace
        // `slot.running` before the synth drained, then the synth
        // would clobber the new job's state and leave the per-Sub
        // counter desynced. The fix routes synth-Reaps directly
        // through `handle_reap_inner` on the controller call stack.
        //
        // This test exercises the simpler post-failure observable:
        // after a spawn-failure on (Sub=7, Profile=1, Resource=1), a
        // follow-up Effect for the *same* Sub on (Profile=2,
        // Resource=2) must spawn promptly. If `running_per_sub[7]`
        // were stuck above zero, the per-Sub gate would defer the
        // second submit indefinitely.
        let mut h = Harness::new(4);
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile(7, 1, 1, 1));
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.spawner.clear_spawn_error();
        h.submit(make_effect_perfile(7, 2, 2, 2));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    // ---------- tmp diff file ----------

    #[test]
    fn effect_with_diff_passes_specter_diff_path() {
        use compact_str::CompactString;
        use smallvec::smallvec;
        use specter_core::{Diff, EntryKind, EntryRef};

        let mut h = Harness::new(4);
        let diff = Arc::new(Diff {
            created: smallvec![EntryRef {
                segment: CompactString::from("a.rs"),
                kind: EntryKind::File,
                inode: 1,
            }],
            ..Default::default()
        });
        let mut e = make_effect_perfile(1, 1, 1, 7);
        e.diff = Some(Arc::clone(&diff));
        h.submit(e);
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        let env = &s[0].env;
        let path = env
            .iter()
            .find(|(k, _)| k == "SPECTER_DIFF_PATH")
            .expect("SPECTER_DIFF_PATH set")
            .1
            .clone();
        // File was written by the actuator.
        assert!(std::path::Path::new(&path).exists(), "tmp file exists");
        let contents = std::fs::read_to_string(&path).unwrap();
        assert!(contents.starts_with("created\ta.rs\t1"));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        // After the wait thread finishes, the tmp file should be cleaned up.
        let deadline = Instant::now() + Duration::from_secs(1);
        while std::path::Path::new(&path).exists() {
            assert!(Instant::now() < deadline, "tmp file not cleaned up: {path}");
            thread::sleep(Duration::from_millis(5));
        }
        h.shutdown();
    }

    #[test]
    fn effect_without_diff_does_not_set_specter_diff_path() {
        let mut h = Harness::new(4);
        let e = make_effect_perfile(1, 1, 1, 1); // diff: None
        h.submit(e);
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        assert!(s[0].env.iter().all(|(k, _)| k != "SPECTER_DIFF_PATH"));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }
}
