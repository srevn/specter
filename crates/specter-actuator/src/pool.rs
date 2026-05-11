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
use crate::env::EnvSnapshot;
use crate::spawner::Spawner;
use crossbeam::channel::{Receiver, Sender};
use specter_core::{CorrelationId, DedupKey, Effect, EffectOutcome, Input, SubId};
use state::ActuatorState;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Sentinel for "default concurrency" passed to [`SubprocessActuator::new`].
/// Resolved to `2 * num_cpus` (or `4` if `num_cpus` is unavailable). The
/// bin's CLI flag passes a non-zero value when set.
pub const DEFAULT_CONCURRENCY: usize = 0;

/// SIGTERM → SIGKILL grace, pinned in one place so the shutdown drain
/// and per-step timer threads can't drift apart.
///
/// Read by:
/// - [`SubprocessActuator::shutdown`] (the SIGTERM → grace → SIGKILL
///   sequence).
/// - [`crate::timer::spawn_timer`] (per-step deadline enforcement).
///
/// Default for production; tests may override via
/// [`SubprocessActuator::new_with_grace`] /
/// [`SubprocessActuator::new_with_grace_and_env`].
pub(crate) const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Signal from a wait thread back to the controller.
#[derive(Debug)]
pub struct Reaped {
    pub key: DedupKey,
    pub sub: SubId,
    pub correlation: CorrelationId,
    pub outcome: EffectOutcome,
}

/// Resolve a `concurrency: usize` knob into a [`NonZeroUsize`]:
/// `0` ⇒ `2 × available_parallelism()` (fallback `4`); non-zero ⇒
/// pass-through. Shared between [`SubprocessActuator::new`] and the
/// test constructors so the resolution logic stays single-source.
fn resolve_concurrency(concurrency: usize) -> NonZeroUsize {
    let fallback = NonZeroUsize::new(4).expect("4 is non-zero");
    NonZeroUsize::new(concurrency).unwrap_or_else(|| {
        std::thread::available_parallelism()
            .ok()
            .and_then(|n| NonZeroUsize::new(n.get().saturating_mul(2)))
            .unwrap_or(fallback)
    })
}

/// The actuator's controller. One per process. Owns the slot map, ready
/// queue, per-Sub counter, and global semaphore. Blocks in [`Self::run`]
/// for the lifetime of the bin process.
#[derive(Debug)]
pub struct SubprocessActuator {
    state: ActuatorState,
    reap_tx: Sender<Reaped>,
    reap_rx: Receiver<Reaped>,
}

impl SubprocessActuator {
    /// Construct with `concurrency` global permits. Pass
    /// [`DEFAULT_CONCURRENCY`] (`0`) to resolve to `2 * num_cpus`; non-zero
    /// values pass through. The `0`-sentinel is the only place this
    /// crate resolves "default concurrency"; everything below
    /// [`ActuatorState::new`] receives a [`NonZeroUsize`] and trusts it.
    ///
    /// Captures the current process env via [`EnvSnapshot::capture`] —
    /// invoked exactly once per actuator. The snapshot is shared by
    /// `Arc` across the resolver's per-step calls.
    #[must_use]
    pub fn new(concurrency: usize) -> Self {
        let resolved = resolve_concurrency(concurrency);
        let (reap_tx, reap_rx) = crossbeam::channel::bounded::<Reaped>(64);
        Self {
            state: ActuatorState::new(resolved, Arc::new(EnvSnapshot::capture()), SHUTDOWN_GRACE),
            reap_tx,
            reap_rx,
        }
    }

    /// Test-only constructor with both a custom shutdown grace and a
    /// preconstructed env snapshot. Used by tests that need to assert
    /// shutdown timing or `${env.<NAME>}` resolution (strict-unset →
    /// Failed, default rendering, etc.) without depending on the
    /// ambient process env.
    ///
    /// Gated to match the test module (`cfg(all(test, feature = "testkit"))`)
    /// — without `testkit`, the test module that consumes this constructor
    /// is excluded too, so the function would otherwise be flagged as
    /// dead code under `cargo test --lib` (no features).
    #[cfg(all(test, feature = "testkit"))]
    pub(crate) fn new_with_grace_and_env(
        concurrency: usize,
        grace: Duration,
        env: Arc<EnvSnapshot>,
    ) -> Self {
        let resolved = resolve_concurrency(concurrency);
        let (reap_tx, reap_rx) = crossbeam::channel::bounded::<Reaped>(64);
        Self {
            state: ActuatorState::new(resolved, env, grace),
            reap_tx,
            reap_rx,
        }
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
        self.shutdown(&engine_in, hard, &hard_shutdown_rx, spawner);
    }

    fn shutdown(
        &mut self,
        engine_in: &Sender<Input>,
        hard: bool,
        hard_shutdown_rx: &Receiver<()>,
        spawner: &dyn Spawner,
    ) {
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
        let deadline = Instant::now() + self.state.shutdown_grace;
        let mut grace = !hard;
        while self.has_running() && grace {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            crossbeam::select! {
                recv(self.reap_rx) -> r => match r {
                    Ok(r) => self.state.handle_reap_no_pump(r, engine_in, spawner, &self.reap_tx),
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
        let final_deadline = Instant::now() + self.state.shutdown_grace;
        while self.has_running() {
            let now = Instant::now();
            if now >= final_deadline {
                tracing::warn!("shutdown phase 4: final-drain deadline elapsed");
                break;
            }
            match self.reap_rx.recv_timeout(final_deadline - now) {
                Ok(r) => self
                    .state
                    .handle_reap_no_pump(r, engine_in, spawner, &self.reap_tx),
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
    use crate::env::EnvSnapshot;
    use crate::testkit::{MockSpawner, SignalRecord};
    use compact_str::CompactString;
    use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::testkit::{predicate_then_program, single_exec_program};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, CorrelationId, DedupKey, Effect, EffectOutcome,
        ExecAction, Input, ProfileId, ResourceId, ResourceKind, SubId,
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

    fn literal_program() -> Arc<ActionProgram> {
        n_step_program(1)
    }

    /// Build an `n`-op program whose every op is a literal `/bin/true`
    /// Exec, chained `on_ok = Continue` (final op `on_ok = Escape`);
    /// every `on_failed = Terminate`. Used by multi-op tests to drive
    /// the actuator's advance / terminate path without caring about
    /// argv shape.
    fn n_step_program(n: usize) -> Arc<ActionProgram> {
        assert!(n >= 1, "n_step_program requires at least one step");
        let mut b = ProgramBuilder::new();
        let mut prev: Option<specter_core::program::OpHandle> = None;
        for _ in 0..n {
            if let Some(ph) = prev {
                let next = b.continue_to_next();
                b.patch_on_ok(ph, next).unwrap();
            }
            let h = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/true"),
            ])])));
            b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
            prev = Some(h);
        }
        if let Some(last) = prev {
            b.patch_on_ok(last, BranchTarget::Escape).unwrap();
        }
        Arc::new(b.build().unwrap())
    }

    fn make_effect_perfile(sub_seed: u64, profile_seed: u64, res_seed: u64, corr: u64) -> Effect {
        make_effect_perfile_with_program(sub_seed, profile_seed, res_seed, corr, literal_program())
    }

    fn make_effect_perfile_with_program(
        sub_seed: u64,
        profile_seed: u64,
        res_seed: u64,
        corr: u64,
        program: Arc<ActionProgram>,
    ) -> Effect {
        let resource = unique_resource_id(res_seed);
        Effect {
            key: DedupKey::PerFile {
                sub: unique_sub_id(sub_seed),
                profile: unique_profile_id(profile_seed),
                resource,
            },
            target: resource,
            forced: false,
            correlation: CorrelationId(corr),
            diff: None,
            capture_output: false,
            sub_name: CompactString::new(""),
            program,
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            target_relative: CompactString::new(""),
            exclude: Arc::from(Vec::<CompactString>::new()),
        }
    }

    fn make_effect_subtree(sub_seed: u64, profile_seed: u64, corr: u64) -> Effect {
        Effect {
            key: DedupKey::Subtree {
                sub: unique_sub_id(sub_seed),
                profile: unique_profile_id(profile_seed),
            },
            target: unique_resource_id(profile_seed),
            forced: false,
            correlation: CorrelationId(corr),
            diff: None,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: literal_program(),
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            target_relative: CompactString::new(""),
            exclude: Arc::from(Vec::<CompactString>::new()),
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

    /// Empty env snapshot — convenience for the majority of tests that
    /// don't exercise `${env.<NAME>}` resolution.
    fn empty_env() -> Arc<EnvSnapshot> {
        Arc::new(EnvSnapshot::from_map::<_, &str, &str>([]))
    }

    impl Harness {
        fn new(concurrency: usize) -> Self {
            Self::new_with_grace_and_env(concurrency, Duration::from_secs(5), empty_env())
        }

        fn new_with_grace(concurrency: usize, grace: Duration) -> Self {
            Self::new_with_grace_and_env(concurrency, grace, empty_env())
        }

        fn new_with_grace_and_env(
            concurrency: usize,
            grace: Duration,
            env: Arc<EnvSnapshot>,
        ) -> Self {
            let (effects_tx, effects_rx) = bounded::<Effect>(1024);
            let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
            let (hard_shutdown_tx, hard_shutdown_rx) = bounded::<()>(1);
            let (engine_tx, engine_rx) = unbounded::<Input>();
            let spawner = Arc::new(MockSpawner::new());
            let spawner_clone = Arc::clone(&spawner);
            let join = thread::Builder::new()
                .name("test-actuator-controller".into())
                .spawn(move || {
                    let mut a = SubprocessActuator::new_with_grace_and_env(concurrency, grace, env);
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

    // ---------- multi-step plans ----------

    /// Multi-step happy path: a 3-step plan reaps each step Ok, the
    /// actuator advances through steps 0 → 1 → 2 in sequence, and
    /// emits exactly one `EffectComplete::Ok` after the last step.
    #[test]
    fn three_step_plan_runs_steps_sequentially_and_emits_one_complete() {
        let mut h = Harness::new(4);
        let plan = n_step_program(3);
        h.submit(make_effect_perfile_with_program(1, 1, 1, 1, plan));

        // Step 0 spawns, reaps Ok → step 1 spawns, reaps Ok → step 2
        // spawns, reaps Ok → terminal EffectComplete.
        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();
        let s2 = h.wait_for_spawns(3, Duration::from_secs(1));
        h.spawner.complete(s2[2].pid, EffectOutcome::Ok).unwrap();

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert_eq!(
            completions.len(),
            1,
            "exactly one EffectComplete per Effect"
        );
        match &completions[0] {
            Input::EffectComplete { result, .. } => {
                assert!(matches!(result, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete::Ok; got {other:?}"),
        }
        // Verify no extra EffectCompletes are queued.
        thread::sleep(Duration::from_millis(50));
        assert!(
            h.engine_in.try_recv().is_err(),
            "no extra EffectComplete after terminal",
        );
        h.shutdown();
    }

    /// Stop-on-fail: a 3-step plan whose step 1 fails terminates the
    /// plan immediately. Step 2 is never spawned. Engine sees one
    /// `EffectComplete::Failed`.
    #[test]
    fn three_step_plan_stops_on_first_failure() {
        let mut h = Harness::new(4);
        let plan = n_step_program(3);
        h.submit(make_effect_perfile_with_program(2, 2, 2, 1, plan));

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        h.spawner
            .complete(
                s1[1].pid,
                EffectOutcome::Failed {
                    exit_code: Some(7),
                    signal: None,
                },
            )
            .unwrap();

        // No third spawn — the plan halted on step 1's failure.
        thread::sleep(Duration::from_millis(50));
        assert_eq!(h.spawner.spawns().len(), 2, "step 2 was not spawned");

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete { result, .. } => assert!(matches!(
                result,
                EffectOutcome::Failed {
                    exit_code: Some(7),
                    signal: None,
                }
            )),
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
        h.shutdown();
    }

    /// Multi-step plan with diff: tmp file is materialised once at
    /// plan start, every step's env has the same `SPECTER_DIFF_PATH`,
    /// the file is cleaned exactly once after the terminal step.
    #[test]
    fn multi_step_plan_shares_tmp_diff_path_and_cleans_at_terminus() {
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
        let plan = n_step_program(2);
        let mut e = make_effect_perfile_with_program(3, 3, 3, 7, plan);
        e.diff = Some(Arc::clone(&diff));
        h.submit(e);

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        let path0 = s0[0]
            .env
            .iter()
            .find(|(k, _)| k == "SPECTER_DIFF_PATH")
            .expect("SPECTER_DIFF_PATH set on step 0")
            .1
            .clone();
        assert!(
            std::path::Path::new(&path0).exists(),
            "tmp file exists during step 0",
        );
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();

        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        let path1 = s1[1]
            .env
            .iter()
            .find(|(k, _)| k == "SPECTER_DIFF_PATH")
            .expect("SPECTER_DIFF_PATH set on step 1")
            .1
            .clone();
        assert_eq!(path0, path1, "step 1 sees the same tmp path as step 0");
        // Mid-plan: the file MUST still exist (cleanup is at terminal).
        assert!(
            std::path::Path::new(&path0).exists(),
            "tmp file persists across steps",
        );
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));

        // After the terminal step, the file is cleaned (poll briefly
        // since cleanup happens on the controller thread post-reap).
        let deadline = Instant::now() + Duration::from_secs(1);
        while std::path::Path::new(&path0).exists() {
            assert!(
                Instant::now() < deadline,
                "tmp file not cleaned up: {path0}",
            );
            thread::sleep(Duration::from_millis(5));
        }
        h.shutdown();
    }

    /// Mid-plan submit-coalesce: a fresh same-key submit during a
    /// running plan replaces `pending` only. The current plan runs to
    /// terminus before pending fires (plan-atomicity invariant).
    #[test]
    fn submit_during_running_plan_replaces_pending_runs_after_terminal() {
        let mut h = Harness::new(4);
        let plan_a = n_step_program(2);
        let effect_a = make_effect_perfile_with_program(4, 4, 4, 100, plan_a);
        h.submit(effect_a);

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        // While step 0 is running, submit a fresh effect for the same
        // key. Latest-coalesce stores it as pending; plan_a continues.
        let plan_b = n_step_program(1);
        let effect_b = make_effect_perfile_with_program(4, 4, 4, 200, plan_b);
        h.submit(effect_b);
        // Also submit a third same-key effect — should replace pending.
        let plan_c = n_step_program(1);
        let effect_c = make_effect_perfile_with_program(4, 4, 4, 300, plan_c);
        h.submit(effect_c);

        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            1,
            "no second spawn while plan_a's step 0 is running",
        );

        // Reap step 0. plan_a advances to step 1.
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();

        // plan_a's terminal EffectComplete arrives, then pending
        // (effect_c, latest) spawns — its single step runs.
        let s2 = h.wait_for_spawns(3, Duration::from_secs(1));
        h.spawner.complete(s2[2].pid, EffectOutcome::Ok).unwrap();

        // Two EffectCompletes total: one for plan_a, one for plan_c.
        // plan_b was dropped by Latest-coalesce (replaced by plan_c).
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    /// Multi-step plan + cap=1 + concurrent fresh Sub: the plan's
    /// advance-step branch picks up the freshly-released permit
    /// (reap-side path is on-stack in the controller, so it always
    /// wins over pump's queue scan for the same permit). The
    /// concurrent Sub's plan starts only after the multi-step plan
    /// terminates.
    ///
    /// This is the deterministic shape of "multi-step plan-atomicity
    /// under contention": all steps of plan A run before plan B
    /// starts. The race-on-select shape (where pump's submit-handler
    /// for B beats handle_reap's advance, forcing plan_continue) is
    /// covered deterministically in the unit-level
    /// `step_ok_not_last_with_no_permit_defers_via_plan_continue`
    /// test in `pool/state.rs`.
    #[test]
    fn multi_step_plan_runs_to_terminus_before_concurrent_sub_starts() {
        let mut h = Harness::new(1); // cap=1: one global permit
        // Sub A: 2-step plan. Step 0 spawns, holding the only permit.
        let plan = n_step_program(2);
        h.submit(make_effect_perfile_with_program(5, 5, 5, 1, plan));
        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));

        // Sub B: 1-step plan submitted concurrently. Has to wait for
        // the permit.
        h.submit(make_effect_perfile(6, 6, 6, 2));
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            1,
            "B is blocked on the only permit",
        );

        // Reap A's step 0. The wait thread releases the permit, then
        // sends Reaped. The controller's reap handler is already on
        // the call stack and re-acquires the permit before pump runs
        // — so step 1 of A spawns next, B still blocked.
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            2,
            "A step 1 took the freed permit; B still blocked",
        );

        // Reap A's step 1. Plan A terminates; permit released; pump
        // runs B's step 0.
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();
        let s2 = h.wait_for_spawns(3, Duration::from_secs(1));
        h.spawner.complete(s2[2].pid, EffectOutcome::Ok).unwrap();

        // Two EffectCompletes total: one for A's 2-step plan, one for B's.
        h.wait_for_effect_completes(2, Duration::from_secs(1));
        h.shutdown();
    }

    /// Multi-step plan under shutdown drop policy: step 0 reaps under
    /// `Drop` policy, no advance, terminal arm emits the reaped
    /// outcome. Subsequent steps are abandoned.
    #[test]
    fn shutdown_mid_plan_abandons_remaining_steps() {
        let mut h = Harness::new(4);
        let plan = n_step_program(3);
        h.submit(make_effect_perfile_with_program(7, 7, 7, 1, plan));
        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));

        // Trigger shutdown; complete step 0 mid-shutdown.
        let shutdown_tx = h.shutdown_tx.clone();
        let spawner = Arc::clone(&h.spawner);
        let pid = s0[0].pid;
        let waiter = thread::spawn(move || {
            thread::sleep(Duration::from_millis(20));
            spawner.complete(pid, EffectOutcome::Ok).unwrap();
        });
        shutdown_tx.send(()).unwrap();
        h.join.take().unwrap().join().expect("controller join");
        waiter.join().unwrap();

        // The shutdown reap path uses Drop policy: step 0's reap
        // emits EffectComplete with Ok, no step 1 spawn.
        let mut received = Vec::new();
        while let Ok(i) = h.engine_in.try_recv() {
            received.push(i);
        }
        assert_eq!(
            received.len(),
            1,
            "exactly one EffectComplete from drained step 0"
        );
        match &received[0] {
            Input::EffectComplete { result, .. } => {
                assert!(matches!(result, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete::Ok; got {other:?}"),
        }
        // Total spawns: 1 (only step 0 — no step 1 under Drop).
        assert_eq!(h.spawner.spawns().len(), 1);
    }

    // ---------- ${env.<NAME>} strict + default ----------

    /// Build a single-op program whose argv is the one given
    /// [`ArgPart`]. The actuator-level env tests need to inject precise
    /// `EnvVar` placeholders without routing through the config layer.
    fn env_var_program(name: &str, default: Option<&str>) -> Arc<ActionProgram> {
        single_exec_program([ArgTemplate::new([ArgPart::EnvVar {
            name: name.into(),
            default: default.map(CompactString::from),
        }])])
    }

    /// Strict-unset: an Effect that references an unset env var with
    /// no default terminates the plan with `EffectOutcome::Failed`
    /// before any spawn happens — the resolver fails fast.
    #[test]
    fn env_var_unset_no_default_terminates_plan_failed_before_spawn() {
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile_with_program(
            1,
            1,
            1,
            1,
            env_var_program("UNSET_VAR_AVOID_AMBIENT_COLLISION", None),
        ));
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete { result, .. } => assert!(matches!(
                result,
                EffectOutcome::Failed {
                    exit_code: None,
                    signal: None,
                }
            )),
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
        // Resolver failed before the spawner was reached.
        assert!(
            h.spawner.spawns().is_empty(),
            "no spawn recorded for unresolved env",
        );
        h.shutdown();
    }

    /// Default-bearing form renders the default literal into argv —
    /// the spawn proceeds normally and reaps Ok.
    #[test]
    fn env_var_unset_with_default_renders_default_in_argv() {
        let mut h = Harness::new_with_grace_and_env(4, Duration::from_secs(5), empty_env());
        h.submit(make_effect_perfile_with_program(
            2,
            2,
            2,
            2,
            env_var_program("UNSET_VAR_AVOID_AMBIENT_COLLISION", Some("/tmp")),
        ));
        let spawns = h.wait_for_spawns(1, Duration::from_secs(1));
        assert_eq!(spawns[0].argv, vec!["/tmp".to_string()]);
        h.spawner
            .complete(spawns[0].pid, EffectOutcome::Ok)
            .unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    /// Snapshot-present: env value substitutes into argv. Confirms the
    /// resolver reads from the injected snapshot, not the ambient
    /// process env.
    #[test]
    fn env_var_present_substitutes_from_injected_snapshot() {
        let mut h = Harness::new_with_grace_and_env(
            4,
            Duration::from_secs(5),
            Arc::new(EnvSnapshot::from_map([("SPECTER_TEST_X", "value-x")])),
        );
        h.submit(make_effect_perfile_with_program(
            3,
            3,
            3,
            3,
            env_var_program("SPECTER_TEST_X", None),
        ));
        let spawns = h.wait_for_spawns(1, Duration::from_secs(1));
        assert_eq!(spawns[0].argv, vec!["value-x".to_string()]);
        h.spawner
            .complete(spawns[0].pid, EffectOutcome::Ok)
            .unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    // ---------- per-step timeout ----------

    /// Build a single-op program with `timeout` set. Mirrors what the
    /// config layer would emit for `{ exec = ["..."], timeout = "..." }`.
    fn timeout_program(d: Duration) -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Exec(
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/true")])]).with_timeout(d),
        ));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// A child that doesn't complete within `timeout` receives SIGTERM
    /// from the per-step timer thread. The `MockSpawner` tracks the
    /// signal; the test confirms SIGTERM arrives by the time we
    /// observe it (poll-with-deadline since the timer is a separate
    /// thread).
    #[test]
    fn step_timeout_sigterms_unfinished_child_after_deadline() {
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile_with_program(
            10,
            10,
            10,
            10,
            timeout_program(Duration::from_millis(50)),
        ));
        let spawns = h.wait_for_spawns(1, Duration::from_secs(1));
        let pid = spawns[0].pid;

        // Wait for the timer to fire — at most deadline+slack. The
        // signal lands asynchronously from a detached thread.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            if h.spawner
                .signals()
                .iter()
                .any(|s| matches!(s, SignalRecord::Term(p) if *p == pid))
            {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "timer never delivered SIGTERM; got {:?}",
                h.spawner.signals(),
            );
            thread::sleep(Duration::from_millis(10));
        }

        // Complete the child so the wait thread can shut down and
        // reap. The signaler's MockChildSignaler::signal_term path
        // recorded `Term`; completing here drains the engine channel
        // so the harness's shutdown drop is clean.
        h.spawner
            .complete(
                pid,
                EffectOutcome::Failed {
                    exit_code: None,
                    signal: Some(15),
                },
            )
            .unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    /// Natural completion before the deadline short-circuits the
    /// timer's signal path via `ChildSignaler::is_dead`. No SIGTERM
    /// observed.
    #[test]
    fn step_timeout_short_circuits_when_child_completes_before_deadline() {
        // Long deadline; complete the child immediately.
        let mut h = Harness::new(4);
        h.submit(make_effect_perfile_with_program(
            11,
            11,
            11,
            11,
            timeout_program(Duration::from_mins(1)),
        ));
        let spawns = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner
            .complete(spawns[0].pid, EffectOutcome::Ok)
            .unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));

        // Allow the controller a moment in case the timer thread is
        // still in flight (sleep racing the dead flag). Even with a
        // generous 100ms window we expect zero signals: 60s deadline
        // dominates.
        thread::sleep(Duration::from_millis(50));
        assert!(
            !h.spawner
                .signals()
                .iter()
                .any(|s| matches!(s, SignalRecord::Term(_) | SignalRecord::Kill(_))),
            "timer must not signal a naturally-completed child; got {:?}",
            h.spawner.signals(),
        );
        h.shutdown();
    }

    // ---------- conditional dispatch (predicate edges) ----------

    /// Build a program for `when=W; then=[T]` (no else): predicate op
    /// (Exec) with `on_failed = Escape` (the "branch, not guard"
    /// outcome elision — predicate Failed terminates the plan Ok
    /// without propagation), then the then-Exec.
    fn predicate_then_no_else(when_label: &str, then_label: &str) -> Arc<ActionProgram> {
        predicate_then_program(
            ExecAction::new([ArgTemplate::new([ArgPart::literal(when_label)])]),
            ExecAction::new([ArgTemplate::new([ArgPart::literal(then_label)])]),
        )
    }

    /// Build a program for `when=W; then=[T]; else=[E]`: three ops in
    /// CFG-shape.
    ///
    /// - op 0: predicate `W` — `on_ok = Continue(1)` (then), `on_failed
    ///   = Continue(2)` (else).
    /// - op 1: then-Exec `T` — `on_ok = Escape` (skip else), `on_failed
    ///   = Terminate`.
    /// - op 2: else-Exec `E` — `on_ok = Escape`, `on_failed =
    ///   Terminate`.
    fn predicate_then_else(
        when_label: &str,
        then_label: &str,
        else_label: &str,
    ) -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let pred = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
            ArgPart::literal(when_label),
        ])])));
        // then enters at cursor 1
        let then_first = b.continue_to_next();
        b.patch_on_ok(pred, then_first).unwrap();
        let then_h = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
            ArgPart::literal(then_label),
        ])])));
        // else enters at cursor 2 — patch predicate's on_failed to it,
        // and then-Exec's on_ok is Escape (skip past else).
        let else_first = b.continue_to_next();
        b.patch_on_failed(pred, else_first).unwrap();
        b.patch_on_ok(then_h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(then_h, BranchTarget::Terminate).unwrap();
        let else_h = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
            ArgPart::literal(else_label),
        ])])));
        b.patch_on_ok(else_h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(else_h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// Predicate reaping Ok enters the then-branch: the actuator
    /// spawns the then-exec after the predicate reaps. Exactly one
    /// EffectComplete is emitted (Ok) at plan terminus.
    #[test]
    fn predicate_ok_spawns_then_branch_and_terminates_ok() {
        let mut h = Harness::new(4);
        let program = predicate_then_else("/bin/check", "/bin/then", "/bin/else");
        h.submit(make_effect_perfile_with_program(1, 1, 1, 1, program));

        // Predicate (cursor 0) spawns first.
        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        assert_eq!(s0[0].argv, vec!["/bin/check".to_string()]);
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();

        // Predicate Ok → enter then-branch. Then-exec spawns.
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s1[1].argv, vec!["/bin/then".to_string()]);
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();

        // After then-exec reaps, the Jump (cursor 2) skips else;
        // cursor 4 is past end → terminate Ok.
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(matches!(
            completions[0],
            Input::EffectComplete {
                result: EffectOutcome::Ok,
                ..
            }
        ));
        // Else-exec was never spawned.
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            2,
            "else-exec must not run when predicate Ok",
        );
        h.shutdown();
    }

    /// Predicate reaping Failed jumps to the else-branch (no
    /// propagation). The else-exec spawns; the predicate's Failed
    /// outcome does NOT surface as `EffectComplete::Failed`. Plan
    /// terminates Ok after else-exec reaps Ok.
    #[test]
    fn predicate_failed_spawns_else_branch_outcome_does_not_propagate() {
        let mut h = Harness::new(4);
        let program = predicate_then_else("/bin/check", "/bin/then", "/bin/else");
        h.submit(make_effect_perfile_with_program(2, 2, 2, 1, program));

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner
            .complete(
                s0[0].pid,
                EffectOutcome::Failed {
                    exit_code: Some(99),
                    signal: None,
                },
            )
            .unwrap();

        // Predicate Failed → jump to else_start. Else-exec spawns;
        // then-exec is skipped.
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s1[1].argv, vec!["/bin/else".to_string()]);
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Ok,
                    ..
                }
            ),
            "predicate Failed must not propagate to EffectComplete; got {:?}",
            completions[0],
        );
        h.shutdown();
    }

    /// Predicate reaping Failed with no else-branch terminates the
    /// plan Ok (predicate's `on_failed = Escape` — the "branch, not
    /// guard" outcome elision). The reaped Failed outcome stays out
    /// of `EffectComplete`.
    #[test]
    fn predicate_failed_no_else_terminates_ok_without_propagation() {
        let mut h = Harness::new(4);
        let program = predicate_then_no_else("/bin/check", "/bin/then");
        h.submit(make_effect_perfile_with_program(3, 3, 3, 1, program));

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner
            .complete(
                s0[0].pid,
                EffectOutcome::Failed {
                    exit_code: Some(7),
                    signal: None,
                },
            )
            .unwrap();

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Ok,
                    ..
                }
            ),
            "predicate Failed past plan end must terminate Ok; got {:?}",
            completions[0],
        );
        thread::sleep(Duration::from_millis(50));
        assert_eq!(
            h.spawner.spawns().len(),
            1,
            "then-exec must not run when predicate Failed",
        );
        h.shutdown();
    }

    /// Predicate spawn failure routes through the same dispatch as a
    /// natural Failed reap — the predicate's outcome does NOT
    /// propagate to plan terminus.
    ///
    /// Deterministic shape: a no-else conditional whose predicate
    /// spawn-fails (via injected `ErrorKind::NotFound`). The dispatch
    /// at cursor 0 sees the predicate op's synth-Failed outcome and
    /// reads `op.target(&Failed) = on_failed = Escape` (the no-else
    /// "branch, not guard" elision), so the plan terminates with
    /// `EffectOutcome::Ok`. If predicate spawn-failure had short-
    /// circuited to terminate directly (the pre-PR3 behaviour), this
    /// would emit `EffectComplete::Failed`. The Ok outcome is the
    /// no-propagation invariant in observable form.
    ///
    /// Zero spawns are recorded — the injection short-circuits
    /// `MockSpawner::spawn` before the `SpawnRecord` push.
    #[test]
    fn predicate_spawn_failure_does_not_propagate_no_else() {
        let mut h = Harness::new(4);
        let program = predicate_then_no_else("/bin/check", "/bin/then");
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile_with_program(4, 4, 4, 1, program));

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Ok,
                    ..
                }
            ),
            "predicate spawn-failure must terminate Ok via dispatch; got {:?}",
            completions[0],
        );
        assert!(
            h.spawner.spawns().is_empty(),
            "no spawn recorded — both injection short-circuits before push",
        );
        h.shutdown();
    }

    /// Predicate **resolver-failure** with an else-branch present
    /// cascades to else; the resolver's [`crate::resolve::ResolveError`]
    /// routes through the same `advance_or_terminate` dispatch as a
    /// natural Failed reap and a spawn-failure.
    ///
    /// Shape: `when` references `${env.MISSING}` (no default) against
    /// an empty [`EnvSnapshot`]; the resolver returns `UnsetEnvVar`
    /// before any process spawns. The actuator synthesises `Failed` at
    /// cursor 0 → predicate op's `on_failed` resolves to `Continue(2)`
    /// (the else-branch's first op) → spawn the else-branch
    /// (literal `/bin/else`).
    ///
    /// **Why this is a distinct test from
    /// [`predicate_spawn_failure_does_not_propagate_no_else`]**:
    /// resolver failure short-circuits in
    /// [`crate::resolve::resolve_step`] before `Spawner::spawn` is
    /// reached at all (different code path from OS-level
    /// spawn-failure). And **why distinct from
    /// [`predicate_failed_spawns_else_branch_outcome_does_not_propagate`]**:
    /// that test reaps a natural Failed from a real spawn; this test
    /// has zero predicate spawns recorded — the synth-Failed dispatch
    /// must work without any in-flight `RunningJob` for cursor 0.
    /// Together they pin "the dispatch loop is uniform on bytecode
    /// shape" across all three Failed-at-cursor-0 sources.
    #[test]
    fn predicate_resolver_failure_cascades_to_else_branch() {
        // Three-op CFG: predicate(${env.MISSING}) → on_ok = Continue(1)
        // (then), on_failed = Continue(2) (else). Then-Exec on_ok =
        // Escape, on_failed = Terminate. Else-Exec on_ok = Escape,
        // on_failed = Terminate.
        let program = {
            let mut b = ProgramBuilder::new();
            let pred = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::EnvVar {
                    name: CompactString::new("MISSING"),
                    default: None,
                },
            ])])));
            let then_first = b.continue_to_next();
            b.patch_on_ok(pred, then_first).unwrap();
            let then_h = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/then"),
            ])])));
            let else_first = b.continue_to_next();
            b.patch_on_failed(pred, else_first).unwrap();
            b.patch_on_ok(then_h, BranchTarget::Escape).unwrap();
            b.patch_on_failed(then_h, BranchTarget::Terminate).unwrap();
            let else_h = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/else"),
            ])])));
            b.patch_on_ok(else_h, BranchTarget::Escape).unwrap();
            b.patch_on_failed(else_h, BranchTarget::Terminate).unwrap();
            Arc::new(b.build().unwrap())
        };
        let mut h = Harness::new_with_grace_and_env(
            4,
            Duration::from_secs(5),
            Arc::new(EnvSnapshot::from_map::<_, &str, &str>([])),
        );
        h.submit(make_effect_perfile_with_program(5, 5, 5, 1, program));

        // The else-branch spawn must be the only spawn recorded — the
        // predicate's resolver-failure short-circuits before any
        // `MockSpawner::spawn` call.
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        assert_eq!(
            s.len(),
            1,
            "exactly one spawn recorded (the else-branch); predicate's resolver-failure \
             must not reach the spawner",
        );
        assert_eq!(
            s[0].argv,
            vec!["/bin/else".to_string()],
            "the spawn is the else-branch, not the then-branch (cursor 1 was skipped)",
        );
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Ok,
                    ..
                }
            ),
            "plan terminates Ok — predicate's resolver-failure does not propagate; \
             got {:?}",
            completions[0],
        );
        h.shutdown();
    }

    /// Multi-instruction plan with a conditional in the middle:
    /// `[Exec(A), Predicate(B), Exec(C)]` (B with no else, jump past
    /// C). When B fires Ok, C runs as the predicate's then-branch.
    /// When B fires Failed, the plan terminates Ok after B (without
    /// running C).
    ///
    /// This pins the "predicate is one instruction within a larger
    /// sequence" shape — the predicate slot at cursor 1 doesn't end
    /// the plan in either outcome; the dispatcher decides based on
    /// the conditional's structure.
    #[test]
    fn predicate_within_sequence_skips_or_runs_then() {
        let prog_path = || {
            // CFG-shape mirror of `[exec=a, when=b then=[exec=c]]`:
            //   op 0: Exec(a) — on_ok = Continue(1), on_failed = Terminate
            //   op 1: predicate b — on_ok = Continue(2) (then),
            //                       on_failed = Escape (no-else branch elision)
            //   op 2: Exec(c) — on_ok = Escape, on_failed = Terminate
            let mut b = ProgramBuilder::new();
            let a = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/a"),
            ])])));
            let after_a = b.continue_to_next();
            b.patch_on_ok(a, after_a).unwrap();
            b.patch_on_failed(a, BranchTarget::Terminate).unwrap();
            let pred = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/b"),
            ])])));
            let after_pred = b.continue_to_next();
            b.patch_on_ok(pred, after_pred).unwrap();
            b.patch_on_failed(pred, BranchTarget::Escape).unwrap();
            let c = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/c"),
            ])])));
            b.patch_on_ok(c, BranchTarget::Escape).unwrap();
            b.patch_on_failed(c, BranchTarget::Terminate).unwrap();
            Arc::new(b.build().unwrap())
        };

        // Path 1: B reaps Ok → C runs.
        {
            let mut h = Harness::new(4);
            h.submit(make_effect_perfile_with_program(10, 10, 10, 1, prog_path()));
            let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
            assert_eq!(s0[0].argv, vec!["/bin/a".to_string()]);
            h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
            let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
            assert_eq!(s1[1].argv, vec!["/bin/b".to_string()]);
            h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();
            let s2 = h.wait_for_spawns(3, Duration::from_secs(1));
            assert_eq!(s2[2].argv, vec!["/bin/c".to_string()]);
            h.spawner.complete(s2[2].pid, EffectOutcome::Ok).unwrap();
            let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
            assert!(matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Ok,
                    ..
                }
            ));
            h.shutdown();
        }

        // Path 2: B reaps Failed → C is skipped, plan terminates Ok.
        {
            let mut h = Harness::new(4);
            h.submit(make_effect_perfile_with_program(11, 11, 11, 1, prog_path()));
            let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
            h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
            let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
            h.spawner
                .complete(
                    s1[1].pid,
                    EffectOutcome::Failed {
                        exit_code: Some(1),
                        signal: None,
                    },
                )
                .unwrap();
            // C must not spawn.
            thread::sleep(Duration::from_millis(50));
            assert_eq!(
                h.spawner.spawns().len(),
                2,
                "C must not run when predicate B fails",
            );
            let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
            assert!(
                matches!(
                    completions[0],
                    Input::EffectComplete {
                        result: EffectOutcome::Ok,
                        ..
                    }
                ),
                "predicate Failed past plan end ⇒ Ok terminus; got {:?}",
                completions[0],
            );
            h.shutdown();
        }
    }

    // ---------- pipe dispatch (Pipe body) ----------
    //
    // The pipe variant is the heaviest behavioural slice of PR4: a
    // single op with `SpawnBody::Pipe` triggers N spawns, an
    // aggregating waiter, a combined signaler for shutdown, and
    // optional per-stage timers. These tests exercise the dispatcher
    // wiring against the testkit `MockSpawner::spawn_pipe`.

    /// Build a single-op program wrapping a pipe body. `on_ok = Escape`,
    /// `on_failed = Terminate`.
    fn pipe_program(stages: Arc<[ExecAction]>) -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Pipe(stages));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// Two-stage pipe with both stages Ok: aggregated outcome is Ok;
    /// the actuator emits exactly one EffectComplete (the engine's
    /// per-Effect accounting is unchanged under pipe vs single-exec).
    #[test]
    fn pipe_two_stages_both_ok_emits_single_ok_completion() {
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])]),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])]),
        ]);
        let program = pipe_program(stages);

        let mut h = Harness::new(4);
        h.submit(make_effect_perfile_with_program(50, 50, 50, 1, program));
        // Both stages spawn at once.
        let spawns = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(spawns.len(), 2);
        assert_eq!(spawns[0].argv, vec!["/bin/a".to_string()]);
        assert_eq!(spawns[1].argv, vec!["/bin/b".to_string()]);
        // The mock's per-stage completion channels are independent;
        // the aggregating waiter drains in spawn order, so completing
        // stage 0 first matches the production sequence.
        h.spawner
            .complete(spawns[0].pid, EffectOutcome::Ok)
            .unwrap();
        h.spawner
            .complete(spawns[1].pid, EffectOutcome::Ok)
            .unwrap();
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert_eq!(completions.len(), 1, "exactly one EffectComplete per pipe");
        assert!(matches!(
            completions[0],
            Input::EffectComplete {
                result: EffectOutcome::Ok,
                ..
            }
        ));
        h.shutdown();
    }

    /// Two-stage pipe with stage 0 Failed: aggregated outcome is
    /// Failed; the cascade SIGTERMs stage 1 before its mock
    /// completion lands (so the test records the signal). After the
    /// cascade, the test completes stage 1 with a Failed-by-signal
    /// outcome to unblock the aggregator.
    #[test]
    fn pipe_first_stage_failed_cascades_sigterm_to_siblings() {
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])]),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])]),
        ]);
        let program = pipe_program(stages);

        let mut h = Harness::new(4);
        h.submit(make_effect_perfile_with_program(51, 51, 51, 1, program));
        let spawns = h.wait_for_spawns(2, Duration::from_secs(1));
        let stage0_pid = spawns[0].pid;
        let stage1_pid = spawns[1].pid;

        // Complete stage 0 Failed; the aggregating waiter will
        // observe this and cascade SIGTERM to stage 1.
        h.spawner
            .complete(
                stage0_pid,
                EffectOutcome::Failed {
                    exit_code: Some(7),
                    signal: None,
                },
            )
            .unwrap();
        // Wait for the cascade SIGTERM to land. The mock signaler
        // records Term(pid) on `signal_term`. Poll briefly.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let signals = h.spawner.signals();
            if signals.contains(&SignalRecord::Term(stage1_pid)) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "expected SIGTERM cascade to stage 1; signals={signals:?}",
            );
            thread::sleep(Duration::from_millis(5));
        }
        // Complete stage 1 (as if SIGTERM took effect) so the
        // aggregator's wait finishes and the EffectComplete arrives.
        h.spawner
            .complete(
                stage1_pid,
                EffectOutcome::Failed {
                    exit_code: None,
                    signal: Some(15),
                },
            )
            .unwrap();
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete { result, .. } => {
                assert!(matches!(
                    result,
                    EffectOutcome::Failed {
                        exit_code: Some(7),
                        signal: Some(15),
                    }
                ));
            }
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
        h.shutdown();
    }

    /// Pipe spawn fails: the actuator routes through the standard
    /// `SpawnError::Failed(SpawnFailureCause::OsSpawn)` path and emits
    /// one Failed completion. No spawns are recorded against the mock
    /// (the injected error short-circuits before stages are minted).
    #[test]
    fn pipe_spawn_failure_terminates_plan_failed() {
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])]),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])]),
        ]);
        let program = pipe_program(stages);

        let mut h = Harness::new(4);
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile_with_program(52, 52, 52, 1, program));
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(matches!(
            completions[0],
            Input::EffectComplete {
                result: EffectOutcome::Failed {
                    exit_code: None,
                    signal: None,
                },
                ..
            }
        ));
        // No stages recorded — the inject_spawn_error path short-
        // circuits MockSpawner::spawn_pipe before allocate_spawn.
        assert_eq!(
            h.spawner.spawns().len(),
            0,
            "no stages recorded when pipe spawn fails",
        );
        h.shutdown();
    }

    /// Pipe followed by another action in the same program: pipe
    /// Ok ⇒ next action runs; pipe Failed ⇒ next action is skipped
    /// (stop-on-failure semantics, same as a plain Exec).
    #[test]
    fn pipe_followed_by_exec_runs_only_on_pipe_ok() {
        let pipe_stages: Arc<[ExecAction]> = Arc::from(vec![
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])]),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])]),
        ]);
        let program = {
            let mut b = ProgramBuilder::new();
            let p = b.emit(SpawnBody::Pipe(pipe_stages));
            let after = b.continue_to_next();
            b.patch_on_ok(p, after).unwrap();
            b.patch_on_failed(p, BranchTarget::Terminate).unwrap();
            let exec_after = b.emit(SpawnBody::Exec(ExecAction::new([ArgTemplate::new([
                ArgPart::literal("/bin/after"),
            ])])));
            b.patch_on_ok(exec_after, BranchTarget::Escape).unwrap();
            b.patch_on_failed(exec_after, BranchTarget::Terminate)
                .unwrap();
            Arc::new(b.build().unwrap())
        };

        // Path 1: pipe Ok → /bin/after runs.
        {
            let mut h = Harness::new(4);
            h.submit(make_effect_perfile_with_program(
                53,
                53,
                53,
                1,
                Arc::clone(&program),
            ));
            let pipe_spawns = h.wait_for_spawns(2, Duration::from_secs(1));
            h.spawner
                .complete(pipe_spawns[0].pid, EffectOutcome::Ok)
                .unwrap();
            h.spawner
                .complete(pipe_spawns[1].pid, EffectOutcome::Ok)
                .unwrap();
            let after_spawns = h.wait_for_spawns(3, Duration::from_secs(1));
            assert_eq!(after_spawns[2].argv, vec!["/bin/after".to_string()]);
            h.spawner
                .complete(after_spawns[2].pid, EffectOutcome::Ok)
                .unwrap();
            let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
            assert!(matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Ok,
                    ..
                }
            ));
            h.shutdown();
        }

        // Path 2: pipe Failed → /bin/after must NOT run.
        {
            let mut h = Harness::new(4);
            h.submit(make_effect_perfile_with_program(54, 54, 54, 1, program));
            let pipe_spawns = h.wait_for_spawns(2, Duration::from_secs(1));
            h.spawner
                .complete(
                    pipe_spawns[0].pid,
                    EffectOutcome::Failed {
                        exit_code: Some(1),
                        signal: None,
                    },
                )
                .unwrap();
            h.spawner
                .complete(pipe_spawns[1].pid, EffectOutcome::Ok)
                .unwrap();
            let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
            assert!(matches!(
                completions[0],
                Input::EffectComplete {
                    result: EffectOutcome::Failed { .. },
                    ..
                }
            ));
            // /bin/after must not have spawned. Recorded spawns = 2
            // (the two pipe stages); a third would mean stop-on-fail
            // broke for SpawnPipe.
            thread::sleep(Duration::from_millis(50));
            assert_eq!(
                h.spawner.spawns().len(),
                2,
                "post-pipe SpawnExec must not run when pipe Failed",
            );
            h.shutdown();
        }
    }

    /// Per-stage timeout: the pipe carries a stage whose
    /// `ExecAction.timeout` is set. The per-stage timer thread fires
    /// at the deadline and signals SIGTERM. The test verifies the
    /// recorded signal lands on the right pid.
    #[test]
    fn pipe_stage_timeout_sigterms_unfinished_stage() {
        let timeout = Duration::from_millis(60);
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])]),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])]).with_timeout(timeout),
        ]);
        let program = pipe_program(stages);

        // Short shutdown_grace so the SIGKILL escalation also lands
        // inside the test window if SIGTERM doesn't take effect.
        let mut h = Harness::new_with_grace(4, Duration::from_millis(20));
        h.submit(make_effect_perfile_with_program(55, 55, 55, 1, program));
        let spawns = h.wait_for_spawns(2, Duration::from_secs(1));
        let stage0_pid = spawns[0].pid;
        let stage1_pid = spawns[1].pid;

        // Wait for the per-stage timer to deliver SIGTERM to stage 1.
        let deadline = Instant::now() + Duration::from_secs(2);
        loop {
            let signals = h.spawner.signals();
            if signals.contains(&SignalRecord::Term(stage1_pid)) {
                break;
            }
            assert!(
                Instant::now() < deadline,
                "expected SIGTERM from per-stage timer on stage 1; signals={signals:?}",
            );
            thread::sleep(Duration::from_millis(5));
        }
        // Stage 0 has no timeout — must not receive a SIGTERM from
        // the per-stage timer. The cascade-on-failure path also
        // doesn't reach stage 0 (idx 0 → siblings idx+1..n, which
        // doesn't include stage 0 itself).
        let signals = h.spawner.signals();
        assert!(
            !signals.contains(&SignalRecord::Term(stage0_pid)),
            "stage 0 (no timeout) must not receive timer-driven SIGTERM",
        );

        // Complete both stages so the aggregator finishes.
        // Stage 1 reports as Failed-by-signal (the timeout took effect).
        h.spawner
            .complete(
                stage1_pid,
                EffectOutcome::Failed {
                    exit_code: None,
                    signal: Some(15),
                },
            )
            .unwrap();
        // The aggregator on stage 1's failure cascades SIGTERM to
        // *later* siblings (none here), then continues draining.
        // Stage 0 hasn't completed yet — drain it so the wait
        // finishes.
        h.spawner.complete(stage0_pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(2));
        h.shutdown();
    }
}
