//! Subprocess pool controller — single thread, drains submits + reaps +
//! shutdown, owns slot state.
//!
//! Channel topology (`N` = resolved concurrency, [`default_concurrency`]):
//!
//! ```text
//! bin --(effects, bounded(1024))--> Controller
//! Controller --(EffectCompleteSender)--> Engine
//! Controller <--(reap_rx, bounded(N))-- WaitThread × N
//! bin --(shutdown, bounded(1) broadcast)--> Controller
//! ```
//!
//! The engine-bound completion edge is a trait
//! ([`crate::EffectCompleteSender`]) rather than a concrete channel —
//! the actuator does not name the engine's `Input` vocabulary. The bin
//! owns the wrapper that lifts the [`EffectCompletion`] envelope into
//! `Input::EffectComplete`; this crate ships the envelope unchanged
//! from the wait thread all the way to the trait boundary.
//!
//! Shutdown sequence: SIGTERM all running, drain reaps for 5s,
//! SIGKILL stragglers, drain remaining reaps.

mod state;
use crate::EffectCompleteSender;
use crate::env::EnvSnapshot;
use crate::spawner::Spawner;
use crossbeam::channel::{Receiver, Sender};
use specter_core::{EffectCompletion, EffectOp};
use state::ActuatorState;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Production default for [`SubprocessActuator::new`] when the bin's
/// `--concurrency` flag is unset: `2 × available_parallelism()`.
///
/// Falls back to `4` if either query fails. The bin calls this and
/// passes the [`NonZeroUsize`] result into [`SubprocessActuator::new`]
/// directly — the `0`-as-sentinel pattern is typed away; everything
/// below the constructor receives a [`NonZeroUsize`] and trusts it.
#[must_use]
pub fn default_concurrency() -> NonZeroUsize {
    let fallback = NonZeroUsize::new(4).expect("4 is non-zero");
    std::thread::available_parallelism()
        .ok()
        .and_then(|n| NonZeroUsize::new(n.get().saturating_mul(2)))
        .unwrap_or(fallback)
}

/// SIGTERM → SIGKILL grace, pinned in one place so the shutdown drain
/// and per-step timer threads can't drift apart.
///
/// Read by:
/// - [`SubprocessActuator::shutdown`] (the SIGTERM → grace → SIGKILL
///   sequence).
/// - [`crate::timer::arm_timer`] (per-step deadline enforcement).
///
/// Default for production; tests may override via
/// `SubprocessActuator::new_with_grace` /
/// `SubprocessActuator::new_with_grace_and_env`.
pub(crate) const SHUTDOWN_GRACE: Duration = Duration::from_secs(5);

/// Channels the actuator's controller owns for the lifetime of
/// [`SubprocessActuator::run`].
///
/// Bundles the four reactive-surface channels — the effects pipe and the
/// three shutdown-handshake legs — that the bin passes as a unit. The bin's
/// `crate::pool`-paired transport bundle ([`channels::ActuatorIO::pair`] in
/// `specter-bin`) returns the matching [`RunWiring`] directly so the
/// actuator's contract is one owned struct, not four positional arguments.
///
/// [`channels::ActuatorIO::pair`]: # "in specter-bin"
#[derive(Debug)]
#[must_use]
pub struct RunWiring {
    /// Effects pipe. The controller drains [`EffectOp`]s via `select!`
    /// against the shutdown legs.
    pub effects_rx: Receiver<EffectOp>,
    /// Soft-shutdown pulse. Drained once at the start of the graceful
    /// stop arm (SIGTERM-then-wait fanout with the grace window).
    pub shutdown_rx: Receiver<()>,
    /// Hard-shutdown pulse. Operator double-Ctrl-C: pre-empts the
    /// grace window and proceeds directly to phase 3's SIGKILL fanout.
    pub hard_shutdown_rx: Receiver<()>,
    /// Phase-3 fanout confirmation. The controller pulses once after
    /// SIGKILL fanout completes; the bin's hard-exit path waits on the
    /// paired receiver before `process::exit(130)` so the parent never
    /// aborts mid-fanout.
    pub hard_shutdown_done_tx: Sender<()>,
}

/// Build the [`EffectCompletion`] back-channel sized to the resolved
/// concurrency.
///
/// One slot per in-flight `pool::state::RunningJob`: every wait
/// thread sends exactly one [`EffectCompletion`] in its lifetime
/// (`state::wait_loop` after `drop(permit)`), and the live wait-thread
/// count is bounded by the permit cap. So a fully-saturated pool
/// draining in lock-step never blocks a permit-released wait thread on
/// `reap_tx.send` waiting for the controller to consume —
/// backpressure becomes a single-slot blip only when the controller has
/// fallen behind the spawn rate, and the bin's bounded(1024)
/// `effects_rx` upstream sets the operator-visible backpressure ceiling
/// anyway.
///
/// Shared between [`SubprocessActuator::new`] and the test constructors
/// so the cap stays single-source — the upper bound is a property of
/// the wait-thread protocol, not the entry point.
fn reap_channel(resolved: NonZeroUsize) -> (Sender<EffectCompletion>, Receiver<EffectCompletion>) {
    crossbeam::channel::bounded::<EffectCompletion>(resolved.get())
}

/// The actuator's controller. One per process. Owns the slot map, ready
/// queue, per-Sub counter, and global semaphore. Blocks in [`Self::run`]
/// for the lifetime of the bin process.
#[derive(Debug)]
pub struct SubprocessActuator {
    state: ActuatorState,
    reap_tx: Sender<EffectCompletion>,
    reap_rx: Receiver<EffectCompletion>,
}

impl SubprocessActuator {
    /// Construct with `concurrency` global permits. The bin resolves
    /// "unset → default" via [`default_concurrency`] before this call;
    /// the [`NonZeroUsize`] type retires the `0`-as-sentinel pattern,
    /// so everything below `ActuatorState::new` receives a non-zero
    /// value and trusts it.
    ///
    /// Captures three pieces of startup-immutable process state — the
    /// env snapshot, the temp directory, and the actuator pid — once
    /// here so the spawn path makes no `getenv` / `getpid` syscall per
    /// Effect. All three live on `ActuatorState` for the actuator's
    /// lifetime; the env snapshot is shared by `Arc` across resolver
    /// calls, `temp_dir` is shared by `Arc<Path>` across
    /// `DiffTmpFile::create` calls, and `actuator_pid` is a copy.
    #[must_use]
    pub fn new(concurrency: NonZeroUsize) -> Self {
        let (reap_tx, reap_rx) = reap_channel(concurrency);
        Self {
            state: ActuatorState::new(
                concurrency,
                Arc::new(EnvSnapshot::capture()),
                Arc::from(std::env::temp_dir().into_boxed_path()),
                std::process::id(),
                SHUTDOWN_GRACE,
            ),
            reap_tx,
            reap_rx,
        }
    }

    /// Test-only constructor with a custom shutdown grace and a
    /// preconstructed env snapshot; `temp_dir` and `actuator_pid`
    /// default to ambient process values. Used by tests that need
    /// to assert shutdown timing or `${env.<NAME>}` resolution
    /// (strict-unset → Failed, default rendering, etc.) without
    /// depending on the ambient process env. For tests that *also*
    /// need to override the tmp directory (the tmp-dir-from-state
    /// fence), see [`Self::new_with_grace_and_env_and_tmp`].
    ///
    /// Gated to match the test module (`cfg(all(test, feature = "testkit"))`)
    /// — without `testkit`, the test module that consumes this constructor
    /// is excluded too, so the function would otherwise be flagged as
    /// dead code under `cargo test --lib` (no features).
    #[cfg(all(test, feature = "testkit"))]
    pub(crate) fn new_with_grace_and_env(
        concurrency: NonZeroUsize,
        grace: Duration,
        env: Arc<EnvSnapshot>,
    ) -> Self {
        Self::new_with_grace_and_env_and_tmp(
            concurrency,
            grace,
            env,
            Arc::from(std::env::temp_dir().into_boxed_path()),
            std::process::id(),
        )
    }

    /// Test-only constructor that lets callers override every piece
    /// of startup-immutable state — `temp_dir` and `actuator_pid`
    /// alongside the env and grace. The single use is the fence that
    /// asserts `DiffTmpFile::create` reads from
    /// `ActuatorState.temp_dir`, not `std::env::temp_dir()`.
    #[cfg(all(test, feature = "testkit"))]
    pub(crate) fn new_with_grace_and_env_and_tmp(
        concurrency: NonZeroUsize,
        grace: Duration,
        env: Arc<EnvSnapshot>,
        temp_dir: Arc<std::path::Path>,
        actuator_pid: u32,
    ) -> Self {
        let (reap_tx, reap_rx) = reap_channel(concurrency);
        Self {
            state: ActuatorState::new(concurrency, env, temp_dir, actuator_pid, grace),
            reap_tx,
            reap_rx,
        }
    }

    /// Block until shutdown. Drains [`EffectOp`]s (submit + cancel)
    /// off `wiring.effects_rx`, dispatches to spawner / cancel handler,
    /// reaps wait threads, propagates [`EffectCompletion`] envelopes
    /// through `engine_in`. Returns when `wiring.effects_rx` disconnects
    /// or `wiring.shutdown_rx` signals; performs the SIGTERM → 5s grace
    /// → SIGKILL sequence on the way out. If `wiring.hard_shutdown_rx`
    /// fires (operator pressed Ctrl-C twice within `HARD_EXIT_WINDOW`),
    /// the grace is pre-empted: the loop breaks immediately, the
    /// SIGTERM phase still runs (cheap; gives well-behaved children a
    /// chance to exit cleanly), then phase 2's grace becomes a
    /// near-zero wait before phase 3 SIGKILLs everything still alive.
    ///
    /// `wiring.hard_shutdown_done_tx` is the back-channel to the signal
    /// thread: the actuator pulses it once at the close of phase 3
    /// SIGKILL fanout (trigger-agnostic — pulse fires whenever phase 3
    /// runs, regardless of soft/hard origin). On the hard-exit path
    /// the signal thread waits for this pulse (or the sender-drop
    /// that follows thread exit) before `process::exit(130)`, so the
    /// parent never aborts mid-fanout and leaves orphans on PID 1.
    ///
    /// `engine_in` is `&dyn` — the controller borrows the sink for the
    /// duration of [`Self::run`] without owning it. The closure
    /// surrounding the spawn site in the bin keeps the
    /// `Box<dyn EffectCompleteSender>` owned for the call's lifetime
    /// and passes `&*box` here, symmetric with the `&dyn Spawner`
    /// calling convention this function already uses. Wait threads send
    /// [`EffectCompletion`] envelopes through `self.reap_tx`; only the
    /// controller calls `engine_in.send(...)`, so a single-threaded
    /// `&dyn` is sufficient.
    ///
    /// `wiring` is taken by value: the controller owns the channels for
    /// the lifetime of [`Self::run`], so the caller hands off and is
    /// freed from any borrow-tracking. The `select!` block borrows
    /// channels by reference per-iteration, which is what trips
    /// clippy's `needless_pass_by_value` — keep the allow to express
    /// "owned by run, used by-ref inside the body" honestly.
    #[allow(clippy::needless_pass_by_value)]
    pub fn run(
        &mut self,
        wiring: RunWiring,
        engine_in: &dyn EffectCompleteSender,
        spawner: &dyn Spawner,
    ) {
        let RunWiring {
            effects_rx,
            shutdown_rx,
            hard_shutdown_rx,
            hard_shutdown_done_tx,
        } = wiring;
        let mut hard = false;
        loop {
            crossbeam::select! {
                recv(effects_rx) -> msg => match msg {
                    Ok(EffectOp::Submit(effect)) => {
                        self.state.handle_submit(effect, spawner, &self.reap_tx, engine_in);
                    }
                    Ok(EffectOp::Cancel { profile }) => {
                        // Engine-driven abandon: SIGTERM in-flight effects
                        // for `profile`, drop queued work for the same
                        // profile. The wait threads still drive natural
                        // reap → handle_reap → terminate_plan, which emits
                        // EffectComplete; the engine routes that late
                        // completion to EffectCompleteOutsideAwaiting.
                        self.state.handle_cancel(profile);
                    }
                    Err(crossbeam::channel::RecvError) => {
                        // Bin closed `effects_tx` — the driver's
                        // `ActuatorIO` dropped. Distinct trigger from
                        // the `shutdown_rx` / `hard_shutdown_rx` arms
                        // below: those are operator-signal-driven and
                        // arrive AHEAD of the channel close; this arm
                        // fires when the driver's shutdown path either
                        // bypassed the soft-pulse (panic mid-`run`) or
                        // already drained the pulse before disconnect.
                        // Logged at info so operators reading the tail
                        // can disambiguate the shutdown trigger from
                        // the channel-close vs signal-pulse paths.
                        tracing::info!("effects_rx disconnected; entering shutdown");
                        break;
                    }
                },
                recv(self.reap_rx) -> msg => match msg {
                    Ok(completion) => self.state.handle_reap(completion, engine_in, spawner, &self.reap_tx),
                    Err(_) => unreachable!(
                        "self.reap_tx keeps reap_rx connected for run's lifetime",
                    ),
                },
                recv(shutdown_rx) -> _ => break,
                recv(hard_shutdown_rx) -> _ => { hard = true; break; }
            }
        }
        self.shutdown(engine_in, hard, &hard_shutdown_rx, &hard_shutdown_done_tx);
    }

    fn shutdown(
        &mut self,
        engine_in: &dyn EffectCompleteSender,
        hard: bool,
        hard_shutdown_rx: &Receiver<()>,
        hard_shutdown_done_tx: &Sender<()>,
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
                    Ok(r) => self.state.handle_reap_drop(r, engine_in),
                    Err(crossbeam::channel::RecvError) => break,
                },
                recv(hard_shutdown_rx) -> _ => { grace = false; }
                default(deadline - now) => break,
            }
        }
        // Phase 3: SIGKILL stragglers, then pulse the back-channel.
        // Trigger-agnostic: the pulse semantics are "phase 3 ran", not
        // "phase 3 ran because of hard". The signal thread only reads
        // on the hard-exit path; a soft-shutdown pulse fills the
        // bounded(1) slot, nobody drains it, no semantic impact.
        //
        // Pulse-before-phase-4: SIGKILL is uninterruptible, so the
        // kernel will reap regardless of whether the actuator finishes
        // phase 4's reap drain. Waiting for phase 4 would bottleneck
        // the signal thread's `recv_timeout` on a 5s reap-drain
        // deadline; pulsing here releases it within microseconds of
        // the last `signal_kill`.
        if self.has_running() {
            tracing::info!("shutdown phase 3: SIGKILL stragglers");
            for slot in self.state.slots.values() {
                if let Some(job) = slot.running.as_ref()
                    && let Err(e) = job.signaler.signal_kill()
                {
                    tracing::debug!(pid = job.pid, ?e, "SIGKILL failed");
                }
            }
            let _ = hard_shutdown_done_tx.try_send(());
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
                Ok(r) => self.state.handle_reap_drop(r, engine_in),
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

impl Drop for SubprocessActuator {
    /// Safety net for drop paths the explicit `run → shutdown`
    /// pipeline can't reach: a panic mid-[`Self::run`] unwinds past
    /// the explicit `shutdown` call without firing it; a boot-fail
    /// that constructs but never runs the controller drops the
    /// actuator with empty state. The fanout below SIGTERMs then
    /// SIGKILLs every still-running child so wait threads' blocked
    /// `waitpid` calls return and the kernel reaps anything left
    /// over `_exit`.
    ///
    /// **No grace window, no reap drain.** Drop is the panic-recovery
    /// shape — clean exit goes through `Self::shutdown`, which owns
    /// the SIGTERM → grace → SIGKILL → reap-drain phasing. Here the
    /// invariant is "make the kernel-side cleanup unblockable, then
    /// return." Wait threads each hold a `reap_tx` clone; the channel
    /// stays connected for their final sends (the controller's clone
    /// drops only with this struct's `reap_tx` field, which is after
    /// Drop returns). Detached wait threads finish either inline with
    /// the kernel's reap or, in pathological cases, get reaped by the
    /// kernel on this process's `_exit`.
    ///
    /// On the happy path, `Self::shutdown` has already drained
    /// `state.slots` of running children (its phase 4 reaps every
    /// completion before returning), so the iteration below runs
    /// zero times.
    fn drop(&mut self) {
        for slot in self.state.slots.values() {
            let Some(job) = slot.running.as_ref() else {
                continue;
            };
            if let Err(e) = job.signaler.signal_term() {
                tracing::debug!(pid = job.pid, ?e, "drop-fallback SIGTERM failed",);
            }
            if let Err(e) = job.signaler.signal_kill() {
                tracing::debug!(pid = job.pid, ?e, "drop-fallback SIGKILL failed",);
            }
        }
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
    use super::RunWiring;
    use crate::env::EnvSnapshot;
    use crate::testkit::{MockSpawner, SignalRecord};
    use crate::{EffectCompleteSender, SendError, SubprocessActuator};
    use compact_str::CompactString;
    use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
    use specter_core::program::{BranchTarget, MultiStage, ProgramBuilder, SpawnBody};
    use specter_core::testkit::{predicate_then_program, single_exec_program};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, CorrelationId, Diff, Effect, EffectCommon,
        EffectCompletion, EffectOp, EffectOutcome, EffectTarget, ExecAction, Input, ProfileId,
        ResourceId, ResourceKind, SubId, Termination,
    };
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::thread;
    use std::time::{Duration, Instant};

    const fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("non-zero literal in test fixture")
    }

    /// Test adapter that lifts an [`EffectCompletion`] envelope into
    /// the engine-side `Input::EffectComplete` so the harness's
    /// `Receiver<Input>` continues to observe completions in the
    /// engine's vocabulary. Mirrors the bin's
    /// [`WakingEffectCompleteSender`] without dragging in the bin's
    /// transport identity.
    ///
    /// [`WakingEffectCompleteSender`]: # "in specter-bin"
    struct TestEngineIn(Sender<Input>);
    impl EffectCompleteSender for TestEngineIn {
        fn send(&self, completion: EffectCompletion) -> Result<(), SendError> {
            self.0
                .send(Input::EffectComplete(completion))
                .map_err(|_| SendError::Disconnected)
        }
    }

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
            let h = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/true")])],
                None,
            )));
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
        let common = EffectCommon {
            sub: unique_sub_id(sub_seed),
            profile: unique_profile_id(profile_seed),
            anchor: resource,
            correlation: CorrelationId::from(corr),
            forced: false,
            capture_output: false,
            sub_name: CompactString::new(""),
            program,
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            exclude: Arc::from(Vec::<CompactString>::new()),
        };
        Effect::per_file(
            common,
            resource,
            CompactString::new(""),
            Arc::new(Diff::default()),
        )
    }

    fn make_effect_subtree(sub_seed: u64, profile_seed: u64, corr: u64) -> Effect {
        let common = EffectCommon {
            sub: unique_sub_id(sub_seed),
            profile: unique_profile_id(profile_seed),
            anchor: unique_resource_id(profile_seed),
            correlation: CorrelationId::from(corr),
            forced: false,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: literal_program(),
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            exclude: Arc::from(Vec::<CompactString>::new()),
        };
        Effect::subtree(common, None)
    }

    /// Spawn the controller in a thread; return the channels + a join
    /// handle. `concurrency` is the global cap.
    ///
    /// Tests submit Effects via [`Self::submit`], which lifts `Effect`
    /// into [`EffectOp::Submit`] at the channel boundary; cancel-arm
    /// coverage lives in `pool/state.rs`'s direct `handle_cancel`
    /// tests against pre-loaded state.
    struct Harness {
        effects_tx: Sender<EffectOp>,
        shutdown_tx: Sender<()>,
        hard_shutdown_tx: Sender<()>,
        hard_shutdown_done_rx: Receiver<()>,
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
        fn new(concurrency: NonZeroUsize) -> Self {
            Self::new_with_grace_and_env(concurrency, Duration::from_secs(5), empty_env())
        }

        fn new_with_grace(concurrency: NonZeroUsize, grace: Duration) -> Self {
            Self::new_with_grace_and_env(concurrency, grace, empty_env())
        }

        fn new_with_grace_and_env(
            concurrency: NonZeroUsize,
            grace: Duration,
            env: Arc<EnvSnapshot>,
        ) -> Self {
            Self::spawn_controller(move || {
                SubprocessActuator::new_with_grace_and_env(concurrency, grace, env)
            })
        }

        /// Inject a custom `temp_dir` into the actuator state — the
        /// only call site is the fence that pins `start_plan` reading
        /// from `ActuatorState.temp_dir` rather than calling
        /// `std::env::temp_dir()` per Effect.
        fn new_with_grace_and_env_and_tmp(
            concurrency: NonZeroUsize,
            grace: Duration,
            env: Arc<EnvSnapshot>,
            temp_dir: Arc<Path>,
        ) -> Self {
            let pid = std::process::id();
            Self::spawn_controller(move || {
                SubprocessActuator::new_with_grace_and_env_and_tmp(
                    concurrency,
                    grace,
                    env,
                    temp_dir,
                    pid,
                )
            })
        }

        /// Common backbone: spawn the controller thread around a
        /// freshly-built actuator and wire the channels.
        fn spawn_controller(build: impl FnOnce() -> SubprocessActuator + Send + 'static) -> Self {
            let (effects_tx, effects_rx) = bounded::<EffectOp>(1024);
            let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
            let (hard_shutdown_tx, hard_shutdown_rx) = bounded::<()>(1);
            let (hard_shutdown_done_tx, hard_shutdown_done_rx) = bounded::<()>(1);
            let (engine_tx, engine_rx) = unbounded::<Input>();
            let engine_in: Box<dyn EffectCompleteSender> = Box::new(TestEngineIn(engine_tx));
            let spawner = Arc::new(MockSpawner::new());
            let spawner_clone = Arc::clone(&spawner);
            let join = thread::Builder::new()
                .name("test-actuator-controller".into())
                .spawn(move || {
                    let mut a = build();
                    let wiring = RunWiring {
                        effects_rx,
                        shutdown_rx,
                        hard_shutdown_rx,
                        hard_shutdown_done_tx,
                    };
                    a.run(wiring, &*engine_in, spawner_clone.as_ref());
                })
                .expect("spawn controller");
            Self {
                effects_tx,
                shutdown_tx,
                hard_shutdown_tx,
                hard_shutdown_done_rx,
                engine_in: engine_rx,
                spawner,
                join: Some(join),
            }
        }

        fn submit(&self, e: Effect) {
            self.effects_tx.send(EffectOp::Submit(e)).expect("submit");
        }

        /// Drop the controller's `effects_rx` peer by overwriting
        /// `self.effects_tx` with an orphan sender (one whose receiver
        /// was immediately dropped). The original tx — the actuator
        /// thread's only producer — drops here, so the controller's
        /// `effects_rx` observes `Disconnected` on the next select.
        ///
        /// The orphan tx is retained on `self` so an accidental
        /// subsequent `submit()` fails loudly with `SendError` instead
        /// of silently succeeding against a dead actuator.
        fn close_effects_tx_for_test(&mut self) {
            let (orphan_tx, orphan_rx) = bounded::<EffectOp>(1);
            drop(orphan_rx);
            let _orig = std::mem::replace(&mut self.effects_tx, orphan_tx);
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(4));
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete(c) => {
                assert!(matches!(c.outcome, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete; got {other:?}"),
        }
        h.shutdown();
    }

    // ---------- concurrency ----------

    #[test]
    fn concurrency_cap_blocks_excess() {
        let mut h = Harness::new(nz(2));
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(4));
        h.shutdown();
        assert!(
            h.spawner.signals().is_empty(),
            "no signals when nothing is running"
        );
    }

    // ---------- effects_rx disconnect (engine-thread-exit path) ----------

    #[test]
    fn effects_rx_disconnect_exits_controller_cleanly() {
        // The actuator's `effects_rx` observing Disconnected (driver-side
        // `effects_tx` dropped) routes through the new explicit `Err`
        // arm. Distinct from the `shutdown_rx` and `hard_shutdown_rx`
        // arms exercised by the existing shutdown tests. The trigger
        // is the engine-thread-exit path: production reaches this state
        // when the driver's `ActuatorIO` drops without a prior soft
        // pulse (panic mid-`run`, or a future shutdown shape that
        // bypasses the pulse).
        let mut h = Harness::new(nz(4));
        h.close_effects_tx_for_test();

        // Controller exits via the effects_rx Disconnected arm; the
        // shutdown phase observes no running children and returns fast.
        let join = h.join.take().expect("controller thread handle");
        join.join().expect("clean exit on effects_rx disconnect");
    }

    #[test]
    fn shutdown_sigterms_running_then_drains_reap() {
        let mut h = Harness::new(nz(4));
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
                .complete(pid, EffectOutcome::Failed(Termination::Signal(15)))
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
        let mut h = Harness::new_with_grace(nz(4), Duration::from_millis(150));
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
                .complete(pid, EffectOutcome::Failed(Termination::Signal(9)))
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
        // wait (phase 2), SIGKILL stragglers (phase 3), and emit the
        // `hard_shutdown_done_tx` pulse so the signal thread can proceed
        // with `process::exit` instead of relying on a sleep heuristic.
        // With a long grace (5s) configured, this test asserts that
        // SIGKILL lands *well* before the grace would have elapsed.
        let mut h = Harness::new_with_grace(nz(4), Duration::from_secs(5));
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
                .complete(pid, EffectOutcome::Failed(Termination::Signal(9)))
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
        // Phase 3 SIGKILL fanout pulse must reach the back-channel —
        // the signal thread's `recv_timeout` proof that the kernel has
        // been told to kill everyone, no sleep-heuristic required.
        assert!(
            h.hard_shutdown_done_rx.try_recv().is_ok(),
            "phase 3 fanout pulse must reach hard_shutdown_done_rx"
        );
    }

    #[test]
    fn shutdown_drops_pending_effects() {
        let mut h = Harness::new(nz(1));
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
        let mut h = Harness::new(nz(4));
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile(1, 1, 1, 1));
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete(c) => {
                assert!(matches!(
                    c.outcome,
                    EffectOutcome::Failed(Termination::Internal)
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
        let mut h = Harness::new(nz(1));
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
        // gate desynced. The fix routes synth-Failed dispatch directly
        // through `advance_or_terminate` on the controller call stack.
        //
        // This test exercises the simpler post-failure observable:
        // after a spawn-failure on (Sub=7, Profile=1, Resource=1), a
        // follow-up Effect for the *same* Sub on (Profile=2,
        // Resource=2) must spawn promptly. If `running_subs` still
        // contained Sub 7, the per-Sub gate would defer the second
        // submit indefinitely.
        let mut h = Harness::new(nz(4));
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
        use specter_core::{EntryKind, EntryRef, FsIdentity};

        let mut h = Harness::new(nz(4));
        let diff = Arc::new(Diff {
            created: smallvec![EntryRef {
                segment: CompactString::from("a.rs"),
                kind: EntryKind::File,
                fs_id: FsIdentity::synthetic(1, 0),
            }],
            ..Default::default()
        });
        let mut e = make_effect_perfile(1, 1, 1, 7);
        let resource = e.sort_key().1;
        let segment = CompactString::from(e.relative());
        e.target = EffectTarget::PerFile {
            resource,
            segment,
            diff: Arc::clone(&diff),
        };
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
        let mut h = Harness::new(nz(4));
        let e = make_effect_subtree(1, 1, 1);
        h.submit(e);
        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        assert!(s[0].env.iter().all(|(k, _)| k != "SPECTER_DIFF_PATH"));
        h.spawner.complete(s[0].pid, EffectOutcome::Ok).unwrap();
        h.wait_for_effect_completes(1, Duration::from_secs(1));
        h.shutdown();
    }

    /// `start_plan` must read from
    /// [`crate::pool::state::ActuatorState::temp_dir`] (captured
    /// once at actuator startup) when handing
    /// `DiffTmpFile::create` its `temp_dir` argument — not call
    /// `std::env::temp_dir()` per Effect. With a custom temp_dir
    /// distinct from the ambient one, a regression that reads from
    /// the process env would land the tmp file under `$TMPDIR`
    /// instead of the override, failing the prefix check below.
    #[test]
    fn diff_tmp_path_lives_under_actuator_state_temp_dir() {
        use compact_str::CompactString;
        use smallvec::smallvec;
        use specter_core::{EntryKind, EntryRef, FsIdentity};

        let custom = tempfile::tempdir().expect("custom tempdir");
        let custom_arc: Arc<Path> = Arc::from(custom.path().to_path_buf().into_boxed_path());

        let mut h = Harness::new_with_grace_and_env_and_tmp(
            nz(4),
            Duration::from_secs(5),
            empty_env(),
            Arc::clone(&custom_arc),
        );
        let diff = Arc::new(Diff {
            created: smallvec![EntryRef {
                segment: CompactString::from("a.rs"),
                kind: EntryKind::File,
                fs_id: FsIdentity::synthetic(1, 0),
            }],
            ..Default::default()
        });
        let mut e = make_effect_perfile(1, 1, 1, 7);
        let resource = e.sort_key().1;
        let segment = CompactString::from(e.relative());
        e.target = EffectTarget::PerFile {
            resource,
            segment,
            diff: Arc::clone(&diff),
        };
        h.submit(e);

        let s = h.wait_for_spawns(1, Duration::from_secs(1));
        let path = s[0]
            .env
            .iter()
            .find(|(k, _)| k == "SPECTER_DIFF_PATH")
            .expect("SPECTER_DIFF_PATH set")
            .1
            .clone();
        assert!(
            Path::new(&path).starts_with(custom.path()),
            "tmp path under custom temp_dir; got {path} vs {}",
            custom.path().display(),
        );
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
        let mut h = Harness::new(nz(4));
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
            Input::EffectComplete(c) => {
                assert!(matches!(c.outcome, EffectOutcome::Ok));
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
        let mut h = Harness::new(nz(4));
        let plan = n_step_program(3);
        h.submit(make_effect_perfile_with_program(2, 2, 2, 1, plan));

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        h.spawner
            .complete(s1[1].pid, EffectOutcome::Failed(Termination::Exit(7)))
            .unwrap();

        // No third spawn — the plan halted on step 1's failure.
        thread::sleep(Duration::from_millis(50));
        assert_eq!(h.spawner.spawns().len(), 2, "step 2 was not spawned");

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete(c) => assert!(matches!(
                c.outcome,
                EffectOutcome::Failed(Termination::Exit(7))
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
        use specter_core::{EntryKind, EntryRef, FsIdentity};

        let mut h = Harness::new(nz(4));
        let diff = Arc::new(Diff {
            created: smallvec![EntryRef {
                segment: CompactString::from("a.rs"),
                kind: EntryKind::File,
                fs_id: FsIdentity::synthetic(1, 0),
            }],
            ..Default::default()
        });
        let plan = n_step_program(2);
        let mut e = make_effect_perfile_with_program(3, 3, 3, 7, plan);
        let resource = e.sort_key().1;
        let segment = CompactString::from(e.relative());
        e.target = EffectTarget::PerFile {
            resource,
            segment,
            diff: Arc::clone(&diff),
        };
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
        let mut h = Harness::new(nz(4));
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
        let mut h = Harness::new(nz(1)); // cap=1: one global permit
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
        let mut h = Harness::new(nz(4));
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
            Input::EffectComplete(c) => {
                assert!(matches!(c.outcome, EffectOutcome::Ok));
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
        let mut h = Harness::new(nz(4));
        h.submit(make_effect_perfile_with_program(
            1,
            1,
            1,
            1,
            env_var_program("UNSET_VAR_AVOID_AMBIENT_COLLISION", None),
        ));
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete(c) => assert!(matches!(
                c.outcome,
                EffectOutcome::Failed(Termination::Internal)
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
        let mut h = Harness::new_with_grace_and_env(nz(4), Duration::from_secs(5), empty_env());
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
            nz(4),
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
        let h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal("/bin/true")])],
            Some(d),
        )));
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
        let mut h = Harness::new(nz(4));
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
            .complete(pid, EffectOutcome::Failed(Termination::Signal(15)))
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
        let mut h = Harness::new(nz(4));
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
            ExecAction::new([ArgTemplate::new([ArgPart::literal(when_label)])], None),
            ExecAction::new([ArgTemplate::new([ArgPart::literal(then_label)])], None),
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
        let pred = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal(when_label)])],
            None,
        )));
        // then enters at cursor 1
        let then_first = b.continue_to_next();
        b.patch_on_ok(pred, then_first).unwrap();
        let then_h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal(then_label)])],
            None,
        )));
        // else enters at cursor 2 — patch predicate's on_failed to it,
        // and then-Exec's on_ok is Escape (skip past else).
        let else_first = b.continue_to_next();
        b.patch_on_failed(pred, else_first).unwrap();
        b.patch_on_ok(then_h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(then_h, BranchTarget::Terminate).unwrap();
        let else_h = b.emit(SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal(else_label)])],
            None,
        )));
        b.patch_on_ok(else_h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(else_h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    /// Predicate reaping Ok enters the then-branch: the actuator
    /// spawns the then-exec after the predicate reaps. Exactly one
    /// EffectComplete is emitted (Ok) at plan terminus.
    #[test]
    fn predicate_ok_spawns_then_branch_and_terminates_ok() {
        let mut h = Harness::new(nz(4));
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
            &completions[0],
            Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
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
        let mut h = Harness::new(nz(4));
        let program = predicate_then_else("/bin/check", "/bin/then", "/bin/else");
        h.submit(make_effect_perfile_with_program(2, 2, 2, 1, program));

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner
            .complete(s0[0].pid, EffectOutcome::Failed(Termination::Exit(99)))
            .unwrap();

        // Predicate Failed → jump to else_start. Else-exec spawns;
        // then-exec is skipped.
        let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
        assert_eq!(s1[1].argv, vec!["/bin/else".to_string()]);
        h.spawner.complete(s1[1].pid, EffectOutcome::Ok).unwrap();

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
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
        let mut h = Harness::new(nz(4));
        let program = predicate_then_no_else("/bin/check", "/bin/then");
        h.submit(make_effect_perfile_with_program(3, 3, 3, 1, program));

        let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
        h.spawner
            .complete(s0[0].pid, EffectOutcome::Failed(Termination::Exit(7)))
            .unwrap();

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
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
    /// `EffectOutcome::Ok`. Short-circuiting spawn-failure straight
    /// to terminus would emit `EffectComplete::Failed` instead; the
    /// Ok outcome here is the no-propagation invariant in observable
    /// form.
    ///
    /// Zero spawns are recorded — the injection short-circuits
    /// `MockSpawner::spawn` before the `SpawnRecord` push.
    #[test]
    fn predicate_spawn_failure_does_not_propagate_no_else() {
        let mut h = Harness::new(nz(4));
        let program = predicate_then_no_else("/bin/check", "/bin/then");
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile_with_program(4, 4, 4, 1, program));

        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(
            matches!(
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
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
            let pred = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::EnvVar {
                    name: CompactString::new("MISSING"),
                    default: None,
                }])],
                None,
            )));
            let then_first = b.continue_to_next();
            b.patch_on_ok(pred, then_first).unwrap();
            let then_h = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/then")])],
                None,
            )));
            let else_first = b.continue_to_next();
            b.patch_on_failed(pred, else_first).unwrap();
            b.patch_on_ok(then_h, BranchTarget::Escape).unwrap();
            b.patch_on_failed(then_h, BranchTarget::Terminate).unwrap();
            let else_h = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/else")])],
                None,
            )));
            b.patch_on_ok(else_h, BranchTarget::Escape).unwrap();
            b.patch_on_failed(else_h, BranchTarget::Terminate).unwrap();
            Arc::new(b.build().unwrap())
        };
        let mut h = Harness::new_with_grace_and_env(
            nz(4),
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
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
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
            let a = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/a")])],
                None,
            )));
            let after_a = b.continue_to_next();
            b.patch_on_ok(a, after_a).unwrap();
            b.patch_on_failed(a, BranchTarget::Terminate).unwrap();
            let pred = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/b")])],
                None,
            )));
            let after_pred = b.continue_to_next();
            b.patch_on_ok(pred, after_pred).unwrap();
            b.patch_on_failed(pred, BranchTarget::Escape).unwrap();
            let c = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/c")])],
                None,
            )));
            b.patch_on_ok(c, BranchTarget::Escape).unwrap();
            b.patch_on_failed(c, BranchTarget::Terminate).unwrap();
            Arc::new(b.build().unwrap())
        };

        // Path 1: B reaps Ok → C runs.
        {
            let mut h = Harness::new(nz(4));
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
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
            ));
            h.shutdown();
        }

        // Path 2: B reaps Failed → C is skipped, plan terminates Ok.
        {
            let mut h = Harness::new(nz(4));
            h.submit(make_effect_perfile_with_program(11, 11, 11, 1, prog_path()));
            let s0 = h.wait_for_spawns(1, Duration::from_secs(1));
            h.spawner.complete(s0[0].pid, EffectOutcome::Ok).unwrap();
            let s1 = h.wait_for_spawns(2, Duration::from_secs(1));
            h.spawner
                .complete(s1[1].pid, EffectOutcome::Failed(Termination::Exit(1)))
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
                    &completions[0],
                    Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
                ),
                "predicate Failed past plan end ⇒ Ok terminus; got {:?}",
                completions[0],
            );
            h.shutdown();
        }
    }

    // ---------- pipe dispatch (Pipe body) ----------
    //
    // A single op with `SpawnBody::Pipe` triggers N spawns, an
    // aggregating waiter, a combined signaler for shutdown, and
    // optional per-stage timers. These tests exercise the dispatcher
    // wiring against the testkit `MockSpawner::spawn_pipe`.

    /// Build a single-op program wrapping a pipe body. `on_ok = Escape`,
    /// `on_failed = Terminate`.
    fn pipe_program(stages: Arc<[ExecAction]>) -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Pipe(
            MultiStage::new(stages).expect("test pipe has >=2 stages"),
        ));
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
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])], None),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])], None),
        ]);
        let program = pipe_program(stages);

        let mut h = Harness::new(nz(4));
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
            &completions[0],
            Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
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
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])], None),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])], None),
        ]);
        let program = pipe_program(stages);

        let mut h = Harness::new(nz(4));
        h.submit(make_effect_perfile_with_program(51, 51, 51, 1, program));
        let spawns = h.wait_for_spawns(2, Duration::from_secs(1));
        let stage0_pid = spawns[0].pid;
        let stage1_pid = spawns[1].pid;

        // Complete stage 0 Failed; the aggregating waiter will
        // observe this and cascade SIGTERM to stage 1.
        h.spawner
            .complete(stage0_pid, EffectOutcome::Failed(Termination::Exit(7)))
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
            .complete(stage1_pid, EffectOutcome::Failed(Termination::Signal(15)))
            .unwrap();
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        match &completions[0] {
            Input::EffectComplete(c) => {
                assert!(matches!(
                    c.outcome,
                    EffectOutcome::Failed(Termination::PipeMixed {
                        last_exit: 7,
                        first_signal: 15,
                    })
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
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])], None),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])], None),
        ]);
        let program = pipe_program(stages);

        let mut h = Harness::new(nz(4));
        h.spawner.inject_spawn_error(std::io::ErrorKind::NotFound);
        h.submit(make_effect_perfile_with_program(52, 52, 52, 1, program));
        let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
        assert!(matches!(
            &completions[0],
            Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Failed(Termination::Internal))
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
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])], None),
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/b")])], None),
        ]);
        let program = {
            let mut b = ProgramBuilder::new();
            let p = b.emit(SpawnBody::Pipe(
                MultiStage::new(pipe_stages).expect("test pipe has >=2 stages"),
            ));
            let after = b.continue_to_next();
            b.patch_on_ok(p, after).unwrap();
            b.patch_on_failed(p, BranchTarget::Terminate).unwrap();
            let exec_after = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/after")])],
                None,
            )));
            b.patch_on_ok(exec_after, BranchTarget::Escape).unwrap();
            b.patch_on_failed(exec_after, BranchTarget::Terminate)
                .unwrap();
            Arc::new(b.build().unwrap())
        };

        // Path 1: pipe Ok → /bin/after runs.
        {
            let mut h = Harness::new(nz(4));
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
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Ok)
            ));
            h.shutdown();
        }

        // Path 2: pipe Failed → /bin/after must NOT run.
        {
            let mut h = Harness::new(nz(4));
            h.submit(make_effect_perfile_with_program(54, 54, 54, 1, program));
            let pipe_spawns = h.wait_for_spawns(2, Duration::from_secs(1));
            h.spawner
                .complete(
                    pipe_spawns[0].pid,
                    EffectOutcome::Failed(Termination::Exit(1)),
                )
                .unwrap();
            h.spawner
                .complete(pipe_spawns[1].pid, EffectOutcome::Ok)
                .unwrap();
            let completions = h.wait_for_effect_completes(1, Duration::from_secs(1));
            assert!(matches!(
                &completions[0],
                Input::EffectComplete(c) if matches!(c.outcome, EffectOutcome::Failed(_))
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
            ExecAction::new([ArgTemplate::new([ArgPart::literal("/bin/a")])], None),
            ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/b")])],
                Some(timeout),
            ),
        ]);
        let program = pipe_program(stages);

        // Short shutdown_grace so the SIGKILL escalation also lands
        // inside the test window if SIGTERM doesn't take effect.
        let mut h = Harness::new_with_grace(nz(4), Duration::from_millis(20));
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
            .complete(stage1_pid, EffectOutcome::Failed(Termination::Signal(15)))
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
