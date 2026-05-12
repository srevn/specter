//! Actuator state machine: slot map, ready queue, per-Sub running counter,
//! global semaphore.
//!
//! All mutations happen on the controller thread. The wait threads send
//! [`Reaped`] events through `reap_tx`; the controller pulls them off
//! `reap_rx` (also held inside [`super::SubprocessActuator`]) and routes
//! to [`ActuatorState::handle_reap`].
//!
//! `ready_queue` orders slots that want to spawn — submit-FIFO. The
//! `in_ready_queue` flag dedups: a key already queued (e.g., a slot
//! whose pending was just replaced) doesn't get pushed twice.
//!
//! # Programs, cursors, and accounting
//!
//! An [`Effect`] carries an [`specter_core::ActionProgram`]: a flat
//! `Box<[ProgramOp]>` walked by a `u32` cursor. Each op carries a
//! [`SpawnBody`] (single Exec or N-stage Pipe) plus explicit `on_ok` /
//! `on_failed` branch targets — dispatch after a reap is a single
//! [`ProgramOp::target`] lookup on the outcome. The actuator walks the
//! program with stop-on-failure semantics encoded by the lowering pass
//! (Exec/Pipe `on_failed = Terminate`; predicate `on_failed` ≠
//! Terminate so the predicate outcome doesn't propagate).
//!
//! - **Per-Effect-stable** state (per-Sub counter bump, diff tmp file)
//!   is owned by [`ActuatorState::start_plan`]: bump on plan start,
//!   release on plan terminus.
//! - **Per-op** state (permit, OS process, wait thread) is owned by
//!   [`ActuatorState::spawn_step_with_permit`]: each op acquires a
//!   fresh permit, the wait thread releases it on reap.
//! - **One [`Input::EffectComplete`] per Effect**: emitted exactly once
//!   at plan terminus (any [`BranchTarget::Terminate`] or
//!   [`BranchTarget::Escape`], or any reap under shutdown's `Drop`
//!   policy). The engine's `outstanding` accounting is unchanged under
//!   multi-op programs — the engine doesn't know programs have multiple
//!   ops.
//!
//! Between two adjacent ops the slot may be in an intermediate state
//! ([`Slot::plan_continue`]) when the wait-thread has reaped op N but no
//! permit is available for op N+1. The pump's plan-continue arm has
//! priority over fresh `pending`: continuation work bypasses the per-Sub
//! gate (it's the same program, already admitted) but still respects the
//! global permit cap.

use crate::env::EnvSnapshot;
use crate::permits::{Permit, Permits};
use crate::resolve;
use crate::spawner::{ChildSignaler, ChildWaiter, EnvVar, Spawner, StageSpec};
use crate::timer;
use crossbeam::channel::Sender;
use specter_core::program::{BranchTarget, ExecAction, SpawnBody};
use specter_core::{CommandResolved, CorrelationId, DedupKey, Effect, EffectOutcome, Input, SubId};
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

/// Policy for [`ActuatorState::handle_reap_inner`]: during normal
/// operation we re-queue pending and let the pump dispatch the next
/// spawn; during shutdown we drop pending and clean up the slot.
///
/// `Pump` also gates the mid-plan advance branch: under shutdown
/// (`Drop`) the reap of step N reports the reaped outcome via the
/// terminal arm and skips spawning step N+1, so partial plans drain
/// cleanly without leaking "step N+1 must run" intent past shutdown.
#[derive(Copy, Clone)]
enum ReapPolicy {
    Pump,
    Drop,
}

/// Outcome of an attempted instruction spawn. Returned by
/// [`ActuatorState::try_spawn_step`] (which acquires a permit).
///
/// The `Failed` variant carries a typed [`SpawnFailureCause`]
/// discriminant: the synth-Failed dispatch sites log it alongside the
/// synthesised `EffectOutcome::Failed { exit_code: None, signal: None }`
/// so an operator triaging "this predicate took the else-branch
/// unexpectedly" can match against the cause-side `error!` log line
/// (resolver, OS spawn, wait-thread) and tell "predicate binary
/// missing" from "predicate exited 1 cleanly".
///
/// The cause is **internal-only**: the engine never sees this type. The
/// wire format remains `EffectOutcome::Failed { exit_code: None,
/// signal: None }` regardless of cause. Splitting cause from outcome
/// here is telemetry-only — it lets the synth-Failed log carry a
/// discriminant without changing engine-side dispatch.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SpawnError {
    /// Permit semaphore at capacity. The caller defers the instruction
    /// into [`Slot::plan_continue`] and re-queues the slot.
    Deferred,
    /// Spawn (or pre-spawn) failure with a typed cause. The caller
    /// routes through [`ActuatorState::advance_or_terminate`] with a
    /// synthesised `EffectOutcome::Failed`; the dispatch then decides
    /// terminate vs continue based on the op's `on_failed` edge at
    /// the failing cursor — predicate spawn-failures still get their
    /// no-propagation semantics through that dispatch.
    Failed(SpawnFailureCause),
}

/// Why a spawn attempt failed. Surfaces at three synthesis sites
/// ([`ActuatorState::start_plan`], [`ActuatorState::spawn_continuation`],
/// [`ActuatorState::advance_or_terminate`]); each site emits a
/// `tracing::warn!` carrying this discriminant so the synthesised
/// `EffectOutcome::Failed { exit_code: None, signal: None }` can be
/// correlated against the cause-side `error!` log line.
///
/// **Not part of the engine wire format.** `EffectOutcome::Failed`
/// carries no cause discriminant; the engine's dispatch reads only the
/// op's `on_failed` edge. Predicate spawn-failure and predicate
/// non-zero-exit are observationally identical to the engine, by design
/// (the op's edge decides routing without inspecting cause).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SpawnFailureCause {
    /// Argv / env substitution failed before any child or wait thread
    /// was spawned. Today the only resolver error is
    /// [`crate::resolve::ResolveError::UnsetEnvVar`] — a strict
    /// `${env.<NAME>}` reference against an unset key with no `:-`
    /// default. Future resolver-time errors (e.g., path canonicalisation)
    /// would land here.
    Resolver,
    /// OS-level process spawn failed —
    /// [`crate::spawner::Spawner::spawn`] / `spawn_pipe` returned an
    /// error (ENOENT on the binary, EAGAIN, EMFILE, …). For pipes the
    /// spawner has already rolled back any partially-spawned stages
    /// before the error reaches this discriminant.
    OsSpawn,
    /// `thread::Builder::spawn` for the wait thread failed. The
    /// spawned child is alive but its paired
    /// [`crate::spawner::ChildWaiter`] was dropped; the recovery branch
    /// SIGKILLs and synchronously reaps the orphan before this
    /// discriminant surfaces.
    WaitThread,
}

/// Parameter bundle for arming a per-step timer thread. Bundled as an
/// `Option<TimerSpec>` at the install site so the arm-or-skip intent
/// is declarative: `Some(spec) ⇒ arm; None ⇒ skip`. Replaces a trio
/// of loose locals (`timeout`, `timer_grace`, `timer_signaler`) that
/// the previous shape carried in parallel — easy to typo-mismatch.
struct TimerSpec {
    deadline: Duration,
    grace: Duration,
    signaler: Arc<dyn ChildSignaler>,
}

/// Per-`DedupKey` actuator slot.
///
/// At most one in-flight child ([`running`]) plus a single
/// Latest-coalesce next-plan slot ([`pending`]) plus, between adjacent
/// instructions of an in-flight plan when the global permit cap is
/// exhausted, a [`plan_continue`] hand-off.
///
/// **Three slots, three roles:**
///
/// - [`running`] is the currently-spawned instruction's bookkeeping
///   (pid, signaler for shutdown SIGTERM/SIGKILL, plus the per-plan
///   snapshot needed to advance to the next instruction).
/// - [`plan_continue`] is "this plan's next instruction, deferred on
///   permit." Bypasses the per-Sub gate (same program, already admitted
///   by `start_plan`) but respects the global permit cap.
/// - [`pending`] is the user's next intent. Latest-coalesced on submit;
///   never replaces a running instruction or a `plan_continue`.
///
/// **Plan-atomicity invariant.** A new submit during a running plan
/// replaces `pending` only; `plan_continue` is never touched by
/// coalesce. Once started, a plan runs all its instructions before
/// `pending` fires.
///
/// **Engine-side twin.** Every `Effect` the actuator runs corresponds
/// to a `+1` on the engine's `PostFirePhase::Awaiting { outstanding }`
/// counter for the owning Profile. The slot retires the plan
/// (or drops the pending Effect on shutdown) and emits exactly one
/// `Input::EffectComplete` per Effect — multi-instruction programs
/// don't change the engine's accounting.
#[derive(Debug, Default)]
pub(crate) struct Slot {
    pub running: Option<RunningJob>,
    pub plan_continue: Option<PlanContinuation>,
    pub pending: Option<Effect>,
    pub in_ready_queue: bool,
}

/// Bookkeeping for one in-flight op of a plan.
///
/// With the CFG-shaped IR, outcome routing (propagate / branch / no-op)
/// lives on the op's edges ([`ProgramOp::on_ok`] / [`ProgramOp::on_failed`]),
/// not in the running job's variant tag. The reap-path reads the
/// edge directly via [`ProgramOp::target`], so there's nothing here
/// that depends on which spawn shape produced the running child.
///
/// Carries:
///
/// - **`pid`** — the operator-facing pid. For [`SpawnBody::Exec`],
///   the child's pid; for [`SpawnBody::Pipe`], the *last* stage's pid
///   (what `ps` would label "the pipe"). Intermediate-stage pids stay
///   inside the per-stage signalers (used only for the per-stage
///   timer threads at install time, then dropped).
/// - **`signaler`** — the signaler the controller uses for shutdown
///   SIGTERM / SIGKILL. For Exec this is the single-child signaler;
///   for Pipe this is the combined fan-out signaler that signals
///   every stage. Per-stage signalers DO NOT live here: pipe install
///   collects them as locals, arms per-stage timer threads against
///   each (cloning the Arc), then drops the locals when install
///   returns. The aggregating `PipeWaiter` owns its own per-stage
///   signaler clones for the SIGTERM-cascade-on-first-failure path,
///   independent of this combined signaler.
/// - **`effect`** — the plan's shared `Arc<Effect>`. The advance branch
///   in [`ActuatorState::handle_reap_inner`] re-resolves op N+1's argv
///   + env from the same snapshot without re-fetching.
/// - **`cursor`** — `u32` index into `effect.program.ops`.
/// - **`diff_tmp_path`** — `Some` iff `start_plan` materialised a diff
///   tmp file. Shared across all ops so every step reads the same
///   `SPECTER_DIFF_PATH`; cleaned at plan terminus in
///   [`ActuatorState::terminate_plan`].
///
/// `signaler` is `Arc<dyn>` so the controller's installed-side
/// reference and the per-step timer thread's clone are independent
/// co-owners; either may outlive the other.
pub(crate) struct RunningJob {
    pub pid: u32,
    pub signaler: Arc<dyn ChildSignaler>,
    pub effect: Arc<Effect>,
    pub cursor: u32,
    pub diff_tmp_path: Option<PathBuf>,
}

impl std::fmt::Debug for RunningJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningJob")
            .field("pid", &self.pid)
            .field("cursor", &self.cursor)
            .field("sub", &self.effect.key.sub())
            .field("correlation", &self.effect.correlation)
            .finish_non_exhaustive()
    }
}

/// Hand-off slot between two adjacent instructions when no permit is
/// available at advance time. The pump's plan-continue arm consumes
/// this in priority over [`Slot::pending`] — same program, already
/// admitted, just waiting on the global cap.
#[derive(Debug)]
pub(crate) struct PlanContinuation {
    pub effect: Arc<Effect>,
    pub cursor: u32,
    pub diff_tmp_path: Option<PathBuf>,
}

#[derive(Debug)]
pub(crate) struct ActuatorState {
    pub slots: BTreeMap<DedupKey, Slot>,
    pub ready_queue: VecDeque<DedupKey>,
    pub running_per_sub: BTreeMap<SubId, u32>,
    pub permits: Permits,
    /// Captured operator env, threaded into every resolver call for
    /// `${env.<NAME>}` substitution. Shared by `Arc` because the
    /// snapshot is immutable for the actuator's lifetime; the rare
    /// test override case constructs a fresh snapshot rather than
    /// mutating the existing one.
    pub env_snapshot: Arc<EnvSnapshot>,
    /// SIGTERM → SIGKILL grace. Reads:
    /// - shutdown drain ([`super::SubprocessActuator::shutdown`]);
    /// - per-step timer thread grace ([`crate::timer::spawn_timer`]).
    ///
    /// Pinned in one place so the two paths can't drift on the
    /// constant.
    pub shutdown_grace: Duration,
    /// Scratch deque reused across [`Self::pump`] calls to hold keys
    /// blocked this round on permit / per-Sub gate unavailability.
    /// Restored to the ready queue at the end of `pump`. Living on the
    /// state (rather than allocated fresh inside `pump`) amortises the
    /// `VecDeque::new()` heap allocation across high-frequency
    /// same-Sub submit bursts. Empty between pump calls; the
    /// `debug_assert!` at pump entry pins the invariant.
    pub blocked_scratch: VecDeque<DedupKey>,
}

impl ActuatorState {
    pub fn new(
        concurrency: NonZeroUsize,
        env_snapshot: Arc<EnvSnapshot>,
        shutdown_grace: Duration,
    ) -> Self {
        Self {
            slots: BTreeMap::new(),
            ready_queue: VecDeque::new(),
            running_per_sub: BTreeMap::new(),
            permits: Permits::new(concurrency),
            env_snapshot,
            shutdown_grace,
            blocked_scratch: VecDeque::new(),
        }
    }

    /// Submit handler — enqueue or coalesce. Always end with `pump`.
    ///
    /// Plan-atomicity: a fresh submit during a running plan replaces
    /// `pending` only. Both `running` and `plan_continue` (an in-flight
    /// plan deferred between steps) keep the slot in "plan in flight"
    /// state from the coalesce point of view.
    pub fn handle_submit(
        &mut self,
        effect: Effect,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
        let key = effect.key.clone();
        tracing::trace!(?key, "submit");
        let slot = self.slots.entry(key.clone()).or_default();
        if slot.running.is_some() || slot.plan_continue.is_some() {
            // Plan in flight; Latest-coalesce — drop old pending if
            // present. Never touches `running` or `plan_continue`: the
            // current plan runs to terminus before pending fires.
            slot.pending = Some(effect);
        } else {
            slot.pending = Some(effect);
            if !slot.in_ready_queue {
                slot.in_ready_queue = true;
                self.ready_queue.push_back(key);
            }
        }
        self.pump(spawner, reap_tx, engine_in);
    }

    /// Reap handler — advance to next step or terminate the plan.
    /// Followed by `pump` to dispatch any newly-ready work.
    pub fn handle_reap(
        &mut self,
        reaped: super::Reaped,
        engine_in: &Sender<Input>,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) {
        self.handle_reap_inner(reaped, engine_in, spawner, reap_tx, ReapPolicy::Pump);
        self.pump(spawner, reap_tx, engine_in);
    }

    /// Shutdown-phase reap handler. Forces the plan to terminus on
    /// the reaped step's outcome — no advance, no pending re-queue.
    /// Subsequent steps are abandoned.
    ///
    /// `spawner` and `reap_tx` are unused under `Drop` policy (the
    /// advance branch is gated on `Pump`); they're threaded through
    /// for signature-symmetry with [`Self::handle_reap`].
    pub fn handle_reap_no_pump(
        &mut self,
        reaped: super::Reaped,
        engine_in: &Sender<Input>,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) {
        self.handle_reap_inner(reaped, engine_in, spawner, reap_tx, ReapPolicy::Drop);
    }

    /// The reap pipeline. Two main exits:
    ///
    /// 1. **Advance**: the op's [`ProgramOp::target`] for the reaped
    ///    outcome is [`BranchTarget::Continue`], so the plan continues
    ///    at the named slot. Handed to [`Self::try_spawn_step`]; a
    ///    `SpawnError::Failed` here loops the dispatch with a
    ///    synthesised `Failed` outcome for the new cursor — a predicate
    ///    spawn-failure cascade naturally walks to its own
    ///    [`BranchTarget::Continue`] (the else-branch's first op), an
    ///    exec spawn-failure walks to its [`BranchTarget::Terminate`]
    ///    and the plan terminates with the synth Failed.
    /// 2. **Terminate**: the op's edge target is [`BranchTarget::Terminate`]
    ///    (carried outcome propagates) or [`BranchTarget::Escape`]
    ///    (terminate Ok regardless of carried outcome — the "branch,
    ///    not guard" outcome elision), or any reap under shutdown's
    ///    `Drop` policy. `terminate_plan` emits one `EffectComplete`,
    ///    decrements the per-Sub counter, cleans the diff tmp file,
    ///    and either re-queues the slot's `pending` (Pump policy) or
    ///    removes the slot (Drop policy).
    ///
    /// **Defensive no-job**: a stale Reaped after slot removal falls
    /// through directly to `terminate_plan` without a job — preserves
    /// the "always emit EffectComplete" invariant for the engine.
    fn handle_reap_inner(
        &mut self,
        reaped: super::Reaped,
        engine_in: &Sender<Input>,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        policy: ReapPolicy,
    ) {
        tracing::trace!(?reaped.key, ?reaped.outcome, "reap");
        let super::Reaped {
            key, sub, outcome, ..
        } = reaped;

        // Take the running job from its slot. A missing slot or absent
        // running is the "stale Reaped" defensive case.
        let job = match self.slots.get_mut(&key) {
            Some(slot) => slot.running.take(),
            None => None,
        };
        let Some(job) = job else {
            self.terminate_plan(key, sub, None, outcome, policy, engine_in);
            return;
        };

        let RunningJob {
            effect,
            cursor,
            diff_tmp_path,
            ..
        } = job;

        // Drop policy (shutdown): no advance, no dispatch. Pass the
        // reaped outcome straight through to terminate. Branch-elision
        // (Escape) is inert under shutdown — the engine is tearing
        // down and only counts EffectCompletes.
        if matches!(policy, ReapPolicy::Drop) {
            self.terminate_plan(
                key,
                sub,
                diff_tmp_path.as_deref(),
                outcome,
                policy,
                engine_in,
            );
            return;
        }

        self.advance_or_terminate(
            key,
            sub,
            effect,
            diff_tmp_path,
            cursor,
            outcome,
            spawner,
            reap_tx,
            engine_in,
        );
    }

    /// Drive the post-reap / post-spawn-failure dispatch loop.
    ///
    /// `cursor` and `outcome` define "where we are" and "what just
    /// happened." The op's edge ([`ProgramOp::target`] on the outcome)
    /// decides:
    ///
    /// - [`BranchTarget::Terminate`] → propagate `outcome` to
    ///   `EffectComplete` and return.
    /// - [`BranchTarget::Escape`] → terminate with
    ///   [`EffectOutcome::Ok`] regardless of the carried outcome (the
    ///   "branch, not guard" outcome elision pinned by lowering).
    /// - [`BranchTarget::Continue`] → attempt to spawn the named op:
    ///   - **Ok**: the wait thread now drives the next reap; return.
    ///   - **Deferred** (permit cap): park in
    ///     [`Slot::plan_continue`] and return.
    ///   - **Failed** (OS spawn / resolver / wait-thread failure): loop
    ///     with a synthesised `Failed` outcome at the new cursor.
    ///
    /// The loop is bounded: each [`BranchTarget::Continue`] edge points
    /// forward (builder invariant: `target > origin`) and within bounds
    /// (`target < ops.len()`), so the cursor strictly increases. A
    /// pathological program is impossible by construction.
    ///
    /// Called only under [`ReapPolicy::Pump`]. The Drop arm in
    /// [`Self::handle_reap_inner`] bypasses dispatch entirely.
    fn advance_or_terminate(
        &mut self,
        key: DedupKey,
        sub: SubId,
        effect: Arc<Effect>,
        diff_tmp_path: Option<PathBuf>,
        mut cursor: u32,
        mut outcome: EffectOutcome,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
        loop {
            let op = &effect.program.ops()[cursor as usize];
            match op.target(&outcome) {
                BranchTarget::Terminate => {
                    self.terminate_plan(
                        key,
                        sub,
                        diff_tmp_path.as_deref(),
                        outcome,
                        ReapPolicy::Pump,
                        engine_in,
                    );
                    return;
                }
                BranchTarget::Escape => {
                    self.terminate_plan(
                        key,
                        sub,
                        diff_tmp_path.as_deref(),
                        EffectOutcome::Ok,
                        ReapPolicy::Pump,
                        engine_in,
                    );
                    return;
                }
                BranchTarget::Continue(next_idx) => {
                    let next = next_idx.get();
                    // Forward-only-and-in-bounds is structurally
                    // enforced at builder patch time. Defensive assert
                    // here as a tripwire if a future variant addition
                    // bypasses the builder's edge validation.
                    debug_assert!(
                        next > cursor && (next as usize) < effect.program.ops().len(),
                        "forward-only + in-bounds (builder invariant)",
                    );
                    match self.try_spawn_step(
                        &key,
                        sub,
                        &effect,
                        next,
                        diff_tmp_path.as_deref(),
                        spawner,
                        reap_tx,
                    ) {
                        Ok(()) => return,
                        Err(SpawnError::Deferred) => {
                            self.queue_plan_continue(
                                key,
                                PlanContinuation {
                                    effect,
                                    cursor: next,
                                    diff_tmp_path,
                                },
                            );
                            return;
                        }
                        Err(SpawnError::Failed(cause)) => {
                            // Synthesise Failed for `next` and loop.
                            // The next iteration reads `next`'s
                            // `on_failed` edge — for a predicate-Failed
                            // synth this walks to the else-branch
                            // (Continue) or to Escape (no-else); for an
                            // Exec/Pipe synth it walks to Terminate
                            // (stop-on-failure propagation).
                            //
                            // Log at warn with the typed cause so the
                            // operator can correlate this dispatch
                            // decision against the cause-side error log
                            // line emitted at the spawn boundary
                            // (resolver / OS spawn / wait thread).
                            tracing::warn!(
                                ?key,
                                cursor = next,
                                ?cause,
                                "synthesised EffectOutcome::Failed (no clean exit); dispatching on op's on_failed edge",
                            );
                            cursor = next;
                            outcome = EffectOutcome::Failed {
                                exit_code: None,
                                signal: None,
                            };
                        }
                    }
                }
            }
        }
    }

    /// Park a plan's next instruction into [`Slot::plan_continue`] and
    /// queue the slot for the next pump cycle. Called from the advance
    /// branch when no permit was available at reap time.
    fn queue_plan_continue(&mut self, key: DedupKey, cont: PlanContinuation) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.plan_continue = Some(cont);
            if !slot.in_ready_queue {
                slot.in_ready_queue = true;
                self.ready_queue.push_back(key);
            }
        }
    }

    /// Terminal arm of a plan: emit one `EffectComplete`, decrement
    /// the per-Sub counter, clean the diff tmp file, and either re-queue
    /// pending (Pump policy + non-empty pending) or remove the slot.
    ///
    /// Called from three sites:
    ///
    /// 1. [`Self::handle_reap_inner`] when no advance is possible.
    /// 2. [`Self::start_plan`] / [`Self::spawn_continuation`] on spawn
    ///    failure — slot.running was never installed (or was rolled
    ///    back), so the take-then-terminate dance handle_reap_inner
    ///    does isn't needed; the caller hands us its locally-scoped
    ///    `diff_tmp_path` directly.
    /// 3. [`Self::handle_reap_inner`] advance branch when
    ///    [`Self::try_spawn_step`] returned `SpawnError::Failed`.
    ///
    /// `diff_tmp_path` is taken by `Option<&Path>` because all callers
    /// retain ownership; cleanup just borrows for the unlink syscall.
    /// The per-Sub counter decrement is `saturating_sub` against an
    /// `Option<&mut u32>` so spawn-failure-before-counter-bump paths
    /// (fresh plan whose step 0 spawn failed) are no-ops as desired.
    fn terminate_plan(
        &mut self,
        key: DedupKey,
        sub: SubId,
        diff_tmp_path: Option<&Path>,
        outcome: EffectOutcome,
        policy: ReapPolicy,
        engine_in: &Sender<Input>,
    ) {
        let _ = engine_in.send(Input::EffectComplete {
            sub,
            key: key.clone(),
            result: outcome,
        });
        if let Some(c) = self.running_per_sub.get_mut(&sub) {
            // The counter bump at `start_plan` precedes every spawn
            // attempt for this Sub, including spawn-failure paths that
            // route through `advance_or_terminate` → `terminate_plan`.
            // A decrement without a prior bump would be a controller
            // accounting bug; debug_assert tripwires it in tests.
            debug_assert!(*c > 0, "running_per_sub decrement without prior bump");
            *c -= 1;
            if *c == 0 {
                self.running_per_sub.remove(&sub);
            }
        }
        if let Some(p) = diff_tmp_path {
            crate::tmp::cleanup(p);
        }
        // The slot may still exist (handle_reap_inner already took
        // running; spawn-failure paths never installed it). Decide:
        // re-queue if pending under Pump, otherwise remove.
        let Some(slot) = self.slots.get_mut(&key) else {
            return;
        };
        match policy {
            ReapPolicy::Pump if slot.pending.is_some() => {
                if !slot.in_ready_queue {
                    slot.in_ready_queue = true;
                    self.ready_queue.push_back(key);
                }
            }
            _ => {
                self.slots.remove(&key);
            }
        }
    }

    /// Spawn ready slots while permits + per-Sub gates allow.
    ///
    /// Two arms per slot:
    ///
    /// - **Plan-continue** (`slot.plan_continue.is_some()`): the slot
    ///   holds an in-flight plan's next step, deferred at reap time on
    ///   permit unavailability. Bypasses the per-Sub gate (continuation
    ///   of an admitted plan; never racing another plan for the Sub by
    ///   construction). Permit gate still applies.
    /// - **Fresh plan** (`slot.pending.is_some()`, `plan_continue` empty):
    ///   per-Sub gate, then permit gate, then [`Self::start_plan`].
    ///
    /// Items blocked by either gate are deferred to a transient buffer
    /// and restored at end so FIFO is preserved across pump invocations.
    /// The blocked-buffer logic is per-arm: a permit-blocked plan-continue
    /// short-circuits the loop the same way a permit-blocked fresh plan
    /// does, since both contend for the same global semaphore.
    pub fn pump(
        &mut self,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
        debug_assert!(
            self.blocked_scratch.is_empty(),
            "blocked_scratch must be empty at pump entry; previous call must have drained it",
        );
        while let Some(key) = self.ready_queue.pop_front() {
            let sub = sub_of_key(&key);
            let Some(slot) = self.slots.get_mut(&key) else {
                continue;
            };

            // Plan-continue: bypass per-Sub gate.
            if slot.plan_continue.is_some() {
                let Some(permit) = self.permits.try_acquire() else {
                    self.blocked_scratch.push_back(key);
                    while let Some(k) = self.ready_queue.pop_front() {
                        self.blocked_scratch.push_back(k);
                    }
                    break;
                };
                slot.in_ready_queue = false;
                let cont = slot
                    .plan_continue
                    .take()
                    .expect("plan_continue checked Some directly above");
                self.spawn_continuation(key, sub, cont, permit, spawner, reap_tx, engine_in);
                continue;
            }

            // Fresh plan: per-Sub gate.
            if self.running_per_sub.get(&sub).copied().unwrap_or(0) > 0 {
                self.blocked_scratch.push_back(key);
                continue;
            }
            // Global gate.
            let Some(permit) = self.permits.try_acquire() else {
                // No more permits this round; defer this and the
                // remaining queued items (FIFO preserved).
                self.blocked_scratch.push_back(key);
                while let Some(k) = self.ready_queue.pop_front() {
                    self.blocked_scratch.push_back(k);
                }
                break;
            };
            slot.in_ready_queue = false;
            let Some(effect) = slot.pending.take() else {
                drop(permit);
                continue;
            };
            self.start_plan(
                key,
                sub,
                Arc::new(effect),
                permit,
                spawner,
                reap_tx,
                engine_in,
            );
        }
        // Drain (don't consume) so the deque retains its capacity for the
        // next pump. The flag is already true (we set it when we pushed
        // and only cleared it on successful spawn). Defensive: ensure it.
        while let Some(k) = self.blocked_scratch.pop_front() {
            if let Some(slot) = self.slots.get_mut(&k) {
                slot.in_ready_queue = true;
            }
            self.ready_queue.push_back(k);
        }
    }

    /// Start a plan: materialise the diff tmp file (if needed), bump
    /// the per-Sub counter, spawn instruction 0 with the given permit.
    ///
    /// **Per-Sub counter is bumped unconditionally** before the spawn
    /// attempt — predicate spawn-failure semantics may continue the
    /// plan via [`Self::advance_or_terminate`], and any in-progress
    /// continuation needs the per-Sub gate to hold same-Sub fresh
    /// plans behind it. On failure, the dispatch loop's terminate
    /// arms decrement normally; the controller is single-threaded so
    /// the bump-then-decrement is atomic from any observer's
    /// perspective.
    ///
    /// On spawn failure, routes through [`Self::advance_or_terminate`]
    /// with a synthesised `EffectOutcome::Failed`. The dispatcher reads
    /// op 0's `on_failed` edge — propagates the synth Failed when the
    /// edge is `Terminate` (Exec/Pipe stop-on-failure), jumps to the
    /// named slot when the edge is `Continue` (a predicate's
    /// else-branch), or terminates Ok when the edge is `Escape` (a
    /// predicate with no else).
    ///
    /// `effect` is taken by value so the caller (pump) hands off the
    /// freshly-constructed `Arc<Effect>` and forgets about it; on
    /// success the Arc is cloned into [`Slot::running`], on failure it
    /// drops or moves into the advance loop. Passing by reference
    /// would force pump to keep the Arc alive past the call for no
    /// reason.
    #[allow(clippy::needless_pass_by_value)]
    fn start_plan(
        &mut self,
        key: DedupKey,
        sub: SubId,
        effect: Arc<Effect>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
        // Materialise the diff tmp file before the first instruction's
        // spawn so the resolver can slot SPECTER_DIFF_PATH into its
        // alphabetical position. Best-effort: on write failure proceed
        // with `None`, the resolver omits the env var. The path lives
        // for the whole plan's lifetime — every instruction shares it;
        // cleaned exactly once at terminate_plan.
        let diff_tmp_path = effect.diff.as_ref().and_then(|diff| {
            let path = crate::tmp::tmp_path(effect.correlation);
            match crate::tmp::write_diff_file(&path, diff) {
                Ok(()) => Some(path),
                Err(e) => {
                    tracing::warn!(
                        ?path,
                        ?e,
                        "tmp diff write failed; proceeding without SPECTER_DIFF_PATH"
                    );
                    None
                }
            }
        });

        // Counter bump symmetric with `terminate_plan`'s decrement;
        // overflow would require billions of concurrent plans per Sub,
        // which is structurally impossible (concurrency cap +
        // per-Sub gate hold at most `permits.cap()` Subs at once).
        let counter = self.running_per_sub.entry(sub).or_insert(0);
        debug_assert!(*counter < u32::MAX, "running_per_sub counter overflow");
        *counter += 1;
        match self.spawn_step_with_permit(
            &key,
            sub,
            &effect,
            0,
            diff_tmp_path.as_deref(),
            permit,
            spawner,
            reap_tx,
        ) {
            Ok(()) => {}
            Err(cause) => {
                // OS spawn / resolver / wait-thread failure at the
                // first instruction. Hand off to the dispatch loop
                // with synthesised Failed — a predicate at cursor 0
                // still jumps to its else-branch via this path. The
                // typed `cause` discriminant accompanies the synth in
                // the operator log so triage can correlate this
                // decision with the cause-side error line above.
                tracing::warn!(
                    ?key,
                    cursor = 0,
                    ?cause,
                    "synthesised EffectOutcome::Failed at plan start; dispatching on op 0's on_failed edge",
                );
                self.advance_or_terminate(
                    key,
                    sub,
                    effect,
                    diff_tmp_path,
                    0,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    },
                    spawner,
                    reap_tx,
                    engine_in,
                );
            }
        }
    }

    /// Spawn the next instruction of a plan that was deferred via
    /// [`Slot::plan_continue`]. Distinct from [`Self::start_plan`]:
    /// no per-Sub counter bump (already bumped at the original
    /// `start_plan`), no tmp materialisation (path inherited from the
    /// `PlanContinuation`).
    ///
    /// On spawn failure, routes through [`Self::advance_or_terminate`]
    /// — predicate spawn-failure at the continuation's cursor jumps
    /// to its else-branch, exec/pipe spawn-failure propagates to
    /// plan terminus. Either way the per-Sub counter (which is at +1
    /// from the original start_plan) decrements at terminate, and
    /// any subsequent advance reuses the inherited tmp path.
    fn spawn_continuation(
        &mut self,
        key: DedupKey,
        sub: SubId,
        cont: PlanContinuation,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
        let PlanContinuation {
            effect,
            cursor,
            diff_tmp_path,
        } = cont;
        match self.spawn_step_with_permit(
            &key,
            sub,
            &effect,
            cursor,
            diff_tmp_path.as_deref(),
            permit,
            spawner,
            reap_tx,
        ) {
            Ok(()) => {}
            Err(cause) => {
                tracing::warn!(
                    ?key,
                    cursor,
                    ?cause,
                    "synthesised EffectOutcome::Failed at plan continuation; dispatching on op's on_failed edge",
                );
                self.advance_or_terminate(
                    key,
                    sub,
                    effect,
                    diff_tmp_path,
                    cursor,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    },
                    spawner,
                    reap_tx,
                    engine_in,
                );
            }
        }
    }

    /// Acquire-and-spawn helper used by the reap-time advance branch.
    ///
    /// Returns:
    /// - `Ok(())` — instruction is in flight (slot.running installed,
    ///   wait thread alive).
    /// - `Err(SpawnError::Deferred)` — permit semaphore was at capacity;
    ///   caller defers via [`Slot::plan_continue`].
    /// - `Err(SpawnError::Failed(cause))` — OS-level spawn, resolver,
    ///   or wait-thread startup failed; caller terminates the plan with
    ///   synthesised `EffectOutcome::Failed` and logs `cause` at the
    ///   synth site.
    ///
    /// The Deferred branch returns before consuming any of the borrowed
    /// inputs — caller-owned values stay live for the
    /// `PlanContinuation` hand-off. `SpawnFailureCause` is lifted into
    /// the wider [`SpawnError::Failed`] variant via `map_err` — the
    /// inner [`Self::spawn_step_with_permit`] cannot defer (its permit
    /// is already acquired), so its return type is the tighter
    /// `Result<(), SpawnFailureCause>`.
    fn try_spawn_step(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        diff_path: Option<&Path>,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) -> Result<(), SpawnError> {
        let Some(permit) = self.permits.try_acquire() else {
            return Err(SpawnError::Deferred);
        };
        self.spawn_step_with_permit(
            key, sub, effect, cursor, diff_path, permit, spawner, reap_tx,
        )
        .map_err(SpawnError::Failed)
    }

    /// Spawn one op of a plan with a pre-acquired permit. Installs
    /// [`Slot::running`] on success.
    ///
    /// Dispatches on the op's [`SpawnBody`] at `cursor`:
    ///
    /// - [`SpawnBody::Exec`] → [`Self::spawn_exec_with_permit`]: one
    ///   resolver call, one [`Spawner::spawn`], one [`RunningJob`]
    ///   installed, one wait thread, one optional timer thread.
    /// - [`SpawnBody::Pipe`] → [`Self::spawn_pipe_with_permit`]: N
    ///   resolver calls, one [`Spawner::spawn_pipe`], one
    ///   [`RunningJob`] (with combined signaler for shutdown fan-out),
    ///   one aggregating wait thread, and per-stage timer threads for
    ///   stages with a `timeout`.
    ///
    /// At the IR level there is no predicate distinction — predicate
    /// behavior is the op's `on_failed` edge, read by the reap-path.
    ///
    /// `now: SystemTime` is sampled at the dispatcher and threaded
    /// into every resolver call so a single pipe sees one shared
    /// `${specter.time}` across all stages — the documented contract
    /// pins "the wall-clock instant immediately before the kernel
    /// runs the user's command," which for a pipe is the instant all
    /// stages start.
    fn spawn_step_with_permit(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        diff_path: Option<&Path>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) -> Result<(), SpawnFailureCause> {
        let now = std::time::SystemTime::now();
        let cwd: &Path = resolve::compute_cwd(&effect.anchor_path, effect.anchor_kind);
        let correlation = effect.correlation;
        let capture_output = effect.capture_output;

        let op = &effect.program.ops()[cursor as usize];
        match &op.body {
            SpawnBody::Exec(exec) => self.spawn_exec_with_permit(
                key,
                sub,
                effect,
                cursor,
                exec,
                now,
                cwd,
                correlation,
                capture_output,
                diff_path,
                permit,
                spawner,
                reap_tx,
            ),
            SpawnBody::Pipe(stages) => {
                // Borrow the stages slice via the Arc held inside the
                // op body. The Arc lifetime is tied to `effect`; the
                // slice survives the resolve/spawn_pipe sequence.
                let stages_slice: &[ExecAction] = stages.as_ref();
                self.spawn_pipe_with_permit(
                    key,
                    sub,
                    effect,
                    cursor,
                    stages_slice,
                    now,
                    cwd,
                    correlation,
                    capture_output,
                    diff_path,
                    permit,
                    spawner,
                    reap_tx,
                )
            }
        }
    }

    /// Single-process spawn path for [`SpawnBody::Exec`]. Outcome
    /// routing (propagate / branch / no-op) lives on the op's edges;
    /// this function is shape-only.
    ///
    /// Sequencing pinned: slot.running is installed **before** the
    /// wait thread is spawned, so a fast-completing wait thread
    /// (mock under test, or a child that exits between fork and
    /// wait) can't send `Reaped` before the controller knows about
    /// it.
    ///
    /// On wait-thread spawn failure: the freshly-spawned child is
    /// alive but has no waiter (the closure that owned it has been
    /// dropped by `Builder::spawn`'s `Err` path). The recovery
    /// branch SIGKILLs the orphan via the signaler held in
    /// `slot.running`, then synchronously reaps it via
    /// [`crate::spawner::ChildSignaler::reap_blocking`] so the OS
    /// doesn't leak a zombie. `slot.running` is then cleared (the
    /// terminate_plan caller expects it to be `None`) and
    /// `SpawnError::Failed` returns.
    ///
    /// **Slot invariant.** All `self.slots.get_mut(key)` lookups in
    /// this function assume the slot was just touched by the caller
    /// (the controller is single-threaded; no Reap or Submit can
    /// interleave between caller's `pump` / `handle_reap_inner` and
    /// here). A missing slot is a programming error, surfaced via
    /// `expect` rather than silently masked — silent masking would
    /// otherwise leak the signaler and leave the child unreachable
    /// from shutdown signaling.
    #[allow(clippy::too_many_arguments)]
    fn spawn_exec_with_permit(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        exec: &ExecAction,
        now: std::time::SystemTime,
        cwd: &Path,
        correlation: CorrelationId,
        capture_output: bool,
        diff_path: Option<&Path>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) -> Result<(), SpawnFailureCause> {
        let (CommandResolved { argv }, env) =
            match resolve::resolve_step(effect, exec, now, diff_path, &self.env_snapshot) {
                Ok(resolved) => resolved,
                Err(e) => {
                    // Strict `${env.<NAME>}` failure: no spawn, no
                    // wait thread, no timer. Permit drops at the end
                    // of this scope; caller routes through
                    // `advance_or_terminate` with synthesised
                    // `EffectOutcome::Failed`.
                    tracing::error!(?key, cursor, %e, "resolver error; aborting step");
                    drop(permit);
                    return Err(SpawnFailureCause::Resolver);
                }
            };

        let handles = match spawner.spawn(&argv, &env, cwd, capture_output) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(?key, cursor, ?cwd, ?e, "spawn failed");
                drop(permit);
                return Err(SpawnFailureCause::OsSpawn);
            }
        };
        let crate::spawner::SpawnHandles {
            pid,
            waiter,
            signaler,
        } = handles;

        // The signaler Arc is also handed to the optional per-step
        // timer thread below; cloning before the move into RunningJob
        // keeps the controller's installed-side reference live
        // regardless of whether the timer is armed.
        let timer_spec: Option<TimerSpec> = exec.timeout.map(|deadline| TimerSpec {
            deadline,
            grace: self.shutdown_grace,
            signaler: Arc::clone(&signaler),
        });
        let diff_tmp_path: Option<PathBuf> = diff_path.map(Path::to_path_buf);
        let slot = self
            .slots
            .get_mut(key)
            .expect("slot present at install (single-threaded controller just dispatched here)");
        slot.running = Some(RunningJob {
            pid,
            signaler,
            effect: Arc::clone(effect),
            cursor,
            diff_tmp_path,
        });

        self.spawn_wait_thread_after_install(
            key,
            sub,
            correlation,
            pid,
            cursor,
            waiter,
            permit,
            reap_tx,
        )?;

        // Per-step timer: spawn AFTER the wait thread is alive so the
        // wait thread's `dead` flag is the natural-completion signal
        // the timer short-circuits on. Best-effort — see
        // [`crate::timer`] module docs for the spawn-failure policy.
        if let Some(spec) = timer_spec {
            // Thread name budget: Linux pthread_setname_np truncates
            // to 15 chars + null, so we shape the name around what
            // `ps -T` / `gdb info threads` can render without
            // truncation. `c{cursor}-pid{pid}` fits a 9-char pid
            // unscathed. The sub identifier is intentionally omitted
            // — adding it would push the name past the Linux ceiling.
            // Sub/key cross-reference is via `tracing` logs keyed on
            // the same `pid` (the `tracing::debug!` line below emits
            // ?key + pid; live system inspection follows pid → log
            // for the sub identity).
            let timer_name = format!("c{cursor}-pid{pid}");
            if let Err(e) =
                timer::spawn_timer(&timer_name, spec.deadline, spec.grace, spec.signaler)
            {
                tracing::error!(
                    ?key,
                    cursor,
                    pid,
                    timeout = ?exec.timeout,
                    ?e,
                    "per-step timer thread spawn failed; deadline not enforced",
                );
            }
        }

        tracing::debug!(?key, cursor, pid, "spawned instruction");
        Ok(())
    }

    /// Multi-stage spawn path for [`SpawnBody::Pipe`].
    ///
    /// The shape mirrors [`Self::spawn_exec_with_permit`] at every
    /// step, scaled to N stages:
    ///
    /// 1. Resolve every stage's argv + env against the shared `now`
    ///    (so `${specter.time}` agrees across stages — see
    ///    [`Spawner::spawn_pipe`] for the contract).
    /// 2. Call [`Spawner::spawn_pipe`] which mints N processes, an
    ///    aggregating [`crate::spawner::ChildWaiter`], a combined
    ///    [`crate::spawner::ChildSignaler`] for shutdown fan-out,
    ///    and per-stage signalers for per-stage timer threads.
    /// 3. Install [`RunningJob`] BEFORE spawning the wait thread
    ///    (slot.running invariant: the wait thread must not be able
    ///    to send `Reaped` before the controller has the job in
    ///    hand). The job carries the combined signaler only — the
    ///    per-stage signalers are locals to this function, cloned
    ///    into any per-stage timer thread and dropped on return.
    /// 4. Spawn one wait thread that drains the aggregating waiter
    ///    and surfaces a single `Reaped` event to the controller —
    ///    the engine's accounting is "one EffectComplete per Effect"
    ///    and that holds regardless of pipe vs single-process.
    /// 5. For each stage with a `timeout`, spawn one detached timer
    ///    thread that observes the stage's `dead` flag and signals
    ///    the stage individually. The aggregating waiter sees the
    ///    resulting per-stage Failed and cascades SIGTERM to alive
    ///    siblings; the engine receives one aggregated Failed
    ///    outcome.
    ///
    /// Resolver failure on **any** stage aborts the entire pipe (no
    /// processes have spawned yet at that point). Pipe-spawn failure
    /// (returned by [`Spawner::spawn_pipe`]) means the spawner has
    /// already rolled back any partially-spawned stages — the caller
    /// just returns `SpawnError::Failed`.
    #[allow(clippy::too_many_arguments)]
    fn spawn_pipe_with_permit(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        stages: &[ExecAction],
        now: std::time::SystemTime,
        cwd: &Path,
        correlation: CorrelationId,
        capture_output: bool,
        diff_path: Option<&Path>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) -> Result<(), SpawnFailureCause> {
        debug_assert!(
            stages.len() >= 2,
            "validation rejects empty / single-stage pipes",
        );

        // Resolve every stage's argv + env. The result tuples own
        // the argv `Vec<String>` and the env `Vec<EnvVar<'_>>`; the
        // env's `Cow::Borrowed` slots borrow from `effect`, the
        // resolver's owned per-stage `parent_str` / `time_str`
        // (moved into the env Cow::Owned slots), and `diff_path` (if
        // present). All borrowed sources outlive this function's
        // body, so the resolved Vec is stable across the
        // `spawn_pipe` call.
        let resolved: Vec<(CommandResolved, Vec<EnvVar<'_>>)> = match stages
            .iter()
            .map(|stage| resolve::resolve_step(effect, stage, now, diff_path, &self.env_snapshot))
            .collect::<Result<Vec<_>, _>>()
        {
            Ok(r) => r,
            Err(e) => {
                tracing::error!(?key, cursor, %e, "resolver error in pipe stage; aborting step");
                drop(permit);
                return Err(SpawnFailureCause::Resolver);
            }
        };
        let stage_specs: Vec<StageSpec<'_>> = resolved
            .iter()
            .map(|(cmd, env)| StageSpec {
                argv: cmd.argv.as_slice(),
                env: env.as_slice(),
            })
            .collect();

        let handles = match spawner.spawn_pipe(&stage_specs, cwd, capture_output) {
            Ok(h) => h,
            Err(e) => {
                // Partial-spawn rollback already happened inside
                // `spawn_pipe` (every prior stage SIGKILLed + reaped,
                // every pipe fd closed in the parent).
                tracing::error!(?key, cursor, ?cwd, ?e, "pipe spawn failed");
                drop(permit);
                return Err(SpawnFailureCause::OsSpawn);
            }
        };
        // We're done with `stage_specs` and `resolved` — let them
        // drop here so per-stage env Vecs / argvs aren't kept alive
        // past the spawn call. (They don't carry per-process state;
        // the spawner has dup'd the argv/env into the children.)
        drop(stage_specs);
        drop(resolved);

        let crate::spawner::PipeSpawnHandles {
            last_pid,
            waiter,
            combined_signaler,
            stage_signalers,
        } = handles;

        // Pre-clone the per-stage Arcs that the optional per-stage
        // timer threads will hold. `stage_signalers` is a local
        // owning the Box<[Arc<...>]> — each `Arc::clone` here mints
        // a fresh handle for the timer thread; the locals' Arcs drop
        // when this function returns, leaving the timer threads (and
        // the aggregating PipeWaiter, which has its own per-stage
        // clones) as the per-stage signaler co-owners.
        //
        // `filter_map` collects only the stages that carry a timeout —
        // most pipes don't, and reserving slot space for absent specs
        // would waste a `Vec<Option<…>>` over `Vec<(usize, …)>`. The
        // `usize` is the stage index so the consumer loop can name
        // threads by stage.
        let timer_specs: Vec<(usize, TimerSpec)> = stages
            .iter()
            .zip(stage_signalers.iter())
            .enumerate()
            .filter_map(|(idx, (stage, signaler))| {
                stage.timeout.map(|deadline| {
                    (
                        idx,
                        TimerSpec {
                            deadline,
                            grace: self.shutdown_grace,
                            signaler: Arc::clone(signaler),
                        },
                    )
                })
            })
            .collect();

        let diff_tmp_path: Option<PathBuf> = diff_path.map(Path::to_path_buf);
        let slot = self
            .slots
            .get_mut(key)
            .expect("slot present at install (single-threaded controller just dispatched here)");
        slot.running = Some(RunningJob {
            pid: last_pid,
            signaler: combined_signaler,
            effect: Arc::clone(effect),
            cursor,
            diff_tmp_path,
        });

        self.spawn_wait_thread_after_install(
            key,
            sub,
            correlation,
            last_pid,
            cursor,
            waiter,
            permit,
            reap_tx,
        )?;
        // `stage_signalers` (the install-time local Arc handles) drop
        // here; the aggregating waiter and any armed timer threads
        // keep the per-stage signalers alive through reap.
        drop(stage_signalers);

        // Per-stage timers. Best-effort: a spawn-failure for one
        // stage's timer leaves that stage without a deadline but the
        // rest of the pipe and the other timers are unaffected.
        for (idx, spec) in timer_specs {
            let timer_name = format!("pipe-c{cursor}-s{idx}-pid{last_pid}");
            if let Err(e) =
                timer::spawn_timer(&timer_name, spec.deadline, spec.grace, spec.signaler)
            {
                tracing::error!(
                    ?key,
                    cursor,
                    stage = idx,
                    ?e,
                    "per-stage timer thread spawn failed; deadline not enforced",
                );
            }
        }

        tracing::debug!(
            ?key,
            cursor,
            last_pid,
            stages = stages.len(),
            "spawned pipe"
        );
        Ok(())
    }

    /// Spawn the wait thread for an already-installed
    /// [`Slot::running`]. On `thread::Builder::spawn` failure, take
    /// the running job back via the slot, recover the orphan child
    /// (SIGKILL + `reap_blocking`), and return
    /// [`SpawnFailureCause::WaitThread`] so the caller routes through
    /// `advance_or_terminate` with a synthesised
    /// `EffectOutcome::Failed`.
    ///
    /// `pid` is used only for the wait-thread's OS name
    /// (`act-wait-{pid}`); for single-process steps it's the spawned
    /// child's pid; for pipes it's the last stage's pid (the
    /// operator-facing "the pid of this pipe"). The `key` is needed
    /// to look up the slot in the recovery branch.
    ///
    /// The function is `&mut self` because the recovery branch
    /// mutates `self.slots[key].running`. The slot lookup is an
    /// `expect` for the same reason as `spawn_exec_with_permit`:
    /// the controller is single-threaded and the caller has just
    /// installed `slot.running`.
    #[allow(clippy::too_many_arguments, clippy::needless_pass_by_value)]
    fn spawn_wait_thread_after_install(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        correlation: CorrelationId,
        pid: u32,
        cursor: u32,
        waiter: Box<dyn ChildWaiter>,
        permit: Permit,
        reap_tx: &Sender<super::Reaped>,
    ) -> Result<(), SpawnFailureCause> {
        let reap_tx_for_thread = reap_tx.clone();
        let wait_key = key.clone();
        if let Err(e) = std::thread::Builder::new()
            // Linux pthread_setname_np truncates to 15 chars + null;
            // `act-wait-` is 9 chars, leaving room for a 6-digit pid
            // unscathed. macOS allows 64 bytes.
            .name(format!("act-wait-{pid}"))
            .spawn(move || {
                wait_loop(
                    waiter,
                    wait_key,
                    sub,
                    correlation,
                    permit,
                    reap_tx_for_thread,
                );
            })
        {
            tracing::error!(
                ?key,
                cursor,
                pid,
                ?e,
                "wait thread spawn failed; SIGKILL + sync reap orphan, synth Failed",
            );
            let slot = self
                .slots
                .get_mut(key)
                .expect("slot installed by this call; controller is single-threaded");
            let job = slot
                .running
                .take()
                .expect("slot.running installed unconditionally above");
            recover_orphan_after_wait_thread_failure(job);
            return Err(SpawnFailureCause::WaitThread);
        }
        Ok(())
    }
}

/// Recovery path for the wait-thread-spawn-failure case in
/// [`ActuatorState::spawn_step_with_permit`]. The child is alive but
/// its paired [`crate::spawner::ChildWaiter`] was dropped along with
/// the failed `Builder::spawn` closure — so the controller must
/// SIGKILL it and synchronously reap it through the signaler. Without
/// the reap the OS would leak a zombie until process exit.
///
/// Both syscalls are best-effort: errors are logged and swallowed.
/// The caller's synthesised `EffectOutcome::Failed` is what the engine
/// observes; this function exists only for OS resource hygiene.
///
/// Extracted as a free function so it can be unit-tested in isolation
/// without standing up the full spawn flow (the actual `thread::Builder::spawn`
/// failure path is rare and not directly injectable in tests).
///
/// `job` is taken by value to express the ownership transfer: the
/// caller has just `slot.running.take()`-ed and hands the
/// in-flight bookkeeping over for tear-down; once we return, the
/// signaler / effect Arc / diff-tmp path all drop. Borrowing would
/// force the caller into a take-then-restore dance for no behavioural
/// gain.
///
/// Same recovery shape for Exec and Pipe: the [`RunningJob::signaler`]
/// is either the single-child signaler or the combined fan-out
/// signaler, and both implement SIGKILL + `reap_blocking` correctly
/// for their underlying children.
#[allow(clippy::needless_pass_by_value)]
fn recover_orphan_after_wait_thread_failure(job: RunningJob) {
    let pid = job.pid;
    if let Err(e) = job.signaler.signal_kill() {
        tracing::warn!(pid, ?e, "orphan SIGKILL failed");
    }
    if let Err(e) = job.signaler.reap_blocking() {
        tracing::warn!(pid, ?e, "orphan reap_blocking failed");
    }
}

/// Wait-thread body. Block on `waiter.wait()`; on return, release the
/// permit and send a [`super::Reaped`] to the controller.
///
/// Two orderings are load-bearing:
///
/// 1. The waiter sets `dead = true` (production impl) before returning,
///    so a controller signal racing this thread observes `dead = true`
///    and short-circuits — preventing a stale signal against a reaped
///    (and possibly pid-reused) child.
///
/// 2. Permit release precedes reap notification. Spawns for *other*
///    Subs can dispatch immediately on the freed permit even if the
///    reap channel is briefly saturated. Spawns for the *same* Sub
///    still wait for the controller to drain `running_per_sub[sub]`
///    when it processes the [`super::Reaped`] — by design (per-Sub
///    serialization). The brief stale-counter window between
///    `drop(permit)` and `handle_reap` is benign: same-Sub items
///    defer one extra pump cycle, no over-spawning.
///
/// **Tmp-file cleanup is NOT this thread's responsibility.** The diff
/// tmp file lives for the whole plan (multiple steps may read it) —
/// the wait thread can't see "is this the last step", only the
/// controller can. Cleanup runs in
/// [`ActuatorState::terminate_plan`] exactly once per plan.
#[allow(clippy::needless_pass_by_value)] // closure-spawned: arguments owned for the thread
fn wait_loop(
    waiter: Box<dyn ChildWaiter>,
    key: DedupKey,
    sub: SubId,
    correlation: specter_core::CorrelationId,
    permit: Permit,
    reap_tx: Sender<super::Reaped>,
) {
    let outcome = match std::panic::catch_unwind(AssertUnwindSafe(|| waiter.wait())) {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            tracing::warn!(?key, ?e, "wait failed");
            EffectOutcome::Failed {
                exit_code: None,
                signal: None,
            }
        }
        Err(_) => {
            tracing::error!(?key, "wait panicked");
            EffectOutcome::Failed {
                exit_code: None,
                signal: None,
            }
        }
    };
    drop(permit);
    let _ = reap_tx.send(super::Reaped {
        key,
        sub,
        correlation,
        outcome,
    });
}

#[inline]
pub(crate) const fn sub_of_key(key: &DedupKey) -> SubId {
    match *key {
        DedupKey::PerFile { sub, .. } | DedupKey::Subtree { sub, .. } => sub,
    }
}

#[cfg(test)]
mod tests {
    //! Direct tests for [`ActuatorState::handle_reap_inner`] — the
    //! teardown that both the success and failure spawn paths route
    //! through. The synth-Reap-equivalent paths (spawn-failure inline
    //! and wait-thread-spawn-failure inline) are exercised here against
    //! pre-loaded state, since neither has a fault-injection seam in
    //! the controller harness.
    use super::super::{Reaped, SHUTDOWN_GRACE};
    use super::{ActuatorState, PlanContinuation, ReapPolicy, RunningJob, Slot};
    use crate::env::EnvSnapshot;
    use crate::spawner::{ChildSignaler, ChildWaiter, EnvVar, SpawnHandles, Spawner};
    use compact_str::CompactString;
    use crossbeam::channel::{Sender, unbounded};
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, CorrelationId, DedupKey, Effect, EffectOutcome,
        ExecAction, Input, ProfileId, ResourceId, ResourceKind, SubId,
    };
    use std::io;
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test setup: n must be non-zero")
    }

    /// Construct an [`ActuatorState`] with an empty env snapshot and the
    /// shared `SHUTDOWN_GRACE`. The tests in this module exercise state-
    /// machine transitions, not env resolution or timeout enforcement,
    /// so a single empty snapshot covers every call site. Env-aware
    /// tests live in the higher-level pool harness and inject snapshots
    /// explicitly via [`super::SubprocessActuator::new_with_grace_and_env`].
    fn test_state(concurrency: NonZeroUsize) -> ActuatorState {
        ActuatorState::new(
            concurrency,
            Arc::new(EnvSnapshot::from_map::<_, &str, &str>([])),
            SHUTDOWN_GRACE,
        )
    }

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

    fn perfile_key(sub_seed: u64, profile_seed: u64, res_seed: u64) -> DedupKey {
        DedupKey::PerFile {
            sub: unique_sub_id(sub_seed),
            profile: unique_profile_id(profile_seed),
            resource: unique_resource_id(res_seed),
        }
    }

    /// Program with `n` literal `/bin/true` Exec ops chained on `on_ok =
    /// Continue` (final op `on_ok = Escape`); every `on_failed =
    /// Terminate`. Used by tests that exercise multi-op advance /
    /// terminate.
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

    fn dummy_effect(key: DedupKey, target: ResourceId, corr: u64) -> Effect {
        dummy_effect_with_steps(key, target, corr, 1)
    }

    fn dummy_effect_with_steps(
        key: DedupKey,
        target: ResourceId,
        corr: u64,
        steps: usize,
    ) -> Effect {
        Effect {
            key,
            target,
            forced: false,
            correlation: CorrelationId(corr),
            diff: None,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: n_step_program(steps),
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            target_relative: CompactString::new(""),
            exclude: Arc::from(Vec::<CompactString>::new()),
        }
    }

    /// No-op Spawner stub for tests that go through `handle_reap_inner`
    /// on paths where advance is not attempted (last step, Drop policy,
    /// or non-Ok outcome). The spawner is plumbed through the function
    /// signature; if a test path were to actually invoke `spawn`, the
    /// `unreachable!` would fire and surface the regression.
    #[derive(Default)]
    struct UnusedSpawner;
    impl Spawner for UnusedSpawner {
        fn spawn(
            &self,
            _argv: &[String],
            _env: &[EnvVar<'_>],
            _cwd: &Path,
            _capture_output: bool,
        ) -> io::Result<SpawnHandles> {
            unreachable!("UnusedSpawner used by a test that didn't expect spawn()")
        }
        fn spawn_pipe(
            &self,
            _stages: &[crate::spawner::StageSpec<'_>],
            _cwd: &Path,
            _capture_output: bool,
        ) -> io::Result<crate::spawner::PipeSpawnHandles> {
            unreachable!("UnusedSpawner used by a test that didn't expect spawn_pipe()")
        }
    }

    /// Spawner stub that records every spawn and returns handles whose
    /// waiter is driven via `complete(pid, outcome)`. Used by the
    /// multi-step advance tests where `try_spawn_step` actually runs.
    struct ScriptedSpawner {
        next_pid: std::sync::atomic::AtomicU32,
        spawns: std::sync::Mutex<Vec<u32>>,
        completions: std::sync::Mutex<
            std::collections::BTreeMap<u32, crossbeam::channel::Sender<EffectOutcome>>,
        >,
        inject_err: std::sync::Mutex<Option<io::ErrorKind>>,
    }
    impl ScriptedSpawner {
        fn new() -> Self {
            Self {
                next_pid: std::sync::atomic::AtomicU32::new(20_000),
                spawns: std::sync::Mutex::new(Vec::new()),
                completions: std::sync::Mutex::new(std::collections::BTreeMap::new()),
                inject_err: std::sync::Mutex::new(None),
            }
        }
        fn spawned(&self) -> Vec<u32> {
            self.spawns.lock().unwrap().clone()
        }
        fn complete(&self, pid: u32, outcome: EffectOutcome) {
            let tx = self
                .completions
                .lock()
                .unwrap()
                .get(&pid)
                .cloned()
                .expect("pid must have been spawned");
            tx.send(outcome).expect("waiter still listening");
        }
        fn inject_spawn_error(&self, kind: io::ErrorKind) {
            *self.inject_err.lock().unwrap() = Some(kind);
        }
    }
    impl Spawner for ScriptedSpawner {
        fn spawn(
            &self,
            _argv: &[String],
            _env: &[EnvVar<'_>],
            _cwd: &Path,
            _capture_output: bool,
        ) -> io::Result<SpawnHandles> {
            // Copy out of the lock before checking — the MutexGuard's
            // significant Drop would otherwise live across the if-let.
            let injected = *self.inject_err.lock().unwrap();
            if let Some(kind) = injected {
                return Err(io::Error::from(kind));
            }
            let pid = self.next_pid.fetch_add(1, Ordering::SeqCst);
            self.spawns.lock().unwrap().push(pid);
            let (tx, rx) = crossbeam::channel::bounded::<EffectOutcome>(1);
            self.completions.lock().unwrap().insert(pid, tx);
            let dead = Arc::new(std::sync::atomic::AtomicBool::new(false));
            Ok(SpawnHandles {
                pid,
                waiter: Box::new(ScriptedWaiter {
                    rx,
                    dead: Arc::clone(&dead),
                }),
                signaler: Arc::new(ScriptedSignaler { dead }),
            })
        }
        fn spawn_pipe(
            &self,
            _stages: &[crate::spawner::StageSpec<'_>],
            _cwd: &Path,
            _capture_output: bool,
        ) -> io::Result<crate::spawner::PipeSpawnHandles> {
            // The multi-step pure-state tests in this module don't
            // exercise pipe dispatch — they use single-Exec programs
            // only. Treat as a regression catcher: if a future test
            // accidentally enables pipe dispatch against this stub,
            // surface the missing scaffolding instead of silently
            // succeeding.
            unreachable!("ScriptedSpawner used by a test that didn't expect spawn_pipe()")
        }
    }
    struct ScriptedWaiter {
        rx: crossbeam::channel::Receiver<EffectOutcome>,
        dead: Arc<std::sync::atomic::AtomicBool>,
    }
    impl ChildWaiter for ScriptedWaiter {
        fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
            let r = self.rx.recv();
            self.dead.store(true, Ordering::SeqCst);
            r.map_err(|_| io::Error::other("waiter channel dropped"))
        }
    }
    struct ScriptedSignaler {
        dead: Arc<std::sync::atomic::AtomicBool>,
    }
    impl ChildSignaler for ScriptedSignaler {
        fn signal_term(&self) -> io::Result<()> {
            if self.dead.load(Ordering::SeqCst) {
                return Ok(());
            }
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            if self.dead.load(Ordering::SeqCst) {
                return Ok(());
            }
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            // ScriptedSpawner's waiter drives reap via the completion
            // channel; this method is the recovery-path only and
            // should not be invoked under the tests that use this
            // stub. A no-op is correct for shape-only conformance.
            self.dead.store(true, Ordering::SeqCst);
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.load(Ordering::SeqCst)
        }
    }

    /// Counts SIGTERM / SIGKILL / reap_blocking invocations; never
    /// errors. Shared across `RunningJob` constructions in tests so
    /// teardown assertions can distinguish which signaler methods
    /// fired (e.g. the wait-thread-spawn-failure recovery test
    /// asserts both `kill` and `reap` bumped by 1).
    #[derive(Default)]
    struct CountingSignaler {
        term: AtomicUsize,
        kill: AtomicUsize,
        reap: AtomicUsize,
    }
    impl ChildSignaler for CountingSignaler {
        fn signal_term(&self) -> io::Result<()> {
            self.term.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            self.kill.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            self.reap.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }
        fn is_dead(&self) -> bool {
            // The counted fixture has no `dead` flag: tests using it
            // never exercise paths that probe completion. Returning
            // `false` keeps the per-step timer's `is_dead` short-
            // circuit inert under these tests — the timer's signal
            // path is what's being asserted, not the short-circuit.
            false
        }
    }

    /// Build a stub `RunningJob` that mimics a freshly-spawned step
    /// of a single-step plan. Counted signaler shared so tests can
    /// assert no SIGTERM/SIGKILL during pure-state teardown.
    fn stub_running_job(effect: Arc<Effect>, signaler: Arc<CountingSignaler>) -> RunningJob {
        RunningJob {
            pid: 99_999,
            signaler,
            effect,
            cursor: 0,
            diff_tmp_path: None,
        }
    }

    /// Channel pair sized for the controller's reap channel; rarely
    /// drained in these tests since most paths don't actually spawn.
    fn reap_channel() -> (Sender<Reaped>, crossbeam::channel::Receiver<Reaped>) {
        crossbeam::channel::bounded(64)
    }

    // ---------- wait-thread-spawn-failure recovery ----------

    /// Direct test for [`super::recover_orphan_after_wait_thread_failure`].
    /// The production-path trigger (`thread::Builder::spawn` returning
    /// `Err`) is rare and not directly injectable in the controller
    /// harness, so we exercise the recovery helper in isolation against
    /// a stub [`RunningJob`].
    ///
    /// The bug the helper closes: pre-fix, the recovery branch only
    /// called `signal_kill`. The waiter (the sole reap path) was
    /// dropped along with the failed thread closure, so the SIGKILL'd
    /// orphan was never `waitpid`-ed — leaking a zombie until process
    /// exit. The helper now drives both `signal_kill` and
    /// `reap_blocking` through the controller-held signaler.
    #[test]
    fn recover_orphan_after_wait_thread_failure_kills_and_reaps() {
        let key = perfile_key(50, 50, 50);
        let res = unique_resource_id(50);
        let effect = Arc::new(dummy_effect(key, res, 1));
        let signaler = Arc::new(CountingSignaler::default());
        let job = stub_running_job(effect, Arc::clone(&signaler));

        super::recover_orphan_after_wait_thread_failure(job);

        assert_eq!(
            signaler.kill.load(Ordering::SeqCst),
            1,
            "orphan must receive exactly one SIGKILL",
        );
        assert_eq!(
            signaler.reap.load(Ordering::SeqCst),
            1,
            "orphan must be synchronously reaped via waitpid",
        );
        assert_eq!(
            signaler.term.load(Ordering::SeqCst),
            0,
            "recovery never sends SIGTERM; it's a hard-fail path",
        );
    }

    /// Stale-Reaped shape: slot exists with no running job and no
    /// counter bump (today's spawn-failure-before-bump entry).
    /// Terminal arm runs unconditionally — emits EffectComplete with
    /// the reaped outcome, removes the slot, and leaves counters intact.
    #[test]
    fn handle_reap_inner_stale_for_unspawned_slot_clears_state() {
        let mut state = test_state(nz(2));
        let key = perfile_key(1, 1, 1);
        let sub = unique_sub_id(1);
        state.slots.insert(key.clone(), Slot::default());
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key: key.clone(),
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Failed {
                    exit_code: None,
                    signal: None,
                },
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        assert!(state.slots.is_empty(), "slot removed");
        assert!(
            state.running_per_sub.is_empty(),
            "counter not underflowed by saturating_sub against absent entry",
        );
        assert!(state.ready_queue.is_empty());
        match rx.try_recv() {
            Ok(Input::EffectComplete {
                sub: s,
                key: k,
                result,
            }) => {
                assert_eq!(s, sub);
                assert_eq!(k, key);
                assert!(matches!(
                    result,
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    }
                ));
            }
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    /// Single-step plan, Failed outcome: terminal arm runs, signaler is
    /// dropped without sending SIGTERM/SIGKILL (the child already died,
    /// we just got the reap), counter decrements to zero and entry is
    /// removed.
    #[test]
    fn handle_reap_inner_failed_single_step_decrements_and_removes() {
        let mut state = test_state(nz(2));
        let key = perfile_key(2, 2, 2);
        let sub = unique_sub_id(2);
        let res = unique_resource_id(2);
        let signaler = Arc::new(CountingSignaler::default());
        let effect = Arc::new(dummy_effect(key.clone(), res, 5));
        let slot = Slot {
            running: Some(stub_running_job(effect, Arc::clone(&signaler))),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key,
                sub,
                correlation: CorrelationId(5),
                outcome: EffectOutcome::Failed {
                    exit_code: None,
                    signal: None,
                },
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        assert!(state.slots.is_empty(), "slot removed");
        assert!(state.running_per_sub.is_empty(), "counter cleared");
        assert_eq!(
            signaler.term.load(Ordering::SeqCst),
            0,
            "no SIGTERM during teardown",
        );
        assert_eq!(
            signaler.kill.load(Ordering::SeqCst),
            0,
            "no SIGKILL during teardown",
        );
        let _ = rx.try_recv().expect("EffectComplete emitted");
    }

    /// Pump policy with non-empty pending re-queues the slot for the
    /// next pump cycle; running is cleared but the slot stays alive
    /// so handle_submit's Latest coalesce continues to work.
    #[test]
    fn handle_reap_inner_pump_with_pending_requeues_for_respawn() {
        let mut state = test_state(nz(2));
        let key = perfile_key(3, 3, 3);
        let sub = unique_sub_id(3);
        let res = unique_resource_id(3);
        let signaler = Arc::new(CountingSignaler::default());
        let effect = Arc::new(dummy_effect(key.clone(), res, 7));
        let slot = Slot {
            running: Some(stub_running_job(effect, signaler)),
            pending: Some(dummy_effect(key.clone(), res, 8)),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, _rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key: key.clone(),
                sub,
                correlation: CorrelationId(7),
                outcome: EffectOutcome::Ok,
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        let slot_after = state.slots.get(&key).expect("slot preserved with pending");
        assert!(slot_after.running.is_none(), "running cleared");
        assert!(
            slot_after.pending.is_some(),
            "pending preserved for re-spawn"
        );
        assert!(slot_after.in_ready_queue);
        assert_eq!(
            state.ready_queue.iter().collect::<Vec<_>>(),
            vec![&key],
            "key re-queued for next pump",
        );
        assert!(state.running_per_sub.is_empty());
    }

    /// Drop policy (shutdown phase) removes the slot regardless of
    /// pending; pending is silently dropped, mirroring the
    /// `handle_reap_no_pump` shutdown contract.
    #[test]
    fn handle_reap_inner_drop_policy_removes_slot_even_with_pending() {
        let mut state = test_state(nz(2));
        let key = perfile_key(4, 4, 4);
        let sub = unique_sub_id(4);
        let res = unique_resource_id(4);
        let signaler = Arc::new(CountingSignaler::default());
        let effect = Arc::new(dummy_effect(key.clone(), res, 11));
        let slot = Slot {
            running: Some(stub_running_job(effect, signaler)),
            pending: Some(dummy_effect(key.clone(), res, 12)),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, _rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key,
                sub,
                correlation: CorrelationId(11),
                outcome: EffectOutcome::Failed {
                    exit_code: None,
                    signal: None,
                },
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Drop,
        );

        assert!(state.slots.is_empty(), "slot removed under Drop policy");
        assert!(state.running_per_sub.is_empty());
        assert!(state.ready_queue.is_empty(), "no re-queue under Drop");
    }

    // ---------- multi-step advance / terminate ----------

    /// Step Ok and not last under Pump policy: handle_reap_inner takes
    /// the running, calls try_spawn_step which acquires a fresh permit
    /// and spawns instruction N+1. Slot.running is reinstalled with cursor
    /// incremented; per-Sub counter stays at +1 (one bump per program, not
    /// per instruction); no EffectComplete is emitted.
    #[test]
    fn step_ok_not_last_advances_to_next_step() {
        let mut state = test_state(nz(2));
        let key = perfile_key(10, 10, 10);
        let sub = unique_sub_id(10);
        let res = unique_resource_id(10);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 100,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp_path: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = ScriptedSpawner::new();

        state.handle_reap_inner(
            Reaped {
                key: key.clone(),
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Ok,
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        assert_eq!(spawner.spawned().len(), 1, "step 1 was spawned");
        let slot_after = state
            .slots
            .get(&key)
            .expect("slot preserved during advance");
        let running = slot_after.running.as_ref().expect("running reinstalled");
        assert_eq!(running.cursor, 1, "cursor advanced");
        assert_eq!(
            state.running_per_sub.get(&sub).copied(),
            Some(1),
            "per-Sub counter unchanged across step advance",
        );
        assert!(rx.try_recv().is_err(), "no EffectComplete emitted mid-plan");
        // Drain the wait thread so the test doesn't hang on Drop.
        spawner.complete(running.pid, EffectOutcome::Ok);
    }

    /// Step Failed mid-plan: terminal arm runs with the reaped Failed
    /// outcome — no advance attempted. Counter decrements; EffectComplete
    /// emitted; slot removed.
    #[test]
    fn step_failed_mid_plan_terminates_without_advance() {
        let mut state = test_state(nz(2));
        let key = perfile_key(11, 11, 11);
        let sub = unique_sub_id(11);
        let res = unique_resource_id(11);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 200,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp_path: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key,
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Failed {
                    exit_code: Some(2),
                    signal: None,
                },
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        assert!(state.slots.is_empty(), "slot removed on terminal");
        assert!(state.running_per_sub.is_empty(), "counter cleared");
        match rx.try_recv() {
            Ok(Input::EffectComplete { result, .. }) => assert!(matches!(
                result,
                EffectOutcome::Failed {
                    exit_code: Some(2),
                    signal: None,
                }
            )),
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    /// Last step Ok: terminal arm runs (no advance possible). Counter
    /// decrements; EffectComplete emitted with Ok.
    #[test]
    fn last_step_ok_terminates() {
        let mut state = test_state(nz(2));
        let key = perfile_key(12, 12, 12);
        let sub = unique_sub_id(12);
        let res = unique_resource_id(12);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 300,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 2, // last instruction (0-indexed) of a 3-instruction program
                diff_tmp_path: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key,
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Ok,
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        assert!(state.slots.is_empty(), "slot removed after last step");
        assert!(state.running_per_sub.is_empty());
        match rx.try_recv() {
            Ok(Input::EffectComplete { result, .. }) => {
                assert!(matches!(result, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete::Ok; got {other:?}"),
        }
    }

    /// Permit unavailable mid-program: try_spawn_step returns Deferred,
    /// the slot's plan_continue is set to (effect, cursor+1, diff),
    /// the slot is queued for the next pump cycle, no EffectComplete is
    /// emitted, the counter stays at +1.
    #[test]
    fn step_ok_not_last_with_no_permit_defers_via_plan_continue() {
        // cap=1 with another job already holding the only permit.
        let mut state = test_state(nz(1));
        let _hold = state
            .permits
            .try_acquire()
            .expect("acquire hogs the only permit");
        let key = perfile_key(13, 13, 13);
        let sub = unique_sub_id(13);
        let res = unique_resource_id(13);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 2));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 400,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp_path: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key: key.clone(),
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Ok,
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        let slot_after = state.slots.get(&key).expect("slot preserved");
        assert!(slot_after.running.is_none(), "running was taken");
        let cont = slot_after
            .plan_continue
            .as_ref()
            .expect("plan_continue installed");
        assert_eq!(cont.cursor, 1, "deferred at instruction 1");
        assert!(slot_after.in_ready_queue);
        assert_eq!(state.ready_queue.iter().collect::<Vec<_>>(), vec![&key]);
        assert_eq!(
            state.running_per_sub.get(&sub).copied(),
            Some(1),
            "counter stays at +1 across deferral",
        );
        assert!(rx.try_recv().is_err(), "no EffectComplete on deferral");
    }

    /// `handle_submit` during plan_continue replaces pending only;
    /// plan_continue is left intact (plan-atomicity invariant).
    #[test]
    fn submit_during_plan_continue_replaces_pending_only() {
        let mut state = test_state(nz(1));
        let _hold = state.permits.try_acquire().expect("acquire");
        let key = perfile_key(14, 14, 14);
        let sub = unique_sub_id(14);
        let res = unique_resource_id(14);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 2));
        let slot = Slot {
            plan_continue: Some(PlanContinuation {
                effect: Arc::clone(&effect),
                cursor: 1,
                diff_tmp_path: None,
            }),
            in_ready_queue: true,
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.ready_queue.push_back(key.clone());
        state.running_per_sub.insert(sub, 1);
        let (tx, _rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        // Submit a new effect for the same key.
        let new_effect = dummy_effect(key.clone(), res, 99);
        state.handle_submit(new_effect, &spawner, &reap_tx, &tx);

        let slot_after = state.slots.get(&key).expect("slot preserved");
        let cont = slot_after
            .plan_continue
            .as_ref()
            .expect("plan_continue NOT touched by submit");
        assert_eq!(cont.cursor, 1);
        let pending = slot_after.pending.as_ref().expect("pending set");
        assert_eq!(
            pending.correlation,
            CorrelationId(99),
            "pending replaced by new submit",
        );
    }

    /// Drop policy mid-plan: terminal arm runs immediately with the
    /// reaped outcome; advance is skipped under shutdown so subsequent
    /// steps are abandoned.
    #[test]
    fn step_ok_not_last_under_drop_policy_skips_advance() {
        let mut state = test_state(nz(2));
        let key = perfile_key(15, 15, 15);
        let sub = unique_sub_id(15);
        let res = unique_resource_id(15);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 500,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp_path: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.handle_reap_inner(
            Reaped {
                key,
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Ok,
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Drop,
        );

        assert!(state.slots.is_empty(), "slot removed under Drop");
        assert!(state.running_per_sub.is_empty());
        match rx.try_recv() {
            Ok(Input::EffectComplete { result, .. }) => {
                assert!(matches!(result, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete::Ok; got {other:?}"),
        }
    }

    /// Spawn-failure on next step (try_spawn_step returns Failed):
    /// terminate_plan runs with synthesised `Failed`, counter decrements,
    /// slot removed.
    #[test]
    fn step_ok_not_last_with_spawn_failure_synthesises_failed() {
        let mut state = test_state(nz(2));
        let key = perfile_key(16, 16, 16);
        let sub = unique_sub_id(16);
        let res = unique_resource_id(16);
        let effect = Arc::new(dummy_effect_with_steps(key.clone(), res, 1, 2));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 600,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp_path: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = ScriptedSpawner::new();
        spawner.inject_spawn_error(io::ErrorKind::NotFound);

        state.handle_reap_inner(
            Reaped {
                key,
                sub,
                correlation: CorrelationId(1),
                outcome: EffectOutcome::Ok,
            },
            &tx,
            &spawner,
            &reap_tx,
            ReapPolicy::Pump,
        );

        assert!(state.slots.is_empty(), "slot removed after synth Failed");
        assert!(state.running_per_sub.is_empty(), "counter cleared");
        match rx.try_recv() {
            Ok(Input::EffectComplete { result, .. }) => assert!(matches!(
                result,
                EffectOutcome::Failed {
                    exit_code: None,
                    signal: None,
                }
            )),
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    // The old `dispatch_outcome` + `next_spawnable` pure-function tests
    // pinned the bytecode dispatch table. Under the CFG-shaped IR these
    // helpers are gone — dispatch is `ProgramOp::target(&outcome)` which
    // returns a `BranchTarget` directly. Routing coverage moved to
    // `specter-core::program::op::tests`; end-to-end behaviour is
    // covered by the multi-step advance/terminate tests above plus the
    // controller-level tests in `pool.rs`.
}
