//! Actuator state machine: slot map, ready queue, per-Sub running set, global semaphore.
//!
//! All mutations happen on the controller thread. The wait threads send `Reaped` events through
//! `reap_tx`; the controller pulls them off `reap_rx` (also held inside
//! [`super::SubprocessActuator`]) and routes to [`ActuatorState::handle_reap`].
//!
//! `ready_queue` orders slots that want to spawn — submit-FIFO. The `in_ready_queue` flag dedups: a
//! key already queued (e.g., a slot whose pending was just replaced) doesn't get pushed twice.
//!
//! # Programs, cursors, and accounting
//!
//! An [`Effect`] carries an [`specter_core::ActionProgram`]: a flat `Box<[ProgramOp]>` walked by a
//! `u32` cursor. Each op carries a [`SpawnBody`] (single Exec or N-stage Pipe) plus explicit
//! `on_ok` / `on_failed` branch targets — dispatch after a reap is a single
//! [`ProgramOp::target`](specter_core::program::ProgramOp::target) lookup on the outcome. The
//! actuator walks the program with stop-on-failure semantics encoded by the lowering pass
//! (Exec/Pipe `on_failed = Terminate`; predicate `on_failed` ≠ Terminate so the predicate outcome
//! doesn't propagate).
//!
//! - **Per-Effect-stable** state (per-Sub set membership, diff tmp file) is owned by
//!   [`ActuatorState::start_plan`]: insert on plan start, remove on plan terminus.
//! - **Per-op** state (permit, OS process, wait thread) is owned by
//!   [`ActuatorState::spawn_step_with_permit`]: each op acquires a fresh permit, the wait thread
//!   releases it on reap.
//! - **One completion per Effect**: emitted exactly once via [`EffectCompleteSender::send`] at plan
//!   terminus (any [`BranchTarget::Terminate`] or [`BranchTarget::Escape`], or any reap under
//!   shutdown's `Drop` policy). The engine's `outstanding` accounting is unchanged under multi-op
//!   programs — the engine doesn't know programs have multiple ops.
//!
//! Between two adjacent ops the slot may be in an intermediate state ([`Slot::plan_continue`]) when
//! the wait-thread has reaped op N but no permit is available for op N+1. The pump's plan-continue
//! arm has priority over fresh `pending`: continuation work bypasses the per-Sub gate (it's the
//! same program, already admitted) but still respects the global permit cap.

use crate::EffectCompleteSender;
use crate::env::EnvSnapshot;
use crate::permits::{Permit, Permits};
use crate::resolve::{self, CommandResolved};
use crate::spawner::{ChildSignaler, ChildWaiter, EnvVar, Spawner, StageSpec};
use crate::timer;
use crate::tmp::DiffTmpFile;
use crossbeam::channel::Sender;
use specter_core::program::{BranchTarget, ExecAction, SpawnBody};
use specter_core::{
    DedupKey, Effect, EffectCompletion, EffectOutcome, ProfileId, SubId, Termination,
};
use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Policy for [`ActuatorState::terminate_plan`]: under `Pump` re-queue the slot's pending Effect
/// (if any) for the next pump cycle; under `Drop` (shutdown drain) drop pending and remove the
/// slot. Only [`ActuatorState::handle_reap_drop`] passes `Drop`; every other terminate site routes
/// through the Pump-policy advance pipeline.
#[derive(Copy, Clone)]
enum ReapPolicy {
    Pump,
    Drop,
}

/// Outcome of an attempted instruction spawn. Returned by [`ActuatorState::try_spawn_step`] (which
/// acquires a permit).
///
/// The `Failed` variant carries a typed [`SpawnFailureCause`] discriminant: the synth-Failed
/// dispatch sites log it alongside the synthesised `EffectOutcome::Failed(Termination::Internal)`
/// so an operator triaging "this predicate took the else-branch unexpectedly" can match against the
/// cause-side `error!` log line (resolver, OS spawn, wait-thread) and tell "predicate binary
/// missing" from "predicate exited 1 cleanly".
///
/// The cause is **internal-only**: the engine never sees this type. The wire outcome is
/// `EffectOutcome::Failed(Termination::Internal)` regardless of cause. Splitting cause from outcome
/// here is telemetry-only — it lets the synth-Failed log carry a discriminant without changing
/// engine-side dispatch.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SpawnError {
    /// Permit semaphore at capacity. The caller defers the instruction into [`Slot::plan_continue`]
    /// and re-queues the slot.
    Deferred,
    /// Spawn (or pre-spawn) failure with a typed cause. The caller routes through
    /// [`ActuatorState::advance_or_terminate`] with a synthesised `EffectOutcome::Failed`; the
    /// dispatch then decides terminate vs continue based on the op's `on_failed` edge at the
    /// failing cursor — predicate spawn-failures still get their no-propagation semantics through
    /// that dispatch.
    Failed(SpawnFailureCause),
}

/// Why a spawn attempt failed. Surfaces at three synthesis sites ([`ActuatorState::start_plan`],
/// [`ActuatorState::spawn_continuation`], [`ActuatorState::advance_or_terminate`]); each site emits
/// a `tracing::warn!` carrying this discriminant so the synthesised
/// `EffectOutcome::Failed(Termination::Internal)` can be correlated against the cause-side `error!`
/// log line.
///
/// **Not part of the engine wire format.** `EffectOutcome::Failed` carries no cause discriminant;
/// the engine's dispatch reads only the op's `on_failed` edge. Predicate spawn-failure and
/// predicate non-zero-exit are observationally identical to the engine, by design (the op's edge
/// decides routing without inspecting cause).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SpawnFailureCause {
    /// Argv / env substitution failed before any child or wait thread was spawned. Today the only
    /// resolver error is [`crate::resolve::ResolveError::UnsetEnvVar`] — a strict `${env.<NAME>}`
    /// reference against an unset key with no `:-` default. Future resolver-time errors (e.g., path
    /// canonicalisation) would land here.
    Resolver,
    /// OS-level process spawn failed — [`crate::spawner::Spawner::spawn`] / `spawn_pipe` returned
    /// an error (ENOENT on the binary, EAGAIN, EMFILE, …). For pipes the spawner has already rolled
    /// back any partially-spawned stages before the error reaches this discriminant.
    OsSpawn,
    /// `thread::Builder::spawn` for the wait thread failed. The spawned child is alive but its
    /// paired [`crate::spawner::ChildWaiter`] was dropped; the recovery branch SIGKILLs and
    /// synchronously reaps the orphan before this discriminant surfaces.
    WaitThread,
}

/// Per-`DedupKey` actuator slot.
///
/// At most one in-flight child (`running`) plus a single Latest-coalesce next-plan slot (`pending`)
/// plus, between adjacent instructions of an in-flight plan when the global permit cap is
/// exhausted, a `plan_continue` hand-off.
///
/// **Three slots, three roles:**
///
/// - `running` is the currently-spawned instruction's bookkeeping (pid, signaler for shutdown
///   SIGTERM/SIGKILL, plus the per-plan snapshot needed to advance to the next instruction).
/// - `plan_continue` is "this plan's next instruction, deferred on permit." Bypasses the per-Sub
///   gate (same program, already admitted by `start_plan`) but respects the global permit cap.
/// - `pending` is the user's next intent. Latest-coalesced on submit; never replaces a running
///   instruction or a `plan_continue`.
///
/// **Plan-atomicity invariant.** A new submit during a running plan replaces `pending` only;
/// `plan_continue` is never touched by coalesce. Once started, a plan runs all its instructions
/// before `pending` fires.
///
/// **Engine-side twin.** Every `Effect` the actuator runs corresponds to a `+1` on the engine's
/// `PostFirePhase::Awaiting { outstanding }` counter for the owning Profile. The slot retires the
/// plan (or drops the pending Effect on shutdown) and calls [`EffectCompleteSender::send`] exactly
/// once per Effect — multi-instruction programs don't change the engine's accounting.
#[derive(Debug, Default)]
pub(crate) struct Slot {
    pub running: Option<RunningJob>,
    pub plan_continue: Option<PlanContinuation>,
    pub pending: Option<Effect>,
    pub in_ready_queue: bool,
}

/// Bookkeeping for one in-flight op of a plan.
///
/// With the CFG-shaped IR, outcome routing (propagate / branch / no-op) lives on the op's edges
/// ([`ProgramOp::on_ok`](specter_core::program::ProgramOp::on_ok) /
/// [`ProgramOp::on_failed`](specter_core::program::ProgramOp::on_failed)), not in the running job's
/// variant tag. The reap-path reads the edge directly via
/// [`ProgramOp::target`](specter_core::program::ProgramOp::target), so there's nothing here that
/// depends on which spawn shape produced the running child.
///
/// Carries:
///
/// - **`pid`** — the operator-facing pid. For [`SpawnBody::Exec`], the child's pid; for
///   [`SpawnBody::Pipe`], the *last* stage's pid (what `ps` would label "the pipe").
///   Intermediate-stage pids stay inside the per-stage signalers (used only for the per-stage timer
///   threads at install time, then dropped).
/// - **`signaler`** — the signaler the controller uses for shutdown SIGTERM / SIGKILL. For Exec
///   this is the single-child signaler; for Pipe this is the combined fan-out signaler that signals
///   every stage. Per-stage signalers DO NOT live here: pipe install collects them as locals, arms
///   per-stage timer threads against each (cloning the Arc), then drops the locals when install
///   returns. The aggregating `PipeWaiter` owns its own per-stage signaler clones for the
///   SIGTERM-cascade-on-first-failure path, independent of this combined signaler.
/// - **`effect`** — the plan's shared `Arc<Effect>`. The advance branch in
///   [`ActuatorState::advance_or_terminate`] re-resolves op N+1's argv + env from the same snapshot
///   without re-fetching.
/// - **`cursor`** — `u32` index into `effect.program.ops`.
/// - **`diff_tmp`** — `Some` iff `start_plan` materialised a diff tmp file. The handle is co-owned by
///   `Slot::running` and `Slot::plan_continue` across the plan's steps via `Arc<DiffTmpFile>`, so
///   every step reads the same `SPECTER_DIFF_PATH`. The Arc's `Drop` impl unlinks the file when the
///   last co-owner is dropped — at plan terminus, after the final [`ActuatorState::terminate_plan`]
///   has emitted `EffectComplete`. See [`crate::tmp`] for the lifecycle invariant.
///
/// `signaler` is `Arc<dyn>` so the controller's installed-side reference and the per-step timer
/// thread's clone are independent co-owners; either may outlive the other.
pub(crate) struct RunningJob {
    pub pid: u32,
    pub signaler: Arc<dyn ChildSignaler>,
    pub effect: Arc<Effect>,
    pub cursor: u32,
    pub diff_tmp: Option<Arc<DiffTmpFile>>,
}

impl std::fmt::Debug for RunningJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningJob")
            .field("pid", &self.pid)
            .field("cursor", &self.cursor)
            .field("sub", &self.effect.sub)
            .field("correlation", &self.effect.correlation)
            .finish_non_exhaustive()
    }
}

/// Hand-off slot between two adjacent instructions when no permit is available at advance time. The
/// pump's plan-continue arm consumes this in priority over [`Slot::pending`] — same program,
/// already admitted, just waiting on the global cap.
///
/// `diff_tmp` carries the plan-bound tmp handle across the permit-deferred boundary. On
/// continuation, the next step's `RunningJob` clones the Arc into its own `diff_tmp` slot; the
/// `PlanContinuation`'s Arc drops with the destructure that consumes it. The file persists because
/// the new `RunningJob` holds a clone.
#[derive(Debug)]
pub(crate) struct PlanContinuation {
    pub effect: Arc<Effect>,
    pub cursor: u32,
    pub diff_tmp: Option<Arc<DiffTmpFile>>,
}

#[derive(Debug)]
pub(crate) struct ActuatorState {
    pub slots: BTreeMap<DedupKey, Slot>,
    pub ready_queue: VecDeque<DedupKey>,
    /// Per-Sub serialization gate: the set of Subs whose plan is in flight. [`Self::start_plan`]
    /// inserts on entry; [`Self::terminate_plan`] removes on the live-plan exit; [`Self::pump`]'s
    /// fresh-plan arm reads via `contains`. The pump's gate enforces at-most-one concurrent fresh
    /// plan per Sub, so the set's cardinality per member is structurally `{absent, present}` — set
    /// membership is the exact shape (a counted multimap would over-type the gate).
    ///
    /// **Stale-Reaped arms do NOT touch this set**: they route through [`Self::terminate_stale`],
    /// which performs engine accounting only. A stale Reaped's `sub` may still own a live plan at a
    /// different `DedupKey`; removing it here would silently clobber the live plan's gate hold.
    pub running_subs: BTreeSet<SubId>,
    pub permits: Permits,
    /// Captured operator env, threaded into every resolver call for `${env.<NAME>}` substitution.
    /// Shared by `Arc` because the snapshot is immutable for the actuator's lifetime; the rare test
    /// override case constructs a fresh snapshot rather than mutating the existing one.
    pub env_snapshot: Arc<EnvSnapshot>,
    /// Captured operator `$TMPDIR` (`std::env::temp_dir`) at actuator startup. Threaded into every
    /// [`DiffTmpFile::create`] call so the spawn path makes no `getenv(TMPDIR)` syscall per Effect.
    /// Lives at the same lifetime tier as [`Self::env_snapshot`]; shared by `Arc<Path>` for the
    /// same reason — immutable across the actuator's lifetime.
    pub temp_dir: Arc<Path>,
    /// Captured `std::process::id()` at actuator startup. Used in the tmp diff filename
    /// (`specter-{pid}-{corr:016x}.diff`) — the actuator's daemon pid, not the spawned child's
    /// (which isn't known until after `Command::spawn` but the env var must be set *before* spawn).
    pub actuator_pid: u32,
    /// SIGTERM → SIGKILL grace. Reads:
    /// - shutdown drain ([`super::SubprocessActuator::shutdown`]);
    /// - per-step timer thread grace ([`crate::timer::arm_timer`]).
    ///
    /// Pinned in one place so the two paths can't drift on the constant.
    pub shutdown_grace: Duration,
    /// Scratch deque reused across [`Self::pump`] calls to hold keys blocked this round on permit /
    /// per-Sub gate unavailability. Restored to the ready queue at the end of `pump`. Living on the
    /// state (rather than allocated fresh inside `pump`) amortises the `VecDeque::new()` heap
    /// allocation across high-frequency same-Sub submit bursts. Empty between pump calls; the
    /// `debug_assert!` at pump entry pins the invariant.
    pub blocked_scratch: VecDeque<DedupKey>,
}

impl ActuatorState {
    pub fn new(
        concurrency: NonZeroUsize,
        env_snapshot: Arc<EnvSnapshot>,
        temp_dir: Arc<Path>,
        actuator_pid: u32,
        shutdown_grace: Duration,
    ) -> Self {
        Self {
            slots: BTreeMap::new(),
            ready_queue: VecDeque::new(),
            running_subs: BTreeSet::new(),
            permits: Permits::new(concurrency),
            env_snapshot,
            temp_dir,
            actuator_pid,
            shutdown_grace,
            blocked_scratch: VecDeque::new(),
        }
    }

    /// Submit handler — enqueue or coalesce. Always end with `pump`.
    ///
    /// Plan-atomicity: a fresh submit during a running plan replaces `pending` only. Both `running`
    /// and `plan_continue` (an in-flight plan deferred between steps) keep the slot in "plan in
    /// flight" state from the coalesce point of view.
    pub fn handle_submit(
        &mut self,
        effect: Effect,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
        engine_in: &dyn EffectCompleteSender,
    ) {
        let key = effect.key();
        tracing::trace!(?key, "submit");
        let slot = self.slots.entry(key).or_default();
        if slot.running.is_some() || slot.plan_continue.is_some() {
            // Plan in flight; Latest-coalesce — drop old pending if present. Never touches
            // `running` or `plan_continue`: the current plan runs to terminus before pending fires.
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

    /// Reap handler — advance to next step or terminate the plan, then pump any newly-ready work.
    /// The two-step shape (`reap_pump` then `pump`) is so the on-stack reap can re-acquire its
    /// just-freed permit before `pump` runs (plan-atomicity under contention).
    pub fn handle_reap(
        &mut self,
        completion: EffectCompletion,
        engine_in: &dyn EffectCompleteSender,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
    ) {
        self.reap_pump(completion, engine_in, spawner, reap_tx);
        self.pump(spawner, reap_tx, engine_in);
    }

    /// Engine-driven per-profile abandon of in-flight effects.
    ///
    /// Sole caller: the controller's `effects_rx` arm on [`specter_core::EffectOp::Cancel`], fed by
    /// the engine's `handle_gate_deadline` emission. The engine is the authority on what's worth
    /// running; this handler is the obeyer.
    ///
    /// **Per matching slot** (slots whose [`DedupKey::profile`] equals `profile`):
    ///
    /// 1. Drop `pending` and `plan_continue` — queued work the engine has already given up on;
    ///    either becoming live would respawn work the engine doesn't track.
    /// 2. Clear `in_ready_queue` — paired with the `ready_queue` `retain` below; together they
    ///    preserve the "in queue ⇔ flag set" invariant the pump's blocked-scratch drain pins.
    /// 3. SIGTERM the running child if any. The signaler short- circuits on `is_dead`, so a race
    ///    with a natural reap is benign (SIGTERM on a reaped pid is a documented no-op).
    /// 4. Leave `slot.running` in place. The wait thread will deliver `Reaped` through the existing
    ///    pipeline; [`Self::handle_reap`] will run [`Self::terminate_plan`] and emit one completion
    ///    via [`EffectCompleteSender::send`], which the engine routes to
    ///    `EffectCompleteOutsideAwaiting` (the Profile has left `Awaiting` by then).
    ///
    /// **Ready-queue cleanup is load-bearing.** The pump's spawn arm unconditionally calls
    /// `slot.pending.take().expect(...)` after the `plan_continue` branch. If this handler left a
    /// key in `ready_queue` with `pending = None && plan_continue = None`, the next pump would pop
    /// that key and panic on the `expect`. So this handler MUST `retain` the queue to drop
    /// cancelled-profile keys; clearing `in_ready_queue` in step 2 keeps the "in queue ⇔ flag set"
    /// invariant the blocked-scratch drain pins.
    ///
    /// **`running_subs` is NOT touched.** A SIGTERMed child still drives reap → [`Self::handle_reap`]
    /// → [`Self::terminate_plan`], which owns the `running_subs` removal. Removing here would race
    /// with the concurrent reap and could clobber a same-Sub fresh plan at a different [`DedupKey`] —
    /// the precise scenario `terminate_stale` guards against (see its rustdoc on why the per-Sub
    /// gate's bump pairs to a specific `(sub, key)` plan, not to `sub` alone).
    ///
    /// **Idempotent.** Cancel for a profile with no in-flight effects is a no-op: the filter
    /// produces an empty key set, the `retain` walks the queue with no removals, no signals fire.
    pub fn handle_cancel(&mut self, profile: ProfileId) {
        // Snapshot matching keys before mutation — iterating `self.slots` and mutating individual
        // slots in the same scope would alias. The collected `Vec` is bounded by the number of
        // matching slots (≤ concurrency), so the heap allocation is small and rare. `DedupKey` is
        // `Copy`, so the collect is a memcpy of slotmap-keyed handles, not an unbounded clone.
        let keys: Vec<DedupKey> = self
            .slots
            .keys()
            .filter_map(|k| (k.profile() == profile).then_some(*k))
            .collect();
        if keys.is_empty() {
            return;
        }

        for key in &keys {
            // `get_mut` rather than `expect`: the snapshot was taken immediately above on the same
            // controller thread, but the `Option` keeps `handle_cancel` total against any future
            // refactor that touches the slot map between snapshot and iteration.
            let Some(slot) = self.slots.get_mut(key) else {
                continue;
            };
            slot.pending = None;
            slot.plan_continue = None;
            slot.in_ready_queue = false;
            if let Some(job) = slot.running.as_ref()
                && let Err(e) = job.signaler.signal_term()
            {
                // Best-effort: a closed signaler (already-reaped child) is benign; the natural reap
                // continues to drive teardown through `handle_reap` → `terminate_plan`.
                tracing::debug!(?key, pid = job.pid, ?e, "cancel SIGTERM failed");
            }
        }

        // Drop every cancelled-profile key from the ready queue. O(N) where N = current queue
        // length, bounded by concurrency plus queue headroom. Cancel is rare (gate-deadline only),
        // so the walk is fine; `retain` preserves the FIFO order of the surviving keys.
        self.ready_queue.retain(|k| k.profile() != profile);

        tracing::debug!(
            ?profile,
            count = keys.len(),
            "handle_cancel: SIGTERM + ready-queue clean complete"
        );
    }

    /// Shutdown-phase reap handler. Forces the plan to terminus on the reaped step's outcome — no
    /// advance, no pending re-queue, no follow-on pump. Subsequent steps are abandoned and the slot
    /// is removed so phase 3's SIGKILL fan-out won't re-signal an already-reaped child.
    ///
    /// No `spawner` / `reap_tx` parameter: the Drop branch never spawns, so the shutdown caller
    /// threads neither.
    pub fn handle_reap_drop(
        &mut self,
        completion: EffectCompletion,
        engine_in: &dyn EffectCompleteSender,
    ) {
        tracing::trace!(?completion.key, ?completion.outcome, "reap drop");
        // `key` is `Copy` (slotmap handle); read it off the envelope for the slot lookup, then
        // thread the envelope through unchanged into the terminal arm — no destructure-and-rebuild.
        let key = completion.key;
        // Consume the running job (if present). The signaler / effect / waiter fields drop with the
        // job — phase 1 already SIGTERMed; this thread has already reaped the child kernel-side, so
        // dropping the signaler co-owner here is safe. Stale completion (running already taken)
        // routes through `terminate_stale` — same shape as `reap_pump`'s stale arm.
        let Some(job) = self.take_running(&key) else {
            self.terminate_stale(completion, engine_in);
            return;
        };
        self.terminate_plan(completion, ReapPolicy::Drop, engine_in);
        // `job` carries the live-plan's last `Arc<DiffTmpFile>` (the slot's `plan_continue` is
        // structurally `None` while `running` is `Some`, so no second co-owner exists at this
        // point). Dropping it explicitly after `terminate_plan` has emitted `EffectComplete` orders
        // the on-disk unlink after the wire event — see [`DiffTmpFile::drop`].
        drop(job);
    }

    /// The Pump-policy reap pipeline. Two main exits:
    ///
    /// 1. **Advance**: the op's [`ProgramOp::target`](specter_core::program::ProgramOp::target) for
    ///    the reaped outcome is [`BranchTarget::Continue`], so the plan continues at the named
    ///    slot. Handed to [`Self::try_spawn_step`]; a `SpawnError::Failed` here loops the dispatch
    ///    with a synthesised `Failed` outcome for the new cursor — a predicate spawn-failure
    ///    cascade naturally walks to its own [`BranchTarget::Continue`] (the else-branch's first
    ///    op), an exec spawn-failure walks to its [`BranchTarget::Terminate`] and the plan
    ///    terminates with the synth Failed.
    /// 2. **Terminate**: the op's edge target is [`BranchTarget::Terminate`] (carried outcome
    ///    propagates) or [`BranchTarget::Escape`] (terminate Ok regardless of carried outcome — the
    ///    "branch, not guard" outcome elision). `terminate_plan` emits one `EffectComplete`,
    ///    removes the Sub from the per-Sub gate, cleans the diff tmp file, and re-queues the slot's
    ///    `pending` (if any) or removes the slot.
    ///
    /// **Defensive no-job**: a stale completion after slot removal routes through
    /// [`Self::terminate_stale`] (not `terminate_plan`) — emits `EffectComplete` for engine
    /// accounting without touching the per-Sub gate (which a live plan for the same Sub may still
    /// hold at a different key) or the tmp file.
    fn reap_pump(
        &mut self,
        completion: EffectCompletion,
        engine_in: &dyn EffectCompleteSender,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
    ) {
        tracing::trace!(?completion.key, ?completion.outcome, "reap");
        // `key` is `Copy`; read off the envelope for the slot lookup and thread the envelope
        // unchanged into the terminal arm.
        let key = completion.key;
        let Some(job) = self.take_running(&key) else {
            // Stale completion: slot already removed (or running already taken).
            // Engine-accounting-only — see `terminate_stale`'s contract for why this can't go
            // through `terminate_plan` under the set-membership gate.
            self.terminate_stale(completion, engine_in);
            return;
        };
        let RunningJob {
            effect,
            cursor,
            diff_tmp,
            ..
        } = job;
        self.advance_or_terminate(
            completion, effect, diff_tmp, cursor, spawner, reap_tx, engine_in,
        );
    }

    /// Take the running job from its slot, leaving `slot.running = None` and the slot itself in
    /// place. Returns `None` if the slot is absent (the "stale Reaped" path — slot was removed by a
    /// prior terminate before this Reaped landed) or if the slot is present but its `running` was
    /// already taken.
    ///
    /// Callers own the rest of the slot's lifecycle: re-queue under Pump if pending is set,
    /// otherwise remove via [`Self::terminate_plan`].
    fn take_running(&mut self, key: &DedupKey) -> Option<RunningJob> {
        self.slots.get_mut(key).and_then(|s| s.running.take())
    }

    /// Drive the post-reap / post-spawn-failure dispatch loop.
    ///
    /// `cursor` and `outcome` define "where we are" and "what just happened." The op's edge
    /// ([`ProgramOp::target`](specter_core::program::ProgramOp::target) on the outcome) decides:
    ///
    /// - [`BranchTarget::Terminate`] → propagate `outcome` to `EffectComplete` and return.
    /// - [`BranchTarget::Escape`] → terminate with [`EffectOutcome::Ok`] regardless of the carried
    ///   outcome (the "branch, not guard" outcome elision pinned by lowering).
    /// - [`BranchTarget::Continue`] → attempt to spawn the named op:
    ///   - **Ok**: the wait thread now drives the next reap; return.
    ///   - **Deferred** (permit cap): park in [`Slot::plan_continue`] and return.
    ///   - **Failed** (OS spawn / resolver / wait-thread failure): loop with a synthesised `Failed`
    ///     outcome at the new cursor.
    ///
    /// The loop is bounded: each [`BranchTarget::Continue`] edge points forward (builder invariant:
    /// `target > origin`) and within bounds (`target < ops.len()`), so the cursor strictly
    /// increases. A pathological program is impossible by construction.
    ///
    /// Called only by the Pump-policy reap pipeline ([`Self::reap_pump`]) and the spawn-failure
    /// paths in [`Self::start_plan`] / [`Self::spawn_continuation`]. The shutdown drain
    /// ([`Self::handle_reap_drop`]) bypasses dispatch entirely.
    ///
    /// Takes the [`EffectCompletion`] envelope by value so the terminate arms thread it through
    /// unchanged to [`Self::terminate_plan`]. The two synthesising arms (`Escape` and the `Continue
    /// → Failed` synth loop) mutate `completion.outcome` in place — the envelope's identity (`(sub,
    /// key)`) is preserved, only the carried outcome shifts.
    fn advance_or_terminate(
        &mut self,
        mut completion: EffectCompletion,
        effect: Arc<Effect>,
        diff_tmp: Option<Arc<DiffTmpFile>>,
        mut cursor: u32,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
        engine_in: &dyn EffectCompleteSender,
    ) {
        loop {
            let op = &effect.program.ops()[cursor as usize];
            match op.target(&completion.outcome) {
                BranchTarget::Terminate => {
                    self.terminate_plan(completion, ReapPolicy::Pump, engine_in);
                    return;
                }
                BranchTarget::Escape => {
                    // "Branch, not guard" outcome elision — the predicate's carried outcome is
                    // irrelevant; the plan terminates Ok. Re-stamp the envelope's outcome so the
                    // engine receives the synthesised verdict, not the raw reap outcome.
                    completion.outcome = EffectOutcome::Ok;
                    self.terminate_plan(completion, ReapPolicy::Pump, engine_in);
                    return;
                }
                BranchTarget::Continue(next_idx) => {
                    let next = next_idx.get();
                    // Forward-only-and-in-bounds is structurally enforced at builder patch time.
                    // Defensive assert here as a tripwire if a future variant addition bypasses the
                    // builder's edge validation.
                    debug_assert!(
                        next > cursor && (next as usize) < effect.program.ops().len(),
                        "forward-only + in-bounds (builder invariant)",
                    );
                    match self.try_spawn_step(
                        &completion.key,
                        completion.sub,
                        &effect,
                        next,
                        diff_tmp.as_ref(),
                        spawner,
                        reap_tx,
                    ) {
                        Ok(()) => return,
                        Err(SpawnError::Deferred) => {
                            // `diff_tmp` moves into PlanContinuation — the Arc keeps the file alive
                            // across the permit wait, and the next step's spawn clones it back out.
                            self.queue_plan_continue(
                                completion.key,
                                PlanContinuation {
                                    effect,
                                    cursor: next,
                                    diff_tmp,
                                },
                            );
                            return;
                        }
                        Err(SpawnError::Failed(cause)) => {
                            // Synthesise Failed for `next` and loop. The next iteration reads
                            // `next`'s `on_failed` edge — for a predicate-Failed synth this walks
                            // to the else-branch (Continue) or to Escape (no-else); for an
                            // Exec/Pipe synth it walks to Terminate (stop-on-failure propagation).
                            //
                            // Log at warn with the typed cause so the operator can correlate this
                            // dispatch decision against the cause-side error log line emitted at
                            // the spawn boundary (resolver / OS spawn / wait thread).
                            tracing::warn!(
                                key = ?completion.key,
                                cursor = next,
                                ?cause,
                                "synthesised EffectOutcome::Failed (no clean exit); dispatching on op's on_failed edge",
                            );
                            cursor = next;
                            completion.outcome = EffectOutcome::Failed(Termination::Internal);
                        }
                    }
                }
            }
        }
    }

    /// Park a plan's next instruction into [`Slot::plan_continue`] and queue the slot for the next
    /// pump cycle. Called from the advance branch when no permit was available at reap time.
    fn queue_plan_continue(&mut self, key: DedupKey, cont: PlanContinuation) {
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.plan_continue = Some(cont);
            if !slot.in_ready_queue {
                slot.in_ready_queue = true;
                self.ready_queue.push_back(key);
            }
        }
    }

    /// Terminal arm of a *live* plan: emit one `EffectComplete`, remove the per-Sub gate hold, and
    /// either re-queue pending (Pump policy + non-empty pending) or remove the slot.
    ///
    /// Called from three sites, every one of which has a paired [`Self::start_plan`] bump for the
    /// same `(sub, key)`:
    ///
    /// 1. [`Self::advance_or_terminate`] Terminate / Escape arms (the canonical live-plan exit
    ///    after a real Reaped or a synth Failed).
    /// 2. [`Self::handle_reap_drop`] shutdown-drain teardown when [`Self::take_running`] returns a
    ///    live job.
    /// 3. [`Self::start_plan`] / [`Self::spawn_continuation`] spawn- failure paths route through
    ///    [`Self::advance_or_terminate`] (which hits #1 above).
    ///
    /// **Tmp-file cleanup is NOT this function's responsibility.** The plan's `Arc<DiffTmpFile>`
    /// (if any) lives on the caller's stack for the canonical exit (the local in
    /// [`Self::advance_or_terminate`] / [`Self::handle_reap_drop`]) and drops on the caller's scope
    /// exit. The file is unlinked by [`crate::tmp::DiffTmpFile::drop`] when the last `Arc` co-owner
    /// is dropped — see the [`RunningJob::diff_tmp`] doc for the lifecycle.
    ///
    /// **Stale-Reaped arms do NOT call here**: a Reaped without a paired live `RunningJob` (slot
    /// absent or `slot.running` already taken) routes through [`Self::terminate_stale`]. That path's
    /// `sub` is *not* guaranteed to hold the per-Sub gate (a live plan for the same Sub may own it on
    /// a different key), so a blanket `running_subs.remove(&sub)` would silently clobber the live
    /// plan's hold. Splitting "live plan teardown" from "engine accounting only" lets the
    /// `running_subs.remove` here be total — guarded by an unconditional `debug_assert!(was_present)`
    /// — pinning the bump/remove pairing under the set-membership shape.
    fn terminate_plan(
        &mut self,
        completion: EffectCompletion,
        policy: ReapPolicy,
        engine_in: &dyn EffectCompleteSender,
    ) {
        // Snapshot the `Copy` identity scalars before the envelope moves into the wire `send`. Both
        // fields stay on the stack for the subsequent per-Sub gate remove + slot decision.
        let key = completion.key;
        let sub = completion.sub;
        // `let _` deliberately drops the `SendError`: a closed `engine_in` means the engine has
        // been torn down. The actuator's `run` loop observes that fact via one of its own exits
        // (`effects_rx` Disconnected, `shutdown_actuator_rx`, `hard_shutdown_actuator_rx`) and
        // breaks out — the swallow keeps per-step accounting consistent until then. Mirrors the
        // engine driver's `effects_tx` degradable-on-disconnect policy in `specter-bin`'s
        // `forward.rs`: both directions of the engine ↔ actuator channel degrade rather than
        // escalate, and the actuator cannot persist with the engine dead.
        let _ = engine_in.send(completion);
        // Live-plan teardown: every reachable call site has a paired `start_plan` insert for this
        // Sub. `BTreeSet::remove` returns `false` only on an unpaired remove — a controller
        // accounting bug, tripwired here. The set-membership shape makes the underflow that a `*c
        // -= 1` admitted unrepresentable.
        let was_present = self.running_subs.remove(&sub);
        debug_assert!(
            was_present,
            "per-Sub gate underflow: terminate_plan for Sub not in running_subs (stale completion path must route to terminate_stale)",
        );
        // The slot may still exist (the reap pipeline already took running; spawn-failure paths
        // never installed it). Decide: re-queue if pending under Pump, otherwise remove.
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

    /// Engine-accounting-only terminal arm for an [`EffectCompletion`] envelope that arrives
    /// without a paired live `RunningJob` ([`Self::take_running`] returned `None` — slot absent or
    /// `slot.running` already taken). Emits one `EffectComplete` so the engine's outstanding
    /// counter stays in lockstep, then removes the slot (idempotent).
    ///
    /// Distinct from [`Self::terminate_plan`] in two ways:
    ///
    /// 1. **No per-Sub gate touch.** A stale completion's `sub` is not guaranteed to hold a gate
    ///    entry. The Sub might own a live plan at a *different* `DedupKey` (the per-Sub gate is
    ///    structurally at-most-one across all keys, but the gate's bump is paired to a specific
    ///    `(sub, key)` plan — not to `sub` alone). A blanket `running_subs.remove(&sub)` here would
    ///    silently clobber that live plan's hold, releasing the gate ahead of its real terminus.
    /// 2. **No diff tmp cleanup.** Same reasoning: the live plan owns the only `Arc<DiffTmpFile>`
    ///    chain (via its `Slot::running` or `Slot::plan_continue`); the stale arm holds no Arc to
    ///    drop, so unlink can't happen here — exactly what's needed to avoid pulling the file out
    ///    from under a live reader.
    ///
    /// **Production reachability.** Defensive: every successful spawn installs `slot.running` and
    /// every wait-thread send is paired with a `take_running` at the controller. The path fires
    /// only against a manufactured `Slot::default` (no running) — see
    /// `reap_pump_stale_for_unspawned_slot_clears_state` in the test module. Production callers
    /// (real reap pipeline, [`Self::handle_reap_drop`]) reach this arm only if the same completion
    /// would otherwise have been delivered twice — a controller invariant violation. The accounting
    /// emit + slot remove keeps the engine in lockstep under that defensive case.
    ///
    /// In production the slot is `Slot::default()` (or absent) — never carries pending — so the
    /// unconditional `slots.remove` here drops no live state. `pending` becomes reachable only via
    /// `handle_submit`, which is on the same single controller thread; the stale-completion path is
    /// a no-op against that arm.
    fn terminate_stale(
        &mut self,
        completion: EffectCompletion,
        engine_in: &dyn EffectCompleteSender,
    ) {
        // `key` is `Copy`; snapshot before the envelope moves into the wire send so the slot-remove
        // below stays addressable.
        let key = completion.key;
        // Same drop-on-disconnect policy as [`Self::terminate_plan`] — see that site's rustdoc for
        // the cross-actor symmetry rationale.
        let _ = engine_in.send(completion);
        self.slots.remove(&key);
    }

    /// Spawn ready slots while permits + per-Sub gates allow.
    ///
    /// Two arms per slot:
    ///
    /// - **Plan-continue** (`slot.plan_continue.is_some()`): the slot holds an in-flight plan's
    ///   next step, deferred at reap time on permit unavailability. Bypasses the per-Sub gate
    ///   (continuation of an admitted plan; never racing another plan for the Sub by construction).
    ///   Permit gate still applies.
    /// - **Fresh plan** (`slot.pending.is_some()`, `plan_continue` empty): per-Sub gate, then
    ///   permit gate, then [`Self::start_plan`].
    ///
    /// Items blocked by either gate are deferred to a transient buffer and restored at end so FIFO
    /// is preserved across pump invocations. The blocked-buffer logic is per-arm: a permit-blocked
    /// plan-continue short-circuits the loop the same way a permit-blocked fresh plan does, since
    /// both contend for the same global semaphore.
    pub fn pump(
        &mut self,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
        engine_in: &dyn EffectCompleteSender,
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
            if self.running_subs.contains(&sub) {
                self.blocked_scratch.push_back(key);
                continue;
            }
            // Global gate.
            let Some(permit) = self.permits.try_acquire() else {
                // No more permits this round; defer this and the remaining queued items (FIFO
                // preserved).
                self.blocked_scratch.push_back(key);
                while let Some(k) = self.ready_queue.pop_front() {
                    self.blocked_scratch.push_back(k);
                }
                break;
            };
            slot.in_ready_queue = false;
            let effect = slot.pending.take().expect(
                "popped slot carries pending (plan_continue branch handled the alternative)",
            );
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
        // Drain (don't consume) so the deque retains its capacity for the next pump. The flag is
        // already true (we set it when we pushed and only cleared it on successful spawn). Re-stamp
        // it to keep "in queue ⇔ flag set" sealed across future refactors.
        while let Some(k) = self.blocked_scratch.pop_front() {
            let slot = self
                .slots
                .get_mut(&k)
                .expect("deferred slot is present (only the active key can be removed mid-pump)");
            slot.in_ready_queue = true;
            self.ready_queue.push_back(k);
        }
    }

    /// Start a plan: materialise the diff tmp file (if needed), insert the Sub into the per-Sub
    /// gate, spawn instruction 0 with the given permit.
    ///
    /// **The per-Sub gate is inserted unconditionally** before the spawn attempt — predicate
    /// spawn-failure semantics may continue the plan via [`Self::advance_or_terminate`], and any
    /// in-progress continuation needs the per-Sub gate to hold same-Sub fresh plans behind it. On
    /// failure, the dispatch loop's terminate arms remove the Sub via [`Self::terminate_plan`]; the
    /// controller is single-threaded so the insert-then-remove is atomic from any observer's
    /// perspective.
    ///
    /// On spawn failure, routes through [`Self::advance_or_terminate`] with a synthesised
    /// `EffectOutcome::Failed`. The dispatcher reads op 0's `on_failed` edge — propagates the synth
    /// Failed when the edge is `Terminate` (Exec/Pipe stop-on-failure), jumps to the named slot
    /// when the edge is `Continue` (a predicate's else-branch), or terminates Ok when the edge is
    /// `Escape` (a predicate with no else).
    ///
    /// `effect` is taken by value so the caller (pump) hands off the freshly-constructed
    /// `Arc<Effect>` and forgets about it; on success the Arc is cloned into [`Slot::running`], on
    /// failure it drops or moves into the advance loop. Passing by reference would force pump to
    /// keep the Arc alive past the call for no reason.
    #[allow(clippy::needless_pass_by_value)]
    fn start_plan(
        &mut self,
        key: DedupKey,
        sub: SubId,
        effect: Arc<Effect>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
        engine_in: &dyn EffectCompleteSender,
    ) {
        // Materialise the diff tmp file before the first instruction's spawn so the resolver can
        // slot SPECTER_DIFF_PATH into its alphabetical position. Best-effort: on write failure
        // proceed with `None`, the resolver omits the env var. The handle is shared across every
        // step of the plan via `Arc<DiffTmpFile>`; `DiffTmpFile::drop` unlinks the file when the
        // last co-owner is dropped at plan terminus.
        let diff_tmp: Option<Arc<DiffTmpFile>> = effect.diff().and_then(|diff| {
            match DiffTmpFile::create(&self.temp_dir, self.actuator_pid, effect.correlation, diff) {
                Ok(handle) => Some(Arc::new(handle)),
                Err(e) => {
                    tracing::warn!(
                        correlation = ?effect.correlation,
                        ?e,
                        "tmp diff write failed; proceeding without SPECTER_DIFF_PATH",
                    );
                    None
                }
            }
        });

        // Per-Sub gate insert symmetric with `terminate_plan`'s remove. The pump's gate
        // (`running_subs.contains` in the fresh-plan arm) blocks any other start_plan for this Sub
        // until the live plan terminates, so the insert is structurally first-of-its-kind.
        // `BTreeSet::insert` returning `false` would mean the pump dispatched a fresh plan past its
        // own gate — a controller bug, tripwired here.
        let inserted = self.running_subs.insert(sub);
        debug_assert!(
            inserted,
            "per-Sub gate violation: start_plan for Sub already in running_subs",
        );
        match self.spawn_step_with_permit(
            &key,
            sub,
            &effect,
            0,
            diff_tmp.as_ref(),
            permit,
            spawner,
            reap_tx,
        ) {
            Ok(()) => {}
            Err(cause) => {
                // OS spawn / resolver / wait-thread failure at the first instruction. Hand off to
                // the dispatch loop with synthesised Failed — a predicate at cursor 0 still jumps
                // to its else-branch via this path. The typed `cause` discriminant accompanies the
                // synth in the operator log so triage can correlate this decision with the
                // cause-side error line above.
                tracing::warn!(
                    ?key,
                    cursor = 0,
                    ?cause,
                    "synthesised EffectOutcome::Failed at plan start; dispatching on op 0's on_failed edge",
                );
                self.advance_or_terminate(
                    EffectCompletion {
                        sub,
                        key,
                        outcome: EffectOutcome::Failed(Termination::Internal),
                    },
                    effect,
                    diff_tmp,
                    0,
                    spawner,
                    reap_tx,
                    engine_in,
                );
            }
        }
    }

    /// Spawn the next instruction of a plan that was deferred via [`Slot::plan_continue`]. Distinct
    /// from [`Self::start_plan`]: no per-Sub gate insert (already held since the original
    /// `start_plan`), no tmp materialisation (path inherited from the `PlanContinuation`).
    ///
    /// On spawn failure, routes through [`Self::advance_or_terminate`] — predicate spawn-failure at
    /// the continuation's cursor jumps to its else-branch, exec/pipe spawn-failure propagates to
    /// plan terminus. Either way the per-Sub gate (held since the original `start_plan`) is
    /// released at terminate, and any subsequent advance reuses the inherited tmp path.
    fn spawn_continuation(
        &mut self,
        key: DedupKey,
        sub: SubId,
        cont: PlanContinuation,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
        engine_in: &dyn EffectCompleteSender,
    ) {
        let PlanContinuation {
            effect,
            cursor,
            diff_tmp,
        } = cont;
        match self.spawn_step_with_permit(
            &key,
            sub,
            &effect,
            cursor,
            diff_tmp.as_ref(),
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
                    EffectCompletion {
                        sub,
                        key,
                        outcome: EffectOutcome::Failed(Termination::Internal),
                    },
                    effect,
                    diff_tmp,
                    cursor,
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
    /// - `Ok(())` — instruction is in flight (slot.running installed, wait thread alive).
    /// - `Err(SpawnError::Deferred)` — permit semaphore was at capacity; caller defers via
    ///   [`Slot::plan_continue`].
    /// - `Err(SpawnError::Failed(cause))` — OS-level spawn, resolver, or wait-thread startup
    ///   failed; caller terminates the plan with synthesised `EffectOutcome::Failed` and logs
    ///   `cause` at the synth site.
    ///
    /// The Deferred branch returns before consuming any of the borrowed inputs — caller-owned
    /// values stay live for the `PlanContinuation` hand-off. `SpawnFailureCause` is lifted into the
    /// wider [`SpawnError::Failed`] variant via `map_err` — the inner
    /// [`Self::spawn_step_with_permit`] cannot defer (its permit is already acquired), so its
    /// return type is the tighter `Result<(), SpawnFailureCause>`.
    fn try_spawn_step(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        diff_tmp: Option<&Arc<DiffTmpFile>>,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
    ) -> Result<(), SpawnError> {
        let Some(permit) = self.permits.try_acquire() else {
            return Err(SpawnError::Deferred);
        };
        self.spawn_step_with_permit(key, sub, effect, cursor, diff_tmp, permit, spawner, reap_tx)
            .map_err(SpawnError::Failed)
    }

    /// Spawn one op of a plan with a pre-acquired permit. Installs [`Slot::running`] on success.
    ///
    /// Dispatches on the op's [`SpawnBody`] at `cursor`:
    ///
    /// - [`SpawnBody::Exec`] → [`Self::spawn_exec_with_permit`]: one resolver call, one
    ///   [`Spawner::spawn`], one [`RunningJob`] installed, one wait thread, one optional timer
    ///   thread.
    /// - [`SpawnBody::Pipe`] → [`Self::spawn_pipe_with_permit`]: N resolver calls, one
    ///   [`Spawner::spawn_pipe`], one [`RunningJob`] (with combined signaler for shutdown fan-out),
    ///   one aggregating wait thread, and per-stage timer threads for stages with a `timeout`.
    ///
    /// At the IR level there is no predicate distinction — predicate behavior is the op's
    /// `on_failed` edge, read by the reap-path.
    ///
    /// `now: SystemTime` is sampled at the dispatcher and threaded into every resolver call so a
    /// single pipe sees one shared `${specter.time}` across all stages — the documented contract
    /// pins "the wall-clock instant immediately before the kernel runs the user's command," which
    /// for a pipe is the instant all stages start.
    fn spawn_step_with_permit(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        diff_tmp: Option<&Arc<DiffTmpFile>>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
    ) -> Result<(), SpawnFailureCause> {
        let now = std::time::SystemTime::now();
        let cwd: &Path = resolve::compute_cwd(effect);
        let capture_output = effect.capture_output;

        let op = &effect.program.ops()[cursor as usize];
        match op.body() {
            SpawnBody::Exec(exec) => self.spawn_exec_with_permit(
                key,
                sub,
                effect,
                cursor,
                exec,
                now,
                cwd,
                capture_output,
                diff_tmp,
                permit,
                spawner,
                reap_tx,
            ),
            SpawnBody::Pipe(stages) => {
                // `MultiStage::stages()` borrows the shared stage slice; its Arc lifetime is tied
                // to `effect`, so the slice outlives the resolve/spawn_pipe sequence. The slice is
                // ≥2 by construction — `MultiStage::new` is the sole producer of `SpawnBody::Pipe`
                // and rejects fewer — so the pipe path's stdout→stdin / pipefail assumptions hold
                // with no runtime arity check on this path.
                let stages_slice: &[ExecAction] = stages.stages();
                self.spawn_pipe_with_permit(
                    key,
                    sub,
                    effect,
                    cursor,
                    stages_slice,
                    now,
                    cwd,
                    capture_output,
                    diff_tmp,
                    permit,
                    spawner,
                    reap_tx,
                )
            }
        }
    }

    /// Single-process spawn path for [`SpawnBody::Exec`]. Outcome routing (propagate / branch /
    /// no-op) lives on the op's edges; this function is shape-only.
    ///
    /// Sequencing pinned: slot.running is installed **before** the wait thread is spawned, so a
    /// fast-completing wait thread (mock under test, or a child that exits between fork and wait)
    /// can't send `Reaped` before the controller knows about it.
    ///
    /// On wait-thread spawn failure: the freshly-spawned child is alive but has no waiter (the
    /// closure that owned it has been dropped by `Builder::spawn`'s `Err` path). The recovery
    /// branch SIGKILLs the orphan via the signaler held in `slot.running`, then synchronously reaps
    /// it via [`crate::spawner::ChildSignaler::reap_blocking`] so the OS doesn't leak a zombie.
    /// `slot.running` is then cleared (the terminate_plan caller expects it to be `None`) and
    /// `SpawnError::Failed` returns.
    ///
    /// **Slot invariant.** All `self.slots.get_mut(key)` lookups in this function assume the slot
    /// was just touched by the caller (the controller is single-threaded; no Reap or Submit can
    /// interleave between caller's `pump` / `reap_pump` and here). A missing slot is a programming
    /// error, surfaced via `expect` rather than silently masked — silent masking would otherwise
    /// leak the signaler and leave the child unreachable from shutdown signaling.
    fn spawn_exec_with_permit(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        exec: &ExecAction,
        now: std::time::SystemTime,
        cwd: &Path,
        capture_output: bool,
        diff_tmp: Option<&Arc<DiffTmpFile>>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
    ) -> Result<(), SpawnFailureCause> {
        let diff_path: Option<&Path> = diff_tmp.map(|h| h.path());
        let (CommandResolved { argv }, env) =
            match resolve::resolve_step(effect, exec, now, diff_path, &self.env_snapshot) {
                Ok(resolved) => resolved,
                Err(e) => {
                    // Strict `${env.<NAME>}` failure: no spawn, no wait thread, no timer. Permit
                    // drops at the end of this scope; caller routes through `advance_or_terminate`
                    // with synthesised `EffectOutcome::Failed`.
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

        // The signaler Arc has up to three co-owners: the installed [`RunningJob::signaler`]
        // (consumed by shutdown / recovery), the optional per-step timer thread (drops on natural
        // completion via the `is_dead` short-circuit), and the wait thread (publishes the
        // dead-ratchet backstop after `catch_unwind`). Clone the two extras here, then move the
        // original into `RunningJob` — keeps the controller's install-side reference live
        // regardless of whether either sibling spawn succeeds. The timer clone is conditional (only
        // paid when `exec.timeout().is_some()`) and pairs the cloned Arc with its deadline so the
        // two can't drift apart.
        let timer_arm: Option<(Duration, Arc<dyn ChildSignaler>)> = exec
            .timeout()
            .map(|deadline| (deadline, Arc::clone(&signaler)));
        let signaler_for_wait = Arc::clone(&signaler);
        let slot = self
            .slots
            .get_mut(key)
            .expect("slot present at install (single-threaded controller just dispatched here)");
        slot.running = Some(RunningJob {
            pid,
            signaler,
            effect: Arc::clone(effect),
            cursor,
            diff_tmp: diff_tmp.map(Arc::clone),
        });

        self.spawn_wait_thread_after_install(
            key,
            sub,
            pid,
            cursor,
            waiter,
            signaler_for_wait,
            permit,
            reap_tx,
        )?;

        // Per-step timer: arm AFTER the wait thread is alive so the wait thread's `dead` flag is
        // the natural-completion signal the timer short-circuits on. Best-effort + log-and-proceed
        // policy lives inside [`timer::arm_timer`].
        if let Some((deadline, timer_signaler)) = timer_arm {
            timer::arm_timer(
                deadline,
                self.shutdown_grace,
                timer_signaler,
                timer::TimerContext::Exec {
                    key: *key,
                    cursor,
                    pid,
                },
            );
        }

        tracing::debug!(?key, cursor, pid, "spawned instruction");
        Ok(())
    }

    /// Multi-stage spawn path for [`SpawnBody::Pipe`].
    ///
    /// `stages` is ≥2 by construction: `MultiStage::new` is the sole producer of
    /// [`SpawnBody::Pipe`] and rejects fewer, and the lone caller passes `MultiStage::stages()`.
    /// The stdout→stdin wiring and pipefail aggregation below assume that arity — it is sealed at
    /// the type's constructor, not re-checked on this path.
    ///
    /// The shape mirrors [`Self::spawn_exec_with_permit`] at every step, scaled to N stages:
    ///
    /// 1. Resolve every stage's argv + env against the shared `now` (so `${specter.time}` agrees
    ///    across stages — see [`Spawner::spawn_pipe`] for the contract).
    /// 2. Call [`Spawner::spawn_pipe`] which mints N processes, an aggregating
    ///    [`crate::spawner::ChildWaiter`], a combined [`crate::spawner::ChildSignaler`] for
    ///    shutdown fan-out, and per-stage signalers for per-stage timer threads.
    /// 3. Install [`RunningJob`] BEFORE spawning the wait thread (slot.running invariant: the wait
    ///    thread must not be able to send `Reaped` before the controller has the job in hand). The
    ///    job carries the combined signaler only — the per-stage signalers are locals to this
    ///    function, cloned into any per-stage timer thread and dropped on return.
    /// 4. Spawn one wait thread that drains the aggregating waiter and surfaces a single `Reaped`
    ///    event to the controller — the engine's accounting is "one EffectComplete per Effect" and
    ///    that holds regardless of pipe vs single-process.
    /// 5. For each stage with a `timeout`, spawn one detached timer thread that observes the
    ///    stage's `dead` flag and signals the stage individually. The aggregating waiter sees the
    ///    resulting per-stage Failed and cascades SIGTERM to alive siblings; the engine receives
    ///    one aggregated Failed outcome.
    ///
    /// Resolver failure on **any** stage aborts the entire pipe (no processes have spawned yet at
    /// that point). Pipe-spawn failure (returned by [`Spawner::spawn_pipe`]) means the spawner has
    /// already rolled back any partially-spawned stages — the caller just returns
    /// `SpawnError::Failed`.
    fn spawn_pipe_with_permit(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        effect: &Arc<Effect>,
        cursor: u32,
        stages: &[ExecAction],
        now: std::time::SystemTime,
        cwd: &Path,
        capture_output: bool,
        diff_tmp: Option<&Arc<DiffTmpFile>>,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<EffectCompletion>,
    ) -> Result<(), SpawnFailureCause> {
        let diff_path: Option<&Path> = diff_tmp.map(|h| h.path());
        // Resolve every stage's argv + env. The result tuples own the argv `Vec<String>` and the
        // env `Vec<EnvVar<'_>>`; the env's `Cow::Borrowed` slots borrow from `effect`, the
        // resolver's owned per-stage `parent_str` / `time_str` (moved into the env Cow::Owned
        // slots), and `diff_path` (if present). All borrowed sources outlive this function's body,
        // so the resolved Vec is stable across the `spawn_pipe` call.
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
                // Partial-spawn rollback already happened inside `spawn_pipe` (every prior stage
                // SIGKILLed + reaped, every pipe fd closed in the parent).
                tracing::error!(?key, cursor, ?cwd, ?e, "pipe spawn failed");
                drop(permit);
                return Err(SpawnFailureCause::OsSpawn);
            }
        };
        // We're done with `stage_specs` and `resolved` — let them drop here so per-stage env Vecs /
        // argvs aren't kept alive past the spawn call. (They don't carry per-process state; the
        // spawner has dup'd the argv/env into the children.)
        drop(stage_specs);
        drop(resolved);

        let crate::spawner::PipeSpawnHandles {
            last_pid,
            waiter,
            combined_signaler,
            stage_signalers,
        } = handles;

        // Clone the combined signaler for the wait-thread's dead-ratchet backstop before moving the
        // original into `RunningJob`. `CombinedSignaler::mark_dead` fans out to every per-stage
        // flag, so this single Arc covers the whole pipe at the outer `wait_loop` layer (per-stage
        // closures inside `PipeWaiter::wait` additionally backstop their own `catch_unwind` sites
        // with the per-stage signalers).
        let signaler_for_wait = Arc::clone(&combined_signaler);
        let slot = self
            .slots
            .get_mut(key)
            .expect("slot present at install (single-threaded controller just dispatched here)");
        slot.running = Some(RunningJob {
            pid: last_pid,
            signaler: combined_signaler,
            effect: Arc::clone(effect),
            cursor,
            diff_tmp: diff_tmp.map(Arc::clone),
        });

        self.spawn_wait_thread_after_install(
            key,
            sub,
            last_pid,
            cursor,
            waiter,
            signaler_for_wait,
            permit,
            reap_tx,
        )?;

        // Per-stage timers: arm AFTER the wait thread is alive so the wait thread's per-stage
        // `dead` flags are the natural- completion signal each timer short-circuits on. The wait-
        // thread `?` above also ensures we don't arm timers for a pipe with no waiter (would leave
        // undrained orphans).
        //
        // Best-effort + log-and-proceed policy lives inside [`timer::arm_timer`]. The per-stage
        // `Arc::clone(sig)` is paid only on stages that carry a timeout — most pipes don't.
        // `stage_signalers` (the install-time local `Arc` handles) drop at function exit; the
        // aggregating waiter and any armed timer threads keep the per-stage signalers alive through
        // reap.
        for (idx, (stage, sig)) in stages.iter().zip(stage_signalers.iter()).enumerate() {
            if let Some(deadline) = stage.timeout() {
                let stage = u32::try_from(idx)
                    .expect("pipe stage count is bounded by spawn arity (≤ u32::MAX)");
                timer::arm_timer(
                    deadline,
                    self.shutdown_grace,
                    Arc::clone(sig),
                    timer::TimerContext::PipeStage {
                        key: *key,
                        cursor,
                        stage,
                        pid: last_pid,
                    },
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

    /// Spawn the wait thread for an already-installed [`Slot::running`]. On
    /// `thread::Builder::spawn` failure, take the running job back via the slot, recover the orphan
    /// child (SIGKILL + `reap_blocking`), and return [`SpawnFailureCause::WaitThread`] so the
    /// caller routes through `advance_or_terminate` with a synthesised `EffectOutcome::Failed`.
    ///
    /// `pid` is used only for the wait-thread's OS name (`act-wait-{pid}`); for single-process
    /// steps it's the spawned child's pid; for pipes it's the last stage's pid (the operator-facing
    /// "the pid of this pipe"). The `key` is needed to look up the slot in the recovery branch.
    ///
    /// `signaler` is a clone of the same [`Arc`] the controller installed on
    /// [`RunningJob::signaler`]. The wait thread owns it solely to publish the dead-ratchet
    /// backstop after the `catch_unwind` of [`ChildWaiter::wait`] — see [`wait_loop`] for the
    /// protocol contract. Cloning at the call site (rather than at the
    /// [`SpawnHandles`](crate::SpawnHandles) origin) keeps the controller's install-side reference
    /// live regardless of whether the spawn-the-wait-thread step succeeds; on `Builder::spawn`
    /// failure the closure-owned clone drops, the recovery branch drives the install-side clone
    /// through SIGKILL + `reap_blocking`, and nothing leaks.
    ///
    /// The function is `&mut self` because the recovery branch mutates `self.slots[key].running`.
    /// The slot lookup is an `expect` for the same reason as `spawn_exec_with_permit`: the
    /// controller is single-threaded and the caller has just installed `slot.running`.
    #[allow(clippy::needless_pass_by_value)]
    fn spawn_wait_thread_after_install(
        &mut self,
        key: &DedupKey,
        sub: SubId,
        pid: u32,
        cursor: u32,
        waiter: Box<dyn ChildWaiter>,
        signaler: Arc<dyn ChildSignaler>,
        permit: Permit,
        reap_tx: &Sender<EffectCompletion>,
    ) -> Result<(), SpawnFailureCause> {
        let reap_tx_for_thread = reap_tx.clone();
        let wait_key = *key;
        if let Err(e) = std::thread::Builder::new()
            // Linux pthread_setname_np truncates to 15 chars + null; `act-wait-` is 9 chars,
            // leaving room for a 6-digit pid unscathed. macOS allows 64 bytes.
            .name(format!("act-wait-{pid}"))
            .spawn(move || {
                wait_loop(waiter, signaler, wait_key, sub, permit, reap_tx_for_thread);
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

/// Recovery path for the wait-thread-spawn-failure case in [`ActuatorState::spawn_step_with_permit`].
/// The child is alive but its paired [`crate::spawner::ChildWaiter`] was dropped along with the
/// failed `Builder::spawn` closure — so the controller must SIGKILL it and synchronously reap it
/// through the signaler. Without the reap the OS would leak a zombie until process exit.
///
/// Both syscalls are best-effort: errors are logged and swallowed. The caller's synthesised
/// `EffectOutcome::Failed` is what the engine observes; this function exists only for OS resource
/// hygiene.
///
/// Extracted as a free function so it can be unit-tested in isolation without standing up the full
/// spawn flow (the actual `thread::Builder::spawn` failure path is rare and not directly injectable
/// in tests).
///
/// `job` is taken by value to express the ownership transfer: the caller has just
/// `slot.running.take()`-ed and hands the in-flight bookkeeping over for tear-down; once we return,
/// the signaler / effect Arc / diff-tmp path all drop. Borrowing would force the caller into a
/// take-then-restore dance for no behavioural gain.
///
/// Same recovery shape for Exec and Pipe: the [`RunningJob::signaler`] is either the single-child
/// signaler or the combined fan-out signaler, and both implement SIGKILL + `reap_blocking`
/// correctly for their underlying children.
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

/// Wait-thread body. Block on `waiter.wait()`; on return, publish the dead-ratchet backstop,
/// release the permit, and send a `Reaped` to the controller.
///
/// Three orderings are load-bearing:
///
/// 1. The waiter sets `dead = true` (production impl) before returning, so a controller signal
///    racing this thread observes `dead = true` and short-circuits — preventing a stale signal
///    against a reaped (and possibly pid-reused) child. This is the canonical *early* publish.
///
/// 2. [`ChildSignaler::mark_dead`] runs after the `catch_unwind` *unconditionally*, regardless of
///    the wait outcome. This is the additive wrapper-layer backstop: a panicking `wait` impl never
///    reaches its own publish, and even a clean `Err` return can leave a custom impl that publishes
///    only on the happy path racing PID-reuse. Calling `mark_dead` here closes the window at the
///    protocol layer; idempotent against the waiter's own early publish on the happy path. For the
///    pipe shape, the wrapping signaler is the [`crate::pipe::CombinedSignaler`] and the call fans
///    out to every per-stage flag so the outer backstop closes every stage's window even when
///    [`crate::pipe::PipeWaiter::wait`] aggregated correctly; the per-stage closures inside
///    `PipeWaiter` additionally backstop their own catch_unwind sites.
///
/// 3. Permit release precedes reap notification. Spawns for *other* Subs can dispatch immediately on
///    the freed permit even if the reap channel is briefly saturated. Spawns for the *same* Sub still
///    wait for the controller to drop `sub` from `running_subs` when it processes the `Reaped` — by
///    design (per-Sub serialization). The brief stale-membership window between `drop(permit)` and
///    `handle_reap` is benign: same-Sub items defer one extra pump cycle, no over-spawning.
///
/// **Tmp-file cleanup is NOT this thread's responsibility.** The diff tmp file lives for the whole
/// plan (multiple steps may read it) — the wait thread carries no `Arc<DiffTmpFile>` co-owner. The
/// controller's `Slot::running` / `Slot::plan_continue` are the canonical co-owners; on plan
/// terminus the last Arc drops at [`ActuatorState::advance_or_terminate`] /
/// [`ActuatorState::handle_reap_drop`] function exit and [`crate::tmp::DiffTmpFile::drop`] unlinks.
/// A wait-thread panic-unwind caught here cannot trigger an early unlink because no Arc lives on
/// this side of the channel.
#[allow(clippy::needless_pass_by_value)] // closure-spawned: arguments owned for the thread
fn wait_loop(
    waiter: Box<dyn ChildWaiter>,
    signaler: Arc<dyn ChildSignaler>,
    key: DedupKey,
    sub: SubId,
    permit: Permit,
    reap_tx: Sender<EffectCompletion>,
) {
    let outcome = match std::panic::catch_unwind(AssertUnwindSafe(|| waiter.wait())) {
        Ok(Ok(o)) => o,
        Ok(Err(e)) => {
            tracing::warn!(?key, ?e, "wait failed");
            EffectOutcome::Failed(Termination::Internal)
        }
        Err(_) => {
            tracing::error!(?key, "wait panicked");
            EffectOutcome::Failed(Termination::Internal)
        }
    };
    // Backstop publish: idempotent against the waiter's own happy-path mark. Closes the PID-reuse
    // window even when the impl panicked before reaching its own write site. See comment 2 above
    // for the full ordering contract.
    signaler.mark_dead();
    drop(permit);
    if let Err(e) = reap_tx.send(EffectCompletion { sub, key, outcome }) {
        // The controller has shut down ahead of us (post-shutdown orphan: shutdown's drain closed
        // `reap_rx`; the wait thread is the only `Reaped` writer). The reap is no longer
        // interesting — emit at `debug!` so a future refactor that splits the controller /
        // reap-channel lifetime is observable in operator logs instead of silently swallowed.
        tracing::debug!(?key, ?e, "reap_tx send after controller exit");
    }
}

#[inline]
pub(crate) const fn sub_of_key(key: &DedupKey) -> SubId {
    match *key {
        DedupKey::PerFile { sub, .. } | DedupKey::Subtree { sub, .. } => sub,
    }
}

#[cfg(test)]
mod tests {
    //! Direct tests for [`ActuatorState::reap_pump`] and [`ActuatorState::handle_reap_drop`] — the
    //! teardown entries that both the success and failure spawn paths route through. The
    //! synth-Reap-equivalent paths (spawn-failure inline and wait-thread-spawn-failure inline) are
    //! exercised here against pre-loaded state, since neither has a fault-injection seam in the
    //! controller harness.
    use super::super::SHUTDOWN_GRACE;
    use super::{ActuatorState, PlanContinuation, RunningJob, Slot};
    use crate::env::EnvSnapshot;
    use crate::spawner::{ChildSignaler, ChildWaiter, EnvVar, SpawnHandles, Spawner};
    use crate::{EffectCompleteSender, SendError};
    use compact_str::CompactString;
    use crossbeam::channel::{Sender, unbounded};
    use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, CorrelationId, DedupKey, Diff, Effect, EffectCommon,
        EffectCompletion, EffectOutcome, ExecAction, Input, ProfileId, ResourceId, ResourceKind,
        SubId, Termination,
    };
    use std::io;
    use std::num::NonZeroUsize;
    use std::path::{Path, PathBuf};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    const fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test setup: n must be non-zero")
    }

    /// Test adapter that lifts an [`EffectCompletion`] envelope into the engine-side
    /// `Input::EffectComplete` so the test's `Receiver<Input>` continues to observe completions in
    /// the engine's vocabulary. Mirrors the bin's [`WakingEffectCompleteSender`] without dragging
    /// in the bin's transport identity.
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

    /// Construct an [`ActuatorState`] with an empty env snapshot and the shared `SHUTDOWN_GRACE`.
    /// The tests in this module exercise state- machine transitions, not env resolution or timeout
    /// enforcement, so a single empty snapshot covers every call site. Env-aware tests live in the
    /// higher-level pool harness and inject snapshots explicitly via
    /// [`super::SubprocessActuator::new_with_grace_and_env`].
    fn test_state(concurrency: NonZeroUsize) -> ActuatorState {
        ActuatorState::new(
            concurrency,
            Arc::new(EnvSnapshot::from_map::<_, &str, &str>([])),
            Arc::from(std::env::temp_dir().into_boxed_path()),
            std::process::id(),
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

    /// Program with `n` literal `/bin/true` Exec ops chained on `on_ok = Continue` (final op `on_ok
    /// = Escape`); every `on_failed = Terminate`. Used by tests that exercise multi-op advance /
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

    fn dummy_effect(key: DedupKey, target: ResourceId, corr: u64) -> Effect {
        dummy_effect_with_steps(key, target, corr, 1)
    }

    fn dummy_effect_with_steps(
        key: DedupKey,
        target: ResourceId,
        corr: u64,
        steps: usize,
    ) -> Effect {
        // Every caller passes a `perfile_key(N, N, N)` whose `resource` equals the `target` (both
        // `unique_resource_id(N)`), so the derived `key()` reproduces the original `perfile_key`.
        let common = EffectCommon {
            sub: key.sub(),
            profile: key.profile(),
            anchor: target,
            correlation: CorrelationId::from(corr),
            forced: false,
            capture_output: false,
            sub_name: CompactString::new(""),
            program: n_step_program(steps),
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            exclude: Arc::from(Vec::<CompactString>::new()),
        };
        Effect::per_file(
            common,
            target,
            CompactString::new(""),
            Arc::new(Diff::default()),
        )
    }

    /// No-op Spawner stub for tests that go through `reap_pump` on paths where advance is not
    /// attempted (last step or non-Ok outcome). The spawner is plumbed through the function
    /// signature; if a test path were to actually invoke `spawn`, the `unreachable!` would fire and
    /// surface the regression.
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

    /// Spawner stub that records every spawn and returns handles whose waiter is driven via
    /// `complete(pid, outcome)`. Used by the multi-step advance tests where `try_spawn_step`
    /// actually runs.
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
            // Copy out of the lock before checking — the MutexGuard's significant Drop would
            // otherwise live across the if-let.
            let injected = *self.inject_err.lock().unwrap();
            if let Some(kind) = injected {
                return Err(io::Error::from(kind));
            }
            let pid = self.next_pid.fetch_add(1, Ordering::SeqCst);
            self.spawns.lock().unwrap().push(pid);
            let (tx, rx) = crossbeam::channel::bounded::<EffectOutcome>(1);
            self.completions.lock().unwrap().insert(pid, tx);
            let dead = crate::lifecycle::DeadFlag::new();
            Ok(SpawnHandles {
                pid,
                waiter: Box::new(ScriptedWaiter {
                    rx,
                    dead: dead.clone(),
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
            // The multi-step pure-state tests in this module don't exercise pipe dispatch — they
            // use single-Exec programs only. Treat as a regression catcher: if a future test
            // accidentally enables pipe dispatch against this stub, surface the missing scaffolding
            // instead of silently succeeding.
            unreachable!("ScriptedSpawner used by a test that didn't expect spawn_pipe()")
        }
    }
    struct ScriptedWaiter {
        rx: crossbeam::channel::Receiver<EffectOutcome>,
        dead: crate::lifecycle::DeadFlag,
    }
    impl ChildWaiter for ScriptedWaiter {
        fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
            let r = self.rx.recv();
            self.dead.mark_dead();
            r.map_err(|_| io::Error::other("waiter channel dropped"))
        }
    }
    struct ScriptedSignaler {
        dead: crate::lifecycle::DeadFlag,
    }
    impl ChildSignaler for ScriptedSignaler {
        fn signal_term(&self) -> io::Result<()> {
            if self.dead.is_dead() {
                return Ok(());
            }
            Ok(())
        }
        fn signal_kill(&self) -> io::Result<()> {
            if self.dead.is_dead() {
                return Ok(());
            }
            Ok(())
        }
        fn reap_blocking(&self) -> io::Result<()> {
            // ScriptedSpawner's waiter drives reap via the completion channel; this method is the
            // recovery-path only and should not be invoked under the tests that use this stub. A
            // no-op is correct for shape-only conformance.
            self.dead.mark_dead();
            Ok(())
        }
        fn is_dead(&self) -> bool {
            self.dead.is_dead()
        }
        fn mark_dead(&self) {
            self.dead.mark_dead();
        }
    }

    /// Counts SIGTERM / SIGKILL / reap_blocking invocations; never errors. Shared across `RunningJob`
    /// constructions in tests so teardown assertions can distinguish which signaler methods fired
    /// (e.g. the wait-thread-spawn-failure recovery test asserts both `kill` and `reap` bumped by 1).
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
            // The counted fixture has no `dead` flag: tests using it never exercise paths that
            // probe completion. Returning `false` keeps the per-step timer's `is_dead` short-
            // circuit inert under these tests — the timer's signal path is what's being asserted,
            // not the short-circuit.
            false
        }
        fn mark_dead(&self) {
            // No-op: paired with the always-`false` `is_dead` above. The lifecycle ratchet is
            // intentionally inert in this fixture so signal-count assertions don't get masked by
            // the wait-thread / wrapper backstops that publish here on a real signaler. Tests using
            // this fixture don't drive [`super::wait_loop`] and never observe `is_dead == true`, so
            // leaving both methods inert is the consistent shape.
        }
    }

    /// Build a stub `RunningJob` that mimics a freshly-spawned step of a single-step plan. Counted
    /// signaler shared so tests can assert no SIGTERM/SIGKILL during pure-state teardown.
    fn stub_running_job(effect: Arc<Effect>, signaler: Arc<CountingSignaler>) -> RunningJob {
        RunningJob {
            pid: 99_999,
            signaler,
            effect,
            cursor: 0,
            diff_tmp: None,
        }
    }

    /// Channel pair sized for the controller's reap channel; rarely drained in these tests since
    /// most paths don't actually spawn.
    fn reap_channel() -> (
        Sender<EffectCompletion>,
        crossbeam::channel::Receiver<EffectCompletion>,
    ) {
        crossbeam::channel::bounded(64)
    }

    // ---------- wait-thread-spawn-failure recovery ----------

    /// Direct test for [`super::recover_orphan_after_wait_thread_failure`]. The production-path
    /// trigger (`thread::Builder::spawn` returning `Err`) is rare and not directly injectable in
    /// the controller harness, so we exercise the recovery helper in isolation against a stub
    /// [`RunningJob`].
    ///
    /// The bug the helper closes: pre-fix, the recovery branch only called `signal_kill`. The
    /// waiter (the sole reap path) was dropped along with the failed thread closure, so the
    /// SIGKILL'd orphan was never `waitpid`-ed — leaking a zombie until process exit. The helper
    /// now drives both `signal_kill` and `reap_blocking` through the controller-held signaler.
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

    /// `wait_loop` wraps the waiter in `catch_unwind` and publishes `signaler.mark_dead()` after
    /// the catch. The trait-level backstop is the only writer that can flip the dead-ratchet when
    /// the waiter panics before reaching its own self-mark — without it a racing controller signal
    /// could land on a recycled PID.
    #[test]
    fn wait_loop_marks_dead_on_panicking_waiter() {
        struct PanickingWaiter;
        impl ChildWaiter for PanickingWaiter {
            fn wait(self: Box<Self>) -> io::Result<EffectOutcome> {
                panic!("PanickingWaiter: intentional panic to exercise wait_loop backstop");
            }
        }

        let key = perfile_key(60, 60, 60);
        let sub = unique_sub_id(60);
        let signaler: Arc<dyn ChildSignaler> = Arc::new(ScriptedSignaler {
            dead: crate::lifecycle::DeadFlag::new(),
        });
        let permits = crate::permits::Permits::new(nz(1));
        let permit = permits.try_acquire().expect("permit available");
        let (reap_tx, _reap_rx) = reap_channel();

        super::wait_loop(
            Box::new(PanickingWaiter),
            Arc::clone(&signaler),
            key,
            sub,
            permit,
            reap_tx,
        );

        assert!(
            signaler.is_dead(),
            "wait_loop's backstop must publish `mark_dead` even when `wait` panicked",
        );
    }

    /// Stale-Reaped shape: slot exists with no running job and no paired per-Sub gate hold. The
    /// stale arm routes through `terminate_stale` — emits EffectComplete with the reaped outcome,
    /// removes the slot, and does not touch `running_subs` (no bump to undo).
    #[test]
    fn reap_pump_stale_for_unspawned_slot_clears_state() {
        let mut state = test_state(nz(2));
        let key = perfile_key(1, 1, 1);
        let sub = unique_sub_id(1);
        state.slots.insert(key, Slot::default());
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Failed(Termination::Internal),
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert!(state.slots.is_empty(), "slot removed");
        assert!(
            state.running_subs.is_empty(),
            "stale arm did not touch running_subs (no paired bump to undo)",
        );
        assert!(state.ready_queue.is_empty());
        match rx.try_recv() {
            Ok(Input::EffectComplete(c)) => {
                assert_eq!(c.sub, sub);
                assert_eq!(c.key, key);
                assert!(matches!(
                    c.outcome,
                    EffectOutcome::Failed(Termination::Internal)
                ));
            }
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    /// Stale-completion against an empty slot for `Sub A`, while a live plan for the *same* `Sub A`
    /// holds `running_subs` at a different (here unrelated) `DedupKey`. The stale arm must NOT
    /// touch `running_subs` — clobbering the live plan's hold here would release the per-Sub gate
    /// prematurely, letting a fresh same-Sub plan dispatch alongside the live one (violates the
    /// per-Sub serialization invariant).
    ///
    /// This pins the `terminate_stale` contract: engine accounting emit + slot remove, no per-Sub
    /// gate touch, no tmp cleanup. Co-located with the existing stale test so the pair documents
    /// the two arms — empty-set and held-set — of the stale path.
    #[test]
    fn reap_pump_stale_does_not_release_per_sub_gate_for_live_plan() {
        let mut state = test_state(nz(2));
        let stale_key = perfile_key(20, 20, 20);
        let sub = unique_sub_id(20);
        // Live plan holds the per-Sub gate (no slot here — the stale arm only inspects
        // `running_subs`, not the live key).
        state.running_subs.insert(sub);
        state.slots.insert(stale_key, Slot::default());
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key: stale_key,
                sub,
                outcome: EffectOutcome::Failed(Termination::Internal),
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert!(!state.slots.contains_key(&stale_key), "stale slot removed");
        assert!(
            state.running_subs.contains(&sub),
            "live plan's per-Sub gate hold preserved across stale Reaped",
        );
        match rx.try_recv() {
            Ok(Input::EffectComplete(c)) => {
                assert_eq!(c.sub, sub);
                assert_eq!(c.key, stale_key);
                assert!(matches!(
                    c.outcome,
                    EffectOutcome::Failed(Termination::Internal)
                ));
            }
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    /// Single-step plan, Failed outcome: terminal arm runs, signaler is dropped without sending
    /// SIGTERM/SIGKILL (the child already died, we just got the reap), the per-Sub gate hold is
    /// released, and the slot is removed.
    #[test]
    fn reap_pump_failed_single_step_decrements_and_removes() {
        let mut state = test_state(nz(2));
        let key = perfile_key(2, 2, 2);
        let sub = unique_sub_id(2);
        let res = unique_resource_id(2);
        let signaler = Arc::new(CountingSignaler::default());
        let effect = Arc::new(dummy_effect(key, res, 5));
        let slot = Slot {
            running: Some(stub_running_job(effect, Arc::clone(&signaler))),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Failed(Termination::Internal),
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert!(state.slots.is_empty(), "slot removed");
        assert!(state.running_subs.is_empty(), "per-Sub gate cleared");
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

    /// Pump policy with non-empty pending re-queues the slot for the next pump cycle; running is
    /// cleared but the slot stays alive so handle_submit's Latest coalesce continues to work.
    #[test]
    fn reap_pump_with_pending_requeues_for_respawn() {
        let mut state = test_state(nz(2));
        let key = perfile_key(3, 3, 3);
        let sub = unique_sub_id(3);
        let res = unique_resource_id(3);
        let signaler = Arc::new(CountingSignaler::default());
        let effect = Arc::new(dummy_effect(key, res, 7));
        let slot = Slot {
            running: Some(stub_running_job(effect, signaler)),
            pending: Some(dummy_effect(key, res, 8)),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, _rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Ok,
            },
            &engine_in,
            &spawner,
            &reap_tx,
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
        assert!(state.running_subs.is_empty());
    }

    /// Drop policy (shutdown phase) removes the slot regardless of pending; pending is silently
    /// dropped, mirroring the `handle_reap_drop` shutdown contract.
    #[test]
    fn handle_reap_drop_removes_slot_even_with_pending() {
        let mut state = test_state(nz(2));
        let key = perfile_key(4, 4, 4);
        let sub = unique_sub_id(4);
        let res = unique_resource_id(4);
        let signaler = Arc::new(CountingSignaler::default());
        let effect = Arc::new(dummy_effect(key, res, 11));
        let slot = Slot {
            running: Some(stub_running_job(effect, signaler)),
            pending: Some(dummy_effect(key, res, 12)),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, _rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);

        state.handle_reap_drop(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Failed(Termination::Internal),
            },
            &engine_in,
        );

        assert!(state.slots.is_empty(), "slot removed under Drop policy");
        assert!(state.running_subs.is_empty());
        assert!(state.ready_queue.is_empty(), "no re-queue under Drop");
    }

    // ---------- multi-step advance / terminate ----------

    /// Step Ok and not last: `reap_pump` takes the running, calls try_spawn_step which acquires a
    /// fresh permit and spawns instruction N+1. Slot.running is reinstalled with cursor
    /// incremented; per-Sub gate stays held (one insert per program, not per instruction); no
    /// EffectComplete is emitted.
    #[test]
    fn step_ok_not_last_advances_to_next_step() {
        let mut state = test_state(nz(2));
        let key = perfile_key(10, 10, 10);
        let sub = unique_sub_id(10);
        let res = unique_resource_id(10);
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 100,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = ScriptedSpawner::new();

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Ok,
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert_eq!(spawner.spawned().len(), 1, "step 1 was spawned");
        let slot_after = state
            .slots
            .get(&key)
            .expect("slot preserved during advance");
        let running = slot_after.running.as_ref().expect("running reinstalled");
        assert_eq!(running.cursor, 1, "cursor advanced");
        assert!(
            state.running_subs.contains(&sub),
            "per-Sub gate hold preserved across step advance",
        );
        assert!(rx.try_recv().is_err(), "no EffectComplete emitted mid-plan");
        // Drain the wait thread so the test doesn't hang on Drop.
        spawner.complete(running.pid, EffectOutcome::Ok);
    }

    /// Step Failed mid-plan: terminal arm runs with the reaped Failed outcome — no advance
    /// attempted. Counter decrements; EffectComplete emitted; slot removed.
    #[test]
    fn step_failed_mid_plan_terminates_without_advance() {
        let mut state = test_state(nz(2));
        let key = perfile_key(11, 11, 11);
        let sub = unique_sub_id(11);
        let res = unique_resource_id(11);
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 200,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Failed(Termination::Exit(2)),
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert!(state.slots.is_empty(), "slot removed on terminal");
        assert!(state.running_subs.is_empty(), "per-Sub gate cleared");
        match rx.try_recv() {
            Ok(Input::EffectComplete(c)) => assert!(matches!(
                c.outcome,
                EffectOutcome::Failed(Termination::Exit(2))
            )),
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    /// Last step Ok: terminal arm runs (no advance possible). Counter decrements; EffectComplete
    /// emitted with Ok.
    #[test]
    fn last_step_ok_terminates() {
        let mut state = test_state(nz(2));
        let key = perfile_key(12, 12, 12);
        let sub = unique_sub_id(12);
        let res = unique_resource_id(12);
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 300,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 2, // last instruction (0-indexed) of a 3-instruction program
                diff_tmp: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Ok,
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert!(state.slots.is_empty(), "slot removed after last step");
        assert!(state.running_subs.is_empty());
        match rx.try_recv() {
            Ok(Input::EffectComplete(c)) => {
                assert!(matches!(c.outcome, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete::Ok; got {other:?}"),
        }
    }

    /// Permit unavailable mid-program: try_spawn_step returns Deferred, the slot's plan_continue is
    /// set to (effect, cursor+1, diff), the slot is queued for the next pump cycle, no
    /// EffectComplete is emitted, the per-Sub gate stays held.
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
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 2));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 400,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Ok,
            },
            &engine_in,
            &spawner,
            &reap_tx,
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
        assert!(
            state.running_subs.contains(&sub),
            "per-Sub gate hold preserved across deferral",
        );
        assert!(rx.try_recv().is_err(), "no EffectComplete on deferral");
    }

    /// `handle_submit` during plan_continue replaces pending only; plan_continue is left intact
    /// (plan-atomicity invariant).
    #[test]
    fn submit_during_plan_continue_replaces_pending_only() {
        let mut state = test_state(nz(1));
        let _hold = state.permits.try_acquire().expect("acquire");
        let key = perfile_key(14, 14, 14);
        let sub = unique_sub_id(14);
        let res = unique_resource_id(14);
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 2));
        let slot = Slot {
            plan_continue: Some(PlanContinuation {
                effect: Arc::clone(&effect),
                cursor: 1,
                diff_tmp: None,
            }),
            in_ready_queue: true,
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.ready_queue.push_back(key);
        state.running_subs.insert(sub);
        let (tx, _rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = UnusedSpawner;

        // Submit a new effect for the same key.
        let new_effect = dummy_effect(key, res, 99);
        state.handle_submit(new_effect, &spawner, &reap_tx, &engine_in);

        let slot_after = state.slots.get(&key).expect("slot preserved");
        let cont = slot_after
            .plan_continue
            .as_ref()
            .expect("plan_continue NOT touched by submit");
        assert_eq!(cont.cursor, 1);
        let pending = slot_after.pending.as_ref().expect("pending set");
        assert_eq!(
            pending.correlation,
            CorrelationId::from(99),
            "pending replaced by new submit",
        );
    }

    /// Drop policy mid-plan: terminal arm runs immediately with the reaped outcome; advance is
    /// skipped under shutdown so subsequent steps are abandoned.
    #[test]
    fn step_ok_not_last_under_drop_policy_skips_advance() {
        let mut state = test_state(nz(2));
        let key = perfile_key(15, 15, 15);
        let sub = unique_sub_id(15);
        let res = unique_resource_id(15);
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 3));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 500,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);

        state.handle_reap_drop(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Ok,
            },
            &engine_in,
        );

        assert!(state.slots.is_empty(), "slot removed under Drop");
        assert!(state.running_subs.is_empty());
        match rx.try_recv() {
            Ok(Input::EffectComplete(c)) => {
                assert!(matches!(c.outcome, EffectOutcome::Ok));
            }
            other => panic!("expected EffectComplete::Ok; got {other:?}"),
        }
    }

    /// Spawn-failure on next step (try_spawn_step returns Failed): terminate_plan runs with
    /// synthesised `Failed`, the per-Sub gate is released, slot removed.
    #[test]
    fn step_ok_not_last_with_spawn_failure_synthesises_failed() {
        let mut state = test_state(nz(2));
        let key = perfile_key(16, 16, 16);
        let sub = unique_sub_id(16);
        let res = unique_resource_id(16);
        let effect = Arc::new(dummy_effect_with_steps(key, res, 1, 2));
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(RunningJob {
                pid: 600,
                signaler: Arc::clone(&signaler) as Arc<dyn ChildSignaler>,
                effect: Arc::clone(&effect),
                cursor: 0,
                diff_tmp: None,
            }),
            ..Slot::default()
        };
        state.slots.insert(key, slot);
        state.running_subs.insert(sub);
        let (tx, rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(tx);
        let (reap_tx, _reap_rx) = reap_channel();
        let spawner = ScriptedSpawner::new();
        spawner.inject_spawn_error(io::ErrorKind::NotFound);

        state.reap_pump(
            EffectCompletion {
                key,
                sub,
                outcome: EffectOutcome::Ok,
            },
            &engine_in,
            &spawner,
            &reap_tx,
        );

        assert!(state.slots.is_empty(), "slot removed after synth Failed");
        assert!(state.running_subs.is_empty(), "per-Sub gate cleared");
        match rx.try_recv() {
            Ok(Input::EffectComplete(c)) => assert!(matches!(
                c.outcome,
                EffectOutcome::Failed(Termination::Internal)
            )),
            other => panic!("expected EffectComplete::Failed; got {other:?}"),
        }
    }

    // Dispatch is `ProgramOp::target(&outcome)`, which returns a `BranchTarget` directly. Routing
    // coverage lives in `specter-core::program::op::tests`; end-to-end behaviour is covered by the
    // multi-step advance/terminate tests above plus the controller-level tests in `pool.rs`.

    // ---------- handle_cancel (engine-driven per-profile abandon) ----------

    /// When the engine's `handle_gate_deadline` fires, the actuator must SIGTERM every in-flight
    /// child for the cancelled profile AND drop queued work so the pump's blocked-scratch invariant
    /// (`in queue ⇔ flag set ⇔ pending|plan_continue is Some`) holds. `running_subs` stays untouched
    /// here — `terminate_plan`, driven by the natural reap that follows SIGTERM, owns that lifecycle.
    ///
    /// One test pins all three load-bearing invariants in one shot:
    /// SIGTERM-only-for-matching-profile, ready-queue cleanup, per-Sub gate hold preservation.
    #[test]
    fn handle_cancel_sigterms_matching_profile_cleans_queue_preserves_running_subs() {
        let mut state = test_state(nz(4));

        // Profile P: two distinct DedupKeys at the same profile — K1: running (will receive
        // SIGTERM) K2: pending only, in ready_queue (will be queue-cleaned) Profile Q: one running
        // key (untouched — different profile)
        let p_sub = unique_sub_id(100);
        let p_profile = unique_profile_id(100);
        let p_k1 = DedupKey::PerFile {
            sub: p_sub,
            profile: p_profile,
            resource: unique_resource_id(1),
        };
        let p_k2 = DedupKey::PerFile {
            sub: p_sub,
            profile: p_profile,
            resource: unique_resource_id(2),
        };
        let q_sub = unique_sub_id(200);
        let q_profile = unique_profile_id(200);
        let q_key = DedupKey::PerFile {
            sub: q_sub,
            profile: q_profile,
            resource: unique_resource_id(3),
        };

        // P's running slot — carries a counting signaler so we can assert exactly one SIGTERM was
        // delivered through this child.
        let p_k1_signaler = Arc::new(CountingSignaler::default());
        let p_k1_effect = Arc::new(dummy_effect(p_k1, unique_resource_id(1), 1));
        state.slots.insert(
            p_k1,
            Slot {
                running: Some(stub_running_job(p_k1_effect, Arc::clone(&p_k1_signaler))),
                ..Slot::default()
            },
        );

        // P's queued slot — pending only, in ready_queue. The load-bearing case: handle_cancel MUST
        // purge this key, else the next pump's `slot.pending.take().expect(...)` panics.
        let p_k2_pending = dummy_effect(p_k2, unique_resource_id(2), 2);
        state.slots.insert(
            p_k2,
            Slot {
                pending: Some(p_k2_pending),
                in_ready_queue: true,
                ..Slot::default()
            },
        );
        state.ready_queue.push_back(p_k2);

        // Q's running slot — different profile, must not receive a signal. Distinct counting
        // signaler so cross-profile bleed would show up as a non-zero `term` count.
        let q_signaler = Arc::new(CountingSignaler::default());
        let q_effect = Arc::new(dummy_effect(q_key, unique_resource_id(3), 3));
        state.slots.insert(
            q_key,
            Slot {
                running: Some(stub_running_job(q_effect, Arc::clone(&q_signaler))),
                ..Slot::default()
            },
        );

        // Per-Sub gate holds for both live plans — handle_cancel must preserve both holds;
        // `terminate_plan` (driven by the post-SIGTERM natural reap) owns the removal.
        state.running_subs.insert(p_sub);
        state.running_subs.insert(q_sub);

        state.handle_cancel(p_profile);

        // (1) P's running child got exactly one SIGTERM.
        assert_eq!(
            p_k1_signaler.term.load(Ordering::SeqCst),
            1,
            "P's running child must receive exactly one SIGTERM",
        );
        // (2) Q's running child got NO signal — cross-profile bleed would invert the
        // cancel-by-profile contract.
        assert_eq!(
            q_signaler.term.load(Ordering::SeqCst),
            0,
            "Q's running child must not receive a SIGTERM (different profile)",
        );

        // (3) P's queued slot was cleaned: pending dropped, in_ready_queue cleared, ready_queue
        // purged. The load-bearing panic-prevention invariant — the next pump's
        // `pending.take().expect()` would panic if the key remained in the queue with an empty slot.
        let p_k2_slot = state.slots.get(&p_k2).expect("P-k2 slot preserved");
        assert!(
            p_k2_slot.pending.is_none(),
            "P-k2's pending must be dropped",
        );
        assert!(
            !p_k2_slot.in_ready_queue,
            "P-k2's in_ready_queue must be cleared (pump invariant)",
        );
        assert!(
            !state.ready_queue.iter().any(|k| *k == p_k2),
            "ready_queue must not contain a P-keyed slot (pump invariant)",
        );

        // (4) P's running slot still carries the job — terminate_plan (driven by the natural reap
        // that follows SIGTERM) owns the running-side teardown. Removing it here would race with
        // the concurrent reap.
        let p_k1_slot = state.slots.get(&p_k1).expect("P-k1 slot preserved");
        assert!(
            p_k1_slot.running.is_some(),
            "P-k1's running stays in place — terminate_plan owns the take",
        );

        // (5) `running_subs` UNCHANGED — both live plans keep their per-Sub gate hold across the
        // cancel. terminate_plan (Pump-policy, via handle_reap) is the sole remover.
        assert!(
            state.running_subs.contains(&p_sub),
            "P's per-Sub gate hold preserved (terminate_plan owns the remove)",
        );
        assert!(
            state.running_subs.contains(&q_sub),
            "Q's per-Sub gate hold preserved (different profile, untouched)",
        );
    }
}
