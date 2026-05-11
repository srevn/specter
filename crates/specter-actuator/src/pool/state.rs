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
//! An [`Effect`] carries an [`specter_core::ActionProgram`] of one or
//! more [`specter_core::Instruction`]s. The actuator walks the program
//! via a `u32` cursor with stop-on-failure semantics:
//!
//! - **Per-Effect-stable** state (per-Sub counter bump, diff tmp file)
//!   is owned by [`ActuatorState::start_plan`]: bump on plan start,
//!   release on plan terminus.
//! - **Per-instruction** state (permit, OS process, wait thread) is
//!   owned by [`ActuatorState::spawn_step_with_permit`]: each
//!   instruction acquires a fresh permit, the wait thread releases it
//!   on reap.
//! - **One [`Input::EffectComplete`] per Effect**: emitted exactly once
//!   at plan terminus (last instruction Ok, any instruction Failed
//!   under stop-on-fail, or shutdown's Drop policy). The engine's
//!   `outstanding` accounting is unchanged under multi-instruction
//!   programs — the engine doesn't know programs have multiple
//!   instructions.
//!
//! Between two adjacent instructions the slot may be in an intermediate
//! state ([`Slot::plan_continue`]) when the wait-thread has reaped
//! instruction N but no permit is available for instruction N+1. The
//! pump's plan-continue arm has priority over fresh `pending`:
//! continuation work bypasses the per-Sub gate (it's the same program,
//! already admitted) but still respects the global permit cap.

use crate::env::EnvSnapshot;
use crate::permits::{Permit, Permits};
use crate::resolve;
use crate::spawner::{ChildSignaler, ChildWaiter, Spawner};
use crate::timer;
use crossbeam::channel::Sender;
use specter_core::{CommandResolved, DedupKey, Effect, EffectOutcome, Input, Instruction, SubId};
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
/// [`ActuatorState::try_spawn_step`] (which acquires a permit) and
/// [`ActuatorState::spawn_step_with_permit`] (which receives a
/// pre-acquired permit).
///
/// Both failure variants carry no detail: a deferred attempt has no
/// outcome to report (nothing happened), and an OS-level spawn / wait-
/// thread spawn failure is uniformly surfaced as
/// `EffectOutcome::Failed { exit_code: None, signal: None }`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum SpawnError {
    /// Permit semaphore at capacity. The caller defers the instruction
    /// into [`Slot::plan_continue`] and re-queues the slot.
    Deferred,
    /// OS-level process spawn or wait-thread spawn failed. The caller
    /// terminates the plan with `EffectOutcome::Failed` via
    /// [`ActuatorState::terminate_plan`].
    Failed,
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
/// to a `+1` on the engine's `BurstPhase::Awaiting { outstanding }`
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

/// Bookkeeping for one in-flight instruction of a plan.
///
/// `effect` is `Arc`-shared so the next-instruction advance branch in
/// [`ActuatorState::handle_reap_inner`] can re-resolve instruction
/// `N+1`'s argv + env without re-fetching the Effect: the same
/// snapshot drives every instruction. `cursor` is a `u32` index into
/// `effect.program.instructions`.
///
/// `diff_tmp_path` is `Some` iff `start_plan` materialised a diff tmp
/// file for this Effect; shared across all instructions so the user's
/// command reads the same `SPECTER_DIFF_PATH` from cursor 0 to plan
/// terminus. Cleaned at the terminal arm in
/// [`ActuatorState::terminate_plan`] — not by the wait thread, which
/// can't see "is this the last instruction".
pub(crate) struct RunningJob {
    pub pid: u32,
    /// Shared with the per-step timer thread (when [`ExecAction::timeout`]
    /// is set) — the timer needs its own handle to deliver SIGTERM /
    /// SIGKILL when the deadline elapses. The controller holds the
    /// install-side clone; the timer thread holds another; either may
    /// outlive the other (the timer thread can fire after the
    /// controller has dropped the job; the controller's shutdown can
    /// fire before the timer wakes). `Arc<dyn>` over `Box<dyn>` is the
    /// minimum-cost expression of "two co-owners, neither dominates";
    /// `Box → Arc` is allocation-free at the install boundary via
    /// `Arc::from`.
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

    /// The reap pipeline. Three exits:
    ///
    /// 1. **Advance**: instruction N reaped Ok, more instructions
    ///    remain (Pump only). Try-acquire a fresh permit; on Ok, spawn
    ///    instruction N+1 and return — the wait thread will reap it.
    ///    On `Deferred`, defer via [`Slot::plan_continue`] and re-queue
    ///    the slot. On `Failed`, the instruction's spawn raced into an
    ///    OS error; route to terminate with a synthesised `Failed`
    ///    outcome.
    /// 2. **Terminate**: last instruction (or any failure, or
    ///    shutdown) — emit `EffectComplete`, decrement per-Sub counter,
    ///    clean tmp, re-queue pending or remove slot per policy.
    /// 3. **Defensive no-job**: stale Reaped after slot removal — fall
    ///    through to terminate with no diff cleanup. Preserves the
    ///    "always emit EffectComplete" invariant for the engine.
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

        let program_len = job.effect.program.instructions.len();
        let last_instruction = (job.cursor as usize) + 1 >= program_len;
        let try_advance = matches!(policy, ReapPolicy::Pump)
            && matches!(outcome, EffectOutcome::Ok)
            && !last_instruction;

        if try_advance {
            let next_cursor = job.cursor + 1;
            // try_spawn_step takes references; on Ok it consumes the
            // Arc/PathBuf clones into a fresh RunningJob installed in
            // slot.running. On Err the originals stay on `job` so we
            // can route them to plan_continue / terminate_plan without
            // re-cloning.
            match self.try_spawn_step(
                &key,
                sub,
                &job.effect,
                next_cursor,
                job.diff_tmp_path.as_deref(),
                spawner,
                reap_tx,
            ) {
                Ok(()) => return, // wait thread now drives the next reap.
                Err(SpawnError::Deferred) => {
                    self.queue_plan_continue(
                        key,
                        PlanContinuation {
                            effect: job.effect,
                            cursor: next_cursor,
                            diff_tmp_path: job.diff_tmp_path,
                        },
                    );
                    return;
                }
                Err(SpawnError::Failed) => {
                    // Instruction N+1 spawn failed mid-plan. Terminate
                    // the plan with synthesised Failed; tmp cleanup
                    // happens here (the failure path consumed nothing
                    // from job).
                    self.terminate_plan(
                        key,
                        sub,
                        job.diff_tmp_path.as_deref(),
                        EffectOutcome::Failed {
                            exit_code: None,
                            signal: None,
                        },
                        policy,
                        engine_in,
                    );
                    return;
                }
            }
        }

        // Terminal: last instruction, any failure, or shutdown.
        self.terminate_plan(
            key,
            sub,
            job.diff_tmp_path.as_deref(),
            outcome,
            policy,
            engine_in,
        );
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
            *c = c.saturating_sub(1);
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
        let mut blocked: VecDeque<DedupKey> = VecDeque::new();
        while let Some(key) = self.ready_queue.pop_front() {
            let sub = sub_of_key(&key);
            let Some(slot) = self.slots.get_mut(&key) else {
                continue;
            };

            // Plan-continue: bypass per-Sub gate.
            if slot.plan_continue.is_some() {
                let Some(permit) = self.permits.try_acquire() else {
                    blocked.push_back(key);
                    while let Some(k) = self.ready_queue.pop_front() {
                        blocked.push_back(k);
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
                blocked.push_back(key);
                continue;
            }
            // Global gate.
            let Some(permit) = self.permits.try_acquire() else {
                // No more permits this round; defer this and the
                // remaining queued items (FIFO preserved).
                blocked.push_back(key);
                while let Some(k) = self.ready_queue.pop_front() {
                    blocked.push_back(k);
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
        for k in blocked {
            // The flag is already true (we set it when we pushed and only
            // cleared it on successful spawn). Defensive: ensure it.
            if let Some(slot) = self.slots.get_mut(&k) {
                slot.in_ready_queue = true;
            }
            self.ready_queue.push_back(k);
        }
    }

    /// Start a plan: materialise the diff tmp file (if needed), spawn
    /// step 0 with the given permit, bump the per-Sub counter on
    /// success.
    ///
    /// On spawn failure, calls [`Self::terminate_plan`] with synthesised
    /// `EffectOutcome::Failed`; the per-Sub counter never bumped (so
    /// the saturating_sub there is a no-op against the absent entry —
    /// counter stays consistent).
    ///
    /// The counter bump is sequenced **after** spawn success so the
    /// failure path's terminate doesn't need a counter rollback. The
    /// controller is single-threaded, so no Reaped can race in between.
    ///
    /// `effect` is taken by value so the caller (pump) hands off the
    /// freshly-constructed `Arc<Effect>` and forgets about it; on
    /// success the Arc is cloned into [`Slot::running`], on failure it
    /// drops at end of scope. Passing by reference would work but
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
            Ok(()) => {
                // Spawn succeeded: bump per-Sub counter (one bump per
                // plan, mirrored by terminate_plan's decrement).
                *self.running_per_sub.entry(sub).or_insert(0) += 1;
            }
            Err(_) => {
                // OS spawn or wait-thread spawn failed at the first
                // instruction. The counter was never bumped;
                // terminate_plan's saturating_sub no-ops against the
                // absent entry.
                self.terminate_plan(
                    key,
                    sub,
                    diff_tmp_path.as_deref(),
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    },
                    ReapPolicy::Pump,
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
    /// On spawn failure, [`Self::terminate_plan`] decrements the
    /// per-Sub counter (which is at +1 from the original start_plan),
    /// cleans the inherited tmp, and removes the slot.
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
            Err(_) => {
                self.terminate_plan(
                    key,
                    sub,
                    diff_tmp_path.as_deref(),
                    EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    },
                    ReapPolicy::Pump,
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
    /// - `Err(SpawnError::Failed)` — OS-level spawn or wait-thread
    ///   startup failed; caller terminates the plan with synthesised
    ///   `EffectOutcome::Failed`.
    ///
    /// The Deferred branch returns before consuming any of the borrowed
    /// inputs — caller-owned values stay live for the
    /// `PlanContinuation` hand-off.
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
    }

    /// Spawn one instruction of a plan with a pre-acquired permit.
    /// Installs [`Slot::running`] on success.
    ///
    /// Sequencing pinned: slot.running is installed **before** the wait
    /// thread is spawned, so a fast-completing wait thread (mock under
    /// test, or a child that exits between fork and wait) can't send
    /// `Reaped` before the controller knows about it.
    ///
    /// `now: SystemTime` is sampled here per instruction — the contract
    /// on `${specter.time}` / `SPECTER_TIME` is "the wall-clock instant
    /// immediately before the kernel runs the user's command", which
    /// must hold per instruction in a multi-instruction program.
    ///
    /// # Dispatch
    ///
    /// PR1 only produces [`Instruction::SpawnExec`] from validation;
    /// the dispatch matches on that variant. The non-`SpawnExec` arms
    /// are `unreachable!()` — PR2's pipe / conditional support will
    /// light them up by adding lowering paths in `specter-config` and
    /// branching here.
    ///
    /// On wait-thread spawn failure: the freshly-spawned child is
    /// alive but has no waiter (the closure that owned it has been
    /// dropped by `Builder::spawn`'s `Err` path). The recovery branch
    /// SIGKILLs the orphan via the signaler held in `slot.running`,
    /// then synchronously reaps it via
    /// [`crate::spawner::ChildSignaler::reap_blocking`] so the OS
    /// doesn't leak a zombie. `slot.running` is then cleared (the
    /// terminate_plan caller expects it to be `None`) and
    /// `SpawnError::Failed` returns.
    ///
    /// **Slot invariant.** Both `self.slots.get_mut(key)` lookups in
    /// this function assume the slot was just touched by the caller
    /// (the controller is single-threaded; no Reap or Submit can
    /// interleave between caller's `pump` / `handle_reap_inner` and
    /// here). A missing slot is a programming error, surfaced via
    /// `expect` rather than silently masked — silent masking would
    /// otherwise leak the signaler and leave the child unreachable
    /// from shutdown signaling.
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
    ) -> Result<(), SpawnError> {
        let now = std::time::SystemTime::now();
        let cwd: &Path = resolve::compute_cwd(&effect.anchor_path, effect.anchor_kind);
        let correlation = effect.correlation;
        let capture_output = effect.capture_output;

        let exec = match &effect.program.instructions[cursor as usize] {
            Instruction::SpawnExec(exec) => exec,
            Instruction::SpawnPipe(_)
            | Instruction::SpawnPredicate { .. }
            | Instruction::Jump { .. } => {
                unreachable!("PR1 lowering only emits SpawnExec; pipe/predicate/jump lands in PR2",)
            }
        };
        let (CommandResolved { argv }, env) =
            match resolve::resolve_step(effect, exec, now, diff_path, &self.env_snapshot) {
                Ok(resolved) => resolved,
                Err(e) => {
                    // Strict `${env.<NAME>}` failure: no spawn, no
                    // wait thread, no timer. Permit drops at the end
                    // of this scope; caller terminates the plan with
                    // synthesised `EffectOutcome::Failed`.
                    tracing::error!(?key, cursor, %e, "resolver error; aborting step");
                    drop(permit);
                    return Err(SpawnError::Failed);
                }
            };

        let handles = match spawner.spawn(&argv, &env, cwd, capture_output) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(?key, cursor, ?cwd, ?e, "spawn failed");
                drop(permit);
                return Err(SpawnError::Failed);
            }
        };
        let crate::spawner::SpawnHandles {
            pid,
            waiter,
            signaler,
        } = handles;

        // Install RunningJob BEFORE spawning the wait thread to close
        // the race where a fast-completing waiter could send Reaped
        // before the controller knows about it. PathBuf is allocated
        // here (one per instruction transition); `effect` is
        // Arc-cloned (cheap). `Arc::from(box)` reuses the box's
        // allocation — no second heap hop relative to the prior `Box`
        // field shape.
        //
        // The Arc is also handed to the optional per-step timer thread
        // below; cloning before the move into RunningJob keeps the
        // controller's installed-side reference live regardless of
        // whether the timer is armed.
        let signaler: Arc<dyn ChildSignaler> = Arc::from(signaler);
        let timer_signaler = exec.timeout.is_some().then(|| Arc::clone(&signaler));
        let timeout = exec.timeout;
        let timer_grace = self.shutdown_grace;
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
                .expect("slot present at wait-thread-spawn-failure recovery");
            let job = slot
                .running
                .take()
                .expect("slot.running installed unconditionally above");
            recover_orphan_after_wait_thread_failure(job);
            return Err(SpawnError::Failed);
        }

        // Per-step timer: spawn AFTER the wait thread is alive so the
        // wait thread's `dead` flag is the natural-completion signal
        // the timer short-circuits on. Best-effort — see
        // [`crate::timer`] module docs for the spawn-failure policy.
        if let (Some(d), Some(sig)) = (timeout, timer_signaler) {
            // `pid` is unique within the actuator process; cursor
            // distinguishes steps within an Effect. Sub is implied by
            // the wait thread's `act-wait-{pid}` name visible alongside.
            let timer_name = format!("c{cursor}-pid{pid}");
            if let Err(e) = timer::spawn_timer(&timer_name, d, timer_grace, sig) {
                tracing::error!(
                    ?key,
                    cursor,
                    pid,
                    timeout = ?d,
                    ?e,
                    "per-step timer thread spawn failed; deadline not enforced",
                );
            }
        }

        tracing::debug!(?key, cursor, pid, "spawned instruction");
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
/// caller has just `slot.running.take()`-ed the in-flight bookkeeping
/// and hands it over for tear-down; once we return, the signaler /
/// effect Arc / diff-tmp path all drop. Borrowing would force the
/// caller into a take-then-restore dance for no behavioural gain.
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
    //!
    //! The PR2 multi-step advance/terminate tests are in the dedicated
    //! `multi_step` submodule below.
    use super::super::{Reaped, SHUTDOWN_GRACE};
    use super::{ActuatorState, PlanContinuation, ReapPolicy, RunningJob, Slot};
    use crate::env::EnvSnapshot;
    use crate::spawner::{ChildSignaler, ChildWaiter, EnvVar, SpawnHandles, Spawner};
    use compact_str::CompactString;
    use crossbeam::channel::{Sender, unbounded};
    use specter_core::{
        ActionProgram, ArgPart, ArgTemplate, CorrelationId, DedupKey, Effect, EffectOutcome,
        ExecAction, Input, Instruction, ProfileId, ResourceId, ResourceKind, SubId,
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

    /// Program with `n` literal `/bin/true` `SpawnExec` instructions.
    /// Used by tests that exercise multi-instruction advance / terminate.
    fn n_step_program(n: usize) -> Arc<ActionProgram> {
        let instructions: Vec<Instruction> = (0..n)
            .map(|_| {
                Instruction::SpawnExec(ExecAction::new([ArgTemplate::new([ArgPart::literal(
                    "/bin/true",
                )])]))
            })
            .collect();
        Arc::new(ActionProgram::new(instructions))
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
                signaler: Box::new(ScriptedSignaler { dead }),
            })
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

    /// Build a stub `RunningJob` that mimics a freshly-spawned step of
    /// a single-step plan. Counted signaler shared so tests can assert
    /// no SIGTERM/SIGKILL during pure-state teardown.
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
    /// a pre-constructed [`RunningJob`].
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
}
