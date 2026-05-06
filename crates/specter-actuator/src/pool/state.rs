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

use crate::permits::{Permit, Permits};
use crate::spawner::{ChildSignaler, Spawner};
use crossbeam::channel::Sender;
use specter_core::{CorrelationId, DedupKey, Effect, EffectOutcome, Input, SubId};
use std::collections::{BTreeMap, VecDeque};
use std::path::PathBuf;

/// Policy for [`ActuatorState::handle_reap_inner`]: during normal
/// operation we re-queue pending and let the pump dispatch the next
/// spawn; during shutdown we drop pending and clean up the slot.
#[derive(Copy, Clone)]
enum ReapPolicy {
    Pump,
    Drop,
}

/// Per-`DedupKey` actuator slot: at most one in-flight child plus a
/// single Latest-coalesce pending Effect.
///
/// **Engine-side twin.** Every `Effect` the actuator runs corresponds
/// to a `+1` on the engine's `BurstPhase::Awaiting { outstanding }`
/// counter for the owning Profile. The slot retires the running job
/// (or drops the pending Effect on shutdown) and emits
/// `Input::EffectComplete`; the engine then decrements `outstanding`
/// and either stays in `Awaiting` or transitions to `Rebasing` when
/// the count hits zero. The two bookkeepings are intentionally
/// disjoint: this slot is per-(Sub, DedupKey) and lives on the
/// actuator thread; the Awaiting counter is per-Profile and lives on
/// the engine thread. Neither side sees the other's bookkeeping
/// directly — the `EffectComplete` message is the sole synchronisation
/// point.
#[derive(Debug, Default)]
pub(crate) struct Slot {
    pub running: Option<RunningJob>,
    pub pending: Option<Effect>,
    pub in_ready_queue: bool,
}

#[allow(dead_code)] // `pid` and `tmp_path` are diagnostic-only fields.
pub(crate) struct RunningJob {
    pub pid: u32,
    pub sub: SubId,
    pub correlation: CorrelationId,
    pub signaler: Box<dyn ChildSignaler>,
    /// `Some` iff a tmp diff file was created and `SPECTER_DIFF_PATH` set.
    /// The wait thread owns the canonical path (for cleanup); this copy
    /// is retained for shutdown-time visibility.
    pub tmp_path: Option<PathBuf>,
}

impl std::fmt::Debug for RunningJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningJob")
            .field("pid", &self.pid)
            .field("sub", &self.sub)
            .field("correlation", &self.correlation)
            .field("tmp_path", &self.tmp_path)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) struct ActuatorState {
    pub slots: BTreeMap<DedupKey, Slot>,
    pub ready_queue: VecDeque<DedupKey>,
    pub running_per_sub: BTreeMap<SubId, u32>,
    pub permits: Permits,
}

impl ActuatorState {
    pub fn new(concurrency: usize) -> Self {
        Self {
            slots: BTreeMap::new(),
            ready_queue: VecDeque::new(),
            running_per_sub: BTreeMap::new(),
            permits: Permits::new(concurrency),
        }
    }

    /// Submit handler — enqueue or coalesce. Always end with `pump`.
    pub fn handle_submit(
        &mut self,
        effect: Effect,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) {
        let key = effect.key.clone();
        tracing::trace!(?key, "submit");
        let slot = self.slots.entry(key.clone()).or_default();
        if slot.running.is_some() {
            // Latest coalesce — drop old pending if present.
            slot.pending = Some(effect);
        } else {
            slot.pending = Some(effect);
            if !slot.in_ready_queue {
                slot.in_ready_queue = true;
                self.ready_queue.push_back(key);
            }
        }
        self.pump(spawner, reap_tx);
    }

    /// Reap handler — emit [`Input::EffectComplete`], clear running,
    /// decrement per-Sub counter, optionally re-queue pending and pump.
    pub fn handle_reap(
        &mut self,
        reaped: super::Reaped,
        engine_in: &Sender<Input>,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) {
        self.handle_reap_inner(reaped, engine_in, ReapPolicy::Pump);
        self.pump(spawner, reap_tx);
    }

    /// Shutdown-phase reap handler — emit [`Input::EffectComplete`] and
    /// clear running, but **do not** re-queue pending or pump.
    /// Pending effects are dropped on shutdown.
    pub fn handle_reap_no_pump(&mut self, reaped: super::Reaped, engine_in: &Sender<Input>) {
        self.handle_reap_inner(reaped, engine_in, ReapPolicy::Drop);
    }

    fn handle_reap_inner(
        &mut self,
        reaped: super::Reaped,
        engine_in: &Sender<Input>,
        policy: ReapPolicy,
    ) {
        tracing::trace!(?reaped.key, ?reaped.outcome, "reap");
        // 1. Emit EffectComplete to the engine.
        let _ = engine_in.send(Input::EffectComplete {
            sub: reaped.sub,
            key: reaped.key.clone(),
            result: reaped.outcome,
        });

        // 2. Clear running; decrement per-Sub counter.
        let Some(slot) = self.slots.get_mut(&reaped.key) else {
            return;
        };
        slot.running = None;
        if let Some(c) = self.running_per_sub.get_mut(&reaped.sub) {
            *c = c.saturating_sub(1);
            if *c == 0 {
                self.running_per_sub.remove(&reaped.sub);
            }
        }

        // 3. Re-queue if pending and policy permits; otherwise drop.
        match policy {
            ReapPolicy::Pump if slot.pending.is_some() => {
                if !slot.in_ready_queue {
                    slot.in_ready_queue = true;
                    self.ready_queue.push_back(reaped.key);
                }
            }
            _ => {
                self.slots.remove(&reaped.key);
            }
        }
    }

    /// Spawn ready slots while permits + per-Sub gates allow. Items
    /// blocked by either gate are deferred to a transient buffer and
    /// restored at end so FIFO is preserved across pump invocations.
    pub fn pump(&mut self, spawner: &dyn Spawner, reap_tx: &Sender<super::Reaped>) {
        let mut blocked: VecDeque<DedupKey> = VecDeque::new();
        while let Some(key) = self.ready_queue.pop_front() {
            // Per-Sub gate.
            let sub = sub_of_key(&key);
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
            // Acquired — flag the slot, take pending, spawn.
            let Some(slot) = self.slots.get_mut(&key) else {
                drop(permit);
                continue;
            };
            slot.in_ready_queue = false;
            let Some(effect) = slot.pending.take() else {
                drop(permit);
                continue;
            };
            self.spawn_effect(key, sub, effect, permit, spawner, reap_tx);
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

    #[allow(clippy::needless_pass_by_value)] // effect's fields are cloned individually; ownership transfer documents the responsibility
    fn spawn_effect(
        &mut self,
        key: DedupKey,
        sub: SubId,
        effect: Effect,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
    ) {
        // Materialize the diff tmp file (best-effort: on write failure we
        // proceed without SPECTER_DIFF_PATH; the user's command sees no
        // diff file and reports a missing-var error on its own if it
        // requires one).
        let tmp_path = effect.diff.as_ref().and_then(|diff| {
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

        // Build the final env (clone effect.env + maybe SPECTER_DIFF_PATH).
        let mut env = effect.env.clone();
        if let Some(p) = tmp_path.as_ref() {
            env.push((
                "SPECTER_DIFF_PATH".to_owned(),
                p.to_string_lossy().into_owned(),
            ));
        }

        let argv = effect.command.argv.clone();
        let cwd = effect.cwd.clone();
        let correlation = effect.correlation;
        let capture_output = effect.capture_output;

        let handles = match spawner.spawn(&argv, &env, &cwd, capture_output) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(?key, ?cwd, ?e, "spawn failed");
                if let Some(p) = tmp_path.as_ref() {
                    crate::tmp::cleanup(p);
                }
                drop(permit);
                // Synthesize a Reaped::Failed and route through the
                // normal reap path — handle_reap is reentrant via pump
                // but we're already inside pump, so we send-and-let-the-
                // controller-pick-up rather than calling handle_reap
                // directly. Sending to reap_tx is non-blocking in the
                // typical case (bounded(64)).
                let _ = reap_tx.send(super::Reaped {
                    key,
                    sub,
                    correlation,
                    outcome: EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    },
                });
                return;
            }
        };

        let pid = handles.pid;
        let waiter = handles.waiter;
        let signaler = handles.signaler;

        // Increment per-Sub counter and stash the signaler.
        *self.running_per_sub.entry(sub).or_insert(0) += 1;
        if let Some(slot) = self.slots.get_mut(&key) {
            slot.running = Some(RunningJob {
                pid,
                sub,
                correlation,
                signaler,
                tmp_path: tmp_path.clone(),
            });
        }

        tracing::debug!(?key, pid, "spawned");

        // Spawn the wait thread. The wait_loop owns the waiter, the
        // permit (released on drop), and the tmp_path (for cleanup
        // post-wait).
        let reap_tx = reap_tx.clone();
        let tmp_path_for_wait = tmp_path;
        let wait_key = key.clone();
        if let Err(e) = std::thread::Builder::new()
            .name(format!("specter-actuator-wait-{pid}"))
            .spawn(move || {
                super::wait_loop(
                    waiter,
                    wait_key,
                    sub,
                    correlation,
                    tmp_path_for_wait,
                    permit,
                    reap_tx,
                );
            })
        {
            // Couldn't spawn the wait thread (EAGAIN — RLIMIT_NPROC).
            // The child is running; we have no one to wait for it. The
            // best we can do is log and synthesize Failed; the OS will
            // eventually reap the zombie when the actuator process
            // exits.
            tracing::error!(?key, pid, ?e, "wait thread spawn failed");
            // The signaler is in the slot; controller's shutdown path
            // will kill the orphan if reachable.
        }
    }
}

#[inline]
pub(crate) const fn sub_of_key(key: &DedupKey) -> SubId {
    match *key {
        DedupKey::PerFile { sub, .. } | DedupKey::Subtree { sub, .. } => sub,
    }
}
