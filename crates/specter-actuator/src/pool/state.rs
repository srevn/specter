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
use crate::resolve;
use crate::spawner::{ChildSignaler, ChildWaiter, Spawner};
use crossbeam::channel::Sender;
use specter_core::{CommandResolved, CorrelationId, DedupKey, Effect, EffectOutcome, Input, SubId};
use std::collections::{BTreeMap, VecDeque};
use std::num::NonZeroUsize;
use std::panic::AssertUnwindSafe;
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

pub(crate) struct RunningJob {
    pub pid: u32,
    // `sub` and `correlation` are debug-only — read by the manual
    // [`Debug`] impl below to surface job context in tracing dumps;
    // not consulted by reap or shutdown paths.
    pub sub: SubId,
    pub correlation: CorrelationId,
    pub signaler: Box<dyn ChildSignaler>,
}

impl std::fmt::Debug for RunningJob {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RunningJob")
            .field("pid", &self.pid)
            .field("sub", &self.sub)
            .field("correlation", &self.correlation)
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
    pub fn new(concurrency: NonZeroUsize) -> Self {
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
        engine_in: &Sender<Input>,
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
        self.pump(spawner, reap_tx, engine_in);
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
        self.pump(spawner, reap_tx, engine_in);
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
    pub fn pump(
        &mut self,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
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
            self.spawn_effect(key, sub, &effect, permit, spawner, reap_tx, engine_in);
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

    fn spawn_effect(
        &mut self,
        key: DedupKey,
        sub: SubId,
        effect: &Effect,
        permit: Permit,
        spawner: &dyn Spawner,
        reap_tx: &Sender<super::Reaped>,
        engine_in: &Sender<Input>,
    ) {
        // Resolution at the spawn boundary preserves "render late" as
        // an architectural invariant: a same-key Effect that raced into
        // `pending` and was replaced by `handle_submit`'s Latest-coalesce
        // path never reaches this point, so the bytes the resolver
        // allocates are guaranteed to back a real syscall.
        //
        // `SystemTime::now()` is sampled here — once per spawn — and
        // threaded through to both the `$time` argv slot and the
        // `SPECTER_TIME` env value so they agree on the wall-clock
        // instant immediately before the kernel runs the user's command.
        let now = std::time::SystemTime::now();
        let cwd = resolve::compute_cwd(&effect.anchor_path, effect.anchor_kind);
        let correlation = effect.correlation;
        let capture_output = effect.capture_output;

        // Materialise the diff tmp file before resolving so the resolver
        // can slot `SPECTER_DIFF_PATH` into its alphabetical position
        // (between SPECTER_CORRELATION and SPECTER_EVENT_KIND). Best-
        // effort: on write failure we proceed with `diff_path = None`,
        // which the resolver translates to "omit the env var entirely";
        // the user's command sees no diff file and reports a missing-var
        // error on its own if it requires one. The tmp_path is retained
        // for cleanup on the spawn-failure / wait-thread-failure paths.
        let tmp_path = effect.diff.as_ref().and_then(|diff| {
            let path = crate::tmp::tmp_path(correlation);
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

        let (CommandResolved { argv }, env) =
            resolve::resolve_effect(effect, now, tmp_path.as_deref());

        let handles = match spawner.spawn(&argv, &env, &cwd, capture_output) {
            Ok(h) => h,
            Err(e) => {
                tracing::error!(?key, ?cwd, ?e, "spawn failed");
                if let Some(p) = tmp_path.as_ref() {
                    crate::tmp::cleanup(p);
                }
                drop(permit);
                // Inline teardown rather than `reap_tx.send`. A channel
                // round-trip would let a same-key submit drain off
                // `effects_rx` before this synth Reap drains off
                // `reap_rx`, repopulating `slot.running` with a fresh
                // job; the picked-up Reap would then clobber that
                // job's signaler and `slots.remove` it, leaking the
                // SIGTERM target on shutdown and double-allowing
                // same-Sub spawns past the per-Sub gate. Direct call
                // collapses the window. `running_per_sub` was not
                // bumped on this branch (saturating_sub no-ops on the
                // absent entry); pending was taken in pump, so Pump
                // policy removes the slot.
                self.handle_reap_inner(
                    super::Reaped {
                        key,
                        sub,
                        correlation,
                        outcome: EffectOutcome::Failed {
                            exit_code: None,
                            signal: None,
                        },
                    },
                    engine_in,
                    ReapPolicy::Pump,
                );
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
            });
        }

        tracing::debug!(?key, pid, "spawned");

        // Pre-clone what the wait_loop closure consumes; `Builder::spawn`
        // drops the closure (and its captures) on failure, but the
        // failure branch below still needs `tmp_path` for cleanup and
        // `key` for the synth Reap.
        let tmp_path_for_thread = tmp_path.clone();
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
                    tmp_path_for_thread,
                    permit,
                    reap_tx_for_thread,
                );
            })
        {
            tracing::error!(
                ?key,
                pid,
                ?e,
                "wait thread spawn failed; SIGKILL orphan + synth Failed",
            );
            // The child is alive but has no waiter. Without SIGKILL its
            // user command runs unmonitored to completion (writing to
            // the watched tree), then sits as an unreaped zombie until
            // the actuator exits. The signaler lives in `slot.running`
            // until `handle_reap_inner` clears the slot, so SIGKILL
            // must precede teardown.
            if let Some(slot) = self.slots.get(&key)
                && let Some(job) = slot.running.as_ref()
                && let Err(kill_err) = job.signaler.signal_kill()
            {
                tracing::warn!(?key, pid, ?kill_err, "orphan SIGKILL failed");
            }
            if let Some(p) = tmp_path.as_ref() {
                crate::tmp::cleanup(p);
            }
            self.handle_reap_inner(
                super::Reaped {
                    key,
                    sub,
                    correlation,
                    outcome: EffectOutcome::Failed {
                        exit_code: None,
                        signal: None,
                    },
                },
                engine_in,
                ReapPolicy::Pump,
            );
        }
    }
}

/// Wait-thread body. Block on `waiter.wait()`; on return, clean up the
/// tmp file, release the permit, send a [`super::Reaped`] to the
/// controller.
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
#[allow(clippy::needless_pass_by_value)] // closure-spawned: arguments owned for the thread
fn wait_loop(
    waiter: Box<dyn ChildWaiter>,
    key: DedupKey,
    sub: SubId,
    correlation: CorrelationId,
    tmp_path: Option<PathBuf>,
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
    if let Some(p) = tmp_path.as_ref() {
        crate::tmp::cleanup(p);
    }
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
    //! through. The two synth-Reap callers (spawn-failure inline and
    //! wait-thread-spawn-failure inline) are exercised here against
    //! pre-loaded state, since neither has a fault-injection seam in
    //! the controller harness.
    use super::super::Reaped;
    use super::{ActuatorState, ReapPolicy, RunningJob, Slot};
    use crate::spawner::ChildSignaler;
    use compact_str::CompactString;
    use crossbeam::channel::unbounded;
    use specter_core::{
        ArgPart, ArgTemplate, CommandTemplate, CorrelationId, DedupKey, Effect, EffectOutcome,
        Input, ProfileId, ResourceId, ResourceKind, SubId,
    };
    use std::io;
    use std::num::NonZeroUsize;
    use std::path::PathBuf;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn nz(n: usize) -> NonZeroUsize {
        NonZeroUsize::new(n).expect("test setup: n must be non-zero")
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

    fn dummy_effect(key: DedupKey, target: ResourceId, corr: u64) -> Effect {
        Effect {
            key,
            target,
            forced: false,
            correlation: CorrelationId(corr),
            diff: None,
            capture_output: false,
            sub_name: CompactString::new(""),
            command: Arc::new(CommandTemplate::new([ArgTemplate::new([
                ArgPart::literal("/bin/true"),
            ])])),
            anchor_path: Arc::from(PathBuf::from("/tmp")),
            anchor_kind: ResourceKind::Dir,
            target_relative: CompactString::new(""),
            exclude: Arc::from(Vec::<CompactString>::new()),
        }
    }

    /// Counts SIGTERM / SIGKILL invocations; never errors. Shared
    /// across `RunningJob` constructions in tests so we can assert
    /// neither was sent during pure-state teardown.
    #[derive(Default)]
    struct CountingSignaler {
        term: AtomicUsize,
        kill: AtomicUsize,
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
    }

    fn stub_running_job(sub: SubId, corr: u64, signaler: Arc<CountingSignaler>) -> RunningJob {
        struct Adapter(Arc<CountingSignaler>);
        impl ChildSignaler for Adapter {
            fn signal_term(&self) -> io::Result<()> {
                self.0.signal_term()
            }
            fn signal_kill(&self) -> io::Result<()> {
                self.0.signal_kill()
            }
        }
        RunningJob {
            pid: 99_999,
            sub,
            correlation: CorrelationId(corr),
            signaler: Box::new(Adapter(signaler)),
        }
    }

    /// Spawn-failure shape: the slot exists (created in `handle_submit`)
    /// but `running` was never set and `running_per_sub` was never
    /// bumped (the spawn failed before the increment). The synth Reap
    /// must remove the slot, leave the counter map empty (no
    /// underflow), and emit `EffectComplete::Failed` to the engine.
    #[test]
    fn handle_reap_inner_synth_for_unspawned_slot_clears_state() {
        let mut state = ActuatorState::new(nz(2));
        let key = perfile_key(1, 1, 1);
        let sub = unique_sub_id(1);
        state.slots.insert(key.clone(), Slot::default());
        let (tx, rx) = unbounded::<Input>();

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

    /// Wait-thread-spawn-failure shape: spawn succeeded so `running`
    /// is set and `running_per_sub[sub] == 1`, but the wait thread
    /// failed to start. The synth Reap must drop the signaler from
    /// `running`, decrement the counter to zero (removing the entry),
    /// and remove the slot.
    #[test]
    fn handle_reap_inner_synth_for_running_slot_decrements_and_removes() {
        let mut state = ActuatorState::new(nz(2));
        let key = perfile_key(2, 2, 2);
        let sub = unique_sub_id(2);
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(stub_running_job(sub, 5, Arc::clone(&signaler))),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, rx) = unbounded::<Input>();

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
        let mut state = ActuatorState::new(nz(2));
        let key = perfile_key(3, 3, 3);
        let sub = unique_sub_id(3);
        let res = unique_resource_id(3);
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(stub_running_job(sub, 7, signaler)),
            pending: Some(dummy_effect(key.clone(), res, 8)),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, _rx) = unbounded::<Input>();

        state.handle_reap_inner(
            Reaped {
                key: key.clone(),
                sub,
                correlation: CorrelationId(7),
                outcome: EffectOutcome::Ok,
            },
            &tx,
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
        let mut state = ActuatorState::new(nz(2));
        let key = perfile_key(4, 4, 4);
        let sub = unique_sub_id(4);
        let res = unique_resource_id(4);
        let signaler = Arc::new(CountingSignaler::default());
        let slot = Slot {
            running: Some(stub_running_job(sub, 11, signaler)),
            pending: Some(dummy_effect(key.clone(), res, 12)),
            ..Slot::default()
        };
        state.slots.insert(key.clone(), slot);
        state.running_per_sub.insert(sub, 1);
        let (tx, _rx) = unbounded::<Input>();

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
            ReapPolicy::Drop,
        );

        assert!(state.slots.is_empty(), "slot removed under Drop policy");
        assert!(state.running_per_sub.is_empty());
        assert!(state.ready_queue.is_empty(), "no re-queue under Drop");
    }
}
