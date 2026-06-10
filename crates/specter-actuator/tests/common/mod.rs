//! Shared test fixtures for the integration suite. Real subprocesses via [`OsSpawner`].

#![allow(
    dead_code,
    clippy::manual_assert,
    clippy::missing_panics_doc,
    clippy::module_name_repetitions,
    clippy::needless_pass_by_value,
    clippy::redundant_clone,
    clippy::useless_conversion,
    clippy::wildcard_enum_match_arm
)]

use compact_str::CompactString;
use crossbeam::channel::{Receiver, Sender, bounded, unbounded};
use specter_actuator::{
    EffectCompleteSender, OsSpawner, RunWiring, SendError as ActSendError, SubprocessActuator,
};
use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
use specter_core::testkit::single_exec_program;
use specter_core::{
    ActionProgram, ArgPart, ArgTemplate, CorrelationId, Diff, Effect, EffectCommon,
    EffectCompletion, EffectOp, ExecAction, Input, ProfileId, ResourceId, ResourceKind, SubId,
};
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use std::time::{Duration, Instant};

/// Test adapter that lifts the actuator's `(sub, key, outcome)` triple into the engine-side
/// `Input::EffectComplete` envelope so the test's `Receiver<Input>` continues to observe
/// completions in the pre-trait shape. Mirrors the bin's `DriverEffectSender` without dragging in
/// the bin's transport identity.
struct TestEngineIn(Sender<Input>);

impl EffectCompleteSender for TestEngineIn {
    fn send(&self, completion: EffectCompletion) -> Result<(), ActSendError> {
        self.0
            .send(Input::EffectComplete(completion))
            .map_err(|_| ActSendError::Disconnected)
    }
}

/// Process-wide monotonic correlation counter for tests. Production correlations come from the
/// engine; tests need their own unique stream so parallel tests within one binary don't collide on
/// the actuator's `(pid, correlation)`-keyed tmp file path.
static NEXT_CORR: AtomicU64 = AtomicU64::new(0xdead_0000);

pub fn next_corr() -> u64 {
    NEXT_CORR.fetch_add(1, Ordering::SeqCst)
}

pub const fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).expect("non-zero literal in test fixture")
}

pub fn unique_sub_id(seed: u64) -> SubId {
    use slotmap::KeyData;
    SubId::from(KeyData::from_ffi(seed))
}

pub fn unique_resource_id(seed: u64) -> ResourceId {
    use slotmap::KeyData;
    ResourceId::from(KeyData::from_ffi(seed))
}

pub fn unique_profile_id(seed: u64) -> ProfileId {
    use slotmap::KeyData;
    ProfileId::from(KeyData::from_ffi(seed))
}

pub struct Harness {
    pub effects_tx: Sender<EffectOp>,
    pub shutdown_tx: Sender<()>,
    pub hard_shutdown_tx: Sender<()>,
    pub hard_shutdown_done_rx: Receiver<()>,
    pub engine_in: Receiver<Input>,
    pub join: Option<thread::JoinHandle<()>>,
}

impl Harness {
    pub fn new(concurrency: NonZeroUsize) -> Self {
        let (effects_tx, effects_rx) = bounded::<EffectOp>(1024);
        let (shutdown_tx, shutdown_rx) = bounded::<()>(1);
        let (hard_shutdown_tx, hard_shutdown_rx) = bounded::<()>(1);
        let (hard_shutdown_done_tx, hard_shutdown_done_rx) = bounded::<()>(1);
        let (engine_tx, engine_rx) = unbounded::<Input>();
        let engine_in = TestEngineIn(engine_tx);
        let join = thread::Builder::new()
            .name("test-actuator-controller".into())
            .spawn(move || {
                let spawner = Arc::new(OsSpawner::new());
                let mut a = SubprocessActuator::new(concurrency);
                a.run(
                    RunWiring {
                        effects_rx,
                        shutdown_rx,
                        hard_shutdown_rx,
                        hard_shutdown_done_tx,
                    },
                    &engine_in,
                    spawner.as_ref(),
                );
            })
            .expect("spawn controller");
        Self {
            effects_tx,
            shutdown_tx,
            hard_shutdown_tx,
            hard_shutdown_done_rx,
            engine_in: engine_rx,
            join: Some(join),
        }
    }

    pub fn submit(&self, e: Effect) {
        self.effects_tx.send(EffectOp::Submit(e)).expect("submit");
    }

    pub fn shutdown(&mut self) {
        let _ = self.shutdown_tx.send(());
        if let Some(j) = self.join.take() {
            j.join().expect("controller join");
        }
    }

    pub fn wait_for_effect_completes(&self, n: usize, dur: Duration) -> Vec<Input> {
        let deadline = Instant::now() + dur;
        let mut received = Vec::new();
        while received.len() < n {
            let now = Instant::now();
            if now >= deadline {
                panic!("expected {n} EffectCompletes; saw {}", received.len());
            }
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

/// Wrap a `Vec<String>` argv as a literal-only single-op [`ActionProgram`].
///
/// The resolver renders each `ArgTemplate(Literal(s))` as the slot `s`, so `literal_program(["foo",
/// "bar"])` resolves to `argv = ["foo", "bar"]` byte-for-byte. Used by the integration helpers
/// below to satisfy `Effect.program: Arc<ActionProgram>` while keeping fixture call sites'
/// `Vec<String>` ergonomics intact.
fn literal_program(argv: Vec<String>) -> Arc<ActionProgram> {
    single_exec_program(
        argv.into_iter()
            .map(|s| ArgTemplate::new([ArgPart::literal(s)])),
    )
}

/// Wrap a sequence of literal argvs as a multi-op [`ActionProgram`] — one `Exec` op per inner vec,
/// chained on `on_ok = Continue` with the final op `on_ok = Escape`; every `on_failed` is
/// `Terminate`. The actuator walks them sequentially with stop-on-failure semantics.
pub fn literal_multi_program(steps: Vec<Vec<String>>) -> Arc<ActionProgram> {
    assert!(!steps.is_empty(), "literal_multi_program requires >=1 step");
    let mut b = ProgramBuilder::new();
    let mut prev: Option<specter_core::program::OpHandle> = None;
    for argv in steps {
        if let Some(ph) = prev {
            let next = b.continue_to_next();
            b.patch_on_ok(ph, next).unwrap();
        }
        let exec = ExecAction::new(
            argv.into_iter()
                .map(|s| ArgTemplate::new([ArgPart::literal(s)])),
            None,
        );
        let h = b.emit(SpawnBody::Exec(exec));
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        prev = Some(h);
    }
    if let Some(last) = prev {
        b.patch_on_ok(last, BranchTarget::Escape).unwrap();
    }
    Arc::new(b.build().unwrap())
}

/// PerFile Effect with an arbitrary (possibly multi-instruction) program. Mirrors [`perfile_effect`]
/// but lets the caller supply a pre-built `Arc<ActionProgram>` directly — needed for tests that need
/// to assert program-snapshot invariants by re-using the same `Arc` across multiple fixtures.
pub fn perfile_effect_with_program(
    sub_seed: u64,
    profile_seed: u64,
    res_seed: u64,
    corr: u64,
    program: Arc<ActionProgram>,
    cwd: PathBuf,
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
        anchor_path: Arc::from(cwd),
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

/// Build a PerFile Effect with a literal `argv` and the given correlation.
///
/// `profile_seed` mints the `DedupKey::PerFile.profile` field via [`unique_profile_id`]; tests that
/// don't care about Profile identity can pass any stable value (e.g., the same as `sub_seed`).
///
/// `cwd` is mapped onto `anchor_path` with `anchor_kind = Dir`, so the actuator's `compute_cwd`
/// returns the same path. The fixture leaves `target_relative` empty — `SPECTER_PATH` then mirrors
/// `anchor_path` (the resolver derives `target_path` from `anchor_path` when `target_relative` is
/// empty). Tests asserting on `SPECTER_PATH` set `target_relative` directly to introduce a per-file
/// segment.
pub fn perfile_effect(
    sub_seed: u64,
    profile_seed: u64,
    res_seed: u64,
    corr: u64,
    argv: Vec<String>,
    cwd: PathBuf,
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
        program: literal_program(argv),
        anchor_path: Arc::from(cwd),
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

/// Build a Subtree Effect with a literal `argv`.
///
/// The actuator does not consult `target`; the field is set to a stable per-Profile sentinel
/// (`unique_resource_id(profile_seed)`) so fixtures remain comparable across calls without leaking
/// engine-internal anchor identity into the actuator's tests.
pub fn subtree_effect(
    sub_seed: u64,
    profile_seed: u64,
    corr: u64,
    argv: Vec<String>,
    cwd: PathBuf,
) -> Effect {
    let common = EffectCommon {
        sub: unique_sub_id(sub_seed),
        profile: unique_profile_id(profile_seed),
        anchor: unique_resource_id(profile_seed),
        correlation: CorrelationId::from(corr),
        forced: false,
        capture_output: false,
        sub_name: CompactString::new(""),
        program: literal_program(argv),
        anchor_path: Arc::from(cwd),
        anchor_kind: ResourceKind::Dir,
        exclude: Arc::from(Vec::<CompactString>::new()),
    };
    Effect::subtree(common, None)
}
