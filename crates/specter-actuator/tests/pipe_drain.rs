//! A pipe end to end, through `OsSpawner` and real children: a failing stage stops the stages
//! upstream of it, the stages after it read what it wrote, and the pipe ends as its rightmost failed
//! stage did. `pipe::tests` pins the same rule over stub stages.

mod common;

use common::{Harness, next_corr, nz, perfile_effect_with_program, unique_sub_id};
use specter_core::program::{BranchTarget, MultiStage, ProgramBuilder, SpawnBody};
use specter_core::{
    ActionProgram, ArgPart, ArgTemplate, EffectOutcome, ExecAction, Input, Termination,
};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Build a single-op program whose `SpawnBody` is a Pipe of the given literal-argv stages. `on_ok =
/// Escape`, `on_failed = Terminate` — the test only cares about the pipe's aggregated outcome.
fn pipe_program(stages: Vec<Vec<String>>) -> Arc<ActionProgram> {
    assert!(stages.len() >= 2, "pipe_program requires >=2 stages");
    let action_stages: Arc<[ExecAction]> = stages
        .into_iter()
        .map(|argv| {
            ExecAction::new(argv.into_iter().map(|s| ArgTemplate::new([ArgPart::literal(s)])), None)
        })
        .collect::<Vec<_>>()
        .into();
    let mut b = ProgramBuilder::new();
    let h =
        b.emit(SpawnBody::Pipe(MultiStage::new(action_stages).expect("test pipe has >=2 stages")));
    b.patch_on_ok(h, BranchTarget::Escape).unwrap();
    b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
    Arc::new(b.build().unwrap())
}

/// Stage 0 blocks without writing (`sleep 60`), so no SIGPIPE would ever end it; stage 1 fails after
/// ~2s. Its failure stops stage 0, so the pipe ends well inside the sleep, and as stage 1 ended:
/// `Exit(7)`, not the SIGTERM that stopped stage 0.
#[test]
fn pipe_drain_does_not_block_on_hung_stage_0() {
    let mut harness = Harness::new(nz(2));
    let program = pipe_program(vec![
        vec!["/bin/sleep".into(), "60".into()],
        vec!["/bin/sh".into(), "-c".into(), "sleep 2; exit 7".into()],
    ]);
    let sub_seed: u64 = 0xf1_0001;
    harness.submit(perfile_effect_with_program(
        sub_seed,
        sub_seed,
        sub_seed,
        next_corr(),
        program,
        PathBuf::from("/tmp"),
    ));

    let start = Instant::now();
    // Generous upper bound: stage 1 exits at ~2s and its failure stops stage 0 at once; anywhere
    // near 60s, nothing stopped it.
    let inputs = harness.wait_for_effect_completes(1, Duration::from_secs(30));
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "pipe drain exceeded sanity bound (elapsed = {elapsed:?})",
    );

    match &inputs[0] {
        Input::EffectComplete(c) => {
            assert_eq!(c.sub, unique_sub_id(sub_seed));
            assert_eq!(c.outcome, EffectOutcome::Failed(Termination::Exit(7)));
        }
        other => panic!("expected EffectComplete, got {other:?}"),
    }
    harness.shutdown();
}

/// Two stages that both succeed (`echo hi | cat`): `cat` reads to the end of its input, which comes
/// when `echo` exits, and the pipe is `Ok`.
#[test]
fn pipe_drain_all_ok_completes_ok() {
    let mut harness = Harness::new(nz(2));
    let program =
        pipe_program(vec![vec!["/bin/echo".into(), "hi".into()], vec!["/bin/cat".into()]]);
    let sub_seed: u64 = 0xf1_0002;
    harness.submit(perfile_effect_with_program(
        sub_seed,
        sub_seed,
        sub_seed,
        next_corr(),
        program,
        PathBuf::from("/tmp"),
    ));
    let inputs = harness.wait_for_effect_completes(1, Duration::from_secs(10));
    match &inputs[0] {
        Input::EffectComplete(c) => {
            assert_eq!(c.outcome, EffectOutcome::Ok);
        }
        other => panic!("expected EffectComplete, got {other:?}"),
    }
    harness.shutdown();
}

/// Stage 0 writes 500 lines, fewer bytes than any pipe buffer holds, and exits 3 at once; stage 1
/// sleeps through that failure before it counts, and fails unless every line arrived. The pipe ends
/// as stage 0 did: had the failure stopped its reader, or the reader read short, it would end as
/// the reader did.
#[test]
fn pipe_drain_reader_reads_all_a_failed_stage_wrote() {
    let mut harness = Harness::new(nz(2));
    let program = pipe_program(vec![
        vec![
            "/bin/sh".into(),
            "-c".into(),
            "i=0; while [ $i -lt 500 ]; do echo line-$i; i=$((i+1)); done; exit 3".into(),
        ],
        vec!["/bin/sh".into(), "-c".into(), "sleep 1; n=$(wc -l); [ $n -eq 500 ]".into()],
    ]);
    let sub_seed: u64 = 0xf1_0003;
    harness.submit(perfile_effect_with_program(
        sub_seed,
        sub_seed,
        sub_seed,
        next_corr(),
        program,
        PathBuf::from("/tmp"),
    ));
    let inputs = harness.wait_for_effect_completes(1, Duration::from_secs(20));
    match &inputs[0] {
        Input::EffectComplete(c) => {
            assert_eq!(c.outcome, EffectOutcome::Failed(Termination::Exit(3)));
        }
        other => panic!("expected EffectComplete, got {other:?}"),
    }
    harness.shutdown();
}
