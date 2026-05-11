//! F1 regression: pipe drain must not deadlock when stage 0's child
//! blocks indefinitely.
//!
//! The parallel `PipeWaiter` (one OS thread per stage) lets stage 1's
//! prompt failure fire the cascade SIGTERM that unblocks stage 0.
//! Under the prior sequential design, this test would hang until
//! stage 0's `sleep 60` finished — effectively forever for any
//! per-test timeout.
//!
//! Pairs with the in-crate unit test
//! `pipe::tests::blocked_first_stage_unblocked_by_cascade_does_not_deadlock`,
//! which pins the same property against synthetic `BlockingWaiter`s
//! at the `PipeWaiter` abstraction layer. This test exercises the
//! property end-to-end through `OsSpawner` + real children.

mod common;

use common::{Harness, next_corr, perfile_effect_with_program, unique_sub_id};
use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
use specter_core::{ActionProgram, ArgPart, ArgTemplate, EffectOutcome, ExecAction, Input};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Build a single-op program whose `SpawnBody` is a Pipe of the given
/// literal-argv stages. `on_ok = Escape`, `on_failed = Terminate` —
/// the test only cares about the pipe's aggregated outcome.
fn pipe_program(stages: Vec<Vec<String>>) -> Arc<ActionProgram> {
    assert!(stages.len() >= 2, "pipe_program requires >=2 stages");
    let action_stages: Arc<[ExecAction]> = stages
        .into_iter()
        .map(|argv| {
            ExecAction::new(
                argv.into_iter()
                    .map(|s| ArgTemplate::new([ArgPart::literal(s)])),
            )
        })
        .collect::<Vec<_>>()
        .into();
    let mut b = ProgramBuilder::new();
    let h = b.emit(SpawnBody::Pipe(action_stages));
    b.patch_on_ok(h, BranchTarget::Escape).unwrap();
    b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
    Arc::new(b.build().unwrap())
}

/// F1 regression: stage 0 blocks (`sleep 60`); stage 1 fails after
/// ~2s. The parallel `PipeWaiter` must report `Failed` well under the
/// 60-second sleep window.
///
/// **Outcome aggregation pins.** Stage 0 (sleep, no stdout) ignores
/// SIGPIPE — it exits only because the cascade SIGTERM lands on it,
/// so its outcome is `Failed { signal: Some(15) }`. Stage 1 exits
/// cleanly with `Failed { exit_code: Some(7) }`. Spawn-order
/// aggregation (last non-zero exit / first observed signal) yields
/// `exit_code = 7` and `signal = 15`.
#[test]
fn pipe_drain_does_not_block_on_hung_stage_0() {
    let mut harness = Harness::new(2);
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
    // Generous upper bound — stage 1 exits at ~2s; the cascade
    // SIGTERM should make stage 0 exit within scheduling slop.
    // Anything anywhere near 60s indicates the sequential deadlock
    // would have hit.
    let inputs = harness.wait_for_effect_completes(1, Duration::from_secs(30));
    let elapsed = start.elapsed();
    assert!(
        elapsed < Duration::from_secs(30),
        "pipe drain exceeded sanity bound (elapsed = {elapsed:?})",
    );

    match &inputs[0] {
        Input::EffectComplete { sub, result, .. } => {
            assert_eq!(*sub, unique_sub_id(sub_seed));
            match result {
                EffectOutcome::Failed { exit_code, signal } => {
                    assert_eq!(*exit_code, Some(7), "stage 1's exit dominates");
                    assert_eq!(*signal, Some(15), "stage 0's cascade SIGTERM surfaces");
                }
                EffectOutcome::Ok => panic!("expected Failed, got Ok"),
            }
        }
        other => panic!("expected EffectComplete, got {other:?}"),
    }
    harness.shutdown();
}

/// Happy-path baseline. A pipe of two cleanly-completing stages
/// (`echo hi | cat`) reports `Ok` — pins that the parallel design
/// doesn't break the no-failure case. The kernel's SIGPIPE chain on
/// stage 0's natural exit propagates EOF to stage 1's stdin without
/// the cascade firing.
#[test]
fn pipe_drain_all_ok_completes_ok() {
    let mut harness = Harness::new(2);
    let program = pipe_program(vec![
        vec!["/bin/echo".into(), "hi".into()],
        vec!["/bin/cat".into()],
    ]);
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
        Input::EffectComplete { result, .. } => {
            assert_eq!(*result, EffectOutcome::Ok);
        }
        other => panic!("expected EffectComplete, got {other:?}"),
    }
    harness.shutdown();
}
