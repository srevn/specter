//! Integration: spawn-failure / non-zero exit / signal-killed paths.

#![cfg(unix)]

mod common;

use common::*;
use specter_core::{EffectOutcome, Input, Termination};
use std::path::PathBuf;
use std::time::Duration;

#[test]
fn non_existent_command_returns_failed() {
    let mut h = Harness::new(2);
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/no/such/binary".into()],
        std::env::temp_dir(),
    ));
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(2));
    match &completions[0] {
        Input::EffectComplete(c) => assert!(matches!(
            c.outcome,
            EffectOutcome::Failed(Termination::Internal)
        )),
        other => panic!("expected EffectComplete::Failed; got {other:?}"),
    }
    h.shutdown();
}

#[test]
fn non_zero_exit_returns_failed_with_exit_code() {
    let mut h = Harness::new(2);
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), "exit 42".into()],
        std::env::temp_dir(),
    ));
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete(c) => match &c.outcome {
            EffectOutcome::Failed(Termination::Exit(exit_code)) => {
                assert_eq!(*exit_code, 42);
            }
            other => panic!("expected Failed(Exit(42)); got {other:?}"),
        },
        other => panic!("expected EffectComplete; got {other:?}"),
    }
    h.shutdown();
}

#[test]
fn zero_exit_returns_ok() {
    let mut h = Harness::new(2);
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), "exit 0".into()],
        std::env::temp_dir(),
    ));
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete(c) => assert_eq!(c.outcome, EffectOutcome::Ok),
        other => panic!("expected EffectComplete; got {other:?}"),
    }
    h.shutdown();
}

#[test]
fn nonexistent_cwd_returns_failed() {
    let mut h = Harness::new(2);
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), "exit 0".into()],
        PathBuf::from("/this/path/does/not/exist"),
    ));
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(2));
    match &completions[0] {
        Input::EffectComplete(c) => assert!(matches!(
            c.outcome,
            EffectOutcome::Failed(Termination::Internal)
        )),
        other => panic!("expected EffectComplete::Failed; got {other:?}"),
    }
    h.shutdown();
}

#[test]
fn empty_argv_returns_failed() {
    let mut h = Harness::new(2);
    h.submit(perfile_effect(1, 1, 1, 1, vec![], std::env::temp_dir()));
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(2));
    match &completions[0] {
        Input::EffectComplete(c) => assert!(matches!(
            c.outcome,
            EffectOutcome::Failed(Termination::Internal)
        )),
        other => panic!("expected Failed; got {other:?}"),
    }
    h.shutdown();
}
