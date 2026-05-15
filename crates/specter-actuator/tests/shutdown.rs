//! Integration: SIGTERM-then-SIGKILL within 5s grace; pending dropped.

#![cfg(unix)]
#![allow(clippy::match_wildcard_for_single_variants)]

mod common;

use common::*;
use specter_core::{EffectOutcome, Input, Termination};
use std::time::{Duration, Instant};

#[test]
fn shutdown_sigkills_term_resistant_child() {
    // Child traps SIGTERM (ignores it) and loops forever. Shutdown
    // forces SIGKILL after the 5s grace.
    let mut h = Harness::new(2);
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    // Trap TERM (no-op) and loop. After SIGKILL it dies signal=9.
    let script = "trap '' TERM; while :; do sleep 0.05; done".to_string();
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), script],
        cwd,
    ));
    std::thread::sleep(Duration::from_millis(100));
    let start = Instant::now();
    h.shutdown_tx.send(()).expect("shutdown");
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(8));
    let elapsed = start.elapsed();
    if let Some(j) = h.join.take() {
        j.join().expect("controller join");
    }
    assert_eq!(completions.len(), 1);
    match &completions[0] {
        Input::EffectComplete { result, .. } => match result {
            EffectOutcome::Failed(
                Termination::Signal(signal)
                | Termination::PipeMixed {
                    first_signal: signal,
                    ..
                },
            ) => {
                assert_eq!(*signal, 9, "SIGKILL delivered");
            }
            other => panic!("expected Failed{{signal=9}}; got {other:?}"),
        },
        other => panic!("expected EffectComplete; got {other:?}"),
    }
    // Within the 5s grace + a bit (CI slack).
    assert!(
        elapsed >= Duration::from_secs(4),
        "shutdown returned too quickly ({elapsed:?}) — grace not honored"
    );
    assert!(
        elapsed < Duration::from_secs(7),
        "shutdown took too long ({elapsed:?}) — SIGKILL not delivered"
    );
}
