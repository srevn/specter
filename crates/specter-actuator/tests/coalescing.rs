//! Integration: distinct-key parallelism under real subprocesses.

#![cfg(unix)]
#![allow(clippy::redundant_clone, clippy::useless_conversion)]

mod common;

use common::*;
use specter_core::{EffectOutcome, Input};
use std::time::Duration;

#[test]
fn distinct_keys_run_concurrently_under_cap() {
    let mut h = Harness::new(4);
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    let mk = |marker: &str| format!("touch {}/{} && sleep 0.1", dir.path().display(), marker);
    // Two distinct keys.
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), mk("a")],
        cwd.clone(),
    ));
    h.submit(perfile_effect(
        2,
        2,
        2,
        2,
        vec!["/bin/sh".into(), "-c".into(), mk("b")],
        cwd.clone(),
    ));
    let completions = h.wait_for_effect_completes(2, Duration::from_secs(5));
    for completion in &completions {
        match completion {
            Input::EffectComplete(c) => assert_eq!(c.outcome, EffectOutcome::Ok),
            other => panic!("expected EffectComplete; got {other:?}"),
        }
    }
    h.shutdown();
    assert!(dir.path().join("a").exists());
    assert!(dir.path().join("b").exists());
}
