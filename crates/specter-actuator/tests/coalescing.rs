//! Integration: Latest coalesce semantics under real subprocesses.

#![cfg(unix)]
#![allow(clippy::redundant_clone, clippy::useless_conversion)]

mod common;

use common::*;
use specter_core::{EffectOutcome, Input};
use std::time::Duration;

#[test]
fn three_effects_on_one_key_run_first_and_last() {
    // Submit three effects on the same DedupKey rapidly. The first runs
    // (eff1); while it's running, eff2 lands as pending; eff3 replaces
    // eff2 (Latest coalesce). On reap, eff3 runs. Net: 2 EffectComplete
    // arrive at the engine.
    let mut h = Harness::new(4);
    // Use marker files in cwd to verify which effects ran.
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    let script = |marker: &str| format!("touch {}/{} && sleep 0.1", dir.path().display(), marker);
    h.submit(perfile_effect(
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), script("eff1")],
        cwd.clone(),
    ));
    // Brief wait so eff1 actually starts before we submit eff2.
    std::thread::sleep(Duration::from_millis(50));
    h.submit(perfile_effect(
        1,
        1,
        2,
        vec!["/bin/sh".into(), "-c".into(), script("eff2")],
        cwd.clone(),
    ));
    h.submit(perfile_effect(
        1,
        1,
        3,
        vec!["/bin/sh".into(), "-c".into(), script("eff3")],
        cwd.clone(),
    ));
    let completions = h.wait_for_effect_completes(2, Duration::from_secs(5));
    for c in &completions {
        match c {
            Input::EffectComplete { result, .. } => assert_eq!(*result, EffectOutcome::Ok),
            other => panic!("expected EffectComplete; got {other:?}"),
        }
    }
    h.shutdown();
    // Markers left by the children:
    assert!(dir.path().join("eff1").exists(), "eff1 ran");
    assert!(dir.path().join("eff3").exists(), "eff3 ran (Latest)");
    assert!(!dir.path().join("eff2").exists(), "eff2 was coalesced away");
}

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
        vec!["/bin/sh".into(), "-c".into(), mk("a")],
        cwd.clone(),
    ));
    h.submit(perfile_effect(
        2,
        2,
        2,
        vec!["/bin/sh".into(), "-c".into(), mk("b")],
        cwd.clone(),
    ));
    let completions = h.wait_for_effect_completes(2, Duration::from_secs(5));
    for c in &completions {
        match c {
            Input::EffectComplete { result, .. } => assert_eq!(*result, EffectOutcome::Ok),
            other => panic!("expected EffectComplete; got {other:?}"),
        }
    }
    h.shutdown();
    assert!(dir.path().join("a").exists());
    assert!(dir.path().join("b").exists());
}
