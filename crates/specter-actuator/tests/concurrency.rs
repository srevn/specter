//! Integration: global concurrency cap + per-Sub serialization with
//! real subprocesses.

#![cfg(unix)]
#![allow(clippy::redundant_clone, clippy::useless_conversion)]

mod common;

use common::*;
use specter_core::Input;
use std::time::{Duration, Instant};

#[test]
fn cap_two_with_four_distinct_subs_serializes_two_at_a_time() {
    // With cap=2 and four distinct (sub, resource) pairs, the wall-clock
    // time should be ~2x a single child's runtime, not ~4x.
    let mut h = Harness::new(2);
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    let script = "sleep 0.2".to_string();
    let start = Instant::now();
    for i in 0..4 {
        h.submit(perfile_effect(
            i + 1,
            i + 1,
            i + 1,
            u64::from(i + 1),
            vec!["/bin/sh".into(), "-c".into(), script.clone()],
            cwd.clone(),
        ));
    }
    h.wait_for_effect_completes(4, Duration::from_secs(5));
    let elapsed = start.elapsed();
    h.shutdown();
    // 4 × 200ms / cap=2 = 400ms minimum. Allow generous slack for CI.
    assert!(
        elapsed >= Duration::from_millis(400),
        "elapsed {elapsed:?} < 400ms — cap not enforced"
    );
    assert!(
        elapsed < Duration::from_millis(1500),
        "elapsed {elapsed:?} suggests serial execution under cap=2"
    );
}

#[test]
fn per_sub_serializes_per_file_burst() {
    // Same Sub, four distinct resources → 4 PerFile keys; cap=4 so the
    // global gate isn't binding. Per-Sub semaphore (1) serializes all 4
    // anyway. Expect wall-time ~4x a single child's runtime.
    let mut h = Harness::new(4);
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    let script = "sleep 0.15".to_string();
    let start = Instant::now();
    for i in 0..4 {
        h.submit(perfile_effect(
            42, // same Sub
            42,
            i + 1,
            u64::from(i + 1),
            vec!["/bin/sh".into(), "-c".into(), script.clone()],
            cwd.clone(),
        ));
    }
    h.wait_for_effect_completes(4, Duration::from_secs(5));
    let elapsed = start.elapsed();
    h.shutdown();
    // Serial: 4 × 150ms = 600ms minimum.
    assert!(
        elapsed >= Duration::from_millis(550),
        "elapsed {elapsed:?} < ~600ms — per-Sub gate not enforced"
    );
}

#[test]
fn distinct_subs_run_concurrently_under_cap() {
    // Distinct Subs and cap=4 → all 4 run concurrently; wall-time ~1x.
    let mut h = Harness::new(4);
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    let script = "sleep 0.15".to_string();
    let start = Instant::now();
    for i in 0..4 {
        h.submit(perfile_effect(
            i + 1, // distinct Subs
            i + 1,
            i + 1,
            u64::from(i + 1),
            vec!["/bin/sh".into(), "-c".into(), script.clone()],
            cwd.clone(),
        ));
    }
    let completions = h.wait_for_effect_completes(4, Duration::from_secs(5));
    let elapsed = start.elapsed();
    h.shutdown();
    assert_eq!(completions.len(), 4);
    // Concurrent: 1 × 150ms ≈ 150–300ms allowing CI slack.
    assert!(
        elapsed < Duration::from_millis(550),
        "elapsed {elapsed:?} suggests serialization despite distinct Subs"
    );
}

#[test]
fn engine_receives_one_effect_complete_per_spawn() {
    // Sanity: the actuator emits exactly one EffectComplete per Effect
    // (regardless of coalescing victims); the wait_for helper counts.
    let mut h = Harness::new(4);
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    let script = "exit 0".to_string();
    h.submit(perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), script.clone()],
        cwd.clone(),
    ));
    h.submit(perfile_effect(
        2,
        2,
        2,
        2,
        vec!["/bin/sh".into(), "-c".into(), script.clone()],
        cwd.clone(),
    ));
    let completions = h.wait_for_effect_completes(2, Duration::from_secs(5));
    h.shutdown();
    assert_eq!(completions.len(), 2);
    for c in &completions {
        assert!(matches!(c, Input::EffectComplete { .. }));
    }
}
