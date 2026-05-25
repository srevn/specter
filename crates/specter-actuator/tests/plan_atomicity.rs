//! Integration regression: plan-atomicity across hot-reload-shaped
//! coalesce.
//!
//! The reshape introduced `Effect.program: Arc<ActionProgram>` and a
//! per-instruction actuator advance loop. Operators can change a
//! watch's `actions` list via SIGHUP while a plan is in flight; the
//! engine emits a new Effect with the new program, which lands in
//! the actuator's per-slot `pending`. The invariant under test:
//! **once started, a plan runs all its ops before `pending` fires**,
//! regardless of new submits. Equivalently: `Effect.program` is a
//! frozen snapshot — the in-flight step's `effect.program.ops()[N+1]` is
//! sourced from the same `Arc` installed at plan start, never from a
//! later submit's program.
//!
//! At the actuator's boundary, "hot reload" manifests as a fresh
//! submit for the same `DedupKey` carrying the new plan. The slot's
//! `running` (and `plan_continue` between steps) is never replaced by
//! coalesce — only `pending` is. This test drives that distinction
//! end-to-end through real subprocesses.

#![cfg(unix)]
#![allow(
    clippy::redundant_clone,
    clippy::similar_names,
    clippy::useless_conversion
)]

mod common;

use common::*;
use specter_core::{EffectOutcome, Input};
use std::time::Duration;

/// A 2-step plan_a is in flight; a fresh submit on the same key
/// arrives with plan_b carrying a different argv. The actuator must
/// finish plan_a (both steps) before plan_b spawns; the marker-file
/// order on disk witnesses the structural Arc-snapshot freeze.
///
/// Concretely:
///
/// 1. Submit Effect 1 (plan_a, 2 steps: write `a0`, then write `a1`).
///    `a0` uses a long-ish `sleep` so we have a deterministic window
///    to fire submit-2 before step 1 spawns.
/// 2. Mid-flight, submit Effect 2 (plan_b, 1 step: write `b0`). It
///    coalesces into the slot's `pending` — `running` (plan_a's step 0)
///    is untouched, and once step 0 reaps, `plan_continue` advances to
///    plan_a's step 1.
/// 3. The on-disk order must be `a0 < a1 < b0`. Any reordering
///    (`a0 < b0 < a1`, or `a1` missing entirely) would mean plan_a's
///    snapshot was replaced mid-flight or pending was dispatched
///    ahead of plan_continue — both are bugs the structural
///    `Arc<ActionProgram>` is meant to prevent.
/// 4. The engine sees exactly two `EffectComplete::Ok` — one per
///    Effect — preserving the per-Effect outstanding accounting.
#[test]
fn pending_submit_during_running_plan_does_not_replace_in_flight_steps() {
    let mut h = Harness::new(nz(4));
    let dir = tempfile::tempdir().expect("tempdir");
    let cwd = dir.path().to_path_buf();
    // `/bin/sleep` accepts decimal-seconds on BSD, GNU coreutils, and
    // macOS — we hand it the literal seconds string so we don't need
    // a `u64 → f64` cast.
    let touch = |marker: &str, sleep_secs: &str| {
        vec![
            "/bin/sh".into(),
            "-c".into(),
            format!(
                "touch {}/{marker} && /bin/sleep {sleep_secs}",
                dir.path().display(),
            ),
        ]
    };
    // plan_a step 0 sleeps long enough that submit-2 is comfortably
    // queued into pending before step 0 reaps. Steps 1 and b0 are
    // brief — we just need the ordering invariant.
    let plan_a = literal_multi_program(vec![touch("a0", "0.2"), touch("a1", "0.05")]);
    let plan_b = literal_multi_program(vec![touch("b0", "0.05")]);

    // Effect 1 (plan_a) and Effect 2 (plan_b) share a DedupKey so
    // submit-2 hits the same slot. Distinct correlations preserve
    // engine-side identity.
    let key_seeds = (7, 7, 7);
    h.submit(perfile_effect_with_program(
        key_seeds.0,
        key_seeds.1,
        key_seeds.2,
        next_corr(),
        plan_a,
        cwd.clone(),
    ));
    // Give step 0 a head start so submit-2 races into `pending`
    // (not into the empty slot that would have spawned plan_b first).
    // 60ms is comfortably less than step 0's 200ms sleep, so step 0
    // is still in flight when we submit.
    std::thread::sleep(Duration::from_millis(60));
    h.submit(perfile_effect_with_program(
        key_seeds.0,
        key_seeds.1,
        key_seeds.2,
        next_corr(),
        plan_b,
        cwd.clone(),
    ));

    let completions = h.wait_for_effect_completes(2, Duration::from_secs(10));
    for completion in &completions {
        match completion {
            Input::EffectComplete(c) => {
                assert_eq!(c.outcome, EffectOutcome::Ok, "every plan terminus is Ok");
            }
            other => panic!("expected EffectComplete; got {other:?}"),
        }
    }
    h.shutdown();

    // All three markers landed.
    let a0 = dir.path().join("a0");
    let a1 = dir.path().join("a1");
    let b0 = dir.path().join("b0");
    assert!(a0.exists(), "plan_a step 0 must run");
    assert!(
        a1.exists(),
        "plan_a step 1 must run (was the in-flight plan)"
    );
    assert!(
        b0.exists(),
        "plan_b step 0 must run after plan_a terminates"
    );

    // Order: a0 < a1 < b0. mtime granularity is finite (HFS+/APFS
    // ~1ns, ext4 ~1ns with `noatime`; tmpfs ~1ns), so the ~50ms gaps
    // we built in are safely above the floor.
    let mtime = |p: &std::path::Path| {
        std::fs::metadata(p)
            .expect("marker stat")
            .modified()
            .expect("modified")
    };
    let plan_a_step0 = mtime(&a0);
    let plan_a_step1 = mtime(&a1);
    let plan_b_step0 = mtime(&b0);
    assert!(
        plan_a_step0 <= plan_a_step1,
        "plan_a step 0 must precede plan_a step 1 \
         ({plan_a_step0:?} vs {plan_a_step1:?})",
    );
    assert!(
        plan_a_step1 <= plan_b_step0,
        "plan_a's full run must precede plan_b — pending never replaces in-flight steps \
         ({plan_a_step1:?} vs {plan_b_step0:?})",
    );
}
