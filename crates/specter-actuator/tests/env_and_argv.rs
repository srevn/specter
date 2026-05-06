//! Integration: env vars + argv substitution + tmp diff file.

#![cfg(unix)]

mod common;

use common::*;
use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::{Diff, EffectOutcome, EntryKind, EntryRef, Input};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Spawn a script that writes the given env-var to a captured file, then
/// asserts on its content.
fn assert_env_var_received(name: &str, expected: &str) {
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out");
    let script = format!(
        "printf '%s' \"${{{name}}}\" > {out}",
        name = name,
        out = out_path.display()
    );
    let mut h = Harness::new(2);
    let mut e = perfile_effect(
        1,
        1,
        1,
        1,
        vec!["/bin/sh".into(), "-c".into(), script],
        dir.path().to_path_buf(),
    );
    e.env.push((name.to_owned(), expected.to_owned()));
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete { result, .. } => assert_eq!(*result, EffectOutcome::Ok),
        other => panic!("expected Ok; got {other:?}"),
    }
    h.shutdown();
    let captured = std::fs::read_to_string(&out_path).expect("read captured");
    assert_eq!(captured, expected);
}

#[test]
fn child_receives_specter_path() {
    assert_env_var_received("SPECTER_PATH", "/abs/proj/src/a.c");
}

#[test]
fn child_receives_specter_anchor() {
    assert_env_var_received("SPECTER_ANCHOR", "/abs/proj");
}

#[test]
fn child_receives_specter_rel_path() {
    assert_env_var_received("SPECTER_REL_PATH", "src/a.c");
}

#[test]
fn child_receives_specter_correlation_decimal() {
    assert_env_var_received("SPECTER_CORRELATION", "12345");
}

#[test]
fn child_receives_specter_forced_zero() {
    assert_env_var_received("SPECTER_FORCED", "0");
}

#[test]
fn child_receives_specter_diff_path_when_diff_present() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out");
    let dbg_path = dir.path().join("out.dbg");
    let cwd = dir.path().to_path_buf();
    // Script: record the path the child saw, then copy file contents.
    let script = format!(
        "printf 'path=%s\\nexists=' \"$SPECTER_DIFF_PATH\" > {dbg}; \
         [ -f \"$SPECTER_DIFF_PATH\" ] && printf 'yes' >> {dbg} || printf 'no' >> {dbg}; \
         cat \"$SPECTER_DIFF_PATH\" > {out}",
        dbg = dbg_path.display(),
        out = out_path.display(),
    );
    let mut h = Harness::new(2);
    let diff = Arc::new(Diff {
        created: smallvec![EntryRef {
            segment: CompactString::from("a.rs"),
            kind: EntryKind::File,
            inode: 7,
        }],
        ..Default::default()
    });
    // Use a per-test unique correlation to avoid colliding on the
    // tmp file path with parallel-running tests in the same binary
    // (path = `specter-{pid}-{corr}.diff`; pid is shared per-binary).
    let mut e = perfile_effect(
        1,
        1,
        1,
        next_corr(),
        vec!["/bin/sh".into(), "-c".into(), script],
        cwd,
    );
    e.diff = Some(diff);
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    let dbg_content =
        std::fs::read_to_string(&dbg_path).unwrap_or_else(|e| format!("read dbg: {e}"));
    match &completions[0] {
        Input::EffectComplete { result, .. } => {
            assert_eq!(*result, EffectOutcome::Ok, "dbg: {dbg_content}");
        }
        other => panic!("expected Ok; got {other:?}; dbg: {dbg_content}"),
    }
    h.shutdown();
    let captured = std::fs::read_to_string(&out_path).expect("read captured");
    assert_eq!(captured, "created\ta.rs\t7\n");
}

#[test]
fn child_does_not_receive_specter_diff_path_without_diff() {
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out");
    let cwd = dir.path().to_path_buf();
    // Script writes the variable's value (or an empty string if unset).
    let script = format!(
        "printf '%s' \"${{SPECTER_DIFF_PATH-}}\" > {out}",
        out = out_path.display()
    );
    let mut h = Harness::new(2);
    let e = perfile_effect(1, 1, 1, 1, vec!["/bin/sh".into(), "-c".into(), script], cwd);
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete { result, .. } => assert_eq!(*result, EffectOutcome::Ok),
        other => panic!("expected Ok; got {other:?}"),
    }
    h.shutdown();
    let captured = std::fs::read_to_string(&out_path).expect("read captured");
    assert_eq!(captured, "", "SPECTER_DIFF_PATH unset when diff is None");
}

#[test]
fn tmp_diff_file_cleaned_up_after_completion() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path_record = dir.path().join("path");
    let cwd = dir.path().to_path_buf();
    // Echo the path so we can read it post-completion.
    let script = format!(
        "printf '%s' \"$SPECTER_DIFF_PATH\" > {p}",
        p = path_record.display()
    );
    let mut h = Harness::new(2);
    let diff = Arc::new(Diff {
        created: smallvec![EntryRef {
            segment: CompactString::from("a"),
            kind: EntryKind::File,
            inode: 1,
        }],
        ..Default::default()
    });
    let mut e = perfile_effect(
        1,
        1,
        1,
        next_corr(),
        vec!["/bin/sh".into(), "-c".into(), script],
        cwd,
    );
    e.diff = Some(diff);
    h.submit(e);
    h.wait_for_effect_completes(1, Duration::from_secs(5));
    h.shutdown();
    let recorded_path = std::fs::read_to_string(&path_record).expect("read path");
    assert!(!recorded_path.is_empty(), "child captured a path");
    // Wait briefly for the wait thread's cleanup. The shutdown above
    // joins the controller, but the wait thread runs cleanup before
    // sending Reaped, so by the time EffectComplete arrives the file
    // is gone.
    let p = Path::new(&recorded_path);
    assert!(!p.exists(), "tmp file removed after wait");
}
