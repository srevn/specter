//! Integration: env vars + argv substitution + tmp diff file.

#![cfg(unix)]

mod common;

use common::*;
use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::{
    CorrelationId, Diff, Effect, EffectOutcome, EffectTarget, EntryKind, EntryRef, FsIdentity,
    Input,
};
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

/// Build a `setup` closure that rewrites a PerFile [`Effect`]'s segment
/// so the resolver-derived `relative()` / `target_path()` reflect `seg`.
/// `relative` is no longer a stored field — it is the
/// [`EffectTarget::PerFile`] segment — so the closure rebuilds the
/// target in place, preserving the fixture's resource and (empty) diff.
fn set_relative(seg: &'static str) -> impl FnOnce(&mut Effect) {
    move |e: &mut Effect| {
        let resource = e.sort_key().1;
        let diff = Arc::clone(e.diff().expect("perfile_effect carries a diff"));
        e.target = EffectTarget::PerFile {
            resource,
            segment: CompactString::from(seg),
            diff,
        };
    }
}

/// Spawn a script that writes the given env-var to a captured file, then
/// asserts on its content.
///
/// The actuator-side resolver derives every `SPECTER_*` env value from the
/// substitution-domain projection on [`Effect`] — there is no `Effect.env`
/// field to push pairs onto. To assert "the child sees `name=expected`"
/// the helper takes a `setup` closure that mutates the resolver's
/// *inputs* (e.g., set `e.target_relative` to assert on `SPECTER_PATH`,
/// since `target_path` is now derived as `anchor_path.join(target_relative)`);
/// the resolver's render-time output is what the spawned process observes
/// via `getenv`.
///
/// `expected_for` derives the expected value from the spawn cwd so that
/// `SPECTER_ANCHOR` (which doubles as cwd) can assert against the
/// tempdir's actual path; cases whose expected value is independent of
/// the cwd ignore the argument.
fn assert_env_var_received(
    name: &str,
    expected_for: impl FnOnce(&Path) -> String,
    setup: impl FnOnce(&mut Effect),
) {
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
    setup(&mut e);
    let expected = expected_for(dir.path());
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete(c) => assert_eq!(c.outcome, EffectOutcome::Ok),
        other => panic!("expected Ok; got {other:?}"),
    }
    h.shutdown();
    let captured = std::fs::read_to_string(&out_path).expect("read captured");
    assert_eq!(captured, expected);
}

#[test]
fn child_receives_specter_path() {
    // `SPECTER_PATH` mirrors the resolver-derived `target_path` —
    // `anchor_path.join(target_relative)`. The fixture defaults
    // `anchor_path = cwd` (the tempdir) and `target_relative = ""`, so
    // setting `target_relative = "src/a.c"` produces
    // `SPECTER_PATH = <tempdir>/src/a.c`. The cwd-derivation
    // `expected_for` keeps the assertion symmetrical with the actual
    // join the resolver performs.
    assert_env_var_received(
        "SPECTER_PATH",
        |dir| dir.join("src/a.c").to_string_lossy().into_owned(),
        set_relative("src/a.c"),
    );
}

#[test]
fn child_receives_specter_anchor() {
    // `SPECTER_ANCHOR` mirrors `Effect.anchor_path`, which doubles as
    // the spawn cwd. Asserting against an arbitrary string (the prior
    // helper's "/abs/proj" placeholder) would set `cwd = /abs/proj`
    // and the spawn would fail with ENOENT. The integration here
    // verifies the cwd-anchor coupling end-to-end.
    assert_env_var_received(
        "SPECTER_ANCHOR",
        |dir| dir.to_string_lossy().into_owned(),
        |_e| {},
    );
}

#[test]
fn child_receives_specter_relative_path() {
    assert_env_var_received(
        "SPECTER_RELATIVE_PATH",
        |_dir| "src/a.c".to_owned(),
        set_relative("src/a.c"),
    );
}

#[test]
fn child_receives_specter_correlation_decimal() {
    assert_env_var_received(
        "SPECTER_CORRELATION",
        |_dir| "12345".to_owned(),
        |e| e.correlation = CorrelationId::from(12345),
    );
}

#[test]
fn child_receives_specter_forced_zero() {
    assert_env_var_received(
        "SPECTER_FORCED",
        |_dir| "0".to_owned(),
        |e| e.forced = false,
    );
}

#[test]
fn child_receives_specter_created_newline_separated() {
    // End-to-end witness for the diff-derived list env vars: a populated
    // `Effect.diff.created` lands in the spawned child's `SPECTER_CREATED`
    // newline-joined, no trailing newline. Resolver-level rendering is
    // pinned by the unit tests; this test pins that the
    // `cmd.envs(env.iter()...)` plumbing in `OsSpawner` propagates the
    // value byte-for-byte through `execve`. The four sibling vars
    // (`DELETED`/`MODIFIED`/`RENAMED_FROM`/`RENAMED_TO`) ride the same
    // plumbing — no separate integration test is justified.
    let dir = tempfile::tempdir().expect("tempdir");
    let out_path = dir.path().join("out");
    let cwd = dir.path().to_path_buf();
    let script = format!(
        "printf '%s' \"$SPECTER_CREATED\" > {out}",
        out = out_path.display()
    );
    let mut h = Harness::new(2);
    let diff = Arc::new(Diff {
        created: smallvec![
            EntryRef {
                segment: CompactString::from("a.rs"),
                kind: EntryKind::File,
                fs_id: FsIdentity::synthetic(1, 0),
            },
            EntryRef {
                segment: CompactString::from("src/b.rs"),
                kind: EntryKind::File,
                fs_id: FsIdentity::synthetic(2, 0),
            },
        ],
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
    e.target = EffectTarget::PerFile {
        resource: e.sort_key().1,
        segment: CompactString::from(e.relative()),
        diff,
    };
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete(c) => assert_eq!(c.outcome, EffectOutcome::Ok),
        other => panic!("expected Ok; got {other:?}"),
    }
    h.shutdown();
    let captured = std::fs::read_to_string(&out_path).expect("read captured");
    assert_eq!(captured, "a.rs\nsrc/b.rs");
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
            fs_id: FsIdentity::synthetic(7, 0),
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
    e.target = EffectTarget::PerFile {
        resource: e.sort_key().1,
        segment: CompactString::from(e.relative()),
        diff,
    };
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    let dbg_content =
        std::fs::read_to_string(&dbg_path).unwrap_or_else(|e| format!("read dbg: {e}"));
    match &completions[0] {
        Input::EffectComplete(c) => {
            assert_eq!(c.outcome, EffectOutcome::Ok, "dbg: {dbg_content}");
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
    // Subtree is the only shape that can carry no diff: a PerFile
    // effect's diff is mandatory, so "no diff" is exercised here via a
    // diff-less Subtree fire.
    let e = subtree_effect(1, 1, 1, vec!["/bin/sh".into(), "-c".into(), script], cwd);
    h.submit(e);
    let completions = h.wait_for_effect_completes(1, Duration::from_secs(5));
    match &completions[0] {
        Input::EffectComplete(c) => assert_eq!(c.outcome, EffectOutcome::Ok),
        other => panic!("expected Ok; got {other:?}"),
    }
    h.shutdown();
    let captured = std::fs::read_to_string(&out_path).expect("read captured");
    assert_eq!(
        captured, "",
        "SPECTER_DIFF_PATH unset when the effect carries no diff"
    );
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
            fs_id: FsIdentity::synthetic(1, 0),
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
    e.target = EffectTarget::PerFile {
        resource: e.sort_key().1,
        segment: CompactString::from(e.relative()),
        diff,
    };
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
