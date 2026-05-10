//! Sibling unit tests for [`super::resolve`]. Pure data work — all
//! fixtures are inline; no I/O.

#![allow(
    clippy::items_after_statements,
    clippy::missing_const_for_fn,
    clippy::too_many_lines
)]

use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::{
    ArgPart, ArgTemplate, CommandTemplate, CorrelationId, DedupKey, Diff, Effect, EffectScope,
    EntryKind, EntryRef, Placeholder, Rename, ResourceId, ResourceKind,
};
use std::path::Path;
use std::sync::Arc;

fn make_effect(
    sub_name: &str,
    scope: EffectScope,
    argv: Vec<ArgTemplate>,
    anchor_path: &Path,
    target_path: &Path,
    target_relative: &str,
    forced: bool,
    correlation: CorrelationId,
    diff: Option<Arc<Diff>>,
) -> Effect {
    Effect {
        key: DedupKey::default(),
        target: ResourceId::default(),
        forced,
        correlation,
        diff,
        capture_output: false,
        sub_name: CompactString::from(sub_name),
        command: Arc::new(CommandTemplate::new(argv)),
        scope,
        anchor_path: anchor_path.to_path_buf(),
        anchor_kind: ResourceKind::Dir,
        target_path: target_path.to_path_buf(),
        target_relative: CompactString::from(target_relative),
        exclude: Arc::from(Vec::<CompactString>::new()),
    }
}

fn lit(s: &str) -> ArgPart {
    ArgPart::literal(s)
}
fn ph(p: Placeholder) -> ArgPart {
    ArgPart::Placeholder(p)
}
fn arg(parts: Vec<ArgPart>) -> ArgTemplate {
    ArgTemplate::new(parts)
}

fn entry_ref(seg: &str, inode: u64) -> EntryRef {
    EntryRef {
        segment: CompactString::from(seg),
        kind: EntryKind::File,
        inode,
    }
}

// ---------- argv substitution ----------

#[test]
fn resolve_simple_literal_passes_through() {
    let e = make_effect(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("make")])],
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["make".to_string()]);
}

#[test]
fn resolve_with_path_placeholder() {
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Path)])],
        Path::new("/proj"),
        Path::new("/proj/src/a.c"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = super::resolve_effect(&e);
    assert_eq!(
        cmd.argv,
        vec!["fmt".to_string(), "/proj/src/a.c".to_string()]
    );
}

#[test]
fn resolve_with_relative_placeholder() {
    let e = make_effect(
        "log",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("log")]), arg(vec![ph(Placeholder::Relative)])],
        Path::new("/proj"),
        Path::new("/proj/src/a.c"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["log".to_string(), "src/a.c".to_string()]);
}

#[test]
fn resolve_with_anchor_placeholder() {
    let e = make_effect(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("build")]), arg(vec![ph(Placeholder::Anchor)])],
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["build".to_string(), "/proj".to_string()]);
}

#[test]
fn resolve_with_concatenated_literal_and_placeholder() {
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--input="), ph(Placeholder::Path)])],
        Path::new("/proj"),
        Path::new("/proj/a.c"),
        "a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["--input=/proj/a.c".to_string()]);
}

#[test]
fn resolve_with_created_expands_to_n_argv() {
    let diff = Diff {
        created: smallvec![
            entry_ref("a.rs", 1),
            entry_ref("b.rs", 2),
            entry_ref("c.rs", 3)
        ],
        ..Default::default()
    };
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _env) = super::resolve_effect(&e);
    assert_eq!(
        cmd.argv,
        vec![
            "fmt".to_string(),
            "a.rs".to_string(),
            "b.rs".to_string(),
            "c.rs".to_string()
        ]
    );
}

#[test]
fn resolve_with_deleted_expands_to_n_argv() {
    let diff = Diff {
        deleted: smallvec![entry_ref("x", 9), entry_ref("y", 10)],
        ..Default::default()
    };
    let e = make_effect(
        "rmlog",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Deleted)])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["x".to_string(), "y".to_string()]);
}

#[test]
fn resolve_with_modified_expands_to_n_argv() {
    let diff = Diff {
        modified: smallvec![entry_ref("m.rs", 1)],
        ..Default::default()
    };
    let e = make_effect(
        "lint",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Modified)])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["m.rs".to_string()]);
}

#[test]
fn resolve_with_renamed_from_and_to_expands_independently() {
    let diff = Diff {
        renamed: smallvec![
            Rename {
                from: entry_ref("a", 1),
                to: entry_ref("A", 1),
            },
            Rename {
                from: entry_ref("b", 2),
                to: entry_ref("B", 2),
            },
        ],
        ..Default::default()
    };
    let e = make_effect(
        "mv",
        EffectScope::PerStableFile,
        vec![
            arg(vec![lit("mv")]),
            arg(vec![ph(Placeholder::RenamedFrom)]),
            arg(vec![ph(Placeholder::RenamedTo)]),
        ],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(
        cmd.argv,
        vec![
            "mv".to_string(),
            "a".to_string(),
            "b".to_string(),
            "A".to_string(),
            "B".to_string()
        ]
    );
}

#[test]
fn resolve_with_diff_placeholder_and_no_diff_yields_zero_args() {
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["fmt".to_string()]);
}

#[test]
fn resolve_with_empty_diff_placeholder_yields_zero_args() {
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(Diff::default())),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["fmt".to_string()]);
}

#[test]
fn resolve_with_multivalue_in_separate_args_emits_literals_as_standalone_slots() {
    let diff = Diff {
        created: smallvec![entry_ref("a", 1), entry_ref("b", 2)],
        ..Default::default()
    };
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![
            arg(vec![lit("pre")]),
            arg(vec![ph(Placeholder::Created)]),
            arg(vec![lit("post")]),
        ],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(
        cmd.argv,
        vec![
            "pre".to_string(),
            "a".to_string(),
            "b".to_string(),
            "post".to_string()
        ]
    );
}

#[test]
fn resolve_with_multivalue_having_prefix_literal_tiles_per_value() {
    let diff = Diff {
        created: smallvec![entry_ref("a", 1), entry_ref("b", 2)],
        ..Default::default()
    };
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert_eq!(cmd.argv, vec!["--out=a".to_string(), "--out=b".to_string()]);
}

#[test]
fn resolve_with_multivalue_having_prefix_and_empty_diff_yields_zero_slots() {
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(Diff::default())),
    );
    let (cmd, _) = super::resolve_effect(&e);
    assert!(cmd.argv.is_empty());
}

// ---------- env vars ----------

#[test]
fn env_contains_specter_path_for_subtree_root() {
    let e = make_effect(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("make")])],
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    let path = env.iter().find(|(k, _)| k == "SPECTER_PATH").unwrap();
    assert_eq!(path.1, "/proj");
}

#[test]
fn env_contains_specter_path_for_per_stable_file() {
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")])],
        Path::new("/proj"),
        Path::new("/proj/a.c"),
        "a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    let path = env.iter().find(|(k, _)| k == "SPECTER_PATH").unwrap();
    assert_eq!(path.1, "/proj/a.c");
}

#[test]
fn env_specter_relative_path_empty_for_subtree_root() {
    let e = make_effect(
        "b",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("x")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter()
            .find(|(k, _)| k == "SPECTER_RELATIVE_PATH")
            .unwrap()
            .1,
        ""
    );
}

#[test]
fn env_specter_relative_path_for_per_stable_file() {
    let e = make_effect(
        "f",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("x")])],
        Path::new("/p"),
        Path::new("/p/src/a.c"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter()
            .find(|(k, _)| k == "SPECTER_RELATIVE_PATH")
            .unwrap()
            .1,
        "src/a.c"
    );
}

#[test]
fn env_specter_anchor_for_both_scopes() {
    for scope in [EffectScope::SubtreeRoot, EffectScope::PerStableFile] {
        let e = make_effect(
            "x",
            scope,
            vec![arg(vec![lit("y")])],
            Path::new("/anchor/dir"),
            Path::new("/anchor/dir"),
            "",
            false,
            CorrelationId(1),
            None,
        );
        let (_, env) = super::resolve_effect(&e);
        let v = env.iter().find(|(k, _)| k == "SPECTER_ANCHOR").unwrap();
        assert_eq!(v.1, "/anchor/dir", "scope = {scope:?}");
    }
}

#[test]
fn env_specter_watch_uses_sub_name() {
    let e = make_effect(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_WATCH").unwrap().1,
        "build"
    );
}

#[test]
fn env_specter_forced_zero_when_unforced() {
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_FORCED").unwrap().1,
        "0"
    );
}

#[test]
fn env_specter_forced_one_when_forced() {
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        true,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_FORCED").unwrap().1,
        "1"
    );
}

#[test]
fn env_specter_event_kind_dir_subtree_for_subtree_root() {
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter()
            .find(|(k, _)| k == "SPECTER_EVENT_KIND")
            .unwrap()
            .1,
        "dir-subtree"
    );
}

#[test]
fn env_specter_event_kind_file_for_per_stable_file() {
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p/a"),
        "a",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter()
            .find(|(k, _)| k == "SPECTER_EVENT_KIND")
            .unwrap()
            .1,
        "file"
    );
}

#[test]
fn env_specter_correlation_decimal_for_v1() {
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(42),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    assert_eq!(
        env.iter()
            .find(|(k, _)| k == "SPECTER_CORRELATION")
            .unwrap()
            .1,
        "42"
    );
}

#[test]
fn env_does_not_contain_specter_diff_path() {
    let diff = Diff {
        created: smallvec![entry_ref("a", 1)],
        ..Default::default()
    };
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p/a"),
        "a",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (_, env) = super::resolve_effect(&e);
    assert!(env.iter().all(|(k, _)| k != "SPECTER_DIFF_PATH"));
}

#[test]
fn env_order_is_alphabetical() {
    let e = make_effect(
        "watch",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e);
    let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "SPECTER_ANCHOR",
            "SPECTER_CORRELATION",
            "SPECTER_EVENT_KIND",
            "SPECTER_FORCED",
            "SPECTER_PATH",
            "SPECTER_RELATIVE_PATH",
            "SPECTER_WATCH",
        ]
    );
}
