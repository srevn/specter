//! Sibling unit tests for [`crate::effect::resolve`]. Pure data work —
//! all fixtures are inline; no I/O.

#![allow(
    clippy::items_after_statements,
    clippy::missing_const_for_fn,
    clippy::too_many_lines
)]

use super::resolve::resolve_effect;
use crate::diff::{Diff, EntryRef, Rename};
use crate::effect::CorrelationId;
use crate::ids::{ProfileId, SubId};
use crate::snapshot::EntryKind;
use crate::sub::{ArgPart, ArgTemplate, ClassSet, CommandTemplate, EffectScope, Placeholder, Sub};
use compact_str::CompactString;
use smallvec::smallvec;
use std::path::Path;
use std::time::Duration;

fn sub(name: &str, scope: EffectScope, argv: Vec<ArgTemplate>) -> Sub {
    Sub::new(
        SubId::default(),
        name,
        ProfileId::default(),
        CommandTemplate::new(argv),
        scope,
        Duration::from_millis(100),
        Duration::from_secs(6),
        ClassSet::EMPTY,
    )
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
    let s = sub(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("make")])],
    );
    let (cmd, _env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(cmd.argv, vec!["make".to_string()]);
}

#[test]
fn resolve_with_path_placeholder() {
    let s = sub(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Path)])],
    );
    let (cmd, _env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj/src/a.c"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(
        cmd.argv,
        vec!["fmt".to_string(), "/proj/src/a.c".to_string()]
    );
}

#[test]
fn resolve_with_rel_placeholder() {
    let s = sub(
        "log",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("log")]), arg(vec![ph(Placeholder::Rel)])],
    );
    let (cmd, _env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj/src/a.c"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(cmd.argv, vec!["log".to_string(), "src/a.c".to_string()]);
}

#[test]
fn resolve_with_anchor_placeholder() {
    let s = sub(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("build")]), arg(vec![ph(Placeholder::Anchor)])],
    );
    let (cmd, _env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(cmd.argv, vec!["build".to_string(), "/proj".to_string()]);
}

#[test]
fn resolve_with_concatenated_literal_and_placeholder() {
    let s = sub(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--input="), ph(Placeholder::Path)])],
    );
    let (cmd, _env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj/a.c"),
        "a.c",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(cmd.argv, vec!["--input=/proj/a.c".to_string()]);
}

#[test]
fn resolve_with_created_expands_to_n_argv() {
    let s = sub(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
    );
    let diff = Diff {
        created: smallvec![
            entry_ref("a.rs", 1),
            entry_ref("b.rs", 2),
            entry_ref("c.rs", 3)
        ],
        ..Default::default()
    };
    let (cmd, _env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
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
    let s = sub(
        "rmlog",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Deleted)])],
    );
    let diff = Diff {
        deleted: smallvec![entry_ref("x", 9), entry_ref("y", 10)],
        ..Default::default()
    };
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
    assert_eq!(cmd.argv, vec!["x".to_string(), "y".to_string()]);
}

#[test]
fn resolve_with_modified_expands_to_n_argv() {
    let s = sub(
        "lint",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Modified)])],
    );
    let diff = Diff {
        modified: smallvec![entry_ref("m.rs", 1)],
        ..Default::default()
    };
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
    assert_eq!(cmd.argv, vec!["m.rs".to_string()]);
}

#[test]
fn resolve_with_renamed_from_and_to_expands_independently() {
    // `["mv", "$renamed_from", "$renamed_to"]` with two renames →
    // `["mv", from1, from2, to1, to2]` — independent expansion, no zip.
    let s = sub(
        "mv",
        EffectScope::PerStableFile,
        vec![
            arg(vec![lit("mv")]),
            arg(vec![ph(Placeholder::RenamedFrom)]),
            arg(vec![ph(Placeholder::RenamedTo)]),
        ],
    );
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
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
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
    let s = sub(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
    );
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(cmd.argv, vec!["fmt".to_string()]);
}

#[test]
fn resolve_with_empty_diff_placeholder_yields_zero_args() {
    let s = sub(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
    );
    let diff = Diff::default();
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
    assert_eq!(cmd.argv, vec!["fmt".to_string()]);
}

#[test]
fn resolve_with_multivalue_in_separate_args_emits_literals_as_standalone_slots() {
    let s = sub(
        "x",
        EffectScope::PerStableFile,
        vec![
            arg(vec![lit("pre")]),
            arg(vec![ph(Placeholder::Created)]),
            arg(vec![lit("post")]),
        ],
    );
    let diff = Diff {
        created: smallvec![entry_ref("a", 1), entry_ref("b", 2)],
        ..Default::default()
    };
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
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
    // `["--out=$created"]` with [a, b] → `["--out=a", "--out=b"]`. The
    // prefix literal tiles per emitted value when the multi-value
    // placeholder shares an ArgTemplate with a leading literal.
    let s = sub(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
    );
    let diff = Diff {
        created: smallvec![entry_ref("a", 1), entry_ref("b", 2)],
        ..Default::default()
    };
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
    assert_eq!(cmd.argv, vec!["--out=a".to_string(), "--out=b".to_string()]);
}

#[test]
fn resolve_with_multivalue_having_prefix_and_empty_diff_yields_zero_slots() {
    // `["--out=$created"]` with empty diff → no slots (no values to emit).
    let s = sub(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
    );
    let diff = Diff::default();
    let (cmd, _) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(&diff),
    );
    assert!(cmd.argv.is_empty());
}

// ---------- env vars ----------

#[test]
fn env_contains_specter_path_for_subtree_root() {
    let s = sub(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("make")])],
    );
    let (_, env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let path = env.iter().find(|(k, _)| k == "SPECTER_PATH").unwrap();
    assert_eq!(path.1, "/proj");
}

#[test]
fn env_contains_specter_path_for_per_stable_file() {
    let s = sub(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")])],
    );
    let (_, env) = resolve_effect(
        &s,
        Path::new("/proj"),
        Path::new("/proj/a.c"),
        "a.c",
        false,
        CorrelationId(1),
        None,
    );
    let path = env.iter().find(|(k, _)| k == "SPECTER_PATH").unwrap();
    assert_eq!(path.1, "/proj/a.c");
}

#[test]
fn env_specter_rel_path_empty_for_subtree_root() {
    let s = sub("b", EffectScope::SubtreeRoot, vec![arg(vec![lit("x")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_REL_PATH").unwrap().1,
        ""
    );
}

#[test]
fn env_specter_rel_path_for_per_stable_file() {
    let s = sub("f", EffectScope::PerStableFile, vec![arg(vec![lit("x")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p/src/a.c"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_REL_PATH").unwrap().1,
        "src/a.c"
    );
}

#[test]
fn env_specter_anchor_for_both_scopes() {
    for scope in [EffectScope::SubtreeRoot, EffectScope::PerStableFile] {
        let s = sub("x", scope, vec![arg(vec![lit("y")])]);
        let (_, env) = resolve_effect(
            &s,
            Path::new("/anchor/dir"),
            Path::new("/anchor/dir"),
            "",
            false,
            CorrelationId(1),
            None,
        );
        let v = env.iter().find(|(k, _)| k == "SPECTER_ANCHOR").unwrap();
        assert_eq!(v.1, "/anchor/dir", "scope = {scope:?}");
    }
}

#[test]
fn env_specter_sub_uses_sub_name() {
    let s = sub("build", EffectScope::SubtreeRoot, vec![arg(vec![lit("y")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_SUB").unwrap().1,
        "build"
    );
}

#[test]
fn env_specter_forced_zero_when_unforced() {
    let s = sub("x", EffectScope::SubtreeRoot, vec![arg(vec![lit("y")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_FORCED").unwrap().1,
        "0"
    );
}

#[test]
fn env_specter_forced_one_when_forced() {
    let s = sub("x", EffectScope::SubtreeRoot, vec![arg(vec![lit("y")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        true,
        CorrelationId(1),
        None,
    );
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_FORCED").unwrap().1,
        "1"
    );
}

#[test]
fn env_specter_event_kind_dir_subtree_for_subtree_root() {
    let s = sub("x", EffectScope::SubtreeRoot, vec![arg(vec![lit("y")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
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
    let s = sub("x", EffectScope::PerStableFile, vec![arg(vec![lit("y")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p/a"),
        "a",
        false,
        CorrelationId(1),
        None,
    );
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
    let s = sub("x", EffectScope::SubtreeRoot, vec![arg(vec![lit("y")])]);
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(42),
        None,
    );
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
    // Even when diff is Some, the resolver does NOT emit
    // SPECTER_DIFF_PATH — that's the actuator's responsibility.
    let s = sub("x", EffectScope::PerStableFile, vec![arg(vec![lit("y")])]);
    let diff = Diff {
        created: smallvec![entry_ref("a", 1)],
        ..Default::default()
    };
    let (_, env) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p/a"),
        "a",
        false,
        CorrelationId(1),
        Some(&diff),
    );
    assert!(env.iter().all(|(k, _)| k != "SPECTER_DIFF_PATH"));
}

#[test]
fn env_order_is_stable_across_calls() {
    let s = sub("x", EffectScope::SubtreeRoot, vec![arg(vec![lit("y")])]);
    let (_, e1) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, e2) = resolve_effect(
        &s,
        Path::new("/p"),
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let keys1: Vec<&String> = e1.iter().map(|(k, _)| k).collect();
    let keys2: Vec<&String> = e2.iter().map(|(k, _)| k).collect();
    assert_eq!(keys1, keys2);
}
