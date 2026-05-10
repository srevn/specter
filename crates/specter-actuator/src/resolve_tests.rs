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
    ArgPart, ArgTemplate, CommandResolved, CommandTemplate, CorrelationId, DedupKey, Diff, Effect,
    EffectScope, EntryKind, EntryRef, Placeholder, ProfileId, Rename, ResourceId, ResourceKind,
    SubId,
};
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

/// Convenience wrapper for tests that don't exercise `$time` /
/// `SPECTER_TIME` rendering — pins `now` to the Unix epoch and omits the
/// diff tmp file. Time-sensitive tests call [`super::resolve_effect`]
/// directly with the instant they want; diff-tmp-aware tests pass
/// `diff_path: Some(_)` directly.
fn resolve(e: &Effect) -> (CommandResolved, Vec<(String, String)>) {
    super::resolve_effect(e, SystemTime::UNIX_EPOCH, None)
}

/// `target_path` is no longer a field on [`Effect`] — the resolver
/// derives it from `(anchor_path, target_relative)` at spawn time. Tests
/// pass the anchor + relative pair; the helper does no extra dispatch.
///
/// `scope` selects the [`DedupKey`] variant (Subtree ⇒ no per-file
/// resource, PerStableFile ⇒ default per-file resource); the resolver
/// then derives `SPECTER_EVENT_KIND` from the variant.
fn make_effect(
    sub_name: &str,
    scope: EffectScope,
    argv: Vec<ArgTemplate>,
    anchor_path: &Path,
    target_relative: &str,
    forced: bool,
    correlation: CorrelationId,
    diff: Option<Arc<Diff>>,
) -> Effect {
    let key = match scope {
        EffectScope::SubtreeRoot => DedupKey::Subtree {
            sub: SubId::default(),
            profile: ProfileId::default(),
        },
        EffectScope::PerStableFile => DedupKey::PerFile {
            sub: SubId::default(),
            profile: ProfileId::default(),
            resource: ResourceId::default(),
        },
    };
    Effect {
        key,
        target: ResourceId::default(),
        forced,
        correlation,
        diff,
        capture_output: false,
        sub_name: CompactString::from(sub_name),
        command: Arc::new(CommandTemplate::new(argv)),
        anchor_path: Arc::from(anchor_path.to_path_buf()),
        anchor_kind: ResourceKind::Dir,
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
    assert_eq!(cmd.argv, vec!["make".to_string()]);
}

#[test]
fn resolve_with_path_placeholder() {
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Path)])],
        Path::new("/proj"),
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
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
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
    assert_eq!(cmd.argv, vec!["log".to_string(), "src/a.c".to_string()]);
}

#[test]
fn resolve_with_anchor_placeholder() {
    let e = make_effect(
        "build",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("build")]), arg(vec![ph(Placeholder::Anchor)])],
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
    assert_eq!(cmd.argv, vec!["build".to_string(), "/proj".to_string()]);
}

// ---------- $excluded / SPECTER_EXCLUDE ----------

#[test]
fn resolve_excluded_one_arg_per_pattern() {
    // `--exclude=$excluded` tiles the literal prefix per pattern,
    // mirroring the diff-derived multi-value behaviour.
    let mut e = make_effect(
        "rsync",
        EffectScope::SubtreeRoot,
        vec![
            arg(vec![lit("rsync")]),
            arg(vec![lit("--exclude="), ph(Placeholder::Excluded)]),
            arg(vec![lit("/src/")]),
        ],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    e.exclude = vec![
        CompactString::from("*.tmp"),
        CompactString::from("cache/"),
        CompactString::from("**/.git/"),
    ]
    .into();
    let (cmd, _) = resolve(&e);
    assert_eq!(
        cmd.argv,
        vec![
            "rsync".to_string(),
            "--exclude=*.tmp".to_string(),
            "--exclude=cache/".to_string(),
            "--exclude=**/.git/".to_string(),
            "/src/".to_string(),
        ]
    );
}

#[test]
fn resolve_excluded_empty_drops_slot() {
    // Empty exclude list mirrors empty-diff: drop the entire
    // `--exclude=$excluded` slot rather than emit `--exclude=`.
    let e = make_effect(
        "rsync",
        EffectScope::SubtreeRoot,
        vec![
            arg(vec![lit("rsync")]),
            arg(vec![lit("--exclude="), ph(Placeholder::Excluded)]),
            arg(vec![lit("/src/")]),
        ],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    // exclude defaults empty in make_effect.
    let (cmd, _) = resolve(&e);
    assert_eq!(
        cmd.argv,
        vec!["rsync".to_string(), "/src/".to_string()],
        "empty $excluded drops the surrounding slot"
    );
}

#[test]
fn env_exclude_newline_separated() {
    // Newline-separated source strings, no trailing newline. Survives
    // any pattern content (commas, spaces, apostrophes) that's legal in
    // glob source strings.
    let mut e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    e.exclude = vec![
        CompactString::from("*.tmp"),
        CompactString::from("cache/"),
        CompactString::from("**/.git/"),
    ]
    .into();
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_EXCLUDE").unwrap().1,
        "*.tmp\ncache/\n**/.git/",
        "no trailing newline; entries joined by single \\n",
    );
}

#[test]
fn env_exclude_empty_is_empty_string() {
    // Empty exclude list ⇒ empty env value, NOT a blank line.
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_EXCLUDE").unwrap().1,
        "",
    );
}

// ---------- $time / SPECTER_TIME ----------

/// Unix timestamp 1_700_000_000 = 2023-11-14T22:13:20Z. Chosen for
/// readability in the assert; the format is RFC 3339 second-precision.
const FIXED_NOW_SECS: u64 = 1_700_000_000;
const FIXED_NOW_RFC3339: &str = "2023-11-14T22:13:20Z";

#[test]
fn resolve_time_uses_injected_now() {
    let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(FIXED_NOW_SECS);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Time)])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = super::resolve_effect(&e, now, None);
    assert_eq!(cmd.argv, vec![FIXED_NOW_RFC3339.to_owned()]);
}

#[test]
fn env_specter_time_uses_injected_now() {
    let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(FIXED_NOW_SECS);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = super::resolve_effect(&e, now, None);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_TIME").unwrap().1,
        FIXED_NOW_RFC3339
    );
}

#[test]
fn format_now_clamps_pre_epoch() {
    // humantime::format_rfc3339_seconds panics on pre-epoch SystemTime.
    // Production never sees pre-epoch on a sane Unix host, but tests can
    // construct one. The resolver clamps to UNIX_EPOCH so the spawn path
    // can't panic on a hostile clock.
    let pre = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(1);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Time)])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = super::resolve_effect(&e, pre, None);
    assert_eq!(cmd.argv, vec!["1970-01-01T00:00:00Z".to_owned()]);
}

// ---------- $parent ----------
//
// Documented edge cases (see Placeholder::Parent rustdoc):
//   PerFile  | /anchor  | foo.rs       | $parent = /anchor
//   PerFile  | /        | foo.rs       | $parent = /        (NOT empty)
//   Subtree  | /anchor  | n/a          | $parent = /
//   Subtree  | /        | n/a          | $parent = ""       (only empty case)

#[test]
fn resolve_parent_is_target_dir_for_perfile() {
    // PerFile target = anchor.join(segment); $parent = the directory
    // immediately containing the file that triggered the fire.
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/anchor"),
        "foo.rs",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["/anchor".to_string()]);
}

#[test]
fn resolve_parent_is_anchor_parent_for_subtree() {
    // Subtree target_path == anchor_path; $parent = parent of the anchor
    // (one level above the watch root).
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/proj/sub"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["/proj".to_string()]);
}

#[test]
fn resolve_parent_for_perfile_at_root_is_root() {
    // Filesystem-root anchor with PerFile scope: target_path = "/foo.rs",
    // parent = "/" (NOT empty). Guards against the easy misreading that
    // any anchor at root yields empty $parent.
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/"),
        "foo.rs",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["/".to_string()]);
}

#[test]
fn resolve_parent_empty_only_for_subtree_at_root() {
    // The only configuration that yields an empty $parent: Subtree scope
    // anchored at filesystem root (target_path = "/", which has no parent).
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    // Empty parent → ArgTemplate produces a single empty argv slot
    // (single-value placeholders never drop the slot, only multi-values
    // with zero entries do).
    assert_eq!(cmd.argv, vec![String::new()]);
}

#[test]
fn env_parent_empty_only_for_subtree_at_root() {
    // SPECTER_PARENT mirrors $parent: empty string only at fs root for
    // Subtree scope; "/" everywhere else at the root level.
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_PARENT").unwrap().1,
        ""
    );
}

#[test]
fn env_parent_for_perfile_is_target_directory() {
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("noop")])],
        Path::new("/anchor"),
        "src/foo.rs",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter().find(|(k, _)| k == "SPECTER_PARENT").unwrap().1,
        "/anchor/src"
    );
}

#[test]
fn resolve_substitutes_watch_name() {
    // `$watch` substitutes `effect.sub_name` — mirrors `$SPECTER_WATCH`
    // env value but in argv form.
    let e = make_effect(
        "build",
        EffectScope::SubtreeRoot,
        vec![
            arg(vec![lit("notify-send")]),
            arg(vec![ph(Placeholder::Watch), lit(" settled")]),
        ],
        Path::new("/proj"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
    assert_eq!(
        cmd.argv,
        vec!["notify-send".to_string(), "build settled".to_string()]
    );
}

#[test]
fn resolve_with_concatenated_literal_and_placeholder() {
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--input="), ph(Placeholder::Path)])],
        Path::new("/proj"),
        "a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _env) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["fmt".to_string()]);
}

#[test]
fn resolve_with_empty_diff_placeholder_yields_zero_args() {
    let e = make_effect(
        "fmt",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(Diff::default())),
    );
    let (cmd, _) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["--out=a".to_string(), "--out=b".to_string()]);
}

#[test]
fn resolve_with_multivalue_having_prefix_and_empty_diff_yields_zero_slots() {
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        Some(Arc::new(Diff::default())),
    );
    let (cmd, _) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "src/a.c",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
            "",
            false,
            CorrelationId(1),
            None,
        );
        let (_, env) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "",
        true,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "a",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
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
        "",
        false,
        CorrelationId(42),
        None,
    );
    let (_, env) = resolve(&e);
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
        "a",
        false,
        CorrelationId(1),
        Some(Arc::new(diff)),
    );
    let (_, env) = resolve(&e);
    assert!(env.iter().all(|(k, _)| k != "SPECTER_DIFF_PATH"));
}

#[test]
fn env_order_is_alphabetical() {
    let e = make_effect(
        "watch",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let (_, env) = resolve(&e);
    let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "SPECTER_ANCHOR",
            "SPECTER_CORRELATION",
            "SPECTER_EVENT_KIND",
            "SPECTER_EXCLUDE",
            "SPECTER_FORCED",
            "SPECTER_PARENT",
            "SPECTER_PATH",
            "SPECTER_RELATIVE_PATH",
            "SPECTER_TIME",
            "SPECTER_WATCH",
        ]
    );
}

#[test]
fn env_order_with_diff_path_is_alphabetical() {
    // With `diff_path: Some(_)`, SPECTER_DIFF_PATH joins the env in
    // alphabetical position (between SPECTER_CORRELATION and
    // SPECTER_EVENT_KIND), keeping a total order across the spawn-time
    // set the child observes.
    let e = make_effect(
        "watch",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId(1),
        None,
    );
    let diff_path = Path::new("/tmp/specter-1234-deadbeef.diff");
    let (_, env) = super::resolve_effect(&e, SystemTime::UNIX_EPOCH, Some(diff_path));
    let keys: Vec<&str> = env.iter().map(|(k, _)| k.as_str()).collect();
    assert_eq!(
        keys,
        vec![
            "SPECTER_ANCHOR",
            "SPECTER_CORRELATION",
            "SPECTER_DIFF_PATH",
            "SPECTER_EVENT_KIND",
            "SPECTER_EXCLUDE",
            "SPECTER_FORCED",
            "SPECTER_PARENT",
            "SPECTER_PATH",
            "SPECTER_RELATIVE_PATH",
            "SPECTER_TIME",
            "SPECTER_WATCH",
        ]
    );
    assert_eq!(
        env.iter()
            .find(|(k, _)| k == "SPECTER_DIFF_PATH")
            .unwrap()
            .1,
        "/tmp/specter-1234-deadbeef.diff"
    );
}
