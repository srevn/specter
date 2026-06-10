//! Sibling unit tests for [`super::resolve`]. Pure data work — all fixtures are inline; no I/O.

#![allow(
    clippy::items_after_statements,
    clippy::missing_const_for_fn,
    clippy::too_many_lines
)]

use super::CommandResolved;
use crate::env::EnvSnapshot;
use crate::spawner::EnvVar;
use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::program::SpawnBody;
use specter_core::testkit::single_exec_program;
use specter_core::{
    ArgPart, ArgTemplate, CorrelationId, Diff, Effect, EffectCommon, EffectScope, EntryKind,
    EntryRef, ExecAction, FsIdentity, Placeholder, ProfileId, Rename, ResourceId, ResourceKind,
    SubId,
};
use std::borrow::Cow;
use std::path::Path;
use std::sync::Arc;
use std::time::SystemTime;

/// Empty env snapshot used by tests that don't exercise `${env.<NAME>}` resolution. Constructed
/// once via `OnceLock` and threaded through the `resolve` helper.
fn empty_env() -> EnvSnapshot {
    EnvSnapshot::from_map::<_, &str, &str>([])
}

/// Convenience wrapper for tests that don't exercise `${specter.time}` / `SPECTER_TIME` rendering —
/// pins `now` to the Unix epoch, omits the diff tmp file, and uses an empty env snapshot.
/// Time-sensitive tests call [`super::resolve_step`] directly with the instant they want;
/// diff-tmp-aware tests pass `diff_path: Some(_)` directly; env-aware tests build a fresh snapshot
/// inline.
fn resolve(e: &Effect) -> (CommandResolved, Vec<EnvVar<'_>>) {
    let exec = exec_of(e);
    super::resolve_step(e, exec, SystemTime::UNIX_EPOCH, None, &empty_env())
        .expect("test fixtures don't exercise the strict-env failure path")
}

/// Borrow the single [`ExecAction`] inside an [`Effect`]'s program. Tests build effects with
/// exactly one `SpawnBody::Exec` op; this is a fixture-side accessor, not a production API.
fn exec_of(e: &Effect) -> &ExecAction {
    match &e.program.ops()[0].body() {
        SpawnBody::Exec(exec) => exec,
        SpawnBody::Pipe(_) => panic!("test fixtures use only Exec body"),
    }
}

/// The resolver derives `target_path` from `(anchor_path, relative())` at spawn time. Tests pass
/// the anchor + relative pair; the helper does no extra dispatch.
///
/// `scope` selects the `EffectTarget` shape (Subtree ⇒ no per-file segment, PerStableFile ⇒
/// per-file segment); the resolver then derives `SPECTER_EVENT_KIND` from the shape.
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
    let common = EffectCommon {
        sub: SubId::default(),
        profile: ProfileId::default(),
        anchor: ResourceId::default(),
        correlation,
        forced,
        capture_output: false,
        sub_name: CompactString::from(sub_name),
        program: single_exec_program(argv),
        anchor_path: Arc::from(anchor_path.to_path_buf()),
        anchor_kind: ResourceKind::Dir,
        exclude: Arc::from(Vec::<CompactString>::new()),
    };
    match scope {
        EffectScope::SubtreeRoot => Effect::subtree(common, diff),
        EffectScope::PerStableFile => {
            // PerFile diff is mandatory. Callers that passed `None` did not reference a
            // diff-derived placeholder; an empty `Diff::default()` renders those placeholders
            // identically to the old absent-diff path.
            let diff = diff.unwrap_or_else(|| Arc::new(Diff::default()));
            Effect::per_file(
                common,
                ResourceId::default(),
                CompactString::from(target_relative),
                diff,
            )
        }
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
        fs_id: FsIdentity::synthetic(inode, 0),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
        None,
    );
    let (cmd, _env) = resolve(&e);
    assert_eq!(cmd.argv, vec!["build".to_string(), "/proj".to_string()]);
}

// ---------- ${specter.excluded} / SPECTER_EXCLUDED ----------

#[test]
fn resolve_excluded_one_arg_per_pattern() {
    // `--exclude=${specter.excluded}` tiles the literal prefix per pattern, mirroring the
    // diff-derived multi-value behaviour.
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
        CorrelationId::from(1),
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
    // Empty exclude list mirrors empty-diff: drop the entire `--exclude=${specter.excluded}` slot
    // rather than emit `--exclude=`.
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
        CorrelationId::from(1),
        None,
    );
    // exclude defaults empty in make_effect.
    let (cmd, _) = resolve(&e);
    assert_eq!(
        cmd.argv,
        vec!["rsync".to_string(), "/src/".to_string()],
        "empty ${{specter.excluded}} drops the surrounding slot"
    );
}

#[test]
fn env_exclude_newline_separated() {
    // Newline-separated source strings, no trailing newline. Survives any pattern content (commas,
    // spaces, apostrophes) that's legal in glob source strings.
    let mut e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
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
        env.iter()
            .find(|e| e.key == "SPECTER_EXCLUDED")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_EXCLUDED")
            .unwrap()
            .value,
        "",
    );
}

// ---------- ${specter.time} / SPECTER_TIME ----------

/// Unix timestamp 1_700_000_000 = 2023-11-14T22:13:20Z. Chosen for readability in the assert; the
/// format is RFC 3339 second-precision.
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
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = super::resolve_step(&e, exec_of(&e), now, None, &empty_env())
        .expect("test fixtures don\'t exercise the strict-env failure path");
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = super::resolve_step(&e, exec_of(&e), now, None, &empty_env())
        .expect("test fixtures don\'t exercise the strict-env failure path");
    assert_eq!(
        env.iter().find(|e| e.key == "SPECTER_TIME").unwrap().value,
        FIXED_NOW_RFC3339
    );
}

#[test]
fn format_now_clamps_pre_epoch() {
    // humantime::format_rfc3339_seconds panics on pre-epoch SystemTime. Production never sees
    // pre-epoch on a sane Unix host, but tests can construct one. The resolver clamps to UNIX_EPOCH
    // so the spawn path can't panic on a hostile clock.
    let pre = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(1);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Time)])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = super::resolve_step(&e, exec_of(&e), pre, None, &empty_env())
        .expect("test fixtures don\'t exercise the strict-env failure path");
    assert_eq!(cmd.argv, vec!["1970-01-01T00:00:00Z".to_owned()]);
}

// ---------- ${specter.parent} ----------
//
// Documented edge cases (see Placeholder::Parent rustdoc):
//   PerFile  | /anchor  | foo.rs       | ${specter.parent} = /anchor
//   PerFile  | /        | foo.rs       | ${specter.parent} = /        (NOT empty)
//   Subtree  | /anchor  | n/a          | ${specter.parent} = /
//   Subtree  | /        | n/a          | ${specter.parent} = ""       (only empty case)

#[test]
fn resolve_parent_is_target_dir_for_perfile() {
    // PerFile target = anchor.join(segment); ${specter.parent} = the directory immediately
    // containing the file that triggered the fire.
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/anchor"),
        "foo.rs",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["/anchor".to_string()]);
}

#[test]
fn resolve_parent_is_anchor_parent_for_subtree() {
    // Subtree target_path == anchor_path; ${specter.parent} = parent of the anchor (one level above
    // the watch root).
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/proj/sub"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["/proj".to_string()]);
}

#[test]
fn resolve_parent_for_perfile_at_root_is_root() {
    // Filesystem-root anchor with PerFile scope: target_path = "/foo.rs", parent = "/" (NOT empty).
    // Guards against the easy misreading that any anchor at root yields empty ${specter.parent}.
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/"),
        "foo.rs",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    assert_eq!(cmd.argv, vec!["/".to_string()]);
}

#[test]
fn resolve_parent_empty_only_for_subtree_at_root() {
    // The only configuration that yields an empty ${specter.parent}: Subtree scope anchored at
    // filesystem root (target_path = "/", which has no parent).
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ph(Placeholder::Parent)])],
        Path::new("/"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = resolve(&e);
    // Empty parent → ArgTemplate produces a single empty argv slot (single-value placeholders never
    // drop the slot, only multi-values with zero entries do).
    assert_eq!(cmd.argv, vec![String::new()]);
}

#[test]
fn env_parent_empty_only_for_subtree_at_root() {
    // SPECTER_PARENT mirrors ${specter.parent}: empty string only at fs root for Subtree scope; "/"
    // everywhere else at the root level.
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_PARENT")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_PARENT")
            .unwrap()
            .value,
        "/anchor/src"
    );
}

#[test]
fn resolve_substitutes_watch_name() {
    // `${specter.watch}` substitutes `effect.sub_name` — mirrors `$SPECTER_WATCH` env value but in
    // argv form.
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
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
        CorrelationId::from(1),
        Some(Arc::new(Diff::default())),
    );
    let (cmd, _) = resolve(&e);
    assert!(cmd.argv.is_empty());
}

// ---------- diff-derived env vars ----------

#[test]
fn env_specter_created_newline_separated() {
    // Diff-derived multi-value env var mirrors the argv form: each entry's segment, joined by `\n`,
    // no trailing newline. Empty list ⇒ empty string (asserted in env_diff_lists_*); populated list
    // ⇒ the segments.
    let diff = Diff {
        created: smallvec![
            entry_ref("a.rs", 1),
            entry_ref("src/b.rs", 2),
            entry_ref("c", 3),
        ],
        ..Default::default()
    };
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        Some(Arc::new(diff)),
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_CREATED")
            .unwrap()
            .value,
        "a.rs\nsrc/b.rs\nc",
        "no trailing newline; entries joined by single \\n",
    );
}

#[test]
fn env_specter_deleted_and_modified_render_their_categories() {
    // One Diff carrying entries for two categories; each env var pulls from its own list. Asserts
    // the dispatch in `diff_env_segs` doesn't cross-contaminate.
    let diff = Diff {
        deleted: smallvec![entry_ref("d1", 1), entry_ref("d2", 2)],
        modified: smallvec![entry_ref("m1", 3)],
        ..Default::default()
    };
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        Some(Arc::new(diff)),
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_DELETED")
            .unwrap()
            .value,
        "d1\nd2",
    );
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_MODIFIED")
            .unwrap()
            .value,
        "m1",
    );
    // Categories not populated stay empty even though the diff is present.
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_CREATED")
            .unwrap()
            .value,
        "",
    );
}

#[test]
fn env_specter_renamed_from_and_to_use_correct_sides() {
    // Two renames, each with distinct from/to segments. The two env vars must each pull their
    // respective side; cross-contamination would mean the from/to projection in `diff_env_renames`
    // is broken.
    let diff = Diff {
        renamed: smallvec![
            Rename {
                from: entry_ref("old1", 1),
                to: entry_ref("new1", 1),
            },
            Rename {
                from: entry_ref("old2", 2),
                to: entry_ref("new2", 2),
            },
        ],
        ..Default::default()
    };
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        Some(Arc::new(diff)),
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_RENAMED_FROM")
            .unwrap()
            .value,
        "old1\nold2",
    );
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_RENAMED_TO")
            .unwrap()
            .value,
        "new1\nnew2",
    );
}

#[test]
fn env_diff_lists_empty_when_no_diff() {
    // `Effect.diff = None` (Sub doesn't reference any diff-derived placeholder and isn't
    // `per-stable-file`). All five list env vars emit as empty strings — always-emit policy mirrors
    // SPECTER_EXCLUDED and avoids `set -u` surprises in the spawned shell. The
    // `Some(Diff::default())` variant exits the same way through `join_with_newlines`'s empty-iter
    // branch (already pinned by `env_exclude_empty_is_empty_string`).
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    for k in [
        "SPECTER_CREATED",
        "SPECTER_DELETED",
        "SPECTER_MODIFIED",
        "SPECTER_RENAMED_FROM",
        "SPECTER_RENAMED_TO",
    ] {
        assert_eq!(
            env.iter().find(|e| e.key == k).unwrap().value,
            "",
            "{k} must be empty when Effect.diff is None",
        );
    }
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
    assert_eq!(path.value, "/proj");
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
    assert_eq!(path.value, "/proj/a.c");
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_RELATIVE_PATH")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_RELATIVE_PATH")
            .unwrap()
            .value,
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
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        let v = env.iter().find(|e| e.key == "SPECTER_ANCHOR").unwrap();
        assert_eq!(v.value, "/anchor/dir", "scope = {scope:?}");
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter().find(|e| e.key == "SPECTER_WATCH").unwrap().value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_FORCED")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_FORCED")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_EVENT_KIND")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_EVENT_KIND")
            .unwrap()
            .value,
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
        CorrelationId::from(42),
        None,
    );
    let (_, env) = resolve(&e);
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_CORRELATION")
            .unwrap()
            .value,
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
        CorrelationId::from(1),
        Some(Arc::new(diff)),
    );
    let (_, env) = resolve(&e);
    assert!(env.iter().all(|e| e.key != "SPECTER_DIFF_PATH"));
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
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    let keys: Vec<&str> = env.iter().map(|e| e.key).collect();
    assert_eq!(
        keys,
        vec![
            "SPECTER_ANCHOR",
            "SPECTER_CORRELATION",
            "SPECTER_CREATED",
            "SPECTER_DELETED",
            "SPECTER_EVENT_KIND",
            "SPECTER_EXCLUDED",
            "SPECTER_FORCED",
            "SPECTER_MODIFIED",
            "SPECTER_PARENT",
            "SPECTER_PATH",
            "SPECTER_RELATIVE_PATH",
            "SPECTER_RENAMED_FROM",
            "SPECTER_RENAMED_TO",
            "SPECTER_TIME",
            "SPECTER_WATCH",
        ]
    );
}

#[test]
fn env_order_with_diff_path_is_alphabetical() {
    // With `diff_path: Some(_)`, SPECTER_DIFF_PATH joins the env in alphabetical position (between
    // SPECTER_DELETED and SPECTER_EVENT_KIND), keeping a total order across the spawn-time set the
    // child observes.
    let e = make_effect(
        "watch",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("y")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let diff_path = Path::new("/tmp/specter-1234-deadbeef.diff");
    let (_, env) = super::resolve_step(
        &e,
        exec_of(&e),
        SystemTime::UNIX_EPOCH,
        Some(diff_path),
        &empty_env(),
    )
    .expect("test fixtures don\'t exercise the strict-env failure path");
    let keys: Vec<&str> = env.iter().map(|e| e.key).collect();
    assert_eq!(
        keys,
        vec![
            "SPECTER_ANCHOR",
            "SPECTER_CORRELATION",
            "SPECTER_CREATED",
            "SPECTER_DELETED",
            "SPECTER_DIFF_PATH",
            "SPECTER_EVENT_KIND",
            "SPECTER_EXCLUDED",
            "SPECTER_FORCED",
            "SPECTER_MODIFIED",
            "SPECTER_PARENT",
            "SPECTER_PATH",
            "SPECTER_RELATIVE_PATH",
            "SPECTER_RENAMED_FROM",
            "SPECTER_RENAMED_TO",
            "SPECTER_TIME",
            "SPECTER_WATCH",
        ]
    );
    assert_eq!(
        env.iter()
            .find(|e| e.key == "SPECTER_DIFF_PATH")
            .unwrap()
            .value,
        "/tmp/specter-1234-deadbeef.diff"
    );
}

// ---------- Cow borrow discipline ----------
//
// When `Effect::target_path` is `Cow::Borrowed` (Subtree fire), `SPECTER_PATH` / `SPECTER_PARENT`
// propagate the borrow into `Cow::Borrowed` on the UTF-8 fast path; when it is `Cow::Owned` (PerFile
// fire), both fields own. The two assertions below pin one Subtree case (path + parent in one
// resolve) and one PerFile case so a future regression that forces an unconditional `into_owned()` on
// either field surfaces in the test suite. The empty-multivalue short-circuit gets its own small
// assertion since the borrow-vs-owned property there is independent of `target_path`.

#[test]
fn env_specter_path_and_parent_borrow_for_subtree() {
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/proj/sub"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
    let parent = env.iter().find(|e| e.key == "SPECTER_PARENT").unwrap();
    assert!(
        matches!(path.value, Cow::Borrowed(_)),
        "Subtree SPECTER_PATH should borrow from effect.anchor_path on the UTF-8 fast path",
    );
    assert!(
        matches!(parent.value, Cow::Borrowed(_)),
        "Subtree SPECTER_PARENT should borrow from effect.anchor_path on the UTF-8 fast path",
    );
    assert_eq!(path.value, "/proj/sub");
    assert_eq!(parent.value, "/proj");
}

#[test]
fn env_specter_path_and_parent_own_for_perfile() {
    // PerFile `target_path = anchor.join(segment)` is a freshly-joined `PathBuf` living only on the
    // resolve stack; both fields must own their bytes for the env vec to outlive the resolve call.
    let e = make_effect(
        "x",
        EffectScope::PerStableFile,
        vec![arg(vec![lit("noop")])],
        Path::new("/proj"),
        "a.c",
        false,
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
    let parent = env.iter().find(|e| e.key == "SPECTER_PARENT").unwrap();
    assert!(matches!(path.value, Cow::Owned(_)));
    assert!(matches!(parent.value, Cow::Owned(_)));
    assert_eq!(path.value, "/proj/a.c");
    assert_eq!(parent.value, "/proj");
}

#[test]
fn env_multivalue_borrows_empty_string_when_no_entries() {
    // `env_multivalue` short-circuits the empty case to `Cow::Borrowed("")` instead of allocating
    // an empty `String`. The no-diff resolve emits six empty multi-value env vars; this saves six
    // `String::new()` allocations per resolve on the common path for Subs that don't reference diff
    // placeholders. One probe per category is enough — they all route through the same helper.
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![lit("noop")])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (_, env) = resolve(&e);
    for k in [
        "SPECTER_CREATED",
        "SPECTER_DELETED",
        "SPECTER_MODIFIED",
        "SPECTER_RENAMED_FROM",
        "SPECTER_RENAMED_TO",
        "SPECTER_EXCLUDED",
    ] {
        let v = env.iter().find(|e| e.key == k).unwrap();
        assert!(
            matches!(v.value, Cow::Borrowed(_)),
            "{k} must be Cow::Borrowed when its source list is empty",
        );
    }
}

// ---------- ${env.<NAME>} ----------

/// `${env.NAME}` resolves to the snapshot's value when present. Default-bearing form is exercised
/// below; together they cover both lexer branches in the resolver pass.
#[test]
fn resolve_env_var_substitutes_from_snapshot() {
    let env = EnvSnapshot::from_map([("HOME", "/home/op")]);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ArgPart::EnvVar {
            name: "HOME".into(),
            default: None,
        }])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = super::resolve_step(&e, exec_of(&e), SystemTime::UNIX_EPOCH, None, &env)
        .expect("HOME present in snapshot");
    assert_eq!(cmd.argv, vec!["/home/op".to_string()]);
}

/// Strict default: unset env var with no `:-` default fails the resolve — the caller maps
/// `ResolveError::UnsetEnvVar` to `EffectOutcome::Failed`.
#[test]
fn resolve_env_var_unset_without_default_returns_unset_env_var_error() {
    let env = EnvSnapshot::from_map::<_, &str, &str>([]);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![arg(vec![ArgPart::EnvVar {
            name: "MISSING".into(),
            default: None,
        }])],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let err = super::resolve_step(&e, exec_of(&e), SystemTime::UNIX_EPOCH, None, &env)
        .expect_err("unset env var must fail strict resolve");
    assert_eq!(
        err,
        crate::resolve::ResolveError::UnsetEnvVar {
            name: "MISSING".into(),
        }
    );
}

/// Unset env var with a `:-default` renders the default literal — explicit lenient opt-in. Empty
/// default (`${env.X:-}`) renders empty.
#[test]
fn resolve_env_var_unset_with_default_renders_default() {
    let env = EnvSnapshot::from_map::<_, &str, &str>([]);
    let e = make_effect(
        "x",
        EffectScope::SubtreeRoot,
        vec![
            arg(vec![ArgPart::EnvVar {
                name: "MISSING".into(),
                default: Some("/tmp".into()),
            }]),
            arg(vec![ArgPart::EnvVar {
                name: "ALSO_MISSING".into(),
                default: Some(CompactString::new("")),
            }]),
        ],
        Path::new("/p"),
        "",
        false,
        CorrelationId::from(1),
        None,
    );
    let (cmd, _) = super::resolve_step(&e, exec_of(&e), SystemTime::UNIX_EPOCH, None, &env)
        .expect("default rendered when env unset");
    assert_eq!(cmd.argv, vec!["/tmp".to_string(), String::new()]);
}
