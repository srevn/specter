//! Integration tests: parse + validate against TOML fixtures.

use specter_config::{Config, ConfigError, IssueKind};
use specter_core::{ArgPart, ClassSet, EffectScope, Instruction, Placeholder};
use std::path::{Path, PathBuf};
use std::time::Duration;

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn fixture(name: &str) -> PathBuf {
    fixtures_dir().join(name)
}

fn validation_errors(err: ConfigError) -> Vec<specter_config::ValidationIssue> {
    match err {
        ConfigError::Validate { errors, .. } => errors,
        other => panic!("expected Validate, got {other:?}"),
    }
}

#[test]
fn minimal_fixture_round_trips() {
    let cfg = Config::from_path(&fixture("minimal.toml")).unwrap();
    assert_eq!(cfg.watches.len(), 1);
    assert_eq!(cfg.watches[0].name, "build");
}

#[test]
fn full_fixture_round_trips_every_field() {
    let cfg = Config::from_path(&fixture("full.toml")).unwrap();
    assert_eq!(cfg.log.level, specter_config::LogLevel::Debug);
    assert_eq!(cfg.log.destination, specter_config::LogDestination::Stderr);
    assert_eq!(cfg.watches.len(), 1);
    let w = &cfg.watches[0];
    assert_eq!(w.name, "build");
    assert_eq!(w.scope, EffectScope::SubtreeRoot);
    assert_eq!(w.settle, Duration::from_millis(500));
    assert_eq!(w.max_settle, Duration::from_secs(30));
    assert!(w.scan.recursive);
    assert!(!w.scan.hidden);
    assert!(w.scan.pattern.is_some());
    assert_eq!(w.scan.exclude.len(), 2);
    assert_eq!(w.scan.max_depth, Some(5));
    assert_eq!(w.events, ClassSet::STRUCTURE | ClassSet::CONTENT);
    let Instruction::SpawnExec(exec) = &w.program.instructions[0] else {
        panic!("expected SpawnExec instruction");
    };
    assert_eq!(exec.argv.len(), 3);
    assert_eq!(exec.argv[0].parts[0], ArgPart::literal("make"));
    assert_eq!(exec.argv[1].parts[0], ArgPart::literal("--input="));
    assert_eq!(
        exec.argv[1].parts[1],
        ArgPart::Placeholder(Placeholder::Path)
    );
    assert_eq!(
        exec.argv[2].parts[0],
        ArgPart::Placeholder(Placeholder::Created)
    );
}

#[test]
fn three_watches_preserves_source_order() {
    let cfg = Config::from_path(&fixture("three-watches.toml")).unwrap();
    let names: Vec<&str> = cfg.watches.iter().map(|w| w.name.as_str()).collect();
    assert_eq!(names, vec!["build", "lint", "fmt"]);
    assert_eq!(cfg.watches[2].scope, EffectScope::PerStableFile);
    assert_eq!(cfg.watches[0].events, ClassSet::DEFAULT_SUBTREE_ROOT);
    assert_eq!(cfg.watches[1].events, ClassSet::DEFAULT_SUBTREE_ROOT);
    assert_eq!(cfg.watches[2].events, ClassSet::DEFAULT_PER_FILE);
}

#[test]
fn unicode_name_preserved_byte_equal() {
    let cfg = Config::from_path(&fixture("unicode-name.toml")).unwrap();
    assert_eq!(cfg.watches[0].name, "build-🚀");
}

#[test]
fn pending_path_round_trips_via_lenient_canonicalization() {
    let td = tempfile::tempdir().unwrap();
    let pending = td.path().join("missing").join("leaf.txt");
    let toml = format!(
        "[[watch]]\nname = \"p\"\npath = \"{}\"\nactions = [{{ exec = [\"echo\"] }}]",
        pending.display(),
    );
    let cfg = Config::from_str(&toml).unwrap();
    assert!(
        cfg.watches[0].path.ends_with(Path::new("missing/leaf.txt")),
        "got {}",
        cfg.watches[0].path.display(),
    );
    let canon_td = td.path().canonicalize().unwrap();
    assert_eq!(
        cfg.watches[0].path,
        canon_td.join("missing").join("leaf.txt"),
    );
}

#[test]
fn invalid_glob_fixture_yields_validate_error() {
    let err = Config::from_path(&fixture("invalid-glob.toml")).unwrap_err();
    let errors = validation_errors(err);
    assert!(errors.iter().any(|e| e.kind == IssueKind::InvalidGlob));
}

#[test]
fn unknown_placeholder_fixture_yields_validate_error() {
    let err = Config::from_path(&fixture("unknown-placeholder.toml")).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::UnknownPlaceholder)
    );
}

#[test]
fn unknown_field_fixture_yields_parse_error() {
    let err = Config::from_path(&fixture("unknown-field.toml")).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }));
}

#[test]
fn duplicate_name_fixture_yields_validate_error() {
    let err = Config::from_path(&fixture("duplicate-name.toml")).unwrap_err();
    let errors = validation_errors(err);
    assert!(errors.iter().any(|e| e.kind == IssueKind::DuplicateName));
}

#[test]
fn all_defaults_fixture_applies_documented_defaults() {
    let cfg = Config::from_path(&fixture("all-defaults.toml")).unwrap();
    let w = &cfg.watches[0];
    assert!(w.scan.recursive);
    assert!(!w.scan.hidden);
    assert!(w.scan.pattern.is_none());
    assert!(w.scan.exclude.is_empty());
    assert_eq!(w.scan.max_depth, None);
    assert_eq!(w.scope, EffectScope::SubtreeRoot);
    assert_eq!(w.settle, Duration::from_millis(200));
    assert_eq!(w.max_settle, Duration::from_hours(1));
    assert_eq!(w.events, ClassSet::DEFAULT_SUBTREE_ROOT);
}

#[test]
fn missing_file_yields_io_error() {
    let err = Config::from_path(Path::new("/nonexistent/specter-test.toml")).unwrap_err();
    assert!(matches!(err, ConfigError::Io { .. }));
}

/// `enabled` is typed as `Option<bool>` in `RawWatch`; toml's standard
/// type-check rejects non-bool values at parse time, so the most
/// common typo (quoting the bool by reflex) surfaces as a parse
/// error rather than a validation issue.
#[test]
fn enabled_string_value_yields_parse_error() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"/\"\n\
                actions = [{ exec = [\"echo\"] }]\nenabled = \"true\"";
    let err = Config::from_str(toml).unwrap_err();
    assert!(matches!(err, ConfigError::Parse { .. }), "got {err:?}");
}

/// Per-action `timeout` (humantime at the TOML layer) round-trips
/// through validation into [`ExecAction::timeout`]. Pinned in one
/// test: humantime mapping, per-action threading, AND
/// omitted-as-`None` are all observable in the same program — the
/// middle step carries `None`, the outer steps carry distinct
/// durations.
#[test]
fn exec_timeout_threads_per_action_via_humantime_serde() {
    let toml = "[[watch]]\nname = \"t\"\npath = \"/\"\n\
                actions = [\n\
                  { exec = [\"a\"], timeout = \"500ms\" },\n\
                  { exec = [\"b\"] },\n\
                  { exec = [\"c\"], timeout = \"30s\" },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let timeouts: Vec<Option<Duration>> = cfg.watches[0]
        .program
        .instructions
        .iter()
        .map(|i| match i {
            Instruction::SpawnExec(e) => e.timeout,
            other => panic!("expected SpawnExec, got {other:?}"),
        })
        .collect();
    assert_eq!(
        timeouts,
        vec![
            Some(Duration::from_millis(500)),
            None,
            Some(Duration::from_secs(30)),
        ],
    );
}

/// Zero-duration timeouts (`"0s"`, `"0ms"`) are a near-certain typo —
/// the SIGTERM would fire before the child can make progress. Surface
/// as [`IssueKind::TimeoutZero`] rather than silently parsing.
#[test]
fn exec_zero_timeout_is_validation_error() {
    let toml = "[[watch]]\nname = \"t\"\npath = \"/\"\n\
                actions = [{ exec = [\"echo\"], timeout = \"0s\" }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors.iter().any(|e| e.kind == IssueKind::TimeoutZero),
        "got {errors:?}"
    );
}

/// `${env.HOME:-/tmp}` lowers through validation into
/// [`ArgPart::EnvVar`] inside the program. Confirms the lexer + lowering
/// path is wired end-to-end; the default-bearing form covers both the
/// `${env.NAME}` and `${env.NAME:-default}` lexer branches in one test.
#[test]
fn exec_env_placeholder_lowers_into_program() {
    let toml = "[[watch]]\nname = \"e\"\npath = \"/\"\n\
                actions = [{ exec = [\"echo\", \"${env.HOME:-/tmp}\"] }]";
    let cfg = Config::from_str(toml).unwrap();
    let exec = match &cfg.watches[0].program.instructions[0] {
        Instruction::SpawnExec(e) => e,
        other => panic!("expected SpawnExec, got {other:?}"),
    };
    match &exec.argv[1].parts[0] {
        ArgPart::EnvVar { name, default } => {
            assert_eq!(name, "HOME");
            assert_eq!(default.as_deref(), Some("/tmp"));
        }
        other => panic!("expected ArgPart::EnvVar, got {other:?}"),
    }
}

/// Malformed env-var name (`1HOME`) surfaces as a validation issue
/// rather than panicking. The template layer's `InvalidEnvName` is
/// collapsed onto [`IssueKind::UnknownPlaceholder`] so operator-facing
/// output stays consistent across both namespaces; the detail message
/// carries the specific cause.
#[test]
fn exec_env_invalid_name_yields_validation_error() {
    let toml = "[[watch]]\nname = \"e\"\npath = \"/\"\n\
                actions = [{ exec = [\"echo\", \"${env.1HOME}\"] }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::UnknownPlaceholder && e.detail.contains("1HOME")),
        "got {errors:?}",
    );
}
