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

// ----- Conditional actions (`when` / `then` / `else`) -----

/// Conditional with `when` + `then` lowers to a two-instruction
/// program: `SpawnPredicate(when, jump=2)` then the then-branch's
/// `SpawnExec`. With no `else`, the predicate's `jump_target` is
/// `len = 2` — the natural "skip past end" form treated by the
/// actuator's reap-path as terminate-Ok.
#[test]
fn conditional_when_then_lowers_to_predicate_plus_then() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, then = [{ exec = [\"yes\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.instructions.len(), 2);
    match &p.instructions[0] {
        Instruction::SpawnPredicate { jump_target, .. } => {
            assert_eq!(*jump_target, 2, "no-else: jump past end");
        }
        other => panic!("expected SpawnPredicate, got {other:?}"),
    }
    assert!(matches!(p.instructions[1], Instruction::SpawnExec(_)));
}

/// Conditional with `when` + `then` + `else` lowers to a four-
/// instruction program with the Jump-after-then skipping the
/// else-branch on the Ok path: `SpawnPredicate(jump=3)`,
/// `SpawnExec(then)`, `Jump(target=4)`, `SpawnExec(else)`.
#[test]
fn conditional_with_else_lowers_to_predicate_then_jump_else() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, \
                    then = [{ exec = [\"yes\"] }], \
                    else = [{ exec = [\"no\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.instructions.len(), 4);
    match &p.instructions[0] {
        Instruction::SpawnPredicate { jump_target, .. } => assert_eq!(*jump_target, 3),
        other => panic!("expected SpawnPredicate, got {other:?}"),
    }
    assert!(matches!(p.instructions[1], Instruction::SpawnExec(_)));
    match &p.instructions[2] {
        Instruction::Jump { target } => assert_eq!(*target, 4),
        other => panic!("expected Jump, got {other:?}"),
    }
    assert!(matches!(p.instructions[3], Instruction::SpawnExec(_)));
}

/// `when` carries its own per-step `timeout` inside the nested
/// `RawExec`. Confirms the predicate's `ExecAction.timeout` is
/// threaded from TOML, distinct from the surrounding action's
/// top-level (forbidden) `timeout`.
#[test]
fn conditional_predicate_carries_per_step_timeout() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"], timeout = \"2s\" }, \
                    then = [{ exec = [\"yes\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    match &cfg.watches[0].program.instructions[0] {
        Instruction::SpawnPredicate { exec, .. } => {
            assert_eq!(exec.timeout, Some(Duration::from_secs(2)));
        }
        other => panic!("expected SpawnPredicate, got {other:?}"),
    }
}

/// `when` without `then` is rejected at the variant-completeness
/// gate as [`IssueKind::ConditionalIncomplete`]. Operators get a
/// single, clear "you wrote half a conditional" error.
#[test]
fn conditional_when_without_then_is_rejected() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [{ when = { exec = [\"check\"] } }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::ConditionalIncomplete),
        "got {errors:?}",
    );
}

/// `then` without `when` is the symmetric case — also flagged as
/// [`IssueKind::ConditionalIncomplete`].
#[test]
fn conditional_then_without_when_is_rejected() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [{ then = [{ exec = [\"y\"] }] }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::ConditionalIncomplete),
        "got {errors:?}",
    );
}

/// Conditional with empty `then` and no `else` (or empty `else`) is
/// pointless — the predicate would have no observable effect.
/// Rejected as [`IssueKind::EmptyConditional`].
#[test]
fn conditional_empty_then_and_no_else_is_rejected() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [{ when = { exec = [\"check\"] }, then = [] }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors.iter().any(|e| e.kind == IssueKind::EmptyConditional),
        "got {errors:?}",
    );
}

/// Empty `then` with non-empty `else` is allowed — operationally
/// equivalent to a negated predicate (run else iff predicate failed).
/// Lowers to a 3-instruction program: predicate, Jump, else.
#[test]
fn conditional_empty_then_nonempty_else_is_allowed() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, then = [], else = [{ exec = [\"no\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.instructions.len(), 3);
    match &p.instructions[0] {
        Instruction::SpawnPredicate { jump_target, .. } => {
            assert_eq!(*jump_target, 2, "predicate jumps directly to else_start");
        }
        other => panic!("expected SpawnPredicate, got {other:?}"),
    }
    match &p.instructions[1] {
        Instruction::Jump { target } => {
            assert_eq!(*target, 3, "Jump (Ok path) skips past else");
        }
        other => panic!("expected Jump, got {other:?}"),
    }
    assert!(matches!(p.instructions[2], Instruction::SpawnExec(_)));
}

/// Conditional with both `exec` and `when` set is ambiguous — flagged
/// as [`IssueKind::ActionAmbiguousVariant`]. The exactly-one-variant
/// rule applies across exec / pipe / conditional uniformly.
#[test]
fn exec_and_conditional_together_is_rejected() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { exec = [\"a\"], when = { exec = [\"b\"] }, then = [{ exec = [\"c\"] }] },\n\
                ]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::ActionAmbiguousVariant),
        "got {errors:?}",
    );
}

/// Top-level `timeout` on a conditional action is rejected — the
/// predicate carries its own per-step timeout inside the nested
/// `RawExec`. Operators sometimes try `timeout` at the outer scope;
/// the validator catches it as [`IssueKind::TimeoutNotApplicable`].
#[test]
fn conditional_top_level_timeout_is_rejected() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, \
                    then = [{ exec = [\"y\"] }], timeout = \"5s\" },\n\
                ]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::TimeoutNotApplicable),
        "got {errors:?}",
    );
}

/// Nested conditional inside `then` lowers recursively — each
/// nesting level produces its own `SpawnPredicate` with correctly
/// backpatched jumps. Pinned to detect off-by-one drift in
/// [`crate::action::lower_actions`] under nested input.
#[test]
fn nested_conditional_inside_then_lowers_recursively() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"outer\"] }, \
                    then = [{ when = { exec = [\"inner\"] }, \
                              then = [{ exec = [\"y\"] }] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    // [SpawnPredicate(outer, jump=3), SpawnPredicate(inner, jump=3), SpawnExec(y)]
    assert_eq!(p.instructions.len(), 3);
    let preds: Vec<u32> = p
        .instructions
        .iter()
        .filter_map(|i| match i {
            Instruction::SpawnPredicate { jump_target, .. } => Some(*jump_target),
            _ => None,
        })
        .collect();
    assert_eq!(preds, vec![3, 3], "both predicates jump past plan end");
}

/// Empty `else` array (explicit `else = []`) is normalised to "no
/// else" — lowering omits the Jump instruction since the else-body
/// would be a zero-length sequence. The predicate's `jump_target`
/// points one past the then-branch end, mirroring the no-`else` form.
#[test]
fn conditional_explicit_empty_else_normalises_to_no_else() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, \
                    then = [{ exec = [\"y\"] }], else = [] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.instructions.len(), 2, "no Jump emitted for empty else");
    assert!(matches!(
        p.instructions[0],
        Instruction::SpawnPredicate { .. }
    ));
    assert!(matches!(p.instructions[1], Instruction::SpawnExec(_)));
}

/// Two-stage pipe lowers to a single `Instruction::SpawnPipe` with
/// the same number of stages as the TOML array. Each stage's argv
/// is preserved verbatim.
#[test]
fn pipe_two_stages_lowers_to_single_instruction() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [\n\
                  { pipe = [\n\
                    { exec = [\"grep\", \"foo\"] },\n\
                    { exec = [\"sort\"] },\n\
                  ] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.instructions.len(), 1);
    match &p.instructions[0] {
        Instruction::SpawnPipe(stages) => {
            assert_eq!(stages.len(), 2);
            assert_eq!(stages[0].argv.len(), 2);
            assert_eq!(stages[1].argv.len(), 1);
            assert_eq!(stages[0].argv[0].parts[0], ArgPart::literal("grep"));
            assert_eq!(stages[1].argv[0].parts[0], ArgPart::literal("sort"));
        }
        other => panic!("expected SpawnPipe; got {other:?}"),
    }
}

/// Each pipe stage carries its own per-stage `timeout`. Validation
/// threads the Duration onto the `ExecAction.timeout` field of the
/// corresponding stage. Stages without a timeout have `None`.
#[test]
fn pipe_stage_timeouts_threaded_per_stage() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [\n\
                  { pipe = [\n\
                    { exec = [\"a\"], timeout = \"500ms\" },\n\
                    { exec = [\"b\"] },\n\
                    { exec = [\"c\"], timeout = \"2s\" },\n\
                  ] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let Instruction::SpawnPipe(stages) = &cfg.watches[0].program.instructions[0] else {
        panic!("expected SpawnPipe");
    };
    assert_eq!(stages[0].timeout, Some(Duration::from_millis(500)));
    assert_eq!(stages[1].timeout, None);
    assert_eq!(stages[2].timeout, Some(Duration::from_secs(2)));
}

/// Empty pipe is rejected as `IssueKind::EmptyPipe`.
#[test]
fn pipe_empty_rejected() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [{ pipe = [] }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors.iter().any(|e| e.kind == IssueKind::EmptyPipe),
        "expected EmptyPipe; got {errors:?}",
    );
}

/// Single-stage pipe is rejected as `IssueKind::SingleStagePipe` —
/// degenerate, the operator should use top-level `exec` directly.
#[test]
fn pipe_single_stage_rejected() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [{ pipe = [{ exec = [\"solo\"] }] }]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors.iter().any(|e| e.kind == IssueKind::SingleStagePipe),
        "expected SingleStagePipe; got {errors:?}",
    );
}

/// Top-level `timeout` on a pipe action is rejected — pipe stages
/// each set their own `timeout` on the nested `RawExec`.
#[test]
fn pipe_top_level_timeout_rejected() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [\n\
                  { pipe = [\n\
                    { exec = [\"a\"] },\n\
                    { exec = [\"b\"] },\n\
                  ], timeout = \"5s\" },\n\
                ]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::TimeoutNotApplicable),
        "expected TimeoutNotApplicable; got {errors:?}",
    );
}

/// `exec` and `pipe` set simultaneously is rejected as
/// `IssueKind::ActionAmbiguousVariant`.
#[test]
fn pipe_with_exec_simultaneously_rejected() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [\n\
                  { exec = [\"a\"], pipe = [\n\
                    { exec = [\"b\"] },\n\
                    { exec = [\"c\"] },\n\
                  ] },\n\
                ]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors
            .iter()
            .any(|e| e.kind == IssueKind::ActionAmbiguousVariant),
        "expected ActionAmbiguousVariant; got {errors:?}",
    );
}

/// An empty `exec` argv inside a pipe stage surfaces as
/// `IssueKind::EmptyArgv` with a path label that locates the offending
/// stage. The structural pipe check (>=2 stages) still passes — the
/// per-stage validation catches the empty argv.
#[test]
fn pipe_stage_with_empty_argv_rejected() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [\n\
                  { pipe = [\n\
                    { exec = [\"a\"] },\n\
                    { exec = [] },\n\
                  ] },\n\
                ]";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert!(
        errors.iter().any(|e| e.kind == IssueKind::EmptyArgv),
        "expected EmptyArgv on stage 1; got {errors:?}",
    );
}
