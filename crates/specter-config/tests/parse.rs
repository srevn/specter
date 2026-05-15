//! Integration tests: parse + validate against TOML fixtures.

use specter_config::{Config, ConfigError, IssueKind};
use specter_core::program::{BranchTarget, SpawnBody};
use specter_core::{ArgPart, ClassSet, EffectScope, Placeholder};
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
    let SpawnBody::Exec(exec) = &w.program.ops()[0].body() else {
        panic!("expected SpawnBody::Exec");
    };
    assert_eq!(exec.argv().len(), 3);
    assert_eq!(exec.argv()[0].parts()[0], ArgPart::literal("make"));
    assert_eq!(exec.argv()[1].parts()[0], ArgPart::literal("--input="));
    assert_eq!(
        exec.argv()[1].parts()[1],
        ArgPart::Placeholder(Placeholder::Path)
    );
    assert_eq!(
        exec.argv()[2].parts()[0],
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
        .ops()
        .iter()
        .map(|op| match op.body() {
            SpawnBody::Exec(e) => e.timeout(),
            other @ SpawnBody::Pipe(_) => panic!("expected SpawnBody::Exec, got {other:?}"),
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
    let exec = match cfg.watches[0].program.ops()[0].body() {
        SpawnBody::Exec(e) => e,
        other @ SpawnBody::Pipe(_) => panic!("expected SpawnBody::Exec, got {other:?}"),
    };
    match &exec.argv()[1].parts()[0] {
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

/// Conditional with `when` + `then` lowers to a two-op program:
/// predicate op (Exec body, `on_failed = Escape` — "branch, not guard")
/// then the then-branch's Exec op. With no `else`, the predicate's
/// `on_failed = Escape` (the "branch, not guard" outcome elision —
/// predicate Failed terminates the plan Ok without propagating).
#[test]
fn conditional_when_then_lowers_to_predicate_plus_then() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, then = [{ exec = [\"yes\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.ops().len(), 2);
    // Predicate op (cursor 0): Exec body; on_ok continues to then-branch
    // (op 1); on_failed = Escape (no-else: terminate Ok without propagation).
    assert!(matches!(p.ops()[0].body(), SpawnBody::Exec(_)));
    match p.ops()[0].on_ok() {
        BranchTarget::Continue(idx) => assert_eq!(idx.get(), 1, "predicate Ok enters then"),
        other => panic!("expected Continue(1), got {other:?}"),
    }
    assert_eq!(p.ops()[0].on_failed(), BranchTarget::Escape);
    // Then-exec (cursor 1): on_ok = Escape (top-level natural completion);
    // on_failed = Terminate (stop-on-failure, outcome propagates).
    assert!(matches!(p.ops()[1].body(), SpawnBody::Exec(_)));
    assert_eq!(p.ops()[1].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[1].on_failed(), BranchTarget::Terminate);
}

/// Conditional with `when` + `then` + `else` lowers to a three-op
/// program (the explicit Jump opcode is gone — the CFG-shaped IR encodes
/// the skip via edges instead). Layout:
///
/// - op 0: predicate Exec — `on_ok = Continue(1)` (enter then),
///   `on_failed = Continue(2)` (enter else).
/// - op 1: then-Exec — `on_ok = Escape` (skip past else), `on_failed =
///   Terminate` (stop-on-failure).
/// - op 2: else-Exec — `on_ok = Escape`, `on_failed = Terminate`.
#[test]
fn conditional_with_else_lowers_to_predicate_then_else_via_edges() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, \
                    then = [{ exec = [\"yes\"] }], \
                    else = [{ exec = [\"no\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.ops().len(), 3);
    // op 0: predicate
    assert!(matches!(p.ops()[0].body(), SpawnBody::Exec(_)));
    match p.ops()[0].on_ok() {
        BranchTarget::Continue(idx) => assert_eq!(idx.get(), 1),
        other => panic!("expected Continue(1), got {other:?}"),
    }
    match p.ops()[0].on_failed() {
        BranchTarget::Continue(idx) => assert_eq!(idx.get(), 2),
        other => panic!("expected Continue(2), got {other:?}"),
    }
    // op 1: then-Exec; on_ok = Escape skips past else.
    assert!(matches!(p.ops()[1].body(), SpawnBody::Exec(_)));
    assert_eq!(p.ops()[1].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[1].on_failed(), BranchTarget::Terminate);
    // op 2: else-Exec
    assert!(matches!(p.ops()[2].body(), SpawnBody::Exec(_)));
    assert_eq!(p.ops()[2].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[2].on_failed(), BranchTarget::Terminate);
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
    // The predicate op (cursor 0) carries an Exec body whose timeout
    // round-trips from TOML through validation into ExecAction.timeout.
    let p = &cfg.watches[0].program;
    let SpawnBody::Exec(exec) = &p.ops()[0].body() else {
        panic!("expected SpawnBody::Exec for predicate");
    };
    assert_eq!(exec.timeout(), Some(Duration::from_secs(2)));
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
/// Lowers to a 2-op program: predicate then else-Exec. The empty
/// then-block contributes no ops; the predicate's `on_ok` resolves to
/// `Escape` (the empty then-tail's escape — skip past else on Ok).
/// `on_failed` continues into the else-Exec.
#[test]
fn conditional_empty_then_nonempty_else_is_allowed() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, then = [], else = [{ exec = [\"no\"] }] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.ops().len(), 2);
    // op 0: predicate
    assert!(matches!(p.ops()[0].body(), SpawnBody::Exec(_)));
    // The empty then-block has no ops — predicate on_ok resolves
    // to Escape (skip past else on the Ok path).
    assert_eq!(p.ops()[0].on_ok(), BranchTarget::Escape);
    match p.ops()[0].on_failed() {
        BranchTarget::Continue(idx) => {
            assert_eq!(idx.get(), 1, "predicate Failed enters else-branch");
        }
        other => panic!("expected Continue(1), got {other:?}"),
    }
    // op 1: else-Exec
    assert!(matches!(p.ops()[1].body(), SpawnBody::Exec(_)));
    assert_eq!(p.ops()[1].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[1].on_failed(), BranchTarget::Terminate);
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
/// nesting level produces its own predicate op with edges patched to
/// the right surrounding context. Pinned to detect off-by-one drift in
/// [`crate::action::lower_actions`] under nested input.
///
/// Shape for outer-when=outer, outer-then=[inner-when, inner-then=[y]]:
///
/// - op 0: outer predicate Exec — `on_ok = Continue(1)` (enter outer-then),
///   `on_failed = Escape` (outer has no else, "branch-not-guard").
/// - op 1: inner predicate Exec — `on_ok = Continue(2)` (enter inner-then),
///   `on_failed = Escape` (inner has no else either).
/// - op 2: inner-then Exec `/bin/y` — `on_ok = Escape`, `on_failed = Terminate`.
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
    assert_eq!(p.ops().len(), 3);
    for op in p.ops() {
        assert!(matches!(op.body(), SpawnBody::Exec(_)));
    }
    // Outer predicate's on_failed is Escape (no-else "branch, not guard").
    assert_eq!(p.ops()[0].on_failed(), BranchTarget::Escape);
    match p.ops()[0].on_ok() {
        BranchTarget::Continue(idx) => assert_eq!(idx.get(), 1),
        other => panic!("op 0 on_ok: expected Continue(1), got {other:?}"),
    }
    // Inner predicate's on_failed is also Escape (its own no-else branch).
    assert_eq!(p.ops()[1].on_failed(), BranchTarget::Escape);
    match p.ops()[1].on_ok() {
        BranchTarget::Continue(idx) => assert_eq!(idx.get(), 2),
        other => panic!("op 1 on_ok: expected Continue(2), got {other:?}"),
    }
    // Inner-then Exec terminates the plan.
    assert_eq!(p.ops()[2].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[2].on_failed(), BranchTarget::Terminate);
}

/// Empty `else` array (explicit `else = []`) is normalised to "no
/// else" — the lowered shape is identical to the omitted-`else` form
/// (predicate + then, with predicate's `on_failed = Escape`).
#[test]
fn conditional_explicit_empty_else_normalises_to_no_else() {
    let toml = "[[watch]]\nname = \"c\"\npath = \"/\"\n\
                actions = [\n\
                  { when = { exec = [\"check\"] }, \
                    then = [{ exec = [\"y\"] }], else = [] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.ops().len(), 2);
    assert!(matches!(p.ops()[0].body(), SpawnBody::Exec(_)));
    // Predicate on_failed escapes (no-else "branch, not guard").
    assert_eq!(p.ops()[0].on_failed(), BranchTarget::Escape);
    match p.ops()[0].on_ok() {
        BranchTarget::Continue(idx) => assert_eq!(idx.get(), 1),
        other => panic!("expected Continue(1), got {other:?}"),
    }
    assert!(matches!(p.ops()[1].body(), SpawnBody::Exec(_)));
    assert_eq!(p.ops()[1].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[1].on_failed(), BranchTarget::Terminate);
}

/// Two-stage pipe lowers to a single op with `SpawnBody::Pipe` carrying
/// every stage. Each stage's argv is preserved verbatim.
#[test]
fn pipe_two_stages_lowers_to_single_op() {
    let toml = "[[watch]]\nname = \"p\"\npath = \"/data\"\n\
                actions = [\n\
                  { pipe = [\n\
                    { exec = [\"grep\", \"foo\"] },\n\
                    { exec = [\"sort\"] },\n\
                  ] },\n\
                ]";
    let cfg = Config::from_str(toml).unwrap();
    let p = &cfg.watches[0].program;
    assert_eq!(p.ops().len(), 1);
    match &p.ops()[0].body() {
        SpawnBody::Pipe(stages) => {
            assert_eq!(stages.len(), 2);
            assert_eq!(stages[0].argv().len(), 2);
            assert_eq!(stages[1].argv().len(), 1);
            assert_eq!(stages[0].argv()[0].parts()[0], ArgPart::literal("grep"));
            assert_eq!(stages[1].argv()[0].parts()[0], ArgPart::literal("sort"));
        }
        other @ SpawnBody::Exec(_) => panic!("expected SpawnBody::Pipe; got {other:?}"),
    }
    // Top-level pipe is the only op — on_ok escapes, on_failed terminates.
    assert_eq!(p.ops()[0].on_ok(), BranchTarget::Escape);
    assert_eq!(p.ops()[0].on_failed(), BranchTarget::Terminate);
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
    let SpawnBody::Pipe(stages) = &cfg.watches[0].program.ops()[0].body() else {
        panic!("expected SpawnBody::Pipe");
    };
    assert_eq!(stages[0].timeout(), Some(Duration::from_millis(500)));
    assert_eq!(stages[1].timeout(), None);
    assert_eq!(stages[2].timeout(), Some(Duration::from_secs(2)));
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
