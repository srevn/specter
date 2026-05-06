//! Integration tests: every `IssueKind` produced by an explicit fixture.

use specter_config::{Config, ConfigError, IssueKind, ValidationIssue};

const ROOT: &str = "/";

fn validation_errors(err: ConfigError) -> Vec<ValidationIssue> {
    match err {
        ConfigError::Validate { errors, .. } => errors,
        other => panic!("expected Validate, got {other:?}"),
    }
}

fn assert_kinds(toml: &str, expected: &[IssueKind]) {
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
    for k in expected {
        assert!(
            kinds.contains(k),
            "expected {k:?} in {kinds:?} (issues: {errors:?})",
        );
    }
}

#[test]
fn issue_kind_empty_for_blank_name() {
    let toml = format!("[[watch]]\nname = \"\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]");
    assert_kinds(&toml, &[IssueKind::Empty]);
}

#[test]
fn issue_kind_empty_command() {
    let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = []");
    assert_kinds(&toml, &[IssueKind::EmptyCommand]);
}

#[test]
fn issue_kind_empty_argv() {
    let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"\"]");
    assert_kinds(&toml, &[IssueKind::EmptyArgv]);
}

#[test]
fn issue_kind_non_absolute() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"src\"\ncommand = [\"echo\"]";
    assert_kinds(toml, &[IssueKind::NonAbsolute]);
}

#[test]
fn issue_kind_invalid_glob_pattern() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\npattern = \"[bad\""
    );
    assert_kinds(&toml, &[IssueKind::InvalidGlob]);
}

#[test]
fn issue_kind_invalid_glob_exclude() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nexclude = [\"[bad\"]"
    );
    assert_kinds(&toml, &[IssueKind::InvalidGlob]);
}

#[test]
fn issue_kind_unknown_placeholder() {
    // Only lowercase non-catalog names trigger the typo error. Uppercase
    // names (`$Path`, `$SPECTER_PATH`, `$HOME`) pass through as literal
    // so the spawned shell can expand them.
    let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"$paht\"]");
    assert_kinds(&toml, &[IssueKind::UnknownPlaceholder]);
}

#[test]
fn issue_kind_settle_too_small() {
    let toml =
        format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nsettle_ms = 0");
    assert_kinds(&toml, &[IssueKind::SettleTooSmall]);
}

#[test]
fn issue_kind_max_settle_too_small() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
         settle_ms = 100\nmax_settle_ms = 200"
    );
    assert_kinds(&toml, &[IssueKind::MaxSettleTooSmall]);
}

#[test]
fn issue_kind_max_settle_too_large() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nmax_settle_ms = 4000000"
    );
    assert_kinds(&toml, &[IssueKind::MaxSettleTooLarge]);
}

#[test]
fn issue_kind_max_depth_zero() {
    let toml =
        format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nmax_depth = 0");
    assert_kinds(&toml, &[IssueKind::MaxDepthZero]);
}

#[test]
fn issue_kind_duplicate_name() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
         [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]"
    );
    assert_kinds(&toml, &[IssueKind::DuplicateName]);
}

#[test]
fn issue_kind_invalid_enum_log_level() {
    let toml = "[log]\nlevel = \"verbose\"";
    assert_kinds(toml, &[IssueKind::InvalidEnum]);
}

#[test]
fn issue_kind_invalid_enum_scope() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nscope = \"weekly\""
    );
    assert_kinds(&toml, &[IssueKind::InvalidEnum]);
}

#[test]
fn issue_kind_events_empty() {
    let toml =
        format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\nevents = []");
    assert_kinds(&toml, &[IssueKind::EventsEmpty]);
}

#[test]
fn issue_kind_duplicate_event_class() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
         events = [\"content\", \"content\"]"
    );
    assert_kinds(&toml, &[IssueKind::DuplicateEventClass]);
}

#[test]
fn issue_kind_invalid_enum_event_class() {
    // Unknown event-class strings reuse `InvalidEnum`, the same family
    // as scope/log-level — keeps the operator-experience symmetrical.
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\ncommand = [\"echo\"]\n\
         events = [\"strcuture\"]"
    );
    assert_kinds(&toml, &[IssueKind::InvalidEnum]);
}

#[test]
fn kitchen_sink_collects_five_distinct_issues() {
    let toml = "[[watch]]\nname = \"\"\npath = \"src\"\ncommand = []\nsettle_ms = 0\nmax_depth = 0";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert_eq!(errors.len(), 5, "got {errors:?}");
    let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
    for expected in [
        IssueKind::Empty,
        IssueKind::NonAbsolute,
        IssueKind::EmptyCommand,
        IssueKind::SettleTooSmall,
        IssueKind::MaxDepthZero,
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
}
