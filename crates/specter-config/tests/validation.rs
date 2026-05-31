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
fn issue_kind_empty_name() {
    let toml =
        format!("[[watch]]\nname = \"\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]");
    assert_kinds(&toml, &[IssueKind::EmptyName]);
}

#[test]
fn issue_kind_empty_command() {
    let toml = format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [] }}]");
    assert_kinds(&toml, &[IssueKind::EmptyArgv]);
}

#[test]
fn issue_kind_empty_argv() {
    let toml =
        format!("[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"\"] }}]");
    assert_kinds(&toml, &[IssueKind::EmptyArgv]);
}

#[test]
fn issue_kind_non_absolute() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"src\"\nactions = [{ exec = [\"echo\"] }]";
    assert_kinds(toml, &[IssueKind::NonAbsolute]);
}

#[test]
fn issue_kind_empty_path() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"\"\nactions = [{ exec = [\"echo\"] }]";
    assert_kinds(toml, &[IssueKind::EmptyPath]);
}

#[test]
fn issue_kind_path_contains_parent_dir() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"/srv/..\"\nactions = [{ exec = [\"echo\"] }]";
    assert_kinds(toml, &[IssueKind::PathContainsParentDir]);
}

/// `/<regular-file>/missing` surfaces ENOTDIR from the kernel — a
/// non-`NotFound` `io::Error` — which routes through
/// [`PathError::Inaccessible`] to [`IssueKind::PathInaccessible`].
/// Exercises the same arm `chmod 0` (EACCES) hits, without the
/// root-skip gymnastics; the EACCES surface is verified manually
/// (we can't drop privileges inside a test process).
#[cfg(unix)]
#[test]
fn issue_kind_path_inaccessible() {
    let td = tempfile::tempdir().unwrap();
    let canon = td.path().canonicalize().unwrap();
    let file = canon.join("regular-file");
    std::fs::write(&file, b"hi").unwrap();
    let bad = file.join("missing-child");
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{}\"\nactions = [{{ exec = [\"echo\"] }}]",
        bad.display(),
    );
    assert_kinds(&toml, &[IssueKind::PathInaccessible]);
}

#[test]
fn issue_kind_invalid_glob_pattern() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\npattern = \"[bad\""
    );
    assert_kinds(&toml, &[IssueKind::InvalidGlob]);
}

#[test]
fn issue_kind_invalid_glob_exclude() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nexclude = [\"[bad\"]"
    );
    assert_kinds(&toml, &[IssueKind::InvalidGlob]);
}

#[test]
fn issue_kind_unknown_placeholder() {
    // Only lowercase non-catalog names trigger the typo error. Uppercase
    // names (`$Path`, `$SPECTER_PATH`, `$HOME`) pass through as literal
    // so the spawned shell can expand them.
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"${{specter.paht}}\"] }}]"
    );
    assert_kinds(&toml, &[IssueKind::UnknownPlaceholder]);
}

#[test]
fn issue_kind_settle_too_small() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nsettle = \"0ms\""
    );
    assert_kinds(&toml, &[IssueKind::SettleTooSmall]);
}

#[test]
fn issue_kind_max_settle_too_small() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
         settle = \"100ms\"\nmax_settle = \"200ms\""
    );
    assert_kinds(&toml, &[IssueKind::MaxSettleTooSmall]);
}

#[test]
fn issue_kind_max_depth_zero() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nmax_depth = 0"
    );
    assert_kinds(&toml, &[IssueKind::MaxDepthZero]);
}

#[test]
fn issue_kind_duplicate_name() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
         [[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]"
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
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nscope = \"weekly\""
    );
    assert_kinds(&toml, &[IssueKind::InvalidEnum]);
}

#[test]
fn issue_kind_events_empty() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\nevents = []"
    );
    assert_kinds(&toml, &[IssueKind::EventsEmpty]);
}

#[test]
fn issue_kind_duplicate_event_class() {
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
         events = [\"content\", \"content\"]"
    );
    assert_kinds(&toml, &[IssueKind::DuplicateEventClass]);
}

#[test]
fn issue_kind_invalid_enum_event_class() {
    // Unknown event-class strings reuse `InvalidEnum`, the same family
    // as scope/log-level — keeps the operator-experience symmetrical.
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
         events = [\"strcuture\"]"
    );
    assert_kinds(&toml, &[IssueKind::InvalidEnum]);
}

/// Disabled entries flow through every validator unchanged: a
/// structural error in a disabled entry surfaces at load time rather
/// than silently at re-enable time. Path is the most common typo
/// shape; one anchoring case is enough — the validator dispatcher
/// runs the full pipeline regardless of `enabled`.
#[test]
fn disabled_entry_does_not_waive_validation() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"src\"\n\
                actions = [{ exec = [\"echo\"] }]\nenabled = false";
    assert_kinds(toml, &[IssueKind::NonAbsolute]);
}

/// Duplicate-name detection spans enabled + disabled entries: two
/// watches with the same `name` (one enabled, one disabled) would
/// conflict at the bin's `name → SubId` map on flip, so preventing
/// the ambiguity at load time is the desired behavior.
#[test]
fn duplicate_name_across_enabled_and_disabled_rejected() {
    let toml = format!(
        "[[watch]]\nname = \"build\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"v1\"] }}]\nenabled = false\n\
         [[watch]]\nname = \"build\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"v2\"] }}]",
    );
    assert_kinds(&toml, &[IssueKind::DuplicateName]);
}

#[test]
fn kitchen_sink_collects_five_distinct_issues() {
    let toml = "[[watch]]\nname = \"\"\npath = \"src\"\nactions = [{ exec = [] }]\nsettle = \"0ms\"\nmax_depth = 0";
    let err = Config::from_str(toml).unwrap_err();
    let errors = validation_errors(err);
    assert_eq!(errors.len(), 5, "got {errors:?}");
    let kinds: Vec<IssueKind> = errors.iter().map(|e| e.kind).collect();
    for expected in [
        IssueKind::EmptyName,
        IssueKind::NonAbsolute,
        IssueKind::EmptyArgv,
        IssueKind::SettleTooSmall,
        IssueKind::MaxDepthZero,
    ] {
        assert!(
            kinds.contains(&expected),
            "missing {expected:?} in {kinds:?}"
        );
    }
}
