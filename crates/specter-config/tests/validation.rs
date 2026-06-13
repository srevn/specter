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

/// `/<regular-file>/missing` surfaces ENOTDIR from the kernel — a non-`NotFound` `io::Error` —
/// which routes through [`PathError::Inaccessible`] to [`IssueKind::PathInaccessible`]. Exercises
/// the same arm `chmod 0` (EACCES) hits, without the root-skip gymnastics; the EACCES surface is
/// verified manually (we can't drop privileges inside a test process).
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
    // Only lowercase non-catalog names trigger the typo error. Uppercase names (`$Path`,
    // `$SPECTER_PATH`, `$HOME`) pass through as literal so the spawned shell can expand them.
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
    // Unknown event-class strings reuse `InvalidEnum`, the same family as scope/log-level — keeps
    // the operator-experience symmetrical.
    let toml = format!(
        "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"echo\"] }}]\n\
         events = [\"strcuture\"]"
    );
    assert_kinds(&toml, &[IssueKind::InvalidEnum]);
}

/// Disabled entries flow through every validator unchanged: a structural error in a disabled entry
/// surfaces at load time rather than silently at re-enable time. Path is the most common typo
/// shape; one anchoring case is enough — the validator dispatcher runs the full pipeline regardless
/// of `enabled`.
#[test]
fn disabled_entry_does_not_waive_validation() {
    let toml = "[[watch]]\nname = \"a\"\npath = \"src\"\n\
                actions = [{ exec = [\"echo\"] }]\nenabled = false";
    assert_kinds(toml, &[IssueKind::NonAbsolute]);
}

/// Duplicate-name detection spans enabled + disabled entries: two watches with the same `name` (one
/// enabled, one disabled) would conflict at the bin's `name → SubId` map on flip, so preventing the
/// ambiguity at load time is the desired behavior.
#[test]
fn duplicate_name_across_enabled_and_disabled_rejected() {
    let toml = format!(
        "[[watch]]\nname = \"build\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"v1\"] }}]\nenabled = false\n\
         [[watch]]\nname = \"build\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"v2\"] }}]",
    );
    assert_kinds(&toml, &[IssueKind::DuplicateName]);
}

/// Dynamic-prefix canonicalisation, the composition payoff: a dynamic pattern's literal prefix is
/// symlink-resolved at lowering exactly like a static path, so a dynamic watch and a static watch
/// over the *same* symlinked tree anchor the same Tree branch (they compose). The
/// divergent-anchor advisory that the verbatim regime needed has nothing left to report — the only
/// remaining advisory channel is the events-incomplete mask.
#[cfg(unix)]
#[test]
fn dynamic_prefix_canonicalises_and_composes_with_static() {
    let td = tempfile::tempdir().unwrap();
    let canon_root = td.path().canonicalize().unwrap();
    let target = canon_root.join("target");
    std::fs::create_dir(&target).unwrap();
    let link = canon_root.join("link");
    std::os::unix::fs::symlink(&target, &link).unwrap();

    let toml = format!(
        "[[watch]]\nname = \"dynamic\"\npath = \"{link}/*/log\"\nactions = [{{ exec = [\"echo\"] }}]\n\
         [[watch]]\nname = \"static\"\npath = \"{link}\"\nactions = [{{ exec = [\"echo\"] }}]",
        link = link.display(),
    );
    let cfg = Config::from_str(&toml).unwrap();

    // Both anchors resolved through the symlink onto the same canonical branch.
    assert_eq!(
        cfg.watches[0].path, target,
        "the dynamic pattern's literal prefix resolves to the link target",
    );
    assert_eq!(
        cfg.watches[1].path, target,
        "the static sibling resolves to the same target — they compose",
    );

    // Nothing diverges, so the once-divergent advisory is silent; the default mask witnesses
    // CONTENT, so the events advisory stays silent too.
    assert!(
        cfg.warnings().is_empty(),
        "canonical anchors produce no advisory: {:?}",
        cfg.warnings(),
    );
}

/// A dynamic pattern whose literal prefix genuinely faults — here ENOTDIR, the prefix traversing a
/// regular file — fails to load with the same fatal [`IssueKind::PathInaccessible`] a static path
/// produces. Canonicalisation is symmetric across the static/dynamic split: a prefix that cannot
/// resolve is a load error, not a silent runtime park.
#[cfg(unix)]
#[test]
fn dynamic_prefix_inaccessible_is_fatal() {
    let td = tempfile::tempdir().unwrap();
    let file = td.path().join("not-a-dir");
    std::fs::write(&file, b"x").unwrap();
    // Literal prefix `<file>/sub` descends through a regular file → ENOTDIR (non-`NotFound`).
    let toml = format!(
        "[[watch]]\nname = \"d\"\npath = \"{}/sub/*.log\"\nactions = [{{ exec = [\"echo\"] }}]",
        file.display(),
    );
    let errors = validation_errors(Config::from_str(&toml).unwrap_err());
    assert_eq!(errors.len(), 1, "got {errors:?}");
    assert_eq!(errors[0].kind, IssueKind::PathInaccessible);
    assert_eq!(errors[0].field, "path");
}

/// [`Config::warnings`], the events-incomplete advisory: a mask that cannot witness its scan shape's
/// quiescence classes engages the hash-channel safety net (two consecutive agreeing full subtree
/// walks per fire), which is supported but expensive — the warning surfaces the cost. The checked
/// identity is the one the firing Profiles run under: a static entry's own `events`, a dynamic
/// entry's template `events`. The scope-conditional default mask carries CONTENT and stays silent.
#[test]
fn warnings_flag_events_incomplete_mask() {
    let toml = format!(
        "[[watch]]\nname = \"structure-only\"\npath = \"{ROOT}\"\n\
         actions = [{{ exec = [\"echo\"] }}]\nevents = [\"structure\"]\n\
         [[watch]]\nname = \"defaulted\"\npath = \"{ROOT}\"\n\
         actions = [{{ exec = [\"echo\"] }}]\n\
         [[watch]]\nname = \"dynamic\"\npath = \"{ROOT}*/log\"\n\
         actions = [{{ exec = [\"echo\"] }}]\nevents = [\"structure\"]",
    );
    let warnings = Config::from_str(&toml).unwrap().warnings();
    assert_eq!(
        warnings.len(),
        2,
        "the explicit structure-only static mask and the structure-only template warn; \
         the CONTENT-bearing default is silent: {warnings:?}",
    );
    for (w, idx) in warnings.iter().zip([0usize, 2]) {
        assert_eq!(w.kind, IssueKind::EventsIncompleteMask);
        assert_eq!(w.watch_index, Some(idx));
        assert_eq!(w.field, "events");
    }
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
