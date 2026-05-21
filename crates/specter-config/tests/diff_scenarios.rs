//! Integration tests: hot-reload diff across config-pair scenarios.

use specter_config::{Config, diff};
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load(name: &str) -> Config {
    Config::from_path(&fixtures_dir().join(name)).expect("fixture parses")
}

#[test]
fn three_watches_against_minimal_classifies_each_correctly() {
    // minimal.toml has watch `build` with actions = [{ exec = ["echo"] }].
    // three-watches.toml has `build` (cmd ["make"]), `lint`, `fmt`.
    // small → big: `build`'s command changed (params-only ⇒ modified_params),
    // `lint`+`fmt` are added.
    let small = load("minimal.toml");
    let big = load("three-watches.toml");

    let going_up = diff(&small, &big);
    assert_eq!(going_up.subs.added.len(), 2);
    assert!(going_up.subs.modified_identity.is_empty());
    assert_eq!(going_up.subs.modified_params.len(), 1);
    assert!(going_up.subs.removed.is_empty());

    let going_down = diff(&big, &small);
    assert!(going_down.subs.added.is_empty());
    assert!(going_down.subs.modified_identity.is_empty());
    assert_eq!(going_down.subs.modified_params.len(), 1);
    assert_eq!(going_down.subs.removed.len(), 2);
}

#[test]
fn identical_fixture_yields_no_diff() {
    let a = load("three-watches.toml");
    let b = load("three-watches.toml");
    let d = diff(&a, &b);
    assert!(d.subs.added.is_empty());
    assert!(d.subs.removed.is_empty());
    assert!(d.subs.modified_identity.is_empty());
    assert!(d.subs.modified_params.is_empty());
    assert!(d.promoters.added.is_empty());
    assert!(d.promoters.removed.is_empty());
    assert!(d.promoters.modified.is_empty());
}

#[test]
fn reorder_only_yields_no_diff() {
    let a = load("three-watches.toml");
    let mut b_watches = a.watches.clone();
    b_watches.reverse();
    let b = Config {
        log: a.log.clone(),
        watches: b_watches,
        promoters: a.promoters.clone(),
    };
    let d = diff(&a, &b);
    assert!(d.subs.added.is_empty(), "added: {:?}", d.subs.added);
    assert!(d.subs.removed.is_empty(), "removed: {:?}", d.subs.removed);
    assert!(
        d.subs.modified_identity.is_empty(),
        "modified_identity: {:?}",
        d.subs.modified_identity,
    );
    assert!(
        d.subs.modified_params.is_empty(),
        "modified_params: {:?}",
        d.subs.modified_params,
    );
}

#[test]
fn changing_only_command_marks_modified_params() {
    let toml_a = "[[watch]]\nname = \"a\"\npath = \"/\"\nactions = [{ exec = [\"echo\"] }]";
    let toml_b = "[[watch]]\nname = \"a\"\npath = \"/\"\nactions = [{ exec = [\"fmt\"] }]";
    let a = Config::from_str(toml_a).unwrap();
    let b = Config::from_str(toml_b).unwrap();
    let d = diff(&a, &b);
    assert!(d.subs.modified_identity.is_empty());
    assert_eq!(d.subs.modified_params.len(), 1);
    assert_eq!(d.subs.modified_params[0].params.name, "a");
}
