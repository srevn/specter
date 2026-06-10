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
    // minimal.toml has watch `build` with actions = [{ exec = ["echo"] }]. three-watches.toml has
    // `build` (cmd ["make"]), `lint`, `fmt`. small → big: `build`'s command changed (params-only ⇒
    // modified_params), `lint`+`fmt` are added.
    let small = load("minimal.toml");
    let big = load("three-watches.toml");

    let going_up = diff(&small, &big);
    assert_eq!(going_up.added.len(), 2);
    assert!(going_up.modified_identity.is_empty());
    assert_eq!(going_up.modified_params.len(), 1);
    assert!(going_up.removed.is_empty());

    let going_down = diff(&big, &small);
    assert!(going_down.added.is_empty());
    assert!(going_down.modified_identity.is_empty());
    assert_eq!(going_down.modified_params.len(), 1);
    assert_eq!(going_down.removed.len(), 2);
}

#[test]
fn identical_fixture_yields_no_diff() {
    let a = load("three-watches.toml");
    let b = load("three-watches.toml");
    let d = diff(&a, &b);
    assert!(d.added.is_empty());
    assert!(d.removed.is_empty());
    assert!(d.modified_identity.is_empty());
    assert!(d.modified_params.is_empty());
}

#[test]
fn reorder_only_yields_no_diff() {
    let a = load("three-watches.toml");
    let mut b_watches = a.watches.clone();
    b_watches.reverse();
    let b = Config {
        log: a.log.clone(),
        watches: b_watches,
    };
    let d = diff(&a, &b);
    assert!(d.added.is_empty(), "added: {:?}", d.added);
    assert!(d.removed.is_empty(), "removed: {:?}", d.removed);
    assert!(
        d.modified_identity.is_empty(),
        "modified_identity: {:?}",
        d.modified_identity,
    );
    assert!(
        d.modified_params.is_empty(),
        "modified_params: {:?}",
        d.modified_params,
    );
}

#[test]
fn changing_only_command_marks_modified_params() {
    let toml_a = "[[watch]]\nname = \"a\"\npath = \"/\"\nactions = [{ exec = [\"echo\"] }]";
    let toml_b = "[[watch]]\nname = \"a\"\npath = \"/\"\nactions = [{ exec = [\"fmt\"] }]";
    let a = Config::from_str(toml_a).unwrap();
    let b = Config::from_str(toml_b).unwrap();
    let d = diff(&a, &b);
    assert!(d.modified_identity.is_empty());
    assert_eq!(d.modified_params.len(), 1);
    assert_eq!(d.modified_params[0].params.name, "a");
}
