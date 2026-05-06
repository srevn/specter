//! Integration tests: hot-reload diff across config-pair scenarios.

use compact_str::CompactString;
use slotmap::KeyData;
use specter_config::{Config, diff};
use specter_core::SubId;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

fn fixtures_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load(name: &str) -> Config {
    Config::from_path(&fixtures_dir().join(name)).expect("fixture parses")
}

fn id(n: u64) -> SubId {
    SubId::from(KeyData::from_ffi(n))
}

fn ids_for(cfg: &Config, start: u64) -> BTreeMap<CompactString, SubId> {
    cfg.watches
        .iter()
        .enumerate()
        .map(|(i, w)| (w.name.clone(), id(start + i as u64)))
        .collect()
}

#[test]
fn three_watches_against_minimal_classifies_each_correctly() {
    // minimal.toml has watch `build` with command = ["echo"].
    // three-watches.toml has `build` (cmd ["make"]), `lint`, `fmt`.
    // small → big: `build` is modified, `lint`+`fmt` are added.
    let small = load("minimal.toml");
    let big = load("three-watches.toml");

    let going_up = diff(&small, &big, &ids_for(&small, 1));
    assert_eq!(going_up.added.len(), 2);
    assert_eq!(going_up.modified.len(), 1);
    assert!(going_up.removed.is_empty());

    let going_down = diff(&big, &small, &ids_for(&big, 1));
    assert!(going_down.added.is_empty());
    assert_eq!(going_down.modified.len(), 1);
    assert_eq!(going_down.removed.len(), 2);
}

#[test]
fn identical_fixture_yields_no_diff() {
    let a = load("three-watches.toml");
    let b = load("three-watches.toml");
    let d = diff(&a, &b, &ids_for(&a, 1));
    assert!(d.added.is_empty());
    assert!(d.removed.is_empty());
    assert!(d.modified.is_empty());
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
    let d = diff(&a, &b, &ids_for(&a, 1));
    assert!(d.added.is_empty(), "added: {:?}", d.added);
    assert!(d.removed.is_empty(), "removed: {:?}", d.removed);
    assert!(d.modified.is_empty(), "modified: {:?}", d.modified);
}

#[test]
fn changing_only_command_marks_modified() {
    let toml_a = "[[watch]]\nname = \"a\"\npath = \"/\"\ncommand = [\"echo\"]";
    let toml_b = "[[watch]]\nname = \"a\"\npath = \"/\"\ncommand = [\"fmt\"]";
    let a = Config::from_str(toml_a).unwrap();
    let b = Config::from_str(toml_b).unwrap();
    let ids = ids_for(&a, 1);
    let d = diff(&a, &b, &ids);
    assert_eq!(d.modified.len(), 1);
    assert_eq!(d.modified[0].0, id(1));
}
