//! Integration tests: hot-reload diff across config-pair scenarios.

use compact_str::CompactString;
use slotmap::KeyData;
use specter_config::{Config, diff};
use specter_core::{PromoterId, SubId};
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

fn sub_ids_for(cfg: &Config, start: u64) -> BTreeMap<CompactString, SubId> {
    cfg.watches
        .iter()
        .enumerate()
        .map(|(i, w)| (w.name.clone(), id(start + i as u64)))
        .collect()
}

const fn empty_promoter_ids() -> BTreeMap<CompactString, PromoterId> {
    BTreeMap::new()
}

#[test]
fn three_watches_against_minimal_classifies_each_correctly() {
    // minimal.toml has watch `build` with command = ["echo"].
    // three-watches.toml has `build` (cmd ["make"]), `lint`, `fmt`.
    // small → big: `build` is modified, `lint`+`fmt` are added.
    let small = load("minimal.toml");
    let big = load("three-watches.toml");

    let going_up = diff(&small, &big, &sub_ids_for(&small, 1), &empty_promoter_ids());
    assert_eq!(going_up.subs.added.len(), 2);
    assert_eq!(going_up.subs.modified.len(), 1);
    assert!(going_up.subs.removed.is_empty());

    let going_down = diff(&big, &small, &sub_ids_for(&big, 1), &empty_promoter_ids());
    assert!(going_down.subs.added.is_empty());
    assert_eq!(going_down.subs.modified.len(), 1);
    assert_eq!(going_down.subs.removed.len(), 2);
}

#[test]
fn identical_fixture_yields_no_diff() {
    let a = load("three-watches.toml");
    let b = load("three-watches.toml");
    let d = diff(&a, &b, &sub_ids_for(&a, 1), &empty_promoter_ids());
    assert!(d.subs.added.is_empty());
    assert!(d.subs.removed.is_empty());
    assert!(d.subs.modified.is_empty());
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
    let d = diff(&a, &b, &sub_ids_for(&a, 1), &empty_promoter_ids());
    assert!(d.subs.added.is_empty(), "added: {:?}", d.subs.added);
    assert!(d.subs.removed.is_empty(), "removed: {:?}", d.subs.removed);
    assert!(
        d.subs.modified.is_empty(),
        "modified: {:?}",
        d.subs.modified
    );
}

#[test]
fn changing_only_command_marks_modified() {
    let toml_a = "[[watch]]\nname = \"a\"\npath = \"/\"\ncommand = [\"echo\"]";
    let toml_b = "[[watch]]\nname = \"a\"\npath = \"/\"\ncommand = [\"fmt\"]";
    let a = Config::from_str(toml_a).unwrap();
    let b = Config::from_str(toml_b).unwrap();
    let ids = sub_ids_for(&a, 1);
    let d = diff(&a, &b, &ids, &empty_promoter_ids());
    assert_eq!(d.subs.modified.len(), 1);
    assert_eq!(d.subs.modified[0].0, id(1));
}
