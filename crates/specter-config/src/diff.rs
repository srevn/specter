use crate::config::{Config, PromoterSpec, SubSpec};
use compact_str::CompactString;
use specter_core::{
    PromoterAttachRequest, PromoterId, PromoterRegistryDiff, SubAttachRequest, SubId,
    SubRegistryDiff, WatchRegistryDiff,
};
use std::collections::BTreeMap;

/// Compute the full hot-reload diff between two validated [`Config`]s.
///
/// The returned [`WatchRegistryDiff`] carries both halves:
/// - [`WatchRegistryDiff::subs`] — Sub adds / removes / modifications
///   keyed against `sub_ids` (the bin's `name → SubId` map for
///   currently-attached static Subs).
/// - [`WatchRegistryDiff::promoters`] — Promoter adds / removes /
///   modifications keyed against `promoter_ids` (the analogous
///   `name → PromoterId` map).
///
/// **Static ↔ dynamic migration via path edit.** A `[[watch]]` whose
/// path edits across the `is_dynamic` boundary (e.g., `/foo` → `/foo/*`)
/// moves between [`Config::watches`] and [`Config::promoters`]
/// between reloads. The diff produces `subs.removed + promoters.added`
/// (or the reverse): a wholesale teardown then attach. The path
/// semantics meaningfully changed; merging across the boundary would
/// hide that.
///
/// Determinism: each side's `removed` and `modified` lists are sorted
/// by id; `added` preserves new-source order.
#[must_use]
pub fn diff(
    old: &Config,
    new: &Config,
    sub_ids: &BTreeMap<CompactString, SubId>,
    promoter_ids: &BTreeMap<CompactString, PromoterId>,
) -> WatchRegistryDiff {
    WatchRegistryDiff {
        subs: diff_subs(old, new, sub_ids),
        promoters: diff_promoters(old, new, promoter_ids),
    }
}

/// Static-Sub half of the watch-registry diff. Extracted so the two
/// halves are independently testable; the public entry point composes
/// both.
fn diff_subs(old: &Config, new: &Config, ids: &BTreeMap<CompactString, SubId>) -> SubRegistryDiff {
    let old_by_name: BTreeMap<&CompactString, &SubSpec> =
        old.watches.iter().map(|s| (&s.name, s)).collect();
    let new_by_name: BTreeMap<&CompactString, &SubSpec> =
        new.watches.iter().map(|s| (&s.name, s)).collect();

    let mut added: Vec<SubAttachRequest> = Vec::new();
    let mut modified: Vec<(SubId, SubAttachRequest)> = Vec::new();
    let mut removed: Vec<SubId> = Vec::new();

    for spec in &new.watches {
        match old_by_name.get(&spec.name) {
            None => added.push(spec.to_attach_request()),
            Some(old_spec) if **old_spec != *spec => {
                if let Some(&id) = ids.get(&spec.name) {
                    modified.push((id, spec.to_attach_request()));
                }
            }
            Some(_) => {}
        }
    }

    for name in old_by_name.keys() {
        if !new_by_name.contains_key(name)
            && let Some(&id) = ids.get(*name)
        {
            removed.push(id);
        }
    }

    removed.sort_unstable();
    modified.sort_by_key(|(id, _)| *id);

    SubRegistryDiff {
        added,
        removed,
        modified,
    }
}

/// Promoter half of the watch-registry diff. Mirrors [`diff_subs`]
/// against [`Config::promoters`] / `promoter_ids`.
fn diff_promoters(
    old: &Config,
    new: &Config,
    ids: &BTreeMap<CompactString, PromoterId>,
) -> PromoterRegistryDiff {
    let old_by_name: BTreeMap<&CompactString, &PromoterSpec> =
        old.promoters.iter().map(|p| (&p.name, p)).collect();
    let new_by_name: BTreeMap<&CompactString, &PromoterSpec> =
        new.promoters.iter().map(|p| (&p.name, p)).collect();

    let mut added: Vec<PromoterAttachRequest> = Vec::new();
    let mut modified: Vec<(PromoterId, PromoterAttachRequest)> = Vec::new();
    let mut removed: Vec<PromoterId> = Vec::new();

    for spec in &new.promoters {
        match old_by_name.get(&spec.name) {
            None => added.push(spec.to_attach_request()),
            Some(old_spec) if **old_spec != *spec => {
                if let Some(&id) = ids.get(&spec.name) {
                    modified.push((id, spec.to_attach_request()));
                }
            }
            Some(_) => {}
        }
    }

    for name in old_by_name.keys() {
        if !new_by_name.contains_key(name)
            && let Some(&id) = ids.get(*name)
        {
            removed.push(id);
        }
    }

    removed.sort_unstable();
    modified.sort_by_key(|(id, _)| *id);

    PromoterRegistryDiff {
        added,
        removed,
        modified,
    }
}

#[cfg(test)]
mod tests {
    use super::diff;
    use crate::config::Config;
    use compact_str::CompactString;
    use slotmap::KeyData;
    use specter_core::{PromoterId, SubId};
    use std::collections::BTreeMap;

    const ROOT: &str = "/";

    fn cfg(blocks: &[&str]) -> Config {
        let mut s = String::new();
        for b in blocks {
            s.push_str(b);
            s.push('\n');
        }
        Config::from_str(&s).expect("config valid")
    }

    fn block(name: &str, command: &str) -> String {
        format!("[[watch]]\nname = \"{name}\"\npath = \"{ROOT}\"\ncommand = [\"{command}\"]")
    }

    /// Build a watch block with an explicit `settle`. `max_settle` is
    /// left to the default (1h), which is comfortably above the floor
    /// for any reasonable `settle`.
    fn block_full(name: &str, command: &str, settle: &str) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{ROOT}\"\n\
             command = [\"{command}\"]\nsettle = \"{settle}\"\n",
        )
    }

    fn dyn_block(name: &str, pattern: &str, command: &str) -> String {
        format!("[[watch]]\nname = \"{name}\"\npath = \"{pattern}\"\ncommand = [\"{command}\"]")
    }

    fn dyn_block_full(name: &str, pattern: &str, command: &str, settle: &str) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{pattern}\"\n\
             command = [\"{command}\"]\nsettle = \"{settle}\"\n",
        )
    }

    fn sid(n: u64) -> SubId {
        SubId::from(KeyData::from_ffi(n))
    }

    fn pid(n: u64) -> PromoterId {
        PromoterId::from(KeyData::from_ffi(n))
    }

    fn sub_ids_of(pairs: &[(&str, SubId)]) -> BTreeMap<CompactString, SubId> {
        pairs
            .iter()
            .map(|(n, id)| (CompactString::new(n), *id))
            .collect()
    }

    fn promoter_ids_of(pairs: &[(&str, PromoterId)]) -> BTreeMap<CompactString, PromoterId> {
        pairs
            .iter()
            .map(|(n, id)| (CompactString::new(n), *id))
            .collect()
    }

    /// Compose a `WatchRegistryDiff` against the trivial maps. Convenience
    /// for the legacy test bodies that only exercise the static side.
    fn diff_subs_only(
        old: &Config,
        new: &Config,
        ids: &BTreeMap<CompactString, SubId>,
    ) -> super::WatchRegistryDiff {
        diff(old, new, ids, &BTreeMap::new())
    }

    // ---- Static (Sub) side — preserves all pre-Phase-10 coverage. ----

    #[test]
    fn empty_vs_empty_is_no_diff() {
        let a = cfg(&[]);
        let b = cfg(&[]);
        let d = diff_subs_only(&a, &b, &BTreeMap::new());
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified.is_empty());
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.removed.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    #[test]
    fn add_only_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [block("a", "echo"), block("b", "echo")];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff_subs_only(&old, &new, &BTreeMap::new());
        assert_eq!(d.subs.added.len(), 2);
        assert_eq!(d.subs.added[0].name, "a");
        assert_eq!(d.subs.added[1].name, "b");
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified.is_empty());
    }

    #[test]
    fn remove_only_populates_removed_with_matching_ids() {
        let old_blocks = [block("a", "echo")];
        let refs: Vec<&str> = old_blocks.iter().map(String::as_str).collect();
        let old = cfg(&refs);
        let new = cfg(&[]);

        let ids = sub_ids_of(&[("a", sid(1))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert!(d.subs.added.is_empty());
        assert_eq!(d.subs.removed, vec![sid(1)]);
        assert!(d.subs.modified.is_empty());
    }

    #[test]
    fn identical_configs_yield_empty_diff() {
        let blocks = [block("a", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff_subs_only(&a, &b, &sub_ids_of(&[("a", sid(1))]));
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified.is_empty());
    }

    #[test]
    fn different_command_at_same_name_yields_modified() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = sub_ids_of(&[("a", sid(1))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert_eq!(d.subs.modified.len(), 1);
        assert_eq!(d.subs.modified[0].0, sid(1));
        assert_eq!(d.subs.modified[0].1.name, "a");
    }

    #[test]
    fn add_remove_modify_mix() {
        let old_blocks = [block("a", "echo"), block("b", "echo"), block("c", "echo")];
        let new_blocks = [block("a", "fmt"), block("c", "echo"), block("d", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = sub_ids_of(&[("a", sid(1)), ("b", sid(2)), ("c", sid(3))]);
        let d = diff_subs_only(&old, &new, &ids);

        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].name, "d");

        assert_eq!(d.subs.removed, vec![sid(2)]);

        assert_eq!(d.subs.modified.len(), 1);
        assert_eq!(d.subs.modified[0].0, sid(1));
        assert_eq!(d.subs.modified[0].1.name, "a");
    }

    #[test]
    fn reorder_only_yields_empty_diff() {
        let order_a = [block("a", "echo"), block("b", "echo")];
        let order_b = [block("b", "echo"), block("a", "echo")];
        let old = cfg(&order_a.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&order_b.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = sub_ids_of(&[("a", sid(1)), ("b", sid(2))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified.is_empty());
    }

    #[test]
    fn rename_yields_added_plus_removed() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("z", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = sub_ids_of(&[("a", sid(1))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].name, "z");
        assert_eq!(d.subs.removed, vec![sid(1)]);
        assert!(d.subs.modified.is_empty());
    }

    #[test]
    fn settle_change_marks_modified() {
        let old_blocks = [block_full("a", "echo", "200ms")];
        let new_blocks = [block_full("a", "echo", "500ms")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = sub_ids_of(&[("a", sid(1))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert_eq!(d.subs.modified.len(), 1);
    }

    #[test]
    fn missing_id_for_removed_silently_skips() {
        let old_blocks = [block("a", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let d = diff_subs_only(&old, &new, &BTreeMap::new());
        assert!(
            d.subs.removed.is_empty(),
            "missing id_map silences `removed`"
        );
    }

    #[test]
    fn missing_id_for_modified_silently_skips() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff_subs_only(&old, &new, &BTreeMap::new());
        assert!(d.subs.modified.is_empty());
    }

    #[test]
    fn removed_sorted_by_subid() {
        let old_blocks = [block("a", "echo"), block("b", "echo"), block("c", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let ids = sub_ids_of(&[("a", sid(3)), ("b", sid(1)), ("c", sid(2))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert_eq!(d.subs.removed, vec![sid(1), sid(2), sid(3)]);
    }

    #[test]
    fn modified_sorted_by_subid() {
        let old_blocks = [block("a", "echo"), block("b", "echo")];
        let new_blocks = [block("a", "fmt"), block("b", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let ids = sub_ids_of(&[("a", sid(2)), ("b", sid(1))]);
        let d = diff_subs_only(&old, &new, &ids);
        let order: Vec<SubId> = d.subs.modified.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![sid(1), sid(2)]);
    }

    #[test]
    fn events_change_marks_modified_in_diff() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             command = [\"echo\"]\nevents = [\"content\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = sub_ids_of(&[("a", sid(1))]);
        let d = diff_subs_only(&old, &new, &ids);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert_eq!(d.subs.modified.len(), 1);
        assert_eq!(d.subs.modified[0].0, sid(1));
        assert_eq!(d.subs.modified[0].1.events, specter_core::ClassSet::CONTENT);
    }

    #[test]
    fn explicit_events_equal_to_default_yields_no_diff() {
        // A user adding `events = ["structure", "content"]` cosmetically
        // (matching the implicit subtree-root default) must not churn
        // the Profile.
        let old_blocks = [block("a", "echo")];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             command = [\"echo\"]\nevents = [\"structure\", \"content\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff_subs_only(&old, &new, &sub_ids_of(&[("a", sid(1))]));
        assert!(d.subs.modified.is_empty(), "got {:?}", d.subs.modified);
    }

    #[test]
    fn events_class_order_does_not_affect_diff() {
        // Class set is bitmask-equality, not list-order — the parser
        // collapses both orderings to the same ClassSet.
        let old_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             command = [\"echo\"]\nevents = [\"structure\", \"content\"]"
        )];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             command = [\"echo\"]\nevents = [\"content\", \"structure\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff_subs_only(&old, &new, &sub_ids_of(&[("a", sid(1))]));
        assert!(d.subs.modified.is_empty(), "got {:?}", d.subs.modified);
    }

    // ---- Promoter (dynamic) side ----

    /// Adding a fresh dynamic [[watch]] populates `promoters.added` in
    /// source order. The `promoter_ids` map is empty (no Promoter is
    /// attached on the old side), so no removed / modified entries.
    #[test]
    fn promoter_added_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [
            dyn_block("logs", "/var/log/*", "echo"),
            dyn_block("sites", "/srv/*/site", "fmt"),
        ];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff(&old, &new, &BTreeMap::new(), &BTreeMap::new());
        assert!(d.subs.added.is_empty());
        assert_eq!(d.promoters.added.len(), 2);
        assert_eq!(d.promoters.added[0].name, "logs");
        assert_eq!(d.promoters.added[1].name, "sites");
        assert!(d.promoters.removed.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    /// Removing a dynamic [[watch]] populates `promoters.removed` with
    /// the looked-up `PromoterId`.
    #[test]
    fn promoter_removed_populates_removed_with_matching_ids() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);

        let promoter_ids = promoter_ids_of(&[("logs", pid(1))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);
        assert_eq!(d.promoters.removed, vec![pid(1)]);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    /// Modifying any field on a dynamic [[watch]] surfaces the entry on
    /// `promoters.modified`. Wholesale replace at the engine layer.
    #[test]
    fn promoter_command_change_yields_modified() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let new_blocks = [dyn_block("logs", "/var/log/*", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let promoter_ids = promoter_ids_of(&[("logs", pid(1))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.removed.is_empty());
        assert_eq!(d.promoters.modified.len(), 1);
        assert_eq!(d.promoters.modified[0].0, pid(1));
        assert_eq!(d.promoters.modified[0].1.name, "logs");
    }

    /// Pattern source change is a structural modification (different
    /// `pattern_spec.source`); diff surfaces it as `modified`. The
    /// engine wholesale-replaces, which drains and re-mints dynamic
    /// Subs against the new pattern.
    #[test]
    fn promoter_pattern_change_yields_modified() {
        let old_blocks = [dyn_block("logs", "/var/log/*.log", "echo")];
        let new_blocks = [dyn_block("logs", "/var/log/*.json", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let promoter_ids = promoter_ids_of(&[("logs", pid(1))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);
        assert_eq!(d.promoters.modified.len(), 1);
        assert_eq!(
            d.promoters.modified[0].1.pattern_spec.source(),
            "/var/log/*.json",
        );
    }

    /// Settle change on a dynamic watch surfaces as modified.
    #[test]
    fn promoter_settle_change_marks_modified() {
        let old_blocks = [dyn_block_full("logs", "/var/log/*", "echo", "200ms")];
        let new_blocks = [dyn_block_full("logs", "/var/log/*", "echo", "500ms")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let promoter_ids = promoter_ids_of(&[("logs", pid(1))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);
        assert_eq!(d.promoters.modified.len(), 1);
    }

    /// Identical promoter configs yield no diff.
    #[test]
    fn promoter_identical_configs_yield_empty_diff() {
        let blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff(
            &a,
            &b,
            &BTreeMap::new(),
            &promoter_ids_of(&[("logs", pid(1))]),
        );
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.removed.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    /// `promoters.removed` sorts by id (mirrors `subs.removed`).
    #[test]
    fn promoter_removed_sorted_by_promoter_id() {
        let old_blocks = [
            dyn_block("a", "/a/*", "echo"),
            dyn_block("b", "/b/*", "echo"),
            dyn_block("c", "/c/*", "echo"),
        ];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let promoter_ids = promoter_ids_of(&[("a", pid(3)), ("b", pid(1)), ("c", pid(2))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);
        assert_eq!(d.promoters.removed, vec![pid(1), pid(2), pid(3)]);
    }

    /// `promoters.modified` sorts by id (mirrors `subs.modified`).
    #[test]
    fn promoter_modified_sorted_by_promoter_id() {
        let old_blocks = [
            dyn_block("a", "/a/*", "echo"),
            dyn_block("b", "/b/*", "echo"),
        ];
        let new_blocks = [dyn_block("a", "/a/*", "fmt"), dyn_block("b", "/b/*", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let promoter_ids = promoter_ids_of(&[("a", pid(2)), ("b", pid(1))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);
        let order: Vec<PromoterId> = d.promoters.modified.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![pid(1), pid(2)]);
    }

    /// Missing id for removed promoter silently skips (mirrors Sub side).
    #[test]
    fn promoter_missing_id_for_removed_silently_skips() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let d = diff(&old, &new, &BTreeMap::new(), &BTreeMap::new());
        assert!(d.promoters.removed.is_empty());
    }

    /// Missing id for modified promoter silently skips.
    #[test]
    fn promoter_missing_id_for_modified_silently_skips() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let new_blocks = [dyn_block("logs", "/var/log/*", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new, &BTreeMap::new(), &BTreeMap::new());
        assert!(d.promoters.modified.is_empty());
    }

    // ---- Cross-kind: static ↔ dynamic migration via path edit ----

    /// Static → dynamic via path edit: name `foo` was static; new
    /// config has `foo` as dynamic. Same name, but the path crossed the
    /// `is_dynamic` boundary so the entry moves between
    /// `Config.watches` and `Config.promoters`. Diff produces
    /// `subs.removed + promoters.added`.
    #[test]
    fn static_to_dynamic_migration_yields_subs_removed_plus_promoters_added() {
        let old_blocks = [block("foo", "echo")]; // path = "/" (static)
        let new_blocks = [dyn_block("foo", "/foo/*", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let sub_ids = sub_ids_of(&[("foo", sid(1))]);
        let d = diff(&old, &new, &sub_ids, &BTreeMap::new());

        assert_eq!(d.subs.removed, vec![sid(1)]);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.modified.is_empty());
        assert_eq!(d.promoters.added.len(), 1);
        assert_eq!(d.promoters.added[0].name, "foo");
        assert!(d.promoters.removed.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    /// Reverse direction: dynamic → static via path edit. The dynamic
    /// `foo` from old becomes a static `foo` in new. Diff produces
    /// `promoters.removed + subs.added`.
    #[test]
    fn dynamic_to_static_migration_yields_promoters_removed_plus_subs_added() {
        let old_blocks = [dyn_block("foo", "/foo/*", "echo")];
        let new_blocks = [block("foo", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let promoter_ids = promoter_ids_of(&[("foo", pid(1))]);
        let d = diff(&old, &new, &BTreeMap::new(), &promoter_ids);

        assert_eq!(d.promoters.removed, vec![pid(1)]);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.modified.is_empty());
        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].name, "foo");
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified.is_empty());
    }

    /// Mixed reload: one Sub modify, one Promoter add, one of each
    /// removed. Each half stands on its own; the diff composes them
    /// without interaction.
    #[test]
    fn mixed_sub_and_promoter_changes_compose_independently() {
        let old_blocks = [
            block("static_keep", "echo"),
            block("static_drop", "echo"),
            dyn_block("dyn_keep", "/keep/*", "echo"),
            dyn_block("dyn_drop", "/drop/*", "echo"),
        ];
        let new_blocks = [
            block("static_keep", "fmt"),              // modified
            dyn_block("dyn_keep", "/keep/*", "echo"), // unchanged
            dyn_block("dyn_new", "/new/*", "echo"),   // added
        ];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let sub_ids = sub_ids_of(&[("static_keep", sid(1)), ("static_drop", sid(2))]);
        let promoter_ids = promoter_ids_of(&[("dyn_keep", pid(1)), ("dyn_drop", pid(2))]);
        let d = diff(&old, &new, &sub_ids, &promoter_ids);

        assert_eq!(d.subs.removed, vec![sid(2)]);
        assert!(d.subs.added.is_empty());
        assert_eq!(d.subs.modified.len(), 1);
        assert_eq!(d.subs.modified[0].0, sid(1));

        assert_eq!(d.promoters.removed, vec![pid(2)]);
        assert_eq!(d.promoters.added.len(), 1);
        assert_eq!(d.promoters.added[0].name, "dyn_new");
        assert!(d.promoters.modified.is_empty());
    }
}
