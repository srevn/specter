use crate::config::{Config, SubSpec};
use compact_str::CompactString;
use specter_core::{SubAttachRequest, SubId, SubRegistryDiff};
use std::collections::BTreeMap;

/// Compute the hot-reload diff between two validated [`Config`]s.
///
/// `ids` is the bin's `name → SubId` map for currently-attached Subs (whose
/// names match `old`). The resulting [`SubRegistryDiff::removed`] /
/// [`SubRegistryDiff::modified`] carry those `SubId`s; `added` carries
/// fresh [`SubAttachRequest`]s whose ids the engine mints at attach time.
///
/// Determinism: `removed` and `modified` are sorted by `SubId`; `added`
/// preserves `new`-source order.
#[must_use]
pub fn diff(old: &Config, new: &Config, ids: &BTreeMap<CompactString, SubId>) -> SubRegistryDiff {
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

#[cfg(test)]
mod tests {
    use super::diff;
    use crate::config::Config;
    use compact_str::CompactString;
    use slotmap::KeyData;
    use specter_core::SubId;
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

    fn block_full(name: &str, command: &str, settle_ms: u64) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{ROOT}\"\n\
             command = [\"{command}\"]\nsettle_ms = {settle_ms}\n\
             max_settle_ms = {}",
            settle_ms.saturating_mul(60).min(3_600_000),
        )
    }

    fn id(n: u64) -> SubId {
        SubId::from(KeyData::from_ffi(n))
    }

    fn ids_of(pairs: &[(&str, SubId)]) -> BTreeMap<CompactString, SubId> {
        pairs
            .iter()
            .map(|(n, id)| (CompactString::new(n), *id))
            .collect()
    }

    #[test]
    fn empty_vs_empty_is_no_diff() {
        let a = cfg(&[]);
        let b = cfg(&[]);
        let d = diff(&a, &b, &BTreeMap::new());
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified.is_empty());
    }

    #[test]
    fn add_only_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [block("a", "echo"), block("b", "echo")];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff(&old, &new, &BTreeMap::new());
        assert_eq!(d.added.len(), 2);
        assert_eq!(d.added[0].name, "a");
        assert_eq!(d.added[1].name, "b");
        assert!(d.removed.is_empty());
        assert!(d.modified.is_empty());
    }

    #[test]
    fn remove_only_populates_removed_with_matching_ids() {
        let old_blocks = [block("a", "echo")];
        let refs: Vec<&str> = old_blocks.iter().map(String::as_str).collect();
        let old = cfg(&refs);
        let new = cfg(&[]);

        let ids = ids_of(&[("a", id(1))]);
        let d = diff(&old, &new, &ids);
        assert!(d.added.is_empty());
        assert_eq!(d.removed, vec![id(1)]);
        assert!(d.modified.is_empty());
    }

    #[test]
    fn identical_configs_yield_empty_diff() {
        let blocks = [block("a", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff(&a, &b, &ids_of(&[("a", id(1))]));
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified.is_empty());
    }

    #[test]
    fn different_command_at_same_name_yields_modified() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = ids_of(&[("a", id(1))]);
        let d = diff(&old, &new, &ids);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert_eq!(d.modified.len(), 1);
        assert_eq!(d.modified[0].0, id(1));
        assert_eq!(d.modified[0].1.name, "a");
    }

    #[test]
    fn add_remove_modify_mix() {
        let old_blocks = [block("a", "echo"), block("b", "echo"), block("c", "echo")];
        let new_blocks = [block("a", "fmt"), block("c", "echo"), block("d", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = ids_of(&[("a", id(1)), ("b", id(2)), ("c", id(3))]);
        let d = diff(&old, &new, &ids);

        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].name, "d");

        assert_eq!(d.removed, vec![id(2)]);

        assert_eq!(d.modified.len(), 1);
        assert_eq!(d.modified[0].0, id(1));
        assert_eq!(d.modified[0].1.name, "a");
    }

    #[test]
    fn reorder_only_yields_empty_diff() {
        let order_a = [block("a", "echo"), block("b", "echo")];
        let order_b = [block("b", "echo"), block("a", "echo")];
        let old = cfg(&order_a.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&order_b.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = ids_of(&[("a", id(1)), ("b", id(2))]);
        let d = diff(&old, &new, &ids);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified.is_empty());
    }

    #[test]
    fn rename_yields_added_plus_removed() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("z", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = ids_of(&[("a", id(1))]);
        let d = diff(&old, &new, &ids);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].name, "z");
        assert_eq!(d.removed, vec![id(1)]);
        assert!(d.modified.is_empty());
    }

    #[test]
    fn settle_change_marks_modified() {
        let old_blocks = [block_full("a", "echo", 200)];
        let new_blocks = [block_full("a", "echo", 500)];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let ids = ids_of(&[("a", id(1))]);
        let d = diff(&old, &new, &ids);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert_eq!(d.modified.len(), 1);
    }

    #[test]
    fn missing_id_for_removed_silently_skips() {
        let old_blocks = [block("a", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let d = diff(&old, &new, &BTreeMap::new());
        assert!(d.removed.is_empty(), "missing id_map silences `removed`");
    }

    #[test]
    fn missing_id_for_modified_silently_skips() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new, &BTreeMap::new());
        assert!(d.modified.is_empty());
    }

    #[test]
    fn removed_sorted_by_subid() {
        let old_blocks = [block("a", "echo"), block("b", "echo"), block("c", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let ids = ids_of(&[("a", id(3)), ("b", id(1)), ("c", id(2))]);
        let d = diff(&old, &new, &ids);
        assert_eq!(d.removed, vec![id(1), id(2), id(3)]);
    }

    #[test]
    fn modified_sorted_by_subid() {
        let old_blocks = [block("a", "echo"), block("b", "echo")];
        let new_blocks = [block("a", "fmt"), block("b", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let ids = ids_of(&[("a", id(2)), ("b", id(1))]);
        let d = diff(&old, &new, &ids);
        let order: Vec<SubId> = d.modified.iter().map(|(id, _)| *id).collect();
        assert_eq!(order, vec![id(1), id(2)]);
    }
}
