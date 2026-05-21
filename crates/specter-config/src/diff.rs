use crate::config::{Config, PromoterSpec, SubSpec};
use compact_str::CompactString;
use specter_core::{
    PromoterAttachRequest, PromoterRegistryDiff, SubAttachRequest, SubRegistryDiff,
    WatchRegistryDiff,
};
use std::collections::BTreeMap;

/// Compute the full hot-reload diff between two validated [`Config`]s.
///
/// A pure function of `(old, new)` — no id maps. The returned
/// [`WatchRegistryDiff`] is **name-keyed**: `removed` carries operator
/// names; `added` / `modified` carry pre-id requests whose name lives
/// inside the request. The engine resolves name → id at apply time
/// through its own authoritative `by_name` index — identity
/// resolution is a registry-owner operation, not the loader's.
///
/// **Static ↔ dynamic migration via path edit.** A `[[watch]]` whose
/// path edits across the `is_dynamic` boundary (e.g., `/foo` → `/foo/*`)
/// moves between [`Config::watches`] and [`Config::promoters`]
/// between reloads. The diff produces `subs.removed + promoters.added`
/// (or the reverse): a wholesale teardown then attach. The path
/// semantics meaningfully changed; merging across the boundary would
/// hide that.
///
/// Determinism: each side's `removed` is name-ordered structurally —
/// it is built from a `BTreeMap`'s `keys()` iterator (ascending by
/// API contract), so the order is established at construction and a
/// debug-mode `is_sorted` check pins the invariant without paying for
/// a runtime sort. `modified` is name-sorted load-bearing — built from
/// source-order `active_*`, sorted at the end via `sort_unstable_by`
/// (names are unique by validation, so stability is unobservable).
/// `added` preserves new-source order.
#[must_use]
pub fn diff(old: &Config, new: &Config) -> WatchRegistryDiff {
    WatchRegistryDiff {
        subs: diff_subs(old, new),
        promoters: diff_promoters(old, new),
    }
}

/// Static-Sub half of the watch-registry diff. Extracted so the two
/// halves are independently testable; the public entry point composes
/// both.
///
/// Filters both sides through [`Config::active_watches`] before the
/// name-keyed comparison: a disabled entry on either side is
/// structurally equivalent to "absent." Flipping `enabled = true →
/// false` therefore surfaces as `subs.removed`; the reverse as
/// `subs.added`. Edits to fields on a disabled entry are invisible
/// to the diff (the entry isn't in either filtered set) — they
/// apply on the next `false → true` transition via the fresh
/// attach.
///
/// Modified entries are partitioned by
/// [`SubSpec::requires_new_profile`]:
///
/// - **`modified_identity`** — path / scan / max_settle / events
///   differ; the Sub must move to a different Profile partition. The
///   engine validates the new anchor's parse, then runs
///   `detach_old → attach_new`.
/// - **`modified_params`** — only per-Sub fields differ (`program`,
///   `scope`, `settle`, `log_output`). The engine rebinds the live
///   Sub in place; no Profile churn, no kernel-watch flap, no
///   baseline loss.
///
/// The partition is exhaustive and disjoint: every modified entry
/// lands in exactly one bucket.
fn diff_subs(old: &Config, new: &Config) -> SubRegistryDiff {
    let old_by_name: BTreeMap<&CompactString, &SubSpec> =
        old.active_watches().map(|s| (&s.name, s)).collect();
    let new_by_name: BTreeMap<&CompactString, &SubSpec> =
        new.active_watches().map(|s| (&s.name, s)).collect();

    let mut added: Vec<SubAttachRequest> = Vec::new();
    let mut modified_identity: Vec<SubAttachRequest> = Vec::new();
    let mut modified_params: Vec<SubAttachRequest> = Vec::new();
    let mut removed: Vec<CompactString> = Vec::new();

    for spec in new.active_watches() {
        match old_by_name.get(&spec.name) {
            None => added.push(spec.to_attach_request()),
            Some(old_spec) if **old_spec == *spec => {} // unchanged
            Some(old_spec) if old_spec.requires_new_profile(spec) => {
                modified_identity.push(spec.to_attach_request());
            }
            Some(_) => modified_params.push(spec.to_attach_request()),
        }
    }

    for name in old_by_name.keys() {
        if !new_by_name.contains_key(name) {
            removed.push((*name).clone());
        }
    }

    // `removed` is collected from a `BTreeMap`'s keys, so it is already
    // name-ordered by construction — the debug assertion pins the
    // invariant without paying for a runtime sort. Both `modified_*`
    // buckets are built from source-order `active_watches()` and are
    // the load-bearing sort: replay stability keys on config content,
    // never slotmap mint order. Names are unique by validation, so
    // `sort_unstable_by` is observably indistinguishable from a stable
    // sort and strictly cheaper.
    debug_assert!(
        removed.is_sorted(),
        "removed must inherit BTreeMap key order",
    );
    modified_identity.sort_unstable_by(|a, b| a.params.name.cmp(&b.params.name));
    modified_params.sort_unstable_by(|a, b| a.params.name.cmp(&b.params.name));

    SubRegistryDiff {
        added,
        removed,
        modified_identity,
        modified_params,
    }
}

/// Promoter half of the watch-registry diff. Mirrors [`diff_subs`]
/// against [`Config::active_promoters`]. Same `enabled`-as-absent
/// semantics: a disabled Promoter on either side is filtered before
/// comparison; flipping the flag surfaces as `promoters.added` /
/// `promoters.removed`.
fn diff_promoters(old: &Config, new: &Config) -> PromoterRegistryDiff {
    let old_by_name: BTreeMap<&CompactString, &PromoterSpec> =
        old.active_promoters().map(|p| (&p.name, p)).collect();
    let new_by_name: BTreeMap<&CompactString, &PromoterSpec> =
        new.active_promoters().map(|p| (&p.name, p)).collect();

    let mut added: Vec<PromoterAttachRequest> = Vec::new();
    let mut modified: Vec<PromoterAttachRequest> = Vec::new();
    let mut removed: Vec<CompactString> = Vec::new();

    for spec in new.active_promoters() {
        match old_by_name.get(&spec.name) {
            None => added.push(spec.to_attach_request()),
            Some(old_spec) if **old_spec != *spec => modified.push(spec.to_attach_request()),
            Some(_) => {}
        }
    }

    for name in old_by_name.keys() {
        if !new_by_name.contains_key(name) {
            removed.push((*name).clone());
        }
    }

    // Sort rationale mirrors `diff_subs`: `removed` is BTreeMap-keyed
    // and so already name-ordered (debug-only assertion pins the
    // invariant); `modified` is load-bearing on `active_promoters()`
    // source order and uses `sort_unstable_by` since names are unique.
    debug_assert!(
        removed.is_sorted(),
        "removed must inherit BTreeMap key order",
    );
    modified.sort_unstable_by(|a, b| a.name.cmp(&b.name));

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

    const ROOT: &str = "/";

    fn cfg(blocks: &[&str]) -> Config {
        let mut s = String::new();
        for b in blocks {
            s.push_str(b);
            s.push('\n');
        }
        Config::from_str(&s).expect("config valid")
    }

    fn block(name: &str, exec: &str) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{ROOT}\"\nactions = [{{ exec = [\"{exec}\"] }}]"
        )
    }

    /// Build a watch block with an explicit `settle`. `max_settle` is
    /// left to the default (1h), which is comfortably above the floor
    /// for any reasonable `settle`.
    fn block_full(name: &str, exec: &str, settle: &str) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"{exec}\"] }}]\nsettle = \"{settle}\"\n",
        )
    }

    fn dyn_block(name: &str, pattern: &str, exec: &str) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{pattern}\"\nactions = [{{ exec = [\"{exec}\"] }}]"
        )
    }

    fn dyn_block_full(name: &str, pattern: &str, exec: &str, settle: &str) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{pattern}\"\n\
             actions = [{{ exec = [\"{exec}\"] }}]\nsettle = \"{settle}\"\n",
        )
    }

    // ---- Static (Sub) side ----

    #[test]
    fn empty_vs_empty_is_no_diff() {
        let a = cfg(&[]);
        let b = cfg(&[]);
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
    fn add_only_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [block("a", "echo"), block("b", "echo")];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff(&old, &new);
        assert_eq!(d.subs.added.len(), 2);
        assert_eq!(d.subs.added[0].params.name, "a");
        assert_eq!(d.subs.added[1].params.name, "b");
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    #[test]
    fn remove_only_populates_removed_with_watch_name() {
        let old_blocks = [block("a", "echo")];
        let refs: Vec<&str> = old_blocks.iter().map(String::as_str).collect();
        let old = cfg(&refs);
        let new = cfg(&[]);

        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert_eq!(d.subs.removed, vec![CompactString::from("a")]);
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    #[test]
    fn identical_configs_yield_empty_diff() {
        let blocks = [block("a", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff(&a, &b);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    /// Command change ⇒ per-Sub field only ⇒ `modified_params`. The
    /// identity bucket stays empty: the partition is exhaustive and
    /// disjoint per watch name.
    #[test]
    fn different_command_lands_in_modified_params() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert_eq!(d.subs.modified_params.len(), 1);
        assert_eq!(d.subs.modified_params[0].params.name, "a");
    }

    #[test]
    fn add_remove_modify_mix() {
        let old_blocks = [block("a", "echo"), block("b", "echo"), block("c", "echo")];
        let new_blocks = [block("a", "fmt"), block("c", "echo"), block("d", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);

        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].params.name, "d");

        assert_eq!(d.subs.removed, vec![CompactString::from("b")]);

        // `block("a", "echo") → block("a", "fmt")` is a program-only
        // change ⇒ `modified_params`.
        assert!(d.subs.modified_identity.is_empty());
        assert_eq!(d.subs.modified_params.len(), 1);
        assert_eq!(d.subs.modified_params[0].params.name, "a");
    }

    #[test]
    fn reorder_only_yields_empty_diff() {
        let order_a = [block("a", "echo"), block("b", "echo")];
        let order_b = [block("b", "echo"), block("a", "echo")];
        let old = cfg(&order_a.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&order_b.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    #[test]
    fn rename_yields_added_plus_removed() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("z", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].params.name, "z");
        assert_eq!(d.subs.removed, vec![CompactString::from("a")]);
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    /// `settle` is a per-Sub field, not a Profile-identity field ⇒
    /// `modified_params`. Pinning the bucket guards against a future
    /// `settle`-into-identity drift that would silently re-route this
    /// case through `modified_identity` (detach+attach with baseline
    /// loss) instead of in-place rebind.
    #[test]
    fn settle_change_lands_in_modified_params() {
        let old_blocks = [block_full("a", "echo", "200ms")];
        let new_blocks = [block_full("a", "echo", "500ms")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert_eq!(d.subs.modified_params.len(), 1);
    }

    #[test]
    fn removed_sorted_by_name() {
        let old_blocks = [block("c", "echo"), block("a", "echo"), block("b", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let d = diff(&old, &new);
        assert_eq!(
            d.subs.removed,
            vec![
                CompactString::from("a"),
                CompactString::from("b"),
                CompactString::from("c"),
            ]
        );
    }

    #[test]
    fn modified_params_sorted_by_name() {
        let old_blocks = [block("b", "echo"), block("a", "echo")];
        let new_blocks = [block("b", "fmt"), block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new);
        let order: Vec<&str> = d
            .subs
            .modified_params
            .iter()
            .map(|r| r.params.name.as_str())
            .collect();
        assert_eq!(order, vec!["a", "b"]);
    }

    /// `events` folds into `ProfileIdentity::config_hash` ⇒
    /// `modified_identity`. Same role guard as
    /// [`settle_change_lands_in_modified_params`]: pinning the bucket
    /// makes a future identity-into-params drift visible at the diff
    /// layer.
    #[test]
    fn events_change_lands_in_modified_identity() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"echo\"] }}]\nevents = [\"content\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_params.is_empty());
        assert_eq!(d.subs.modified_identity.len(), 1);
        assert_eq!(d.subs.modified_identity[0].params.name, "a");
        assert_eq!(
            d.subs.modified_identity[0].identity.events,
            specter_core::ClassSet::CONTENT
        );
    }

    #[test]
    fn explicit_events_equal_to_default_yields_no_diff() {
        // A user adding `events = ["structure", "content"]` cosmetically
        // (matching the implicit subtree-root default) must not churn
        // the Profile.
        let old_blocks = [block("a", "echo")];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"echo\"] }}]\nevents = [\"structure\", \"content\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new);
        assert!(
            d.subs.modified_identity.is_empty(),
            "identity bucket: {:?}",
            d.subs.modified_identity,
        );
        assert!(
            d.subs.modified_params.is_empty(),
            "params bucket: {:?}",
            d.subs.modified_params,
        );
    }

    #[test]
    fn events_class_order_does_not_affect_diff() {
        // Class set is bitmask-equality, not list-order — the parser
        // collapses both orderings to the same ClassSet.
        let old_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"echo\"] }}]\nevents = [\"structure\", \"content\"]"
        )];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"echo\"] }}]\nevents = [\"content\", \"structure\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new);
        assert!(
            d.subs.modified_identity.is_empty(),
            "identity bucket: {:?}",
            d.subs.modified_identity,
        );
        assert!(
            d.subs.modified_params.is_empty(),
            "params bucket: {:?}",
            d.subs.modified_params,
        );
    }

    /// `scan` (recursive flag) folds into `ProfileIdentity::config_hash`
    /// ⇒ `modified_identity`. Same name, same path, different scan ⇒
    /// the Sub must move to a different Profile. Distinct from
    /// `events_change_lands_in_modified_identity` because `scan` is a
    /// nested struct (`ScanConfig`) where `events` is a `ClassSet`
    /// bitmask — exercising a struct-equality path the bitmask test
    /// doesn't.
    #[test]
    fn scan_change_lands_in_modified_identity() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"echo\"] }}]\nrecursive = false\n",
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.subs.modified_params.is_empty());
        assert_eq!(d.subs.modified_identity.len(), 1);
        assert_eq!(d.subs.modified_identity[0].params.name, "a");
    }

    // ---- Promoter (dynamic) side ----

    /// Adding a fresh dynamic [[watch]] populates `promoters.added` in
    /// source order. Nothing on the old side, so no removed / modified
    /// entries.
    #[test]
    fn promoter_added_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [
            dyn_block("logs", "/var/log/*", "echo"),
            dyn_block("sites", "/srv/*/site", "fmt"),
        ];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert_eq!(d.promoters.added.len(), 2);
        assert_eq!(d.promoters.added[0].name, "logs");
        assert_eq!(d.promoters.added[1].name, "sites");
        assert!(d.promoters.removed.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    /// Removing a dynamic [[watch]] populates `promoters.removed` with
    /// the operator Promoter name.
    #[test]
    fn promoter_removed_populates_removed_with_promoter_name() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);

        let d = diff(&old, &new);
        assert_eq!(d.promoters.removed, vec![CompactString::from("logs")]);
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

        let d = diff(&old, &new);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.removed.is_empty());
        assert_eq!(d.promoters.modified.len(), 1);
        assert_eq!(d.promoters.modified[0].name, "logs");
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

        let d = diff(&old, &new);
        assert_eq!(d.promoters.modified.len(), 1);
        assert_eq!(
            d.promoters.modified[0].pattern_spec.source(),
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

        let d = diff(&old, &new);
        assert_eq!(d.promoters.modified.len(), 1);
    }

    /// Identical promoter configs yield no diff.
    #[test]
    fn promoter_identical_configs_yield_empty_diff() {
        let blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff(&a, &b);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.removed.is_empty());
        assert!(d.promoters.modified.is_empty());
    }

    /// `promoters.removed` sorts by name (mirrors `subs.removed`).
    #[test]
    fn promoter_removed_sorted_by_name() {
        let old_blocks = [
            dyn_block("c", "/c/*", "echo"),
            dyn_block("a", "/a/*", "echo"),
            dyn_block("b", "/b/*", "echo"),
        ];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let d = diff(&old, &new);
        assert_eq!(
            d.promoters.removed,
            vec![
                CompactString::from("a"),
                CompactString::from("b"),
                CompactString::from("c"),
            ]
        );
    }

    /// `promoters.modified` sorts by name (mirrors the Sub-side
    /// `modified_*` buckets, both individually sorted in `diff_subs`).
    #[test]
    fn promoter_modified_sorted_by_name() {
        let old_blocks = [
            dyn_block("b", "/b/*", "echo"),
            dyn_block("a", "/a/*", "echo"),
        ];
        let new_blocks = [dyn_block("b", "/b/*", "fmt"), dyn_block("a", "/a/*", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new);
        let order: Vec<&str> = d
            .promoters
            .modified
            .iter()
            .map(|r| r.name.as_str())
            .collect();
        assert_eq!(order, vec!["a", "b"]);
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

        let d = diff(&old, &new);

        assert_eq!(d.subs.removed, vec![CompactString::from("foo")]);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
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

        let d = diff(&old, &new);

        assert_eq!(d.promoters.removed, vec![CompactString::from("foo")]);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.modified.is_empty());
        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].params.name, "foo");
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
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

        let d = diff(&old, &new);

        assert_eq!(d.subs.removed, vec![CompactString::from("static_drop")]);
        assert!(d.subs.added.is_empty());
        // `static_keep`: echo → fmt is a program-only change ⇒ params bucket.
        assert!(d.subs.modified_identity.is_empty());
        assert_eq!(d.subs.modified_params.len(), 1);
        assert_eq!(d.subs.modified_params[0].params.name, "static_keep");

        assert_eq!(d.promoters.removed, vec![CompactString::from("dyn_drop")]);
        assert_eq!(d.promoters.added.len(), 1);
        assert_eq!(d.promoters.added[0].name, "dyn_new");
        assert!(d.promoters.modified.is_empty());
    }

    // ---- Enabled-toggle transitions ----

    fn block_with_enabled(name: &str, exec: &str, enabled: bool) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"{exec}\"] }}]\nenabled = {enabled}\n",
        )
    }

    fn dyn_block_with_enabled(name: &str, pattern: &str, exec: &str, enabled: bool) -> String {
        format!(
            "[[watch]]\nname = \"{name}\"\npath = \"{pattern}\"\n\
             actions = [{{ exec = [\"{exec}\"] }}]\nenabled = {enabled}\n",
        )
    }

    /// Flipping `enabled = true → false` filters the entry out of the
    /// new effective set, surfacing as `subs.removed` — the same shape
    /// the engine sees for an outright deletion. This is the
    /// load-bearing case of the feature.
    #[test]
    fn enabled_true_to_false_yields_subs_removed() {
        let old = cfg(&[block_with_enabled("a", "echo", true).as_str()]);
        let new = cfg(&[block_with_enabled("a", "echo", false).as_str()]);
        let d = diff(&old, &new);
        assert_eq!(d.subs.removed, vec![CompactString::from("a")]);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    /// Reverse: `false → true` re-introduces the entry, surfacing as
    /// `subs.added`.
    #[test]
    fn enabled_false_to_true_yields_subs_added() {
        let old = cfg(&[block_with_enabled("a", "echo", false).as_str()]);
        let new = cfg(&[block_with_enabled("a", "echo", true).as_str()]);
        let d = diff(&old, &new);
        assert_eq!(d.subs.added.len(), 1);
        assert_eq!(d.subs.added[0].params.name, "a");
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    /// Edits to other fields while the entry is disabled produce no
    /// diff: both sides are filtered out before the comparison, so
    /// the engine sees nothing. The new field values apply on the
    /// next `false → true` transition via the fresh attach.
    #[test]
    fn disabled_to_disabled_with_field_change_yields_empty_diff() {
        let old = cfg(&[block_with_enabled("a", "echo", false).as_str()]);
        let new = cfg(&[block_with_enabled("a", "fmt", false).as_str()]);
        let d = diff(&old, &new);
        assert!(d.subs.added.is_empty());
        assert!(d.subs.removed.is_empty());
        assert!(d.subs.modified_identity.is_empty());
        assert!(d.subs.modified_params.is_empty());
    }

    /// Promoter-side enabled flip mirrors the static side. One
    /// transition is enough — the code path is structurally identical
    /// (`active_promoters` filter ahead of the same name-keyed
    /// matching).
    #[test]
    fn promoter_enabled_true_to_false_yields_promoters_removed() {
        let old = cfg(&[dyn_block_with_enabled("logs", "/var/log/*", "echo", true).as_str()]);
        let new = cfg(&[dyn_block_with_enabled("logs", "/var/log/*", "echo", false).as_str()]);
        let d = diff(&old, &new);
        assert_eq!(d.promoters.removed, vec![CompactString::from("logs")]);
        assert!(d.promoters.added.is_empty());
        assert!(d.promoters.modified.is_empty());
    }
}
