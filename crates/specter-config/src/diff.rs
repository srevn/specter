use crate::config::{Config, SubSpec};
use compact_str::CompactString;
use specter_core::{SubAttachRequest, SubRegistryDiff};
use std::collections::BTreeMap;

/// Compute the hot-reload diff between two validated [`Config`]s.
///
/// A pure function of `(old, new)` — no id maps. The returned [`SubRegistryDiff`] is
/// **name-keyed**: `removed` carries operator names; `added` / `modified_*` carry pre-id requests
/// whose name lives inside the request. The engine resolves name → id at apply time through its own
/// authoritative `by_name` index — identity resolution is a registry-owner operation, not the
/// loader's. Dynamic `[[watch]]` blocks ride the same buckets: a discovery spec is a
/// template-bearing [`SubSpec`] in the one [`Config::watches`] list.
///
/// Filters both sides through [`Config::active_watches`] before the name-keyed comparison: a
/// disabled entry on either side is structurally equivalent to "absent." Flipping `enabled = true →
/// false` therefore surfaces as `removed`; the reverse as `added`. Edits to fields on a disabled
/// entry are invisible to the diff (the entry isn't in either filtered set) — they apply on the
/// next `false → true` transition via the fresh attach.
///
/// Modified entries are partitioned by [`SubSpec::requires_new_profile`]:
///
/// - **`modified_identity`** — path / scan / max_settle / events differ, **or either spec is
///   template-bearing**: any field change on a discovery spec is a wholesale replace (minted Subs
///   hold `Arc`s of the template Sub's program — an in-place rebind would strand them), and a path
///   edit across the `is_dynamic` boundary flips template presence, landing here too. The engine
///   validates the new anchor's parse, then runs `detach_old → attach_new`.
/// - **`modified_params`** — only per-Sub fields differ (`program`, `scope`, `settle`,
///   `log_output`) on a template-free pair. The engine rebinds the live Sub in place; no Profile
///   churn, no kernel-watch flap, no baseline loss.
///
/// The partition is exhaustive and disjoint: every modified entry lands in exactly one bucket.
///
/// Determinism: `removed` is name-ordered structurally — it is built from a `BTreeMap`'s `keys()`
/// iterator (ascending by API contract), so the order is established at construction and a
/// debug-mode `is_sorted` check pins the invariant without paying for a runtime sort. `modified_*`
/// is name-sorted load-bearing — built from source-order `active_watches`, sorted at the end via
/// `sort_unstable_by` (names are unique by validation, so stability is unobservable). `added`
/// preserves new-source order.
#[must_use]
pub fn diff(old: &Config, new: &Config) -> SubRegistryDiff {
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

    // `removed` is collected from a `BTreeMap`'s keys, so it is already name-ordered by construction
    // — the debug assertion pins the invariant without paying for a runtime sort. Both `modified_*`
    // buckets are built from source-order `active_watches()` and are the load-bearing sort: replay
    // stability keys on config content, never slotmap mint order. Names are unique by validation, so
    // `sort_unstable_by` is observably indistinguishable from a stable sort and strictly cheaper.
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

#[cfg(test)]
mod tests {
    use super::diff;
    use crate::config::Config;
    use compact_str::CompactString;
    use specter_core::ScanConfig;

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

    /// Build a watch block with an explicit `settle`. `max_settle` is left to the default (1h),
    /// which is comfortably above the floor for any reasonable `settle`.
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
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    #[test]
    fn add_only_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [block("a", "echo"), block("b", "echo")];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff(&old, &new);
        assert_eq!(d.added.len(), 2);
        assert_eq!(d.added[0].params.name, "a");
        assert_eq!(d.added[1].params.name, "b");
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    #[test]
    fn remove_only_populates_removed_with_watch_name() {
        let old_blocks = [block("a", "echo")];
        let refs: Vec<&str> = old_blocks.iter().map(String::as_str).collect();
        let old = cfg(&refs);
        let new = cfg(&[]);

        let d = diff(&old, &new);
        assert!(d.added.is_empty());
        assert_eq!(d.removed, vec![CompactString::from("a")]);
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    #[test]
    fn identical_configs_yield_empty_diff() {
        let blocks = [block("a", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff(&a, &b);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// Command change ⇒ per-Sub field only ⇒ `modified_params`. The identity bucket stays empty:
    /// the partition is exhaustive and disjoint per watch name.
    #[test]
    fn different_command_lands_in_modified_params() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("a", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert_eq!(d.modified_params.len(), 1);
        assert_eq!(d.modified_params[0].params.name, "a");
    }

    #[test]
    fn add_remove_modify_mix() {
        let old_blocks = [block("a", "echo"), block("b", "echo"), block("c", "echo")];
        let new_blocks = [block("a", "fmt"), block("c", "echo"), block("d", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);

        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].params.name, "d");

        assert_eq!(d.removed, vec![CompactString::from("b")]);

        // `block("a", "echo") → block("a", "fmt")` is a program-only change ⇒ `modified_params`.
        assert!(d.modified_identity.is_empty());
        assert_eq!(d.modified_params.len(), 1);
        assert_eq!(d.modified_params[0].params.name, "a");
    }

    #[test]
    fn reorder_only_yields_empty_diff() {
        let order_a = [block("a", "echo"), block("b", "echo")];
        let order_b = [block("b", "echo"), block("a", "echo")];
        let old = cfg(&order_a.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&order_b.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    #[test]
    fn rename_yields_added_plus_removed() {
        let old_blocks = [block("a", "echo")];
        let new_blocks = [block("z", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].params.name, "z");
        assert_eq!(d.removed, vec![CompactString::from("a")]);
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// `settle` is a per-Sub field, not a Profile-identity field ⇒ `modified_params`. Pinning the
    /// bucket guards against a future `settle`-into-identity drift that would silently re-route
    /// this case through `modified_identity` (detach+attach with baseline loss) instead of in-place
    /// rebind.
    #[test]
    fn settle_change_lands_in_modified_params() {
        let old_blocks = [block_full("a", "echo", "200ms")];
        let new_blocks = [block_full("a", "echo", "500ms")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert_eq!(d.modified_params.len(), 1);
    }

    #[test]
    fn removed_sorted_by_name() {
        let old_blocks = [block("c", "echo"), block("a", "echo"), block("b", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);
        let d = diff(&old, &new);
        assert_eq!(
            d.removed,
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
            .modified_params
            .iter()
            .map(|r| r.params.name.as_str())
            .collect();
        assert_eq!(order, vec!["a", "b"]);
    }

    /// `events` folds into `ProfileIdentity::config_hash` ⇒ `modified_identity`. Same role guard as
    /// [`settle_change_lands_in_modified_params`]: pinning the bucket makes a future
    /// identity-into-params drift visible at the diff layer.
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
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_params.is_empty());
        assert_eq!(d.modified_identity.len(), 1);
        assert_eq!(d.modified_identity[0].params.name, "a");
        assert_eq!(
            d.modified_identity[0].identity.events,
            specter_core::ClassSet::CONTENT
        );
    }

    #[test]
    fn explicit_events_equal_to_default_yields_no_diff() {
        // A user adding `events = ["structure", "content"]` cosmetically (matching the implicit
        // subtree-root default) must not churn the Profile.
        let old_blocks = [block("a", "echo")];
        let new_blocks = [format!(
            "[[watch]]\nname = \"a\"\npath = \"{ROOT}\"\n\
             actions = [{{ exec = [\"echo\"] }}]\nevents = [\"structure\", \"content\"]"
        )];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let d = diff(&old, &new);
        assert!(
            d.modified_identity.is_empty(),
            "identity bucket: {:?}",
            d.modified_identity,
        );
        assert!(
            d.modified_params.is_empty(),
            "params bucket: {:?}",
            d.modified_params,
        );
    }

    #[test]
    fn events_class_order_does_not_affect_diff() {
        // Class set is bitmask-equality, not list-order — the parser collapses both orderings to
        // the same ClassSet.
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
            d.modified_identity.is_empty(),
            "identity bucket: {:?}",
            d.modified_identity,
        );
        assert!(
            d.modified_params.is_empty(),
            "params bucket: {:?}",
            d.modified_params,
        );
    }

    /// `scan` (recursive flag) folds into `ProfileIdentity::config_hash` ⇒ `modified_identity`.
    /// Same name, same path, different scan ⇒ the Sub must move to a different Profile. Distinct
    /// from `events_change_lands_in_modified_identity` because `scan` is a nested struct
    /// (`ScanConfig`) where `events` is a `ClassSet` bitmask — exercising a struct-equality path
    /// the bitmask test doesn't.
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
        assert!(d.modified_params.is_empty());
        assert_eq!(d.modified_identity.len(), 1);
        assert_eq!(d.modified_identity[0].params.name, "a");
    }

    // ---- Discovery (dynamic) side ----

    /// Adding a fresh dynamic [[watch]] populates `added` in source order with template-bearing
    /// requests. Nothing on the old side, so no removed / modified entries.
    #[test]
    fn discovery_added_populates_added_in_source_order() {
        let old = cfg(&[]);
        let new_blocks = [
            dyn_block("logs", "/var/log/*", "echo"),
            dyn_block("sites", "/srv/*/site", "fmt"),
        ];
        let refs: Vec<&str> = new_blocks.iter().map(String::as_str).collect();
        let new = cfg(&refs);

        let d = diff(&old, &new);
        assert_eq!(d.added.len(), 2);
        assert_eq!(d.added[0].params.name, "logs");
        assert_eq!(d.added[1].params.name, "sites");
        assert!(d.added.iter().all(|r| r.params.template.is_some()));
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// Removing a dynamic [[watch]] populates `removed` with the operator name — the engine's
    /// detach cascade reaps the minted set from the template Sub.
    #[test]
    fn discovery_removed_populates_removed_with_name() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&[]);

        let d = diff(&old, &new);
        assert_eq!(d.removed, vec![CompactString::from("logs")]);
        assert!(d.added.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// A **program-only** change on a template-bearing pair classifies `modified_identity` — the same
    /// edit on a static spec lands in `modified_params` (see
    /// [`different_command_lands_in_modified_params`]). Minted Subs hold `Arc`s of the template Sub's
    /// program; an in-place rebind would strand them, so any discovery edit is a wholesale replace.
    #[test]
    fn discovery_command_change_lands_in_modified_identity() {
        let old_blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let new_blocks = [dyn_block("logs", "/var/log/*", "fmt")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_params.is_empty());
        assert_eq!(d.modified_identity.len(), 1);
        assert_eq!(d.modified_identity[0].params.name, "logs");
    }

    /// Pattern source change re-anchors the discovery Profile; diff surfaces it as
    /// `modified_identity` carrying the fresh `MatchChain` scan. The engine wholesale-replaces,
    /// which reaps the old minted set and re-mints against the new pattern.
    #[test]
    fn discovery_pattern_change_lands_in_modified_identity() {
        let old_blocks = [dyn_block("logs", "/var/log/*.log", "echo")];
        let new_blocks = [dyn_block("logs", "/var/log/*.json", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert_eq!(d.modified_identity.len(), 1);
        let ScanConfig::MatchChain(spec) = d.modified_identity[0].identity.config.as_ref() else {
            panic!("discovery request carries MatchChain");
        };
        assert_eq!(spec.source(), "/var/log/*.json");
    }

    /// A user-`settle`-only change on a dynamic watch classifies `modified_identity` — the user's
    /// `settle` lives on the *template* (the minted Subs' debounce), so a params-class rebind can
    /// never carry it. Wholesale before the unification, identity now; never an in-place rebind.
    #[test]
    fn discovery_settle_change_lands_in_modified_identity() {
        let old_blocks = [dyn_block_full("logs", "/var/log/*", "echo", "200ms")];
        let new_blocks = [dyn_block_full("logs", "/var/log/*", "echo", "500ms")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);
        assert!(d.modified_params.is_empty());
        assert_eq!(d.modified_identity.len(), 1);
    }

    /// Identical dynamic blocks yield no diff — `PatternSpec` equality routes through `source` and
    /// `TemplateSpec` is structural, so re-parsing the same TOML is diff-invisible.
    #[test]
    fn discovery_identical_configs_yield_empty_diff() {
        let blocks = [dyn_block("logs", "/var/log/*", "echo")];
        let refs: Vec<&str> = blocks.iter().map(String::as_str).collect();
        let a = cfg(&refs);
        let b = cfg(&refs);
        let d = diff(&a, &b);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    // ---- Cross-kind: static ↔ dynamic migration via path edit ----

    /// Static → dynamic via path edit: same name, but the path crossed the `is_dynamic` boundary,
    /// so template presence differs across the pair and the head guard classifies
    /// `modified_identity` — a wholesale teardown then attach. The path semantics meaningfully
    /// changed; an in-place rebind would hide that.
    #[test]
    fn static_to_dynamic_migration_lands_in_modified_identity() {
        let old_blocks = [block("foo", "echo")]; // path = "/" (static)
        let new_blocks = [dyn_block("foo", "/foo/*", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);

        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_params.is_empty());
        assert_eq!(d.modified_identity.len(), 1);
        assert_eq!(d.modified_identity[0].params.name, "foo");
        assert!(d.modified_identity[0].params.template.is_some());
    }

    /// Reverse direction: dynamic → static via path edit. Same head guard (template presence on the
    /// *old* side), same `modified_identity` bucket; the request is now template-free.
    #[test]
    fn dynamic_to_static_migration_lands_in_modified_identity() {
        let old_blocks = [dyn_block("foo", "/foo/*", "echo")];
        let new_blocks = [block("foo", "echo")];
        let old = cfg(&old_blocks.iter().map(String::as_str).collect::<Vec<_>>());
        let new = cfg(&new_blocks.iter().map(String::as_str).collect::<Vec<_>>());

        let d = diff(&old, &new);

        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_params.is_empty());
        assert_eq!(d.modified_identity.len(), 1);
        assert_eq!(d.modified_identity[0].params.name, "foo");
        assert!(d.modified_identity[0].params.template.is_none());
    }

    /// Mixed reload: one static modify, one discovery add, one of each removed. Static and
    /// discovery entries ride the same buckets without interaction.
    #[test]
    fn mixed_static_and_discovery_changes_compose_independently() {
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

        let removed: Vec<&str> = d.removed.iter().map(CompactString::as_str).collect();
        assert_eq!(removed, vec!["dyn_drop", "static_drop"]);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].params.name, "dyn_new");
        // `static_keep`: echo → fmt is a program-only change on a static spec ⇒ params bucket.
        assert!(d.modified_identity.is_empty());
        assert_eq!(d.modified_params.len(), 1);
        assert_eq!(d.modified_params[0].params.name, "static_keep");
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

    /// Flipping `enabled = true → false` filters the entry out of the new effective set, surfacing
    /// as `subs.removed` — the same shape the engine sees for an outright deletion. This is the
    /// load-bearing case of the feature.
    #[test]
    fn enabled_true_to_false_yields_subs_removed() {
        let old = cfg(&[block_with_enabled("a", "echo", true).as_str()]);
        let new = cfg(&[block_with_enabled("a", "echo", false).as_str()]);
        let d = diff(&old, &new);
        assert_eq!(d.removed, vec![CompactString::from("a")]);
        assert!(d.added.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// Reverse: `false → true` re-introduces the entry, surfacing as `subs.added`.
    #[test]
    fn enabled_false_to_true_yields_subs_added() {
        let old = cfg(&[block_with_enabled("a", "echo", false).as_str()]);
        let new = cfg(&[block_with_enabled("a", "echo", true).as_str()]);
        let d = diff(&old, &new);
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].params.name, "a");
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// Edits to other fields while the entry is disabled produce no diff: both sides are filtered
    /// out before the comparison, so the engine sees nothing. The new field values apply on the
    /// next `false → true` transition via the fresh attach.
    #[test]
    fn disabled_to_disabled_with_field_change_yields_empty_diff() {
        let old = cfg(&[block_with_enabled("a", "echo", false).as_str()]);
        let new = cfg(&[block_with_enabled("a", "fmt", false).as_str()]);
        let d = diff(&old, &new);
        assert!(d.added.is_empty());
        assert!(d.removed.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());
    }

    /// Dynamic-block enabled flip mirrors the static side — discovery specs ride the same
    /// `active_watches` filter, so the flip surfaces as removed/added, never modified. The removed
    /// name reaches the engine as a plain detach whose cascade reaps the minted set.
    #[test]
    fn discovery_enabled_flip_yields_removed_then_added() {
        let on = dyn_block_with_enabled("logs", "/var/log/*", "echo", true);
        let off = dyn_block_with_enabled("logs", "/var/log/*", "echo", false);
        let d = diff(&cfg(&[on.as_str()]), &cfg(&[off.as_str()]));
        assert_eq!(d.removed, vec![CompactString::from("logs")]);
        assert!(d.added.is_empty());
        assert!(d.modified_identity.is_empty());
        assert!(d.modified_params.is_empty());

        let d = diff(&cfg(&[off.as_str()]), &cfg(&[on.as_str()]));
        assert_eq!(d.added.len(), 1);
        assert_eq!(d.added[0].params.name, "logs");
        assert!(d.added[0].params.template.is_some());
        assert!(d.removed.is_empty());
    }
}
