//! Cross-module integration: shared Profile via `config_hash`, distinct
//! Profile across `max_settle`/`pattern`, detach clears both indices, slot
//! semantics under Profile anchoring.
//!
//! These tests intentionally exercise the Sub addition flow (find then
//! attach if absent) so any future shape change here is visible at this
//! seam.

use specter_core::{
    ArgPart, ArgTemplate, ClassSet, CommandTemplate, EffectScope, GlobPattern, Placeholder,
    Profile, ProfileMap, ResourceRole, ScanConfig, Sub, SubRegistry, Tree, compute_config_hash,
};
use std::time::Duration;

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn bare_cfg() -> ScanConfig {
    ScanConfig::builder().build()
}

fn build_template() -> CommandTemplate {
    CommandTemplate::new([ArgTemplate::new([
        ArgPart::literal("/bin/build"),
        ArgPart::Placeholder(Placeholder::Path),
    ])])
}

#[test]
fn shared_profile_via_config_hash() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let mut subs = SubRegistry::new();

    let r = tree.ensure(None, "/anchor", ResourceRole::User);
    let cfg = bare_cfg();
    let hash = compute_config_hash(&cfg, MAX_SETTLE, NO_EVENTS);

    // Sub A: creates the Profile (find = None).
    let pid_a = profiles.find(r, hash).unwrap_or_else(|| {
        profiles.attach(
            &mut tree,
            Profile::new(r, cfg.clone(), MAX_SETTLE, SETTLE, NO_EVENTS),
        )
    });
    profiles.get_mut(pid_a).unwrap().sub_refcount += 1;
    let _sid_a = subs.insert(|id| {
        Sub::new(
            id,
            "build-a",
            pid_a,
            build_template(),
            EffectScope::SubtreeRoot,
            SETTLE,
            MAX_SETTLE,
            NO_EVENTS,
        )
    });

    // Sub B: same (resource, hash); reuses the Profile.
    let pid_b = profiles
        .find(r, hash)
        .expect("Profile exists from Sub A's attach");
    assert_eq!(pid_a, pid_b, "shared Profile across matching configs");
    profiles.get_mut(pid_b).unwrap().sub_refcount += 1;

    assert_eq!(profiles.get(pid_a).unwrap().sub_refcount, 2);
    assert_eq!(profiles.len(), 1);
    assert_eq!(tree.get(r).unwrap().profiles().len(), 1);
}

#[test]
fn distinct_profile_for_distinct_max_settle() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();

    let r = tree.ensure(None, "/anchor", ResourceRole::User);

    let pid_short = profiles.attach(
        &mut tree,
        Profile::new(r, bare_cfg(), Duration::from_secs(6), SETTLE, NO_EVENTS),
    );
    let pid_long = profiles.attach(
        &mut tree,
        Profile::new(r, bare_cfg(), Duration::from_secs(12), SETTLE, NO_EVENTS),
    );

    assert_ne!(pid_short, pid_long);
    assert_eq!(profiles.len(), 2);
    assert_eq!(tree.get(r).unwrap().profiles().len(), 2);
}

#[test]
fn distinct_profile_for_distinct_pattern() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();

    let r = tree.ensure(None, "/anchor", ResourceRole::User);

    let cfg_rs = ScanConfig::builder()
        .pattern(GlobPattern::compile("*.rs").unwrap())
        .build();
    let cfg_txt = ScanConfig::builder()
        .pattern(GlobPattern::compile("*.txt").unwrap())
        .build();

    let pid_rs = profiles.attach(
        &mut tree,
        Profile::new(r, cfg_rs, MAX_SETTLE, SETTLE, NO_EVENTS),
    );
    let pid_txt = profiles.attach(
        &mut tree,
        Profile::new(r, cfg_txt, MAX_SETTLE, SETTLE, NO_EVENTS),
    );

    assert_ne!(pid_rs, pid_txt);
}

#[test]
fn detach_clears_back_references_on_both_sides() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();

    let r = tree.ensure(None, "/anchor", ResourceRole::User);
    let pid = profiles.attach(
        &mut tree,
        Profile::new(r, bare_cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
    );

    profiles.detach(&mut tree, pid);

    assert!(profiles.get(pid).is_none());
    assert!(
        profiles
            .find(r, compute_config_hash(&bare_cfg(), MAX_SETTLE, NO_EVENTS))
            .is_none()
    );
    assert!(tree.get(r).unwrap().profiles().is_empty());
}

#[test]
fn rename_after_detach_yields_fresh_id() {
    // Mimics the engine's rename handling: detach Profile → vacate +
    // try_reap → ensure with a new segment.
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let parent = tree.ensure(None, "/dir", ResourceRole::User);

    let id_old = tree.ensure(Some(parent), "foo.c", ResourceRole::User);
    let pid = profiles.attach(
        &mut tree,
        Profile::new(id_old, bare_cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
    );

    // Rename: engine detaches the Profile, then vacates + try_reaps the slot.
    profiles.detach(&mut tree, pid);
    tree.vacate(id_old);
    assert!(tree.try_reap(id_old), "post-detach reap must succeed");

    let id_new = tree.ensure(Some(parent), "bar.c", ResourceRole::User);
    assert_ne!(id_old, id_new, "renamed slot yields fresh id");
}

#[test]
fn recreate_at_anchored_slot_keeps_id() {
    // Mimics `touch foo.c` recreating after a delete that didn't detach the
    // Profile.
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let parent = tree.ensure(None, "/dir", ResourceRole::User);
    let id = tree.ensure(Some(parent), "foo.c", ResourceRole::User);
    let _pid = profiles.attach(
        &mut tree,
        Profile::new(id, bare_cfg(), MAX_SETTLE, SETTLE, NO_EVENTS),
    );

    // Vacate without detach: slot is anchored by Profile, try_reap refused.
    tree.vacate(id);
    assert!(!tree.try_reap(id), "Profile-anchored slot must not reap");

    // Same (parent, segment) returns the same id.
    let id_again = tree.ensure(Some(parent), "foo.c", ResourceRole::User);
    assert_eq!(id, id_again, "anchored slot reused on re-ensure");
}
