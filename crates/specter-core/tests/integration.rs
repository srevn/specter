//! Cross-module integration: shared Profile via `config_hash`, distinct Profile across
//! `max_settle`/`pattern`, detach clears both indices, slot semantics under Profile anchoring.
//!
//! These tests intentionally exercise the Sub addition flow (find then attach if absent) so any
//! future shape change here is visible at this seam.

use specter_core::program::{BranchTarget, ProgramBuilder, SpawnBody};
use specter_core::{
    ActionProgram, ArgPart, ArgTemplate, ClassSet, EffectScope, ExecAction, GlobPattern,
    Placeholder, Profile, ProfileIdentity, ProfileMap, ResourceRole, ScanConfig, StepOutput, Sub,
    SubParams, SubRegistry, Tree,
};
use std::sync::Arc;
use std::time::Duration;

const SETTLE: Duration = Duration::from_millis(100);
const MAX_SETTLE: Duration = Duration::from_secs(6);
const NO_EVENTS: ClassSet = ClassSet::EMPTY;

fn bare_cfg() -> ScanConfig {
    ScanConfig::builder().build()
}

fn build_program() -> Arc<ActionProgram> {
    let mut b = ProgramBuilder::new();
    let h = b.emit(SpawnBody::Exec(ExecAction::new(
        [ArgTemplate::new([
            ArgPart::literal("/bin/build"),
            ArgPart::Placeholder(Placeholder::Path),
        ])],
        None,
    )));
    b.patch_on_ok(h, BranchTarget::Escape).unwrap();
    b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
    Arc::new(b.build().unwrap())
}

#[test]
fn shared_profile_via_config_hash() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let mut subs = SubRegistry::new();

    let r = tree.ensure_root("/anchor", ResourceRole::User);
    let cfg = bare_cfg();
    let hash = ProfileIdentity::new(cfg.clone(), MAX_SETTLE, NO_EVENTS).config_hash();

    // Sub A: creates the Profile (find = None).
    let pid_a = profiles.find(r, hash).unwrap_or_else(|| {
        profiles.attach(
            &mut tree,
            Profile::new(
                r,
                ProfileIdentity::new(cfg.clone(), MAX_SETTLE, NO_EVENTS),
                SETTLE,
                None,
            ),
        )
    });
    let _sid_a = subs.insert(Sub::from_request(
        pid_a,
        SubParams {
            name: "build-a".into(),
            program: build_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            template: None,
            source_discovery: None,
        },
    ));

    // Sub B: same (resource, hash); reuses the Profile.
    let pid_b = profiles
        .find(r, hash)
        .expect("Profile exists from Sub A's attach");
    assert_eq!(pid_a, pid_b, "shared Profile across matching configs");

    assert_eq!(profiles.len(), 1);
    assert_eq!(tree.get(r).unwrap().profiles().len(), 1);
}

#[test]
fn distinct_profile_for_distinct_max_settle() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();

    let r = tree.ensure_root("/anchor", ResourceRole::User);

    let pid_short = profiles.attach(
        &mut tree,
        Profile::new(
            r,
            ProfileIdentity::new(bare_cfg(), Duration::from_secs(6), NO_EVENTS),
            SETTLE,
            None,
        ),
    );
    let pid_long = profiles.attach(
        &mut tree,
        Profile::new(
            r,
            ProfileIdentity::new(bare_cfg(), Duration::from_secs(12), NO_EVENTS),
            SETTLE,
            None,
        ),
    );

    assert_ne!(pid_short, pid_long);
    assert_eq!(profiles.len(), 2);
    assert_eq!(tree.get(r).unwrap().profiles().len(), 2);
}

#[test]
fn distinct_profile_for_distinct_pattern() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();

    let r = tree.ensure_root("/anchor", ResourceRole::User);

    let cfg_rs = ScanConfig::builder()
        .pattern(GlobPattern::compile("*.rs").unwrap())
        .build();
    let cfg_txt = ScanConfig::builder()
        .pattern(GlobPattern::compile("*.txt").unwrap())
        .build();

    let pid_rs = profiles.attach(
        &mut tree,
        Profile::new(
            r,
            ProfileIdentity::new(cfg_rs, MAX_SETTLE, NO_EVENTS),
            SETTLE,
            None,
        ),
    );
    let pid_txt = profiles.attach(
        &mut tree,
        Profile::new(
            r,
            ProfileIdentity::new(cfg_txt, MAX_SETTLE, NO_EVENTS),
            SETTLE,
            None,
        ),
    );

    assert_ne!(pid_rs, pid_txt);
}

#[test]
fn detach_clears_back_references_on_both_sides() {
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();

    let r = tree.ensure_root("/anchor", ResourceRole::User);
    let pid = profiles.attach(
        &mut tree,
        Profile::new(
            r,
            ProfileIdentity::new(bare_cfg(), MAX_SETTLE, NO_EVENTS),
            SETTLE,
            None,
        ),
    );

    profiles.detach(&mut tree, pid);

    assert!(profiles.get(pid).is_none());
    assert!(
        profiles
            .find(
                r,
                ProfileIdentity::new(bare_cfg(), MAX_SETTLE, NO_EVENTS).config_hash()
            )
            .is_none()
    );
    assert!(tree.get(r).unwrap().profiles().is_empty());
}

#[test]
fn rename_after_detach_yields_fresh_id() {
    // Mimics the engine's rename handling: detach Profile → vacate + try_reap → ensure with a new
    // segment. The cascade in `try_reap` also frees the now-orphaned `/dir` parent (no other
    // claims), so the post-reap re-attach re-creates the full path — the engine's own
    // `attach_sub_inner` does exactly this via `materialize_path_or_pending`.
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let parent = tree.ensure_root("/dir", ResourceRole::User);

    let id_old = tree
        .ensure_child(parent, "foo.c", ResourceRole::User)
        .expect("test live parent");
    let pid = profiles.attach(
        &mut tree,
        Profile::new(
            id_old,
            ProfileIdentity::new(bare_cfg(), MAX_SETTLE, NO_EVENTS),
            SETTLE,
            None,
        ),
    );

    // Rename: engine detaches the Profile, then try_reaps the slot.
    profiles.detach(&mut tree, pid);
    assert!(
        tree.try_reap(id_old, &mut StepOutput::default()),
        "post-detach reap must succeed",
    );
    assert!(
        tree.get(parent).is_none(),
        "cascade reaped the now-orphaned parent",
    );

    // Re-ensure the path (the engine's own re-attach flow). The fresh `/dir` slot gets a new id;
    // the renamed `bar.c` under it likewise.
    let parent_fresh = tree.ensure_root("/dir", ResourceRole::User);
    let id_new = tree
        .ensure_child(parent_fresh, "bar.c", ResourceRole::User)
        .expect("test live parent");
    assert_ne!(parent, parent_fresh, "parent slot was reaped and re-minted");
    assert_ne!(id_old, id_new, "renamed slot yields fresh id");
}

#[test]
fn recreate_at_anchored_slot_keeps_id() {
    // Mimics `touch foo.c` recreating after a delete that didn't detach the Profile.
    let mut tree = Tree::new();
    let mut profiles = ProfileMap::new();
    let parent = tree.ensure_root("/dir", ResourceRole::User);
    let id = tree
        .ensure_child(parent, "foo.c", ResourceRole::User)
        .expect("test live parent");
    let _pid = profiles.attach(
        &mut tree,
        Profile::new(
            id,
            ProfileIdentity::new(bare_cfg(), MAX_SETTLE, NO_EVENTS),
            SETTLE,
            None,
        ),
    );

    // try_reap without detach: slot is anchored by Profile, refused.
    assert!(
        !tree.try_reap(id, &mut StepOutput::default()),
        "Profile-anchored slot must not reap",
    );

    // Same (parent, segment) returns the same id.
    let id_again = tree
        .ensure_child(parent, "foo.c", ResourceRole::User)
        .expect("test live parent");
    assert_eq!(id, id_again, "anchored slot reused on re-ensure");
}
