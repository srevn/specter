//! Fixtures the bin's unit tests share.

use specter_core::testkit::dir_snap;
use specter_core::{EntryKind, SubAttachAnchor, SubId};
use specter_engine::Engine;
use specter_engine::testkit::{attach_discovery, mint_spec, pre_place_dir, seed_to_idle};
use std::time::Instant;

/// Mint one Sub the way the daemon does: a template `disc` on `/srv/*` attaches at a pre-placed
/// `/srv`, and its Seed's walk lists one match, which the pass mints as `disc@/srv/a`. Returns the
/// minted Sub; its own Seed is left in flight.
pub(crate) fn mint_one(engine: &mut Engine) -> SubId {
    let now = Instant::now();
    let srv = pre_place_dir(engine, &["srv"]);
    let (template, discovery) = attach_discovery(
        engine,
        "disc",
        SubAttachAnchor::Resource(srv),
        "/srv/*",
        mint_spec(),
        now,
    );
    let _ = seed_to_idle(
        engine,
        discovery,
        &dir_snap(&[("a", EntryKind::Dir, 1)]),
        now,
    );
    engine
        .subs()
        .minted(template)
        .next()
        .expect("the pass mints the one match")
}
