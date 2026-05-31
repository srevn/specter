//! Sub-count derivation invariance.
//!
//! `SubRegistry::at(profile).len()` is the *sole* source of a Profile's
//! live-Sub count, and the `detach_sub_inner` choreography derives its
//! post-detach count from it directly. This pins the invariant: across
//! an arbitrary attach/detach permutation on Subs sharing one Profile,
//! the derived count tracks the live set exactly — no drift, since
//! there is no second counter to drift from.

use specter_core::{Input, ResourceRole, ScanConfig, SubAttachAnchor, SubId};
use specter_engine::Engine;
use specter_engine::testkit::{self, NO_EVENTS};
use std::time::Instant;

#[test]
fn subs_at_len_is_the_sole_derived_count_across_attach_detach() {
    let mut e = Engine::new();
    let r = e.tree_mut().ensure_root("src", ResourceRole::User);
    e.tree_mut().set_kind(r, specter_core::ResourceKind::Dir);
    // Identical (resource, ScanConfig, max_settle, settle, events) ⇒
    // identical config_hash ⇒ every Sub attaches to one shared Profile.
    let cfg = ScanConfig::builder().recursive(true).build();
    let now = Instant::now();

    let attach = |e: &mut Engine, name: &str| -> SubId {
        testkit::attach(
            e,
            name,
            SubAttachAnchor::Resource(r),
            cfg.clone(),
            NO_EVENTS,
            testkit::MAX_SETTLE,
            now,
        )
        .0
    };

    let s1 = attach(&mut e, "A");
    let pid = e.subs().get(s1).expect("Sub A live").profile();
    assert_eq!(e.subs().at(pid).len(), 1, "one Sub attached");

    let s2 = attach(&mut e, "B");
    let s3 = attach(&mut e, "C");
    assert_eq!(
        e.subs().get(s2).unwrap().profile(),
        pid,
        "B shares the Profile (same config_hash)",
    );
    assert_eq!(
        e.subs().get(s3).unwrap().profile(),
        pid,
        "C shares the Profile"
    );
    assert_eq!(e.subs().at(pid).len(), 3, "count tracks three live Subs");

    // Detach the middle Sub: derived count drops to 2, A and C survive.
    e.step(Input::DetachSub(s2), now);
    assert_eq!(
        e.subs().at(pid).len(),
        2,
        "count derives the live set, not a mirror"
    );
    assert!(e.subs().get(s2).is_none(), "B gone from the registry");
    for s in [s1, s3] {
        assert_eq!(
            e.subs().get(s).unwrap().profile(),
            pid,
            "survivors still share the Profile",
        );
    }

    // Re-attach pushes the derived count back up — symmetric.
    let s4 = attach(&mut e, "D");
    assert_eq!(
        e.subs().get(s4).unwrap().profile(),
        pid,
        "D shares the Profile"
    );
    assert_eq!(e.subs().at(pid).len(), 3, "count climbs back to three");

    // Drain to empty. Each detach derives the next count from the
    // registry alone; the last one leaves zero.
    for (sid, expected) in [(s1, 2), (s3, 1), (s4, 0)] {
        e.step(Input::DetachSub(sid), now);
        assert_eq!(
            e.subs().at(pid).len(),
            expected,
            "post-detach count is the registry-derived live count",
        );
    }
    let _ = e.cancel_all_in_flight_probes();
}
