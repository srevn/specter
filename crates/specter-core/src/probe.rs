//! At-most-one-in-flight probe slot.
//!
//! [`ProbeSlot`] is the single home for one owner-state carrier's probe
//! liveness *and* identity. It collapses two facts the engine otherwise
//! tracks apart — "is a probe in flight for this carrier?" and "which
//! correlation is it?" — into one `Option`:
//!
//! - **empty** ⇒ idle: no probe out.
//! - **armed** ⇒ in flight: holds the [`ProbeCorrelation`] the response
//!   must echo, plus a `Tag` for any per-probe dispatch key the variant
//!   itself does not already supply.
//!
//! [`ProbeSlot::disarm`] is the one consume primitive: it takes the slot
//! idle and yields the prior correlation. Routing a correlation to a
//! dispatch handler is "disarm, then act on the returned correlation" —
//! once disarmed, the same correlation cannot be routed again, because
//! it is no longer in the slot.
//!
//! "At most one probe per owner" is a representability property: one
//! owner-state carrier holds exactly one slot, so two simultaneous
//! probes for one carrier are unconstructable. The slot is **linear**
//! (non-`Copy`, non-`Clone`) and guards both linear edges:
//! [`ProbeSlot::arm`] backstops *re-acquire* with an unconditional
//! assert — re-arming a still-armed slot would orphan the prior
//! correlation (its response would then be rejected as stale even
//! though the engine asked for it). The [`Drop`] tripwire backstops
//! *destroy* with the same discipline — a slot dropped while still
//! armed orphans its correlation identically, so it crashes just as
//! loudly. Convention is not relied on at either edge.
//!
//! `Tag` is `()` for carriers whose state variant is itself the routing
//! class; it carries a real key (e.g. a [`ResourceId`]) only where the
//! response handler needs a dispatch datum the variant does not encode.

use crate::ids::ProbeCorrelation;

/// At most one in-flight probe for one owner-state carrier.
///
/// A **linear** (consume-once) value held *inside* the owning state
/// variant and mutated in place through a `&mut` to that variant. It is
/// deliberately **not** [`Copy`] and **not** [`Clone`]: the correlation
/// it carries must be consumed exactly once, so the slot cannot be
/// duplicated — it is consumed where it lives, never via a snapshot.
///
/// [`Self::disarm`] is the one consume. The protocol is guarded at both
/// linear edges: [`Self::arm`] guards *re-acquire* (a re-arm without an
/// intervening disarm orphans the prior correlation); the [`Drop`]
/// tripwire guards *destroy* (a slot reaching drop still armed orphans
/// its correlation just the same — its response would stale-detect even
/// though the engine asked for it, silently drifting a fire). The two
/// are duals and both crash loudly, by one "surface in every build"
/// discipline.
///
/// `Tag` defaults to `()` — the variant is the routing class. A carrier
/// whose response handler needs a dispatch key the variant does not
/// supply parameterises the slot with that key's type (which must be
/// [`Copy`]).
#[derive(Debug, Eq, PartialEq)]
pub struct ProbeSlot<Tag: Copy = ()> {
    inner: Option<(ProbeCorrelation, Tag)>,
}

/// The *destroy*-edge linearity guard — the structural dual of
/// [`ProbeSlot::arm`]'s *re-acquire*-edge assert. An armed slot reaching
/// drop means its correlation was never consumed: the engine emitted a
/// probe whose response will now stale-detect, silently drifting the
/// fire it gates. That is the same failure class and severity as a
/// double-`arm`, so it crashes loudly in every build (operator-visible,
/// fail-stop) rather than letting a daemon whose whole purpose is a
/// trustworthy absence-of-change proof carry on with a broken one.
/// `core` forbids I/O, so there is no fallback log; the panic carries
/// the orphaned correlation for triage.
///
/// `!panicking()` is the *only* reason the crash is conditional: a
/// second panic while already unwinding aborts the process and masks
/// the first failure — strictly less diagnosable, the opposite of the
/// intent. The engine `step` is single-threaded, so an orphan observed
/// only during an unrelated unwind is moot — no further step runs.
///
/// This is the **sole** explicit `Drop` in `core`/`engine`. Every
/// enclosing carrier (`PreFirePhase`, `PostFirePhase`, `PreFireBurst`,
/// `PostFireBurst`, `ProfileState`, `DescentState`, …) reaches this
/// guard through auto drop-glue, *not* its own `impl Drop`: partial
/// moves out of those carriers (draining a sibling field at burst-end)
/// depend on the absence of an explicit `Drop` (E0509). An `impl Drop`
/// anywhere up the tree would both break those moves and be redundant —
/// drop-glue already propagates this tripwire.
impl<Tag: Copy> Drop for ProbeSlot<Tag> {
    fn drop(&mut self) {
        let Some((correlation, _)) = &self.inner else {
            return; // idle / disarmed — the sanctioned terminal state.
        };
        if std::thread::panicking() {
            // Already unwinding: a second panic aborts the process and
            // masks the first failure — stay silent instead.
            return;
        }
        panic!(
            "ProbeSlot dropped while armed: probe correlation \
             {correlation:?} orphaned — a probe-bearing state \
             variant was destroyed or overwritten without a \
             preceding disarm; the response path \
             (take_owner_probe) or an abandon site \
             (cancel_owner_probe) must consume the slot before \
             teardown",
        );
    }
}

/// `Default` is the idle slot, regardless of `Tag`. Hand-written rather
/// than derived so it requires only `Tag: Copy` (the struct's own
/// bound): an empty slot holds no `Tag` value, so `Tag: Default` would
/// be a spurious bound a derived impl imposes.
impl<Tag: Copy> Default for ProbeSlot<Tag> {
    fn default() -> Self {
        Self::empty()
    }
}

impl<Tag: Copy> ProbeSlot<Tag> {
    /// An idle slot — no probe in flight.
    #[must_use]
    pub const fn empty() -> Self {
        Self { inner: None }
    }

    /// An armed slot carrying `c` and its dispatch `tag`.
    #[must_use]
    pub const fn armed(c: ProbeCorrelation, tag: Tag) -> Self {
        Self {
            inner: Some((c, tag)),
        }
    }

    /// Identity of the in-flight probe, or `None` if idle.
    #[must_use]
    pub const fn correlation(&self) -> Option<ProbeCorrelation> {
        match &self.inner {
            Some((c, _)) => Some(*c),
            None => None,
        }
    }

    /// Dispatch tag of the in-flight probe, or `None` if idle.
    #[must_use]
    pub const fn tag(&self) -> Option<Tag> {
        match &self.inner {
            Some((_, t)) => Some(*t),
            None => None,
        }
    }

    /// Arm an idle slot. Unconditional assert on a double-arm: a
    /// re-arm without an intervening [`Self::disarm`] would orphan the
    /// prior correlation, so this is a programming error that must
    /// surface in every build, not a silent overwrite.
    pub fn arm(&mut self, c: ProbeCorrelation, tag: Tag) {
        assert!(
            self.inner.is_none(),
            "I5 violated: ProbeSlot armed while already armed \
             (prior correlation would be orphaned, its response stale-detected)",
        );
        self.inner = Some((c, tag));
    }

    /// The single consume primitive: take the slot idle and return the
    /// prior correlation (`None` if it was already idle).
    #[must_use = "the disarmed probe correlation must be routed or explicitly discarded"]
    pub const fn disarm(&mut self) -> Option<ProbeCorrelation> {
        match self.inner.take() {
            Some((c, _)) => Some(c),
            None => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeSlot;
    use crate::ids::{ProbeCorrelation, ResourceId};
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn corr(n: u64) -> ProbeCorrelation {
        ProbeCorrelation::from(n)
    }

    /// `empty()` is idle: not armed, no correlation, no tag. An idle
    /// slot reaching drop is silent — the linear protocol only fires
    /// the tripwire on an *armed* drop.
    #[test]
    fn empty_is_idle() {
        let s: ProbeSlot = ProbeSlot::empty();
        assert_eq!(s.correlation(), None);
        assert_eq!(s.tag(), None);
    }

    /// `armed(c, ())` reports armed, surfaces the correlation, and the
    /// unit tag round-trips as `Some(())`. The slot is disarmed before
    /// it drops — an armed slot reaching drop trips the linearity
    /// tripwire, and this test's point (the projections) is proven
    /// before the consume.
    #[test]
    fn armed_unit_tag_reports_correlation() {
        let mut s: ProbeSlot = ProbeSlot::armed(corr(7), ());
        assert_eq!(s.correlation(), Some(corr(7)));
        assert_eq!(s.tag(), Some(()));
        let _ = s.disarm();
    }

    /// A non-unit `Tag` round-trips verbatim through `tag()`. Disarmed
    /// before drop to satisfy the linear protocol.
    #[test]
    fn armed_carries_non_unit_tag() {
        let target = ResourceId::default();
        let mut s: ProbeSlot<ResourceId> = ProbeSlot::armed(corr(3), target);
        assert_eq!(s.correlation(), Some(corr(3)));
        assert_eq!(s.tag(), Some(target));
        let _ = s.disarm();
    }

    /// `arm` on an idle slot makes it armed with the supplied values.
    /// Disarmed before drop to satisfy the linear protocol.
    #[test]
    fn arm_idle_slot_makes_it_armed() {
        let mut s: ProbeSlot = ProbeSlot::empty();
        s.arm(corr(11), ());
        assert_eq!(s.correlation(), Some(corr(11)));
        let _ = s.disarm();
    }

    /// `arm` on an already-armed slot panics unconditionally — a
    /// re-arm would orphan the prior correlation.
    #[test]
    #[should_panic(expected = "armed while already armed")]
    fn arm_panics_on_double_arm() {
        let mut s: ProbeSlot = ProbeSlot::armed(corr(1), ());
        s.arm(corr(2), ()); // panics
    }

    /// `disarm` on an armed slot returns the prior correlation and
    /// leaves the slot idle.
    #[test]
    fn disarm_returns_prior_and_idles() {
        let mut s: ProbeSlot = ProbeSlot::armed(corr(9), ());
        assert_eq!(s.disarm(), Some(corr(9)));
        assert_eq!(s.correlation(), None);
        assert_eq!(s.tag(), None);
    }

    /// `disarm` on an idle slot is a clean `None`, slot stays idle.
    #[test]
    fn disarm_idle_slot_returns_none() {
        let mut s: ProbeSlot = ProbeSlot::empty();
        assert_eq!(s.disarm(), None);
        assert!(s.correlation().is_none());
    }

    /// A slot can be re-armed after a disarm (the consume-then-mint
    /// cycle descent advance relies on). The re-arm leaves the slot
    /// armed, so it is disarmed again before drop.
    #[test]
    fn rearm_after_disarm_is_allowed() {
        let mut s: ProbeSlot = ProbeSlot::armed(corr(1), ());
        assert_eq!(s.disarm(), Some(corr(1)));
        s.arm(corr(2), ());
        assert_eq!(s.correlation(), Some(corr(2)));
        let _ = s.disarm();
    }

    /// `Default` is `empty()` and does not require `Tag: Default` — a
    /// `Copy` tag with no `Default` impl still yields an idle slot.
    #[test]
    fn default_is_empty() {
        let s: ProbeSlot = ProbeSlot::default();
        assert_eq!(s, ProbeSlot::empty());

        let r: ProbeSlot<ResourceId> = ProbeSlot::default();
        assert!(r.correlation().is_none());
    }

    /// Dropping an **armed** slot (not during an unwind) trips the
    /// linearity tripwire: it panics, and the payload names both the
    /// "dropped while armed" class and the orphaned correlation. This
    /// is the *destroy*-edge dual of the double-`arm` assert — an armed
    /// slot reaching drop orphans its correlation just as a re-arm
    /// would.
    #[test]
    fn drop_while_armed_panics_with_orphaned_correlation() {
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            let _s: ProbeSlot = ProbeSlot::armed(corr(42), ());
            // `_s` drops here, still armed → tripwire fires.
        }));
        let payload = panicked.expect_err("dropping an armed slot must panic");
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .expect("panic payload is a string");
        assert!(
            msg.contains("ProbeSlot dropped while armed"),
            "payload must name the linearity class, got: {msg}",
        );
        assert!(
            msg.contains(&format!("{:?}", corr(42))),
            "payload must carry the orphaned correlation, got: {msg}",
        );
    }

    /// Dropping a **disarmed** slot is silent, and so is dropping an
    /// `empty()` slot — the tripwire fires only on an armed drop, so
    /// the normal consume-then-drop path never crashes.
    #[test]
    fn drop_when_disarmed_or_empty_is_silent() {
        let quiet = catch_unwind(AssertUnwindSafe(|| {
            let mut s: ProbeSlot = ProbeSlot::armed(corr(8), ());
            let _ = s.disarm();
            // `s` drops here, idle → silent.
            let _e: ProbeSlot = ProbeSlot::empty();
            // `_e` drops here, never armed → silent.
        }));
        assert!(
            quiet.is_ok(),
            "an idle/empty slot reaching drop must not panic",
        );
    }

    /// The `!std::thread::panicking()` guard: an armed slot dropped
    /// *while a panic is already unwinding* must stay silent, so the
    /// original panic propagates intact rather than a double-panic
    /// aborting the process and masking it. The holder's armed slot
    /// drops as the `"primary"` unwind tears the frame down; the caught
    /// payload must still be `"primary"`, proving the ProbeSlot `Drop`
    /// observed `panicking() == true` and did not abort.
    #[test]
    fn drop_while_armed_during_unwind_does_not_double_panic() {
        struct Holder {
            _slot: ProbeSlot,
        }

        let payload = catch_unwind(AssertUnwindSafe(|| {
            let _h = Holder {
                _slot: ProbeSlot::armed(corr(99), ()),
            };
            panic!("primary");
            // Unwinding past here drops `_h` (and its armed slot)
            // *while panicking* → the tripwire must observe
            // `panicking() == true` and stay silent.
        }))
        .expect_err("the primary panic must propagate");
        let msg = payload
            .downcast_ref::<&str>()
            .copied()
            .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
            .expect("panic payload is a string");
        assert_eq!(
            msg, "primary",
            "the original panic must propagate; the in-unwind armed-drop \
             must neither abort nor replace it",
        );
    }

    /// Symmetry with the enumeration slot: a generic tagged
    /// `ProbeSlot<ResourceId>` dropped while armed trips the same
    /// tripwire — the guard is on `ProbeSlot<Tag>`, not only the
    /// unit-tag specialisation.
    #[test]
    fn drop_while_armed_panics_for_tagged_slot() {
        let target = ResourceId::default();
        let panicked = catch_unwind(AssertUnwindSafe(move || {
            let _s: ProbeSlot<ResourceId> = ProbeSlot::armed(corr(13), target);
            // `_s` drops here, still armed → tripwire fires.
        }));
        let payload = panicked.expect_err("dropping an armed tagged slot must panic");
        let msg = payload
            .downcast_ref::<String>()
            .map(String::as_str)
            .or_else(|| payload.downcast_ref::<&str>().copied())
            .expect("panic payload is a string");
        assert!(
            msg.contains("ProbeSlot dropped while armed"),
            "tagged slot must trip the same linearity tripwire, got: {msg}",
        );
        assert!(
            msg.contains(&format!("{:?}", corr(13))),
            "payload must carry the orphaned correlation, got: {msg}",
        );
    }
}
