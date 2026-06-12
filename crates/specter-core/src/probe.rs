//! At-most-one-in-flight probe slot.
//!
//! [`ProbeSlot`] is the single home for one owner-state carrier's probe liveness *and* identity. It
//! collapses two facts the engine otherwise tracks apart — "is a probe in flight for this carrier?"
//! and "which correlation is it?" — into one `Option`:
//!
//! - **empty** ⇒ idle: no probe out.
//! - **armed** ⇒ in flight: holds the [`ProbeCorrelation`] the response must echo.
//!
//! [`ProbeSlot::disarm`] is the one consume primitive: it takes the slot idle and yields the prior
//! correlation. Routing a correlation to a dispatch handler is "disarm, then act on the returned
//! correlation" — once disarmed, the same correlation cannot be routed again, because it is no
//! longer in the slot.
//!
//! "At most one probe per owner" is a representability property: one owner-state carrier holds
//! exactly one slot, so two simultaneous probes for one carrier are unconstructable. The slot is
//! **linear** (non-`Copy`, non-`Clone`) and guards both linear edges: [`ProbeSlot::arm`] backstops
//! *re-acquire* with an unconditional assert — re-arming a still-armed slot would orphan the prior
//! correlation (its response would then be rejected as stale even though the engine asked for it).
//! The [`Drop`] tripwire backstops *destroy* with the same discipline — a slot dropped while still
//! armed orphans its correlation identically, so it crashes just as loudly. Convention is not
//! relied on at either edge.

use crate::ids::ProbeCorrelation;

/// At most one in-flight probe for one owner-state carrier.
///
/// A **linear** (consume-once) value held *inside* the owning state variant and mutated in place
/// through a `&mut` to that variant. It is deliberately **not** [`Copy`] and **not** [`Clone`]: the
/// correlation it carries must be consumed exactly once, so the slot cannot be duplicated — it is
/// consumed where it lives, never via a snapshot.
///
/// [`Self::disarm`] is the one consume. The protocol is guarded at both linear edges: [`Self::arm`]
/// guards *re-acquire* (a re-arm without an intervening disarm orphans the prior correlation); the
/// [`Drop`] tripwire guards *destroy* (a slot reaching drop still armed orphans its correlation just
/// the same — its response would stale-detect even though the engine asked for it, silently drifting
/// a fire). The two are duals and both crash loudly, by one "surface in every build" discipline.
///
/// [`Default`] is [`Self::empty`] (the idle slot) — derived, since the sole field defaults to `None`.
/// Load-bearing: every `#[derive(Default)]` on a carrier that holds a `ProbeSlot` must default it
/// idle, or the carrier's first drop would trip the [`Drop`] tripwire on a never-armed slot.
#[derive(Debug, Default, Eq, PartialEq)]
pub struct ProbeSlot {
    inner: Option<ProbeCorrelation>,
}

/// The *destroy*-edge linearity guard — the structural dual of [`ProbeSlot::arm`]'s
/// *re-acquire*-edge assert. An armed slot reaching drop means its correlation was never consumed:
/// the engine emitted a probe whose response will now stale-detect, silently drifting the fire it
/// gates. That is the same failure class and severity as a double-`arm`, so it crashes loudly in
/// every build (operator-visible, fail-stop) rather than letting a daemon whose whole purpose is a
/// trustworthy absence-of-change proof carry on with a broken one. `core` forbids I/O, so there is
/// no fallback log; the panic carries the orphaned correlation for triage.
///
/// `!panicking()` is the *only* reason the crash is conditional: a second panic while already
/// unwinding aborts the process and masks the first failure — strictly less diagnosable, the
/// opposite of the intent. The engine `step` is single-threaded, so an orphan observed only during
/// an unrelated unwind is moot — no further step runs, **because no `catch_unwind` wraps the engine
/// driver: a mid-`step` panic terminates the process** (the driver's `run`/`tick` carry a matching
/// no-`catch_unwind` note). The `catch_unwind`s that *do* exist (sensor prober workers, actuator
/// pipe/pool waiters, the bin's watcher / config-watcher / actuator supervision loops) are all on
/// threads that hold no `ProbeSlot`; this silence-in-unwind carve-out depends on that separation.
///
/// This is the **sole** explicit `Drop` in `core`/`engine`. Every enclosing carrier
/// (`PreFirePhase`, `PostFirePhase`, `PreFireBurst`, `PostFireBurst`, `ProfileState`,
/// `DescentState`, …) reaches this guard through auto drop-glue, *not* its own `impl Drop`: partial
/// moves out of those carriers (draining a sibling field at burst-end) depend on the absence of an
/// explicit `Drop` (E0509). An `impl Drop` anywhere up the tree would both break those moves and be
/// redundant — drop-glue already propagates this tripwire.
impl Drop for ProbeSlot {
    fn drop(&mut self) {
        let Some(correlation) = &self.inner else {
            return; // idle / disarmed — the sanctioned terminal state.
        };
        if std::thread::panicking() {
            // Already unwinding: a second panic aborts the process and masks the first failure —
            // stay silent instead.
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

impl ProbeSlot {
    /// An idle slot — no probe in flight.
    #[must_use]
    pub const fn empty() -> Self {
        Self { inner: None }
    }

    /// An armed slot carrying correlation `c`.
    #[must_use]
    pub const fn armed(c: ProbeCorrelation) -> Self {
        Self { inner: Some(c) }
    }

    /// Identity of the in-flight probe, or `None` if idle.
    #[must_use]
    pub const fn correlation(&self) -> Option<ProbeCorrelation> {
        self.inner
    }

    /// Arm an idle slot. Unconditional assert on a double-arm: a re-arm without an intervening
    /// [`Self::disarm`] would orphan the prior correlation, so this is a programming error that
    /// must surface in every build, not a silent overwrite.
    pub fn arm(&mut self, c: ProbeCorrelation) {
        assert!(
            self.inner.is_none(),
            "I5 violated: ProbeSlot armed while already armed \
             (prior correlation would be orphaned, its response stale-detected)",
        );
        self.inner = Some(c);
    }

    /// The single consume primitive: take the slot idle and return the prior correlation (`None` if
    /// it was already idle).
    #[must_use = "the disarmed probe correlation must be routed or explicitly discarded"]
    pub const fn disarm(&mut self) -> Option<ProbeCorrelation> {
        self.inner.take()
    }
}

#[cfg(test)]
mod tests {
    use super::ProbeSlot;
    use crate::ids::ProbeCorrelation;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    fn corr(n: u64) -> ProbeCorrelation {
        ProbeCorrelation::from(n)
    }

    /// `empty()` is idle: not armed, no correlation. An idle slot reaching drop is silent — the
    /// linear protocol only fires the tripwire on an *armed* drop.
    #[test]
    fn empty_is_idle() {
        let s = ProbeSlot::empty();
        assert_eq!(s.correlation(), None);
    }

    /// `armed(c)` reports armed and surfaces the correlation. The slot is disarmed before it drops
    /// — an armed slot reaching drop trips the linearity tripwire, and this test's point (the
    /// projection) is proven before the consume.
    #[test]
    fn armed_reports_correlation() {
        let mut s = ProbeSlot::armed(corr(7));
        assert_eq!(s.correlation(), Some(corr(7)));
        let _ = s.disarm();
    }

    /// `arm` on an idle slot makes it armed with the supplied correlation. Disarmed before drop to
    /// satisfy the linear protocol.
    #[test]
    fn arm_idle_slot_makes_it_armed() {
        let mut s = ProbeSlot::empty();
        s.arm(corr(11));
        assert_eq!(s.correlation(), Some(corr(11)));
        let _ = s.disarm();
    }

    /// `arm` on an already-armed slot panics unconditionally — a re-arm would orphan the prior
    /// correlation.
    #[test]
    #[should_panic(expected = "armed while already armed")]
    fn arm_panics_on_double_arm() {
        let mut s = ProbeSlot::armed(corr(1));
        s.arm(corr(2)); // panics
    }

    /// `disarm` on an armed slot returns the prior correlation and leaves the slot idle.
    #[test]
    fn disarm_returns_prior_and_idles() {
        let mut s = ProbeSlot::armed(corr(9));
        assert_eq!(s.disarm(), Some(corr(9)));
        assert_eq!(s.correlation(), None);
    }

    /// `disarm` on an idle slot is a clean `None`, slot stays idle.
    #[test]
    fn disarm_idle_slot_returns_none() {
        let mut s = ProbeSlot::empty();
        assert_eq!(s.disarm(), None);
        assert!(s.correlation().is_none());
    }

    /// A slot can be re-armed after a disarm (the consume-then-mint cycle descent advance relies
    /// on). The re-arm leaves the slot armed, so it is disarmed again before drop.
    #[test]
    fn rearm_after_disarm_is_allowed() {
        let mut s = ProbeSlot::armed(corr(1));
        assert_eq!(s.disarm(), Some(corr(1)));
        s.arm(corr(2));
        assert_eq!(s.correlation(), Some(corr(2)));
        let _ = s.disarm();
    }

    /// `Default` is `empty()`. Load-bearing: any `#[derive(Default)]` on a carrier holding a
    /// `ProbeSlot` must default it idle — an armed default would trip the Drop tripwire the moment
    /// that carrier is dropped.
    #[test]
    fn default_is_empty() {
        let s = ProbeSlot::default();
        assert_eq!(s, ProbeSlot::empty());
        assert!(s.correlation().is_none());
    }

    /// Dropping an **armed** slot (not during an unwind) trips the linearity tripwire: it panics,
    /// and the payload names both the "dropped while armed" class and the orphaned correlation.
    /// This is the *destroy*-edge dual of the double-`arm` assert — an armed slot reaching drop
    /// orphans its correlation just as a re-arm would.
    #[test]
    fn drop_while_armed_panics_with_orphaned_correlation() {
        let panicked = catch_unwind(AssertUnwindSafe(|| {
            let _s = ProbeSlot::armed(corr(42));
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

    /// Dropping a **disarmed** slot is silent, and so is dropping an `empty()` slot — the tripwire
    /// fires only on an armed drop, so the normal consume-then-drop path never crashes.
    #[test]
    fn drop_when_disarmed_or_empty_is_silent() {
        let quiet = catch_unwind(AssertUnwindSafe(|| {
            let mut s = ProbeSlot::armed(corr(8));
            let _ = s.disarm();
            // `s` drops here, idle → silent.
            let _e = ProbeSlot::empty();
            // `_e` drops here, never armed → silent.
        }));
        assert!(
            quiet.is_ok(),
            "an idle/empty slot reaching drop must not panic",
        );
    }

    /// The `!std::thread::panicking()` guard: an armed slot dropped *while a panic is already
    /// unwinding* must stay silent, so the original panic propagates intact rather than a
    /// double-panic aborting the process and masking it. The holder's armed slot drops as the
    /// `"primary"` unwind tears the frame down; the caught payload must still be `"primary"`,
    /// proving the ProbeSlot `Drop` observed `panicking() == true` and did not abort.
    #[test]
    fn drop_while_armed_during_unwind_does_not_double_panic() {
        struct Holder {
            _slot: ProbeSlot,
        }

        let payload = catch_unwind(AssertUnwindSafe(|| {
            let _h = Holder {
                _slot: ProbeSlot::armed(corr(99)),
            };
            panic!("primary");
            // Unwinding past here drops `_h` (and its armed slot) *while panicking* → the tripwire
            // must observe `panicking() == true` and stay silent.
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
}
