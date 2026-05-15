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
//! probes for one carrier are unconstructable. [`ProbeSlot::arm`]
//! backstops it with an unconditional assert — re-arming a still-armed
//! slot would orphan the prior correlation (its response would then be
//! rejected as stale even though the engine asked for it), so crashing
//! loudly is the only correct outcome.
//!
//! `Tag` is `()` for carriers whose state variant is itself the routing
//! class; it carries a real key (e.g. a [`ResourceId`]) only where the
//! response handler needs a dispatch datum the variant does not encode.

use crate::ids::ProbeCorrelation;

/// At most one in-flight probe for one owner-state carrier.
///
/// A value type held *inside* the owning state variant and mutated in
/// place through a `&mut` to that variant. It is [`Copy`]: disarming a
/// copy leaves the original armed, so callers consume the slot where it
/// lives, never a snapshot of it.
///
/// `Tag` defaults to `()` — the variant is the routing class. A carrier
/// whose response handler needs a dispatch key the variant does not
/// supply parameterises the slot with that key's type (which must be
/// [`Copy`]).
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct ProbeSlot<Tag: Copy = ()> {
    inner: Option<(ProbeCorrelation, Tag)>,
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

    /// `true` iff a probe is in flight (the slot is armed).
    #[must_use]
    pub const fn is_armed(&self) -> bool {
        self.inner.is_some()
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

    fn corr(n: u64) -> ProbeCorrelation {
        ProbeCorrelation::from(n)
    }

    /// `empty()` is idle: not armed, no correlation, no tag.
    #[test]
    fn empty_is_idle() {
        let s: ProbeSlot = ProbeSlot::empty();
        assert!(!s.is_armed());
        assert_eq!(s.correlation(), None);
        assert_eq!(s.tag(), None);
    }

    /// `armed(c, ())` reports armed, surfaces the correlation, and the
    /// unit tag round-trips as `Some(())`.
    #[test]
    fn armed_unit_tag_reports_correlation() {
        let s: ProbeSlot = ProbeSlot::armed(corr(7), ());
        assert!(s.is_armed());
        assert_eq!(s.correlation(), Some(corr(7)));
        assert_eq!(s.tag(), Some(()));
    }

    /// A non-unit `Tag` round-trips verbatim through `tag()`.
    #[test]
    fn armed_carries_non_unit_tag() {
        let target = ResourceId::default();
        let s: ProbeSlot<ResourceId> = ProbeSlot::armed(corr(3), target);
        assert!(s.is_armed());
        assert_eq!(s.correlation(), Some(corr(3)));
        assert_eq!(s.tag(), Some(target));
    }

    /// `arm` on an idle slot makes it armed with the supplied values.
    #[test]
    fn arm_idle_slot_makes_it_armed() {
        let mut s: ProbeSlot = ProbeSlot::empty();
        s.arm(corr(11), ());
        assert!(s.is_armed());
        assert_eq!(s.correlation(), Some(corr(11)));
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
        assert!(!s.is_armed());
        assert_eq!(s.correlation(), None);
        assert_eq!(s.tag(), None);
    }

    /// `disarm` on an idle slot is a clean `None`, slot stays idle.
    #[test]
    fn disarm_idle_slot_returns_none() {
        let mut s: ProbeSlot = ProbeSlot::empty();
        assert_eq!(s.disarm(), None);
        assert!(!s.is_armed());
    }

    /// A slot can be re-armed after a disarm (the consume-then-mint
    /// cycle descent advance relies on).
    #[test]
    fn rearm_after_disarm_is_allowed() {
        let mut s: ProbeSlot = ProbeSlot::armed(corr(1), ());
        assert_eq!(s.disarm(), Some(corr(1)));
        s.arm(corr(2), ());
        assert_eq!(s.correlation(), Some(corr(2)));
    }

    /// `Default` is `empty()` and does not require `Tag: Default` — a
    /// `Copy` tag with no `Default` impl still yields an idle slot.
    #[test]
    fn default_is_empty() {
        let s: ProbeSlot = ProbeSlot::default();
        assert_eq!(s, ProbeSlot::empty());
        assert!(!s.is_armed());

        let r: ProbeSlot<ResourceId> = ProbeSlot::default();
        assert!(!r.is_armed());
    }

    /// `ProbeSlot` is `Copy`: disarming a copy must not disarm the
    /// original. Pins the value-semantics so a reader knows the slot is
    /// consumed where it lives, never via a snapshot.
    #[test]
    fn disarm_on_copy_leaves_original_armed() {
        let original: ProbeSlot = ProbeSlot::armed(corr(5), ());
        let mut copy = original;
        assert_eq!(copy.disarm(), Some(corr(5)));
        assert!(!copy.is_armed());
        assert!(
            original.is_armed(),
            "Copy semantics: the original slot is independent of the copy",
        );
        assert_eq!(original.correlation(), Some(corr(5)));
    }
}
