//! Monotonic-id counter. Engine-internal abstraction shared by every
//! correlation-token mint site — probe correlations
//! ([`specter_core::ProbeCorrelation`]), effect correlations
//! ([`specter_core::CorrelationId`]), and timer ids
//! ([`specter_core::TimerId`]). The three id types are declared by the
//! `monotonic_id!` macro in `specter_core::ids`; their shared
//! `From<u64>` impl is what this counter mints through.
//!
//! Counter saturation is treated as fatal. A 64-bit counter advanced one
//! tick per nanosecond saturates after ~580 years, so the assertion exists
//! to catch programming errors (a counter wired into an unbounded retry
//! loop, a fuzzer running for weeks) rather than realistic workload
//! exhaustion. Pausing the engine on saturation would be fictional
//! recovery — at the saturation boundary the counter's identity space is
//! already corrupted, so the only honest outcome is to panic and surface
//! the bug.

use std::any::type_name;
use std::marker::PhantomData;

/// Engine-resident monotonic id counter.
///
/// `T` is the typed wrapper the counter produces — [`specter_core::ProbeCorrelation`],
/// [`specter_core::CorrelationId`], or [`specter_core::TimerId`]. The
/// phantom marker pins the produced type at compile time: passing a
/// `MonotonicCounter<ProbeCorrelation>` where a
/// `MonotonicCounter<CorrelationId>` is expected is a type error,
/// closing the cross-space-confusion hazard the typed wrappers were
/// introduced to forbid.
///
/// `PhantomData<fn() -> T>` is the variance-correct marker for "produces
/// `T`, never owns `T`" — covariant in `T` and `Send + Sync` regardless
/// of `T`'s thread-safety. The choice keeps [`Engine`](crate::Engine)
/// auto-derivable as `Send + Sync` without bubbling a `T: Send + Sync`
/// bound up to the counter's owner.
pub(crate) struct MonotonicCounter<T> {
    value: u64,
    _marker: PhantomData<fn() -> T>,
}

impl<T> Default for MonotonicCounter<T> {
    fn default() -> Self {
        Self {
            value: 0,
            _marker: PhantomData,
        }
    }
}

impl<T> std::fmt::Debug for MonotonicCounter<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MonotonicCounter<{}>({})", type_name::<T>(), self.value)
    }
}

impl<T: From<u64>> MonotonicCounter<T> {
    /// Mint the next id. Bumps the counter monotonically and wraps the
    /// new value via `T::from(u64)`.
    ///
    /// **Panics** when the counter is already at [`u64::MAX`]. The
    /// assertion is unconditional (not `debug_assert!`) so the check
    /// survives release builds — silent saturation (via
    /// `saturating_add`) produces duplicate ids and corrupts every
    /// consumer that relies on per-counter uniqueness: stale-response
    /// detection (probe), lazy invalidation (timer), and actuator-side
    /// coalescing (effect).
    #[must_use]
    pub(crate) fn next(&mut self) -> T {
        assert!(
            self.value < u64::MAX,
            "MonotonicCounter<{}> saturated at u64::MAX; engine cannot mint \
             further ids without corrupting the id space",
            type_name::<T>(),
        );
        self.value += 1;
        T::from(self.value)
    }
}

#[cfg(test)]
impl<T> MonotonicCounter<T> {
    /// Read the current counter value without minting. Test-only — the
    /// `engine_default_constructible_has_empty_state` fixture asserts
    /// against this for the "fresh Engine starts at zero" contract.
    pub(crate) fn peek(&self) -> u64 {
        self.value
    }

    /// Set the counter to `value`. The next [`Self::next`] call returns
    /// `T::from(value + 1)`. Test-only — saturation tests use this to
    /// jump near [`u64::MAX`] without `u64::MAX - 1` wasted mints.
    pub(crate) fn prime(&mut self, value: u64) {
        self.value = value;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Local stand-in token, kept inside this module so the counter's
    /// unit tests don't drag in `specter-core`. Mirrors the shape of the
    /// real wrappers (tuple struct over `u64` with a `From<u64>` impl).
    #[derive(Debug, Eq, PartialEq)]
    struct StubId(u64);

    impl From<u64> for StubId {
        fn from(v: u64) -> Self {
            Self(v)
        }
    }

    #[test]
    fn default_counter_starts_at_zero() {
        let c: MonotonicCounter<StubId> = MonotonicCounter::default();
        assert_eq!(c.peek(), 0);
    }

    #[test]
    fn first_mint_returns_one() {
        let mut c: MonotonicCounter<StubId> = MonotonicCounter::default();
        assert_eq!(c.next(), StubId(1));
        assert_eq!(c.peek(), 1);
    }

    #[test]
    fn mints_advance_monotonically() {
        let mut c: MonotonicCounter<StubId> = MonotonicCounter::default();
        let a = c.next();
        let b = c.next();
        let d = c.next();
        assert_eq!(a, StubId(1));
        assert_eq!(b, StubId(2));
        assert_eq!(d, StubId(3));
    }

    #[test]
    fn last_mint_succeeds_at_u64_max() {
        // Boundary check: priming at u64::MAX - 1 lets exactly one
        // more mint succeed, returning u64::MAX. The NEXT call panics.
        let mut c: MonotonicCounter<StubId> = MonotonicCounter::default();
        c.prime(u64::MAX - 1);
        assert_eq!(c.next(), StubId(u64::MAX));
        assert_eq!(c.peek(), u64::MAX);
    }

    /// The whole point of the abstraction: release-runnable panic on
    /// saturation. Deliberately no `cfg_attr(not(debug_assertions),
    /// ignore)` gate — the `assert!` in `next()` fires unconditionally.
    #[test]
    #[should_panic(expected = "MonotonicCounter")]
    fn panics_when_already_at_u64_max() {
        let mut c: MonotonicCounter<StubId> = MonotonicCounter::default();
        c.prime(u64::MAX);
        let _ = c.next();
    }

    #[test]
    fn debug_carries_type_name_and_value() {
        let mut c: MonotonicCounter<StubId> = MonotonicCounter::default();
        let _ = c.next();
        let s = format!("{c:?}");
        assert!(s.contains("StubId"), "debug should mention type T: {s}");
        assert!(s.ends_with("(1)"), "debug should mention value: {s}");
    }
}
