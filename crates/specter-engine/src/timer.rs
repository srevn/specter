//! Timer heap. Tie-break on `(deadline, ProfileId, TimerId)`;
//! cancelled timers are not removed eagerly, only invalidated on pop.

use specter_core::{ProfileId, TimerId};
use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::time::Instant;

/// One pending timer.
///
/// `Ord` is derived in field-declaration order: `(deadline, profile, id)`.
/// That tuple matches the required tie-break, so the derived implementation
/// is correct — no manual `cmp` needed.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
pub struct TimerEntry {
    pub deadline: Instant,
    pub profile: ProfileId,
    pub id: TimerId,
}

/// Min-heap of pending timers.
///
/// Lazy invalidation: a cancelled timer stays in the heap until
/// [`pop_top`](Self::pop_top) returns it; the engine then validates
/// against the owning Profile's burst and silently drops stale entries.
/// `O(log n)` removal would force a slot map alongside the heap; the
/// spec's lazy form is cheaper for the typical "schedule, fire,
/// occasionally cancel" workload.
///
/// Sizing: at most two timers per Active Profile (settle + burst-deadline)
/// plus stale entries pending lazy collection.
#[derive(Debug, Default)]
pub struct TimerHeap {
    inner: BinaryHeap<Reverse<TimerEntry>>,
    /// Monotonic counter for `TimerId` minting. Saturates at `u64::MAX` —
    /// at one billion timers per second that boundary is ~580 years out,
    /// safely past any plausible v1 deployment.
    counter: u64,
}

impl TimerHeap {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Schedule a fresh timer. Returns the minted [`TimerId`]; the engine
    /// stores this on the owning Profile's burst so `pop_expired` can
    /// recognize live timers from cancelled ones.
    ///
    /// The minted id is unique within this heap's lifetime (modulo counter
    /// saturation, which is unreachable in any realistic deployment) —
    /// monotonic-counter minting via [`TimerId::from_counter`] sidesteps
    /// `slotmap`'s generation re-use semantics.
    pub fn schedule(&mut self, deadline: Instant, profile: ProfileId) -> TimerId {
        self.counter = self.counter.saturating_add(1);
        let id = TimerId::from_counter(self.counter);
        self.inner.push(Reverse(TimerEntry {
            deadline,
            profile,
            id,
        }));
        id
    }

    #[must_use]
    pub fn peek_top(&self) -> Option<&TimerEntry> {
        self.inner.peek().map(|r| &r.0)
    }

    pub fn pop_top(&mut self) -> Option<TimerEntry> {
        self.inner.pop().map(|r| r.0)
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;
    use slotmap::KeyData;
    use specter_core::ProfileId;
    use std::time::{Duration, Instant};

    fn pid(n: u64) -> ProfileId {
        ProfileId::from(KeyData::from_ffi(n))
    }

    #[test]
    fn empty_heap_peek_and_pop_return_none() {
        let mut h = TimerHeap::new();
        assert!(h.peek_top().is_none());
        assert!(h.pop_top().is_none());
        assert_eq!(h.len(), 0);
        assert!(h.is_empty());
    }

    #[test]
    fn schedule_returns_distinct_ids() {
        let mut h = TimerHeap::new();
        let now = Instant::now();
        let a = h.schedule(now, ProfileId::default());
        let b = h.schedule(now, ProfileId::default());
        let c = h.schedule(now, ProfileId::default());
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_ne!(a, c);
    }

    #[test]
    fn len_and_is_empty_track_schedules_and_pops() {
        let mut h = TimerHeap::new();
        let now = Instant::now();
        assert!(h.is_empty());
        h.schedule(now, ProfileId::default());
        h.schedule(now, ProfileId::default());
        assert_eq!(h.len(), 2);
        assert!(!h.is_empty());
        h.pop_top();
        assert_eq!(h.len(), 1);
        h.pop_top();
        assert!(h.is_empty());
    }

    #[test]
    fn monotonic_counter_persists_across_pops() {
        // Schedule → pop → schedule. The second-minted id must differ from
        // the first; the counter does not recycle on pop.
        let mut h = TimerHeap::new();
        let now = Instant::now();
        let a = h.schedule(now, ProfileId::default());
        h.pop_top();
        let b = h.schedule(now, ProfileId::default());
        assert_ne!(a, b);
    }

    #[test]
    fn peek_top_is_smallest_after_schedules() {
        let mut h = TimerHeap::new();
        let base = Instant::now();
        let later = base + Duration::from_millis(50);
        let earlier = base + Duration::from_millis(10);
        h.schedule(later, ProfileId::default());
        h.schedule(earlier, ProfileId::default());
        let top = h.peek_top().unwrap();
        assert_eq!(top.deadline, earlier);
    }

    #[test]
    fn pop_breaks_ties_by_profile_then_id() {
        // Same deadline, different profile: the smaller profile pops first
        // even when scheduled later — confirms profile-tier tie-break.
        let mut h = TimerHeap::new();
        let when = Instant::now();
        let p_high = pid(0xdead_beef);
        let p_low = pid(0x0001);
        let id_first = h.schedule(when, p_high);
        let id_second = h.schedule(when, p_low);
        let first = h.pop_top().unwrap();
        let second = h.pop_top().unwrap();
        assert_eq!(first.profile, p_low);
        assert_eq!(first.id, id_second);
        assert_eq!(second.profile, p_high);
        assert_eq!(second.id, id_first);
    }

    #[test]
    fn pop_breaks_ties_by_id_within_same_profile() {
        // Same deadline, same profile, two timers: the smaller-Ord TimerId
        // pops first. Don't assume the direction of TimerId's Ord (slotmap's
        // KeyData encoding is opaque); verify smaller-of-the-two-comes-first.
        let mut h = TimerHeap::new();
        let when = Instant::now();
        let p = pid(1);
        let id_a = h.schedule(when, p);
        let id_b = h.schedule(when, p);
        let first = h.pop_top().unwrap().id;
        let second = h.pop_top().unwrap().id;
        let (smaller, larger) = if id_a < id_b {
            (id_a, id_b)
        } else {
            (id_b, id_a)
        };
        assert_eq!(first, smaller);
        assert_eq!(second, larger);
    }

    proptest! {
        #[test]
        fn prop_pop_drains_in_non_decreasing_order(
            deltas in prop::collection::vec(0u64..1_000_000, 1..32),
        ) {
            let mut h = TimerHeap::new();
            let base = Instant::now();
            for d in &deltas {
                h.schedule(base + Duration::from_micros(*d), ProfileId::default());
            }
            let mut prev: Option<TimerEntry> = None;
            while let Some(top) = h.pop_top() {
                if let Some(p) = prev {
                    prop_assert!(p <= top, "out of order: {p:?} then {top:?}");
                }
                prev = Some(top);
            }
            prop_assert!(h.is_empty());
        }

        #[test]
        fn prop_schedule_returns_distinct_ids(n in 1usize..64) {
            let mut h = TimerHeap::new();
            let now = Instant::now();
            let mut ids = Vec::with_capacity(n);
            for _ in 0..n {
                ids.push(h.schedule(now, ProfileId::default()));
            }
            ids.sort();
            ids.dedup();
            prop_assert_eq!(ids.len(), n);
        }

        #[test]
        fn prop_peek_matches_pop(
            deltas in prop::collection::vec(0u64..1_000_000, 1..16),
        ) {
            let mut h = TimerHeap::new();
            let base = Instant::now();
            for d in &deltas {
                h.schedule(base + Duration::from_micros(*d), ProfileId::default());
            }
            while !h.is_empty() {
                let peeked = *h.peek_top().unwrap();
                let popped = h.pop_top().unwrap();
                prop_assert_eq!(peeked, popped);
            }
        }
    }
}
