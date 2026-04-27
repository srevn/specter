//! Generational slotmap key types for the three identity tiers plus
//! the engine's internal timer handle. All four are deterministic
//! generational handles — a stale id looks up to `None`.

use slotmap::KeyData;

slotmap::new_key_type! {
    pub struct ResourceId;
    pub struct ProfileId;
    pub struct SubId;
    pub struct TimerId;
}

impl TimerId {
    /// Mint a `TimerId` from a monotonic counter. The engine's `TimerHeap`
    /// uses this to produce deterministic, totally-ordered handles
    /// without backing each id with a `SlotMap` allocation: timers come and
    /// go far faster than the structural Resource/Profile/Sub tiers, and
    /// lazy invalidation means cancelled ids must merely be
    /// distinguishable, not reusable.
    ///
    /// The caller is responsible for monotonicity and uniqueness within one
    /// `TimerHeap`'s lifetime; `slotmap`'s generational re-use semantics are
    /// bypassed entirely.
    #[must_use]
    pub fn from_counter(counter: u64) -> Self {
        Self::from(KeyData::from_ffi(counter))
    }
}
