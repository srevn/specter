//! Clock abstraction. The Engine never reads wall-clock time directly —
//! it consumes `now: Instant` from the bin loop. Everything else
//! that needs `Instant::now()` goes through [`Clock`] so tests can drive
//! time deterministically via `MockClock`.

use std::time::Instant;

pub trait Clock: Send + Sync {
    fn now(&self) -> Instant;
}

#[derive(Debug, Default, Clone, Copy)]
pub struct SystemClock;

impl Clock for SystemClock {
    fn now(&self) -> Instant {
        Instant::now()
    }
}
