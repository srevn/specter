//! Deterministic time source for tests.

use crate::time::Clock;
use std::sync::Mutex;
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct MockClock {
    state: Mutex<Instant>,
}

impl MockClock {
    #[must_use]
    pub const fn new(start: Instant) -> Self {
        Self {
            state: Mutex::new(start),
        }
    }

    /// Convenience constructor anchored at the host's wall-clock `now`.
    #[must_use]
    pub fn at_zero() -> Self {
        Self::new(Instant::now())
    }

    pub fn advance(&self, by: Duration) {
        let mut t = self.state.lock().expect("MockClock mutex poisoned");
        *t += by;
    }

    pub fn set(&self, when: Instant) {
        let mut t = self.state.lock().expect("MockClock mutex poisoned");
        *t = when;
    }
}

impl Clock for MockClock {
    fn now(&self) -> Instant {
        *self.state.lock().expect("MockClock mutex poisoned")
    }
}

#[cfg(test)]
mod tests {
    use super::MockClock;
    use crate::time::Clock;
    use std::time::{Duration, Instant};

    #[test]
    fn advance_increments_now() {
        let start = Instant::now();
        let clock = MockClock::new(start);
        assert_eq!(clock.now(), start);
        clock.advance(Duration::from_millis(50));
        assert_eq!(clock.now(), start + Duration::from_millis(50));
    }

    #[test]
    fn set_replaces_now() {
        let start = Instant::now();
        let clock = MockClock::new(start);
        let later = start + Duration::from_mins(1);
        clock.set(later);
        assert_eq!(clock.now(), later);
    }
}
