//! Pure test fixtures.
//!
//! Gated behind the `testkit` feature so the module never leaks into
//! release builds. Inherits `unsafe_code = forbid` from `specter-core`
//! — `MockClock` and `MockSensor` are pure too.

pub mod clock;
pub mod sensor;

pub use clock::MockClock;
pub use sensor::MockSensor;
