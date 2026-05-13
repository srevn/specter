//! Pure test fixtures.
//!
//! Gated behind the `testkit` feature so the module never leaks into
//! release builds. Inherits `unsafe_code = forbid` from `specter-core`
//! — `MockClock` and `MockSensor` are pure too.

pub mod clock;
pub mod diagnostics;
pub mod program;
pub mod sensor;

pub use clock::MockClock;
pub use diagnostics::{first_attached_promoter, first_attached_sub};
pub use program::{predicate_then_program, single_exec_program};
pub use sensor::MockSensor;
