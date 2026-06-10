//! Pure test fixtures.
//!
//! Gated behind the `testkit` feature so the module never leaks into release builds. Inherits
//! `unsafe_code = forbid` from `specter-core` — the fixtures here are pure too.

pub mod builders;
pub mod diagnostics;
pub mod program;
pub mod sensor;

pub use builders::{
    anchor_ok, covered, dir_snap, dir_snap_nested, dirty_provenance, empty_program, enumerated,
    file_leaf, fresh_profile_id, fresh_profile_ids, leaf, proven, uncovered,
};
pub use diagnostics::{first_attached_promoter, first_attached_sub};
pub use program::{predicate_then_program, single_exec_program};
pub use sensor::MockSensor;
