//! `specter-engine` — pure step machine.

// I1: zero I/O in engine. `forbid` is compile-time enforcement.
#![forbid(unsafe_code)]

mod burst;
mod claims;
mod counter;
mod coverage;
mod descent;
mod engine;
mod path;
mod probe;
mod promoter;
mod promoter_claims;
mod reconcile;
mod refcounts;
mod timer;
mod transitions;

// `covers` is the only cross-crate coverage primitive. `nearest_covering_ancestor` is engine-internal
// (`pub(crate)` in `coverage`): it is the reconfirm query core, with no external consumer.
pub use coverage::covers;
pub use engine::Engine;
// `TimerHeap` itself is engine-internal (`pub(crate)`); only `TimerEntry` crosses the crate
// boundary — the bin layer reads its fields off the `Engine::pop_expired` return value.
pub use timer::TimerEntry;

#[cfg(feature = "testkit")]
pub mod testkit;
