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
mod probe_channel;
mod promoter;
mod promoter_claims;
mod reconcile;
mod refcounts;
mod stability;
mod timer;
mod transitions;

pub use coverage::{covers, nearest_covering_ancestor};
pub use engine::Engine;
// `TimerHeap` itself is engine-internal (`pub(crate)`); only `TimerEntry`
// crosses the crate boundary — the bin layer reads its fields off the
// `Engine::pop_expired` return value.
pub use timer::TimerEntry;
