//! `specter-engine` — pure step machine.

// I1: zero I/O in engine. `forbid` is compile-time enforcement.
#![forbid(unsafe_code)]

mod burst;
mod claims;
mod coverage;
mod descent;
mod engine;
mod probe_channel;
mod reconcile;
mod refcounts;
mod stability;
mod timer;
mod transitions;

pub use coverage::{covers, nearest_covering_ancestor};
pub use engine::Engine;
pub use stability::StabilityIndex;
pub use timer::{TimerEntry, TimerHeap};

// Re-export `SubAttachRequest` from `core` for back-compat with sites
// that imported it from `specter_engine`.
pub use specter_core::SubAttachRequest;
