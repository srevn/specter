//! `specter-core` — types and traits shared by the engine, the actors, and
//! the binary. I1: zero I/O, no syscalls, no time except `Instant` passed in.

// Stricter than the workspace default (`deny`). `forbid` cannot be lifted
// at any inner scope, making I1 a compile-time guarantee for this crate
// (and for `core::testkit`, which inherits).
#![forbid(unsafe_code)]

pub mod effect;
pub mod hash;
pub mod program;

mod diag;
mod diff;
mod ids;
mod input;
mod op;
mod output;
mod pattern;
mod profile;
mod promoter;
mod resource;
mod scan_config;
mod snapshot;
mod sub;
mod time;
mod tree;

pub use diag::{ClaimKind, Diagnostic, PromoterClaimKind};
pub use diff::{Diff, EntryRef, Rename};
pub use effect::{CommandResolved, CorrelationId, DedupKey, Effect, EffectOutcome};
pub use ids::{ProfileId, PromoterId, ResourceId, SubId, TimerId};
pub use input::{FsEvent, Input, OverflowScope, WatchRegistryDiff};
pub use op::{
    ProbeCorrelation, ProbeOp, ProbeOutcome, ProbeOwner, ProbeRequest, ProbeResponse, WatchFailure,
    WatchOp,
};
pub use output::StepOutput;
pub use pattern::{PatternComponent, PatternError, PatternSpec};
pub use profile::{
    AnchorClaim, Burst, BurstIntent, BurstPhase, DescentRemaining, DescentState, Profile,
    ProfileMap, ProfileState, TimerKind,
};
pub use program::{ActionProgram, ArgPart, ArgTemplate, ExecAction, Placeholder};
pub use promoter::{
    Promoter, PromoterAttachRequest, PromoterRegistry, PromoterRegistryDiff, PromoterState,
    ProxyState,
};
pub use resource::{ContribKey, Resource, ResourceKind, ResourceRole};
pub use scan_config::{
    ConfigError, GlobPattern, ScanConfig, ScanConfigBuilder, compute_config_hash,
};
pub use snapshot::EntryKind;
pub use snapshot::tree::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, LeafEntry, SpliceResult, TreeSnapshot,
    diff_dir_pair, diff_tree, splice, subtree_at_dir,
};
pub use sub::{ClassSet, EffectScope, Sub, SubAttachRequest, SubRegistry, SubRegistryDiff};
pub use time::{Clock, SystemClock};
pub use tree::Tree;

#[cfg(feature = "testkit")]
pub mod testkit;
