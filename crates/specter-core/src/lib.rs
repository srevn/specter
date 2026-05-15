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
mod fs_id;
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

pub use diag::{
    BurstHelper, ClaimKind, Diagnostic, LcaIntegritySource, PromoterClaimKind, SpliceFailureCause,
};
pub use diff::{Diff, EntryRef, Rename};
pub use effect::{
    CommandResolved, DedupKey, Effect, EffectCommon, EffectOutcome, EffectTarget, Termination,
};
pub use fs_id::FsIdentity;
pub use ids::{CorrelationId, ProbeCorrelation, ProfileId, PromoterId, ResourceId, SubId, TimerId};
pub use input::{FsEvent, Input, OverflowScope, WatchRegistryDiff};
pub use op::{
    ProbeOp, ProbeOutcome, ProbeOwner, ProbeRequest, ProbeResponse, WatchFailure, WatchOp,
};
pub use output::{SortedEffects, StepOutput, StepOutputParts};
pub use pattern::{PatternComponent, PatternError, PatternSpec};
pub use profile::{
    ActiveBurst, AnchorClaim, BurstFinish, BurstIntent, DescentRemaining, DescentState,
    DetachLifecycle, PostFireBurst, PostFirePhase, PreFireBurst, PreFirePhase, Profile, ProfileMap,
    ProfileState, ProfileStateDiscriminant, ReapTrigger, TimerKind,
};
pub use program::{ActionProgram, ArgPart, ArgTemplate, ExecAction, Placeholder};
pub use promoter::{
    Promoter, PromoterAttachRequest, PromoterRegistry, PromoterRegistryDiff, PromoterState,
    ProxyState,
};
pub use resource::{ContribKey, Resource, ResourceKind, ResourceRole};
pub use scan_config::{ConfigError, GlobPattern, ProfileIdentity, ScanConfig, ScanConfigBuilder};
pub use snapshot::EntryKind;
pub use snapshot::tree::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, LeafEntry, SpliceResult, TreeSnapshot,
    diff_dir_pair, diff_tree, splice, subtree_at_dir,
};
pub use sub::{
    ClassSet, EffectScope, Sub, SubAttachAnchor, SubAttachRequest, SubParams, SubRegistry,
    SubRegistryDiff,
};
pub use time::{Clock, SystemClock};
pub use tree::{AttachPathError, FS_ROOT_SEGMENT, StaleIdError, Tree, TreePath};

#[cfg(feature = "testkit")]
pub mod testkit;
