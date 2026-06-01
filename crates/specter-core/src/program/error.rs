//! Shared construction-error taxonomy for the `program` module.
//!
//! [`ProgramError`] is the failure surface for **program construction**,
//! raised by the two sealed constructors that shape an
//! [`super::ActionProgram`]:
//!
//! - [`super::ProgramBuilder`] — op-graph hygiene (empty program;
//!   unpatched, backward, or out-of-bounds edges; a stale op handle).
//! - [`super::MultiStage::new`] — spawn-body shape (a `Pipe` reified
//!   with fewer than two stages).
//!
//! It lives in its own module — not in `builder` — because it is owned
//! by neither sibling constructor: both raise it and one external
//! mapper (`specter-config` lowering) consumes it. Every variant is a
//! *construction-hygiene* bug: a correct lowering pass never produces
//! one from valid input. The validator captures them as load errors
//! rather than letting them panic the daemon, so an internal lowering
//! defect degrades to a contained config-load failure.
//!
//! The "program exceeds `u32::MAX` ops" case is deliberately *not*
//! representable here: [`super::ProgramBuilder::emit`] enforces
//! `pending.len() <= u32::MAX` as a precondition, panicking otherwise.
//! Such a program is physically impossible to load (~128 GiB of
//! builder state); treating it as a precondition failure rather than a
//! recoverable error keeps this surface clean.

use super::builder::Edge;
use std::fmt;

/// Failure modes for [`super::ActionProgram`] construction.
///
/// Raised by [`super::ProgramBuilder`] (op-graph edges and handles) and
/// [`super::MultiStage::new`] (pipe arity). [`Self::EmptyProgram`] is
/// the only variant a non-lowering caller can trigger (calling
/// [`super::ProgramBuilder::build`] on a fresh builder); every other
/// variant signals a lowering-hygiene bug. See the module docs for why
/// the "program too large" case is absent.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ProgramError {
    /// [`super::ProgramBuilder::build`] called on a builder with no
    /// emits.
    EmptyProgram,
    /// An op was emitted but `edge` was never patched. The `op_index`
    /// is the position of the offending op in emission order.
    UnpatchedEdge { op_index: u32, edge: Edge },
    /// A `Continue(target)` patch points at or before the origin op.
    /// `origin` is the op being patched; `target` is the requested
    /// target index.
    BackwardEdge { origin: u32, target: u32 },
    /// A `Continue(target)` patch points past the emitted region (and,
    /// at build time, past the final op count). `len` is the bound at
    /// the time the error was raised — current pending length for
    /// patch-time rejections, final op count for build-time rejections.
    OutOfBoundsEdge { origin: u32, target: u32, len: u32 },
    /// [`super::ProgramBuilder::patch`] was called with an
    /// [`super::OpHandle`] that does not index this builder's pending
    /// ops. Only obtainable by reusing a handle minted by a *different*
    /// builder — a construction-hygiene bug. Surfaced (not panicked) so
    /// a lowering defect degrades to a contained config-load error
    /// rather than a daemon abort mid-reload. `handle` is the offending
    /// index; `len` is the builder's pending-op count.
    StaleHandle { handle: u32, len: u32 },
    /// [`super::MultiStage::new`] was called with fewer than two
    /// stages. The config validator forbids 0/1-stage pipes upstream
    /// (`IssueKind::EmptyPipe` / `SingleStagePipe`) before an
    /// `Action::Pipe` is built, so reaching the constructor with a
    /// degenerate list is a lowering-hygiene bug. `stages` is the
    /// rejected stage count — a diagnostic only, never an index, so no
    /// `u32` width invariant attaches.
    DegeneratePipe { stages: usize },
}

impl fmt::Display for ProgramError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyProgram => f.write_str("program has no ops"),
            Self::UnpatchedEdge { op_index, edge } => {
                write!(f, "op {op_index} has unpatched `{edge}` edge")
            }
            Self::BackwardEdge { origin, target } => {
                write!(f, "backward edge from op {origin} to op {target}")
            }
            Self::OutOfBoundsEdge {
                origin,
                target,
                len,
            } => write!(
                f,
                "out-of-bounds edge from op {origin} to op {target} (bound: {len})"
            ),
            Self::StaleHandle { handle, len } => {
                write!(f, "stale op handle {handle} (builder holds {len} ops)")
            }
            Self::DegeneratePipe { stages } => {
                write!(f, "pipe lowered with {stages} stage(s); >=2 required")
            }
        }
    }
}

impl std::error::Error for ProgramError {}
