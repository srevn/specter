//! Program ops — CFG nodes with two outgoing edges.
//!
//! A [`ProgramOp`] carries a [`SpawnBody`] (single `Exec` or N-stage
//! `Pipe`) plus a pair of [`BranchTarget`]s — `on_ok` and `on_failed` —
//! chosen by the dispatcher on the spawned-process outcome. Branch
//! targets are explicit and total: there is no implicit "fall through
//! to the next index" — the dispatcher reads exactly the edge that
//! matches the outcome.
//!
//! Three terminal shapes the dispatcher can land on:
//!
//! - [`BranchTarget::Continue`] — advance to an in-program index;
//!   builder enforces forward-only-and-in-bounds at patch time.
//! - [`BranchTarget::Terminate`] — early exit; the carried outcome
//!   propagates to the plan's `EffectComplete`. Lowering emits this on
//!   the `on_failed` edge of Exec/Pipe (stop-on-failure).
//! - [`BranchTarget::Escape`] — natural completion; terminate with
//!   `EffectOutcome::Ok` regardless of the carried outcome. This is
//!   the top-level escape — also the no-else fall-through of a
//!   conditional ("branch, not guard" outcome elision).

use super::exec::ExecAction;
use crate::effect::EffectOutcome;
use std::sync::Arc;

/// One CFG node — a spawn body plus the two edges the dispatcher reads
/// on outcome.
///
/// Structural `Eq` propagates from [`SpawnBody`] and [`BranchTarget`];
/// consumed by `SubRegistryDiff` for hot-reload no-op suppression
/// (two `Arc<ActionProgram>`s with byte-equal ops compare equal even
/// when the Arc allocations differ).
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramOp {
    pub body: SpawnBody,
    pub on_ok: BranchTarget,
    pub on_failed: BranchTarget,
}

impl ProgramOp {
    /// Pick the edge that matches `outcome`.
    ///
    /// `Ok ⇒ on_ok`, `Failed ⇒ on_failed`. The exit-code / signal
    /// payload on `Failed` is irrelevant to routing — it propagates
    /// verbatim when the edge terminates the plan.
    #[must_use]
    pub const fn target(&self, outcome: &EffectOutcome) -> BranchTarget {
        match outcome {
            EffectOutcome::Ok => self.on_ok,
            EffectOutcome::Failed { .. } => self.on_failed,
        }
    }

    /// `true` iff the body's argv references any diff-derived
    /// placeholder. `Pipe` ratchets if any stage does.
    #[must_use]
    pub fn references_diff_derived(&self) -> bool {
        match &self.body {
            SpawnBody::Exec(e) => e.references_diff_derived(),
            SpawnBody::Pipe(stages) => stages.iter().any(ExecAction::references_diff_derived),
        }
    }
}

/// Spawn shape — what the actuator actually launches.
///
/// `Exec` is one process; `Pipe` is N processes wired stdout→stdin
/// with pipefail-on aggregation (last non-zero exit, first observed
/// signal). Single-stage pipes are legal but pointless — lowering
/// produces them only from explicit `pipe = [...]` config.
///
/// `Pipe` carries `Arc<[ExecAction]>` so coalesced Effects share one
/// stages allocation; the Arc travels from `lower_to_program` into
/// every emitted `Effect` without rewrapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpawnBody {
    Exec(ExecAction),
    Pipe(Arc<[ExecAction]>),
}

/// Edge endpoint chosen on outcome.
///
/// `Continue(BranchIndex)` is the only in-program target; `BranchIndex`
/// can be constructed only by [`super::ProgramBuilder`], which enforces
/// the forward-only-and-in-bounds invariant at patch time. The two
/// no-op terminal variants — `Terminate` and `Escape` — can be minted
/// directly: they carry no payload that needs builder validation.
///
/// `Escape` is the typed encoding of "natural completion" — a
/// conditional whose `then` branch ran and there is no `else`, or
/// a top-level reach past the last op. Without it, the IR would need
/// an implicit past-end-substitutes-Ok mechanic in the dispatcher;
/// with it, every edge is total.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum BranchTarget {
    /// In-program target. `Continue(i)` ⇒ `i > origin_index` AND
    /// `i < ops.len()`; enforced by [`super::ProgramBuilder`] at patch
    /// time.
    Continue(BranchIndex),
    /// Early exit; the carried outcome propagates to `EffectComplete`.
    /// Emitted as `on_failed` for `Exec` / `Pipe` (stop-on-failure).
    Terminate,
    /// Natural completion; terminate with `EffectOutcome::Ok` regardless
    /// of the carried outcome. Top-level escape; a `Conditional`'s
    /// no-else fall-through ("branch, not guard" outcome elision).
    Escape,
}

/// Validated index into an [`super::ActionProgram`]'s `ops` slice.
///
/// The inner field is private; the only constructor — `new` — is
/// `pub(super)` so only `program::*` can mint values. External code
/// can construct [`BranchTarget::Terminate`]
/// and [`BranchTarget::Escape`] directly, but cannot construct
/// [`BranchTarget::Continue`] without going through the builder. This
/// is the type-level enforcement of the forward-only-and-in-bounds
/// invariant: every `Continue` in a built program was patched by
/// [`super::ProgramBuilder`], which validated the index.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct BranchIndex(u32);

impl BranchIndex {
    /// Mint a fresh index. Builder-only — see [`Self`] for the
    /// encapsulation rationale.
    #[must_use]
    pub(super) const fn new(i: u32) -> Self {
        Self(i)
    }

    /// Read the underlying op index as `u32`. Used by the dispatcher
    /// to advance the cursor.
    #[must_use]
    pub const fn get(self) -> u32 {
        self.0
    }
}

#[cfg(test)]
mod tests {
    use super::{BranchIndex, BranchTarget, ProgramOp, SpawnBody};
    use crate::effect::EffectOutcome;
    use crate::program::exec::{ArgPart, ArgTemplate, ExecAction, Placeholder};
    use std::sync::Arc;

    fn exec_with(part: ArgPart) -> ExecAction {
        ExecAction::new([ArgTemplate::new([part])])
    }

    fn op_with_edges(body: SpawnBody, on_ok: BranchTarget, on_failed: BranchTarget) -> ProgramOp {
        ProgramOp {
            body,
            on_ok,
            on_failed,
        }
    }

    #[test]
    fn target_routes_ok_to_on_ok() {
        let op = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::literal("/bin/true"))),
            BranchTarget::Continue(BranchIndex::new(3)),
            BranchTarget::Terminate,
        );
        assert_eq!(
            op.target(&EffectOutcome::Ok),
            BranchTarget::Continue(BranchIndex::new(3)),
        );
    }

    #[test]
    fn target_routes_failed_to_on_failed_regardless_of_payload() {
        let op = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::literal("/bin/true"))),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        // Routing is by-discriminant — exit code / signal don't change
        // the edge selection.
        let failed_exit = EffectOutcome::Failed {
            exit_code: Some(1),
            signal: None,
        };
        let failed_signal = EffectOutcome::Failed {
            exit_code: None,
            signal: Some(15),
        };
        assert_eq!(op.target(&failed_exit), BranchTarget::Terminate);
        assert_eq!(op.target(&failed_signal), BranchTarget::Terminate);
    }

    #[test]
    fn references_diff_derived_false_for_anchor_only_exec() {
        let op = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::Placeholder(Placeholder::Path))),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        assert!(!op.references_diff_derived());
    }

    #[test]
    fn references_diff_derived_true_for_diff_placeholder_exec() {
        let op = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::Placeholder(Placeholder::Created))),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        assert!(op.references_diff_derived());
    }

    /// `Pipe` ratchets `references_diff_derived` on any stage; pinning
    /// the OR-semantic prevents a future refactor from regressing to
    /// `&&` (which would silently miss diff-derived placeholders in
    /// later stages).
    #[test]
    fn references_diff_derived_pipe_or_across_stages() {
        let plain = exec_with(ArgPart::literal("/bin/cat"));
        let diff = exec_with(ArgPart::Placeholder(Placeholder::Modified));

        let plain_then_diff: Arc<[ExecAction]> =
            Arc::from(vec![plain.clone(), diff].into_boxed_slice());
        let op = op_with_edges(
            SpawnBody::Pipe(plain_then_diff),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        assert!(op.references_diff_derived());

        let plain_only: Arc<[ExecAction]> =
            Arc::from(vec![plain.clone(), plain].into_boxed_slice());
        let op = op_with_edges(
            SpawnBody::Pipe(plain_only),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        assert!(!op.references_diff_derived());
    }

    /// Empty `Pipe` stages slice — `iter().any()` returns `false`. The
    /// type allows it but the builder/lowering enforces non-empty pipes.
    /// Pinning the predicate's response to the degenerate shape catches
    /// any future refactor that would special-case empty pipes.
    #[test]
    fn references_diff_derived_pipe_false_for_empty_stages() {
        let empty: Arc<[ExecAction]> = Arc::from(Vec::<ExecAction>::new().into_boxed_slice());
        let op = op_with_edges(
            SpawnBody::Pipe(empty),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        assert!(!op.references_diff_derived());
    }

    /// `BranchIndex` round-trips through `new` / `get`.
    #[test]
    fn branch_index_round_trip() {
        for i in [0_u32, 1, 7, u32::MAX] {
            assert_eq!(BranchIndex::new(i).get(), i);
        }
    }

    /// Two equal-shape ops compare equal — structural `Eq` is preserved
    /// across `ProgramOp` / `SpawnBody` / `BranchTarget`. The
    /// `SubRegistryDiff` hot-reload no-op suppression relies on this.
    #[test]
    fn program_op_eq_is_structural() {
        let a = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::literal("/bin/true"))),
            BranchTarget::Continue(BranchIndex::new(2)),
            BranchTarget::Terminate,
        );
        let b = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::literal("/bin/true"))),
            BranchTarget::Continue(BranchIndex::new(2)),
            BranchTarget::Terminate,
        );
        assert_eq!(a, b);

        // Mutate one edge → no longer equal.
        let c = op_with_edges(
            SpawnBody::Exec(exec_with(ArgPart::literal("/bin/true"))),
            BranchTarget::Continue(BranchIndex::new(2)),
            BranchTarget::Escape,
        );
        assert_ne!(a, c);
    }
}
