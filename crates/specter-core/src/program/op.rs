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

use super::error::ProgramError;
use super::exec::ExecAction;
use crate::effect::EffectOutcome;
use std::sync::Arc;

/// One CFG node — a spawn body plus the two edges the dispatcher reads
/// on outcome.
///
/// Builder-only: `Self::new` is `pub(super)`, so a `ProgramOp` is
/// minted solely by [`super::ProgramBuilder::build`]. A bare op
/// outside a built [`super::ActionProgram`] is inert — nothing
/// consumes one — so exposing a public constructor would be a
/// dead-end affordance. Reads go through [`Self::target`] (semantic,
/// outcome-routed) or the structural [`Self::body`] / [`Self::on_ok`]
/// / [`Self::on_failed`] accessors.
///
/// Structural `Eq` propagates from [`SpawnBody`] and [`BranchTarget`];
/// consumed by `SubRegistryDiff` for hot-reload no-op suppression
/// (two `Arc<ActionProgram>`s with byte-equal ops compare equal even
/// when the Arc allocations differ).
///
/// ```compile_fail
/// use specter_core::program::{ProgramOp, SpawnBody, ExecAction, ArgTemplate, ArgPart, BranchTarget};
/// // must not compile: `ProgramOp` fields are private, no `pub` constructor
/// let _ = ProgramOp {
///     body: SpawnBody::Exec(ExecAction::new([ArgTemplate::new([ArgPart::literal("x")])], None)),
///     on_ok: BranchTarget::Escape,
///     on_failed: BranchTarget::Terminate,
/// };
/// ```
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProgramOp {
    body: SpawnBody,
    on_ok: BranchTarget,
    on_failed: BranchTarget,
}

impl ProgramOp {
    /// Assemble a node. Builder-only — see [`Self`] for why no public
    /// constructor exists.
    #[must_use]
    pub(super) const fn new(body: SpawnBody, on_ok: BranchTarget, on_failed: BranchTarget) -> Self {
        Self {
            body,
            on_ok,
            on_failed,
        }
    }

    /// The spawn body — single `Exec` or N-stage `Pipe`.
    #[must_use]
    pub const fn body(&self) -> &SpawnBody {
        &self.body
    }

    /// The `on_ok` edge — taken when the spawned process reaps `Ok`.
    #[must_use]
    pub const fn on_ok(&self) -> BranchTarget {
        self.on_ok
    }

    /// The `on_failed` edge — taken when the process reaps `Failed`.
    #[must_use]
    pub const fn on_failed(&self) -> BranchTarget {
        self.on_failed
    }

    /// Pick the edge that matches `outcome`.
    ///
    /// `Ok ⇒ on_ok`, `Failed ⇒ on_failed`. The exit-code / signal
    /// payload on `Failed` is irrelevant to routing — it propagates
    /// verbatim when the edge terminates the plan.
    #[must_use]
    pub const fn target(&self, outcome: &EffectOutcome) -> BranchTarget {
        match outcome {
            EffectOutcome::Ok => self.on_ok,
            EffectOutcome::Failed(_) => self.on_failed,
        }
    }

    /// `true` iff the body's argv references any diff-derived
    /// placeholder. `Pipe` ratchets if any stage does.
    #[must_use]
    pub fn references_diff_derived(&self) -> bool {
        match &self.body {
            SpawnBody::Exec(e) => e.references_diff_derived(),
            SpawnBody::Pipe(stages) => stages
                .stages()
                .iter()
                .any(ExecAction::references_diff_derived),
        }
    }
}

/// A pipe's stage list — **≥2 stages, guaranteed by construction**.
///
/// The payload of [`SpawnBody::Pipe`]. Wraps the shared
/// `Arc<[ExecAction]>` so coalesced Effects share one stages allocation
/// (the newtype is zero-cost over the `Arc`; [`Self::shared`] proves
/// it). The sole constructor [`Self::new`] enforces the arity the
/// actuator's pipe path assumes — stdout→stdin wiring and pipefail
/// aggregation are meaningless below two stages. This is the
/// spawn-body-shape analogue of the control-flow seal [`BranchIndex`]
/// and [`ProgramOp`] already apply: the type admits no bad value
/// because its constructor, not the caller, does.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MultiStage(Arc<[ExecAction]>);

impl MultiStage {
    /// Reify a validated stage list. `Err(ProgramError::DegeneratePipe)`
    /// on fewer than two stages.
    ///
    /// The config validator rejects 0/1-stage pipes upstream
    /// (`IssueKind::EmptyPipe` / `SingleStagePipe`) before an
    /// `Action::Pipe` is built, so reaching here with <2 is a
    /// lowering-hygiene bug — surfaced (not panicked) so it rides the
    /// contained `LoweringInternal` channel like every other
    /// [`ProgramError`].
    ///
    /// `pub` (not `pub(super)`): the sole producer is `specter-config`
    /// lowering, a different crate — the same crate-boundary rationale
    /// as [`ExecAction::new`]. ([`BranchIndex`] / [`ProgramOp`] are
    /// sealed `pub(super)` because the in-crate builder is their only
    /// producer; a pipe body is reified cross-crate.)
    pub fn new(stages: Arc<[ExecAction]>) -> Result<Self, ProgramError> {
        if stages.len() < 2 {
            return Err(ProgramError::DegeneratePipe {
                stages: stages.len(),
            });
        }
        Ok(Self(stages))
    }

    /// The stage slice — spawn order, stage 0's stdout feeds stage 1's
    /// stdin, etc. Length is ≥2 by construction.
    #[must_use]
    pub fn stages(&self) -> &[ExecAction] {
        &self.0
    }

    /// The shared backing `Arc`. Exposed so a consumer can prove two
    /// coalesced Effects share one stages allocation (no per-Effect
    /// re-clone) — the optimization this newtype exists to preserve.
    #[must_use]
    pub const fn shared(&self) -> &Arc<[ExecAction]> {
        &self.0
    }
}

/// Spawn shape — what the actuator actually launches.
///
/// `Exec` is one process; `Pipe` is N processes wired stdout→stdin
/// with pipefail-on aggregation (last non-zero exit, first observed
/// signal). A `Pipe` always has ≥2 stages: [`MultiStage`] is its sole
/// producer and rejects fewer, so a single-stage "pipe" is
/// unrepresentable — one command lowers to a top-level `Exec`, not a
/// `Pipe`.
///
/// `Pipe`'s [`MultiStage`] is a zero-cost wrapper over the shared
/// `Arc<[ExecAction]>`, so coalesced Effects still share one stages
/// allocation; the Arc travels from `lower_to_program` into every
/// emitted `Effect` without rewrapping.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum SpawnBody {
    Exec(ExecAction),
    Pipe(MultiStage),
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
///
/// ```compile_fail
/// use specter_core::program::{BranchTarget, BranchIndex};
/// // must not compile: `BranchIndex::new` is `pub(super)`
/// let _ = BranchTarget::Continue(BranchIndex::new(0));
/// ```
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
    use super::{BranchIndex, BranchTarget, MultiStage, ProgramOp, SpawnBody};
    use crate::effect::{EffectOutcome, Termination};
    use crate::program::ProgramError;
    use crate::program::exec::{ArgPart, ArgTemplate, ExecAction, Placeholder};
    use std::sync::Arc;

    fn exec_with(part: ArgPart) -> ExecAction {
        ExecAction::new([ArgTemplate::new([part])], None)
    }

    fn op_with_edges(body: SpawnBody, on_ok: BranchTarget, on_failed: BranchTarget) -> ProgramOp {
        ProgramOp::new(body, on_ok, on_failed)
    }

    fn exec_arc(literals: &[&str]) -> Arc<[ExecAction]> {
        Arc::from(
            literals
                .iter()
                .map(|s| exec_with(ArgPart::literal(*s)))
                .collect::<Vec<_>>(),
        )
    }

    /// `MultiStage::new` is the spawn-body-shape seal: a pipe with
    /// fewer than two stages is a lowering-hygiene bug, surfaced as
    /// `DegeneratePipe` (never panicked) with the rejected count.
    #[test]
    fn multi_stage_new_rejects_fewer_than_two_stages() {
        let zero: Arc<[ExecAction]> = Arc::from(Vec::<ExecAction>::new());
        assert_eq!(
            MultiStage::new(zero),
            Err(ProgramError::DegeneratePipe { stages: 0 }),
        );

        let one = exec_arc(&["/bin/solo"]);
        assert_eq!(
            MultiStage::new(one),
            Err(ProgramError::DegeneratePipe { stages: 1 }),
        );
    }

    /// ≥2 stages construct; `stages()` returns the slice verbatim with
    /// length preserved.
    #[test]
    fn multi_stage_new_accepts_two_or_more_stages() {
        let stages = exec_arc(&["/bin/a", "/bin/b"]);
        let ms = MultiStage::new(Arc::clone(&stages)).expect(">=2 stages is valid");
        assert_eq!(ms.stages().len(), 2);
        assert_eq!(ms.stages(), &stages[..]);

        let three = exec_arc(&["/bin/a", "/bin/b", "/bin/c"]);
        let ms3 = MultiStage::new(three).expect(">=3 stages is valid");
        assert_eq!(ms3.stages().len(), 3);
    }

    /// `shared()` exposes the *same* backing allocation — the zero-cost
    /// wrapping that is the entire reason `MultiStage` wraps the `Arc`
    /// rather than reshaping the body. Coalesced Effects depend on this:
    /// the stage vector is never re-cloned per Effect.
    #[test]
    fn multi_stage_shared_preserves_the_backing_arc() {
        let stages = exec_arc(&["/bin/a", "/bin/b"]);
        let ms = MultiStage::new(Arc::clone(&stages)).expect(">=2 stages is valid");
        assert!(
            Arc::ptr_eq(ms.shared(), &stages),
            "MultiStage must wrap the original Arc, not re-allocate",
        );
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
        let failed_exit = EffectOutcome::Failed(Termination::Exit(1));
        let failed_signal = EffectOutcome::Failed(Termination::Signal(15));
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
            SpawnBody::Pipe(MultiStage::new(plain_then_diff).expect("test pipe has >=2 stages")),
            BranchTarget::Escape,
            BranchTarget::Terminate,
        );
        assert!(op.references_diff_derived());

        let plain_only: Arc<[ExecAction]> =
            Arc::from(vec![plain.clone(), plain].into_boxed_slice());
        let op = op_with_edges(
            SpawnBody::Pipe(MultiStage::new(plain_only).expect("test pipe has >=2 stages")),
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
