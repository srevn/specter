//! Builder for [`super::ActionProgram`].
//!
//! Emit each op once, then patch its two outgoing edges. The builder
//! enforces:
//!
//! - **Bounded size.** Every emitted op gets a `u32` index. The
//!   builder upholds the post-condition `pending.len() <= u32::MAX`
//!   after every successful emit by refusing to push when
//!   `pending.len() == u32::MAX`. This is a precondition failure
//!   (panic, not Result) — a program with `u32::MAX + 1` ops is
//!   physically impossible to load (~128 GiB of in-memory builder
//!   state). Downstream casts of `pending.len()` to `u32` rely on
//!   this invariant.
//! - **Forward-only.** A `Continue(target)` may only point past the
//!   origin op (`target > origin`).
//! - **In-bounds.** A `Continue(target)` must land on an emitted op
//!   (`target < final_ops_len`). The patch-time check is the loose
//!   bound `target <= pending.len()` — the `==` case is the
//!   "future slot" produced by [`Self::continue_to_next`], promised
//!   to be filled by a follow-up emit. [`Self::build`] re-checks the
//!   strict bound and reports [`ProgramError::OutOfBoundsEdge`] if the
//!   promise was broken (no emit filled the deferred slot).
//! - **Total patching.** Every emitted op must have both edges patched
//!   before [`Self::build`]; an unpatched edge surfaces as
//!   [`ProgramError::UnpatchedEdge`] with the offending edge identity.
//!
//! The terminal targets [`super::BranchTarget::Terminate`] and
//! [`super::BranchTarget::Escape`] carry no payload and are always
//! valid — they get no bounds check.

use super::ActionProgram;
use super::error::ProgramError;
use super::op::{BranchIndex, BranchTarget, ProgramOp, SpawnBody};
use std::fmt;

/// Stepwise constructor for an [`ActionProgram`].
///
/// Build by interleaving [`Self::emit`] (push an op with both edges
/// pending) and [`Self::patch_on_ok`] / [`Self::patch_on_failed`] (set
/// each edge). Finalise with [`Self::build`].
///
/// The builder is the *sole* construction path for [`ActionProgram`]
/// values that contain [`BranchTarget::Continue`] edges:
/// [`BranchIndex`]'s constructor is sealed to `program::*`, so external
/// callers cannot mint a `Continue` target without routing through
/// [`Self::continue_to_next`].
#[derive(Debug, Default)]
pub struct ProgramBuilder {
    pending: Vec<PendingOp>,
}

/// Opaque handle to an emitted op.
///
/// Returned by [`ProgramBuilder::emit`] and consumed by the patch
/// methods. `Copy` so the same handle can be passed to both
/// [`ProgramBuilder::patch_on_ok`] and [`ProgramBuilder::patch_on_failed`]
/// without re-emitting.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub struct OpHandle(u32);

#[derive(Debug)]
struct PendingOp {
    body: SpawnBody,
    on_ok: Option<BranchTarget>,
    on_failed: Option<BranchTarget>,
}

/// Identifies which edge of a [`ProgramOp`] an error refers to. Carried
/// on [`ProgramError::UnpatchedEdge`].
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
pub enum Edge {
    OnOk,
    OnFailed,
}

impl fmt::Display for Edge {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OnOk => f.write_str("on_ok"),
            Self::OnFailed => f.write_str("on_failed"),
        }
    }
}

impl ProgramBuilder {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Push an op with both edges pending. The returned handle is the
    /// only way to later patch this op's edges; it is `Copy`, so both
    /// patches can share one handle.
    ///
    /// # Panics
    ///
    /// Panics if the builder already holds `u32::MAX` pending ops —
    /// the next index would not fit in `u32`, and the post-condition
    /// `pending.len() <= u32::MAX` would break. In practice this is
    /// unreachable (such a program would need 100+ GiB of memory to
    /// even hold); treating it as a precondition failure keeps the
    /// downstream casts of `pending.len()` to `u32` infallible.
    pub fn emit(&mut self, body: SpawnBody) -> OpHandle {
        // Two-step bound: the new index must fit in `u32`, AND the
        // resulting `pending.len()` after push must also fit (so the
        // post-condition `pending.len() <= u32::MAX` holds — downstream
        // casts of `pending.len()` rely on it). `checked_add(1)` over
        // the converted index covers both in one path.
        let index =
            u32::try_from(self.pending.len()).expect("program length cannot exceed u32::MAX");
        index
            .checked_add(1)
            .expect("emit would grow pending past u32::MAX");
        self.pending.push(PendingOp {
            body,
            on_ok: None,
            on_failed: None,
        });
        OpHandle(index)
    }

    /// Patch the `on_ok` edge of `h`. The internal `patch_target_check`
    /// helper enforces the patch-time invariants.
    pub fn patch_on_ok(&mut self, h: OpHandle, target: BranchTarget) -> Result<(), ProgramError> {
        self.patch(h, Edge::OnOk, target)
    }

    /// Patch the `on_failed` edge of `h`. The internal
    /// `patch_target_check` helper enforces the patch-time invariants.
    pub fn patch_on_failed(
        &mut self,
        h: OpHandle,
        target: BranchTarget,
    ) -> Result<(), ProgramError> {
        self.patch(h, Edge::OnFailed, target)
    }

    /// Patch the named `edge` of `h`. Equivalent to
    /// [`Self::patch_on_ok`] / [`Self::patch_on_failed`] but selects
    /// the edge by [`Edge`] value — for callers (the lowering pass)
    /// that carry the edge as a runtime token alongside the handle.
    ///
    /// Two independent preconditions:
    ///
    /// - **Origin in range.** `h` must index this builder's pending ops
    ///   (this method indexes `pending[h.0]`). A handle that does not is
    ///   only obtainable by reusing one minted by a different builder;
    ///   it is rejected as [`ProgramError::StaleHandle`]. This is
    ///   `patch`'s own precondition, totally checked here.
    /// - **Target valid.** Delegated to [`Self::patch_target_check`],
    ///   which validates `Continue` targets and is a no-op for the
    ///   payload-free `Terminate` / `Escape`.
    ///
    /// The origin check runs *first* and unconditionally: a stale
    /// handle is a stale handle regardless of the target, and the
    /// target check early-returns for `Terminate` / `Escape`, so
    /// deferring to it would leave `pending[h.0]` to panic on those
    /// targets (release-mode: silent out-of-bounds).
    pub fn patch(
        &mut self,
        h: OpHandle,
        edge: Edge,
        target: BranchTarget,
    ) -> Result<(), ProgramError> {
        let len = u32::try_from(self.pending.len())
            .expect("program length cannot exceed u32::MAX; emit() enforces this");
        if h.0 >= len {
            return Err(ProgramError::StaleHandle { handle: h.0, len });
        }
        self.patch_target_check(h.0, target)?;
        // Origin range-checked above; this slice index is in bounds.
        let slot = &mut self.pending[h.0 as usize];
        match edge {
            Edge::OnOk => slot.on_ok = Some(target),
            Edge::OnFailed => slot.on_failed = Some(target),
        }
        Ok(())
    }

    /// `BranchTarget::Continue(<next-emission-position>)`. Use to wire
    /// the current op's `on_ok` to the slot the next emit will fill.
    /// The returned target is provisional: if no follow-up emit lands,
    /// [`Self::build`] reports it as [`ProgramError::OutOfBoundsEdge`].
    #[must_use]
    pub fn continue_to_next(&self) -> BranchTarget {
        let next = u32::try_from(self.pending.len())
            .expect("program length cannot exceed u32::MAX; emit() enforces this");
        BranchTarget::Continue(BranchIndex::new(next))
    }

    /// Convert the pending sequence into a finalised [`ActionProgram`].
    ///
    /// Validates: at least one op ([`ProgramError::EmptyProgram`]);
    /// every op's edges are patched ([`ProgramError::UnpatchedEdge`]);
    /// every `Continue(target)` lands within the final op count
    /// ([`ProgramError::OutOfBoundsEdge`]).
    pub fn build(self) -> Result<ActionProgram, ProgramError> {
        if self.pending.is_empty() {
            return Err(ProgramError::EmptyProgram);
        }
        let final_len = u32::try_from(self.pending.len())
            .expect("program length cannot exceed u32::MAX; emit() enforces this");

        let mut ops: Vec<ProgramOp> = Vec::with_capacity(self.pending.len());
        for (idx, pending_op) in self.pending.into_iter().enumerate() {
            // `idx < pending.len()`, which fit `u32` above.
            let origin = u32::try_from(idx).expect("idx < pending.len() <= u32::MAX");
            let on_ok = pending_op.on_ok.ok_or(ProgramError::UnpatchedEdge {
                op_index: origin,
                edge: Edge::OnOk,
            })?;
            let on_failed = pending_op.on_failed.ok_or(ProgramError::UnpatchedEdge {
                op_index: origin,
                edge: Edge::OnFailed,
            })?;
            // Build-time strict in-bounds check — catches the
            // "continue_to_next never filled" case.
            check_final_in_bounds(origin, on_ok, final_len)?;
            check_final_in_bounds(origin, on_failed, final_len)?;
            ops.push(ProgramOp::new(pending_op.body, on_ok, on_failed));
        }
        Ok(ActionProgram {
            ops: ops.into_boxed_slice(),
        })
    }

    /// Patch-time invariant check for a `Continue` target: forward of
    /// origin AND within the loose bound `target <= pending.len()`. The
    /// upper-bound `==` case is the deferred future slot produced by
    /// [`Self::continue_to_next`] — accepted here, re-checked strictly
    /// at build.
    ///
    /// `Terminate` / `Escape` never carry an index, so the check is a
    /// no-op for them.
    fn patch_target_check(&self, origin: u32, target: BranchTarget) -> Result<(), ProgramError> {
        let BranchTarget::Continue(idx) = target else {
            return Ok(());
        };
        let target_idx = idx.get();
        if target_idx <= origin {
            return Err(ProgramError::BackwardEdge {
                origin,
                target: target_idx,
            });
        }
        // `pending.len()` is the loose bound — equal-to is the deferred
        // future slot; greater-than is unambiguously out of range.
        let pending_len = u32::try_from(self.pending.len())
            .expect("program length cannot exceed u32::MAX; emit() enforces this");
        if target_idx > pending_len {
            return Err(ProgramError::OutOfBoundsEdge {
                origin,
                target: target_idx,
                len: pending_len,
            });
        }
        Ok(())
    }
}

const fn check_final_in_bounds(
    origin: u32,
    target: BranchTarget,
    final_len: u32,
) -> Result<(), ProgramError> {
    let BranchTarget::Continue(idx) = target else {
        return Ok(());
    };
    let target_idx = idx.get();
    if target_idx >= final_len {
        return Err(ProgramError::OutOfBoundsEdge {
            origin,
            target: target_idx,
            len: final_len,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Edge, OpHandle, ProgramBuilder, ProgramError};
    use crate::program::exec::{ArgPart, ArgTemplate, ExecAction};
    use crate::program::op::{BranchIndex, BranchTarget, SpawnBody};

    fn exec_body() -> SpawnBody {
        SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal("/bin/true")])],
            None,
        ))
    }

    /// Empty builder → `EmptyProgram` on build.
    #[test]
    fn build_empty_returns_empty_program() {
        let b = ProgramBuilder::new();
        assert!(matches!(b.build(), Err(ProgramError::EmptyProgram)));
    }

    /// `continue_to_next` returns `Continue(pending.len())` — the slot
    /// the next `emit` will fill. After one emit, the next slot is 1.
    #[test]
    fn continue_to_next_returns_pending_emission_position() {
        let mut b = ProgramBuilder::new();
        assert_eq!(
            b.continue_to_next(),
            BranchTarget::Continue(BranchIndex::new(0))
        );
        let _h = b.emit(exec_body());
        assert_eq!(
            b.continue_to_next(),
            BranchTarget::Continue(BranchIndex::new(1))
        );
        let _h2 = b.emit(exec_body());
        assert_eq!(
            b.continue_to_next(),
            BranchTarget::Continue(BranchIndex::new(2))
        );
    }

    /// Forward-only-in-bounds patches succeed and build produces a
    /// program of matching length with edges that match the patches.
    #[test]
    fn forward_in_bounds_patches_succeed() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        let h1 = b.emit(exec_body());

        // h0.on_ok → h1 (forward, in-bounds).
        b.patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(1)))
            .expect("forward in-bounds Continue should patch");
        b.patch_on_failed(h0, BranchTarget::Terminate)
            .expect("Terminate target needs no bounds check");

        b.patch_on_ok(h1, BranchTarget::Escape)
            .expect("Escape target needs no bounds check");
        b.patch_on_failed(h1, BranchTarget::Terminate)
            .expect("Terminate target needs no bounds check");

        let program = b.build().expect("all edges patched, all targets in bounds");
        assert_eq!(program.ops.len(), 2);
        assert_eq!(
            program.ops[0].on_ok(),
            BranchTarget::Continue(BranchIndex::new(1))
        );
        assert_eq!(program.ops[0].on_failed(), BranchTarget::Terminate);
        assert_eq!(program.ops[1].on_ok(), BranchTarget::Escape);
        assert_eq!(program.ops[1].on_failed(), BranchTarget::Terminate);
    }

    /// `patch_on_ok` with a backward `Continue` target → `BackwardEdge`.
    /// `target == origin` is also backward (self-loop is not forward).
    #[test]
    fn patch_on_ok_backward_target_returns_backward_edge() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        let _h1 = b.emit(exec_body());

        // target == origin: self-loop, classified as backward.
        let err = b
            .patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(0)))
            .expect_err("self-loop must be rejected");
        assert_eq!(
            err,
            ProgramError::BackwardEdge {
                origin: 0,
                target: 0
            }
        );

        // target < origin: strictly backward.
        let h2 = b.emit(exec_body());
        let err = b
            .patch_on_ok(h2, BranchTarget::Continue(BranchIndex::new(1)))
            .expect_err("backward edge must be rejected");
        assert_eq!(
            err,
            ProgramError::BackwardEdge {
                origin: 2,
                target: 1
            }
        );
    }

    /// `patch_on_failed` validates the same way as `patch_on_ok` — both
    /// route through `patch_target_check`. Pinning this ensures
    /// drift between the two paths is caught.
    #[test]
    fn patch_on_failed_backward_target_returns_backward_edge() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());

        let err = b
            .patch_on_failed(h0, BranchTarget::Continue(BranchIndex::new(0)))
            .expect_err("on_failed self-loop must be rejected");
        assert_eq!(
            err,
            ProgramError::BackwardEdge {
                origin: 0,
                target: 0
            }
        );
    }

    /// `patch_on_ok` with a target past `pending.len() + 0` (the
    /// loose patch-time bound — `==` is the deferred slot) →
    /// `OutOfBoundsEdge`.
    #[test]
    fn patch_on_ok_target_past_pending_returns_out_of_bounds() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body()); // pending.len() == 1

        // target == 2, pending.len() == 1 → beyond the deferred slot.
        let err = b
            .patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(2)))
            .expect_err("target beyond pending.len() must be rejected");
        assert_eq!(
            err,
            ProgramError::OutOfBoundsEdge {
                origin: 0,
                target: 2,
                len: 1,
            }
        );
    }

    /// At patch time, `target == pending.len()` is the *deferred future
    /// slot* — accepted by the patch. If no follow-up emit fills it,
    /// `build` catches the unfilled promise via `OutOfBoundsEdge`.
    /// This test pins both sides of the contract.
    #[test]
    fn patch_to_pending_len_accepted_then_caught_at_build_if_unfilled() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body()); // pending.len() == 1

        // Patch to the deferred slot (target == 1, pending.len() == 1).
        b.patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(1)))
            .expect("deferred future-slot is accepted at patch time");
        b.patch_on_failed(h0, BranchTarget::Terminate)
            .expect("Terminate target is unconditional");

        // No follow-up emit. Build catches the unfilled promise.
        let err = b
            .build()
            .expect_err("unfilled deferred slot must be caught at build");
        assert_eq!(
            err,
            ProgramError::OutOfBoundsEdge {
                origin: 0,
                target: 1,
                len: 1,
            }
        );
    }

    /// When the deferred slot IS filled by a follow-up emit, the
    /// program builds successfully.
    #[test]
    fn deferred_slot_filled_builds_successfully() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        // Defer h0.on_ok to the next slot.
        let after = b.continue_to_next();
        b.patch_on_ok(h0, after)
            .expect("deferred slot accepted at patch time");
        b.patch_on_failed(h0, BranchTarget::Terminate)
            .expect("Terminate is unconditional");

        let h1 = b.emit(exec_body());
        b.patch_on_ok(h1, BranchTarget::Escape).expect("Escape ok");
        b.patch_on_failed(h1, BranchTarget::Terminate)
            .expect("Terminate ok");

        let program = b.build().expect("deferred slot filled by follow-up emit");
        assert_eq!(program.ops.len(), 2);
        assert_eq!(
            program.ops[0].on_ok(),
            BranchTarget::Continue(BranchIndex::new(1))
        );
    }

    /// An emitted op with one (or both) edge unpatched → `UnpatchedEdge`
    /// at build, with the edge identity in the error.
    #[test]
    fn unpatched_on_ok_at_build_reports_edge_identity() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        b.patch_on_failed(h0, BranchTarget::Terminate)
            .expect("Terminate ok");
        // on_ok is unpatched.
        let err = b.build().expect_err("unpatched on_ok must be caught");
        assert_eq!(
            err,
            ProgramError::UnpatchedEdge {
                op_index: 0,
                edge: Edge::OnOk,
            }
        );
    }

    #[test]
    fn unpatched_on_failed_at_build_reports_edge_identity() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        b.patch_on_ok(h0, BranchTarget::Escape).expect("Escape ok");
        // on_failed is unpatched.
        let err = b.build().expect_err("unpatched on_failed must be caught");
        assert_eq!(
            err,
            ProgramError::UnpatchedEdge {
                op_index: 0,
                edge: Edge::OnFailed,
            }
        );
    }

    /// When both edges are unpatched, `on_ok` is reported first
    /// (deterministic — UnpatchedEdge enumeration order is by edge
    /// then op).
    #[test]
    fn build_reports_first_unpatched_edge_deterministically() {
        let mut b = ProgramBuilder::new();
        let _h = b.emit(exec_body());
        let err = b.build().expect_err("both edges unpatched");
        // The implementation checks on_ok first within each op.
        assert_eq!(
            err,
            ProgramError::UnpatchedEdge {
                op_index: 0,
                edge: Edge::OnOk,
            }
        );
    }

    /// The earliest-op rule: with two unpatched ops, op 0 surfaces
    /// first.
    #[test]
    fn build_reports_earliest_unpatched_op_first() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        let _h1 = b.emit(exec_body());
        // Patch h0 fully, leave h1 unpatched.
        b.patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(1)))
            .unwrap();
        b.patch_on_failed(h0, BranchTarget::Terminate).unwrap();
        let err = b.build().expect_err("h1 unpatched");
        assert_eq!(
            err,
            ProgramError::UnpatchedEdge {
                op_index: 1,
                edge: Edge::OnOk,
            }
        );
    }

    /// `Terminate` and `Escape` skip every bounds check — they carry
    /// no payload that needs validation.
    #[test]
    fn terminate_and_escape_targets_unconditionally_accepted() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        b.patch_on_ok(h0, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h0, BranchTarget::Terminate).unwrap();
        let program = b.build().expect("Terminate/Escape never bounds-check");
        assert_eq!(program.ops.len(), 1);
        assert_eq!(program.ops[0].on_ok(), BranchTarget::Escape);
        assert_eq!(program.ops[0].on_failed(), BranchTarget::Terminate);
    }

    /// Re-patching an edge overwrites the previous value. Documented
    /// behaviour — lowering produces each patch exactly once by
    /// construction, but the API itself is permissive.
    #[test]
    fn re_patching_overwrites_previous_value() {
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(exec_body());
        let h1 = b.emit(exec_body());

        b.patch_on_ok(h0, BranchTarget::Terminate).unwrap();
        // Overwrite with a Continue target.
        b.patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(1)))
            .unwrap();
        b.patch_on_failed(h0, BranchTarget::Terminate).unwrap();

        // Finish h1 so build succeeds.
        b.patch_on_ok(h1, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h1, BranchTarget::Terminate).unwrap();

        let program = b.build().expect("re-patch resolves to final value");
        assert_eq!(
            program.ops[0].on_ok(),
            BranchTarget::Continue(BranchIndex::new(1))
        );
    }

    /// A handle minted by one builder used against a smaller one is
    /// *guaranteed* detected — the origin range-check is total and runs
    /// before the target is examined, so detection no longer depends on
    /// the target accidentally tripping a bounds check. Here the target
    /// is a forward `Continue` that, under the old target-first order,
    /// would have surfaced `OutOfBoundsEdge`; now the stale origin is
    /// caught first. The `Terminate` / `Escape` targets (where the
    /// target check is a no-op) are pinned by
    /// `stale_handle_on_terminate_or_escape_is_reported_not_panicked` —
    /// together they cover every `BranchTarget`.
    #[test]
    fn handle_with_out_of_range_index_rejected_at_patch() {
        // `OpHandle` is `Copy` and minted only by `emit`; reuse one from
        // a larger builder against a smaller one to forge the
        // cross-builder misuse without constructing a handle by hand.
        let mut larger = ProgramBuilder::new();
        let _ = larger.emit(exec_body());
        let _ = larger.emit(exec_body());
        let h_into_larger = larger.emit(exec_body()); // OpHandle(2)

        let mut smaller = ProgramBuilder::new();
        let _ = smaller.emit(exec_body()); // pending.len() == 1

        // h_into_larger has index 2; smaller holds 1 op. The origin
        // check fires first and rejects the handle as stale — the
        // forward `Continue(5)` target is never reached.
        let err = smaller
            .patch_on_ok(h_into_larger, BranchTarget::Continue(BranchIndex::new(5)))
            .expect_err("cross-builder handle must be rejected as stale");
        assert_eq!(err, ProgramError::StaleHandle { handle: 2, len: 1 });
    }

    /// A handle that does not index this builder's pending ops is the
    /// caller's hygiene bug (only obtainable by reusing a handle minted
    /// by a *different* builder). `patch` owns this precondition because
    /// it indexes `pending[h.0]`; `patch_target_check` validates only
    /// the TARGET and early-returns for `Terminate` / `Escape`, so
    /// without an explicit origin range-check those targets would
    /// panic-index a non-existent slot (release-mode: silent UB). Both
    /// terminal targets must surface `StaleHandle` instead of panicking.
    #[test]
    fn stale_handle_on_terminate_or_escape_is_reported_not_panicked() {
        let mut larger = ProgramBuilder::new();
        let _ = larger.emit(exec_body());
        let _ = larger.emit(exec_body());
        let stale = larger.emit(exec_body()); // OpHandle(2)

        let mut smaller = ProgramBuilder::new();
        let _ = smaller.emit(exec_body()); // pending.len() == 1

        // Terminate: `patch_target_check` is a no-op for it, so the
        // origin range-check is the only thing between this and a
        // panic-index of `pending[2]`.
        let err = smaller
            .patch_on_failed(stale, BranchTarget::Terminate)
            .expect_err("stale handle on Terminate must be reported, not panic");
        assert_eq!(err, ProgramError::StaleHandle { handle: 2, len: 1 });

        // Escape: the other no-op terminal target, same path.
        let err = smaller
            .patch_on_ok(stale, BranchTarget::Escape)
            .expect_err("stale handle on Escape must be reported, not panic");
        assert_eq!(err, ProgramError::StaleHandle { handle: 2, len: 1 });
    }

    /// `OpHandle` is `Copy` — the same handle can be used for both
    /// patches without re-emitting.
    #[test]
    fn op_handle_is_copy() {
        let mut b = ProgramBuilder::new();
        let h: OpHandle = b.emit(exec_body());
        let h_copy: OpHandle = h;
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h_copy, BranchTarget::Terminate).unwrap();
        let program = b.build().unwrap();
        assert_eq!(program.ops.len(), 1);
    }

    /// `Edge`'s Display formatting — surfaces in `UnpatchedEdge`
    /// rendering. Pinned to catch accidental rename.
    #[test]
    fn edge_display_strings() {
        assert_eq!(Edge::OnOk.to_string(), "on_ok");
        assert_eq!(Edge::OnFailed.to_string(), "on_failed");
    }
}
