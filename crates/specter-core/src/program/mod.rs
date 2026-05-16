//! Action program — CFG-shaped bytecode IR.
//!
//! An [`ActionProgram`] is a flat `Box<[ProgramOp]>` walked by a `u32`
//! cursor at the actuator. Each [`ProgramOp`] carries a [`SpawnBody`]
//! (Exec or Pipe) plus two outgoing edges ([`BranchTarget`]s) — the
//! dispatcher reads exactly the edge that matches the spawned-process
//! outcome. There is no implicit fall-through.
//!
//! Programs are constructed via [`ProgramBuilder`]: each op is emitted
//! with both edges pending, then patched. Builder invariants ensure
//! every `Continue` edge points forward and lands on an emitted op.
//!
//! ```ignore
//! use specter_core::program::{
//!     ProgramBuilder, BranchTarget, SpawnBody,
//! };
//!
//! let mut b = ProgramBuilder::new();
//! let h = b.emit(SpawnBody::Exec(/* exec */));
//! b.patch_on_ok(h, BranchTarget::Escape)?;
//! b.patch_on_failed(h, BranchTarget::Terminate)?;
//! let program = b.build()?;
//! ```

mod builder;
mod error;
mod exec;
mod op;

pub use builder::{Edge, OpHandle, ProgramBuilder};
pub use error::ProgramError;
pub use exec::{ArgPart, ArgTemplate, ExecAction, Placeholder};
pub use op::{BranchIndex, BranchTarget, MultiStage, ProgramOp, SpawnBody};

/// Lowered execution program — a CFG-shaped bytecode IR.
///
/// Built once at config validation, shared by-Arc across every emitted
/// [`crate::Effect`]. The actuator walks the op slice by `u32` cursor;
/// each op carries explicit `on_ok` / `on_failed` branch targets, so
/// dispatch after a process reaps is a single lookup on the outcome.
///
/// The op slice is `Box<[ProgramOp]>` — the program shape is fixed at
/// construction time, and the field is private so [`ProgramBuilder`]
/// is the sole construction path. [`ProgramOp`]'s own constructor is
/// `pub(super)` and [`BranchIndex`] is sealed, so the guarantee holds
/// at every level: every value of this type provably came from a
/// builder that validated forward-only edges and in-bounds Continue
/// targets; the dispatcher does not need to handle backward jumps or
/// out-of-bounds cursors.
///
/// Structural `Eq` propagates from [`ProgramOp`]: two programs with
/// byte-equal ops compare equal even when their Arc allocations differ.
/// Consumed by hot-reload diffing to suppress no-op replacements.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ActionProgram {
    pub(super) ops: Box<[ProgramOp]>,
}

impl ActionProgram {
    /// Borrow the op slice. Cursor-indexed by the actuator; iterated
    /// for read-only scans (e.g., diff-derived placeholder detection).
    #[must_use]
    pub fn ops(&self) -> &[ProgramOp] {
        &self.ops
    }

    /// `true` iff any op in the program references a diff-derived
    /// placeholder (see [`Placeholder::is_diff_derived`]). Linear scan;
    /// called once at `Sub` construction to derive `Sub.needs_diff`.
    #[must_use]
    pub fn references_diff_derived(&self) -> bool {
        self.ops.iter().any(ProgramOp::references_diff_derived)
    }
}

#[cfg(test)]
mod tests {
    use super::{ActionProgram, BranchIndex, BranchTarget, ProgramBuilder, ProgramOp, SpawnBody};
    use crate::program::exec::{ArgPart, ArgTemplate, ExecAction, Placeholder};

    fn build_one_op_program(body: SpawnBody) -> ActionProgram {
        let mut b = ProgramBuilder::new();
        let h = b.emit(body);
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        b.build().unwrap()
    }

    #[test]
    fn references_diff_derived_false_for_anchor_only_program() {
        let body = SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([
                ArgPart::literal("/bin/build"),
                ArgPart::Placeholder(Placeholder::Path),
            ])],
            None,
        ));
        let program = build_one_op_program(body);
        assert!(!program.references_diff_derived());
    }

    #[test]
    fn references_diff_derived_true_for_diff_placeholder_program() {
        for p in [
            Placeholder::Created,
            Placeholder::Deleted,
            Placeholder::Modified,
            Placeholder::RenamedFrom,
            Placeholder::RenamedTo,
        ] {
            let body = SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::Placeholder(p)])],
                None,
            ));
            let program = build_one_op_program(body);
            assert!(
                program.references_diff_derived(),
                "ActionProgram::references_diff_derived must propagate from {p:?}"
            );
        }
    }

    /// Structural `Eq` across the program — equal-shape programs from
    /// independent builders compare equal. Hot-reload no-op suppression
    /// depends on this.
    #[test]
    fn action_program_structural_equality() {
        let make = || {
            let mut b = ProgramBuilder::new();
            let h0 = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/true")])],
                None,
            )));
            let h1 = b.emit(SpawnBody::Exec(ExecAction::new(
                [ArgTemplate::new([ArgPart::literal("/bin/false")])],
                None,
            )));
            b.patch_on_ok(h0, BranchTarget::Continue(BranchIndex::new(1)))
                .unwrap();
            b.patch_on_failed(h0, BranchTarget::Terminate).unwrap();
            b.patch_on_ok(h1, BranchTarget::Escape).unwrap();
            b.patch_on_failed(h1, BranchTarget::Terminate).unwrap();
            b.build().unwrap()
        };

        let a = make();
        let b = make();
        assert_eq!(a, b);
    }

    /// `ProgramOp` is exposed as a pure-`Clone` value (the inner
    /// `Pipe(Arc<[ExecAction]>)` body forbids `Copy`). The plan banks
    /// on this — coalesced Effects share one Arc rather than duplicating
    /// stage vectors.
    #[test]
    fn program_op_is_clone_not_copy() {
        // The lack of `Copy` is enforced by the type system; this test
        // just exercises the `Clone` path so the trait bound stays
        // referenced and any accidental Copy-derive surfaces.
        let body = SpawnBody::Exec(ExecAction::new(
            [ArgTemplate::new([ArgPart::literal("/bin/true")])],
            None,
        ));
        let op = ProgramOp::new(body, BranchTarget::Escape, BranchTarget::Terminate);
        let cloned = op.clone();
        assert_eq!(op, cloned);
    }
}
