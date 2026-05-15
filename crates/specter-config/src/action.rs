//! Validation-only surface-syntax tree for `[[watch.actions]]`.
//!
//! The TOML `actions = [{ ... }, ...]` array deserialises into a flat
//! sequence of [`crate::raw::RawAction`] entries. Validation rewrites
//! each entry into one [`Action`] node (a discriminated union over the
//! variants the surface grammar supports); [`lower_to_program`] then
//! folds the tree into the engine/actuator-side [`ActionProgram`].
//!
//! # Why a separate AST
//!
//! The runtime sees only the lowered CFG-shaped op program — there is no
//! benefit to teaching the engine or actuator about pipes, conditionals,
//! or other surface-grammar shapes. Keeping the AST in `specter-config`
//! means future variants land here without bloating `specter-core`'s
//! public surface; the lowering pass is the single seam between the two
//! layers.
//!
//! # Lowering invariants
//!
//! Each op has explicit `on_ok` / `on_failed` edges. Lowering produces:
//!
//! - **`Exec` / `Pipe`** — `on_ok` continues to the next action's first
//!   op (or [`BranchTarget::Escape`] at scope tail); `on_failed` is
//!   [`BranchTarget::Terminate`] (stop-on-failure, outcome propagates).
//! - **`Conditional`** — the predicate (lowered as `SpawnBody::Exec`)'s
//!   edges depend on then/else presence:
//!     - `on_ok` → then-branch's first op when non-empty; otherwise the
//!       post-conditional slot (`after`).
//!     - `on_failed` → else-branch's first op when non-empty; otherwise
//!       `after`. When `after` resolves to [`BranchTarget::Escape`], the
//!       "branch, not guard" outcome elision is preserved: a Failed
//!       predicate with no else terminates Ok rather than propagating
//!       Failed.
//!
//!   Then-body and else-body tails are patched with the post-conditional
//!   `after` slot the same way sequence tails are.
//!
//! The [`ProgramBuilder`] enforces forward-only edges and in-bounds
//! Continue targets at patch time; backward jumps are unrepresentable
//! by construction (no `Loop`-like surface).

use specter_core::program::{
    ActionProgram, BranchTarget, Edge, ExecAction, OpHandle, ProgramBuilder, ProgramError,
    SpawnBody,
};
use std::sync::Arc;

/// Surface-syntax tree produced by validation. Lives in `specter-config`
/// because it's a validation artifact — it never reaches the engine or
/// actuator. After validation, lowering folds it into an
/// `Arc<ActionProgram>`; the tree is dropped at the function boundary.
///
/// `pub(crate)` because the tree never escapes this crate. Future
/// variants land here as additional arms; the runtime stays oblivious.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Action {
    /// Spawn a single process with this argv template.
    Exec(ExecAction),
    /// Spawn N processes wired stdout→stdin. `stages` is the spawn
    /// order — stage 0's stdout feeds stage 1's stdin, etc. Each
    /// stage carries its own [`ExecAction::timeout`]; the actuator's
    /// per-stage timer thread enforces them independently.
    ///
    /// `stages: Arc<[ExecAction]>` (not `Box<[…]>`) so lowering can
    /// `Arc::clone` directly into [`SpawnBody::Pipe`] without
    /// re-allocating the leaf vector. Validation guarantees
    /// `stages.len() >= 2` (single-stage pipes are rejected as
    /// [`crate::error::IssueKind::SingleStagePipe`] — use top-level
    /// `exec` instead).
    Pipe { stages: Arc<[ExecAction]> },
    /// Predicate + then-branch + optional else-branch.
    ///
    /// `when` runs first; on Ok the `then` body runs to completion (in
    /// order, stop-on-failure); on Failed (or spawn failure / resolver
    /// failure) the `otherwise` body runs (if `Some`), or the
    /// conditional is skipped entirely (if `None`). The predicate's own
    /// Failed outcome does NOT propagate to plan terminus — the
    /// conditional is a branch, not a guard.
    ///
    /// `then` / `otherwise` are `Box<[Action]>` because lowering walks
    /// them element-by-element; the slice doesn't need to survive past
    /// the tree's drop.
    Conditional {
        when: ExecAction,
        then: Box<[Self]>,
        /// TOML field name is `else` (Rust keyword); the Rust struct
        /// field on [`crate::raw::RawAction`] is `otherwise` via
        /// `#[serde(rename = "else")]`. `None` means no else-branch was
        /// supplied; empty `Some(_)` is normalised to `None` by
        /// validation so the lowering pass has one shape to handle.
        otherwise: Option<Box<[Self]>>,
    },
}

/// One unpatched edge from a lowered op. Returned by [`lower_action`]
/// and propagated up to the enclosing block, which patches each tail
/// with the slot where execution should continue after this body.
///
/// The `edge` discriminates the two flavours: a no-else conditional's
/// pred carries its tail on `on_failed` (Failed predicate falls
/// through); every other case has its tail on `on_ok`.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
struct Tail {
    handle: OpHandle,
    edge: Edge,
}

impl Tail {
    const fn on_ok(handle: OpHandle) -> Self {
        Self {
            handle,
            edge: Edge::OnOk,
        }
    }

    const fn on_failed(handle: OpHandle) -> Self {
        Self {
            handle,
            edge: Edge::OnFailed,
        }
    }
}

/// Walk a validated `[Action]` sequence and emit the equivalent
/// [`ActionProgram`]. The Arc is the same allocation that the engine's
/// `Sub.program` and every emitted `Effect.program` references — one
/// shared CFG-shaped op program per validated Sub.
///
/// Returns [`ProgramError`] on builder-hygiene violations (unreachable
/// from a correct lowering pass); the caller maps these to validation
/// issues so an internal bug surfaces as a config-load error rather
/// than a panic.
pub(crate) fn lower_to_program(tree: &[Action]) -> Result<Arc<ActionProgram>, ProgramError> {
    let mut b = ProgramBuilder::new();
    let top_tails = lower_block(tree, &mut b)?;
    // Top-level tails escape — natural completion terminates Ok.
    for tail in top_tails {
        b.patch(tail.handle, tail.edge, BranchTarget::Escape)?;
    }
    Ok(Arc::new(b.build()?))
}

/// Lower a block (a slice of [`Action`]s in execution order), returning
/// the tails of the block's last action that need to be patched with
/// "where execution continues after this block."
///
/// For each non-last action, the tails are patched inline to point at
/// the next action's first op slot — known at iteration end via
/// [`ProgramBuilder::continue_to_next`] (the next iteration's first
/// emit fills the deferred slot).
fn lower_block(actions: &[Action], b: &mut ProgramBuilder) -> Result<Vec<Tail>, ProgramError> {
    let n = actions.len();
    let mut returned_tails: Vec<Tail> = Vec::new();

    for (idx, action) in actions.iter().enumerate() {
        let is_last = idx + 1 == n;
        let action_tails = lower_action(action, b)?;

        if is_last {
            returned_tails.extend(action_tails);
        } else {
            // The next iteration's first emit lands at the slot
            // `continue_to_next` names right now — patch-time
            // deferred-slot promise (accepted as `target == pending.len()`,
            // filled by the upcoming emit).
            let next_slot = b.continue_to_next();
            for tail in action_tails {
                b.patch(tail.handle, tail.edge, next_slot)?;
            }
        }
    }

    Ok(returned_tails)
}

/// Lower one [`Action`], returning its tail handles.
///
/// `Exec` and `Pipe` produce one op with `on_failed = Terminate` and
/// `on_ok` unpatched (the single tail) via [`lower_spawn`].
///
/// `Conditional` produces the predicate op plus the lowered then- and
/// else-bodies. The predicate's edges are patched here for the
/// non-empty-body sides; the empty-body side becomes a tail (pred's
/// `on_ok` for empty-then, pred's `on_failed` for no-else). Body tails
/// bubble up to the caller for patching with the post-conditional slot.
fn lower_action(action: &Action, b: &mut ProgramBuilder) -> Result<Vec<Tail>, ProgramError> {
    match action {
        Action::Exec(e) => lower_spawn(SpawnBody::Exec(e.clone()), b),
        Action::Pipe { stages } => lower_spawn(SpawnBody::Pipe(Arc::clone(stages)), b),
        Action::Conditional {
            when,
            then,
            otherwise,
        } => {
            let pred = b.emit(SpawnBody::Exec(when.clone()));
            let mut tails: Vec<Tail> = Vec::new();

            // pred.on_ok: chain to then's first slot, or surface as a
            // tail for empty-then (the caller patches it with `after`).
            if then.is_empty() {
                tails.push(Tail::on_ok(pred));
            } else {
                let then_first = b.continue_to_next();
                b.patch_on_ok(pred, then_first)?;
                tails.extend(lower_block(then, b)?);
            }

            // pred.on_failed: chain to else's first slot when present and
            // non-empty; otherwise surface as a tail (no-else fall-through
            // → `after`, which preserves "branch, not guard" outcome
            // elision when `after` resolves to `Escape`).
            match otherwise {
                Some(body) if !body.is_empty() => {
                    let else_first = b.continue_to_next();
                    b.patch_on_failed(pred, else_first)?;
                    tails.extend(lower_block(body, b)?);
                }
                _ => {
                    tails.push(Tail::on_failed(pred));
                }
            }

            Ok(tails)
        }
    }
}

/// Lower a "spawn-and-stop-on-failure" body (Exec or Pipe) — emit one
/// op, patch `on_failed = Terminate`, surface `on_ok` as the tail.
/// Shared by [`Action::Exec`] and [`Action::Pipe`] since they differ
/// only in the [`SpawnBody`] discriminant.
fn lower_spawn(body: SpawnBody, b: &mut ProgramBuilder) -> Result<Vec<Tail>, ProgramError> {
    let h = b.emit(body);
    b.patch_on_failed(h, BranchTarget::Terminate)?;
    Ok(vec![Tail::on_ok(h)])
}

#[cfg(test)]
mod tests {
    use super::{Action, lower_to_program};
    use specter_core::program::{
        ActionProgram, ArgPart, ArgTemplate, BranchTarget, ExecAction, ProgramOp, SpawnBody,
    };
    use std::sync::Arc;

    fn exec_with_literal(literal: &str) -> ExecAction {
        ExecAction::new([ArgTemplate::new([ArgPart::literal(literal)])], None)
    }

    /// Build a `Continue(idx)` for assertion ergonomics. The
    /// `BranchIndex::new` constructor is sealed to `program::*`, so test
    /// assertions reach for the [`ProgramOp`]'s actual edge enum.
    fn ops(program: &Arc<ActionProgram>) -> &[ProgramOp] {
        program.ops()
    }

    fn assert_continue(target: BranchTarget, expected: u32) {
        match target {
            BranchTarget::Continue(idx) => assert_eq!(
                idx.get(),
                expected,
                "expected Continue({expected}), got Continue({})",
                idx.get(),
            ),
            other => panic!("expected Continue({expected}), got {other:?}"),
        }
    }

    /// Single Exec lowers to a one-op program. on_ok escapes; on_failed
    /// terminates (stop-on-failure).
    #[test]
    fn single_exec_lowers_to_one_op() {
        let tree = [Action::Exec(exec_with_literal("/bin/build"))];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 1);
        assert!(matches!(ops(&program)[0].body(), SpawnBody::Exec(_)));
        assert_eq!(ops(&program)[0].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[0].on_failed(), BranchTarget::Terminate);
    }

    /// Multiple Execs chain via `Continue` on_ok; the last one escapes.
    /// on_failed is Terminate at every step (stop-on-failure).
    #[test]
    fn multi_exec_chains_via_continue() {
        let tree = [
            Action::Exec(exec_with_literal("/bin/first")),
            Action::Exec(exec_with_literal("/bin/second")),
            Action::Exec(exec_with_literal("/bin/third")),
        ];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 3);

        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_continue(ops(&program)[1].on_ok(), 2);
        assert_eq!(ops(&program)[2].on_ok(), BranchTarget::Escape);

        for op in ops(&program) {
            assert_eq!(op.on_failed(), BranchTarget::Terminate);
        }
    }

    /// `Action::Pipe` lowers to a single `SpawnBody::Pipe` carrying the
    /// same `Arc` — no per-leaf re-allocation.
    #[test]
    fn pipe_lowers_to_one_op_sharing_arc() {
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            exec_with_literal("/bin/a"),
            exec_with_literal("/bin/b"),
        ]);
        let stages_for_assert = Arc::clone(&stages);
        let tree = [Action::Pipe { stages }];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 1);
        match ops(&program)[0].body() {
            SpawnBody::Pipe(s) => {
                assert_eq!(s.len(), 2);
                assert!(
                    Arc::ptr_eq(s, &stages_for_assert),
                    "lowering must Arc::clone the stages slice, not re-allocate",
                );
            }
            other @ SpawnBody::Exec(_) => panic!("expected SpawnBody::Pipe, got {other:?}"),
        }
        assert_eq!(ops(&program)[0].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[0].on_failed(), BranchTarget::Terminate);
    }

    /// Pipe and Exec mix freely; each lowers to one op; order preserved.
    #[test]
    fn pipe_and_exec_mixed_preserves_order() {
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            exec_with_literal("/bin/pipe-a"),
            exec_with_literal("/bin/pipe-b"),
        ]);
        let tree = [
            Action::Exec(exec_with_literal("/bin/pre")),
            Action::Pipe { stages },
            Action::Exec(exec_with_literal("/bin/post")),
        ];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 3);
        assert!(matches!(ops(&program)[0].body(), SpawnBody::Exec(_)));
        assert!(matches!(ops(&program)[1].body(), SpawnBody::Pipe(_)));
        assert!(matches!(ops(&program)[2].body(), SpawnBody::Exec(_)));

        // Each non-last op chains forward; the last escapes.
        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_continue(ops(&program)[1].on_ok(), 2);
        assert_eq!(ops(&program)[2].on_ok(), BranchTarget::Escape);
    }

    /// Conditional with no else: predicate's on_failed = Escape ("branch,
    /// not guard" outcome elision); on_ok chains into then-body, whose
    /// tail escapes. At top level there's no propagation of predicate
    /// Failed.
    #[test]
    fn conditional_no_else_predicate_failed_escapes() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
            otherwise: None,
        }];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 2);
        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_eq!(
            ops(&program)[0].on_failed(),
            BranchTarget::Escape,
            "no-else predicate falls through to Escape (no propagation)",
        );
        assert_eq!(ops(&program)[1].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[1].on_failed(), BranchTarget::Terminate);
    }

    /// Conditional with else: predicate routes Ok→then-first,
    /// Failed→else-first. Both then-tail and else-tail escape at the
    /// post-conditional slot (= Escape at top level).
    #[test]
    fn conditional_with_else_routes_both_branches() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
            otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/else"))])),
        }];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 3);

        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_continue(ops(&program)[0].on_failed(), 2);

        assert_eq!(ops(&program)[1].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[1].on_failed(), BranchTarget::Terminate);

        assert_eq!(ops(&program)[2].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[2].on_failed(), BranchTarget::Terminate);
    }

    /// Conditional with empty then and non-empty else: predicate's
    /// on_ok escapes directly (no then-body to enter); on_failed enters
    /// the else-branch. Validation rejects empty-then-empty-else, so
    /// this shape always has a non-empty else when then is empty.
    #[test]
    fn conditional_empty_then_nonempty_else() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([]),
            otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/else"))])),
        }];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 2);

        assert_eq!(
            ops(&program)[0].on_ok(),
            BranchTarget::Escape,
            "empty-then: pred.on_ok bypasses to post-conditional slot",
        );
        assert_continue(ops(&program)[0].on_failed(), 1);

        assert_eq!(ops(&program)[1].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[1].on_failed(), BranchTarget::Terminate);
    }

    /// `Some(empty)` else is shape-equivalent to `None`: lowering emits
    /// pred.on_failed = Escape (the no-else fall-through). Validation
    /// rejects this shape upstream (EmptyConditional fires for
    /// then=[]+else=[]); pinning the lowering's response keeps the IR
    /// well-formed even if the validator's check is bypassed.
    #[test]
    fn conditional_some_empty_else_falls_through_like_none() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
            otherwise: Some(Box::new([])),
        }];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 2);
        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_eq!(ops(&program)[0].on_failed(), BranchTarget::Escape);
    }

    /// Nested conditional in the then-branch produces 5 ops with the
    /// CFG-shaped IR (no intermediate Jump opcodes needed). Outer
    /// predicate routes Ok→inner-pred, Failed→outer-else. Inner
    /// predicate routes Ok→inner-then, Failed→inner-else. All body tails
    /// escape at the top level.
    #[test]
    fn nested_conditional_in_then_lowers_to_five_ops() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/outer"),
            then: Box::new([Action::Conditional {
                when: exec_with_literal("/bin/inner"),
                then: Box::new([Action::Exec(exec_with_literal("/bin/inner-then"))]),
                otherwise: Some(Box::new([Action::Exec(exec_with_literal(
                    "/bin/inner-else",
                ))])),
            }]),
            otherwise: Some(Box::new([Action::Exec(exec_with_literal(
                "/bin/outer-else",
            ))])),
        }];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 5, "5 ops (was 7 with Jumps)");

        // 0: outer pred. on_ok = 1 (inner pred). on_failed = 4 (outer else).
        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_continue(ops(&program)[0].on_failed(), 4);

        // 1: inner pred. on_ok = 2 (inner then). on_failed = 3 (inner else).
        assert_continue(ops(&program)[1].on_ok(), 2);
        assert_continue(ops(&program)[1].on_failed(), 3);

        // 2-4: bodies, each escape on Ok, terminate on Failed.
        for idx in 2..5 {
            assert_eq!(ops(&program)[idx].on_ok(), BranchTarget::Escape);
            assert_eq!(ops(&program)[idx].on_failed(), BranchTarget::Terminate);
        }
    }

    /// Conditional inside a non-last sequence position: the conditional's
    /// tails (then-tail and else-tail / pred.on_failed for no-else) chain
    /// to the next iteration's first slot, not to Escape. The next
    /// iteration's own tail is what escapes.
    #[test]
    fn conditional_in_non_last_position_chains_to_next() {
        let tree = [
            Action::Conditional {
                when: exec_with_literal("/bin/check"),
                then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
                otherwise: None,
            },
            Action::Exec(exec_with_literal("/bin/after")),
        ];
        let program = lower_to_program(&tree).expect("lowering succeeds");
        assert_eq!(program.ops().len(), 3);

        // 0: pred. on_ok = 1 (then). on_failed = 2 (post-conditional = after).
        assert_continue(ops(&program)[0].on_ok(), 1);
        assert_continue(ops(&program)[0].on_failed(), 2);

        // 1: then. on_ok = 2 (after). on_failed = Terminate.
        assert_continue(ops(&program)[1].on_ok(), 2);
        assert_eq!(ops(&program)[1].on_failed(), BranchTarget::Terminate);

        // 2: after. on_ok = Escape (top-level tail). on_failed = Terminate.
        assert_eq!(ops(&program)[2].on_ok(), BranchTarget::Escape);
        assert_eq!(ops(&program)[2].on_failed(), BranchTarget::Terminate);
    }

    /// Empty surface trees lower to a [`ProgramError::EmptyProgram`].
    /// The `validate_actions` entry point rejects the empty case before
    /// calling `lower_to_program`, so this is purely a defensive contract
    /// test for the lowering primitive.
    #[test]
    fn empty_tree_returns_empty_program_error() {
        let err = lower_to_program(&[]).expect_err("empty tree must error");
        assert!(matches!(
            err,
            specter_core::program::ProgramError::EmptyProgram
        ));
    }
}
