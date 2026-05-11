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
//! The runtime sees only the lowered bytecode IR — there is no benefit
//! to teaching the engine or actuator about pipes, conditionals, or
//! other surface-grammar shapes. Keeping the AST in `specter-config`
//! means future variants land here without bloating `specter-core`'s
//! public surface; the lowering pass is the single seam between the
//! two layers.
//!
//! # Lowering invariants
//!
//! - **Forward-only jumps.** Every jump target (predicate `jump_target`
//!   and unconditional `Jump.target`) is strictly greater than the
//!   position of the jump-emitting instruction. The lowering walk is
//!   the sole producer; backward jumps are unrepresentable by
//!   construction (no `Loop`-like surface).
//! - **`target == instructions.len()`** is the legal "skip past end"
//!   form — the actuator's `next_spawnable` walks past it and the
//!   reap-path terminates Ok. Used by an unconditional `Conditional`
//!   whose `then` runs to plan-end with no `else`, and by an `else`
//!   that itself runs to plan-end.

use specter_core::{ActionProgram, ExecAction, Instruction};
use std::sync::Arc;

/// Surface-syntax tree produced by validation. Lives in `specter-config`
/// because it's a validation artifact — it never reaches the engine or
/// actuator. After validation, lowering folds it into an
/// `Arc<ActionProgram>`; the tree is dropped at the function boundary.
///
/// `pub(crate)` because the tree never escapes this crate. Future
/// variants (`Pipe`) land here as additional arms; the runtime stays
/// oblivious.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Action {
    /// Spawn a single process with this argv template.
    Exec(ExecAction),
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

/// Walk a validated `[Action]` sequence and emit the equivalent
/// `ActionProgram`. The Arc is the same allocation that the engine's
/// `Sub.program` and every emitted `Effect.program` references — one
/// shared bytecode IR per validated Sub.
pub(crate) fn lower_to_program(tree: &[Action]) -> Arc<ActionProgram> {
    let mut out: Vec<Instruction> = Vec::with_capacity(tree.len());
    lower_actions(tree, &mut out);
    Arc::new(ActionProgram::new(out))
}

/// Recursive lowering pass.
///
/// `Action::Exec` lowers to a single [`Instruction::SpawnExec`].
///
/// `Action::Conditional` lowers to a [`Instruction::SpawnPredicate`]
/// whose `jump_target` is backpatched to the start of the else-branch
/// (or to one past the then-branch when no else exists). When an
/// else-branch is present, an unconditional [`Instruction::Jump`] is
/// emitted immediately after the then-branch to skip past the
/// else-branch on the predicate-Ok path; its `target` is backpatched
/// to one past the else-branch.
fn lower_actions(actions: &[Action], out: &mut Vec<Instruction>) {
    for action in actions {
        match action {
            Action::Exec(e) => out.push(Instruction::SpawnExec(e.clone())),

            Action::Conditional {
                when,
                then,
                otherwise,
            } => {
                let pred_idx = out.len();
                // Backpatched below; the placeholder target is never
                // observed since the next mutation rewrites it.
                out.push(Instruction::SpawnPredicate {
                    exec: when.clone(),
                    jump_target: 0,
                });
                lower_actions(then, out);

                match otherwise {
                    Some(else_body) if !else_body.is_empty() => {
                        // Emit Jump after the then-branch to skip past
                        // the else-branch on the Ok path; backpatched
                        // after we know `after_else`.
                        let skip_idx = out.len();
                        out.push(Instruction::Jump { target: 0 });

                        let else_start = u32_or_program_overflow(out.len());
                        lower_actions(else_body, out);
                        let after_else = u32_or_program_overflow(out.len());

                        backpatch_jump(&mut out[pred_idx], else_start);
                        backpatch_jump(&mut out[skip_idx], after_else);
                    }
                    _ => {
                        // No else-branch (or empty after normalisation):
                        // predicate's Failed cursor lands one past the
                        // then-branch — the natural "skip past end of
                        // conditional" form.
                        let after_then = u32_or_program_overflow(out.len());
                        backpatch_jump(&mut out[pred_idx], after_then);
                    }
                }
            }
        }
    }
}

/// Rewrite the jump-bearing field of an [`Instruction`] in place.
/// Called from [`lower_actions`] for backpatching after the target
/// position is known.
///
/// The `unreachable` arm enforces that the caller indexed the right
/// slot — only [`Instruction::SpawnPredicate`] and
/// [`Instruction::Jump`] carry a jump-bearing field. A caller
/// indexing wrong is a lowering bug, not a config-author error.
fn backpatch_jump(insn: &mut Instruction, target: u32) {
    match insn {
        Instruction::SpawnPredicate { jump_target, .. } => *jump_target = target,
        Instruction::Jump { target: t } => *t = target,
        Instruction::SpawnExec(_) | Instruction::SpawnPipe(_) => {
            unreachable!("backpatch site must be a SpawnPredicate or Jump instruction")
        }
    }
}

/// Convert a `usize` instruction-vector length to `u32`. The `u32`
/// width is the lowering invariant — every `cursor` and jump target
/// is `u32`, so the program slice is also `u32`-indexable. The
/// `expect` is structurally unreachable: a single `[[watch]]` block
/// at `u32::MAX` instructions would already exceed any plausible
/// validator-enforced ceiling and would OOM long before. The panic
/// is preferred over `unwrap_or(u32::MAX)`-style truncation, which
/// would silently produce invalid jump targets.
fn u32_or_program_overflow(len: usize) -> u32 {
    u32::try_from(len).expect("program length fits in u32 by construction")
}

#[cfg(test)]
mod tests {
    use super::{Action, lower_to_program};
    use specter_core::{ArgPart, ArgTemplate, ExecAction, Instruction};

    fn exec_with_literal(literal: &str) -> ExecAction {
        ExecAction::new([ArgTemplate::new([ArgPart::literal(literal)])])
    }

    /// Single Exec lowers to a one-instruction program — the natural
    /// shape of every v1 single-action watch.
    #[test]
    fn single_exec_lowers_to_one_spawn_exec() {
        let tree = [Action::Exec(exec_with_literal("/bin/build"))];
        let program = lower_to_program(&tree);
        assert_eq!(program.instructions.len(), 1);
        assert!(matches!(program.instructions[0], Instruction::SpawnExec(_)));
    }

    /// Multiple Execs preserve order and produce one instruction per
    /// surface entry — the actuator walks them sequentially with
    /// stop-on-failure semantics.
    #[test]
    fn multi_exec_lowers_preserving_order() {
        let tree = [
            Action::Exec(exec_with_literal("/bin/first")),
            Action::Exec(exec_with_literal("/bin/second")),
            Action::Exec(exec_with_literal("/bin/third")),
        ];
        let program = lower_to_program(&tree);
        assert_eq!(program.instructions.len(), 3);
        for (idx, expected) in ["/bin/first", "/bin/second", "/bin/third"]
            .iter()
            .enumerate()
        {
            let Instruction::SpawnExec(exec) = &program.instructions[idx] else {
                panic!(
                    "instruction {idx}: expected SpawnExec, got {:?}",
                    program.instructions[idx]
                );
            };
            let Some(ArgPart::Literal(s)) = exec.argv[0].parts.first() else {
                panic!("argv[0].parts[0] not Literal");
            };
            assert_eq!(s.as_str(), *expected);
        }
    }

    /// Empty surface trees lower to zero-instruction programs. The
    /// `validate_actions` entry point rejects the empty case before
    /// calling `lower_to_program`, so this is purely a defensive
    /// contract test for the lowering primitive.
    #[test]
    fn empty_tree_lowers_to_empty_program() {
        let program = lower_to_program(&[]);
        assert_eq!(program.instructions.len(), 0);
    }

    /// Conditional with no else-branch lowers to predicate + then
    /// instructions; predicate's `jump_target` is the natural
    /// `instructions.len()` "skip past end" form, which the actuator
    /// treats as terminate-Ok.
    #[test]
    fn conditional_no_else_predicate_jumps_past_then() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
            otherwise: None,
        }];
        let program = lower_to_program(&tree);
        // [SpawnPredicate(check, jump=2), SpawnExec(then)]
        assert_eq!(program.instructions.len(), 2);
        match &program.instructions[0] {
            Instruction::SpawnPredicate { jump_target, .. } => {
                assert_eq!(*jump_target, 2, "no-else: jump past then-branch end");
            }
            other => panic!("expected SpawnPredicate, got {other:?}"),
        }
        assert!(matches!(program.instructions[1], Instruction::SpawnExec(_)));
    }

    /// Conditional with an else-branch lowers to predicate + then +
    /// Jump + else; predicate jumps to `else_start`, Jump targets past
    /// the else.
    #[test]
    fn conditional_with_else_predicate_jumps_to_else_jump_skips_else() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
            otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/else"))])),
        }];
        let program = lower_to_program(&tree);
        // [SpawnPredicate(check, jump=3), SpawnExec(then), Jump(target=4), SpawnExec(else)]
        assert_eq!(program.instructions.len(), 4);
        match &program.instructions[0] {
            Instruction::SpawnPredicate { jump_target, .. } => {
                assert_eq!(*jump_target, 3, "predicate jumps to else_start");
            }
            other => panic!("expected SpawnPredicate, got {other:?}"),
        }
        assert!(matches!(program.instructions[1], Instruction::SpawnExec(_)));
        match &program.instructions[2] {
            Instruction::Jump { target } => {
                assert_eq!(*target, 4, "Jump skips past else-branch");
            }
            other => panic!("expected Jump, got {other:?}"),
        }
        assert!(matches!(program.instructions[3], Instruction::SpawnExec(_)));
    }

    /// Conditional with empty `then` and non-empty `else` lowers to
    /// predicate + Jump + else. Predicate's `jump_target` points to the
    /// else-start, which is `pred_idx + 2` (one past the Jump). The
    /// Jump immediately following the predicate is unreachable on the
    /// Ok path (zero-instruction then-branch); on the Failed path the
    /// predicate jumps directly into else.
    #[test]
    fn conditional_empty_then_nonempty_else() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([]),
            otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/else"))])),
        }];
        let program = lower_to_program(&tree);
        // [SpawnPredicate(check, jump=2), Jump(target=3), SpawnExec(else)]
        assert_eq!(program.instructions.len(), 3);
        match &program.instructions[0] {
            Instruction::SpawnPredicate { jump_target, .. } => {
                assert_eq!(*jump_target, 2, "predicate jumps to else_start");
            }
            other => panic!("expected SpawnPredicate, got {other:?}"),
        }
        match &program.instructions[1] {
            Instruction::Jump { target } => {
                assert_eq!(*target, 3, "Jump (Ok path) targets after else");
            }
            other => panic!("expected Jump, got {other:?}"),
        }
        assert!(matches!(program.instructions[2], Instruction::SpawnExec(_)));
    }

    /// Conditional with `Some(empty)` else is treated the same as
    /// `None`: lowering omits the Jump. Validation normalises this
    /// before calling lower, but the lowering pass also handles it
    /// directly via the `_` arm of the otherwise match.
    #[test]
    fn conditional_some_empty_else_treated_as_none() {
        let tree = [Action::Conditional {
            when: exec_with_literal("/bin/check"),
            then: Box::new([Action::Exec(exec_with_literal("/bin/then"))]),
            otherwise: Some(Box::new([])),
        }];
        let program = lower_to_program(&tree);
        // Same as `None` — no Jump emitted.
        assert_eq!(program.instructions.len(), 2);
        match &program.instructions[0] {
            Instruction::SpawnPredicate { jump_target, .. } => {
                assert_eq!(*jump_target, 2);
            }
            other => panic!("expected SpawnPredicate, got {other:?}"),
        }
    }

    /// Nested conditional in the `then` branch produces two predicates
    /// with correctly-backpatched jump targets. Outer predicate jumps
    /// past the entire inner conditional + outer Jump; inner predicate
    /// jumps to inner else_start.
    #[test]
    fn nested_conditional_in_then_backpatches_correctly() {
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
        let program = lower_to_program(&tree);
        // [
        //   0: SpawnPredicate(outer, jump=6),
        //   1: SpawnPredicate(inner, jump=4),
        //   2: SpawnExec(inner-then),
        //   3: Jump(target=5),
        //   4: SpawnExec(inner-else),
        //   5: Jump(target=7),
        //   6: SpawnExec(outer-else),
        // ]
        assert_eq!(program.instructions.len(), 7);
        match &program.instructions[0] {
            Instruction::SpawnPredicate { jump_target, .. } => assert_eq!(*jump_target, 6),
            other => panic!("expected SpawnPredicate, got {other:?}"),
        }
        match &program.instructions[1] {
            Instruction::SpawnPredicate { jump_target, .. } => assert_eq!(*jump_target, 4),
            other => panic!("expected SpawnPredicate, got {other:?}"),
        }
        assert!(matches!(program.instructions[2], Instruction::SpawnExec(_)));
        match &program.instructions[3] {
            Instruction::Jump { target } => assert_eq!(*target, 5),
            other => panic!("expected Jump, got {other:?}"),
        }
        assert!(matches!(program.instructions[4], Instruction::SpawnExec(_)));
        match &program.instructions[5] {
            Instruction::Jump { target } => assert_eq!(*target, 7),
            other => panic!("expected Jump, got {other:?}"),
        }
        assert!(matches!(program.instructions[6], Instruction::SpawnExec(_)));
    }

    /// Lowering invariant: every jump emitted is forward (`target` >
    /// position of jump-emitting instruction). Pinned across the
    /// representative shapes — single conditional, with-else,
    /// empty-then, nested.
    #[test]
    fn all_jumps_are_forward() {
        let trees: Vec<Vec<Action>> = vec![
            vec![Action::Conditional {
                when: exec_with_literal("/bin/c"),
                then: Box::new([Action::Exec(exec_with_literal("/bin/t"))]),
                otherwise: None,
            }],
            vec![Action::Conditional {
                when: exec_with_literal("/bin/c"),
                then: Box::new([Action::Exec(exec_with_literal("/bin/t"))]),
                otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/e"))])),
            }],
            vec![Action::Conditional {
                when: exec_with_literal("/bin/c"),
                then: Box::new([]),
                otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/e"))])),
            }],
            vec![Action::Conditional {
                when: exec_with_literal("/bin/outer"),
                then: Box::new([Action::Conditional {
                    when: exec_with_literal("/bin/inner"),
                    then: Box::new([Action::Exec(exec_with_literal("/bin/it"))]),
                    otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/ie"))])),
                }]),
                otherwise: Some(Box::new([Action::Exec(exec_with_literal("/bin/oe"))])),
            }],
        ];
        for tree in trees {
            let program = lower_to_program(&tree);
            for (idx, insn) in program.instructions.iter().enumerate() {
                let pos = u32::try_from(idx).unwrap();
                match insn {
                    Instruction::SpawnPredicate { jump_target, .. } => {
                        assert!(
                            *jump_target > pos,
                            "predicate jump at {pos} is backward to {jump_target}",
                        );
                    }
                    Instruction::Jump { target } => {
                        assert!(*target > pos, "Jump at {pos} is backward to {target}");
                    }
                    Instruction::SpawnExec(_) | Instruction::SpawnPipe(_) => {}
                }
            }
        }
    }
}
