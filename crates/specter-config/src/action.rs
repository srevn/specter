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
//! # PR1 scope
//!
//! Only [`Action::Exec`] is implemented in PR1; future variants (`Pipe`,
//! `Conditional`, ...) extend the enum without retrofitting the
//! existing arm. Lowering is correspondingly trivial — each `Exec`
//! becomes one [`Instruction::SpawnExec`].

use specter_core::{ActionProgram, ExecAction, Instruction};
use std::sync::Arc;

/// Surface-syntax tree produced by validation. Lives in `specter-config`
/// because it's a validation artifact — it never reaches the engine or
/// actuator. After validation, lowering folds it into an
/// `Arc<ActionProgram>`; the tree is dropped at the function boundary.
///
/// `pub(crate)` because the tree never escapes this crate. Future
/// variants (`Pipe`, `Conditional`) land here as additional arms; the
/// runtime stays oblivious.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum Action {
    /// Spawn a single process with this argv template.
    Exec(ExecAction),
}

/// Walk a validated `[Action]` sequence and emit the equivalent
/// `ActionProgram`. The Arc is the same allocation that the engine's
/// `Sub.program` and every emitted `Effect.program` references — one
/// shared bytecode IR per validated Sub.
///
/// PR1 produces only [`Instruction::SpawnExec`]; the function is
/// shaped to absorb future tree variants without changing its caller
/// signature.
pub(crate) fn lower_to_program(tree: &[Action]) -> Arc<ActionProgram> {
    let mut out: Vec<Instruction> = Vec::with_capacity(tree.len());
    lower_actions(tree, &mut out);
    Arc::new(ActionProgram::new(out))
}

fn lower_actions(actions: &[Action], out: &mut Vec<Instruction>) {
    for action in actions {
        match action {
            Action::Exec(e) => out.push(Instruction::SpawnExec(e.clone())),
        }
    }
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
}
