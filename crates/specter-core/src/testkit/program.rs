//! Test-only [`ActionProgram`] constructors.
//!
//! The production lowering path lives in `specter-config`; consumers of
//! the engine and actuator don't depend on `specter-config`, so they need
//! a backdoor for fixture construction. These helpers are the canonical
//! shape — a fixture built via `single_exec_program(argv)` is
//! operationally identical to one produced by config lowering of a
//! single `[[watch.actions]] exec = [...]` entry.

use crate::sub::{ActionProgram, ArgTemplate, ExecAction, Instruction};
use std::sync::Arc;

/// Single-exec program with no per-step timeout.
///
/// Covers the common fixture shape used across engine and actuator
/// tests. The returned `Arc` is the same shape `lower_to_program`
/// mints, so it can flow directly into [`crate::Sub::new`] /
/// [`crate::SubAttachRequest`] / [`crate::Effect`].
#[must_use]
pub fn single_exec_program(argv: impl IntoIterator<Item = ArgTemplate>) -> Arc<ActionProgram> {
    Arc::new(ActionProgram::new([Instruction::SpawnExec(
        ExecAction::new(argv),
    )]))
}

/// Two-instruction `[Predicate(when, jump=2), Exec(then)]` program.
///
/// Mirrors the lowering of `{ when = ..., then = [{ exec = ... }] }`
/// with no `else` branch — the predicate's `jump_target` points one
/// past the last instruction (the natural "skip past end" form), so
/// a Failed predicate terminates the plan Ok without running the
/// then-exec.
///
/// Convenience fixture for actuator tests that exercise predicate
/// dispatch without routing through the config layer.
#[must_use]
pub fn predicate_then_program(when: ExecAction, then_exec: ExecAction) -> Arc<ActionProgram> {
    Arc::new(ActionProgram::new([
        Instruction::SpawnPredicate {
            exec: when,
            jump_target: 2,
        },
        Instruction::SpawnExec(then_exec),
    ]))
}
