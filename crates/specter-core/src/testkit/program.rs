//! Test-only [`ActionProgram`] constructors.
//!
//! The production lowering path lives in `specter-config`; consumers of
//! the engine and actuator don't depend on `specter-config`, so they need
//! a backdoor for fixture construction. These helpers are the canonical
//! shape — a fixture built via `single_exec_program(argv)` is
//! operationally identical to one produced by config lowering of a
//! single `[[watch.actions]] exec = [...]` entry.

use crate::program::{
    ActionProgram, ArgTemplate, BranchTarget, ExecAction, ProgramBuilder, SpawnBody,
};
use std::sync::Arc;

/// Single-exec program with no per-step timeout.
///
/// Covers the common fixture shape used across engine and actuator
/// tests. The returned `Arc` is the same shape `lower_to_program` mints,
/// so it can flow directly into [`crate::Sub::from_request`] /
/// [`crate::SubAttachRequest`] / [`crate::Effect`].
///
/// Edges:
/// - `on_ok = Escape` (natural completion at top level).
/// - `on_failed = Terminate` (stop-on-failure, outcome propagates).
#[must_use]
pub fn single_exec_program(argv: impl IntoIterator<Item = ArgTemplate>) -> Arc<ActionProgram> {
    let mut b = ProgramBuilder::new();
    let h = b.emit(SpawnBody::Exec(ExecAction::new(argv, None)));
    b.patch_on_ok(h, BranchTarget::Escape)
        .expect("Escape target is unconditionally accepted");
    b.patch_on_failed(h, BranchTarget::Terminate)
        .expect("Terminate target is unconditionally accepted");
    Arc::new(b.build().expect("one-op program with both edges patched"))
}

/// Two-op `[pred (on_ok=Continue(1), on_failed=Escape), then-exec
/// (on_ok=Escape, on_failed=Terminate)]` program.
///
/// Mirrors the lowering of `{ when = ..., then = [{ exec = ... }] }`
/// with no `else` branch — the predicate's `on_failed` is `Escape` (the
/// "branch, not guard" outcome elision: a Failed predicate terminates
/// the plan Ok rather than propagating Failed). On predicate Ok, the
/// then-exec runs; its own Failed propagates (stop-on-failure).
///
/// Convenience fixture for actuator tests that exercise predicate
/// dispatch without routing through the config layer.
#[must_use]
pub fn predicate_then_program(when: ExecAction, then_exec: ExecAction) -> Arc<ActionProgram> {
    let mut b = ProgramBuilder::new();
    let pred = b.emit(SpawnBody::Exec(when));
    // Use the deferred-slot promise: target == pending.len() (= 1),
    // filled by the upcoming `then` emit.
    let then_first = b.continue_to_next();
    b.patch_on_ok(pred, then_first)
        .expect("deferred slot target == pending.len() is accepted");
    b.patch_on_failed(pred, BranchTarget::Escape)
        .expect("Escape target is unconditionally accepted");

    let then_handle = b.emit(SpawnBody::Exec(then_exec));
    b.patch_on_ok(then_handle, BranchTarget::Escape)
        .expect("Escape target is unconditionally accepted");
    b.patch_on_failed(then_handle, BranchTarget::Terminate)
        .expect("Terminate target is unconditionally accepted");

    Arc::new(
        b.build()
            .expect("two-op program with all edges patched and in bounds"),
    )
}
