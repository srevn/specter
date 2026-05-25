//! Render an [`ActionProgram`] to operator-readable lines for
//! [`crate::ipc::protocol::SubDetails::program`].
//!
//! One line per [`ProgramOp`]. The argv parts use the same
//! `${specter.<name>}` / `${env.<NAME>}` vocabulary the source TOML
//! uses, but the rendered form is **operator-readable, not
//! round-trippable**: adjacent literals in one `ArgTemplate`
//! concatenate without a separator (see
//! [`tests::render_pipe_joins_stages_with_pipe_separator`] for
//! `/bin/grepfoo`), so the config lexer remains the authoritative
//! parse boundary.
//!
//! The reverse-direction placeholder table mirrors the forward table
//! consumed by the TOML lexer; both are exhaustive matches against
//! the shared [`Placeholder`] enum, so a new variant is a compile
//! error in this module AND in the lexer.

use std::fmt::Write as _;

use specter_core::program::{BranchTarget, ProgramOp, SpawnBody};
use specter_core::{ActionProgram, ArgPart, ArgTemplate, ExecAction, Placeholder};

/// Render every op in the program as one line. Caller threads the
/// returned slice into [`crate::ipc::protocol::SubDetails::program`].
pub(crate) fn render(program: &ActionProgram) -> Vec<String> {
    program
        .ops()
        .iter()
        .enumerate()
        .map(|(i, op)| render_op(i, op))
        .collect()
}

/// `"[i] <body>  ok→<edge> fail→<edge>"` — the canonical per-op
/// shape every renderer reuses.
fn render_op(index: usize, op: &ProgramOp) -> String {
    let mut out = String::with_capacity(64);
    let _ = write!(out, "[{index}] ");
    write_body(&mut out, op.body());
    out.push_str("  ok→");
    write_target(&mut out, op.on_ok());
    out.push_str(" fail→");
    write_target(&mut out, op.on_failed());
    out
}

/// `exec <argv>` or `pipe(N) <stage0> | <stage1> …`.
fn write_body(out: &mut String, body: &SpawnBody) {
    match body {
        SpawnBody::Exec(exec) => {
            out.push_str("exec ");
            write_exec(out, exec);
        }
        SpawnBody::Pipe(ms) => {
            let stages = ms.stages();
            let _ = write!(out, "pipe({}) ", stages.len());
            for (i, stage) in stages.iter().enumerate() {
                if i > 0 {
                    out.push_str(" | ");
                }
                write_exec(out, stage);
            }
        }
    }
}

/// Argv joined by a single space, then an optional ` (timeout …)`
/// suffix on a per-stage basis. Default-untimed Execs render
/// unannotated; operators see per-stage timeouts in the pipe form.
fn write_exec(out: &mut String, exec: &ExecAction) {
    for (i, arg) in exec.argv().iter().enumerate() {
        if i > 0 {
            out.push(' ');
        }
        write_template(out, arg);
    }
    if let Some(d) = exec.timeout() {
        let _ = write!(out, " (timeout {})", humantime::format_duration(d));
    }
}

/// One argv slot's parts written back-to-back. Two adjacent parts (a
/// literal `--flag=` followed by `${specter.path}`) collapse into a
/// single argv slot — the writer does not insert separators.
fn write_template(out: &mut String, t: &ArgTemplate) {
    for part in t.parts() {
        write_part(out, part);
    }
}

fn write_part(out: &mut String, p: &ArgPart) {
    match p {
        ArgPart::Literal(s) => out.push_str(s),
        ArgPart::Placeholder(ph) => {
            let _ = write!(out, "${{specter.{}}}", placeholder_token(*ph));
        }
        ArgPart::EnvVar {
            name,
            default: None,
        } => {
            let _ = write!(out, "${{env.{name}}}");
        }
        ArgPart::EnvVar {
            name,
            default: Some(d),
        } => {
            let _ = write!(out, "${{env.{name}:-{d}}}");
        }
    }
}

/// In-program target prints as `#<index>`; the two no-op terminals
/// print their operator-facing keyword.
fn write_target(out: &mut String, t: BranchTarget) {
    match t {
        BranchTarget::Continue(idx) => {
            let _ = write!(out, "#{}", idx.get());
        }
        BranchTarget::Terminate => out.push_str("terminate"),
        BranchTarget::Escape => out.push_str("escape"),
    }
}

/// Reverse direction of the TOML template lexer's placeholder
/// catalog. A new [`Placeholder`] variant added to `specter-core`
/// without a matching arm here is a compile error — the exhaustive
/// match is the structural seam that keeps the round-trip vocabulary
/// in sync.
const fn placeholder_token(p: Placeholder) -> &'static str {
    match p {
        Placeholder::Path => "path",
        Placeholder::Relative => "relative",
        Placeholder::Anchor => "anchor",
        Placeholder::Watch => "watch",
        Placeholder::Parent => "parent",
        Placeholder::Time => "time",
        Placeholder::Created => "created",
        Placeholder::Deleted => "deleted",
        Placeholder::Modified => "modified",
        Placeholder::RenamedFrom => "renamed_from",
        Placeholder::RenamedTo => "renamed_to",
        Placeholder::Excluded => "excluded",
    }
}

#[cfg(test)]
mod tests {
    use super::render;
    use specter_core::program::{BranchTarget, MultiStage, ProgramBuilder, SpawnBody};
    use specter_core::{ActionProgram, ArgPart, ArgTemplate, ExecAction, Placeholder};
    use std::sync::Arc;
    use std::time::Duration;

    fn exec_with(parts: impl IntoIterator<Item = ArgPart>) -> ExecAction {
        ExecAction::new([ArgTemplate::new(parts)], None)
    }

    fn exec_with_args(args: impl IntoIterator<Item = ArgTemplate>) -> ExecAction {
        ExecAction::new(args, None)
    }

    fn exec_timed(parts: impl IntoIterator<Item = ArgPart>, timeout: Duration) -> ExecAction {
        ExecAction::new([ArgTemplate::new(parts)], Some(timeout))
    }

    fn single_op_program(
        body: SpawnBody,
        on_ok: BranchTarget,
        on_failed: BranchTarget,
    ) -> ActionProgram {
        let mut b = ProgramBuilder::new();
        let h = b.emit(body);
        b.patch_on_ok(h, on_ok).unwrap();
        b.patch_on_failed(h, on_failed).unwrap();
        b.build().unwrap()
    }

    /// `${specter.<name>}` projects back from [`Placeholder`] using the
    /// same vocabulary the TOML lexer consumes. Catches a future rename of
    /// a `Placeholder` variant that forgets to update the reverse table.
    #[test]
    fn render_exec_with_literal_and_placeholder() {
        let body = SpawnBody::Exec(exec_with([
            ArgPart::literal("/bin/build"),
            ArgPart::Placeholder(Placeholder::Path),
        ]));
        let prog = single_op_program(body, BranchTarget::Escape, BranchTarget::Terminate);
        let lines = render(&prog);
        assert_eq!(lines.len(), 1);
        assert_eq!(
            lines[0],
            "[0] exec /bin/build${specter.path}  ok→escape fail→terminate",
        );
    }

    /// `pipe(N)` prefix carries the stage count and stages join on ` | `.
    /// Pins both the operator-visible separator and the stage counting.
    #[test]
    fn render_pipe_joins_stages_with_pipe_separator() {
        let stages: Arc<[ExecAction]> = Arc::from(vec![
            exec_with([ArgPart::literal("/bin/cat")]),
            exec_with([ArgPart::literal("/bin/grep"), ArgPart::literal("foo")]),
            exec_with([ArgPart::literal("/bin/wc")]),
        ]);
        let body = SpawnBody::Pipe(MultiStage::new(stages).unwrap());
        let prog = single_op_program(body, BranchTarget::Escape, BranchTarget::Terminate);
        let lines = render(&prog);
        assert_eq!(lines.len(), 1);
        // Adjacent literals in one ArgTemplate concatenate (no space).
        assert_eq!(
            lines[0],
            "[0] pipe(3) /bin/cat | /bin/grepfoo | /bin/wc  ok→escape fail→terminate",
        );
    }

    /// Argv parts from separate [`ArgTemplate`]s join on a single space;
    /// parts within one template concatenate. Two `ArgTemplate`s with
    /// `[Literal("--input="), Placeholder(Path)]` and `[Literal("/log")]`
    /// render as `--input=${specter.path} /log`.
    #[test]
    fn render_exec_splits_args_on_template_boundaries() {
        let body = SpawnBody::Exec(exec_with_args([
            ArgTemplate::new([
                ArgPart::literal("--input="),
                ArgPart::Placeholder(Placeholder::Path),
            ]),
            ArgTemplate::new([ArgPart::literal("/log")]),
        ]));
        let prog = single_op_program(body, BranchTarget::Escape, BranchTarget::Terminate);
        let lines = render(&prog);
        assert_eq!(
            lines[0],
            "[0] exec --input=${specter.path} /log  ok→escape fail→terminate",
        );
    }

    /// Both env forms render with the right delimiter. `default = None` is
    /// `${env.NAME}`; `default = Some(d)` is `${env.NAME:-d}`.
    #[test]
    fn render_env_var_with_and_without_default() {
        let body = SpawnBody::Exec(exec_with_args([
            ArgTemplate::new([ArgPart::EnvVar {
                name: "HOME".into(),
                default: None,
            }]),
            ArgTemplate::new([ArgPart::EnvVar {
                name: "EDITOR".into(),
                default: Some("vi".into()),
            }]),
        ]));
        let prog = single_op_program(body, BranchTarget::Escape, BranchTarget::Terminate);
        let lines = render(&prog);
        assert_eq!(
            lines[0],
            "[0] exec ${env.HOME} ${env.EDITOR:-vi}  ok→escape fail→terminate",
        );
    }

    /// Branch targets render to `#N` / `terminate` / `escape`. `Continue`
    /// is built via the builder's `continue_to_next` helper (the only public
    /// path — `BranchIndex` is sealed); `Terminate` and `Escape` are the two
    /// no-op terminals.
    #[test]
    fn render_branch_targets_continue_terminate_escape() {
        // Two-op program so `Continue` points at the next emit.
        let mut b = ProgramBuilder::new();
        let h0 = b.emit(SpawnBody::Exec(exec_with([ArgPart::literal("/bin/a")])));
        let next = b.continue_to_next();
        let h1 = b.emit(SpawnBody::Exec(exec_with([ArgPart::literal("/bin/b")])));
        b.patch_on_ok(h0, next).unwrap();
        b.patch_on_failed(h0, BranchTarget::Terminate).unwrap();
        b.patch_on_ok(h1, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h1, BranchTarget::Terminate).unwrap();
        let prog = b.build().unwrap();
        let lines = render(&prog);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0], "[0] exec /bin/a  ok→#1 fail→terminate");
        assert_eq!(lines[1], "[1] exec /bin/b  ok→escape fail→terminate");
    }

    /// `(timeout …)` suffix only appears when the operator set one.
    /// Default-untimed Execs render unannotated.
    #[test]
    fn render_exec_with_timeout_appends_suffix() {
        let body = SpawnBody::Exec(exec_timed(
            [ArgPart::literal("/bin/slow")],
            Duration::from_secs(5),
        ));
        let prog = single_op_program(body, BranchTarget::Escape, BranchTarget::Terminate);
        let lines = render(&prog);
        assert_eq!(
            lines[0],
            "[0] exec /bin/slow (timeout 5s)  ok→escape fail→terminate",
        );

        // Negative control: default-untimed exec carries no suffix.
        let untimed = SpawnBody::Exec(exec_with([ArgPart::literal("/bin/fast")]));
        let prog2 = single_op_program(untimed, BranchTarget::Escape, BranchTarget::Terminate);
        let lines2 = render(&prog2);
        assert!(
            !lines2[0].contains("(timeout"),
            "untimed exec must not render (timeout …) — got {:?}",
            lines2[0],
        );
    }
}
