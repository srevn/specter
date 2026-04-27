//! Pure resolver — turns a `(Sub, paths, diff, correlation, forced)` tuple
//! into [`CommandResolved`] argv plus the standard `SPECTER_*` env-var set.
//!
//! Lives next to the data types it touches; pure data work — no
//! `std::env`, no `std::process`, no I/O. The Engine invokes at emission
//! time (stable-verdict path); tests drive directly.
//!
//! # `SPECTER_DIFF_PATH` is **not** set here
//!
//! The actuator at spawn time materializes the diff tmp file (its path
//! depends on the actuator process's pid) and appends `SPECTER_DIFF_PATH`
//! to the env it passes to `Command::envs`. The resolver returns env
//! without that var; the actuator decides whether the file write
//! succeeded and conditionally extends env.
//!
//! # Argv substitution semantics
//!
//! Each [`ArgTemplate`] in `Sub.command.argv` produces one or more argv
//! slots. The walk is single-pass with a prefix accumulator:
//!
//! - **Literals** and **single-value placeholders** (`$path`, `$rel`,
//!   `$anchor`) append to the prefix.
//! - **Multi-value placeholders** (`$created`, `$deleted`, `$modified`,
//!   `$renamed_from`, `$renamed_to`) emit one argv slot per matching
//!   `Diff` entry, each prefixed by the accumulated prefix; then the
//!   accumulator resets to empty.
//! - At end-of-template: if anything was ever emitted from a multi-value,
//!   any remaining accumulator becomes a standalone trailing slot. If
//!   nothing was emitted (no multi-value found), the single-slot
//!   prefix is the one slot for this template.
//! - An [`ArgTemplate`] containing a multi-value placeholder that yields
//!   zero entries (empty diff list, or `diff = None`) produces zero argv
//!   slots — there's no value to emit, and dropping the surrounding
//!   prefix is the principle-of-least-surprise (`["fmt", "$created"]`
//!   with no created entries is just `["fmt"]`).
//!
//! Two multi-value placeholders within one template (exotic; e.g.
//! `["mv $renamed_from $renamed_to"]` as one quoted arg) expand
//! **independently** — no parallel zip. Users wanting per-pair semantics
//! use `EffectScope::PerStableFile`.

use crate::diff::Diff;
use crate::effect::{CommandResolved, CorrelationId};
use crate::sub::{ArgPart, ArgTemplate, EffectScope, Placeholder, Sub};
use std::path::Path;

/// Resolve an Effect's command + env from the firing Sub and the burst's
/// positional context. See module docs.
#[must_use]
pub fn resolve_effect(
    sub: &Sub,
    anchor_path: &Path,
    target_path: &Path,
    target_rel: &str,
    forced: bool,
    correlation: CorrelationId,
    diff: Option<&Diff>,
) -> (CommandResolved, Vec<(String, String)>) {
    let argv = substitute_argv(
        &sub.command.argv,
        anchor_path,
        target_path,
        target_rel,
        diff,
    );
    let env = build_env(
        sub,
        anchor_path,
        target_path,
        target_rel,
        forced,
        correlation,
    );
    (CommandResolved { argv }, env)
}

/// Substitute placeholders into argv slots.
fn substitute_argv(
    template: &[ArgTemplate],
    anchor_path: &Path,
    target_path: &Path,
    target_rel: &str,
    diff: Option<&Diff>,
) -> Vec<String> {
    let mut argv = Vec::with_capacity(template.len());
    for arg in template {
        substitute_one(arg, anchor_path, target_path, target_rel, diff, &mut argv);
    }
    argv
}

/// Render one [`ArgTemplate`] into zero or more argv slots, appending to
/// `out`.
fn substitute_one(
    arg: &ArgTemplate,
    anchor_path: &Path,
    target_path: &Path,
    target_rel: &str,
    diff: Option<&Diff>,
    out: &mut Vec<String>,
) {
    let mut prefix = String::new();
    let mut emitted_any = false;
    for part in &arg.parts {
        match part {
            ArgPart::Literal(s) => prefix.push_str(s),
            ArgPart::Placeholder(p) => match p {
                Placeholder::Path => prefix.push_str(&target_path.to_string_lossy()),
                Placeholder::Rel => prefix.push_str(target_rel),
                Placeholder::Anchor => prefix.push_str(&anchor_path.to_string_lossy()),
                Placeholder::Created
                | Placeholder::Deleted
                | Placeholder::Modified
                | Placeholder::RenamedFrom
                | Placeholder::RenamedTo => {
                    let values = multivalue_values(*p, diff);
                    for v in values {
                        let mut slot = prefix.clone();
                        slot.push_str(&v);
                        out.push(slot);
                        emitted_any = true;
                    }
                    prefix.clear();
                }
            },
        }
    }
    // If a multi-value placeholder emitted at least one slot, a non-empty
    // trailing prefix becomes its own standalone slot. Otherwise the
    // prefix is the single slot for this ArgTemplate.
    if emitted_any {
        if !prefix.is_empty() {
            out.push(prefix);
        }
    } else {
        // No multi-value placeholders consumed (or all yielded zero
        // entries with no leading literals/single-values). If we saw
        // *any* multi-value at all (even with zero values), the
        // ArgTemplate produces zero slots — `emitted_any == false &&
        // arg has a multi-value` is the empty-diff case.
        if has_multivalue(arg) {
            // Drop the prefix; zero-slot output.
        } else {
            out.push(prefix);
        }
    }
}

fn has_multivalue(arg: &ArgTemplate) -> bool {
    arg.parts.iter().any(ArgPart::is_diff_placeholder)
}

/// Resolve a multi-value placeholder against `diff` to its segment list.
/// Returns an empty Vec if `diff` is None or the corresponding list is
/// empty.
fn multivalue_values(p: Placeholder, diff: Option<&Diff>) -> Vec<String> {
    let Some(d) = diff else {
        return Vec::new();
    };
    match p {
        Placeholder::Created => d.created.iter().map(|e| e.segment.to_string()).collect(),
        Placeholder::Deleted => d.deleted.iter().map(|e| e.segment.to_string()).collect(),
        Placeholder::Modified => d.modified.iter().map(|e| e.segment.to_string()).collect(),
        Placeholder::RenamedFrom => d
            .renamed
            .iter()
            .map(|r| r.from.segment.to_string())
            .collect(),
        Placeholder::RenamedTo => d.renamed.iter().map(|r| r.to.segment.to_string()).collect(),
        // Unreachable: caller already filtered to multi-value variants.
        Placeholder::Path | Placeholder::Rel | Placeholder::Anchor => Vec::new(),
    }
}

/// Build the standard `SPECTER_*` env-var set excluding `SPECTER_DIFF_PATH`
/// (the actuator appends that at spawn time).
fn build_env(
    sub: &Sub,
    anchor_path: &Path,
    target_path: &Path,
    target_rel: &str,
    forced: bool,
    correlation: CorrelationId,
) -> Vec<(String, String)> {
    let event_kind = match sub.scope {
        EffectScope::SubtreeRoot => "dir-subtree",
        EffectScope::PerStableFile => "file",
    };
    vec![
        (
            "SPECTER_PATH".to_owned(),
            target_path.to_string_lossy().into_owned(),
        ),
        ("SPECTER_REL_PATH".to_owned(), target_rel.to_owned()),
        (
            "SPECTER_ANCHOR".to_owned(),
            anchor_path.to_string_lossy().into_owned(),
        ),
        ("SPECTER_SUB".to_owned(), sub.name.to_string()),
        (
            "SPECTER_FORCED".to_owned(),
            if forced { "1" } else { "0" }.to_owned(),
        ),
        ("SPECTER_EVENT_KIND".to_owned(), event_kind.to_owned()),
        ("SPECTER_CORRELATION".to_owned(), correlation.0.to_string()),
    ]
}
