//! Pure resolver — turns the substitution-domain projection on [`Effect`] plus one [`ExecAction`]
//! step into [`CommandResolved`] argv plus the standard `SPECTER_*` env-var set.
//!
//! Lives next to the actuator because resolution runs immediately before `spawner.spawn` —
//! Latest-coalesce drops `pending` Effects before they reach the syscall, so resolving at submit time
//! wastes the bytes; resolving at spawn time preserves "render late" as the architectural invariant.
//!
//! Pure data work — no `std::env`, no `std::process`, no I/O. The actuator's `spawn_effect` is the
//! sole production caller; sibling unit tests drive directly.
//!
//! # The `(Effect, ExecAction)` split
//!
//! Per-Effect-stable values live on [`Effect`] (`anchor_path`, `target_relative`, `sub_name`,
//! `exclude`, `diff`, `key`, `forced`, `correlation`); the per-op argv template lives on
//! [`ExecAction`]. The caller extracts the op at `effect.program.ops()[cursor]`, reaches into its
//! [`specter_core::program::SpawnBody`] for the `ExecAction(s)`, and passes both references in.
//! Single-process ops ([`specter_core::program::SpawnBody::Exec`]) hand one `ExecAction`; multi-stage
//! pipes ([`specter_core::program::SpawnBody::Pipe`]) pass each stage's [`ExecAction`] in turn.
//!
//! # `target_path` is derived, not stored
//!
//! The Effect carries `anchor_path: Arc<Path>` and `target_relative: CompactString`; the resolver
//! derives `target_path` (`${specter.path}` / `SPECTER_PATH`) by joining the two when
//! `target_relative` is non-empty, or by borrowing `anchor_path` when it is. Subtree fires (always
//! empty relative) avoid the `PathBuf` allocation entirely; PerFile fires allocate exactly once per
//! resolve, at the spawn boundary where Latest-coalesce has already filtered Effects that won't run.
//!
//! # `SPECTER_DIFF_PATH` slots in alphabetically
//!
//! The actuator's `spawn_effect` materialises the diff tmp file (path depends on the actuator
//! process's pid, so it can't be derived purely) and passes the resulting `&Path` to
//! [`resolve_step`] as `diff_path`. The resolver inserts `SPECTER_DIFF_PATH` at its alphabetical
//! position in the env vec rather than relying on the caller to append after the fact. The
//! env-order golden test (`tests::env_order_is_alphabetical`) is then a guarantee about the bytes
//! the spawned child sees, not just a property of the resolver's standalone output.
//!
//! # Argv substitution semantics
//!
//! Each [`ArgTemplate`] in `ExecAction.argv` produces one or more argv slots. The walk is
//! single-pass with a prefix accumulator:
//!
//! - **Literals** and **single-value placeholders** (`${specter.path}`, `${specter.relative}`,
//!   `${specter.anchor}`, `${specter.watch}`, `${specter.parent}`, `${specter.time}`) append to the
//!   prefix.
//! - **Multi-value placeholders** (`${specter.created}`, `${specter.deleted}`, `${specter.modified}`,
//!   `${specter.renamed_from}`, `${specter.renamed_to}`, `${specter.excluded}`) emit one argv slot
//!   per source entry, each prefixed by the accumulated prefix; then the accumulator resets to empty.
//!   The first five source from `Diff`; `${specter.excluded}` sources from `effect.exclude`.
//! - At end-of-template: if anything was ever emitted from a multi-value, any remaining accumulator
//!   becomes a standalone trailing slot. If nothing was emitted (no multi-value found), the
//!   single-slot prefix is the one slot for this template.
//! - An [`ArgTemplate`] containing a multi-value placeholder that yields zero entries (empty diff
//!   list, `diff = None`, or empty exclude list) produces zero argv slots — there's no value to
//!   emit, and dropping the surrounding prefix is the principle-of-least-surprise (`["fmt",
//!   "${specter.created}"]` with no created entries is just `["fmt"]`).
//!
//! Two multi-value placeholders within one template (exotic; e.g. `["mv ${specter.renamed_from}
//! ${specter.renamed_to}"]` as one quoted arg) expand **independently** — no parallel zip. Users
//! wanting per-pair semantics use `EffectScope::PerStableFile`.
//!
//! # Env catalog
//!
//! Every multi-value placeholder has an env-var counterpart whose value is the same source list
//! joined by `\n` (no trailing newline). Empty list (or absent diff) renders as the empty string —
//! unlike the argv path, which drops the surrounding slot. Always-emit avoids `set -u` surprises
//! and lets shell scripts iterate uniformly with `while IFS= read -r ...`. The mapping is:
//!
//! | Placeholder              | Env var                |
//! |--------------------------|------------------------|
//! | `${specter.created}`     | `SPECTER_CREATED`      |
//! | `${specter.deleted}`     | `SPECTER_DELETED`      |
//! | `${specter.modified}`    | `SPECTER_MODIFIED`     |
//! | `${specter.renamed_from}`| `SPECTER_RENAMED_FROM` |
//! | `${specter.renamed_to}`  | `SPECTER_RENAMED_TO`   |
//! | `${specter.excluded}`    | `SPECTER_EXCLUDED`     |
//!
//! The env-side surface carries segments only — inodes and rename pairing live in the line-oriented
//! `SPECTER_DIFF_PATH` tmp file.
//!
//! Segments carrying `\n` corrupt the newline-joined value — see [`crate::tmp`]'s module-level note
//! on embedded delimiters.
//!
//! # Single-pass multi-value dispatch
//!
//! Both surfaces — argv prefix-tiling and env newline-joining — funnel through
//! [`for_each_multivalue`], which iterates a placeholder's source list once and yields each value
//! as `&str` via callback. The argv consumer ([`substitute_one`]) tiles a clone of the accumulated
//! prefix per emitted value; the env consumer ([`env_multivalue`]) newline-joins into one owned
//! `String`. The `Placeholder → source list` mapping has one definition — argv and env can't drift.
//!
//! # Single rendering pass per resolve
//!
//! All four shared single-value derivations — `${specter.path}` / `SPECTER_PATH`,
//! `${specter.anchor}` / `SPECTER_ANCHOR`, `${specter.parent}` / `SPECTER_PARENT`, and
//! `${specter.time}` / `SPECTER_TIME` — are rendered exactly once at the top of [`resolve_step`]
//! and threaded through both `substitute_argv` and `build_env` as a [`Derived<'a>`] bundle whose
//! `Cow<'a, str>` fields propagate the input borrow: Subtree fires (where `target_path` and
//! `anchor_path` already live as `&'a Path`) emit `Cow::Borrowed` on the UTF-8 fast path, so
//! `SPECTER_PATH`, `SPECTER_ANCHOR`, and `SPECTER_PARENT` cost zero allocations. PerFile fires
//! (`target_path = anchor.join(segment)`, a stack-local `PathBuf`) own. The argv and env surfaces
//! share the same source string for each placeholder by construction — no risk of one surface
//! formatting differently from the other under future edits. The `Option<&Diff>` carrying the
//! diff-derived placeholders is unpacked the same way, so a pipe's per-stage resolves and an argv
//! template's per-arg loops both see the same `Arc` deref once.
//!
//! Multi-value env values ([`env_multivalue`]) follow the same discipline: empty source lists
//! short-circuit to `Cow::Borrowed("")` rather than allocating a zero-byte `String`; populated
//! lists emit `Cow::Owned` after newline-joining.

use crate::env::EnvSnapshot;
use crate::spawner::EnvVar;
use compact_str::CompactString;
use specter_core::{
    ArgPart, ArgTemplate, Diff, Effect, EffectTarget, ExecAction, Placeholder, ResourceKind,
};
use std::borrow::Cow;
use std::path::Path;
use std::time::SystemTime;

/// The argv [`resolve_step`] renders from an [`Effect`] plus one [`ExecAction`] — handed straight
/// to the spawner.
///
/// Only `Debug` is derived (the strict-env-failure test asserts via `Result::expect_err`, which
/// bounds the `Ok` type `Debug`); nothing clones, compares, or defaults it.
#[derive(Debug)]
pub(crate) struct CommandResolved {
    pub argv: Vec<String>,
}

/// Resolver-side failure. Surfaces only causes the resolver can detect without spawning a process;
/// OS-level spawn failures take the `SpawnFailureCause::OsSpawn` route at the `spawner.spawn`
/// boundary in `pool::state`. Resolver errors surface as `SpawnFailureCause::Resolver` at the same
/// boundary, distinguishing "argv could not be rendered" from "binary could not be exec'd".
///
/// v1 has one variant — strict `${env.<NAME>}` references that found neither an entry in the captured
/// snapshot nor a `:-default` literal. The caller maps every variant to `EffectOutcome::Failed {
/// exit_code: None, signal: None }` after a `tracing::error!` log line; the operator sees a clear
/// "Specter could not satisfy a required env reference" message in their journal, followed by a
/// `tracing::warn!` at the synth-Failed dispatch site carrying the `Resolver` cause discriminant.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) enum ResolveError {
    /// `${env.<NAME>}` references an unset env var with no default literal. The captured snapshot
    /// returned `None` and the template carries `default: None`. Strict mode: a misconfigured TOML
    /// must not silently render the empty string — operators opt into lenient explicitly with
    /// `${env.NAME:-}` (empty default) or `${env.NAME:-fallback}` (literal default).
    UnsetEnvVar { name: CompactString },
}

impl std::fmt::Display for ResolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnsetEnvVar { name } => write!(
                f,
                "env var `{name}` is unset and the placeholder has no `:-` default",
            ),
        }
    }
}

impl std::error::Error for ResolveError {}

/// Bundle of the four single-value derivations [`resolve_step`] renders once and threads through
/// [`substitute_argv`] and [`build_env`]. The argv and env surfaces share the same byte sequence
/// for each placeholder by construction — no second source-of-truth can drift under future edits.
///
/// All three path-derived fields share one borrow discipline: when [`Effect::target_path`] returns
/// `Cow::Borrowed` (Subtree fire shape, borrowing from `effect.anchor_path`), `path_lossy` /
/// `parent_lossy` propagate the borrow into `Cow::Borrowed(&'a str)` on the UTF-8 fast path; when
/// it returns `Cow::Owned` (PerFile fire shape, where `target_path = anchor_path.join(segment)`
/// lives only on this resolve's stack), both fields own. `anchor_lossy` borrows from
/// `effect.anchor_path` directly on the UTF-8 fast path. The unified `Cow<'a, str>` shape lets
/// [`build_env`] move each field into `EnvVar<'a>::value` regardless of which arm is present, so
/// the downstream env surface neither knows nor cares which fire shape produced the bytes.
struct Derived<'a> {
    /// `${specter.path}` / `SPECTER_PATH`.
    path_lossy: Cow<'a, str>,
    /// `${specter.anchor}` / `SPECTER_ANCHOR`.
    anchor_lossy: Cow<'a, str>,
    /// `${specter.parent}` / `SPECTER_PARENT`. Empty when `target_path.parent()` is `None` (Subtree
    /// scope at filesystem root).
    parent_lossy: Cow<'a, str>,
    /// `${specter.time}` / `SPECTER_TIME` — RFC 3339 second-precision.
    time_str: String,
}

/// Resolve one [`ExecAction`] step against its owning [`Effect`] — rendering argv plus the standard
/// `SPECTER_*` env-var set. See module docs.
///
/// `now` is sampled by the actuator's `spawn_effect` immediately before `spawner.spawn` and reused
/// for the `${specter.time}` argv slot AND the `SPECTER_TIME` env value — they agree on the
/// wall-clock instant immediately before the kernel runs the user's command. Tests inject a
/// deterministic `now`; production sources `SystemTime::now()`.
///
/// `diff_path` is the absolute path of the actuator-materialised diff tmp file when the Effect
/// carries a [`Diff`] AND the file write succeeded; otherwise `None`. The resolver emits
/// `SPECTER_DIFF_PATH` in alphabetical position iff this is `Some`, keeping env order total across
/// the spawn-time set the child observes.
pub(crate) fn resolve_step<'a>(
    effect: &'a Effect,
    exec: &'a ExecAction,
    now: SystemTime,
    diff_path: Option<&'a Path>,
    env_snapshot: &EnvSnapshot,
) -> Result<(CommandResolved, Vec<EnvVar<'a>>), ResolveError> {
    // Unpack the `Arc<Diff>` once and thread `Option<&Diff>` through both surfaces — pipe stages
    // otherwise pay the same Arc deref per stage, and per-arg argv loops pay it per template.
    let diff: Option<&'a Diff> = effect.diff().map(|d| &**d);

    // Materialise the four shared single-value derivations once. The `path_lossy` / `parent_lossy`
    // pair is rendered together so the input `Cow<'a, Path>`'s borrow arm dispatches once — see
    // [`render_target_paths`].
    let (path_lossy, parent_lossy) = render_target_paths(effect.target_path());
    let derived = Derived {
        path_lossy,
        anchor_lossy: effect.anchor_path.to_string_lossy(),
        parent_lossy,
        time_str: format_now(now),
    };

    let argv = substitute_argv(exec, effect, &derived, diff, env_snapshot)?;
    // `build_env` consumes `derived` — each field moves into the env vec instead of being cloned at
    // the per-key push site.
    let env = build_env(effect, derived, diff, diff_path);
    Ok((CommandResolved { argv }, env))
}

/// Render `target_path` and its parent as a `(path_lossy, parent_lossy)` pair of UTF-8-lossy
/// `Cow<'a, str>`s, preserving the input `Cow`'s borrow discipline.
///
/// - **`Cow::Borrowed`** (Subtree fires — `target_path` is the anchor, borrowed from
///   `effect.anchor_path`): both projections borrow with lifetime `'a` on the UTF-8 fast path (Unix
///   paths are UTF-8 in practice), so `SPECTER_PATH` and `SPECTER_PARENT` cost zero allocations.
/// - **`Cow::Owned`** (PerFile fires — `target_path` is the joined `PathBuf` living only on the
///   resolve stack): both projections allocate. `path_lossy` consumes the `PathBuf` via
///   `OsString::into_string()`, which transmutes the underlying `Vec<u8>` into a `String` on valid
///   UTF-8 (strictly cheaper than `.to_string_lossy().into_owned()`, which always copies);
///   `parent_lossy` falls through `to_string_lossy().into_owned()` on the parent slice because the
///   `PathBuf` itself is still needed for the path-side consume that follows.
///
/// Returning both projections from one match keeps the borrow analysis linear: the Subtree arm
/// yields the borrow once for both keys, so a future caller cannot split the arm across two helper
/// invocations and re-walk the same `Path` twice.
fn render_target_paths(target_path: Cow<'_, Path>) -> (Cow<'_, str>, Cow<'_, str>) {
    match target_path {
        Cow::Borrowed(p) => (
            p.to_string_lossy(),
            p.parent().map_or(Cow::Borrowed(""), Path::to_string_lossy),
        ),
        Cow::Owned(pb) => {
            // Compute `parent` while `pb` is still owned here — the closure copies the slice's
            // bytes into a fresh `String`, so once `map_or_else` returns, no borrow of `pb` remains
            // and the consume below is unencumbered.
            let parent = pb.parent().map_or_else(
                || Cow::Owned(String::new()),
                |q| Cow::Owned(q.to_string_lossy().into_owned()),
            );
            // Consume `pb` for the path side. On valid UTF-8, `OsString::into_string()` reuses the
            // underlying `Vec<u8>` as the `String`'s buffer — zero additional allocation past the
            // `PathBuf` the caller already paid for. On non-UTF-8 we fall back to lossy-copy with
            // replacement chars.
            let path = match pb.into_os_string().into_string() {
                Ok(s) => Cow::Owned(s),
                Err(os) => Cow::Owned(os.to_string_lossy().into_owned()),
            };
            (path, parent)
        }
    }
}

/// Choose the spawn cwd for `effect`.
///
/// `Command::current_dir` requires a directory; spawn fails with `ENOTDIR` otherwise. For
/// File-anchored Profiles the parent directory is the natural cwd (user scripts use `$SPECTER_PATH`
/// to locate the file). `Dir`-anchored Profiles anchor at the path itself.
///
/// Every branch is structurally a borrow of `effect.anchor_path` (the path itself, or its parent),
/// so the function returns `&Path` and the caller passes it straight to `Spawner::spawn` without an
/// owning hop.
///
/// Takes `&Effect` rather than the two flat fields it reads (`anchor_path` + `anchor_kind`): every
/// production caller already has the `Effect` in hand, and bundling avoids one parameter shape
/// where a future cwd-relevant Effect field (e.g. an explicit `working_dir` override) would
/// otherwise force a signature churn.
///
/// **`ResourceKind::Unknown` is unreachable.** The engine's `emit_effects` reads `Profile.kind` via
/// `unwrap_or(ResourceKind::Dir)` (see `transitions.rs::emit_effects`); Pending Profiles whose
/// anchor has not classified are additionally filtered at `covering_profiles` before any Effect is
/// constructed. Together these guarantee that every Effect reaching the actuator carries
/// `anchor_kind ∈ { File, Dir }`. The `Unknown` arm of this match exists as a typed tripwire — a
/// future emit path that forgets the `unwrap_or` will panic here in dev rather than surface as an
/// opaque `ENOTDIR` at spawn time.
#[must_use]
pub(crate) fn compute_cwd(effect: &Effect) -> &Path {
    let anchor_path: &Path = &effect.anchor_path;
    match effect.anchor_kind {
        ResourceKind::File => anchor_path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .unwrap_or(anchor_path),
        ResourceKind::Dir => anchor_path,
        ResourceKind::Unknown => unreachable!(
            "Effect.anchor_kind is structurally constrained to File or Dir: \
             emit_effects defaults the Profile's kind via unwrap_or(Dir), and \
             Pending Profiles are filtered at covering_profiles before any \
             Effect is constructed",
        ),
    }
}

/// Initial capacity for the resolver's per-resolve scratch buffer ([`substitute_argv`]). Sized for
/// "typical short argv slot" — long enough that most slots accumulate in-place without growth,
/// small enough that the per-resolve baseline isn't wasteful. Slots longer than this grow scratch's
/// allocation once; the grown capacity persists across subsequent slot renders within the same
/// resolve.
const SUBSTITUTE_SCRATCH_CAPACITY: usize = 64;

/// Substitute placeholders into argv slots for one step.
///
/// Threads a single [`String`] scratch buffer through [`substitute_one`]. The buffer's allocation
/// is reused across slot renders within this call: short slots accumulate without growth, and the
/// slot pushed to `out` is a `clone` of scratch's current content (exact-size allocation, no
/// over-reserved trailing capacity).
fn substitute_argv(
    exec: &ExecAction,
    effect: &Effect,
    derived: &Derived<'_>,
    diff: Option<&Diff>,
    env_snapshot: &EnvSnapshot,
) -> Result<Vec<String>, ResolveError> {
    let template = exec.argv();
    let mut argv = Vec::with_capacity(template.len());
    let mut scratch = String::with_capacity(SUBSTITUTE_SCRATCH_CAPACITY);
    for arg in template {
        substitute_one(
            arg,
            effect,
            derived,
            diff,
            env_snapshot,
            &mut argv,
            &mut scratch,
        )?;
    }
    Ok(argv)
}

/// Render one [`ArgTemplate`] into zero or more argv slots, appending to `out`. Returns `Err` on
/// the first [`ArgPart::EnvVar`] that resolves to neither a snapshot entry nor a `:-` default
/// literal — short- circuits to the caller, which fails the plan with `EffectOutcome::Failed`.
///
/// `scratch` is a caller-owned [`String`] reused across calls. Cleared at the top; built up via
/// `push_str`; cloned (not moved) into `out` at slot emission so the buffer's allocation survives
/// for the next call. The multi-value arm allocates a fresh slot per emitted entry — the shared
/// scratch prefix and the per-entry suffix are sized exactly so no growth pass is needed.
fn substitute_one(
    arg: &ArgTemplate,
    effect: &Effect,
    derived: &Derived<'_>,
    diff: Option<&Diff>,
    env_snapshot: &EnvSnapshot,
    out: &mut Vec<String>,
    scratch: &mut String,
) -> Result<(), ResolveError> {
    scratch.clear();
    let mut emitted_any = false;
    for part in arg.parts() {
        match part {
            ArgPart::Literal(s) => scratch.push_str(s),
            ArgPart::Placeholder(p) => match p {
                Placeholder::Path => scratch.push_str(&derived.path_lossy),
                Placeholder::Relative => scratch.push_str(effect.relative()),
                Placeholder::Anchor => scratch.push_str(&derived.anchor_lossy),
                Placeholder::Watch => scratch.push_str(&effect.sub_name),
                Placeholder::Parent => scratch.push_str(&derived.parent_lossy),
                Placeholder::Time => scratch.push_str(&derived.time_str),
                Placeholder::Created
                | Placeholder::Deleted
                | Placeholder::Modified
                | Placeholder::RenamedFrom
                | Placeholder::RenamedTo
                | Placeholder::Excluded => {
                    // Exact-size allocation per emitted entry. The scratch prefix is the accumulated
                    // literal+single- value content up to this point; `v` is the per-entry suffix.
                    // Sizing precisely here is one alloc per emitted slot, no growth.
                    for_each_multivalue(*p, effect, diff, |v| {
                        let mut slot = String::with_capacity(scratch.len() + v.len());
                        slot.push_str(scratch);
                        slot.push_str(v);
                        out.push(slot);
                        emitted_any = true;
                    });
                    scratch.clear();
                }
            },
            ArgPart::EnvVar { name, default } => {
                // Single-value substitution: append to `scratch`, identical to `Literal` /
                // single-value placeholder semantics. The lexer's grammar guarantees `name` is a
                // byte-identical match candidate against the snapshot's `BTreeMap` keys.
                match env_snapshot.get(name) {
                    Some(value) => scratch.push_str(value),
                    None => match default {
                        Some(d) => scratch.push_str(d),
                        None => {
                            return Err(ResolveError::UnsetEnvVar { name: name.clone() });
                        }
                    },
                }
            }
        }
    }
    // If a multi-value placeholder emitted at least one slot, a non-empty trailing scratch becomes
    // its own standalone slot. Otherwise the scratch is the single slot for this ArgTemplate. The
    // clone keeps scratch's capacity alive for the next ArgTemplate.
    if emitted_any {
        if !scratch.is_empty() {
            out.push(scratch.clone());
        }
    } else if has_multivalue(arg) {
        // A multi-value placeholder consumed but yielded zero entries: drop the entire ArgTemplate
        // (zero argv slots). Empty-diff / empty-exclude case.
    } else {
        out.push(scratch.clone());
    }
    Ok(())
}

fn has_multivalue(arg: &ArgTemplate) -> bool {
    arg.parts().iter().any(ArgPart::is_multivalue)
}

/// Walk a multi-value placeholder's source list, yielding each value as `&str` via `emit`. Zero
/// allocation: every value is borrowed in-place from `effect.exclude` (Excluded) or `diff` (the
/// five diff-derived variants).
///
/// The single-value arms (`Path`, `Relative`, `Anchor`, `Watch`, `Parent`, `Time`) are unreachable
/// by caller contract — argv routes only the six multi-value variants here, env's
/// [`env_multivalue`] likewise. The empty fallback satisfies Rust's exhaustiveness.
///
/// Argv ([`substitute_one`]) tiles a clone of the accumulated prefix per emitted value; env
/// ([`env_multivalue`]) newline-joins into one owned `String`. Both surfaces share this dispatch —
/// argv and env can't drift on the `Placeholder → source list` mapping.
fn for_each_multivalue(
    p: Placeholder,
    effect: &Effect,
    diff: Option<&Diff>,
    mut emit: impl FnMut(&str),
) {
    match p {
        Placeholder::Excluded => {
            for s in effect.exclude.iter() {
                emit(s.as_str());
            }
        }
        Placeholder::Created => {
            if let Some(d) = diff {
                for e in &d.created {
                    emit(e.segment.as_str());
                }
            }
        }
        Placeholder::Deleted => {
            if let Some(d) = diff {
                for e in &d.deleted {
                    emit(e.segment.as_str());
                }
            }
        }
        Placeholder::Modified => {
            if let Some(d) = diff {
                for e in &d.modified {
                    emit(e.segment.as_str());
                }
            }
        }
        Placeholder::RenamedFrom => {
            if let Some(d) = diff {
                for r in &d.renamed {
                    emit(r.from.segment.as_str());
                }
            }
        }
        Placeholder::RenamedTo => {
            if let Some(d) = diff {
                for r in &d.renamed {
                    emit(r.to.segment.as_str());
                }
            }
        }
        Placeholder::Path
        | Placeholder::Relative
        | Placeholder::Anchor
        | Placeholder::Watch
        | Placeholder::Parent
        | Placeholder::Time => {}
    }
}

/// Newline-joined render of a multi-value placeholder's source list, no trailing newline. Empty
/// list (or absent diff) short-circuits to `Cow::Borrowed("")` — distinguishable from "one empty
/// entry" for shell consumers reading via `while read`, and one fewer `String::new()` allocation
/// per resolve in the common path where the Sub doesn't reference diff-derived placeholders (five
/// empty multi-value categories collapse to five borrowed empty strings). Non-empty lists allocate
/// one `String` and emit `Cow::Owned`.
///
/// The `first` flag (instead of `out.is_empty()`) preserves the separator when an interior entry is
/// itself empty.
///
/// Returns `Cow<'static, str>` because neither arm borrows from the caller's inputs: the empty arm
/// is `&'static str` and the non-empty arm is fully owned. The surrounding [`EnvVar<'a>`] field
/// coerces via the standard `'static: 'a` covariance.
fn env_multivalue(p: Placeholder, effect: &Effect, diff: Option<&Diff>) -> Cow<'static, str> {
    let mut out: Option<String> = None;
    let mut first = true;
    for_each_multivalue(p, effect, diff, |v| {
        let buf = out.get_or_insert_with(String::new);
        if !first {
            buf.push('\n');
        }
        first = false;
        buf.push_str(v);
    });
    out.map_or(Cow::Borrowed(""), Cow::Owned)
}

/// Render `now` as RFC 3339 UTC second-precision (`2026-05-10T12:34:56Z`).
///
/// `humantime::format_rfc3339_seconds` panics on pre-epoch `SystemTime`. Production
/// `SystemTime::now()` never returns pre-epoch on a sane Unix host, but a hostile clock or a test
/// fixture can construct one — clamp to the Unix epoch so the spawn path can't panic. The clamp
/// emits a `tracing::warn!` at the moment it fires so a wedged clock surfaces in the operator's
/// journal instead of silently rendering `1970-01-01T00:00:00Z` for every spawn.
fn format_now(now: SystemTime) -> String {
    let clamped = now.max(SystemTime::UNIX_EPOCH);
    if clamped != now {
        tracing::warn!(
            ?now,
            "system clock pre-epoch; clamping SPECTER_TIME to UNIX_EPOCH",
        );
    }
    humantime::format_rfc3339_seconds(clamped).to_string()
}

/// Unconditional `SPECTER_*` key count emitted by [`build_env`] — every key landing alphabetically
/// except the optional `SPECTER_DIFF_PATH`, which adds one when present. Used to pre-size the env
/// [`Vec`]; the env-order golden tests (`tests::env_order_is_alphabetical` and
/// `tests::env_order_with_diff_path_is_alphabetical`) pin the actual keys, so this constant is a
/// sizing hint, not a contract — drifting it from the push count would silently trigger one `Vec`
/// resize per resolve but not break correctness. Bump when adding or removing an unconditional
/// `env.push(EnvVar { … })` site below.
const SPECTER_ENV_BASE_COUNT: usize = 15;

/// Build the standard `SPECTER_*` env-var set. Keys land in alphabetical order by name — pinned by
/// `tests::env_order_is_alphabetical`.
///
/// `SPECTER_DIFF_PATH` slots into its alphabetical position when `diff_path` is `Some`; absent when
/// `None`. The env order is total across both cases — the spawned child observes alphabetical keys
/// regardless of whether a diff is present. Order is fixed for golden-test stability and operator
/// predictability (e.g., `env | sort` matches positional `getenv` reads).
///
/// `derived` carries the four single-value derivations [`resolve_step`] pre-rendered (`path_lossy`,
/// `anchor_lossy`, `parent_lossy`, `time_str`); each field moves into the env vec at its
/// `SPECTER_*` push site so this function and [`substitute_argv`] emit identical bytes for the
/// corresponding placeholders.
///
/// Values are `Cow<'a, str>` so the resolver borrows whenever the byte sequence already lives
/// somewhere stable across the resolve call — the [`Effect`]'s `target_relative` / `sub_name`, the
/// `diff_path` argument, the static `event_kind` / `SPECTER_FORCED` literals, the empty-multivalue
/// short-circuit, and `derived`'s `path_lossy` / `anchor_lossy` / `parent_lossy` fields when
/// [`render_target_paths`] returned `Cow::Borrowed` (Subtree fires on the UTF-8 fast path). Owned
/// strings are emitted only when synthesising bytes that did not previously exist
/// (`SPECTER_CORRELATION`'s decimal render, populated multi-value joins, `SPECTER_TIME`'s RFC 3339
/// string), or when `derived`'s path-derived fields fall through PerFile's Owned arm. Keys are
/// always `&'static str`.
fn build_env<'a>(
    effect: &'a Effect,
    derived: Derived<'a>,
    diff: Option<&Diff>,
    diff_path: Option<&'a Path>,
) -> Vec<EnvVar<'a>> {
    let Derived {
        path_lossy,
        anchor_lossy,
        parent_lossy,
        time_str,
    } = derived;
    // `event_kind` derives from the fire shape — Subtree/PerFile agree with the originating Sub's
    // EffectScope by construction, so there is no second source-of-truth to consult.
    let event_kind: &'static str = match effect.target {
        EffectTarget::Subtree { .. } => "dir-subtree",
        EffectTarget::PerFile { .. } => "file",
    };
    let cap = SPECTER_ENV_BASE_COUNT + usize::from(diff_path.is_some());
    let mut env: Vec<EnvVar<'a>> = Vec::with_capacity(cap);
    env.push(EnvVar {
        key: "SPECTER_ANCHOR",
        value: anchor_lossy,
    });
    env.push(EnvVar {
        key: "SPECTER_CORRELATION",
        value: Cow::Owned(effect.correlation.as_u64().to_string()),
    });
    env.push(EnvVar {
        key: "SPECTER_CREATED",
        value: env_multivalue(Placeholder::Created, effect, diff),
    });
    env.push(EnvVar {
        key: "SPECTER_DELETED",
        value: env_multivalue(Placeholder::Deleted, effect, diff),
    });
    if let Some(p) = diff_path {
        env.push(EnvVar {
            key: "SPECTER_DIFF_PATH",
            value: p.to_string_lossy(),
        });
    }
    env.push(EnvVar {
        key: "SPECTER_EVENT_KIND",
        value: Cow::Borrowed(event_kind),
    });
    env.push(EnvVar {
        key: "SPECTER_EXCLUDED",
        value: env_multivalue(Placeholder::Excluded, effect, diff),
    });
    env.push(EnvVar {
        key: "SPECTER_FORCED",
        value: Cow::Borrowed(if effect.forced { "1" } else { "0" }),
    });
    env.push(EnvVar {
        key: "SPECTER_MODIFIED",
        value: env_multivalue(Placeholder::Modified, effect, diff),
    });
    env.push(EnvVar {
        key: "SPECTER_PARENT",
        value: parent_lossy,
    });
    env.push(EnvVar {
        key: "SPECTER_PATH",
        value: path_lossy,
    });
    env.push(EnvVar {
        key: "SPECTER_RELATIVE_PATH",
        value: Cow::Borrowed(effect.relative()),
    });
    env.push(EnvVar {
        key: "SPECTER_RENAMED_FROM",
        value: env_multivalue(Placeholder::RenamedFrom, effect, diff),
    });
    env.push(EnvVar {
        key: "SPECTER_RENAMED_TO",
        value: env_multivalue(Placeholder::RenamedTo, effect, diff),
    });
    env.push(EnvVar {
        key: "SPECTER_TIME",
        value: Cow::Owned(time_str),
    });
    env.push(EnvVar {
        key: "SPECTER_WATCH",
        value: Cow::Borrowed(effect.sub_name.as_str()),
    });
    env
}

#[cfg(test)]
mod tests {
    //! Sibling unit tests for [`super::resolve`]. Pure data work — all fixtures are inline; no I/O.
    #![allow(
        clippy::items_after_statements,
        clippy::missing_const_for_fn,
        clippy::too_many_lines
    )]

    use super::CommandResolved;
    use crate::env::EnvSnapshot;
    use crate::spawner::EnvVar;
    use compact_str::CompactString;
    use smallvec::smallvec;
    use specter_core::program::SpawnBody;
    use specter_core::testkit::single_exec_program;
    use specter_core::{
        ArgPart, ArgTemplate, CorrelationId, Diff, Effect, EffectCommon, EffectScope, EntryKind,
        EntryRef, ExecAction, FsIdentity, Placeholder, ProfileId, Rename, ResourceId, ResourceKind,
        SubId,
    };
    use std::borrow::Cow;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::SystemTime;

    /// Empty env snapshot used by tests that don't exercise `${env.<NAME>}` resolution. Constructed
    /// once via `OnceLock` and threaded through the `resolve` helper.
    fn empty_env() -> EnvSnapshot {
        EnvSnapshot::from_map::<_, &str, &str>([])
    }

    /// Convenience wrapper for tests that don't exercise `${specter.time}` / `SPECTER_TIME`
    /// rendering — pins `now` to the Unix epoch, omits the diff tmp file, and uses an empty env
    /// snapshot. Time-sensitive tests call [`super::resolve_step`] directly with the instant they
    /// want; diff-tmp-aware tests pass `diff_path: Some(_)` directly; env-aware tests build a fresh
    /// snapshot inline.
    fn resolve(e: &Effect) -> (CommandResolved, Vec<EnvVar<'_>>) {
        let exec = exec_of(e);
        super::resolve_step(e, exec, SystemTime::UNIX_EPOCH, None, &empty_env())
            .expect("test fixtures don't exercise the strict-env failure path")
    }

    /// Borrow the single [`ExecAction`] inside an [`Effect`]'s program. Tests build effects with
    /// exactly one `SpawnBody::Exec` op; this is a fixture-side accessor, not a production API.
    fn exec_of(e: &Effect) -> &ExecAction {
        match &e.program.ops()[0].body() {
            SpawnBody::Exec(exec) => exec,
            SpawnBody::Pipe(_) => panic!("test fixtures use only Exec body"),
        }
    }

    /// The resolver derives `target_path` from `(anchor_path, relative())` at spawn time. Tests
    /// pass the anchor + relative pair; the helper does no extra dispatch.
    ///
    /// `scope` selects the `EffectTarget` shape (Subtree ⇒ no per-file segment, PerStableFile ⇒
    /// per-file segment); the resolver then derives `SPECTER_EVENT_KIND` from the shape.
    fn make_effect(
        sub_name: &str,
        scope: EffectScope,
        argv: Vec<ArgTemplate>,
        anchor_path: &Path,
        target_relative: &str,
        forced: bool,
        correlation: CorrelationId,
        diff: Option<Arc<Diff>>,
    ) -> Effect {
        let common = EffectCommon {
            sub: SubId::default(),
            profile: ProfileId::default(),
            anchor: ResourceId::default(),
            correlation,
            forced,
            capture_output: false,
            sub_name: CompactString::from(sub_name),
            program: single_exec_program(argv),
            anchor_path: Arc::from(anchor_path.to_path_buf()),
            anchor_kind: ResourceKind::Dir,
            exclude: Arc::from(Vec::<CompactString>::new()),
        };
        match scope {
            EffectScope::SubtreeRoot => Effect::subtree(common, diff),
            EffectScope::PerStableFile => {
                // PerFile diff is mandatory. Callers that passed `None` did not reference a
                // diff-derived placeholder; an empty `Diff::default()` renders those placeholders
                // identically to the old absent-diff path.
                let diff = diff.unwrap_or_else(|| Arc::new(Diff::default()));
                Effect::per_file(
                    common,
                    ResourceId::default(),
                    CompactString::from(target_relative),
                    diff,
                )
            }
        }
    }

    fn lit(s: &str) -> ArgPart {
        ArgPart::literal(s)
    }
    fn ph(p: Placeholder) -> ArgPart {
        ArgPart::Placeholder(p)
    }
    fn arg(parts: Vec<ArgPart>) -> ArgTemplate {
        ArgTemplate::new(parts)
    }

    fn entry_ref(seg: &str, inode: u64) -> EntryRef {
        EntryRef {
            segment: CompactString::from(seg),
            kind: EntryKind::File,
            fs_id: FsIdentity::synthetic(inode, 0),
        }
    }

    // ---------- argv substitution ----------

    #[test]
    fn resolve_simple_literal_passes_through() {
        let e = make_effect(
            "build",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("make")])],
            Path::new("/proj"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(cmd.argv, vec!["make".to_string()]);
    }

    #[test]
    fn resolve_with_path_placeholder() {
        let e = make_effect(
            "fmt",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Path)])],
            Path::new("/proj"),
            "src/a.c",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec!["fmt".to_string(), "/proj/src/a.c".to_string()]
        );
    }

    #[test]
    fn resolve_with_relative_placeholder() {
        let e = make_effect(
            "log",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("log")]), arg(vec![ph(Placeholder::Relative)])],
            Path::new("/proj"),
            "src/a.c",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(cmd.argv, vec!["log".to_string(), "src/a.c".to_string()]);
    }

    #[test]
    fn resolve_with_anchor_placeholder() {
        let e = make_effect(
            "build",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("build")]), arg(vec![ph(Placeholder::Anchor)])],
            Path::new("/proj"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(cmd.argv, vec!["build".to_string(), "/proj".to_string()]);
    }

    // ---------- ${specter.excluded} / SPECTER_EXCLUDED ----------

    #[test]
    fn resolve_excluded_one_arg_per_pattern() {
        // `--exclude=${specter.excluded}` tiles the literal prefix per pattern, mirroring the
        // diff-derived multi-value behaviour.
        let mut e = make_effect(
            "rsync",
            EffectScope::SubtreeRoot,
            vec![
                arg(vec![lit("rsync")]),
                arg(vec![lit("--exclude="), ph(Placeholder::Excluded)]),
                arg(vec![lit("/src/")]),
            ],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        e.exclude = vec![
            CompactString::from("*.tmp"),
            CompactString::from("cache/"),
            CompactString::from("**/.git/"),
        ]
        .into();
        let (cmd, _) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec![
                "rsync".to_string(),
                "--exclude=*.tmp".to_string(),
                "--exclude=cache/".to_string(),
                "--exclude=**/.git/".to_string(),
                "/src/".to_string(),
            ]
        );
    }

    #[test]
    fn resolve_excluded_empty_drops_slot() {
        // Empty exclude list mirrors empty-diff: drop the entire `--exclude=${specter.excluded}`
        // slot rather than emit `--exclude=`.
        let e = make_effect(
            "rsync",
            EffectScope::SubtreeRoot,
            vec![
                arg(vec![lit("rsync")]),
                arg(vec![lit("--exclude="), ph(Placeholder::Excluded)]),
                arg(vec![lit("/src/")]),
            ],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        // exclude defaults empty in make_effect.
        let (cmd, _) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec!["rsync".to_string(), "/src/".to_string()],
            "empty ${{specter.excluded}} drops the surrounding slot"
        );
    }

    #[test]
    fn env_exclude_newline_separated() {
        // Newline-separated source strings, no trailing newline. Survives any pattern content
        // (commas, spaces, apostrophes) that's legal in glob source strings.
        let mut e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        e.exclude = vec![
            CompactString::from("*.tmp"),
            CompactString::from("cache/"),
            CompactString::from("**/.git/"),
        ]
        .into();
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_EXCLUDED")
                .unwrap()
                .value,
            "*.tmp\ncache/\n**/.git/",
            "no trailing newline; entries joined by single \\n",
        );
    }

    #[test]
    fn env_exclude_empty_is_empty_string() {
        // Empty exclude list ⇒ empty env value, NOT a blank line.
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_EXCLUDED")
                .unwrap()
                .value,
            "",
        );
    }

    // ---------- ${specter.time} / SPECTER_TIME ----------

    /// Unix timestamp 1_700_000_000 = 2023-11-14T22:13:20Z. Chosen for readability in the assert;
    /// the format is RFC 3339 second-precision.
    const FIXED_NOW_SECS: u64 = 1_700_000_000;
    const FIXED_NOW_RFC3339: &str = "2023-11-14T22:13:20Z";

    #[test]
    fn resolve_time_uses_injected_now() {
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(FIXED_NOW_SECS);
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![ph(Placeholder::Time)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = super::resolve_step(&e, exec_of(&e), now, None, &empty_env())
            .expect("test fixtures don\'t exercise the strict-env failure path");
        assert_eq!(cmd.argv, vec![FIXED_NOW_RFC3339.to_owned()]);
    }

    #[test]
    fn env_specter_time_uses_injected_now() {
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(FIXED_NOW_SECS);
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = super::resolve_step(&e, exec_of(&e), now, None, &empty_env())
            .expect("test fixtures don\'t exercise the strict-env failure path");
        assert_eq!(
            env.iter().find(|e| e.key == "SPECTER_TIME").unwrap().value,
            FIXED_NOW_RFC3339
        );
    }

    #[test]
    fn format_now_clamps_pre_epoch() {
        // humantime::format_rfc3339_seconds panics on pre-epoch SystemTime. Production never sees
        // pre-epoch on a sane Unix host, but tests can construct one. The resolver clamps to
        // UNIX_EPOCH so the spawn path can't panic on a hostile clock.
        let pre = SystemTime::UNIX_EPOCH - std::time::Duration::from_secs(1);
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![ph(Placeholder::Time)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = super::resolve_step(&e, exec_of(&e), pre, None, &empty_env())
            .expect("test fixtures don\'t exercise the strict-env failure path");
        assert_eq!(cmd.argv, vec!["1970-01-01T00:00:00Z".to_owned()]);
    }

    // ---------- ${specter.parent} ----------
    //
    // Documented edge cases (see Placeholder::Parent rustdoc):
    //   PerFile  | /anchor  | foo.rs       | ${specter.parent} = /anchor
    //   PerFile  | /        | foo.rs       | ${specter.parent} = /        (NOT empty)
    //   Subtree  | /anchor  | n/a          | ${specter.parent} = /
    //   Subtree  | /        | n/a          | ${specter.parent} = ""       (only empty case)

    #[test]
    fn resolve_parent_is_target_dir_for_perfile() {
        // PerFile target = anchor.join(segment); ${specter.parent} = the directory immediately
        // containing the file that triggered the fire.
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![ph(Placeholder::Parent)])],
            Path::new("/anchor"),
            "foo.rs",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["/anchor".to_string()]);
    }

    #[test]
    fn resolve_parent_is_anchor_parent_for_subtree() {
        // Subtree target_path == anchor_path; ${specter.parent} = parent of the anchor (one level
        // above the watch root).
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![ph(Placeholder::Parent)])],
            Path::new("/proj/sub"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["/proj".to_string()]);
    }

    #[test]
    fn resolve_parent_for_perfile_at_root_is_root() {
        // Filesystem-root anchor with PerFile scope: target_path = "/foo.rs", parent = "/" (NOT
        // empty). Guards against the easy misreading that any anchor at root yields empty
        // ${specter.parent}.
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![ph(Placeholder::Parent)])],
            Path::new("/"),
            "foo.rs",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["/".to_string()]);
    }

    #[test]
    fn resolve_parent_empty_only_for_subtree_at_root() {
        // The only configuration that yields an empty ${specter.parent}: Subtree scope anchored at
        // filesystem root (target_path = "/", which has no parent).
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![ph(Placeholder::Parent)])],
            Path::new("/"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = resolve(&e);
        // Empty parent → ArgTemplate produces a single empty argv slot (single-value placeholders
        // never drop the slot, only multi-values with zero entries do).
        assert_eq!(cmd.argv, vec![String::new()]);
    }

    #[test]
    fn env_parent_empty_only_for_subtree_at_root() {
        // SPECTER_PARENT mirrors ${specter.parent}: empty string only at fs root for Subtree scope;
        // "/" everywhere else at the root level.
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_PARENT")
                .unwrap()
                .value,
            ""
        );
    }

    #[test]
    fn env_parent_for_perfile_is_target_directory() {
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("noop")])],
            Path::new("/anchor"),
            "src/foo.rs",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_PARENT")
                .unwrap()
                .value,
            "/anchor/src"
        );
    }

    #[test]
    fn resolve_substitutes_watch_name() {
        // `${specter.watch}` substitutes `effect.sub_name` — mirrors `$SPECTER_WATCH` env value but
        // in argv form.
        let e = make_effect(
            "build",
            EffectScope::SubtreeRoot,
            vec![
                arg(vec![lit("notify-send")]),
                arg(vec![ph(Placeholder::Watch), lit(" settled")]),
            ],
            Path::new("/proj"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec!["notify-send".to_string(), "build settled".to_string()]
        );
    }

    #[test]
    fn resolve_with_concatenated_literal_and_placeholder() {
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("--input="), ph(Placeholder::Path)])],
            Path::new("/proj"),
            "a.c",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(cmd.argv, vec!["--input=/proj/a.c".to_string()]);
    }

    #[test]
    fn resolve_with_created_expands_to_n_argv() {
        let diff = Diff {
            created: smallvec![
                entry_ref("a.rs", 1),
                entry_ref("b.rs", 2),
                entry_ref("c.rs", 3)
            ],
            ..Default::default()
        };
        let e = make_effect(
            "fmt",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
            Path::new("/proj"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (cmd, _env) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec![
                "fmt".to_string(),
                "a.rs".to_string(),
                "b.rs".to_string(),
                "c.rs".to_string()
            ]
        );
    }

    #[test]
    fn resolve_with_deleted_expands_to_n_argv() {
        let diff = Diff {
            deleted: smallvec![entry_ref("x", 9), entry_ref("y", 10)],
            ..Default::default()
        };
        let e = make_effect(
            "rmlog",
            EffectScope::PerStableFile,
            vec![arg(vec![ph(Placeholder::Deleted)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["x".to_string(), "y".to_string()]);
    }

    #[test]
    fn resolve_with_modified_expands_to_n_argv() {
        let diff = Diff {
            modified: smallvec![entry_ref("m.rs", 1)],
            ..Default::default()
        };
        let e = make_effect(
            "lint",
            EffectScope::PerStableFile,
            vec![arg(vec![ph(Placeholder::Modified)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["m.rs".to_string()]);
    }

    #[test]
    fn resolve_with_renamed_from_and_to_expands_independently() {
        let diff = Diff {
            renamed: smallvec![
                Rename {
                    from: entry_ref("a", 1),
                    to: entry_ref("A", 1),
                },
                Rename {
                    from: entry_ref("b", 2),
                    to: entry_ref("B", 2),
                },
            ],
            ..Default::default()
        };
        let e = make_effect(
            "mv",
            EffectScope::PerStableFile,
            vec![
                arg(vec![lit("mv")]),
                arg(vec![ph(Placeholder::RenamedFrom)]),
                arg(vec![ph(Placeholder::RenamedTo)]),
            ],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec![
                "mv".to_string(),
                "a".to_string(),
                "b".to_string(),
                "A".to_string(),
                "B".to_string()
            ]
        );
    }

    #[test]
    fn resolve_with_diff_placeholder_and_no_diff_yields_zero_args() {
        let e = make_effect(
            "fmt",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["fmt".to_string()]);
    }

    #[test]
    fn resolve_with_empty_diff_placeholder_yields_zero_args() {
        let e = make_effect(
            "fmt",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("fmt")]), arg(vec![ph(Placeholder::Created)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(Diff::default())),
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["fmt".to_string()]);
    }

    #[test]
    fn resolve_with_multivalue_in_separate_args_emits_literals_as_standalone_slots() {
        let diff = Diff {
            created: smallvec![entry_ref("a", 1), entry_ref("b", 2)],
            ..Default::default()
        };
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![
                arg(vec![lit("pre")]),
                arg(vec![ph(Placeholder::Created)]),
                arg(vec![lit("post")]),
            ],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(
            cmd.argv,
            vec![
                "pre".to_string(),
                "a".to_string(),
                "b".to_string(),
                "post".to_string()
            ]
        );
    }

    #[test]
    fn resolve_with_multivalue_having_prefix_literal_tiles_per_value() {
        let diff = Diff {
            created: smallvec![entry_ref("a", 1), entry_ref("b", 2)],
            ..Default::default()
        };
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (cmd, _) = resolve(&e);
        assert_eq!(cmd.argv, vec!["--out=a".to_string(), "--out=b".to_string()]);
    }

    #[test]
    fn resolve_with_multivalue_having_prefix_and_empty_diff_yields_zero_slots() {
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("--out="), ph(Placeholder::Created)])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(Diff::default())),
        );
        let (cmd, _) = resolve(&e);
        assert!(cmd.argv.is_empty());
    }

    // ---------- diff-derived env vars ----------

    #[test]
    fn env_specter_created_newline_separated() {
        // Diff-derived multi-value env var mirrors the argv form: each entry's segment, joined by
        // `\n`, no trailing newline. Empty list ⇒ empty string (asserted in env_diff_lists_*);
        // populated list ⇒ the segments.
        let diff = Diff {
            created: smallvec![
                entry_ref("a.rs", 1),
                entry_ref("src/b.rs", 2),
                entry_ref("c", 3),
            ],
            ..Default::default()
        };
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_CREATED")
                .unwrap()
                .value,
            "a.rs\nsrc/b.rs\nc",
            "no trailing newline; entries joined by single \\n",
        );
    }

    #[test]
    fn env_specter_deleted_and_modified_render_their_categories() {
        // One Diff carrying entries for two categories; each env var pulls from its own list.
        // Asserts the dispatch in `diff_env_segs` doesn't cross-contaminate.
        let diff = Diff {
            deleted: smallvec![entry_ref("d1", 1), entry_ref("d2", 2)],
            modified: smallvec![entry_ref("m1", 3)],
            ..Default::default()
        };
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_DELETED")
                .unwrap()
                .value,
            "d1\nd2",
        );
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_MODIFIED")
                .unwrap()
                .value,
            "m1",
        );
        // Categories not populated stay empty even though the diff is present.
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_CREATED")
                .unwrap()
                .value,
            "",
        );
    }

    #[test]
    fn env_specter_renamed_from_and_to_use_correct_sides() {
        // Two renames, each with distinct from/to segments. The two env vars must each pull their
        // respective side; cross-contamination would mean the from/to projection in
        // `diff_env_renames` is broken.
        let diff = Diff {
            renamed: smallvec![
                Rename {
                    from: entry_ref("old1", 1),
                    to: entry_ref("new1", 1),
                },
                Rename {
                    from: entry_ref("old2", 2),
                    to: entry_ref("new2", 2),
                },
            ],
            ..Default::default()
        };
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_RENAMED_FROM")
                .unwrap()
                .value,
            "old1\nold2",
        );
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_RENAMED_TO")
                .unwrap()
                .value,
            "new1\nnew2",
        );
    }

    #[test]
    fn env_diff_lists_empty_when_no_diff() {
        // `Effect.diff = None` (Sub doesn't reference any diff-derived placeholder and isn't
        // `per-stable-file`). All five list env vars emit as empty strings — always-emit policy
        // mirrors SPECTER_EXCLUDED and avoids `set -u` surprises in the spawned shell. The
        // `Some(Diff::default())` variant exits the same way through `join_with_newlines`'s
        // empty-iter branch (already pinned by `env_exclude_empty_is_empty_string`).
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        for k in [
            "SPECTER_CREATED",
            "SPECTER_DELETED",
            "SPECTER_MODIFIED",
            "SPECTER_RENAMED_FROM",
            "SPECTER_RENAMED_TO",
        ] {
            assert_eq!(
                env.iter().find(|e| e.key == k).unwrap().value,
                "",
                "{k} must be empty when Effect.diff is None",
            );
        }
    }

    // ---------- env vars ----------

    #[test]
    fn env_contains_specter_path_for_subtree_root() {
        let e = make_effect(
            "build",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("make")])],
            Path::new("/proj"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
        assert_eq!(path.value, "/proj");
    }

    #[test]
    fn env_contains_specter_path_for_per_stable_file() {
        let e = make_effect(
            "fmt",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("fmt")])],
            Path::new("/proj"),
            "a.c",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
        assert_eq!(path.value, "/proj/a.c");
    }

    #[test]
    fn env_specter_relative_path_empty_for_subtree_root() {
        let e = make_effect(
            "b",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("x")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_RELATIVE_PATH")
                .unwrap()
                .value,
            ""
        );
    }

    #[test]
    fn env_specter_relative_path_for_per_stable_file() {
        let e = make_effect(
            "f",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("x")])],
            Path::new("/p"),
            "src/a.c",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_RELATIVE_PATH")
                .unwrap()
                .value,
            "src/a.c"
        );
    }

    #[test]
    fn env_specter_anchor_for_both_scopes() {
        for scope in [EffectScope::SubtreeRoot, EffectScope::PerStableFile] {
            let e = make_effect(
                "x",
                scope,
                vec![arg(vec![lit("y")])],
                Path::new("/anchor/dir"),
                "",
                false,
                CorrelationId::from(1),
                None,
            );
            let (_, env) = resolve(&e);
            let v = env.iter().find(|e| e.key == "SPECTER_ANCHOR").unwrap();
            assert_eq!(v.value, "/anchor/dir", "scope = {scope:?}");
        }
    }

    #[test]
    fn env_specter_watch_uses_sub_name() {
        let e = make_effect(
            "build",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter().find(|e| e.key == "SPECTER_WATCH").unwrap().value,
            "build"
        );
    }

    #[test]
    fn env_specter_forced_zero_when_unforced() {
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_FORCED")
                .unwrap()
                .value,
            "0"
        );
    }

    #[test]
    fn env_specter_forced_one_when_forced() {
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            true,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_FORCED")
                .unwrap()
                .value,
            "1"
        );
    }

    #[test]
    fn env_specter_event_kind_dir_subtree_for_subtree_root() {
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_EVENT_KIND")
                .unwrap()
                .value,
            "dir-subtree"
        );
    }

    #[test]
    fn env_specter_event_kind_file_for_per_stable_file() {
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "a",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_EVENT_KIND")
                .unwrap()
                .value,
            "file"
        );
    }

    #[test]
    fn env_specter_correlation_decimal_for_v1() {
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(42),
            None,
        );
        let (_, env) = resolve(&e);
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_CORRELATION")
                .unwrap()
                .value,
            "42"
        );
    }

    #[test]
    fn env_does_not_contain_specter_diff_path() {
        let diff = Diff {
            created: smallvec![entry_ref("a", 1)],
            ..Default::default()
        };
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "a",
            false,
            CorrelationId::from(1),
            Some(Arc::new(diff)),
        );
        let (_, env) = resolve(&e);
        assert!(env.iter().all(|e| e.key != "SPECTER_DIFF_PATH"));
    }

    #[test]
    fn env_order_is_alphabetical() {
        let e = make_effect(
            "watch",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        let keys: Vec<&str> = env.iter().map(|e| e.key).collect();
        assert_eq!(
            keys,
            vec![
                "SPECTER_ANCHOR",
                "SPECTER_CORRELATION",
                "SPECTER_CREATED",
                "SPECTER_DELETED",
                "SPECTER_EVENT_KIND",
                "SPECTER_EXCLUDED",
                "SPECTER_FORCED",
                "SPECTER_MODIFIED",
                "SPECTER_PARENT",
                "SPECTER_PATH",
                "SPECTER_RELATIVE_PATH",
                "SPECTER_RENAMED_FROM",
                "SPECTER_RENAMED_TO",
                "SPECTER_TIME",
                "SPECTER_WATCH",
            ]
        );
    }

    #[test]
    fn env_order_with_diff_path_is_alphabetical() {
        // With `diff_path: Some(_)`, SPECTER_DIFF_PATH joins the env in alphabetical position
        // (between SPECTER_DELETED and SPECTER_EVENT_KIND), keeping a total order across the
        // spawn-time set the child observes.
        let e = make_effect(
            "watch",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("y")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let diff_path = Path::new("/tmp/specter-1234-deadbeef.diff");
        let (_, env) = super::resolve_step(
            &e,
            exec_of(&e),
            SystemTime::UNIX_EPOCH,
            Some(diff_path),
            &empty_env(),
        )
        .expect("test fixtures don\'t exercise the strict-env failure path");
        let keys: Vec<&str> = env.iter().map(|e| e.key).collect();
        assert_eq!(
            keys,
            vec![
                "SPECTER_ANCHOR",
                "SPECTER_CORRELATION",
                "SPECTER_CREATED",
                "SPECTER_DELETED",
                "SPECTER_DIFF_PATH",
                "SPECTER_EVENT_KIND",
                "SPECTER_EXCLUDED",
                "SPECTER_FORCED",
                "SPECTER_MODIFIED",
                "SPECTER_PARENT",
                "SPECTER_PATH",
                "SPECTER_RELATIVE_PATH",
                "SPECTER_RENAMED_FROM",
                "SPECTER_RENAMED_TO",
                "SPECTER_TIME",
                "SPECTER_WATCH",
            ]
        );
        assert_eq!(
            env.iter()
                .find(|e| e.key == "SPECTER_DIFF_PATH")
                .unwrap()
                .value,
            "/tmp/specter-1234-deadbeef.diff"
        );
    }

    // ---------- Cow borrow discipline ----------
    //
    // When `Effect::target_path` is `Cow::Borrowed` (Subtree fire), `SPECTER_PATH` /
    // `SPECTER_PARENT` propagate the borrow into `Cow::Borrowed` on the UTF-8 fast path; when it is
    // `Cow::Owned` (PerFile fire), both fields own. The two assertions below pin one Subtree case
    // (path + parent in one resolve) and one PerFile case so a future regression that forces an
    // unconditional `into_owned()` on either field surfaces in the test suite. The empty-multivalue
    // short-circuit gets its own small assertion since the borrow-vs-owned property there is
    // independent of `target_path`.

    #[test]
    fn env_specter_path_and_parent_borrow_for_subtree() {
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/proj/sub"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
        let parent = env.iter().find(|e| e.key == "SPECTER_PARENT").unwrap();
        assert!(
            matches!(path.value, Cow::Borrowed(_)),
            "Subtree SPECTER_PATH should borrow from effect.anchor_path on the UTF-8 fast path",
        );
        assert!(
            matches!(parent.value, Cow::Borrowed(_)),
            "Subtree SPECTER_PARENT should borrow from effect.anchor_path on the UTF-8 fast path",
        );
        assert_eq!(path.value, "/proj/sub");
        assert_eq!(parent.value, "/proj");
    }

    #[test]
    fn env_specter_path_and_parent_own_for_perfile() {
        // PerFile `target_path = anchor.join(segment)` is a freshly-joined `PathBuf` living only on
        // the resolve stack; both fields must own their bytes for the env vec to outlive the
        // resolve call.
        let e = make_effect(
            "x",
            EffectScope::PerStableFile,
            vec![arg(vec![lit("noop")])],
            Path::new("/proj"),
            "a.c",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        let path = env.iter().find(|e| e.key == "SPECTER_PATH").unwrap();
        let parent = env.iter().find(|e| e.key == "SPECTER_PARENT").unwrap();
        assert!(matches!(path.value, Cow::Owned(_)));
        assert!(matches!(parent.value, Cow::Owned(_)));
        assert_eq!(path.value, "/proj/a.c");
        assert_eq!(parent.value, "/proj");
    }

    #[test]
    fn env_multivalue_borrows_empty_string_when_no_entries() {
        // `env_multivalue` short-circuits the empty case to `Cow::Borrowed("")` instead of
        // allocating an empty `String`. The no-diff resolve emits six empty multi-value env vars;
        // this saves six `String::new()` allocations per resolve on the common path for Subs that
        // don't reference diff placeholders. One probe per category is enough — they all route
        // through the same helper.
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![lit("noop")])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (_, env) = resolve(&e);
        for k in [
            "SPECTER_CREATED",
            "SPECTER_DELETED",
            "SPECTER_MODIFIED",
            "SPECTER_RENAMED_FROM",
            "SPECTER_RENAMED_TO",
            "SPECTER_EXCLUDED",
        ] {
            let v = env.iter().find(|e| e.key == k).unwrap();
            assert!(
                matches!(v.value, Cow::Borrowed(_)),
                "{k} must be Cow::Borrowed when its source list is empty",
            );
        }
    }

    // ---------- ${env.<NAME>} ----------

    /// `${env.NAME}` resolves to the snapshot's value when present. Default-bearing form is
    /// exercised below; together they cover both lexer branches in the resolver pass.
    #[test]
    fn resolve_env_var_substitutes_from_snapshot() {
        let env = EnvSnapshot::from_map([("HOME", "/home/op")]);
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![ArgPart::EnvVar {
                name: "HOME".into(),
                default: None,
            }])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = super::resolve_step(&e, exec_of(&e), SystemTime::UNIX_EPOCH, None, &env)
            .expect("HOME present in snapshot");
        assert_eq!(cmd.argv, vec!["/home/op".to_string()]);
    }

    /// Strict default: unset env var with no `:-` default fails the resolve — the caller maps
    /// `ResolveError::UnsetEnvVar` to `EffectOutcome::Failed`.
    #[test]
    fn resolve_env_var_unset_without_default_returns_unset_env_var_error() {
        let env = EnvSnapshot::from_map::<_, &str, &str>([]);
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![arg(vec![ArgPart::EnvVar {
                name: "MISSING".into(),
                default: None,
            }])],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let err = super::resolve_step(&e, exec_of(&e), SystemTime::UNIX_EPOCH, None, &env)
            .expect_err("unset env var must fail strict resolve");
        assert_eq!(
            err,
            crate::resolve::ResolveError::UnsetEnvVar {
                name: "MISSING".into(),
            }
        );
    }

    /// Unset env var with a `:-default` renders the default literal — explicit lenient opt-in.
    /// Empty default (`${env.X:-}`) renders empty.
    #[test]
    fn resolve_env_var_unset_with_default_renders_default() {
        let env = EnvSnapshot::from_map::<_, &str, &str>([]);
        let e = make_effect(
            "x",
            EffectScope::SubtreeRoot,
            vec![
                arg(vec![ArgPart::EnvVar {
                    name: "MISSING".into(),
                    default: Some("/tmp".into()),
                }]),
                arg(vec![ArgPart::EnvVar {
                    name: "ALSO_MISSING".into(),
                    default: Some(CompactString::new("")),
                }]),
            ],
            Path::new("/p"),
            "",
            false,
            CorrelationId::from(1),
            None,
        );
        let (cmd, _) = super::resolve_step(&e, exec_of(&e), SystemTime::UNIX_EPOCH, None, &env)
            .expect("default rendered when env unset");
        assert_eq!(cmd.argv, vec!["/tmp".to_string(), String::new()]);
    }
}
