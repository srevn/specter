use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::{ArgPart, ArgTemplate, Placeholder};
use std::fmt;

/// Namespace prefix that opens the Specter placeholder grammar. Anything
/// not matching this exact byte sequence (including the trailing dot)
/// falls through as a literal `$`.
const NS_SPECTER: &str = "${specter.";

/// Namespace prefix for the operator-env placeholder grammar
/// (`${env.NAME}` / `${env.NAME:-default}`). Resolved against the
/// actuator's `EnvSnapshot` at spawn time; the lexer's job is to
/// distinguish a well-formed env reference from arbitrary `$`-bearing
/// shell syntax.
const NS_ENV: &str = "${env.";

/// `:-` is the boundary between an `${env.NAME}` and an optional literal
/// default. Mirrors POSIX shell `${VAR:-default}` for operator
/// familiarity. The default is a frozen literal — nested placeholders
/// are rejected at the lexer (no `${env.HOME:-${env.USER}}`); operators
/// wanting composition wrap with shell.
const ENV_DEFAULT_SEP: &str = ":-";

/// Failures the lexer can surface from inside a recognised namespace
/// placeholder.
///
/// Outside the recognised namespaces (`${specter.<name>}`,
/// `${env.<name>}`), the lexer never errors — any other `$`-bearing
/// byte sequence passes through verbatim, freeing operators to write
/// arbitrary shell / awk / perl `$` syntax in argv slots without a
/// Specter typo tax.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TemplateError {
    /// `${specter.<name>}` where `<name>` is not in the placeholder
    /// catalog. Catches mistypes (`${specter.ptah}`) and members of the
    /// catalog that aren't yet implemented.
    UnknownPlaceholder { name: String },
    /// A recognised namespace opener (`${specter.…}` or `${env.…}`)
    /// reached end-of-string without a closing `}`. `partial` is the
    /// substring from the opening `${` to end-of-input.
    UnterminatedPlaceholder { partial: String },
    /// `${specter.}` or `${env.}` — the namespace was opened but no
    /// name follows. The empty-name policy is shared across namespaces.
    EmptyPlaceholderName,
    /// `${specter.<name>}` where `<name>` contains a character outside
    /// `[a-z0-9_]`. The first offending character is reported.
    InvalidPlaceholderChar { name: String, ch: char },
    /// `${env.<NAME>}` where `<NAME>` is not a well-formed env-var
    /// identifier (`[A-Za-z_][A-Za-z0-9_]*`). The first offending
    /// character is reported.
    InvalidEnvName { name: String, ch: char },
    /// `${env.<NAME>:-<default>}` whose default contains a reserved
    /// single character (`$`, `{`) or any ASCII / Unicode control
    /// character (`is_control()`). Nested substitution inside defaults
    /// isn't supported in v1 — the default is a literal byte sequence
    /// up to the first closing `}`. Operators wanting composition wrap
    /// with shell. Control chars are rejected so an unintended newline
    /// or tab in a default doesn't silently make it into argv.
    InvalidEnvDefault { default: String, ch: char },
    /// `${env.<NAME>:-<default>}` whose default contains the literal
    /// `:-` separator substring. The separator is reserved for the
    /// name/default split itself; allowing it inside defaults would
    /// invite ambiguity (`${env.X:-a:-b}` could read as name=X /
    /// default=`a:-b` or name=X / default=`a` / trailing). v1 rejects
    /// the case outright. Operators wanting `:-` literal wrap with
    /// shell.
    EnvDefaultContainsSeparator { default: String },
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPlaceholder { name } => {
                write!(f, "unknown placeholder `${{specter.{name}}}`")
            }
            Self::UnterminatedPlaceholder { partial } => {
                write!(f, "unterminated placeholder `{partial}` (missing `}}`)")
            }
            Self::EmptyPlaceholderName => f.write_str(
                "empty placeholder name (expected `${specter.<name>}` or `${env.<name>}`)",
            ),
            Self::InvalidPlaceholderChar { name, ch } => {
                write!(
                    f,
                    "invalid character `{ch}` in placeholder name `{name}` \
                     (expected `[a-z0-9_]`)",
                )
            }
            Self::InvalidEnvName { name, ch } => {
                write!(
                    f,
                    "invalid character `{ch}` in `${{env.{name}}}` \
                     (expected `[A-Za-z_][A-Za-z0-9_]*`)",
                )
            }
            Self::InvalidEnvDefault { default, ch } => {
                // Render both the offending char and the surrounding
                // default via `escape_default` so control chars (e.g.,
                // `\n`, `\t`, DEL) appear as readable escapes rather
                // than literal bytes that would wreck the log line.
                // Printable chars like `$` / `{` pass through unchanged.
                let ch_repr: String = ch.escape_default().collect();
                let default_repr: String = default.escape_default().collect();
                write!(
                    f,
                    "invalid character `{ch_repr}` in `${{env.<NAME>:-<default>}}` literal `{default_repr}` \
                     (defaults are literal-only; `$`, `{{`, and control characters are reserved)",
                )
            }
            Self::EnvDefaultContainsSeparator { default } => {
                let default_repr: String = default.escape_default().collect();
                write!(
                    f,
                    "default `{default_repr}` contains reserved separator `:-` \
                     in `${{env.<NAME>:-<default>}}` (defaults containing `:-` are \
                     unrepresentable in v1; wrap with shell for composition)",
                )
            }
        }
    }
}

impl std::error::Error for TemplateError {}

/// Parse one TOML argv string into an [`ArgTemplate`].
///
/// Recognises three `$`-prefix patterns; everything else passes through
/// as a literal:
///
/// - `${specter.<name>}` — the Specter placeholder namespace. `<name>`
///   must be a non-empty `[a-z0-9_]` sequence and must match a catalog
///   entry (`path`, `relative`, `anchor`, `watch`, `parent`, `time`,
///   `created`, `deleted`, `modified`, `renamed_from`, `renamed_to`,
///   `excluded`); anything else inside the namespace returns an error.
/// - `${env.<NAME>}` or `${env.<NAME>:-<default>}` — operator-env
///   reference resolved at spawn time against the actuator's captured
///   [`crate::EnvSnapshot`]. `<NAME>` must be a `[A-Za-z_][A-Za-z0-9_]*`
///   identifier. Strict by default: missing env var with no default ⇒
///   the plan fails (`EffectOutcome::Failed`). Explicit lenient opt-in
///   via `${env.HOME:-}` (empty default) or `${env.HOME:-/tmp}`.
/// - `$$` — escapes a literal `$`. The only way to write a single `$`
///   that the spawned shell will not interpret as the start of an env
///   var name; doubles up shell `$$` (PID expansion) as `$$$$`.
///
/// Every other `$`-bearing sequence is a literal: `$HOME`, `$path`,
/// `$5`, `${VAR}`, `${specter}` (no dot), `${SPECTER.path}` (uppercase),
/// `${capture.foo}` (unrecognised namespace) all pass through verbatim.
/// The strict typo guard fires only inside the recognised namespaces;
/// the shell can expand the rest as it likes.
pub fn parse_arg(s: &str) -> Result<ArgTemplate, TemplateError> {
    let mut parts: smallvec::SmallVec<[ArgPart; 2]> = smallvec![];
    let mut buf = CompactString::new("");
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        // Fast-forward to the next `$`. Bytes are pure ASCII for `$` and
        // `{`, so byte-level scanning is correct without char boundary
        // checks; non-ASCII content lives in `buf` as bytes which
        // re-assemble into valid UTF-8 because we never split a code
        // point.
        if bytes[i] != b'$' {
            // Push as a single char to preserve UTF-8 boundaries.
            let ch_start = i;
            let ch_end = next_char_boundary(s, i);
            buf.push_str(&s[ch_start..ch_end]);
            i = ch_end;
            continue;
        }

        // `$$` — escape to literal `$`.
        if matches!(bytes.get(i + 1), Some(&b'$')) {
            buf.push('$');
            i += 2;
            continue;
        }

        // Recognised namespace opener?
        if let Some((kind, body_start)) = try_open_namespace(s, i) {
            let Some(rel_end) = s[body_start..].find('}') else {
                return Err(TemplateError::UnterminatedPlaceholder {
                    partial: s[i..].to_owned(),
                });
            };
            let body = &s[body_start..body_start + rel_end];
            let part = dispatch_namespace(kind, body)?;
            flush_literal(&mut parts, &mut buf);
            parts.push(part);
            i = body_start + rel_end + 1;
            continue;
        }

        // Lone `$` — literal pass-through.
        buf.push('$');
        i += 1;
    }

    if !buf.is_empty() || parts.is_empty() {
        parts.push(ArgPart::Literal(buf));
    }
    Ok(ArgTemplate { parts })
}

/// Tag identifying which namespace `parse_arg` just opened. The match
/// is exhaustive across the recognised set; unknown openers (e.g.
/// `${capture.…}`, `${SCOPE.…}`) fall through to literal pass-through
/// before this enum is ever consulted.
#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum NamespaceKind {
    Specter,
    Env,
}

/// Look for a recognised namespace opener at byte offset `i`. Returns
/// `(kind, body_start)` where `body_start` is the byte index of the
/// first character after the namespace dot (the start of `<name>`).
fn try_open_namespace(s: &str, i: usize) -> Option<(NamespaceKind, usize)> {
    if s[i..].starts_with(NS_SPECTER) {
        Some((NamespaceKind::Specter, i + NS_SPECTER.len()))
    } else if s[i..].starts_with(NS_ENV) {
        Some((NamespaceKind::Env, i + NS_ENV.len()))
    } else {
        None
    }
}

/// Parse the body of an opened namespace (everything between the
/// namespace dot and the closing `}`) into an [`ArgPart`].
fn dispatch_namespace(kind: NamespaceKind, body: &str) -> Result<ArgPart, TemplateError> {
    match kind {
        NamespaceKind::Specter => parse_specter_name(body).map(ArgPart::Placeholder),
        NamespaceKind::Env => parse_env_body(body),
    }
}

/// Validate `name` against the Specter namespace grammar and resolve it
/// to a catalog [`Placeholder`].
///
/// `[a-z0-9_]+` and a catalog entry. Any deviation surfaces a typed
/// error so the validator can render a useful operator-facing message.
fn parse_specter_name(name: &str) -> Result<Placeholder, TemplateError> {
    if name.is_empty() {
        return Err(TemplateError::EmptyPlaceholderName);
    }
    for ch in name.chars() {
        let ok = ch.is_ascii_lowercase() || ch.is_ascii_digit() || ch == '_';
        if !ok {
            return Err(TemplateError::InvalidPlaceholderChar {
                name: name.to_owned(),
                ch,
            });
        }
    }
    catalog_lookup(name).ok_or_else(|| TemplateError::UnknownPlaceholder {
        name: name.to_owned(),
    })
}

/// Parse an `${env.…}` body: split on `:-` to separate `<NAME>` from
/// an optional literal default, validate each side, and produce
/// [`ArgPart::EnvVar`].
///
/// `:-` is recognised on its first occurrence and consumed as the
/// name/default split. A default body containing a second `:-` is
/// rejected by [`validate_env_default`] as
/// [`TemplateError::EnvDefaultContainsSeparator`] — defaults are
/// representable as a literal byte sequence with `:-` reserved.
/// `}` cannot appear inside a default because the closing brace
/// terminates the placeholder before this function ever sees it; a
/// default that wants `}` is unrepresentable in v1.
fn parse_env_body(body: &str) -> Result<ArgPart, TemplateError> {
    let (name, default) = match body.find(ENV_DEFAULT_SEP) {
        Some(idx) => (&body[..idx], Some(&body[idx + ENV_DEFAULT_SEP.len()..])),
        None => (body, None),
    };
    validate_env_name(name)?;
    if let Some(d) = default {
        validate_env_default(d)?;
    }
    Ok(ArgPart::EnvVar {
        name: name.into(),
        default: default.map(CompactString::from),
    })
}

/// `[A-Za-z_][A-Za-z0-9_]*` — POSIX env-var identifier grammar.
fn validate_env_name(name: &str) -> Result<(), TemplateError> {
    if name.is_empty() {
        return Err(TemplateError::EmptyPlaceholderName);
    }
    let mut chars = name.chars();
    let first = chars
        .next()
        .expect("non-empty checked immediately above; iterator yields at least one char");
    if !(first.is_ascii_alphabetic() || first == '_') {
        return Err(TemplateError::InvalidEnvName {
            name: name.to_owned(),
            ch: first,
        });
    }
    for ch in chars {
        if !(ch.is_ascii_alphanumeric() || ch == '_') {
            return Err(TemplateError::InvalidEnvName {
                name: name.to_owned(),
                ch,
            });
        }
    }
    Ok(())
}

/// Defaults are literal-only in v1. Three reject classes:
///
/// 1. The literal `:-` separator substring. The first `:-` is already
///    consumed by `parse_env_body` as the name/default split. A second
///    occurrence anywhere in the default body would invite the ambiguity
///    "is that part of the default or did the operator mean to split
///    again?", so we reject outright. `${env.X:-a:-b}` ⇒
///    [`TemplateError::EnvDefaultContainsSeparator`].
/// 2. `$` and `{`. Both hint at nested substitution / a stray opener —
///    feedback to the operator that defaults are literal-only.
/// 3. Any control character (`is_control()`). An unintended `\n` or
///    `\t` would silently make it into argv and (worse) into log
///    lines if the default ever appears in a diagnostic; rendering as
///    a Rust-source escape (via Display's `escape_default`) keeps the
///    operator-facing error readable.
///
/// The closing `}` cannot appear inside the default because the
/// placeholder lexer terminates on the first one — this branch never
/// sees it.
fn validate_env_default(d: &str) -> Result<(), TemplateError> {
    if d.contains(ENV_DEFAULT_SEP) {
        return Err(TemplateError::EnvDefaultContainsSeparator {
            default: d.to_owned(),
        });
    }
    if let Some(ch) = d.chars().find(|c| matches!(c, '$' | '{') || c.is_control()) {
        return Err(TemplateError::InvalidEnvDefault {
            default: d.to_owned(),
            ch,
        });
    }
    Ok(())
}

/// Resolve a syntactically valid lowercase name to a [`Placeholder`].
/// `None` means the name is well-formed but not in the catalog —
/// surfaced as [`TemplateError::UnknownPlaceholder`] by the caller.
const fn catalog_lookup(name: &str) -> Option<Placeholder> {
    // `match` over `&str` lets the catalog stay one source of truth; the
    // compiler folds it to a hash-free dispatch.
    Some(match name.as_bytes() {
        b"path" => Placeholder::Path,
        b"relative" => Placeholder::Relative,
        b"anchor" => Placeholder::Anchor,
        b"watch" => Placeholder::Watch,
        b"parent" => Placeholder::Parent,
        b"time" => Placeholder::Time,
        b"created" => Placeholder::Created,
        b"deleted" => Placeholder::Deleted,
        b"modified" => Placeholder::Modified,
        b"renamed_from" => Placeholder::RenamedFrom,
        b"renamed_to" => Placeholder::RenamedTo,
        b"excluded" => Placeholder::Excluded,
        _ => return None,
    })
}

/// Flush the in-flight literal buffer as one [`ArgPart::Literal`] when
/// non-empty. Adjacent literals always coalesce because every literal
/// byte goes into the same `buf` and we only flush at placeholder
/// boundaries / end-of-input.
fn flush_literal(parts: &mut smallvec::SmallVec<[ArgPart; 2]>, buf: &mut CompactString) {
    if !buf.is_empty() {
        parts.push(ArgPart::Literal(std::mem::take(buf)));
    }
}

/// Return the byte index just past the UTF-8 character starting at `i`.
/// `s.is_char_boundary` is the canonical Rust API; we trust the input
/// `s` is valid UTF-8 (it came from a `&str`) so a forward scan over
/// continuation bytes is sufficient.
fn next_char_boundary(s: &str, i: usize) -> usize {
    let bytes = s.as_bytes();
    let mut j = i + 1;
    while j < bytes.len() && (bytes[j] & 0b1100_0000) == 0b1000_0000 {
        j += 1;
    }
    j
}

#[cfg(test)]
mod tests {
    use super::{TemplateError, parse_arg};
    use compact_str::CompactString;
    use proptest::prelude::*;
    use specter_core::{ArgPart, ArgTemplate, Placeholder};

    fn lit(s: &str) -> ArgPart {
        ArgPart::literal(s)
    }

    fn ph(p: Placeholder) -> ArgPart {
        ArgPart::Placeholder(p)
    }

    fn parts(t: ArgTemplate) -> Vec<ArgPart> {
        t.parts.into_iter().collect()
    }

    // === Catalog membership ===

    #[test]
    fn each_catalog_placeholder_alone() {
        for (s, p) in [
            ("${specter.path}", Placeholder::Path),
            ("${specter.relative}", Placeholder::Relative),
            ("${specter.anchor}", Placeholder::Anchor),
            ("${specter.watch}", Placeholder::Watch),
            ("${specter.parent}", Placeholder::Parent),
            ("${specter.time}", Placeholder::Time),
            ("${specter.created}", Placeholder::Created),
            ("${specter.deleted}", Placeholder::Deleted),
            ("${specter.modified}", Placeholder::Modified),
            ("${specter.renamed_from}", Placeholder::RenamedFrom),
            ("${specter.renamed_to}", Placeholder::RenamedTo),
            ("${specter.excluded}", Placeholder::Excluded),
        ] {
            assert_eq!(parts(parse_arg(s).unwrap()), vec![ph(p)], "input {s}");
        }
    }

    #[test]
    fn unknown_placeholder_inside_namespace() {
        let err = parse_arg("${specter.unknown}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::UnknownPlaceholder {
                name: "unknown".to_owned()
            }
        );
    }

    // === Literal pass-through (the strict break) ===

    #[test]
    fn pure_literal_input() {
        assert_eq!(parts(parse_arg("hello").unwrap()), vec![lit("hello")]);
    }

    #[test]
    fn empty_input_is_single_empty_literal() {
        assert_eq!(parts(parse_arg("").unwrap()), vec![lit("")]);
    }

    #[test]
    fn bare_dollar_name_is_literal() {
        // Under the new grammar, `$<name>` is shell territory regardless
        // of catalog membership. The lexer never touches it.
        assert_eq!(parts(parse_arg("$path").unwrap()), vec![lit("$path")]);
        assert_eq!(parts(parse_arg("$watch").unwrap()), vec![lit("$watch")]);
        assert_eq!(parts(parse_arg("$created").unwrap()), vec![lit("$created")]);
        assert_eq!(parts(parse_arg("$HOME").unwrap()), vec![lit("$HOME")]);
        assert_eq!(parts(parse_arg("$Path").unwrap()), vec![lit("$Path")]);
        assert_eq!(parts(parse_arg("$_x").unwrap()), vec![lit("$_x")]);
        assert_eq!(parts(parse_arg("$5").unwrap()), vec![lit("$5")]);
    }

    #[test]
    fn brace_var_outside_namespace_is_literal() {
        // `${VAR}`, `${HOME}` — shell-style braced expansion. The lexer
        // doesn't open a namespace here (the prefix is `${specter.`,
        // requiring the lowercase `specter.` exactly), so the `$`
        // becomes a single literal char and `{HOME}` follows as plain
        // bytes.
        assert_eq!(parts(parse_arg("${HOME}").unwrap()), vec![lit("${HOME}")]);
        assert_eq!(parts(parse_arg("${VAR}").unwrap()), vec![lit("${VAR}")]);
    }

    #[test]
    fn brace_namespace_no_dot_is_literal() {
        // `${specter}` (no dot) is NOT the namespace opener. The dot is
        // load-bearing.
        assert_eq!(
            parts(parse_arg("${specter}").unwrap()),
            vec![lit("${specter}")]
        );
    }

    #[test]
    fn uppercase_namespace_is_literal() {
        // `${SPECTER.path}` is uppercase; the namespace prefix is
        // lowercase only. Falls through as literal.
        assert_eq!(
            parts(parse_arg("${SPECTER.path}").unwrap()),
            vec![lit("${SPECTER.path}")]
        );
    }

    #[test]
    fn space_after_dollar_is_literal() {
        // `$ {specter.path}` — must be `${` adjacent. The space breaks
        // the prefix; lone `$` becomes literal.
        assert_eq!(
            parts(parse_arg("$ {specter.path}").unwrap()),
            vec![lit("$ {specter.path}")]
        );
    }

    // === Escape rules ===

    #[test]
    fn double_dollar_collapses_to_literal() {
        assert_eq!(parts(parse_arg("$$").unwrap()), vec![lit("$")]);
        assert_eq!(parts(parse_arg("$$$$").unwrap()), vec![lit("$$")]);
    }

    #[test]
    fn double_dollar_then_namespace_escapes_namespace() {
        // `$${specter.path}` — `$$` consumes the leading `$`, leaving a
        // literal `$` followed by `{specter.path}` plain bytes.
        assert_eq!(
            parts(parse_arg("$${specter.path}").unwrap()),
            vec![lit("${specter.path}")]
        );
    }

    #[test]
    fn triple_dollar_then_namespace() {
        // `$$$` — `$$` collapses to literal `$`, then a lone `$`
        // followed by `{specter.path}` opens the namespace and
        // resolves to a placeholder.
        assert_eq!(
            parts(parse_arg("$$${specter.path}").unwrap()),
            vec![lit("$"), ph(Placeholder::Path)]
        );
    }

    // === Composition with literals and adjacent placeholders ===

    #[test]
    fn literal_prefix_then_placeholder() {
        assert_eq!(
            parts(parse_arg("--input=${specter.path}").unwrap()),
            vec![lit("--input="), ph(Placeholder::Path)]
        );
    }

    #[test]
    fn placeholder_then_literal_suffix() {
        assert_eq!(
            parts(parse_arg("${specter.path}/foo").unwrap()),
            vec![ph(Placeholder::Path), lit("/foo")]
        );
    }

    #[test]
    fn literal_around_placeholder_yields_three_parts() {
        // The adjacent-literal coalescing invariant: one literal before,
        // one placeholder, one literal after — exactly three parts.
        assert_eq!(
            parts(parse_arg("abc${specter.path}xyz").unwrap()),
            vec![lit("abc"), ph(Placeholder::Path), lit("xyz")]
        );
    }

    #[test]
    fn adjacent_placeholders() {
        assert_eq!(
            parts(parse_arg("${specter.path}${specter.relative}").unwrap()),
            vec![ph(Placeholder::Path), ph(Placeholder::Relative)]
        );
    }

    #[test]
    fn unicode_literal_preserved() {
        assert_eq!(
            parts(parse_arg("build-🚀-${specter.path}").unwrap()),
            vec![lit("build-🚀-"), ph(Placeholder::Path)]
        );
    }

    #[test]
    fn bare_trailing_dollar_is_literal() {
        assert_eq!(parts(parse_arg("$").unwrap()), vec![lit("$")]);
    }

    // === Errors inside the namespace ===

    #[test]
    fn unterminated_placeholder() {
        let err = parse_arg("${specter.path").unwrap_err();
        assert_eq!(
            err,
            TemplateError::UnterminatedPlaceholder {
                partial: "${specter.path".to_owned(),
            }
        );
    }

    #[test]
    fn empty_placeholder_name() {
        let err = parse_arg("${specter.}").unwrap_err();
        assert_eq!(err, TemplateError::EmptyPlaceholderName);
    }

    #[test]
    fn invalid_placeholder_char_uppercase() {
        let err = parse_arg("${specter.PATH}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidPlaceholderChar {
                name: "PATH".to_owned(),
                ch: 'P',
            }
        );
    }

    #[test]
    fn invalid_placeholder_char_dot() {
        // Dots inside the name are reserved — no nested namespace
        // (`${specter.renamed.from}` is not how `renamed_from` is
        // spelled).
        let err = parse_arg("${specter.renamed.from}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidPlaceholderChar {
                name: "renamed.from".to_owned(),
                ch: '.',
            }
        );
    }

    #[test]
    fn invalid_placeholder_char_dollar() {
        // Nested-dollar sentinel inside the name fails at the first `$`.
        let err = parse_arg("${specter.${specter.foo}}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidPlaceholderChar {
                name: "${specter.foo".to_owned(),
                ch: '$',
            }
        );
    }

    // === Display rendering ===

    #[test]
    fn template_error_display_renders_specter_namespace() {
        assert_eq!(
            TemplateError::UnknownPlaceholder {
                name: "Foo".to_owned()
            }
            .to_string(),
            "unknown placeholder `${specter.Foo}`"
        );
        assert_eq!(
            TemplateError::EmptyPlaceholderName.to_string(),
            "empty placeholder name (expected `${specter.<name>}` or `${env.<name>}`)"
        );
        assert_eq!(
            TemplateError::InvalidEnvName {
                name: "1HOME".to_owned(),
                ch: '1',
            }
            .to_string(),
            "invalid character `1` in `${env.1HOME}` (expected `[A-Za-z_][A-Za-z0-9_]*`)"
        );
    }

    // === `${env.<NAME>}` namespace ===

    fn env(name: &str, default: Option<&str>) -> ArgPart {
        ArgPart::EnvVar {
            name: name.into(),
            default: default.map(CompactString::from),
        }
    }

    #[test]
    fn env_var_simple_no_default() {
        assert_eq!(
            parts(parse_arg("${env.HOME}").unwrap()),
            vec![env("HOME", None)]
        );
    }

    #[test]
    fn env_var_with_literal_default() {
        assert_eq!(
            parts(parse_arg("${env.HOME:-/tmp}").unwrap()),
            vec![env("HOME", Some("/tmp"))]
        );
    }

    /// `${env.HOME:-}` — explicit lenient opt-in. The default is the
    /// empty string; resolver renders empty when HOME is unset rather
    /// than failing the plan.
    #[test]
    fn env_var_empty_default_is_explicit_lenient_opt_in() {
        assert_eq!(
            parts(parse_arg("${env.HOME:-}").unwrap()),
            vec![env("HOME", Some(""))]
        );
    }

    #[test]
    fn env_var_underscore_name_allowed() {
        assert_eq!(
            parts(parse_arg("${env._PRIVATE_X}").unwrap()),
            vec![env("_PRIVATE_X", None)]
        );
    }

    #[test]
    fn env_var_mixed_case_name_allowed() {
        // POSIX env-var grammar is case-sensitive and case-mixed;
        // unlike `${specter.…}` (lowercase-only), `${env.…}` accepts
        // any well-formed identifier.
        assert_eq!(
            parts(parse_arg("${env.PathSeparator2}").unwrap()),
            vec![env("PathSeparator2", None)]
        );
    }

    #[test]
    fn env_var_empty_name_is_empty_placeholder_name() {
        // `${env.}` shares the empty-name policy with `${specter.}`.
        let err = parse_arg("${env.}").unwrap_err();
        assert_eq!(err, TemplateError::EmptyPlaceholderName);
    }

    /// `${env.}` with `:-default` — the name side is still empty,
    /// reported as the canonical empty-name error.
    #[test]
    fn env_var_empty_name_with_default_is_empty_placeholder_name() {
        let err = parse_arg("${env.:-fallback}").unwrap_err();
        assert_eq!(err, TemplateError::EmptyPlaceholderName);
    }

    #[test]
    fn env_var_name_must_not_start_with_digit() {
        let err = parse_arg("${env.1HOME}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidEnvName {
                name: "1HOME".to_owned(),
                ch: '1',
            }
        );
    }

    #[test]
    fn env_var_name_rejects_dash() {
        let err = parse_arg("${env.MY-VAR}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidEnvName {
                name: "MY-VAR".to_owned(),
                ch: '-',
            }
        );
    }

    #[test]
    fn env_var_unterminated_is_unterminated_placeholder() {
        let err = parse_arg("${env.HOME").unwrap_err();
        assert_eq!(
            err,
            TemplateError::UnterminatedPlaceholder {
                partial: "${env.HOME".to_owned(),
            }
        );
    }

    #[test]
    fn env_var_default_rejects_nested_dollar() {
        // `${env.HOME:-${env.USER}}` — the default contains `$`,
        // which we reject for unambiguity. The closing `}` after
        // `USER` terminates the outer placeholder, so the lexer sees
        // `default = "${env.USER"`.
        let err = parse_arg("${env.HOME:-${env.USER}}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidEnvDefault {
                default: "${env.USER".to_owned(),
                ch: '$',
            }
        );
    }

    #[test]
    fn env_var_default_rejects_open_brace() {
        // Lone `{` is also reserved — even without a leading `$`,
        // it's a signal that the operator might be confused about
        // the literal-only contract.
        let err = parse_arg("${env.HOME:-{foo}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::InvalidEnvDefault {
                default: "{foo".to_owned(),
                ch: '{',
            }
        );
    }

    /// `${env.X:-a:-b}` — the first `:-` consumed as the name/default
    /// split leaves `default = "a:-b"`. A second `:-` inside the default
    /// body invites ambiguity ("was `b` meant as a trailing field?"),
    /// rejected outright by the strict v1 grammar.
    #[test]
    fn env_var_default_rejects_separator_substring() {
        let err = parse_arg("${env.X:-a:-b}").unwrap_err();
        assert_eq!(
            err,
            TemplateError::EnvDefaultContainsSeparator {
                default: "a:-b".to_owned(),
            }
        );
    }

    /// Control characters in defaults would silently end up in argv and
    /// (worse) corrupt log lines if surfaced in diagnostics. Strict v1
    /// rejects every `is_control()` char; newline and tab are the cases
    /// an operator is most likely to typo.
    #[test]
    fn env_var_default_rejects_control_chars() {
        for (input, expected_ch) in [
            ("${env.X:-line\nbreak}", '\n'),
            ("${env.X:-col\tumn}", '\t'),
            ("${env.X:-\x00null}", '\0'),
        ] {
            let err = parse_arg(input).unwrap_err();
            match err {
                TemplateError::InvalidEnvDefault { ch, .. } => assert_eq!(
                    ch, expected_ch,
                    "input `{input}` should reject `{expected_ch:?}`"
                ),
                other => panic!("input `{input}`: expected InvalidEnvDefault, got {other:?}"),
            }
        }
    }

    /// Display rendering for control chars uses `escape_default` so the
    /// error message stays readable in a log stream. A literal newline
    /// in the message would otherwise wreck downstream line-oriented
    /// parsing.
    #[test]
    fn invalid_env_default_display_escapes_control_chars() {
        let msg = TemplateError::InvalidEnvDefault {
            default: "x\ny".to_owned(),
            ch: '\n',
        }
        .to_string();
        assert!(
            msg.contains(r"\n"),
            "Display should escape `\\n` literally, got `{msg}`"
        );
        assert!(
            !msg.contains('\n'),
            "Display message must not contain a raw newline, got `{msg}`"
        );
    }

    #[test]
    fn env_var_default_allows_slash_dot_colon_etc() {
        // Common default-path/value shapes survive: `/tmp`, `.cache`,
        // `0`, `127.0.0.1:8080`, and unicode tokens.
        for (input, expected_default) in [
            ("${env.X:-/tmp}", "/tmp"),
            ("${env.X:-.cache}", ".cache"),
            ("${env.PORT:-0}", "0"),
            ("${env.ADDR:-127.0.0.1:8080}", "127.0.0.1:8080"),
            ("${env.LANG:-en_US.UTF-8}", "en_US.UTF-8"),
        ] {
            let parsed = parse_arg(input).unwrap();
            let env_part = parsed.parts.into_iter().next().expect("at least one part");
            match env_part {
                ArgPart::EnvVar { default, .. } => assert_eq!(
                    default.as_deref(),
                    Some(expected_default),
                    "input `{input}`"
                ),
                other => panic!("input `{input}`: expected EnvVar, got {other:?}"),
            }
        }
    }

    /// Unknown namespaces (`${capture.foo}`, `${SCOPE.bar}`, anything
    /// the lexer doesn't recognise) pass through verbatim — the strict
    /// typo guard only fires inside known namespaces. Future additive
    /// namespaces land without breaking existing TOML.
    #[test]
    fn unknown_namespace_passes_through_as_literal() {
        for s in [
            "${capture.foo}",
            "${scope.bar}",
            "${ENV.HOME}", // uppercase namespace prefix → not recognised
            "${Specter.path}",
        ] {
            assert_eq!(
                parts(parse_arg(s).unwrap()),
                vec![lit(s)],
                "unknown namespace `{s}` must pass through"
            );
        }
    }

    #[test]
    fn mixed_specter_and_env_namespaces() {
        // Operator mixes both namespaces in one argv slot — each
        // resolves independently into its own ArgPart.
        assert_eq!(
            parts(parse_arg("${specter.path}-${env.USER:-anon}").unwrap()),
            vec![ph(Placeholder::Path), lit("-"), env("USER", Some("anon")),]
        );
    }

    /// Two `${env.…}` placeholders adjacent — no separator, no
    /// confused parsing.
    #[test]
    fn adjacent_env_var_placeholders() {
        assert_eq!(
            parts(parse_arg("${env.A}${env.B}").unwrap()),
            vec![env("A", None), env("B", None)]
        );
    }

    /// `$${env.HOME}` — the `$$` escape consumes the leading `$`,
    /// leaving a literal `$` followed by `{env.HOME}` as plain bytes.
    /// Matches the existing `$${specter.path}` escape contract.
    #[test]
    fn double_dollar_then_env_namespace_escapes_namespace() {
        assert_eq!(
            parts(parse_arg("$${env.HOME}").unwrap()),
            vec![lit("${env.HOME}")]
        );
    }

    // === Property tests ===

    proptest! {
        /// Total: any UTF-8 input parses without panic.
        #[test]
        fn prop_parser_is_total(s in "[\\PC]{0,64}") {
            let _ = parse_arg(&s);
        }

        /// Bare `$<name>` — regardless of casing, digits, underscores —
        /// passes through verbatim. The lexer never opens a placeholder
        /// without the `${specter.` prefix.
        #[test]
        fn prop_bare_dollar_name_literal(name in "[A-Za-z_][A-Za-z0-9_]{0,15}") {
            let s = format!("${name}");
            let parsed = parse_arg(&s).unwrap();
            prop_assert_eq!(parts(parsed), vec![lit(&s)]);
        }

        /// `${VAR}` braced shell-style expansion is literal regardless of
        /// content — only `${specter.…}` and `${env.…}` open a namespace.
        #[test]
        fn prop_brace_non_namespace_literal(name in "[A-Z_][A-Z0-9_]{0,15}") {
            let s = format!("${{{name}}}");
            let parsed = parse_arg(&s).unwrap();
            prop_assert_eq!(parts(parsed), vec![lit(&s)]);
        }

        /// Every well-formed `${env.<NAME>}` round-trips to an
        /// `ArgPart::EnvVar { name, default: None }`. Captures the
        /// grammar invariant: no parse path produces a Placeholder or
        /// Literal for these inputs.
        #[test]
        fn prop_env_var_no_default_round_trips(name in "[A-Za-z_][A-Za-z0-9_]{0,15}") {
            let s = format!("${{env.{name}}}");
            let parsed = parse_arg(&s).unwrap();
            prop_assert_eq!(
                parts(parsed),
                vec![ArgPart::EnvVar {
                    name: CompactString::from(name.as_str()),
                    default: None,
                }],
            );
        }

        /// Every `${env.<NAME>:-<default>}` whose default avoids the
        /// reserved chars (`$`, `{`, `}`), control characters, and the
        /// `:-` separator substring round-trips with the default
        /// preserved verbatim.
        #[test]
        fn prop_env_var_with_safe_default_round_trips(
            name in "[A-Za-z_][A-Za-z0-9_]{0,15}",
            default in "[A-Za-z0-9_/.:= -]{0,32}",
        ) {
            // Strict v1: defaults containing `:-` are rejected by
            // `validate_env_default` (the separator must occur exactly
            // once between name and default, never inside the default
            // body). Filter generated cases that violate the invariant.
            prop_assume!(!default.contains(":-"));
            let s = format!("${{env.{name}:-{default}}}");
            let parsed = parse_arg(&s).unwrap();
            prop_assert_eq!(
                parts(parsed),
                vec![ArgPart::EnvVar {
                    name: CompactString::from(name.as_str()),
                    default: Some(CompactString::from(default.as_str())),
                }],
            );
        }
    }
}
