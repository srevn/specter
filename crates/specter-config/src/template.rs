use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::{ArgPart, ArgTemplate, Placeholder};
use std::fmt;

/// Namespace prefix that opens a Specter placeholder. Anything not
/// matching this exact byte sequence (including the trailing dot) falls
/// through as a literal `$` — bare `$NAME`, `$home`, `${VAR}`, `${specter}`
/// (no dot), `${SPECTER.path}` (uppercase), etc. are all literals.
const NAMESPACE: &str = "${specter.";

/// Failures the lexer can surface from inside a `${specter.…}` placeholder.
///
/// Outside the namespace, the lexer never errors — every other
/// `$`-bearing byte sequence passes through verbatim, freeing operators
/// to write arbitrary shell / awk / perl `$` syntax in argv slots
/// without a Specter typo tax.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TemplateError {
    /// `${specter.<name>}` where `<name>` is not in the placeholder
    /// catalog. Catches mistypes (`${specter.ptah}`) and members of the
    /// catalog that aren't yet implemented.
    UnknownPlaceholder { name: String },
    /// `${specter.<name>` reached end-of-string without a closing `}`.
    /// `partial` is the substring from the opening `${` to end-of-input.
    UnterminatedPlaceholder { partial: String },
    /// `${specter.}` — the namespace was opened but no name follows.
    EmptyPlaceholderName,
    /// `${specter.<name>}` where `<name>` contains a character outside
    /// `[a-z0-9_]`. The first offending character is reported.
    InvalidPlaceholderChar { name: String, ch: char },
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
            Self::EmptyPlaceholderName => {
                f.write_str("empty placeholder name `${specter.}` (expected `${specter.<name>}`)")
            }
            Self::InvalidPlaceholderChar { name, ch } => {
                write!(
                    f,
                    "invalid character `{ch}` in placeholder name `{name}` \
                     (expected `[a-z0-9_]`)",
                )
            }
        }
    }
}

impl std::error::Error for TemplateError {}

/// Parse one TOML argv string into an [`ArgTemplate`].
///
/// Recognises exactly two `$`-prefix patterns; everything else passes
/// through as a literal:
///
/// - `${specter.<name>}` — the Specter placeholder namespace. `<name>`
///   must be a non-empty `[a-z0-9_]` sequence and must match a catalog
///   entry (`path`, `relative`, `anchor`, `watch`, `parent`, `time`,
///   `created`, `deleted`, `modified`, `renamed_from`, `renamed_to`,
///   `excluded`); anything else inside the namespace returns an error.
/// - `$$` — escapes a literal `$`. The only way to write a single `$`
///   that the spawned shell will not interpret as the start of an env
///   var name; doubles up shell `$$` (PID expansion) as `$$$$`.
///
/// Every other `$`-bearing sequence is a literal: `$HOME`, `$path`,
/// `$5`, `${VAR}`, `${specter}`, `${SPECTER.path}` all pass through
/// verbatim. The strict typo guard fires only inside the namespace; the
/// shell can expand the rest as it likes.
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

        // `${specter.` — open namespace.
        if s[i..].starts_with(NAMESPACE) {
            let name_start = i + NAMESPACE.len();
            let Some(rel_end) = s[name_start..].find('}') else {
                return Err(TemplateError::UnterminatedPlaceholder {
                    partial: s[i..].to_owned(),
                });
            };
            let name = &s[name_start..name_start + rel_end];
            let placeholder = parse_namespace_name(name)?;
            flush_literal(&mut parts, &mut buf);
            parts.push(ArgPart::Placeholder(placeholder));
            i = name_start + rel_end + 1;
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

/// Validate `name` against the namespace grammar and resolve it to a
/// catalog [`Placeholder`].
///
/// `[a-z0-9_]+` and a catalog entry. Any deviation surfaces a typed
/// error so the validator can render a useful operator-facing message.
fn parse_namespace_name(name: &str) -> Result<Placeholder, TemplateError> {
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
            "empty placeholder name `${specter.}` (expected `${specter.<name>}`)"
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
        /// content — only `${specter.…}` opens the namespace.
        #[test]
        fn prop_brace_non_namespace_literal(name in "[A-Z_][A-Z0-9_]{0,15}") {
            let s = format!("${{{name}}}");
            let parsed = parse_arg(&s).unwrap();
            prop_assert_eq!(parts(parsed), vec![lit(&s)]);
        }
    }
}
