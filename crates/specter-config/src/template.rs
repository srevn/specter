use compact_str::CompactString;
use smallvec::smallvec;
use specter_core::{ArgPart, ArgTemplate, Placeholder};
use std::fmt;

#[derive(Debug, Clone, Eq, PartialEq)]
pub enum TemplateError {
    UnknownPlaceholder { name: String },
}

impl fmt::Display for TemplateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownPlaceholder { name } => write!(f, "unknown placeholder `${name}`"),
        }
    }
}

impl std::error::Error for TemplateError {}

/// Parse one TOML argv string into an [`ArgTemplate`].
///
/// Lexer rules:
/// - `$<name>` where `<name>` exactly matches a catalog entry
///   (lowercase: `path`, `rel`, `anchor`, `created`, `deleted`, `modified`,
///   `renamed_from`, `renamed_to`) → [`Placeholder`].
/// - `$<name>` where `<name>` contains any ASCII uppercase letter → literal
///   `$<name>`. Preserves shell-expansion of env vars
///   (`$SPECTER_PATH`, `$SPECTER_FORCED`, etc.) and conventional uppercase
///   shell vars (`$HOME`, `$PATH`, `$USER`).
/// - `$<name>` where `<name>` is all-lowercase (with optional digits /
///   underscores) but not in the catalog → [`TemplateError::UnknownPlaceholder`].
///   Catches typos of catalog names (`$pat`, `$rell`, etc.) since the
///   catalog is lowercase by convention.
/// - `$` followed by a digit, punctuation, end-of-string, or any non-name-
///   start character is a literal `$`.
/// - Empty input → `[Literal("")]` (rejected by the validator, but the
///   lexer stays total).
pub fn parse_arg(s: &str) -> Result<ArgTemplate, TemplateError> {
    let mut parts: smallvec::SmallVec<[ArgPart; 2]> = smallvec![];
    let mut buf = CompactString::new("");
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c != '$' {
            buf.push(c);
            continue;
        }

        let starts_name = chars
            .peek()
            .is_some_and(|c| c.is_ascii_alphabetic() || *c == '_');
        if !starts_name {
            buf.push('$');
            continue;
        }

        let mut name = String::new();
        while let Some(c) = chars.peek() {
            if c.is_ascii_alphanumeric() || *c == '_' {
                name.push(*c);
                chars.next();
            } else {
                break;
            }
        }

        let placeholder = match name.as_str() {
            "path" => Placeholder::Path,
            "rel" => Placeholder::Rel,
            "anchor" => Placeholder::Anchor,
            "created" => Placeholder::Created,
            "deleted" => Placeholder::Deleted,
            "modified" => Placeholder::Modified,
            "renamed_from" => Placeholder::RenamedFrom,
            "renamed_to" => Placeholder::RenamedTo,
            // Names with any uppercase ASCII letter pass through as literal
            // — they are env vars (`SPECTER_PATH`) or conventional shell
            // vars (`HOME`, `PATH`). Typo detection still applies to
            // all-lowercase non-catalog names below.
            other if other.bytes().any(|b| b.is_ascii_uppercase()) => {
                buf.push('$');
                buf.push_str(other);
                continue;
            }
            _ => return Err(TemplateError::UnknownPlaceholder { name }),
        };

        if !buf.is_empty() {
            parts.push(ArgPart::Literal(std::mem::take(&mut buf)));
        }
        parts.push(ArgPart::Placeholder(placeholder));
    }

    if !buf.is_empty() || parts.is_empty() {
        parts.push(ArgPart::Literal(buf));
    }
    Ok(ArgTemplate { parts })
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

    #[test]
    fn pure_literal_input() {
        assert_eq!(parts(parse_arg("hello").unwrap()), vec![lit("hello")]);
    }

    #[test]
    fn empty_input_is_single_empty_literal() {
        assert_eq!(parts(parse_arg("").unwrap()), vec![lit("")]);
    }

    #[test]
    fn each_catalog_placeholder_alone() {
        for (s, p) in [
            ("$path", Placeholder::Path),
            ("$rel", Placeholder::Rel),
            ("$anchor", Placeholder::Anchor),
            ("$created", Placeholder::Created),
            ("$deleted", Placeholder::Deleted),
            ("$modified", Placeholder::Modified),
            ("$renamed_from", Placeholder::RenamedFrom),
            ("$renamed_to", Placeholder::RenamedTo),
        ] {
            assert_eq!(parts(parse_arg(s).unwrap()), vec![ph(p)], "input {s}");
        }
    }

    #[test]
    fn literal_prefix_then_placeholder() {
        assert_eq!(
            parts(parse_arg("--input=$path").unwrap()),
            vec![lit("--input="), ph(Placeholder::Path)]
        );
    }

    #[test]
    fn placeholder_then_literal_suffix() {
        assert_eq!(
            parts(parse_arg("$path/foo").unwrap()),
            vec![ph(Placeholder::Path), lit("/foo")]
        );
    }

    #[test]
    fn adjacent_single_value_placeholders() {
        assert_eq!(
            parts(parse_arg("$path$rel").unwrap()),
            vec![ph(Placeholder::Path), ph(Placeholder::Rel)]
        );
    }

    #[test]
    fn adjacent_multi_value_placeholders() {
        assert_eq!(
            parts(parse_arg("$created$deleted").unwrap()),
            vec![ph(Placeholder::Created), ph(Placeholder::Deleted)]
        );
    }

    #[test]
    fn dollar_dollar_then_placeholder() {
        assert_eq!(
            parts(parse_arg("$$path").unwrap()),
            vec![lit("$"), ph(Placeholder::Path)]
        );
    }

    #[test]
    fn double_dollar_alone_is_two_literal_dollars() {
        assert_eq!(parts(parse_arg("$$").unwrap()), vec![lit("$$")]);
    }

    #[test]
    fn dollar_followed_by_digit_is_literal() {
        assert_eq!(parts(parse_arg("$5").unwrap()), vec![lit("$5")]);
    }

    #[test]
    fn bare_trailing_dollar_is_literal() {
        assert_eq!(parts(parse_arg("$").unwrap()), vec![lit("$")]);
    }

    #[test]
    fn dollar_in_middle_followed_by_digit_is_literal() {
        assert_eq!(parts(parse_arg("abc$5xy").unwrap()), vec![lit("abc$5xy")]);
    }

    #[test]
    fn capitalized_name_passes_through_as_literal_for_shell_expansion() {
        // Env vars (uppercase) and conventional shell vars must reach the
        // spawned shell unchanged. Names containing ANY uppercase letter
        // bypass the catalog lookup; the catalog is lowercase-only.
        assert_eq!(parts(parse_arg("$Path").unwrap()), vec![lit("$Path")]);
        assert_eq!(
            parts(parse_arg("$SPECTER_ANCHOR").unwrap()),
            vec![lit("$SPECTER_ANCHOR")]
        );
        assert_eq!(parts(parse_arg("$HOME").unwrap()), vec![lit("$HOME")]);
    }

    #[test]
    fn mixed_case_with_literal_neighbors() {
        // Uppercase identifiers concatenate with surrounding literal context.
        assert_eq!(
            parts(parse_arg("export $HOME=$path").unwrap()),
            vec![lit("export $HOME="), ph(Placeholder::Path)]
        );
    }

    #[test]
    fn unknown_placeholder_rejected() {
        let err = parse_arg("$unknown").unwrap_err();
        assert_eq!(
            err,
            TemplateError::UnknownPlaceholder {
                name: "unknown".to_owned()
            }
        );
    }

    #[test]
    fn name_starting_with_underscore_rejected_when_not_in_catalog() {
        let err = parse_arg("$_path").unwrap_err();
        assert_eq!(
            err,
            TemplateError::UnknownPlaceholder {
                name: "_path".to_owned()
            }
        );
    }

    #[test]
    fn literal_separates_placeholder_from_suffix() {
        assert_eq!(
            parts(parse_arg("abc$path-suffix").unwrap()),
            vec![lit("abc"), ph(Placeholder::Path), lit("-suffix")]
        );
    }

    #[test]
    fn unicode_literal_preserved() {
        assert_eq!(
            parts(parse_arg("build-🚀-$path").unwrap()),
            vec![lit("build-🚀-"), ph(Placeholder::Path)]
        );
    }

    #[test]
    fn template_error_display_renders_dollar_prefix() {
        let err = TemplateError::UnknownPlaceholder {
            name: "Foo".to_owned(),
        };
        assert_eq!(err.to_string(), "unknown placeholder `$Foo`");
    }

    const CATALOG: &[&str] = &[
        "path",
        "rel",
        "anchor",
        "created",
        "deleted",
        "modified",
        "renamed_from",
        "renamed_to",
    ];

    proptest! {
        #[test]
        fn prop_literal_only_inputs_round_trip(s in "[a-zA-Z0-9_/\\-=. ]{0,32}") {
            let parsed = parse_arg(&s).unwrap();
            prop_assert_eq!(parts(parsed), vec![lit(&s)]);
        }

        #[test]
        fn prop_unknown_lowercase_placeholder_rejected(name in "[a-z_][a-z0-9_]{0,15}") {
            // Lowercase-only names not in the catalog → typo error. The
            // catalog is exclusively lowercase, so this gate catches `$pat`,
            // `$paht`, `$rell`, etc.
            if CATALOG.contains(&name.as_str()) {
                return Ok(());
            }
            let s = format!("${name}");
            let err = parse_arg(&s).unwrap_err();
            prop_assert_eq!(err, TemplateError::UnknownPlaceholder { name });
        }

        #[test]
        fn prop_uppercase_or_mixed_passes_through_as_literal(
            name in "[A-Za-z_][A-Za-z0-9_]{0,15}"
        ) {
            // Names containing any uppercase letter pass through as literal
            // `$<name>` so the spawned shell can expand them as env vars.
            // Pure-lowercase names land in the lowercase prop above.
            if !name.bytes().any(|b| b.is_ascii_uppercase()) {
                return Ok(());
            }
            let s = format!("${name}");
            let expected_lit = format!("${name}");
            prop_assert_eq!(parts(parse_arg(&s).unwrap()), vec![lit(&expected_lit)]);
        }
    }
}
