//! Path-pattern parsing for dynamic watches.
//!
//! `PatternSpec` carries the canonical source string alongside its decomposed
//! `Vec<PatternComponent>` and the `literal_prefix_len` — the number of leading consecutive
//! `Literal` segments (root included). These two derived fields are a deterministic function of
//! `source`, so equality routes through `source` alone.
//!
//! The parser screens a few invariants beyond glob compilation:
//! - **Absolute only.** Patterns must begin with `/`.
//! - **Globstar rejected.** `**` is unsupported in v1.
//! - **No empty / `.` / `..` segments.** `//foo`, `/./x`, `/../x` are all rejected at parse.
//! - **No Windows prefix.** A `:` inside any segment fails parse.
//!
//! Brace expansion (`{a,b}`) stays as a *single* `Glob` component; globset compiles it natively.
//! The parser does not enumerate brace alternatives.

use crate::scan_config::{ConfigError, GlobPattern};
use compact_str::CompactString;
use std::fmt;
use std::path::{Path, PathBuf};

/// Decomposed glob path pattern. `components.len() >= 2` post-parse — a synthetic `Literal("/")` at
/// index 0 plus at least one segment.
///
/// Equality is over `source` only; the `components` and `literal_prefix_len` fields are
/// deterministic functions of `source` (the parser is pure), so two `PatternSpec`s with equal
/// `source` strings have byte-equal decompositions.
#[derive(Clone, Debug)]
pub struct PatternSpec {
    source: CompactString,
    components: Vec<PatternComponent>,
    literal_prefix_len: usize,
}

/// One segment of a parsed `PatternSpec`. `Literal` carries the raw segment name (or `/` for the
/// synthetic root); `Glob` carries the compiled `globset::GlobMatcher` plus its source.
///
/// Brace patterns (`{a,b,c}`) compile to one `Glob` — globset matches alternatives natively. The
/// parser never enumerates the alternation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatternComponent {
    Literal(CompactString),
    Glob(GlobPattern),
}

/// Parse / classification errors. Surfaced through the config layer's `IssueKind::InvalidPattern`;
/// never reaches the engine.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum PatternError {
    /// `**` — recursive globbing is unsupported in v1.
    GlobstarUnsupported,
    /// `globset::Glob::new` rejected the segment.
    InvalidGlob { source: String, message: String },
    /// Empty source string.
    EmptyPattern,
    /// Source did not begin with `/`.
    NonAbsolute,
    /// `.` or `..` segment.
    NonCanonical,
    /// Empty segment between `/`s (e.g., `//foo`, trailing `/`).
    EmptySegment,
    /// Windows-style prefix detected (`:` inside a segment).
    WindowsPrefix,
    /// Source is pure-literal — none of the `*`, `?`, `[`, `{` discriminators — so it is a static
    /// anchor, not a dynamic pattern. Callers route the static/dynamic split on
    /// [`PatternSpec::is_dynamic`] upstream, so this is a defense-in-depth signal for direct
    /// callers that bypass that dispatch; production never reaches it.
    NotDynamic,
}

impl fmt::Display for PatternError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::GlobstarUnsupported => {
                f.write_str("`**` (recursive glob) is not supported in v1")
            }
            Self::InvalidGlob { source, message } => {
                write!(f, "invalid glob segment `{source}`: {message}")
            }
            Self::EmptyPattern => f.write_str("pattern must not be empty"),
            Self::NonAbsolute => f.write_str("pattern must be absolute (start with `/`)"),
            Self::NonCanonical => f.write_str("`.` and `..` segments are not allowed"),
            Self::EmptySegment => {
                f.write_str("empty path segment (consecutive `/` or trailing `/`)")
            }
            Self::WindowsPrefix => f.write_str("Windows-style prefix (`:`) is not allowed"),
            Self::NotDynamic => f.write_str(
                "pattern is pure-literal (no `*`, `?`, `[`, `{`); not a dynamic watch pattern",
            ),
        }
    }
}

impl std::error::Error for PatternError {}

/// `GlobPattern::compile` fails with [`ConfigError`]; a malformed glob segment inside a dynamic
/// pattern surfaces as [`PatternError::InvalidGlob`]. Naming the conversion keeps the cross-error
/// coupling explicit and exhaustiveness-checked — a future `ConfigError` variant forces a new arm
/// here, where an inline single-arm `match` would have silently compiled through.
///
/// [`ConfigError::UnreachableGlob`] is folded into [`PatternError::InvalidGlob`] with `message =
/// reason`: every shape `GlobPattern::compile` rejects as unreachable is *already* caught by
/// [`PatternSpec::parse`]'s earlier gates (`EmptyPattern`, `EmptySegment`, `NonAbsolute`,
/// `NonCanonical`, `GlobstarUnsupported`), so the arm is structurally unreachable from this
/// conversion site — but the exhaustive match keeps the coupling from drifting if a future
/// `ConfigError` variant lands.
impl From<ConfigError> for PatternError {
    fn from(e: ConfigError) -> Self {
        match e {
            ConfigError::InvalidGlob { source, message } => Self::InvalidGlob { source, message },
            ConfigError::UnreachableGlob { source, reason } => Self::InvalidGlob {
                source,
                message: reason,
            },
        }
    }
}

impl PatternSpec {
    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    #[must_use]
    pub fn components(&self) -> &[PatternComponent] {
        &self.components
    }

    /// Count of leading consecutive `Literal` components, root included. Always `>= 1` (the
    /// synthetic root `/` is `Literal`) and strictly `<` the total component count: [`Self::parse`]
    /// rejects pure-literal sources with [`PatternError::NotDynamic`], so every constructed
    /// `PatternSpec` has at least one non-`Literal` component.
    #[must_use]
    pub const fn literal_prefix_len(&self) -> usize {
        self.literal_prefix_len
    }

    /// The literal-prefix anchor path: `/` plus the leading consecutive `Literal` segments — where
    /// a discovery Sub for this pattern attaches. Total by the parse invariant
    /// (`components[0..literal_prefix_len]` are all `Literal`, root included), so the fold is
    /// always absolute; a `Glob` inside the prefix is an invariant breach (`debug_assert!`),
    /// skipped rather than rendered in release.
    #[must_use]
    pub fn literal_prefix_path(&self) -> PathBuf {
        let mut p = PathBuf::new();
        for comp in &self.components[..self.literal_prefix_len] {
            match comp {
                PatternComponent::Literal(s) => p.push(s.as_str()),
                PatternComponent::Glob(_) => {
                    debug_assert!(false, "glob in literal prefix violates parse invariant");
                }
            }
        }
        p
    }

    /// Chain levels below the literal-prefix anchor — the anchor-relative depth at which a match
    /// terminates. Always `>= 1`: [`Self::parse`] rejects pure-literal sources, so at least one
    /// component sits past the literal prefix.
    #[must_use]
    pub fn terminus_depth(&self) -> u32 {
        u32::try_from(self.components.len() - self.literal_prefix_len).unwrap_or(u32::MAX)
    }

    /// True iff `segment` matches the positional component at anchor-relative `depth` (`1 ..=
    /// terminus_depth`). Total: out-of-range depths return `false` — `0` included, since the anchor
    /// *is* the literal prefix, not a chain position.
    ///
    /// `Literal` compares byte-equality against the bare segment. A literal component never carries
    /// a glob discriminator (the parser routes those to `Glob`), but glob-special non-discriminator
    /// bytes such as `\` stay literal here — never escape-interpreted. `Glob` runs the compiled
    /// matcher against the bare segment, so brace alternation, `?`, and character classes apply
    /// per-position.
    #[must_use]
    pub fn matches_at(&self, depth: u32, segment: &str) -> bool {
        if depth == 0 || depth > self.terminus_depth() {
            return false;
        }
        // In range ⇒ the index lands in `literal_prefix_len ..= components.len() − 1` (`depth <=
        // terminus_depth = len − lpl`), so the direct index cannot panic.
        match &self.components[self.literal_prefix_len + depth as usize - 1] {
            PatternComponent::Literal(lit) => lit.as_str() == segment,
            PatternComponent::Glob(g) => g.matches_path(Path::new(segment)),
        }
    }

    /// Validator dispatch gate: returns `true` iff `source` contains any of the four glob
    /// discriminator characters: `*`, `?`, `[`, `{`. `{` is part of the set so brace patterns route
    /// to the dynamic parser.
    ///
    /// `const fn` so the validator can fold the dispatch decision into a `const` context if needed;
    /// the byte-by-byte scan avoids the non-const `Iterator::any` machinery.
    #[must_use]
    pub const fn is_dynamic(source: &str) -> bool {
        let bytes = source.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            if matches!(bytes[i], b'*' | b'?' | b'[' | b'{') {
                return true;
            }
            i += 1;
        }
        false
    }

    /// Parse `source` into a structurally-classified `PatternSpec`.
    ///
    /// A pure-literal `source` (none of the `*`, `?`, `[`, `{` discriminators) is rejected with
    /// [`PatternError::NotDynamic`] — release-true, never a panic. Callers route the static/dynamic
    /// split on [`Self::is_dynamic`] upstream, so `NotDynamic` is a defense-in-depth signal for
    /// direct callers that bypass that dispatch. On `Ok`, the invariant `literal_prefix_len <
    /// components.len()` is discharged by the parser itself, not by a release-stripped convention.
    pub fn parse(source: &str) -> Result<Self, PatternError> {
        if source.is_empty() {
            return Err(PatternError::EmptyPattern);
        }
        if !source.starts_with('/') {
            return Err(PatternError::NonAbsolute);
        }
        if source.contains("**") {
            return Err(PatternError::GlobstarUnsupported);
        }

        let parts: Vec<&str> = source[1..].split('/').collect();
        let mut components: Vec<PatternComponent> = Vec::with_capacity(parts.len() + 1);
        components.push(PatternComponent::Literal(CompactString::from("/")));

        for part in parts {
            if part.is_empty() {
                return Err(PatternError::EmptySegment);
            }
            if part == "." || part == ".." {
                return Err(PatternError::NonCanonical);
            }
            if part.contains(':') {
                return Err(PatternError::WindowsPrefix);
            }
            // `is_dynamic` is the canonical predicate for the same byte set; reusing it here keeps
            // the discriminator definition single-source.
            if Self::is_dynamic(part) {
                components.push(PatternComponent::Glob(GlobPattern::compile(part)?));
            } else {
                components.push(PatternComponent::Literal(CompactString::from(part)));
            }
        }

        let literal_prefix_len = components
            .iter()
            .take_while(|c| matches!(c, PatternComponent::Literal(_)))
            .count();

        // Pure-literal source ⇒ every component a `Literal` ⇒ the leading run spans the whole vec.
        // Rejecting here upholds the `literal_prefix_len < components.len()` invariant callers rely
        // on; `>=` faithfully negates that strict bound (`>` is structurally unreachable —
        // `take_while().count() <= len`).
        if literal_prefix_len >= components.len() {
            return Err(PatternError::NotDynamic);
        }

        Ok(Self {
            source: CompactString::from(source),
            components,
            literal_prefix_len,
        })
    }
}

impl PartialEq for PatternSpec {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for PatternSpec {}

#[cfg(test)]
mod tests {
    use super::{PatternComponent, PatternError, PatternSpec};

    /// `is_dynamic` flips `true` for each of the four discriminator chars and stays `false` for
    /// plain absolute paths.
    #[test]
    fn is_dynamic_flips_on_each_glob_discriminator() {
        assert!(PatternSpec::is_dynamic("/srv/*/data"));
        assert!(PatternSpec::is_dynamic("/srv/?/data"));
        assert!(PatternSpec::is_dynamic("/srv/[a-z]/data"));
        assert!(PatternSpec::is_dynamic("/srv/{a,b}/data"));
    }

    #[test]
    fn is_dynamic_false_for_pure_literal_paths() {
        assert!(!PatternSpec::is_dynamic("/var/log/myapp"));
        assert!(!PatternSpec::is_dynamic("/"));
        assert!(!PatternSpec::is_dynamic(""));
    }

    /// `parse` produces the synthetic root + segments and computes `literal_prefix_len` over the
    /// leading consecutive `Literal`s.
    #[test]
    fn parse_mixed_pattern_decomposes_components() {
        let spec = PatternSpec::parse("/srv/staging/*/data/*/log").expect("valid pattern");
        assert_eq!(spec.components().len(), 7);
        assert!(matches!(
            spec.components()[0],
            PatternComponent::Literal(ref s) if s == "/",
        ));
        assert!(matches!(
            spec.components()[1],
            PatternComponent::Literal(ref s) if s == "srv",
        ));
        assert!(matches!(
            spec.components()[2],
            PatternComponent::Literal(ref s) if s == "staging",
        ));
        assert!(matches!(spec.components()[3], PatternComponent::Glob(_)));
        assert!(matches!(
            spec.components()[4],
            PatternComponent::Literal(ref s) if s == "data",
        ));
        assert!(matches!(spec.components()[5], PatternComponent::Glob(_)));
        assert!(matches!(
            spec.components()[6],
            PatternComponent::Literal(ref s) if s == "log",
        ));
        assert_eq!(spec.literal_prefix_len(), 3);
    }

    #[test]
    fn parse_leading_glob_has_minimal_literal_prefix() {
        // `/srv/*/site` ⇒ literal_prefix_len = 2.
        let spec = PatternSpec::parse("/srv/*/site").expect("valid pattern");
        assert_eq!(spec.literal_prefix_len(), 2);
    }

    #[test]
    fn parse_brace_expansion_stays_one_glob_component() {
        // Brace expansion is one Glob component, not multiple. globset matches alternatives natively.
        let spec = PatternSpec::parse("/var/log/{app,system}/access.log").expect("valid pattern");
        assert_eq!(spec.components().len(), 5);
        assert_eq!(spec.literal_prefix_len(), 3);
        assert!(matches!(spec.components()[3], PatternComponent::Glob(_)));
    }

    #[test]
    fn parse_rejects_empty() {
        assert_eq!(PatternSpec::parse(""), Err(PatternError::EmptyPattern));
    }

    #[test]
    fn parse_rejects_non_absolute() {
        assert_eq!(
            PatternSpec::parse("var/log/*"),
            Err(PatternError::NonAbsolute),
        );
    }

    #[test]
    fn parse_rejects_globstar() {
        assert_eq!(
            PatternSpec::parse("/var/log/**/x"),
            Err(PatternError::GlobstarUnsupported),
        );
    }

    #[test]
    fn parse_rejects_double_slash_as_empty_segment() {
        assert_eq!(
            PatternSpec::parse("//var/log/*"),
            Err(PatternError::EmptySegment),
        );
    }

    #[test]
    fn parse_rejects_trailing_slash_as_empty_segment() {
        assert_eq!(
            PatternSpec::parse("/var/log/*/"),
            Err(PatternError::EmptySegment),
        );
    }

    #[test]
    fn parse_rejects_dot_segment() {
        assert_eq!(
            PatternSpec::parse("/var/./log/*"),
            Err(PatternError::NonCanonical),
        );
    }

    #[test]
    fn parse_rejects_dotdot_segment() {
        assert_eq!(
            PatternSpec::parse("/var/../log/*"),
            Err(PatternError::NonCanonical),
        );
    }

    #[test]
    fn parse_rejects_windows_prefix() {
        assert_eq!(
            PatternSpec::parse("/c:/Users/*"),
            Err(PatternError::WindowsPrefix),
        );
    }

    #[test]
    fn parse_rejects_invalid_glob_segment() {
        // Unbalanced `[` triggers globset's parse error.
        assert!(matches!(
            PatternSpec::parse("/var/log/[unbalanced"),
            Err(PatternError::InvalidGlob { .. }),
        ));
    }

    /// `/*` is the FS-root pattern. It parses to one literal segment (root) plus one glob, with
    /// `literal_prefix_len = 1`.
    #[test]
    fn parse_accepts_root_glob_pattern() {
        let spec = PatternSpec::parse("/*").expect("valid pattern");
        assert_eq!(spec.components().len(), 2);
        assert_eq!(spec.literal_prefix_len(), 1);
        assert!(matches!(
            spec.components()[0],
            PatternComponent::Literal(ref s) if s == "/",
        ));
        assert!(matches!(spec.components()[1], PatternComponent::Glob(_)));
    }

    /// Consecutive globs build a deeper proxy chain. literal_prefix_len = 2; subsequent globs at
    /// idx 2, 3, etc.
    #[test]
    fn parse_consecutive_globs_after_literal_prefix() {
        let spec = PatternSpec::parse("/data/*/*/log").expect("valid pattern");
        assert_eq!(spec.components().len(), 5);
        assert_eq!(spec.literal_prefix_len(), 2);
        assert!(matches!(spec.components()[2], PatternComponent::Glob(_)));
        assert!(matches!(spec.components()[3], PatternComponent::Glob(_)));
        assert!(matches!(
            spec.components()[4],
            PatternComponent::Literal(ref s) if s == "log",
        ));
    }

    /// `PartialEq` routes through `source` only — equal source strings produce byte-equal
    /// `PatternSpec`s.
    #[test]
    fn equality_routes_through_source() {
        let a = PatternSpec::parse("/srv/*/data").expect("valid");
        let b = PatternSpec::parse("/srv/*/data").expect("valid");
        assert_eq!(a, b);
    }

    /// Post-condition contract: a pure-literal source (no glob discriminator) is rejected with
    /// [`PatternError::NotDynamic`] — release-true, never a panic. The `literal_prefix_len <
    /// components.len()` invariant is discharged by the parser, not a release-stripped
    /// `debug_assert!`.
    #[test]
    fn parse_returns_not_dynamic_err() {
        assert_eq!(
            PatternSpec::parse("/var/log/myapp"),
            Err(PatternError::NotDynamic),
        );
    }

    /// `literal_prefix_path` renders the leading `Literal` run as an absolute anchor path: the
    /// synthetic root plus each literal segment for a nested prefix, and bare `/` for the `lpl = 1`
    /// root pattern (the no-parent-edge anchor case).
    #[test]
    fn literal_prefix_path_renders_anchor_for_each_prefix_shape() {
        let nested = PatternSpec::parse("/srv/staging/*/data/*/log").expect("valid pattern");
        assert_eq!(
            nested.literal_prefix_path(),
            std::path::PathBuf::from("/srv/staging"),
        );
        let root = PatternSpec::parse("/*").expect("valid pattern");
        assert_eq!(root.literal_prefix_path(), std::path::PathBuf::from("/"));
    }

    /// `terminus_depth` is the chain length below the literal-prefix anchor — pinned across the
    /// three prefix shapes (mixed literal/glob, root-anchored `/*`, consecutive globs) so the `len
    /// − lpl` arithmetic can't silently drift against the parser's decomposition.
    #[test]
    fn terminus_depth_measures_chain_below_literal_prefix() {
        let mixed = PatternSpec::parse("/srv/staging/*/data/*/log").expect("valid pattern");
        assert_eq!(mixed.terminus_depth(), 4);
        let root = PatternSpec::parse("/*").expect("valid pattern");
        assert_eq!(root.terminus_depth(), 1);
        let consecutive = PatternSpec::parse("/data/*/*/log").expect("valid pattern");
        assert_eq!(consecutive.terminus_depth(), 3);
    }

    /// The `Literal` arm of `matches_at` is byte-equality, not glob matching. `\` is glob-special
    /// but not a parse discriminator, so `a\b` stays a `Literal` component; a glob-interpreting arm
    /// would escape it to match `ab`. Case and prefix mismatches pin plain equality.
    #[test]
    fn matches_at_literal_is_byte_equality_not_glob() {
        let spec = PatternSpec::parse(r"/srv/*/a\b").expect("valid pattern");
        // Depth 2 is the literal `a\b` component.
        assert!(spec.matches_at(2, r"a\b"));
        assert!(
            !spec.matches_at(2, "ab"),
            r"`\` must stay a literal byte, never a glob escape",
        );
        let plain = PatternSpec::parse("/srv/*/data").expect("valid pattern");
        assert!(plain.matches_at(2, "data"));
        assert!(!plain.matches_at(2, "Data"));
        assert!(!plain.matches_at(2, "dat"));
        assert!(!plain.matches_at(2, "database"));
    }

    /// The `Glob` arm runs the compiled matcher against the bare segment — one pin per
    /// discriminator (`*`, `?`, `[a-z]`, `{a,b}`; the brace alternation stays one component).
    #[test]
    fn matches_at_glob_matches_per_discriminator() {
        let star = PatternSpec::parse("/srv/app*").expect("valid pattern");
        assert!(star.matches_at(1, "app1"));
        assert!(!star.matches_at(1, "web1"));

        let question = PatternSpec::parse("/srv/app?").expect("valid pattern");
        assert!(question.matches_at(1, "app1"));
        assert!(!question.matches_at(1, "app12"));

        let class = PatternSpec::parse("/srv/vol[a-z]").expect("valid pattern");
        assert!(class.matches_at(1, "vola"));
        assert!(!class.matches_at(1, "vol1"));

        let brace = PatternSpec::parse("/srv/{app,web}").expect("valid pattern");
        assert!(brace.matches_at(1, "app"));
        assert!(brace.matches_at(1, "web"));
        assert!(!brace.matches_at(1, "db"));
    }

    /// `matches_at` is total over depth: `0` (the anchor — not a chain position) and anything
    /// beyond the terminus return `false` rather than panicking on index arithmetic.
    #[test]
    fn matches_at_out_of_range_depths_are_false() {
        let spec = PatternSpec::parse("/srv/*/log").expect("valid pattern");
        assert_eq!(spec.terminus_depth(), 2);
        assert!(!spec.matches_at(0, "srv"));
        assert!(!spec.matches_at(3, "log"));
        assert!(!spec.matches_at(u32::MAX, "log"));
    }

    /// `Display` produces human-readable, operator-friendly messages. The config validator routes
    /// parse errors through `IssueKind:: InvalidPattern`; the rendered detail surfaces these
    /// strings, so a regression in wording would silently degrade error UX.
    #[test]
    fn pattern_error_display_is_human_readable() {
        for (err, needle) in [
            (PatternError::GlobstarUnsupported, "recursive glob"),
            (PatternError::EmptyPattern, "must not be empty"),
            (PatternError::NonAbsolute, "absolute"),
            (PatternError::NonCanonical, "`.` and `..`"),
            (PatternError::EmptySegment, "empty path segment"),
            (PatternError::WindowsPrefix, "Windows"),
            (PatternError::NotDynamic, "pure-literal"),
        ] {
            let s = err.to_string();
            assert!(
                s.contains(needle),
                "expected `{needle}` in display of {err:?}, got `{s}`",
            );
        }
        let invalid = PatternError::InvalidGlob {
            source: "[bad".to_owned(),
            message: "unbalanced `[`".to_owned(),
        };
        let s = invalid.to_string();
        assert!(s.contains("[bad"), "got `{s}`");
        assert!(s.contains("unbalanced"), "got `{s}`");
    }
}
