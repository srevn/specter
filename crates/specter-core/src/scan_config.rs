//! Scan configuration.
//!
//! [`compute_config_hash`] is the canonical hash of `(ScanConfig, max_settle, events)`;
//! [`ProfileIdentity`] is its reified preimage and [`ProfileIdentity::config_hash`] the sole public
//! route. Equal hash ⇒ Subs share one Profile (snapshot, burst lifecycle). Both fold sites
//! destructure exhaustively, so a new identity-bearing field is a compile error until folded — the
//! fold-completeness ratchet.
//!
//! `GlobPattern` carries the canonical `source` string alongside the compiled
//! `globset::GlobMatcher`. Equality and ordering are over `source` only — the matcher is a
//! transient compiled artifact. There is no `Hash` impl: the single hashing path through these
//! types is `compute_config_hash`, which reads `source` directly via `core::hash::hasher`.

use crate::hash::hasher;
use crate::resource::ResourceKind;
use crate::sub::ClassSet;
use compact_str::CompactString;
use globset::{Glob, GlobMatcher};
use std::cmp::Ordering;
use std::path::Path;
use std::time::Duration;

/// A glob pattern: the canonical `source` text and its compiled matcher.
///
/// `source` is the sole identity axis across `PartialEq`/`Eq`/`Ord` and the `compute_config_hash`
/// fold. `matcher` is a transient compiled artifact derived from `source` — never key any of those
/// four on it.
#[derive(Clone, Debug)]
pub struct GlobPattern {
    source: CompactString,
    matcher: GlobMatcher,
}

impl GlobPattern {
    /// Compile a glob pattern. Two failure paths:
    ///
    /// - [`ConfigError::UnreachableGlob`] — the source is syntactically *valid* per `globset` but
    ///   cannot match any non-empty relative path under the walker's anchor-relative semantics
    ///   (empty, `"."`/`".."`, leading `/`, trailing `/` without `**`). Rejected at the floor so
    ///   every downstream consumer of a compiled `GlobPattern` is one that can potentially match
    ///   something — the gitignore-canonical `target/` footgun in particular fails fast rather than
    ///   silently doing nothing.
    /// - [`ConfigError::InvalidGlob`] — `globset` rejected the source itself (unbalanced brackets,
    ///   malformed alternation, etc.). The `globset::Error` is not `Clone`, so its display message
    ///   is rendered into a `String`.
    ///
    /// The unreachability check runs *before* `globset::Glob::new` so the variants are unambiguous:
    /// if `globset` accepts it, the only remaining failure is structural unreachability, and vice
    /// versa.
    pub fn compile(source: impl Into<CompactString>) -> Result<Self, ConfigError> {
        let source: CompactString = source.into();
        validate_source_reachability(source.as_str())?;
        let glob = Glob::new(&source).map_err(|e| ConfigError::InvalidGlob {
            source: source.to_string(),
            message: e.to_string(),
        })?;
        Ok(Self {
            source,
            matcher: glob.compile_matcher(),
        })
    }

    #[must_use]
    pub fn source(&self) -> &str {
        &self.source
    }

    #[must_use]
    pub fn matches_path(&self, path: &Path) -> bool {
        self.matcher.is_match(path)
    }
}

impl PartialEq for GlobPattern {
    fn eq(&self, other: &Self) -> bool {
        self.source == other.source
    }
}

impl Eq for GlobPattern {}

impl Ord for GlobPattern {
    fn cmp(&self, other: &Self) -> Ordering {
        self.source.cmp(&other.source)
    }
}

impl PartialOrd for GlobPattern {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// The scan predicate — what the walker reads and what `covers` tests, as a sum over scan shapes.
///
/// Every consumer goes through the named predicates ([`Self::accepts`],
/// [`Self::accepts_structural`], [`Self::accepts_kinded`], [`Self::descends_into`],
/// [`Self::exclude_globs`]) or the hash; no consumer destructures the variants — shape dispatch
/// lives entirely in this module, so a new shape is a compile error here and nowhere else.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScanConfig {
    /// The cumulative recursive predicate over one subtree: every filter narrows what a recursive
    /// walk admits. The user-facing shape — [`ScanConfigBuilder`] builds it, config lowering and
    /// Profile identity carry it.
    Subtree {
        recursive: bool,
        hidden: bool,
        /// Sorted by source string at builder time.
        exclude: Vec<GlobPattern>,
        pattern: Option<GlobPattern>,
        max_depth: Option<u32>,
    },
    /// Admit-every-dirent, single level — the descent probe's preset. Descent searches for the next
    /// path component of a not-yet-existing anchor, so the user-facing filters (which would mask
    /// the very segment being searched for) deliberately collapse to no-ops. Prober-internal: never
    /// wrapped in a `ProfileIdentity`, never on the wire, never partitions Profiles.
    Descent,
}

impl ScanConfig {
    #[must_use]
    pub fn builder() -> ScanConfigBuilder {
        ScanConfigBuilder::default()
    }

    /// True iff an entry at cumulative relative path `rel` of `kind` at `depth` (anchor = 0, direct
    /// child = 1, …) is in scope.
    ///
    /// The single source of the scope predicate, composed of its two halves
    /// ([`Self::accepts_structural`] ∧ [`Self::accepts_kinded`]). Two callers:
    /// - `specter_engine::coverage` tests every prefix from anchor → target — `kind` is in hand,
    ///   single call.
    /// - The walker (`specter-sensor::prober`) tests each dirent — `kind` is only known after
    ///   `lstat`, so it calls the halves directly: [`Self::accepts_structural`] pre-`lstat`,
    ///   [`Self::accepts_kinded`] post-`lstat`; see the structural half's doc for the rationale.
    ///
    /// **Depth 0 bypasses every filter.** The anchor is part of the Profile's scope by
    /// construction. For `depth > 0`, `Subtree` folds `max_depth`, `recursive`, `hidden`
    /// (last-segment basename test), `exclude` (full-`rel` test), and `pattern` (final-`File` only
    /// — directories are always covered; we descend through them); `Descent` admits one level.
    ///
    /// `ResourceKind::Unknown` is collapsed by upstream callers (`Resource::kind_or_file`,
    /// `From<EntryKind>`) before reaching this method, so `kind != Dir` here means the same thing
    /// as the walker's `!is_dir`.
    #[must_use]
    pub fn accepts(&self, rel: &Path, kind: ResourceKind, depth: u32) -> bool {
        self.accepts_structural(rel, depth)
            && self.accepts_kinded(rel, matches!(kind, ResourceKind::Dir))
    }

    /// The kind-independent half of [`Self::accepts`]. For `Subtree`, folds the four gates that
    /// don't need a `ResourceKind`: `max_depth`, `recursive`, `hidden`, and `exclude`. For
    /// `Descent`, admits one level (the walk never descends, so only depths 0 and 1 are queried).
    ///
    /// Exists for the walker, which doesn't know `is_dir` until after the per-dirent `lstat`.
    /// Calling [`Self::accepts`] there with a guessed kind would either skip a covered Dir (guess
    /// `File` + `pattern.matches == false`) or admit a pattern-violating File (guess `Dir`).
    /// Splitting lets the walker reject hidden / excluded dirents pre-`lstat` — saving the syscall
    /// on a `target/` tree in a Cargo project (thousands of excluded dirents) — and gate the kinded
    /// half post-`lstat` against the known kind. `covers` has `kind` in hand and calls
    /// [`Self::accepts`] directly.
    ///
    /// **Fold-completeness ratchet, predicate tier.** The `Subtree` arm destructures exhaustively:
    /// a new field on the variant is a compile error here until the author decides which half it
    /// folds into — structural (a clause below) or kind-dependent (an arm in
    /// [`Self::accepts_kinded`], where `pattern` lives). A new *variant* is a non-exhaustive-match
    /// compile error across every predicate and the hash at once.
    #[must_use]
    pub fn accepts_structural(&self, rel: &Path, depth: u32) -> bool {
        // Anchor depth bypasses every filter for every shape — the anchor is in scope by
        // construction.
        if depth == 0 {
            return true;
        }
        match self {
            // `pattern` lives here only to discharge the destructure; its gate is the kinded half.
            Self::Subtree {
                recursive,
                hidden,
                exclude,
                pattern: _,
                max_depth,
            } => {
                if let Some(max) = *max_depth
                    && depth > max
                {
                    return false;
                }
                if depth > 1 && !*recursive {
                    return false;
                }
                if !*hidden
                    && rel
                        .file_name()
                        .and_then(|n| n.to_str())
                        .is_some_and(|s| s.starts_with('.'))
                {
                    return false;
                }
                if exclude.iter().any(|g| g.matches_path(rel)) {
                    return false;
                }
                true
            }
            Self::Descent => depth <= 1,
        }
    }

    /// The kind-dependent half of [`Self::accepts`]: the residual gate that needs to know whether
    /// the entry is a directory. For `Subtree`, the `pattern` arm — files must match the pattern
    /// when one is set; directories are always covered (the walker descends through them, and
    /// `covers` checks them as intermediate prefixes). `Descent` admits every kind.
    ///
    /// One home for both consumers: [`Self::accepts`] passes `kind == Dir`, and the walker calls
    /// this directly post-`lstat` with the freshly-observed `is_dir` — the two agree because
    /// `ResourceKind` folds every non-directory (symlinks included) to `File`, so `kind != Dir` ⟺
    /// `!is_dir` for every kind `accepts` can receive.
    #[must_use]
    pub fn accepts_kinded(&self, rel: &Path, is_dir: bool) -> bool {
        match self {
            Self::Subtree { pattern, .. } => {
                is_dir || pattern.as_ref().is_none_or(|pat| pat.matches_path(rel))
            }
            Self::Descent => true,
        }
    }

    /// True iff the walk's coverage extends below a directory sitting at `child_depth` (anchor = 0,
    /// direct child = 1, …) — the recursion-edge decision, distinct from [`Self::accepts`]'s
    /// per-entry inclusion decision. `Subtree` descends while `recursive` and under `max_depth`;
    /// `Descent` never descends.
    ///
    /// The walker's `should_recurse` conjoins this with its cross-filesystem gate (the engine's
    /// `Tree` slots don't carry `device`, so the device axis cannot live here); negation drives
    /// `DirChild::Uncovered(fs_id)` emission.
    #[must_use]
    pub fn descends_into(&self, child_depth: u32) -> bool {
        match self {
            Self::Subtree {
                recursive,
                max_depth,
                ..
            } => *recursive && child_depth < max_depth.unwrap_or(u32::MAX),
            Self::Descent => false,
        }
    }

    /// The exclude globs this scan shape filters with — `Subtree`'s `exclude` list; the empty slice
    /// for shapes that carry none. Effect-env projection (`Profile::exclude_strings`) consumes this
    /// so it never destructures the shape.
    #[must_use]
    pub fn exclude_globs(&self) -> &[GlobPattern] {
        match self {
            Self::Subtree { exclude, .. } => exclude,
            Self::Descent => &[],
        }
    }
}

/// Builder for [`ScanConfig::Subtree`] — the user-facing shape; the other variants are presets
/// constructed directly.
///
/// Sorts `exclude` by source on `build()` so equal logical configs are byte-equal —
/// `compute_config_hash` reads in already-sorted order.
#[derive(Debug, Default)]
pub struct ScanConfigBuilder {
    recursive: bool,
    hidden: bool,
    exclude: Vec<GlobPattern>,
    pattern: Option<GlobPattern>,
    max_depth: Option<u32>,
}

impl ScanConfigBuilder {
    #[must_use]
    pub const fn recursive(mut self, v: bool) -> Self {
        self.recursive = v;
        self
    }

    #[must_use]
    pub const fn hidden(mut self, v: bool) -> Self {
        self.hidden = v;
        self
    }

    #[must_use]
    pub fn exclude(mut self, g: GlobPattern) -> Self {
        self.exclude.push(g);
        self
    }

    #[must_use]
    pub fn excludes<I: IntoIterator<Item = GlobPattern>>(mut self, gs: I) -> Self {
        self.exclude.extend(gs);
        self
    }

    #[must_use]
    pub fn pattern(mut self, g: GlobPattern) -> Self {
        self.pattern = Some(g);
        self
    }

    #[must_use]
    pub const fn max_depth(mut self, d: Option<u32>) -> Self {
        self.max_depth = d;
        self
    }

    #[must_use]
    pub fn build(mut self) -> ScanConfig {
        self.exclude.sort();
        ScanConfig::Subtree {
            recursive: self.recursive,
            hidden: self.hidden,
            exclude: self.exclude,
            pattern: self.pattern,
            max_depth: self.max_depth,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigError {
    /// `globset` rejected the glob source (e.g. unbalanced brackets, malformed alternation).
    /// `message` is the rendered `globset::Error` text (the original is not `Clone`).
    InvalidGlob { source: String, message: String },
    /// The glob is syntactically valid per `globset` but cannot match any non-empty relative path
    /// in the walker's semantics. Surfaced at the [`GlobPattern::compile`] floor so a typo
    /// (`target/` vs `target/**`) or a misunderstanding (leading `/` in a relative- path glob)
    /// fails fast rather than silently doing nothing. `reason` is an operator-facing explanation of
    /// the specific shape that's unreachable.
    UnreachableGlob { source: String, reason: String },
}

/// Reject globs that `globset` would happily compile but that cannot match any non-empty relative
/// path the walker produces. The four shapes are empirically derived against `globset` 0.4 — each
/// parses as a valid `Glob`, but their matchers return false on every plausible input. Catching
/// them here is the cheapest way to give the operator a "this glob does nothing" error instead of a
/// silent no-op.
///
/// Universal-match globs (`**`, `**/*`) are deliberately not rejected: they're equivalent to no
/// pattern, which is sometimes the intended shape (e.g. for the exclude list's "match everything
/// below" form `<name>/**`).
fn validate_source_reachability(s: &str) -> Result<(), ConfigError> {
    // Bytes test on the tail-`/` arm: `ends_with(char)` is cheap, and the `!ends_with("**")` clause
    // keeps `**` (universal-match) and `<name>/**` (a valid exclude shape) out of the rejection set.
    let reason = match s {
        "" => Some("glob is empty — matches no entry"),
        "." | ".." => Some(
            "lone `.` or `..` matches nothing — globs are matched against entry paths \
             relative to the anchor, which never equal `.` or `..`",
        ),
        s if s.starts_with('/') => Some(
            "leading `/` makes the glob absolute, but glob patterns are matched against \
             paths relative to the watch anchor — drop the `/` (e.g. `foo/**`, not `/foo/**`)",
        ),
        s if s.ends_with('/') && !s.ends_with("**") => Some(
            "trailing `/` matches no entry — use `<name>/**` to exclude contents, or \
             `<name>` to match the directory itself (gitignore-style `target/` is not supported)",
        ),
        _ => None,
    };
    if let Some(reason) = reason {
        return Err(ConfigError::UnreachableGlob {
            source: s.to_owned(),
            reason: reason.to_owned(),
        });
    }
    Ok(())
}

/// Canonical hash of `(ScanConfig, max_settle, events)` — the crate-internal hashing kernel.
/// [`ProfileIdentity::config_hash`] is the sole public route; production threads a
/// `ProfileIdentity` through that rather than calling this directly.
///
/// Inputs are folded in fixed order through [`crate::hash::hasher`]: a 1-byte scan-shape
/// discriminant, the variant's own fields (for `Subtree`: `recursive`, `hidden`, `len(exclude)` as
/// `u32`, each `exclude.source`, then `pattern` and `max_depth` each as a 1-byte presence flag plus
/// optional payload), then the shape-independent identity axes — `max_settle.as_nanos()` and
/// `events.bits()`.
///
/// The discriminant byte is chosen explicitly in each arm (not `mem::discriminant`), so the digest
/// is stable across variant reordering; it leads the fold so no two shapes' field encodings can
/// collide byte-for-byte. `events` is folded last so two Subs differing only on event-class mask
/// fork separate Profiles ("Profile-union infection" defence).
#[must_use]
pub(crate) fn compute_config_hash(
    scan: &ScanConfig,
    max_settle: Duration,
    events: ClassSet,
) -> u64 {
    let mut h = hasher();

    match scan {
        // Fold-completeness ratchet: this exhaustive destructure makes a new `Subtree` field a
        // compile error (E0027) until it is folded into the digest below — promoting
        // Profile-partition completeness from a hand-maintained test convention to a compiler
        // invariant. Never add `..`: it silently re-opens the silent-Profile-merge hole. A new
        // *variant* is a non-exhaustive-match compile error until it picks its own discriminant byte.
        ScanConfig::Subtree {
            recursive,
            hidden,
            exclude,
            pattern,
            max_depth,
        } => {
            h.put_u8(0);
            h.put_u8(u8::from(*recursive));
            h.put_u8(u8::from(*hidden));

            // Canonical width: u32 for the count. Saturate on the absurd overflow case — the
            // alternative is an explicit panic, which buys nothing for a config layer that cannot
            // realistically reach 2^32 globs.
            let exclude_count = u32::try_from(exclude.len()).unwrap_or(u32::MAX);
            h.put_u32(exclude_count);
            for g in exclude {
                h.put_str(g.source.as_str());
            }

            match pattern {
                Some(g) => {
                    h.put_u8(1);
                    h.put_str(g.source.as_str());
                }
                None => {
                    h.put_u8(0);
                }
            }

            match max_depth {
                Some(d) => {
                    h.put_u8(1);
                    h.put_u32(*d);
                }
                None => {
                    h.put_u8(0);
                }
            }
        }
        // Never hashed in production — a `Descent` is prober-internal and anchors no Profile. The
        // arm is total rather than `unreachable!` so a future caller hashing one gets a stable
        // digest, not a panic.
        ScanConfig::Descent => {
            h.put_u8(1);
        }
    }

    h.put_u128(max_settle.as_nanos());
    h.put_u8(events.bits());

    h.finish_u64()
}

/// The Profile partition key's config half, reified.
///
/// The inputs whose canonical hash decides which Profile a Sub joins (equal hash ⇒ shared
/// Profile/snapshot/burst). Deliberately neither `Hash` nor `Eq`/`Ord`: [`Self::config_hash`] is
/// the sole identity operation — a structural derive would be a second identity route that could
/// diverge from the hash the partition actually keys on.
#[derive(Clone, Debug)]
pub struct ProfileIdentity {
    pub config: ScanConfig,
    pub max_settle: Duration,
    pub events: ClassSet,
}

impl ProfileIdentity {
    /// Canonical hash of this identity — the single public hashing route for Profile partitioning.
    #[must_use]
    pub fn config_hash(&self) -> u64 {
        // Fold-completeness ratchet, identity tier: a new axis on `ProfileIdentity` is a compile
        // error here until it is threaded into the canonical hash below (the `ScanConfig` ratchet
        // does not cover an axis that is neither scan-config, max_settle, nor events).
        let Self {
            config,
            max_settle,
            events,
        } = self;
        compute_config_hash(config, *max_settle, *events)
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassSet, ConfigError, GlobPattern, ProfileIdentity, ResourceKind, ScanConfig,
        compute_config_hash,
    };
    use std::path::Path;
    use std::time::Duration;

    fn glob(source: &str) -> GlobPattern {
        GlobPattern::compile(source).expect("test glob compiles")
    }

    const SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    #[test]
    fn glob_compile_failure_returns_invalid_glob() {
        let err = GlobPattern::compile("[invalid").expect_err("malformed glob must fail");
        let ConfigError::InvalidGlob { source, message } = err else {
            panic!("expected InvalidGlob, got {err:?}");
        };
        assert_eq!(source, "[invalid");
        assert!(!message.is_empty());
    }

    #[test]
    fn glob_eq_over_source_only() {
        let a = glob("*.rs");
        let b = glob("*.rs");
        assert_eq!(a, b);
        assert_eq!(a.cmp(&b), std::cmp::Ordering::Equal);
    }

    #[test]
    fn glob_matches_path() {
        let g = glob("*.rs");
        assert!(g.matches_path(std::path::Path::new("foo.rs")));
        assert!(!g.matches_path(std::path::Path::new("foo.txt")));
    }

    #[test]
    fn excludes_sorted_at_build_regardless_of_insertion_order() {
        let cfg = ScanConfig::builder()
            .exclude(glob("z"))
            .exclude(glob("a"))
            .exclude(glob("m"))
            .build();
        let sources: Vec<&str> = cfg
            .exclude_globs()
            .iter()
            .map(GlobPattern::source)
            .collect();
        assert_eq!(sources, vec!["a", "m", "z"]);
    }

    #[test]
    fn empty_builder_yields_default_shaped_config() {
        let cfg = ScanConfig::builder().build();
        let ScanConfig::Subtree {
            recursive,
            hidden,
            exclude,
            pattern,
            max_depth,
        } = &cfg
        else {
            panic!("builder builds Subtree, got {cfg:?}");
        };
        assert!(!recursive);
        assert!(!hidden);
        assert!(exclude.is_empty());
        assert!(pattern.is_none());
        assert!(max_depth.is_none());
    }

    #[test]
    fn hash_deterministic() {
        let a = ScanConfig::builder().recursive(true).build();
        let b = ScanConfig::builder().recursive(true).build();
        assert_eq!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
    }

    #[test]
    fn hash_canonical_excludes() {
        let a = ScanConfig::builder()
            .exclude(glob("a"))
            .exclude(glob("z"))
            .exclude(glob("m"))
            .build();
        let b = ScanConfig::builder()
            .exclude(glob("z"))
            .exclude(glob("m"))
            .exclude(glob("a"))
            .build();
        assert_eq!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_max_settle() {
        let cfg = ScanConfig::builder().build();
        assert_ne!(
            compute_config_hash(&cfg, Duration::from_secs(1), NO_EVENTS),
            compute_config_hash(&cfg, Duration::from_secs(2), NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_recursive() {
        let a = ScanConfig::builder().recursive(true).build();
        let b = ScanConfig::builder().recursive(false).build();
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_hidden() {
        let a = ScanConfig::builder().hidden(true).build();
        let b = ScanConfig::builder().hidden(false).build();
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_pattern() {
        let a = ScanConfig::builder().pattern(glob("*.rs")).build();
        let b = ScanConfig::builder().pattern(glob("*.txt")).build();
        let c = ScanConfig::builder().build();
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&c, SETTLE, NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_max_depth() {
        let a = ScanConfig::builder().max_depth(Some(3)).build();
        let b = ScanConfig::builder().max_depth(Some(4)).build();
        let c = ScanConfig::builder().max_depth(None).build();
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&c, SETTLE, NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_exclude_set() {
        let a = ScanConfig::builder().exclude(glob("a")).build();
        let b = ScanConfig::builder().build();
        assert_ne!(
            compute_config_hash(&a, SETTLE, NO_EVENTS),
            compute_config_hash(&b, SETTLE, NO_EVENTS),
        );
    }

    /// `events` is part of `config_hash`. Two Subs differing only on the class mask must fork
    /// separate Profiles ("Profile-union infection" defence).
    #[test]
    fn hash_distinguishes_events_mask() {
        let cfg = ScanConfig::builder().build();
        let empty = compute_config_hash(&cfg, SETTLE, ClassSet::EMPTY);
        let content = compute_config_hash(&cfg, SETTLE, ClassSet::CONTENT);
        let metadata = compute_config_hash(&cfg, SETTLE, ClassSet::METADATA);
        let content_meta =
            compute_config_hash(&cfg, SETTLE, ClassSet::CONTENT | ClassSet::METADATA);
        // Pairwise distinct — every distinct mask produces a distinct hash.
        assert_ne!(empty, content);
        assert_ne!(empty, metadata);
        assert_ne!(empty, content_meta);
        assert_ne!(content, metadata);
        assert_ne!(content, content_meta);
        assert_ne!(metadata, content_meta);
    }

    /// `compute_config_hash` is order-stable across `events` mask: the canonical bit representation
    /// determines the fold, not call order.
    #[test]
    fn hash_events_is_canonical_over_or_order() {
        let cfg = ScanConfig::builder().build();
        let a = ClassSet::CONTENT | ClassSet::METADATA;
        let b = ClassSet::METADATA | ClassSet::CONTENT;
        assert_eq!(
            compute_config_hash(&cfg, SETTLE, a),
            compute_config_hash(&cfg, SETTLE, b),
        );
    }

    /// Discrimination complement to the fold-completeness destructure: the exhaustive pattern makes
    /// a new field a *compile error* until folded; this test makes the fold *distinguish*, so the
    /// ratchet cannot be satisfied by folding a new field as a constant.
    ///
    /// The base is fully-populated and non-default: toggling any single field must move the hash.
    /// (`hash_known_good` pins a *default* config, where a new field left at its default would
    /// never shift the digest — this is the structural complement that closes that gap.)
    ///
    /// Not itself ratcheted: a new field still needs a new `assert_ne!` line here. The compile
    /// error from the destructure is the forcing function that drives the author to add it; this
    /// test then guards that the fold actually discriminates.
    #[test]
    fn hash_discriminates_every_populated_field() {
        fn populated() -> ProfileIdentity {
            ProfileIdentity {
                config: ScanConfig::builder()
                    .recursive(true)
                    .hidden(true)
                    .exclude(glob("a"))
                    .exclude(glob("b"))
                    .pattern(glob("*.rs"))
                    .max_depth(Some(7))
                    .build(),
                max_settle: Duration::from_secs(7),
                events: ClassSet::CONTENT | ClassSet::METADATA,
            }
        }

        fn rehash(base: &ProfileIdentity, mutate: impl FnOnce(&mut ProfileIdentity)) -> u64 {
            let mut id = base.clone();
            mutate(&mut id);
            id.config_hash()
        }

        let base = populated();
        let h0 = base.config_hash();

        assert_ne!(
            h0,
            rehash(&base, |id| {
                let ScanConfig::Subtree { recursive, .. } = &mut id.config else {
                    unreachable!("builder builds Subtree")
                };
                *recursive = false;
            }),
            "recursive must discriminate"
        );
        assert_ne!(
            h0,
            rehash(&base, |id| {
                let ScanConfig::Subtree { hidden, .. } = &mut id.config else {
                    unreachable!("builder builds Subtree")
                };
                *hidden = false;
            }),
            "hidden must discriminate"
        );
        assert_ne!(
            h0,
            rehash(&base, |id| {
                let ScanConfig::Subtree { exclude, .. } = &mut id.config else {
                    unreachable!("builder builds Subtree")
                };
                exclude.push(glob("c"));
            }),
            "exclude must discriminate"
        );
        assert_ne!(
            h0,
            rehash(&base, |id| {
                let ScanConfig::Subtree { pattern, .. } = &mut id.config else {
                    unreachable!("builder builds Subtree")
                };
                *pattern = None;
            }),
            "pattern must discriminate"
        );
        assert_ne!(
            h0,
            rehash(&base, |id| {
                let ScanConfig::Subtree { max_depth, .. } = &mut id.config else {
                    unreachable!("builder builds Subtree")
                };
                *max_depth = Some(8);
            }),
            "max_depth must discriminate"
        );
        assert_ne!(
            h0,
            rehash(&base, |id| id.max_settle = Duration::from_secs(8)),
            "max_settle must discriminate"
        );
        assert_ne!(
            h0,
            rehash(&base, |id| id.events = ClassSet::CONTENT),
            "events must discriminate"
        );
    }

    /// The 1-byte scan-shape discriminant leads the fold: two shapes whose remaining byte streams
    /// could coincide still digest apart. Pinned on the only variant pair that exists today; a new
    /// shape extends this with its own pair.
    #[test]
    fn hash_distinguishes_scan_shape() {
        let subtree = ScanConfig::builder().build();
        assert_ne!(
            compute_config_hash(&subtree, SETTLE, NO_EVENTS),
            compute_config_hash(&ScanConfig::Descent, SETTLE, NO_EVENTS),
        );
    }

    /// Golden test — pins the canonical `compute_config_hash` encoding. Drift here changes every
    /// Profile's `config_hash` this binary computes; update only this constant after a deliberate
    /// review.
    #[test]
    fn hash_known_good() {
        let cfg = ScanConfig::builder().build();
        let h = compute_config_hash(&cfg, Duration::from_secs(1), ClassSet::EMPTY);
        assert_eq!(h, GOLDEN_HASH);
    }

    /// The public route (`ProfileIdentity::config_hash`) and the sealed kernel agree bit-for-bit on
    /// the golden preimage — sealing the kernel did not perturb the canonical encoding.
    #[test]
    fn profile_identity_config_hash_matches_golden() {
        let identity = ProfileIdentity {
            config: ScanConfig::builder().build(),
            max_settle: Duration::from_secs(1),
            events: ClassSet::EMPTY,
        };
        assert_eq!(identity.config_hash(), GOLDEN_HASH);
    }

    const GOLDEN_HASH: u64 = 0x2BE2_4E94_4F1C_30EB;

    /// Fold-completeness ratchet for [`ScanConfig::accepts`]. Mirrors
    /// [`hash_discriminates_every_populated_field`]: a fully-populated `ScanConfig` plus a neutral
    /// target; toggling each filter field must move at least one `accepts` verdict on a small test
    /// grid.
    ///
    /// The exhaustive destructure inside `accepts_structural` makes a new field a *compile error*
    /// until folded; this test makes the fold *discriminate*, so the ratchet cannot be satisfied by
    /// folding a new field as a constant.
    ///
    /// Not itself ratcheted: a new field still needs a new mutation closure here. The compile error
    /// from the destructure is the forcing function that drives the author to add it; this test
    /// then guards that the fold actually discriminates.
    #[test]
    fn accepts_reads_every_field() {
        fn populated() -> ScanConfig {
            ScanConfig::builder()
                .recursive(true)
                .hidden(true)
                .exclude(glob("excluded"))
                .pattern(glob("*.rs"))
                .max_depth(Some(7))
                .build()
        }

        fn verdicts(cfg: &ScanConfig, probes: &[(&str, ResourceKind, u32)]) -> Vec<bool> {
            probes
                .iter()
                .map(|(rel, kind, depth)| cfg.accepts(Path::new(rel), *kind, *depth))
                .collect()
        }

        fn shifted(
            base_v: &[bool],
            cfg: &ScanConfig,
            probes: &[(&str, ResourceKind, u32)],
        ) -> bool {
            verdicts(cfg, probes) != base_v
        }

        // Grid of probes: (rel, kind, depth). Each probe is chosen so a *single* field toggle
        // shifts its verdict — the other fields are picked to "pass" so the predicate stops on the
        // field under test. Verdicts overlap by design (the predicate short-circuits), so a probe
        // that's already rejected by a *different* field cannot discriminate.
        let probes: &[(&str, ResourceKind, u32)] = &[
            ("foo.rs", ResourceKind::File, 5),
            (".hidden_dir", ResourceKind::Dir, 1),
            ("foo.rs", ResourceKind::File, 7),
            ("excluded", ResourceKind::Dir, 1),
            ("foo.txt", ResourceKind::File, 1),
        ];

        let base = populated();
        let base_v = verdicts(&base, probes);

        let mut mut_recursive = populated();
        {
            let ScanConfig::Subtree { recursive, .. } = &mut mut_recursive else {
                unreachable!("builder builds Subtree")
            };
            *recursive = false;
        }
        assert!(
            shifted(&base_v, &mut_recursive, probes),
            "recursive must discriminate"
        );

        let mut mut_hidden = populated();
        {
            let ScanConfig::Subtree { hidden, .. } = &mut mut_hidden else {
                unreachable!("builder builds Subtree")
            };
            *hidden = false;
        }
        assert!(
            shifted(&base_v, &mut_hidden, probes),
            "hidden must discriminate"
        );

        let mut_exclude = ScanConfig::builder()
            .recursive(true)
            .hidden(true)
            .pattern(glob("*.rs"))
            .max_depth(Some(7))
            .build();
        assert!(
            shifted(&base_v, &mut_exclude, probes),
            "exclude must discriminate"
        );

        let mut_pattern = ScanConfig::builder()
            .recursive(true)
            .hidden(true)
            .exclude(glob("excluded"))
            .max_depth(Some(7))
            .build();
        assert!(
            shifted(&base_v, &mut_pattern, probes),
            "pattern must discriminate"
        );

        let mut mut_max_depth = populated();
        {
            let ScanConfig::Subtree { max_depth, .. } = &mut mut_max_depth else {
                unreachable!("builder builds Subtree")
            };
            *max_depth = Some(6);
        }
        assert!(
            shifted(&base_v, &mut_max_depth, probes),
            "max_depth must discriminate"
        );

        // Anchor-depth (`depth == 0`) bypasses every filter unconditionally. Pinning the load-bearing
        // "anchor is in scope by construction" property at the predicate's first line, so a future
        // structural rewrite that drops the short-circuit surfaces as a localised failure here.
        assert!(
            base.accepts(Path::new(""), ResourceKind::Dir, 0),
            "depth 0 must accept regardless of filters"
        );
    }
}
