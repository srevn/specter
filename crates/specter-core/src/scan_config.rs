//! Scan configuration.
//!
//! `ScanConfig` is the only data folded by [`compute_config_hash`], the
//! canonical hash of `(scan_config, max_settle)`. Two Subs whose hashes match
//! share one Profile; different `max_settle` values produce different hashes.
//!
//! `GlobPattern` carries the canonical `source` string alongside the compiled
//! `globset::GlobMatcher`. Equality and ordering are over `source` only — the
//! matcher is a transient compiled artifact. There is no `Hash` impl: the
//! single hashing path through these types is `compute_config_hash`, which
//! reads `source` directly via `core::hash::hasher`.

use crate::hash::hasher;
use crate::sub::ClassSet;
use compact_str::CompactString;
use globset::{Glob, GlobMatcher};
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::path::Path;
use std::time::Duration;

#[derive(Clone, Debug)]
pub struct GlobPattern {
    source: CompactString,
    matcher: GlobMatcher,
}

impl GlobPattern {
    /// Compile a glob pattern. Returns [`ConfigError::InvalidGlob`] with the
    /// `globset` error message rendered into a `String` (the original
    /// `globset::Error` is not `Clone`).
    pub fn compile(source: impl Into<CompactString>) -> Result<Self, ConfigError> {
        let source: CompactString = source.into();
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

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ScanConfig {
    pub recursive: bool,
    pub hidden: bool,
    /// Sorted by source string at builder time.
    pub exclude: Vec<GlobPattern>,
    pub pattern: Option<GlobPattern>,
    pub max_depth: Option<u32>,
}

impl ScanConfig {
    #[must_use]
    pub fn builder() -> ScanConfigBuilder {
        ScanConfigBuilder::default()
    }
}

/// Builder for [`ScanConfig`]. Sorts `exclude` by source on `build()` so
/// equal logical configs are byte-equal — `compute_config_hash` reads in
/// already-sorted order.
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
        ScanConfig {
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
    InvalidGlob { source: String, message: String },
}

/// Canonical hash of `(ScanConfig, max_settle, events)`.
/// The single hashing entry point in `core`/`engine` — all other paths
/// route through this for `config_hash` derivation.
///
/// Inputs are folded in fixed order through [`crate::hash::hasher`]:
///   `recursive`, `hidden`, `len(exclude)` as `u32`, each `exclude.source`,
///   pattern presence-byte plus optional source, `max_depth`,
///   `max_settle.as_nanos()`, `events.bits()`.
///
/// `events` is folded last so two Subs differing only on event-class mask
/// fork separate Profiles ("Profile-union infection" defence).
#[must_use]
pub fn compute_config_hash(scan: &ScanConfig, max_settle: Duration, events: ClassSet) -> u64 {
    let mut h = hasher();

    scan.recursive.hash(&mut h);
    scan.hidden.hash(&mut h);

    // Canonical width: u32 for the count. Saturate on the absurd
    // overflow case — the alternative is an explicit panic, which buys
    // nothing for a config layer that cannot realistically reach 2^32 globs.
    let exclude_count = u32::try_from(scan.exclude.len()).unwrap_or(u32::MAX);
    exclude_count.hash(&mut h);
    for g in &scan.exclude {
        g.source.as_str().hash(&mut h);
    }

    match &scan.pattern {
        Some(g) => {
            1u8.hash(&mut h);
            g.source.as_str().hash(&mut h);
        }
        None => {
            0u8.hash(&mut h);
        }
    }

    scan.max_depth.hash(&mut h);
    max_settle.as_nanos().hash(&mut h);
    events.bits().hash(&mut h);

    h.finish()
}

#[cfg(test)]
mod tests {
    use super::{ClassSet, ConfigError, GlobPattern, ScanConfig, compute_config_hash};
    use std::time::Duration;

    fn glob(source: &str) -> GlobPattern {
        GlobPattern::compile(source).expect("test glob compiles")
    }

    const SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    #[test]
    fn glob_compile_failure_returns_invalid_glob() {
        let err = GlobPattern::compile("[invalid").expect_err("malformed glob must fail");
        let ConfigError::InvalidGlob { source, message } = err;
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
        let sources: Vec<&str> = cfg.exclude.iter().map(GlobPattern::source).collect();
        assert_eq!(sources, vec!["a", "m", "z"]);
    }

    #[test]
    fn empty_builder_yields_default_shaped_config() {
        let cfg = ScanConfig::builder().build();
        assert!(!cfg.recursive);
        assert!(!cfg.hidden);
        assert!(cfg.exclude.is_empty());
        assert!(cfg.pattern.is_none());
        assert!(cfg.max_depth.is_none());
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

    /// `events` is part of `config_hash`. Two Subs differing only on
    /// the class mask must fork separate Profiles ("Profile-union
    /// infection" defence).
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

    /// `compute_config_hash` is order-stable across `events` mask: the
    /// canonical bit representation determines the fold, not call order.
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

    /// Golden test — pins `SipHash` key + canonical encoding. Drift here is a
    /// *breaking* change for any persisted Profile hash; rotate intentionally
    /// and update only this constant.
    #[test]
    fn hash_known_good() {
        let cfg = ScanConfig::builder().build();
        let h = compute_config_hash(&cfg, Duration::from_secs(1), ClassSet::EMPTY);
        assert_eq!(h, GOLDEN_HASH);
    }

    const GOLDEN_HASH: u64 = 0x35A4_7E13_BE87_7324;
}
