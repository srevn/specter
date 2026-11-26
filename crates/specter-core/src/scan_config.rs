//! Scan configuration.
//!
//! [`compute_config_hash`] is the canonical hash of `(ScanConfig, max_settle, events)`;
//! [`ProfileIdentity`] is its reified preimage, sealed over the digest at construction —
//! [`ProfileIdentity::config_hash`] is the sole public route and reads the sealed value. Equal hash
//! ⇒ Subs share one Profile (snapshot, burst lifecycle). Both fold sites force completeness — the
//! kernel's exhaustive destructure per scan shape, the constructor's exhaustive struct literal per
//! identity axis — so a new identity-bearing field is a compile error until folded — the
//! fold-completeness ratchet.
//!
//! A [`SubtreeScan`] is a `[[watch]]`'s scan knobs; a [`ScanConfig`] is what a walk reads — a
//! subtree scan, or a pattern's chain. An identity is built from either: [`ProfileIdentity::new`]
//! from a subtree scan, the identity of a Profile that fires, and [`ProfileIdentity::discovery`]
//! from a pattern alone, whose mask and ceiling it fixes.
//!
//! `GlobPattern` carries the canonical `source` string alongside the compiled
//! `globset::GlobMatcher`. Equality and ordering are over `source` only — the matcher is a
//! transient compiled artifact. There is no `Hash` impl: the single hashing path through these
//! types is `compute_config_hash`, which reads `source` directly via `core::hash::hasher`.

use crate::hash::hasher;
use crate::pattern::PatternSpec;
use crate::resource::ResourceKind;
use crate::sub::ClassSet;
use compact_str::CompactString;
use globset::{Glob, GlobMatcher};
use std::cmp::Ordering;
use std::num::NonZeroU32;
use std::path::Path;
use std::sync::Arc;
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

/// A `[[watch]]`'s scan knobs: the cumulative predicate one walk of a subtree admits.
///
/// Every filter narrows what the walk lists. A firing Profile's identity is built from one
/// ([`ProfileIdentity::new`]); a walk reads it as [`ScanConfig::Subtree`]. [`SubtreeScanBuilder`]
/// is its one constructor, so `exclude` is in the canonical order the hash folds.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SubtreeScan {
    hidden: bool,
    exclude: Vec<GlobPattern>,
    pattern: Option<GlobPattern>,
    /// How deep the walk lists, the anchor's own entries being depth one; `None` is unbounded.
    max_depth: Option<NonZeroU32>,
}

impl SubtreeScan {
    #[must_use]
    pub fn builder() -> SubtreeScanBuilder {
        SubtreeScanBuilder::default()
    }

    /// The classes a mask must cover for silence to witness a subtree's quiescence: the `Subtree`
    /// arm of [`ScanConfig::quiescence_witness_classes`], for a holder of the knobs alone.
    #[must_use]
    pub const fn quiescence_witness_classes(&self) -> ClassSet {
        ClassSet::IN_PLACE_WRITES
    }
}

/// The one constructor of a [`SubtreeScan`].
///
/// `build` canonicalises `exclude` — sorted by source, adjacent duplicates dropped — so equal
/// logical knobs are byte-equal under `compute_config_hash`, which folds the list in that order.
///
/// Its default is a `[[watch]]` that writes none of these knobs: every depth, no dotfiles, no filter.
#[derive(Debug, Default)]
pub struct SubtreeScanBuilder {
    hidden: bool,
    exclude: Vec<GlobPattern>,
    pattern: Option<GlobPattern>,
    max_depth: Option<NonZeroU32>,
}

impl SubtreeScanBuilder {
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
    pub const fn max_depth(mut self, bound: Option<NonZeroU32>) -> Self {
        self.max_depth = bound;
        self
    }

    #[must_use]
    pub fn build(mut self) -> SubtreeScan {
        // Sort then collapse duplicates: `GlobPattern` equality is source-keyed, so after the sort
        // every duplicate source is adjacent and `dedup` removes it exactly. This is the
        // canonicalisation `compute_config_hash` relies on — a logically equal exclude set must
        // produce one Profile, not fork on accidental repetition.
        self.exclude.sort();
        self.exclude.dedup();
        SubtreeScan {
            hidden: self.hidden,
            exclude: self.exclude,
            pattern: self.pattern,
            max_depth: self.max_depth,
        }
    }
}

impl From<SubtreeScan> for ScanConfig {
    fn from(scan: SubtreeScan) -> Self {
        Self::Subtree(scan)
    }
}

/// What a walk reads and what `covers` tests: a subtree scan, or a pattern's chain.
///
/// Every consumer goes through the named predicates ([`Self::accepts`],
/// [`Self::accepts_structural`], [`Self::accepts_kinded`], [`Self::descends_into`],
/// [`Self::exclude_globs`], [`Self::quiescence_witness_classes`], [`Self::denial`],
/// [`Self::match_chain`]) or the hash; no consumer destructures the variants — shape dispatch lives
/// entirely in this module, so a new shape is a compile error here and nowhere else.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ScanConfig {
    /// One walk of a subtree under a `[[watch]]`'s knobs.
    Subtree(SubtreeScan),
    /// Positional per-segment chain over a parsed absolute pattern, anchored at the pattern's
    /// literal prefix. A dirent at anchor-relative depth `d` tests pattern component
    /// `literal_prefix_len + d − 1` ([`PatternSpec::matches_at`]); coverage ends at the terminus
    /// depth ([`PatternSpec::terminus_depth`]), where matched directories stay unexplored
    /// (`DirChild::Uncovered`) and matched files are ordinary leaves — the pruned walk observes
    /// chain membership, not subtree content.
    ///
    /// `Arc`: one parsed pattern travels from a dynamic watch's target into its chain's identity,
    /// each hop a refcount bump instead of a copy of the parsed components and their matchers.
    MatchChain(Arc<PatternSpec>),
}

impl ScanConfig {
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
    /// **Depth 0 bypasses every filter.** The anchor is part of the Profile's scope by construction.
    /// For `depth > 0`, `Subtree` folds `max_depth`, `hidden` (last-segment basename test), `exclude`
    /// (full-`rel` test), and `pattern` (final-`File` only — directories are always covered; we
    /// descend through them); `MatchChain` tests the last segment against its positional component
    /// and applies the chain kind rule (mid-chain must be Dir; the terminus admits any kind).
    ///
    /// `ResourceKind::Unknown` is collapsed by upstream callers (`Resource::kind_or_file`,
    /// `From<EntryKind>`) before reaching this method, so `kind != Dir` here means the same thing
    /// as the walker's `!is_dir`.
    #[must_use]
    pub fn accepts(&self, rel: &Path, kind: ResourceKind, depth: u32) -> bool {
        self.accepts_structural(rel, depth)
            && self.accepts_kinded(rel, matches!(kind, ResourceKind::Dir), depth)
    }

    /// The kind-independent half of [`Self::accepts`]. For `Subtree`, folds the three gates that
    /// don't need a `ResourceKind`: `max_depth`, `hidden`, and `exclude`. For `MatchChain`, the
    /// positional segment test — the last segment of `rel` against the pattern component at `depth`
    /// (the depth bound is [`PatternSpec::matches_at`]'s own totality: out-of-range depths are
    /// `false`).
    ///
    /// Exists for the walker, which doesn't know `is_dir` until after the per-dirent `lstat`. Calling
    /// [`Self::accepts`] there with a guessed kind would either skip a covered Dir (guess `File` +
    /// `pattern.matches == false`) or admit a pattern-violating File (guess `Dir`). Splitting lets
    /// the walker reject out-of-scope dirents pre-`lstat` — saving the syscall on a `target/` tree in
    /// a Cargo project (thousands of excluded dirents) — and gate the kinded half post-`lstat`
    /// against the known kind. `covers` has `kind` in hand and calls [`Self::accepts`] directly.
    ///
    /// **Fold-completeness ratchet, predicate tier.** The `Subtree` arm destructures exhaustively:
    /// a new field of [`SubtreeScan`] is a compile error here until the author decides which half
    /// it folds into — structural (a clause below) or kind-dependent (an arm in
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
            Self::Subtree(SubtreeScan {
                hidden,
                exclude,
                pattern: _,
                max_depth,
            }) => {
                if max_depth.is_some_and(|max| depth > max.get()) {
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
            // The positional test is the whole arm: `matches_at` already folds the depth bound
            // (beyond-terminus ⇒ false). Non-UTF-8 names are conservatively out of scope — the
            // walker drops them before any predicate and Tree names are UTF-8 by construction, so
            // the `None` arm is totality, not a live path.
            Self::MatchChain(spec) => rel
                .file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|seg| spec.matches_at(depth, seg)),
        }
    }

    /// The kind-dependent half of [`Self::accepts`]: the residual gate that needs to know whether
    /// the entry is a directory. For `Subtree`, the `pattern` arm — files must match the pattern
    /// when one is set; directories are always covered (the walker descends through them, and
    /// `covers` checks them as intermediate prefixes). For `MatchChain`, the positional kind rule —
    /// an intermediate chain position names a directory the walk descends through, so a mid-chain
    /// non-dir is out of scope; the terminus admits any kind (a match terminates there regardless).
    ///
    /// One home for both consumers: [`Self::accepts`] passes `kind == Dir`, and the walker calls
    /// this directly post-`lstat` with the freshly-observed `is_dir` — the two agree because
    /// `ResourceKind` folds every non-directory (symlinks included) to `File`, so `kind != Dir` ⟺
    /// `!is_dir` for every kind `accepts` can receive. `depth` shares the caller's one derivation
    /// (the walker's `rel`-derived `entry_depth`; `covers`' chain position) — never recomputed
    /// here, symmetric with [`Self::accepts_structural`].
    #[must_use]
    pub fn accepts_kinded(&self, rel: &Path, is_dir: bool, depth: u32) -> bool {
        // Anchor depth bypasses the kinded gate too — same "in scope by construction" contract as
        // the structural half, now expressible since the depth is in hand.
        if depth == 0 {
            return true;
        }
        match self {
            Self::Subtree(SubtreeScan { pattern, .. }) => {
                is_dir || pattern.as_ref().is_none_or(|pat| pat.matches_path(rel))
            }
            Self::MatchChain(spec) => is_dir || depth >= spec.terminus_depth(),
        }
    }

    /// True iff the walk's coverage extends below a directory sitting at `child_depth` (anchor = 0,
    /// direct child = 1, …) — the recursion-edge decision, distinct from [`Self::accepts`]'s
    /// per-entry inclusion decision.
    ///
    /// The walker supplies `same_device` — its per-dirent observation that the child's device
    /// equals the recursion root's. The observation cannot originate here (the engine's `Tree`
    /// slots don't carry `device`), but whether it *matters* is the shape's policy:
    ///
    /// - `Subtree` descends under `max_depth` and on the same device — an unbounded walk must not
    ///   wander into mounts.
    /// - `MatchChain` descends strictly above the terminus depth, device-blind: the walk is bounded
    ///   by the chain length (no runaway-cost concern), and chain positions may legitimately sit on
    ///   distinct mounts (one matched volume per level).
    ///
    /// Negation drives `DirChild::Uncovered(fs_id)` emission in the walker — uncovered-by-mount,
    /// beyond-`max_depth`, and the chain terminus all fall out of the same edge.
    #[must_use]
    pub fn descends_into(&self, child_depth: u32, same_device: bool) -> bool {
        match self {
            Self::Subtree(SubtreeScan { max_depth, .. }) => {
                max_depth.is_none_or(|max| child_depth < max.get()) && same_device
            }
            Self::MatchChain(spec) => child_depth < spec.terminus_depth(),
        }
    }

    /// The exclude globs this scan shape filters with — `Subtree`'s `exclude` list; the empty slice
    /// for shapes that carry none. Effect-env projection (`Profile::exclude_strings`) consumes this
    /// so it never destructures the shape.
    #[must_use]
    pub fn exclude_globs(&self) -> &[GlobPattern] {
        match self {
            Self::Subtree(SubtreeScan { exclude, .. }) => exclude,
            Self::MatchChain(_) => &[],
        }
    }

    /// The positional pattern iff this is the `MatchChain` shape: the discovery discriminant. Every
    /// consumer that dispatches on the discovery shape, or reads its pattern, asks this projection,
    /// so no variant destructure leaves this module.
    #[must_use]
    pub const fn match_chain(&self) -> Option<&Arc<PatternSpec>> {
        match self {
            Self::MatchChain(spec) => Some(spec),
            Self::Subtree(_) => None,
        }
    }

    /// The event classes a Profile's mask must cover for settle-window silence to witness this
    /// shape's quiescence — the shape half of the witness criterion the verdict floor reads through
    /// `Profile::events_witness_quiescence`.
    ///
    /// The scan shape determines the proof object, and the proof object determines which change
    /// classes could cross a settle window invisibly:
    /// - `Subtree` proves a content hash ⇒ [`ClassSet::IN_PLACE_WRITES`]: in-place writes are the
    ///   one change kind that moves `leaf_hash` without a point event.
    /// - `MatchChain` proves a match set ⇒ [`ClassSet::MEMBERSHIP_CHANGES`]: membership changes are
    ///   STRUCTURE point events; no in-place-write analog exists for set membership (the
    ///   leaf-content residual is documented at that constant).
    ///
    /// The returned set is always non-empty. [`ClassSet::contains`] reports `false` for an empty
    /// argument, so a hypothetical "silence always witnesses" shape must not encode itself as
    /// `EMPTY` here — it would read as *never* witnessing. Such a shape needs an explicit fold at
    /// the projection instead.
    #[must_use]
    pub const fn quiescence_witness_classes(&self) -> ClassSet {
        match self {
            Self::Subtree(scan) => scan.quiescence_witness_classes(),
            Self::MatchChain(_) => ClassSet::MEMBERSHIP_CHANGES,
        }
    }

    /// What a directory the walk could not read means to this shape's proof object.
    #[must_use]
    pub const fn denial(&self) -> Denial {
        match self {
            Self::Subtree(_) => Denial::Hole,
            Self::MatchChain(_) => Denial::Bound,
        }
    }
}

/// What a directory the walk could not read means to a shape's proof object, and so what a proof
/// that still cannot read it does at its ceiling.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Denial {
    /// The proof object is content, and a denied directory is a hole in it: the proof refuses.
    Hole,
    /// The proof object is the set of matches the daemon may read, and a denied directory bounds
    /// it: the proof goes on over what was read, as a shell glob goes on past a directory it may
    /// not list.
    Bound,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ConfigError {
    /// `globset` rejected the glob source (e.g. unbalanced brackets, malformed alternation).
    /// `message` is the rendered `globset::Error` text (the original is not `Clone`).
    InvalidGlob { source: String, message: String },
    /// The glob is syntactically valid per `globset` but cannot match any non-empty relative path
    /// in the walker's semantics. Surfaced at the [`GlobPattern::compile`] floor so a typo
    /// (`target/` vs `target/**`) or a misunderstanding (leading `/` in a relative-path glob)
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
    // The tail-`/` arm rejects any glob whose last byte is `/`. `ends_with('/')` is a cheap
    // single-byte test and needs no companion guard: `**` (universal-match) and `<name>/**` (the
    // valid exclude shape) end in `*`, not `/`, so they can never reach this arm in the first place.
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
        s if s.ends_with('/') => Some(
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
/// `ProfileIdentity`'s two constructors are its only production callers: each seals the digest onto
/// the identity, and [`ProfileIdentity::config_hash`] reads that sealed value.
///
/// The discriminant byte is chosen explicitly in each arm (not `mem::discriminant`), so the digest
/// is stable across variant reordering; it leads the fold so no two shapes' field encodings can
/// collide byte-for-byte. `events` folds too, so two Subs differing only on their event-class mask
/// fork separate Profiles ("Profile-union infection" defence).
#[must_use]
pub(crate) fn compute_config_hash(
    scan: &ScanConfig,
    max_settle: Duration,
    events: ClassSet,
) -> u64 {
    let mut h = hasher();

    match scan {
        // Fold-completeness ratchet: this exhaustive destructure makes a new `SubtreeScan` field a
        // compile error (E0027) until it is folded into the digest below — promoting
        // Profile-partition completeness from a hand-maintained test convention to a compiler
        // invariant. Never add `..`: it silently re-opens the silent-Profile-merge hole. A new
        // *variant* is a non-exhaustive-match compile error until it picks its own discriminant byte.
        ScanConfig::Subtree(SubtreeScan {
            hidden,
            exclude,
            pattern,
            max_depth,
        }) => {
            h.put_u8(0);
            h.put_u8(u8::from(*hidden));

            // Canonical width: u32 for the count. The list arrives sorted *and* deduplicated from
            // `SubtreeScanBuilder::build`, so this count is over the canonical set — `["a", "a"]` and
            // `["a"]` fold identically rather than forking two Profiles on a logically equal exclude
            // set. Saturate on the absurd overflow case — the alternative is an explicit panic, which
            // buys nothing for a config layer that cannot realistically reach 2^32 globs.
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

            // Zero is no bound, so the bound's niche is the encoding of "unbounded".
            h.put_u32(max_depth.map_or(0, NonZeroU32::get));
        }
        // `source` is the complete identity encoding of the positional predicate: it spells the
        // canonical literal prefix plus the glob tail, and equal source ⇒ byte-equal decomposition
        // within the uniformly re-anchored production population (config lowering always
        // canonicalises then `reanchor`s). Folding the string therefore folds the whole predicate.
        // `source` isn't re-parsed and is not guaranteed to round-trip through `PatternSpec::parse`
        // (a canonical prefix segment may carry a glob metacharacter); the redundant prefix half is
        // harmless here because the Profile partition key's anchor-resource component independently
        // separates distinct anchors — this fold's job is to separate distinct *tails* under one
        // anchor (`/a/*.log` vs `/a/*.txt`).
        ScanConfig::MatchChain(spec) => {
            h.put_u8(1);
            h.put_str(spec.source());
        }
    }

    h.put_u128(max_settle.as_nanos());
    h.put_u8(events.bits());

    h.finish_u64()
}

/// The settle a discovery Profile is founded at, and a template's debounce.
///
/// Never a watch's own `settle`, which debounces its reaction: a `settle = "30s"` must not become
/// thirty seconds of mint latency. 150 ms coalesces an untar-style burst of entries into one pass.
pub const DISCOVERY_SETTLE: Duration = Duration::from_millis(150);
/// A discovery Profile's ceiling: it bounds mint latency under sustained chain churn.
const DISCOVERY_MAX_SETTLE: Duration = Duration::from_secs(2);
// The order `Profile::new` asserts of every Profile: no settle above its ceiling.
const _: () = assert!(DISCOVERY_SETTLE.as_nanos() <= DISCOVERY_MAX_SETTLE.as_nanos());

/// The Profile partition key's config half, reified.
///
/// The inputs whose canonical hash decides which Profile a Sub joins (equal hash ⇒ shared
/// Profile/snapshot/burst). Deliberately neither `Hash` nor `Eq`/`Ord`: [`Self::config_hash`] is
/// the sole identity operation — a structural derive would be a second identity route that could
/// diverge from the hash the partition actually keys on.
///
/// Sealed: private fields, two constructors, and the hash computed once inside them. "The hash
/// agrees with its preimage" is thereby a structural property, not a documented convention — no
/// construction or mutation path can produce an identity whose `hash` disagrees with its axes, so
/// every carrier reads the same precomputed key instead of re-deriving it.
///
/// `config` sits behind an `Arc`, so an identity, and the scan it ships onto every probe, travels
/// as a refcount bump, never a deep copy; the other fields are `Copy`, so `ProfileIdentity: Clone`
/// is cheap by construction.
#[derive(Clone, Debug)]
pub struct ProfileIdentity {
    config: Arc<ScanConfig>,
    max_settle: Duration,
    events: ClassSet,
    /// The Profile partition key itself — [`compute_config_hash`] over the three axes above.
    hash: u64,
}

impl ProfileIdentity {
    /// A firing Profile's identity: a subtree scan under a ceiling and a mask. It takes no chain:
    ///
    /// ```compile_fail
    /// use specter_core::{ClassSet, PatternSpec, ProfileIdentity, ScanConfig};
    /// use std::{sync::Arc, time::Duration};
    /// let chain = ScanConfig::MatchChain(Arc::new(PatternSpec::parse("/srv/*").unwrap()));
    /// let _ = ProfileIdentity::new(chain, Duration::from_secs(2), ClassSet::STRUCTURE);
    /// ```
    #[must_use]
    pub fn new(scan: SubtreeScan, max_settle: Duration, events: ClassSet) -> Self {
        Self::seal(ScanConfig::Subtree(scan), max_settle, events)
    }

    /// The identity of the Profile that walks `pattern`'s chain: its mask is the chain's witness
    /// classes and its ceiling the discovery pair's, so one pattern is one Profile whatever the
    /// watches over it say, and settle-window silence witnesses its match set.
    #[must_use]
    pub fn discovery(pattern: Arc<PatternSpec>) -> Self {
        Self::seal(
            ScanConfig::MatchChain(pattern),
            DISCOVERY_MAX_SETTLE,
            ClassSet::MEMBERSHIP_CHANGES,
        )
    }

    /// Freeze the three identity axes and seal their canonical hash in one step.
    fn seal(config: ScanConfig, max_settle: Duration, events: ClassSet) -> Self {
        // Fold-completeness ratchet, identity tier: a new axis on `ProfileIdentity` is a compile
        // error at this literal until it is given a field here — and it must also be threaded into
        // the canonical hash above, or the axis silently merges Profiles (the `ScanConfig` ratchet
        // inside `compute_config_hash` does not cover an axis that is neither scan-config,
        // max_settle, nor events).
        let hash = compute_config_hash(&config, max_settle, events);
        Self {
            config: Arc::new(config),
            max_settle,
            events,
            hash,
        }
    }

    /// The frozen [`ScanConfig`] axis. Borrow for the shape predicates (`match_chain`,
    /// `exclude_globs`, the witness classes); consumers never take ownership through this. Not
    /// `const`: the read derefs through the `Arc`.
    #[must_use]
    pub fn config(&self) -> &ScanConfig {
        &self.config
    }

    /// The same frozen [`ScanConfig`] behind its sharing handle — for carriers that ship the config
    /// onward as a refcount bump (the Profile's `config_shared`, and through it the probe wire).
    /// Every plain reader borrows through [`Self::config`] instead.
    #[must_use]
    pub const fn config_shared(&self) -> &Arc<ScanConfig> {
        &self.config
    }

    /// The settle-deadline ceiling axis (folds into [`Self::config_hash`]).
    #[must_use]
    pub const fn max_settle(&self) -> Duration {
        self.max_settle
    }

    /// The user-declared event-class mask axis (folds into [`Self::config_hash`]).
    #[must_use]
    pub const fn events(&self) -> ClassSet {
        self.events
    }

    /// Canonical hash of this identity — the single public hashing route for Profile partitioning.
    /// A field read: the fold ran once at [`Self::new`].
    #[must_use]
    pub const fn config_hash(&self) -> u64 {
        self.hash
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ClassSet, ConfigError, GlobPattern, PatternSpec, ProfileIdentity, ResourceKind, ScanConfig,
        SubtreeScan, compute_config_hash,
    };
    use std::num::NonZeroU32;
    use std::path::Path;
    use std::sync::Arc;
    use std::time::Duration;

    fn glob(source: &str) -> GlobPattern {
        GlobPattern::compile(source).expect("test glob compiles")
    }

    fn pattern(source: &str) -> Arc<PatternSpec> {
        Arc::new(PatternSpec::parse(source).expect("test pattern parses"))
    }

    fn chain(source: &str) -> ScanConfig {
        ScanConfig::MatchChain(pattern(source))
    }

    const SETTLE: Duration = Duration::from_secs(6);
    const NO_EVENTS: ClassSet = ClassSet::EMPTY;

    /// `scan`'s hash under the suite's ceiling and an empty mask.
    fn hash(scan: SubtreeScan) -> u64 {
        compute_config_hash(&scan.into(), SETTLE, NO_EVENTS)
    }

    #[test]
    fn glob_compile_failure_returns_invalid_glob() {
        let err = GlobPattern::compile("[invalid").expect_err("malformed glob must fail");
        let ConfigError::InvalidGlob { source, message } = err else {
            panic!("expected InvalidGlob, got {err:?}");
        };
        assert_eq!(source, "[invalid");
        assert_ne!(message, "");
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
        let scan = SubtreeScan::builder()
            .exclude(glob("z"))
            .exclude(glob("a"))
            .exclude(glob("m"))
            .build();
        let sources: Vec<&str> = scan.exclude.iter().map(GlobPattern::source).collect();
        assert_eq!(sources, vec!["a", "m", "z"]);
    }

    #[test]
    fn empty_builder_yields_default_shaped_config() {
        let SubtreeScan {
            hidden,
            exclude,
            pattern,
            max_depth,
        } = SubtreeScan::builder().build();
        assert!(!hidden);
        assert_eq!(exclude.as_slice(), []);
        assert!(pattern.is_none());
        assert!(max_depth.is_none());
    }

    #[test]
    fn hash_deterministic() {
        assert_eq!(
            hash(SubtreeScan::builder().build()),
            hash(SubtreeScan::builder().build()),
        );
    }

    #[test]
    fn hash_canonical_excludes() {
        let a = SubtreeScan::builder()
            .exclude(glob("a"))
            .exclude(glob("z"))
            .exclude(glob("m"))
            .build();
        let b = SubtreeScan::builder()
            .exclude(glob("z"))
            .exclude(glob("m"))
            .exclude(glob("a"))
            .build();
        assert_eq!(hash(a), hash(b));
    }

    /// Duplicate excludes collapse at `build()`: `["a", "a", "z"]` and `["a", "z"]` are the same
    /// logical config, so they must share one Profile. Without the `dedup()`, `compute_config_hash`
    /// folds the count (`3` vs `2`) and each source, forking two Profiles on a logically equal set.
    /// Pins both the canonical list shape and the hash equality.
    #[test]
    fn build_collapses_duplicate_excludes() {
        let with_dup = SubtreeScan::builder()
            .exclude(glob("a"))
            .exclude(glob("a"))
            .exclude(glob("z"))
            .build();
        let without_dup = SubtreeScan::builder()
            .exclude(glob("a"))
            .exclude(glob("z"))
            .build();

        let sources: Vec<&str> = with_dup.exclude.iter().map(GlobPattern::source).collect();
        assert_eq!(sources, vec!["a", "z"], "duplicate dropped at build");

        assert_eq!(
            hash(with_dup),
            hash(without_dup),
            "a duplicated exclude must not fork a second Profile",
        );
    }

    #[test]
    fn hash_distinguishes_max_settle() {
        let cfg = ScanConfig::from(SubtreeScan::builder().build());
        assert_ne!(
            compute_config_hash(&cfg, Duration::from_secs(1), NO_EVENTS),
            compute_config_hash(&cfg, Duration::from_secs(2), NO_EVENTS),
        );
    }

    #[test]
    fn hash_distinguishes_hidden() {
        assert_ne!(
            hash(SubtreeScan::builder().hidden(true).build()),
            hash(SubtreeScan::builder().hidden(false).build()),
        );
    }

    #[test]
    fn hash_distinguishes_pattern() {
        let rs = hash(SubtreeScan::builder().pattern(glob("*.rs")).build());
        assert_ne!(
            rs,
            hash(SubtreeScan::builder().pattern(glob("*.txt")).build())
        );
        assert_ne!(rs, hash(SubtreeScan::builder().build()));
    }

    /// Bounds that walk differently hash apart.
    #[test]
    fn hash_distinguishes_max_depth() {
        let [unbounded, one, three] = [None, NonZeroU32::new(1), NonZeroU32::new(3)]
            .map(|bound| hash(SubtreeScan::builder().max_depth(bound).build()));
        assert_ne!(unbounded, one);
        assert_ne!(unbounded, three);
        assert_ne!(one, three);
    }

    #[test]
    fn hash_distinguishes_exclude_set() {
        assert_ne!(
            hash(SubtreeScan::builder().exclude(glob("a")).build()),
            hash(SubtreeScan::builder().build()),
        );
    }

    /// `events` is part of `config_hash`. Two Subs differing only on the class mask must fork
    /// separate Profiles ("Profile-union infection" defence).
    #[test]
    fn hash_distinguishes_events_mask() {
        let cfg = ScanConfig::from(SubtreeScan::builder().build());
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
        let cfg = ScanConfig::from(SubtreeScan::builder().build());
        let a = ClassSet::CONTENT | ClassSet::METADATA;
        let b = ClassSet::METADATA | ClassSet::CONTENT;
        assert_eq!(
            compute_config_hash(&cfg, SETTLE, a),
            compute_config_hash(&cfg, SETTLE, b),
        );
    }

    /// Discrimination complement to the fold-completeness ratchets: the compile errors at the fold
    /// sites force a new field into the fold; this test makes the fold *distinguish*, so the
    /// ratchet cannot be satisfied by folding a new field as a constant.
    ///
    /// The base is fully-populated and non-default: toggling any single field must move the hash.
    /// (`hash_known_good` pins a *default* config, where a new field left at its default would
    /// never shift the digest — this is the structural complement that closes that gap.)
    ///
    /// Each variation builds a fresh identity from mutated *inputs* — the hash seals at
    /// construction, so every probe runs the same public route production uses
    /// (`ProfileIdentity::new` → `config_hash`).
    ///
    /// Not itself ratcheted: a new field still needs a new `assert_ne!` line here. The compile
    /// error from the fold sites is the forcing function that drives the author to add it; this
    /// test then guards that the fold actually discriminates.
    #[test]
    fn hash_discriminates_every_populated_field() {
        const BASE_MAX_SETTLE: Duration = Duration::from_secs(7);
        fn base_events() -> ClassSet {
            ClassSet::CONTENT | ClassSet::METADATA
        }
        fn populated_scan() -> SubtreeScan {
            SubtreeScan::builder()
                .hidden(true)
                .exclude(glob("a"))
                .exclude(glob("b"))
                .pattern(glob("*.rs"))
                .max_depth(NonZeroU32::new(7))
                .build()
        }
        fn rehash_scan(mutate: impl FnOnce(&mut SubtreeScan)) -> u64 {
            let mut scan = populated_scan();
            mutate(&mut scan);
            ProfileIdentity::new(scan, BASE_MAX_SETTLE, base_events()).config_hash()
        }

        let h0 =
            ProfileIdentity::new(populated_scan(), BASE_MAX_SETTLE, base_events()).config_hash();

        assert_ne!(
            h0,
            rehash_scan(|scan| scan.hidden = false),
            "hidden must discriminate"
        );
        assert_ne!(
            h0,
            rehash_scan(|scan| scan.exclude.push(glob("c"))),
            "exclude must discriminate"
        );
        assert_ne!(
            h0,
            rehash_scan(|scan| scan.pattern = None),
            "pattern must discriminate"
        );
        assert_ne!(
            h0,
            rehash_scan(|scan| scan.max_depth = NonZeroU32::new(8)),
            "max_depth must discriminate"
        );
        assert_ne!(
            h0,
            ProfileIdentity::new(populated_scan(), Duration::from_secs(8), base_events())
                .config_hash(),
            "max_settle must discriminate"
        );
        assert_ne!(
            h0,
            ProfileIdentity::new(populated_scan(), BASE_MAX_SETTLE, ClassSet::CONTENT)
                .config_hash(),
            "events must discriminate"
        );
    }

    /// The 1-byte scan-shape discriminant leads the fold: two shapes whose remaining byte streams
    /// could coincide still digest apart. Pairwise over every variant; a new shape extends this
    /// with its own pairs.
    #[test]
    fn hash_distinguishes_scan_shape() {
        let chain_h = compute_config_hash(&chain("/srv/*/log"), SETTLE, NO_EVENTS);
        assert_ne!(hash(SubtreeScan::builder().build()), chain_h);
    }

    /// Two positional sources digest apart — `source` is `MatchChain`'s complete encoding (parse
    /// purity: equal source ⇒ byte-equal decomposition), so distinct patterns must fork distinct
    /// Profiles.
    #[test]
    fn hash_distinguishes_match_chain_sources() {
        assert_ne!(
            compute_config_hash(&chain("/srv/*/log"), SETTLE, NO_EVENTS),
            compute_config_hash(&chain("/srv/*/data"), SETTLE, NO_EVENTS),
        );
    }

    /// A chain's identity is its pattern alone: STRUCTURE, the classes that witness a match set,
    /// under the discovery ceiling. Two parses of one pattern are one Profile.
    #[test]
    fn discovery_identity_is_the_pattern_under_its_own_mask_and_ceiling() {
        let identity = ProfileIdentity::discovery(pattern("/srv/*/log"));
        assert_eq!(identity.config(), &chain("/srv/*/log"));
        assert_eq!(identity.events(), ClassSet::STRUCTURE);
        assert_eq!(identity.max_settle(), Duration::from_secs(2));
        assert_eq!(
            identity.config_hash(),
            ProfileIdentity::discovery(pattern("/srv/*/log")).config_hash(),
        );
    }

    /// Predicate grid for the positional shape over `/srv/*/data/*/log` (terminus depth 4): a
    /// matching chain is admitted at every position, a non-matching segment rejects, scope ends at
    /// the terminus, the kind rule drops mid-chain non-dirs while the terminus admits any kind, and
    /// depth 0 bypasses both predicate halves.
    #[test]
    fn match_chain_accepts_positional_grid() {
        let cfg = chain("/srv/*/data/*/log");

        // The matching chain, Dir-kinded, at every depth.
        assert!(cfg.accepts(Path::new("app1"), ResourceKind::Dir, 1));
        assert!(cfg.accepts(Path::new("app1/data"), ResourceKind::Dir, 2));
        assert!(cfg.accepts(Path::new("app1/data/box1"), ResourceKind::Dir, 3));
        assert!(cfg.accepts(Path::new("app1/data/box1/log"), ResourceKind::Dir, 4));
        // The terminus admits any kind — a match terminates there regardless.
        assert!(cfg.accepts(Path::new("app1/data/box1/log"), ResourceKind::File, 4));

        // A non-matching segment rejects at its position.
        assert!(!cfg.accepts(Path::new("app1/etc"), ResourceKind::Dir, 2));
        assert!(!cfg.accepts(Path::new("app1/data/box1/current"), ResourceKind::File, 4));

        // Beyond the terminus is out of scope for every kind.
        assert!(!cfg.accepts(Path::new("app1/data/box1/log/below"), ResourceKind::Dir, 5));
        assert!(!cfg.accepts(Path::new("app1/data/box1/log/below"), ResourceKind::File, 5));

        // Mid-chain non-dir: an intermediate position names a directory the chain descends through;
        // a file there leads nowhere.
        assert!(!cfg.accepts(Path::new("app1"), ResourceKind::File, 1));
        assert!(!cfg.accepts(Path::new("app1/data"), ResourceKind::File, 2));

        // Depth 0 bypasses both halves — the anchor is in scope by construction, even File-kinded
        // (the kinded bypass; the structural bypass alone would still reject this).
        assert!(cfg.accepts(Path::new(""), ResourceKind::Dir, 0));
        assert!(cfg.accepts(Path::new(""), ResourceKind::File, 0));
    }

    /// The recursion edge is shape policy over the walker's device observation: `MatchChain` is
    /// bounded by the terminus and device-blind (chain levels may legitimately sit on distinct
    /// mounts; the walk cost is bounded by the chain length), while `Subtree` keeps the
    /// cross-filesystem refusal — the widened signature must not invert that gate's polarity.
    #[test]
    fn descends_into_is_terminus_bounded_and_device_blind_for_match_chain() {
        let cfg = chain("/srv/*/data/*/log");
        assert!(cfg.descends_into(0, false));
        assert!(cfg.descends_into(3, false));
        assert!(
            !cfg.descends_into(4, true),
            "the terminus depth never descends",
        );

        let subtree = ScanConfig::from(SubtreeScan::builder().build());
        assert!(subtree.descends_into(1, true));
        assert!(
            !subtree.descends_into(1, false),
            "Subtree keeps the cross-device refusal under the widened signature",
        );
    }

    /// Golden test — pins the canonical `compute_config_hash` encoding. Drift here changes every
    /// Profile's `config_hash` this binary computes; update only this constant after a deliberate
    /// review.
    #[test]
    fn hash_known_good() {
        let cfg = ScanConfig::from(SubtreeScan::builder().build());
        let h = compute_config_hash(&cfg, Duration::from_secs(1), ClassSet::EMPTY);
        assert_eq!(h, GOLDEN_HASH);
    }

    /// The public route (`ProfileIdentity::config_hash`) and the sealed kernel agree bit-for-bit on
    /// the golden preimage — sealing the kernel did not perturb the canonical encoding.
    #[test]
    fn profile_identity_config_hash_matches_golden() {
        let identity = ProfileIdentity::new(
            SubtreeScan::builder().build(),
            Duration::from_secs(1),
            ClassSet::EMPTY,
        );
        assert_eq!(identity.config_hash(), GOLDEN_HASH);
    }

    const GOLDEN_HASH: u64 = 0x49FE_0EB9_4F17_13E0;

    /// Fold-completeness ratchet for [`ScanConfig::accepts`]. Mirrors
    /// [`hash_discriminates_every_populated_field`]: a fully-populated [`SubtreeScan`] plus a
    /// neutral target; toggling each filter field must move at least one `accepts` verdict on a
    /// small test grid.
    ///
    /// The exhaustive destructure inside `accepts_structural` makes a new field a *compile error*
    /// until folded; this test makes the fold *discriminate*, so the ratchet cannot be satisfied by
    /// folding a new field as a constant.
    ///
    /// Not itself ratcheted: a new field still needs a new mutation here. The compile error from
    /// the destructure is the forcing function that drives the author to add it; this test then
    /// guards that the fold actually discriminates.
    #[test]
    fn accepts_reads_every_field() {
        fn populated() -> SubtreeScan {
            SubtreeScan::builder()
                .hidden(true)
                .exclude(glob("excluded"))
                .pattern(glob("*.rs"))
                .max_depth(NonZeroU32::new(7))
                .build()
        }

        fn verdicts(scan: SubtreeScan, probes: &[(&str, ResourceKind, u32)]) -> Vec<bool> {
            let cfg = ScanConfig::from(scan);
            probes
                .iter()
                .map(|(rel, kind, depth)| cfg.accepts(Path::new(rel), *kind, *depth))
                .collect()
        }

        // Grid of probes: (rel, kind, depth). Each probe is chosen so a *single* field toggle
        // shifts its verdict — the other fields are picked to "pass" so the predicate stops on the
        // field under test. Verdicts overlap by design (the predicate short-circuits), so a probe
        // that's already rejected by a *different* field cannot discriminate.
        let probes: &[(&str, ResourceKind, u32)] = &[
            (".hidden_dir", ResourceKind::Dir, 1),
            ("foo.rs", ResourceKind::File, 7),
            ("excluded", ResourceKind::Dir, 1),
            ("foo.txt", ResourceKind::File, 1),
        ];
        let base_v = verdicts(populated(), probes);
        let shifted = |mutate: fn(&mut SubtreeScan)| {
            let mut scan = populated();
            mutate(&mut scan);
            verdicts(scan, probes) != base_v
        };

        assert!(
            shifted(|scan| scan.hidden = false),
            "hidden must discriminate"
        );
        assert!(
            shifted(|scan| scan.exclude.clear()),
            "exclude must discriminate"
        );
        assert!(
            shifted(|scan| scan.pattern = None),
            "pattern must discriminate"
        );
        assert!(
            shifted(|scan| scan.max_depth = NonZeroU32::new(6)),
            "max_depth must discriminate"
        );

        // Anchor-depth (`depth == 0`) bypasses every filter unconditionally. Pinning the load-bearing
        // "anchor is in scope by construction" property at the predicate's first line, so a future
        // structural rewrite that drops the short-circuit surfaces as a localised failure here.
        assert!(
            ScanConfig::from(populated()).accepts(Path::new(""), ResourceKind::Dir, 0),
            "depth 0 must accept regardless of filters"
        );
    }
}
