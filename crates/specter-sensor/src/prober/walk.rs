//! `probe_anchor_file`, `probe_subtree`, `probe_descent` — pure-IO walkers.
//!
//! Each returns a [`ProbeOutcome`] typed to its query kind: `probe_anchor_file` →
//! `AnchorOk(LeafEntry)`; `probe_subtree` → `SubtreeProven { snapshot, authority }` where the
//! [`ProofAuthority`] certifies whether the response discharged its proof obligation; `probe_descent`
//! → `DirEnumerated(arc)` — a structural query is not a quiescence observation and the type carries
//! no certificate. Kind mismatches and absent paths collapse to `Vanished`; a root-anchor I/O error
//! is `Failed { errno }`. Mid-walk faults skip-and-continue and are accounted in the [`ProofLedger`]
//! (`exclude` is the user-facing surface for declaring expected-EACCES paths).
//!
//! Three controls live on [`specter_core::ProbeRequest::Subtree`]:
//! - `baseline_subtree`: the engine's last-known view. Equal `root_meta` against the freshly
//!   `lstat`-ed directory ⇒ return `Arc::clone(prior)` (mtime-skip), cascading into recursion via
//!   each child's `DirChild::Covered(arc)`, looked up by name through
//!   [`specter_core::DirSnapshot::lookup_covered_dir`].
//! - `obligation`: the subtrees that MUST be freshly observed for the response to certify
//!   quiescence. The walker refuses mtime-skip at any frame at-or-above a
//!   [`ProofObligation::Chains`] path (or anywhere, for [`ProofObligation::WholeSubtree`]);
//!   [`certify`] folds the [`ProofLedger`] against it into the response's [`ProofAuthority`].
//! - `forced`: defensive bypass for max-settle force-fire — every frame enumerates regardless of
//!   `baseline_subtree` or `obligation`.
//!
//! [`specter_core::ProbeRequest::AnchorFile`] runs a single `lstat` (no controls — a leaf has no
//! descendants to skip). [`specter_core::ProbeRequest::Descent`] walks under
//! [`WalkPolicy::DescentLevel`] (admit every dirent, one level, never descend) — the Profile's
//! user-facing filters would mask the very segment descent is searching for.
//!
//! Symlinks are never traversed (`symlink_metadata` ≡ `lstat`); they appear as `EntryKind::Symlink`
//! leaves when encountered as direct children. v1 has no `follow_symlinks` opt-in. Cross-filesystem
//! descent is refused: subdir entries with a `dev` differing from the root anchor's `dev` are
//! emitted as `DirChild::Uncovered(fs_id)` (uncovered-by-mount).
//!
//! Per-dirent scope filtering goes through [`WalkPolicy`]: a shape walk delegates to
//! [`ScanConfig::accepts_structural`] (pre-`lstat`) plus [`ScanConfig::accepts_kinded`]
//! (post-`lstat`, when `is_dir` is known). `covers` (engine) calls the full [`ScanConfig::accepts`]
//! — the composition of the same two halves; both consumers run the same predicate body **and**
//! measure `rel` from the same basis — the Profile anchor, shipped on `ProbeRequest::Subtree`'s
//! `anchor_path` and re-derived per dirent via `strip_prefix`. Matching the predicate body alone is
//! not enough: scope inputs must share an origin too, or an anchor-relative glob desyncs from an
//! LCA-relative `rel`. The shared basis keeps walker and engine in lockstep across the *structural*
//! scope axes (name + depth + frozen config); [`WalkContext::note_structural_filter_drop`] is the
//! runtime witness of that lockstep — an obligation-chain leaf the structural gate nonetheless
//! filters degrades the frame to `Undischarged` rather than silently dropping it. The kinded gate
//! ([`ScanConfig::accepts_kinded`]) is exempt: kind is time-varying (an atomic replace can flip a
//! chained Dir to a pattern-failing file between event capture and probe), so its drop is a
//! legitimate identity change the walker omits as observed-absent-from-scope.

use crate::ProbeFailureExt;
use compact_str::CompactString;
use specter_core::{
    ChildEntry, DirChild, DirMeta, DirSnapshot, FsIdentity, LeafEntry, ProbeFailure, ProbeOutcome,
    ProofAuthority, ProofObligation, ScanConfig,
};
use std::collections::{BTreeMap, BTreeSet};
use std::io;
use std::os::unix::fs::MetadataExt;
use std::path::Path;
use std::sync::Arc;

/// `captured_with` stamp for every `DirSnapshot` a [`probe_descent`] walk returns — an explicit
/// reserved marker, not any Profile's identity hash. A descent enumeration keys no Profile, and its
/// snapshot is consumed structurally (`entries.get(name)`) and dropped before any
/// [`specter_core::DirSnapshot::dir_hash`] fold could read the stamp, so the field's only job is to
/// be recognisably not-a-Profile in a debug dump. A live `config_hash` landing on this exact value
/// would be the same ~2⁻⁶⁴ accident the hash route itself tolerates — and nothing compares descent
/// stamps against Profile hashes anyway.
const DESCENT_CAPTURED_WITH: u64 = u64::MAX;

/// The scope policy one walk runs under — the walker-mode axis, distinct from the engine's
/// identity-bearing [`ScanConfig`] shapes.
///
/// [`Shape`](Self::Shape) honours a Profile's frozen scan shape: the three per-dirent decisions
/// delegate to the shape's named projections, and every frame stamps the engine-computed identity
/// hash shipped on the wire. [`DescentLevel`](Self::DescentLevel) is the pending-descent
/// enumeration mode: admit every dirent, one level, never descend. Descent searches for the next
/// path component of a not-yet-existing anchor, so the user-facing filters (which would mask the
/// very segment being searched for) must not apply — and a walker mode is not a Profile identity,
/// so it lives here rather than as a `ScanConfig` variant whose arms every identity projection and
/// the config hash would have to carry.
///
/// Pairing `captured_with` with the config inside `Shape` is load-bearing: a descent walk *cannot*
/// stamp a Profile's identity hash and a shape walk *cannot* stamp the descent marker — the
/// stamp/policy agreement is structural, not a call-site convention.
#[derive(Clone, Copy)]
enum WalkPolicy<'a> {
    Shape {
        config: &'a ScanConfig,
        /// `Profile.config_hash` at emission time, from `ProbeRequest::Subtree` — the walker cannot
        /// derive it (the hash folds identity axes the wire doesn't carry).
        captured_with: u64,
    },
    DescentLevel,
}

impl WalkPolicy<'_> {
    /// The kind-independent per-dirent gate — [`ScanConfig::accepts_structural`] for a shape walk.
    /// Descent admits exactly one level; the walk never descends, so deeper depths are never
    /// queried (the bound is totality, not a live filter).
    #[must_use]
    fn accepts_structural(&self, rel: &Path, depth: u32) -> bool {
        match self {
            Self::Shape { config, .. } => config.accepts_structural(rel, depth),
            Self::DescentLevel => depth <= 1,
        }
    }

    /// The kind-dependent per-dirent gate — [`ScanConfig::accepts_kinded`] for a shape walk.
    /// Descent admits every kind: the next path component may be anything; kind resolution is the
    /// engine's job at dispatch.
    #[must_use]
    fn accepts_kinded(&self, rel: &Path, is_dir: bool, depth: u32) -> bool {
        match self {
            Self::Shape { config, .. } => config.accepts_kinded(rel, is_dir, depth),
            Self::DescentLevel => true,
        }
    }

    /// The recursion edge — [`ScanConfig::descends_into`] (the single home of the shape
    /// recursion-edge policy) for a shape walk. Descent never descends: it is a one-level
    /// structural query, so every Dir dirent surfaces as `DirChild::Uncovered`.
    #[must_use]
    fn descends_into(&self, child_depth: u32, same_device: bool) -> bool {
        match self {
            Self::Shape { config, .. } => config.descends_into(child_depth, same_device),
            Self::DescentLevel => false,
        }
    }

    /// The `captured_with` stamp every frame of this walk writes onto its `DirSnapshot` — the
    /// engine-computed identity hash for a shape walk, [`DESCENT_CAPTURED_WITH`] for descent.
    #[must_use]
    const fn captured_with(&self) -> u64 {
        match self {
            Self::Shape { captured_with, .. } => *captured_with,
            Self::DescentLevel => DESCENT_CAPTURED_WITH,
        }
    }
}

/// Recursion-invariant inputs shared across every frame of one subtree probe. Built once at probe
/// entry ([`walk_root`]) from the `ProbeRequest::Subtree` payload, then threaded by reference into
/// [`snapshot_dir`], [`enumerate_dir`], and [`build_dir_child`]. Per-frame inputs (`path`,
/// `baseline`, `cmeta`, `name`) stay as positional arguments to those callees; per-dirent depth
/// derives from `rel` at the dirent (see [`enumerate_dir`]), not a threaded counter; the non-`Copy`
/// [`ProofLedger`] threads as a separate `&mut`.
///
/// Separating invariant from per-frame at the type level makes the distinction structural: a reader
/// at any call site sees `ctx` (unchanging across the recursion) plus the dirent-scope inputs that
/// vary. The methods name the walker's coverage decisions: [`should_recurse`](Self::should_recurse)
/// (the `Covered`/`Uncovered(fs_id)` gate at the dirent) and
/// [`try_mtime_skip`](Self::try_mtime_skip) (the no-op-when-unchanged primitive). The two
/// obligation-sensitive decisions — [`obligation_at_or_under`](Self::obligation_at_or_under) (the
/// mtime-skip refusal) and [`note_structural_filter_drop`](Self::note_structural_filter_drop) (the
/// structural on-chain filter-drop tripwire) — both project the obligation through the single
/// [`chain_through`](Self::chain_through) query, so the chain structure is read one way.
///
/// `anchor_path` is the **scope basis** — the Profile anchor the per-dirent `rel` is measured from
/// (`child_path.strip_prefix(anchor_path)`). It is distinct from both the per-frame `path` (where
/// the recursion currently sits) and the walk root (the `target_path` `read_dir` first descended
/// into): a Standard burst roots the walk at the dirty-LCA for speed but must still measure
/// `exclude` / `pattern` / depth from the anchor, or its scope desyncs from the engine's `covers`.
///
/// `root_dev` is the **walk root's** device — `target_path`'s, captured once in [`walk_root`] from
/// the top-level `lstat`. It is the recursion root's device, not the anchor's; the two coincide
/// only when the walk roots at the anchor (`target == anchor`), while a Standard burst can root at
/// a dirty-LCA below the anchor. The cross-filesystem gate
/// ([`should_recurse`](Self::should_recurse)) refuses to descend into any child whose device
/// differs from `root_dev`, leaving a sub-mount below the recursion root uncovered.
///
/// `Copy + Clone` because the struct is a handful of words — two thin references, the
/// [`WalkPolicy`] (a reference plus the stamp), one `u64`, one `bool`. Passing by reference at
/// recursion frequency is the convention here; the `Copy` derive is for the cheap "snapshot a `ctx`
/// value into a closure" cases that arise during evolution.
#[derive(Clone, Copy)]
struct WalkContext<'a> {
    anchor_path: &'a Path,
    policy: WalkPolicy<'a>,
    obligation: &'a ProofObligation,
    forced: bool,
    root_dev: u64,
}

/// Faithfulness of one directory level's own read — the value [`enumerate_dir`] returns and
/// [`snapshot_dir`] folds into the [`ProofLedger`]. `enumerate_dir` *reports*; `snapshot_dir`
/// *accumulates* (separation of concerns: the ledger never enters `enumerate_dir`'s own writes,
/// only threads through it to recursive frames).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Completeness {
    Complete,
    Incomplete,
}

/// Non-observation accounting for one subtree probe — the record of every way [`snapshot_dir`] can
/// yield a snapshot that is not a faithful complete read. [`certify`] folds it against the
/// obligation into a [`ProofAuthority`].
///
/// Written only by [`snapshot_dir`]'s degrade choke: a frame whose own read was unfaithful (the
/// [`Completeness::Incomplete`] arm of [`enumerate_dir`]). The mtime-skip predicate
/// ([`WalkContext::try_mtime_skip`]) refuses to skip at any frame on an obligation chain by
/// construction, so a sound off-chain skip is not a non-observation and is never recorded.
#[derive(Default)]
struct ProofLedger {
    degraded: BTreeSet<Arc<Path>>,
}

/// Fold the non-observation ledger against the obligation into the response's [`ProofAuthority`].
///
/// Unidirectional: `Undischarged` iff a degraded frame `f` lies at-or-**above** an obligation path
/// `p` (`p.starts_with(f)`) — a hole on a chain we had to prove. `f` at-or-below `p` sits off-chain
/// and must **not** flag. `WholeSubtree` (no trusted prior) treats any degraded frame anywhere as a
/// hole — the existential is correct there. `first_unread` is the obligation path whose proof we
/// could not discharge (`Chains`) or the offending frame itself (`WholeSubtree`, where no
/// obligation path exists to name).
fn certify(obligation: &ProofObligation, l: &ProofLedger) -> ProofAuthority {
    match obligation {
        ProofObligation::Chains(chains) => chains
            .iter()
            .find(|p| l.degraded.iter().any(|f| p.starts_with(f)))
            .map_or(ProofAuthority::Authoritative, |p| {
                ProofAuthority::Undischarged {
                    first_unread: Arc::clone(p),
                }
            }),
        ProofObligation::WholeSubtree => {
            l.degraded
                .iter()
                .next()
                .map_or(ProofAuthority::Authoritative, |f| {
                    ProofAuthority::Undischarged {
                        first_unread: Arc::clone(f),
                    }
                })
        }
    }
}

impl WalkContext<'_> {
    /// True iff a child directory at `depth_after_descent` on `child_dev` is in-scope for recursive
    /// descent — a pure delegation to [`WalkPolicy::descends_into`] (which, for a shape walk, is
    /// [`ScanConfig::descends_into`], the single home of the shape recursion-edge policy). The walker
    /// contributes the one observation core cannot make (`child_dev == self.root_dev`; the engine's
    /// `Tree` slots don't carry `device`); whether the observation *matters* is the policy's decision
    /// (`Subtree` is device-gated, `MatchChain` is device-blind, descent never descends).
    ///
    /// Negation drives `DirChild::Uncovered(fs_id)` emission in [`build_dir_child`]. This is the
    /// only source of `Uncovered` emissions in the walker; transient I/O (raced unlink, EACCES,
    /// ENOTDIR mid-walk) surfaces as `Covered(empty_or_partial_arc)` instead, via
    /// [`enumerate_dir`]'s benign-empty contract.
    ///
    /// The recursion-edge decision is deliberately separate from `ScanConfig::accepts` (*whether to
    /// include the dirent in the snapshot*): the per-dirent predicate runs depth-bounded gates too,
    /// but this decision is *whether to descend into the subtree* and is consulted exactly once per
    /// directory dirent.
    #[must_use]
    fn should_recurse(&self, depth_after_descent: u32, child_dev: u64) -> bool {
        self.policy
            .descends_into(depth_after_descent, child_dev == self.root_dev)
    }

    /// Returns `Some(Arc::clone(baseline))` when the directory at `path` with freshly-`lstat`ed
    /// `root_meta` is observationally identical to the baseline subtree. Three predicates folded:
    /// - `!self.forced` (no defensive bypass), AND
    /// - `!self.obligation_at_or_under(path)` (this frame is not on a proof obligation; the
    ///   obligation set is scanned at most once per call, and not at all when `self.forced`
    ///   short-circuits), AND
    /// - `baseline.root_meta == *root_meta` (mtime + inode + device).
    ///
    /// On `Some`, the caller short-circuits one whole recursion frame: zero readdir, zero leaf
    /// `lstat`, zero allocation. Composes recursively through each child's `DirChild::Covered(arc)`
    /// — an equal-mtime tree elides the entire walk.
    #[must_use]
    fn try_mtime_skip(
        &self,
        path: &Path,
        root_meta: &DirMeta,
        baseline: Option<&Arc<DirSnapshot>>,
    ) -> Option<Arc<DirSnapshot>> {
        if self.forced || self.obligation_at_or_under(path) {
            return None;
        }
        let prior = baseline?;
        if prior.root_meta() != *root_meta {
            return None;
        }
        Some(Arc::clone(prior))
    }

    /// True iff an obligation chain runs at-or-through `path` — some chain path is at-or-below it
    /// (`NonEmptyChainSet::any_chain_starts_with`). `WholeSubtree` carries no enumerable chain, so
    /// its projection is empty (`false`).
    ///
    /// The single read of the obligation's chain structure, shared by the two obligation-sensitive
    /// decisions: [`obligation_at_or_under`](Self::obligation_at_or_under) (the mtime-skip refusal)
    /// and [`note_structural_filter_drop`](Self::note_structural_filter_drop) (the structural
    /// on-chain filter-drop tripwire).
    ///
    /// Component-wise `Path::starts_with`, not byte-lex: probing `/a` must match a chain `/a/b/c`
    /// (we descend toward the leaf, so `/a` may not be skipped) but not a sibling `/ab`. A
    /// `BTreeSet::range` byte-prefix would conflate the two.
    #[must_use]
    fn chain_through(&self, path: &Path) -> bool {
        match self.obligation {
            ProofObligation::Chains(chains) => chains.any_chain_starts_with(path),
            ProofObligation::WholeSubtree => false,
        }
    }

    /// True iff [`try_mtime_skip`](Self::try_mtime_skip) must refuse to skip `path`: it lies
    /// at-or-above an obligation chain ([`chain_through`](Self::chain_through)), or the obligation
    /// is `WholeSubtree` (no trusted prior anywhere ⇒ every frame must be freshly read, even off
    /// any chain).
    #[must_use]
    fn obligation_at_or_under(&self, path: &Path) -> bool {
        self.chain_through(path) || matches!(self.obligation, ProofObligation::WholeSubtree)
    }

    /// Fold a **structural** filter drop into the running [`Completeness`], tripping when the drop
    /// is a scope regression.
    ///
    /// [`enumerate_dir`] calls this at the `accepts_structural` `continue` only. That gate reads
    /// nothing but the dirent's name and depth against the frozen config (exclude / hidden / depth
    /// bound / positional segment) — all time-invariant for a chained slot. The walker measures
    /// scope from the anchor on the wire, the same basis the engine's `covers` uses, so every
    /// obligation-chain leaf the engine tracks is structurally in scope for the walker too. A
    /// rename drops the old chained name from `read_dir` entirely (observed-absent, never reaching
    /// a filter arm), so a structural filter dropping a *present* on-chain dirent can only mean the
    /// scope basis desynced — a regression, never a legitimate exclusion.
    ///
    /// The kinded gate ([`ScanConfig::accepts_kinded`]) carries no such tripwire: it reads the
    /// dirent's current `is_dir`, which an atomic replace can flip between the chaining event and
    /// this probe. Its drop is a legitimate identity change, so [`enumerate_dir`] omits the entry
    /// inline rather than routing it here.
    ///
    /// On regression it is loud in dev (`debug_assert`) and degrades in release — returns
    /// [`Completeness::Incomplete`] so [`snapshot_dir`] records this frame (an ancestor of
    /// `child_path`) and [`certify`] flags `Undischarged` ⇒ refuse to fire. Otherwise returns
    /// `level` unchanged. A delete never reaches a filter arm, so a legitimately-removed chain leaf
    /// never trips this — absence and filtering are distinguishable here by construction.
    #[must_use]
    fn note_structural_filter_drop(&self, child_path: &Path, level: Completeness) -> Completeness {
        let on_chain = self.chain_through(child_path);
        debug_assert!(
            !on_chain,
            "scope regression: filter dropped on-chain dirent {} (obligation {:?})",
            child_path.display(),
            self.obligation
        );
        if on_chain {
            Completeness::Incomplete
        } else {
            level
        }
    }
}

/// Anchor-file probe. Single `lstat` against `target_path`.
///
/// Returns:
/// - `AnchorOk(LeafEntry)` for a regular file.
/// - `Vanished` when the path doesn't exist *or* is not a regular file (kind mismatch — symlink,
///   directory, FIFO, etc.).
/// - `Failed { errno }` for any other I/O error.
pub(super) fn probe_anchor_file(target_path: &Path) -> ProbeOutcome {
    let meta = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return ProbeOutcome::Vanished,
        Err(e) => return ProbeOutcome::Failed(ProbeFailure::from_io(&e)),
    };
    if !meta.is_file() {
        return ProbeOutcome::Vanished;
    }
    // The `is_file` guard above upholds `from_metadata`'s non-directory precondition;
    // `entry_kind_from_file_type` resolves it to `File`.
    let leaf = LeafEntry::from_metadata(&meta);
    ProbeOutcome::AnchorOk(leaf)
}

/// Shared root entry for both directory walks: root `lstat`, kind check, [`WalkContext`]
/// construction, then the recursive [`snapshot_dir`]. Returns the built subtree, or the
/// early-terminal [`ProbeOutcome`] (`Vanished` on absent/kind-mismatch, `Failed` on any other root
/// I/O error) — the caller wraps the `Ok` arm in its query-kind-specific outcome.
///
/// The walk *roots* at `target_path` (the `read_dir` start) but *scopes* from `anchor_path` (the
/// basis every dirent's `rel` is `strip_prefix`-ed against). They are the same path for Seed /
/// Rebase / Descent and for a Standard burst whose dirty-LCA is the anchor; a Standard burst rooted
/// at a deeper LCA passes the true anchor as the second argument so scope stays anchor-relative.
///
/// The non-`Copy` [`ProofLedger`] is the caller's: `probe_subtree` `certify`s it; `probe_descent`
/// discards it. Splitting the wrap from the walk is what makes `probe_descent` *not* a
/// `probe_subtree` delegation — a descent can no longer produce a `SubtreeProven`.
fn walk_root<'a>(
    target_path: &'a Path,
    anchor_path: &'a Path,
    policy: WalkPolicy<'a>,
    baseline: Option<&Arc<DirSnapshot>>,
    obligation: &'a ProofObligation,
    forced: bool,
    ledger: &mut ProofLedger,
) -> Result<Arc<DirSnapshot>, ProbeOutcome> {
    let root_meta_raw = match std::fs::symlink_metadata(target_path) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Err(ProbeOutcome::Vanished),
        Err(e) => return Err(ProbeOutcome::Failed(ProbeFailure::from_io(&e))),
    };
    if !root_meta_raw.is_dir() {
        return Err(ProbeOutcome::Vanished);
    }
    let root_meta = DirMeta::from_metadata(&root_meta_raw);
    let ctx = WalkContext {
        anchor_path,
        policy,
        obligation,
        forced,
        // Walk root's device (from `lstat(target_path)`), deliberately not the anchor's: cross-fs
        // refusal is scoped to sub-mounts below the recursion root, which is `target_path`.
        root_dev: root_meta.fs_id().device(),
    };
    Ok(snapshot_dir(&ctx, target_path, root_meta, baseline, ledger))
}

/// Subtree probe. Recursive DFS walk rooted at `target_path` honoring `recursive`, `hidden`,
/// `exclude`, `pattern`, and `max_depth` — each measured against `anchor_path`, the Profile anchor
/// every dirent's `rel` is `strip_prefix`-ed from (equal to `target_path` unless the walk roots at
/// a dirty-LCA below the anchor).
///
/// Each recursion frame may short-circuit via mtime-skip when `!forced`, the frame is not
/// at-or-above an `obligation` path, and a baseline subtree is provided whose `root_meta` (mtime +
/// inode + device) equals the freshly-`lstat`ed directory — returning `Arc::clone(baseline)` (zero
/// allocation/readdir/leaf-`lstat`), composing recursively through each child's
/// `DirChild::Covered(arc)`. Otherwise it enumerates one level, stamps a fresh `DirSnapshot`, and
/// recurses for covered Dir children.
///
/// Returns `SubtreeProven { snapshot, authority }` where `authority` is [`certify`]'s fold of the
/// [`ProofLedger`] against `obligation`: `Authoritative` iff no non-observation (mtime-skip of an
/// obligation frame, or a degraded enumeration level) lies on an obligation chain. Root errors
/// propagate as `Vanished` / `Failed`. Mid-walk `read_dir` / per-child faults skip-and-continue and
/// degrade the affected level (`DirChild::Covered(empty_or_partial_arc)`); the uncovered variant
/// `DirChild::Uncovered(fs_id)` stays reserved for the static gates in [`build_dir_child`]
/// (`!recursive`, beyond `max_depth`, cross-fs).
pub(super) fn probe_subtree(
    target_path: &Path,
    anchor_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    obligation: &ProofObligation,
    forced: bool,
) -> ProbeOutcome {
    let mut ledger = ProofLedger::default();
    match walk_root(
        target_path,
        anchor_path,
        WalkPolicy::Shape {
            config,
            captured_with,
        },
        baseline,
        obligation,
        forced,
        &mut ledger,
    ) {
        Ok(snapshot) => ProbeOutcome::SubtreeProven {
            snapshot,
            authority: certify(obligation, &ledger),
        },
        Err(outcome) => outcome,
    }
}

/// Descent prefix probe. Single-level enumeration of `target_path` under
/// [`WalkPolicy::DescentLevel`], which admits *every* dirent at one level and never descends —
/// descent is searching for the next path segment, so the engine's user-facing filters (which would
/// mask the very segment we're looking for) must not apply. Descent dispatch reads
/// `arc.entries.get(name)` directly and (for Profile descent) discards the snapshot.
///
/// Returns [`ProbeOutcome::DirEnumerated`] — a structural query is not a quiescence observation, so
/// it carries **no** [`ProofAuthority`]. It still threads the shared recursion core, so its
/// `ProofLedger` is written-then-discarded: a descent `read_dir` fault can populate `degraded`, but
/// the *type* (no `authority` field) is the guarantee, not an empty ledger. `WholeSubtree` is inert
/// here — the descent policy never descends and `baseline=None` makes mtime-skip unreachable, so it
/// never refuses a skip that could matter.
///
/// Every frame stamps [`DESCENT_CAPTURED_WITH`], the reserved not-a-Profile marker — paired with
/// the policy inside [`WalkPolicy::captured_with`], so this call site cannot mis-stamp.
pub(super) fn probe_descent(target_path: &Path) -> ProbeOutcome {
    let mut sink = ProofLedger::default();
    // Descent roots and scopes at the same path: target == anchor, so every dirent's `rel` is its
    // bare segment. The descent policy admits every dirent regardless, so the basis is inert here.
    match walk_root(
        target_path,
        target_path,
        WalkPolicy::DescentLevel,
        None,
        &ProofObligation::WholeSubtree,
        false,
        &mut sink,
    ) {
        Ok(snapshot) => ProbeOutcome::DirEnumerated(snapshot),
        Err(outcome) => outcome,
    }
}

/// Build one directory's snapshot frame. Shared by two callers:
/// 1. [`walk_root`], after the root `lstat` produces `root_meta` from the freshly-`lstat`ed anchor.
/// 2. [`build_dir_child`], with a `cmeta`-derived `root_meta` for a covered subdir dirent.
///
/// **Owns the [`ProofLedger`] degrade choke** (`enumerate_dir` reports, `snapshot_dir`
/// accumulates): an `Incomplete` level (this frame's own read was not faithful) writes
/// `ledger.degraded`. The mtime-skip arm is recorded nowhere — the obligation guard inside
/// [`WalkContext::try_mtime_skip`] makes a sound skip not a non-observation by construction.
///
/// Infallible by construction. Any failure inside the recursive [`enumerate_dir`] routes through
/// the `Covered(empty_or_partial_arc)` contract and (for non-benign faults) the degrade choke;
/// `DirChild::Uncovered(fs_id)` stays reserved for the recursion-edge refusals fronted by
/// [`WalkContext::should_recurse`] (`Subtree`'s `recursive=false` / `max_depth` / cross-fs gates,
/// `MatchChain`'s terminus depth) and is never minted for transient I/O.
#[must_use]
fn snapshot_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    root_meta: DirMeta,
    baseline: Option<&Arc<DirSnapshot>>,
    ledger: &mut ProofLedger,
) -> Arc<DirSnapshot> {
    if let Some(prior) = ctx.try_mtime_skip(path, &root_meta, baseline) {
        return prior;
    }
    let (entries, completeness) = enumerate_dir(ctx, path, baseline.map(Arc::as_ref), ledger);
    if completeness == Completeness::Incomplete {
        ledger.degraded.insert(Arc::from(path));
    }
    Arc::new(DirSnapshot::new(
        root_meta,
        ctx.policy.captured_with(),
        entries,
    ))
}

/// Read one directory level, applying filters and recursing into covered Dir children. Returns the
/// constructed entries map **and** this level's own [`Completeness`] (`enumerate_dir` reports; the
/// caller [`snapshot_dir`] folds it into the [`ProofLedger`] — the ledger is never written here,
/// only threaded through to recursive frames).
///
/// Errors at this level are skip-and-continue. The level is `Incomplete` iff its own read was
/// unfaithful: `read_dir` failed (non-NotFound), a dirent / non-UTF-8 / `strip_prefix` / per-child
/// `lstat` (non-NotFound) fault dropped an entry, or [`WalkContext::note_structural_filter_drop`]
/// caught the structural gate dropping an on-chain dirent (a should-never-happen scope regression,
/// loud in dev). Three drops stay `Complete` because they are *observed-absent*, not blindness, and
/// self-correct (short snapshot hash-differs → `Retry` → converge): `read_dir` `NotFound`
/// (raced-empty dir), a per-child `lstat` `NotFound` (a child unlinked between `read_dir` and the
/// `lstat` — a raced delete during `rm -rf` / `rsync --delete` / log-rotate), and a kinded-gate
/// drop (the dirent's current kind left scope, e.g. an atomic replace swapped a chained Dir for a
/// pattern-failing file). Degrading any of them would wedge a common production lifecycle into
/// permanent `Undischarged`/never-fire. The partially-populated `BTreeMap` becomes
/// `DirChild::Covered(empty_or_partial_arc)`; the uncovered variant stays the static-config gates'
/// (`build_dir_child`).
fn enumerate_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    baseline: Option<&DirSnapshot>,
    ledger: &mut ProofLedger,
) -> (BTreeMap<CompactString, ChildEntry>, Completeness) {
    let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    let mut completeness = Completeness::Complete;

    let read_dir = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        // Observed-absent, self-correcting: a raced-empty dir's empty snapshot hash-differs from a
        // non-empty prior ⇒ Retry ⇒ converge. Degrading it would be a liveness regression.
        Err(e) if e.kind() == io::ErrorKind::NotFound => {
            return (entries, Completeness::Complete);
        }
        Err(e) => {
            tracing::warn!(
                anchor = ?ctx.anchor_path,
                ?path,
                ?e,
                "probe_subtree readdir failed; degrading level"
            );
            return (entries, Completeness::Incomplete);
        }
    };

    for dirent_result in read_dir {
        let dirent = match dirent_result {
            Ok(d) => d,
            Err(e) => {
                tracing::warn!(
                    anchor = ?ctx.anchor_path,
                    ?path,
                    ?e,
                    "probe_subtree dirent error; degrading level"
                );
                completeness = Completeness::Incomplete;
                continue;
            }
        };
        let child_path = dirent.path();
        let name_os = dirent.file_name();
        let Some(name_str) = name_os.to_str() else {
            tracing::trace!(
                ?child_path,
                "probe_subtree non-UTF-8 filename; degrading level"
            );
            completeness = Completeness::Incomplete;
            continue;
        };
        let Ok(rel) = child_path.strip_prefix(ctx.anchor_path) else {
            tracing::trace!(
                ?child_path,
                "probe_subtree strip_prefix failed; degrading level"
            );
            completeness = Completeness::Incomplete;
            continue;
        };
        // Depth shares `rel` with the exclude/pattern gates — one source, so it cannot desync.
        // `try_from` saturates the PATH_MAX-bounded count (only a future iterative walker could
        // approach u32::MAX).
        let entry_depth = u32::try_from(rel.components().count()).unwrap_or(u32::MAX);
        // Pre-`lstat` scope gate — the kind-independent half of the shape's predicate (Subtree:
        // hidden / exclude; MatchChain: the positional segment match). Skipping here saves the
        // per-dirent `lstat` syscall on out-of-scope subtrees (a `target/` tree in a Cargo project
        // is thousands of dirents). Agreement invariant with the recursion edge: the structural
        // depth bound admits every depth `descends_into` reaches (`descends_into(d−1) ⇒ dirents
        // enumerate at depth d ⇒ the bound admits d`), so a drop here is always a genuine scope
        // filter, never a depth-bound desync — `note_structural_filter_drop` is the runtime
        // tripwire for exactly that regression.
        if !ctx.policy.accepts_structural(rel, entry_depth) {
            completeness = ctx.note_structural_filter_drop(&child_path, completeness);
            continue;
        }
        let cmeta = match std::fs::symlink_metadata(&child_path) {
            Ok(m) => m,
            // A child unlinked between `read_dir` and this `lstat` is observed-absent —
            // structurally identical to the `read_dir` NotFound arm. Benign, self-correcting; the
            // level stays `Complete`.
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                tracing::trace!(
                    ?child_path,
                    "probe_subtree child vanished before lstat; omitting"
                );
                continue;
            }
            // A non-NotFound fault (EACCES/EIO/ELOOP) is a true non-observation: we cannot tell
            // what is there.
            Err(e) => {
                tracing::warn!(
                    anchor = ?ctx.anchor_path,
                    ?child_path,
                    ?e,
                    "probe_subtree child symlink_metadata failed; degrading level"
                );
                completeness = Completeness::Incomplete;
                continue;
            }
        };
        let is_dir = cmeta.file_type().is_dir();

        // Kinded half of `accepts`, run post-`lstat` now that `is_dir` is known — the pre-`lstat`
        // gate above has already discharged the structural half, so re-running `accepts` here would
        // re-evaluate it for nothing. Same `entry_depth` both halves read — one derivation from
        // `rel`, so the positional gates cannot desync.
        //
        // A drop here carries no scope-regression tripwire: `accepts_kinded` reads the dirent's
        // *current* kind, which can flip between the event that chained this path and this probe
        // (an atomic replace — a same-named pattern-failing file swapped over a chained Dir, or a
        // mid-chain position turned non-dir). The engine chained the path against its prior kind;
        // the walker observes the new one. That divergence is a legitimate identity change, not a
        // basis desync, so the entry is observed-absent-from-scope: omit it and leave the level
        // `Complete` (the raced-unlink arm's semantics). The omission hash-differs from a baseline
        // that still holds the prior in-scope entity, firing the change through the ordinary diff
        // path instead of wedging the obligation in `Undischarged`.
        if !ctx.policy.accepts_kinded(rel, is_dir, entry_depth) {
            tracing::trace!(
                ?child_path,
                "probe_subtree dirent kind now out of scope (atomic replace); omitting"
            );
            continue;
        }

        let key = CompactString::new(name_str);
        let child_entry = if is_dir {
            build_dir_child(
                ctx,
                &child_path,
                baseline,
                entry_depth,
                &cmeta,
                name_str,
                ledger,
            )
        } else {
            build_leaf_child(&cmeta, name_str, baseline)
        };

        entries.insert(key, child_entry);
    }

    (entries, completeness)
}

/// Build a `ChildEntry::Dir` for one directory dirent. Recurses via [`snapshot_dir`] when the entry
/// is in-scope per [`WalkContext::should_recurse`] (recursive walk, within `max_depth`, same
/// filesystem); emits `DirChild::Uncovered(fs_id)` otherwise.
///
/// `Uncovered(fs_id)` is emitted iff [`WalkContext::should_recurse`] returns `false`. Every other
/// path enters [`snapshot_dir`], whose infallible return is wrapped unconditionally in
/// `DirChild::Covered(arc)`. Transient I/O failures inside the recursive walk surface as
/// `DirChild::Covered(empty_or_partial_arc)` via [`enumerate_dir`]'s benign-empty contract, never
/// as `Uncovered`.
fn build_dir_child(
    ctx: &WalkContext<'_>,
    child_path: &Path,
    baseline: Option<&DirSnapshot>,
    child_depth: u32,
    cmeta: &std::fs::Metadata,
    name: &str,
    ledger: &mut ProofLedger,
) -> ChildEntry {
    let fs_id = FsIdentity::from_metadata(cmeta);
    if !ctx.should_recurse(child_depth, cmeta.dev()) {
        // Uncovered branch: not recursive, beyond max_depth, or cross-fs. Walker stores the entry
        // but does not recurse — the dirent is still observed, so this is not a filter drop and
        // carries no `note_structural_filter_drop` tripwire: any out-of-scope descendant below an
        // Uncovered dir is on no obligation chain.
        return ChildEntry::Dir(DirChild::Uncovered(fs_id));
    }
    // Build the subdir's DirMeta from the caller-held `cmeta`: a second
    // `symlink_metadata(child_path)` would be redundant in the happy path and a race surface in the
    // unhappy one (concurrent unlink / kind-flip could make it disagree with the is_dir just
    // checked).
    let root_meta = DirMeta::from_metadata(cmeta);
    // Pull the child's prior subtree from baseline so mtime-skip composes recursively. BTreeMap key
    // match by string segment is the snapshot's native lookup; `lookup_covered_dir` collapses the
    // "Dir entry + covered" gate into one named operation.
    let child_baseline = baseline.and_then(|b| b.lookup_covered_dir(name));
    let arc = snapshot_dir(ctx, child_path, root_meta, child_baseline, ledger);
    ChildEntry::Dir(DirChild::Covered(arc))
}

/// Build a `ChildEntry::Leaf` for one non-directory dirent. Inherits the baseline leaf's
/// `leaf_hash` when the prior entry's identity matches — re-enumeration of an unchanged leaf elides
/// the SipHash24 fold the walker would otherwise pay. Identity mismatch recomputes the hash from
/// the freshly-`lstat`ed fields. Kind, size, mtime, and `fs_id` all derive from the one `cmeta`, so
/// the leaf is atomic by construction.
///
/// The caller's `is_dir` dispatch in [`enumerate_dir`] upholds `LeafEntry::from_metadata`'s
/// non-directory precondition (dirents with `is_dir` route to [`build_dir_child`], never here).
fn build_leaf_child(
    cmeta: &std::fs::Metadata,
    name: &str,
    baseline: Option<&DirSnapshot>,
) -> ChildEntry {
    let baseline_leaf = baseline.and_then(|b| b.lookup_leaf(name));
    ChildEntry::Leaf(LeafEntry::from_metadata_or_inherit(cmeta, baseline_leaf))
}

#[cfg(test)]
mod tests {
    use super::{WalkContext, WalkPolicy};
    use specter_core::{PatternSpec, ProofObligation, ScanConfig};
    use std::path::Path;
    use std::sync::Arc;

    /// Real mounts are untestable in unit CI, so the cross-device pin is the predicate composition:
    /// `should_recurse` translates the walker's device observation into `descends_into`'s
    /// `same_device` and the shape applies its policy. A device-mismatched child must still descend
    /// under `MatchChain` (bounded chain walk; levels may cross mounts) while the same mismatch
    /// refuses under `Subtree` — and the same-device control proves the `child_dev == root_dev`
    /// translation isn't polarity-inverted. Lives inline because `WalkContext` is module-private by
    /// design.
    #[test]
    fn should_recurse_device_gate_is_shape_policy() {
        let obligation = ProofObligation::WholeSubtree;
        let chain_cfg = ScanConfig::MatchChain(Arc::new(
            PatternSpec::parse("/srv/*/data/*/log").expect("test pattern parses"),
        ));
        let chain = WalkContext {
            anchor_path: Path::new("/srv"),
            policy: WalkPolicy::Shape {
                config: &chain_cfg,
                captured_with: 0,
            },
            obligation: &obligation,
            forced: false,
            root_dev: 1,
        };
        assert!(
            chain.should_recurse(1, 2),
            "MatchChain is device-blind above the terminus",
        );
        assert!(
            !chain.should_recurse(4, 2),
            "the terminus depth never descends",
        );

        let subtree_cfg = ScanConfig::builder().recursive(true).build();
        let subtree = WalkContext {
            policy: WalkPolicy::Shape {
                config: &subtree_cfg,
                captured_with: 0,
            },
            ..chain
        };
        assert!(
            !subtree.should_recurse(1, 2),
            "an unbounded recursive walk refuses a cross-device child",
        );
        assert!(
            subtree.should_recurse(1, 1),
            "same-device control: the device translation is not inverted",
        );
    }
}
