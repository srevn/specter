//! `probe_anchor_file`, `probe_subtree`, `probe_descent` — pure-IO walkers.
//!
//! Each returns a [`ProbeOutcome`] typed to its query kind:
//! `probe_anchor_file` → `AnchorOk(LeafEntry)`; `probe_subtree` →
//! `SubtreeProven { snapshot, authority }` where the [`ProofAuthority`]
//! certifies whether the response discharged its proof obligation;
//! `probe_descent` → `DirEnumerated(arc)` — a structural query is not a
//! quiescence observation and the type carries no certificate. Kind
//! mismatches and absent paths collapse to `Vanished`; a root-anchor
//! I/O error is `Failed { errno }`. Mid-walk faults skip-and-continue
//! and are accounted in the [`ProofLedger`] (`exclude` is the
//! user-facing surface for declaring expected-EACCES paths).
//!
//! Three controls live on [`ProbeRequest::Subtree`]:
//! - `baseline_subtree`: the engine's last-known view. Equal
//!   `root_meta` against the freshly `lstat`-ed directory ⇒ return
//!   `Arc::clone(prior)` (mtime-skip), cascading into recursion via
//!   each child's `DirChild::Covered(arc)`, looked up by name through
//!   [`specter_core::DirSnapshot::lookup_covered_dir`].
//! - `obligation`: the subtrees that MUST be freshly observed for the
//!   response to certify quiescence. The walker refuses mtime-skip at
//!   any frame at-or-above a [`ProofObligation::Chains`] path (or
//!   anywhere, for [`ProofObligation::WholeSubtree`]); [`certify`]
//!   folds the [`ProofLedger`] against it into the response's
//!   [`ProofAuthority`].
//! - `forced`: defensive bypass for max-settle force-fire — every
//!   frame enumerates regardless of `baseline_subtree` or `obligation`.
//!
//! [`ProbeRequest::AnchorFile`] runs a single `lstat` (no controls — a
//! leaf has no descendants to skip). [`ProbeRequest::Descent`] hardcodes
//! a minimal override config (`recursive=false`, `hidden=true`, no
//! exclude/pattern, no `max_depth`) — the Profile's user-facing filters
//! would mask the very segment descent is searching for.
//!
//! Symlinks are never traversed (`symlink_metadata` ≡ `lstat`); they
//! appear as `EntryKind::Symlink` leaves when encountered as direct
//! children. v1 has no `follow_symlinks` opt-in. Cross-filesystem descent
//! is refused: subdir entries with a `dev` differing from the root anchor's
//! `dev` are emitted as `DirChild::Uncovered(fs_id)` (uncovered-by-mount).
//!
//! Per-dirent scope filtering is delegated to
//! [`ScanConfig::accepts_structural`] (pre-`lstat`) plus the pattern arm
//! inline (post-`lstat`, when `is_dir` is known). `covers` (engine) calls
//! the full [`ScanConfig::accepts`]; both consume the same predicate body,
//! keeping walker and engine in lockstep across every scope axis.

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

/// Recursion-invariant inputs shared across every frame of one subtree
/// probe. Built once at probe entry ([`walk_root`]) from the
/// `ProbeRequest::Subtree` payload, then threaded by reference into
/// [`snapshot_dir`], [`enumerate_dir`], and [`build_dir_child`].
/// Per-frame inputs (`path`, `baseline`, `depth`, `cmeta`, `name`) stay
/// as positional arguments to those callees; the non-`Copy`
/// [`ProofLedger`] threads as a separate `&mut`.
///
/// Separating invariant from per-frame at the type level makes the
/// distinction structural: a reader at any call site sees `ctx`
/// (unchanging across the recursion) plus the dirent-scope inputs that
/// vary. The three methods name the walker's three coverage decisions:
/// [`should_recurse`](Self::should_recurse) (the
/// `Covered`/`Uncovered(fs_id)` gate at the dirent),
/// [`try_mtime_skip`](Self::try_mtime_skip) (the no-op-when-unchanged
/// primitive), and
/// [`obligation_at_or_under`](Self::obligation_at_or_under) (the proof
/// obligation that refuses skip).
///
/// `root_dev` is the *anchor*'s device, captured once in [`walk_root`]
/// from the top-level `lstat`. It is intentionally distinct from each
/// recursion frame's `root_meta.fs_id.device` — the cross-filesystem
/// gate refuses to descend whenever a child's device differs from the
/// anchor's, regardless of whether the recursion has already crossed a
/// sub-mount earlier.
///
/// `Copy + Clone` because the struct is three thin/fat pointers + two
/// `u64`s + one `bool` (`obligation` is a thin `&ProofObligation`).
/// Passing by reference at recursion frequency is the convention here;
/// the `Copy` derive is for the cheap "snapshot a `ctx` value into a
/// closure" cases that arise during evolution.
#[derive(Clone, Copy)]
struct WalkContext<'a> {
    anchor_path: &'a Path,
    config: &'a ScanConfig,
    obligation: &'a ProofObligation,
    forced: bool,
    captured_with: u64,
    root_dev: u64,
}

/// Faithfulness of one directory level's own read — the value
/// [`enumerate_dir`] returns and [`snapshot_dir`] folds into the
/// [`ProofLedger`]. `enumerate_dir` *reports*; `snapshot_dir`
/// *accumulates* (separation of concerns: the ledger never enters
/// `enumerate_dir`'s own writes, only threads through it to recursive
/// frames).
#[derive(Debug, Clone, Copy, Eq, PartialEq)]
enum Completeness {
    Complete,
    Incomplete,
}

/// Non-observation accounting for one subtree probe — the record of
/// every way [`snapshot_dir`] can yield a snapshot that is not a
/// faithful complete read. [`certify`] folds it against the obligation
/// into a [`ProofAuthority`].
///
/// Written only by [`snapshot_dir`]'s degrade choke: a frame whose own
/// read was unfaithful (the [`Completeness::Incomplete`] arm of
/// [`enumerate_dir`]). The mtime-skip predicate
/// ([`WalkContext::try_mtime_skip`]) refuses to skip at any frame on
/// an obligation chain by construction, so a sound off-chain skip is
/// not a non-observation and is never recorded.
#[derive(Default)]
struct ProofLedger {
    degraded: BTreeSet<Arc<Path>>,
}

/// Fold the non-observation ledger against the obligation into the
/// response's [`ProofAuthority`].
///
/// Unidirectional: `Undischarged` iff a degraded frame `f` lies
/// at-or-**above** an obligation path `p` (`p.starts_with(f)`) — a hole
/// on a chain we had to prove. `f` at-or-below `p` sits off-chain and
/// must **not** flag. `WholeSubtree` (no trusted prior) treats any
/// degraded frame anywhere as a hole — the existential is correct
/// there. `first_unread` is the obligation path whose proof we could
/// not discharge (`Chains`) or the offending frame itself
/// (`WholeSubtree`, where no obligation path exists to name).
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
    /// True iff a child directory at `depth_after_descent` on
    /// `child_dev` is in-scope for recursive descent. Folds three
    /// statically-knowable gates:
    /// - `self.config.recursive`
    /// - `depth_after_descent < max_depth.unwrap_or(u32::MAX)`
    /// - `child_dev == self.root_dev` (cross-filesystem refusal)
    ///
    /// Negation drives `DirChild::Uncovered(fs_id)` emission in
    /// [`build_dir_child`]. This is the only source of `Uncovered`
    /// emissions in the walker; transient I/O (raced unlink, EACCES,
    /// ENOTDIR mid-walk) surfaces as `Covered(empty_or_partial_arc)`
    /// instead, via [`enumerate_dir`]'s benign-empty contract.
    ///
    /// Cross-filesystem refusal is walker-specific (the engine's `Tree`
    /// slots don't carry `device`, so `ScanConfig::accepts` cannot fold
    /// it). The `recursive` and `max_depth` gates here exactly mirror
    /// what `ScanConfig::accepts` enforces per dirent — kept inline
    /// because this decision is *whether to descend into the subtree*
    /// (a recursion-edge concern) rather than *whether to include the
    /// dirent in the snapshot*; the two are deliberately separate.
    #[must_use]
    fn should_recurse(&self, depth_after_descent: u32, child_dev: u64) -> bool {
        self.config.recursive
            && depth_after_descent < self.config.max_depth.unwrap_or(u32::MAX)
            && child_dev == self.root_dev
    }

    /// Returns `Some(Arc::clone(baseline))` when the directory at
    /// `path` with freshly-`lstat`ed `root_meta` is observationally
    /// identical to the baseline subtree. Three predicates folded:
    /// - `!self.forced` (no defensive bypass), AND
    /// - `!self.obligation_at_or_under(path)` (this frame is not on a
    ///   proof obligation; the obligation set is scanned at most once
    ///   per call, and not at all when `self.forced` short-circuits),
    ///   AND
    /// - `baseline.root_meta == *root_meta` (mtime + inode + device).
    ///
    /// On `Some`, the caller short-circuits one whole recursion frame:
    /// zero readdir, zero leaf `lstat`, zero allocation. Composes
    /// recursively through each child's `DirChild::Covered(arc)` — an
    /// equal-mtime tree elides the entire walk.
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

    /// Returns `true` iff `path` lies at-or-above an obligation path
    /// (`Chains`) or unconditionally (`WholeSubtree`, no trusted
    /// prior). When true, [`try_mtime_skip`](Self::try_mtime_skip)
    /// refuses to skip `path`.
    ///
    /// Why `Path::starts_with` and not `==`: imagine `path = /a` and
    /// the obligation is `{/a/b/c}`. If we skip at `/a`, we never
    /// recurse into `/a/b/c` and miss the kernel's signal. Component-
    /// wise `starts_with` catches this — at `/a`,
    /// `(/a/b/c).starts_with(/a)` is true ⇒ refuse skip ⇒ enumerate
    /// children. At `/a/b`, the same path triggers the same refusal
    /// until we reach `/a/b/c`'s leaf, after which sibling subtrees
    /// are mtime-skip-eligible again.
    ///
    /// Byte-lex via `BTreeSet::range` would erroneously match `/ab`
    /// when probing `/a`; we need component-wise `Path::starts_with`.
    #[must_use]
    fn obligation_at_or_under(&self, path: &Path) -> bool {
        match self.obligation {
            ProofObligation::Chains(chains) => chains.any_chain_starts_with(path),
            ProofObligation::WholeSubtree => true,
        }
    }
}

/// Anchor-file probe. Single `lstat` against `target_path`.
///
/// Returns:
/// - `AnchorOk(LeafEntry)` for a regular file.
/// - `Vanished` when the path doesn't exist *or* is not a regular file
///   (kind mismatch — symlink, directory, FIFO, etc.).
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
    // The `is_file` guard above upholds `from_metadata`'s non-directory
    // precondition; `entry_kind_from_file_type` resolves it to `File`.
    let leaf = LeafEntry::from_metadata(&meta);
    ProbeOutcome::AnchorOk(leaf)
}

/// Shared root entry for both directory walks: root `lstat`, kind
/// check, [`WalkContext`] construction, then the recursive
/// [`snapshot_dir`]. Returns the built subtree, or the early-terminal
/// [`ProbeOutcome`] (`Vanished` on absent/kind-mismatch, `Failed` on
/// any other root I/O error) — the caller wraps the `Ok` arm in its
/// query-kind-specific outcome.
///
/// The non-`Copy` [`ProofLedger`] is the caller's: `probe_subtree`
/// `certify`s it; `probe_descent` discards it. Splitting the wrap from
/// the walk is what makes `probe_descent` *not* a `probe_subtree`
/// delegation — a descent can no longer produce a `SubtreeProven`.
fn walk_root<'a>(
    target_path: &'a Path,
    config: &'a ScanConfig,
    captured_with: u64,
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
        anchor_path: target_path,
        config,
        obligation,
        forced,
        captured_with,
        root_dev: root_meta.fs_id().device(),
    };
    Ok(snapshot_dir(
        &ctx,
        target_path,
        root_meta,
        baseline,
        0,
        ledger,
    ))
}

/// Subtree probe. Recursive DFS walk against `target_path` honoring
/// `recursive`, `hidden`, `exclude`, `pattern`, and `max_depth`.
///
/// Each recursion frame may short-circuit via mtime-skip when
/// `!forced`, the frame is not at-or-above an `obligation` path, and a
/// baseline subtree is provided whose `root_meta` (mtime + inode +
/// device) equals the freshly-`lstat`ed directory — returning
/// `Arc::clone(baseline)` (zero allocation/readdir/leaf-`lstat`),
/// composing recursively through each child's `DirChild::Covered(arc)`.
/// Otherwise it enumerates one level, stamps a fresh `DirSnapshot`, and
/// recurses for covered Dir children.
///
/// Returns `SubtreeProven { snapshot, authority }` where `authority` is
/// [`certify`]'s fold of the [`ProofLedger`] against `obligation`:
/// `Authoritative` iff no non-observation (mtime-skip of an obligation
/// frame, or a degraded enumeration level) lies on an obligation chain.
/// Root errors propagate as `Vanished` / `Failed`. Mid-walk `read_dir`
/// / per-child faults skip-and-continue and degrade the affected level
/// (`DirChild::Covered(empty_or_partial_arc)`); the uncovered variant
/// `DirChild::Uncovered(fs_id)` stays reserved for the static gates in
/// [`build_dir_child`] (`!recursive`, beyond `max_depth`, cross-fs).
pub(super) fn probe_subtree(
    target_path: &Path,
    config: &ScanConfig,
    captured_with: u64,
    baseline: Option<&Arc<DirSnapshot>>,
    obligation: &ProofObligation,
    forced: bool,
) -> ProbeOutcome {
    let mut ledger = ProofLedger::default();
    match walk_root(
        target_path,
        config,
        captured_with,
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

/// Sentinel `captured_with` value stamped on every `DirSnapshot`
/// returned from a [`probe_descent`] walk.
///
/// Descent dispatch never reads the field: the snapshot is consumed
/// by `arc.entries.get(name)` and dropped before any
/// [`specter_core::DirSnapshot::dir_hash`] computation could fold it
/// in. Any value is therefore sound today. The constant exists so the
/// call site reads as the named contract rather than a bare `0`,
/// guarding a future caller that pulls a descent snapshot through a
/// `dir_hash` comparison: an accidental collision with a real
/// `Profile.config_hash` would be inferable from the name, not from
/// re-deriving the obligation chain.
const DESCENT_CAPTURED_WITH: u64 = 0;

/// Descent prefix probe. Single-level enumeration of `target_path` with
/// a hardcoded override config: `recursive=false`, `hidden=true`, no
/// `exclude`, no `pattern`, no `max_depth`. The override config is what
/// drives the unified [`ScanConfig::accepts_structural`] predicate to
/// admit *every* dirent — descent is searching for the next path
/// segment, so the engine's user-facing filters (which would mask the
/// very segment we're looking for) deliberately collapse to no-ops
/// here. Descent dispatch reads `arc.entries.get(name)` directly and
/// (for Profile descent) discards the snapshot.
///
/// Returns [`ProbeOutcome::DirEnumerated`] — a structural query is not
/// a quiescence observation, so it carries **no** [`ProofAuthority`].
/// It still threads the shared recursion core, so its `ProofLedger` is
/// written-then-discarded: a descent `read_dir` fault can populate
/// `degraded`, but the *type* (no `authority` field) is the guarantee,
/// not an empty ledger. `WholeSubtree` is inert here — `recursive=false`
/// stops at one level and `baseline=None` makes mtime-skip unreachable,
/// so it never refuses a skip that could matter.
///
/// `captured_with` is stamped as [`DESCENT_CAPTURED_WITH`] — descent
/// dispatch never reads the field (the snapshot is consumed by the
/// engine and dropped before any consumer compares hashes), so the
/// value is observationally irrelevant. Callers should not rely on a
/// particular sentinel.
pub(super) fn probe_descent(target_path: &Path) -> ProbeOutcome {
    let cfg = ScanConfig::builder()
        .recursive(false)
        .hidden(true)
        .max_depth(None)
        .build();
    let mut sink = ProofLedger::default();
    match walk_root(
        target_path,
        &cfg,
        DESCENT_CAPTURED_WITH,
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
/// 1. [`walk_root`], after the root `lstat` produces `root_meta` from
///    the freshly-`lstat`ed anchor.
/// 2. [`build_dir_child`], with a `cmeta`-derived `root_meta` for a
///    covered subdir dirent.
///
/// **Owns the [`ProofLedger`] degrade choke** (`enumerate_dir`
/// reports, `snapshot_dir` accumulates): an `Incomplete` level (this
/// frame's own read was not faithful) writes `ledger.degraded`. The
/// mtime-skip arm is recorded nowhere — the obligation guard inside
/// [`WalkContext::try_mtime_skip`] makes a sound skip not a
/// non-observation by construction.
///
/// Infallible by construction. Any failure inside the recursive
/// [`enumerate_dir`] routes through the `Covered(empty_or_partial_arc)`
/// contract and (for non-benign faults) the degrade choke;
/// `DirChild::Uncovered(fs_id)` stays reserved for the static-config
/// gates fronted by [`WalkContext::should_recurse`] (`recursive=false`,
/// `max_depth`, cross-fs) and is never minted for transient I/O.
#[must_use]
fn snapshot_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    root_meta: DirMeta,
    baseline: Option<&Arc<DirSnapshot>>,
    depth: u32,
    ledger: &mut ProofLedger,
) -> Arc<DirSnapshot> {
    if let Some(prior) = ctx.try_mtime_skip(path, &root_meta, baseline) {
        return prior;
    }
    let (entries, completeness) =
        enumerate_dir(ctx, path, baseline.map(Arc::as_ref), depth, ledger);
    if completeness == Completeness::Incomplete {
        ledger.degraded.insert(Arc::from(path));
    }
    Arc::new(DirSnapshot::new(root_meta, ctx.captured_with, entries))
}

/// Read one directory level, applying filters and recursing into covered
/// Dir children. Returns the constructed entries map **and** this
/// level's own [`Completeness`] (`enumerate_dir` reports; the caller
/// [`snapshot_dir`] folds it into the [`ProofLedger`] — the ledger is
/// never written here, only threaded through to recursive frames).
///
/// Errors at this level are skip-and-continue. The level is `Incomplete`
/// iff its own read was unfaithful: `read_dir` failed (non-NotFound),
/// or a dirent / non-UTF-8 / `strip_prefix` / per-child `lstat`
/// (non-NotFound) fault dropped an entry. Two faults stay `Complete`
/// because they are *observed-absent*, not blindness, and self-correct
/// (empty/short snapshot hash-differs → `Unstable` → converge):
/// `read_dir` `NotFound` (raced-empty dir) and a per-child `lstat`
/// `NotFound` (a child unlinked between `read_dir` and the `lstat` — a
/// raced delete during `rm -rf` / `rsync --delete` / log-rotate).
/// Degrading either would wedge that common scenario into permanent
/// `Undischarged`/never-fire. The partially-populated `BTreeMap`
/// becomes `DirChild::Covered(empty_or_partial_arc)`; the uncovered
/// variant stays the static-config gates' (`build_dir_child`).
fn enumerate_dir(
    ctx: &WalkContext<'_>,
    path: &Path,
    baseline: Option<&DirSnapshot>,
    depth: u32,
    ledger: &mut ProofLedger,
) -> (BTreeMap<CompactString, ChildEntry>, Completeness) {
    let mut entries: BTreeMap<CompactString, ChildEntry> = BTreeMap::new();
    let mut completeness = Completeness::Complete;

    let read_dir = match std::fs::read_dir(path) {
        Ok(rd) => rd,
        // Observed-absent, self-correcting: a raced-empty dir's empty
        // snapshot hash-differs from a non-empty prior ⇒ Unstable ⇒
        // converge. Degrading it would be a liveness regression.
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

    // Each dirent's depth is one below the directory's own — depth-0
    // here means we're enumerating the anchor itself (dirents land at
    // depth 1). Saturating add keeps the predicate well-typed at
    // pathological depths; `should_recurse` (the recursion edge that
    // produces the recursive calls into this function) already caps
    // descent at `max_depth`, so reaching `u32::MAX` here is purely
    // defensive.
    let entry_depth = depth.saturating_add(1);
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
        // Pre-`lstat` scope gate: hidden / exclude / recursive /
        // max_depth (the last two are no-ops at this site — the
        // walker only reaches dirents at depths `should_recurse` has
        // already cleared — but the same predicate runs for `covers`
        // where they bite). Skipping here saves the per-dirent `lstat`
        // syscall on excluded subtrees (a `target/` tree in a Cargo
        // project is thousands of dirents).
        if !ctx.config.accepts_structural(rel, entry_depth) {
            continue;
        }
        let cmeta = match std::fs::symlink_metadata(&child_path) {
            Ok(m) => m,
            // A child unlinked between `read_dir` and this `lstat` is
            // observed-absent — structurally identical to the
            // `read_dir` NotFound arm. Benign, self-correcting; the
            // level stays `Complete`.
            Err(e) if e.kind() == io::ErrorKind::NotFound => {
                tracing::trace!(
                    ?child_path,
                    "probe_subtree child vanished before lstat; omitting"
                );
                continue;
            }
            // A non-NotFound fault (EACCES/EIO/ELOOP) is a true
            // non-observation: we cannot tell what is there.
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

        // Pattern arm of `accepts`, run post-`lstat` now that `is_dir`
        // is known. Kept inline (rather than calling `accepts` again
        // and re-evaluating the structural gates) — the pre-`lstat`
        // gate above has already discharged the structural half.
        if !is_dir
            && let Some(pat) = &ctx.config.pattern
            && !pat.matches_path(rel)
        {
            continue;
        }

        let key = CompactString::new(name_str);
        let child_entry = if is_dir {
            build_dir_child(ctx, &child_path, baseline, depth, &cmeta, name_str, ledger)
        } else {
            build_leaf_child(&cmeta, name_str, baseline)
        };

        entries.insert(key, child_entry);
    }

    (entries, completeness)
}

/// Build a `ChildEntry::Dir` for one directory dirent. Recurses via
/// [`snapshot_dir`] when the entry is in-scope per
/// [`WalkContext::should_recurse`] (recursive walk, within `max_depth`,
/// same filesystem); emits `DirChild::Uncovered(fs_id)` otherwise.
///
/// `Uncovered(fs_id)` is emitted iff [`WalkContext::should_recurse`]
/// returns `false`. Every other path enters [`snapshot_dir`], whose
/// infallible return is wrapped unconditionally in
/// `DirChild::Covered(arc)`. Transient I/O failures inside the
/// recursive walk surface as `DirChild::Covered(empty_or_partial_arc)`
/// via [`enumerate_dir`]'s benign-empty contract, never as
/// `Uncovered`.
fn build_dir_child(
    ctx: &WalkContext<'_>,
    child_path: &Path,
    baseline: Option<&DirSnapshot>,
    depth: u32,
    cmeta: &std::fs::Metadata,
    name: &str,
    ledger: &mut ProofLedger,
) -> ChildEntry {
    let fs_id = FsIdentity::from_metadata(cmeta);
    // Saturating: a recursion-based walker can never reach `u32::MAX`
    // (the kernel's path-length limit caps depth far below that), but
    // a future iterative walker could; computing once also kills the
    // duplicate addition the two call sites would otherwise repeat.
    let next_depth = depth.saturating_add(1);
    if !ctx.should_recurse(next_depth, cmeta.dev()) {
        // Uncovered branch: not recursive, beyond max_depth, or cross-fs.
        // Walker stores the entry but does not recurse.
        return ChildEntry::Dir(DirChild::Uncovered(fs_id));
    }
    // Build the subdir's DirMeta from the caller-held `cmeta`: a second
    // `symlink_metadata(child_path)` would be redundant in the happy
    // path and a race surface in the unhappy one (concurrent unlink /
    // kind-flip could make it disagree with the is_dir just checked).
    let root_meta = DirMeta::from_metadata(cmeta);
    // Pull the child's prior subtree from baseline so mtime-skip composes
    // recursively. BTreeMap key match by string segment is the snapshot's
    // native lookup; `lookup_covered_dir` collapses the "Dir entry + covered"
    // gate into one named operation.
    let child_baseline = baseline.and_then(|b| b.lookup_covered_dir(name));
    let arc = snapshot_dir(
        ctx,
        child_path,
        root_meta,
        child_baseline,
        next_depth,
        ledger,
    );
    ChildEntry::Dir(DirChild::Covered(arc))
}

/// Build a `ChildEntry::Leaf` for one non-directory dirent. Inherits
/// the baseline leaf's `leaf_hash` when the prior entry's identity
/// matches — re-enumeration of an unchanged leaf elides the SipHash24
/// fold the walker would otherwise pay. Identity mismatch recomputes
/// the hash from the freshly-`lstat`ed fields. Kind, size, mtime, and
/// `fs_id` all derive from the one `cmeta`, so the leaf is atomic by
/// construction.
///
/// The caller's `is_dir` dispatch in [`enumerate_dir`] upholds
/// `LeafEntry::from_metadata`'s non-directory precondition (dirents
/// with `is_dir` route to [`build_dir_child`], never here).
fn build_leaf_child(
    cmeta: &std::fs::Metadata,
    name: &str,
    baseline: Option<&DirSnapshot>,
) -> ChildEntry {
    let baseline_leaf = baseline.and_then(|b| b.lookup_leaf(name));
    ChildEntry::Leaf(LeafEntry::from_metadata_or_inherit(cmeta, baseline_leaf))
}
