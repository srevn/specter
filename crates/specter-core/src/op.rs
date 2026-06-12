//! Watch and Probe ops, plus their request/response payloads.

use crate::effect::Effect;
use crate::ids::{ProbeCorrelation, ProfileId, ResourceId};
use crate::resource::ResourceKind;
use crate::scan_config::ScanConfig;
use crate::snapshot::tree::{DirSnapshot, LeafEntry};
use crate::sub::ClassSet;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// Non-empty carrier for the [`ProofObligation::Chains`] chain set.
///
/// The wrapper's value *is* the invariant: a `Chains(NonEmptyChainSet)` always carries at least one
/// chain. Empty input is rejected at construction ([`Self::new`] returns `None`), with the contract
/// that the caller MUST degrade to [`ProofObligation::WholeSubtree`] — passing an empty chain set
/// to the walker would silently certify [`ProofAuthority::Authoritative`] (the walker's `certify`
/// returns `Authoritative` when no chain matches a degraded frame, and an empty chain set matches
/// nothing), defeating the proof obligation.
///
/// No public mutators — the wrapper is constructed once at the engine's probe choke and consumed
/// read-only by the walker. The invariant composes through `Clone` because [`BTreeSet`]'s
/// non-emptiness is preserved by clone.
///
/// Member order follows `BTreeSet<Arc<Path>>` (lex on the path bytes) so the walker's first-hit
/// search ([`certify`](#)) is deterministic across replays.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NonEmptyChainSet(BTreeSet<Arc<Path>>);

impl NonEmptyChainSet {
    /// Wrap a [`BTreeSet`] of chain paths, returning `None` when the input is empty. The `None`
    /// case is the caller's contract: degrade to [`ProofObligation::WholeSubtree`].
    #[must_use]
    pub fn new(set: BTreeSet<Arc<Path>>) -> Option<Self> {
        (!set.is_empty()).then_some(Self(set))
    }

    /// Borrowing iterator over every chain path, in [`BTreeSet`] lex order. The wrapper exposes no
    /// other iteration shape — the walker's `certify` and tests' membership / cardinality
    /// assertions compose through this projection.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<Path>> + '_ {
        self.0.iter()
    }

    /// Number of chain paths. Always `>= 1` by construction; pair with [`Self::is_empty`] to
    /// satisfy clippy's `len_without_is_empty` lint with a structural proof rather than `#[allow]`.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Always `false` — the wrapper's load-bearing invariant. `const` so the proof is inspectable at
    /// compile time; pairs with [`Self::len`] to satisfy `len_without_is_empty` without `#[allow]`.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// True iff `frame` is at-or-above some chain path. Composes the walker's
    /// `obligation_at_or_under` predicate: a frame on the recursion path that is an ancestor of (or
    /// equal to) any chain path must not be mtime-skipped, lest the kernel's signal at the chain
    /// leaf be missed.
    #[must_use]
    pub fn any_chain_starts_with(&self, frame: &Path) -> bool {
        self.0.iter().any(|p| p.starts_with(frame))
    }

    /// True iff `path` is byte-equal to some chain path. O(log n) via `BTreeSet::contains` (lookup
    /// keyed by `Path` through `Arc<Path>: Borrow<Path>`).
    #[must_use]
    pub fn contains(&self, path: &Path) -> bool {
        self.0.contains(path)
    }
}

/// What a [`ProbeRequest::Subtree`] walk must freshly observe for its response to certify quiescence.
///
/// The quiescence verdict is sound only if every entry folding into the response hash was *observed
/// at this probe* — an mtime-skipped or degraded frame is a non-observation. This enum is the
/// engine's statement of which subtrees may not be skipped:
///
/// - [`Self::Chains`] — Standard. The dirty root→leaf chains (resources whose `FsEvent` drove the
///   burst, projected to paths). The walker refuses mtime-skip at any directory at-or-above a chain
///   path; off-chain siblings stay skip-eligible. The [`NonEmptyChainSet`] carrier makes the empty
///   case unrepresentable: an empty chain set would silently certify
///   [`ProofAuthority::Authoritative`] (no chain to match a degraded frame), defeating the proof
///   obligation, so the engine degrades to [`Self::WholeSubtree`] when the source projection yields
///   nothing.
/// - [`Self::WholeSubtree`] — Seed / Rebase. No trustworthy prior exists, so nothing under the
///   anchor may be skipped: the whole subtree is unproven until freshly read. Seed has never
///   observed the tree; the post-fire rebase must prove the *post-command* tree quiescent, and the
///   command just mutated it (an in-place descendant edit need not bump an ancestor mtime, so a
///   chains-only skip would re-clone a stale subtree and certify a false quiet).
///
/// `Chains` entries are `Arc::clone`s of the engine's slot paths (shipped without re-allocating);
/// the underlying `BTreeSet` orders them deterministically for replay.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ProofObligation {
    Chains(NonEmptyChainSet),
    WholeSubtree,
}

/// Engine→walker probe request.
///
/// The variant is the contract: the walker dispatches on it; the engine reads it back at no point.
/// Each variant carries exactly the fields its walker arm consumes — no over-fetching.
///
/// Boxing the heavy `Subtree` variant was considered and rejected: every non-Descent burst produces
/// one, at most one Probe is in flight per burst, and `Arc<DirSnapshot>` baselines are already the
/// dominant payload. The enclosing `ProbeOp` lives in a `StepOutput::probe_ops` `BTreeMap` node —
/// heap-allocated on insert regardless of variant width — so the variant width never rides an
/// inline slot.
#[derive(Debug, Clone)]
pub enum ProbeRequest {
    /// File-anchor verify / Seed / Rebase. The walker runs a single `lstat` and returns
    /// `ProbeOutcome::AnchorOk(LeafEntry)` (or `Vanished` on absent / kind-mismatch / `Failed {
    /// errno }` on I/O). No baseline (a leaf has no descendants to skip), no `obligation` (a single
    /// `lstat` is definitionally authoritative — no subtree to discharge a proof over), no `forced`
    /// (mtime-skip is not a concept for `lstat`).
    AnchorFile {
        /// Profile the engine demuxes the response back to. Echoed back on `ProbeResponse` and used
        /// by the Sensor's expectation-map insertion.
        owner: ProfileId,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the anchor at probe-emission time. `Arc::clone` of the slot's
        /// materialised path (`tree.path_of(profile.resource)`) — a refcount bump, no rebuild.
        target_path: Arc<Path>,
    },
    /// Subtree verify / Seed / Rebase / Standard. Recursive Dir walk honouring `scan_config`.
    /// Walker returns `ProbeOutcome::SubtreeProven { snapshot, authority }` rooted at `target_path`
    /// (or `Vanished` / `Failed`); the `authority` certifies whether the response discharged the
    /// `obligation`.
    Subtree {
        /// Profile the engine demuxes the response back to. Echoed back on `ProbeResponse` and used
        /// by the Sensor's expectation-map insertion.
        owner: ProfileId,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the directory to walk — the **recursion root** and graft point. The
        /// deepest start that still covers every dirty path (the dirty-LCA for a Standard burst;
        /// the anchor for Seed / Rebase), at-or-under `anchor_path`. `Arc::clone` of
        /// `tree.path_of(target)` shipped on the wire — the walker has no `Tree` and never needs
        /// the engine's `ResourceId`. An empty path is the lone failure mode (the engine's stale-id
        /// sentinel); the walker treats empty as `Vanished`.
        target_path: Arc<Path>,
        /// Filesystem path of the Profile **anchor** — the scope basis, distinct from
        /// `target_path`. The walker measures every dirent's `rel` (hence its depth) as
        /// `child_path.strip_prefix(anchor_path)`, so `exclude` / `pattern` / `max_depth` /
        /// `recursive` resolve against the same origin the engine's `covers` uses. Measuring from
        /// `target_path` instead would silently desync the walker from `covers` whenever the
        /// recursion root sits below the anchor — an anchor-relative glob re-read against an
        /// LCA-relative `rel` drops an in-scope obligation leaf and certifies a region it never
        /// observed.
        ///
        /// Equals `target_path` exactly when the walk roots at the anchor — Seed, post-fire Rebase,
        /// and any Standard burst whose dirty-LCA resolves to the anchor; strictly above it when a
        /// Standard burst's dirty events share a subtree deeper than the anchor.
        ///
        /// **Invariant: `target_path` is at-or-under `anchor_path`.** Producer-guaranteed — the
        /// engine resolves the recursion root to a covered descendant of the anchor — so the
        /// walker's `strip_prefix(anchor_path)` is total over the subtree it reads. A violation is
        /// not unsound: the per-dirent `strip_prefix` fails, drops the dirent, and degrades the
        /// level ⇒ the proof refuses to fire.
        anchor_path: Arc<Path>,
        /// `ScanConfig` to honour (recursive, hidden, exclude, pattern, `max_depth`). The Profile's
        /// frozen config behind its sharing handle — a refcount bump at emit time; workers read the
        /// same allocation concurrently across probes.
        scan_config: Arc<ScanConfig>,
        /// `Profile.config_hash` at emission time. Walker stamps every `DirSnapshot.captured_with`
        /// so two Profiles sharing a Resource with different filters cannot produce identical
        /// `dir_hash` for divergent in-scope content.
        captured_with: u64,
        /// Engine's last-known view of `target_path`'s subtree. The walker consults
        /// `baseline_subtree.root_meta` for mtime-skip and propagates child baselines via each
        /// child's `DirChild::Covered(arc)`, resolved by name through
        /// [`crate::DirSnapshot::lookup_covered_dir`]. `None` means "no prior observation": first
        /// Seed of a fresh Profile.
        ///
        /// Cheap to ship — `Arc::clone` on the channel send. Multiple workers may hold the same Arc
        /// concurrently: `DirSnapshot` is fully immutable post-construction (hashes are eager
        /// fields, not a lazy cache).
        baseline_subtree: Option<Arc<DirSnapshot>>,
        /// The proof obligation: which subtrees this probe MUST freshly observe (refusing
        /// mtime-skip) for the response to certify quiescence. Populated from the burst's dirty
        /// resources ([`ProofObligation::Chains`], Standard) or set to
        /// [`ProofObligation::WholeSubtree`] when there is no trustworthy prior (Seed / Rebase).
        ///
        /// The walker refuses mtime-skip at any directory at-or-above a `Chains` path (O(N)
        /// prefix-match per directory, N typically 1–5) or anywhere within the subtree for
        /// `WholeSubtree`, and stamps the response's [`ProofAuthority`] `Undischarged` iff a
        /// skipped / degraded frame lies on an obligation chain.
        obligation: ProofObligation,
        /// `true` ⇒ walker bypasses mtime-skip at every directory regardless of `baseline_subtree`
        /// and `obligation`. Engine sets this when `PreFireBurst.forced` is true (max-settle
        /// deadline elapsed; force-fire).
        ///
        /// Defensive: mtime-skip is correct under normal semantics, but a forced probe wants the
        /// freshest possible snapshot regardless of cost.
        forced: bool,
    },
    /// Pending-descent prefix probe. Walker enumerates one level of `target_path` — every dirent
    /// admitted, no recursion — and returns `ProbeOutcome::DirEnumerated(arc)` containing the
    /// prefix's direct children — descent dispatch reads `arc.entries.get(name)` and discards the
    /// snapshot (it is never spliced into `Profile.current`). No `obligation` (a structural query is
    /// not a quiescence observation), and no `ScanConfig`: the Profile's user-facing filters would
    /// mask the very segment descent is searching for, so the admit-all policy lives walker-side.
    Descent {
        /// Profile the engine demuxes the response back to. Echoed back on `ProbeResponse` and used
        /// by the Sensor's expectation-map insertion.
        owner: ProfileId,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the descent prefix at probe-emission time. The engine routes
        /// responses by `(owner, correlation)` against the owner's state-resident `ProbeSlot` (the
        /// descent prefix lives on `DescentState`); the walker only needs the path. `Arc::clone` of
        /// the slot's materialised path — no rebuild.
        target_path: Arc<Path>,
    },
}

impl ProbeRequest {
    /// Profile the engine demuxes the response back to. Determinism-sort key for
    /// [`crate::StepOutput::probe_ops`] (via [`ProbeOp::owner`]).
    #[must_use]
    pub const fn owner(&self) -> ProfileId {
        match self {
            Self::AnchorFile { owner, .. }
            | Self::Subtree { owner, .. }
            | Self::Descent { owner, .. } => *owner,
        }
    }

    /// Correlation token. Used by the bin's expectation-map insertion in the sensor's
    /// `Prober::submit` (via `WorkerProber`) and by the worker's post-run cleanup. Never read by
    /// the engine after emit.
    #[must_use]
    pub const fn correlation(&self) -> ProbeCorrelation {
        match self {
            Self::AnchorFile { correlation, .. }
            | Self::Subtree { correlation, .. }
            | Self::Descent { correlation, .. } => *correlation,
        }
    }

    /// Filesystem path the walker probes. Every variant carries one; returns the borrowed path
    /// verbatim. The wire is path-keyed — this is the load-bearing identifier the walker dispatches
    /// on.
    #[must_use]
    pub fn target_path(&self) -> &Path {
        match self {
            Self::AnchorFile { target_path, .. }
            | Self::Subtree { target_path, .. }
            | Self::Descent { target_path, .. } => target_path,
        }
    }
}

/// Walker→engine probe response. Flat — `(owner, correlation)` is the staleness key the engine
/// gates against the owner's in-flight `ProbeSlot`; `outcome` carries the per-variant payload.
#[derive(Debug, Clone)]
pub struct ProbeResponse {
    pub owner: ProfileId,
    pub correlation: ProbeCorrelation,
    pub outcome: ProbeOutcome,
}

/// Walker-stamped certificate riding a [`ProbeOutcome::SubtreeProven`] (and engine-injected
/// `Authoritative` for `AnchorOk`).
///
/// `Authoritative` ⟺ every entry that folds into the response hash was freshly observed at this
/// probe — equivalently, no non-observation (mtime-skip clone or degraded enumeration) lies on an
/// obligation chain. `Undischarged` is the refuse-to-fire tripwire: the walker could not discharge
/// the proof obligation at `first_unread`, so the engine must not derive a quiescence verdict from
/// this response.
///
/// Rides **only** the inbound proof outcome — never stamped onto a stored `DirSnapshot` (`current`
/// / `baseline` stay pure content; stamping would corrupt `dir_hash` equality and conflate "what
/// the tree is" with "how well we saw it").
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ProofAuthority {
    Authoritative,
    Undischarged { first_unread: Arc<Path> },
}

/// Typed failure stamped on [`ProbeOutcome::Failed`].
///
/// The engine routes `Failed` uniformly today (log + teardown), but the variant names the routing
/// target so a future retry path can fork at the dispatch site without re-classifying errnos inside
/// the engine. Backends translate libc errnos to this variant once, at the trait boundary in
/// `specter-sensor`; the engine stays free of kernel vocabulary.
///
/// Cross-crate dual of [`WatchFailure`]. Naming follows the same rule — each variant carries "what
/// the engine should do," not the kernel's error-class name. `errno` is diagnostic context
/// (operator-visible integer on the IPC wire), not a behavioural switch.
///
/// The `From<io::Error>` translation lives in `specter-sensor` (via `ProbeFailureExt::from_io`)
/// because errno-name matching needs `libc`, which is banned in `core` per `deny.toml`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ProbeFailure {
    /// Path-fatal at the probe root: a non-`NotFound` I/O error from the root `lstat` (`EACCES` /
    /// `ELOOP` / `ENOTDIR` / `EIO`). Engine routes to its per-route `dispatch_*_failed` cleanup —
    /// release the anchor's `watch_demand`, surface the diagnostic, finish the burst.
    ///
    /// `ENOENT` is *not* an `Anchor` failure: the walker collapses "path absent" into
    /// [`ProbeOutcome::Vanished`] before this enum is reached.
    Anchor { errno: i32 },
    /// Backpressure or transient kernel-resource failure: the process-wide or system-wide FD
    /// ceiling was hit at the root-`lstat` syscall, or the walker retried into a transient
    /// rate-limit (`EMFILE` / `ENFILE` / `ENOSPC` / `EAGAIN`).
    ///
    /// v1 dispatches identically to [`Self::Anchor`]; the variant names the routing target for a
    /// future retry path. Calling it out at the trait boundary keeps the sensor's kernel-vocabulary
    /// classifier the single source — the engine never re-derives the retry signal from a raw `i32`.
    Transient { errno: i32 },
}

impl ProbeFailure {
    /// Underlying errno carried by every variant. Convenience for diagnostic logging and the IPC
    /// wire (which carries the integer, not the variant kind: the routing target is engine-internal
    /// today, not operator-actionable).
    ///
    /// Equivalent to a two-arm `match`; the `const` shape mirrors [`WatchFailure::errno`].
    #[must_use]
    pub const fn errno(&self) -> i32 {
        match self {
            Self::Anchor { errno } | Self::Transient { errno } => *errno,
        }
    }
}

/// Walker outcome.
///
/// Five variants. `Vanished` / `Failed` are intent-agnostic (the engine routes those by
/// `Profile.state` discriminator + pre-/post-fire phase, not by request shape — a vanished anchor
/// is a vanished anchor regardless of whether the walker was looking at a file or a directory). The
/// two directory outcomes are **type-distinct by query kind**: a `Subtree` proof carries its
/// [`ProofAuthority`] certificate; a `Descent` enumeration cannot even name one.
#[derive(Debug, Clone)]
pub enum ProbeOutcome {
    /// `AnchorFile` request returned a leaf observation. Sole producer is the walker's
    /// `probe_anchor_file`. A single `lstat` has no mtime-skip concept, so an anchor read is
    /// definitionally authoritative — the engine injects [`ProofAuthority::Authoritative`] at
    /// dispatch; the wire carries no certificate here.
    AnchorOk(LeafEntry),
    /// `Subtree` request returned a directory observation **plus** the walker-stamped
    /// [`ProofAuthority`] certifying whether every entry that folds into the response hash was
    /// freshly observed at this probe. Sole producer is the walker's `probe_subtree`. The
    /// `authority` rides only this inbound outcome — never stamped onto the stored snapshot.
    SubtreeProven {
        snapshot: Arc<DirSnapshot>,
        authority: ProofAuthority,
    },
    /// `Descent` request returned one prefix level. Sole producer is the walker's `probe_descent`.
    /// Descent dispatch reads `arc.entries.get(name)` and discards the snapshot (never spliced into
    /// `Profile.current`), so a descent enumeration carries **no** proof — the absence of an
    /// `authority` field is the type-level statement that structural queries are not quiescence
    /// observations.
    DirEnumerated(Arc<DirSnapshot>),
    /// Path absent (`ENOENT`) or kind mismatch (file probe found dir, dir probe found file). Routed
    /// to whichever `dispatch_*_vanished` corresponds to the Profile's state.
    Vanished,
    /// I/O error at the *root* of the probe (root `lstat`, permission, `EIO`). Mid-walk errors don't
    /// surface here — they skip-and-continue with `tracing::warn!`. The inner [`ProbeFailure`] is the
    /// sensor's classified routing target; the engine never inspects the raw `errno`.
    Failed(ProbeFailure),
}

#[derive(Debug, Clone)]
pub enum WatchOp {
    /// Install (or re-register) a watch on `resource` at `path`.
    ///
    /// `kind` is the engine's authoritative classification of the slot (`File` / `Dir` / `Unknown`).
    /// The sensor uses it as a verification step against the inode its `O_PATH` / `open` fd resolved
    /// to — rejecting installs where the path's on-disk kind diverges from the engine's expectation.
    /// `Unknown` is a wildcard: the engine emits it for slots it has not yet classified (descent
    /// prefix placeholder, post-`add_watch` before the first probe), and the sensor accepts whatever
    /// inode resolves while caching the observed kind for normalization / mask translation.
    ///
    /// `events` is the carrier for the per-Resource event-class union: the engine ships
    /// [`crate::Resource::events_union`] on every `Watch` op, the sensor diffs the cached per-FD
    /// mask, and re-registers iff different. `ClassSet::EMPTY` degrades to identity-floor-only
    /// delivery (kqueue: `NOTE_DELETE | NOTE_RENAME | NOTE_REVOKE`).
    Watch {
        resource: ResourceId,
        path: Arc<Path>,
        kind: ResourceKind,
        events: ClassSet,
    },
    Unwatch {
        resource: ResourceId,
    },
}

impl WatchOp {
    /// The Resource this op targets. Every variant carries one — the match is exhaustive and
    /// `const`. This is the determinism-sort key for [`crate::StepOutput::watch_ops`].
    #[must_use]
    pub const fn resource(&self) -> ResourceId {
        match self {
            Self::Watch { resource, .. } | Self::Unwatch { resource } => *resource,
        }
    }
}

/// Typed failure of a [`WatchOp::Watch`] install.
///
/// The engine demuxes on the variant — never on `errno` — so backends can map their kernel-specific
/// errno values once at the trait boundary without forcing the engine to learn each kernel's
/// vocabulary. Each variant names *what the engine should do*; the inner `errno` is diagnostic
/// context, not a behavioural switch.
///
/// The `From<io::Error>` translation lives in `specter-sensor` (via `WatchFailureExt::from_io`)
/// because errno-name matching needs `libc`, which is banned in `core` per `deny.toml`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum WatchFailure {
    /// Backpressure: kernel resource limit reached. Engine clamps `watch_demand := 0` on the
    /// affected resource and waits for natural retry on the parent's next reconcile.
    ///
    /// kqueue: `EMFILE` / `ENFILE` from `open(O_EVTONLY)`. inotify: `ENOSPC` from
    /// `inotify_add_watch` (`max_user_watches`).
    Pressure { errno: i32 },

    /// Path-fatal: the path doesn't resolve to an inode the engine expects. Engine treats as
    /// terminal for the resource and re-resolves via descent (anchor case ⇒ `finalize_anchor_lost`;
    /// descendant case ⇒ clamp + wait for parent).
    ///
    /// kqueue: `ENOENT` / `EACCES` / `ELOOP` / `ENOTDIR`. inotify: same set, plus `ENOTDIR` under
    /// `IN_ONLYDIR`.
    Resource { errno: i32 },

    /// Programmer error or trait-misuse: the watcher's invariant has been violated. Engine logs at
    /// error level and clamps the slot; in practice these never fire on a healthy bin. Examples:
    /// path with embedded NUL, `EBADF` against the inotify_fd, double-mapping of one wd to two
    /// `ResourceId`s (hardlink aliasing).
    Invariant { errno: i32 },
}

impl WatchFailure {
    /// Underlying errno carried by every variant. Convenience for diagnostic logging — equivalent
    /// to a three-arm `match`.
    #[must_use]
    pub const fn errno(&self) -> i32 {
        match self {
            Self::Pressure { errno } | Self::Resource { errno } | Self::Invariant { errno } => {
                *errno
            }
        }
    }
}

// `ProbeRequest::Subtree` carries baseline_subtree / obligation / forced etc., so `Probe` dwarfs
// `Cancel`. Boxing it would add an allocation per probe (every burst emits one `Probe`; `Cancel` is
// sparse) for no gain: a `ProbeOp` lives in a `StepOutput::probe_ops` BTreeMap node, heap-allocated
// on insert regardless of variant width — the size never rides an inline slot. Accept the delta
// explicitly.
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum ProbeOp {
    Probe { request: ProbeRequest },
    Cancel { owner: ProfileId },
}

impl ProbeOp {
    /// The Profile this op addresses. Both variants carry one (the `Probe` variant via its nested
    /// [`ProbeRequest`]). This is the determinism-sort key for [`crate::StepOutput::probe_ops`].
    #[must_use]
    pub const fn owner(&self) -> ProfileId {
        match self {
            Self::Probe { request } => request.owner(),
            Self::Cancel { owner } => *owner,
        }
    }
}

/// Engine→actuator wire vocabulary.
///
/// Structurally parallel to [`WatchOp`] and [`ProbeOp`]: submit and cancel ride as variants of one
/// enum, so a single FIFO channel preserves causal order between same-profile submit and cancel
/// without any extra synchronisation. The actuator's controller dispatches on the variant —
/// [`Self::Submit`] enters `handle_submit`, [`Self::Cancel`] enters `handle_cancel`.
///
/// `Cancel` is emitted by the engine at the `handle_gate_deadline` edge (the sole abandonment site
/// — the engine waits for natural completion otherwise). The actuator's `handle_cancel` walks every
/// slot whose [`crate::DedupKey::profile`] matches, drops queued `pending` / `plan_continue` (work
/// the engine has given up on), SIGTERMs the running child (if any), and lets the existing reap
/// pipeline deliver `EffectComplete` naturally. The engine routes that late completion to
/// `EffectCompleteOutsideAwaiting` (the Profile has already left `Awaiting` by then).
///
/// **Memory.** `Submit(Effect)` is the dominant variant width; the engine→actuator FIFO channel
/// slot size is dictated by it. `Cancel { profile }` pays only the discriminant on top of an 8-byte
/// `ProfileId`. Boxing `Submit` would add a heap allocation per emitted Effect for no gain — bursts
/// emit Effects in batches; `Cancel` is sparse (gate-deadline only).
#[allow(clippy::large_enum_variant)]
#[derive(Debug, Clone)]
pub enum EffectOp {
    /// Engine emits an Effect for the actuator to coalesce + spawn.
    Submit(Effect),
    /// Engine abandons every in-flight effect belonging to `profile`. Actuator SIGTERMs any running
    /// child for keys whose [`crate::DedupKey::profile`] matches and drops queued `pending` /
    /// `plan_continue` work for the same key set.
    Cancel { profile: ProfileId },
}

#[cfg(test)]
mod non_empty_chain_set_tests {
    use super::NonEmptyChainSet;
    use std::collections::BTreeSet;
    use std::path::Path;
    use std::sync::Arc;

    /// `new` rejects empty input — the wrapper's load-bearing invariant. The probe choke relies on
    /// this `None` to degrade to `WholeSubtree`; flipping `new`'s contract would silently emit a
    /// chain-less `Chains` obligation and the walker would certify Authoritative against a no-op
    /// proof. Behavioural integration tests cover the downstream effects; this is the focused
    /// constructor pin.
    #[test]
    fn new_rejects_empty_and_wraps_non_empty() {
        assert!(NonEmptyChainSet::new(BTreeSet::new()).is_none());

        let mut populated: BTreeSet<Arc<Path>> = BTreeSet::new();
        populated.insert(Arc::from(Path::new("/w/a")));
        let wrapped = NonEmptyChainSet::new(populated).expect("non-empty wraps");
        assert_eq!(wrapped.len(), 1);
    }
}
