//! Watch and Probe ops, plus their request/response payloads.

use crate::effect::Effect;
use crate::ids::{ProbeCorrelation, ProfileId, ResourceId};
use crate::resource::ResourceKind;
use crate::scan_config::ScanConfig;
use crate::snapshot::EntryKind;
use crate::snapshot::tree::{DirSnapshot, LeafEntry};
use crate::sub::ClassSet;
use compact_str::CompactString;
use std::collections::BTreeSet;
use std::path::Path;
use std::sync::Arc;

/// A non-empty set of paths, in lexicographic order.
///
/// It carries the chains a walk must read afresh ([`ProofObligation::Chains`]), and the directories
/// it had cause to read and could not ([`ProofAuthority::Undischarged`]). [`Self::new`] rejects an
/// empty set, so each caller says what none means.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct NonEmptyPathSet(BTreeSet<Arc<Path>>);

impl NonEmptyPathSet {
    /// Wrap a [`BTreeSet`] of paths, returning `None` when the input is empty.
    #[must_use]
    pub fn new(set: BTreeSet<Arc<Path>>) -> Option<Self> {
        (!set.is_empty()).then_some(Self(set))
    }

    /// Every path, in order.
    pub fn iter(&self) -> impl Iterator<Item = &Arc<Path>> + '_ {
        self.0.iter()
    }

    /// The first path in lex order. Total: the set is never empty.
    #[must_use]
    pub fn first(&self) -> &Arc<Path> {
        self.0
            .first()
            .expect("a NonEmptyPathSet holds at least one path by construction")
    }

    /// Number of paths. Always `>= 1` by construction; pair with [`Self::is_empty`] to satisfy
    /// clippy's `len_without_is_empty` lint with a structural proof rather than `#[allow]`.
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

    /// True iff some path is at-or-under `frame`, component-wise: `/a` holds `/a/b/c` and not `/ab`.
    #[must_use]
    pub fn any_at_or_under(&self, frame: &Path) -> bool {
        self.0.iter().any(|p| p.starts_with(frame))
    }
}

/// What a [`ProbeRequest::Subtree`] walk must read afresh. Any other directory whose `lstat` has
/// not moved since its earlier listing may stand on that listing unread.
///
/// - [`Self::Chains`] — the root→leaf chains of the paths the burst's events named: off them, the
///   event stream vouches for what an unmoved directory holds.
/// - [`Self::WholeSubtree`] — every directory, where nothing vouches for an unmoved one: a walk
///   with no prior to trust, one whose events cannot witness an in-place write (which need not move
///   any directory's `lstat`), or the rebase of a tree the command just wrote.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ProofObligation {
    Chains(NonEmptyPathSet),
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
    /// Subtree verify / Seed / Rebase / Standard. Recursive Dir walk honouring `scan_config`. Walker
    /// returns `ProbeOutcome::SubtreeProven { snapshot, authority }` rooted at `target_path` (or
    /// `Vanished` / `Failed`); the `authority` names what the walk had cause to read and could not.
    Subtree {
        /// Profile the engine demuxes the response back to. Echoed back on `ProbeResponse` and used
        /// by the Sensor's expectation-map insertion.
        owner: ProfileId,
        /// Engine-monotonic correlation token — pairs request with response.
        correlation: ProbeCorrelation,
        /// Filesystem path of the directory to walk — the **recursion root** and graft point. The
        /// deepest start that still covers every dirty path (for a Standard burst, the deepest
        /// directory the Profile's snapshot lists on the way to the dirty-LCA; for Seed and Rebase,
        /// the anchor), at-or-under `anchor_path`. `Arc::clone` of `tree.path_of(target)` shipped
        /// on the wire — the walker has no `Tree` and never needs the engine's `ResourceId`. An
        /// empty path is the lone failure mode (the engine's stale-id sentinel); the walker treats
        /// empty as `Vanished`.
        target_path: Arc<Path>,
        /// The Profile anchor: the basis every dirent's scope is measured from
        /// (`child_path.strip_prefix(anchor_path)`), as the engine's `covers` measures it. Measured
        /// from `target_path` instead, an anchor-relative glob would drop an in-scope leaf whenever
        /// the walk roots below the anchor. `target_path` is at-or-under it; a dirent outside it is
        /// a walker fault that stops the walk.
        anchor_path: Arc<Path>,
        /// The scan the walk honours: the Profile's frozen config behind its sharing handle — a
        /// refcount bump at emit time; workers read the same allocation concurrently across probes.
        scan_config: Arc<ScanConfig>,
        /// `Profile.config_hash` at emission time. Walker stamps every `DirSnapshot.captured_with`
        /// so two Profiles sharing a Resource with different filters cannot produce identical
        /// `dir_hash` for divergent in-scope content.
        captured_with: u64,
        /// The Profile's listing of `target_path`: the walk's prior, matched to each directory's
        /// earlier self by name and identity ([`crate::PriorDir`]). It stands in for a directory
        /// the walk owes no fresh read, and for one it could not read. `None` only for a Profile
        /// that holds no listing yet, whose walk roots at the anchor: the engine roots a walk below
        /// the anchor only at a directory its snapshot lists.
        baseline_subtree: Option<Arc<DirSnapshot>>,
        /// What the walk must read afresh.
        obligation: ProofObligation,
        /// `true` ⇒ walker bypasses mtime-skip at every directory regardless of `baseline_subtree`
        /// and `obligation`. Engine sets this when `PreFireBurst.forced` is true (max-settle
        /// deadline elapsed; force-fire).
        ///
        /// Defensive: mtime-skip is correct under normal semantics, but a forced probe wants the
        /// freshest possible snapshot regardless of cost.
        forced: bool,
    },
    /// Pending-descent structural query: does `segment` exist under the prefix at `target_path`,
    /// and as what kind? The walker answers with one `lstat(target_path/segment)` —
    /// [`ProbeOutcome::SegmentObserved`]. Descent advances one path component per response, so the
    /// engine's question is always about exactly one name; carrying that name on the wire scopes
    /// the walker's work to the question (the alternative — enumerating the whole prefix level for
    /// the engine to read one entry — pays one `lstat` per dirent on every re-probe, multiplied by
    /// sibling churn at `/tmp`-class prefixes). No `obligation` (a structural query is not a
    /// quiescence observation), and no `ScanConfig`: the Profile's user-facing filters never apply
    /// — there is no enumeration to filter, and the segment is queried by name regardless of what
    /// any pattern would say.
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
        /// The awaited head segment of the descent's remaining path at emission time. Frozen for
        /// the lifetime of the armed probe slot: every head mutation (forward advance, rewind
        /// prepend) re-arms the slot with a freshly minted correlation, so a response that passes
        /// the correlation gate is always an answer about the *current* head — the wire never needs
        /// to echo the segment back. The kernel resolves the name during the walker's `lstat` (case
        /// folding / Unicode normalization per filesystem), keeping descent consistent with watch
        /// installation and anchor probes, which resolve the same user-spelled path.
        segment: CompactString,
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

    /// The filesystem path this probe targets, regardless of variant — every variant carries
    /// exactly one. A **test-only projection** (gated behind `testkit`): production never reads a
    /// `ProbeRequest` back after emit. The walker dispatches on the request *variant* (`run_probe`
    /// destructures each arm and reads the relevant path field directly), not through this
    /// accessor; `MockProber` routing and probe-target assertions use it to recover the path
    /// without re-matching the variant.
    #[cfg(feature = "testkit")]
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

/// The walker's certificate on a [`ProbeOutcome::SubtreeProven`]: the directories it had cause to
/// read and could not.
///
/// It rides the response, never a stored [`DirSnapshot`]: how well a tree was seen is not what the
/// tree holds, and `dir_hash` equality would split on it.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum ProofAuthority {
    /// The walk read every directory it had cause to read.
    Authoritative,
    /// The walk had cause to read these directories and could not.
    Undischarged { denied: NonEmptyPathSet },
}

impl ProofAuthority {
    /// The certificate of a walk that could not read `denied`: [`Self::Authoritative`] iff there is
    /// none.
    #[must_use]
    pub fn certify(denied: BTreeSet<Arc<Path>>) -> Self {
        NonEmptyPathSet::new(denied)
            .map_or(Self::Authoritative, |denied| Self::Undischarged { denied })
    }
}

/// Why a probe could not answer, in what the read said: the path refused it, or the daemon lacked a
/// resource.
///
/// `errno` is the operator's context, never a switch. The sensor classifies the errno
/// (`ProbeFailureExt::from_io`): `core` may not name `libc`.
#[derive(Copy, Clone, Debug, Eq, PartialEq, Hash)]
pub enum ProbeFailure {
    /// The path refused the read (`EACCES`, `EIO`, …): a fact about the path. Absence is the
    /// caller's to draw first. A fault in the prober itself answers as one, with `EIO`.
    Denied { errno: i32 },
    /// The daemon lacked a resource (`EMFILE`, `ENFILE`, `ENOSPC`, `ENOMEM`, `EAGAIN`) at some read
    /// the probe made: it observed nothing, and says nothing of the path.
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
            Self::Denied { errno } | Self::Transient { errno } => *errno,
        }
    }
}

/// Walker outcome.
///
/// Five variants. `Vanished` / `Failed` are intent-agnostic (the engine routes those by
/// `Profile.state` discriminator + pre-/post-fire phase, not by request shape — a vanished anchor
/// is a vanished anchor regardless of whether the walker was looking at a file or a directory). The
/// success outcomes are **type-distinct by query kind**: a `Subtree` proof carries its
/// [`ProofAuthority`] certificate; a `Descent` query gets a structural answer that cannot even name
/// one.
#[derive(Debug, Clone)]
pub enum ProbeOutcome {
    /// `AnchorFile` request returned a leaf observation. Sole producer is the walker's
    /// `probe_anchor_file`. A single `lstat` has no mtime-skip concept, so an anchor read is
    /// definitionally authoritative — the engine injects [`ProofAuthority::Authoritative`] at
    /// dispatch; the wire carries no certificate here.
    AnchorOk(LeafEntry),
    /// `Subtree` request answered: the walk's listing of `target_path`, and its certificate.
    SubtreeProven {
        snapshot: Arc<DirSnapshot>,
        authority: ProofAuthority,
    },
    /// `Descent` request answered: a structural query gets a structural answer, not a snapshot. Sole
    /// producer is the walker's `probe_descent`. `Some(kind)` — `lstat(prefix/segment)` succeeded;
    /// the awaited segment exists with this kind. `None` — the prefix is a healthy directory but the
    /// segment is absent (descent parks on the standing absence observation). A vanished or
    /// non-directory prefix collapses to [`Self::Vanished`] (rewind) and a root I/O error to
    /// [`Self::Failed`], exactly as the other request shapes. Carrying no payload a verdict could
    /// fold is the type-level statement that structural queries are not quiescence observations.
    SegmentObserved { kind: Option<EntryKind> },
    /// Path absent (`ENOENT`) or kind mismatch (file probe found dir, dir probe found file). Routed
    /// to whichever `dispatch_*_vanished` corresponds to the Profile's state.
    Vanished,
    /// The probe could not answer. A walk stops at the first read that wants a resource, and fails
    /// at a root it can neither read whole nor stand on an earlier listing for; a refusal below the
    /// root answers in the snapshot ([`crate::DirChild::Denied`], or the directory's earlier
    /// listing) and the certificate instead.
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
    /// `inotify_add_watch` (`max_user_watches`). Both: `ENOMEM`, when the kernel cannot allocate.
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
mod non_empty_path_set_tests {
    use super::NonEmptyPathSet;
    use std::collections::BTreeSet;
    use std::path::Path;
    use std::sync::Arc;

    /// `new` rejects empty input — the wrapper's load-bearing invariant. The probe choke relies on
    /// this `None` to degrade to `WholeSubtree`; flipping `new`'s contract would silently emit a
    /// chain-less `Chains` obligation and the walker would certify Authoritative against a no-op
    /// proof. Behavioural integration tests cover the downstream effects; this is the focused
    /// constructor pin, and the pin of the total `first` the invariant buys.
    #[test]
    fn new_rejects_empty_and_wraps_non_empty() {
        assert!(NonEmptyPathSet::new(BTreeSet::new()).is_none());

        let populated: BTreeSet<Arc<Path>> = ["/w/b", "/w/a"]
            .into_iter()
            .map(|p| Arc::from(Path::new(p)))
            .collect();
        let wrapped = NonEmptyPathSet::new(populated).expect("non-empty wraps");
        assert_eq!(wrapped.len(), 2);
        assert_eq!(
            wrapped.first().as_ref(),
            Path::new("/w/a"),
            "first is the least path"
        );
    }
}
