//! Subscription and `EffectScope`.
//!
//! A `Sub` is a *reaction declaration*: it names what to watch and what should happen when the
//! watched tree settles. A request ([`SubAttachRequest`]) is addressed to one anchor or to every
//! match of a pattern ([`SubAttachTarget`]) and says a watch's knobs once. At rest the reaction is
//! a two-variant sum ([`Reaction`]): `Spawn` runs a program (a CFG-shaped op IR — [`ActionProgram`]
//! holds a `Box<[ProgramOp]>` walked by a `u32` cursor at the actuator), `Mint` mints a Sub per
//! matched chain terminus (discovery). The surface syntax (validation-side `Action` tree, lives in
//! `specter-config`) folds into the program at validation time; the engine and actuator see only
//! the lowered form.
//!
//! v1 surface is argv-only — no shell variant.

use crate::ids::{ProfileId, ResourceId, SubId};
use crate::pattern::PatternSpec;
use crate::program::ActionProgram;
use crate::scan_config::{DISCOVERY_SETTLE, ProfileIdentity, SubtreeScan};
use compact_str::CompactString;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Where a Sub anchors.
///
/// `Resource` names a slot the caller has already materialised; the engine trusts it after an O(1)
/// liveness check. `Path` names an absolute path the engine resolves at attach time — immediate
/// when every component already exists, otherwise pending descent until the anchor materialises and
/// a Seed burst establishes the baseline.
#[derive(Clone, Debug)]
pub enum SubAttachAnchor {
    Resource(ResourceId),
    Path(PathBuf),
}

/// What a `Spawn` reaction launches when its Profile settles: the program and its scope.
///
/// Construction-sealed: [`Self::new`] derives `needs_diff` once from the scope and the program's
/// placeholder set, and private fields make a desynced `needs_diff` unconstructable — there is no
/// way to swap `program` or `scope` without re-deriving. One derivation per *spec*: a request's is
/// sealed at config lowering, and a template stamps it onto every Sub it mints as a clone of the
/// sealed value — an Arc bump plus `Copy` fields.
#[derive(Clone, Debug)]
pub struct SpawnSpec {
    program: Arc<ActionProgram>,
    scope: EffectScope,
    needs_diff: bool,
}

impl SpawnSpec {
    #[must_use]
    pub fn new(program: Arc<ActionProgram>, scope: EffectScope) -> Self {
        let needs_diff = scope == EffectScope::PerStableFile || program.references_diff_derived();
        Self {
            program,
            scope,
            needs_diff,
        }
    }

    /// The lowered bytecode IR. Borrowed as the `Arc` so emit sites refcount-bump straight onto
    /// each [`crate::Effect`] — one allocation per validated Sub, never re-wrapped.
    #[must_use]
    pub const fn program(&self) -> &Arc<ActionProgram> {
        &self.program
    }

    #[must_use]
    pub const fn scope(&self) -> EffectScope {
        self.scope
    }

    /// Whether emitted Effects need the tree [`crate::Diff`]: true iff `scope` is `PerStableFile`
    /// or the program references a diff-derived placeholder. Derived once at [`Self::new`]; never
    /// re-evaluated.
    #[must_use]
    pub const fn needs_diff(&self) -> bool {
        self.needs_diff
    }
}

/// What a template stamps onto each Sub it mints: the identity of the Profile the Sub fires on, and
/// its params there.
///
/// Sealed by the engine, once per template attach, from the request's knobs ([`Self::new`]): the
/// identity is built from a subtree scan, so no minted Sub sits on a chain. It is the mint dedup's
/// key, and its sealed hash the Profile partition key, one source by construction; it is frozen for
/// the template's lifetime, and an edit of it replaces the template, where an edit of the params
/// rebinds them ([`SubRegistry::rebind`]).
#[derive(Debug)]
pub struct MintSpec {
    identity: ProfileIdentity,
    params: SubParams,
}

impl MintSpec {
    /// The spec a template stamps: its minted Profiles' identity, from `scan` under `max_settle`
    /// and `events`, and the params each Sub it mints runs.
    #[must_use]
    pub fn new(
        scan: SubtreeScan,
        max_settle: Duration,
        events: ClassSet,
        params: SubParams,
    ) -> Self {
        Self {
            identity: ProfileIdentity::new(scan, max_settle, events),
            params,
        }
    }

    /// The identity of the Profile each minted Sub fires on.
    #[must_use]
    pub const fn identity(&self) -> &ProfileIdentity {
        &self.identity
    }

    /// The params each minted Sub runs.
    #[must_use]
    pub const fn params(&self) -> &SubParams {
        &self.params
    }
}

/// Per-Sub Effect fire history — the four spawn-state fields in one home; `Default` is "never fired".
///
/// Lives on the `Spawn` reaction variant, not in [`SubParams`]: params are what an edit replaces
/// and a template stamps onto each Sub it mints, while each Sub keeps its own history through both
/// — state and spec must not share a struct. Mutated only through the registry edges
/// ([`SubRegistry::mark_fired`] / [`SubRegistry::clear_fired`] / [`SubRegistry::record_fired`] /
/// [`SubRegistry::record_dedup_suppressed`]; the registry holds the sole `&mut Sub`), and its
/// lifetime *is* the slotmap entry's — detach drops it with the Sub, so a detached or
/// hot-reload-replaced reaction can never re-fire on a later drift verdict and a revived Profile's
/// fresh Subs start unfired structurally (no purge step to forget).
#[derive(Copy, Clone, Debug, Default)]
pub struct FireHistory {
    /// `true` once this Sub has emitted at least one Effect. Sole load-bearing reader is the B1
    /// dedup suppress (`!forced && nothing_changed && has_fired` — a never-fired Sub is its own
    /// first emission); the per-Profile SeedDrift filter ([`SubRegistry::fired_in`]) and the
    /// recovery-drift short-circuit ([`SubRegistry::any_fired`]) read it Profile-wide. The
    /// invariant is the weakest tier in the engine — drift self-corrects on the next real change —
    /// so a plain `bool` carries it with no edge-method ceremony beyond the registry funnel.
    pub has_fired: bool,
    /// Engine instant of this Sub's last Effect emission; `None` until the first fire. Observational
    /// only — distinct from `has_fired` (the load-bearing B1-dedup signal): this carries the
    /// timestamp the operator-facing `list` UI renders (via the bin's `start_instant`/`start_wall`
    /// reference pair). Written only through [`SubRegistry::record_fired`].
    pub last_fired_at: Option<Instant>,
    /// Cumulative Effect emissions across this Sub's lifetime — `SubtreeRoot` increments by 1 per
    /// fire, `PerStableFile` by the per-file count of the emission pass. Observational; surfaces in
    /// the IPC `list` projection. Saturating-add on increment (`u64` holds a millennium of
    /// microsecond-cadence fires).
    pub fire_count: u64,
    /// Cumulative B1-dedup-suppressed verdicts — bumped when this `SubtreeRoot` Sub's
    /// `fire_decision` resolves to `FireVerdict::SuppressDedup` (unchanged tree, already fired, not
    /// forced). Observational; surfaces in `list --wide`. `PerStableFile` never suppresses (its
    /// dedup is diff-membership), so this counter stays zero on those Subs.
    pub dedup_suppressed_count: u64,
}

/// How a terminus entered a template's match set — what a mint's Seed owes, and what
/// [`crate::Diagnostic::DiscoveryMinted`] says of it.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MintEntry {
    /// Found by an enumeration — before the template holds a certified set, or in a branch the
    /// daemon could not read until now. It may predate the watch, so its Seed pins cold.
    Enumerated,
    /// Missing from a set the template held certified, in a branch the daemon had read: it appeared
    /// under watch, and its Seed owes the first fire.
    Appeared,
}

/// A line a template makes once in its lifetime ([`SubRegistry::narrate_once`]).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MintNarration {
    /// [`crate::Diagnostic::DiscoveryFanoutThreshold`]: its minted set grew past the warning
    /// threshold.
    FanoutThreshold,
    /// [`crate::Diagnostic::DiscoveryUnsupportedAnchorKind`]: a match whose kind can anchor no watch.
    UnsupportedAnchorKind,
}

impl MintNarration {
    /// This narration's bit in [`MintState`]'s record.
    const fn bit(self) -> u8 {
        1 << self as u8
    }
}

/// A template's state beside its [`MintSpec`]: how its next miss enters, and which lines it has made.
///
/// It lives as long as the template's slotmap entry and changes only through the registry's edges
/// ([`SubRegistry::close_enumeration`], [`SubRegistry::narrate_once`]).
#[derive(Debug)]
pub struct MintState {
    /// How a terminus the template has not minted enters its set, in a branch the daemon had read:
    /// `Enumerated` until the template holds a certified set — a completed pass's, or the empty one
    /// of a prefix seen absent — and `Appeared` ever after. It never reopens: the minted set it is
    /// read against outlives anchor loss, overflow and baseline clears, so a miss after a blind
    /// window did appear in it.
    miss: MintEntry,
    /// The [`MintNarration`]s made, one bit each.
    narrated: u8,
}

impl MintState {
    const fn new() -> Self {
        Self {
            miss: MintEntry::Enumerated,
            narrated: 0,
        }
    }

    /// How a terminus this template has not minted enters its set, where the daemon had read the
    /// branch it sits in.
    #[must_use]
    pub const fn miss(&self) -> MintEntry {
        self.miss
    }
}

/// What a firing Sub runs: its debounce and what it launches — the unit an edit of its params
/// replaces, and what a template stamps onto each Sub it mints.
///
/// Carries no Profile-identity field, so a Sub cannot express (or leak) a sibling Profile's
/// config/mask — demonstrated unrepresentable:
///
/// ```compile_fail
/// use specter_core::SubParams;
/// let _ = |p: SubParams| p.events;
/// ```
#[derive(Clone, Debug)]
pub struct SubParams {
    /// The Sub's debounce — the least on a Profile is its floor ([`SubRegistry::settle_floor`]).
    /// Distinct from `max_settle`, which is identity (folds into `config_hash`) and lives on
    /// [`ProfileIdentity`].
    pub settle: Duration,
    pub spawn: SpawnSpec,
}

/// What a request is addressed to — the one sum a `[[watch]]` is.
#[derive(Clone, Debug)]
pub enum SubAttachTarget {
    /// One anchor: the Sub fires there.
    At(SubAttachAnchor),
    /// Every match of a pattern: the Sub is a template, anchored at the pattern's literal prefix,
    /// that mints a Sub at each match.
    Pattern(Arc<PatternSpec>),
}

/// Public-API request to attach a Sub: its target, and a watch's knobs said once.
///
/// At an anchor the knobs describe the one Sub the request attaches; for a pattern, each Sub its
/// template mints, while the template's own Profile takes its identity from the pattern alone
/// ([`ProfileIdentity::discovery`]). The identity of the Profile a Sub fires on travels as its
/// preimage — a subtree scan, a ceiling and a mask — so no request seats a firing Sub on a chain,
/// and the engine seals a template's [`MintSpec`] from these same fields. A request cannot say
/// "minted": a minted Sub comes only from [`Sub::minted`], at the engine's mint entry.
///
/// Lives in `core::sub` (not `engine`) so [`SubRegistryDiff`] can carry pre-id requests via
/// [`crate::Input::ConfigDiff`] without a `core → engine` cycle.
#[derive(Clone, Debug)]
pub struct SubAttachRequest {
    pub name: CompactString,
    pub target: SubAttachTarget,
    pub scan: SubtreeScan,
    pub max_settle: Duration,
    pub events: ClassSet,
    pub params: SubParams,
}

/// Hot-reload diff. Computed by the TOML loader; consumed by `Engine::step(Input::ConfigDiff(_))`.
///
/// Name-keyed: `removed` carries operator watch names; `added`, `modified_identity`, and
/// `modified_params` carry pre-id [`SubAttachRequest`]s, each naming its Sub. The engine resolves
/// name → [`SubId`] at apply time through its own authoritative `by_name` index — identity
/// resolution is a registry-owner operation, not the loader's.
///
/// **The `modified` bucket is split.** Two semantically distinct transformations live behind a
/// "modified watch":
///
/// - **Identity change** (`modified_identity`): the target, the scan, `max_settle` or the events mask
///   differs from the prior spec, for either kind of target. Any of these moves a Sub to a different
///   Profile partition (the partition key is `(anchor_resource, ProfileIdentity::config_hash())`).
///   The engine asks the attach's gate first, then performs `detach_old → attach_new`; a refused
///   target leaves the old Sub in place — structural rollback at the composition layer.
/// - **Params change** (`modified_params`): the target and identity are unchanged; only the
///   [`SubParams`] differ. The engine rebinds the live Sub in place, a template with the set it
///   minted ([`SubRegistry::rebind`]): no Profile churn, no kernel-watch flap, no baseline loss. On
///   the rare case where the prior attach failed and the Sub never entered the registry, the engine
///   degrades the entry to a fresh attach ([`crate::Diagnostic::ConfigDiffRebindFallbackAttach`]
///   narrates the reason).
///
/// Engine processes `removed → modified_params → modified_identity → added` atomically in one step.
/// The four buckets are name-disjoint by diff construction.
#[derive(Clone, Debug, Default)]
pub struct SubRegistryDiff {
    /// Fresh attaches.
    pub added: Vec<SubAttachRequest>,
    /// Detaches by operator watch name.
    pub removed: Vec<CompactString>,
    /// Identity changes: each replaces the Sub of its name.
    pub modified_identity: Vec<SubAttachRequest>,
    /// Params changes: each rebinds the Sub of its name in place.
    pub modified_params: Vec<SubAttachRequest>,
}

impl SubRegistryDiff {
    /// True iff every bucket is empty — the "no Sub-side changes" short-circuit the reload pipeline
    /// tests before handing the diff to the engine. Single point of truth: every bucket future or
    /// present is named here.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.added.is_empty()
            && self.removed.is_empty()
            && self.modified_identity.is_empty()
            && self.modified_params.is_empty()
    }
}

#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub enum EffectScope {
    #[default]
    SubtreeRoot,
    PerStableFile,
}

/// User-facing event-class set on a [`Sub`] — the surface of the event filtering primitive.
///
/// A class set names *what kinds of change* a watch cares about, in backend-agnostic vocabulary.
/// The kqueue translator (sensor side) is the only place that maps the set onto `NOTE_*` fflags;
/// inotify gets a sibling translator. Engine and core never see backend bits.
///
/// Three classes:
/// - **STRUCTURE** — directory entries added / removed / renamed (Dir-only).
/// - **CONTENT**   — file bytes changed *or* file identity changed
///   (delete / rename / revoke). File-only.
/// - **METADATA**  — attribute change (perms, owner, link count,
///   timestamps). Both Files and Dirs.
///
/// The set is backed by a `u8` bitmask — `bits()` is the canonical representation folded into the
/// Profile config hash via [`ProfileIdentity::config_hash`] (two Subs differing only on classes
/// fork separate Profiles).
#[derive(Copy, Clone, Debug, Default, Eq, PartialEq, Ord, PartialOrd)]
pub struct ClassSet(u8);

impl ClassSet {
    pub const EMPTY: Self = Self(0);
    pub const STRUCTURE: Self = Self(1 << 0);
    pub const CONTENT: Self = Self(1 << 1);
    pub const METADATA: Self = Self(1 << 2);

    /// Default for [`EffectScope::SubtreeRoot`] — STRUCTURE | CONTENT. CONTENT implies per-file
    /// FDs, so in-place edits surface as events instead of going unseen until the next probe.
    pub const DEFAULT_SUBTREE_ROOT: Self = Self(0b011);

    /// Default for [`EffectScope::PerStableFile`] — CONTENT | METADATA. The user opted into
    /// per-file granularity; metadata is part of "this file's state changed".
    pub const DEFAULT_PER_FILE: Self = Self(0b110);

    /// Classes whose subscription suffices to witness in-place writes over a settle window — the
    /// semantic mask the verdict floor reads via [`crate::Profile::events_witness_quiescence`] to
    /// decide whether `EventsReliable` or the Layer-C hash channel folds the quiescence verdict.
    ///
    /// Today reduces to [`Self::CONTENT`]: CONTENT events fire for in-place writes — the only
    /// change kind that can span the settle window invisibly. STRUCTURE and METADATA are point
    /// events (atomic creates/renames; chmod/touch) and never bridge a gap. Adding STREAM /
    /// SPARSE_GROW / XATTR to the witness vocabulary is a one-line decision here.
    ///
    /// **Kernel-event-vocabulary assumption.** The criterion assumes the kernel surfaces every
    /// in-place write as a CONTENT-class event at the write boundary (`NOTE_WRITE` / `NOTE_EXTEND` on
    /// kqueue, `IN_MODIFY` / `IN_CLOSE_WRITE` on inotify). `mmap`-driven writes via dirty-page
    /// flushes, async-I/O completions, and `splice(2)` zero-copy paths may not satisfy this on every
    /// supported platform. Workloads with such writers should subscribe to a mask that does *not*
    /// cover [`Self::IN_PLACE_WRITES`] (e.g. `STRUCTURE` only), forcing the hash-channel safety net.
    ///
    /// **Even the hash channel proves lstat-identity quiescence, not byte quiescence**: a walk's
    /// `leaf_hash` folds `(kind, size, mtime, fs_id)`, so a same-size in-place write landing inside
    /// one mtime-granularity window is invisible to any walk — an inherent platform limit, not a
    /// mask choice.
    pub const IN_PLACE_WRITES: Self = Self::CONTENT;

    /// Classes whose subscription suffices to witness *membership* quiescence over a settle window
    /// — the criterion for scan shapes whose proof object is a match set (`ScanConfig::MatchChain`)
    /// rather than a subtree content hash. Membership changes (entries appearing / vanishing /
    /// renaming at chain positions) are all STRUCTURE point events; no in-place-write analog can
    /// span the window invisibly, so a STRUCTURE-covering mask folds the verdict via
    /// `EventsReliable` — N=1, no hash-channel ride.
    ///
    /// **HAZARD — an "equal hash ⇒ skip reconcile" shortcut has two obstructions.** The load-bearing
    /// one is registry-delta starvation: a discovery template can join its Profile with no tree event
    /// in sight (hot reload, mid-life join), and its mints are then owed on an *unchanged* tree — a
    /// hash gate would skip exactly the pass that owes them. The template's [`MintState::miss`] does
    /// not disarm it: an appearance says the template holds a certified set, not that a pass ran for
    /// it. A template that joins a descent whose prefix is then seen absent holds the empty set, and
    /// a prefix found again with the content its settled hash recorded would skip that template's
    /// first pass; a gate needs its own record of which templates a pass has reconciled. The second
    /// obstruction is hash noise: a matched *file* terminus still folds `{mtime, size, leaf_hash}`
    /// into the pruned snapshot hash, so an unwatched in-place append moves the hash without an event
    /// — under this witness the hash is never consulted for the verdict and reconciliation depends
    /// only on names and kinds, so the drift is cosmetic baseline noise, but it keeps "equal hash"
    /// needlessly strict until terminus leaves fold identity-only.
    pub const MEMBERSHIP_CHANGES: Self = Self::STRUCTURE;

    /// True iff every bit in `other` is set in `self` AND `other` is non-empty.
    ///
    /// The `other.0 != 0` clause sidesteps the `bitflags`-crate footgun where `contains(EMPTY) ==
    /// true` for every set: at every call site the question being asked is "do we hold this
    /// *specific*, non-empty class?", and reporting "yes" for `EMPTY` would silently green-light a
    /// no-class translator branch.
    #[must_use]
    pub const fn contains(self, other: Self) -> bool {
        (self.0 & other.0) == other.0 && other.0 != 0
    }

    /// True iff `self` and `other` share at least one bit. `EMPTY` intersects nothing (including
    /// itself).
    #[must_use]
    pub const fn intersects(self, other: Self) -> bool {
        (self.0 & other.0) != 0
    }

    /// Canonical bit representation — folded into `config_hash`.
    #[must_use]
    pub const fn bits(self) -> u8 {
        self.0
    }

    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.0 == 0
    }
}

impl std::ops::BitOr for ClassSet {
    type Output = Self;
    fn bitor(self, rhs: Self) -> Self {
        Self(self.0 | rhs.0)
    }
}

impl std::ops::BitOrAssign for ClassSet {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

impl std::ops::BitAnd for ClassSet {
    type Output = Self;
    fn bitand(self, rhs: Self) -> Self {
        Self(self.0 & rhs.0)
    }
}

impl std::ops::BitAndAssign for ClassSet {
    fn bitand_assign(&mut self, rhs: Self) {
        self.0 &= rhs.0;
    }
}

/// A live Sub's reaction: the params it runs or the spec it stamps, and the state it carries at
/// rest — a fire history on the spawn side, a [`MintState`] on the mint side.
///
/// Every spawn-field consumer dispatches on this sum, so the template case cannot be forgotten — a
/// `Mint` Sub has no `scope`, `program`, or fire history of its own (the minted Subs' reaction
/// lives on its spec; the mint never fires an Effect).
#[derive(Debug)]
pub enum Reaction {
    Spawn {
        params: SubParams,
        /// The template that minted this Sub — written by [`Sub::minted`] alone; `None` for a Sub a
        /// request declared. Living inside `Spawn`, a minted template is unrepresentable and the
        /// detach cascade is depth one: a minted Sub mints nothing.
        minted_by: Option<SubId>,
        history: FireHistory,
    },
    Mint {
        spec: MintSpec,
        state: MintState,
    },
}

impl Reaction {
    /// The params an edit of this Sub replaces: a firing Sub's own, or those a template stamps.
    const fn params_mut(&mut self) -> &mut SubParams {
        match self {
            Self::Spawn { params, .. } => params,
            Self::Mint { spec, .. } => &mut spec.params,
        }
    }
}

#[derive(Debug)]
pub struct Sub {
    pub name: CompactString,
    /// The Profile this Sub attaches to — the join axis of the per-Sub fire bookkeeping onto a
    /// `(Resource, ScanConfig)` partition.
    ///
    /// **Write-once** at construction: re-assigning this would orphan [`SubRegistry::by_profile`]
    /// (the secondary index by `ProfileId`), break the `Sub`-to-`Profile` lifetime presupposition
    /// every dispatcher reads, and silently re-target fire history. The invariant is held by
    /// encapsulation — module-private with no setter — matching the discipline on
    /// [`crate::Profile::resource`] (the same write-once join axis on the Profile side).
    profile: ProfileId,
    /// What this Sub does when its Profile settles. Module-private: external consumers read through
    /// [`Self::reaction`] and the typed projections ([`Self::settle`] / [`Self::minted_by`] /
    /// [`Self::is_template`] / [`Self::spawn_spec`] / [`Self::fire_history`] / [`Self::mint_spec`]
    /// / [`Self::mint_state`]), and every mutation flows through its registry edge (fire history,
    /// the template's state, rebind — the registry holds the sole `&mut Sub`).
    reaction: Reaction,
}

impl Sub {
    /// A Sub a request declared at one anchor: it runs `params`, and starts never fired. The
    /// slotmap key is the Sub's identity authority — there is no `id` field.
    #[must_use]
    pub fn declared(profile: ProfileId, name: CompactString, params: SubParams) -> Self {
        Self {
            name,
            profile,
            reaction: Reaction::Spawn {
                params,
                minted_by: None,
                history: FireHistory::default(),
            },
        }
    }

    /// A template a request declared for a pattern: it mints under `spec`, its enumeration open.
    #[must_use]
    pub const fn template(profile: ProfileId, name: CompactString, spec: MintSpec) -> Self {
        Self {
            name,
            profile,
            reaction: Reaction::Mint {
                spec,
                state: MintState::new(),
            },
        }
    }

    /// A Sub `template` minted: it runs the params the template stamps, and starts never fired. The
    /// one constructor that names a template, so a Sub's provenance is the engine's mint entry's to
    /// write, never a request's.
    #[must_use]
    pub fn minted(
        profile: ProfileId,
        name: CompactString,
        template: SubId,
        spec: &MintSpec,
    ) -> Self {
        Self {
            name,
            profile,
            reaction: Reaction::Spawn {
                params: spec.params.clone(),
                minted_by: Some(template),
                history: FireHistory::default(),
            },
        }
    }

    /// The Profile this Sub attaches to. Write-once at construction; see the field rustdoc for the
    /// load-bearing invariant.
    #[must_use]
    pub const fn profile(&self) -> ProfileId {
        self.profile
    }

    /// This Sub's reaction — the variant dispatch every spawn-field consumer routes through, so the
    /// `Mint` case cannot be silently misread as a firing Sub.
    #[must_use]
    pub const fn reaction(&self) -> &Reaction {
        &self.reaction
    }

    /// This Sub's debounce: its params', or — a template's reaction being its Profile's walk — the
    /// discovery debounce ([`DISCOVERY_SETTLE`]).
    #[must_use]
    pub const fn settle(&self) -> Duration {
        match &self.reaction {
            Reaction::Spawn { params, .. } => params.settle,
            Reaction::Mint { .. } => DISCOVERY_SETTLE,
        }
    }

    /// The discovery template that minted this Sub — `None` for operator-declared Subs, discovery
    /// templates included. Written once, at [`Self::minted`]: the registry files the Sub under it,
    /// in its template's minted set ([`SubRegistry::minted`]).
    #[must_use]
    pub const fn minted_by(&self) -> Option<SubId> {
        match &self.reaction {
            Reaction::Spawn { minted_by, .. } => *minted_by,
            Reaction::Mint { .. } => None,
        }
    }

    /// Whether this Sub is a discovery template — its reaction is minting Subs, never firing
    /// Effects. A template comes from a request's [`SubAttachTarget::Pattern`] alone, which seats
    /// it on the chain Profile its pattern names.
    #[must_use]
    pub const fn is_template(&self) -> bool {
        matches!(self.reaction, Reaction::Mint { .. })
    }

    /// The spawn payload when this Sub's reaction is `Spawn` — `None` for a `Mint` Sub, which has
    /// no program or scope of its own.
    #[must_use]
    pub const fn spawn_spec(&self) -> Option<&SpawnSpec> {
        match &self.reaction {
            Reaction::Spawn { params, .. } => Some(&params.spawn),
            Reaction::Mint { .. } => None,
        }
    }

    /// This Sub's fire history — `None` for a `Mint` Sub (discovery fires attachments, never
    /// Effects).
    #[must_use]
    pub const fn fire_history(&self) -> Option<&FireHistory> {
        match &self.reaction {
            Reaction::Spawn { history, .. } => Some(history),
            Reaction::Mint { .. } => None,
        }
    }

    /// What this template stamps onto each Sub it mints — `None` for a `Spawn` Sub.
    #[must_use]
    pub const fn mint_spec(&self) -> Option<&MintSpec> {
        match &self.reaction {
            Reaction::Mint { spec, .. } => Some(spec),
            Reaction::Spawn { .. } => None,
        }
    }

    /// This template's state — `None` for a `Spawn` Sub.
    #[must_use]
    pub const fn mint_state(&self) -> Option<&MintState> {
        match &self.reaction {
            Reaction::Mint { state, .. } => Some(state),
            Reaction::Spawn { .. } => None,
        }
    }

    /// Whether this Sub has emitted at least one Effect — structurally `false` for a `Mint` Sub.
    /// The total projection of [`FireHistory::has_fired`] the B1-dedup / SeedDrift scans read; a
    /// "never fired" answer for a template is the truth, not a default.
    #[must_use]
    pub const fn has_fired(&self) -> bool {
        match &self.reaction {
            Reaction::Spawn { history, .. } => history.has_fired,
            Reaction::Mint { .. } => false,
        }
    }
}

/// Slotmap-backed Sub store with three secondary indices.
///
/// Each index keys on a field no edge writes after [`Self::insert`] — a Sub's `profile`, `name` and
/// `minted_by` — so `insert` and [`Self::remove`] are the only writers of every index.
///
/// - `by_profile` groups Subs by `ProfileId` (insertion order within a Profile).
/// - `by_name` resolves an operator-facing or synthesised name to its `SubId`. Indexes **every**
///   Sub regardless of provenance: the config validator reserves the `@` byte, so a
///   `[[watch]].name` never carries one and a minted `<template_name>@<matched_path>` always does —
///   the two populations are disjoint by construction and their union is unique. Callers that tell
///   them apart read [`Sub::minted_by`] on the resolved Sub. The index is load-bearing — hot-reload
///   resolves every `removed`/`modified` name to an id through [`Self::find_by_name`] (O(log N)).
/// - `by_template` holds each template's minted set ([`Self::minted`]), keyed on the `minted_by`
///   every minted Sub carries. An entry leaves with its last member, or with its template, whose
///   removal hands the members back ([`Removed::orphans`]).
///
/// A duplicate name is loud in dev — `insert`'s `debug_assert!` — and unreachable in correct
/// operation: the validator (static side) and the discovery reconcile's dedup (dynamic side)
/// prevent it, and the `@`-disjointness keeps the two construction sites from racing each other. A
/// release-mode breach is contained by `remove`'s id-checked clear of `by_name`: the index is
/// last-writer-wins, so the name resolves to the later id, and the id-check refuses to clear that
/// mapping when the earlier (shadowed) id is removed — the *mapping* stays 1:1, the wrong name→id
/// edge never forms. The shadowed Sub is **not** inert, though: its slotmap entry stays in
/// `by_profile` and keeps firing on its Profile's verdicts until that Profile is reaped — it is
/// merely unreachable by name. Containment bounds the damage; it does not erase the duplicate.
#[derive(Debug, Default)]
pub struct SubRegistry {
    subs: SlotMap<SubId, Sub>,
    by_profile: BTreeMap<ProfileId, SmallVec<[SubId; 2]>>,
    by_name: BTreeMap<CompactString, SubId>,
    by_template: BTreeMap<SubId, BTreeSet<SubId>>,
}

/// What [`SubRegistry::remove`] took out: the Sub, and what it minted.
#[derive(Debug)]
#[must_use = "a removed template hands back the Subs it minted; the caller owes their detach"]
pub struct Removed {
    pub sub: Sub,
    /// The Subs that name `sub` their template ([`Sub::minted_by`]), in id order — still
    /// registered, their template gone. Empty for any Sub that minted nothing.
    pub orphans: BTreeSet<SubId>,
}

impl SubRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a Sub and file it in every index; the returned slotmap [`SubId`] is its identity
    /// authority (the Sub carries no `id` field).
    pub fn insert(&mut self, sub: Sub) -> SubId {
        let profile = sub.profile;
        let name = sub.name.clone();
        let template = sub.minted_by();
        let id = self.subs.insert(sub);
        self.by_profile.entry(profile).or_default().push(id);
        debug_assert!(
            !self.by_name.contains_key(&name),
            "duplicate Sub name escaped registration: {name:?}",
        );
        self.by_name.insert(name, id);
        if let Some(template) = template {
            self.by_template.entry(template).or_default().insert(id);
        }
        id
    }

    /// Remove a Sub by id and clear it from every index; `None` for a stale id. A template leaves
    /// with its minted set: the members stay registered and come back as [`Removed::orphans`],
    /// whose detach the caller owes.
    #[must_use]
    pub fn remove(&mut self, id: SubId) -> Option<Removed> {
        let sub = self.subs.remove(id)?;
        if let Some(v) = self.by_profile.get_mut(&sub.profile) {
            v.retain(|sid| *sid != id);
            if v.is_empty() {
                self.by_profile.remove(&sub.profile);
            }
        }
        if self.by_name.get(&sub.name) == Some(&id) {
            self.by_name.remove(&sub.name);
        }
        // No entry is an orphan's: its template's removal already took the set it was filed in.
        if let Some(template) = sub.minted_by()
            && let Some(minted) = self.by_template.get_mut(&template)
        {
            minted.remove(&id);
            if minted.is_empty() {
                self.by_template.remove(&template);
            }
        }
        let orphans = self.by_template.remove(&id).unwrap_or_default();
        Some(Removed { sub, orphans })
    }

    #[must_use]
    pub fn get(&self, id: SubId) -> Option<&Sub> {
        self.subs.get(id)
    }

    /// Subs attached to `profile`, in insertion order. Empty slice if none.
    #[must_use]
    pub fn at(&self, profile: ProfileId) -> &[SubId] {
        self.by_profile.get(&profile).map_or(&[], |v| v.as_slice())
    }

    /// The Subs `template` minted, in id order; empty for a Sub that minted nothing.
    pub fn minted(&self, template: SubId) -> impl ExactSizeIterator<Item = SubId> {
        self.by_template
            .get(&template)
            .map(BTreeSet::iter)
            .unwrap_or_default()
            .copied()
    }

    /// The Sub `template` minted on `profile`, read off the one or two Subs the Profile holds — the
    /// mint dedup. Another Sub sharing the Profile, an operator's or another template's, is no
    /// answer.
    #[must_use]
    pub fn minted_at(&self, template: SubId, profile: ProfileId) -> Option<SubId> {
        self.at(profile).iter().copied().find(|&sid| {
            self.subs
                .get(sid)
                .is_some_and(|s| s.minted_by() == Some(template))
        })
    }

    /// The Profiles `sub`'s watch fires on, in the minted Subs' id order: a firing Sub's own; for a
    /// template, whose own Profile reconciles and never fires, the Profile each Sub it minted sits
    /// on — one per Sub, by the mint dedup ([`Self::minted_at`]). Empty for a stale id.
    pub fn firing_profiles(&self, sub: SubId) -> impl Iterator<Item = ProfileId> {
        let own = self.subs.get(sub).filter(|s| !s.is_template());
        let minted = self
            .minted(sub)
            .map(|minted| self.subs.get(minted).expect("a minted set holds live Subs"));
        own.into_iter().chain(minted).map(Sub::profile)
    }

    pub fn iter(&self) -> impl Iterator<Item = (SubId, &Sub)> {
        self.subs.iter()
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.subs.len()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.subs.is_empty()
    }

    /// Resolve a user-facing `name` to its [`SubId`] in O(log N) via `by_name`. `None` if no Sub
    /// holds `name`.
    ///
    /// A declared Sub's name and a minted one resolve alike — callers that tell them apart read
    /// [`Sub::minted_by`] on the resolved Sub. The config validator reserves the `@` byte, so a
    /// declared name and a minted `<template_name>@<matched_path>` cannot collide; uniqueness
    /// across the union is structural.
    ///
    /// Load-bearing: the engine's hot-reload shim resolves every `removed`/`modified` name through
    /// here.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<SubId> {
        self.by_name.get(name).copied()
    }

    /// The `Spawn` arm's mutable fire history for `sub` — the shared body of the four fire-history
    /// edges. Two misses, two loudnesses: a stale id is a silent `None` (the history already died
    /// with the slotmap entry — a benign detach race), while a `Mint` hit is a **debug-loud**
    /// `None` (a fire-history write on a template is a routing logic error — discovery fires
    /// attachments, never Effects). `edge` names the calling edge in the assert.
    fn spawn_history_mut(&mut self, sub: SubId, edge: &str) -> Option<&mut FireHistory> {
        match &mut self.subs.get_mut(sub)?.reaction {
            Reaction::Spawn { history, .. } => Some(history),
            Reaction::Mint { .. } => {
                debug_assert!(
                    false,
                    "{edge} reached a template — discovery fires attachments, never Effects \
                     (sub = {sub:?})",
                );
                None
            }
        }
    }

    /// The `Mint` arm's mutable state for `sub` — the shared body of the template edges, and the
    /// dual of [`Self::spawn_history_mut`]: a stale id is a silent `None` (the template died with
    /// its slotmap entry — a benign detach race), a `Spawn` hit a **debug-loud** `None` (a
    /// template's state lives only on `Mint`). `edge` names the calling edge in the assert.
    fn mint_state_mut(&mut self, sub: SubId, edge: &str) -> Option<&mut MintState> {
        match &mut self.subs.get_mut(sub)?.reaction {
            Reaction::Mint { state, .. } => Some(state),
            Reaction::Spawn { .. } => {
                debug_assert!(
                    false,
                    "{edge} reached a Spawn Sub — a template's state lives only on Mint \
                     (sub = {sub:?})",
                );
                None
            }
        }
    }

    /// Record that `sub` emitted an Effect — the B1-dedup / SeedDrift fire-history mark, written by
    /// `emit_effects`' SubtreeRoot arm on a successful push. Idempotent. A stale `SubId` (the Sub
    /// detached between the emit decision and here) is a silent no-op: the flag already died with
    /// the slotmap entry. A `Mint` hit is a debug-loud no-op (see `spawn_history_mut`).
    pub fn mark_fired(&mut self, sub: SubId) {
        if let Some(h) = self.spawn_history_mut(sub, "mark_fired") {
            h.has_fired = true;
        }
    }

    /// Record `count` successful Effect emissions on `sub` at `now`. Bumps
    /// [`FireHistory::fire_count`] by `count` (saturating) and writes [`FireHistory::last_fired_at`]
    /// = `Some(now)`. Observational — [`Self::mark_fired`] is the load-bearing B1-dedup edge; this
    /// one carries the per-Sub fire history the operator-facing `list` projection renders.
    ///
    /// Called at most once per Sub per `emit_effects` pass on the emit-side. `count` is `1` for a
    /// `SubtreeRoot` emission and the per-file count for a `PerStableFile` emission (aggregated so
    /// `Diagnostic::SubFired`'s wire stream isn't amplified by N). A stale `SubId` is a silent
    /// no-op — the counter already died with the slotmap entry, mirroring [`Self::mark_fired`].
    pub fn record_fired(&mut self, sub: SubId, count: u32, now: Instant) {
        if let Some(h) = self.spawn_history_mut(sub, "record_fired") {
            h.fire_count = h.fire_count.saturating_add(u64::from(count));
            h.last_fired_at = Some(now);
        }
    }

    /// Bump [`FireHistory::dedup_suppressed_count`] by one — written when `emit_effects` resolves a
    /// `SubtreeRoot` Sub to `FireVerdict::SuppressDedup` (unchanged tree + already-fired + not
    /// forced). Saturating add; observational only. A stale `SubId` is a silent no-op — same shape
    /// as [`Self::mark_fired`] and [`Self::record_fired`].
    pub fn record_dedup_suppressed(&mut self, sub: SubId) {
        if let Some(h) = self.spawn_history_mut(sub, "record_dedup_suppressed") {
            h.dedup_suppressed_count = h.dedup_suppressed_count.saturating_add(1);
        }
    }

    /// Replace `sub`'s params in place, and those of every Sub it minted — the `modified_params`
    /// arm of hot-reload's [`SubRegistryDiff`] split. A template's edit reaches its minted set in
    /// the one edge, so no Sub of the set runs params its template no longer holds.
    ///
    /// **Preserves**: every [`SubId`], `profile`, `name` and `minted_by`, every [`FireHistory`], a
    /// template's [`MintState`], and its spec's identity — the edge is handed none, so the mint
    /// dedup's key outlives every rebind. Ids and Profile joins are invariants of a Sub's lifetime,
    /// and a request carries no provenance to change `minted_by` with; `name` is the key the caller
    /// found `sub` by ([`Self::find_by_name`]), and params carry none. `has_fired` is preserved
    /// because the B1 dedup floor reads it as "this Sub has already announced the current stable
    /// tree state"; a program swap changes *what runs*, not *whether the tree changed*. The three
    /// observational counters are preserved for the same reason — they record a Sub's history under
    /// its operator-facing name. A template's enumeration stands as it was, so a match that appears
    /// after the edit is still an appearance.
    ///
    /// **Replaces**: the [`SubParams`] wholesale — a firing Sub's own, or a template's spec's and
    /// each of its minted Subs' copy. The sealed [`SpawnSpec`] carries `needs_diff` pre-derived at
    /// [`SpawnSpec::new`], so there is no recompute here to forget.
    ///
    /// Returns the params `sub` held; `None` for a stale id. The settle floors of the Profiles the
    /// edge reached — `sub`'s, and each minted Sub's — are the caller's to refold.
    #[must_use]
    pub fn rebind(&mut self, sub: SubId, params: SubParams) -> Option<SubParams> {
        for &minted in self.by_template.get(&sub).into_iter().flatten() {
            self.subs
                .get_mut(minted)
                .expect("a minted set holds live Subs")
                .reaction
                .params_mut()
                .clone_from(&params);
        }
        let live = self.subs.get_mut(sub)?.reaction.params_mut();
        Some(std::mem::replace(live, params))
    }

    /// Clear `sub`'s fire history — the `EffectComplete::Failed` clear. A failed Effect produced no
    /// observation worth deduplicating against, so the next stable verdict at this Sub must re-fire
    /// even on an unchanged tree. No-op on a stale `SubId` (already detached ⇒ its history is
    /// already gone).
    pub fn clear_fired(&mut self, sub: SubId) {
        if let Some(h) = self.spawn_history_mut(sub, "clear_fired") {
            h.has_fired = false;
        }
    }

    /// Whether any Sub on `profile` has fired — the fast `seed_drift_observed` short-circuit
    /// ("never fired ⇒ no prior emission to re-fire on recovery"). The scan legitimately runs
    /// Profile-wide: a `Mint` Sub reports never-fired through [`Sub::has_fired`] — no assert.
    #[must_use]
    pub fn any_fired(&self, profile: ProfileId) -> bool {
        self.at(profile)
            .iter()
            .any(|sid| self.subs.get(*sid).is_some_and(Sub::has_fired))
    }

    /// The Subs on `profile` that have fired — the SeedDrift conservative-recovery fire-filter
    /// basis. `Mint` Subs naturally never qualify ([`Sub::has_fired`] is structurally `false`).
    ///
    /// **Order is membership only.** The caller filters with `.contains`; the observable Effect order
    /// is established globally by [`crate::StepOutput::sort_for_emission`] (the load-bearing `(SubId,
    /// ResourceId)` canonicalisation every step applies before returning), so the insertion order
    /// `at` yields here is sufficient and deterministic — there is no per-call re-sort to justify.
    #[must_use]
    pub fn fired_in(&self, profile: ProfileId) -> SmallVec<[SubId; 2]> {
        self.at(profile)
            .iter()
            .copied()
            .filter(|sid| self.subs.get(*sid).is_some_and(Sub::has_fired))
            .collect()
    }

    /// The least settle of the Subs on `profile` — the floor its Profile debounces at; `None` for a
    /// Profile with no Sub. Loud, not filtered: `insert` and `remove` keep `by_profile` and the
    /// slotmap in lockstep, so an id one holds and the other does not is a breach of this
    /// registry's own invariant, and a floor folded past it would be silently too high.
    #[must_use]
    pub fn settle_floor(&self, profile: ProfileId) -> Option<Duration> {
        self.at(profile)
            .iter()
            .map(|&sid| {
                self.subs
                    .get(sid)
                    .expect("by_profile and the slotmap move in lockstep")
                    .settle()
            })
            .min()
    }

    /// Whether `template` makes `what` now: `true` the first time, `false` ever after — the check
    /// and the record are one edge, so a line is made once per template lifetime by construction. A
    /// stale id is `false`.
    pub fn narrate_once(&mut self, template: SubId, what: MintNarration) -> bool {
        self.mint_state_mut(template, "narrate_once")
            .is_some_and(|state| {
                let first = state.narrated & what.bit() == 0;
                state.narrated |= what.bit();
                first
            })
    }

    /// `template` holds a certified set: from now on a miss in a branch the daemon had read is an
    /// appearance ([`MintState::miss`]). Idempotent. A set is certified by a pass that walked a
    /// certified snapshot, or by a prefix seen absent — the empty set; a pass that walked nothing
    /// established none.
    pub fn close_enumeration(&mut self, template: SubId) {
        if let Some(state) = self.mint_state_mut(template, "close_enumeration") {
            state.miss = MintEntry::Appeared;
        }
    }

    /// Whether `profile` has at least one attached Sub that *reacts* per-stable-file — the scope
    /// test behind the per-file recovery-drop signal. A `Mint` Sub has no scope of its own (its
    /// template's per-file scope is the *minted* Subs' reaction, answered on their own Profiles),
    /// so the variant dispatch excludes it structurally.
    ///
    /// **Must not be collapsed into [`crate::Profile::has_per_file_fds`].** That predicate is
    /// events-mask derived (`CONTENT | METADATA` present) and a `SubtreeRoot` Sub watching
    /// `CONTENT` sets it just as much as a `PerStableFile` Sub does — it is *necessary* for
    /// per-file FDs but *not sufficient* for "this Profile carries a per-file-*scoped* reaction".
    /// Swapping this scan for `has_per_file_fds` would false-positive the recovery-drop diagnostic
    /// on Subtree-only Profiles that happen to watch content. The spawn scope is the only sound
    /// witness; the scan stays.
    #[must_use]
    pub fn has_per_stable_file_sub(&self, profile: ProfileId) -> bool {
        self.at(profile).iter().any(|sid| {
            self.subs.get(*sid).is_some_and(|s| {
                s.spawn_spec()
                    .is_some_and(|spec| spec.scope() == EffectScope::PerStableFile)
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActionProgram, ClassSet, EffectScope, MintEntry, MintNarration, MintSpec, MintState,
        SpawnSpec, Sub, SubAttachAnchor, SubAttachRequest, SubAttachTarget, SubParams, SubRegistry,
        SubRegistryDiff, SubtreeScan,
    };
    use crate::ids::{ProfileId, SubId};
    use crate::program::{
        ArgPart, ArgTemplate, BranchTarget, ExecAction, Placeholder, ProgramBuilder, SpawnBody,
    };
    use compact_str::CompactString;
    use std::num::NonZeroU32;
    use std::sync::Arc;
    use std::time::{Duration, Instant};

    const SETTLE: Duration = Duration::from_millis(100);

    /// Build a one-op program holding a single Exec body. Equivalent to the lowering of a single
    /// `[[watch.actions]] exec = [...]` entry.
    fn single_exec_program(exec: ExecAction) -> Arc<ActionProgram> {
        let mut b = ProgramBuilder::new();
        let h = b.emit(SpawnBody::Exec(exec));
        b.patch_on_ok(h, BranchTarget::Escape).unwrap();
        b.patch_on_failed(h, BranchTarget::Terminate).unwrap();
        Arc::new(b.build().unwrap())
    }

    fn anchor_only_program() -> Arc<ActionProgram> {
        single_exec_program(ExecAction::new(
            [ArgTemplate::new([
                ArgPart::literal("/bin/build"),
                ArgPart::Placeholder(Placeholder::Path),
            ])],
            None,
        ))
    }

    fn program_with(p: Placeholder) -> Arc<ActionProgram> {
        single_exec_program(ExecAction::new(
            [ArgTemplate::new([ArgPart::Placeholder(p)])],
            None,
        ))
    }

    /// What a fixture Sub runs: `program` at `scope`, debounced at `SETTLE`.
    fn params(program: Arc<ActionProgram>, scope: EffectScope) -> SubParams {
        SubParams {
            settle: SETTLE,
            spawn: SpawnSpec::new(program, scope),
        }
    }

    /// A Sub a request declared on `profile`, running the fixture program.
    fn declared(profile: ProfileId, name: &str) -> Sub {
        Sub::declared(
            profile,
            name.into(),
            params(anchor_only_program(), EffectScope::SubtreeRoot),
        )
    }

    #[test]
    fn references_diff_derived_for_each_diff_placeholder() {
        for p in [
            Placeholder::Created,
            Placeholder::Deleted,
            Placeholder::Modified,
            Placeholder::RenamedFrom,
            Placeholder::RenamedTo,
        ] {
            assert!(
                program_with(p).references_diff_derived(),
                "references_diff_derived must be true for {p:?}"
            );
        }
    }

    #[test]
    fn references_diff_derived_false_for_anchor_only_program() {
        assert!(!anchor_only_program().references_diff_derived());
        // The full non-diff-derived set: every single-value placeholder PLUS `Excluded`
        // (multi-value but not diff-derived). Including `Excluded` here is the load-bearing
        // assertion of the `is_multivalue` / `is_diff_derived` split — using the `Excluded` variant
        // in a template must NOT ratchet `Sub.needs_diff` true.
        for p in [
            Placeholder::Path,
            Placeholder::Relative,
            Placeholder::Anchor,
            Placeholder::Watch,
            Placeholder::Parent,
            Placeholder::Time,
            Placeholder::Excluded,
        ] {
            assert!(
                !program_with(p).references_diff_derived(),
                "references_diff_derived must be false for non-diff-derived {p:?}"
            );
        }
    }

    #[test]
    fn needs_diff_set_for_per_stable_file_scope() {
        let spec = SpawnSpec::new(anchor_only_program(), EffectScope::PerStableFile);
        assert!(spec.needs_diff());
    }

    #[test]
    fn needs_diff_set_for_diff_placeholder_in_subtree_scope() {
        let spec = SpawnSpec::new(program_with(Placeholder::Created), EffectScope::SubtreeRoot);
        assert!(spec.needs_diff());
    }

    #[test]
    fn needs_diff_false_for_anchor_subtree_combo() {
        let spec = SpawnSpec::new(anchor_only_program(), EffectScope::SubtreeRoot);
        assert!(!spec.needs_diff());
    }

    /// A freshly built Sub starts with no fire history — the B1-dedup / SeedDrift baseline.
    /// Relocated from the deleted per-Profile `new_profile_initialises_fired_subs_empty`: the
    /// history now lives per-Sub, so the "starts empty" contract is asserted on the Sub, not the
    /// Profile. Also pins the three observational counters' fresh state — `record_fired` /
    /// `record_dedup_suppressed` are the only writers, so a fresh Sub can never carry inherited
    /// history from a slotmap slot's prior occupant.
    #[test]
    fn fresh_sub_starts_unfired() {
        let sub = declared(ProfileId::default(), "build");
        let h = sub.fire_history().expect("spawn reaction carries history");
        assert!(!h.has_fired, "fresh Sub: no prior Effect emission");
        assert!(h.last_fired_at.is_none(), "no fire timestamp");
        assert_eq!(h.fire_count, 0, "no cumulative fires");
        assert_eq!(h.dedup_suppressed_count, 0, "no suppressed verdicts");
    }

    /// `record_fired` accumulates per-pass counts into `fire_count` and stamps `last_fired_at` with
    /// the supplied instant. The B1-dedup `has_fired` is untouched — `mark_fired` and
    /// `record_fired` are disjoint edge methods on disjoint pieces of fire history.
    #[test]
    fn record_fired_bumps_count_and_stamps_last_fired() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(declared(pid, "build"));
        let t0 = Instant::now();
        reg.record_fired(sid, 1, t0);
        let h = reg.get(sid).and_then(Sub::fire_history).expect("Sub alive");
        assert_eq!(h.fire_count, 1, "first fire bumps count by 1");
        assert_eq!(h.last_fired_at, Some(t0), "first fire stamps timestamp");
        assert!(
            !h.has_fired,
            "record_fired must NOT touch has_fired (mark_fired owns it)",
        );

        // A PerStableFile-style aggregation: count=3 adds to the running total, timestamp advances.
        let t1 = t0 + Duration::from_millis(10);
        reg.record_fired(sid, 3, t1);
        let h = reg.get(sid).and_then(Sub::fire_history).expect("Sub alive");
        assert_eq!(h.fire_count, 4, "second fire aggregates: 1 + 3 = 4");
        assert_eq!(h.last_fired_at, Some(t1), "timestamp advances");

        // Stale id is a silent no-op.
        let _ = reg.remove(sid).expect("removed");
        reg.record_fired(sid, 1, t1); // would otherwise panic on missing entry
    }

    /// `record_dedup_suppressed` increments the dedicated counter and touches no other field — the
    /// SuppressDedup arm signals "Sub would have fired but the dedup floor said no", distinct from
    /// fires (`record_fired`) and the B1 flag (`mark_fired`).
    #[test]
    fn record_dedup_suppressed_bumps_only_its_own_counter() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(declared(pid, "build"));
        reg.record_dedup_suppressed(sid);
        reg.record_dedup_suppressed(sid);
        let h = reg.get(sid).and_then(Sub::fire_history).expect("Sub alive");
        assert_eq!(h.dedup_suppressed_count, 2);
        assert_eq!(h.fire_count, 0, "suppression does not bump fire_count");
        assert!(
            h.last_fired_at.is_none(),
            "suppression does not stamp last_fired_at",
        );
        assert!(!h.has_fired, "suppression does not touch has_fired");

        let _ = reg.remove(sid).expect("removed");
        reg.record_dedup_suppressed(sid); // stale id is a silent no-op
    }

    #[test]
    fn registry_at_groups_by_profile() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();

        let s1 = reg.insert(declared(pid, "a"));
        let s2 = reg.insert(declared(pid, "b"));

        let mut got = reg.at(pid).to_vec();
        got.sort();
        let mut expected = vec![s1, s2];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn registry_at_empty_for_unknown_profile() {
        let reg = SubRegistry::new();
        assert_eq!(reg.at(ProfileId::default()), []);
    }

    /// After a multi-insert/remove sequence, every key `iter()` yields re-looks-up via `get` to the
    /// same Sub, and `at(profile)` equals the live key set. The slotmap key is the sole identity
    /// authority (a `Sub` carries no `id`).
    #[test]
    fn registry_iter_keys_round_trip_through_get() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mk = |name: &str| declared(pid, name);
        let a = reg.insert(mk("a"));
        let b = reg.insert(mk("b"));
        let c = reg.insert(mk("c"));
        let _ = reg.remove(b);

        let mut iter_keys: Vec<SubId> = reg
            .iter()
            .map(|(k, s)| {
                assert_eq!(
                    reg.get(k).expect("iter key resolves via get").name,
                    s.name,
                    "get(k) returns the same entry iter yielded",
                );
                k
            })
            .collect();
        iter_keys.sort();

        let mut want = vec![a, c];
        want.sort();
        assert_eq!(iter_keys, want, "iter yields exactly the live keys");

        let mut at = reg.at(pid).to_vec();
        at.sort();
        assert_eq!(at, want, "at(profile) agrees with the live key set");

        assert!(reg.get(b).is_none(), "removed key no longer resolves");
        assert_eq!(reg.len(), 2);
    }

    #[test]
    fn find_by_name_returns_some_for_match() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let id = reg.insert(declared(pid, "build"));
        assert_eq!(reg.find_by_name("build"), Some(id));
    }

    #[test]
    fn find_by_name_returns_none_for_absent() {
        let reg = SubRegistry::new();
        assert!(reg.find_by_name("missing").is_none());
    }

    #[test]
    fn find_by_name_returns_none_after_remove() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let id = reg.insert(declared(pid, "build"));
        let _ = reg.remove(id);
        assert!(reg.find_by_name("build").is_none());
    }

    /// `by_name` indexes every Sub regardless of provenance — both an operator's name and a minted
    /// `<template_name>@<matched_path>` resolve through `find_by_name`. The two populations are
    /// disjoint by the config validator's `@`-byte reservation, so their indexed union is unique.
    #[test]
    fn by_name_indexes_declared_and_minted_subs() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let declared_id = reg.insert(declared(pid, "foo"));
        let minted_id = reg.insert(minted(pid, "p@/tmp/x", SubId::default()));

        assert_eq!(
            reg.find_by_name("foo"),
            Some(declared_id),
            "a declared name resolves",
        );
        assert_eq!(
            reg.find_by_name("p@/tmp/x"),
            Some(minted_id),
            "a synthesised minted name resolves",
        );
        assert!(
            reg.find_by_name("nope").is_none(),
            "absent name yields None",
        );
    }

    /// `remove` drops a minted Sub's `by_name` entry just like a declared one's — and `by_profile`
    /// accounting is symmetric across the two.
    #[test]
    fn remove_clears_by_name_for_minted_sub() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let minted_id = reg.insert(minted(pid, "p@/tmp/x", SubId::default()));
        assert_eq!(reg.find_by_name("p@/tmp/x"), Some(minted_id));

        let _ = reg.remove(minted_id).expect("minted removed");
        assert!(
            reg.find_by_name("p@/tmp/x").is_none(),
            "the minted Sub's by_name entry dropped on remove",
        );
        assert!(
            reg.at(pid).is_empty(),
            "by_profile bucket dropped when last Sub leaves",
        );
    }

    #[test]
    fn registry_remove_clears_by_profile_and_drops_empty_bucket() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(declared(pid, "build"));

        let removed = reg.remove(sid);
        assert!(removed.is_some());
        assert!(reg.get(sid).is_none());
        assert_eq!(reg.at(pid), []);
        assert_eq!(reg.len(), 0);
    }

    /// The minted set follows the two edges: `insert` files a minted Sub under its template and
    /// `remove` takes it out; a template's removal hands back what it still minted — registered
    /// until its detach — and no entry outlives the Subs it holds.
    #[test]
    fn minted_set_follows_insert_and_remove_and_leaves_with_its_template() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let template = reg.insert(template(pid, "disc"));
        let a = reg.insert(minted(pid, "disc@/a", template));
        let b = reg.insert(minted(pid, "disc@/b", template));
        let _ = reg.insert(declared(pid, "build"));
        assert_eq!(reg.minted(template).collect::<Vec<_>>(), [a, b]);

        let removed = reg.remove(a).expect("live");
        assert!(removed.orphans.is_empty(), "a minted Sub mints nothing");
        assert_eq!(reg.minted(template).collect::<Vec<_>>(), [b]);

        let removed = reg.remove(template).expect("live");
        assert_eq!(
            removed.orphans.into_iter().collect::<Vec<_>>(),
            [b],
            "the template leaves with what it still minted",
        );
        assert_eq!(reg.minted(template).len(), 0);
        assert!(
            reg.get(b).is_some(),
            "an orphan stays registered until its detach"
        );

        let _ = reg.remove(b).expect("live");
        assert!(
            reg.by_template.is_empty(),
            "no entry outlives the Subs it holds"
        );
    }

    /// `minted_at` answers with the template's own Sub on a Profile: an operator's Sub or another
    /// template's mint there is no answer, and a Profile the template never minted on has none.
    #[test]
    fn minted_at_reads_only_the_templates_own_sub_on_a_profile() {
        let profile = |n| ProfileId::from(slotmap::KeyData::from_ffi(n));
        let (discovery, shared) = (profile(1), profile(2));
        let mut reg = SubRegistry::new();
        let alpha = reg.insert(template(discovery, "alpha"));
        let beta = reg.insert(template(discovery, "beta"));
        let _ = reg.insert(declared(shared, "build"));
        assert_eq!(reg.minted_at(alpha, shared), None, "an operator's Sub");
        let beta_mint = reg.insert(minted(shared, "beta@/x", beta));
        assert_eq!(
            reg.minted_at(alpha, shared),
            None,
            "another template's mint"
        );
        assert_eq!(reg.minted_at(beta, shared), Some(beta_mint));
        let alpha_mint = reg.insert(minted(shared, "alpha@/x", alpha));
        assert_eq!(reg.minted_at(alpha, shared), Some(alpha_mint));
        assert_eq!(
            reg.minted_at(alpha, discovery),
            None,
            "nothing minted there"
        );
    }

    /// The spawn program is reference-counted: cloning it bumps the strong count without copying
    /// the inner [`ActionProgram`].
    #[test]
    fn sub_program_is_arc_wrapped() {
        let sub = declared(ProfileId::default(), "build");

        let program = sub.spawn_spec().expect("spawn reaction").program();
        let initial = Arc::strong_count(program);
        let bumped = Arc::clone(program);
        assert_eq!(
            Arc::strong_count(program),
            initial + 1,
            "Arc::clone increments strong_count on the spec",
        );
        assert!(
            Arc::ptr_eq(&bumped, program),
            "the clone and the spec point at the same allocation",
        );
    }

    /// `Sub::declared` does not re-wrap the program: the caller's Arc is the same allocation the Sub
    /// stores. The minted Arc from the config layer's `lower_to_program` flows through without churn.
    #[test]
    fn sub_new_does_not_rewrap_program_arc() {
        let program = anchor_only_program();
        let before = Arc::as_ptr(&program);
        let sub = Sub::declared(
            ProfileId::default(),
            "build".into(),
            params(Arc::clone(&program), EffectScope::SubtreeRoot),
        );
        assert!(
            std::ptr::eq(
                before,
                Arc::as_ptr(sub.spawn_spec().expect("spawn reaction").program()),
            ),
            "Sub::declared must not allocate a new ActionProgram",
        );
    }

    /// Diff is plain data — pins the `Default` shape and the [`SubRegistryDiff::is_empty`] contract
    /// in one place. **Each** of the four buckets independently flips the predicate, so a future
    /// bucket addition that forgets to extend `is_empty` is caught here.
    #[test]
    fn sub_registry_diff_is_empty_per_bucket() {
        assert!(SubRegistryDiff::default().is_empty(), "default is empty");

        let req = || SubAttachRequest {
            name: "a".into(),
            target: SubAttachTarget::At(SubAttachAnchor::Path("/a".into())),
            scan: SubtreeScan::builder().build(),
            max_settle: Duration::from_hours(1),
            events: ClassSet::DEFAULT_SUBTREE_ROOT,
            params: params(anchor_only_program(), EffectScope::SubtreeRoot),
        };

        for (label, d) in [
            (
                "added",
                SubRegistryDiff {
                    added: vec![req()],
                    ..Default::default()
                },
            ),
            (
                "removed",
                SubRegistryDiff {
                    removed: vec![CompactString::from("a")],
                    ..Default::default()
                },
            ),
            (
                "modified_identity",
                SubRegistryDiff {
                    modified_identity: vec![req()],
                    ..Default::default()
                },
            ),
            (
                "modified_params",
                SubRegistryDiff {
                    modified_params: vec![req()],
                    ..Default::default()
                },
            ),
        ] {
            assert!(
                !d.is_empty(),
                "populating `{label}` must flip is_empty to false",
            );
        }
    }

    /// `SubRegistry::rebind` replaces the params and preserves the structural fields — including
    /// `has_fired`, which the B1 dedup floor reads as "this Sub has already announced the current
    /// stable tree state." A program swap changes *what runs*, not *whether the tree changed*, so
    /// the flag must not reset on rebind.
    #[test]
    fn rebind_replaces_per_sub_fields_and_preserves_has_fired() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(declared(pid, "build"));
        reg.mark_fired(sid);
        let fired_at = Instant::now();
        reg.record_fired(sid, 7, fired_at);
        reg.record_dedup_suppressed(sid);

        let new_program = program_with(Placeholder::Created);
        let new_settle = SETTLE + SETTLE;
        let prior = reg.rebind(
            sid,
            SubParams {
                settle: new_settle,
                spawn: SpawnSpec::new(Arc::clone(&new_program), EffectScope::PerStableFile),
            },
        );

        let prior = prior.expect("a live Sub rebinds");
        assert_eq!(
            (prior.settle, prior.spawn.scope()),
            (SETTLE, EffectScope::SubtreeRoot),
            "rebind hands back the params it replaced",
        );
        let s = reg.get(sid).expect("Sub preserved across rebind");
        let h = s.fire_history().expect("spawn reaction carries history");
        assert!(h.has_fired, "has_fired preserved across rebind");
        assert_eq!(
            h.last_fired_at,
            Some(fired_at),
            "last_fired_at preserved — operator-facing fire history",
        );
        assert_eq!(h.fire_count, 7, "fire_count preserved across rebind");
        assert_eq!(
            h.dedup_suppressed_count, 1,
            "dedup_suppressed_count preserved across rebind",
        );
        assert_eq!(s.name, "build", "name preserved");
        assert_eq!(s.profile, pid, "profile preserved");
        assert!(s.minted_by().is_none(), "minted_by preserved");
        let spec = s.spawn_spec().expect("spawn reaction carries a spec");
        assert_eq!(spec.scope(), EffectScope::PerStableFile, "scope replaced");
        assert_eq!(s.settle(), new_settle, "settle replaced");
        assert!(
            Arc::ptr_eq(spec.program(), &new_program),
            "program Arc replaced (no rewrap)",
        );
        assert!(
            spec.needs_diff(),
            "needs_diff travels with the sealed spec — PerStableFile alone sets it true",
        );
    }

    /// Stale `SubId` returns `None`. The dispatcher resolves through `find_by_name` in the same
    /// step as the rebind, so this surface is rarely hit in production; the engine wraps it in a
    /// `RebindUnknownSub` diagnostic when it does.
    #[test]
    fn rebind_returns_none_on_stale_sub_id() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(declared(pid, "build"));
        let _ = reg.remove(sid).expect("removed");
        let res = reg.rebind(sid, params(anchor_only_program(), EffectScope::SubtreeRoot));
        assert!(res.is_none(), "stale SubId yields None");
    }

    /// A mint spec over an arbitrary subtree scan, its mints running the fixture program at `scope`
    /// — these pins exercise registry mechanics only.
    fn mint_spec(scope: EffectScope) -> MintSpec {
        MintSpec::new(
            SubtreeScan::builder().max_depth(NonZeroU32::new(1)).build(),
            SETTLE * 4,
            ClassSet::EMPTY,
            params(anchor_only_program(), scope),
        )
    }

    /// A template on `profile` whose mints run the fixture program.
    fn template(profile: ProfileId, name: &str) -> Sub {
        Sub::template(profile, name.into(), mint_spec(EffectScope::SubtreeRoot))
    }

    /// A Sub `template` minted on `profile`.
    fn minted(profile: ProfileId, name: &str, template: SubId) -> Sub {
        Sub::minted(
            profile,
            name.into(),
            template,
            &mint_spec(EffectScope::SubtreeRoot),
        )
    }

    /// Each line is made once per template lifetime, and making one leaves the other unmade.
    #[test]
    fn narrate_once_makes_each_line_once() {
        let mut reg = SubRegistry::new();
        let sid = reg.insert(template(ProfileId::default(), "disc"));
        assert!(reg.narrate_once(sid, MintNarration::FanoutThreshold));
        assert!(!reg.narrate_once(sid, MintNarration::FanoutThreshold));
        assert!(reg.narrate_once(sid, MintNarration::UnsupportedAnchorKind));
        assert!(!reg.narrate_once(sid, MintNarration::UnsupportedAnchorKind));
    }

    /// A template edge aimed at a `Spawn` Sub is a routing error: loud in a debug build, and in a
    /// release build no line is made.
    #[test]
    #[cfg_attr(debug_assertions, should_panic(expected = "reached a Spawn Sub"))]
    fn narrate_once_on_a_spawn_sub_is_loud_in_debug_and_makes_no_line() {
        let mut reg = SubRegistry::new();
        let plain = reg.insert(declared(ProfileId::default(), "build"));
        assert!(!reg.narrate_once(plain, MintNarration::FanoutThreshold));
    }

    /// A stale id makes no line, quietly: the template died with its slotmap entry.
    #[test]
    fn narrate_once_makes_no_line_for_a_stale_id() {
        let mut reg = SubRegistry::new();
        let sid = reg.insert(template(ProfileId::default(), "disc"));
        let _ = reg.remove(sid).expect("removed");
        assert!(!reg.narrate_once(sid, MintNarration::FanoutThreshold));
    }

    /// A fresh template's miss is an enumeration; closing its enumeration makes every later miss an
    /// appearance, for good.
    #[test]
    fn close_enumeration_makes_a_miss_an_appearance_for_good() {
        let mut reg = SubRegistry::new();
        let sid = reg.insert(template(ProfileId::default(), "disc"));
        let miss = |reg: &SubRegistry| reg.get(sid).unwrap().mint_state().unwrap().miss();
        assert_eq!(miss(&reg), MintEntry::Enumerated);
        reg.close_enumeration(sid);
        assert_eq!(miss(&reg), MintEntry::Appeared);
        reg.close_enumeration(sid);
        assert_eq!(miss(&reg), MintEntry::Appeared);
    }

    /// A template's rebind gives the params to its spec and to every Sub it minted, and to no other
    /// Sub on their Profiles — an operator's, or another template's mint; the template's identity
    /// and state stand, as does each minted Sub's history.
    #[test]
    fn rebind_of_a_template_reaches_what_it_minted_and_nothing_beside() {
        let profile = |n| ProfileId::from(slotmap::KeyData::from_ffi(n));
        let (discovery, at_a, at_b) = (profile(1), profile(2), profile(3));
        let mut reg = SubRegistry::new();
        let disc = reg.insert(template(discovery, "disc"));
        let other = reg.insert(template(discovery, "other"));
        let a = reg.insert(minted(at_a, "disc@/a", disc));
        let b = reg.insert(minted(at_b, "disc@/b", disc));
        let beside = [
            reg.insert(minted(at_a, "other@/a", other)),
            reg.insert(declared(at_a, "op")),
        ];
        reg.mark_fired(a);
        reg.close_enumeration(disc);
        let identity = |reg: &SubRegistry| {
            let spec = reg
                .get(disc)
                .and_then(Sub::mint_spec)
                .expect("live template");
            spec.identity().config_hash()
        };
        let frozen = identity(&reg);

        let program = program_with(Placeholder::Created);
        let prior = reg.rebind(
            disc,
            SubParams {
                settle: SETTLE * 2,
                spawn: SpawnSpec::new(Arc::clone(&program), EffectScope::SubtreeRoot),
            },
        );

        assert_eq!(
            prior.map(|p| p.settle),
            Some(SETTLE),
            "what the template stamped comes back"
        );
        let runs = |sid| {
            reg.get(sid)
                .and_then(Sub::spawn_spec)
                .is_some_and(|spec| Arc::ptr_eq(spec.program(), &program))
        };
        assert!(
            runs(a) && runs(b),
            "every Sub it minted runs the new program"
        );
        assert_eq!(
            [a, b].map(|sid| reg.get(sid).map(Sub::settle)),
            [Some(SETTLE * 2); 2],
            "and the new settle",
        );
        assert!(
            !beside.into_iter().any(runs),
            "no other Sub on their Profiles moves"
        );
        let stamped = reg
            .get(disc)
            .and_then(Sub::mint_spec)
            .expect("live template");
        assert!(
            Arc::ptr_eq(stamped.params().spawn.program(), &program),
            "the template stamps the new params from now on",
        );
        assert_eq!(identity(&reg), frozen, "its identity stands");
        assert_eq!(
            reg.get(disc).and_then(Sub::mint_state).map(MintState::miss),
            Some(MintEntry::Appeared),
            "its state stands",
        );
        assert!(
            reg.get(a).is_some_and(Sub::has_fired),
            "a minted Sub's history stands"
        );
    }

    /// A per-file scope on a template's *spawn* is minted-reaction payload, not this Profile's
    /// reaction — a `Mint` Sub has no scope of its own and must not trip the per-file recovery-drop
    /// predicate. A plain per-file Sub on the same Profile still does.
    #[test]
    fn has_per_stable_file_sub_excludes_template_bearing_subs() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        reg.insert(Sub::template(
            pid,
            "disc".into(),
            mint_spec(EffectScope::PerStableFile),
        ));
        assert!(
            !reg.has_per_stable_file_sub(pid),
            "a template's reaction is minting, never a per-file Effect",
        );
        reg.insert(Sub::declared(
            pid,
            "fmt".into(),
            params(anchor_only_program(), EffectScope::PerStableFile),
        ));
        assert!(
            reg.has_per_stable_file_sub(pid),
            "a plain per-file Sub still trips the predicate",
        );
    }
}

#[cfg(test)]
mod class_set_tests {
    use super::ClassSet;

    #[test]
    fn empty_is_default() {
        assert_eq!(ClassSet::default(), ClassSet::EMPTY);
        assert_eq!(ClassSet::EMPTY.bits(), 0);
        assert!(ClassSet::EMPTY.is_empty());
    }

    #[test]
    fn distinct_bit_positions() {
        // All four named values pairwise distinct: each occupies its own bit position (verifies the
        // constants haven't drifted).
        let all = [
            ClassSet::EMPTY,
            ClassSet::STRUCTURE,
            ClassSet::CONTENT,
            ClassSet::METADATA,
        ];
        for (i, a) in all.iter().enumerate() {
            for (j, b) in all.iter().enumerate() {
                if i != j {
                    assert_ne!(a, b, "bit constants must be pairwise distinct");
                }
            }
        }
    }

    #[test]
    fn or_combines_bits() {
        let s = ClassSet::STRUCTURE | ClassSet::CONTENT;
        assert!(s.intersects(ClassSet::STRUCTURE));
        assert!(s.intersects(ClassSet::CONTENT));
        assert!(!s.intersects(ClassSet::METADATA));
    }

    #[test]
    fn or_assign_combines_in_place() {
        let mut s = ClassSet::STRUCTURE;
        s |= ClassSet::CONTENT;
        assert_eq!(s, ClassSet::STRUCTURE | ClassSet::CONTENT);
    }

    #[test]
    fn and_intersects_bits() {
        let s = ClassSet::STRUCTURE | ClassSet::CONTENT;
        assert_eq!(s & ClassSet::STRUCTURE, ClassSet::STRUCTURE);
        assert_eq!(s & ClassSet::METADATA, ClassSet::EMPTY);
    }

    #[test]
    fn and_assign_intersects_in_place() {
        let mut s = ClassSet::STRUCTURE | ClassSet::CONTENT;
        s &= ClassSet::CONTENT | ClassSet::METADATA;
        assert_eq!(s, ClassSet::CONTENT);
    }

    #[test]
    fn contains_requires_full_membership() {
        let s = ClassSet::STRUCTURE | ClassSet::CONTENT;
        assert!(s.contains(ClassSet::STRUCTURE));
        assert!(s.contains(ClassSet::CONTENT));
        assert!(s.contains(ClassSet::STRUCTURE | ClassSet::CONTENT));
        assert!(!s.contains(ClassSet::METADATA));
        assert!(!s.contains(ClassSet::CONTENT | ClassSet::METADATA));
    }

    /// `contains(EMPTY)` returns `false` — guards against the bitflags footgun where
    /// `contains(EMPTY) == true` for every set.
    #[test]
    fn contains_empty_is_false() {
        assert!(!ClassSet::EMPTY.contains(ClassSet::EMPTY));
        assert!(!ClassSet::STRUCTURE.contains(ClassSet::EMPTY));
        let full = ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA;
        assert!(!full.contains(ClassSet::EMPTY));
    }

    #[test]
    fn intersects_empty_is_false() {
        assert!(!ClassSet::EMPTY.intersects(ClassSet::EMPTY));
        assert!(!ClassSet::EMPTY.intersects(ClassSet::STRUCTURE));
        assert!(!ClassSet::STRUCTURE.intersects(ClassSet::EMPTY));
    }

    #[test]
    fn intersects_overlap_is_true() {
        let s = ClassSet::STRUCTURE | ClassSet::CONTENT;
        assert!(s.intersects(ClassSet::STRUCTURE));
        assert!(s.intersects(ClassSet::STRUCTURE | ClassSet::METADATA));
        assert!(!s.intersects(ClassSet::METADATA));
    }

    #[test]
    fn bits_round_trip_through_or() {
        // Each case pairs a `ClassSet` with an *independently spelled* bitmask. Asserting the exact
        // `u8` (not popcount) catches a bit-position swap in the constants or `BitOr` — an
        // equal-popcount pair such as 0b011 / 0b101 would hide it.
        let cases: [(ClassSet, u8); 8] = [
            (ClassSet::EMPTY, 0b000),
            (ClassSet::STRUCTURE, 0b001),
            (ClassSet::CONTENT, 0b010),
            (ClassSet::METADATA, 0b100),
            (ClassSet::STRUCTURE | ClassSet::CONTENT, 0b011),
            (ClassSet::CONTENT | ClassSet::METADATA, 0b110),
            (ClassSet::STRUCTURE | ClassSet::METADATA, 0b101),
            (
                ClassSet::STRUCTURE | ClassSet::CONTENT | ClassSet::METADATA,
                0b111,
            ),
        ];
        for (set, expected) in cases {
            assert_eq!(
                set.bits(),
                expected,
                "{set:?} must encode as {expected:#05b}"
            );
        }
    }

    /// Pinned defaults — drift here is a user-facing semantic change.
    #[test]
    fn defaults_pin_expected_classes() {
        assert_eq!(
            ClassSet::DEFAULT_SUBTREE_ROOT,
            ClassSet::STRUCTURE | ClassSet::CONTENT,
            "subtree-root default must include STRUCTURE+CONTENT \
             (CONTENT drives the per-file FDs that surface in-place edits)"
        );
        assert_eq!(
            ClassSet::DEFAULT_PER_FILE,
            ClassSet::CONTENT | ClassSet::METADATA,
            "per-stable-file default must include CONTENT+METADATA"
        );
    }
}
