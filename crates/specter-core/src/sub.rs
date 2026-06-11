//! Subscription and `EffectScope`.
//!
//! A `Sub` is a *reaction declaration*: it names what to watch and what program should run when the
//! watched tree settles. The program is a CFG-shaped op IR — [`ActionProgram`] holds a
//! `Box<[ProgramOp]>` walked by a `u32` cursor at the actuator. The surface syntax (validation-side
//! `Action` tree, lives in `specter-config`) folds into the program at validation time; the engine
//! and actuator see only the lowered form.
//!
//! `Sub.needs_diff` is derived at construction: true iff the `EffectScope` is `PerStableFile` *or*
//! the program references any diff-derived placeholder
//! (`Created`/`Deleted`/`Modified`/`RenamedFrom`/`RenamedTo`).
//!
//! v1 surface is argv-only — no shell variant.

use crate::ids::{ProfileId, ResourceId, SubId};
use crate::program::ActionProgram;
use crate::scan_config::{ProfileIdentity, ScanConfig};
use compact_str::CompactString;
use slotmap::SlotMap;
use smallvec::SmallVec;
use std::collections::BTreeMap;
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

/// The identity a discovery Sub mints its dynamic Subs with — the second identity a dynamic
/// `[[watch]]` carries beyond the discovery Sub's own.
///
/// A discovery Sub's *reaction* is to mint Subs that run its program on each match; its Profile
/// fires attachments, never Effects. The Sub's own `program` / `scope` / `log_output` therefore
/// double as the minted Subs' reaction spec, while this template carries what a plain `SubParams`
/// cannot: the minted Profiles' identity (the user's scan / events / `max_settle` knobs) and the
/// minted Subs' debounce. The discovery Sub's own identity is pinned to discovery constants at
/// config lowering — every user knob lands here instead, so the `[[watch]]` surface keeps its
/// meaning (`settle` debounces the *reaction*, not the discovery walk).
#[derive(Clone, Debug)]
pub struct MintTemplate {
    /// Minted Profiles' identity (the user's scan / events / `max_settle` knobs).
    pub identity: ProfileIdentity,
    /// Minted Subs' debounce (the user's `settle`). Together with `identity.max_settle` it must
    /// satisfy the config layer's `validate_settle` floor — enforced at lowering and debug-asserted
    /// by `Profile::new` at every mint.
    pub settle: Duration,
}

/// The per-Sub reaction declaration: everything that is *not* Profile identity or the anchor.
///
/// `name` is `CompactString`, moved end to end: `SubSpec.name` (config) is already `CompactString`,
/// so the attach request carries it without a `String` round-trip and [`Sub::from_request`] moves
/// it into `Sub.name` unchanged. `program` is `Arc<ActionProgram>` so the Arc minted by the config
/// layer's `lower_to_program` flows through to `Sub.program` without a re-allocation.
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
    pub name: CompactString,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    /// Per-Sub debounce floor — min-folded across the Profile's Subs by the engine's
    /// `recompute_profile_settle`. Distinct from `max_settle`, which is identity (folds into
    /// `config_hash`) and lives on [`ProfileIdentity`].
    pub settle: Duration,
    /// Forward subprocess stdout/stderr to Specter's own stdio (`Stdio::inherit()`); `false` routes
    /// child output to `/dev/null`. Threaded to `Effect.capture_output`; not identity.
    pub log_output: bool,
    /// `Some` ⇒ this Sub is a discovery template; its Profile's scan shape is `MatchChain` — the
    /// two are coupled iff (the engine's attach boundary asserts both directions: a template on a
    /// non-chain Profile and a plain Sub on a chain Profile are equally unconstructable). Minted
    /// Subs always carry `None`, so the detach cascade is structurally one level deep — no
    /// transitive discovery.
    ///
    /// `Arc`: every reconcile pass collects the Profile's template set before minting — a refcount
    /// bump per pass instead of re-borrowing the registry per mint, and `SubParams: Clone` stays
    /// cheap.
    pub template: Option<Arc<MintTemplate>>,
    /// Discovery Sub that minted this Sub — `None` for operator-declared Subs. Read at the engine's
    /// recovery fan-out ([`Sub::is_dynamic`]) and by the detach cascade (`source_discovery ==
    /// detached id` reaps the minted set).
    pub source_discovery: Option<SubId>,
}

impl SubParams {
    /// Params for a Sub whose reaction is spawning its own program — both discovery fields `None`.
    /// The construction funnel for everything that is neither a template nor minted: call sites
    /// name only the spawn fields, so the provenance axes cannot be half-filled by accident.
    #[must_use]
    pub const fn spawn(
        name: CompactString,
        program: Arc<ActionProgram>,
        scope: EffectScope,
        settle: Duration,
        log_output: bool,
    ) -> Self {
        Self {
            name,
            program,
            scope,
            settle,
            log_output,
            template: None,
            source_discovery: None,
        }
    }

    /// Params for a discovery-minted Sub — [`Self::spawn`] plus the minting template's id. The sole
    /// production caller is the discovery reconcile's mint arm; tests injecting dynamic Subs
    /// directly build the same shape through it.
    #[must_use]
    pub const fn minted(
        name: CompactString,
        program: Arc<ActionProgram>,
        scope: EffectScope,
        settle: Duration,
        log_output: bool,
        minted_by: SubId,
    ) -> Self {
        Self {
            name,
            program,
            scope,
            settle,
            log_output,
            template: None,
            source_discovery: Some(minted_by),
        }
    }

    /// The discovery template these params attribute the Sub to — `None` for operator-declared
    /// Subs. Pre-attach twin of [`Sub::minted_by`]; consumers read provenance through it rather
    /// than binding to the field encoding.
    #[must_use]
    pub const fn minted_by(&self) -> Option<SubId> {
        self.source_discovery
    }

    /// Whether these params declare a discovery template — the mint reaction. The engine's attach
    /// boundary asserts the iff-coupling with the `MatchChain` Profile shape on exactly this
    /// predicate.
    #[must_use]
    pub const fn is_template(&self) -> bool {
        self.template.is_some()
    }
}

/// Public-API request to attach a Sub.
///
/// Three orthogonal parts: *where* ([`SubAttachAnchor`]), *which Profile* ([`ProfileIdentity`]),
/// *what the Sub does* ([`SubParams`]). Identity decides Profile partitioning; the anchor resolves
/// separately (not in the hash preimage); params are per-Sub. The split makes a Sub leaking a
/// sibling's identity field structurally unrepresentable.
///
/// Lives in `core::sub` (not `engine`) so [`SubRegistryDiff`] can carry pre-id requests via
/// [`crate::Input::ConfigDiff`] without a `core → engine` cycle. `Clone` serves the rare
/// multi-Engine fan-out; production consumes by value.
#[derive(Clone, Debug)]
pub struct SubAttachRequest {
    pub anchor: SubAttachAnchor,
    pub identity: ProfileIdentity,
    pub params: SubParams,
}

impl SubAttachRequest {
    /// Canonical constructor. [`Self::for_anchor`] is the flat-argument ergonomic over this for the
    /// config layer and tests; the engine's discovery reconcile builds minted Subs through this
    /// directly.
    #[must_use]
    pub const fn from_parts(
        anchor: SubAttachAnchor,
        identity: ProfileIdentity,
        params: SubParams,
    ) -> Self {
        Self {
            anchor,
            identity,
            params,
        }
    }

    /// Build a static (operator-declared) attach request — [`SubParams::spawn`] params. Discovery
    /// templates (config lowering) and minted Subs ([`SubParams::minted`], discovery reconcile)
    /// build their params explicitly and flow through [`Self::from_parts`]. Not `const`: minting
    /// the identity's config handle allocates.
    #[must_use]
    pub fn for_anchor(
        name: CompactString,
        anchor: SubAttachAnchor,
        config: ScanConfig,
        max_settle: Duration,
        settle: Duration,
        program: Arc<ActionProgram>,
        scope: EffectScope,
        events: ClassSet,
        log_output: bool,
    ) -> Self {
        Self::from_parts(
            anchor,
            ProfileIdentity::new(config, max_settle, events),
            SubParams::spawn(name, program, scope, settle, log_output),
        )
    }
}

/// Hot-reload diff. Computed by the TOML loader; consumed by `Engine::step(Input::ConfigDiff(_))`.
///
/// Name-keyed: `removed` carries operator watch names; `added`, `modified_identity`, and
/// `modified_params` carry pre-id [`SubAttachRequest`]s (the name lives inside `params.name`). The
/// engine resolves name → [`SubId`] at apply time through its own authoritative `by_name` index —
/// identity resolution is a registry-owner operation, not the loader's.
///
/// **The `modified` bucket is split.** Two semantically distinct transformations live behind a
/// "modified watch":
///
/// - **Identity change** (`modified_identity`): the anchor path, scan config, max_settle, or events
///   mask differs from the prior spec. Any of these forces the Sub onto a different Profile
///   partition (the partition key is `(anchor_resource, ProfileIdentity::config_hash())`). The
///   engine validates the new anchor's parse first, then performs `detach_old → attach_new`.
///   Validation failure leaves the old Sub in place — structural rollback at the composition layer.
/// - **Params change** (`modified_params`): the anchor and identity are unchanged; only per-Sub
///   fields (`program`, `scope`, `settle`, `log_output`) differ. The engine rebinds the live Sub in
///   place via [`SubRegistry::rebind`]: no Profile churn, no kernel-watch flap, no baseline loss.
///   On the rare case where the prior attach failed and the Sub never entered the registry, the
///   engine degrades the entry to a fresh attach
///   ([`crate::Diagnostic::ConfigDiffRebindFallbackAttach`] narrates the reason).
///
/// Engine processes `removed → modified_params → modified_identity → added` atomically in one step.
/// The four buckets are name-disjoint by diff construction.
#[derive(Clone, Debug, Default)]
pub struct SubRegistryDiff {
    /// Fresh attaches.
    pub added: Vec<SubAttachRequest>,
    /// Detaches by operator watch name.
    pub removed: Vec<CompactString>,
    /// Path / scan / max_settle / events changed — the Sub must move to a different Profile
    /// partition. Engine validates the new anchor's parse, then detaches the old Sub and attaches
    /// the new. Validation failure leaves the old Sub in place (rollback).
    pub modified_identity: Vec<SubAttachRequest>,
    /// Per-Sub fields only (`program`, `scope`, `settle`, `log_output`); anchor and identity
    /// unchanged. Engine rebinds in place via [`SubRegistry::rebind`] — no Profile churn, no
    /// kernel-watch flap. When the named Sub is unexpectedly absent from the registry (prior attach
    /// failed), the engine degrades the entry to a fresh attach.
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
    pub const IN_PLACE_WRITES: Self = Self::CONTENT;

    /// Classes whose subscription suffices to witness *membership* quiescence over a settle window
    /// — the criterion for scan shapes whose proof object is a match set (`ScanConfig::MatchChain`)
    /// rather than a subtree content hash. Membership changes (entries appearing / vanishing /
    /// renaming at chain positions) are all STRUCTURE point events; no in-place-write analog can
    /// span the window invisibly, so a STRUCTURE-covering mask folds the verdict via
    /// `EventsReliable` — N=1, no hash-channel ride.
    ///
    /// **HAZARD — the membership witness does not cover leaf content.** A matched *file* terminus
    /// still folds `{mtime, size, leaf_hash}` into the pruned snapshot hash, so an unwatched
    /// in-place append moves the hash without an event. Under this witness the hash is never
    /// consulted for the verdict and match-set reconciliation depends only on names and kinds, so
    /// that drift is cosmetic baseline noise — but never add an "equal hash ⇒ skip reconcile"
    /// shortcut on this witness without first making terminus leaves fold identity-only.
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

/// A discovery template's runtime carrier on [`Sub`]: the intent ([`MintTemplate`], frozen at
/// attach) plus the per-template-lifetime state that has no meaning off a template.
///
/// The two warning latches share one discipline — same as [`Sub::has_fired`]: a plain bool whose
/// lifetime *is* the slotmap entry's, mutated only through its single registry edge (the registry
/// holds the sole `&mut Sub`), one-shot per template lifetime so a steady-state pathological
/// pattern narrates once, not once per reconcile. Homing the latches here rather than as `Sub`
/// fields makes a latched non-template unrepresentable.
#[derive(Debug)]
pub struct DiscoveryTemplate {
    pub spec: Arc<MintTemplate>,
    /// `true` once this template's live minted-Sub count first crossed the fan-out warning
    /// threshold. Mutated only through [`SubRegistry::latch_fanout_warning`].
    pub fanout_warned: bool,
    /// `true` once a reconcile pass observed a `Symlink`/`Other` chain terminus under this template
    /// and narrated the skip. Mutated only through [`SubRegistry::latch_unsupported_kind_warning`].
    /// Gates only the *diagnostic*, never the skip itself — the terminus kind is read fresh off the
    /// snapshot each pass, so a symlink later replaced by a regular file at the same path mints
    /// normally.
    pub unsupported_kind_warned: bool,
}

#[derive(Debug)]
pub struct Sub {
    pub name: CompactString,
    /// The Profile this Sub attaches to — the join axis of the per-Sub fire bookkeeping onto a
    /// `(Resource, ScanConfig)` partition.
    ///
    /// **Write-once** at [`Sub::from_request`]: re-assigning this would orphan
    /// [`SubRegistry::by_profile`] (the secondary index by `ProfileId`), break the `Sub`-to-`Profile`
    /// lifetime presupposition every dispatcher reads, and silently re-target fire history. The
    /// invariant is held by encapsulation — module-private with no setter — matching the discipline
    /// on [`crate::Profile::resource`] (the same write-once join axis on the Profile side).
    profile: ProfileId,
    pub program: Arc<ActionProgram>,
    pub scope: EffectScope,
    /// Per-Sub debounce floor. `max_settle` and `events` are *not* stored here — they are Profile
    /// identity (fold into `config_hash`, invariant for the Profile's lifetime); read them off the
    /// Profile.
    pub settle: Duration,
    pub needs_diff: bool,
    /// Forward subprocess stdout/stderr to Specter's own stdio. Threaded onto each emitted
    /// [`crate::Effect`] as `capture_output`; the actuator switches between `Stdio::null()` (false,
    /// the default) and `Stdio::inherit()` (true).
    pub log_output: bool,
    /// `Some` ⇒ this Sub is a discovery template (see [`SubParams::template`] for the ⟺ coupling
    /// with the `MatchChain` Profile shape). The carried `spec` is never mutated post-attach: a
    /// template change is an identity change at the config layer (reap + reattach), never an
    /// in-place rebind — the minted Subs hold `Arc`s of the template's program, so a rebind would
    /// strand them on stale reaction state. The carrier's warning latches are the only runtime
    /// fields, each mutated only through its registry edge ([`SubRegistry::latch_fanout_warning`] /
    /// [`SubRegistry::latch_unsupported_kind_warning`]).
    ///
    /// Module-private: external consumers read through [`Self::is_template`] /
    /// [`Self::discovery_template`], so they bind to the predicate, not the optional-field encoding.
    template: Option<DiscoveryTemplate>,
    /// Discovery Sub that minted this Sub — `None` for operator-declared Subs. Never mutated
    /// post-attach. Module-private: external consumers read through [`Self::minted_by`] /
    /// [`Self::is_dynamic`].
    source_discovery: Option<SubId>,
    /// The per-Sub Effect fire history: `true` once this Sub has emitted at least one Effect. Sole
    /// load-bearing reader is the B1 dedup suppress (`!forced && nothing_changed && has_fired` — a
    /// never-fired Sub is its own first emission); the per-Profile SeedDrift filter
    /// ([`SubRegistry::fired_in`]) and the recovery-drift short-circuit
    /// ([`SubRegistry::any_fired`]) read it Profile-wide.
    ///
    /// Lives here, not on [`crate::Profile`]: a Sub attaches to exactly one Profile, so "has this
    /// Sub fired?" is per-Sub, not a per-Profile join table. Its lifetime *is* the slotmap entry's
    /// — detach ([`SubRegistry::remove`]) drops it with the Sub, so a detached or
    /// hot-reload-modified reaction can never re-fire on a later drift verdict and a revived
    /// Profile's fresh Subs start `false` structurally (no purge step to forget). Mutated only
    /// through [`SubRegistry::mark_fired`] / [`SubRegistry::clear_fired`] (the registry holds the
    /// sole `&mut Sub`). The invariant is the weakest tier in the engine — drift self-corrects on
    /// the next real change, no refcount/state-machine corruption — so a plain `bool` carries it
    /// with no edge-method ceremony.
    pub has_fired: bool,
    /// Engine instant at which this Sub last emitted an Effect, or `None` for a Sub that has never
    /// fired. Observational only — distinct from [`Self::has_fired`] (the load-bearing B1-dedup
    /// signal): `last_fired_at` is `Some` iff `has_fired` is true, but the value carries the
    /// timestamp the operator-facing `list` UI renders as a relative or wall-clock instant (via the
    /// bin's `start_instant`/`start_wall` reference pair). Written only through
    /// [`SubRegistry::record_fired`].
    pub last_fired_at: Option<Instant>,
    /// Cumulative Effect emissions across this Sub's lifetime — `SubtreeRoot` increments by 1 per
    /// fire, `PerStableFile` by the per-file count of the emission pass. Observational; surfaces in
    /// the IPC `list` projection. Saturating-add on increment (`u64` holds a millennium of
    /// microsecond-cadence fires); written only through [`SubRegistry::record_fired`].
    pub fire_count: u64,
    /// Cumulative B1-dedup-suppressed verdicts — bumped when this `SubtreeRoot` Sub's
    /// `fire_decision` resolves to `FireVerdict::SuppressDedup` (unchanged tree, already fired, not
    /// forced). Observational; surfaces in `list --wide` for operators tuning dedup behaviour.
    /// `PerStableFile` never suppresses (its dedup is diff-membership), so this counter stays zero
    /// on those Subs. Written only through [`SubRegistry::record_dedup_suppressed`].
    pub dedup_suppressed_count: u64,
}

impl Sub {
    /// Construct a Sub from its [`ProfileId`] and the per-Sub [`SubParams`]. `needs_diff` is
    /// derived: true iff `scope == PerStableFile` OR the program references any diff-derived
    /// placeholder. Pre-computed once; never re-evaluated.
    ///
    /// The slotmap key is the Sub's identity authority — there is no `id` field. `params.name`
    /// (`CompactString`) and `params.program`'s Arc both move through unchanged (no re-allocation,
    /// no Arc re-wrap); one Arc per Sub, refcount-bumped on each emitted [`crate::Effect`].
    #[must_use]
    pub fn from_request(profile: ProfileId, params: SubParams) -> Self {
        let needs_diff =
            params.scope == EffectScope::PerStableFile || params.program.references_diff_derived();
        Self {
            name: params.name,
            profile,
            program: params.program,
            scope: params.scope,
            settle: params.settle,
            needs_diff,
            log_output: params.log_output,
            template: params.template.map(|spec| DiscoveryTemplate {
                spec,
                fanout_warned: false,
                unsupported_kind_warned: false,
            }),
            source_discovery: params.source_discovery,
            has_fired: false,
            last_fired_at: None,
            fire_count: 0,
            dedup_suppressed_count: 0,
        }
    }

    /// The Profile this Sub attaches to. Write-once at [`Self::from_request`]; see the field
    /// rustdoc for the load-bearing invariant.
    #[must_use]
    pub const fn profile(&self) -> ProfileId {
        self.profile
    }

    /// Whether this Sub was minted by a discovery reconcile rather than declared by the operator.
    /// The anchor-terminal recovery fan-out reads it to split all-dynamic Profiles (wholesale
    /// teardown; the source re-mints on reappearance) from mixed/static ones (the static Sub's
    /// `watch_root_parent` recovery channel keeps the Profile alive). A discovery *template* is
    /// operator-declared (`false`) — its recovery is the static channel, exactly like any other
    /// `[[watch]]`.
    #[must_use]
    pub const fn is_dynamic(&self) -> bool {
        self.minted_by().is_some()
    }

    /// The discovery template that minted this Sub — `None` for operator-declared Subs, discovery
    /// templates included. The canonical Sub-side provenance read: the detach cascade reaps a
    /// detached template's minted set by it, the mint dedup resolves "already minted for this
    /// anchor?" through it, and the IPC projections render it.
    #[must_use]
    pub const fn minted_by(&self) -> Option<SubId> {
        self.source_discovery
    }

    /// Whether this Sub is a discovery template — its reaction is minting Subs, never firing
    /// Effects. Coupled iff with the Profile's `MatchChain` shape at the engine's attach boundary.
    #[must_use]
    pub const fn is_template(&self) -> bool {
        self.template.is_some()
    }

    /// The discovery-template carrier when this Sub is a template: the frozen mint spec plus the
    /// per-lifetime warning latches the reconcile pass reads.
    #[must_use]
    pub const fn discovery_template(&self) -> Option<&DiscoveryTemplate> {
        self.template.as_ref()
    }
}

/// Slotmap-backed Sub store with two secondary indices.
///
/// - `by_profile` groups Subs by `ProfileId` (insertion order within a Profile).
/// - `by_name` resolves an operator-facing or synthesised name to its `SubId`. Indexes **every**
///   Sub regardless of `source_discovery`: the config validator reserves the `@` byte, so a
///   `[[watch]].name` never carries one and a minted `<template_name>@<matched_path>` always does —
///   the two populations are disjoint by construction and their union is unique. Callers that need
///   the static-vs-dynamic discrimination read [`Sub::is_dynamic`] on the resolved Sub. The index
///   is load-bearing — hot-reload resolves every `removed`/`modified` name to an id through
///   [`Self::find_by_name`] (O(log N)).
///
/// `by_name` mirrors the slotmap entry's lifetime: [`Self::insert`] populates it, [`Self::remove`]
/// clears it id-checked. The `insert` `debug_assert!` is the dev/CI signal for a duplicate name;
/// the validator (static side) and the discovery reconcile's registry-derived dedup (dynamic side)
/// make a collision unreachable in correct operation, and the `@`-disjointness keeps the two
/// construction sites from racing each other. A release-mode breach is contained by the id-checked
/// `remove`: the *mapping* stays 1:1; only a hypothetical orphaned slotmap entry (never the wrong
/// name→id edge) could survive.
#[derive(Debug, Default)]
pub struct SubRegistry {
    subs: SlotMap<SubId, Sub>,
    by_profile: BTreeMap<ProfileId, SmallVec<[SubId; 2]>>,
    by_name: BTreeMap<CompactString, SubId>,
}

impl SubRegistry {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert a Sub; the returned slotmap [`SubId`] is its identity authority (the Sub carries no
    /// `id` field). Both secondary indices update in lockstep — `by_profile` and `by_name` are
    /// populated for every Sub.
    ///
    /// The `debug_assert!` fires on a duplicate name — the dev/CI signal only. Static-name
    /// uniqueness is validator-enforced; dynamic-name uniqueness is reconcile-enforced (the
    /// registry-derived dedup mints one Sub per terminus per template); cross-population uniqueness
    /// is structural via the `@`-byte reservation. A release-mode breach is contained by the
    /// id-checked [`Self::remove`] (the mapping stays consistent).
    pub fn insert(&mut self, sub: Sub) -> SubId {
        let profile = sub.profile;
        let name = sub.name.clone();
        let id = self.subs.insert(sub);
        self.by_profile.entry(profile).or_default().push(id);
        debug_assert!(
            !self.by_name.contains_key(&name),
            "duplicate Sub name escaped registration: {name:?}",
        );
        self.by_name.insert(name, id);
        id
    }

    /// Remove a Sub by id, returning the owned value. Clears both secondary indices. The `by_name`
    /// clear is **id-checked** — the entry drops only if it still points at `id`, so removing a
    /// duplicate-name escape's shadowed id (a release-mode diff bug) cannot clobber the live id's
    /// mapping. Returns `None` for a stale id.
    pub fn remove(&mut self, id: SubId) -> Option<Sub> {
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
        Some(sub)
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
    /// Returns hits for both static and dynamic Subs — callers that need the discrimination read
    /// [`Sub::is_dynamic`] on the resolved Sub. The config validator reserves the `@` byte, so a
    /// static name and a minted `<template_name>@<matched_path>` cannot collide; uniqueness across
    /// the union is structural.
    ///
    /// Load-bearing: the engine's hot-reload shim resolves every `removed`/`modified` name through
    /// here.
    #[must_use]
    pub fn find_by_name(&self, name: &str) -> Option<SubId> {
        self.by_name.get(name).copied()
    }

    /// Record that `sub` emitted an Effect — the B1-dedup / SeedDrift fire-history mark, written by
    /// `emit_effects`' SubtreeRoot arm on a successful push. Idempotent. A stale `SubId` (the Sub
    /// detached between the emit decision and here) is a silent no-op: the flag already died with
    /// the slotmap entry.
    pub fn mark_fired(&mut self, sub: SubId) {
        if let Some(s) = self.subs.get_mut(sub) {
            s.has_fired = true;
        }
    }

    /// Record `count` successful Effect emissions on `sub` at `now`. Bumps [`Sub::fire_count`] by
    /// `count` (saturating) and writes [`Sub::last_fired_at`] = `Some(now)`. Observational —
    /// [`Self::mark_fired`] is the load-bearing B1-dedup edge; this one carries the per-Sub fire
    /// history the operator-facing `list` projection renders.
    ///
    /// Called at most once per Sub per `emit_effects` pass on the emit-side. `count` is `1` for a
    /// `SubtreeRoot` emission and the per-file count for a `PerStableFile` emission (aggregated so
    /// `Diagnostic::SubFired`'s wire stream isn't amplified by N). A stale `SubId` is a silent
    /// no-op — the counter already died with the slotmap entry, mirroring [`Self::mark_fired`].
    pub fn record_fired(&mut self, sub: SubId, count: u32, now: Instant) {
        if let Some(s) = self.subs.get_mut(sub) {
            s.fire_count = s.fire_count.saturating_add(u64::from(count));
            s.last_fired_at = Some(now);
        }
    }

    /// Bump [`Sub::dedup_suppressed_count`] by one — written when `emit_effects` resolves a
    /// `SubtreeRoot` Sub to `FireVerdict::SuppressDedup` (unchanged tree + already-fired + not
    /// forced). Saturating add; observational only. A stale `SubId` is a silent no-op — same shape
    /// as [`Self::mark_fired`] and [`Self::record_fired`].
    pub fn record_dedup_suppressed(&mut self, sub: SubId) {
        if let Some(s) = self.subs.get_mut(sub) {
            s.dedup_suppressed_count = s.dedup_suppressed_count.saturating_add(1);
        }
    }

    /// Replace `sub`'s per-Sub fields with `new_params` in place — the `modified_params` arm of
    /// hot-reload's [`SubRegistryDiff`] split.
    ///
    /// **Preserves**: [`SubId`], `profile`, `name`, `source_discovery`, `has_fired`,
    /// `last_fired_at`, `fire_count`, `dedup_suppressed_count`. The first two are structural (the
    /// slotmap key and the Profile join are invariants of this Sub's lifetime); `name` and
    /// `source_discovery` are pinned by the rebind invariant (callers route through
    /// [`Self::find_by_name`], which keys on `name`, and a `source_discovery` change would cross
    /// the static↔dynamic boundary the diff already maps to add+remove). `has_fired` is preserved
    /// because the B1 dedup floor reads it as "this Sub has already announced the current stable
    /// tree state"; a program swap changes *what runs*, not *whether the tree changed*. The three
    /// observational counters (`last_fired_at` / `fire_count` / `dedup_suppressed_count`) are
    /// preserved for the same reason — they record this Sub's history under its operator-facing
    /// name, and a `modified_params` rebind leaves both identity and history intact.
    ///
    /// **Replaces**: `program`, `scope`, `settle`, `log_output`; recomputes `needs_diff` (derived
    /// from `scope` + the program's diff-placeholder set).
    ///
    /// Returns `Some((prior_settle, profile))` on success: `prior_settle` is the per-Sub settle the
    /// caller compares against `new_params.settle` to gate a Profile-settle recompute; `profile` is
    /// the rebound Sub's host Profile, threaded out so the wrapper avoids a second `get(sub)` for
    /// the recompute target. Both reads fold into the same `get_mut` the mutation uses — the
    /// wrapper's recompute gate becomes a single comparison with no follow-up lookup. `Sub.profile`
    /// is invariant for the Sub's lifetime, so returning it costs nothing observable on the rebind
    /// itself.
    ///
    /// Returns `None` on a stale [`SubId`]. The invariant is that the dispatcher resolves through
    /// [`Self::find_by_name`] in the same step as the rebind, so a stale id is structurally
    /// unexpected; the caller surfaces it via [`crate::Diagnostic::RebindUnknownSub`].
    ///
    /// `debug_assert!`s pin the `name` / `source_discovery` invariants — a release-mode breach would
    /// silently rewrite the identifying fields under the registry's `by_name` index, leaving the
    /// index pointing at the wrong [`SubId`]; the assertions catch the breach at the call site. The
    /// template assertion is both-`None`, not equality (`ProfileIdentity` deliberately has no `Eq`):
    /// any field change on a template-bearing spec classifies as `modified_identity` (wholesale reap
    /// + reattach) at the config diff, never an in-place rebind — the minted Subs hold `Arc`s of the
    /// template's program, so a rebind would strand them on stale reaction state.
    pub fn rebind(&mut self, sub: SubId, new_params: SubParams) -> Option<(Duration, ProfileId)> {
        let s = self.subs.get_mut(sub)?;
        debug_assert_eq!(
            s.name, new_params.name,
            "rebind cannot change Sub name (rebind identity invariant)",
        );
        debug_assert_eq!(
            s.source_discovery, new_params.source_discovery,
            "rebind cannot change source_discovery (static↔dynamic boundary)",
        );
        debug_assert!(
            s.template.is_none() && new_params.template.is_none(),
            "rebind never touches a template-bearing Sub — template changes classify \
             as modified_identity (reap + reattach), never an in-place rebind",
        );
        let prior_settle = s.settle;
        let profile = s.profile;
        s.program = new_params.program;
        s.scope = new_params.scope;
        s.settle = new_params.settle;
        s.log_output = new_params.log_output;
        s.needs_diff = s.scope == EffectScope::PerStableFile || s.program.references_diff_derived();
        Some((prior_settle, profile))
    }

    /// Clear `sub`'s fire history — the `EffectComplete::Failed` clear. A failed Effect produced no
    /// observation worth deduplicating against, so the next stable verdict at this Sub must re-fire
    /// even on an unchanged tree. No-op on a stale `SubId` (already detached ⇒ its history is
    /// already gone).
    pub fn clear_fired(&mut self, sub: SubId) {
        if let Some(s) = self.subs.get_mut(sub) {
            s.has_fired = false;
        }
    }

    /// Whether any Sub on `profile` has fired — the fast `seed_drift_observed` short-circuit
    /// ("never fired ⇒ no prior emission to re-fire on recovery").
    #[must_use]
    pub fn any_fired(&self, profile: ProfileId) -> bool {
        self.at(profile)
            .iter()
            .any(|sid| self.subs.get(*sid).is_some_and(|s| s.has_fired))
    }

    /// The Subs on `profile` that have fired — the SeedDrift conservative-recovery fire-filter basis.
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
            .filter(|sid| self.subs.get(*sid).is_some_and(|s| s.has_fired))
            .collect()
    }

    /// One-shot fan-out warning latch for the discovery template `sub`.
    ///
    /// `count` is the caller's *live* minted-Sub tally for this template, derived from registry
    /// truth — no mirror to drift. Returns `Some(count)` the first time `count` exceeds `threshold`
    /// and latches [`DiscoveryTemplate::fanout_warned`] so later crossings return `None` — a
    /// pathological pattern warns once per template lifetime. The check-and-latch is atomic here,
    /// so the one-shot property is structural rather than a caller convention; the engine reads the
    /// latch off the template carrier as a pre-gate so it can skip computing `count` once warned. A
    /// stale `SubId` or a non-template Sub is a silent `None` — the latch lives on the template
    /// carrier, so the miss mirrors [`Self::mark_fired`]'s died-with-the-entry contract.
    pub fn latch_fanout_warning(
        &mut self,
        sub: SubId,
        threshold: usize,
        count: usize,
    ) -> Option<usize> {
        let t = self.subs.get_mut(sub)?.template.as_mut()?;
        (count > threshold && !t.fanout_warned).then(|| {
            t.fanout_warned = true;
            count
        })
    }

    /// One-shot unsupported-terminus warning latch for the discovery template `sub` — the sibling
    /// of [`Self::latch_fanout_warning`] for the `Symlink`/`Other` mint-skip narration.
    ///
    /// Returns `true` exactly once per template lifetime: the first call latches
    /// [`DiscoveryTemplate::unsupported_kind_warned`] and reports "newly latched"; later calls
    /// return `false`. The check-and-latch is atomic here, so the one-shot property is structural
    /// rather than a caller convention. A stale `SubId` or a non-template Sub is a silent `false` —
    /// the latch lives on the template carrier, mirroring [`Self::latch_fanout_warning`]'s
    /// died-with-the-entry contract.
    pub fn latch_unsupported_kind_warning(&mut self, sub: SubId) -> bool {
        let Some(t) = self.subs.get_mut(sub).and_then(|s| s.template.as_mut()) else {
            return false;
        };
        !std::mem::replace(&mut t.unsupported_kind_warned, true)
    }

    /// Whether `profile` has at least one attached Sub that *reacts* per-stable-file — the scope
    /// test behind the per-file recovery-drop signal.
    ///
    /// **Template-bearing Subs are excluded.** A discovery Sub stores the user's scope as the
    /// minted Subs' reaction spec (the double-duty params reading), but its own reaction is minting
    /// — it never fires a per-file Effect, so a per-file template scope must not trip the
    /// recovery-drop diagnostic on the discovery Profile. The minted Subs answer for themselves on
    /// their own Profiles, where their stored scope *is* their reaction.
    ///
    /// **Must not be collapsed into [`crate::Profile::has_per_file_fds`].** That predicate is
    /// events-mask derived (`CONTENT | METADATA` present) and a `SubtreeRoot` Sub watching
    /// `CONTENT` sets it just as much as a `PerStableFile` Sub does — it is *necessary* for
    /// per-file FDs but *not sufficient* for "this Profile carries a per-file-*scoped* reaction".
    /// Swapping this scan for `has_per_file_fds` would false-positive the recovery-drop diagnostic
    /// on Subtree-only Profiles that happen to watch content. The `scope` field is the only sound
    /// witness; the scan stays.
    #[must_use]
    pub fn has_per_stable_file_sub(&self, profile: ProfileId) -> bool {
        self.at(profile).iter().any(|sid| {
            self.subs
                .get(*sid)
                .is_some_and(|s| s.template.is_none() && s.scope == EffectScope::PerStableFile)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActionProgram, ClassSet, EffectScope, MintTemplate, ProfileIdentity, ScanConfig, Sub,
        SubParams, SubRegistry, SubRegistryDiff,
    };
    use crate::ids::{ProfileId, SubId};
    use crate::program::{
        ArgPart, ArgTemplate, BranchTarget, ExecAction, Placeholder, ProgramBuilder, SpawnBody,
    };
    use compact_str::CompactString;
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
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "fmt".into(),
                anchor_only_program(),
                EffectScope::PerStableFile,
                SETTLE,
                false,
            ),
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_set_for_diff_placeholder_in_subtree_scope() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "report".into(),
                program_with(Placeholder::Created),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        );
        assert!(sub.needs_diff);
    }

    #[test]
    fn needs_diff_false_for_anchor_subtree_combo() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        );
        assert!(!sub.needs_diff);
    }

    /// A freshly built Sub starts with no fire history — the B1-dedup / SeedDrift baseline.
    /// Relocated from the deleted per-Profile `new_profile_initialises_fired_subs_empty`: the
    /// history now lives per-Sub, so the "starts empty" contract is asserted on the Sub, not the
    /// Profile. Also pins the three observational counters' fresh state — `record_fired` /
    /// `record_dedup_suppressed` are the only writers, so a fresh Sub can never carry inherited
    /// history from a slotmap slot's prior occupant.
    #[test]
    fn fresh_sub_starts_unfired() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        );
        assert!(!sub.has_fired, "fresh Sub: no prior Effect emission");
        assert!(sub.last_fired_at.is_none(), "no fire timestamp");
        assert_eq!(sub.fire_count, 0, "no cumulative fires");
        assert_eq!(sub.dedup_suppressed_count, 0, "no suppressed verdicts");
    }

    /// `record_fired` accumulates per-pass counts into `fire_count` and stamps `last_fired_at` with
    /// the supplied instant. The B1-dedup `has_fired` is untouched — `mark_fired` and
    /// `record_fired` are disjoint edge methods on disjoint pieces of fire history.
    #[test]
    fn record_fired_bumps_count_and_stamps_last_fired() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        let t0 = Instant::now();
        reg.record_fired(sid, 1, t0);
        let s = reg.get(sid).expect("Sub alive");
        assert_eq!(s.fire_count, 1, "first fire bumps count by 1");
        assert_eq!(s.last_fired_at, Some(t0), "first fire stamps timestamp");
        assert!(
            !s.has_fired,
            "record_fired must NOT touch has_fired (mark_fired owns it)",
        );

        // A PerStableFile-style aggregation: count=3 adds to the running total, timestamp advances.
        let t1 = t0 + Duration::from_millis(10);
        reg.record_fired(sid, 3, t1);
        let s = reg.get(sid).expect("Sub alive");
        assert_eq!(s.fire_count, 4, "second fire aggregates: 1 + 3 = 4");
        assert_eq!(s.last_fired_at, Some(t1), "timestamp advances");

        // Stale id is a silent no-op.
        reg.remove(sid).expect("removed");
        reg.record_fired(sid, 1, t1); // would otherwise panic on missing entry
    }

    /// `record_dedup_suppressed` increments the dedicated counter and touches no other field — the
    /// SuppressDedup arm signals "Sub would have fired but the dedup floor said no", distinct from
    /// fires (`record_fired`) and the B1 flag (`mark_fired`).
    #[test]
    fn record_dedup_suppressed_bumps_only_its_own_counter() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        reg.record_dedup_suppressed(sid);
        reg.record_dedup_suppressed(sid);
        let s = reg.get(sid).expect("Sub alive");
        assert_eq!(s.dedup_suppressed_count, 2);
        assert_eq!(s.fire_count, 0, "suppression does not bump fire_count");
        assert!(
            s.last_fired_at.is_none(),
            "suppression does not stamp last_fired_at",
        );
        assert!(!s.has_fired, "suppression does not touch has_fired");

        reg.remove(sid).expect("removed");
        reg.record_dedup_suppressed(sid); // stale id is a silent no-op
    }

    #[test]
    fn registry_at_groups_by_profile() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();

        let s1 = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "a".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        let s2 = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "b".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));

        let mut got = reg.at(pid).to_vec();
        got.sort();
        let mut expected = vec![s1, s2];
        expected.sort();
        assert_eq!(got, expected);
    }

    #[test]
    fn registry_at_empty_for_unknown_profile() {
        let reg = SubRegistry::new();
        assert!(reg.at(ProfileId::default()).is_empty());
    }

    /// After a multi-insert/remove sequence, every key `iter()` yields re-looks-up via `get` to the
    /// same Sub, and `at(profile)` equals the live key set. The slotmap key is the sole identity
    /// authority (a `Sub` carries no `id`).
    #[test]
    fn registry_iter_keys_round_trip_through_get() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mk = |name: &str| {
            Sub::from_request(
                pid,
                SubParams::spawn(
                    name.into(),
                    anchor_only_program(),
                    EffectScope::SubtreeRoot,
                    SETTLE,
                    false,
                ),
            )
        };
        let a = reg.insert(mk("a"));
        let b = reg.insert(mk("b"));
        let c = reg.insert(mk("c"));
        reg.remove(b);

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
        let id = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
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
        let id = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        reg.remove(id);
        assert!(reg.find_by_name("build").is_none());
    }

    /// `by_name` indexes every Sub regardless of `source_discovery` — both a static operator name and
    /// a minted `<template_name>@<matched_path>` resolve through `find_by_name`. The two populations
    /// are disjoint by the config validator's `@`-byte reservation, so their indexed union is unique.
    #[test]
    fn by_name_indexes_static_and_dynamic_subs() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mk = |name: &str, source: Option<SubId>| {
            let params = match source {
                Some(src) => SubParams::minted(
                    name.into(),
                    anchor_only_program(),
                    EffectScope::SubtreeRoot,
                    SETTLE,
                    false,
                    src,
                ),
                None => SubParams::spawn(
                    name.into(),
                    anchor_only_program(),
                    EffectScope::SubtreeRoot,
                    SETTLE,
                    false,
                ),
            };
            Sub::from_request(pid, params)
        };

        let static_id = reg.insert(mk("foo", None));
        let dynamic_id = reg.insert(mk("p@/tmp/x", Some(SubId::default())));

        assert_eq!(
            reg.find_by_name("foo"),
            Some(static_id),
            "static name resolves",
        );
        assert_eq!(
            reg.find_by_name("p@/tmp/x"),
            Some(dynamic_id),
            "synthesised dynamic name resolves",
        );
        assert!(
            reg.find_by_name("nope").is_none(),
            "absent name yields None",
        );
    }

    /// `remove` drops the dynamic Sub's `by_name` entry just like a static one — and `by_profile`
    /// accounting is symmetric across the static/dynamic axis.
    #[test]
    fn remove_clears_by_name_for_dynamic_sub() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let dynamic_id = reg.insert(Sub::from_request(
            pid,
            SubParams::minted(
                "p@/tmp/x".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
                SubId::default(),
            ),
        ));
        assert_eq!(reg.find_by_name("p@/tmp/x"), Some(dynamic_id));

        reg.remove(dynamic_id).expect("dynamic removed");
        assert!(
            reg.find_by_name("p@/tmp/x").is_none(),
            "dynamic Sub's by_name entry dropped on remove",
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
        let sid = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));

        let removed = reg.remove(sid);
        assert!(removed.is_some());
        assert!(reg.get(sid).is_none());
        assert!(reg.at(pid).is_empty());
        assert_eq!(reg.len(), 0);
    }

    /// `Sub.program` is reference-counted: cloning the field bumps the strong count without copying
    /// the inner [`ActionProgram`].
    #[test]
    fn sub_program_is_arc_wrapped() {
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        );

        let initial = Arc::strong_count(&sub.program);
        let bumped = Arc::clone(&sub.program);
        assert_eq!(
            Arc::strong_count(&sub.program),
            initial + 1,
            "Arc::clone increments strong_count on the field",
        );
        assert!(
            Arc::ptr_eq(&bumped, &sub.program),
            "the clone and the field point at the same allocation",
        );
    }

    /// `Sub::from_request` does not re-wrap the program: the caller's Arc is the same allocation
    /// the Sub stores. The minted Arc from the config layer's `lower_to_program` flows through
    /// without churn.
    #[test]
    fn sub_new_does_not_rewrap_program_arc() {
        let program = anchor_only_program();
        let before = Arc::as_ptr(&program);
        let sub = Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "build".into(),
                Arc::clone(&program),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        );
        assert!(
            std::ptr::eq(before, Arc::as_ptr(&sub.program)),
            "Sub::from_request must not allocate a new ActionProgram",
        );
    }

    /// Diff is plain data — pins the `Default` shape and the [`SubRegistryDiff::is_empty`] contract
    /// in one place. **Each** of the four buckets independently flips the predicate, so a future
    /// bucket addition that forgets to extend `is_empty` is caught here.
    #[test]
    fn sub_registry_diff_is_empty_per_bucket() {
        assert!(SubRegistryDiff::default().is_empty(), "default is empty");

        let req = || {
            crate::SubAttachRequest::for_anchor(
                "a".into(),
                crate::SubAttachAnchor::Path(std::path::PathBuf::from("/a")),
                crate::ScanConfig::builder().build(),
                Duration::from_hours(1),
                SETTLE,
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                crate::ClassSet::DEFAULT_SUBTREE_ROOT,
                false,
            )
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

    /// `SubRegistry::rebind` replaces the four per-Sub fields and preserves the structural ones —
    /// including `has_fired`, which the B1 dedup floor reads as "this Sub has already announced the
    /// current stable tree state." A program swap changes *what runs*, not *whether the tree
    /// changed*, so the flag must not reset on rebind.
    #[test]
    fn rebind_replaces_per_sub_fields_and_preserves_has_fired() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let original = anchor_only_program();
        let sid = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                Arc::clone(&original),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        reg.mark_fired(sid);
        let fired_at = Instant::now();
        reg.record_fired(sid, 7, fired_at);
        reg.record_dedup_suppressed(sid);

        let new_program = program_with(Placeholder::Created);
        let new_settle = SETTLE + SETTLE;
        let prior = reg.rebind(
            sid,
            SubParams::spawn(
                "build".into(),
                Arc::clone(&new_program),
                EffectScope::PerStableFile,
                new_settle,
                true,
            ),
        );

        assert_eq!(
            prior,
            Some((SETTLE, pid)),
            "rebind returns the prior settle and the host Profile",
        );
        let s = reg.get(sid).expect("Sub preserved across rebind");
        assert!(s.has_fired, "has_fired preserved across rebind");
        assert_eq!(
            s.last_fired_at,
            Some(fired_at),
            "last_fired_at preserved — operator-facing fire history",
        );
        assert_eq!(s.fire_count, 7, "fire_count preserved across rebind");
        assert_eq!(
            s.dedup_suppressed_count, 1,
            "dedup_suppressed_count preserved across rebind",
        );
        assert_eq!(s.name, "build", "name preserved");
        assert_eq!(s.profile, pid, "profile preserved");
        assert!(s.source_discovery.is_none(), "source_discovery preserved");
        assert_eq!(s.scope, EffectScope::PerStableFile, "scope replaced");
        assert_eq!(s.settle, new_settle, "settle replaced");
        assert!(s.log_output, "log_output replaced");
        assert!(
            Arc::ptr_eq(&s.program, &new_program),
            "program Arc replaced (no rewrap)",
        );
        assert!(
            s.needs_diff,
            "needs_diff recomputed — PerStableFile alone sets it true",
        );
    }

    /// Stale `SubId` returns `None`. The dispatcher resolves through `find_by_name` in the same
    /// step as the rebind, so this surface is rarely hit in production; the engine wraps it in a
    /// `RebindUnknownSub` diagnostic when it does.
    #[test]
    fn rebind_returns_none_on_stale_sub_id() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let sid = reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        reg.remove(sid).expect("removed");
        let res = reg.rebind(
            sid,
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        );
        assert!(res.is_none(), "stale SubId yields None");
    }

    /// A template-bearing `SubParams` fixture. The minted identity is an arbitrary `Subtree` scan:
    /// the template ⟺ `MatchChain` coupling is the engine attach boundary's invariant, not the
    /// registry's — these pins exercise registry mechanics only.
    fn template_params(name: &str) -> SubParams {
        SubParams {
            name: name.into(),
            program: anchor_only_program(),
            scope: EffectScope::SubtreeRoot,
            settle: SETTLE,
            log_output: false,
            template: Some(Arc::new(MintTemplate {
                identity: ProfileIdentity::new(
                    ScanConfig::builder().build(),
                    SETTLE * 4,
                    ClassSet::EMPTY,
                ),
                settle: SETTLE,
            })),
            source_discovery: None,
        }
    }

    /// The fan-out latch crossing edge: strict-greater (`count == threshold` does not cross), the
    /// first crossing returns the count and latches, later crossings are silent — one warning per
    /// template lifetime, structural in the check-and-latch.
    #[test]
    fn latch_fanout_warning_crosses_strict_greater_exactly_once() {
        let mut reg = SubRegistry::new();
        let sid = reg.insert(Sub::from_request(
            ProfileId::default(),
            template_params("disc"),
        ));
        assert_eq!(
            reg.latch_fanout_warning(sid, 3, 3),
            None,
            "count == threshold does not cross (strict greater)",
        );
        assert!(
            !reg.get(sid)
                .unwrap()
                .template
                .as_ref()
                .unwrap()
                .fanout_warned,
            "a non-crossing probe leaves the latch open",
        );
        assert_eq!(
            reg.latch_fanout_warning(sid, 3, 4),
            Some(4),
            "first crossing returns the count and latches",
        );
        assert_eq!(
            reg.latch_fanout_warning(sid, 3, 5),
            None,
            "latched: later crossings are silent",
        );
    }

    /// A non-template Sub and a stale id both yield a silent `None` — the latch lives on the
    /// template carrier, so the miss mirrors `mark_fired`'s died-with-the-entry contract.
    #[test]
    fn latch_fanout_warning_none_for_non_template_and_stale_id() {
        let mut reg = SubRegistry::new();
        let plain = reg.insert(Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        assert_eq!(
            reg.latch_fanout_warning(plain, 0, 10),
            None,
            "non-template Sub: the latch has no home",
        );
        reg.remove(plain).expect("removed");
        assert_eq!(
            reg.latch_fanout_warning(plain, 0, 10),
            None,
            "stale id: silent miss",
        );
    }

    /// Rebind tripwire: `source_discovery` is a synthesis-origin identity field — crossing the
    /// static↔dynamic boundary in place would silently re-attribute the Sub's cascade membership.
    #[test]
    #[should_panic(expected = "source_discovery")]
    fn rebind_panics_on_source_discovery_change() {
        let mut reg = SubRegistry::new();
        let sid = reg.insert(Sub::from_request(
            ProfileId::default(),
            SubParams::spawn(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
            ),
        ));
        let _ = reg.rebind(
            sid,
            SubParams::minted(
                "build".into(),
                anchor_only_program(),
                EffectScope::SubtreeRoot,
                SETTLE,
                false,
                SubId::default(),
            ),
        );
    }

    /// Rebind tripwire: a template-bearing Sub never rebinds in place — minted Subs hold `Arc`s of
    /// the template's program, so an in-place swap would strand them on stale reaction state. The
    /// config diff classifies any template-spec change as `modified_identity` (reap + reattach);
    /// this assert is the core-side floor under that rule.
    #[test]
    #[should_panic(expected = "template")]
    fn rebind_panics_on_template_bearing_sub() {
        let mut reg = SubRegistry::new();
        let sid = reg.insert(Sub::from_request(
            ProfileId::default(),
            template_params("disc"),
        ));
        let _ = reg.rebind(sid, template_params("disc"));
    }

    /// A per-file *template* scope is minted-reaction payload, not this Profile's reaction — it
    /// must not trip the per-file recovery-drop predicate. A plain per-file Sub on the same Profile
    /// still does.
    #[test]
    fn has_per_stable_file_sub_excludes_template_bearing_subs() {
        let mut reg = SubRegistry::new();
        let pid = ProfileId::default();
        let mut tp = template_params("disc");
        tp.scope = EffectScope::PerStableFile;
        reg.insert(Sub::from_request(pid, tp));
        assert!(
            !reg.has_per_stable_file_sub(pid),
            "a discovery Sub's reaction is minting, never a per-file Effect",
        );
        reg.insert(Sub::from_request(
            pid,
            SubParams::spawn(
                "fmt".into(),
                anchor_only_program(),
                EffectScope::PerStableFile,
                SETTLE,
                false,
            ),
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
